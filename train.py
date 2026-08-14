"""第一版训练闭环：AdamW + warmup/cosine + FP16 + 梯度累积 + checkpoint。

规格来自 AGENTS.md，超参里的模型部分复用 model.GPTConfig，不要另写一套。
训练部分（学习率、累积步数、checkpoint 间隔）写在本文件，不提前拆 config.py。

    优化器        AdamW，weight_decay 0.1（只打在 2D 权重上，bias / LayerNorm 除外）
    峰值学习率    3e-4
    调度          linear warmup + cosine decay
    精度          FP16（autocast + GradScaler）
    micro batch   16，梯度累积 4，有效 batch 64
    checkpoint    每 1000 steps 存 latest；validation 最优另存 best

一个 optimizer step = 4 个 micro-batch 的梯度累加。学习率、日志、checkpoint
都按 optimizer step 计数，不是按 micro-batch。

用法（仓库根目录，解释器必须是 Mini_GPT 环境）：
    python train.py --max-steps 20 --eval-interval 10 --eval-iters 4 --batch-size 4
    python train.py
    python train.py --max-steps 30000
    python train.py --resume

本机 RTX 3070 Laptop 8GB 上 micro batch 16 的反向会 OOM，验证链路用 --batch-size 4。
完整训练仍按 3090 的默认 micro batch 16。
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from dataset import BATCH_SIZE, BLOCK_SIZE, get_batch, load_tokens
from model import GPTConfig, MiniGPT

PEAK_LR = 3e-4
MIN_LR = 3e-5          # cosine 末端，约为峰值的 10%
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)    # GPT 系常用；AdamW 默认 (0.9, 0.999) 对 LM 偏慢
GRAD_CLIP = 1.0
GRAD_ACCUM = 4
MAX_STEPS = 10_000     # 先跑 1 万步验证链路；基线再显式传 --max-steps 30000
WARMUP_STEPS = 200
EVAL_INTERVAL = 1_000
EVAL_ITERS = 20
CKPT_INTERVAL = 1_000
LOG_INTERVAL = 10

CKPT_DIR = Path("checkpoints")
LOG_DIR = Path("experiments")


@dataclass
class TrainState:
    """需要写进 checkpoint、恢复时原样读回来的训练状态。"""

    step: int
    best_val_loss: float


def configure_optimizer(
    model: nn.Module,
    weight_decay: float,
    learning_rate: float,
    betas: tuple[float, float],
    device: str,
) -> torch.optim.AdamW:
    """AdamW 参数分组。

    权重共享后 tok_emb.weight 和 lm_head.weight 是同一块存储，named_parameters
    会列出两次。用 id 去重，否则优化器会把同一份梯度加两遍。

    2D 矩阵（Embedding / Linear 的 weight）吃 weight_decay；1D 的 bias 和
    LayerNorm 不吃。decay 打在归一化参数上会把尺度往 0 拉，训练更不稳。
    """
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if id(param) in seen:
            continue
        seen.add(id(param))
        if param.ndim >= 2:
            decay.append(param)
        else:
            no_decay.append(param)

    fused = device == "cuda"
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=betas,
        fused=fused,
    )


def get_lr(step: int, warmup_steps: int, max_steps: int) -> float:
    """linear warmup 到 PEAK_LR，再 cosine 降到 MIN_LR。

    step 从 0 起。warmup 阶段用 (step + 1) / warmup_steps，避免第 0 步学习率为 0
    时第一次更新完全空转。
    """
    if step < warmup_steps:
        return PEAK_LR * (step + 1) / max(warmup_steps, 1)
    if step >= max_steps:
        return MIN_LR
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return MIN_LR + coeff * (PEAK_LR - MIN_LR)


@torch.no_grad()
def estimate_loss(
    model: MiniGPT,
    splits: dict[str, object],
    batch_size: int,
    block_size: int,
    eval_iters: int,
    device: str,
) -> dict[str, float]:
    """各 split 上随机抽 eval_iters 个 batch，返回平均 cross-entropy。

    perplexity = exp(loss)。loss 是 token 级平均 NLL，所以 ppl 是「平均每个
    token 的分支数」：随机初始化约 50257，训练后应明显下降。
    """
    model.eval()
    out: dict[str, float] = {}
    for name, tokens in splits.items():
        total = 0.0
        for _ in range(eval_iters):
            x, y = get_batch(tokens, batch_size, block_size, device)
            with autocast(device, dtype=torch.float16):
                _, loss = model(x, y)
            total += float(loss.item())
        out[name] = total / eval_iters
    model.train()
    return out


def save_checkpoint(
    path: Path,
    model: MiniGPT,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    state: TrainState,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "step": state.step,
            "best_val_loss": state.best_val_loss,
            "config": asdict(model.config),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: MiniGPT,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: str,
) -> TrainState:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    return TrainState(step=int(ckpt["step"]), best_val_loss=float(ckpt["best_val_loss"]))


def format_metrics(
    step: int,
    max_steps: int,
    loss: float,
    lr: float,
    tokens_per_sec: float,
    peak_mem_mb: float,
) -> str:
    ppl = math.exp(min(loss, 20.0))  # 截断避免未训练时 exp 溢出打印
    return (
        f"step {step:6d}/{max_steps}  loss {loss:.4f}  ppl {ppl:.1f}  "
        f"lr {lr:.2e}  {tokens_per_sec:.0f} tok/s  peak_mem {peak_mem_mb:.0f}MB"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mini-GPT 第一版训练")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    parser.add_argument("--grad-accum", type=int, default=GRAD_ACCUM)
    parser.add_argument("--eval-interval", type=int, default=EVAL_INTERVAL)
    parser.add_argument("--eval-iters", type=int, default=EVAL_ITERS)
    parser.add_argument("--ckpt-interval", type=int, default=CKPT_INTERVAL)
    parser.add_argument("--log-interval", type=int, default=LOG_INTERVAL)
    parser.add_argument("--resume", action="store_true", help="从 checkpoints/latest.pt 恢复")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cpu":
        raise SystemExit("第一版训练按 CUDA + FP16 设计，当前没有可用 GPU")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "train.log"

    config = GPTConfig(block_size=args.block_size)
    model = MiniGPT(config).to(args.device)
    optimizer = configure_optimizer(model, WEIGHT_DECAY, PEAK_LR, BETAS, args.device)
    scaler = GradScaler(args.device)
    state = TrainState(step=0, best_val_loss=float("inf"))

    latest_path = CKPT_DIR / "latest.pt"
    best_path = CKPT_DIR / "best.pt"
    if args.resume:
        if not latest_path.exists():
            raise SystemExit(f"找不到 {latest_path}，无法 --resume")
        state = load_checkpoint(latest_path, model, optimizer, scaler, args.device)
        print(f"已从 {latest_path} 恢复：step={state.step}  best_val_loss={state.best_val_loss:.4f}")

    train_tokens = load_tokens("train")
    val_tokens = load_tokens("val")
    splits = {"train": train_tokens, "val": val_tokens}

    tokens_per_step = args.batch_size * args.block_size * args.grad_accum
    n_params = model.num_parameters()
    header = (
        f"MiniGPT {n_params / 1e6:.1f}M  device={args.device}  "
        f"micro_batch={args.batch_size}  accum={args.grad_accum}  "
        f"effective_batch={args.batch_size * args.grad_accum}  "
        f"T={args.block_size}  max_steps={args.max_steps}"
    )
    print(header)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(header + "\n")

    model.train()
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    running_loss = 0.0
    t0 = time.perf_counter()

    while state.step < args.max_steps:
        lr = get_lr(state.step, args.warmup_steps, args.max_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # 每个 optimizer step 前做一次 eval：step 0 给出随机初始化基线
        if state.step % args.eval_interval == 0:
            losses = estimate_loss(
                model, splits, args.batch_size, args.block_size, args.eval_iters, args.device
            )
            val_ppl = math.exp(min(losses["val"], 20.0))
            train_ppl = math.exp(min(losses["train"], 20.0))
            line = (
                f"eval step {state.step:6d}  "
                f"train_loss {losses['train']:.4f}  train_ppl {train_ppl:.1f}  "
                f"val_loss {losses['val']:.4f}  val_ppl {val_ppl:.1f}"
            )
            print(line)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

            if losses["val"] < state.best_val_loss:
                state.best_val_loss = losses["val"]
                save_checkpoint(best_path, model, optimizer, scaler, state)
                print(f"  新的 best val_loss={state.best_val_loss:.4f} -> {best_path}")

        optimizer.zero_grad(set_to_none=True)
        micro_loss_sum = 0.0
        for _ in range(args.grad_accum):
            x, y = get_batch(train_tokens, args.batch_size, args.block_size, args.device)
            with autocast(args.device, dtype=torch.float16):
                _, loss = model(x, y)
            micro_loss_sum += float(loss.item())
            scaler.scale(loss / args.grad_accum).backward()

        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        state.step += 1
        step_loss = micro_loss_sum / args.grad_accum
        running_loss += step_loss

        if state.step % args.log_interval == 0:
            dt = time.perf_counter() - t0
            t0 = time.perf_counter()
            avg_loss = running_loss / args.log_interval
            running_loss = 0.0
            tok_s = tokens_per_step * args.log_interval / max(dt, 1e-9)
            peak_mem = (
                torch.cuda.max_memory_allocated() / (1024 * 1024) if args.device == "cuda" else 0.0
            )
            line = format_metrics(state.step, args.max_steps, avg_loss, lr, tok_s, peak_mem)
            print(line)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

        if state.step % args.ckpt_interval == 0 or state.step >= args.max_steps:
            save_checkpoint(latest_path, model, optimizer, scaler, state)
            print(f"  已写入 {latest_path} (step={state.step})")

    # 循环里的 eval 发生在 step 更新之前，退出时永远不会在 max_steps 上 eval。
    # 收尾补一次，短跑和正式训练结束都能看到最终 val ppl。
    losses = estimate_loss(
        model, splits, args.batch_size, args.block_size, args.eval_iters, args.device
    )
    val_ppl = math.exp(min(losses["val"], 20.0))
    train_ppl = math.exp(min(losses["train"], 20.0))
    line = (
        f"eval step {state.step:6d}  "
        f"train_loss {losses['train']:.4f}  train_ppl {train_ppl:.1f}  "
        f"val_loss {losses['val']:.4f}  val_ppl {val_ppl:.1f}"
    )
    print(line)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    if losses["val"] < state.best_val_loss:
        state.best_val_loss = losses["val"]
        save_checkpoint(best_path, model, optimizer, scaler, state)
        print(f"  新的 best val_loss={state.best_val_loss:.4f} -> {best_path}")


if __name__ == "__main__":
    main()
