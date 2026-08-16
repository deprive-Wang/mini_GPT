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
    python train.py --max-steps 20 --eval-interval 10 --eval-iters 4 --batch-size 4 --warmup-steps 2
    python train.py
    python train.py --max-steps 30000
    python train.py --resume

TensorBoard 事件文件写到 experiments/tb/<run-name>（与 train.log 并列，已 gitignore）。
查看曲线（仓库根目录，解释器必须是 Mini_GPT 环境）：
    python -m tensorboard --logdir experiments/tb

本机 RTX 3070 Laptop 8GB 上 micro batch 16 的反向会 OOM，验证链路用 --batch-size 4。
完整训练仍按 3090 的默认 micro batch 16。
"""

from __future__ import annotations

import argparse
import atexit
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from dataset import BATCH_SIZE, BLOCK_SIZE, get_batch, load_tokens
from model import GPTConfig, MiniGPT

PEAK_LR = 3e-4
MIN_LR = 3e-5          # cosine 末端，约为峰值的 10%
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)    # GPT 系常用；AdamW 默认 (0.9, 0.999) 对 LM 偏慢
GRAD_CLIP = 1.0
GRAD_ACCUM = 4
EPOCHS = 1.0            # 未显式指定 max_steps 时的备用目标
MAX_STEPS = 100_000     # 云端正式训练默认 10 万 optimizer steps
WARMUP_STEPS = 200
EVAL_INTERVAL = 1_000
EVAL_ITERS = 20
CKPT_INTERVAL = 1_000
LOG_INTERVAL = 10

PROJECT_DIR = Path(__file__).resolve().parent
CKPT_DIR = PROJECT_DIR / "checkpoints"
LOG_DIR = PROJECT_DIR / "experiments"
TB_DIR = LOG_DIR / "tb"


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
    device: torch.device,
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

    fused = device.type == "cuda"
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
    余弦退火策略--针对学习率衰减的一种策略，学习率先增大后减小。
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
    device: torch.device,
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
            with autocast(device.type, dtype=torch.float16):
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


def format_metrics_header() -> str:
    """与 format_metrics 列宽对齐的表头。"""
    return (
        f"| {'step':^13} | {'loss':^7} | {'ppl':^8} | {'lr':^8} | "
        f"{'tok/s':^6} | {'peak_mem':^8} |"
    )


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
        f"| {f'{step}/{max_steps}':>13} | {loss:7.4f} | {ppl:8.1f} | {lr:8.2e} | "
        f"{tokens_per_sec:6.0f} | {f'{peak_mem_mb:.0f}MB':>8} |"
    )


def write_tb_scalars(writer: SummaryWriter, step: int, scalars: dict[str, float]) -> None:
    """同一 step 的标量写进 TensorBoard。名字用 train/ 与 eval/ 分组，便于对比曲线。"""
    for name, value in scalars.items():
        writer.add_scalar(name, value, step)
    writer.flush()


def validate_args(args: argparse.Namespace) -> torch.device:
    """在启动模型和读数据前校验训练参数，避免循环内才暴露除零或 CUDA 错误。"""
    positive_names = (
        "batch_size",
        "block_size",
        "grad_accum",
        "eval_interval",
        "eval_iters",
        "ckpt_interval",
        "log_interval",
    )
    for name in positive_names:
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} 必须大于 0")
    if args.epochs <= 0:
        raise SystemExit("--epochs 必须大于 0")
    if args.max_steps is not None and args.max_steps <= 0:
        raise SystemExit("--max-steps 必须大于 0")
    if args.warmup_steps < 0:
        raise SystemExit("--warmup-steps 不能小于 0")

    try:
        device = torch.device(args.device)
    except RuntimeError as error:
        raise SystemExit(f"无效的 --device={args.device}: {error}") from error
    if device.type != "cuda":
        raise SystemExit("第一版训练按 CUDA + FP16 设计，请使用 --device cuda 或 --device cuda:0")
    if not torch.cuda.is_available():
        raise SystemExit("未检测到可用 CUDA。请确认云镜像的 PyTorch 能执行 torch.cuda.is_available()。")
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise SystemExit(
            f"--device {device} 不存在；当前只检测到 {torch.cuda.device_count()} 张 CUDA GPU。"
        )
    return device


def build_run_dir(log_dir: Path, run_name: str | None) -> Path:
    """每次训练使用独立 TensorBoard 事件目录，避免曲线互相混写。"""
    name = run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = log_dir / name
    if run_dir.exists():
        raise SystemExit(f"TensorBoard 运行目录已存在：{run_dir}。请指定新的 --run-name。")
    return run_dir


def resolve_max_steps(args: argparse.Namespace, train_token_count: int) -> tuple[int, float, float]:
    """把 token 数换算成 optimizer steps；随机有放回采样下的 epoch 是近似值。"""
    tokens_per_step = args.batch_size * args.block_size * args.grad_accum
    steps_per_epoch = train_token_count / tokens_per_step
    max_steps = args.max_steps or math.ceil(args.epochs * steps_per_epoch)
    effective_epochs = max_steps / steps_per_epoch
    if args.warmup_steps > max_steps:
        raise SystemExit("--warmup-steps 不能超过最终 max_steps")
    return max_steps, steps_per_epoch, effective_epochs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mini-GPT 第一版训练")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=MAX_STEPS,
        help="最大 optimizer steps；不传时根据 --epochs 和 train.bin 自动计算",
    )
    parser.add_argument("--epochs", type=float, default=EPOCHS)
    parser.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    parser.add_argument("--grad-accum", type=int, default=GRAD_ACCUM)
    parser.add_argument("--eval-interval", type=int, default=EVAL_INTERVAL)
    parser.add_argument("--eval-iters", type=int, default=EVAL_ITERS)
    parser.add_argument("--ckpt-interval", type=int, default=CKPT_INTERVAL)
    parser.add_argument("--log-interval", type=int, default=LOG_INTERVAL)
    parser.add_argument("--log-dir", type=Path, default=TB_DIR)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--resume", action="store_true", help="从 checkpoints/latest.pt 恢复")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = validate_args(args)
    run_dir = build_run_dir(args.log_dir, args.run_name)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "train.log"
    writer = SummaryWriter(log_dir=str(run_dir))
    atexit.register(writer.close)

    config = GPTConfig(block_size=args.block_size)
    model = MiniGPT(config).to(device)
    optimizer = configure_optimizer(model, WEIGHT_DECAY, PEAK_LR, BETAS, device)
    scaler = GradScaler(device.type)
    state = TrainState(step=0, best_val_loss=float("inf"))

    latest_path = CKPT_DIR / "latest.pt"
    best_path = CKPT_DIR / "best.pt"
    if args.resume:
        if not latest_path.exists():
            raise SystemExit(f"找不到 {latest_path}，无法 --resume")
        state = load_checkpoint(latest_path, model, optimizer, scaler, device)
        print(f"已从 {latest_path} 恢复：step={state.step}  best_val_loss={state.best_val_loss:.4f}")

    train_tokens = load_tokens("train")
    val_tokens = load_tokens("val")
    splits = {"train": train_tokens, "val": val_tokens}

    args.max_steps, steps_per_epoch, effective_epochs = resolve_max_steps(
        args, len(train_tokens)
    )
    tokens_per_step = args.batch_size * args.block_size * args.grad_accum
    n_params = model.num_parameters()
    header = (
        f"MiniGPT {n_params / 1e6:.1f}M  device={device}  "
        f"micro_batch={args.batch_size}  accum={args.grad_accum}  "
        f"effective_batch={args.batch_size * args.grad_accum}  "
        f"T={args.block_size}  max_steps={args.max_steps}  "
        f"steps_per_epoch={steps_per_epoch:.0f}  epochs={effective_epochs:.2f}"
    )
    print(header)
    print(f"tokens_per_step: {tokens_per_step:,}")
    print(f"TensorBoard: {run_dir}")
    metrics_header = format_metrics_header()
    print(metrics_header)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(header + "\n")
        f.write(f"tokens_per_step: {tokens_per_step:,}\n")
        f.write(f"TensorBoard: {run_dir}\n")
        f.write(metrics_header + "\n")

    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    running_loss = 0.0
    t0 = time.perf_counter()

    while state.step < args.max_steps:
        lr = get_lr(state.step, args.warmup_steps, args.max_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # 每个 optimizer step 前做一次 eval：step 0 给出随机初始化基线
        if state.step % args.eval_interval == 0:
            losses = estimate_loss(
                model, splits, args.batch_size, args.block_size, args.eval_iters, device
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

            write_tb_scalars(
                writer,
                state.step,
                {
                    "eval/train_loss": losses["train"],
                    "eval/train_ppl": train_ppl,
                    "eval/val_loss": losses["val"],
                    "eval/val_ppl": val_ppl,
                    "eval/best_val_loss": state.best_val_loss,
                },
            )

        optimizer.zero_grad(set_to_none=True)
        micro_loss_sum = 0.0
        for _ in range(args.grad_accum):
            x, y = get_batch(train_tokens, args.batch_size, args.block_size, device)
            with autocast(device.type, dtype=torch.float16):
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
                torch.cuda.max_memory_allocated(device) / (1024 * 1024)
                if device.type == "cuda"
                else 0.0
            )
            line = format_metrics(state.step, args.max_steps, avg_loss, lr, tok_s, peak_mem)
            print(line)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            write_tb_scalars(
                writer,
                state.step,
                {
                    "train/loss": avg_loss,
                    "train/ppl": math.exp(min(avg_loss, 20.0)),
                    "train/lr": lr,
                    "train/tokens_per_sec": tok_s,
                    "train/peak_mem_mb": peak_mem,
                },
            )

        if state.step % args.ckpt_interval == 0 or state.step >= args.max_steps:
            save_checkpoint(latest_path, model, optimizer, scaler, state)
            print(f"  已写入 {latest_path} (step={state.step})")

    # 循环里的 eval 发生在 step 更新之前，退出时永远不会在 max_steps 上 eval。
    # 收尾补一次，短跑和正式训练结束都能看到最终 val ppl。
    losses = estimate_loss(
        model, splits, args.batch_size, args.block_size, args.eval_iters, device
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
    write_tb_scalars(
        writer,
        state.step,
        {
            "eval/train_loss": losses["train"],
            "eval/train_ppl": train_ppl,
            "eval/val_loss": losses["val"],
            "eval/val_ppl": val_ppl,
            "eval/best_val_loss": state.best_val_loss,
        },
    )
    writer.close()
    atexit.unregister(writer.close)


if __name__ == "__main__":
    main()
