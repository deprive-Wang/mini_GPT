"""第一版自回归采样：greedy、temperature、top-k。

从 train.py 产出的 checkpoint 读取 MiniGPT 权重和 GPTConfig，使用 GPT-2 BPE
把英文 prompt 编码成 token ID，再逐 token 续写。每次模型最多接收 block_size
个 token；生成超过 512 个 token 时，自动只保留最近的上下文。

当前 checkpoint 是本机 22 步冒烟权重，文本质量差是预期；本文件的验收目标是
三种采样路径能正确加载权重、生成指定 token 数，并成功解码。

用法（仓库根目录，解释器必须是 Mini_GPT 环境）：
    python sample.py --mode greedy
    python sample.py --mode temperature --temperature 0.8 --seed 42
    python sample.py --mode top-k --temperature 0.8 --top-k 40 --seed 42
    python sample.py --checkpoint checkpoints/best.pt --prompt "Once upon a time"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import tiktoken
import torch
import torch.nn.functional as F

from model import GPTConfig, MiniGPT

DEFAULT_PROMPT = "Once upon a time"
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = PROJECT_DIR / "checkpoints" / "latest.pt"


def load_model(checkpoint_path: Path, device: str) -> MiniGPT:
    """从 train.py 保存的 checkpoint 恢复模型并切到 eval 模式。"""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"找不到 checkpoint：{checkpoint_path}。请先运行 train.py 产出 latest.pt。"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "config" not in checkpoint or "model" not in checkpoint:
        raise ValueError(f"{checkpoint_path} 不是 train.py 产出的 Mini-GPT checkpoint")

    config = GPTConfig(**checkpoint["config"])
    model = MiniGPT(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def sample_next_token(
    logits: torch.Tensor,
    mode: str,
    temperature: float,
    top_k: int | None,
) -> torch.Tensor:
    """从最后位置的 logits [1, vocab_size] 采样下一个 token ID [1, 1]。"""
    if mode == "greedy":
        return torch.argmax(logits, dim=-1, keepdim=True)

    if temperature <= 0:
        raise ValueError("temperature 必须大于 0")

    scaled_logits = logits / temperature
    if mode == "top-k":
        if top_k is None or top_k <= 0:
            raise ValueError("top-k 模式必须传入大于 0 的 --top-k")
        if top_k > scaled_logits.size(-1):
            raise ValueError(f"top_k 不能超过词表大小 {scaled_logits.size(-1)}")

        # 低于第 k 大分数的词设成 -inf，softmax 后概率正好为 0。
        threshold = torch.topk(scaled_logits, top_k, dim=-1).values[:, -1:]
        scaled_logits = scaled_logits.masked_fill(scaled_logits < threshold, float("-inf"))

    probs = F.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate(
    model: MiniGPT,
    token_ids: list[int],
    max_new_tokens: int,
    mode: str,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> list[int]:
    """基于 token_ids 续写 max_new_tokens 个 token，返回 prompt 加完整生成序列。"""
    if not token_ids:
        raise ValueError("prompt 编码后不能为空")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens 不能小于 0")
    if mode not in {"greedy", "temperature", "top-k"}:
        raise ValueError(f"不支持的采样模式：{mode}")

    device = next(model.parameters()).device
    generated = torch.tensor([token_ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        # Learned positional embedding 只定义到 block_size，因此上下文必须截断。
        context = generated[:, -model.config.block_size:]
        logits, _ = model(context)
        next_logits = logits[:, -1, :]
        next_token = sample_next_token(next_logits, mode, temperature, top_k)
        generated = torch.cat((generated, next_token), dim=1)

    return generated[0].tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mini-GPT 第一版采样")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument(
        "--mode",
        choices=["greedy", "temperature", "top-k"],
        default="greedy",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.prompt.strip():
        raise SystemExit("--prompt 不能为空")
    if args.max_new_tokens < 0:
        raise SystemExit("--max-new-tokens 不能小于 0")
    if args.mode != "greedy" and args.temperature <= 0:
        raise SystemExit("temperature 必须大于 0")
    if args.mode == "top-k" and args.top_k is None:
        raise SystemExit("top-k 模式必须传入 --top-k")

    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    encoding = tiktoken.get_encoding("gpt2")
    prompt_ids = encoding.encode(args.prompt)
    model = load_model(args.checkpoint, args.device)
    output_ids = generate(
        model=model,
        token_ids=prompt_ids,
        max_new_tokens=args.max_new_tokens,
        mode=args.mode,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    print(f"checkpoint: {args.checkpoint}")
    print(f"mode: {args.mode}")
    if args.mode != "greedy":
        print(f"temperature: {args.temperature}")
    if args.mode == "top-k":
        print(f"top_k: {args.top_k}")
    print(f"prompt tokens: {len(prompt_ids)}")
    print(f"generated tokens: {len(output_ids) - len(prompt_ids)}")
    print("\n--- generated text ---")
    print(encoding.decode(output_ids))


if __name__ == "__main__":
    main()
