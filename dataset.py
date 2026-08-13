"""从 token 流中采样训练 batch。

数据来源是 scripts/prepare_data.py 产出的两个文件：
    data/train.bin  224,512,862 tokens
    data/val.bin      4,765,918 tokens
它们是一条连续的 uint16 token 流，篇与篇之间用 <|endoftext|>(50256) 分隔，
不保存任何篇章边界索引——训练时直接在这条长流上随机截取片段即可。

用法（在仓库根目录运行自检）：
    python dataset.py
"""

import numpy as np
import torch

DATA_DIR = "data"
BLOCK_SIZE = 512    # context 长度 T，与 AGENTS.md 第一版规格一致
BATCH_SIZE = 16     # micro batch，训练时配合梯度累积 4 得到有效 batch 64


def load_tokens(split):
    """以 memmap 方式打开 token 流。

    449MB 的 train.bin 不整个读进内存：memmap 只在实际切片时按页从磁盘取，
    随机采样每次只碰 513 个 token，开销可以忽略。
    """
    return np.memmap(f"{DATA_DIR}/{split}.bin", dtype=np.uint16, mode="r")


def get_batch(tokens, batch_size=BATCH_SIZE, block_size=BLOCK_SIZE, device="cpu"):
    """随机采样一个训练 batch。

    返回 x, y，形状都是 [batch_size, block_size]，dtype 为 int64。

    关键点是 label 错位一格：截取长度 block_size + 1 的片段后，
        x = 片段[:-1]   模型在位置 t 看到的输入
        y = 片段[1:]    模型在位置 t 要预测的下一个 token
    于是 y[t] == x[t + 1]，即 logits[:, t, :] 对应的监督信号是原文里 t 的下一个词。
    这正是自回归语言模型的训练目标；配合因果 mask，一个长度 T 的样本能同时
    提供 T 个预测任务，而不是只预测末尾一个词。
    """
    # 起点上界要留出 block_size + 1 个 token，否则最后一个样本会越界
    max_start = len(tokens) - block_size - 1
    starts = torch.randint(max_start, (batch_size,))

    # memmap 切片是 uint16，PyTorch 不支持该 dtype，且 embedding 的索引必须是 int64
    x = torch.stack([
        torch.from_numpy(tokens[i:i + block_size].astype(np.int64)) for i in starts
    ])
    y = torch.stack([
        torch.from_numpy(tokens[i + 1:i + 1 + block_size].astype(np.int64)) for i in starts
    ])

    if device != "cpu":
        # non_blocking 配合 pin_memory 可以让拷贝与计算重叠，这里先保持简单
        x, y = x.to(device), y.to(device)
    return x, y


def _self_check():
    """形状与错位一格的自检，对应 AGENTS.md 的验收标准第 1 条。"""
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")

    for split in ["train", "val"]:
        tokens = load_tokens(split)
        print(f"{split}.bin: {len(tokens):,} tokens")

    tokens = load_tokens("train")
    x, y = get_batch(tokens)

    print(f"\nx: shape={tuple(x.shape)}, dtype={x.dtype}")
    print(f"y: shape={tuple(y.shape)}, dtype={y.dtype}")
    assert x.shape == (BATCH_SIZE, BLOCK_SIZE)
    assert y.shape == x.shape
    assert x.dtype == torch.int64

    # 错位一格：y 的前 T-1 个 token 应当等于 x 的后 T-1 个 token
    assert torch.equal(y[:, :-1], x[:, 1:]), "label 错位不正确"
    print("错位校验通过：y[:, :-1] == x[:, 1:]")

    # token id 必须落在词表范围内
    assert int(x.max()) < enc.n_vocab, "出现越界 token id"
    print(f"token id 范围: [{int(x.min())}, {int(x.max())}]，词表 {enc.n_vocab}")

    # 抽第 0 条样本的开头，肉眼确认输入与标签的对应关系
    print("\n第 0 条样本前 40 个 token 解码：")
    print(f"  x: {enc.decode(x[0, :40].tolist())!r}")
    print(f"  y: {enc.decode(y[0, :40].tolist())!r}")
    print(f"\nx[0][0]={x[0][0].item()} -> 要预测 y[0][0]={y[0][0].item()} "
          f"({enc.decode([x[0][0].item()])!r} -> {enc.decode([y[0][0].item()])!r})")


if __name__ == "__main__":
    _self_check()
