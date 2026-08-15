"""下载 TinyStories 并用 GPT-2 BPE 编码为 token 序列。

输出（相对路径，需在仓库根目录运行）：
    data/train.bin  uint16 token 流（train split 前 TRAIN_STORIES 篇）
    data/val.bin    uint16 token 流（完整 validation split）

用法：
    python scripts/prepare_data.py
    100 篇冒烟已通过，当前为全量配置；重跑会覆盖 data/train.bin 与 data/val.bin。
"""

from array import array
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
TRAIN_STORIES = 1_000_000   # 全量：train split 前 100 万篇（冒烟阶段用 100 已通过）
SHARD_SIZE = 1_000          # 每编码多少篇打印一次进度
EOT_ID = 50256              # <|endoftext|>，分隔两篇故事


def encode_split(dataset, limit, enc):
    """把数据集中的故事逐篇编码，拼成一条 uint16 token 流。

    limit 为 None 表示全部取用。篇尾手动补 EOT_ID，
    避免模型把上一篇结尾和下一篇开头当成连续文本。
    """
    buf = array("H")  # "H" = uint16；词表 50257 < 65536，2 字节刚好，比 int32 省一半磁盘
    n = len(dataset) if limit is None else min(limit, len(dataset))
    for i, story in enumerate(dataset):
        if i >= n:
            break
        # encode_ordinary：不把原文里的 "<|endoftext|>" 字符串当特殊 token，行为完全可控
        ids = enc.encode_ordinary(story["text"])
        ids.append(EOT_ID)
        buf.extend(ids)
        if (i + 1) % SHARD_SIZE == 0:
            print(f"  已编码 {i + 1}/{n} 篇，累计 {len(buf):,} tokens")
    return np.frombuffer(buf, dtype=np.uint16).copy()


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")
    assert enc.n_vocab == 50257, f"词表大小异常: {enc.n_vocab}"

    print(f"加载 TinyStories（首次会下载约 2GB 到 {DATA_DIR / 'hf_cache'}）...")
    ds = load_dataset("roneneldan/TinyStories", cache_dir=str(DATA_DIR / "hf_cache"))

    for split, limit, out_name in [
        ("train", TRAIN_STORIES, "train.bin"),
        ("validation", None, "val.bin"),
    ]:
        print(f"编码 {split} ...")
        arr = encode_split(ds[split], limit, enc)
        out_path = DATA_DIR / out_name
        arr.tofile(out_path)
        print(f"  已写入 {out_path}: {len(arr):,} tokens, {arr.nbytes / 1e6:.1f} MB")

    # 读回校验：确认写盘格式能被正确解析
    back = np.fromfile(DATA_DIR / "train.bin", dtype=np.uint16)
    print(f"读回校验，train.bin 前 80 个 token 解码：\n{enc.decode(back[:80].tolist())}")


if __name__ == "__main__":
    main()
