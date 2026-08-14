"""6 层 Decoder-only Transformer（第一版 Mini-GPT）。

规格来自 AGENTS.md，第一版不要改：
    层数 6 / d_model 512 / n_head 8（每头 64 维）/ d_ff 2048
    context T 512 / dropout 0.1 / Pre-LayerNorm / 可学习绝对位置编码
    词表 50257（GPT-2 BPE，不重训 tokenizer）

输入 idx 形状 [B, T]，输出 logits 形状 [B, T, 50257]。
位置 t 的 logits 预测的是位置 t 的下一个 token，对应 dataset.get_batch
返回的 y[:, t]；训练时再配合因果 mask，一个长度 T 的样本同时提供 T 个预测。

用法（在仓库根目录运行自检）：
    python model.py
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    """第一版超参数。训练脚本以后直接复用这份，不要另写一套。"""

    vocab_size: int = 50257
    block_size: int = 512
    n_layer: int = 6
    d_model: int = 512
    n_head: int = 8
    d_ff: int = 2048
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    """多头因果自注意力。

    因果 mask 的含义：位置 t 只能看到 0..t，不能看到未来。
    没有这层 mask，训练时模型会直接抄 y 里已经出现的下一个词，loss 会假下降，
    推理时却没有未来可抄，对不上。
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        if config.d_model % config.n_head != 0:
            raise ValueError("d_model 必须能被 n_head 整除")

        self.n_head = config.n_head
        self.head_dim = config.d_model // config.n_head  # 512 / 8 = 64
        self.d_model = config.d_model

        # 一次投影出 Q、K、V，再在最后一维拆成三份，比三个 Linear 少一次 kernel launch
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

        # [T, T] 上三角（不含对角线）为 True，表示「不允许看」。
        # register_buffer：随 .to(device) 一起走，但不算模型参数、不参与优化。
        # persistent=False：存 checkpoint 时不必把这张静态表写进去。
        causal = torch.triu(
            torch.ones(config.block_size, config.block_size, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("causal_mask", causal, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]，C = d_model
        batch_size, seq_len, _ = x.shape

        qkv = self.qkv(x)  # [B, T, 3C]
        q, k, v = qkv.split(self.d_model, dim=-1)  # 各 [B, T, C]

        # 把 C 拆成 n_head 个头，再把「头」维挪到 batch 后面，方便做 [T, T] 注意力
        # [B, T, C] -> [B, T, n_head, head_dim] -> [B, n_head, T, head_dim]
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        # scores: [B, n_head, T, T]
        # 除以 sqrt(head_dim)，避免点积随维度变大、softmax 退化成 one-hot
        scale = self.head_dim ** 0.5
        scores = (q @ k.transpose(-2, -1)) / scale
        scores = scores.masked_fill(self.causal_mask[:seq_len, :seq_len], float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        # [B, n_head, T, head_dim] -> [B, T, n_head, head_dim] -> [B, T, C]
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.resid_drop(self.out_proj(out))


class FeedForward(nn.Module):
    """位置级 MLP：C -> d_ff -> C。每个 token 独立，不混合序列维。"""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Pre-LayerNorm Decoder block。

    Pre-LN：先归一化再进子层，残差在子层外面相加。
        x = x + Attn(LN(x))
        x = x + FFN(LN(x))
    相对 Post-LN（GPT-2 原文）更深时更稳；第一版按计划用 Pre-LN，不要改成 Post-LN。
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x


class MiniGPT(nn.Module):
    """Decoder-only LM。token 嵌入与 lm_head 权重共享，参数量落在约 45M。"""

    def __init__(self, config: GPTConfig | None = None) -> None:
        super().__init__()
        self.config = config or GPTConfig()
        cfg = self.config

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        # 可学习绝对位置编码：每个位置 0..T-1 一个向量，和 token 向量相加。
        # 第一版不用 RoPE；超出 block_size 的位置没有嵌入，forward 会直接报错。
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        # bias=False 才能和 Embedding.weight 形状完全一致，做权重共享
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # 残差投影（attn / ffn 的输出矩阵）再缩小一点，避免 L 层残差叠加把激活打爆
        residual_std = 0.02 / (2 * cfg.n_layer) ** 0.5
        for name, param in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("net.2.weight"):
                nn.init.normal_(param, mean=0.0, std=residual_std)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self) -> int:
        """可训练参数个数。权重共享后 lm_head 不再另计一份。"""
        return sum(param.numel() for param in self.parameters())

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """idx / targets: [B, T] int64。

        logits[b, t, :] 是在只看见 idx[b, :t+1] 的条件下，对下一个 token 的分数。
        若传入 targets（即 dataset 的 y），再返回 token 级交叉熵，方便自检和训练。
        """
        _, seq_len = idx.shape
        if seq_len > self.config.block_size:
            raise ValueError(
                f"序列长度 {seq_len} 超过 block_size {self.config.block_size}"
            )

        # positions: [T]，值是 0, 1, ..., T-1，device 必须和 idx 一致
        positions = torch.arange(seq_len, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(positions))  # [B, T, C]

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # [B, T, vocab_size]

        loss = None
        if targets is not None:
            # flatten 成 [B*T, vocab] vs [B*T]，每个位置一个分类任务
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss


def _self_check() -> None:
    """形状、因果、参数量、单 batch 前向。对应 AGENTS.md 验收标准第 1 条。"""
    config = GPTConfig()
    model = MiniGPT(config)
    model.eval()

    n_params = model.num_parameters()
    print(f"参数量: {n_params:,} ({n_params / 1e6:.1f}M)")
    # 权重共享后约 44.9M，落在「约 45M-55M」；不共享会到 ~70M，超出第一版上限
    assert 40_000_000 < n_params < 55_000_000, f"参数量 {n_params} 不在预期区间"

    # --- 小序列：形状 + 因果（改最后一个 token，前面的 logits 必须不变）---
    small_b, small_t = 2, 32
    idx = torch.randint(0, config.vocab_size, (small_b, small_t))
    with torch.no_grad():
        logits, loss = model(idx)
    assert loss is None
    assert logits.shape == (small_b, small_t, config.vocab_size), logits.shape
    print(f"小序列: idx {tuple(idx.shape)} -> logits {tuple(logits.shape)}")

    idx_changed = idx.clone()
    idx_changed[:, -1] = (idx_changed[:, -1] + 1) % config.vocab_size
    with torch.no_grad():
        logits_changed, _ = model(idx_changed)
    assert torch.allclose(logits[:, :-1], logits_changed[:, :-1], atol=1e-5)
    assert not torch.allclose(logits[:, -1], logits_changed[:, -1], atol=1e-4)
    print("因果校验通过：改最后一个 token 只影响最后一位 logits")

    # --- 单 batch 前向：尽量贴近训练时的 [16, 512] ---
    # 本机 8GB 上 logits [16, 512, 50257] 约 1.65GB，eval + no_grad 通常放得下；
    # 放不下就退到 [2, 512]，形状结论不变。完整训练仍按 3090 设计。
    from dataset import BATCH_SIZE, BLOCK_SIZE, get_batch, load_tokens

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokens = load_tokens("train")
    batch_size = BATCH_SIZE
    x, y = get_batch(tokens, batch_size=batch_size, block_size=BLOCK_SIZE, device=device)
    model = model.to(device)

    try:
        with torch.no_grad():
            logits, loss = model(x, y)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        batch_size = 2
        x, y = get_batch(
            tokens, batch_size=batch_size, block_size=BLOCK_SIZE, device=device
        )
        with torch.no_grad():
            logits, loss = model(x, y)
        print(f"显存不足，已退到 batch_size={batch_size}")

    assert x.shape == (batch_size, BLOCK_SIZE)
    assert logits.shape == (batch_size, BLOCK_SIZE, config.vocab_size), logits.shape
    assert loss is not None and torch.isfinite(loss)
    print(
        f"单 batch: x {tuple(x.shape)} -> logits {tuple(logits.shape)}, "
        f"device={device}, loss={loss.item():.4f}"
    )
    # 未训练模型的 CE 应接近 ln(vocab_size) ≈ 10.82
    print(f"随机初始化参考：ln(vocab_size) = {torch.log(torch.tensor(config.vocab_size)).item():.2f}")


if __name__ == "__main__":
    _self_check()
