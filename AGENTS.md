# Mini-GPT Agent Notes

> 维护入口：本文件。后续改环境、改步骤、改进度，优先更新这里。  
> 对应计划：`D:\holiday_learning\暑期计划.md` 第八节「阶段 2：Mini-GPT」  
> 时间窗口：2026.8.12 - 2026.8.21（阶段第 2 天起）

## 项目概述

在 TinyStories 上从零训练一个能续写英文故事的 Decoder-only Transformer。重点是自己实现模型、因果 mask、训练、恢复训练和采样，而不是只调用训练框架。

本项目独立于 `model-learning/transformer`（英译中 Encoder-Decoder）。不要复用那个 SentencePiece 词表，也不要把依赖装进 `ML` 或 AutoDL 的 `transformer` 环境。

## 当前进度

记录日期：2026-08-13

- [x] 确认项目定位：从零预训练 Mini-GPT，不重训 tokenizer
- [x] 确认独立 conda 环境名：`Mini_GPT`（用户已开始创建）
- [x] 确认 Python 3.11 只是稳妥选择，不是 nanoGPT 官方硬性要求
- [x] 建立项目目录 `D:\holiday_learning\mini_GPT`
- [ ] 完成 `conda create -n Mini_GPT python=3.11 -y`，并确认解释器路径含 `envs\Mini_GPT`
- [ ] 在 `Mini_GPT` 环境安装 `tiktoken`
- [ ] tokenizer 冒烟：`Once upon a time` 编码/解码，`vocab_size == 50257`
- [x] 补充 `.gitignore`，忽略 `data/`、权重、缓存
- [x] 补充 `requirements.txt`（第一阶段仅 `tiktoken`）
- [ ] 下载 TinyStories（须先确认路径；默认本仓库 `data/`）
- [ ] 用 GPT-2 BPE 编码并打包训练 token 序列
- [ ] 验证 `[B, T] -> logits [B, T, 50257]`，以及 next-token label 错位一格
- [ ] 实现 6 层 Decoder-only 骨架并完成单 batch 前向
- [ ] 训练 / validation / checkpoint / 采样闭环

当前目录现状：

```text
mini_GPT/
  .vscode/settings.json   本机 IDE 配置，已 gitignore；conda 管理器已选，尚未绑定 Mini_GPT 解释器
  data/                   空目录，尚未下载数据
  main.py                 空文件
  AGENTS.md               本文件
  .gitignore              已忽略 data/、checkpoints/、experiments/、.vscode/、权重和缓存
  requirements.txt        第一阶段仅 tiktoken
```

## 固定要求

来自暑期计划，第一版不要改这些规格。

### 数据与分词

- 数据：[roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
- 训练集：官方 train split 的前 100 万篇故事
- 验证集：保留官方 validation split
- tokenizer：直接复用 GPT-2 BPE，[openai-community/gpt2](https://huggingface.co/openai-community/gpt2)
- 词表大小固定 50257，不重新训练 tokenizer
- 实现优先用 `tiktoken` 的 `gpt2` encoding，不要先装 `transformers`

### 模型（第一版）

```text
层数          6
d_model       512
n_head        8          每头 64 维
d_ff          2048
context T     512
dropout       0.1
归一化        Pre-LayerNorm
位置编码      可学习绝对位置编码
参数量        约 45M-55M
```

第二版再对照 RoPE、RMSNorm、SwiGLU、KV Cache。第一版不要提前做。

### 训练

```text
优化器        AdamW
weight_decay  0.1
峰值学习率    3e-4
调度          linear warmup + cosine decay
精度          FP16（3090 不原生支持 BF16）
micro batch   16
梯度累积      4
有效 batch    64
先跑          10,000 steps 验证数据和训练链路
再跑          30,000 steps 基线
checkpoint    每 1,000 steps 存 latest；validation 最优另存 best
记录          loss、perplexity、learning rate、tokens/sec、峰值显存
```

训练默认按 AutoDL 单卡 RTX 3090 24GB 设计。本地先做 tokenizer、数据编码和 CPU/小 batch 形状验证。

### 验收标准

1. 输入 `[B, T]` 对应 logits `[B, T, 50257]`，能解释 next-token label 为什么错位一格
2. training loss 下降，validation perplexity 可计算并被记录
3. 能从 latest 恢复模型、优化器和 step；best 可独立加载
4. 支持 greedy、temperature、top-k 三种生成
5. 至少 10 组固定 prompt 的生成样例，比较采样参数差异
6. 完成模型、数据、参数、曲线、显存、生成结果和失败样例的实验记录

## 当前步骤（做到 tokenizer 冒烟为止）

今天只做分词器，先不下载 TinyStories。

### 1. 建环境

用户已执行过 `conda create -n Mini_GPT`（未指定 Python）。不要再建成第二个空环境。

若创建尚未完成或环境是空的，用：

```powershell
conda create -n Mini_GPT python=3.11 -y
conda activate Mini_GPT
python -c "import sys; print(sys.executable); print(sys.version)"
```

若环境已存在但没有 Python：

```powershell
conda activate Mini_GPT
conda install python=3.11 -y
python -c "import sys; print(sys.executable); print(sys.version)"
```

通过标准：路径必须含 `envs\Mini_GPT`，版本为 3.11.x。

说明：nanoGPT README 没有规定 Python 版本；3.11 是本机稳妥选择，不是官方硬性要求。

### 2. 只装分词器

```powershell
pip install tiktoken
```

第一次调用 GPT-2 编码时，`tiktoken` 会拉取很小的编码表。这不是模型权重，也不是 TinyStories。

### 3. 冒烟验证

```powershell
python -c "import tiktoken; enc = tiktoken.get_encoding('gpt2'); ids = enc.encode('Once upon a time'); print('vocab_size:', enc.n_vocab); print('ids:', ids); print('decoded:', enc.decode(ids))"
```

通过标准：

```text
vocab_size: 50257
ids: 一串整数
decoded: Once upon a time
```

`vocab_size` 必须正好是 50257。对不上先停。

### 当前不要做

- 不要装 `torch`、`transformers`、`datasets`
- 不要下载 TinyStories
- 不要训练自己的 tokenizer
- 不要把依赖装进 `base`、`ML` 或 AutoDL `transformer`

## 建议目录

代码和配置提交；数据、权重、缓存不提交。

```text
mini_GPT/
  AGENTS.md
  .gitignore
  requirements.txt          第一阶段仅 tiktoken
  main.py
  config.py                 待建，集中超参数和路径
  model.py                  待建，Decoder-only
  dataset.py                待建
  train.py                  待建
  sample.py                 待建
  scripts/prepare_data.py   待建，下载并编码 TinyStories
  data/                     gitignore；原始文本与 token 序列
  checkpoints/              gitignore；latest.pt / best.pt
  experiments/              gitignore；曲线、生成样例、日志
  .vscode/                  gitignore；本机解释器与 IDE 配置
```

数据默认放本仓库 `data/`，与英译中项目「项目内 data + gitignore」的约定一致。下载前仍须口头确认一次。

## 环境与运行

| 项 | 当前约定 |
|---|---|
| conda 环境 | `Mini_GPT` |
| 建议解释器 | `D:\Anaconda\envs\Mini_GPT\python.exe`（创建完成后核对） |
| 第一阶段依赖 | 仅 `tiktoken` |
| 后续依赖 | `torch`、`numpy`、`datasets`；需要 GPT-2 权重对照时再装 `transformers` |
| 参考实现 | [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)，对照工程，不直接照抄 |
| 论文 | GPT-2 技术报告；TinyStories [arXiv:2305.07759](https://arxiv.org/abs/2305.07759) |

nanoGPT 文档中的安装示例是：

```text
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

本项目第一阶段只装 `tiktoken`，其余按步骤补，避免一次装齐后环境失控。

本仓库任务不要用 Codex 内置 Python（`C:\Users\14000\.cache\codex-runtimes\...`），也不要往 Anaconda `base` 装包。

## 关键约定

- 详细中文教学注释，尤其写清 token ID、shape、device、因果 mask、label 错位
- 数学公式用纯文本或 Unicode，不在对话和本文件里写 LaTeX
- 源码、配置、脚本可以提交；`data/`、权重、HF 缓存、日志、`.vscode/` 不提交
- 改代码只做当前步骤，不提前搭完整训练框架
- 与 `model-learning/transformer` 共用工作区，但环境和词表完全分开
- AutoDL 上若再新建环境，不要复用旧的 `transformer` 环境；注意 conda 镜像和解释器是否一致

## 当前风险或缺口

- `conda create -n Mini_GPT` 可能建成空环境；必须确认 `python` 指向 `envs\Mini_GPT`
- `.vscode/settings.json` 已 gitignore；conda 管理器已选，尚未绑定 `Mini_GPT` 解释器
- TinyStories 体积较大，下载前要确认落在 `mini_GPT/data/`（`.gitignore` 已忽略该目录）
- 完整训练按 3090 设计；本机只做 tokenizer、数据编码和形状验证
- 旧 AutoDL `transformer` 环境出现过「conda 已列出包但 Python 导不进」；Mini-GPT 用 GPT-2 BPE，不依赖 SentencePiece，但装新依赖时仍要核对 `sys.executable`

## 下一步

tokenizer 冒烟通过后，按这个顺序：

1. 把 VS Code / Cursor 解释器指到 `Mini_GPT`
2. 确认 TinyStories 下载路径后，写 `scripts/prepare_data.py`
3. 编码前 100 万篇 train + 官方 validation，检查样本数、token 总量、长度分布
4. 写 Dataset，验证 `[B, 512]` 和 next-token label 错位
5. 搭模型骨架，跑通单 batch 前向：`[B, T, 50257]`
