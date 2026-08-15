# Mini-GPT Agent Notes

> 维护入口：本文件。后续改环境、改步骤、改进度，优先更新这里。  
> 对应计划：`D:\holiday_learning\暑期计划.md` 第八节「阶段 2：Mini-GPT」  
> 时间窗口：2026.8.12 - 2026.8.21（阶段第 2 天起）

## 项目概述

在 TinyStories 上从零训练一个能续写英文故事的 Decoder-only Transformer。重点是自己实现模型、因果 mask、训练、恢复训练和采样，而不是只调用训练框架。

本项目独立于 `model-learning/transformer`（英译中 Encoder-Decoder）。不要复用那个 SentencePiece 词表，也不要把依赖装进 `ML` 或 AutoDL 的 `transformer` 环境。

## 当前进度

记录日期：2026-08-15

- [x] 确认项目定位：从零预训练 Mini-GPT，不重训 tokenizer
- [x] 确认独立 conda 环境名：`Mini_GPT`（用户已开始创建）
- [x] 确认 Python 3.11 只是稳妥选择，不是 nanoGPT 官方硬性要求
- [x] 建立项目目录 `D:\holiday_learning\mini_GPT`
- [x] 完成 `conda create -n Mini_GPT python=3.11 -y`，并确认解释器路径含 `envs\Mini_GPT`
- [x] 在 `Mini_GPT` 环境安装 `tiktoken`
- [x] tokenizer 冒烟：`Once upon a time` 编码/解码，`vocab_size == 50257`
- [x] 补充 `.gitignore`，忽略 `data/`、权重、缓存
- [x] 补充 `requirements.txt`（第一阶段仅 `tiktoken`）
- [x] 安装 `datasets`（顺带引入 `numpy`、`pandas`、`pyarrow`）
- [x] 写 `scripts/prepare_data.py`（相对路径，从仓库根目录运行；已过 `py_compile`）
- [x] 下载 TinyStories（原始 parquet 落在 `HF_HOME=D:\AI_model\huggingface`，arrow 缓存在 `data/hf_cache`）
- [x] 全量编码：`data/train.bin` 224,512,862 tokens / 100 万篇；`data/val.bin` 4,765,918 tokens / 21,990 篇
- [x] 安装 CUDA 版 PyTorch 2.11.0+cu128；本机 RTX 3070 Laptop 8GB，`torch.cuda.is_available()` 为 True
- [x] 写 `dataset.py`，自检通过：`[16, 512]` int64，`y[:, :-1] == x[:, 1:]`，token id 不越界
- [x] 写 `model.py`：6 层 Pre-LN Decoder-only，可学习位置编码，手写因果 mask，token/lm_head 权重共享
- [x] 单 batch 前向自检通过：`[16, 512] -> [16, 512, 50257]`，cuda，loss=10.91（接近 ln(50257)≈10.82），参数量 44.9M
- [x] 写 `train.py`：AdamW、weight_decay 0.1、warmup + cosine、FP16、梯度累积 4、latest/best checkpoint
- [x] 本机小步数验证：micro batch 16 反向 OOM；`--batch-size 4` 下 20 步 loss 10.95→9.81，val ppl 可算；`--resume` 从 latest 恢复后再跑 2 步
- [x] 写 `sample.py`：greedy / temperature / top-k；用 22 步 checkpoint 三条路径都已跑通（生成质量差属预期）
- [x] 训练指标接到 TensorBoard：`experiments/tb`；Mini_GPT 环境已装 `tensorboard`

当前目录现状：

```text
mini_GPT/
  .vscode/settings.json     本机 IDE 配置，已 gitignore
  data/                     已 gitignore；train.bin 449MB、val.bin 9.5MB、hf_cache 1.8GB
  scripts/prepare_data.py   下载并编码 TinyStories，从仓库根目录运行
  dataset.py                memmap 采样 [16, 512]，含错位自检
  model.py                  6 层 Pre-LN Decoder-only；python model.py 自检通过
  train.py                  AdamW + warmup/cosine + FP16 + 梯度累积 + checkpoint + TensorBoard
  sample.py                 greedy / temperature / top-k；python sample.py 可验证
  main.py                   空文件
  AGENTS.md                 本文件
  .gitignore                已忽略 data/、checkpoints/、experiments/、.vscode/、权重和缓存
  requirements.txt          tiktoken / datasets / numpy / torch / tensorboard
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

训练默认按 AutoDL 单卡 RTX 3090 24GB、micro batch 16 设计。本机 RTX 3070 Laptop 8GB 能跑 `[16, 512]` eval 前向，但 FP16 反向会 OOM；本机验证用 `--batch-size 4`。

### 验收标准

1. 输入 `[B, T]` 对应 logits `[B, T, 50257]`，能解释 next-token label 为什么错位一格
2. training loss 下降，validation perplexity 可计算并被记录
3. 能从 latest 恢复模型、优化器和 step；best 可独立加载
4. 支持 greedy、temperature、top-k 三种生成
5. 至少 10 组固定 prompt 的生成样例，比较采样参数差异
6. 完成模型、数据、参数、曲线、显存、生成结果和失败样例的实验记录

## 当前步骤（第一版闭环完成）

数据、模型、训练恢复与采样三条路径均已在本机冒烟通过。训练指标会同时写 `experiments/train.log` 和 `experiments/tb`。当前 `checkpoints/` 只有 22 步权重，生成质量差属预期；下一阶段在 AutoDL RTX 3090 上按默认 micro batch 16 跑正式训练，用 TensorBoard 看 loss / val ppl / lr / tokens/sec / 峰值显存，不提前上第二版结构。

### 当前不要做

- 不要上 RoPE / RMSNorm / SwiGLU / KV Cache
- 不要把依赖装进 `base`、`ML` 或 AutoDL `transformer`
- 不要用 Codex 内置 Python

## 建议目录

代码和配置提交；数据、权重、缓存不提交。

```text
mini_GPT/
  AGENTS.md
  .gitignore
  requirements.txt          tiktoken / datasets / numpy / torch / tensorboard
  main.py                   空文件
  config.py                 待建；第一版超参数目前写在 model.GPTConfig
  model.py                  已建，Decoder-only；python model.py 自检通过
  dataset.py                已建，memmap 采样 [16, 512]
  train.py                  已建；本机验证用 --batch-size 4，默认仍是 16；指标写 experiments/tb
  sample.py                 已建；greedy / temperature / top-k 路径已验证
  scripts/prepare_data.py   已建，下载并编码 TinyStories
  data/                     gitignore；原始文本与 token 序列
  checkpoints/              gitignore；latest.pt / best.pt
  experiments/              gitignore；train.log、TensorBoard 事件文件 tb/、生成样例
  .vscode/                  gitignore；本机解释器与 IDE 配置
```

数据默认放本仓库 `data/`，与英译中项目「项目内 data + gitignore」的约定一致。下载前仍须口头确认一次。

## 环境与运行

| 项 | 当前约定 |
|---|---|
| conda 环境 | `Mini_GPT` |
| 建议解释器 | `D:\Anaconda\envs\Mini_GPT\python.exe` |
| 当前依赖 | `tiktoken`、`datasets`、`numpy`、`torch`（2.11.0+cu128）、`tensorboard` |
| 尚未安装 | 需要 GPT-2 权重对照时再装 `transformers`；需要交互进度条时再装 `tqdm`。训练曲线用 TensorBoard，不装 `wandb` |
| 参考实现 | [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)，对照工程，不直接照抄 |
| 论文 | GPT-2 技术报告；TinyStories [arXiv:2305.07759](https://arxiv.org/abs/2305.07759) |

nanoGPT 文档中的安装示例一次装齐；本项目已按步骤补到 `tiktoken` / `datasets` / `numpy` / `torch` / `tensorboard`。`transformers`、`tqdm` 仍按需再装。

查看训练曲线（仓库根目录，解释器必须是 Mini_GPT 环境）：

```text
python -m tensorboard --logdir experiments/tb
```

本仓库任务不要用 Codex 内置 Python（`C:\Users\14000\.cache\codex-runtimes\...`），也不要往 Anaconda `base` 装包。

## 关键约定

- 详细中文教学注释，尤其写清 token ID、shape、device、因果 mask、label 错位
- 数学公式用纯文本或 Unicode，不在对话和本文件里写 LaTeX
- 源码、配置、脚本可以提交；`data/`、权重、HF 缓存、日志、`.vscode/` 不提交
- 改代码只做当前步骤，不提前搭完整训练框架
- 与 `model-learning/transformer` 共用工作区，但环境和词表完全分开
- AutoDL 上若再新建环境，不要复用旧的 `transformer` 环境；注意 conda 镜像和解释器是否一致

## 当前风险或缺口

- 完整训练按 AutoDL 单卡 RTX 3090 24GB、默认 micro batch 16 设计。本机 RTX 3070 Laptop 8GB 能跑 `[16, 512]` eval 前向，但 FP16 反向会 OOM；本机验证链路用 `--batch-size 4`
- 当前 `checkpoints/` 是 22 步冒烟权重，只能用来测恢复和采样接口，不能当生成质量基线
- 第一版超参数目前写在 `model.GPTConfig`，训练超参写在 `train.py`，尚未拆到独立 `config.py`
- 旧 AutoDL `transformer` 环境出现过「conda 已列出包但 Python 导不进」；Mini-GPT 用 GPT-2 BPE，不依赖 SentencePiece，但装新依赖时仍要核对 `sys.executable`

## 下一步

第一版功能闭环已完成，下一步转到 AutoDL RTX 3090：

1. 用默认 micro batch 16、梯度累积 4 先跑 10,000 steps；用 TensorBoard（`experiments/tb`）看 loss、val perplexity、lr、tokens/sec 和峰值显存
2. 确认链路稳定后，再跑 30,000 steps 基线；继续使用 latest / best checkpoint
3. 用正式 best checkpoint 对至少 10 个固定 prompt 运行 greedy、temperature、top-k，记录参数差异、成功样例和失败样例
