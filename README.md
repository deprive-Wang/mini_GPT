# Mini-GPT

在 TinyStories 上从零预训练一个能续写英文小故事的 Decoder-only Transformer。模型、因果 mask、训练循环、断点恢复和采样全部手写实现，只复用 GPT-2 的 BPE 词表，不调用训练框架。对照工程是 [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)，只对照、不照抄。

本文档同时是项目说明和第一版实验记录。

> **仓库不包含训练数据和模型权重。** `data/`、`checkpoints/`、TensorBoard 日志会在 clone 后重新生成；如果要保留当前 26,000 步基线，请在删除本地目录前把 `checkpoints/best.pt` 和 `checkpoints/latest.pt` 复制到仓库外的可靠存储。`best.pt` 用于推理，`latest.pt` 用于 `--resume` 续训。

## 模型

6 层 Pre-LayerNorm Decoder-only，参数量 44.9M：

| 项 | 值 |
|---|---|
| 层数 | 6 |
| d_model | 512 |
| n_head | 8（每头 64 维） |
| d_ff | 2048 |
| context T | 512 |
| dropout | 0.1 |
| 归一化 | Pre-LayerNorm |
| 位置编码 | 可学习绝对位置编码 |
| 词表 | 50257（GPT-2 BPE，不重训 tokenizer） |
| 权重共享 | tok_emb 与 lm_head 共享（lm_head 无 bias） |

实现要点：

- 手写因果 mask：`[T, T]` 上三角布尔表，位置 t 只能看到 0..t
- label 错位一格：`x = 片段[:-1]`，`y = 片段[1:]`，一个长度 T 的样本提供 T 个预测任务
- 残差投影初始化缩小 `0.02 / sqrt(2*n_layer)`，稳定深层残差叠加

## 数据

- 数据集：[roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
- 训练集：官方 train split 前 100 万篇，编码后 `data/train.bin` 共 **224,512,862 tokens**
- 验证集：完整 validation split 21,990 篇，`data/val.bin` 共 **4,765,918 tokens**
- 分词：`tiktoken` 的 `gpt2` encoding；`encode_ordinary` 编码正文，篇尾手动补 `<|endoftext|>`(50256) 分隔
- 存储：uint16 连续 token 流（词表 50257 < 65536，2 字节够用），memmap 按页读取，训练时随机有放回采样

## 训练配置

| 项 | 值 |
|---|---|
| 优化器 | AdamW（fused），betas (0.9, 0.95) |
| weight_decay | 0.1，只打在 2D 权重上，bias / LayerNorm 不衰减 |
| 学习率 | 峰值 3e-4，linear warmup 200 步 + cosine 降到 3e-5 |
| 梯度裁剪 | 1.0 |
| 精度 | FP16（autocast + GradScaler） |
| micro batch | 16，梯度累积 4，有效 batch 64 |
| 每 step tokens | 16 × 4 × 512 = 32,768 |
| checkpoint | 每 1,000 steps 存 latest；validation 最优另存 best |
| 记录 | loss / ppl / lr / tokens-per-sec / 峰值显存，同时写 train.log 与 TensorBoard |

## 训练结果（baseline，2026-08-16）

| 项 | 值 |
|---|---|
| 硬件 | AutoDL 单卡 RTX 3090 24GB |
| 环境 | PyTorch 2.8.0+cu128，Python 3.12 |
| 训练量 | 26,000 / 计划 100,000 optimizer steps（定时关机中止），约 3.8 个 token epoch，累计约 8.5 亿 tokens |
| 平均速度 | 约 89,000-90,000 tokens/sec（TensorBoard 实测区间 44k-90k） |
| 峰值显存 | 7,799 MB / 24,576 MB |
| 折算时长 | 约 2.7 小时（按平均速度折算的估计值） |
| best | step 24,000，val loss **1.4395**，val ppl **4.22** |
| latest | step 26,000，val loss 1.4446，val ppl 4.24 |

说明：学习率调度按 100,000 步的 cosine 计算，实际停在 26,000 步，此时 lr 仍在 2.6e-4 附近、未进入深度衰减区；曲线后段的提升部分来自数据量本身。

### 验证曲线（每 1,000 steps 评估一次，20 个随机 batch 平均）

| step | train loss | val loss | val ppl |
|---:|---:|---:|---:|
| 0 | 10.867 | 10.870 | 52,592.7 |
| 1000 | 2.222 | 2.242 | 9.4 |
| 2000 | 1.890 | 1.895 | 6.7 |
| 3000 | 1.753 | 1.787 | 6.0 |
| 4000 | 1.695 | 1.727 | 5.6 |
| 5000 | 1.633 | 1.677 | 5.4 |
| 6000 | 1.624 | 1.646 | 5.2 |
| 7000 | 1.572 | 1.633 | 5.1 |
| 8000 | 1.575 | 1.621 | 5.1 |
| 9000 | 1.540 | 1.596 | 4.9 |
| 10000 | 1.534 | 1.557 | 4.7 |
| 11000 | 1.536 | 1.535 | 4.6 |
| 12000 | 1.530 | 1.512 | 4.5 |
| 13000 | 1.478 | 1.550 | 4.7 |
| 14000 | 1.492 | 1.512 | 4.5 |
| 15000 | 1.462 | 1.529 | 4.6 |
| 16000 | 1.491 | 1.510 | 4.5 |
| 17000 | 1.466 | 1.511 | 4.5 |
| 18000 | 1.445 | 1.482 | 4.4 |
| 19000 | 1.434 | 1.464 | 4.3 |
| 20000 | 1.446 | 1.483 | 4.4 |
| 21000 | 1.447 | 1.482 | 4.4 |
| 22000 | 1.412 | 1.456 | 4.3 |
| 23000 | 1.465 | 1.477 | 4.4 |
| 24000 | 1.426 | **1.4395** | **4.22** |
| 25000 | 1.417 | 1.479 | 4.4 |
| 26000 | - | 1.445 | 4.24 |

训练与验证 loss 差距很小（26,000 步时约 0.05），未见明显过拟合；12,000 步后进入平台期，收益放缓。TensorBoard 事件文件在 `experiments/tb/baseline-100k/`（本地保留，不入库）。

## 生成结果

用 `best.pt`（step 24,000）对 10 组固定 prompt 各跑 greedy / temperature 0.8 / top-k 40 三种模式，共 30 份样例（seed 42，可复现）。质量分布约：成功 13 / 部分 7 / 失败 10。完整样例与逐份标注在本机 `experiments/samples/`（gitignore，不入库），要点如下。

成功样例（`02-magic-box.topk40`，prompt: *One day, a little boy found a magic box*）：

> One day, a little boy found a magic box in the park. He was very excited and wanted to see what was inside. He opened it and found a big, soft blanket. ... From that day on, he took the magic blanket with him everywhere he went.

完整的"发现—互动—珍惜"故事弧并自然收尾。多份样例在故事讲完后主动输出 `<|endoftext|>` 并另起新故事，说明模型学到了篇章边界。

失败样例（典型三类）：

1. 实体漂移：`08-old-man-sea`——*The old man ... called his owner*，把老人当宠物写；`09-hungry-rabbit`——鳄鱼登场后中途变成 *the wolf's pond*
2. 重复退化：`06-learn-bike`（temperature/top-k）——*wear your helmet and wear your helmet* 整句循环
3. 主语混乱：`04-dog-morning`——*The little dog ... saw the little dog*，自己看见自己

三种模式对比：greedy 语法最稳但易绕圈和退化字符；temperature 0.8 发散但重复循环集中出现在此模式；top-k 40 整体略稳。同 seed 下 top-k 40 与纯 temperature 在 10 组 prompt 中有 6 组输出完全相同——top-k 截断多数时候不影响被采中的 token，两者只在候选落出前 40 名时分叉。

系统性弱点（各模式共有）：长程实体一致性与物理常识（鸟想学飞、老人有 owner），属于 TinyStories 小模型的预期边界。

## 验收标准对照

| # | 标准 | 状态 |
|---|---|---|
| 1 | logits 形状与 label 错位解释 | 通过，`model.py` / `dataset.py` 自检 |
| 2 | loss 下降、val ppl 可计算被记录 | 通过，见上方曲线 |
| 3 | latest 恢复 / best 独立加载 | 通过，两者均已加载生成 |
| 4 | greedy / temperature / top-k | 通过 |
| 5 | 10 组固定 prompt 样例与参数对比 | 通过，30 份样例 |
| 6 | 完整实验记录 | 本文档 |

## 快速开始

### 从 Git 重新开始（PowerShell）

```powershell
git clone <仓库 URL>
cd mini_GPT

conda create -n Mini_GPT python=3.11 -y
conda activate Mini_GPT
python -m pip install -r requirements.txt
```

`requirements.txt` 不安装 `torch`，避免覆盖本机或云镜像中匹配 CUDA 的版本。安装后先确认 `torch.cuda.is_available()`；没有 CUDA 时可以用很小的 batch 做链路自检，但正式训练建议使用带 CUDA 的 PyTorch 环境。

### 重新准备数据（PowerShell）

```powershell
python scripts/prepare_data.py
```

脚本会下载 TinyStories，并在仓库内生成被 gitignore 的 `data/train.bin`、`data/val.bin` 和 Hugging Face 缓存。数据准备完成后可运行：

```powershell
python model.py
python dataset.py
```

### 本机推理（PowerShell）

需要先把外部备份的 `best.pt` 复制到 `checkpoints/best.pt`：

```powershell
python sample.py --checkpoint checkpoints/best.pt --prompt "Once upon a time" --mode top-k --top-k 40 --temperature 0.8 --max-new-tokens 200 --seed 42
```

### 云端训练（Linux，RTX 3090）

选预装 CUDA PyTorch 的镜像；`requirements.txt` 不含 torch，不会覆盖镜像自带版本：

```bash
python -m pip install -r requirements.txt

# 没有现成 data/*.bin 时先生成（首次下载约 2GB）
python scripts/prepare_data.py

# 20 步 smoke（小卡用 --batch-size 4）
python train.py --max-steps 20 --warmup-steps 2 --eval-interval 10 --eval-iters 4 --batch-size 4 --run-name smoke

# 正式训练（没有外部 checkpoint 时从头开始）
python train.py --max-steps 100000 --run-name baseline-100k

# 从外部备份恢复：先将 latest.pt 放到 checkpoints/latest.pt
python train.py --max-steps 100000 --resume --run-name baseline-100k-resume

# TensorBoard（云平台把 6006 暴露为自定义服务，或走 SSH 隧道）
tensorboard --logdir experiments/tb --host 0.0.0.0 --port 6006
```

### 自检

```bash
python model.py     # 形状 / 因果 mask / 参数量 / 单 batch 前向
python dataset.py   # 形状 / label 错位 / token id 范围
```

## 目录结构

```text
mini_GPT/
  model.py                  模型：6 层 Decoder-only + 自检
  dataset.py                memmap 采样 [16, 512] + 自检
  train.py                  AdamW + warmup/cosine + FP16 + 梯度累积 + checkpoint + TensorBoard
  sample.py                 greedy / temperature / top-k 采样
  scripts/prepare_data.py   下载 TinyStories 并编码为 data/*.bin
  data/                     gitignore；train.bin / val.bin
  checkpoints/              gitignore；latest.pt / best.pt
  experiments/samples/      已提交；30 份固定 prompt 生成样例与标注
  experiments/tb/           gitignore；TensorBoard 事件文件
  experiments/*.log         gitignore；训练日志
  requirements.txt          tiktoken / datasets / numpy / tensorboard
```

代码、实验样例和文档入库；训练数据、Hugging Face 缓存、TensorBoard 日志、训练日志和模型权重不入库。删除本地环境前，请单独备份 `checkpoints/best.pt` 与 `checkpoints/latest.pt`。

## 参考

- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)——对照工程
- TinyStories: [arXiv:2305.07759](https://arxiv.org/abs/2305.07759)
- GPT-2 技术报告；GPT-2 BPE via [tiktoken](https://github.com/openai/tiktoken)
