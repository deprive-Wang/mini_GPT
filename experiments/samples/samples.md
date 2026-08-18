# 生成样例实验记录（验收第 5、6 条）

日期：2026-08-16

## 实验设置

| 项 | 值 |
|---|---|
| checkpoint | `checkpoints/best.pt`（step 24,000） |
| val loss / ppl | 1.4395 / 4.219（baseline-100k 全程最优） |
| 模型 | 6 层 Decoder-only，44.9M 参数，T=512 |
| 训练数据 | TinyStories 前 100 万篇，GPT-2 BPE，约 26,000 optimizer steps（约 3.8 token epoch） |
| 生成长度 | 每份 120 new tokens |
| 随机种子 | temperature / top-k 均固定 seed 42，可复现 |
| 采样参数 | greedy；temperature 0.8；top-k 40 + temperature 0.8 |

文件命名：`NN-<prompt名>.<模式>.txt`，共 10 × 3 = 30 份，与本文同目录。

## 质量标注总表

成功 = 情节连贯、有基本故事结构；部分 = 大体连贯但有明显瑕疵或截断；失败 = 实体混乱、逻辑矛盾或重复退化。

| prompt | greedy | temperature 0.8 | top-k 40 |
|---|---|---|---|
| 01 once-upon-a-time | 部分（连贯但中断） | 部分（moral 未写完） | 部分（同左） |
| 02 magic-box | 成功 | 成功 | 成功（结构最完整） |
| 03 best-friends | 成功 | 成功 | 成功 |
| 04 dog-morning | 失败（狗看见"另一只狗"实体混乱） | 失败（语法断裂 + 实体混乱） | 部分 |
| 05 garden-tree | 部分（对话怪、结尾退化为单字） | 失败（鸟对树说"想像你一样飞"） | 失败（同左） |
| 06 learn-bike | 成功（最有逻辑的一份） | 失败（"wear your helmet" 重复循环） | 失败（同左） |
| 07 sunny-morning | 成功 | 成功 | 成功 |
| 08 old-man-sea | 部分（老人自称 old man） | 失败（老人"打电话给他的 owner"） | 失败（同左） |
| 09 hungry-rabbit | 部分（怪物转折突兀） | 失败（鳄鱼中途变狼） | 部分 |
| 10 little-fox | 成功 | 成功（结尾截断） | 成功（同左） |

合计约：成功 13 / 部分 7 / 失败 10。

## 典型成功样例

- `02-magic-box.topk40.txt`：完整的"发现魔盒—得到毯子—珍惜毯子"故事弧，自然收尾并输出 `<|endoftext|>`。
- `06-learn-bike.greedy.txt`：头盔、安全、母子对话全程逻辑自洽，是 greedy 模式最稳的一份。
- 多份样例（02 greedy/topk、04 greedy、09 temp08/topk）在故事讲完后主动输出 `<|endoftext|>` 并另起新故事，说明模型学到了训练数据中篇与篇的边界。

## 典型失败样例

- `08-old-man-sea.temp08.txt`：*The old man ... called his owner*——把老人当宠物写，训练语料里狗/宠物叙事迁移到了人身上。
- `09-hungry-rabbit.temp08.txt`：鳄鱼登场后中途变成 *the wolf's pond*，捕食者实体中途漂移。
- `06-learn-bike.temp08.txt` / `.topk40.txt`：*wear your helmet and wear your helmet* 整句级重复循环，temperature 采样的典型退化。
- `04-dog-morning.greedy.txt`：*The little dog ... saw the little dog*，主语与宾语自我指代混乱。
- `05-garden-tree.greedy.txt`：结尾退化为孤立字符 `L`，greedy 的 token 级退化。

## 三种模式对比结论

- greedy：语法最稳、无随机性，但重复率高，长文容易原地绕圈或出现退化字符；同一 prompt 结果完全一致。
- temperature 0.8：情节更发散，但代价是重复循环（06）和实体漂移（09）集中出现在该模式；同 seed 可复现。
- top-k 40：整体与 temperature 0.8 质量接近且略稳；同 seed 下多数 prompt 与纯 temperature 输出完全相同（01/03/05/07/08/10），只在采样 token 落出前 40 名时分叉（02/04/06/09）。原因是两种模式同 seed 同初始分布，top-k 截断大多数时候不影响被采到的 token。
- 所有模式共享的系统性弱点：长程实体一致性（谁在做事）和物理常识（鸟想学飞、老人有 owner），这正是 TinyStories 小模型的预期边界。

## 复现命令（PowerShell）

```powershell
conda activate Mini_GPT
cd D:\holiday_learning\mini_GPT
python sample.py --checkpoint checkpoints/best.pt --prompt "Once upon a time" --mode top-k --top-k 40 --temperature 0.8 --max-new-tokens 120 --seed 42
```
