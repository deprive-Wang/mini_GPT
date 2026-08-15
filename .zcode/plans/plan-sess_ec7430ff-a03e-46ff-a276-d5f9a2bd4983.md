目标是在不改变训练行为的前提下，把现有训练指标写入 TensorBoard event 文件，供 AutoDL 的 TensorBoard 面板读取。

1. 仅修改 `train.py` 和 `requirements.txt`：在训练入口创建 `torch.utils.tensorboard.SummaryWriter`，默认日志目录继续使用现有 `experiments/`；新增可选 `--log-dir` 参数，便于 AutoDL 上指定实验目录。

2. 按 optimizer step 写入标量，避免梯度累积导致横轴混乱：
   - 每个 `log_interval`：`train/loss`、`train/perplexity`、`train/learning_rate`、`train/tokens_per_second`、`system/peak_memory_mb`。
   - 每次 validation：`val/loss`、`val/perplexity`，并保留现有文本日志和 best checkpoint 判定。
   继续将 step 0 的 validation 写入 TensorBoard，作为未训练基线。

3. 确保训练正常结束和异常退出时 writer 都会关闭或 flush，避免 event 文件缺尾；不改模型、优化器、学习率调度、checkpoint 格式或采样逻辑。

4. 补充 `requirements.txt` 的 `tensorboard` 依赖。由于这会让 AutoDL 环境需要安装一个新 Python 包，落盘位置由用户的 Mini_GPT/AutoDL 环境决定；代码改完后我会先在当前本地环境检查 `tensorboard` 是否已安装。若未安装，我只提供对应环境的安装命令，不在未确认下载位置前自行安装。

5. 验证：运行 `py_compile`；如果本机已有 TensorBoard，则用极小步训练确认 `experiments/` 下产生 event 文件并使用 `EventAccumulator` 检查预期 tag；否则明确报告因未安装无法完成 event 文件端到端验证，并给出 AutoDL 命令：`tensorboard --logdir experiments --bind_all --port 6006`。同步 `AGENTS.md` 的当前依赖、运行方式和实验记录要求。