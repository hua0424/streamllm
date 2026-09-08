# GPU 实验机交接文档（SCI 补实验，已完成归档）

> 正式补实验已全部完成：P1 v2 结果见 run `sci34_dc52978_20260901_async_prepared_v2` 与 D-015。当前不要再次执行本文档；以下内容只保留为复现与审计流程。
>
> 面向对象：需要独立复现实验的 GPU 运行 agent。
> 详细参考：[GPU_RUNBOOK.md](GPU_RUNBOOK.md)（完整命令）、[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)（设计与验收）、[CLAIMS_MATRIX.md](CLAIMS_MATRIX.md)（允许的论文主张）。
> 阅读顺序：本文档 → GPU_RUNBOOK.md → 遇到主张类问题查 CLAIMS_MATRIX.md。

## 0. 背景与目标（一分钟版）

1. 论文已完成离线数据审计；本轮在 GPU 机上执行三个**新增补实验**：
   - **S-E3** 固定生成轨迹一致性（消除原 E3 两条件生成轨迹不同的混杂）；
   - **S-A1** KV crop + 角色恢复联合计时（raw repeats + median/IQR）；
   - **S-P1** headless 异步播放控制路径微基准。
2. **归档状态**：S-P1 prepared-state v2 已完成并通过 D-015 验收。旧 P1 在播放器启动前发起的异步 `ensure_full()` 尚未完成，第一次 stop 后的同步把这段准备工作错误计入 stop→crop/role；这不是可通过删除首个样本解决的一次性冷启动。独立复现时仍须保持旧 `${CAMPAIGN}_async` 只读，且不得顺带重跑 E3/judge/A1。
3. **明确禁止**：人工评测步骤、A2/E3/A1 重跑、真实声卡播放、在线 TTS 取消、完整音频闭环、任何联网下载模型。

## 1. 环境要求

- 2 × RTX 3090 24 GB（与旧实验机 GPU 相同）；
- Python 3.10 / PyTorch 2.8.0+cu128 / transformers 4.57.x；
- 所有模型必须**本地已存在**，正式 run 强制离线（脚本会设置 `HF_HUB_OFFLINE=1`）。

## 2. 执行步骤

以下命令中的路径为示例，按本机实际路径替换：

- `REPO`：仓库检出位置
- `DIALOGUES`：正式 MultiWOZ 派生对话 `p2_turns.json`
- `MAIN_MODEL`：Qwen2-7B-Instruct 本地目录
- `JUDGE_MODEL`：Mistral-7B-Instruct-v0.3 本地目录

### 步骤 1：签出与干净树

```bash
git fetch origin
git checkout paper2
git pull --ff-only
git rev-parse HEAD        # 记录该 commit
git status --short        # 必须为空
```

> 结果目录已被 `.gitignore` 排除，跑完一个子实验不会污染后续子实验的干净树检查。
> 正式 run 默认拒绝 dirty tree；`--allow-dirty` 仅限调试，其结果不得用于论文。

### 步骤 2：环境与数据检查

```bash
uv sync
```

然后执行 GPU_RUNBOOK.md 第 2 节的检查脚本，确认：

- torch 2.8.0+cu128、CUDA 可用、两张 3090 可见；
- NLTK `punkt` / `punkt_tab` 离线资源存在（缺失则先离线安装，否则 E3 正式模式会直接失败）；
- `DIALOGUES`、`MAIN_MODEL`、`JUDGE_MODEL` 路径存在。

**若 `p2_turns.json` 不在本机**（`datasets/processed` 被 gitignore，很可能没有）：
按 GPU_RUNBOOK.md 用 `prepare_multiwoz_data` 以 `--seed 42 --max-dialogues 100` 重新派生，并记录文件 SHA-256。

### 步骤 3：无模型 smoke（必过才继续）

```bash
uv run python -m py_compile experiments/sci34_supplement/*.py
uv run python -m experiments.sci34_supplement.smoke
uv run python -m src.dialogue.run_timeline_test
```

三项全部 PASS 后才加载 7B。

### 步骤 4：采集重跑前精确环境快照

执行 [P1_PREPARED_RERUN.md](P1_PREPARED_RERUN.md) 的 snapshot 命令，保存 CPU 型号/拓扑、RAM、kernel、NVIDIA driver、每张 GPU 的精确字段以及 GPU 进程清单。快照必须进入本次回传包，不能只依赖 manifest 的概括字段。

### 步骤 5：P1 prepared-state v2（无声卡，180 正式事件）

按 GPU_RUNBOOK.md 第 9 节或 P1_PREPARED_RERUN.md 的命令。新 run-id 必须以 `async_prepared_v2` 结尾；默认每个 `(length, fraction)` 做 3 次 warmup，warmup 不落盘；每次 `ensure_full()` 后必须先同步 GPU 再启动播放器。

验收：

- `records.jsonl` 恰好 180 行，全部 `trial_kind=formal`；
- `protocol=async_prepared_v2`、`prepared_state_synchronized=true`，并有非负 `setup_ms`；
- 0.25/0.75 全部 `partial=true`，0.5 全部 `partial=false`；
- 旧 `${CAMPAIGN}_async` 的路径、时间戳和哈希不变。

中断恢复：完全相同参数加 `--resume`。runner 只对存在缺失记录的 `(length, fraction)` 单元做 3 次 warmup，再补缺失 repeat；已完整单元不 warm、不重跑。

### 步骤 6：分析、后快照与 P1-only 打包

生成 `analysis.json`，采集 after snapshots，然后按 P1_PREPARED_RERUN.md **只打包新 P1 run + 本次日志/快照**并生成 SHA-256。不得混入旧 P1、E3、judge 或 A1。

**不要只传 summary**——需要 manifest、180 条 raw records、analysis、运行日志和完整 snapshots。

## 3. 失败处理速查

| 现象 | 处理 |
|---|---|
| OOM | 清掉其他 GPU 进程重试；仍 OOM 则记录失败，**新 run-id** 换配置，禁止 resume 到旧目录 |
| manifest/config hash mismatch | 说明配置变了：新 run-id 重跑，禁止强行续写 |
| judge parse failure | 保留 raw output；修 prompt/解析后新 judge run-id 重跑 |
| 模型本地缺失 | 停止；在进入正式流程前一次性准备好本地模型，不在正式 run 中联网 |
| E3 断句报错（NLTK 资源） | 正式模式禁止 fallback：先离线装好 `punkt_tab` 再重跑 |
| GPU 有干扰进程 | A1 对负载敏感：空闲后用新 run-id 整体重跑该子实验 |

## 4. 汇报要求

跑完后回传：

1. 新 P1 run-id、commit，以及 snapshots 中的精确 CPU/RAM/kernel/driver/GPU 信息；
2. 180 条 formal records 与 partial/boundary 验收结果；
3. P1：每个 `(length,fraction)` 的 stop ack / stop→crop / stop→role median/P95；
4. warmup 数、prepared-state/setup 字段确认；
5. 任何失败、resume 与处置记录。

## 5. 红线（不可违反）

1. 不修改任何旧 GPU 结果，尤其旧 `async_bargein/${CAMPAIGN}_async`；
2. 不重跑 E1/E2/E3/A1/A2、judge 或其他补实验；
3. 不执行任何人工评测步骤；
4. 不把 P1 结果称为真实声卡停播或完整 barge-in latency（口径见 CLAIMS_MATRIX.md）；
5. 不在正式 run 中联网下载模型或数据；
6. 回传包不得包含旧 P1/E3/judge/A1，只包含新 P1 run 与本轮日志/快照。
