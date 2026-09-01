# GPU 实验机交接文档（SCI 补实验）

> 面向对象：GPU 实验机上的运行 agent。本文档是唯一入口，按顺序执行。
> 详细参考：[GPU_RUNBOOK.md](GPU_RUNBOOK.md)（完整命令）、[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)（设计与验收）、[CLAIMS_MATRIX.md](CLAIMS_MATRIX.md)（允许的论文主张）。
> 阅读顺序：本文档 → GPU_RUNBOOK.md → 遇到主张类问题查 CLAIMS_MATRIX.md。

## 0. 背景与目标（一分钟版）

1. 论文已完成离线数据审计；本轮在 GPU 机上执行三个**新增补实验**：
   - **S-E3** 固定生成轨迹一致性（消除原 E3 两条件生成轨迹不同的混杂）；
   - **S-A1** KV crop + 角色恢复联合计时（raw repeats + median/IQR）；
   - **S-P1** headless 异步播放控制路径微基准。
2. **旧实验数据全部保留有效，不重跑**。本机 CPU 与旧实验机不同属预期情况；manifest 会自动记录环境，无需与旧机对比。
3. **明确禁止**：人工评测步骤（已取消）、A2 重跑、真实声卡播放、在线 TTS 取消、任何联网下载模型。

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

### 步骤 4：E3 三条数据 integration

按 GPU_RUNBOOK.md 第 4 节命令（`--limit 3 --formal`）。

验收：`trajectories.jsonl` = 3 行，`records.jsonl` = 24 行（3 对话 × 4 位置 × 2 条件）。

### 步骤 5：E3 正式 100 条

按 GPU_RUNBOOK.md 第 5 节命令（`--limit 100 --formal`）。

验收：`trajectories.jsonl` = 100，`records.jsonl` = 800，无 `fx*` ID。

中断恢复：**完全相同参数** + `--resume`。参数/数据/模型变化会拒绝恢复。

### 步骤 6：Mistral 裁判（换卡 1）

按 GPU_RUNBOOK.md 第 6 节命令。

验收：`parse_failures` 全为 0。解析失败会立即中止——保留现场，修复后用**新 run-id**重跑，禁止把失败记为 NO。

### 步骤 7：E3 分析

按 GPU_RUNBOOK.md 第 7 节命令。

验收：`summary.json` 生成；`construction_checks.playback_local_unheard_empty` 为 true；记录 `total_pairs / eligible_pairs / empty_target_pairs`。

### 步骤 8：A1 联合计时

先 20 repeats 预跑确认 8k 不 OOM，再正式 50 repeats（`A1_REPEATS=50 A1_WARMUP=5` 或按第 8 节命令显式传参）。

验收：每个长度 4 组 raw 数组长度 = repeats；`analysis.json` 生成；主加速比 = `median(re-prefill raw) / median(joint raw)`。

### 步骤 9：P1 异步播放（无声卡，180 事件）

按 GPU_RUNBOOK.md 第 9 节命令（3 长度 × 3 位置 × 20 repeats，`--time-scale 1`）。

验收：`records.jsonl` = 180 行；所有记录 ack 后游标稳定（脚本内置断言）。

### 步骤 10：打包回传

按 GPU_RUNBOOK.md 第 11 节打包 `e3/judge/a1/async_bargein` 各 run 目录（含 manifest、records、summary、日志）+ sha256，回传 tarball。

**不要只传 summary**——论文更新需要 manifest 和 raw records。

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

1. 各 run 的 run-id、commit、GPU/CPU 信息（manifest 已含，确认无缺）；
2. E3：`summary.json` 关键数字（eligible pairs、各口径 rate、McNemar p、bootstrap CI）；
3. A1：各长度 joint median/IQR 与加速比；
4. P1：stop ack / stop→crop / stop→role 的 median/P95；
5. 任何失败与处置记录。

## 5. 红线（不可违反）

1. 不修改 `experiments/results/` 下任何旧 GPU 结果文件；
2. 不重跑旧 E1/E2/E3/A1/A2 正式实验；
3. 不执行任何人工评测步骤；
4. 不把 P1 结果称为真实声卡停播或完整 barge-in latency（口径见 CLAIMS_MATRIX.md）；
5. 不在正式 run 中联网下载模型或数据。
