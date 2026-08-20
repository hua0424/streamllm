# E3-LA 重跑执行交接（GPU 主机）—— DEV-3 修复后

> **读者**：GPU 主机实验执行者。
> **前置状态**：DEV-3 错帧 + 裁剪幻听双重修复已完成、复审 R2 通过（无阻塞项），批准执行 E0 冒烟与全量重跑。
> **预计耗时**：E0 约 10 分钟；全量约 6 GPU 小时（498 样本，ASR turbo@cuda:0 + LLM Qwen2-7B@cuda:1）。
> **相关文档**：`experiments/review/20260820-E3LA/`（两轮评审）、`E3_LA_BUG_REVIEW.md`（bug 定位 + 修复状态）、
> `../../REVISION_CHANGELOG.md`（2026-08-20 修复条目）、`experiments/GPU_EXPERIMENT_HANDOFF.md` §E3。

---

## 1. 修复摘要（为什么旧结果无效、这次跑的是什么）

- 旧结果（`la_results_20260820_161601.json.INVALID_dev3_frame_bug`）因两个机制无效：
  1. **错帧**：缓冲裁剪后仍用裁剪前帧的 `n_committed` 下标截取新假设 → 中段文本静默丢失（79% 样本大比例丢文本）；
  2. **裁剪幻听**：裁剪把缓冲切成句末残片开头，Whisper turbo 坍缩为水印幻听（'请不吝点赞订阅转发…'）。
- 修复提交 **`6d74c1c`**：绝对时间轴提交 + 标点鲁棒一致比较 + 句界裁剪（`la_max_buffer_s=15.0`）；文档提交 `8a453c0`。
- 本机验证：回归 16/16；典型样本 `crosswoz_10296_turn2` 回放 WER 0.8796→0.0185（`replay_crosswoz_10296_turn2_fixed.json`）。

## 2. 执行步骤

### 步骤 0：代码就位

```bash
cd <repo> && git pull
git log --oneline -2   # 必须看到 8a453c0（或更新）与 6d74c1c
uv run python -m experiments.scripts.test_revision_regressions   # 期望 16/16
```

### 步骤 1：隔离旧现场（防止续传/混淆）

```bash
cd experiments/results/revision/r3_baseline_la
mkdir -p invalid_dev3_frame_bug
mv -f checkpoint.json la_results_*.INVALID_dev3_frame_bug \
      la_summary_*.INVALID_dev3_frame_bug la_statistics_*.INVALID_dev3_frame_bug \
      invalid_dev3_frame_bug/ 2>/dev/null
```

注意：`exp2_ablation_sample_list.json`（498 条清单）和 `system_ab_rerun/` **不要动**。

### 步骤 2：E0 冒烟（评审门槛，必须先跑）

```bash
cd <repo>
uv run python -m experiments.scripts.run_exp_baseline_la \
  --dataset all --sample-list experiments/results/revision/r3_baseline_la/exp2_ablation_sample_list.json \
  --asr-device cuda:0 --llm-device cuda:1 \
  --output-dir experiments/results/revision/r3_baseline_la/e0_smoke --no-resume
```

**E0 验收（全部满足才继续）**：2/2 完成、error 0、无死锁无空转、正常收尾（LLM 产出回复）；
`transcribed_text` 非空且肉眼无中段跳失；结果 config 块含 `trailing_margin_s=0.0`、`la_max_buffer_s=15.0`、
`asr_device=cuda:0`、`llm_device=cuda:1`。

### 步骤 3：全量 E3-LA（约 6h）

```bash
uv run python -m experiments.scripts.run_exp_baseline_la \
  --dataset all --sample-list experiments/results/revision/r3_baseline_la/exp2_ablation_sample_list.json \
  --asr-device cuda:0 --llm-device cuda:1 \
  --output-dir experiments/results/revision/r3_baseline_la --no-resume \
  2>&1 | tee experiments/results/revision/r3_baseline_la/la_run.log
```

说明：`trailing_margin_s=0.0` 在脚本内锁定（等效 System B 的 `--suffix-segments 0`，本脚本无该参数）；
清单缺失样本会**硬失败**（若发生说明数据集目录不完整，停止上报，不要删清单条目凑数）。

## 3. 跑后 QA（评审 R2 要求，逐条执行，不得只看 WER）

| # | 检查项 | 通过标准 |
|---|---|---|
| 1 | 完成度 | 498/498，error 0（任何异常样本逐条给解释） |
| 2 | config 块 | `trailing_margin_s=0.0`、`la_max_buffer_s=15.0`、`asr_device=cuda:0`、`llm_device=cuda:1`、asr_model=turbo、llm_model=Qwen2-7B-Instruct |
| 3 | WER/CER | mean 回到 System B 同机（`system_ab_rerun/`）可解释量级；若 mean 仍 ≫0.2 视为异常 |
| 4 | 转写长度比 | LA/System B 逐样本长度比 mean/median ≈1.0（修复前 mean 0.52、79% <0.7 为无效特征） |
| 5 | divergence_count | 分组统计可解释（偶发）；出现异常爆炸（mean ≫10 或大量样本 >20）视为异常 |
| 6 | 片段抽查 | 抽 3–5 个样本用 `uv run python -m experiments.scripts.replay_la_sample --sample-id <id> --asr-model-size turbo --device cuda:0` 复核：提交片段拼接无重复、无中段跳失、flush 不重复尾部 |
| 7 | TTFT | 分组统计与 System B 同机数字（1573.9ms，long/very_long/extra_long=1464/1551/1638ms）量级可解释 |

**停止规则**（评审 R2 保留项 4）：若 #3/#4/#5 任一项异常，**停止并保留现场**（结果文件加
`.INVALID_<原因>` 后缀、写 RUNINFO.md 说明），不得以"模型随机性"解释后继续入表。

## 4. 完成后登记

1. 在 `experiments/results/revision/r3_baseline_la/` 写 `RUNINFO.md`：完整命令行、代码 commit（`git rev-parse HEAD`）、起止时间、样本数、error 数、上表 7 项 QA 结论；
2. 把 QA 关键数字（WER/CER/长度比/divergence/TTFT 分组）带回本机侧，由本机侧登记 `REVISION_CHANGELOG.md` 与 `PAPER_IMPACT_NOTES.md` 影响项 4；
3. **评审 R2 最终决定**：结果级 QA 完成前，E3-LA 数字不得写入论文表格或最终回复。

## 5. 注意

- 回放脚本（`replay_la_sample.py`）跳过实时休眠，只用于正确性复核，**其数字不能当 TTFT**（评审 R2 保留项 3）。
- 论文方法部分若描述 LA-2 基线，必须写修复后语义（绝对时间轴提交 + 句界裁剪 + `la_max_buffer_s=15.0`），不得只写"LocalAgreement-2"（评审 R2 保留项 2）。
- E4/E5/E6 与 LA 组件零耦合，不受本次重跑影响，可并行推进。
