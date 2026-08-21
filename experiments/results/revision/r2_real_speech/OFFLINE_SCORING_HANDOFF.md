# R2 离线评分 + 中文 CER 口径修正执行交接（GPU 主机，纯 CPU，约 2 分钟）

> **2026-08-21 追加（任务 1 需重跑）**：首轮产物经本机核验发现 scope 口径问题——
> 脚本按 sample_id 前缀分组，把 12 个变体并入干净集，丢失 Table VI 需要的逐条件行。
> 脚本已修正（`--scope-by dir` 为默认，E3 已用 `--scope-by prefix` 重出并验证不变）。
> **请重跑任务 1**（命令不变，git pull 后直接执行即可，约 1 分钟）；任务 2 不受影响，无需重跑。
>
> **2026-08-21 二次追加（glob 引号问题已闭环）**：主机指出的"带引号 glob 只读到最后一个目录"
> 属实（旧 `files[-1]` 对单 glob 展开的多目录取样错误）。脚本已改为**按目录分组取最新**，
> 带不带引号行为一致，本文档命令保持带引号写法即可，无需调整执行方式。
> 任务 1 重跑产物（8036780）经核验正确（30 WER 行 + 102 TTFT 行，逐条件齐全）。

> **读者**：GPU 主机实验执行者。
> **性质**：不是新实验——两个离线统计任务，需要在主机执行的唯一原因是
> librispeech/aishell1 及其 12 个变体的样本 JSON（含参考文本）只存在于主机
>（E2-0 主机自建，未回传；本机只有 crosswoz/multiwoz）。
> **脚本已随 main 推送**：`git pull` 即可，无需任何数据传输。

---

## 任务 1：R2 离线 WER/CER + TTFT 统计（Table VI 用数）

```bash
git pull
uv run python -m experiments.scripts.score_wer_offline --self-test   # 期望 9/9 PASS
uv run python -m experiments.scripts.score_wer_offline \
  --results "experiments/results/revision/r2_real_speech/*/exp1_results_*.json" \
  --out-dir experiments/results/revision/r2_real_speech
```

产物：`r2_real_speech/wer_real.csv` 与 `ttft_real.csv`
（scope=数据集/ALL × mode × 分组；含 mean/std/p50/p95、空转写计数 n_empty）。

口径（与 exp3 一致 + 两处已登记扩展）：英文归一化后大小写折叠；中文 `cer` 直接吃原文；
空转写（babble 零提交样本）WER/CER 记 1.0 并计 n_empty；error 行排除。

**验收**：14 个数据集目录全部读取（clean 2 个各 150 行、变体 12 个各 60 行）；
csv 落盘；无"未在样本 JSON 索引中找到"报错。

## 任务 2：中文 CER 口径修正重算（E2-0 QA 数字勘误）

背景：2026-08-21 本机发现 `qa_real_speech.py` 原中文分支误用
`cer(zh_to_word_seq(ref), zh_to_word_seq(hyp))`——逐字空格污染 `cer` 的字符分母，
E2-0 登记的 "aishell1 CER 6.72%" 口径有误（exp3 原生口径是 `cer(ref, hyp)` 直接吃原文，
见 `run_exp_quality.py:606`）。脚本已修正并加了免模型重算模式。执行：

```bash
uv run python -m experiments.scripts.qa_real_speech \
  --recompute-from-csv experiments/results/revision/r2_real_speech/qa_transcribe.csv
```

产物：`qa_transcribe.corrected.csv`（含完整参考文本 + 修正值），控制台打印旧值→修正值对照。
英文 WER 口径未变，重算值应与原值一致（作为交叉验证）。

**验收与处置**：修正后 mean CER ≤ 0.10 → exit 0，勘误关闭；
若 > 0.10（exit 2）→ **不要删改任何数据**，把 corrected CSV 与控制台输出带回本机侧，
由本机侧评估是否影响 E2-0 验收结论（该 QA 是构建 sanity 门，目的是抓构建错位——
构建错位会表现为 CER 50%+ 而非十几个点）并统一登记 changelog。

## 完成后

把 `wer_real.csv`、`ttft_real.csv`、`qa_transcribe.corrected.csv` 与控制台输出
随 git 提交或回传本机侧；本机侧登记 changelog 并继续 Table VI 装配。
