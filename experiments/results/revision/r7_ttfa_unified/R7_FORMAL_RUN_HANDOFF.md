# R7 正式实验交接文档（50×2 主实验 + 子集补轮 + 匹配文本控制）

- 日期：2026-08-21（待审查复核通过后执行）
- 前置：冒烟结果级核验通过（`experiments/review/20260821-PRE-PAPER-AUDIT/dev-smoke-verification-20260821.md`）
  **且审查方书面放行正式实验**——未获放行前本文档不得执行。
- 代码基线：`git pull` 至 `cdeb927` 或更新
- 脚本：`experiments/scripts/run_ttfa_unified.py`（探活/self-test 期望与 r4 版一致：86 PASS）

## 1. 正式主实验 + 重复子集（单条命令）

```bash
uv run python -m experiments.scripts.run_ttfa_unified \
    --sample-list experiments/results/revision/r1_stats/repeat_subset_ids.json \
    --json-dir experiments/datasets/processed/json \
    --audio-dir experiments/datasets/processed/audio \
    --datasets crosswoz multiwoz \
    --asr-model turbo --asr-device cuda:0 \
    --llm-model Qwen/Qwen2-7B-Instruct --llm-device cuda:1 \
    --tts-url http://127.0.0.1:20401 --tts-spk 晓伊 --tts-speed 0.8 \
    --silero-dir ~/.cache/torch/hub/snakers4_silero-vad_master \
    --output-dir experiments/results/revision/r7_ttfa_unified \
    --run-id r7_main
```

与冒烟的差异（全部由脚本内置，无需额外参数）：

- 无 `--smoke`/`--inject-fault`（正式模式禁用故障注入）；
- 50 样本全量、AB/BA 分层平衡调度（25/25，语言×时长 stratum ≤1）；
- 自动含 10 条重复子集（5 zh + 5 en，repeat 0 计入三轮 + 自动补 repeat 1/2，
  三轮交替 AB/BA/AB 或 BA/AB/BA），schedule 共 50×2×1 + 10×2×2 = **120 任务**；
- 每任务预计 1–4 分钟（A 全文 TTS 15–40s 主导），总计约 **3–5 小时**；
- checkpoint 断点续跑：中断后同 run_id 重跑自动跳过已终态任务；
  **fatal（模型/线程级）会 fail-stop 并把剩余任务补 cancelled，此时停下反馈，勿换 run_id 重试**。

## 2. 匹配文本 TTS 控制（主实验完成后）

对主实验产出的分层 ≥10 条子集（语言×时长平衡），仅做 TTS 调用分离"早启动策略"与
"输入长度"影响。执行方式见下（从 checkpoint 读取 B 首句/A 回复文本后逐条调 TTS）：

```bash
uv run python -m experiments.scripts.run_ttfa_unified \
    --sample-list experiments/results/revision/r1_stats/repeat_subset_ids.json \
    --json-dir experiments/datasets/processed/json \
    --audio-dir experiments/datasets/processed/audio \
    --tts-url http://127.0.0.1:20401 --tts-spk 晓伊 --tts-speed 0.8 \
    --silero-dir ~/.cache/torch/hub/snakers4_silero-vad_master \
    --output-dir experiments/results/revision/r7_ttfa_unified \
    --run-id r7_tts_control --tts-control-only
```

说明：`--tts-control-only` 为只跑控制测量、不跑 A/B 管线的模式（每样本：B 首句文本重测
= 运行方差、A 回复首句、A 全文重测；另加固定校准句中英各一）。该模式本轮由开发侧随
放行一并实现并自测——**若该 flag 尚未在代码中，请勿自行改造，反馈后由开发侧补齐**。

## 3. 验收清单（正式实验）

1. `QA_r7_main.md`：问题 0；任务 120 全部终态（无 cancelled/timeout；error 行保留并逐条反馈）；
2. 50 条主实验每配对键 A/B 双记录、WAV hash 与 seed 一致；
3. 子集 10 条每 (sample, mode) 恰 3 轮成功（`ttfa_subset_cv_r7_main.csv` 有 CV 值）；
4. 成功记录：validate 全过、闭合残差 0、TTFA 非负；
5. TTS 无 error（偶发错误保留记录反馈，不删不改不重试掩盖）；
6. RUNINFO：git/env/Silero artifact/TTS 探活策略/清单 hash 齐全；
7. `-lcuda` 噪声属已知登记项，不影响验收。

## 4. 产物与反馈

- `r7_ttfa_unified/`：checkpoint_r7_main.jsonl、RUNINFO/QA/summary/CV、run.log、
  r7_tts_control 同套产物、tts_probe.json；
- push 后本机做结果级核验 → 装配新 Table VIII（TTFA_playable 主口径）→ 总册 W8 阶段 2
  同步 → 审查复核 → 论文修改放行。
