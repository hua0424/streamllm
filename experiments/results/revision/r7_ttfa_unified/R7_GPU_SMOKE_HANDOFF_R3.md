# R7 统一 TTFA：GPU 冒烟交接文档 r3（PSE Silero 签名修复后）

- 日期：2026-08-21
- 取代：`R7_GPU_SMOKE_HANDOFF_R2.md`（命令与验收不变；本文档为 PSE 修复后的执行版）
- 对应现场报告：`handoff/R7_PSE_SILERO_SIGNATURE_BUG_HANDOFF.md`（已修复，回复见
  `handoff/reply-R7_PSE_SILERO_SIGNATURE_BUG.md`）
- 代码基线：`git pull` 至本次 push（PSE Silero model 透传修复）或更新
- 审查放行范围不变：探活 + 3 条分层冒烟；**正式 50×2 仍待冒烟结果级复核**

## 0. 本次修复（开发侧，已推送）

1. `silero_pse_sample` 透传 Silero `model` 必填位置参数（与分段器调用一致）；
2. `analyze_pse` 对"有函数缺 model"显式拒止（`pse_missing_model`）；
3. self-test 假 Silero 改**签名严格**（漏传 model 即断言失败），新增缺 model 用例；
   self-test 75 → **76 PASS / 0 FAIL**；
4. 本地集成检查加防护：契约类 PSE 错误不允许能量法兜底掩盖。

## 1. 重跑顺序

任务 1（探活）已通过（tts_probe.json 有效），**无需重跑**。

### 任务 2：self-test（期望 **76 PASS / 0 FAIL**）

```bash
uv run python -m experiments.scripts.run_ttfa_unified --self-test
```

### 任务 3：3 条分层 GPU 冒烟（命令与 r2 版完全一致）

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
    --smoke 3 --inject-fault asr_error \
    --output-dir experiments/results/revision/r7_ttfa_unified \
    --run-id r7_smoke
```

PSE 预扫描现在会真实调用 Silero（真实语音样本应有 speech 段；现场已实测
crosswoz_7701_turn3 → n_segments=14、last_end=488920 与能量法一致）。若仍出现
`pse_single_algorithm_failure`，把该样本的 `energy_pse_sample`/`silero_pse_sample`/
`silero_error` 三个字段值反馈，**不要绕过 fail-closed**。

### 验收清单（同 r2 版八项）

QA 0 问题 / 6 记录 / ≥1 成功且 A/B 均有 / 注入任务 error 含 `fault_injection` /
无 `final_drain_empty`·`thread_leak`·`pair_timeout`·schema 错误 / Silero artifact hash
双侧一致（repo_commit=None+注记属预期）/ TTS 无 error / TTFA 量级合理（B 数秒级、
A 15–40s 级）。慢流证据与禁止事项同前。W2 环境记录已采集，无需重复。

## 2. 反馈与提交

- 产物：`r7_ttfa_unified/`（checkpoint_r7_smoke.jsonl、RUNINFO_r7_smoke.md、QA_r7_smoke.md、
  summary、run.log；tts_probe.json 已在）；
- push 后通知本机结果级核验；核验 + 审查复核通过后出正式实验 handoff
  （50×2 + 子集补轮 + 匹配文本控制）。
