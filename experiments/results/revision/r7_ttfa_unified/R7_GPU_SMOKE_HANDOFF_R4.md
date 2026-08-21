# R7 统一 TTFA：GPU 冒烟交接文档 r4（PCM/JSON 误判修复后）

- 日期：2026-08-21
- 取代：`R7_GPU_SMOKE_HANDOFF_R3.md`（命令与验收不变；本文档为 classify_payload 修复后的执行版）
- 对应现场报告：`handoff/R7_TTS_PCM_JSON_MISCLASSIFY_BUG_HANDOFF.md`（已修复，回复见
  `handoff/reply-R7_TTS_PCM_JSON_MISCLASSIFY_BUG.md`）
- 代码基线：`git pull` 至本次 push（classify_payload 严格解析修复）或更新
- 审查放行范围不变：探活 + 3 条分层冒烟；**正式 50×2 仍待冒烟结果级复核**

## 0. 本次修复（开发侧，已推送）

`classify_payload` 由"单字节前缀"改为**结构校验**：JSON 须整段前缀通过 `json.loads`
（音频字节解析失败 → pcm）；HTML 须 `<!doctype`/`<html` 特征；Content-Type 显式声明
json/html/xml 时以响应头为准；WAV 保持 4 字节 RIFF。裸 PCM 首字节 `0x7b`/`0x5b`/`0x3c`
不再误判。self-test 76 → **86 PASS / 0 FAIL**（含 10 项新回归用例）。

## 1. 重跑顺序

任务 1（探活）：r2 版已通过。**建议快速重跑一次**（修复后应稳定 ok）：

```bash
uv run python -m experiments.scripts.run_ttfa_unified \
    --tts-probe --tts-url http://127.0.0.1:20401 --tts-spk 晓伊 --tts-speed 0.8 \
    --output-dir experiments/results/revision/r7_ttfa_unified
```

### 任务 2：self-test（期望 **86 PASS / 0 FAIL**）

```bash
uv run python -m experiments.scripts.run_ttfa_unified --self-test
```

### 任务 3：3 条分层 GPU 冒烟（命令与 r3 版完全一致）

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

### 验收清单（同前八项）

QA 0 问题 / 6 记录 / ≥1 成功且 A/B 均有 / 注入任务 error 含 `fault_injection` /
无 `final_drain_empty`·`thread_leak`·`pair_timeout`·schema 错误（含
`tts_format_not_pcm`——若再现请保留记录并反馈 magic_hex）/ Silero artifact hash 双侧
一致 / TTS 无 error / TTFA 量级合理（B 数秒级、A 15–40s 级）。

## 2. 反馈与提交

- 产物：`r7_ttfa_unified/`（checkpoint_r7_smoke.jsonl、RUNINFO/QA、summary、run.log、
  tts_probe.json）；
- push 后通知本机结果级核验；核验 + 审查复核通过后出正式实验 handoff
  （50×2 + 子集补轮 + 匹配文本控制）。
