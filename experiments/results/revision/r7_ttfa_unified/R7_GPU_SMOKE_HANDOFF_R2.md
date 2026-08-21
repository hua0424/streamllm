# R7 统一 TTFA：GPU 冒烟交接文档 r2（TTS 契约修复后）

- 日期：2026-08-21
- 取代：`R7_GPU_SMOKE_HANDOFF.md`（任务顺序与验收清单不变，本文档为修复后的执行版）
- 对应现场报告：`handoff/R7_TTS_CLIENT_CONTRACT_BUG_HANDOFF.md`（两处契约错配已修复）
- 代码基线：`git pull` 至本次 push（TTS 契约修复 commit）或更新
- 审查放行范围不变：探活 + 3 条分层冒烟（含故障注入）；**正式 50×2 仍待冒烟结果级复核**

## 0. 本次修复内容（开发侧，已推送）

1. **端点拼接**：`_tts_endpoint()` 幂等拼 `/inference_sft`（裸基址/尾斜杠/已带后缀三种输入均正确）；
2. **form 编码**：探活与正式请求均改 `data=`（`application/x-www-form-urlencoded`），
   字段全字符串（`stream="True"`），与 E6 `measure_tts_first_chunk.py` 契约一致；
3. **契约严格假服务**（回归防护）：self-test 的假 TTS 服务现在与真实 FastAPI 一致——
   非法路径 404 JSON、非 form body 422 JSON；新增探活契约测试与端点归一化测试；
   self-test 由 69 → **75 PASS / 0 FAIL**；本机真实组件集成检查复跑 ALL PASS。

## 1. 两件次要事项的处理结论（对应现场报告 §4）

1. **Content-Type: None**：接受并按设计工作——探活把缺头按 `None` **原样固定为允许值**
   （probe 输出新增 `policy_note` 字段说明），正式请求逐项**严格相等**比对；任何偏离
   （头新增、取值变化）记 error，不放宽。旧审查"缺 Content-Type 不得视为可接受"针对的是
   旧实现"缺头即静默放行"；现实现是"探活时固定、正式时严格比对"，满足其意图；
2. **Silero 缓存目录非 git checkout**：可接受——锁定依据转为 **artifact sha256**
   （现场已实测 `e1122837f4154c511485fe0b929c96fbb8d79fbdb336383ebd3720`，2.27MB .jit），
   脚本自动哈希并注入正式分段器；`repo_commit` 记 `None` 并附原因注记
   （`repo_commit_note`）。**沿用 `--silero-dir` 即可，无需 `--silero-ref`**。

## 2. 执行顺序（与原 handoff 相同，命令不变）

### 任务 1：TTS 探活（期望 `"ok": true`、`payload_class: "pcm"`）

```bash
uv run python -m experiments.scripts.run_ttfa_unified \
    --tts-probe --tts-url http://127.0.0.1:20401 --tts-spk 晓伊 --tts-speed 0.8 \
    --output-dir experiments/results/revision/r7_ttfa_unified
```

探活产物 `tts_probe.json` 随冒烟一并提交（现含 endpoint/policy_note 字段）。

### 任务 2：self-test（期望 **75 PASS / 0 FAIL**）

```bash
uv run python -m experiments.scripts.run_ttfa_unified --self-test
```

### 任务 3：3 条分层 GPU 冒烟（命令与原 handoff §4 完全一致）

```bash
uv run python -m experiments.scripts.run_ttfa_unified \
    --sample-list experiments/results/revision/r1_stats/repeat_subset_ids.json \
    --json-dir experiments/datasets/processed/json \
    --audio-dir experiments/results/../../datasets/processed/audio \
    --datasets crosswoz multiwoz \
    --asr-model turbo --asr-device cuda:0 \
    --llm-model Qwen/Qwen2-7B-Instruct --llm-device cuda:1 \
    --tts-url http://127.0.0.1:20401 --tts-spk 晓伊 --tts-speed 0.8 \
    --silero-dir ~/.cache/torch/hub/snakers4_silero-vad_master \
    --smoke 3 --inject-fault asr_error \
    --output-dir experiments/results/revision/r7_ttfa_unified \
    --run-id r7_smoke
```

（`--audio-dir` 用 `experiments/datasets/processed/audio`；上行折行仅为可读。）

### 验收清单（原 handoff §4 八项不变）

QA 0 问题 / 6 记录 / ≥1 成功且 A/B 均有 / 注入任务 error 含 `fault_injection` /
无 `final_drain_empty`·`thread_leak`·`pair_timeout`·schema 错误 / Silero artifact hash
双侧一致（`repo_commit` 为 None 属预期，见 §1.2）/ TTS 无 error / TTFA 量级合理
（B 数秒级、A 视回复长度 15–40s 级）。慢流证据与禁止事项同原 handoff。

### W2 环境记录

同原 handoff §5（lscpu/uname/free/nvidia-smi/governor/pip freeze/TTS 服务 commit
`8555549e` 与启动参数 → `r7_ttfa_unified/env/`）。

## 3. 反馈与提交

- 产物：`r7_ttfa_unified/`（checkpoint_r7_smoke.jsonl、tts_probe.json、RUNINFO/QA、env/）；
- push 后通知本机做结果级核验；核验 + 审查复核通过后才出正式实验（50×2 + 子集补轮 +
  匹配文本控制）handoff；
- 若探活仍非 `ok:true`：**不要改代码/放宽策略**，把完整 JSON 与
  `curl -s -X POST -d 'tts_text=探活&spk_id=晓伊&stream=True&speed=0.8'
  http://127.0.0.1:20401/inference_sft | head -c 32 | xxd` 的输出一并反馈。
