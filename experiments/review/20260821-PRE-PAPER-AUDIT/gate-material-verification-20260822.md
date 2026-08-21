# R7 放行前 Gate 材料包本机核验清单（对应 review-final-gate 终裁）

- 核验日期：2026-08-22
- GPU 材料提交：`51f5d8f`
- handoff 路径修订：`128ff2e`
- 结论：**Gate 材料实质齐备，非末位 fatal/cancelled 证据通过；建议提交审查方最终放行复核。**

## 1. 材料逐项核验

| 材料 | 结果 |
|---|---|
| GPU clean tree / code provenance | ✅ `gate_clean_git.txt`：HEAD=`2e54ac2`，porcelain 为空 |
| GPU clean self-test | ✅ `gate_selftest_gpu.log/.md`：90 PASS / 0 FAIL，exit 0 |
| 非末位 fatal smoke | ✅ 6 记录：`success → error(fault_injection,fatal) → 4×cancelled_after_fatal`；QA 0 问题；cancelled 无事件 |
| TTS provenance | ✅ CosyVoice commit/diff/image/model/spk2info/startup/dependency 材料已提交 |
| Platform conditions | ✅ 双 RTX 3090、驱动/CUDA、Triton fallback、显存/进程/独占声明等已提交 |
| 新 TTS probe | ✅ `ok=true` / `payload_class=pcm`，策略与正式客户端一致 |
| Gate manifest | ✅ 8 项材料均列出，`code_commit=34ea12e`；以 Git blob 原始内容重算全部 hash 一致 |

## 2. 非末位 fatal smoke 独立重算

从 `fatal_smoke/checkpoint_r7_smoke_fatal.jsonl` 逐条核验：

- 状态序列：`success, error, cancelled, cancelled, cancelled, cancelled`；
- error 记录含 `fault_injection:asr_error`，`fatal=true`；
- 后续 4 条均为 `terminal_state=cancelled`、`error=cancelled_after_fatal`；
- cancelled 记录没有 ASR/LLM/TTS 事件、没有 chunk/event 污染；
- success 记录通过 `validate_record`，TTFA/闭合规则正常；
- 独立 run，与正式 `r7_main` 无关。

## 3. 两处非阻塞交付瑕疵

1. §2c 原始归档在 `selftest_archive/`，§0b manifest 引用 `env/gate/gate_selftest_gpu.log`；GPU 侧已**非破坏性复制**到 manifest 路径，原始归档保留。后续 handoff 已统一要求直接写 `env/gate/gate_selftest_gpu.log`；
2. fatal smoke 没有 run.log：原因是 `tee` 先于脚本创建输出目录；checkpoint/RUNINFO/QA/summary/CV 完整，控制台日志在 GPU 任务日志中，**不影响 Gate 验收**。后续 handoff 已要求先 `mkdir -p`，避免再次发生。

## 4. hash 说明

本机 Windows checkout 会受 CRLF 转换影响，直接从工作树读取材料 hash 会与 GPU manifest 文本不同；
以仓库 Git blob 原始内容重算，8/8 manifest hash 全部一致。已加入 `.gitattributes`：

```text
experiments/results/revision/r7_ttfa_unified/** -text
```

后续材料应保持 LF 原样检出，避免再次出现表面 hash 差异。

## 5. 放行申请边界

建议审查方现在复核并书面放行：

- 正式主实验 `r7_main`（50×A/B + 10 条子集补轮）；
- 主实验完成后 `r7_tts_control`（32 次匹配文本 TTS 控制）。

正式 run 必须使用新子目录、新 checkpoint、新探活；不得从旧 smoke checkpoint 续跑。
