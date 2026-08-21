# 冒烟结果级核验报告（开发侧，提交审查复核）

- 日期：2026-08-21
- 核验对象：commit `b1e1206` 产物 `r7_ttfa_unified/`（run_id=r7_smoke，6 任务）
- 核验方式：本机从 `checkpoint_r7_smoke.jsonl` 原始记录独立重算（不转引脚本自产 QA）
- 结论：**八项验收逐项通过，独立重算与 QA 一致，无新增问题。建议审查复核通过后放行正式实验。**

## 一、逐项核验（对照 R7_GPU_SMOKE_HANDOFF_R4 验收清单）

| # | 验收项 | 核验结果 |
|---|---|---|
| 1 | QA 0 问题、记录数 6 | ✅ QA_r7_smoke.md：success 5 / 非成功 1，问题 0；checkpoint 实际 6 条记录 |
| 2 | 成功 ≥1 且 A/B 均有 | ✅ streaming 3 成功（zh 2 + en 1）、non-streaming 2 成功 |
| 3 | 注入任务 error 含 fault_injection | ✅ crosswoz_8672_turn3/non-streaming：error 尾部 `RuntimeError: fault_injection:asr_error`（出自 `transcribe_complete_audio`，即注入点），fatal=True（其为最后一条任务，无剩余需回填） |
| 4 | 无 final_drain_empty / thread_leak / pair_timeout / schema 错误 | ✅ 全部 5 条成功记录逐字段重扫：`final_drain_empty` 均为 False（首扫误报系脚本把字段名当命中，已修正）；error 字段均空；validate_record 独立重跑 0 违规 |
| 5 | Silero artifact hash 双侧一致 | ✅ binding `silero_meta.artifact_sha256=e1122837…d3720`；`repo_commit=None` + 注记（非 git checkout，属 r3 已定夺口径）；RUNINFO segmenter_meta 注入断言在 |
| 6 | TTS 无 error | ✅ 5 条成功记录 tts.error 均空；探活 ok=true/pcm（Content-Type None 策略生效） |
| 7 | TTFA 非负且量级合理 | ✅ B：zh 2582/2610/2555ms、en 3045ms（close→首token 1251–1526ms + TTS→playable 1096–1268ms）；A：22276/23401ms（全文 TTS 13.4–18.7s 主导）；原始 ns 闭合残差 0 |
| 8 | 慢流证据 | ✅ 无任何 TTS 错误记录；如出现假阳性分类已在 r4 修复并有 10 项回归 |

补充核验：配对 seed 三样本 A/B 各自一致（3896906829 / 1462181371 / 3763255375）；
每样本恰 A/B 双模式；语言覆盖 zh+en；双语种 + 双模式 + 成功 + 故障四类路径齐备。

## 二、组件量级快照（冒烟，n 极小，仅量级参考，不入论文）

- B（streaming）：trailing_feed_wait ≈0ms（无追加静音设计下符合预期）→ flush→close ~100ms
  → close→首 token 1251–1526ms（最终段转写+预填）→ 首句冻结 →~40ms 内发出 TTS 请求
  → TTS→playable 1096–1268ms；
- A（non-streaming）：feed_end → 全文 ASR+预填 2751–3239ms → 全文生成 → 全文 TTS
  首包 13405–18711ms → TTFA 22.3–23.4s；
- A−B 差 ≈ 19.6–20.4s，主要由"A 等全文再 TTS"策略贡献——与协议声明一致
  （差异含各自 TTS 调用策略，不全部归因 ASR/KV）。

## 三、-lcuda 噪声（主机顺带说明，非阻塞）

属实登记：`/usr/bin/ld: cannot find -lcuda` 为 ctranslate2 JIT 探测 32 位 libcuda 的
链接噪声，ASR 正常出结果（5 条成功记录转写正确）。**不作为门禁项**；正式实验沿用现状，
若未来某任务因此真失败会以 error 记录显式暴露（fail-closed 语义覆盖），无需预防性处置。

## 四、建议

冒烟八项全过、真实 GPU 路径（固定 Silero PSE+分段器、Whisper final-drain、Qwen 生成、
TTS 契约、故障注入、双语双模式）均有实测证据。**申请审查复核通过后放行正式实验**：
50×2 主实验 + 10 条子集补 repeat 1/2 + 匹配文本 TTS 控制（正式 handoff 已备好待发）。
