# 摘要

## 中文摘要

级联式语音对话系统在用户打断时需要协调语言模型生成、语音合成与软件播放的异步进度。按播放位置截断会话历史与 KV 缓存裁剪均有既有先例，本文不主张这些原则或原语的原创性。本文研究外部进度条件下的联合前缀状态修正：将 software-consumed-sample cursor 经 TTS 片段解析为合法 assistant token 提交边界，并同步修正 KV、attention mask、token ledger、position 与 role/EOT 状态。该合同包括边界解析、联合状态、不变式保持转换和可证伪验证；软件游标不等同于设备已呈现采样或声学上被听到的内容。

唯一核心贡献 C2 实现 `software cursor → TTS fragment → assistant token span → KV crop → role recovery`。在冻结的 Qwen2-7B-Instruct、Transformers、BF16 与 SDPA 条件下，v3 覆盖 24 个 case、27 次 crop event 和 60 个 recovery step。每次事件的 crop 前保留前缀、production crop 后状态与从同一 snapshot 逐层切片但不调用 production crop 的 oracle 在 28 层 K/V 上 bitwise exact；同一 accepted run 内的匹配双臂接受相同 token-ID chunks 与操作序列后，K/V、logits、mask、ledger 及 role/end/content state 亦逐步 exact。该证据仅支持 direct crop integrity 与 within-run matched-arm recovery exactness；v1/v2 clean-reprefill 协议仍按冻结门槛 rejected，v3 不建立 clean-reprefill、continuation 或跨环境等价性。

固定轨迹 E3 作为 C2 的下游支持证据。label-weighted generation−playback 差在 fragment/rule、fragment/judge、proxy/rule 和 proxy/judge 四个并列操作化中分别为 −3.37、−2.02、−1.58 和 −2.63 个百分点，对话聚类 95% CI 均跨零；dialogue-weighted 与 target-specific exact-key 去重用于敏感性分析。支持性 C1 只刻画同步分段文本 harness 中的 oracle 接受前候选生成：token-consistent C-E2 的 candidate-readiness 差为 −0.03 ms（crossed 95% CI [−0.64, 0.61]），同步 oracle 时延下界差为 +20.80 ms（[17.85, 23.65]）；B@0.92 的接受时候选可用率为 335/500，pooled discarded-token ratio 为 2.85%。C-E1 是非 token-equivalent implementation-path comparison。探索性 C3 因生成轨迹混杂不支持策略因果、负向或零效应结论。本文结果不构成人类语义、HCI、真实端点前收益或 production deliverability 证据。

**关键词**：级联式语音对话；用户打断；软件播放游标；联合前缀状态修正；KV 缓存裁剪；角色状态恢复

## Abstract

Barge-in creates asynchronous progress across language-model generation, speech synthesis, and software playback in cascaded spoken-dialogue systems. Playback-conditioned history truncation and KV-cache cropping are established; neither is claimed as novel. We study external-progress-conditioned joint prefix-state repair: a software-consumed-sample cursor resolves a legal TTS-fragment and assistant-token commit boundary, after which the KV cache, attention mask, token ledger, position indices, and role/EOT state are repaired jointly. The contract comprises boundary resolution, joint-state representation, invariant-preserving transitions, and falsifiable validation.

The sole core contribution, C2, implements the path from software cursor to TTS fragment, assistant token span, KV crop, and role recovery. Under a frozen Qwen2-7B-Instruct/Transformers/BF16/SDPA configuration, v3 covered 24 cases, 27 crop events, and 60 recovery steps. For every event, the retained pre-crop prefix and production post-crop state were bitwise exact across 28 K/V layers against an oracle that sliced the same snapshot without calling the production crop interface. Within the accepted run, matched arms receiving identical token-ID chunks and operations also remained stepwise exact in K/V, logits, masks, ledgers, and role/end/content state. These results support only direct crop integrity and within-run matched-arm recovery exactness. Earlier clean-reprefill protocols remain rejected, and v3 establishes neither clean-reprefill nor cross-environment equivalence.

As downstream evidence, fixed-trajectory E3 produced generation-minus-playback differences of −3.37, −2.02, −1.58, and −2.63 percentage points across four target–detector operationalizations; all dialogue-cluster 95% confidence intervals crossed zero. Supporting C1 characterized pre-oracle-acceptance candidate generation in a synchronous segmented-text harness. In token-consistent C-E2, the candidate-readiness difference was −0.03 ms (crossed 95% CI [−0.64, 0.61]), whereas the optimistic oracle-latency lower-bound difference was +20.80 ms ([17.85, 23.65]). Endpoint candidate availability was 335/500, with a 2.85% pooled discarded-token ratio. C-E1 compared non-token-equivalent implementation paths. These automated, software-level results do not establish human-semantic or HCI effects, benefits before real end of speech, device/acoustic boundaries, or production deliverability.

**Keywords**: cascaded spoken dialogue; barge-in; software playback cursor; joint prefix-state repair; KV-cache cropping; role-state recovery
