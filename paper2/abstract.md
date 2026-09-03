# 摘要

## 中文摘要

级联式语音对话系统在用户打断时需要协调语言模型生成、语音合成与软件播放的异步进度。OpenAI、Azure 和 LiveKit 已公开按播放进度截断 transcript 或 session history 的高层实践，KV 裁剪也是既有原语。本文不主张这些原则或原语的原创性，而聚焦一个可检视的软件状态修正问题：从 software-consumed-sample cursor 定位 TTS 文本片段和 assistant token 保留边界，再同步修正 KV 缓存、attention mask、token ledger、position 与 role/EOT state。该边界是 TTS 片段级的软件保留边界，不代表设备已呈现采样或用户声学上实际听到的内容。

核心贡献 C2 实现 software cursor→TTS fragment→assistant token span→KV crop→role recovery 的级联路径。Qwen2-7B-Instruct 上的 v3 direct crop-integrity addendum 在受测 snapshot/backend 下覆盖 24/24 cases、27/27 crop events、3 个 no-op 和 60 个 recovery steps。每次裁剪的保留前缀、production post-crop 状态与独立切片 oracle 在 28 层 K/V 上逐层精确一致；输入相同 token-ID chunks 后，K/V、logits、mask、token ledger 及 role/end/content state 亦精确一致。该证据支持 direct crop integrity 与 matched-recovery determinism，不支持 clean-reprefill numerical equivalence 或跨模型、后端、硬件及在线音频正确性。此前 v1/v2 clean-reprefill 协议均按冻结门槛 rejected；v2 数值门仅通过 42/45，且 control 与 production forward topology 不匹配，v3 不改变其 verdict。

固定轨迹 E3 将后续信息复现定义为 fixed-detector-conditioned 指标。label-weighted 主分析中，generation−playback 的 fragment/rule、fragment/judge、proxy/rule 和 proxy/judge 差异依次为 −3.37、−2.02、−1.58 和 −2.63 个百分点，对话聚类 95% CI 均跨零。按 target-specific unique semantic boundary 去重后，fragment/rule 为 −2.96 个百分点，fragment/judge 为 0.00 个百分点；proxy 仅移除一个重复，结果基本不变。这些结果不支持优效、等效、非劣、伤害或 absence-of-effect 推断，也不构成人类感知或 HCI 结论。

支持性贡献 C1 刻画话轮结束前候选响应生成的 candidate selection/compute-readiness、post-candidate oracle acceptance 与 wasted-token 工作点，而非 speculative decoding 或 production deliverability。确认性设计包含 100 条唯一话语、5 个独立进程 session 和每条件 500 个交叉观测。C-E1 是两条整体 implementation path 的比较：System A 与 B@0.92 的完整输出 token 仅 280/500 一致，A−B 的 arrival→candidate-selection 差为 −34.69 ms（crossed 95% CI [−35.44, −33.95]），不能归因于纯 incremental-prefill effect。token-consistent 的 C-E2 中，never−B@0.92 的 candidate-readiness 差为 −0.03 ms（95% CI [−0.64, 0.61]）；oracle TTFT_eff 乐观下界差为 +20.80 ms（95% CI [17.85, 23.65]），对应 67.0% survival 和 2.85% pooled waste。first-deliverable 与 consumer 时间仅为同步 harness diagnostics。探索性扩展 C3 实现朴素、标记与重写三种历史自然化路径；受条件间生成轨迹混杂影响，现有负结果不支持策略因果比较。

**关键词**：级联式语音对话；软件播放游标；TTS 文本片段；KV 缓存裁剪；角色状态恢复；候选响应生成

## Abstract

Barge-in creates asynchronous progress across language-model generation, speech synthesis, and software playback in cascaded spoken-dialogue systems. Playback-conditioned history truncation and KV-cache cropping are established. This study addresses their inspectable integration: a software-consumed-sample cursor selects a TTS-fragment and assistant-token retention boundary, after which the KV cache, attention mask, token ledger, position indices, and role/EOT state are updated together. The boundary is software- and fragment-level; device-presented samples and acoustically heard content are not measured.

The core contribution, C2, implements this state contract. On Qwen2-7B-Instruct, a direct crop-integrity addendum covered 24 cases and 27 crop events. For every event, the retained pre-crop prefix, production post-crop cache, and independent slicing oracle were bitwise identical across 28 K/V layers. Sixty matched recovery steps also produced exact K/V, logits, masks, token ledgers, and role/end states. This supports direct crop integrity and matched-recovery determinism for the tested snapshot/backend, not clean-reprefill numerical equivalence or cross-system correctness; two earlier clean-reprefill protocols remain rejected under their frozen gates.

E3 measured fixed-detector-conditioned information reproduction. Label-weighted generation-minus-playback effects for fragment/rule, fragment/judge, proxy/rule, and proxy/judge were −3.37, −2.02, −1.58, and −2.63 percentage points, with all dialogue-cluster 95% confidence intervals crossing zero. Deduplicating exact semantic boundaries changed the fragment effects to −2.96 and 0.00 points and left proxy results nearly unchanged. These automated measurements establish neither superiority, equivalence, noninferiority, harm, nor human-perception effects.

The supporting contribution, C1, characterizes pre-end-of-turn candidate computation rather than production deliverability. Across 100 utterances and five process sessions, C-E1 compared non-token-equivalent implementation paths: full outputs matched in 280/500 observations, and System-A-minus-B@0.92 candidate-readiness was −34.69 ms (crossed 95% CI [−35.44, −33.95]). In token-consistent C-E2, never-minus-B@0.92 readiness was −0.03 ms [−0.64, 0.61], while the synchronous-oracle TTFT_eff lower-bound difference was +20.80 ms [17.85, 23.65], with 67.0% survival and 2.85% pooled waste. C3 remains an exploratory negative extension because its history-policy trajectories are confounded.

**Keywords**: cascaded spoken dialogue; software playback cursor; TTS text fragment; KV-cache cropping; role-state recovery; candidate-response generation
