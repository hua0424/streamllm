# 第二章 相关工作

本文相关工作横跨五个相邻问题：播放条件下的会话历史截断、流式与增量语音对话、用户打断与话轮控制、话轮结束前的候选响应计算，以及 KV 缓存裁剪与前缀复用。高层播放感知历史原则和缓存裁剪原语均已有先例；本章据此界定本文跨层实现的范围，而不以单一组件的新颖性立论。

## 2.1 播放条件下的 transcript 与 session-history 截断

OpenAI Realtime API 的 `conversation.item.truncate` 允许客户端提交 `audio_end_ms`，从会话条目中移除未播放音频及其对应 transcript[1]。Azure Voice Live 的 `auto_truncate` 在播放期间检测到用户语音后更新上一轮响应和 session context；其公开文档明确说明截断估算采用实时播放速度假设[2]。LiveKit Agents 也提供截断被打断 transcript/history、使消息状态与 spoken output 相匹配的框架语义[3]。这些资料共同表明，按播放进度修正会话历史是既有工程实践，而非本文提出的新原则。

公开接口语义不能揭示闭源服务的内部推理架构。OpenAI 和 Azure 是否采用级联模型、如何表示 token span、是否原地裁剪 KV，均不能从其文档推断。LiveKit 公开框架层消息处理，但其相关资料不公开 transformer KV、attention mask、token ledger 与 role/EOT state 的联合恢复。因此，本文与这些系统的区别是公开研究对象和证据层级，而不是声称商业或开源框架“没有”某项未公开内部机制。

上述系统使用的播放概念也不完全等同于物理听觉测量。Azure 的实时播放速度假设和 OpenAI 的客户端 `audio_end_ms` 都属于接口或软件时序语义；设备缓冲和声学传播仍需独立测量。本文同样只观测 software-consumed-sample cursor，并将其映射为 TTS-fragment-level software retention boundary。该操作不提供 device-presented samples 或 acoustically heard content 的真值。

## 2.2 流式语音对话、提前计算与打断控制

流式级联研究通常通过语义触发、增量推理或输入预测缩短等待。LTS-VoiceAgent 使用 semantic triggering 和 incremental reasoning 组织 Listen–Think–Speak 流程[4]。RelayS2S 采用双路径 response-level candidate prefix 与验证/续写机制[5]。Personalized Predictive ASR 从 partial ASR 预测完整输入，并预取下游响应；最终识别结果确认预测后采用缓存结果[18]。三者均构成“在输入最终确认前进行下游计算”的先例，但研究信号位于用户输入或候选响应侧，未公开由 assistant 播放游标触发的 token/KV 历史修正。

本文的 supporting C1 与这类工作共享 compute-before-commit 思路，但术语和结论范围更窄。确认性实证对象是同步分段文本 harness 中的 **pre-oracle-acceptance candidate generation with invalidation**，而不是固定 prompt 上由 draft model 与 target model 协作的 speculative decoding，也不识别候选是否在真实 end of speech 之前就绪。`first_token_ready` 是 first-candidate-token selection/candidate compute-readiness；`endpoint_accept` 是候选处理后的同步 oracle acceptance。该实验刻画接受时候选可用率、pooled discarded-token ratio 和 oracle $\mathrm{TTFT}_{\mathrm{eff}}$ 乐观下界，不证明 production consumer delivery、TTS admission 或 acoustic output 得到改善。

打断检测研究主要回答何时停止系统输出。FireRedChat 使用 streaming/personalized VAD 与 interruption control，并在确认打断后控制 TTS[7]。这种检测与本文的状态修正互补：前者产生或确认 interruption event，后者在事件到达后决定 software-fragment prefix 及相关模型状态如何保留。将两者区分可避免把“检测到打断”误写为“完成了多轮历史修复”。

端到端全双工系统提供另一类架构参照。Moshi 通过并行 speech/text streams 支持重叠交互[6]。这种同步建模减少级联 ASR、LLM 和独立 TTS 之间的部分中间错位，但并不据此证明模型流位置、设备呈现位置和声学听觉位置恒等。网络、应用队列、音频 API 和设备缓冲仍可能形成交付差异，因此端到端与级联设计不能仅凭架构标签完成听觉边界比较。

## 2.3 KV 缓存裁剪、前缀复用与跨轮状态

自回归 transformer 的 KV cache 保存历史 token 的中间状态，以免每一步重复计算完整前缀。Hugging Face Transformers 提供 `DynamicCache` 及 `crop` 操作，可将缓存缩短至指定序列长度[8]。因此，KV crop primitive 本身不是本文创新。

PagedAttention 将 KV cache 组织为非连续固定大小块，并通过按需分配与 copy-on-write 改善 serving memory efficiency[19]。SGLang 的 RadixAttention 以 radix tree 管理可跨请求复用的公共 token 前缀[20]。这些方法建立了 KV 内存管理和 prefix reuse 的重要先例，其优化目标主要是吞吐、内存利用或公共前缀共享，而非依据外部 software playback cursor 选择被打断 assistant 的对话提交边界。

IntentKV 处理 text-agent 的 cross-turn intent-aware KV pruning[9]；Speculative Interaction Agents 研究异步工具调用中的推测结果和作废控制[10]。这些工作表明，跨轮 KV 或推测状态可以受外部控制逻辑影响，但其信号来自文本意图或工具状态，不涉及 software-consumed samples、TTS fragment、assistant content span 与 chat role/EOT state 的联合恢复。

本文的技术对象因而不是一种新的缓存数据结构，而是**外部进度条件下的联合前缀状态修正合同**。该合同包含四层：软件游标经 TTS 片段解析为合法 assistant commit boundary；KV、attention mask、完整 token ledger、assistant content span、position 和 role/EOT 构成联合状态；crop 与 close/reopen 必须保持联合不变式；direct slicing oracle、wrong-length negative control 和 matched recovery 提供可证伪检查。C2 v3 检验的仅是每层 K/V 的 direct crop integrity，以及同一 accepted run 内从匹配保留状态出发、接收相同 token-ID chunks 与操作序列时的 matched-arm recovery exactness。该证据只覆盖受测 snapshot/backend，不建立 clean-reprefill numerical equivalence，也不能替代在线音频或跨引擎验证。

## 2.4 Targeted Public-Source Novelty Scan

### 2.4.1 范围、来源与查询族

为评估组合式贡献，本文于 2026-09-03 截止开展 targeted public-source scan，而非 systematic review 或 patent search。目标问题是：公开资料是否同时披露 cascaded ASR→LLM→TTS、software-consumed-sample cursor、cursor→TTS fragment→assistant token span mapping、interruption 后的 in-place KV crop、attention mask/token ledger/role/EOT recovery，以及可复算 crop-integrity/recovery evidence。

检索渠道包括 Google Scholar 或等价跨出版商索引、arXiv、ACL Anthology、ISCA Archive/Interspeech、ACM Digital Library、IEEE Xplore、NeurIPS Proceedings、DOI/Crossref 与出版商元数据页，以及 OpenAI、Microsoft Azure、LiveKit、Hugging Face 和 GitHub 的第一方资料。查询分为五族：

1. cascaded/streaming spoken dialogue、ASR–LLM–TTS 与 latency；
2. barge-in、interruption、turn-taking、endpointing 与 history/context；
3. predictive/speculative response generation、partial ASR 与 before-end-of-turn computation；
4. KV cache crop/truncate/rollback、`DynamicCache.crop`、chunked prefill 与 prefix reuse；
5. playback/listening-aware history、`audio_end_ms`、`conversation.item.truncate`、`auto_truncate` 和 playback/audio cursor。

纳入资料至少涉及上述一个相邻机制，并具有 DOI、官方出版页、官方预印本或第一方 URL。排除项包括纯 draft-target speculative decoding、与对话或 rollback 无关的通用 KV 压缩、只处理 echo cancellation 的 barge-in、仅停止音频却未说明历史语义的播放 API、可由第一方来源替代的二手博客，以及根据营销材料猜测闭源内部架构的陈述。Snowballing 起点包括 Predictive ASR、LTS-VoiceAgent、RelayS2S、FireRedChat、Moshi、OpenAI/Azure/LiveKit truncation 文档、PagedAttention、SGLang 和 Transformers DynamicCache。

本轮检索存在可复查性限制。2026-05-21 的既有 22-source novelty 核查保留了来源与逐项结论，但未保存完整原始查询日志、逐库结果数、去重和逐条排除记录。2026-09-03 补查冻结了查询族，但 Google、ACL、ACM、部分 arXiv/官方页面等渠道出现 reCAPTCHA、超时、403 或工具访问限制；这些渠道不能据此记为“零结果”。因此，下述结论只是在所报告公开来源中的限定性非识别。

### 2.4.2 最近邻与组合边界

**表 2-1　公开最近邻与本文组合机制的边界**

| 来源 | 公开建立的相邻机制 | 未由公开资料共同建立的本文组合要素 |
|---|---|---|
| OpenAI Realtime API[1] | 客户端以 `audio_end_ms` 截断音频与对应 transcript | 未公开 cascaded stack、assistant token span、KV crop 或 role recovery |
| Azure Voice Live[2] | `auto_truncate` 在播放期打断后更新 session context；采用实时播放速度假设 | 未公开内部 token/KV 实现；不能推断设备或声学精度 |
| LiveKit Agents[3] | interrupted transcript/history 与 spoken output 对齐的框架语义 | 未公开 transformer KV、mask、ledger 与 role-state recovery |
| LTS-VoiceAgent[4] | semantic triggering 与 incremental reasoning | 未公开 playback-cursor-driven KV correction |
| RelayS2S[5] | response-level candidate prefix 与验证/续写 | 所审阅公开资料未报告 interruption 后的 playback-history repair |
| Predictive ASR[18] | partial-ASR prediction 与 downstream response prefetch | 所审阅公开资料未报告被打断 assistant 的保留边界修正 |
| FireRedChat[7] | streaming/personalized VAD 与 interruption control | 主要回答何时停止，未公开停止后的 KV/role repair |
| Moshi[6] | speech/text stream 下的全双工重叠交互 | 不能据此推出 device/acoustic delivery 与 model stream 恒等 |
| Transformers DynamicCache[8] | cache abstraction 与 crop primitive | 不提供 cursor→fragment→token 映射或对话状态证据 |
| PagedAttention[19] / SGLang[20] | KV memory management、prefix sharing/reuse | 所审阅公开资料未以 software playback cursor 选择 assistant commit boundary |
| IntentKV[9] | text-agent cross-turn intent-aware KV pruning | 控制信号不是 speech delivery，未覆盖 TTS/role recovery |
| 本文 | software cursor→TTS fragment→assistant token span→in-place KV crop→explicit role/EOT recovery | 证据限于受测软件边界、模型 snapshot/backend 与受控实验 |

截至 2026-09-03，在所报告的公开来源范围内，未识别到一个可检视的级联实现同时公开上述完整状态路径，并提供可复算 direct state-integrity 与 latency artifacts。该陈述不是“全球首次”，也不否认未发表、闭源或因索引访问受限而未收录的系统。完整查询式、来源状态、纳排规则和逐项最近邻说明见 `docs/novelty_search_2026-09-03.md`。

## 2.5 本文定位

相关工作支持分层而非并列的贡献定位。第一，C2 是唯一核心机制，其增量位于外部软件进度、合法提交边界、联合模型状态、角色/EOT 转换和 exact gates 的公开联结，而不在 playback-history 原则或 KV crop primitive。固定轨迹 E3 是 C2 的 downstream 支持性证据，只估计冻结检测器条件下的信息复现。第二，C1 是 supporting characterization：C-E2 报告 token-consistent B 路径内的 candidate selection/compute-readiness、同步 oracle acceptance、endpoint candidate availability 与 pooled discarded-token 工作点；C-E1 审计非 token-equivalent implementation paths，不声称 speculative decoding 或 production latency improvement。第三，C3 是 exploratory extension：A2 对三种历史自然化实现作受混杂的描述，不构成策略优越、负向或零效应证据。

这一定位也限定了可推广性。当前证据适用于 software-consumed-sample cursor 和 TTS-fragment boundary，不适用于 device-presented 或 acoustically heard truth；C2 v3 适用于受测 Qwen2-7B snapshot/backend，不适用于 clean-reprefill equivalence 或跨引擎正确性；C1/E3 适用于同步受控 harness 与冻结 detector/judge 条件，不适用于真实异步语音闭环或 HCI 效果。

## 2.6 本章小结

播放条件下的 transcript/session-history truncation、输入结束前的下游计算、打断检测、KV crop 和 prefix reuse 均有公开先例。本文不将这些单项重新包装为原创，而聚焦其间尚未在本次公开检索中共同识别的实现契约：由 software-consumed-sample cursor 选择 TTS-fragment-level assistant prefix，原地裁剪 KV，并同步恢复 mask、token、position 与 role/EOT state。下一章据此形式化软件边界、状态不变式和分层评测指标。
