# 级联式语音对话打断中的上下文状态修正：从软件播放游标和 TTS 片段到 KV 与角色恢复

> 全文合并草稿（自动生成，勿直接编辑；请修改分章 Markdown 后重新合并）。
> 本文件依据 crossed E1/E2 analysis v2、E3 weighting/dedup analysis v2、accepted C2 v3 及内部审阅意见统一更新。

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
---

# 第一章 绪论

## 1.1 研究背景与问题界定

级联式语音对话系统将流式语音识别（Automatic Speech Recognition，ASR）、大语言模型（Large Language Model，LLM）和流式语音合成（Text-to-Speech，TTS）组合为可独立选择与调优的模块。与端到端语音模型相比，这一路线便于替换组件、观测中间状态并控制部署资源；其交互时延则同时受 ASR 分段、LLM 推理、TTS 产出和播放调度影响。

前期工作使用 ASR 稳定文本段和 LLM KV 缓存增量预填充，减少用户话轮结束后的上下文编码等待。本研究转向输出侧的状态一致性：用户在 assistant 播报期间打断时，模型生成、TTS 合成和软件播放处于不同进度。若系统把软件游标尚未覆盖的生成内容完整写入下一轮历史，模型可能基于未通过该软件播放路径交付的信息继续生成；若删除整轮 assistant 状态，又会丢弃应保留的前缀及其 KV 计算。

本文严格区分 `software-consumed samples`、`device-presented samples` 与 `acoustically heard content`。前者是播放器软件报告的已消费采样；后两者还受音频 API、系统与设备缓冲、传输和声学传播影响。本文只观测第一层。因此，“播放感知”仅指 software-consumed-sample cursor 驱动的 TTS 文本片段级状态操作，不表示逐词或逐 token 的设备呈现或听觉真值。

依据播放进度更新 transcript/session history 已有明确先例。OpenAI Realtime API 允许客户端按 `audio_end_ms` 截断未播放音频及对应 transcript[1]；Azure Voice Live 的 `auto_truncate` 在播放期打断后更新会话上下文，并说明其估算采用实时播放速度假设[2]；LiveKit Agents 也提供使被打断 transcript/history 与 spoken output 相匹配的框架语义[3]。本文既不主张“历史应反映已播放输出”这一高层原则的原创性，也不把 Hugging Face `DynamicCache.crop` 等缓存裁剪原语视为创新[8]。研究对象是**外部进度条件下的联合前缀状态修正合同**：如何把软件游标解析为合法 assistant 提交边界，并在该边界上共同修正 KV、attention mask、token ledger、position 以及 role/EOT 状态。

## 1.2 贡献层级、研究问题与证据结构

本文以 C2 为唯一核心机制贡献，E3 为其 downstream 支持性证据，C-E2 与 C-E1 为支持性 C1 的工作点刻画和路径审计，A2 为探索性 C3 的描述。研究问题按这一证据层级排列：

1. **RQ1（C2）：冻结模型与后端下，外部软件进度驱动的联合前缀状态修正是否满足 direct crop integrity 与同一 accepted run 内的 matched-arm recovery exactness，其固定协议成本和 prepared-state 软件控制路径时延如何？**
2. **RQ2（E3）：在固定首轮轨迹、固定自动检测器和目标特定资格规则下，software-cursor retention 与 generation retention 的后续信息复现率有何差异？**
3. **RQ3（C-E2）：在 token-consistent 的 B 路径内，阈值如何影响同步 oracle 接受前的候选生成、接受时候选可用率、pooled discarded-token ratio 与 $\mathrm{TTFT}_{\mathrm{eff}}$ 乐观下界？**
4. **RQ4（C-E1）：在同步分段文本 harness 中，两条非 token-equivalent implementation paths 的 candidate selection/compute-readiness 及相关诊断量有何差异？** 该比较不识别单一 incremental-prefill 因果效应。
5. **RQ5（A2）：当前探索性运行中，朴素保留、打断标记与重写三种历史自然化实现的连贯性分数和重写耗时如何？** 条件间首轮生成轨迹存在混杂，因而不检验策略改善或零效应。

**表 1-1　贡献—研究问题—实验—证据层级**

| 层级 | 研究问题 | 实验 | 可支持的证据 |
|---|---|---|---|
| 核心 C2 | RQ1 | C2 v3、A1、P1 | direct crop integrity、within-run matched-arm recovery exactness；固定协议成本与软件控制路径描述 |
| C2 downstream 支持 | RQ2 | E3 | fixed-detector-conditioned information reproduction 及加权/去重敏感性 |
| 支持性 C1 | RQ3 | C-E2 | token-consistent B 路径内的有限阈值工作点与同步 oracle 下界 |
| 支持性 C1 路径审计 | RQ4 | C-E1 | 非 token-equivalent 整体实现路径差异，不是纯增量预填充效应 |
| 探索性 C3 | RQ5 | A2 | 受混杂的实现与描述性结果，不作策略因果比较 |

这一层级避免把不同构念合并为端到端效果证明。C2 的 exact gates 检验联合状态转换的受测性质；E3 仅考察固定检测器条件下的 downstream 信息复现；C-E2/C-E1 刻画同步文本 harness 的候选计算；A1/P1 是固定协议或 headless 软件路径计时；A2 只作探索性描述。缓存长度一致只是结构合法性的必要条件，软件游标也不是设备或听觉测量，内部 candidate-readiness 与生产可交付性必须分开报告。

## 1.3 本文工作与贡献

### 1.3.1 核心贡献 C2：外部进度条件下的联合前缀状态修正

C2 将打断处理组织为四层可检验合同。第一，**边界解析**：通过 $p\xrightarrow{\Phi}$ TTS fragment 将 software-consumed-sample cursor 映射为合法 assistant commit boundary。第二，**联合状态**：将 KV、attention mask、完整 token ledger、assistant 内容 span、position 和 role/EOT 视为不可分割状态，而非只操作 KV。第三，**不变式保持转换**：裁剪、结构性 close 和 user-role reopen 必须维持长度、内容 span、位置连续性和角色语义；预测 EOT 不进入 assistant 内容账本或内容 KV，仅进入 `ASSISTANT_EOT_PENDING`，再由唯一 close 入口提交一次。第四，**可证伪验证**：从同一 pre-crop snapshot 逐层切片、但不调用 production crop 接口的 oracle，wrong-length negative control 和 matched recovery 分别检查保留前缀、gate 灵敏度与后续恢复。

在受测 Qwen2-7B-Instruct snapshot、Transformers backend、BF16 和 SDPA 条件下，C2 v3 覆盖 24 个 case、27 次 crop event、3 个 no-op 和 60 个 recovery step。每次事件的 pre-crop retained prefix、production post-crop 状态与 slicing oracle 在 28 层 K/V 上逐层精确一致；keep length、mask、token ledger、sequence length 与 KV length 也一致。同一 accepted run 内的两条匹配臂从精确匹配的保留状态出发，接受相同 token-ID chunks 和相同操作序列后，K/V、logits、mask、token ledger、retained prefix 与 role/end/content state 逐步精确一致。该证据仅支持 direct crop integrity 与 within-run matched-arm recovery exactness。

v1/v2 clean-reprefill 协议均按冻结门槛 rejected。由于 v2 control 与 production forward topology 不匹配，其数值失败既不能定位 crop bug，也不能建立 clean-reprefill equivalence。v3 改为回答可由 direct slicing oracle 识别的问题，不放宽旧门槛，也不改变 v1/v2 判定；它不支持 clean-reprefill numerical equivalence、32-token continuation equivalence或跨模型、后端、硬件、在线音频与生产系统正确性。

A1 与 P1 只补充成本证据。A1 比较固定操作顺序、固定移除 32-token suffix 的 joint crop+role 与重新预填充；P1 描述 prepared-state、headless software cursor 控制路径。二者均不测量声卡、声学停播或生产端到端 barge-in，P1 的每 cell P95 也只是经验顺序统计量而非生产 SLO。

### 1.3.2 C2 的 downstream 支持性证据 E3

固定轨迹 E3 比较 generation retention 与 software-cursor fragment retention 对后续信息复现的影响。四个注入位置为 0.25、0.5、0.75 以及 fragment boundary；fragment 与 character-proportional whitespace-snapped proxy 分别按其目标字段非空确定资格。主 estimand 对每个 eligible `(dialogue, injection position)` 等权，dialogue-weighted 结果先在对话内平均再对对话等权，target-specific exact-key 去重则作为敏感性分析。fragment/proxy 与 lexical rule/`specific-reference-v3` judge 的四个 target×detector 单元是并列的冻结操作化，而非人类语义 reference standard。

label-weighted 主分析中，generation−playback 的四个点估计均低于零且对话聚类 95% CI 均跨零。该结果只描述冻结目标、轨迹、规则、judge、prompt 与 40-token cap 下的 fixed-detector-conditioned information reproduction；不支持 superiority、equivalence、noninferiority、harm、absence-of-effect 或 HCI 推断。去重键由 dialogue/trajectory identity、两条件 history key、目标哈希及 fragment 口径下的 `heard_token_end` 精确组成，因此本文称其为 **target-specific exact-key grouping/deduplication**，不称语义聚类。

### 1.3.3 支持性贡献 C1：同步 harness 中的候选生成刻画

C1 在同步分段文本 harness 中研究 **pre-oracle-acceptance candidate generation with invalidation**，而非证明候选在真实 end-of-speech 前就绪。`first_token_ready` 表示 first-candidate-token selection/candidate compute-readiness；`endpoint_accept` 是候选处理后的同步 oracle 事件。first-deliverable 与 consumer markers 只用于诊断程序执行顺序，不表示 production delivery、TTS admission 或声学输出。

C-E2 在 token-consistent 的 B@0.92 与 B-never 之间比较阈值策略，报告 candidate-readiness、同步 oracle $\mathrm{TTFT}_{\mathrm{eff}}$ 乐观下界、接受时候选可用率和 pooled discarded-token ratio。接受时候选可用率以全部 condition records 为分母，而不是 $P(\text{available}\mid\text{candidate launched})$；pooled discarded-token ratio 在每个 bootstrap replicate 内按 ratio of sums 计算，不代表 FLOPs、GPU 时间、能耗或显存带宽浪费。正式不确定性使用 crossed/product bootstrap，对 100 条唯一话语与 5 个独立初始化进程 session 分别重采样后取笛卡尔积。

C-E1 比较一次性 full-string/full-prefill 与 segment-wise/incremental 两条整体 implementation paths。两者并非 token-equivalent，差异混合 tokenization、forward topology/shape、role boundary、kernel 与 Python scheduling。因此，C-E1 只能作为实现路径审计，不能归因于纯 incremental-prefill effect，也不能按输出是否匹配进行 post-treatment selection。

### 1.3.4 探索性贡献 C3：历史自然化实现

本文实现朴素保留、显式打断标记和轻量模型重写三种历史自然化路径，并记录自动连贯性评分与重写耗时。A2 的不同条件使用了不同首轮和后续生成轨迹，策略效应与内容差异不可分离。因此，该实验仅是受混杂的探索性描述，不能支持策略优越、负向或零效应结论。

## 1.4 研究定位与 novelty 边界

截至 2026-09-03，本文开展了 targeted public-source scan，覆盖跨出版商学术索引、arXiv、ACL Anthology、ISCA Archive/Interspeech、ACM Digital Library、IEEE Xplore、NeurIPS Proceedings、DOI/Crossref 元数据、第一方产品文档和开源仓库。查询族涉及 cascaded/streaming spoken dialogue、barge-in/turn-taking、候选响应提前计算、KV crop/rollback/prefix reuse 以及 playback/listening-aware history。部分渠道受反自动化、超时或访问控制限制，访问失败未被解释为零结果。

检索识别出 playback-conditioned transcript/session-history truncation 的 OpenAI、Azure 和 LiveKit 先例[1–3]，输入侧提前计算与候选响应的 Predictive ASR、LTS-VoiceAgent 和 RelayS2S[4,5,18]，打断检测与全双工架构的 FireRedChat 和 Moshi[6,7]，以及 `DynamicCache.crop`、PagedAttention、RadixAttention 和跨轮 KV pruning 等缓存先例[8,9,19,20]。在所报告公开来源范围内，未识别到同时公开 software cursor→TTS fragment→assistant token span→in-place KV crop→explicit role/EOT recovery，并提供可复算 direct crop-integrity、matched-recovery 与 latency evidence 的级联实现。这是范围受限的非识别结果，不排除未发表、闭源或未收录系统，也不构成全球首次声明。

## 1.5 论文组织结构

第二章界定播放条件下历史截断、候选提前计算、打断控制和 KV 状态管理的已有基础；第三章形式化软件游标、片段边界、联合状态合同及分层 estimand；第四章给出候选调度、联合裁剪与角色恢复、v3 exact gates 以及 E3/A2 方法。后续章节按 C2→E3→C-E2→C-E1→A2 的证据顺序报告实现与结果，并集中讨论构念、内部与外部效度。
---

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
---

# 第三章 问题形式化

## 3.1 系统模型与符号

### 3.1.1 级联流水线

考虑级联式流式语音对话系统

$$
\mathcal{S}=\langle \mathrm{ASR},\mathrm{LLM},\mathrm{CHK},\mathrm{TTS},\mathrm{PLY}\rangle,
$$

其中各模块按流水线衔接。

- **流式 ASR** 将用户语音转写为稳定文本段序列 $U=\langle u_1,u_2,\ldots\rangle$。本文下游仅接收不再修正的 final segment，而不直接消费可撤销的 partial transcript。
- **LLM** 维护对话上下文的 KV 缓存 $\mathcal{K}$。用户文本段到达后增量预填充，assistant 回复以零起始的内容 token 序列 $Y=\langle y_0,\ldots,y_{G-1}\rangle$ 逐步生成。
- **断句器 CHK** 将 assistant 内容 token 流切分为 TTS 文本片段 $F=\langle f_1,\ldots,f_m\rangle$。每个片段关联一个左闭右开的 token 区间：

$$
f_j\mapsto[\operatorname{ts}(f_j),\operatorname{te}(f_j)),\qquad
\operatorname{ts}(f_1)=0,\quad
\operatorname{ts}(f_{j+1})=\operatorname{te}(f_j).
$$

  本文以 TTS 文本片段作为历史裁剪的原子单位。
- **流式 TTS** 接收完整文本片段并输出一个或多个音频块。片段 $f_j$ 在累计软件音频轴上占据采样区间 $[\operatorname{ss}(f_j),\operatorname{se}(f_j))$，采样率记为 $r$。
- **播放器 PLY** 顺序消费已登记音频块并维护软件已消费采样游标 $p\in\mathbb{N}$。$p$ 采用计数语义，即软件播放器已经消费区间 $[0,p)$。

本文固定使用“ASR 稳定文本段”表示 $u_i$，使用“TTS 文本片段”表示 $f_j$，以避免两类片段混淆。

### 3.1.2 三层播放语义与异构进度

本文严格区分三个层级：

1. **软件已消费采样（software-consumed samples）**：播放器线程或 headless harness 报告已经消费的采样数，即本文的 $p$；
2. **设备已呈现采样（device-presented samples）**：已经通过音频 API、操作系统、驱动和设备缓冲并由设备呈现的采样；
3. **声学上被听到的内容（acoustically heard content）**：经过扬声器与传播路径后到达用户并可能被感知的内容。

本文只观测第一层。由于应用队列、音频 API、操作系统、驱动、设备及传播路径均可能引入尚未测量的缓冲或延迟，$p$ 不等于设备已呈现采样数，也不构成用户声学上实际听到内容的真值。“播放感知”在本文中因此专指**软件已消费采样游标驱动的 TTS 片段级状态操作**。

在时刻 $t$，系统同时维护三种原始进度：生成内容 token 端点 $G(t)$、已登记到 TTS/软件播放时间轴的最大文本片段 token 端点 $S(t)$，以及软件已消费采样游标 $p(t)$。三者分别处于 token 域和采样域，不能直接写成 $p(t)\leq S(t)\leq G(t)$。

系统只把软件游标映射到**命中的 TTS 文本片段**，并以该片段的 token 末端作为软件保留边界 $\widehat H(p)$。在“软件播放器只消费已登记片段、片段按生成顺序入队”的假设下，比较对象均转换为 token 端点后有

$$
\widehat H(p(t))\leq S(t)\leq G(t). \tag{3-1}
$$

式（3-1）描述的是本文软件时间轴的生产者顺序约束，而不是设备或声学播放规律。只要打断时 $\widehat H(p)<G$，按生成边界保留历史就可能纳入软件游标尚未覆盖的片段内容。

端到端帧同步模型可以减少独立 TTS 引入的错位，但网络和播放缓冲仍可能使模型产出、设备呈现与声学到达位置不同；因此本文不将“端到端”简单等同于三种进度完全一致。

![图 3-1](figures/fig3_1.png)

**图 3-1　三种异构进度与片段级软件保留边界。** 软件已消费采样游标 $p$ 落在片段 $f_3$ 内，片段级保留边界吸附到 $\widehat H(p)=\operatorname{te}(f_3)$。片段中游标尚未覆盖的文本尾部由定义 3.3 的字符比例—空白边界代理估计；该图不表示设备呈现或声学听觉真值。

## 3.2 软件游标与片段级历史对齐

**定义 3.1（TTS 片段级软件保留边界）**　若软件已消费采样游标 $p$ 落在片段 $f_k$ 的累计采样区间内，即

$$
\operatorname{ss}(f_k)<p\leq\operatorname{se}(f_k),
$$

则定义

$$
\widehat H(p)=\operatorname{te}(f_k). \tag{3-2}
$$

若 $p=\operatorname{se}(f_k)$，软件游标恰好覆盖片段边界；若 $p<\operatorname{se}(f_k)$，则软件仅消费了该片段的一部分。式（3-2）选择把命中片段整体保留在历史中，从而避免在缺少片段内文本—音频对齐时裁剪到任意 token。

当 $p=0$ 时，表示软件播放器尚未消费推测内容，定义 $\widehat H(0)=0$。若游标越过全部已登记音频，则边界钳制到最后一个具有音频记录的片段末端；仅当本轮全部 assistant 内容均已登记时，该端点才等于本轮 assistant 内容结束位置。

**定义 3.2（片段级软件历史对齐）**　设本轮 assistant 内容相对起点的保留范围为 $[0,\widehat H(p))$。若打断后对话历史及其 KV 表示只保留该范围，则称其满足 TTS 片段级软件历史对齐：

$$
\mathcal{H}_{\mathrm{frag}}=
\langle y_0,\ldots,y_{\widehat H(p)-1}\rangle. \tag{3-3}
$$

作为对照，按生成位置保留的历史为 $\mathcal{H}_{\mathrm{gen}}=\langle y_0,\ldots,y_{G-1}\rangle$。当 $G>\widehat H(p)$ 时，区间 $[\widehat H(p),G)$ 对应完整的游标外片段或其后续内容。

需要强调，式（3-3）保证的是**软件游标与片段操作语义下的保留一致性**，不是设备已呈现或声学上被听到内容的逐 token 真值。本文实现没有设备时钟、loopback 波形、TTS 词级 duration 或强制对齐，因而不能从 $p$ 推得真实 token 播放位置。工件中的 legacy 字段 `heard_text`、`n_heard` 与 `strict_unheard` 仅为兼容别名，其操作语义分别限于片段保留或字符比例—空白吸附代理。

**定义 3.3（字符比例—空白边界代理）**　当 $p$ 命中片段 $f_k$ 且软件只消费了该片段的一部分时，定义片段内软件消费比例

$$
\alpha(p)=\frac{p-\operatorname{ss}(f_k)}{
\operatorname{se}(f_k)-\operatorname{ss}(f_k)}.
$$

设片段文本长度为 $L_k$ 个字符，先计算原始字符切点

$$
c_{\mathrm{raw}}=\operatorname{round}\bigl(\alpha(p)L_k\bigr),
$$

再将其向前移动到最近的空白边界，得到 $c_{\mathrm{ws}}$。文本后缀

$$
W_{\mathrm{tail}}(p)=f_k[c_{\mathrm{ws}}:L_k] \tag{3-4}
$$

作为命中片段中软件游标尚未覆盖部分的代理。该口径是**字符比例—空白边界近似**，既不是音素/词级对齐真值，也不是 token 域线性插值，更不测量人类感知。它只用于第六章分析片段级向上吸附可能带来的文本尾部风险。

本文包含两类状态修正事件：播放期用户打断按 $\widehat H(p)$ 保留历史；未被接受的候选响应作废时，缓存回滚到该次推测之前的 user-open 端点，对应 $p=0$ 和 $\widehat H=0$，但其裁剪点来自推测状态快照而非时间轴查询。

## 3.3 反向查询与持久化模型状态

**定义 3.4（软件游标反向查询）**　给定软件已消费采样游标 $p$，关联时间轴执行

$$
\Phi:p\longrightarrow
f_k\ \text{s.t.}\ \operatorname{ss}(f_k)<p\leq\operatorname{se}(f_k)
\longrightarrow[\operatorname{ts}(f_k),\operatorname{te}(f_k))
\longrightarrow\widehat H(p). \tag{3-5}
$$

时间轴记录片段关联的 `chunk_ids`，但当前反查按片段聚合采样区间定位，并不从采样位置解析到某个具体音频块。$\Phi$ 是由生产者不变式维护的关联与反向索引，不是采样、音频块、文本与 token 四层之间的可逆双射。

令持久化模型状态为

$$
\mathcal{Z}=\langle\mathcal{K},M,I,A,\varphi,e,a_0,a_1\rangle,
$$

其中 $\mathcal{K}$ 为 `DynamicCache`，$M$ 为 attention mask，$I$ 为覆盖完整缓存序列的 `token_ids` ledger，$A$ 为仅含当前 assistant **内容 token** 的 `assistant_token_ids` ledger，$\varphi$ 为 `RolePhase`，$e$ 为 `GenerationEndReason`，$[a_0,a_1)$ 为当前 assistant 内容 span。任何稳定状态都必须满足

$$
|I|=|M|=\operatorname{seq}(\mathcal{K}),\qquad
A=I[a_0:a_1].
$$

当前受测适配器的 `RolePhase` 完整状态集合为 `USER_OPEN`、`ASSISTANT_OPEN` 和 `ASSISTANT_EOT_PENDING`。`GenerationEndReason` 显式记录 `NONE`、`EOS`、`MAX_TOKENS`、`CONSUMER_STOP` 或 `CROPPED`，不能再由生成长度或账本末 token 反推。

当生成器选择结构性 EOT 时，该 EOT 只触发 `ASSISTANT_EOT_PENDING` 并把结束原因置为 `EOS`：它不进入 $A$、TTS 时间轴，也不作为 assistant 内容 token forward 进 $\mathcal{K}$。随后 `reopen_user_role()` 是提交 assistant close 的唯一入口；它恰好一次把模板推导出的结构性 EOT 与 user-open token 写入全局 ledger $I$ 和 KV。由此，预测 EOT 与结构 close 不会重复注入，同时结构 token 始终不计入 assistant 内容 span。

设当前 assistant 内容在整段 KV 序列中的绝对起点为 $a_0$，播放期裁剪位置为

$$
N=a_0+\widehat H(p).
$$

状态恢复分为两个阶段。

**定义 3.5（裁剪阶段合法性）**　裁剪至 $N$ 后，应满足：

1. KV 序列长度、注意力掩码长度和全局 `token_ids` ledger 长度均为绝对端点 $N$；本轮 assistant 内容账本长度为 $N-a_0=\widehat H(p)$；
2. 被移除的 assistant 内容 token 不再出现在 KV、掩码、全局 ledger 和 assistant 内容 ledger 中；
3. 下一次预填充使用裁剪后的实际 past length 构造连续位置编码；
4. 裁剪不得落在 role/EOT 等结构 token 内部，`RolePhase`、assistant span 和 `GenerationEndReason.CROPPED` 必须与裁剪后的 token 序列一致。

播放期保留零个 assistant 内容 token 时，裁剪到 $a_0$，assistant role 仍为打开状态，随后由正常 close 路径结束该轮。整段推测作废则裁掉 assistant header，回到推测前的 `USER_OPEN` 端点，以便继续追加用户文本。`CROPPED` 是当前阶段状态而非永久事件日志：crop 后、任何新内容推进前必须可见；`prefill_user_text()` 成功追加新 user 内容后必须立即重置为 `NONE`，避免陈旧裁剪原因污染下一生成阶段。

**定义 3.6（角色恢复阶段合法性）**　设 `reopen_user_role()` 从 tokenizer chat template 推导出的 assistant close 与下一 user-open 结构串包含 $q$ 个 token。该串提交后，KV、注意力掩码和全局 ledger 的绝对端点均为 $N+q$；结构串不计入本轮 assistant 内容账本，账本仍保存 $\widehat H(p)$ 个内容 token。下一轮 user 文本从位置 $N+q$ 开始，角色阶段为 `USER_OPEN`，结束原因为 `NONE`。

## 3.4 评测指标

### 3.4.1 候选计算与接受事件

E1/E2 的同步受控 harness 区分五类事件：

- **最后段到达** $t_{\mathrm{arr}}$（`last_segment_arrival`）：最后一个预切分稳定文本段进入 LLM 输入路径；
- **首候选 token 选择** $t_{\mathrm{cand}}$（legacy `first_token_ready`）：生成循环选出首个 candidate token 后的内部回调。该回调早于 cache-update forward 和 generator `yield`，只表示 first-candidate-token selection / candidate compute-readiness；
- **候选后 oracle 接受** $t_{\mathrm{acc}}$（`endpoint_accept`）：同步 harness 在候选处理之后，以用户话轮真值终点接受或作废候选；它不是自然端点检测器输出，也不是最后文本段到达瞬间；
- **首可交付诊断标记** $t_{\mathrm{diag-deliv}}$（`first_deliverable_token`）：同步程序按自身接受顺序记录的 marker；
- **consumer 诊断标记** $t_{\mathrm{diag-cons}}$（`consumer_delivery`）：同步程序记录的 consumer-observation marker。

后两者只用于诊断 harness 执行顺序，不代表生产 deliverability、TTS admission、设备播放或声学输出。特别是，`first_token_ready` 不应解释为“可被下游消费”。

**到达—首候选选择延迟**是内部计算指标：

$$
L_{\mathrm{arr}\to\mathrm{cand}}=t_{\mathrm{cand}}-t_{\mathrm{arr}}. \tag{3-6}
$$

该指标度量最后段到达后至首个候选 token 被选择的墙钟时间，不度量 gate-authorized token、consumer-observed TTFT 或首块音频响应。

**oracle 接受后候选延迟下界 $\mathrm{TTFT}_{\mathrm{eff}}$** 是同步接受策略下的乐观下界；保留 `TTFT` 符号只为与既有 artifact 兼容，不表示 production first-token delivery：

$$
\mathrm{TTFT}_{\mathrm{eff}}=
\begin{cases}
0, & \text{若 } t_{\mathrm{acc}} \text{ 时存在存活且已选择的候选};\\
t_{\mathrm{diag-deliv}}-t_{\mathrm{acc}}, & \text{否则}.
\end{cases} \tag{3-7}
$$

式（3-7）回答的是“若 post-candidate oracle 在接受时立即采用存活候选，可获得何种条件性下界”，不是实际可交付或用户感知时延。候选首选到 oracle 接受的间隔也仅是同步程序内部间隔，不能解释为自然端点提前量或用户继续说话时长。第六章单列 $t_{\mathrm{diag-deliv}}$ 与 $t_{\mathrm{diag-cons}}$，用于暴露同步执行顺序，而不将其作为系统 headline。

**推测触发到首候选选择延迟 $\mathrm{TTFT}_{\mathrm{spec}}$**：推测阈值被触发到首个候选 token 被选择的墙钟时间。该指标描述触发—候选计算链路，不表示内容已经获准进入 TTS 或播出。

**mouth-to-ear 延迟**：用户话轮结束到首块音频可播放的时间。第六章只将 LLM 计算墙钟时间与 TTS 画像组合建模；该数值不是实际音频闭环、设备呈现或声学到达的端到端实测。

**KV 裁剪操作延迟 $L_{\mathrm{crop}}$**：孤立执行缓存裁剪的墙钟时间。A1 的主要恢复指标 $L_{\mathrm{joint}}$ 在同一 GPU 同步计时区间内依次执行 crop 与角色恢复，并与重新预填充的逐次墙钟计时比较。A1 固定操作顺序、固定移除 32-token suffix，每个上下文长度包含 5 次预热与 50 次重复；因此其结果只描述该固定协议，不代表自然打断位置或其他裁剪长度。

**Prepared-state 软件控制路径延迟**：P1 在播放器启动前完成目标 KV 状态恢复和 CUDA/GPU 设备同步，并把该准备时间记为 $L_{\mathrm{setup}}$，但将其排除在 stop 路径之外。stop 请求发出后，分别记录软件播放器确认 $L_{\mathrm{ack}}$、确认后的 CUDA/GPU 同步 $L_{\mathrm{sync}}$ 和时间轴反查 $L_{\Phi}$；同时定义两个从同一 stop 请求时刻起算的累计端点：

$$
L_{\mathrm{stop\to crop}}=t_{\mathrm{crop\ done}}-t_{\mathrm{stop\ request}},\qquad
L_{\mathrm{stop\to role}}=t_{\mathrm{role\ done}}-t_{\mathrm{stop\ request}}. \tag{3-8}
$$

$L_{\mathrm{stop\to crop}}$ 已嵌套包含软件停播确认、播放器确认后的 CUDA/GPU 同步、$\Phi$ 查询和同步 KV 裁剪；$L_{\mathrm{stop\to role}}$ 又嵌套包含前者及角色恢复。各区间中位数不能相加，P1 与另一 campaign 的 A1 也不能通过相减解释系统开销。P1 只覆盖 9 个 cell、每 cell 20 次的 headless 软件路径；其 P95 是经验性、描述性的 order statistic，主要由每 cell 的一至两个上尾观测决定，不是生产 SLO。

### 3.4.2 固定检测器条件下的信息复现指标

设固定首轮生成轨迹中 playback 片段边界之后、generation 条件额外保留的差异文本为 $W$，其后两轮回复集合为 $R$。固定轨迹 E3 在两种历史条件下使用完全相同的 $W$，并记录后续回复是否复现其中的信息。每条对话设置 0.25、0.50、0.75 和 fragment boundary 四个软件游标注入位置，形成 `(dialogue, injection position)` 单元。本文采用两种目标口径。

- **片段目标（fragment）**：$W_{\mathrm{frag}}$ 为片段级 software-cursor 端点之后的共享差异文本。只有当该目标非空时，配对记录才进入片段目标分析。
- **字符比例—空白边界近似目标（proxy）**：$W_{\mathrm{proxy}}$ 将式（3-4）的命中片段文本尾部与 $W_{\mathrm{frag}}$ 拼接。该口径纳入片段内代理尾部，但不是设备或声学边界；其分析资格依据 $W_{\mathrm{proxy}}$ 自身是否非空确定。

fragment/proxy 与 lexical rule/`specific-reference-v3` judge 组成四个并列冻结的 target×detector 操作化单元。label-weighted 主 estimand 在每个单元内对所有 eligible `(dialogue, injection position)` 等权；dialogue-weighted 敏感性分析先在每条有效对话内平均，再对对话等权。target-specific exact-key 敏感性分析按 `id`、`trajectory_id`、playback/generation `history_key`、目标文本 SHA-256，以及 fragment 口径下的 `heard_token_end` 构造精确键并压缩重复权重；它不是语义相似性聚类或人工判重。

两种自动检测器的操作定义如下。

| 项目 | 词面规则 | `specific-reference-v3` judge |
|---|---|---|
| 输入 | 目标文本 $W$ 与两轮回复合并文本 | `TARGET` 与以分隔符合并的两轮 `REPLY` |
| 判据 | 从目标抽取数字、首字母大写且长度不少于 3 的非停用词，以及长度不少于 5 的非停用内容词；任一词边界命中，或长度不少于 6 的长词子串命中，即判阳性 | 判断 `REPLY` 是否使用、重复或引用 `TARGET` 中的具体信息；一般主题重合不计 |
| 输出 | 布尔值 | greedy 解码首行 `YES`/`NO`；格式不合格时仅追加格式提醒并有界重试一次 |
| 条件信息 | 规则不接收 playback/generation 标签 | prompt 不提供 condition identity |

E3 的 estimand 因而是**固定检测器条件下的信息复现率**，不是人类语义真值或 HCI 效果。区间只表示在冻结规则、裁判、目标、轨迹、提示词与 40-token cap 条件下的 dialogue-sampling uncertainty，不包含检测器误差、提示词/模型变动或人类感知误差。

同时，本文区分“software-cursor 条件是否写入局部完整游标外文本”这一结构合规问题和“共享差异文本是否在后续回复中复现”这一代理后果。前者是可由边界和文本长度直接检查的构造性性质；后者只能由固定规则或模型代理估计。结构检查不得与代理分析的分母合并。

### 3.4.3 效率指标

pooled discarded-token ratio 定义为

$$
\rho=\frac{\sum\text{作废的候选 token 数}}
{\sum\text{作废的候选 token 数}+\sum\text{最终生成 token 数}}. \tag{3-9}
$$

式（3-9）只按 token 数刻画未被采用的候选工作量，不等同于 FLOPs、GPU 时间、能耗、显存带宽或硬件利用率意义上的计算浪费。该 ratio-of-sums 在每个 bootstrap replicate 内重新计算。八个数值阈值和一个 never-speculate 对照构成九个 B-path 工作点：

$$
\bigl(\rho(\theta),\mathrm{TTFT}_{\mathrm{eff}}(\theta)\bigr).
$$

阈值降低通常提高候选生成覆盖率，也可能增加作废 token。第六章同时报告各工作点的到达—首候选选择延迟与**接受时候选可用率**：分子是 oracle 接受时存在可用候选的 condition records，分母是该条件的全部 records，而不是给定候选已经启动后的条件存活概率。有限个测试点只能支持受控工作点刻画，不自动构成连续或严格单调的 Pareto 前沿。

KV 复用收益通过“重新预填充耗时中位数 / 同一计时区间联合执行 crop 与角色恢复的耗时中位数”描述。该比值只适用于 A1 的固定顺序和固定 32-token suffix 协议。本文以联合路径为主要分母，并把 crop-only、role-only 作为局部诊断；不以两个独立中位数之和替代联合路径中位数。

### 3.4.4 实验单位与路径可比性

确认性 E1/E2 采用 $100$ 个唯一话语与 $5$ 个独立初始化进程 session 的交叉设计。每条件共有 $100\times5=500$ 个 session×utterance 观测，但内容采样单位是 100 个唯一话语，session 是技术重复，不把 500 个观测解释为 500 个独立内容样本。正式 `analysis_v2` 使用 crossed/product bootstrap：独立重采样全局 session 与全局话语，再取笛卡尔积；重复 10,000 次，seed 为 20260901，并报告 percentile 95% 区间。

C-E1 比较一次性 full-string tokenization/full-prefill 的 System A 与 segment-wise tokenization/incremental 的 B@0.92。由于两条路径不保证 token 等价，C-E1 是**实现路径比较**，混合 tokenization、forward topology/shape、role boundary、kernel 与 Python scheduling，不能归因为纯 incremental-prefill 效应。C-E2 比较 B@0.92 与 B-never，两者沿相同 B-path 且正式记录的 token 输出一致，可用于 B 路径内部的阈值策略比较。主延迟分析不能只筛选 C-E1 输出相同的记录，因为这会形成结果之后的选择。

### 3.4.5 指标与实验对应关系

| 研究问题 | 主要指标或 gate | 实验 |
|---|---|---|
| RQ1：外部软件进度驱动的联合前缀状态修正是否满足受测性质，其协议成本如何 | direct crop integrity、within-run matched-arm recovery exactness；联合 crop+角色恢复与重新预填充耗时；软件 stop 与恢复端点 | C2 v3、A1、P1 |
| RQ2：固定轨迹下两种历史边界的固定检测器条件信息复现率有何差异 | fragment/proxy × rule/judge；label-weighted 主 estimand 与 dialogue/exact-key 敏感性 | E3 |
| RQ3：阈值如何影响同步 oracle 接受前候选的工作点 | $\rho$、$L_{\mathrm{arr}\to\mathrm{cand}}$、接受时候选可用率、$\mathrm{TTFT}_{\mathrm{eff}}$ | C-E2 |
| RQ4：两条非 token-equivalent 实现路径有何差异 | $L_{\mathrm{arr}\to\mathrm{cand}}$、诊断 markers、$\mathrm{TTFT}_{\mathrm{eff}}$ | C-E1 |
| RQ5：当前探索性运行中三种历史处理实现的描述性表现如何 | 连贯性评分、重写耗时 | A2 |

## 3.5 本章小结

本章把 $p$ 限定为 software-consumed-sample cursor，并将其与 device-presented samples 和 acoustically heard content 分开；$\widehat H(p)$ 仅表示 TTS 片段级软件保留边界。持久化状态显式包含全局 `token_ids` ledger、assistant 内容 ledger、`RolePhase`、`GenerationEndReason` 与内容 span；预测 EOT 进入 `ASSISTANT_EOT_PENDING`，不进入内容账本、时间轴或内容 KV，由 `reopen_user_role()` 唯一提交结构 close。延迟口径改为首候选 token 选择/内部计算就绪、候选后 oracle 接受以及仅供诊断的 first-deliverable/consumer markers，不再作生产可交付性推断。最后，本章明确了 100 个唯一话语与 5 个 session 的交叉设计、C-E1 的非 token-equivalent 实现路径边界、A1 固定 32-token suffix 和 P1 经验 P95 的适用范围。
---

# 第四章 方法设计

## 4.1 总体设计

本文在本项目一期内部实现的“流式 ASR 稳定文本段—LLM 增量 KV 预填充”流水线之上，增加输出断句、流式 TTS、软件播放时间轴和打断状态修正，形成图 4-1 所示闭环。

![图 4-1](figures/fig4_1.png)

**图 4-1　系统总体架构。** 输入侧以稳定 ASR 文本段驱动增量预填充和候选生成；输出侧将内容 token 流切分为 TTS 文本片段并登记音频块；打断侧依据 software-consumed-sample cursor 查询片段级保留边界，执行 KV 裁剪和角色恢复，并可选用历史自然化策略。

系统遵循两项设计原则。第一，打断后的历史按软件游标对应的 TTS 片段边界保留；这一高层思想已有工程先例，本文的核心是把 software cursor、片段、assistant token span、KV、mask、token ledger、position、role 与 EOT 状态组成可检查的跨层合同。第二，尚未提交的候选必须可作废、可计量并可由阈值调节。其架构目标是在自然话轮确认前预计算候选，但本文确认性实证仅识别同步分段文本 harness 中的 **pre-oracle-acceptance candidate generation with invalidation**，不是 draft-target speculative decoding，也不证明真实 end-of-speech 前就绪。

这里的 playback-aware 不表示设备或声学观测。本文只取得软件播放器消费的采样游标；设备已呈现采样和用户声学上听到的内容需要设备时钟、loopback 或其他物理测量，均未由本方法实例化。

## 4.2 可作废的候选响应生成

### 4.2.1 单一推测阈值与候选后接受

TEN Turn Detection 对累计文本给出连续置信度

$$
c_i=\operatorname{conf}(u_1\cdots u_i)\in[0,1],
$$

并以单一推测阈值 $\theta$ 控制候选响应生成。当 $c_i\geq\theta$ 时，系统冻结候选前状态快照，打开 assistant role，并预生成不超过预算 $B$ 的候选内容 token。$B$ 限制单次误触发的计算成本。在线系统可在自然端点确认前执行这一机制；本文实验则由后续同步 oracle 事件决定采用或作废。

首个候选 token 被选择时触发内部回调；回调位于 token selection 之后、cache-update forward 与 generator yield 之前。因此该事件只表示 first-candidate-token selection / candidate compute-readiness，不表示 token 已获准交付、consumer 已观察、TTS 已接纳或已经产生声学输出。

同步实验 harness 在候选处理之后使用用户话轮真值终点执行 `endpoint_accept`。该事件是 **post-candidate oracle acceptance**，不是自然端点检测输出，也不是最后文本段到达瞬间。harness 另记录 `first_deliverable_token` 与 `consumer_delivery`，但二者仅作为暴露同步执行顺序的诊断 marker，不能据此推断 production deliverability。面向在线部署可以另设独立的播出门控；本文没有实现或验证该门控。

话轮置信度计算与当前文本段的增量预填充在目标架构中可以并发，但“可以并发”不等于无条件零开销。实际能否掩蔽触发模型延迟取决于资源隔离、调度和两条计算链的完成时序。

### 4.2.2 推测—作废状态机

对每个到达的 ASR 稳定文本段 $u_i$，系统执行以下步骤。

1. 若存在活跃候选，则新文本段表明用户仍在说话。系统将 KV、mask、完整 `token_ids` ledger、assistant 内容 ledger 及 role 状态原子地裁剪回该次推测前的 `USER_OPEN` 快照，并记录作废内容 token 数。整段作废同时移除 assistant header，而不是在 assistant role 内保留零内容。
2. crop 完成后 `GenerationEndReason.CROPPED` 必须暂时可见；随后在 `USER_OPEN` 阶段增量预填充 $u_i$。用户文本成功追加即表示进入新输入阶段，陈旧的 `CROPPED` 被重置为 `NONE`。
3. 计算 $c_i$；若达到阈值，则记录新的推测前状态，打开 assistant role，并生成至多 $B$ 个候选内容 token。
4. 用户话轮真值终点到达时，post-candidate oracle 若接受存活候选，则沿其持久化状态继续生成；若没有可接受候选，则按需打开 assistant role 并生成。

![图 4-2](figures/fig4_2.png)

**图 4-2　推测—作废状态机。** 新文本段到达会使先前候选作废并恢复至推测前 `USER_OPEN` 状态；真值话轮终点在候选处理之后执行 oracle acceptance。首候选选择、oracle 接受、诊断 deliverable marker 和 consumer marker 是不同事件。

八个数值阈值加一个 never-speculate 对照得到九个有限工作点。它们描述同步受控 harness 中的候选计算、oracle 响应下界、接受时候选可用率与 pooled discarded-token ratio，不预设曲线严格单调，也不证明异步在线系统的可交付时延改善。

## 4.3 软件游标驱动的 KV 缓存管理

### 4.3.1 片段关联时间轴及生产者合同

反向查询 $\Phi$ 由 `PlaybackTimeline` 支撑。每条片段记录包含

$$
\langle f_j,[\operatorname{ts},\operatorname{te}),
\text{音频块列表},[\operatorname{ss},\operatorname{se}),\text{状态}\rangle.
$$

生成侧登记 token 区间，TTS 侧附加音频块并推进累计软件采样轴，播放器维护 software-consumed-sample cursor。打断到达时，时间轴依据累计采样区间定位当前片段，并返回 $\widehat H(p)=\operatorname{te}(f_k)$。这种结构是“四层关联与反向索引”，不是四层之间的严格双向映射，也不把软件采样轴提升为设备或声学真值。

![图 4-3](figures/fig4_3.png)

**图 4-3　片段关联时间轴与反向查询。** TTS 文本片段是内容 token 区间、音频块和累计软件采样区间的共同主键；软件游标通过片段记录解析为 KV 内容裁剪点。

该查询的成立依赖写入侧不变式，接口以 fail-closed 方式强制以下合同：

1. `add_fragment()` 的 token span 必须非空、单调且与上一片段连续，禁止重叠、倒退或跨越未登记 token；
2. 一个片段只有在其 token span 冻结后才接受音频块；已关闭或已进入终态的片段不得再附加 chunk；
3. 每个 `chunk_id` 全局唯一且只能归属于一个 fragment，`attach_chunk()` 必须按片段与 chunk 顺序调用；
4. 音频块 sample range 必须非空、全局单调连续，片段的聚合 $[\operatorname{ss},\operatorname{se})$ 由其 chunk 序列唯一决定；
5. 软件游标只允许单调前进，并钳制在已登记采样范围内；
6. 任一 out-of-order fragment、重复/错属 chunk、sample gap/overlap、关闭后追加或游标倒退均被拒绝，而不是静默重排。

这些约束使式（3-1）成为接口不变式。锁用于序列化短小的轮级时间轴更新，但不取代上述语义验证。

断句器只接收解码文本，不直接感知 token。系统为每个生成 token 累计非空白字符数；断句器产出片段后，再按片段的非空白字符长度在前缀和中定位片段末 token。该方法能够容忍空白归一化，并通过“片段拼接后的非空白字符序列等于原始解码序列”检查守恒性。

部署目标中，生成、TTS 和软件播放可以由独立执行单元并发推进。实验中的 Mock TTS 与 headless player 共享接口和软件边界语义，但模拟不能代表真实 TTS 队列、audio API、OS/驱动/设备缓冲或声学停播。

### 4.3.2 统一持久化状态与 KV 裁剪

缓存容器不是单一 KV 对象，而是

$$
\langle\mathcal{K},M,I,A,\varphi,e,a_0,a_1\rangle,
$$

即 `DynamicCache`、attention mask、完整 `token_ids` ledger、assistant 内容 ledger、`RolePhase`、`GenerationEndReason` 及 assistant 内容 span。所有 token 追加均通过同一 token-ID prefill 核心，并在稳定点检查

$$
|I|=|M|=\operatorname{seq}(\mathcal{K}),\qquad A=I[a_0:a_1].
$$

时间轴返回相对边界 $\widehat H(p)$ 后，系统将其转换为绝对缓存端点 $N=a_0+\widehat H(p)$，并同步执行：

1. 调用 `DynamicCache.crop(N)` 缩短每层 K/V；
2. 将 attention mask 与完整 `token_ids` ledger 裁剪至 $N$；
3. 将 assistant 内容 ledger 与内容 span 裁剪至同一语义边界；
4. 从裁剪后的真实 past length 构造后续 position IDs；
5. 恢复与剩余 token 序列一致的 `RolePhase`，并把当前 end reason 标记为 `CROPPED`。

只裁剪 KV 而不更新 mask 或 token ledger，会破坏缓存长度与可审计 token 序列的一致性；只更新文本历史而保留旧 KV，则模型仍可访问已删除状态。结构 token 内部不是合法 crop 点。播放期保留零个内容 token 时裁剪到 assistant content start，保持 `ASSISTANT_OPEN`，再由正常关闭路径提交 EOT；整段候选作废则裁剪到 assistant role start 之前并移除 assistant header，恢复 `USER_OPEN`。

A1 在每个上下文长度进行 5 次预热和 50 次重复，所有 trial 采用固定 operation order，并固定移除 32-token suffix。CUDA/GPU 前后同步包围同一计时区间内的 crop 与角色恢复。该微基准不包含时间轴查询、播放器停止、线程调度或服务通信，且不能代表自然打断位置、其他 crop length 或随机化执行顺序。

P1 使用 prepared-state 屏障：每次 trial 先恢复目标 KV 状态并完成 CUDA/GPU 同步，再启动 headless 墙钟节拍软件播放器。stop 路径依次记录软件线程确认、确认后的 CUDA/GPU 同步、时间轴反查、同步裁剪和角色恢复完成端点。stop→crop 与 stop→role 是嵌套累计区间。P1 每个 cell 只有 20 个观测，P95 只作为 empirical/descriptive order statistic；它不是稳定的 production SLO，也不覆盖设备呈现或声学停止。

### 4.3.3 EOT、角色边界与多轮恢复

KV 是连续 token 状态，role 信息由 chat template 特殊 token 表达。系统从 tokenizer 的规范 chat template 推导 user-open、assistant-open 与 assistant-close 转换，并验证 token-ID 序列，不依赖手写固定字符串。

生成循环逐 token 执行 selection 与 production append。普通内容 token 被 forward 进 KV，同时追加到全局 `token_ids` ledger 和 assistant 内容 ledger。若所选 token 是结构性 EOT，则执行不同路径：

1. 不把 EOT 作为 assistant 内容 forward 进 KV；
2. 不将其加入 `assistant_token_ids` 或 TTS fragment timeline；
3. 将 `RolePhase` 置为 `ASSISTANT_EOT_PENDING`；
4. 将 `GenerationEndReason` 显式置为 `EOS` 并停止内容生成。

`reopen_user_role()` 是唯一的 close commit：它在 `ASSISTANT_OPEN`（因 crop/consumer stop/max-token 而结束）或 `ASSISTANT_EOT_PENDING`（生成器预测 EOT）状态下，恰好一次预填充模板推导出的 assistant-close 与 user-open 结构 token。由于预测 EOT 尚未进入 KV，这一路径不会产生重复 EOT。结构串进入全局 ledger、mask 和 KV，但不进入 assistant 内容 ledger。完成后进入 `USER_OPEN`，end reason 重置为 `NONE`。

`GenerationEndReason` 是阶段状态而非永久日志。crop 后的 `CROPPED` 必须在新内容推进前可见；成功的 `prefill_user_text()`、`prefill_assistant_text()`、`open_assistant_role()` 或规范 reopen 会根据新阶段重置状态。特别地，user 内容一旦成功追加就清除陈旧 `CROPPED`。编排器另保存候选结束原因快照，避免为了保留审计信息而污染当前运行状态。

![图 4-4](figures/fig4_4.png)

**图 4-4　KV 裁剪与角色边界恢复。** 第一步原子裁剪 K/V、mask、完整 ledger 与 assistant 内容 ledger；第二步依据裁剪语义恢复 `RolePhase`；第三步由 `reopen_user_role()` 唯一提交 assistant close 与下一 user-open。结构 EOT 不属于 assistant 内容。

### 4.3.4 C2 v3 direct crop-integrity 验证方法

C2 的正式 direct crop-integrity 与 within-run matched-arm recovery 验证采用 protocol v3 exact-only addendum，而不是 clean re-prefill 对照。固定 Qwen2-7B snapshot、BF16、SDPA 与 Transformers backend，设置 24 个 ordered cases，覆盖 512/2048/8192 token context、$p=0$、片段边界、中段吸附、reply tail、pending EOT、推测全作废、下一轮与第二次 crop。24 个 case 共产生 27 个 ordered crop events，其中包含 3 个 no-op event；冻结 assistant fixture 共 308 个内容 token，必须全部经 `generate_accumulating` 的 production 路径逐 token append，每个 token 对应一次受控 `_prefill_ids_p2` forward。

每个 crop event 设三方 exact 比较：

1. **pre-crop retained prefix**：crop 前从 production K/V 取得目标长度前缀的逐层 manifest；
2. **production post-crop**：唯一调用生产 `crop_to_token()` 后的状态；
3. **production-interface-independent slicing oracle**：从同一 crop 前 snapshot clone，并按推导出的 keep length 逐层切片；它不调用生产 crop 接口，但不是外部团队或不同实现栈的独立复现。

三方逐层比较 K/V 的 shape、dtype、device、SHA-256 和 runtime `torch.equal`，并要求 keep length、mask、完整 token ledger、sequence length 与 KV length exact。wrong-length disposable negative control 必须对每个 event 被拒绝，以证明 gate 能检测错误保留长度。

crop 后，production arm 与 direct oracle 接收完全相同的 token-ID chunks，执行 60 个 matched-recovery steps。每一步要求 K/V、logits、mask、完整 token ledger、retained-prefix hashes，以及由操作序列独立推导的 role/end/content state bitwise/exact 一致。本文将 **within-run matched-arm recovery exactness** 定义为：同一 accepted run 内，两臂从精确匹配的保留状态出发，接受相同 token-ID chunks 与相同操作序列后逐步保持上述状态一致。它不表示跨进程、跨设备或相对 clean re-prefill 的 determinism。

v1/v2 采用 canonical clean re-prefill 对照，均按冻结门槛 rejected。v2 虽然 24/24 termination probe 和 45/45 token/state/EOT/scenario gate 通过，但单控制的 2× numerical gate 仅 42/45。进一步审查发现 control 按语义 seam 分块并强制末 token 单独 forward，而 production 初始 context 和 role/content append 使用不同 forward topology，故该 control 不能识别三项数值失败来自 crop 还是拓扑差异。v3 没有把门槛事后放宽，也没有改判 v1/v2；它改问可由 exact slicing oracle 识别的问题。因此，v3 **不声称** crop+role 与 clean re-prefill 的数值、logit 或 continuation 等价，也不支持跨模型、dtype、backend、硬件、在线音频或生产端到端正确性。

### 4.3.5 对照条件的语义

- **按生成位置保留（B-gen）**：保留完整已生成回复。若 $G>\widehat H(p)$，则完整的 software-cursor 外片段可能进入历史。
- **重新预填充（B-noKV）**：丢弃 KV，根据裁剪后的文本重新执行模型前向。该条件只用于 A1 固定协议下的计算耗时对比；C2 v3 不以其作为数值等价 oracle。
- **按合成位置保留（B-syn）**：以合成边界修正历史。在本文同步 Mock TTS 条件下，它无法与 B-gen 形成可区分时序，故不作为已验证条件。

## 4.4 被打断历史的处理策略

片段级保留会把 software cursor 命中的当前片段整体写入历史，其中可能包含游标尚未覆盖的文本尾部；某些断句片段也可能在语义上不完整。本文在同一片段边界之上实现三种策略。

- **朴素策略**：直接保留片段级 assistant 前缀。
- **标记策略**：在保留前缀后追加省略号等打断标记。它不调用额外模型，但仍产生少量 tokenization 和预填充开销。
- **重写策略**：使用 Qwen3-0.6B[16] 将保留前缀自然收束，并在提示词中要求不新增事实。该要求是设计约束而非形式保证；KV 需要回退到本轮 assistant 起点，再预填充重写文本。

重写可以在下一轮用户输入期间异步执行，因此具有隐藏部分延迟的潜力；当前实现和实验未测量真实重叠比例。现有 A2 为三种策略分别重新采样首轮和下一轮回复，未隔离策略效应，所以研究问题仅描述当前探索性运行的连贯性分数与重写耗时，不回答“是否改善”。

## 4.5 实验设计边界

确认性 E1/E2 是 100 个唯一话语与 5 个独立初始化进程 session 的交叉设计。每条件有 500 个 session×utterance observations，但 100 个话语是内容采样单位，5 个 session 是技术重复。正式 versioned `analysis_v2` 不覆盖 `analysis_v1`，而是使用 10,000 次 crossed/product bootstrap：独立抽样全局 session 与 utterance，再对二者笛卡尔积计算配对 estimand；seed 为 20260901，区间为 percentile 95%。

C-E1 的两条实现路径不是 token-equivalent：System A 对完整字符串一次 tokenization 并 full prefill；System B 对文本段分别 tokenization 并 incremental prefill。故 C-E1 估计整体 implementation-path difference，混合 tokenization、forward topology/shape、role boundary、kernel 与 Python scheduling，不能隔离纯增量预填充效应。正式输出检查只用于披露路径差异，不能据此过滤主延迟记录。C-E2 在相同 B path 内比较 B@0.92 与 never-speculate，输出 token 序列一致，因此具有更强的路径内可比性。

固定轨迹 E3 使用 label-weighted estimand 与同一口径的 dialogue-cluster bootstrap，并另报 dialogue-weighted 与 target-specific exact-key deduplication sensitivity。四个注入位置为 0.25、0.50、0.75 和 fragment boundary；fragment/proxy 分别按目标字段非空确定资格。词面规则提取数字、首字母大写词和长度不少于 5 的非停用内容词；`specific-reference-v3` judge 对合并的两轮回复作 greedy `YES`/`NO` 判断，输入不暴露条件标签。二者均为冻结自动操作化而非人类 reference standard。E3 量化固定检测器条件下的信息复现率，不支持 superiority、equivalence、noninferiority、harm、absence-of-effect 或人类感知结论。

## 4.6 本章小结

本章将辅助 C1 限定为同步分段文本 harness 中 oracle 接受前候选生成的 selection/compute-readiness、post-candidate oracle acceptance 与 discarded-token 工作点刻画；把核心 C2 定义为 software cursor→TTS fragment→assistant token span→KV/mask/ledger/position/role/EOT 的状态合同。方法强制时间轴生产者顺序、不接受乱序写入；使用完整 token ledger、显式 `RolePhase` 和 `GenerationEndReason`；由 `ASSISTANT_EOT_PENDING` 与唯一 close commit 消除重复 EOT。C2 v3 以 24 cases、27 crop events 的三方 exact slicing 与 matched recovery 验证 direct crop integrity，不将已 rejected 的 clean-reprefill 协议改写为通过。最后，本章把 E1/E2 交叉重复、C-E1 非 token-equivalent 实现路径、A1/P1 协议边界及 A2 探索性定位纳入方法合同。
---

# 第五章 系统实现

## 5.1 模块架构

系统复用一期基于 Whisper[17] 的流式 ASR 与 LLM 增量预填充，并增加对话编排、输出断句、流式 TTS、软件播放状态和实验驱动。表 5-1 将模块与本文的验证对象对应起来。

**表 5-1　主要模块、接口与验证对象**

| 模块 | 主要职责 | 关键验证对象 |
|---|---|---|
| `src/dialogue/timeline.py` | 关联 TTS 文本片段、token 区间、音频块与累计软件采样区间；按 software-consumed cursor 反查 | token/sample 连续性、chunk 唯一归属、乱序拒绝、裁剪端点 |
| `src/llm/stream_llm_inference.py` | 增量预填充、assistant 侧 KV 累积、裁剪、EOT 与角色恢复 | KV/mask/ledger 长度一致，role/end 状态与 position 连续 |
| `src/tts/sentence_chunker.py` | 基于 stream2sentence[21] 将 token 解码流切分为 TTS 文本片段并关联 token 区间 | 非空白字符守恒、空片段处理、末端钳制 |
| `src/tts/streaming_tts.py` | 流式 TTS 接口、Mock TTS 与真实后端适配 | 片段—音频块归属、画像参数读取 |
| `src/player/player.py` | 维护软件已消费采样游标和停止接口 | 采样计数、单调查询与 seek 语义 |
| `src/dialogue/trigger.py` | 计算话轮完成置信度 | 类别 token 配置与阈值触发记录 |
| `src/dialogue/orchestrator.py` | 串联用户输入、候选生成、断句、TTS、打断与状态修正 | 推测作废、同步 oracle 接受、EOT/role 状态和逐轮指标 |
| `experiments/sci34_supplement/` | 固定轨迹 E3、A1/P1、确认性 E1/E2 与 C2 协议 | 配对目标、计时边界、crossed analysis、exact crop/recovery gates |

机制实现与实验实例化相互分离：编排器面向统一接口，实验脚本选择真实后端或画像驱动后端。本文的 playback 状态仅来自软件已消费采样游标，不包含 device-presented sample clock 或 acoustically heard content measurement。

## 5.2 片段时间轴与断句对齐

`PlaybackTimeline` 以 TTS 文本片段为主轴。片段生成后登记文本及 assistant 内容 token 区间；TTS 音频块随后附加到所属片段，并更新累计软件采样区间。播放器维护 software-consumed-sample cursor，打断查询据此定位片段并返回该片段末 token 作为合法保留边界。

采样区间采用 $[0,p)$ 已由软件消费的语义。当 $p=\operatorname{se}(f_k)$ 时，游标恰好覆盖片段 $f_k$，因此保留该片段。实现显式处理片段中部、片段末端、首采样、空时间轴和越过末端等边界。工件中的 `heard_text`、`n_heard` 等字段仅是兼容别名，不表示声学真值。

写入 API 强制生产者合同。`add_fragment()` 要求非空、单调且连续的 token span；重复、重叠、倒退或 gap 均失败。片段 span 冻结后才能 `attach_chunk()`；关闭、丢弃或完成的片段拒绝追加。`chunk_id` 全局唯一且只能归属一个片段，sample range 必须非空并与累计端点连续；软件游标也必须单调且不越过已登记范围。锁只保证短更新的原子性，不替代上述检查。

断句器输出文本，而缓存状态使用 token 端点。系统在逐 token 解码时累计非空白字符数，再以片段的非空白字符长度定位其末 token。该映射不假设 token 边界与标点或词边界一致，但要求全部片段拼接后保持非空白字符序列守恒；纯空白片段被跳过，区间末端被钳制到实际内容 token 数。

## 5.3 KV、token ledger 与角色状态

### 5.3.1 联合状态容器

一期调用方只保留用户侧预填充缓存，assistant 生成 token 的 KV 未持续写回，无法在播放期裁剪。二期 `AccumKVCache` 联合维护 `DynamicCache`、attention mask、完整 `token_ids` ledger、当前 assistant 内容 ledger、`RolePhase`、`GenerationEndReason` 及 assistant role/content 边界。每个稳定操作后均检查

$$
\operatorname{len}(token\_ids)
=\operatorname{len}(pre\_attention\_mask)
=\operatorname{seq\_length}
=\operatorname{DynamicCache.get\_seq\_length}().
$$

同时，assistant 内容 ledger 必须与完整 ledger 中的当前内容 span 一致；裁剪后的 position IDs 按实际 past length 重算。

### 5.3.2 tokenwise append、EOT 与结束原因

`generate_accumulating()` 先选择候选 token。普通内容 token 再经一次 cache-update forward 写入 KV，同步追加 mask 与两个 ledger，随后才由 generator yield。因此首 token 回调表示 candidate selection/internal compute-readiness，而不是 consumer 可见或可交付 token。

结构性 EOT 不作为 assistant 内容写入 KV，不进入内容 ledger、TTS 片段或 timeline。模型选择 EOT 后只进入 `ASSISTANT_EOT_PENDING`，将结束原因记为 `EOS`；`reopen_user_role()` 是唯一 assistant close commit，恰好一次追加从 tokenizer chat template 推导并校验的 assistant-close 与下一 user-open token。max-token、consumer-stop 和 crop 分别记录为 `MAX_TOKENS`、`CONSUMER_STOP` 与 `CROPPED`，避免从生成长度或末 token 反推结束原因。

### 5.3.3 crop 与角色恢复

`crop_to_token(N)` 同步裁剪 K/V、mask、完整 ledger、assistant 内容 ledger 和 span，并由保留序列恢复 role/end state；结构 token 内部不是合法 crop 点。播放期保留零个内容 token 时裁到 assistant content start，保留 assistant header 和 `ASSISTANT_OPEN`，随后正常提交一次 close。整段候选作废则回到推测前的 assistant role start，删除 header 与全部候选内容，并恢复 `USER_OPEN`。

裁剪后的 `CROPPED` 仅保留到下一次合法状态推进；成功追加新 user 内容后恢复 `USER_OPEN + NONE`。`open_assistant_role()`、`prefill_assistant_text()` 与 `reopen_user_role()` 也分别更新 end reason。角色边界由 tokenizer chat template 的规范 tokenization 推导，故当前适用范围限于可可靠提取这些 token 的模板。

## 5.4 编排与事件语义

`DialogueOrchestrator` 提供一次性用户文本路径和增量文本段路径。前者用于 System A，后者用于 B 系列候选生成工作点；输出 token 经断句和 TTS 后登记到时间轴，再按条件执行打断与状态修正。

确认性 E1/E2 runtime 区分以下事件：`last_segment_arrival_ns` 是最后一个预切分文本段到达；legacy `first_token_ready_ns` 是首候选 token 的选择/内部计算就绪回调；`endpoint_accept_ns` 是候选处理后的同步 oracle 接受；`first_deliverable_token_ns` 与 `consumer_delivery_ns` 是同步程序的诊断 marker。后两者受“先处理候选、后接受/消费”的程序顺序支配，不作为生产 deliverability 指标；候选选择到 oracle 接受的间隔也不能解释为自然端点提前量。

Mock TTS 由 CosyVoice2 六句真机画像参数化，以每非空白字符采样数、首块延迟和实时率构造确定性时长近似。该实现便于控制软件游标，但不等价于真实异步合成、应用队列、audio API、OS/驱动/设备缓冲或声学传播；相应 mouth-to-ear 量仅是模型计时与 TTS 画像的组合估计。

## 5.5 实验协议的实现

确认性 E1/E2 使用 100 个唯一 holdout utterances、5 个独立初始化进程 session 和 10 个条件；每个 session 重新加载模型，并在同一组话语上运行 System A、八个数值阈值及 never-speculate B 对照。正式 `analysis_v2.json` 以 crossed/product bootstrap 独立重采样全局 session 与 utterance，再对笛卡尔积重算估计量。C-E1 比较 full-string tokenization/full prefill 与 segment-wise tokenization/incremental forward 两条非 token-equivalent implementation paths；C-E2 则在 token-consistent 的 B@0.92 与 B-never 之间比较。

固定轨迹 E3 先为每条对话生成一次被打断 assistant token 轨迹，再由同一轨迹导出四个注入位置和 playback/generation 两个条件。后续两轮 probe 按 retained-history key 复用；目标字段非空决定 target-specific 资格。正式重分析报告 label-weighted 主 estimand，并以 dialogue-weighted 和 target-specific exact-key deduplication 检查对话权重及重复标签的影响。

A1 在 256–8192 token 上使用 5 次预热、50 次正式重复、固定操作顺序和固定 32-token suffix；crop+role 与 re-prefill 计时均执行 CUDA/GPU 同步。P1 使用 3 个上下文长度、3 个软件游标比例和每单元 20 次重复，共 180 条 prepared-state headless 记录。两者分别是模型侧固定协议微基准与软件控制路径测量，不代表完整 barge-in。

## 5.6 C2 v3 的 exact-only 验证实现

C2 v3 复用冻结的 24-case 网格，包含 27 个 crop event、3 个 no-op 和 60 个后续恢复步骤。fixture 的 308 个 assistant 内容 token 均经 production `generate_accumulating()` 逐 token append。

本文所称 **slicing oracle** 是独立于 production crop 接口、但取自同一 pre-crop snapshot 的逐层切片对照。对目标长度 $N$，oracle 将 snapshot 中每层 key/value 张量沿序列轴直接复制 `[..., :N, :]`，同时复制 `attention_mask[:, :N]` 与 `token_ids[:N]` 并将 sequence length 置为 $N$；它不调用 `crop_to_token()`，也不执行 clean re-prefill。验证器不信任记录中的 stored keep，而根据 case、fragment token partition、role/content boundaries 及第二次裁剪比例独立推导 $N$。

production arm 是对同一 snapshot 唯一调用 `crop_to_token(N)`。每个事件比较 pre-crop retained prefix、production post-crop 与 slicing oracle 的逐层 K/V shape、dtype、device、hash 和运行时 `torch.equal`，并检查 mask、完整 ledger、sequence length 与 KV length。wrong-length disposable control 将 $N$ 改为错误值，验证 shape/manifest gate 能检出偏差。

裁剪后，production 与 oracle 两臂在同一 accepted run 内从精确匹配的保留状态出发，接收相同 token-ID chunks 和相同操作序列；每步比较 K/V、logits、mask、完整 ledger、retained-prefix hash，以及独立按操作序列推导的 role/end/content state。本文将这一性质称为 **within-run matched-arm recovery exactness**。它不表示跨进程、跨设备可重复性，也不表示 clean-reprefill 或不同 forward topology 的数值等价。

## 5.7 环境与复算入口

**表 5-2　正式工件的环境与复算入口**

| 项目 | 冻结信息 |
|---|---|
| 主模型与精度 | Qwen2-7B-Instruct；BF16；SDPA；C2 模型 artifact identity 见 accepted manifest |
| 自动裁判 / 话轮检测 | Mistral-7B-Instruct-v0.3 / TEN Turn Detection；模型 identity 见各 campaign manifest |
| 软件栈 | Python 3.10.18；PyTorch 2.8.0+cu128；Transformers 4.57.1；CUDA 12.8；cuDNN 9.10.2 |
| 设备 | NVIDIA RTX 3090（24 GB）；确认性 E1/E2 使用两张卡，其余 campaign 按各 manifest 记录 |
| 权威工件入口 | 仓库根目录 `REPRODUCIBILITY.md` |
| 关键 accepted analyses | C-E1/C-E2 `analysis_v2.json`；E3 `analysis_weighting_dedup_v2.json`；C2 v3 `analysis_v1.json` 与 `validation.json` |

`REPRODUCIBILITY.md` 是 campaign 状态、正式 run、模型身份、分析文件和复算命令的稳定索引。不同 campaign 不构成可池化的计时总体；分析复算与模型重跑的依赖范围亦以该索引和 campaign manifest 为准。

## 5.8 本章小结

系统将 software-consumed cursor、TTS fragment、assistant token span 和联合缓存解释状态连接起来。时间轴 API 强制连续性、生命周期与唯一归属；缓存容器同步维护 KV、mask、ledger、position、role 与 end reason，并以唯一 EOT close commit 避免结构 token 重复入账。C2 v3 通过同一 pre-crop snapshot 上的 production crop、逐层 slicing oracle、wrong-length control 和 within-run matched-arm recovery 构造可证伪的 exact gate；环境与复算范围由 `REPRODUCIBILITY.md` 统一索引。
---

# 第六章 实验与结果分析

## 6.1 证据层级与实验设置

本章按 **C2→E3→C-E2→C-E1→A2** 报告证据。C2 是唯一核心机制贡献；E3 检查该机制所选择的软件保留边界对后续自动信息复现指标的下游影响；C-E2 与 C-E1 是支持性 C1 刻画；A2 仅为受混杂的探索性描述。研究问题依次为：

- **RQ1（C2）：** 冻结模型与后端下，production crop 是否保持指定缓存前缀，匹配双臂的恢复轨迹是否逐步精确一致？固定协议的模型侧成本和 prepared-state 软件路径时延如何？
- **RQ2（E3）：** 固定被打断轨迹与自动检测器下，software-cursor fragment retention 与 generation retention 的后续信息复现率有何差异？
- **RQ3（C-E2）：** 推测阈值如何影响 pooled discarded-token ratio、接受时候选可用率、候选选择/计算就绪和同步 oracle 时延下界？
- **RQ4（C-E1）：** 一次性预填充与增量推测两条非 token-equivalent implementation paths 的候选计算就绪有何差异？
- **RQ5（A2）：** 三种历史自然化实现的评分和重写耗时在本次探索性运行中呈现何种描述？

实验由 MultiWOZ 2.1[15] 派生。主模型为 Qwen2-7B-Instruct[11]；话轮检测器使用 TEN Turn Detection[12]；E3 自动裁判为 Mistral-7B-Instruct-v0.3[13]；TTS 时长画像由 CosyVoice2-0.5B[14] 采集；A2 重写模型为 Qwen3-0.6B[16]。确认性 C-E1/C-E2 使用 100 条唯一话语、5 个独立初始化进程 session 和 10 个条件，每条件 500 个 session×utterance 观测；内容采样单位仍是 100 条话语。固定轨迹 E3 包含 100 条对话、400 个 `(dialogue,injection_position)` 配对、800 条条件记录和 1600 条裁判记录。C2 v3、A1 与 P1 使用相互独立的正式工件；软件栈与硬件版本见表 5-2。

C-E1/C-E2 是同步预切分文本 harness，不包含真实音频、ASR 墙钟、在线 TEN 前向、TTS、播放器或声卡。raw `first_token_ready` 位于 token selection 之后、cache-update forward 与 generator yield 之前，故本文称其为**首候选 token 选择/内部计算就绪**。`endpoint_accept` 是候选处理后的同步 oracle 接受事件，不是自然端点检测输出；`TTFT_eff` 是该接受规则下的 **oracle latency lower bound**，亦即推测收益的乐观上界。

**表 6-1　证据层级与测量边界**

| 顺序 | 证据 | 角色与测量层级 | 主要限制 |
|---:|---|---|---|
| 1 | C2 v3 | 核心：direct crop integrity 与 within-run matched-arm recovery exactness | 不检验 clean re-prefill，不跨模型、后端、设备或在线系统 |
| 2 | E3 | C2 下游支持：固定检测器条件下的信息复现率 | 自动代理，不是人工或 HCI reference standard |
| 3 | C-E2 | C1：同一 B-path 内的阈值、候选可用性与 oracle 下界 | 同步 oracle，不识别真实端点前收益 |
| 4 | C-E1 | C1：非 token-equivalent implementation-path comparison | 不识别单一 incremental-prefill effect |
| 5 | A2 | C3：受混杂探索性描述 | 不具可识别的策略处理效应 |

播放侧 $p$ 仅表示 software-consumed-sample cursor；它及其片段保留边界均不等同于 device-presented samples 或 acoustically heard content。

## 6.2 RQ1：C2 直接裁剪完整性与匹配恢复

### 6.2.1 C2 v3 exact gate

C2 v3 在 Qwen2-7B-Instruct、BF16、SDPA 和 Transformers `DynamicCache` 的冻结环境中覆盖 24 个 case 与 27 个 crop event。308 个 fixture assistant 内容 token 均经 production append；另含 3 个 no-op、60 个 recovery step 和 27 个 wrong-length negative control。

slicing oracle 独立于 production crop 接口，但与 production arm 取自同一 pre-crop snapshot。对独立推导的 keep length $N$，它将每层 K/V 沿序列轴复制 `[..., :N, :]`，并同步复制 mask 与 token ledger 前缀；它不调用 `crop_to_token()`，也不进行 clean re-prefill。production arm 对对应快照唯一调用 `crop_to_token(N)`。

**表 6-2　C2 协议状态与允许结论**

| 版本 | 冻结结论 | 关键结果 | 允许解释 |
|---|---|---|---|
| v1 | rejected | 冻结等价门未通过 | 仅作失败协议记录 |
| v2 | rejected | 结构检查通过；单控制数值门 42/45 | control 与 production forward topology 不匹配，不能定位 crop 缺陷或声称 clean-reprefill 等价 |
| v3 | accepted | 27/27 crop events、60/60 recovery steps 和 27/27 负控通过 | direct crop integrity 与 within-run matched-arm recovery exactness |

**表 6-3　C2 v3 exact gate 结果**

| 检查 | 结果 |
|---|---:|
| Case / crop event | 24/24；27/27 |
| 逐 token production append | 308 tokens |
| K/V 层数 | 28 |
| Recovery step | 60/60 |
| Wrong-length negative control | 27/27 检出 |
| No-op crop | 3/3 |

每个事件的 pre-crop retained K/V prefix、production post-crop K/V 与 slicing oracle 在 28 层上具有相同 shape、dtype、device、hash 和运行时 `torch.equal` 结果；keep、mask、完整 token ledger、sequence length 与 KV length 也精确一致。裁剪后，两臂在同一正式 run 内从精确匹配的保留状态出发，接收相同 token-ID chunks 与相同操作序列，逐步得到相同 K/V、logits、mask、ledger、retained-prefix hash 及 role/end/content state。该结果不表示跨进程或跨设备确定性，也不支持 clean-reprefill numerical equivalence、32-token continuation equivalence 或生产端到端正确性。v3 回答的是比 v1/v2 更窄的可识别问题，不改变其 rejected verdict。

### 6.2.2 固定协议成本与软件控制路径

A1 覆盖 256–8192 token，每个长度 5 次预热、50 次正式重复；操作顺序固定，每次固定移除 32-token suffix，计时边界执行 CUDA/GPU 同步。

**表 6-4　联合 crop+role 与重新预填充微基准**

| 上下文长度 | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---:|---:|---:|---:|---:|---:|
| 联合路径中位数 (ms) | 31.616 | 31.852 | 31.054 | 31.519 | 36.903 | 48.315 |
| 联合路径 IQR (ms) | 2.356 | 2.162 | 3.099 | 1.197 | 0.635 | 0.928 |
| 重新预填充 / 联合路径中位数 | 2.254× | 4.124× | 7.707× | 15.020× | 25.453× | 40.620× |

![图 6-1](figures/fig6_4.png)

**图 6-1　固定 32-token suffix 协议的联合微基准。** 误差线为 50 次正式重复的 IQR。该比较是时延微基准，不是 clean-reprefill 状态等价检验。

P1 v2 在 512、2048、8192 token 和 0.25、0.50、0.75 三个游标位置上各运行 20 次，共 180 条 prepared-state headless 记录。180/180 次 request 与 acknowledgement 命中目标 software-consumed cursor；`leaked_samples=0` 仅指软件计数器在确认后未继续增加。

**表 6-5　Prepared-state P1 软件路径时延（每单元 n=20）**

| 计时区间 | 单元中位数范围 (ms) | 最大单元 empirical P95 (ms) |
|---|---:|---:|
| headless 播放器线程 stop acknowledgement | 0.055–0.062 | 约 0.077 |
| 播放器确认后的 CUDA/GPU 同步 | 0.167–0.176 | 约 0.352 |
| 时间轴反查 | 0.47–0.50 | 约 0.94 |
| stop→crop 完成 | 2.44–2.53 | 约 3.492 |
| stop→角色恢复完成 | 78.6–80.8 | 约 86.1 |

stop→crop 与 stop→角色恢复是共享起点的嵌套累计区间，不能与组件中位数相加。每单元 n=20 的 empirical P95 主要由 1–2 个上尾值决定，不是生产 SLO。A1 与 P1 均不测 device/acoustic stop、在线 TTS 取消或完整并发 barge-in。

## 6.3 RQ2：固定轨迹 E3 的下游支持证据

E3 的四个注入标签是 0.25、0.50、0.75 三个软件游标比例及一个 fragment boundary。每个对话只生成一次被打断 assistant 轨迹，playback 与 generation 条件共享轨迹、断句时间轴和注入位置。fragment 与 proxy 两种 target 分别按 `unheard_text` 和 `strict_unheard_text` 非空确定资格；两种 target 与 rule/judge 两种 detector 形成四个并列的冻结操作化，未指定单一 reference standard。

主 estimand 为 **label-weighted generation-minus-playback effect**：从某一 target 的全部 eligible `(dialogue,injection_position)` 集合中等权抽取一个标签，计算两条件二元阳性率之差。区间采用 10,000 次 paired dialogue-cluster bootstrap，在对话层重采样并保留对话内标签及条件配对。dialogue-weighted 敏感性先在每条对话内平均其 eligible labels，再令对话等权。

**表 6-6　E3 label-weighted 信息复现率**

| 目标 / 检测器 | software-cursor retention | generation retention | 差值 | 95% dialogue-cluster CI |
|---|---:|---:|---:|---:|
| 片段目标 / 词面规则 | 67.00% (199/297) | 63.64% (189/297) | −3.37 pp | [−10.49, 3.40] pp |
| 片段目标 / 自动裁判 | 42.76% (127/297) | 40.74% (121/297) | −2.02 pp | [−10.70, 6.13] pp |
| 字符比例—空白边界代理 / 词面规则 | 75.26% (286/380) | 73.68% (280/380) | −1.58 pp | [−6.08, 2.67] pp |
| 字符比例—空白边界代理 / 自动裁判 | 43.95% (167/380) | 41.32% (157/380) | −2.63 pp | [−8.57, 2.90] pp |

片段目标有 297 个 eligible labels、96 条对话；proxy 有 380 个 labels、100 条对话。target-specific exact-key 由 `id`、`trajectory_id`、playback/generation 两条件 `history_key` 与 exact target hash 组成，fragment key 另含 `heard_token_end`。该操作是精确键去重，不是语义聚类或人工判重。片段目标从 297 个标签压缩为 169 个 exact-key groups，表示去除了 **128 个额外 label 权重**；proxy 从 380 压缩为 379，仅去除 1 个额外权重。

**表 6-7　E3 对话加权与 exact-key 去重敏感性**

| 目标 / 检测器 | Dialogue-weighted effect [95% CI] | Exact-key retention rates | Exact-key effect [95% CI] |
|---|---:|---:|---:|
| 片段目标 / 词面规则 | −3.21 pp [−9.55, 2.78] | 71.60% / 68.64% | −2.96 pp [−9.04, 2.63] |
| 片段目标 / 自动裁判 | −1.30 pp [−8.94, 6.08] | 43.20% / 43.20% | 0.00 pp [−7.98, 7.47] |
| 代理目标 / 词面规则 | −1.50 pp [−5.75, 2.50] | 75.20% / 73.61% | −1.58 pp [−6.10, 2.69] |
| 代理目标 / 自动裁判 | −2.58 pp [−8.25, 2.67] | 43.80% / 41.16% | −2.64 pp [−8.57, 2.90] |

**表 6-8　E3 检测器的冻结操作定义**

| 检测器 | 输入与操作 | 阳性判据 | 输出 |
|---|---|---|---|
| 词面规则 | 从 TARGET 提取数字、长度≥3 的首字母大写非停用词及长度≥5 的其他非停用内容词；将两轮 probe replies 以空格合并 | 任一 cue 命中 reply 词边界，或长度≥6 的 cue 命中长词子串 | Boolean |
| `specific-reference-v3` judge | 向 Mistral 裁判提供 TARGET 与以分隔符合并的两轮 REPLY；不在 prompt 中提供 condition identity | 判断 REPLY 是否使用、重复或引用 TARGET 的具体信息；generic topical overlap 不计 | greedy 解码，首行严格 YES/NO |

400/400 个 playback 条件在片段边界后的局部完整未保留文本为空，局部规则阳性为 0；这是构造检查，不是语义效果。词面规则与 judge 的 label-level 合并一致数为片段 370/594、proxy 442/760；exact-key level 为 207/338 和 440/758。judge 不是人工 reference standard，这些数值只表示自动代理间一致性。四个主效应点估计均低于零，但区间均跨零，且未预设实质性差异阈值，故不能推断优势、等效、非劣、伤害或差异不存在。

![图 6-2](figures/fig6_1.png)

**图 6-2　E3 效应区间。** 主结果与两类敏感性分析分别见表 6-6 和表 6-7。

## 6.4 RQ3：阈值与同步 oracle 刻画（C-E2）

九个 B-path 工作点包括八个数值阈值和 never-speculate。0.92 在 holdout 结果揭示前冻结，不表示部署最优。**Pooled discarded-token ratio** 定义为 $\sum_i W_i/(\sum_i W_i+\sum_i G_i)$；bootstrap 的每个 replicate 都按 weighted ratio-of-sums 重算。该比例衡量 token 计数，不等同于 FLOPs、GPU 时间、能耗或带宽浪费。

**表 6-9　确认性九点扫描（每点 100 条话语×5 sessions）**

| 阈值 $\theta$ | 0.0052 | 0.1979 | 0.3906 | 0.5833 | 0.7760 | 0.8500 | 0.9200 | 0.9688 | never |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pooled discarded-token ratio | 31.0% | 19.3% | 15.8% | 13.2% | 11.3% | 10.7% | **2.85%** | 0% | 0% |
| 接受时候选可用率 | 100% | 99% | 98% | 97% | 96% | 84% | **67% (335/500)** | 28% | 0% |
| oracle TTFT_eff 下界均值 (ms) | 0.0 | 0.3 | 0.7 | 0.9 | 1.3 | 5.0 | **10.3** | 22.4 | 31.1 |
| arrival→candidate selection/readiness 均值 (ms) | 62.4 | 62.4 | 62.3 | 62.1 | 62.2 | 62.0 | 62.4 | 62.3 | 62.4 |

“接受时候选可用率”以全部 500 个 condition records 为分母，B@0.92 为 335/500；它不是 $P(\text{survive}\mid\text{candidate launched})$。现有 harness 证明的是同步 oracle 接受前候选生成以及接受时候选可用性，不证明候选在真实 end-of-speech 前就绪。

B@0.92 相对 never 的 candidate-readiness 差值（never−B）为 −0.03 ms，crossed 95% CI [−0.64, 0.61] ms；oracle `TTFT_eff` 下界差为 +20.80 ms [17.85, 23.65] ms。pooled discarded-token ratio 为 2.85% [1.12%, 4.73%]，接受时候选可用率为 67% [58%, 76%]。后两个 marker 的均值 257.58 与 265.57 ms 受同步执行顺序支配，仅为诊断量。

![图 6-3](figures/fig6_2.png)

**图 6-3　九点扫描。** 左图是同步 oracle 时延下界，右图是 candidate selection/readiness；描述性离散范围不是 crossed CI。

## 6.5 RQ4：非 token-equivalent 路径比较（C-E1）

C-E1 在 session×utterance 内比较 System A 的 full-string tokenization/full prefill 与 B@0.92 的 segment-wise tokenization/incremental forward。两路径不满足相同 tokenized context 条件，故估计对象是整体 implementation-path difference。

**表 6-10　C-E1 配对实现路径结果**

| 指标 | System A | B@0.92 | A−B | Crossed 95% CI |
|---|---:|---:|---:|---:|
| arrival→candidate selection/readiness 均值 | 27.70 ms | 62.38 ms | −34.69 ms | [−35.44, −33.95] ms |
| oracle TTFT_eff 下界均值 | 27.70 ms | 10.26 ms | +17.44 ms | [14.41, 20.32] ms |

完整 `output_token_ids` 仅 280/500 相同，首 token 为 465/500，长度/EOS/max-token 状态为 495/500；44/100 条唯一话语出现完整输出分岔。B@0.92 与 B-never 则在完整 token、首 token、长度、结束状态和文本上均为 500/500 一致。因此 C-E1 混合 tokenization、forward topology/shape、role boundary、kernel 与 Python scheduling，不能归因于单一 incremental-prefill 操作，也不能按输出一致子集作 post-treatment 筛选。

![图 6-4](figures/fig6_3.png)

**图 6-4　C-E1 双口径实现路径比较。** 方框为配对均值差的 crossed 95% CI；柱内离散范围是描述性统计。

## 6.6 RQ5：A2 受混杂的探索性描述

A2 每种实现包含 100 条记录。单一 Mistral 自动裁判的连贯性均值为朴素保留 3.76、轻量重写 3.62、打断标记 3.29；重写调用均值 639 ms，中位数 670 ms，线性插值 P90 约 935 ms，最大值 1165 ms。

三种实现分别重新生成首轮与下一轮回复，只有 33/100 个对话的兼容 `heard_text` 在三条件相同，朴素与重写成对相同为 49/100。评分差异同时混入首轮内容、断句边界和下一轮生成差异，因而不存在可解释的策略处理效应 estimand。上述数字仅描述本次运行，既不是负结果，也不能证明重写延迟已在真实用户发言期间隐藏。

## 6.7 本章结论

C2 v3 的 direct crop-integrity 与 within-run matched-arm recovery exactness gate 在冻结网格内全部通过；v1/v2 仍为 rejected，且本文不提出 clean-reprefill 等价主张。E3 作为下游支持证据，在 label-weighted 主分析、dialogue-weighted 与 target-specific exact-key 敏感性分析中均未确定方向。C-E2 刻画了同步 oracle 接受下的候选可用性、token 丢弃比例和乐观时延下界；C-E1 仅比较非 token-equivalent 实现路径。A2 只保留受混杂的探索性描述。所有结果均止于软件 runtime 层，不支持设备播放、声学接收或 HCI 推断。
---

# 第七章 讨论

## 7.1 核心贡献及其与已有工作的关系

本文的证据按 C2、E3、C1 和 C3 分层。**C2 是唯一核心机制贡献**：它把 software-consumed-sample cursor 经 TTS 文本片段解析为合法 assistant commit boundary，并将 KV、attention mask、token ledger、position、role 与 EOT 作为联合前缀状态修正。E3 检查该边界在冻结自动检测器下的 downstream 信息复现；C1 只刻画同步分段文本 harness 中的候选选择、oracle acceptance 与 discarded-token 工作点；C3 则是受混杂的探索性实现描述。

OpenAI Realtime API、Azure Voice Live 和 LiveKit Agents 已建立 playback-conditioned transcript/session-history truncation 的高层先例，KV crop 与 prefix reuse 也已有成熟原语。因此，本文不把“历史应反映已交付输出”或缓存裁剪本身列为原创。本文的增量是 **external-progress-conditioned joint prefix-state repair contract**：边界解析、联合状态、不变式保持转换与可证伪验证被组织为一条公开、可复算的级联实现路径。在截至 2026-09-03 的限定性公开来源检索中，未识别同时披露完整状态路径及 direct-integrity、matched-recovery 与时延工件的实现；这不是全球首次声明。

C2 v3 直接检验该合同中可识别的性质。production crop 后的 28 层 K/V 与同一 pre-crop snapshot 的保留前缀及逐层 slicing oracle bitwise exact；同一 accepted run 内，两条匹配臂从相同保留状态出发并接收相同 token-ID chunks 与操作序列后，恢复状态逐步 exact。这一结论是 direct crop integrity 与 within-run matched-arm recovery exactness，不是 clean re-prefill、跨进程、跨设备或生产端到端正确性。

## 7.2 结果解释

### 7.2.1 C2：联合状态合同，而非宽泛系统质量结论

C2 的作用是把“保留哪段历史”落实为推理 runtime 的显式状态转换。software cursor 首先经片段时间轴解析 token 端点；随后 KV length、mask、完整 token ledger、assistant content span、position、role phase 与 generation end reason 必须同步。结构 EOT 不进入 assistant 内容账本或内容 KV，而由唯一 close/reopen 入口提交一次。这些约束防止文本历史、缓存和角色结构在打断后指向不同前缀。

v1/v2 的 rejected 结果说明 clean-reprefill 数值对照并不能在不同 forward topology 下识别 crop 性质。v2 虽通过结构检查，但 2× numerical gate 为 42/45，且 control 与 production 的分块方式不匹配。v3 没有放宽门槛或改判旧实验，而是用同一 pre-crop snapshot 上、独立于 production crop 接口的逐层切片回答更窄的问题。因而，v3 提高的是 direct-integrity 证据的可识别性，不是对 clean-reprefill equivalence 的替代证明。

A1 和 P1 只补充成本边界。A1 表明在固定操作顺序和固定 32-token suffix 下，联合 crop+role 相对重新预填充的中位数比值随上下文长度增大；P1 刻画 prepared-state headless 软件控制路径。二者均不包含声卡、声学停止或完整服务并发，不能合并成生产 barge-in 时延。

### 7.2.2 E3：C2 的下游代理证据

E3 的四个 label-weighted generation-minus-playback 点估计均低于零，但区间均跨零，且未预设实质性差异阈值。dialogue-weighted 与 target-specific exact-key 敏感性分析也未给出稳定方向。因此，结果只说明当前冻结 target、trajectory、rule、judge、prompt 和 40-token cap 下的方向未被确定，不能推出 superiority、equivalence、noninferiority、harm 或 absence of effect。

片段目标与字符比例—空白边界 proxy 回答不同操作化问题；词面规则与单模型 judge 也不是可互换的 reference standard。exact-key 去重只移除重复键的额外 label 权重，不是语义聚类。自动代理的一致数同样不能代替盲法人工标注。E3 的角色因此是检查 C2 边界选择是否在固定自动测量下呈现可观察的后续差异，而非证明人类语义保真或交互质量改善。

### 7.2.3 C1：候选计算与提交时机的分离

C-E2 中，B@0.92 相对 never 的 candidate selection/readiness 差接近零，而 oracle `TTFT_eff` 下界差为正。两者不矛盾：前者测量最后文本段到达后的内部候选 token 选择，后者将 oracle 接受时已经存在的候选计为零等待。B@0.92 的 335/500（67%）是**全部 condition records 中接受时有候选可用的比例**，不是给定候选已启动后的条件存活概率；2.85% 是 pooled discarded-token ratio，不代表 FLOPs、GPU 时间、能耗或带宽损耗。

确认性 harness 由候选处理后的同步 oracle 触发接受，因此只识别 pre-oracle-acceptance candidate generation，不能证明候选在真实 end of speech 前就绪。first-deliverable、consumer 和约 291 ms 的 candidate-to-accept 间隔受同步程序顺序支配，只作诊断。若要估计真实提交前收益，需要独立异步 endpoint gate、consumer、TTS admission 与统一墙钟事件。

C-E1 则是非 token-equivalent implementation-path comparison。完整输出 token 仅 280/500 一致，说明其差异混合 tokenization、forward topology/shape、role boundary、kernel 与 Python scheduling。故 −34.69 ms 的 candidate-readiness 差不能归因于单一 incremental-prefill 操作，也不能通过筛选输出一致子集获得无偏的单机制效应。

### 7.2.4 C3：不可识别的探索性描述

A2 的三种实现分别重新生成首轮和后续回复，只有 33/100 个对话的兼容历史字段在三条件相同。连贯性分数和重写耗时因此只能描述当前运行，不能构成策略处理效应、负结果或零效应证据。重写可与下一轮用户输入并行是架构潜力；现有记录未测真实发言时长或重叠比例，不能声称其时延已经隐藏。

## 7.3 效度威胁

### 7.3.1 构念效度

本文严格区分 software-consumed samples、device-presented samples 与 acoustically heard content。现有游标仅属于第一层；应用队列、audio API、操作系统、驱动、设备缓冲和传播均可能使其偏离后两层。因此，P1 的 `leaked_samples=0` 只指软件计数器，不能转写为零设备或声学泄漏。

E3 的 fragment target 依赖 TTS 片段边界，proxy 依赖字符比例与空白吸附。词面规则可能把任务域词汇重合计为复现，单一 Mistral judge 也可能漏检或误检具体引用。没有盲法人工双标或用户实验，本文不推断真实语义保真、自然度、信任或 HCI 效果。

### 7.3.2 内部与结论效度

E1/E2 的 100 条唯一话语跨 5 个 process sessions 重复，正式 crossed/product bootstrap 同时重采样两个维度，但 session 仍只是技术重复。C-E1 不具 token equivalence；E3 的四个注入位置在部分 target 上形成重复 exact keys；A1 固定操作顺序与裁剪长度；P1 每 cell 仅 20 次，empirical P95 主要由 1–2 个上尾观测决定。相应分析均按其设计报告，不把观察性或描述性差异升级为因果、等效或生产 SLO。

E3 区间仅反映对话抽样不确定性，不包含检测器、提示词、模型或人类感知误差。E1/E2 的 crossed intervals 也只覆盖 100 条话语和已观察到的 5 个 process-initialization levels，不覆盖跨硬件、跨日、负载、并发或部署环境变异。C2 v3 的 exact gate 是有限确定性网格中的完整性证据，不是跨环境错误率估计。

### 7.3.3 外部效度

实验证据主要限于英文任务型 MultiWOZ 对话、Qwen2-7B-Instruct、特定 ChatML 类 role transition、Transformers `DynamicCache`、BF16/SDPA 与 RTX 3090。开放域长回答、中文及其他分词语言、不同 TTS、chat template、dtype、attention backend 或推理引擎都可能形成不同边界和状态。迁移时应重新验证 token serialization、结构 EOT、合法 crop point 与恢复转换。

A1、P1、E3 和 E1/E2 来自独立 campaign，其绝对时间不能池化或通过相减解释系统开销。本文也没有测量真实流式 ASR、在线异步 TTS 取消、声卡停播、loopback 波形、网络拥塞或生产并发。因此，结论适用于受测软件 runtime/prototype，不外推到 acoustic stop、真实 mouth-to-ear、生产 barge-in 或用户体验。

## 7.4 适用条件与后续证据

该合同迁移的最低条件是：TTS 能报告文本片段与音频块归属，推理引擎允许缓存裁剪，并能同步维护 mask、position、token ledger 与 role/EOT 状态。若后端提供词级 duration、设备时钟或 loopback 波形，可把 software-fragment boundary 扩展到更接近 device/acoustic boundary；若音频流不透明，则需要额外对齐与缓冲观测。

现有实验支持本文限定范围内的结论。真实异步音频闭环、固定轨迹 A2、盲法人工/HCI 标注、跨语言/模型/后端复验，以及随机 operation order 和多裁剪长度 A1，将分别增强外部、构念或因果效度，而不是改变本文对当前证据的解释。

## 7.5 本章小结

本文最稳健的发现不是生产时延或真实对话质量得到改善，而是外部软件进度可被组织为一套可检查的联合前缀状态修正合同。C2 v3 支持冻结环境下的 direct crop integrity 与 within-run matched-arm recovery exactness；E3、C1、A1/P1 和 C3 分别提供下游代理、候选工作点、协议成本与探索性描述，均不构成端到端效果证明。
---

# 第八章 总结与展望

## 8.1 主要结论

本文研究级联式语音对话系统在打断后的上下文状态修正。高层 playback-conditioned history truncation 与 KV crop primitive 均有既有先例；本文不以这些单项的新颖性立论，而提出并实现 **external-progress-conditioned joint prefix-state repair contract**：software-consumed-sample cursor 经 TTS fragment 映射为合法 assistant commit boundary，KV、attention mask、token ledger、position、role 与 EOT 在该边界上作为联合状态同步转换。合同由边界解析、联合状态、不变式保持转换和可证伪验证四层组成。

**C2 是唯一核心贡献。** 在冻结 Qwen2-7B-Instruct snapshot、BF16/SDPA、Transformers backend 和 24-case/27-event 网格内，production crop 后的 28 层 K/V 与同一 pre-crop snapshot 的保留前缀及逐层 slicing oracle bitwise exact；27/27 wrong-length negative controls 被检出。同一 accepted run 内，两条匹配臂从精确匹配的保留状态出发，接收相同 token-ID chunks 与操作序列后，其 K/V、logits、mask、token ledger 和 role/end/content state 在 60 个恢复步骤中逐步 exact。该结果只支持 direct crop integrity 与 within-run matched-arm recovery exactness。v1/v2 clean-reprefill 协议仍按冻结门槛 rejected，v3 不改变其 verdict，也不建立 clean-reprefill、continuation 或跨环境等价性。

A1 与 P1 给出该机制的成本边界。固定 32-token suffix 的联合 crop+role 微基准在 256–8192 token 上为 31.054–48.315 ms，重新预填充与联合路径的中位数比值为 2.254–40.620。prepared-state P1 的 stop→crop 和 stop→role 单元中位数分别为 2.44–2.53 ms 与 78.6–80.8 ms。前者是固定 GPU 微基准，后者是 headless 软件路径；两者均不是声卡、声学停止或生产端到端打断时延。

固定轨迹 E3 是 C2 的 downstream 支持性证据。label-weighted generation-minus-playback 点估计在 fragment/rule、fragment/judge、proxy/rule 和 proxy/judge 四个并列操作化中分别为 −3.37、−2.02、−1.58 和 −2.63 个百分点，对话聚类 95% CI 均跨零。dialogue-weighted 与 target-specific exact-key 去重敏感性分析同样未确定方向。结果仅适用于冻结自动检测器和目标构造，不支持 superiority、equivalence、noninferiority、harm、absence of effect、人类语义或 HCI 推断。

**C1 是支持性刻画。** 在 token-consistent C-E2 中，B@0.92 相对 never 的 candidate selection/readiness 差为 −0.03 ms（crossed 95% CI [−0.64, 0.61]），同步 oracle `TTFT_eff` 乐观下界差为 +20.80 ms（[17.85, 23.65]）。B@0.92 的接受时候选可用率为 335/500（67%），pooled discarded-token ratio 为 2.85%。这些量描述 pre-oracle-acceptance candidate generation，不证明真实 end-of-speech 前就绪或 production deliverability。C-E1 的 27.70 与 62.38 ms 是两条非 token-equivalent implementation paths 的 candidate-readiness 均值；由于输出、tokenization 与 forward topology 不完全一致，该差异不能归因于纯 incremental-prefill effect。

**C3 是探索性实现。** A2 报告朴素、重写和标记路径的描述性评分及重写耗时，但三条件使用不同生成轨迹，策略效应不可识别。因此，A2 不构成负结果、零效应或因果比较。

综上，本文建立的是受测软件 runtime 层的状态合同，而非真实听觉边界或完整系统效益证明。software-consumed cursor 不等于 device-presented samples 或 acoustically heard content；当前证据也不覆盖真实异步 ASR/TTS/播放器闭环、生产时延或用户体验。

## 8.2 后续工作

后续研究可从四个层次扩展本文证据。第一，接入在线 ASR、异步 TTS、bounded audio queue、设备时钟或 loopback 波形，统一测量 software、device 与 acoustic stop。第二，在固定 assistant token 轨迹、断句和打断点下重做 A2，并固定或成对控制下一轮解码，以形成可识别的策略比较。第三，采用盲法双标或直接用户研究评估特定信息复现、自然度、信任与交互质量。第四，在不同语言、模型、TTS、chat template、dtype、attention backend 与推理引擎上重新验证状态合同，并为 A1 随机化操作顺序、覆盖更多 crop length。

## 8.3 工件与声明入口

正式 campaign 状态、run identity、分析文件、复算命令与主张边界由仓库根目录 `REPRODUCIBILITY.md` 统一索引；E3 processed input 的可复算入口亦列于其中。作者、机构、许可、伦理、基金、利益冲突、CRediT、公开 artifact URL/DOI 与 AI 使用声明保留在 `paper2/declarations.md`，待目标期刊和责任主体确认后按期刊格式完成。

## 8.4 结语

本文将级联语音对话打断后的历史修正，从文本层原则推进为外部进度条件下的联合前缀状态转换。受控结果表明，合同中由 v3 覆盖的直接裁剪与匹配恢复性质通过 exact gates；其向设备呈现、声学接收、跨运行环境和人类交互效果的推广，仍需对应层级的直接证据。
---

# 参考文献

> 本表是本研究稿的统一引文清单，在线资料访问日期为 2026-09-03。近期预印本按检索时的官方 arXiv 记录标注为 preprint；提交到具体期刊时，应按该期刊格式导出同一元数据，并再次核对其发表状态。

[1] OpenAI. Realtime conversations[EB/OL]. [2026-09-03]. https://developers.openai.com/api/docs/guides/realtime-conversations.

[2] Microsoft. Handle voice interruptions in chat history[EB/OL]. 2026-04-28[2026-09-03]. https://learn.microsoft.com/en-us/azure/ai-services/speech-service/how-to-voice-live-auto-truncation.

[3] LiveKit. Events and interruption handling[EB/OL]. [2026-09-03]. https://docs.livekit.io/agents/build/events/.

[4] Zou W, Miao Y, Ma Z, Xu J, et al. LTS-VoiceAgent: A Listen-Think-Speak Framework for Efficient Streaming Voice Interaction via Semantic Triggering and Incremental Reasoning[EB/OL]. Preprint, arXiv:2601.19952, 2026.

[5] Mai L. RelayS2S: A Dual-Path Speculative Generation for Real-Time Dialogue[EB/OL]. Preprint, arXiv:2603.23346, 2026.

[6] Défossez A, et al. Moshi: A Speech-Text Foundation Model for Real-Time Dialogue[EB/OL]. arXiv:2410.00037, 2024.

[7] Chen J, Hu Y, Li J, et al. FireRedChat: A Pluggable, Full-Duplex Voice Interaction System with Cascaded and Semi-Cascaded Implementations[EB/OL]. arXiv:2509.06502, 2025.

[8] Hugging Face. Transformers v4.57.1 cache utilities: `DynamicCache` and `crop`[CP/OL]. [2026-09-03]. https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/cache_utils.py; Hugging Face. Cache strategies[EB/OL]. [2026-09-03]. https://huggingface.co/docs/transformers/v4.57.1/en/kv_cache.

[9] Li J, Lou J, Li J. IntentKV: Cross-Turn Intent-Aware KV Cache Pruning for Agent Inference[EB/OL]. Preprint, arXiv:2606.09916, 2026.

[10] Hooper C, Kang M, Moon S, et al. Speculative Interaction Agents: Building Real-Time Agents with Asynchronous I/O and Speculative Tool Calling[EB/OL]. Preprint, arXiv:2605.13360, 2026.

[11] Yang A, et al. Qwen2 Technical Report[EB/OL]. arXiv:2407.10671, 2024.

[12] TEN Team. TEN Turn Detection: Turn Detection for Full-Duplex Dialogue Communication[CP/OL]. 2025[2026-09-03]. https://github.com/TEN-framework/ten-turn-detection.

[13] Jiang A Q, et al. Mistral 7B[EB/OL]. arXiv:2310.06825, 2023; Mistral AI. Mistral-7B-Instruct-v0.3 model card[EB/OL]. [2026-09-03]. https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3.

[14] Du Z, et al. CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models[EB/OL]. arXiv:2412.10117, 2024.

[15] Eric M, et al. MultiWOZ 2.1: A Consolidated Multi-Domain Dialogue Dataset with State Corrections and State Tracking Baselines. LREC 2020. arXiv:1907.01669.

[16] Yang A, et al. Qwen3 Technical Report[EB/OL]. arXiv:2505.09388, 2025.

[17] Radford A, Kim J W, Xu T, et al. Robust Speech Recognition via Large-Scale Weak Supervision[C]//Proceedings of the 40th International Conference on Machine Learning. PMLR, 2023, 202: 28492–28518. https://proceedings.mlr.press/v202/radford23a.html.

[18] Schwarz A, He D, Van Segbroeck M, Hethnawi M, Rastrow A. Personalized Predictive ASR for Latency Reduction in Voice Assistants. Interspeech 2023: 745–749. https://doi.org/10.21437/Interspeech.2023-211.

[19] Kwon W, Li Z, Zhuang S, et al. Efficient Memory Management for Large Language Model Serving with PagedAttention. Proceedings of the 29th Symposium on Operating Systems Principles, 2023: 611–626. https://doi.org/10.1145/3600006.3613165.

[20] Zheng L, Yin L, Xie Z, et al. SGLang: Efficient Execution of Structured Language Model Programs. Advances in Neural Information Processing Systems, 2024, 37: 62557–62583. https://doi.org/10.52202/079017-2000.

[21] Beigel K. stream2sentence: Real-Time Processing and Delivery of Sentences from a Continuous Stream of Characters or Text Chunks, version 1.0.0[CP/OL]. 2026-06-24[2026-09-03]. https://github.com/KoljaB/stream2sentence.（本文锁定环境使用 1.0.0。）
