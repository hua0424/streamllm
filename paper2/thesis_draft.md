# 级联式语音对话中软件播放游标与 TTS 片段驱动的 KV 状态修正

> 全文合并草稿（自动生成，勿直接编辑；请修改分章 Markdown 后重新合并）。
> 本文件已于 2026-09-03 依据 crossed E1/E2 analysis v2、E3 weighting/dedup analysis v2、accepted C2 v3 及二审意见统一更新。

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
---

# 第一章 绪论

## 1.1 研究背景与问题界定

级联式语音对话系统将流式语音识别（Automatic Speech Recognition，ASR）、大语言模型（Large Language Model，LLM）和流式语音合成（Text-to-Speech，TTS）组合为可独立选择与调优的模块。与端到端语音模型相比，这一路线便于替换组件、观测中间状态并控制部署资源；其交互时延则同时受 ASR 分段、LLM 推理、TTS 产出和播放调度影响。

前期工作使用 ASR 稳定文本段和 LLM KV 缓存增量预填充，减少用户话轮结束后的上下文编码等待。本研究转向另一个状态问题：用户在 assistant 播报期间打断时，模型生成、TTS 合成和软件播放处于不同进度。若系统把尚未播放的生成内容完整写入下一轮历史，模型可能基于未通过该软件播放路径交付的信息继续生成；若删除整轮 assistant 状态，又会丢弃已保留前缀及其 KV 计算。

本文严格区分三个层级。`software-consumed samples` 是播放器软件报告的已消费采样；`device-presented samples` 是音频设备实际呈现的采样；`acoustically heard content` 还受系统缓冲、设备、传输和声学传播影响。本文观测并使用第一层，不测量后两层。因此，本文所称“播放感知”专指 software-consumed-sample cursor 驱动的 TTS 文本片段级状态操作，不表示逐词或逐 token 的听觉真值。

依据播放进度更新 transcript 或 session history 已有明确先例。OpenAI Realtime API 允许客户端按 `audio_end_ms` 截断未播放音频及对应 transcript[1]；Azure Voice Live 的 `auto_truncate` 在播放期打断后更新会话上下文，并公开说明其估算采用实时播放速度假设[2]；LiveKit Agents 也提供使被打断 transcript/history 与 spoken output 相匹配的框架语义[3]。本文既不主张“历史应反映已播放输出”这一高层原则的原创性，也不把 Hugging Face `DynamicCache.crop` 等缓存裁剪原语视为创新[8]。研究重点是公开级联实现中的跨层状态契约：如何从软件采样游标定位 TTS 片段和 assistant token span，如何原地裁剪 KV，并使 attention mask、token ledger、position 与 role/EOT 状态恢复一致。

## 1.2 研究问题与挑战

本文以 C2 为唯一核心机制贡献，并将 C1 和 C3 分别定位为支持性刻画与探索性扩展。为使问题、实验与结论一一对应，全文采用以下五个研究问题；编号只表示报告顺序，不表示贡献优先级。

1. **RQ1：固定被打断轨迹及固定自动检测器下，software-cursor playback 与 generation 历史的信息复现率有何差异？** 该问题只测 fixed-detector-conditioned information reproduction，并同时考察 label、dialogue 与 unique semantic boundary 三种加权/去重口径。
2. **RQ2：推测阈值如何影响作废计算与首候选 token 选择/内部计算就绪？** 本文的 speculation 是 pre-end-of-turn candidate-response generation with invalidation，不是 draft-target speculative decoding；同步 oracle 接受口径与 candidate-readiness 墙钟口径必须分开。
3. **RQ3：在受控同步文本段中，增量推测实现路径相对一次性预填充实现路径的候选计算就绪延迟有何差异？** 两条路径并非 token-equivalent，因此该问题比较完整 implementation path，而不识别单一 incremental-prefill 因果效应。
4. **RQ4：冻结模型与后端下，软件游标到 KV 裁剪及角色恢复的核心状态操作是否满足直接完整性合同，其模型侧成本和 prepared-state 软件控制路径时延如何？** 系统需要关联 software-consumed-sample cursor、TTS fragment 和 assistant token span，并同步维护逐层 K/V、attention mask、完整 token ledger、position 与 role/EOT state。
5. **RQ5：当前探索性运行中，朴素保留、打断标记与重写三种历史实现的连贯性分数和重写耗时如何？** 三种条件的首轮生成轨迹存在混杂，因此该问题只作描述，不检验策略改善的因果效应。

这些问题面临三类证据挑战。第一，缓存长度一致只是结构合法性的必要条件，不能单独证明逐层状态和恢复 logits 正确。第二，软件游标不是设备或人类听觉测量。第三，推测生成在同步实验中的内部事件顺序可能显著影响 first-deliverable 与 consumer marker，因此 candidate compute-readiness、oracle 下界和生产可交付性必须分别报告。

## 1.3 本文工作与贡献

### 1.3.1 核心贡献 C2：软件游标驱动的 KV 状态修正

本文实现 `software-consumed-sample cursor → TTS text fragment → assistant token span → in-place KV crop → mask/token/position/role/EOT recovery` 的跨层状态路径。片段时间轴负责游标反查；推理层使用 `DynamicCache.crop` 保留目标前缀，并同步维护 attention mask、完整 token ledger、assistant content span 及显式角色状态。EOT 不作为 assistant 内容重复入账；角色恢复根据结束状态只提交一次合法关闭边界。

正式正确性证据来自 C2 v3 direct crop-integrity addendum。该实验在受测 Qwen2-7B-Instruct snapshot/backend 上覆盖 24/24 cases、27/27 crop events、3 个 no-op、60 个 recovery steps，以及 27/27 wrong-length negative controls。每次事件中，pre-crop retained prefix、production post-crop cache 与 independent slicing oracle 在 28 层 K/V 上的 shape、dtype、device、hash 和运行时 `torch.equal` 均精确一致；keep/mask/token/sequence/KV 长度亦一致。向裁剪路径和匹配对照输入相同 token-ID chunks 后，K/V、logits、mask、token ledger、retained prefix 与 role/end/content state 精确一致。该结果支持受测 snapshot/backend 下的 direct crop integrity 与 matched-recovery determinism。

该证据不支持 clean-reprefill numerical equivalence、32-token continuation equivalence、跨模型/后端/硬件推广或在线音频与生产正确性。v1/v2 clean-reprefill 协议均按预先冻结门槛 rejected。v2 虽通过 24/24 probes 和 45/45 token/state/EOT/scenario checks，但预注册的单控制 2× 数值门槛仅通过 42/45；由于 control 与 production forward topology 不匹配，这三项失败既不能定位为 crop bug，也不能建立 clean-reprefill equivalence。v3 是直接裁剪完整性的补充证据，不改变 v1/v2 verdict。

模型侧 A1 微基准进一步刻画固定协议成本：在固定操作顺序、每次移除固定 32-token suffix、5 次 warmup 和 50 次重复下，256–8192 token context 的 joint crop+role median 为 31.054–48.315 ms，IQR 为 0.635–3.099 ms；re-prefill/joint median ratio 为 2.254–40.620。该结果仅适用于受测裁剪量与操作顺序，不代表自然打断位置分布。P1 则覆盖 headless software cursor 与模型状态控制路径；其每 cell 20 个观测的 P95 仅为描述性经验顺序统计量，不是生产 SLO。A1 与 P1 均不测量声卡、声学停播或生产端到端 barge-in。

### 1.3.2 支持性贡献 C1：候选计算就绪与浪费刻画

本文使用话轮检测置信度和单一阈值启动可作废候选响应生成，并用 discarded tokens 描述计算代价。确认性 C-E1/C-E2 采用 100 条唯一话语 × 5 个独立初始化进程 session × 10 个条件的交叉设计；每条件有 500 个观测，但内容采样单位是 100 条话语，session 是技术重复。正式不确定性使用 crossed/product bootstrap，独立重采样全局 session 与 dialogue 后取笛卡尔积。

C-E1 比较一次性 full-string/full-prefill 与 segment-wise/incremental 两条整体 implementation path，而非隔离单一增量预填充因素。System A 与 B@0.92 的完整 `output_token_ids` 仅 280/500 一致，首 token 465/500 一致；44/100 条唯一话语至少出现一次完整输出分岔。A−B@0.92 的 arrival→candidate-selection 差为 −34.6877 ms，crossed 95% CI 为 [−35.4421, −33.9535] ms；两路径差异混合 tokenization、forward topology/shape、role boundary、kernel 和 Python scheduling，不能归因于“纯 incremental-prefill effect”。

C-E2 在 B@0.92 与 B-never 之间保持 500/500 的完整 token、首 token、长度、EOS/max-token 与文本一致性。never−B@0.92 的 arrival→candidate-selection 差为 −0.03349 ms，95% CI 为 [−0.63861, 0.61494] ms。post-candidate oracle acceptance 下，never−B@0.92 的 TTFT_eff 乐观下界差为 +20.8037 ms，95% CI 为 [17.8492, 23.6450] ms；B@0.92 survival 为 67.0% [58.0%, 76.0%]，pooled waste 为 2.8527% [1.1239%, 4.7345%]。该 oracle 结果是条件性收益上界，不是生产 deliverability 改善。同步执行顺序支配的 first-deliverable 与 consumer markers 仅作 diagnostics；candidate-first-selection 到 post-candidate oracle acceptance 的 291 ms 中位内部间隔也不表示自然端点提前量或用户继续说话时长。

### 1.3.3 探索性贡献 C3：历史自然化实现与受混杂负结果

本文实现朴素保留、显式打断标记和轻量模型重写三种历史自然化路径，并记录自动连贯性评分与重写耗时。由于不同条件的首轮回复由独立生成轨迹产生，现有结果受内容差异混杂，不能识别策略的因果效应。该部分因此作为探索性实现与负结果报告，不声称标记或重写改善后续对话。

固定轨迹 E3 另行比较 generation 与 software-fragment playback retention 条件下的信息复现。主 estimand 为 label-weighted、fixed-detector-conditioned information-reproduction rate。fragment/rule、fragment/judge、proxy/rule 和 proxy/judge 的 generation−playback 点估计分别为 −3.37、−2.02、−1.58 和 −2.63 个百分点，对话聚类 95% CI 依次为 [−10.49, 3.40]、[−10.70, 6.13]、[−6.08, 2.67] 和 [−8.57, 2.90]。按 target-specific unique semantic boundary 去重后，fragment/rule 为 −2.96 个百分点 [−9.04, 2.63]，fragment/judge 为 0.00 个百分点 [−7.98, 7.47]；proxy 仅移除一个重复，结果近似不变。这些区间只表示在冻结规则、特定 judge、targets、trajectory、prompt 与 40-token cap 条件下的 dialogue-sampling uncertainty，不包含 detector error、prompt/model variation 或 human-perception error。结果不支持 superiority、equivalence、noninferiority、harm 或 absence-of-effect。

## 1.4 研究定位与 novelty 边界

截至 2026-09-03 的 targeted public-source scan 计划检索跨出版商学术索引、arXiv、ACL Anthology、ISCA Archive/Interspeech、ACM Digital Library、IEEE Xplore、NeurIPS Proceedings、DOI/Crossref 元数据、第一方产品文档和第一方开源仓库；其中部分索引或页面在补查时受反自动化、超时或访问控制限制，未将访问失败解释为零结果。查询族覆盖 cascaded/streaming spoken dialogue、barge-in/turn-taking、pre-end-of-turn candidate response、KV crop/rollback/prefix reuse，以及 playback/listening-aware history；纳入规则要求来源涉及至少一个相邻机制并具有 DOI、正式出版页、官方预印本或第一方 URL，排除了纯 draft-target speculative decoding、与对话回滚无关的通用 KV 压缩、只停止音频而未说明历史修正的接口，以及可由第一方来源替代的二手材料。

该检索识别出 playback-conditioned transcript/session-history truncation 的 OpenAI、Azure 和 LiveKit 先例[1–3]，输入侧提前计算与候选响应的 Predictive ASR、LTS-VoiceAgent 和 RelayS2S[4,5,18]，打断检测与全双工架构的 FireRedChat 和 Moshi[6,7]，以及 `DynamicCache.crop`、PagedAttention、RadixAttention 和跨轮 KV pruning 等缓存先例[8,9,19,20]。在所报告的公开来源范围内，未识别到同时公开 software cursor→TTS fragment→assistant token span→in-place KV crop→explicit role/EOT recovery，并提供可复算 direct state-integrity 与 latency evidence 的级联实现。这是范围受限的非识别结果，不排除未发表、闭源或因索引访问受限而未收录的系统，也不构成全球首次声明。检索过程、查询式、纳排规则、最近邻矩阵和访问限制记录于 `docs/novelty_search_2026-09-03.md`。

## 1.5 论文组织结构

第二章评述播放感知历史、提前响应计算、打断控制和 KV 状态管理的相关工作，并报告 scoped novelty scan；第三章形式化软件游标、TTS 片段和 token 状态边界；第四章说明候选响应调度、KV 裁剪与角色恢复及历史自然化实现；第五章描述系统与验证方法；第六章分别报告 E3、C-E1/C-E2、C2 v3、A1、P1 和 A2，避免跨层级合并推断；第七章讨论构念、内部、外部和结论效度；第八章总结证据边界与后续研究。
---

# 第二章 相关工作

本文相关工作横跨五个相邻问题：播放条件下的会话历史截断、流式与增量语音对话、用户打断与话轮控制、话轮结束前的候选响应计算，以及 KV 缓存裁剪与前缀复用。高层播放感知历史原则和缓存裁剪原语均已有先例；本章据此界定本文跨层实现的范围，而不以单一组件的新颖性立论。

## 2.1 播放条件下的 transcript 与 session-history 截断

OpenAI Realtime API 的 `conversation.item.truncate` 允许客户端提交 `audio_end_ms`，从会话条目中移除未播放音频及其对应 transcript[1]。Azure Voice Live 的 `auto_truncate` 在播放期间检测到用户语音后更新上一轮响应和 session context；其公开文档明确说明截断估算采用实时播放速度假设[2]。LiveKit Agents 也提供截断被打断 transcript/history、使消息状态与 spoken output 相匹配的框架语义[3]。这些资料共同表明，按播放进度修正会话历史是既有工程实践，而非本文提出的新原则。

公开接口语义不能揭示闭源服务的内部推理架构。OpenAI 和 Azure 是否采用级联模型、如何表示 token span、是否原地裁剪 KV，均不能从其文档推断。LiveKit 公开框架层消息处理，但其相关资料不公开 transformer KV、attention mask、token ledger 与 role/EOT state 的联合恢复。因此，本文与这些系统的区别是公开研究对象和证据层级，而不是声称商业或开源框架“没有”某项未公开内部机制。

上述系统使用的播放概念也不完全等同于物理听觉测量。Azure 的实时播放速度假设和 OpenAI 的客户端 `audio_end_ms` 都属于接口或软件时序语义；设备缓冲和声学传播仍需独立测量。本文同样只观测 software-consumed-sample cursor，并将其映射为 TTS-fragment-level software retention boundary。该操作不提供 device-presented samples 或 acoustically heard content 的真值。

## 2.2 流式语音对话、提前计算与打断控制

流式级联研究通常通过语义触发、增量推理或输入预测缩短等待。LTS-VoiceAgent 使用 semantic triggering 和 incremental reasoning 组织 Listen–Think–Speak 流程[4]。RelayS2S 采用双路径 response-level candidate prefix 与验证/续写机制[5]。Personalized Predictive ASR 从 partial ASR 预测完整输入，并预取下游响应；最终识别结果确认预测后采用缓存结果[18]。三者均构成“在输入最终确认前进行下游计算”的先例，但研究信号位于用户输入或候选响应侧，未公开由 assistant 播放游标触发的 token/KV 历史修正。

本文的 supporting C1 与这类工作共享 compute-before-commit 思路，但术语和结论范围更窄。本文 speculation 指 pre-end-of-turn candidate-response generation with invalidation，不是固定 prompt 上由 draft model 与 target model 协作的 speculative decoding。`first_token_ready` 是 first-candidate-token selection/candidate compute-readiness；同步 harness 的 `endpoint_accept` 是 post-candidate oracle acceptance。该实验刻画候选存活、wasted tokens 和 oracle TTFT_eff 乐观下界，不证明 production consumer delivery、TTS admission 或 acoustic output 得到改善。

打断检测研究主要回答何时停止系统输出。FireRedChat 使用 streaming/personalized VAD 与 interruption control，并在确认打断后控制 TTS[7]。这种检测与本文的状态修正互补：前者产生或确认 interruption event，后者在事件到达后决定 software-fragment prefix 及相关模型状态如何保留。将两者区分可避免把“检测到打断”误写为“完成了多轮历史修复”。

端到端全双工系统提供另一类架构参照。Moshi 通过并行 speech/text streams 支持重叠交互[6]。这种同步建模减少级联 ASR、LLM 和独立 TTS 之间的部分中间错位，但并不据此证明模型流位置、设备呈现位置和声学听觉位置恒等。网络、应用队列、音频 API 和设备缓冲仍可能形成交付差异，因此端到端与级联设计不能仅凭架构标签完成听觉边界比较。

## 2.3 KV 缓存裁剪、前缀复用与跨轮状态

自回归 transformer 的 KV cache 保存历史 token 的中间状态，以免每一步重复计算完整前缀。Hugging Face Transformers 提供 `DynamicCache` 及 `crop` 操作，可将缓存缩短至指定序列长度[8]。因此，KV crop primitive 本身不是本文创新。

PagedAttention 将 KV cache 组织为非连续固定大小块，并通过按需分配与 copy-on-write 改善 serving memory efficiency[19]。SGLang 的 RadixAttention 以 radix tree 管理可跨请求复用的公共 token 前缀[20]。这些方法建立了 KV 内存管理和 prefix reuse 的重要先例，其优化目标主要是吞吐、内存利用或公共前缀共享，而非依据外部 software playback cursor 选择被打断 assistant 的对话提交边界。

IntentKV 处理 text-agent 的 cross-turn intent-aware KV pruning[9]；Speculative Interaction Agents 研究异步工具调用中的推测结果和作废控制[10]。这些工作表明，跨轮 KV 或推测状态可以受外部控制逻辑影响，但其信号来自文本意图或工具状态，不涉及 software-consumed samples、TTS fragment、assistant content span 与 chat role/EOT state 的联合恢复。

本文的技术对象因而不是一种新的缓存数据结构，而是跨层状态合同。游标查询先给出 TTS 片段级软件保留边界，再定位 assistant token span；KV crop 必须与 attention mask、完整 token ledger、assistant content ledger、position 和 role/end state 同步。C2 v3 进一步以 independent slicing oracle 检查每层 K/V 的 direct crop integrity，并在 identical token-ID chunks 下检查 matched-recovery determinism。该证据只覆盖受测 snapshot/backend；它不建立 clean-reprefill numerical equivalence，也不能替代在线音频或跨引擎验证。

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

相关工作支持三层贡献定位。第一，C2 是核心机制：其增量位于软件游标、TTS 片段、assistant token span、KV/mask/ledger/position 与 role/EOT state 的公开联结及 direct-integrity evidence，而不在 playback-history 原则或 KV crop primitive。第二，C1 是 supporting characterization：它报告 candidate selection/compute-readiness、post-candidate oracle acceptance、survival 与 wasted-token 工作点，不声称 speculative decoding 或 production latency improvement。第三，C3 是 exploratory extension：三种历史自然化实现及其受混杂负结果不构成策略优越性证据。

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

`RolePhase` 至少区分 user role 已打开、assistant role 已打开以及 `ASSISTANT_EOT_PENDING`。`GenerationEndReason` 显式记录 `NONE`、`EOS`、`MAX_TOKENS`、`CONSUMER_STOP` 或 `CROPPED`，不能再由生成长度或账本末 token 反推。

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

### 3.4.2 一致性指标

设固定首轮生成轨迹中 playback 片段边界之后、generation 条件额外保留的差异文本为 $W$，其后两轮回复集合为 $R$。固定轨迹 E3 在两种历史条件下使用完全相同的 $W$，并记录后续回复是否复现其中的信息。本文采用两种目标口径。

- **片段目标（fragment）**：$W_{\mathrm{frag}}$ 为片段级 software-cursor 端点之后的共享差异文本。只有当该目标非空时，配对记录才进入片段目标分析。
- **字符比例—空白边界近似目标（proxy）**：$W_{\mathrm{proxy}}$ 将式（3-4）的命中片段文本尾部与 $W_{\mathrm{frag}}$ 拼接。该口径纳入片段内代理尾部，但不是设备或声学边界；其分析资格必须依据 $W_{\mathrm{proxy}}$ 自身是否非空确定。

引用判定使用固定词面规则与固定 `specific-reference-v3` Mistral judge。E3 的 estimand 是**固定检测器条件下的信息复现率**，不是人类语义真值或 HCI 效果。区间只表示在冻结规则、裁判、目标、轨迹、提示词与 40-token cap 条件下的 dialogue-sampling uncertainty，不包含检测器误差、提示词/模型变动或人类感知误差。

同时，本文区分“software-cursor 条件是否写入局部完整游标外文本”这一结构合规问题和“共享差异文本是否在后续回复中复现”这一代理后果。前者是可由边界和文本长度直接检查的构造性性质；后者只能由固定规则或模型代理估计。结构检查不得与语义代理分析的分母合并。

### 3.4.3 效率指标

推测浪费率定义为

$$
\rho=\frac{\sum\text{作废的候选 token 数}}
{\sum\text{作废的候选 token 数}+\sum\text{最终生成 token 数}}. \tag{3-9}
$$

式（3-9）的 pooled 口径与确认性 E1/E2 campaign 的正式 estimand 相同。八个数值阈值和一个 never-speculate 对照构成九个 B-path 工作点：

$$
\bigl(\rho(\theta),\mathrm{TTFT}_{\mathrm{eff}}(\theta)\bigr).
$$

阈值降低通常提高候选生成覆盖率，也可能增加作废计算。第六章同时报告各工作点的到达—首候选选择延迟与候选存活率；有限个测试点只能支持受控工作点刻画，不自动构成连续或严格单调的 Pareto 前沿。

KV 复用收益通过“重新预填充耗时中位数 / 同一计时区间联合执行 crop 与角色恢复的耗时中位数”描述。该比值只适用于 A1 的固定顺序和固定 32-token suffix 协议。本文以联合路径为主要分母，并把 crop-only、role-only 作为局部诊断；不以两个独立中位数之和替代联合路径中位数。

### 3.4.4 实验单位与路径可比性

确认性 E1/E2 采用 $100$ 个唯一话语与 $5$ 个独立初始化进程 session 的交叉设计。每条件共有 $100\times5=500$ 个 session×utterance 观测，但内容采样单位是 100 个唯一话语，session 是技术重复，不把 500 个观测解释为 500 个独立内容样本。正式 `analysis_v2` 使用 crossed/product bootstrap：独立重采样全局 session 与全局话语，再取笛卡尔积；重复 10,000 次，seed 为 20260901，并报告 percentile 95% 区间。

C-E1 比较一次性 full-string tokenization/full-prefill 的 System A 与 segment-wise tokenization/incremental 的 B@0.92。由于两条路径不保证 token 等价，C-E1 是**实现路径比较**，混合 tokenization、forward topology/shape、role boundary、kernel 与 Python scheduling，不能归因为纯 incremental-prefill 效应。C-E2 比较 B@0.92 与 B-never，两者沿相同 B-path 且正式记录的 token 输出一致，可用于 B 路径内部的阈值策略比较。主延迟分析不能只筛选 C-E1 输出相同的记录，因为这会形成结果之后的选择。

### 3.4.5 指标与实验对应关系

| 研究问题 | 主要指标 | 实验 |
|---|---|---|
| 固定轨迹下两种历史边界的固定检测器条件信息复现率有何差异 | 片段目标、字符比例—空白边界近似目标 | E3 |
| 推测阈值如何影响候选计算与 oracle 响应下界 | $\rho$、$L_{\mathrm{arr}\to\mathrm{cand}}$、候选存活率、$\mathrm{TTFT}_{\mathrm{eff}}$ | E2（同时作为 A3） |
| 两条非 token-equivalent 实现路径在受控文本输入下的指标有何差异 | $L_{\mathrm{arr}\to\mathrm{cand}}$、诊断 markers、$\mathrm{TTFT}_{\mathrm{eff}}$、建模 mouth-to-ear | E1 |
| KV 状态复用及软件控制路径的时延表现如何 | A1 固定协议下联合 crop+角色恢复/重新预填充耗时；P1 软件 stop 确认、反查及累计恢复端点 | A1、P1 |
| 当前探索性运行中三种历史处理实现的表现如何 | 连贯性评分、重写耗时 | A2 |

## 3.5 本章小结

本章把 $p$ 限定为 software-consumed-sample cursor，并将其与 device-presented samples 和 acoustically heard content 分开；$\widehat H(p)$ 仅表示 TTS 片段级软件保留边界。持久化状态显式包含全局 `token_ids` ledger、assistant 内容 ledger、`RolePhase`、`GenerationEndReason` 与内容 span；预测 EOT 进入 `ASSISTANT_EOT_PENDING`，不进入内容账本、时间轴或内容 KV，由 `reopen_user_role()` 唯一提交结构 close。延迟口径改为首候选 token 选择/内部计算就绪、候选后 oracle 接受以及仅供诊断的 first-deliverable/consumer markers，不再作生产可交付性推断。最后，本章明确了 100 个唯一话语与 5 个 session 的交叉设计、C-E1 的非 token-equivalent 实现路径边界、A1 固定 32-token suffix 和 P1 经验 P95 的适用范围。
---

# 第四章 方法设计

## 4.1 总体设计

本文在本项目一期内部实现的“流式 ASR 稳定文本段—LLM 增量 KV 预填充”流水线之上，增加输出断句、流式 TTS、软件播放时间轴和打断状态修正，形成图 4-1 所示闭环。

![图 4-1](figures/fig4_1.png)

**图 4-1　系统总体架构。** 输入侧以稳定 ASR 文本段驱动增量预填充和候选生成；输出侧将内容 token 流切分为 TTS 文本片段并登记音频块；打断侧依据 software-consumed-sample cursor 查询片段级保留边界，执行 KV 裁剪和角色恢复，并可选用历史自然化策略。

系统遵循两项设计原则。第一，打断后的历史按软件游标对应的 TTS 片段边界保留；这一高层思想已有工程先例，本文的核心是把 software cursor、片段、assistant token span、KV、mask、token ledger、position、role 与 EOT 状态组成可检查的跨层契约。第二，话轮结束前生成的候选必须可作废、可计量并可由阈值调节。本文的 speculation 是 **pre-end-of-turn candidate-response generation with invalidation**，不是 draft-target speculative decoding。

这里的 playback-aware 不表示设备或声学观测。本文只取得软件播放器消费的采样游标；设备已呈现采样和用户声学上听到的内容需要设备时钟、loopback 或其他物理测量，均未由本方法实例化。

## 4.2 可作废的候选响应生成

### 4.2.1 单一推测阈值与候选后接受

TEN Turn Detection 对累计文本给出连续置信度

$$
c_i=\operatorname{conf}(u_1\cdots u_i)\in[0,1],
$$

并以单一推测阈值 $\theta$ 控制话轮结束前的候选响应生成。当 $c_i\geq\theta$ 时，系统冻结推测前状态快照，打开 assistant role，并预生成不超过预算 $B$ 的候选内容 token。$B$ 限制单次误触发的计算成本。

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

八个数值阈值加一个 never-speculate 对照得到九个有限工作点。它们描述同步受控 harness 中的候选计算、oracle 响应下界、存活率与浪费率，不预设曲线严格单调，也不证明异步在线系统的可交付时延改善。

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

`GenerationEndReason` 是阶段状态而非永久日志。crop 后的 `CROPPED` 必须在新内容推进前可见；成功的 `prefill_user_text()`、`prefill_assistant_text()`、`open_assistant_role()` 或规范 reopen 会根据新阶段重置状态，其中 D-022 特别要求 user 内容一旦成功追加就清除陈旧 `CROPPED`。编排器另保存候选结束原因快照，避免为了保留审计信息而污染当前运行状态。

![图 4-4](figures/fig4_4.png)

**图 4-4　KV 裁剪与角色边界恢复。** 第一步原子裁剪 K/V、mask、完整 ledger 与 assistant 内容 ledger；第二步依据裁剪语义恢复 `RolePhase`；第三步由 `reopen_user_role()` 唯一提交 assistant close 与下一 user-open。结构 EOT 不属于 assistant 内容。

### 4.3.4 C2 v3 direct crop-integrity 验证方法

C2 的正式正确性方法采用 protocol v3 exact-only addendum，而不是 clean re-prefill 对照。固定 Qwen2-7B snapshot、BF16、SDPA 与 Transformers backend，设置 24 个 ordered cases，覆盖 512/2048/8192 token context、$p=0$、片段边界、中段吸附、reply tail、pending EOT、推测全作废、下一轮与第二次 crop。24 个 case 共产生 27 个 ordered crop events，其中包含 3 个 no-op event；冻结 assistant fixture 共 308 个内容 token，必须全部经 `generate_accumulating` 的 production 路径逐 token append，每个 token 对应一次受控 `_prefill_ids_p2` forward。

每个 crop event 设三方 exact 比较：

1. **pre-crop retained prefix**：crop 前从 production K/V 取得目标长度前缀的逐层 manifest；
2. **production post-crop**：唯一调用生产 `crop_to_token()` 后的状态；
3. **independent slicing oracle**：从 crop 前 snapshot 独立 clone，并按推导出的 keep length 逐层切片，不调用生产 crop 接口。

三方逐层比较 K/V 的 shape、dtype、device、SHA-256 和 runtime `torch.equal`，并要求 keep length、mask、完整 token ledger、sequence length 与 KV length exact。wrong-length disposable negative control 必须对每个 event 被拒绝，以证明 gate 能检测错误保留长度。

crop 后，production arm 与 direct oracle 接收完全相同的 token-ID chunks，执行 60 个 matched-recovery steps。每一步要求 K/V、logits、mask、完整 token ledger、retained-prefix hashes，以及由操作序列独立推导的 role/end/content state bitwise/exact 一致。该设计验证的是受测 snapshot/backend 下的 direct crop integrity 与 matched-recovery determinism。

v1/v2 采用 canonical clean re-prefill 对照，均按冻结门槛 rejected。v2 虽然 24/24 termination probe 和 45/45 token/state/EOT/scenario gate 通过，但单控制的 2× numerical gate 仅 42/45。进一步审查发现 control 按语义 seam 分块并强制末 token 单独 forward，而 production 初始 context 和 role/content append 使用不同 forward topology，故该 control 不能识别三项数值失败来自 crop 还是拓扑差异。v3 没有把门槛事后放宽，也没有改判 v1/v2；它改问可由 exact slicing oracle 识别的问题。因此，v3 **不声称** crop+role 与 clean re-prefill 的数值、logit 或 continuation 等价，也不支持跨模型、dtype、backend、硬件、在线音频或生产端到端正确性。

### 4.3.5 对照条件的语义

- **按生成位置保留（B-gen）**：保留完整已生成回复。若 $G>\widehat H(p)$，则完整的 software-cursor 外片段可能进入历史。
- **重新预填充（B-noKV）**：丢弃 KV，根据裁剪后的文本重新执行模型前向。该条件只用于 A1 固定协议下的计算耗时对比；C2 v3 不以其作为数值等价 oracle。
- **按合成位置保留（B-syn）**：以合成边界修正历史。在本文同步 Mock TTS 条件下，它无法与 B-gen 形成可区分时序，故不作为已验证条件。

## 4.4 被打断历史的处理策略

片段级保留会把 software cursor 命中的当前片段整体写入历史，其中可能包含游标尚未覆盖的文本尾部；某些断句片段也可能在语义上不完整。本文在同一片段边界之上实现三种策略。

- **朴素策略**：直接保留片段级 assistant 前缀。
- **标记策略**：在保留前缀后追加省略号等打断标记。它不调用额外模型，但仍产生少量 tokenization 和预填充开销。
- **重写策略**：使用 Qwen3-0.6B 将保留前缀自然收束，并在提示词中要求不新增事实。该要求是设计约束而非形式保证；KV 需要回退到本轮 assistant 起点，再预填充重写文本。

重写可以在下一轮用户输入期间异步执行，因此具有隐藏部分延迟的潜力；当前实现和实验未测量真实重叠比例。现有 A2 为三种策略分别重新采样首轮和下一轮回复，未隔离策略效应，所以研究问题仅描述当前探索性运行的连贯性分数与重写耗时，不回答“是否改善”。

## 4.5 实验设计边界

确认性 E1/E2 是 100 个唯一话语与 5 个独立初始化进程 session 的交叉设计。每条件有 500 个 session×utterance observations，但 100 个话语是内容采样单位，5 个 session 是技术重复。正式 versioned `analysis_v2` 不覆盖 `analysis_v1`，而是使用 10,000 次 crossed/product bootstrap：独立抽样全局 session 与 utterance，再对二者笛卡尔积计算配对 estimand；seed 为 20260901，区间为 percentile 95%。

C-E1 的两条实现路径不是 token-equivalent：System A 对完整字符串一次 tokenization 并 full prefill；System B 对文本段分别 tokenization 并 incremental prefill。故 C-E1 估计整体 implementation-path difference，混合 tokenization、forward topology/shape、role boundary、kernel 与 Python scheduling，不能隔离纯增量预填充效应。正式输出检查只用于披露路径差异，不能据此过滤主延迟记录。C-E2 在相同 B path 内比较 B@0.92 与 never-speculate，输出 token 序列一致，因此具有更强的路径内可比性。

固定轨迹 E3 使用 label-weighted estimand 与同一口径的 dialogue-cluster bootstrap，并另报 dialogue-weighted 与 unique-semantic-boundary sensitivity。它量化固定检测器条件下的信息复现率，不支持 superiority、equivalence、noninferiority、harm、absence-of-effect 或人类感知结论。

## 4.6 本章小结

本章将辅助 C1 限定为话轮结束前候选生成的 selection/compute-readiness、post-candidate oracle acceptance 与浪费工作点刻画；把核心 C2 定义为 software cursor→TTS fragment→assistant token span→KV/mask/ledger/position/role/EOT 的状态契约。方法强制时间轴生产者顺序、不接受乱序写入；使用完整 token ledger、显式 `RolePhase` 和 `GenerationEndReason`；由 `ASSISTANT_EOT_PENDING` 与唯一 close commit 消除重复 EOT。C2 v3 以 24 cases、27 crop events 的三方 exact slicing 与 matched recovery 验证 direct crop integrity，不将已 rejected 的 clean-reprefill 协议改写为通过。最后，本章把 E1/E2 交叉重复、C-E1 非 token-equivalent 实现路径、A1/P1 协议边界及 A2 探索性定位纳入方法合同。
---

# 第五章 系统实现

## 5.1 模块架构

系统复用本项目一期内部实现的、基于 Whisper[17] 的流式 ASR 和 LLM 增量预填充能力，并新增对话编排、输出断句、流式 TTS、软件播放状态和实验驱动程序。表 5-1 给出主要模块及其验证对象。

**表 5-1　主要模块、接口与验证对象**

| 模块 | 主要职责 | 关键验证对象 |
|---|---|---|
| `src/dialogue/timeline.py` | 保存 TTS 文本片段、token 区间、音频块和累计软件采样区间；按 software-consumed cursor 反查 | token/sample 连续性、chunk 唯一归属、乱序拒绝、裁剪端点 |
| `src/llm/stream_llm_inference.py` | 增量预填充、assistant 侧 KV 累积、缓存裁剪、EOT 与角色恢复 | KV/mask/完整 ledger/内容 ledger 长度一致，role/end 状态与 position 连续 |
| `src/tts/sentence_chunker.py` | 基于 stream2sentence[21] 将 token 解码流切分为 TTS 文本片段并关联 token 区间 | 非空白字符守恒、空片段和末端钳制 |
| `src/tts/streaming_tts.py` | 定义流式 TTS 接口、Mock TTS 与真实后端适配 | 片段—音频块归属、画像参数读取 |
| `src/player/player.py` | 维护软件已消费采样游标和停止接口 | 采样计数、单调查询与 seek 语义 |
| `src/dialogue/trigger.py` | 计算话轮完成置信度 | 类别 token 配置、阈值触发记录 |
| `src/dialogue/rewriter.py` | 重写被打断轮的保留前缀 | 调用耗时、输出非空和替换路径 |
| `src/dialogue/orchestrator.py` | 串联用户输入、候选生成、断句、TTS、打断和状态修正 | 推测作废、oracle 接受、EOT/role 状态、逐轮指标 |
| `experiments/scripts/` | 旧二期实验驱动程序及离线重分析 | 旧结果保护、fixture 清洗和统计复算 |
| `experiments/sci34_supplement/` | 固定轨迹 E3、联合 A1、prepared-state P1、确认性 E1/E2、C2 v1/v2/v3 | 配对目标、计时边界、crossed analysis、exact crop/recovery gates、seal |

上述结构把机制实现和实验实例化分开：编排器面向统一接口，实验脚本决定使用真实后端还是画像驱动的模拟后端。本文中的 playback 状态只来自软件已消费采样游标，不包含 device-presented sample clock 或 acoustically heard content measurement。

## 5.2 片段时间轴与断句对齐

`PlaybackTimeline` 以 TTS 文本片段记录为主轴。片段生成后首先登记文本和 assistant 内容 token 区间；TTS 每产生一个音频块，就将其附加到对应片段并更新累计软件采样区间；播放器维护 software-consumed-sample cursor。打断查询读取这一游标，在片段采样区间中定位命中记录，并返回片段末 token 作为保留边界。

计数边界采用 $[0,p)$ 已被软件消费的语义。当 $p=\operatorname{se}(f_k)$ 时，软件游标恰好覆盖片段 $f_k$，应保留该片段而不是落到下一片段。实现分别处理片段中部、片段末端、首采样、空时间轴和越过末端等情况，避免大于/大于等于选择导致一个片段偏移。这里的 `heard_text`、`n_heard` 等 legacy 字段仅为 artifact compatibility alias，不能解释为声学真值。

写入 API 不依赖调用方“自觉有序”，而是显式验证生产者合同。`add_fragment()` 要求 token span 非空、单调且与上一片段连续；重复、重叠、倒退或 gap 均抛错。片段 token span 冻结后才能 `attach_chunk()`；已关闭、已丢弃或已完成的片段拒绝追加。每个 `chunk_id` 必须全局唯一并归属唯一 fragment，chunk 按所属片段和生成次序附加。sample range 必须非空且与全局累计采样端点连续，禁止 gap、overlap 和倒序。软件游标必须单调且不越过已登记范围。测试覆盖错序 fragment、重复/错属 chunk、关闭后追加、sample gap/overlap 与 cursor regression，并要求全部 fail-closed，而不是重排或静默修正。

断句器产出文本字符串，LLM 状态使用 token 端点。系统在逐 token 解码时累计非空白字符前缀和；文本片段产出后，用其非空白字符长度定位最后一个被覆盖的 token。该方法不假设 tokenizer token 边界与标点或词边界一致，但要求片段拼接后的非空白字符序列与原始生成文本守恒。实现跳过纯空白片段并把区间末端钳制到实际内容 token 数，以防空片段错误消费下一 token 或产生越界 crop。

时间轴使用一把锁序列化短小的片段与 chunk 更新。锁只解决并发原子性，不替代 token span、fragment lifecycle、chunk ownership 和 sample ordering 验证。第六章未测量高并发尾延迟，因此本文只主张这些生产者不变式与乱序拒绝行为，不声称任意并发规模下的性能。

## 5.3 KV、token ledger 与角色状态

### 5.3.1 统一缓存容器

一期生成循环的调用方只持有用户侧预填充缓存，assistant 生成 token 的 KV 未持续写回，因而无法支持播放期裁剪。二期 `AccumKVCache` 将以下对象作为一个逻辑状态维护：

- `past_key_values: DynamicCache`；
- `pre_attention_mask`；
- 覆盖完整缓存序列的 `token_ids` ledger；
- 仅含当前 assistant 内容的 `assistant_token_ids` ledger；
- `RolePhase`；
- `GenerationEndReason`；
- assistant role start、assistant content start/end 等边界。

所有 user/assistant 文本、role transition 与恢复 token 都经 token-ID prefill 核心追加。每次稳定操作后强制检查

$$
\operatorname{len}(token\_ids)
=\operatorname{len}(pre\_attention\_mask)
=\operatorname{seq\_length}
=\operatorname{DynamicCache.get\_seq\_length}(),
$$

并要求 `assistant_token_ids` 与 `token_ids` 的当前 assistant content span 完全一致。后续 position IDs 由真实 past length 重算，不沿用 crop 前的位置。

### 5.3.2 tokenwise production append 与 EOT

`generate_accumulating()` 先执行 token selection。普通内容 token 随后单独通过 cache-update forward 写入 KV，并同步追加 mask、完整 ledger 与 assistant 内容 ledger，再由 generator yield。首候选 token 回调发生在 selection 后、这次 forward 与 yield 前，所以记录的是 first-candidate-token selection/internal compute-readiness，不是可交付 token。

结构性 EOT 使用专门语义。生成器选择 EOT 后：

1. 不执行将该 EOT 作为 assistant 内容写入 KV 的 forward；
2. 不追加至 `assistant_token_ids`、TTS 文本片段或 timeline；
3. 将 `RolePhase` 置为 `ASSISTANT_EOT_PENDING`；
4. 将 `GenerationEndReason` 置为 `EOS` 并终止内容生成。

`reopen_user_role()` 是唯一 assistant close commit。它根据 tokenizer chat template 推导并校验 assistant-close 和下一 user-open 的 token IDs，恰好一次写入完整 ledger、mask 与 KV，但不写入 assistant 内容 ledger。这样，预测 EOT 不会与角色恢复重复注入。max-token、consumer-stop 和 crop 分别记录 `MAX_TOKENS`、`CONSUMER_STOP` 和 `CROPPED`，不再通过生成 token 数或 ledger 末 token 推断结束原因。

### 5.3.3 crop 与状态恢复

`crop_to_token(N)` 对 K/V、mask、完整 `token_ids` ledger、assistant 内容 ledger 与 span 同步裁剪，并根据保留序列恢复 `RolePhase` 和 end state。结构 token 内部不是合法 crop 点。播放期保留零个内容 token 时，目标是 assistant content start：assistant header 仍在，角色维持 `ASSISTANT_OPEN`，随后通过唯一 close commit 正常结束。整段候选作废时，目标是 assistant role start 之前的推测快照：assistant header 与全部内容一起删除，角色恢复 `USER_OPEN`。

crop 后 `GenerationEndReason.CROPPED` 在下一状态推进前保留。D-022 修复要求 `prefill_user_text()` 成功追加新 user 内容后立即将其清为 `NONE`；`open_assistant_role()`、`prefill_assistant_text()` 与 `reopen_user_role()` 也按各自的新阶段更新 end reason。编排器在追加 user segment 后 fail-closed 检查 `USER_OPEN + NONE`，避免陈旧 `CROPPED` 穿透到下一轮。

角色转换不硬编码固定 ChatML 字符串，而从 tokenizer chat template 的规范 tokenization 推导并验证。当前适配范围因此限定为能够可靠提取这些边界 token 的模板，不声称适用于任意因果语言模型。

## 5.4 编排与事件记录

`DialogueOrchestrator` 提供一次性用户文本路径和增量文本段路径。前者用于 System A，后者用于 B 系列推测工作点。输出侧逐 token 生成，断句器产出 TTS 文本片段，TTS 后端登记音频块与软件采样区间，再按实验条件注入打断并选择 software-cursor 或 generation 历史策略。

确认性 E1/E2 runtime 将事件明确拆开：

- `last_segment_arrival_ns`：最后预切分文本段进入路径；
- legacy `first_token_ready_ns`：首候选 token 选择回调，属于内部 compute-readiness；
- `endpoint_accept_ns`：候选处理之后的 oracle acceptance；
- `first_deliverable_token_ns`：同步 harness 的诊断 marker；
- `consumer_delivery_ns`：同步 harness 的 consumer 诊断 marker。

后两项受同步程序“先处理候选、后接受/消费”的顺序支配，只用于诊断，不作为生产 deliverability headline。尤其不能把候选选择到 oracle 接受的内部间隔解释为自然 endpoint lead 或用户继续说话时长。

实验中的 Mock TTS 由 CosyVoice2 六句真机画像参数化，使用每非空白字符采样数、首块延迟和实时率构造确定性时长近似。它可以统一控制软件游标比例并保留平均时长尺度，但不等价于真实异步合成、应用队列、audio API、OS/驱动/设备缓冲、线程唤醒或声学传播。mouth-to-ear 因此只是“计算时间与 TTS 画像组合建模”，不是完整音频闭环实测。

## 5.5 实验实现与 versioned analysis

### 5.5.1 E1/E2 交叉重复

确认性受控文本段 run `e1e2c_b8c758b_20260901T173306Z` 使用 100 个唯一 holdout utterances、5 个独立初始化 Python process sessions 与 10 个条件。每 session 重新加载模型，并在同一 100 个 utterances 上运行一次性 System A、八个数值阈值和一个 never-speculate B 对照；因此每条件有 500 个 session×utterance observations，总计 5000 条 records。内容采样单位仍是 100 个 utterances，5 个 session 仅是技术重复。

正式统计使用不覆盖 `analysis_v1` 的 `analysis_v2.json`。crossed/product bootstrap 每次独立抽样全局 session 与全局 utterance，再取笛卡尔积并在 cell 内保持条件配对；10,000 repeats，seed 20260901，报告 percentile 95% interval。该实现匹配“同一批话语跨五个进程重复”的交叉设计，而不是把 dialogue 错当为嵌套于 session。

TEN 置信度通过 222 条目的离线 replay cache 逐段回放，不进入延迟窗口。模型固定 Qwen2-7B-Instruct、greedy、BF16 与 SDPA，holdout 从本地 MultiWOZ 2.1 确定性派生，并与旧 E1/E2 及固定轨迹 E3 的样本 ID 隔离。

C-E1 是非 token-equivalent 的 implementation-path comparison。System A 对 full string 一次 tokenization/full prefill；System B 对 segments 分别 tokenization 并走 incremental forward。正式记录中，A 与 B@0.92 的完整 `output_token_ids` exact 为 280/500，首 token exact 为 465/500，长度/EOS/max-token agreement 为 495/500；44/100 个唯一 utterances 至少一次完整输出分岔，且五个 session 中均为相同 44/100。差异可能来自 segment-wise tokenization、forward topology/shape、role boundary、kernel 与 Python scheduling，不能归因为纯 incremental-prefill effect，也不能按 280 个 matched outputs 过滤主延迟。

C-E2 比较 B@0.92 与 B-never；二者完整 tokens、首 token、长度、EOS/max-token 与文本均为 500/500 一致，因此是 token-consistent B-path comparison。该一致性不改变 C1 只属于同步 candidate-readiness/oracle characterization 的范围。

### 5.5.2 E3、A1 与 P1

E3 accepted run 保存 100 dialogues、400 `(dialogue,injection_label)` pairs、800 condition records 与 1600 judge records。versioned weighting/dedup `analysis_v2` 以 label-weighted estimand 和相同 label-weighted dialogue-cluster bootstrap 为主，另报 dialogue-weighted estimand 与 target-specific unique-semantic-boundary sensitivity；10,000 repeats，seed 20260831。它只测 fixed-detector-conditioned information-reproduction rate，不把规则或 Mistral judge 称作人类真值。

A1 对 256–8192 token contexts 使用 5 warmups、50 repeats，固定 operation order，并固定移除 32-token suffix。joint crop+role 与 re-prefill 的计时均有 CUDA/GPU 同步，但结果只适用于这一固定模型侧协议，不能外推到自然打断位置、其他 crop length 或完整 barge-in。

P1 使用 3 个 context lengths、3 个软件游标比例与每 cell 20 repeats，共 180 条 prepared-state headless records。stop→crop 与 stop→role 是嵌套累计区间。P95 是每 cell 20 个数值形成的 empirical/descriptive order statistic，主要由一至两个上尾观测决定，不作为 production SLO。P1 只测 software cursor 与 headless control path，不测设备呈现、声学停播、在线 TTS 取消或真实服务并发。

## 5.6 C2 v3 exact-only 实现与验收

C2 v3 工件位于 `experiments/sci34_supplement/c2_crop_integrity/`。正式 accepted run 为 `c2crop_82103004_20260903T080512Z`，code commit `8210300`，result commit `7d50624`，manifest hash 前缀 `d8c3db4d`，seal hash 前缀 `e0997d41`。运行固定为 Qwen2-7B artifact、BF16、SDPA、Transformers backend 与严格离线环境。

v3 精确复用冻结的 24-case 网格。24/24 ordered cases 产生 27/27 ordered crop events，其中 3 个 no-op；冻结 fixture 的 308 个 assistant 内容 token 全部经 production `generate_accumulating()` 逐 token append，并检查每 token 一次 `_prefill_ids_p2` forward。三方比较实现如下：

1. crop 前对目标 keep length 的 K/V prefix 生成每层 shape、dtype、device 与 hash manifest；
2. production arm 唯一调用 `crop_to_token()`；
3. oracle arm 从 crop 前 clone 直接独立 slicing，不调用生产 crop；
4. validator 不信任 stored keep，而从 case、fragment token partition、role/content boundaries 和 second-crop fraction 独立推导；
5. 每层要求 pre-prefix、production post-crop 与 slicing oracle 的 K/V `torch.equal` 和 hash exact，并要求 `shape[-2]==keep`；
6. mask、完整 token ledger、seq length 与 KV length exact；27/27 wrong-length disposable negative controls 必须被拒绝。

crop 后两臂以相同 token-ID chunks 执行 60 个 recovery steps。每一步逐层比较 K/V，并比较 logits、mask、完整 ledger、retained-prefix hash，以及从操作序列独立推导的 `RolePhase`、`GenerationEndReason` 与 assistant content state。accepted run 中这些 gate 全部 bitwise/exact 通过；28 层 K/V 均在比较范围内。

v1/v2 是 clean re-prefill 协议，结论保持 rejected。v2 的 24/24 termination probe 和 45/45 token/state/EOT/scenario 检查通过，但单控制 2× numerical gate 只有 42/45。其 control 按语义 seam 分块并强制末 token单独 forward，production 则对初始 512/2048/8192 context 一次 forward，并按实际 API chunk 追加 role/content；两者 forward topology 不匹配。因此三项数值失败既不能识别 crop bug，也不能建立 clean-reprefill equivalence。v3 删除不可识别的 numerical re-prefill 比较，改用 direct pre-prefix/production crop/independent slice 的 exact 问题；它不改变 v1/v2 verdict，也不支持 clean-reprefill numerical equivalence、32-token continuation equivalence、跨模型/backend/硬件或在线生产正确性。

## 5.7 Artifact 与可复算范围

仓库根目录 `REPRODUCIBILITY.md` 是 artifact matrix，映射 campaign、accepted run、代码/结果身份、入口与分析文件。E3 的 exact processed input rescue 位于 `experiments/sci34_supplement/results/e3_exact_rescue/README.md`；它用于闭合 accepted E3 输入工件，不改变原始记录。C2 v3 accepted run、validation、analysis、manifest 与 seal 构成 D-023 接受的 direct crop-integrity 证据。确认性 E1/E2 的正式重分析位于 `experiments/sci34_supplement/results/e1e2_confirmatory/e1e2c_b8c758b_20260901T173306Z/analysis_v2.json`；E3 weighting/dedup 重分析位于 `experiments/sci34_supplement/results/e3/sci34_f11ccba_20260901_e3/analysis_weighting_dedup_v2.json`。二者均为 versioned analysis，不得以旧 `analysis_v1` 替代或覆盖旧文件。

不同 campaign 不组成一个可池化的计时总体。确认性 E1/E2、固定轨迹 E3/联合 A1、prepared-state P1 与 C2 v3 各自保留模型、代码、结果和协议身份；Windows checkout 的 CRLF 转换可能使 byte-level C2 seal 在工作树误报，正式复核应使用 LF-preserving checkout、Git blob 或原 tarball。

这些工件支持论文所列的 analysis-only 复算、部分 smoke、accepted-run validation 与 hash/seal 检查，但不能据此声称整个系统已经完整可复现。旧探索性 E1/E2/A2 的部分环境与 exact input 仍不完整，真实 ASR/TTS/设备/声学闭环未实例化。仓库状态也不支持编造或宣称正式 LICENSE、公开/匿名 artifact URL、DOI、作者、基金、利益冲突、伦理或 consent 结论；相关 submission declarations 只在 `paper2/declarations.md` 中作为待作者确认草稿处理。

## 5.8 本章小结

本章给出 software-consumed cursor、片段、token 与 KV 的具体实现，并把 device-presented 与 acoustically heard 层排除在当前观测之外。时间轴 API 强制 token/sample 连续、fragment lifecycle、chunk 唯一归属与乱序拒绝。缓存容器显式维护完整 `token_ids` ledger、assistant 内容 ledger、`RolePhase` 与 `GenerationEndReason`；EOT 不进入内容 KV/ledger，`ASSISTANT_EOT_PENDING` 由 `reopen_user_role()` 唯一提交 close，推测全作废与播放期零内容保留采用不同 crop 语义，user 内容推进会清除陈旧 `CROPPED`。实验实现按 100 utterances×5 sessions 的 crossed `analysis_v2`、非 token-equivalent C-E1 路径、A1/P1 固定协议和 C2 v3 exact-only gate 报告。artifact 章节给出 `REPRODUCIBILITY.md`、E3 rescue 与 C2 accepted run 的可复算入口，但不作完整复现、许可证或公开 URL 主张。
---

# 第六章 实验与结果分析

## 6.1 研究问题与实验设置

本章围绕以下研究问题展开。

- **RQ1：** 在固定被打断回复轨迹及固定自动检测器下，software-cursor playback 与 generation 历史边界的后续信息复现率有何差异？
- **RQ2：** 推测阈值如何影响作废计算与首候选 token 选择/内部计算就绪？实际墙钟口径与同步 oracle 接受口径分别给出什么结论？
- **RQ3：** 在受控同步文本段条件下，增量推测实现路径相对一次性预填充实现路径的候选计算就绪延迟有何差异？
- **RQ4：** 冻结模型与后端下，software-consumed-sample cursor 到 KV 裁剪及角色恢复的核心状态操作是否满足直接完整性合同，其模型侧成本和 prepared-state 软件控制路径时延如何？
- **RQ5：** 本次探索性运行中，朴素保留、打断标记和轻量重写三种实现的连贯性分数与重写耗时如何？

### 6.1.1 硬件、模型与数据

实验使用两张 NVIDIA RTX 3090（24 GB）。主模型为 Qwen2-7B-Instruct[11]；话轮检测器为 TEN Turn Detection[12]（7.6B）；历史重写模型为 Qwen3-0.6B[16]；自动裁判为 Mistral-7B-Instruct-v0.3[13]；TTS 时长画像由 CosyVoice2-0.5B[14]采集。裁判与主模型来自不同模型家族，但这不能消除单裁判和单提示词偏差。

数据由 MultiWOZ 2.1[15]派生。RQ2 与 RQ3 使用确认性 C-E1/C-E2 run `e1e2c_b8c758b_20260901T173306Z`：100 条唯一话语在 5 个独立初始化的进程 session 中重复，每个 session 执行 10 个条件，形成每条件 500 个 session×utterance 观测和 5000 条总记录。100 条话语是内容采样单位，5 个 session 是技术重复。RQ1 使用独立的固定轨迹 E3：100 条对话、400 个 `(dialogue, injection_label)` 配对、800 条条件记录及 1600 条自动裁判记录。RQ4 使用 C2 v3、联合 A1 和 prepared-state P1 三组独立工件。P1 覆盖 3 个上下文长度、3 个软件游标注入位置和每单元 20 次正式重复，共 180 条记录。

TEN 的标定成对正确率 1.00 来自 8 条手工完整句和 8 条手工不完整句形成的 64 个跨类对，仅用于确定阈值扫描范围，不代表独立口语测试集上的端点检测性能。

### 6.1.2 时序、播放边界与统计口径

C-E1/C-E2 使用确定性预切分文本段，不包含真实音频、ASR 墙钟、在线 TEN 前向、TTS、播放器或声卡。raw 字段 `first_token_ready` 的回调位于 token 选择之后、cache-update forward 与 generator yield 之前，因此本文统一称其为**首候选 token 选择/内部计算就绪**，不称可交付 token、consumer observation、TTS admission 或声学输出。`endpoint_accept` 是同步 harness 在候选生成后的 oracle 接受事件，并非自然端点检测输出。`TTFT_eff` 仅为同步 oracle 条件下的时延乐观下界，即推测收益上界。

C-E1/C-E2 的点估计来自完整、未加权的 5 session×100 utterance 网格。不确定性采用 crossed/product bootstrap：独立有放回抽取全局 session 与全局 utterance，再取二者笛卡尔积并重算估计量；重复 10,000 次，seed 为 20260901，报告 percentile 95% CI。该设计不把 500 个观测误作 500 个独立内容样本。

播放侧以 $p$ 表示 **software-consumed-sample cursor**；$\widehat{H}(p)$ 表示 TTS-fragment-level software retention boundary。二者均不等同于 device-presented samples 或 acoustically heard content。工件中的 `heard_text`、`n_heard` 和 `strict_unheard` 仅是兼容字段：分别对应片段保留或字符比例—空白吸附代理，不是人类听觉真值。

**表 6-1　主要证据的测量层级**

| 证据 | 测量层级 | 主要限制 |
|---|---|---|
| C2 v3 | 冻结 Qwen2-7B/BF16/SDPA/Transformers 下的直接 KV 裁剪完整性与匹配恢复 | 不检验 clean re-prefill 数值等价，不覆盖其他模型、后端、在线音频或生产系统 |
| C-E1/C-E2 | 同步文本段 harness 的候选选择/计算就绪及 oracle 下界 | 100 唯一话语×5 技术 session；无真实 ASR、在线 TTS、设备播放或生产可交付性 |
| 固定轨迹 E3 | 固定规则与单一 LLM 裁判条件下的信息复现率 | CI 仅含对话抽样不确定性；无检测器误差、提示词变化或人类感知误差 |
| 联合 A1 | 同步 GPU 的 crop+角色恢复微基准 | 固定执行顺序、固定移除 32-token suffix；不是完整打断路径 |
| P1 v2 | prepared-state、headless 软件游标控制路径 | 无 device/acoustic stop、在线 TTS、真实模块并发或 HCI 测量 |
| A2 | 受混杂探索性评分与重写耗时 | 三种实现的生成轨迹不一致，不支持处理效应解释 |

## 6.2 RQ1：固定检测器条件下的信息复现（E3）

固定轨迹 E3 使同一对话内的 playback 与 generation 条件共享被打断 assistant token 轨迹、断句时间轴和注入标签，后续 probe 使用 greedy 解码。主 estimand 为 label-weighted：每个符合资格的注入标签权重相同；差值统一定义为 generation 减 playback。点估计和区间均使用该 estimand，区间由 10,000 次 paired dialogue-cluster bootstrap 得到。

**表 6-2　固定轨迹 E3 的 label-weighted 信息复现率**

| 目标 / 检测器 | Playback | Generation | 差值 | 95% dialogue-cluster CI |
|---|---:|---:|---:|---:|
| 片段目标 / 词面规则 | 67.00% (199/297) | 63.64% (189/297) | −3.37 pp | [−10.49, 3.40] pp |
| 片段目标 / 自动裁判 | 42.76% (127/297) | 40.74% (121/297) | −2.02 pp | [−10.70, 6.13] pp |
| 字符比例—空白边界代理 / 词面规则 | 75.26% (286/380) | 73.68% (280/380) | −1.58 pp | [−6.08, 2.67] pp |
| 字符比例—空白边界代理 / 自动裁判 | 43.95% (167/380) | 41.32% (157/380) | −2.63 pp | [−8.57, 2.90] pp |

片段目标包含 297 个有效标签，来自 96 条对话；按目标、两条件历史、软件保留边界和轨迹精确去重后为 169 个语义组，即 128 个标签属于重复组。代理目标按自身非空性确定资格，共 380 个标签、100 条对话和 379 个语义组，仅移除 1 个重复标签。四个主区间均跨零，因此这些结果不能确定方向性优势，也不能推出差异不存在。

为检查标签权重和重复边界的影响，表 6-3 同时给出每条有效对话等权的 effect，以及每个唯一语义组等权的条件率与 effect。所有差值仍为 generation 减 playback。

**表 6-3　E3 weighting 与精确去重敏感性**

| 目标 / 检测器 | Dialogue-weighted effect [95% CI] | Unique-group Playback / Generation | Unique-group effect [95% CI] |
|---|---:|---:|---:|
| 片段目标 / 词面规则 | −3.21 pp [−9.55, 2.78] | 71.60% / 68.64% | −2.96 pp [−9.04, 2.63] |
| 片段目标 / 自动裁判 | −1.30 pp [−8.94, 6.08] | 43.20% / 43.20% | 0.00 pp [−7.98, 7.47] |
| 字符比例—空白边界代理 / 词面规则 | −1.50 pp [−5.75, 2.50] | 75.20% / 73.61% | −1.58 pp [−6.10, 2.69] |
| 字符比例—空白边界代理 / 自动裁判 | −2.58 pp [−8.25, 2.67] | 43.80% / 41.16% | −2.64 pp [−8.57, 2.90] |

所有 E3 区间仅表示：在冻结词面规则、`specific-reference-v3` Mistral 自动裁判、目标构造、固定轨迹、提示词和 40-token cap 条件下，由对话抽样产生的不确定性。区间不包含检测器误差、提示词或模型变化，以及人类感知误差。

### 6.2.1 构造检查与自动代理一致性

400/400 个 playback 条件记录在片段边界之后的局部完整未保留文本为空，对应局部规则阳性数为 0。该结果是 software-cursor retention 规则与指标共同定义的 implementation invariant check，不是语义效果或声学边界准确率。

词面规则与自动裁判在 label level 的合并一致数为：片段目标 370/594，代理目标 442/760；在 unique-group level 分别为 207/338 和 440/758。这些数值只描述两个自动代理之间的一致性，不能视为人工验证、检测器校准或 HCI 证据。

![图 6-1](figures/fig6_1.png)

**图 6-1　固定轨迹 E3 的效应区间示意。** 正式 label-weighted 点估计、匹配的 dialogue-cluster percentile 95% CI 及 weighting/dedup 敏感性以表 6-2 和表 6-3 为准。

## 6.3 RQ2：推测浪费与候选计算就绪（C-E2）

确认性扫描包含八个数值阈值和一个 never-speculate 对照，共九个工作点。每点由 100 条唯一话语×5 个独立进程 session 形成 500 个观测。0.92 是在新 holdout 结果揭示前冻结的 confirmatory candidate，不是部署最优阈值。浪费率定义为 pooled 的 $\sum W_i/(\sum W_i+\sum G_i)$。

**表 6-4　确认性九点扫描（每点 100 条唯一话语×5 个 session）**

| 阈值 $\theta$ | 0.0052 | 0.1979 | 0.3906 | 0.5833 | 0.7760 | 0.8500 | 0.9200 | 0.9688 | never |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pooled 浪费率 $\rho$ | 31.0% | 19.3% | 15.8% | 13.2% | 11.3% | 10.7% | **2.85%** | 0% | 0% |
| 候选存活率 | 100% | 99% | 98% | 97% | 96% | 84% | **67%** | 28% | 0% |
| oracle TTFT_eff 均值 (ms) | 0.0 | 0.3 | 0.7 | 0.9 | 1.3 | 5.0 | **10.3** | 22.4 | 31.1 |
| arrival→candidate selection/readiness 均值 (ms) | 62.4 | 62.4 | 62.3 | 62.1 | 62.2 | 62.0 | 62.4 | 62.3 | 62.4 |

C-E2 的 candidate-readiness 差值 `never − B@0.92` 为 −0.03 ms，crossed 95% CI [−0.64, 0.61] ms。oracle TTFT_eff 下界的差值为 +20.80 ms，[17.85, 23.65] ms。B@0.92 的 pooled waste 为 2.85%，[1.12%, 4.73%]；survival 为 67%，[58%, 76%]。因此该候选工作点刻画了同步 oracle 接受语义下的计算浪费与候选可用性，而不是生产可交付延迟改善。

在同一同步 harness 中，B@0.92 的 arrival→first-deliverable marker 均值为 257.58 ms，arrival→consumer marker 均值为 265.57 ms。二者受“先同步生成候选、再 oracle 接受”的执行顺序支配，只作为程序诊断，不作为生产系统 headline。约 291 ms 也仅是 candidate-first-selection 到 post-candidate oracle acceptance 的内部间隔中位数，不能解释为自然端点提前量或用户继续发言时长。

![图 6-2](figures/fig6_2.png)

**图 6-2　确认性九点扫描。** 左图对应同步 oracle 时延乐观下界，右图对应内部 candidate selection/readiness；右图竖线为各条件的 Q1–Q3 描述性离散范围，不是 crossed 95% CI。图中的“平坦”仅指九个 B-path 工作点的均值为 62.0–62.4 ms，不包含 System A。正式 crossed 差值区间以本节文字为准，两种口径不可互换。

## 6.4 RQ3：实现路径比较（C-E1）

C-E1 在每个 session×utterance 单元内配对比较 System A 的 full-string tokenization/full-prefill 路径与 B@0.92 的 segment-wise tokenization/incremental 路径。该比较不满足相同 tokenized context 条件，因此估计的是整体 implementation-path difference，而非单一增量预填充操作的效应。

**表 6-5　C-E1 实现路径配对结果**

| 指标 | System A | B@0.92 | A−B | Crossed 95% CI |
|---|---:|---:|---:|---:|
| arrival→candidate selection/readiness 均值 | 27.70 ms | 62.38 ms | −34.69 ms | [−35.44, −33.95] ms |
| oracle TTFT_eff 下界均值 | 27.70 ms | 10.26 ms | +17.44 ms | [14.41, 20.32] ms |

输出身份诊断进一步界定了这一比较：A 与 B@0.92 的完整 `output_token_ids` 仅 280/500 相同，首 token 为 465/500 相同，长度/EOS/max-token 状态为 495/500 相同；44/100 条唯一话语至少在一个 session 出现完整输出不一致，且每个 session 均为 44/100。相比之下，B@0.92 与 B-never 的完整 token、首 token、长度/EOS/max-token 及文本均为 500/500 相同，支持 C-E2 作为同一 B-path 内部比较。

C-E1 的差异混合了 full-string 与 segment-wise tokenization、forward topology/shape、角色边界、kernel 和 Python scheduling。主延迟分析不按 280 条完整输出一致记录筛选，因为该筛选位于实现路径之后。结果不得排他归因于某一次额外 forward，也不能从 oracle 下界反推真实音频或生产系统的墙钟改善。

![图 6-3](figures/fig6_3.png)

**图 6-3　C-E1 的双口径实现路径比较。** 柱高为条件均值，竖线表示各条件的 Q1–Q3 描述性范围，白色横刻度为中位数；图内方框才是配对均值差的 crossed 95% CI：candidate-readiness [−35.44, −33.95] ms、oracle 下界 [14.41, 20.32] ms。B 条件均为 B@0.92；该图比较非 token-equivalent implementation paths。

### 6.4.1 旧探索性 E1/E2 的口径审计

旧 E1/E2 的 0.581 ms 和 48.3→12.1 ms 将同步推测完成后的时间原点误作用户端点，属于 oracle 口径 artifact。其结果只保留为探索性审计和 0.92 候选来源，不作为正式墙钟证据。旧 mouth-to-ear 数值亦由模型计时与 6 句 TTS 画像组合建模，不是在线音频闭环测量。

## 6.5 RQ4：C2 核心正确性与支持性时延证据

### 6.5.1 C2 v3 direct crop-integrity addendum

正式接受的 C2 v3 run 为 `c2crop_82103004_20260903T080512Z`。协议固定 Qwen2-7B snapshot、BF16、SDPA、Transformers backend、24 个 case 和 27 个 crop event。308 个 fixture assistant token 均逐 token 走 production append；另含 3 个 no-op、60 个 recovery step 和 27 个 wrong-length negative control。

**表 6-6　C2 v3 核心正确性结果**

| 检查 | 结果 |
|---|---:|
| Case / crop event | 24/24；27/27 |
| 逐 token production append | 308 tokens |
| K/V 层数 | 28 |
| Recovery step | 60/60 |
| Wrong-length negative control | 27/27 检出 |
| No-op crop | 3/3 |

在该冻结网格中，每个 event 的 crop 前 retained K/V prefix、production `crop_to_token` 后 K/V 与独立 slicing oracle 在 28 层上逐张量 shape、dtype、device、hash 和 runtime `torch.equal` 均 bitwise exact；keep、attention mask、token ledger、sequence length 与 KV length exact 一致。使用相同 token-ID chunks 继续恢复后，K/V、logits、attention mask、token ledger、retained prefix 以及 role/end/content state 也 exact 一致。该结果只支持受测 snapshot/backend 下的 direct crop integrity 与 matched-recovery determinism。

v1 与 v2 clean-reprefill 协议均按冻结门槛 rejected。v2 的 24/24 termination probe 与 45/45 token/state/EOT/scenario 检查通过，但单控制 2× numerical gate 仅 42/45；由于 control 与 production forward topology 不匹配，三项失败既不能识别 crop 缺陷，也不能建立 clean-reprefill numerical equivalence。v3 是直接 crop-integrity addendum，不改变 v1/v2 verdict，也不支持 32-token continuation、跨模型/后端/硬件、在线音频或生产端到端正确性主张。

### 6.5.2 固定协议的联合微基准（A1）

A1 覆盖 256、512、1024、2048、4096 和 8192 token；每个长度执行 5 次预热和 50 次正式重复。各重复按固定而非随机化的 operation order 运行，并固定移除 32-token suffix。计时区间前后执行设备同步；联合路径在同一区间内执行 KV crop 与角色恢复。

**表 6-7　联合 crop+角色恢复与重新预填充的同步 GPU 微基准**

| 上下文长度 | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---:|---:|---:|---:|---:|---:|
| 联合路径中位数 (ms) | 31.616 | 31.852 | 31.054 | 31.519 | 36.903 | 48.315 |
| 联合路径 IQR (ms) | 2.356 | 2.162 | 3.099 | 1.197 | 0.635 | 0.928 |
| 重新预填充 / 联合路径中位数 | 2.254× | 4.124× | 7.707× | 15.020× | 25.453× | 40.620× |

![图 6-4](figures/fig6_4.png)

**图 6-4　固定 32-token suffix 协议下的联合微基准。** 误差线为 50 次正式重复的 IQR。

结果仅适用于固定执行顺序和固定移除长度，不代表自然打断位置、其他 crop length 或完整 barge-in。时间轴查询、软件停播、服务通信、线程调度和并发负载均不在 A1 窗口内。

### 6.5.3 Prepared-state 软件控制路径（P1）

P1 v2 在 512、2048、8192 token 与 0.25、0.50、0.75 三个 software-cursor 位置上各运行 20 次，共 9 个单元、180 条记录。120 条为片段内位置，60 条为片段边界。播放前屏障完成 setup 和 GPU 同步，并将其排除在 stop 路径之外。180/180 条 request 与 acknowledgement 精确命中目标 software-consumed-sample cursor，`leaked_samples=0`。

**表 6-8　Prepared-state P1 软件控制路径时延（每单元 n=20）**

| 计时区间 | 单元中位数范围 (ms) | 最大单元 empirical P95 (ms) |
|---|---:|---:|
| 软件停播确认 | 0.055–0.062 | 约 0.077 |
| 播放器确认后的 CUDA/GPU 同步 | 0.167–0.176 | 约 0.352 |
| 时间轴反查 | 0.47–0.50 | 约 0.94 |
| stop→crop 完成 | 2.44–2.53 | 约 3.492 |
| stop→角色恢复完成 | 78.6–80.8 | 约 86.1 |

每单元仅有 20 个值，empirical P95 主要由 1–2 个上尾观测决定，只作描述性次序统计，不代表生产 SLO。stop→crop 与 stop→角色恢复是从同一 stop 请求起点计算的嵌套累计区间，不能与组件中位数相加。P1 不测 device-presented 或 acoustically heard boundary，也不包含声卡/扬声器停止、在线 TTS 取消、真实 ASR/LLM/TTS/播放器并发、用户体验或生产端到端 barge-in。

## 6.6 RQ5：三种历史处理实现的描述性结果（A2）

本次 A2 每种实现包含 100 条记录。单一 Mistral 自动裁判的连贯性均值为：朴素保留 3.76、轻量重写 3.62、打断标记 3.29。重写调用耗时均值为 639 ms，中位数为 670 ms，线性插值 P90 约 935 ms，最大值为 1165 ms。

三种实现分别重新生成首轮与下一轮回复，只有 33/100 个对话的兼容字段 `heard_text` 在三条件完全相同，朴素与重写成对相同的仅 49/100。评分差异同时混入首轮内容、断句边界与下一轮采样差异，因而只能描述本次运行，不能解释为策略处理效应。约 0.64 s 的重写调用具备与下一轮用户发言并行的工程可能，但本实验未记录真实发言时长，也未测异步重叠，故不声称其耗时已被隐藏。

## 6.7 本章结论

1. **RQ1：** label-weighted 主分析及 dialogue-weighted、unique-group 敏感性分析的区间均跨零；结果仅是 fixed-detector-conditioned information-reproduction rate，不能确定方向性优势或差异不存在。
2. **RQ2：** 0.92 是预冻结 candidate。B@0.92 的 pooled waste 为 2.85% [1.12%, 4.73%]，survival 为 67% [58%, 76%]；oracle 下界相对 never 的差值为 +20.80 ms [17.85, 23.65]，candidate-readiness 差值为 −0.03 ms [−0.64, 0.61]。同步 deliverable/consumer marker 仅作诊断。
3. **RQ3：** C-E1 是 implementation-path comparison。A−B 的 candidate-readiness 差值为 −34.69 ms [−35.44, −33.95]，oracle 下界差值为 +17.44 ms [14.41, 20.32]；结果不能归因于单一 forward 或外推至生产交付。
4. **RQ4：** C2 v3 在冻结网格内建立了 28 层 K/V 的 direct crop bitwise integrity 与 matched recovery exactness。A1 和 P1 分别提供固定 32-token suffix 模型侧微基准及 headless software-cursor 控制路径的描述性时延，不提供设备、声学或 HCI 结论。
5. **RQ5：** 本次受混杂运行中三种实现的均值为 3.76、3.62 和 3.29，重写均值耗时 639 ms；这些是描述性结果，不是因果比较。
---

# 第七章 讨论

## 7.1 贡献层级与已有工作的关系

本文的贡献按证据成熟度分为三个层级。**C2 是核心贡献**：将 software-consumed-sample cursor 映射到 TTS 文本片段、assistant token span、KV crop、attention mask、token ledger、position 与 role/EOT 状态恢复，并为直接裁剪完整性提供可复算的精确证据。**C1 是支持性 characterization**：刻画 pre-end-of-turn candidate-response generation with invalidation 在同步文本 harness 中的候选选择/计算就绪、oracle acceptance 和 wasted-token 工作点；它不是 draft-target speculative decoding，也不是生产可交付性改善证据。**C3 是探索性扩展**：实现朴素、标记和重写三种历史处理方式，但现有 A2 受生成轨迹混杂，只提供描述性负结果。

OpenAI Realtime API、Azure Voice Live 和 LiveKit Agents 已公开 playback-conditioned transcript/session-history truncation 的高层实践，KV crop 与 prefix reuse 也有既有基础。因此，“历史应反映 delivered/spoken output”与缓存裁剪原语均不作为本文原创原则。本文可辩护的增量限于：在已报告的公开检索范围内，未识别同时公开 software cursor→fragment→token→KV crop→role recovery 及 exact/latency evidence 的级联实现。该表述是范围受限的未识别结果，不是全球首创判断。

C2 v3 进一步明确了这一增量的证据形态：在冻结 Qwen2-7B、BF16/SDPA/Transformers backend 和 24-case/27-event 网格内，production crop 保留的 28 层 K/V 与 crop 前前缀及独立切片 oracle bitwise exact；匹配 token-ID chunk 恢复后的 K/V、logits、mask、ledger 与 role/end state 亦 exact。它证明的是 direct crop integrity 和 matched-recovery determinism，而非 clean re-prefill 数值等价或生产端到端正确性。

## 7.2 结果解释

### 7.2.1 C2：状态合同而非宽泛系统质量结论

C2 的主要价值是把跨模块边界转化为可检查状态合同。software cursor 首先确定 TTS-fragment-level retention boundary，再由片段记录定位 assistant token span；裁剪后，KV length、attention mask、完整 token ledger、内容 span、position 推进、role phase 与 generation end reason 必须一致。结构 EOT 不属于 assistant 内容账本，并由角色恢复路径唯一提交。这样可以把“保留了哪段历史”从文本层决策落实为推理 runtime 的显式状态操作。

v1/v2 clean-reprefill 协议的 rejected 结果也限定了可回答的问题。v2 虽通过 24/24 termination probe 和 45/45 token/state/EOT/scenario 检查，但单控制 2× numerical gate 为 42/45；由于对照与 production forward topology 不匹配，这些差异既不能识别 crop 缺陷，也不能建立 clean-reprefill numerical equivalence。v3 因而改为同一 pre-crop K/V 上的独立切片 oracle，并对匹配恢复逐步执行 exact 检查。该证据更直接地回答裁剪实现是否保留指定缓存前缀，但不回答不同 forward topology 是否数值一致。

### 7.2.2 C1：候选就绪与接受策略的分离

C-E2 显示，B@0.92 相对 never 的 candidate-readiness 差值接近零，而同步 oracle TTFT_eff 下界存在正差值。二者并不矛盾：前者测量最后文本段到达后的内部 token 选择/计算就绪，后者把接受事件前已存在且存活的候选视为零等待。0.92 工作点的 2.85% pooled waste 和 67% survival 应解释为 candidate-selection/oracle-readiness characterization，而不是生产 deliverability 或用户感知时延改善。

同步 harness 中 B@0.92 的 first-deliverable 与 consumer marker 均值分别为 257.58 和 265.57 ms。这些 marker 受“完成固定候选生成后再接受”的程序顺序支配，因此只用于揭示 harness 行为。candidate-first-selection 到 post-candidate oracle acceptance 的约 291 ms 中位内部间隔也不能代替自然端点提前量。若要回答真实可交付收益，需要独立异步 endpoint gate、consumer、TTS admission 与 device/acoustic 时间戳；该问题超出现有实验范围，但不是本文窄化 C2 主张的当前阻塞项。

C-E1 不能解释为纯 incremental-prefill effect。A 与 B@0.92 的完整输出仅 280/500 相同，首 token 为 465/500 相同，44/100 条唯一话语出现完整输出分岔；B@0.92 与 B-never 则为 500/500 一致。由此，C-E1 混合 tokenization、forward topology/shape、role boundary、kernel 与 Python scheduling，应视为 implementation-path comparison。−34.69 ms 的 candidate-readiness 差值不能排他归因于单一额外 forward。

### 7.2.3 E3 与 A2：代理结果的有限解释

E3 的四项 label-weighted 效应均为小负值且区间跨零；dialogue-weighted 和 unique-group 敏感性没有提供稳定的方向性判断，其中片段目标/自动裁判去重效应为 0.00 pp [−7.98, 7.47]。这些区间仅反映冻结检测器、目标、轨迹、提示词和 40-token cap 条件下的对话抽样不确定性。词面规则与自动裁判的一致数也只是 automated-proxy agreement，不是人工校准。

A2 中朴素、重写和标记实现的平均分为 3.76、3.62 和 3.29，重写平均耗时 639 ms。然而三种实现分别生成不同的首轮和下一轮回复，只有 33/100 个对话的兼容历史字段在三条件相同。因此，分数和耗时只描述当前探索性运行，不能回答处理策略是否导致连贯性变化，也不能证明重写耗时在真实用户发言期间已被隐藏。

## 7.3 效度威胁

### 7.3.1 构念效度

本文的软件边界有三个必须区分的层级：software-consumed samples、device-presented samples 和 acoustically heard content。现有 $p$ 仅属于第一层；由其得到的片段保留边界不是设备呈现或用户声学接收真值。应用队列、audio API、OS mixer、驱动、DMA、蓝牙/网络缓冲、扬声器与传播均可能使三者分离。P1 的零软件采样泄漏不能转写为零声学泄漏。

E3 测量 fixed-detector-conditioned information-reproduction rate。片段目标依赖 TTS 片段边界；字符比例—空白边界目标先按片段内软件播放比例切字符，再向前吸附至空白，只是文本代理。词面规则可能把任务域自然重合计为复现，单一 Mistral 裁判也可能误判显式或隐式引用。没有盲法人工双标或直接 HCI 数据，因此结果不能用于推断真实语义保真、自然度、信任或用户体验。

`heard_text`、`n_heard` 和 `strict_unheard` 是工件兼容别名，不改变上述构念层级。正文中的“playback”仅表示 software-cursor-conditioned fragment retention；涉及“用户听到”时只作为设计目标或尚待验证的声学层概念。

### 7.3.2 内部效度

E1/E2 的 100 条唯一话语跨 5 个进程 session 重复，属于交叉而非嵌套设计。正式 v2 使用 crossed/product bootstrap，避免将 500 个观测当作独立内容样本；但 session 仍只是技术重复，不能扩张任务内容多样性。TEN 置信度离线重放，在线检测成本与真实模块竞争均未进入时间窗。

C-E1 的路径不具 token equivalence，故不进行单机制因果归因，也不按输出一致子集筛选主分析。E3 虽共享固定首轮轨迹，但同一对话含四个注入标签，且部分标签映射到相同语义边界；label-weighted 主表、dialogue-weighted estimand 和 exact unique-group dedup 分别回答不同加权问题。所有分析保留对话内配对与聚类结构。

A1 的不同 operation 按固定顺序执行，并固定移除 32-token suffix。其 2.254–40.620 比值只适用于该协议，随机顺序、不同裁剪量和自然打断分布可能产生不同结果。P1 使用播放前 prepared-state 屏障，修复了 v1 的异步准备态污染，但仍是 headless、无争用控制路径。每个 P1 单元 n=20，其 empirical P95 主要由 1–2 个上尾值决定，不能作为稳定尾延迟或生产 SLO。

### 7.3.3 外部效度

实验证据主要限于英文任务型 MultiWOZ 对话、Qwen2-7B-Instruct、特定 ChatML 类 role transition、Transformers `DynamicCache`、BF16/SDPA 和 RTX 3090。E3 的首轮及 probe 均有 40-token cap；开放域长回答、中文或其他非空白分词语言、多人重叠语音、犹豫修正、不同 TTS 与服务端推理引擎可能形成不同片段和缓存状态。

C2 v3 的 exact 结论不跨模型、dtype、attention backend、硬件或 chat template 自动成立。迁移时必须重新验证 token serialization、结构 EOT、裁剪允许边界和恢复状态。A1、P1、E3 与 E1/E2 来自不同 campaign；即使 GPU 型号相同，CPU、OS 和调度环境也不完全一致，绝对毫秒值不得池化或跨 campaign 相减。

本文没有测量真实流式 ASR、在线异步 TTS 取消、声卡停播、loopback 波形、设备缓冲、网络拥塞或生产并发。因此结果适用于可审计的软件 runtime/prototype，不支持真实 acoustic stop、mouth-to-ear、生产 barge-in 或 HCI 泛化。

### 7.3.4 结论效度

E3 label-weighted 主区间为：片段规则 [−10.49, 3.40] pp、片段裁判 [−10.70, 6.13] pp、代理规则 [−6.08, 2.67] pp、代理裁判 [−8.57, 2.90] pp。区间包含零，只能说明在当前样本、检测器和协议下方向未被确定；不能据此证明两种边界没有差异。检测器误差、提示词变化与人类感知误差也不在区间内。

E1/E2 的 crossed CI 同时反映全局 session 和 utterance 重采样，但仅基于 5 个 session 与 100 条话语。oracle TTFT_eff 是由同步接受规则定义的下界，不得与 candidate-readiness 或 marker diagnostic 合并为一个端到端效应。C2 v3 的 exact 检查是有限、确定性网格中的完整性证据，不是概率意义上的跨环境错误率估计。

A1 报告中位数与 IQR，P1 报告单元中位数范围和描述性 empirical P95；两者均未建立生产尾延迟分布。A2 因处理前输入和后续生成不一致，不具有明确的处理效应 estimand。

## 7.4 适用条件与后续证据

时间轴机制可迁移的最低条件是：TTS 能报告文本片段与音频块归属，推理引擎允许裁剪缓存，并能同步维护 mask、position、token ledger 和 role/EOT 状态。后端若提供词级 duration、设备时钟或 loopback 波形，可把 software-fragment boundary 扩展到更接近 device/acoustic boundary；若只提供不透明音频流，则必须另建对齐和缓冲观测。

截至本轮修订，C2 v3 正式 run 已接受并封存，E1/E2 crossed analysis 与 E3 weighting/dedup analysis 已完成；现有主张不需要新的 GPU 工作作为当前阻塞。真实异步音频闭环、固定轨迹 A2、人工/HCI 标注、跨语言/模型/后端复验和多裁剪长度 A1 均是增强外部或构念效度的可选后续工作，而不是既有窄化结论的未完成前提。

## 7.5 本章小结

本文最稳健的贡献不是证明真实对话质量或生产时延得到改善，而是建立并验证 software cursor—TTS fragment—assistant token—KV/role state 的公开状态合同。C2 v3 提供冻结环境下的直接 bitwise crop integrity 与 matched recovery exactness；C1 提供候选计算就绪、oracle 下界和作废计算的支持性刻画；C3 仅保留探索性结果。设备播放、声学接收和用户体验仍与软件游标证据明确分层。
---

# 第八章 总结与展望

## 8.1 全文总结

本文研究级联式流式语音对话系统在打断后的上下文状态管理，并将证据范围限定为 software-cursor-conditioned、TTS-fragment-level runtime/prototype。OpenAI、Azure 和 LiveKit 等系统已公开 playback-conditioned history truncation 的高层实践，KV crop 与 prefix reuse 也不是本文原创；本文的工作重点是把 software-consumed-sample cursor、TTS 文本片段、assistant token span、KV cache、attention mask、token ledger、position、role 和 EOT 状态连接为可审计合同。

贡献层级如下。**C2 是核心贡献**：实现 software cursor→fragment→token→KV crop→role/EOT recovery，并以 C2 v3 direct crop-integrity addendum 检验状态完整性。**C1 是支持性贡献**：对 pre-end-of-turn candidate-response generation with invalidation 的 candidate-selection/compute-readiness、同步 oracle acceptance 与 wasted-token 工作点进行受控刻画，不把其解释为 production deliverability 改善。**C3 是探索性扩展**：实现朴素、标记和重写三种历史处理方式；当前 A2 只提供受混杂描述性结果。

C2 v3 正式 run `c2crop_82103004_20260903T080512Z` 覆盖 24/24 cases、27/27 crop events、3 个 no-op、60 个 recovery steps、308 个逐 token production append 和 27/27 wrong-length negative control。在冻结 Qwen2-7B snapshot、BF16/SDPA/Transformers backend 下，28 层 K/V 的 crop 前 retained prefix、production post-crop 与独立 slicing oracle 逐张量 bitwise exact；匹配 token-ID chunks 恢复后的 K/V、logits、attention mask、token ledger 与 role/end/content state 亦 exact。该结论只涉及 direct crop integrity 与 matched-recovery determinism，不涉及 clean re-prefill numerical equivalence、32-token continuation、跨模型/后端/硬件或在线音频系统正确性。v1/v2 clean-reprefill 协议均保持 rejected；v3 不改变其 verdict。

固定轨迹 E3 的 label-weighted 主表包含片段目标 297 个标签/96 条对话，以及代理目标 380 个标签/100 条对话。generation−playback 的四项效应与 95% dialogue-cluster CI 分别为：片段规则 −3.37 pp [−10.49, 3.40]、片段自动裁判 −2.02 pp [−10.70, 6.13]、代理规则 −1.58 pp [−6.08, 2.67]、代理自动裁判 −2.63 pp [−8.57, 2.90]。精确去重后，片段目标为 169 个语义组、代理目标为 379 个语义组；片段自动裁判的 unique-group effect 为 0.00 pp [−7.98, 7.47]。这些结果仅是 fixed-detector-conditioned information-reproduction rate；区间不含检测器、提示词、模型变化或人类感知误差，不能确定方向性优势或差异不存在。规则与裁判的一致数只描述 automated-proxy agreement。

确认性 E1/E2 采用 100 条唯一话语×5 个独立进程 session 的交叉设计；每条件 500 个观测，正式区间由 crossed/product bootstrap 得到。C-E2 中 `never − B@0.92` 的 candidate-readiness 差值为 −0.03 ms [−0.64, 0.61]，oracle TTFT_eff 下界差值为 +20.80 ms [17.85, 23.65]；B@0.92 pooled waste 为 2.85% [1.12%, 4.73%]，survival 为 67% [58%, 76%]。同步 harness 中 B@0.92 的 first-deliverable 和 consumer marker 均值为 257.58 与 265.57 ms，只是程序执行顺序诊断，不代表生产可交付性。

C-E1 是 implementation-path comparison：System A 与 B@0.92 的 candidate-readiness 均值为 27.70 与 62.38 ms，A−B 为 −34.69 ms [−35.44, −33.95]；oracle 下界 A−B 为 +17.44 ms [14.41, 20.32]。两路径完整输出 token 仅 280/500 相同，首 token 为 465/500 相同，长度/EOS/max-token 状态为 495/500 相同，44/100 条唯一话语出现完整输出不一致；B@0.92 与 B-never 则为 500/500 相同。因此差值混合 tokenization、forward topology/shape、role boundary、kernel 和 Python scheduling，不能归因于单一额外 forward 或视为纯 incremental-prefill effect。

联合 A1 在固定 operation order、固定移除 32-token suffix、5 次预热和每点 50 次重复下，256–8192 token 的 joint crop+role 中位数为 31.054–48.315 ms，IQR 为 0.635–3.099 ms，重新预填充/联合路径中位数比为 2.254–40.620。P1 v2 的 9 个单元各含 20 次记录，software stop→crop 与 stop→role 的单元中位数分别为 2.44–2.53 和 78.6–80.8 ms；P95 仅为 empirical/descriptive order statistic。两组结果都不代表 device/acoustic stop、用户实际接收边界或生产端到端 barge-in。

RQ5 的描述性结果为：朴素、重写和标记实现的连贯性均值分别为 3.76、3.62 和 3.29；重写均值耗时 639 ms。由于三条件历史和下一轮生成不一致，这些数值不支持策略因果比较，也不证明重写延迟已被真实用户发言完全隐藏。

综上，本文最稳健的结论是：在冻结模型和后端下，software cursor 驱动的片段级保留可以被落实为具有 bitwise crop integrity 和 matched-recovery exactness 的 KV/role 状态操作；支持性实验进一步给出了固定协议的模型侧成本、headless 软件控制路径时延及候选生成的浪费—oracle-readiness 工作点。本文未测 device-presented samples、acoustically heard content、真实异步音频闭环或 HCI 效果，因而不对生产时延、声学边界或交互自然度作超出证据的结论。

## 8.2 可选后续工作

现有 C2 v3、E1/E2 crossed analysis、E3 weighting/dedup、A1 和 P1 证据已经满足本文收窄后的结论边界，**无需新增 GPU 工作作为当前提交阻塞**。若资源与目标期刊定位允许，可进一步开展以下研究。

1. **真实异步音频与设备/声学边界。** 接入在线 ASR、TTS、bounded audio queue、设备时钟或 loopback 波形，统一记录 stop request、device stop、acoustic stop、timeline query、KV crop 和 role recovery。
2. **固定轨迹 A2。** 缓存同一 assistant token 流、断句和打断点，并固定下一轮解码或成对随机种子，以形成可识别的策略比较。
3. **人工与 HCI 评测。** 使用盲法双标或直接用户实验测量特定信息复现、自然度、信任和主观交互质量，并报告标注一致性与不确定性。
4. **边界粒度与系统迁移。** 比较片段、词、音素和 token 级对齐，并在不同语言、主模型、TTS、chat template、dtype、attention backend 和推理引擎上重新验证状态合同。
5. **时延分布扩展。** 对 A1 随机化 operation order、覆盖多种 crop length；增加 P1 重复并在真实并发条件下估计稳定的高分位数。

## 8.3 工件可用性与投稿声明

实验工件、accepted/rejected campaign 状态、run ID、代码与结果 commit、hash、复算入口及主张边界的权威索引见仓库根目录 `REPRODUCIBILITY.md`。accepted E3 processed input 保存在 `experiments/sci34_supplement/results/e3_exact_rescue/p2_turns.json`；模型权重与原始 MultiWOZ 数据不在仓库中再分发，第三方资产受各自许可与访问条款约束。C2 v3、固定轨迹 E3、联合 A1、P1 v2 和确认性 E1/E2 工件均已在当前研究仓库中保存并以 run/hash 关联；C2 v1/v2 保持 rejected 状态。公共不可变 release/DOI 仍需作者在投稿前提供。

投稿声明草稿见 `paper2/declarations.md`，在以下事项由作者确认前不得改写为已完成、`none` 或 `not applicable`：公开数据与代码 URL/DOI、immutable release/tag 与访问日期；E3 派生数据再分发许可；仓库权利人和 LICENSE；第三方 notices；伦理审查或豁免、参与者与 consent；funding；每位作者的 competing interests；作者名单、顺序、CRediT 角色与 accountability；生成式 AI/自动化工具披露。以上均保留 **AUTHOR CONFIRM** 边界，不从 Git 元数据、实验内容或本稿推断。

## 8.4 结语

本项目一期内部实现关注用户话轮期间的增量预填充，本文则把研究重点推进到系统播报被打断后的显式状态修正。结果表明，可复查的 software cursor—fragment—token—KV/role 合同能够在受控环境中实现并验证；其向真实设备、声学听觉和用户体验的推广仍需对应层级的直接证据。
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
