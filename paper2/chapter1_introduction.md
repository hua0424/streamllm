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
