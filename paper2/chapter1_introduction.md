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
