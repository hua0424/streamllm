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
