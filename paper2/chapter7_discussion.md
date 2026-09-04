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
