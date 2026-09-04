# Paper 2 初稿内部审阅意见（2026-09-04）

## 0. 审阅对象与结论

- 主稿：`paper2/thesis_draft.md`。
- 权威正文源：`paper2/abstract.md`、`paper2/chapter1_introduction.md` 至 `paper2/chapter8_conclusion.md`。
- 审阅性质：只读内部审阅；本轮未修改正文、分析代码或实验工件。
- 同步性检查：`thesis_draft.md` 与 10 个权威源文件一致。

**总体建议：定位与报告口径上的 Major Revision；不是数据性 Major Revision。**

当前证据已经能够支撑一篇边界严格的**级联语音对话系统 / LLM 推理运行时状态管理论文**。论文最稳健的主张是：在冻结的 Qwen2-7B/Transformers/BF16/SDPA 条件和有限测试网格内，软件游标驱动的片段级保留可以落实为一组联合 KV/元数据状态转换，其中 production crop 具有 direct prefix integrity，匹配双臂的后续恢复轨迹具有 exact equality。

稿件目前没有需要撤回的系统性过度主张，也**不需要新增 GPU 实验作为当前修订的前提**。主要问题是：

1. 标题、RQ 顺序和实验章节顺序尚未真正体现“C2 是唯一核心贡献”；
2. E1/E2 仍有两个估计对象的命名需要闭合；
3. E3 的 `unique semantic group/boundary` 命名超出了实际 exact-key 去重操作，detector 和分析单位定义也不够完整；
4. C2 v1/v2/v3、旧口径审计、工件治理和限制语句在正文重复过多，主线被审计记录淹没；
5. A2 因不可识别而不能称为“负结果”。

本报告将问题分为：**必须修改、建议修改、应保留、条件性新实验**。行号均指当前权威分章文件；合并稿需在修改分章后重新生成。

---

## 1. 对作者列出的五项重点作出直接裁决

### 1.1 标题与研究定位

**裁决：研究定位接受，当前标题建议修改。**

当前标题“级联式语音对话中软件播放游标与 TTS 片段驱动的 KV 状态修正”技术上基本准确，但存在三点不足：

- 未出现读者最容易理解的场景词“打断”；
- “KV 状态”不足以概括 mask、token ledger、position、role/EOT 等联合状态；
- “软件播放游标与 TTS 片段驱动”偏实现说明，没有直接呈现“游标经片段映射到上下文提交边界”的问题结构。

相关动机和完整状态对象实际已在 `paper2/chapter1_introduction.md:7-11`、`paper2/chapter1_introduction.md:27-33` 和 `paper2/chapter7_discussion.md:13-17` 讲清。

**首选标题：**

> **级联式语音对话打断中的上下文状态修正：从软件播放游标和 TTS 片段到 KV 与角色恢复**

**对应英文标题：**

> **Context-State Repair for Barge-In in Cascaded Spoken Dialogue: From a Software Playback Cursor and TTS Fragments to KV and Role Recovery**

**备选标题 1：**

> 从软件播放游标到 KV 状态恢复：级联式语音对话打断的跨层状态合同

**备选标题 2：**

> 级联式语音对话打断后的软件播放感知状态对齐：片段级 KV 裁剪与角色恢复

不建议把论文定位成 HCI、真实全双工用户体验或生产级低延迟论文。当前没有真实异步音频闭环、设备呈现、声学接收或用户实验；论文对此已在 `paper2/chapter7_discussion.md:35-57` 作出合格限制。

### 1.2 C2/C1/C3 贡献层级

**裁决：接受 C2 为唯一核心贡献、C1 为支持性刻画、C3 为探索性扩展。**

文字声明在 `paper2/chapter1_introduction.md:15-21`、`paper2/chapter7_discussion.md:3-9` 和 `paper2/chapter8_conclusion.md:5-9` 基本一致。不过，章节结构没有落实这一层级：当前 RQ1 是 E3，RQ2/RQ3 是 C1，核心 C2 排在 RQ4；实验章也到 `paper2/chapter6_experiments.md:122` 才报告 C2。读者会感到 C1/E3 与 C2 是并列主线。

**建议的逻辑次序：**

1. 核心 RQ：C2 的 direct crop integrity、matched-arm recovery exactness 及固定协议成本（现 RQ4；C2 v3 + A1 + P1）；
2. C2 的 downstream 支持性 RQ：software-cursor retention 与 generation retention 的 fixed-detector-conditioned information reproduction（现 RQ1；E3）；
3. C1 支持性 RQ：阈值、候选可用性、discarded-token ratio 与 oracle 下界（现 RQ2；C-E2）；
4. C1 路径审计 RQ：非 token-equivalent implementation paths 的 readiness 差异（现 RQ3；C-E1）；
5. C3 探索性 RQ：三种历史自然化实现的受混杂描述（现 RQ5；A2）。

最好重排 RQ 和第六章结果顺序；若暂不重编号，至少增加“贡献—RQ—实验—证据层级”映射表，并把 C2 结果前置。

### 1.3 C2 结论边界

**裁决：接受 v3 仅证明 direct crop integrity 与 matched recovery；不恢复 clean re-prefill numerical equivalence。**

正文在 `paper2/abstract.md:7,19`、`paper2/chapter4_method.md:127-141`、`paper2/chapter6_experiments.md:124-141`、`paper2/chapter7_discussion.md:13-17` 和 `paper2/chapter8_conclusion.md:9` 一致保留了这一边界，没有把 v1/v2 事后改判，也没有推广至跨模型、后端、硬件或在线音频正确性。

但需要进一步收窄三个词组：

- “正式正确性证据”改为“正式 direct crop-integrity 与 matched-recovery 证据”（`paper2/chapter1_introduction.md:31`）；
- “C2 核心正确性”改为“C2 直接裁剪完整性与匹配恢复证据”（`paper2/chapter6_experiments.md:122-128`）；
- “状态合同能够……实现并验证”改为“状态合同中受 v3 覆盖的直接裁剪与匹配恢复性质通过 exact gate”（`paper2/chapter8_conclusion.md:41`）。

`matched-recovery determinism` 也应定义为**同一 accepted run 内的 matched-arm exactness**：两臂从精确匹配的保留状态出发，接受相同 token-ID chunks 和相同操作序列时逐步一致；它不是跨进程、跨设备或 clean-reprefill 的可重复性声明。首次定义位置建议放在 `paper2/chapter4_method.md:139`。

上一轮 CF-04 若按原要求“证明 clean re-prefill equivalence”逐字验收，仍不能标为完全解决；正确做法不是恢复该主张或继续调阈值，而是在正式回复中明确：作者不再提出该等价性主张，v1/v2 保持 rejected，并以 v3 回答更窄且可识别的 direct-integrity 问题，请编辑接受这一主张收缩和替代证据。该项属于作者立场/编辑裁决，不是当前必须新增 GPU 的缺口。

### 1.4 E1/E2 的重新解释

**裁决：27.70/62.38 ms、C-E1 和 oracle 的主要解释均已正确；仍有两项必须改名或补定义。**

已经正确的内容包括：

- 27.70/62.38 ms 始终是 `arrival→candidate selection/readiness`，不是 generator yield 或 production TTFT（`paper2/chapter6_experiments.md:99-116`）；
- C-E1 是非 token-equivalent implementation-path comparison，不识别单一 incremental-prefill effect（`paper2/chapter3_formulation.md:239-241`、`paper2/chapter6_experiments.md:99-112`）；
- oracle `TTFT_eff` 被限定为同步接受规则下的乐观下界/推测收益上界（`paper2/chapter3_formulation.md:180-190`、`paper2/chapter6_experiments.md:91-93`）；
- crossed/product bootstrap 与 100 utterances × 5 sessions 的技术重复结构表达正确（`paper2/chapter3_formulation.md:237-241`、`paper2/chapter6_experiments.md:23-25`）。

必须改的两项为：

1. E1/E2 不能把实证对象简称为真实 **pre-end-of-turn** 候选收益。`endpoint_accept` 是候选处理后记录的同步 oracle 事件，现有数据证明的是 **pre-oracle-acceptance candidate availability in a synchronous segmented-text harness**，不证明候选在真实 end-of-speech 之前就绪。`pre-end-of-turn` 可作为机制设计目标保留，但必须说明确认性 harness 未识别真实端点前收益。涉及 `paper2/abstract.md:11,23`、`paper2/chapter4_method.md:25-29`、`paper2/chapter6_experiments.md:23,80-93`。
2. `survival rate=67%` 的正式实现是 `sum(survived)/all condition records`，即 335/500，分母包含未触发候选的单元。它更准确地表示**接受时候选可用率（endpoint candidate availability rate）**，不是通常意义上的 `P(survive | candidate launched)`。涉及 `paper2/chapter3_formulation.md:227-233`、`paper2/chapter6_experiments.md:80-91,185-187` 和 `paper2/chapter8_conclusion.md:13`。

另建议把“pooled waste”统一称为 **pooled discarded-token ratio**；该比率不是 FLOPs、GPU 时间、能耗或显存带宽意义上的计算浪费。公式本身可保留，并补一句它在每个 bootstrap replicate 内按 ratio-of-sums 计算。

### 1.5 E3 的定位

**裁决：接受 label-weighted 为主加权口径、dialogue-weighted 与去重为敏感性分析，且结果不作 HCI/人类语义推断；但 E3 的估计对象和去重术语尚未闭合。**

正确且应保留的内容：

- 主表先给 label-weighted point estimate 和配套 dialogue-cluster CI（`paper2/chapter6_experiments.md:40-53`）；
- dialogue-weighted 与去重结果放在敏感性表（`paper2/chapter6_experiments.md:55-66`）；
- 全文没有从区间跨零推出 equivalence/noninferiority/absence-of-effect（`paper2/chapter7_discussion.md:59-63`）；
- software-consumed、device-presented 和 acoustically heard 三层区分充分（`paper2/chapter3_formulation.md:31-55`、`paper2/chapter7_discussion.md:35-41`）。

必须修改：

1. `unique semantic boundary/group` 应改为 **target-specific exact-key group/deduplication**。分析器实际按 `id`、`trajectory_id`、两条件 `history_key`、目标 hash，以及 fragment 口径下的 `heard_token_end` 组成精确键；没有语义聚类或人工判重。相关正文：`paper2/abstract.md:9,21`、`paper2/chapter1_introduction.md:17,49`、`paper2/chapter4_method.md:165`、`paper2/chapter5_implementation.md:114`、`paper2/chapter6_experiments.md:53-64`、`paper2/chapter7_discussion.md:29,47`、`paper2/chapter8_conclusion.md:11`。
2. `297→169` 的差值 128 表示**去重后减少 128 个额外 label 权重**，不能写成“128 个标签属于重复组”（`paper2/chapter6_experiments.md:53`）。
3. 需要集中定义 E3 分析单位：四个 injection labels 实际是 0.25、0.5、0.75 和 fragment boundary 四种软件游标注入位置；fragment/proxy 各自按目标字段非空确定资格；每个 eligible `(dialogue, injection position)` 在 label-weighted 主分析中等权；dialogue-weighted 是先在对话内平均再对对话等权；四个 target×detector 单元应说明是并列冻结操作化还是有预先主次。
4. 词面规则和 `specific-reference-v3` judge 需要最低限度的操作定义。正文至少应说明：规则从目标提取数字、首字母大写词和长度不少于 5 的非停用内容词，任一词边界/长词子串命中即阳性；judge 接收 TARGET 与合并的两轮 REPLY，判断回复是否使用、重复或引用 TARGET 中的具体信息，generic topical overlap 不算，greedy 输出 YES/NO。完整 prompt 可留工件。建议在 E3 方法中加入一个紧凑 detector 表。

A2 的详细因果限制是正确的，但“受混杂负结果/negative extension”不成立。既然输入和生成轨迹不一致导致处理效应不可识别，就不能把结果称为 null/negative evidence。全文应改成“受混杂的探索性描述”或“未形成可解释策略比较的探索性运行”。涉及 `paper2/abstract.md:11,23`、`paper2/chapter1_introduction.md:45-47`、`paper2/chapter2_related_work.md:74`、`paper2/chapter7_discussion.md:5`。

---

## 2. 必须修改事项（按优先级）

### M1. 用章节顺序落实 C2 的核心地位

**锚点：** `paper2/chapter1_introduction.md:15-21`、`paper2/chapter1_introduction.md:45-49`、`paper2/chapter6_experiments.md:3-11`、`paper2/chapter6_experiments.md:40-141`。

当前贡献标签说 C2 核心，但读者先看到 E3 和两个 C1 RQ，C2 到 RQ4/6.5 才出现。E3 还紧接在 C3 小节之后，容易被看作 C3 附属或第四个未命名贡献。

**最低修改：**增加贡献—RQ—实验映射，并明确 E3 是 C2 的 downstream/supporting evidence；**更优修改：**按 C2→E3→C1→C3 重排 RQ 和第六章。

### M2. 重命名 E1/E2 的“pre-end-of-turn”和“survival”估计对象

**锚点：** `paper2/abstract.md:11,23`、`paper2/chapter4_method.md:25-29`、`paper2/chapter6_experiments.md:80-93`。

确认性实验使用同步、候选后 oracle acceptance。不得把 +20.80 ms 理解为真实端点前就绪或生产收益。67% 是所有单元中在接受时有可用候选的比例，而不是给定曾触发候选后的条件存活率。

**最低修改：**统一使用“同步 oracle 接受前候选生成/接受时候选可用率”，并在表 6-4 和首次定义处写出 335/500 的分子分母。

### M3. 将 E3 “semantic group” 改为 exact-key group，并补齐 estimand

**锚点：** `paper2/chapter3_formulation.md:207-216`、`paper2/chapter6_experiments.md:40-66`。

现有分析是精确键分组，不是语义判重。主分析也需要让读者清楚回答“从什么集合等权抽取哪个单位”。

**最低修改：**重命名、修正 128 的表述，并增加 1 个自然语言 estimand 段落。

### M4. 补齐 E3 rule/judge 的操作定义

**锚点：** `paper2/chapter3_formulation.md:209-216`、`paper2/chapter6_experiments.md:68-72`。

rule 与 judge 绝对阳性率相差约 20–30 个百分点；只报告名称和 agreement counts 不足以解释“信息复现率”测了什么。

**最低修改：**增加 detector 表：输入、目标、判据、输出映射、解码、条件盲化/是否向 judge 暴露 condition、两轮回复的聚合方式。

### M5. 将 C2 抽象成 external-progress-conditioned joint prefix-state repair

**锚点：** `paper2/chapter1_introduction.md:27-33`、`paper2/chapter3_formulation.md:109-156`、`paper2/chapter7_discussion.md:13-17`。

当前贡献陈述容易被评价成“把已知 playback truncation 和已知 DynamicCache.crop 接起来”。第三章已经有足够的学术原料，应把 C2 组织为四层合同：

1. boundary resolution：$p\xrightarrow{\Phi}$ 合法 assistant commit boundary；
2. joint state：KV、mask、token ledger、position 和 role/EOT 是不可分割状态；
3. invariant-preserving transition：crop 与 close/reopen 保持联合不变式；
4. falsifiable validation：slicing oracle、wrong-length negative control 和 matched recovery 分别检查不同性质。

这能提升贡献的可迁移抽象，而不需要发明新缓存算法。

### M6. 收窄 C2 的“正确性”和 `determinism` 用词

**锚点：** `paper2/chapter1_introduction.md:31`、`paper2/chapter4_method.md:139`、`paper2/chapter6_experiments.md:122-141`、`paper2/chapter8_conclusion.md:41`。

主文应始终写出受测性质，不用无定语的“核心正确性”。`matched-recovery determinism` 应定义为 within-run matched-arm exactness，或直接改名为 `matched-arm recovery exactness`。

### M7. A2 不再称“负结果”

**锚点：** `paper2/abstract.md:11,23`、`paper2/chapter1_introduction.md:45-47`、`paper2/chapter2_related_work.md:74`、`paper2/chapter7_discussion.md:5`。

受混杂而不可识别不是 negative/null result。只保留“描述性探索运行”和“不能作策略因果比较”。

---

## 3. 建议修改事项

### m1. RQ—实验映射表遗漏 C2 v3

`paper2/chapter3_formulation.md:243-251` 的 RQ4 行只列 A1/P1，遗漏 direct integrity 的 C2 v3；RQ2 又保留“E2（同时作为 A3）”的旧别名。应统一为：RQ1→E3，RQ2→C-E2，RQ3→C-E1，RQ4→C2 v3+A1+P1，RQ5→A2。

### m2. 英文摘要明确 CI 类型

`paper2/abstract.md:23` 中 C-E2 的 `[-0.64,0.61]` 与 `[17.85,23.65]` 应明确为 crossed 95% CIs。

### m3. 收窄 crossed CI 的推广范围

`paper2/chapter7_discussion.md:63` 已承认只有 5 个技术 session，但还可明确：区间反映 100 utterances 与已观察到的 5 个 process-initialization levels，不覆盖跨硬件、跨日、负载、并发或部署环境变异。现有 `analysis_v2` 的 per-session/leave-one-session-out 结果可放补充材料，无需新 GPU。

### m4. 使 Playback/P1 名称脱离上下文也不被误读

- `paper2/chapter1_introduction.md:7` 的“尚未播放”改为“尚未被软件游标覆盖”；
- 表 6-2 的 `Playback` 首次定义为 `software-cursor fragment-retention condition`；
- 表 6-8 的“软件停播确认”改为“headless 播放器线程 stop acknowledgement”；
- `leaked_samples=0` 就地说明只指 software counter。

### m5. 封闭角色状态集合，删除内部决策编号

`paper2/chapter3_formulation.md:135` 的“RolePhase 至少区分”应改为当前受测适配器的完整支持集合。`paper2/chapter4_method.md:121` 和 `paper2/chapter5_implementation.md:78` 中的内部编号 `D-022` 应删除，保留状态语义即可。

### m6. 限定 slicing oracle 的独立性

“独立切片 oracle”首次出现时应写成“独立于 production crop 接口、从同一 pre-crop snapshot 逐层切片的 oracle”，避免被误解为外部独立复现。

### m7. 修正 novelty scan 时态

`paper2/chapter1_introduction.md:53` 的“计划检索”与后文已经报告检索结果不一致，改为“开展了检索/覆盖以下渠道”。

### m8. 删除论文内项目管理式表述

`paper2/chapter7_discussion.md:71` 和 `paper2/chapter8_conclusion.md:25` 的“无需新增 GPU 工作作为当前提交阻塞”应从学术主文删除，改为“现有实验支持本文限定范围内的结论；以下工作将增强外部或构念效度”。GPU 决策保留在本审稿意见、handoff 或项目文档中。

### m9. “效应均为小负值”不要引入未定义的实质性阈值

`paper2/chapter7_discussion.md:29` 建议改为“点估计均低于零，但区间均跨零，且未预设实质性差异阈值”。

### m10. 最终投稿前补齐主文可见环境表

`paper2/chapter5_implementation.md:122` 和 `paper2/chapter6_experiments.md:13-17` 可用紧凑表列出精确模型 artifact/revision、Transformers、PyTorch/CUDA、BF16、SDPA、RTX 3090 和权威工件入口。值可从已有 manifest 抄录，不需重跑。

---

## 4. 篇幅与可读性裁决

### 4.1 建议压缩目标

当前 `thesis_draft.md` 为 1108 行，正文审计和限制语句密度过高。建议学位论文版整体压缩 **20%–25%**；如衍生期刊稿，建议压缩 **25%–30%**。目标不是删除证据边界，而是让同一边界只承担一次定义、一次结果限定和一次效度讨论。

### 4.2 C2 v1/v2 在正文保留到什么程度

**必须保留：**

- v1/v2 按冻结门槛 rejected；
- v3 回答不同的 direct-integrity 问题，未改门槛、未改判；
- v2 control 与 production forward topology 不匹配，因此数值失败既不能定位 crop bug，也不能建立 clean-reprefill equivalence。

**建议正文分工：**

- 摘要：只留一句 rejected 事实和 v3 不改变 verdict；删 42/45 与 topology 细节；
- 绪论：最多两句，说明主张边界；
- 方法：保留一段解释为什么从不可识别的 clean-reprefill control 改为 direct slicing oracle；
- 结果：用 v1/v2/v3 三行状态表，保留 v2 42/45；
- 讨论：保留一段认识论解释；
- 结论：只留“v1/v2 rejected，v3 不改判”。

完整 frozen gates、case/probe IDs、逐层差异、seam/chunk topology、运行日志与审计链全部移入补充材料和 `REPRODUCIBILITY.md`。

### 4.3 重点压缩清单

| 对象 | 当前锚点 | 主文建议保留比例 |
|---|---|---:|
| C2 v1/v2 protocol genealogy | `abstract.md:7,19`；`chapter1_introduction.md:31-33`；`chapter4_method.md:141`；`chapter5_implementation.md:135`；`chapter6_experiments.md:141`；`chapter7_discussion.md:17`；`chapter8_conclusion.md:9` | 20%–30% |
| C2 v3 方法/实现/结果重复 | `chapter4_method.md:127-141`；`chapter5_implementation.md:120-135`；`chapter6_experiments.md:124-141` | 各章明确分工后总量减约 1/3 |
| C-E1 的 280/500、465/500、44/100 | `abstract.md:11,23`；`chapter1_introduction.md:41`；`chapter5_implementation.md:108`；`chapter6_experiments.md:110`；`chapter7_discussion.md:25`；`chapter8_conclusion.md:15` | 详细数只留第六章；其余留一句边界 |
| readiness/deliverable/consumer/291 ms | `chapter1_introduction.md:23,43`；`chapter3_formulation.md:160-194`；`chapter6_experiments.md:91-93`；`chapter7_discussion.md:21-25`；`chapter8_conclusion.md:13` | 第三章定义、第六章数值、第七章一句外推限制 |
| 三层播放语义 | `chapter1_introduction.md:9`；`chapter2_related_work.md:11`；`chapter3_formulation.md:31-55`；`chapter4_method.md:13`；`chapter5_implementation.md:22`；`chapter6_experiments.md:27`；`chapter7_discussion.md:35-41`；`chapter8_conclusion.md:21` | 摘要一句、第三章完整、效度章完整，其余压至一句或删除重复表述 |
| A1 固定 32-token / P1 empirical P95 | `chapter1_introduction.md:35`；`chapter4_method.md:104-106`；`chapter6_experiments.md:143-175`；`chapter7_discussion.md:49`；`chapter8_conclusion.md:17` | 协议留方法、数字与一次限制留结果、讨论一句 |
| E3 禁止推断清单 | `abstract.md:9,21`；`chapter1_introduction.md:49`；`chapter4_method.md:165`；`chapter6_experiments.md:53,66`；`chapter7_discussion.md:29,61`；`chapter8_conclusion.md:11` | 完整清单只留结果/效度；摘要压成一句 |
| run/commit/hash/seal/CRLF | `chapter5_implementation.md:120-143`；`chapter8_conclusion.md:35-37` | 主文仅留稳定 artifact 入口；治理细节移仓库文档 |
| novelty 访问失败过程 | `chapter1_introduction.md:53-55`；`chapter2_related_work.md:35-49` | 主文留范围/方法/限制一句；日志移附录 |
| AUTHOR CONFIRM 清单 | `chapter8_conclusion.md:33-37` | 移独立 declarations/front matter；结论只留入口 |
| 旧 E1/E2 口径审计 | `chapter6_experiments.md:118-120` | 主文只说因错误时间原点排除；旧数字移补充 |
| A2 详细结果 | `chapter6_experiments.md:177-181` | 学位稿可留简表；期刊稿建议移补充，只留不可识别结论 |

### 4.4 章节职责建议

- 第三章：保留定义、$\mathcal{Z}$ 联合状态、不变式和合法转换；
- 第四章：保留算法/状态机和 v3 oracle 设计；
- 第五章：只保留抽象到 API/模块的映射、环境与复现入口，删除决策日志和平台治理细节；
- 第六章：只报告结果，不重新解释完整实现；
- 第七章：集中处理效度边界；
- 第八章：回答“学到了什么”，不要再复制所有表格数字和项目待办。

---

## 5. 应保留的内容

以下信息是论文可信度的核心，不应因压缩而删除：

1. **三层播放语义**：software-consumed、device-presented、acoustically heard 的严格区分（`paper2/chapter3_formulation.md:31-55`）。
2. **联合状态与不变式**：$\mathcal{Z}$、KV/mask/ledger 长度一致、assistant content span、合法 crop point、position 连续和 close/reopen 约束（`paper2/chapter3_formulation.md:122-156`）。
3. **EOT 唯一提交规则**：预测 EOT 不作为内容重复写入，`ASSISTANT_EOT_PENDING` 与唯一 close commit（`paper2/chapter4_method.md:108-121`）。
4. **三方 v3 关系和负控**：pre-crop retained prefix、production post-crop、独立于 production crop 的 slicing oracle，以及 wrong-length negative control（`paper2/chapter4_method.md:127-139`）。
5. **B-noKV 的证据边界**：只用于 A1 计时，不作为 v3 数值等价 oracle（`paper2/chapter4_method.md:145-147`）。
6. **C-E1 的非 token-equivalent 边界**：不能解释为纯 incremental-prefill effect，也不能按 matched outputs 做 post-treatment selection（`paper2/chapter3_formulation.md:239-241`）。
7. **E3 不作人类/HCI推断**：自动 rule/judge 不是 reference standard，跨零区间不是 equivalence 或 absence-of-effect（`paper2/chapter7_discussion.md:27-41,59-63`）。
8. **范围受限的 novelty 语言**：不把 high-level playback truncation 或 crop primitive 当原创，不声称全球首次（`paper2/chapter2_related_work.md:70-76`）。

---

## 6. 最强反方论证及答复方向

### 反方论证

> C2 v3 有力证明了“给定 keep length 后，production crop 与同一 pre-crop K/V 的逐层切片一致”，但没有独立证明 software cursor 选择的片段就是正确的人类交互边界，也没有证明裁剪状态等价于 canonical clean re-prefill。E3 没有确定方向性后果，C1 只是同步 oracle 下界，A1/P1 又是固定微基准/headless 软件路径。因此，论文可能只是把已知 playback-history 原则和已知 crop primitive 接成一个工程适配层，而没有证明真实对话质量或生产时延改善。

### 建议答复

不要用更强结论反驳。应正面承认：

- 人类交互边界、clean re-prefill equivalence、生产时延与 HCI 效果不在本文主张内；
- 学术增量不是新的 crop primitive，而是 **external-progress-conditioned joint prefix-state repair contract**；
- 贡献在于将外部进度解析、合法提交边界、联合缓存解释状态、角色/EOT 转换与可证伪 exact gate 组织成公开合同；
- v3 直接检验该合同中可识别的实现性质；E3/C1/A1/P1 是不同层级的支持性刻画，不共同构成端到端效果证明。

这种答复比扩大实验结论更可信。

---

## 7. 是否需要新实验

### 7.1 当前论文边界下

**不需要新增 GPU、声学或 HCI 实验作为本轮修订的阻塞项。**

M1–M7 均可以通过重排、重命名、补操作定义、使用现有 raw/analysis 的离线统计以及压缩审计信息解决。尤其：

- 不重跑 C2；
- 不重跑 accepted E1/E2/E3/A1/P1；
- 不把 v1/v2 改判；
- 不为了“语义组”补实验，而是把它准确改名为 exact-key group；
- 若希望补 per-session、leave-one-session-out、条件存活率或 E3 confusion table，优先从既有 raw/analysis 做 versioned 离线补充，不使用 GPU。

### 7.2 只有扩大主张时才需要新证据

| 拟扩大主张 | 所需证据 |
|---|---|
| clean re-prefill numerical equivalence | topology-matched、预冻结的新协议；不能复用 v2 verdict |
| 32-token continuation equivalence | 从 crop/slice/clean states 实际续写并逐步比较 logits/tokens/EOS |
| 跨模型、dtype、backend、硬件普适 | 多模型、多模板、多后端和设备矩阵；跨硬件不宜预设 bitwise equality |
| 真实 end-of-speech 前候选收益或 production TTFT | 独立异步 endpoint gate、consumer、TTS admission 与统一墙钟事件 |
| device/acoustic stop 或“用户实际听到” | 真实异步 TTS/播放器、设备时钟或 loopback 波形 |
| 人类语义保真 | 盲法人工双标、仲裁、一致性与不确定性 |
| HCI 自然度、信任或用户体验 | 直接用户研究 |
| A2 策略效应或“无效/负效应” | 固定首轮 token 轨迹、断句和打断点，并控制/配对下一轮生成 |
| equivalence/noninferiority/harm | 预设 estimand、实质性 margin、功效和对应统计设计 |

---

## 8. 修订优先级清单

### 第一优先级：必须先决定/修改

1. 采用能体现“打断 + 上下文联合状态”的标题；
2. 让 C2 在 RQ 和结果顺序中真正前置，明确 E3 是 C2 的 downstream 支持性证据；
3. 把 E1/E2 的 `survival` 改成接受时候选可用率，并收窄 `pre-end-of-turn` 实证措辞；
4. 把 E3 `semantic group` 改成 exact-key group，修正 297→169 的解释；
5. 补 E3 分析单位和 rule/judge 操作定义；
6. 将 C2 的“正确性/determinism”收窄到 direct crop integrity 与 within-run matched-arm recovery exactness；
7. 将 A2 从“负结果”改为“受混杂、不可识别的探索性描述”。

### 第二优先级：完成主线压缩

1. 摘要删除 v1/v2 的 42/45 与 topology 细节，压缩 C1 数字；
2. C2 v1/v2 过程只在方法/结果/讨论各承担一种职责；
3. 第五章删除 D-022、CRLF、具体 commit/hash 重复和本地治理过程；
4. 第八章删除 GPU 阻塞、AUTHOR CONFIRM 工作流和大段结果复述；
5. 将完整 novelty 查询和访问失败日志移补充材料；
6. 学位稿整体减 20%–25%，期刊稿再减至 25%–30%。

### 第三优先级：投稿前处理

1. 补紧凑环境表和稳定 artifact 入口；
2. 把 per-session/leave-one-session-out、E3 confusion matrix 等放补充材料；
3. 待目标期刊确定后，再完成 reference style、public/anonymous URL、release/DOI、LICENSE、伦理、funding、COI、作者名单与 CRediT、AI disclosure；
4. 由权威分章重新生成 `thesis_draft.md`，不要直接编辑合并稿。

---

## 9. 最终判断

这份初稿已经越过“实验是否站得住”的主要门槛，但还没有越过“读者能否迅速看出唯一核心贡献”的门槛。下一轮修订应避免继续添加审计信息，而应完成三件事：

1. 把 C2 从组件清单提升为**外部进度条件下的联合前缀状态修正合同**；
2. 把 E1/E2 和 E3 的估计对象用准确、可复述的自然语言闭合；
3. 将失败协议、工件治理和重复限制下沉，让标题、摘要、RQ、结果和结论围绕同一条 C2 主线。

在维持当前结论边界的前提下，本报告**不建议安排新的 GPU 实验**。