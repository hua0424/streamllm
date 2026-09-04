# 第二篇论文大纲

**建议题目**：级联式语音对话打断中的上下文状态修正：从软件播放游标和 TTS 片段到 KV 与角色恢复

**English title**: Context-State Repair for Barge-In in Cascaded Spoken Dialogue: From a Software Playback Cursor and TTS Fragments to KV and Role Recovery

**论文定位**：工程与系统型硕士论文。OpenAI、Azure 和 LiveKit 已建立 playback-conditioned transcript/session-history truncation 的高层先例，KV crop 亦是既有原语；本文的核心贡献边界是公开级联栈中的 external-progress-conditioned joint prefix-state repair，即 software-consumed-sample cursor→TTS fragment→assistant token span→KV crop→mask/token/position/role/EOT recovery 及其受控 exact evidence。C1 仅作 candidate selection/compute-readiness、synchronous oracle acceptance 与 discarded-token ratio 的支持性刻画，C3 为受混杂的探索性描述。

**权威正文源**：`abstract.md`、`chapter1_introduction.md` 至 `chapter8_conclusion.md`、`references.md`。`thesis_draft.md` 由 `uv run python -m experiments.scripts.build_thesis_draft` 自动合并；中英文 IEEE 稿为后续衍生版本，不应反向覆盖分章源稿。

---

## 第一章 绪论

### 1.1 研究背景与问题界定
- 级联语音对话的模块化优势与打断后的生成、合成、软件播放状态错位
- 严格区分 software-consumed、device-presented 与 acoustically heard 三层
- 承认 OpenAI、Azure、LiveKit 的 prior art，并将对象收窄到联合前缀状态修正

### 1.2 贡献层级、研究问题与证据结构
- RQ1/C2：direct crop integrity、within-run matched-arm recovery exactness 及 A1/P1 成本边界
- RQ2/E3：固定检测器条件下的 downstream information reproduction
- RQ3/C-E2：候选可用性、pooled discarded-token ratio 与同步 oracle 下界
- RQ4/C-E1：非 token-equivalent implementation-path comparison
- RQ5/A2：受混杂的探索性描述
- 贡献—RQ—实验—证据层级映射表

### 1.3 本文工作与贡献
- 1.3.1 C2（核心）：external-progress-conditioned joint prefix-state repair 的四层合同
- 1.3.2 E3：C2 的 downstream 支持性证据
- 1.3.3 C1（支持）：同步分段文本 harness 中的 pre-oracle-acceptance candidate generation
- 1.3.4 C3（探索）：三种历史自然化实现，不作策略因果、负向或零效应解释

### 1.4 研究定位与 novelty 边界
- 可检视研究工件；novelty 只作截至 2026-09-03 的 scoped public-source non-identification，不作 global-first 主张

### 1.5 论文组织结构

---

## 第二章 相关工作

### 2.1 商用与工程系统中的打断—上下文管理
- OpenAI Realtime API
- Azure Voice Live
- LiveKit Agents
- 比较公开接口语义、实现可见性和研究工件，不猜测闭源内部架构

### 2.2 流式语音对话与打断处理
- 提前触发与增量推理：LTS-VoiceAgent、RelayS2S、Predictive ASR
- 端到端全双工：Moshi；说明帧同步减弱但不自动消除播放缓冲差异
- 打断检测：FireRedChat；与打断后的历史修正互补

### 2.3 LLM 推理与 KV 缓存操作
- `DynamicCache.crop`、前缀缓存、vLLM/SGLang
- IntentKV、Speculative Interaction Agents
- 播放信号驱动的显式 KV 修正与角色恢复

### 2.4 Targeted public-source novelty scan 与本文定位
- 截止日期 2026-09-03；报告数据库/第一方渠道、五组查询族、纳排规则、snowballing 起点与访问限制
- 最近邻矩阵承认 OpenAI/Azure/LiveKit 高层先例、response-level speculation/输入侧预取和 KV crop/prefix-reuse prior art
- novelty 仅表述为报告公开来源范围内未识别完整软件状态路径，不作系统综述、专利检索或全球首次声明

---

## 第三章 问题形式化

### 3.1 系统模型与异构进度
- ASR 稳定文本段、LLM token、TTS 文本片段、音频块和播放采样
- 原始进度不直接比较；映射到 token 域后描述顺序约束

### 3.2 播放边界与片段级历史一致性
- 片段级保留边界 $\widehat H(p)$
- 片段操作语义下的历史对齐
- 字符比例—空白边界代理；明确不是物理 token 真值

### 3.3 反向查询与 KV 合法性
- 片段记录关联与反向索引 $\Phi$
- 裁剪阶段：KV、掩码、账本端点一致
- 恢复阶段：角色串长度 $q$，下一轮 user 从 $N+q$ 开始

### 3.4 评测指标
- 五个事件：last_segment_arrival、first_token_ready、endpoint_accept，以及仅作同步诊断的 first-deliverable/consumer markers
- 主指标 $L_{\mathrm{arr}\to\mathrm{cand}}=t_{\mathrm{cand}}-t_{\mathrm{arr}}$；$\mathrm{TTFT}_{\mathrm{eff}}$ 为同步 oracle 接受下界
- A1 联合恢复与 P1 prepared-state 软件控制路径的区分；stop→crop 与 stop→role 为嵌套区间
- E3：fragment/proxy 目标、label-weighted 主 estimand、dialogue-weighted 与 target-specific exact-key 敏感性，以及 rule/judge 操作定义
- pooled discarded-token ratio = wasted/(wasted+final)，在 bootstrap replicate 内按 ratio of sums 计算
- 接受时候选可用率以全部 condition records 为分母，不是给定候选已启动后的条件存活率

### 3.5 本章小结

---

## 第四章 方法设计

### 4.1 总体设计
- 输入增量流水、输出断句/TTS/播放和打断状态修正

### 4.2 可作废的推测生成调度
- 单一推测阈值与候选预算
- 真值终点接受；在线 TTS 门控为未验证扩展
- 推测作废状态机

### 4.3 播放感知的 KV 缓存管理
- 片段关联时间轴和断句—token 对齐
- `DynamicCache.crop`、掩码和 token 账本同步裁剪
- 角色边界恢复和 assistant 侧 KV 累积
- P1 prepared-state 屏障：播放前 `ensure_full()` + 设备同步，准备耗时排除
- B-gen、B-noKV、B-syn 的语义边界

### 4.4 被打断历史处理策略
- 朴素、标记、重写
- 重写“不新增信息”是提示约束，不是形式保证
- 延迟可重叠是潜力，不声称完全隐藏

---

## 第五章 系统实现

### 5.1 模块架构
- 模块、接口与验证对象表

### 5.2 片段时间轴与断句对齐
- software cursor、片段、token 与 sample 的计数语义和生产者合同
- 非空白字符守恒及边界钳制

### 5.3 KV、token ledger 与角色状态
- 联合状态容器与长度/content-span 不变式
- tokenwise append、EOT pending 与唯一 close commit
- 播放期零内容保留和整段候选作废的不同 crop 语义

### 5.4 编排与事件语义
- System A/B 路径、同步 oracle 与 diagnostic markers
- Mock TTS 是真机画像参数化近似，不代表真实异步音频

### 5.5 实验协议的实现
- C-E1/C-E2 crossed design 与路径可比性
- 固定轨迹 E3、联合 A1 和 prepared-state P1

### 5.6 C2 v3 exact-only 验证实现
- 同一 pre-crop snapshot 上的 production crop、逐层 slicing oracle 与 wrong-length control
- within-run matched-arm recovery exactness

### 5.7 环境与复算入口
- 紧凑环境表与 `REPRODUCIBILITY.md` 稳定入口
- 不同 campaign 的绝对时间不池化

### 5.8 本章小结

---

## 第六章 实验与结果分析

### 6.1 研究问题与实验设置
- RQ1（C2 核心）：外部软件进度能否解析为合法 assistant 提交边界，并在冻结协议中保持 direct crop integrity 与同一 run 内 matched-arm recovery exactness？证据为 C2 v3、A1 与 P1。
- RQ2（C2 下游支持）：software-cursor fragment retention 与 generation retention 在固定检测器条件下的信息复现率有何差异？证据为固定轨迹 E3。
- RQ3（C1 支持）：同步 oracle 接受前候选生成在不同阈值下呈现怎样的候选可用性、pooled discarded-token ratio 与 oracle latency lower bound？证据为 C-E2。
- RQ4（C1 路径审计）：System A 与 B@0.92 两条非 token-equivalent 实现路径的 candidate selection/compute readiness 有何差异？证据为 C-E1。
- RQ5（C3 探索）：三种历史自然化实现的描述性表现如何，现有受混杂轨迹允许哪些结论？证据为 A2。
- 结果按 C2→E3→C1→C3 的证据层级组织；墙钟实测、oracle 下界、微基准、构造检查和描述性探索不互相替代。
- 固定轨迹 E3 使用 100 条纯 MultiWOZ 对话；确认性 E1/E2 使用与既有 E1/E2/E3 零交集的 100 条 holdout，并在 5 个独立进程中重复。

### 6.2 C2：联合前缀状态修正的直接证据与成本边界

#### 6.2.1 C2 v3 exact gate
- accepted run `c2crop_82103004_20260903T080512Z`；24/24 cases、27/27 crop events、3 no-op、60 recovery steps、27/27 wrong-length negative controls。
- 28 层 K/V 的 pre-crop retained prefix、production post-crop 和独立于 production crop 接口、从同一 snapshot 逐层切片的 oracle 三方 bitwise exact。
- 同一 accepted run 内，两臂从精确匹配的保留状态出发并接受相同 token-ID chunks 与操作序列后逐步 exact。
- v1/v2 按冻结门槛 rejected；v3 不改变旧 verdict，也不支持 clean-reprefill、continuation 或跨环境等价性。

#### 6.2.2 固定协议成本与软件控制路径
- A1：256–8192 token，warmup=5、repeats=50，固定 32-token suffix 与 CUDA/GPU 同步。
- P1：3 长度×3 位置×20 次；`leaked_samples=0` 仅指软件计数器。
- 两者分别为模型侧微基准和 headless 软件路径，不代表声卡、声学接收、在线 TTS、真实并发或生产端到端打断。

### 6.3 固定轨迹 E3：C2 的下游检测器条件化证据
- 100 dialogues、400 个 `(dialogue, injection position)` 配对单元、800 条条件记录；四个位置为 0.25、0.50、0.75 和 fragment boundary。
- 每个 target×detector 单元的主 estimand 对符合相应非空目标资格的 `(dialogue, injection position)` 等权；点估计 label-weighted，区间按 dialogue cluster 重采样。
- Fragment/rule：−3.37 pp [−10.49, 3.40]；fragment/judge：−2.02 pp [−10.70, 6.13]；proxy/rule：−1.58 pp [−6.08, 2.67]；proxy/judge：−2.63 pp [−8.57, 2.90]。
- Dialogue-weighted 与 target-specific exact-key deduplication 为敏感性分析；fragment 297 labels→169 groups 表示减少 128 个额外 label 权重，proxy 380→379。
- 规则版与单模型 judge 均为冻结的自动操作化，不是人工 reference standard；结果不支持 superiority、equivalence、noninferiority、harm 或 absence-of-effect。

### 6.4 C-E2：候选可用性、oracle acceptance 与 discarded-token ratio
- crossed design：100 unique utterances×5 process sessions；每条件 500 observations；crossed/product bootstrap 10,000 repeats。
- B@0.92 与 B-never 的 output tokens/text 500/500 一致，支持 token-consistent B-path comparison。
- never−B@0.92 的 arrival→candidate-selection 差为 −0.03349 ms [−0.63861, 0.61494]。
- B@0.92 pooled discarded-token ratio 为 2.8527% [1.1239%, 4.7345%]；接受时候选可用率为 335/500=67.0% [58.0%, 76.0%]。
- oracle `TTFT_eff` latency lower bound 的 never−B@0.92 差为 +20.8037 ms [17.8492, 23.6450]；它对应乐观时延下界或潜在推测收益上界。
- 291 ms 只表示 candidate-first-selection→post-candidate oracle-acceptance 的同步程序内部间隔，不是自然端点 lead。

### 6.5 C-E1：非 token-equivalent implementation-path comparison
- 同一 crossed design；不得把 500 observations 当作 500 个独立内容样本。
- System A vs B@0.92 的完整输出 token 为 280/500 相同，首 token 为 465/500 相同；44/100 unique utterances 至少一次完整输出不一致。
- arrival→candidate-selection 的 A−B@0.92 差为 −34.6877 ms [−35.4421, −33.9535]；oracle latency lower bound 差为 +17.4367 ms [14.4079, 20.3234]。
- 差异混合 tokenization、forward topology/shape、role boundary、kernel 与 Python scheduling，不能归因于纯 incremental-prefill effect；主分析不按 matched outputs 做 post-treatment selection。

### 6.6 C3：探索性历史实现（A2）
- 报告描述性评分与重写耗时。
- 仅 33/100 的三策略 fragment-retention compatibility alias 相同，独立生成混杂使策略效应不可识别。
- 该运行是受混杂的探索性描述，不构成负结果、零效应或因果比较。

### 6.7 按证据层级汇总结论与适用边界

---

## 第七章 讨论

### 7.1 核心贡献及其与已有工作的关系
- C2 是唯一核心合同；高层 playback truncation 与 crop primitive 不作为原创

### 7.2 结果解释
- 7.2.1 C2：联合状态合同，不是宽泛系统质量结论
- 7.2.2 E3：C2 的下游固定检测器条件证据
- 7.2.3 C1：candidate readiness 与提交时机分离
- 7.2.4 C3：不可识别的探索性描述

### 7.3 效度威胁
- 7.3.1 构念效度：software/device/acoustic 三层、边界代理与自动检测器
- 7.3.2 内部与结论效度：crossed design、非 token-equivalent 路径、exact-key 重复、固定 A1 与小样本 P1
- 7.3.3 外部效度：单模型/语言/后端与无真实异步音频闭环

### 7.4 适用条件与后续证据
- TTS 需报告片段—音频归属；推理引擎需支持联合状态裁剪与角色恢复

### 7.5 本章小结

---

## 第八章 总结与展望

### 8.1 主要结论
- C2（核心）、E3（downstream 支持）、C1（支持）与 C3（探索）按证据成熟度总结
- C2 v3 仅支持 tested snapshot/backend 下的 direct crop integrity 与 within-run matched-arm recovery exactness；v1/v2 仍 rejected
- E3 不支持 superiority/equivalence/noninferiority/harm/absence-of-effect；C1 只支持同步 candidate/oracle/discarded-token characterization；C3 为受混杂描述

### 8.2 后续工作
- 真实异步音频与 device/acoustic boundary
- 固定轨迹 A2、盲法人工/HCI 评测
- 跨语言、模型、TTS、chat template、dtype、backend 与引擎复验
- A1 随机操作顺序与更多 crop length

### 8.3 工件与声明入口
- `REPRODUCIBILITY.md` 统一索引；作者/机构/期刊依赖声明保留在 `declarations.md`

### 8.4 结语

---

## 定稿前事项

1. 待目标期刊确定后转换参考文献和版式。
2. 完成作者、机构、许可、伦理、基金、利益冲突、CRediT、公开 artifact URL/DOI 与 AI 使用声明。
3. 从权威 Markdown 生成期刊投稿稿和中英文衍生版本，不反向覆盖分章源稿。
