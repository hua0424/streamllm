# 第二篇论文大纲

**建议题目**：级联式语音对话中软件播放游标与 TTS 片段驱动的 KV 状态修正

**English title**: Software-Playback-Cursor and TTS-Fragment–Driven KV-State Correction for Cascaded Spoken Dialogue

**论文定位**：工程与系统型硕士论文。OpenAI、Azure 和 LiveKit 已建立 playback-conditioned transcript/session-history truncation 的高层先例，KV crop 亦是既有原语；本文的核心贡献边界是公开级联栈中的 software-consumed-sample cursor→TTS fragment→assistant token span→KV crop→mask/token/position/role/EOT recovery 及其受控 exact evidence。C1 仅作 candidate compute-readiness、oracle acceptance 与 waste 的支持性刻画，C3 为受混杂的探索性负扩展。

**权威正文源**：`abstract.md`、`chapter1_introduction.md` 至 `chapter8_conclusion.md`、`references.md`。`thesis_draft.md` 由 `uv run python -m experiments.scripts.build_thesis_draft` 自动合并；中英文 IEEE 稿为后续衍生版本，不应反向覆盖分章源稿。

---

## 第一章 绪论

### 1.1 研究背景与动机
- 级联语音对话的模块化优势与低延迟需求
- 用户打断后的生成、合成、播放状态错位
- 承认 OpenAI、Azure、LiveKit 的 prior art
- 将研究范围收窄到公开级联实现中的显式 KV 状态管理

### 1.2 研究问题与挑战
- 核心问题一：software-consumed-sample cursor、音频块、TTS 文本片段和 assistant token span 的关联与反查
- 核心问题二：KV、mask、token ledger、position 和 role/EOT state 的合法恢复
- 支持性问题：candidate selection/compute-readiness、post-candidate oracle acceptance 与 wasted-token 工作点
- 探索性问题：当前受混杂运行中三种历史实现的描述性表现

### 1.3 本文工作与贡献
- C2（核心）：software cursor→TTS fragment→assistant token span→KV crop→mask/token/position/role/EOT recovery；以 v3 direct crop-integrity 与 matched-recovery exact evidence 验收，且不声称 clean-reprefill equivalence
- C1（支持）：pre-end-of-turn candidate-response generation with invalidation；区分 readiness、oracle acceptance 与同步 diagnostics，不声称 speculative decoding 或 production deliverability 改善
- C3（探索）：三种历史自然化实现及受独立生成混杂的负结果，不作策略因果比较
- 可检视研究工件；novelty 只作截至 2026-09-03 的 scoped public-source non-identification，不作 global-first 主张

### 1.4 论文组织结构

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
- 三个事件：last_segment_arrival、first_token_ready、endpoint_accept（接受不在到达瞬间）
- 主指标 $L_{\mathrm{arr}\to\mathrm{rdy}}=t_{\mathrm{rdy}}-t_{\mathrm{arr}}$（实际墙钟）；$\mathrm{TTFT}_{\mathrm{eff}}$ 为 oracle 接受下界（存活且已就绪为 0，否则端点后按需生成首 token 的时延）
- TTFT 触发口径与 mouth-to-ear 画像建模
- A1 crop-only/联合恢复与 P1 prepared-state 软件控制路径的区分
- stop 确认、stop 后同步、反查、stop→crop、stop→role 的嵌套区间定义；不相加
- 片段口径与字符比例—空白边界近似口径
- 词面检测、LLM 裁判和人工样本均为代理，不称上下界
- 推测浪费率 wasted/(wasted+final)（与确认性 campaign 的正式 estimand 相同）和离散工作点

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
- 模块、接口和验证对象表

### 5.2 片段时间轴与断句对齐
- 边界计数语义
- 非空白字符前缀和
- 部署并发目标与实验模拟的区别

### 5.3 KV 状态操作
- assistant 侧 KV 累积
- KV/掩码/账本统一不变式
- chat template 角色串推导与适配范围

### 5.4 编排与实验记录
- 两条用户输入路径、输出闭环和条件开关
- Mock TTS 是真机画像参数化近似
- 原始结果与离线重分析分离
- 确认性 E1/E2 campaign 工件：run `e1e2c_b8c758b_20260901T173306Z`、代码 `b8c758b`/结果 `62508dc`、5 独立进程×100 条零交集 holdout×10 条件（5000 条 records）、TEN replay cache 222 条目且不进计时窗口、greedy/bfloat16/sdpa、manifest/checksums（72 文件）/环境快照

### 5.5 部署、验证与可复现性
- 双 3090 模型布局
- 四个独立 campaign：旧 E1/E2/A2（探索性旧 campaign）、固定轨迹 E3/联合 A1、prepared-state P1 v2、确认性 E1/E2；绝对时间不池化，A1/P1 不相减
- P1 主机：双路 Xeon Gold 6330、112 逻辑 CPU、约 756 GiB、Ubuntu 22.04.5、driver 580.105.08、双 RTX 3090
- 组件检查与数据级检查的证据边界
- 固定轨迹 E3/联合 A1/P1 v2/确认性 E1/E2 已归档 manifest、运行身份、checksums 与原始记录；旧 E1/E2 口径 artifact 已修正并降级为审计引用，旧 campaign 仍缺部分元数据

---

## 第六章 实验与结果分析

### 6.1 研究问题与实验设置
- RQ1：software cursor→TTS fragment→assistant token state boundary；RQ2：direct KV crop 与 matched recovery correctness；RQ3：candidate-readiness/oracle/waste；RQ4：三种历史实现的描述性表现
- 结果按证据对象组织：固定轨迹 E3；C-E2 supporting characterization；C-E1 implementation-path comparison；C2 v3 exact correctness；A1/P1 latency boundary；A2 exploratory result
- 旧 E2 清除 fixture；固定轨迹 E3 独立使用 100 条纯 MultiWOZ，不与旧 E3 合并
- 确认性 E1/E2（C-E1/C-E2）为第四个独立 campaign：5 独立进程×100 条新 holdout（与旧 E1/E2/E3 零交集）、greedy、TEN 置信度离线回放；旧 E1/E2 明确标为探索性旧 campaign
- 墙钟实测、oracle 下界、微基准、画像建模和构造性结果分类
- 明确同步推测、40-token E3 上限和无真实 ASR 闭环

### 6.2 固定轨迹 E3：detector-conditioned information reproduction
- 100 dialogues、400 `(dialogue,injection_label)` pairs、800 condition records；共享首轮 trajectory/timeline/injection，40-token cap
- 主分析为 label-weighted point estimate + 同 estimand 的 dialogue-cluster bootstrap
- Fragment/rule：−3.37 pp [−10.49, 3.40]；fragment/judge：−2.02 pp [−10.70, 6.13]
- Proxy/rule：−1.58 pp [−6.08, 2.67]；proxy/judge：−2.63 pp [−8.57, 2.90]
- target-specific unique-semantic-boundary sensitivity：fragment 297 labels→169 groups，rule −2.96 pp、judge 0.00 pp；proxy 380 labels→379 groups，结果近似不变
- 所有结果不支持 superiority/equivalence/noninferiority/harm/absence-of-effect；CI 不含 detector、prompt/model 或 human-perception error
- automated proxy agreement 不称 human validation

### 6.3 C-E2：candidate-readiness、oracle acceptance 与 waste
- crossed design：100 unique utterances × 5 process sessions；每条件 500 observations；crossed/product bootstrap 10,000 repeats
- B@0.92 与 B-never 的 output tokens/text 500/500 一致，支持 token-consistent B-path comparison
- never−B@0.92 arrival→candidate-selection：−0.03349 ms [−0.63861, 0.61494]
- B@0.92 pooled waste 2.8527% [1.1239%, 4.7345%]；survival 67.0% [58.0%, 76.0%]
- oracle TTFT_eff lower bound，never−B@0.92：+20.8037 ms [17.8492, 23.6450]
- 291 ms 只称 candidate-first-selection→post-candidate oracle-acceptance 的同步程序内部间隔；不称自然端点 lead
- first-deliverable/consumer markers 只作同步 harness diagnostics，不作 production deliverability headline

### 6.4 C-E1：implementation-path comparison
- 同一 crossed design；不得把 500 observations 当作 500 个独立内容样本
- System A vs B@0.92：full output tokens 280/500、first token 465/500；44/100 unique utterances 至少一次 full-output mismatch
- arrival→candidate-selection，A−B@0.92：−34.6877 ms [−35.4421, −33.9535]
- oracle TTFT_eff lower bound，A−B@0.92：+17.4367 ms [14.4079, 20.3234]
- 差异混合 full-string vs segment-wise tokenization、forward topology/shape、role boundary、kernel 与 Python scheduling；不归因于纯 incremental-prefill effect
- 不按 280 个 matched outputs 过滤主延迟，避免 post-treatment selection

### 6.5 C2：exact correctness、KV 状态复用与软件控制路径

#### 6.5.1 C2 v3：Direct crop-integrity addendum
- accepted run `c2crop_82103004_20260903T080512Z`；24/24 cases、27/27 crop events、3 no-op、60 recovery steps、27/27 wrong-length negative controls
- 28 层 K/V：pre-crop retained prefix = production post-crop = independent slicing oracle，逐层 shape/dtype/device/hash 与 runtime `torch.equal` exact
- identical token-ID chunks 下 K/V、logits、mask、token ledger、retained prefix 与 role/end/content state exact
- 只支持 tested snapshot/backend 下 direct crop integrity 与 matched-recovery determinism；不支持 clean-reprefill numerical equivalence、32-token continuation、跨模型/backend/hardware 或 online-audio correctness
- v1/v2 clean-reprefill protocols 均 rejected；v2 42/45 numerical gates 且 control topology 不匹配；v3 不改变旧 verdict

#### 6.5.2 A1：KV 复用联合微基准
- 表 6-5：256–8192 token；warmup=5，repeats=50；设备同步的同一区间联合计时
- 联合中位数 31.054–48.315 ms，IQR 0.635–3.099 ms
- 重新预填充中位数 / 联合路径中位数 = 2.254–40.620
- 不是播放器链路

#### 6.5.3 P1：Prepared-state 软件打断控制路径
- run `sci34_dc52978_20260901_async_prepared_v2`；代码 `dc52978`；结果 `ee1dcc7`
- 3 长度 × 3 位置 × 20 = 180；120 片段内、60 边界；180/180 精确目标、零软件采样泄漏
- 播放前 setup 原始 40.499–1722.228 ms、单元中位数 41.208–1717.110 ms，明确排除
- 紧凑表 6-6：stop ack 0.055–0.062 / 最大单元 P95≈0.077；post-stop sync 0.167–0.176 / ≈0.352；lookup 0.47–0.50 / ≈0.94；stop→crop 2.44–2.53 / ≈3.492；stop→role 78.6–80.8 / ≈86.1 ms
- 两个 stop 累计区间嵌套；组件与累计端点不相加；不与 A1 池化或相减；不声称上下文无关
- 仅 headless 墙钟节拍软件播放器/模型状态；不代表声卡、声学/用户所听、在线 TTS、真实并发或生产端到端

### 6.6 C3：探索性历史实现（A2）
- 描述性评分与重写耗时
- 仅 33/100 三策略 fragment-retention compatibility alias 相同，明确独立生成混杂
- 作为探索性负结果，不作因果比较

### 6.7 按证据对象汇总结论与适用边界

---

## 第七章 讨论

### 7.1 与已有系统的关系
- 公开研究实现与既有工程实践互补
- 不猜测商业系统内部实现

### 7.2 效度威胁
- 构念效度：边界代理、词面规则、单一 judge v3、无人类双标
- 内部效度：旧 E1/E2 口径 artifact（user_end 记录在同步推测完成后，oracle 误作墙钟）已由确认性 campaign 修正；C-E1 是包含 tokenization、forward topology/shape、role boundary、kernel 与 Python scheduling 的 implementation-path comparison；同步时序、40-token cap、重复 semantic boundary 与 A2 条件轨迹不一致
- 外部效度：单模型、双 3090、英文任务型对话、四个 campaign 不池化
- 结论效度：cluster bootstrap 主、McNemar 描述；不显著不等于等效/非劣，负点估计不等于伤害；oracle 上界收益不得表述为墙钟改善；P1 v1 协议失败只作审计，v2 当前有效但限于软件控制路径

### 7.3 可推广性与适用条件
- TTS 需提供片段—音频归属
- 推理引擎需支持缓存裁剪和角色恢复
- 阈值须按新领域和在线时序重新标定

---

## 第八章 总结与展望

### 8.1 全文总结
- 按 C2（核心）、C1（支持）、C3（探索）的证据成熟度总结
- C2 v3 支持 tested snapshot/backend 下 direct crop integrity 与 matched-recovery determinism，但 v1/v2 clean-reprefill verdict 仍 rejected；A1/P1 只支持各自冻结的软件/模型协议
- E3 不支持 superiority/equivalence/noninferiority/harm/absence-of-effect；C1 只支持 candidate-readiness/oracle/waste characterization；C3 为受混杂的探索性负结果

### 8.2 后续工作
- P1 v2 已完成；后续接入真实声卡/声学停播、在线异步 TTS、真实 ASR/LLM/TTS/播放器并发与生产控制闭环
- 固定轨迹 A2 因果对照（固定 E3 已完成，不再列未来工作）
- 词级/token 级播放边界
- 随机、盲法、双标注员的人评
- 跨语种、跨模型、跨引擎复验和完整运行 manifest
- 完成实测播放位置与实时速度假设的 E4

---

## 定稿前引用与工件待办

1. 为 Predictive ASR、vLLM、SGLang 和 stream2sentence 补原始文献或官方技术文档。
2. 为“一期工作已验证的结果”提供正式论文/章节引用，而不是只引用实验设计文件。
3. 不使用未核实的 CosyVoice2 45 ms 外部数字；如讨论官方 150 ms 宣称或 L20/TensorRT-LLM 约 190–220 ms 基准，须完整注明来源和条件，且不直接代入本文系统对比。
4. 按学校要求把统一文献表转换为 GB/T 7714 等正式格式。
5. 用 `uv run python -m experiments.scripts.build_thesis_draft` 从分章 Markdown 重新生成 `thesis_draft.md`，再同步中文和英文 IEEE 衍生稿。
