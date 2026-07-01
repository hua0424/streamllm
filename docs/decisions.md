# 二期工程决策日志

每次做技术决策时按时间倒序追加一条。每条包含：**日期 / 决策 / 背景 / 理由 / 影响 / 状态**。
状态：`proposed` / `accepted` / `superseded`（被后续决策替代时填写替代条目的日期与编号）。

---

## D-008（2026-05-21）反向映射表数据结构设计（PlaybackTimeline）

**决策**：
- 反向映射表实现为 **`PlaybackTimeline`**，落 `src/dialogue/timeline.py`。主干 = 按生成顺序排列的 **`FragmentRecord` 列表**（片段是截断单位，故以片段为主轴）。
- `FragmentRecord` 字段：`fragment_id / text / token_start,token_end / chunk_ids / sample_start,sample_end / status`（status ∈ SPECULATIVE/SYNTHESIZING/ENQUEUED/PLAYING/PLAYED/DISCARDED）。
- 反向查询：`playback_ms → samples → 二分查找命中片段 → token 边界`（sample_start 单调，O(log n)）。
- 并发：**一把锁罩整个 timeline**（操作极小，对话速率下竞争可忽略，不过早拆锁）；`played_samples` 游标原子 int 单独走。
- "已合成未播放"处理：打断时游标之后的 SYNTHESIZING/ENQUEUED 片段标 DISCARDED、token 被 crop。
- **mid-fragment 截断语义（选 A）**：打断落在片段中间时，该片段算"已听到"，截断到其 `token_end`（物理仍为片段边界）；若该片段被部分播放（partial）则置 rewrite 标记，供贡献3重写。

**背景**：handoff 方向1 的核心数据结构，KV 截断/推测浪费率/播放感知截断都依赖它。CPU + 0.5B 可验证，不需 GPU。

**理由**：片段主轴与"截断单位=片段"一致；单锁避免过早优化；选 A 与核心原则"历史=用户听到内容"及 P2"mid-fragment 触发重写"自洽。

**影响**：
- `src/dialogue/timeline.py` 实现 + `run_timeline_test.py` smoke（纯 Python，本机 CPU 可跑）
- 打断链路 `on_barge_in(playback_samples)` 返回 crop_token_end / discarded_ids / partial 标记，供 KV crop 与重写触发
- 对应 `experiment_design.md` §6 反向映射表落盘埋点

**环境备注**：本机 5070 Ti(sm_120) 当前 torch(cu121,≤sm_90) 不兼容，GPU 暂不可用（这是 handoff 所称"venv 损坏"的真因）。策略：核心 KV 逻辑先 CPU+0.5B 验证；需 CosyVoice2/全链路时再升 torch→cu128（兼容 3090）。

**状态**：accepted

---

## D-007（2026-05-21）实验设计四项基础决策（/experiment-agent plan 模式）

**决策**：
- **P1 打断产生**：确定性程序注入——"用户听到的"=注入时刻前已播放音频，ground truth 确定、可复现、无需真人/伦理审查
- **P2 打断时机**：混合——固定播放比例 25%/50%/75%（含 mid-fragment，触发重写）+ 片段边界对照（干净截断）
- **P3 一致性指标**：客观"未听到内容引用率"为主 + LLM-judge 连贯性为辅 + 人工小样本验证（Cohen's κ）
- **P4 语种/数据**：英文为主（MultiWOZ 派生 + 自构造英文打断集，对齐 HumDial-FDBench）；中文 CrossWOZ 为可砍扩展

**背景**：进入实验设计阶段，先定这四项决定后续所有实验能否测、代码要埋哪些点。完整设计见 `paper2/experiment_design.md`。

**理由**：均服务于"工程/系统贡献 + 一个月 deadline + 可复现"三重约束。程序注入避免真人成本；混合时机同时覆盖贡献2/3；客观主指标最扎实；英文为主对齐 benchmark 且省工。

**影响**：
- 确立被测系统条件 A / B-ours / B-gen / B-syn / B-noKV / B-naive|mark|rewrite
- 产出 instrumentation 埋点清单（`experiment_design.md` §6）作为 `src/dialogue/` 编码验收标准
- E4（buffer 精确映射对比）确认为锦上添花可砍（呼应 D-006）

**状态**：accepted

---

## D-006（2026-05-21）核心创新点重新定位（据 novelty 核查结论）

**决策**：贡献 2 的创新点从"提出'对话历史=用户实际听到的内容'原则"**降级/重新定位**为——

> **"首个开源、可复现的级联式播放感知上下文一致性管理实现 + 具体 KV 机制（`DynamicCache.crop` + `pre_attention_mask`/`position_ids` 同步重算 + ChatML role 边界重建）+ 可量化对比"**

三条硬约束（写作时必须遵守）：
1. **不把"历史=用户听到的内容"当作本论文 insight 来 headline**——Azure Voice Live 官方文档几乎逐字写过。它作为**组织性原则**可用，但必须**引用** OpenAI Realtime / Azure Voice Live / LiveKit 为 prior art。
2. **intro 显式引用上述商用系统先发制人**，堵审稿人。
3. **不得靠"商用系统做得粗/框架只做检测"立论**——这两条已被对抗核查 0-3 驳回（它们确实做了 played-vs-heard 历史管理）。合法差异只有：开源 vs 闭源、显式 KV crop vs 删 transcript、级联 vs 端到端、测量 buffer vs 假设实时速度。

**背景**：deep-research 核查（Task wi2gfobgx）判 (C) 部分重叠。概念被商用系统 pre-empt，但无学术/开源级联先例。完整报告 `docs/research_novelty_check.md`。

**理由**：
- 与 D-005 工程/系统框架完全一致——工程贡献不要求概念首创，要求开源+可复现+系统评测，正是商用闭源系统留下的空间
- 对硕士毕业论文门槛绰绰有余；对标邻居（RelayS2S/LTS-VoiceAgent/FireRedChat）都是 arXiv/workshop 级 preprint
- **代码工作量不变**——重新定位只改 framing 与引用，不改要实现的东西

**影响**：
- `paper2_context.md` §2.1 framing 已加 prior-art 护栏（本决策同批改）
- intro/related work 必须新增一段：商用系统现状 + 本工作与之的精确差异（用 `research_novelty_check.md` §三差异表）
- 论文可投层次：Interspeech / ICASSP / ASRU / SLT 系统方向（若投稿）

**关联决策**：**"最强 novelty 杠杆"实验**（量化 buffer 精确映射 vs 实时速度假设的 context 正确性差异）**列为"锦上添花"，不进主线**——一个月 deadline 下先保主 pipeline 完整；主体跑通且时间有余再做。需额外构造"合成速度≠实时"场景（TTS 快于播放/buffer 堆积），估 +3~5 天。

**状态**：accepted

---

## D-005（2026-05-21）论文定位：以工程/系统贡献为主骨架

**决策**：二期论文**主框架 = 系统贡献**——"在开源级联栈上实现播放感知的打断-上下文一致性管理"；"按用户实际播放位置截断 KV"作为该系统的**技术创新点/亮点**保留，但**不把论文成败押在它'全球首次'上**。贡献层级：
- **贡献 2（主，系统+技术亮点）**：播放感知 KV 缓存管理，必做
- **贡献 1（辅助）**：软触发推测生成，必做
- **贡献 3（扩展，可砍缓冲垫）**：对话历史自然化重写——时间不够时退化为"论文讨论 + 小规模验证"，甚至只保留零成本的标记法

**不采用"收窄到某个更细 novel 子点"路线**。

**背景**：论文目标 = **硕士毕业论文**，**预期一个月内完成编写**。当时 novelty 对抗核查（deep-research，Task wi2gfobgx）尚在跑，但本决策对报告结论 A/B/C 三种输出都鲁棒，故先定。

**理由**：
- 硕士学位论文评价尺度 = 工作量 + 系统完整性 + 实验充分性 + **一定的**创新性，不是顶会 novelty 门槛。~2000 行完整 pipeline + 系统实验本身即合格主体，与一期（流式架构 + KV prefill）同一评价逻辑
- "收窄"需要精密隔离实验证明某细点首创，更耗时且风险高（点被占则无退路），一个月预算承受不起
- 工程框架对 deep-research 结论鲁棒：判 A 则放大亮点，判 B/C 则作安全港，**不必等报告即可定**
- 工程框架允许干净砍范围（贡献 3 作缓冲），适配紧张时间线

**影响**：
- **实验目标简化**：从"证明全球首创"变为"在本系统上关键指标可测量改善 + 消融证明各组件有用"。核心对比实验（按播放位置截断 vs 按生成/合成位置截断，对多轮连贯性的影响）在自有系统内部自洽完成，不依赖外部 novelty
- 论文大纲与实验清单待 deep-research 报告回来后据此调整
- deadline 风险高，需以工程框架主动控范围

**状态**：accepted

---

## D-004（2026-05-21）重写模型选型

**决策**：使用 **Qwen3-0.6B** 作为对话历史自然化重写模型。

**背景**：贡献 3 的"重写法"分支在截断位置语义不完整时启用，并行运行隐藏延迟。输入 ~50 token，输出 ~60 token。

**理由**：
- Qwen3 系列 2025 年发布，比 Qwen2.5 更新，中文质量好
- 0.6B 规模在 3090 上推理 200-300ms，并行隐藏在用户说话期内
- 与主 LLM 同家族（Qwen 系），但**实例独立部署**，符合多服务工程现实
- 不在论文贡献范围内，不做模型选型消融

**影响**：
- `src/dialogue/rewriter.py` 加载 Qwen3-0.6B-Instruct 实例
- 实验机分卡布局：与软触发 + CosyVoice 共驻卡 1

**状态**：accepted

---

## D-003（2026-05-21）软触发模型选型

**决策**：使用 **TEN Turn Detection**（基于 Qwen 0.5B 微调的文本侧端点检测器，Apache 2.0）作为软触发主选；**不做候选模型消融**，软触发不是论文贡献。

**两阈值机制**：模型输出连续置信度，配两个阈值
- **推测阈值**（激进）：超过即触发主 LLM decode 进入"推测生成"
- **提交阈值**（保守）：超过才允许 TTS 开始播放给用户

调整两阈值得到"推测浪费率 vs TTFT"trade-off 曲线（论文核心图之一，paper2_context.md §五）。

**背景**：候选过 Smart-Turn v2（音频侧，~20ms）、TEN（文本侧，50-100ms）、Phoenix-VAD（权重发布不确定）、Qwen prompted（最灵活但慢）。

**理由**：
- 文本侧检测的推理时间**与 KV prefill 并行**，挂在 prefill 的延迟阴影里，**实际零额外成本** —— 这是关键架构观察
- 文本侧错误更易复查与调优（端点判断错时可以打印当前累积文本看原因）
- 中英文双优，Apache 2.0，知名度足，论文里讲故事无争议
- 软触发不是论文贡献，**不需要做模型选型消融实验**

**影响**：
- `src/dialogue/trigger.py` 加载 TEN Turn Detection 实例（卡 1）
- 软触发输入是 ASR final 片段累积文本，触发判断与 LLM `_add_stream_prompt` 并行
- 论文中作为辅助模块描述，**不展开多模型对比**

**状态**：accepted

---

## D-002（2026-05-21）硬件配置、分支、主 LLM 规模策略、模型独立部署

**决策**：
1. **二期工作分支**：`bargeincache`（已切，不污染一期 main）
2. **验证机**：5070 Ti 16GB，主 LLM 用 0.5B 跑通 pipeline
3. **实验机**：3090 24GB × 2 = 48GB，主 LLM 用 7B，与一期实验对齐
4. **三个 LLM 实例完全独立部署**（主 LLM / 软触发 / 重写），不复用权重，模拟真实多服务工程

**3090×2 部署粗算（7B fp16）**：
- 卡 0：主 LLM(~14GB) + 长 KV(2-4GB) + Whisper-small(~1GB) ≈ 17-19GB
- 卡 1：CosyVoice2-0.5B(~2-3GB) + 软触发(~1-2GB) + 重写(~1-2GB) ≈ 5-7GB

**理由**：
- 与一期实验对齐，便于直接对比一期/二期数据
- 多服务独立部署反映工程真实，论文工程价值更可信
- 验证机用 0.5B 跑通，等架构 OK 再上实验机跑 7B，节省迭代时间

**影响**：
- `src/config.py` 二期需要支持**按模块**指定 device（主 LLM、ASR、TTS、trigger、rewriter 各自一项），一期目前只分了 asr_device/llm_device 两路
- 实验脚本要支持单卡（验证）/双卡（实验）两种 device map

**状态**：accepted

---

## D-001（2026-05-21）transformers KV cache 的对象类型与改造路径

**决策**：二期 KV 截断走 `DynamicCache.crop()` 路线。一期 `StreamLLMInference.KVCache` 中的 `past_key_values` 字段保持现状（"transformers 返回什么就用什么"），但二期新增的 KV 操作模块**显式断言**它是 `DynamicCache` 实例；若 transformers 实际返回 legacy tuple，则一开始就 `DynamicCache.from_legacy_cache()` 转换。

**背景**：一期 `src/llm/stream_llm_inference.py` 把 `past_key_values` 当作不透明对象在 `_init_kv_cache` / `_add_stream_prompt` / `generate` 之间传递，从未调用 cache 方法 — 无法从代码静态判断它到底是 DynamicCache 还是 legacy tuple。

**理由**：现代 transformers（4.36+）对 Qwen2.5 默认就返回 `DynamicCache`，`crop()` 自 4.39 起稳定。显式断言/转换让 KV 操作有一个稳定的契约面，二期不再被 transformers 内部默认行为牵着走。

**影响**：
- 二期新增模块（KV 截断、role 重建）依赖 `DynamicCache` API（`crop`、`__len__`、`key_cache` / `value_cache` 访问、`update`）
- 一期的 `KVCache` 数据类需要在二期版本里多带一个字段：**当前 cache 长度**（即 `past_key_values.get_seq_length()`），避免靠 `pre_attention_mask.shape[1]` 间接推断
- 风险：若实际运行的 transformers 版本不返回 DynamicCache，需在加载阶段统一转换

**状态**：accepted

---

## D-000（模板示例）

**决策**：[一句话决定了什么]
**背景**：[当时面临的问题 / 约束]
**理由**：[为什么这么选 — 与备选方案的对比]
**影响**：[改动哪些文件、引入哪些依赖、有哪些后续工作]
**状态**：proposed / accepted / superseded by D-xxx

---

> 这条 D-000 是模板，提交真实决策时删除或保留为占位。
