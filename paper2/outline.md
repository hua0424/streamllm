# 二期论文大纲

> **论文工作标题（暂定）**：播放感知的级联式流式语音对话上下文管理
> **English (tentative)**：Playback-Aware Context Management for Cascaded Streaming Voice Dialogue

**定位**：工程/系统贡献型硕士学位论文（D-005）。创新点已按 novelty 核查重新定位（D-006）：非"提出原则"，而是"**首个开源、可复现的级联式播放感知上下文一致性管理实现 + 具体 KV 机制 + 可量化对比**"。
**关键约束**：一个月内完成编写；prior-art 护栏见 `docs/decisions.md` D-006、`docs/research_novelty_check.md`。

---

## 模板说明（为什么不用一期结构）

一期用的是传统"绪论/相关技术/系统设计/实验/总结"五章式。二期改用**现代 AI 应用-系统类论文的流行结构**，特征：

1. **Introduction 里直接列 bullet 式 contributions**，每条映射到具体章节
2. **Related Work 独立成章且前置**——因为本论文 novelty 是"部分重叠"，必须先把商用系统（OpenAI/Azure/LiveKit）作为 prior art 摆清楚，再讲自己（先发制人堵审稿人）
3. **Problem Formulation 独立成章**（formalize-before-method）——把"生成≠合成≠播放"三进度不一致问题、以及各评测指标先形式化定义，这是 AI 系统论文的"硬核"签名，也让后面方法有的放矢
4. **Design 与 Implementation 分成两章**——工程贡献型论文靠 Implementation 章体现工作量与可复现性
5. **Evaluation 章前置 metric 定义 + baseline + ablation**，核心是 trade-off 曲线
6. **Discussion 章单列**（limitations / threats to validity / 与商用系统的诚实对比）

---

## 第一章 绪论 Introduction

**1.1 研究背景与动机**
- 大模型语音交互从"指令式"到"自然对话"；级联式架构仍是主流（引 ICASSP 2026 HumDial 现状）
- 痛点重定义：不只是 TTFT，而是**打断（barge-in）后的"用户感知一致性"**——系统以为自己说了、但用户没听到，会导致后续对话错乱
- 一期已解决"TTFT 随语音长度线性增长"（简述，作为本文起点）

**1.2 问题与挑战（叙述版，正式定义留到第三章）**
- 级联栈里 LLM 生成进度 ＞ TTS 合成进度 ＞ 音频实际播放进度，三者不一致
- 打断时"哪些内容真正进入了对话历史"没有开源级联系统给出可复现答案

**1.3 本文贡献（bullet，映射章节）**
- **C1（辅助）**：推测生成调度机制——软触发推测阈值，用推测浪费率换 TTFT/流畅度 → 第四章 §4.1、第六章
- **C2（核心）**：播放感知的 KV 缓存管理——反向映射 + `DynamicCache.crop` + ChatML role 边界重建 + 推测作废回滚 → 第四章 §4.2、第五章
- **C3（扩展，可砍缓冲）**：对话历史自然化重写 → 第四章 §4.3
- **贯穿**：首个**开源、可复现**的级联式播放感知上下文一致性实现与系统性评测
- ⚠️ 明确写：本文**不主张**"历史=用户听到内容"这一原则为首创，该原则已见于商用系统（见第二章）

**1.4 论文组织结构**

---

## 第二章 相关工作 Related Work（★ 本论文的防御核心，前置）

> 写作策略：**先摆商用 prior art（承认），再摆学术空白（我的空间），最后差异表定位。** 直接取用 `docs/research_novelty_check.md`。

**2.1 商用/工程系统中的打断-上下文管理（prior art，必须诚实引用）**
- OpenAI Realtime API：`conversation.item.truncate` + `audio_end_ms`，按播放位置删除未播放 transcript
- Azure Voice Live：`auto_truncate`，官方文档"reflect what the user actually heard"
- LiveKit Agents：只提交已播放 transcript，`interrupted=True`
- **明说**：概念已被这些系统实现；它们的局限是**闭源、粗粒度（删 transcript 而非 KV 操作）、位置靠估算（实时速度假设）或客户端上报**
- ⚠️ 护栏：不得贬低成"它们没做"——它们做了，差异在开源/显式 KV/测量精度（D-006）

**2.2 学术界的流式语音对话与打断处理**
- 级联流式延迟优化：LTS-VoiceAgent、RelayS2S（speculative prefix）、Predictive ASR
- 端到端全双工：Moshi/Hibiki——帧同步使"生成=播放"隐式成立，**从架构上回避了本问题**（反证级联场景的独特性）
- barge-in 检测：FireRedChat（只在检测层暂停 TTS）
- 逐篇说清"做了什么/没做什么"——它们**都不做**播放感知 KV 截断

**2.3 LLM 推理与 KV 缓存操作**
- KV cache 原理、transformers `DynamicCache`/`crop`、prefix caching、vLLM/SGLang
- 相关但正交：IntentKV（文本 agent KV 剪枝）、Speculative Interaction Agents（工具作废）
- 结论：外部信号（播放进度）驱动的 KV crop + role 重建，无已记录开源先例

**2.4 差异对比表 + 本文定位**
- 直接放 `research_novelty_check.md` §三 的差异表（截断依据/播放感知/context 粒度/级联/开源）
- 一句话定位：本文填补"开源、可复现、级联、显式 KV 机制"这一格

---

## 第三章 问题形式化 Problem Formulation（★ formalize-before-method）

**3.1 系统与符号定义**
- 级联流水线的形式化：音频流、ASR final 片段流、LLM token 序列、TTS 音频 chunk、播放进度
- 三个进度指针的定义：生成指针 g（token 级）、合成指针 s（片段级）、播放指针 p（毫秒级）；恒有 p ≤ s ≤ g

**3.2 "用户感知一致性"的定义**
- 定义"对话历史应等于播放指针 p 对应的内容边界"
- 定义打断事件、推测作废事件两种上下文截断场景

**3.3 反向映射问题**
- 形式化：给定播放时刻 t → audio chunk → text fragment → LLM token 区间 [i, j)
- 定义"截断到片段边界"的合法性条件（KV 长度、attention mask、position id 一致）

**3.4 评测指标定义（前置，供第六章直接引用）**
- 延迟：TTFT、mouth-to-ear、barge-in 响应延迟
- 效率：**推测浪费率**（作废 token / 总生成 token）、KV 复用率
- 一致性：多轮连贯性（人工 + LLM-as-judge）
- 核心 trade-off：推测激进度 → (推测浪费率, TTFT) 曲线

---

## 第四章 方法设计 Method / System Design

**4.1 推测生成调度（C1，辅助）**
- 软触发推测阈值机制，文本侧检测与 KV prefill 并行（零额外耗时）；本工作确定性模拟中由真值端点触发推测提交，真实部署可增设提交阈值门控 TTS
- 推测生成长度上限策略（限前 N token/首句，降作废成本）
- 选型：TEN Turn Detection（不做选型消融，D-003）

**4.2 播放感知的 KV 缓存管理（C2，核心，占方法章 ~60%）**
- **4.2.1 跨模块反向映射时间轴**：token↔fragment↔chunk↔playback 四向映射的构造与维护
- **4.2.2 基于播放位置的 KV 截断**：`DynamicCache.crop(N)` + `pre_attention_mask` 同步截短 + position_ids 重算（一期 `_add_stream_prompt` 模式复用，见 `docs/paper2_context.md` Q2）
- **4.2.3 ChatML role 边界重建**：截断后注入 `<|im_end|>\n<|im_start|>user\n` 的 KV（Q3 手工字符串模式）
- **4.2.4 推测作废回滚**：TTS 未播放即被打断时整段作废，KV 回滚到 user 输入末尾
- **4.2.5 assistant 端 KV 累积**：改造一期 `generate()` 使其边生成边写回可 crop 的 KVCache（Q4/Q5）

**4.3 对话历史自然化重写（C3，扩展/可砍）**
- 三策略：朴素截断 / 标记法（零成本）/ 重写法（Qwen3-0.6B 并行，D-004）
- 时间不足时退化：只保留标记法 + 论文讨论

---

## 第五章 系统实现 Implementation（★ 工程贡献型论文的工作量体现）

**5.1 开源级联栈整体架构**
- Whisper streaming（一期）+ transformers LLM（一期）+ stream2sentence + CosyVoice 2 + 播放器
- 模块清单与新增代码分布（取 `paper2_context.md` Q8 工作量表）

**5.2 反向映射表的数据结构与并发模型**
- `src/dialogue/timeline.py` 的设计（handoff 方向1：四向映射、锁粒度、"已合成未播放" buffer 处理）
- 与 stream2sentence 对齐的 token 计数难点（不丢字符地标注 fragment→token 区间）

**5.3 KV 操作层实现**
- `DynamicCache` 契约化（D-001）、crop + role 重建的工具函数、position_ids 陷阱

**5.4 编排层与打断链路**
- `src/dialogue/orchestrator.py`：软触发→推测 decode→断句→TTS→播放→打断反查→截断的完整时序

**5.5 部署**
- 验证机（5070 Ti，0.5B）/ 实验机（3090×2，7B）分卡布局（D-002/§3.6）

---

## 第六章 实验与评测 Evaluation

> 工程框架下实验目标 = "本系统上可测量改善 + 消融证各组件有用"，**不追求全球首创**（D-005）。指标定义见第三章。

**6.1 实验设置**
- 数据集：MultiWOZ/CrossWOZ 派生 + 自构造打断场景集（含 HumDial-FDBench 参考）
- baseline：非流式级联；朴素截断；（可选）模拟商用系统的"实时速度假设"截断
- 硬件、预热、共享模型实例

**6.2 场景**：流畅完整句 / 带思考停顿 / **频繁打断（重点）** / 混合

**6.3 【必做】主结果**
- E1 端到端延迟：TTFT、mouth-to-ear、barge-in 响应延迟
- E2 **核心 trade-off 曲线**：软触发激进度 → 推测浪费率 vs TTFT（论文核心图）
- E3 一致性：播放感知截断 vs 按生成位置截断，对多轮连贯性的影响（**自有系统内部自洽，不依赖外部 novelty**）

**6.4 【必做】消融**
- A1 KV 复用 vs 重新 prefill
- A2 三种历史处理策略（朴素/标记/重写）
- A3 软触发激进度扫描

**6.5 【锦上添花，不进主线】最强 novelty 杠杆（D-006）**
- E4：buffer 精确映射 vs 实时速度假设，在 context 正确性（"我说过X"幻觉率）上的差异
- 需构造"合成速度≠实时"场景；主 pipeline 跑通且时间有余再做

---

## 第七章 讨论 Discussion

- **与商用系统的诚实对比**：承认概念重叠，量化本文在开源/显式 KV/测量精度上的具体差异
- **局限与 threats to validity**：0.5B/7B 规模差异、片段级（非 token 级）截断粒度的感知影响、软触发依赖现成模型
- **可推广性**：反向映射 + KV crop 机制对其它级联栈/其它 TTS 的迁移

---

## 第八章 总结与展望 Conclusion & Future Work

- 贡献回顾（对齐 1.3 的 C1/C2/C3）
- 展望：token 级截断粒度、buffer 精确映射的进一步实验、多说话人、与端到端模型的融合

---

## 参考文献 References
- 商用 prior art：OpenAI Realtime、Azure Voice Live、LiveKit（务必引，D-006）
- 学术：RelayS2S、LTS-VoiceAgent、FireRedChat、Moshi、CosyVoice 2、stream2sentence、Attention/KV cache 经典
- 完整源清单见 `docs/research_novelty_check.md` §附

---

## 附：写作与篇幅建议（一个月 deadline）

| 章 | 优先级 | 篇幅占比 | 说明 |
|---|---|---|---|
| 二 相关工作 | 最高 | 15% | 防御核心，材料已备（research_novelty_check.md），可最先写 |
| 三 问题形式化 | 高 | 12% | 决定后面所有严谨度 |
| 四 方法 | 高 | 20% | C2 为主 |
| 五 实现 | 高 | 18% | 工作量体现，边实现边写 |
| 六 实验 | 高 | 20% | 必做部分优先，E4 可砍 |
| 一/七/八 | 中 | 15% | 最后统稿 |

**写作顺序建议**：二（相关工作，材料现成）→ 三（形式化）→ 四/五（方法/实现，与编码并行）→ 六（实验）→ 一/七/八（统稿）。
