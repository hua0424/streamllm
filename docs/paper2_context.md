# 二期论文交接文档

> 本文档用于将一期论文（StreamLLM）的工作背景、二期论文的研究方向、已经讨论清楚的设计决策、以及待澄清的工程问题，完整交接给 Claude Code 继续协作。

---

## 一、研究者背景与项目历史

研究者是硕士研究生，研究方向为**级联式语音对话系统的优化**。

**一期论文（已完成）**：使用流式并行架构改造传统串行级联架构，在用户说话过程中融入 ASR，并对 LLM 的 KV cache 进行流式 prefill 操作，从而缩短 LLM 的 TTFT 耗时。

- 论文地址：https://github.com/hua0424/streamllm/blob/whisper/paper/tougao/thesis.md
- 源代码：https://github.com/hua0424/streamllm/tree/whisper/src
- 技术栈关键点：使用 transformers 库提供的 LLM 模型，基于 transformers 的 KV cache 接口做的改造

---

## 二、二期论文方向（已确定）

### 2.1 论文定位

**稳健的工程导向论文**，定位为一期工作的延续与深化。

**核心 framing**：
> "面向用户感知一致性的级联式流式语音对话系统：基于推测生成与播放感知 KV 缓存管理的优化"

**核心原则**（贯穿全篇，作为**组织性原则**使用）：
> **对话历史 = 用户实际听到的内容**

这是符合人类对话的根本原则：LLM 在内部生成了什么、TTS 合成了什么都不算，**只有让用户听到的内容才能进入对话历史**。这个原则直接定义了 KV 缓存的去留边界。

> **⚠️ prior-art 护栏（D-006，2026-05-21 novelty 核查后加）**：这句原则**不是本论文的 insight，不能作为创新点 headline**。OpenAI Realtime API（`conversation.item.truncate`）、Azure Voice Live（`auto_truncate`，官方文档几乎逐字写过此原则）、LiveKit Agents 均已在商用系统层面实现"按播放位置截断上下文"。本论文的创新点已重新定位为**"首个开源、可复现的级联式播放感知上下文一致性管理实现 + 具体 KV 机制（`DynamicCache.crop` + role 边界重建）+ 可量化对比"**。写 intro/related work 时**必须显式引用这三个商用系统为 prior art**。详见 `docs/research_novelty_check.md` 与 `docs/decisions.md` D-006。

### 2.2 整体 Pipeline 设计

```
用户输入语音片段
  → 流式 ASR（Whisper streaming）得到文本片段
  → LLM 流式 prefill（持续累积 KV，复用一期工作）
  → 软触发判断：当前文本是否构成可回复的语义单元
      ↓ 触发
  → LLM 开始 decode 生成回复（推测性生成，可作废）
  → 文本 token 流入 stream2sentence（断句缓冲）
  → 按句子/片段 chunk 送入 CosyVoice 2 流式合成
  → 流式 TTS 输出音频 chunk + 对应文本片段索引
  → 播放器播放，同时追踪实际播放进度
      ↓ 用户打断
  → 立刻停止 TTS 播放
  → 基于"用户实际听到的片段边界"截断 LLM 的 KV cache
  → 重组 prompt 结构：保留被听到的 assistant 输出，开启新的 user role
  → 并行启动"对话历史自然化重写"（可选）
  → 累积用户新输入，重新触发推测生成
  → 循环
```

**关键设计取舍**：

- **不依赖显式的语音端点判断**：传统 VAD 端点检测被替换为"软触发推测生成"——允许多次推测与作废，端点判断不再是硬决策
- **打断不分类型**：只要 LLM 输出了、用户听到了，都作为历史的一部分。这与人类对话一致——说错了听到了也是历史
- **通过冗余计算换交互流畅度**：可以多次推测、作废，论文要量化"推测浪费率"作为核心指标之一
- **采用句子/片段级 chunking 对接 TTS**：这是业界标准做法（LiveKit、AssemblyAI 等都推荐），用 stream2sentence 实现，CosyVoice 2 做输出端流式合成。**KV 截断和"用户已听到"的判定都按片段边界对齐**，简化实现且符合人类对话感知精度

### 2.3 论文核心贡献（3 点，全部围绕"用户感知一致性"这一 framing）

#### 贡献 1：推测生成调度机制（辅助贡献）

- 使用轻量模型（如 Qwen 3 0.6B 或 Qwen 2.5 0.5B）作为软触发判断器
- 替代传统硬端点检测
- **不做模型微调，直接 prompt 现成模型即可**（保持论文聚焦）
- 在论文中作为辅助模块，不是创新主线

#### 贡献 2：播放感知的 KV 缓存管理（核心创新，占论文 60% 篇幅）

这一部分拆分为以下子问题：

**子问题 A：跨模块时间对齐**

LLM 生成进度（token 级）≠ TTS 合成进度（句子片段级）≠ 音频实际播放进度（毫秒级，含播放器 buffer）

需要建立一个**统一的进度时间轴**，能反向查询"当前实际播放到的音频时刻，对应 LLM 生成的第几个文本片段（以及该片段对应的 LLM token 范围）"。

技术要点：
- 每个送入 TTS 的文本片段记录其对应的 LLM token 范围（start_idx, end_idx）
- TTS 输出每个音频 chunk 时记录其对应的文本片段 ID
- 播放器播放每个音频 chunk 时回报"当前实际播放进度"
- 反向映射：当前播放时刻 → 当前播放片段 ID → 该片段对应的 LLM token 范围
- 处理音频 buffer 中已合成但未播放的部分（被打断时这部分要丢弃）

**子问题 B：基于实际播放位置的 KV 截断**

- 截断单位是**文本片段（sentence/fragment）**，而非单个 token
- 物理上是切到第 N 个 token（片段边界对应的 token 位置）
- 要确保切断后 KV 状态合法（attention mask、position id 等都一致）
- transformers 库的 DynamicCache 类提供 `crop()` 方法可以做这件事

**子问题 C：prompt 结构重组**

KV cache 是连续序列，并不天然知道 role 边界——role 信息在原始 prompt 文本里是通过 chat template 的特殊 token 编码的。

截断后需要：
1. 截断到 token N（用户听到的最后一个片段的末尾 token）
2. 在 KV 后**追加一个 assistant role 的结束 token**（比如 ChatML 的 `<|im_end|>`）的 KV
3. 然后开启新的 user role，prefill 用户的新输入

这一步在工程上是有坑的：追加结束 token 时，position id、attention 都要算对。**这正是论文方法节要详写的工程贡献**。

**子问题 D：推测生成被作废时的 KV 处理**

这是比"用户打断 TTS"更早期的另一种 KV 截断场景：
- 此时 TTS 可能还没开始播放，所以所有 LLM 已生成的输出都要作废
- KV 直接回滚到"用户输入结束、assistant 开始之前"的位置
- 用户新输入 append 到原 user role 上（连续累积）

#### 贡献 3：对话历史自然化重写（扩展贡献）

当 LLM 生成"今天的天气是晴天，温度 25 度，适合出门散步"，用户在播放到"温度 25 度，"这个片段时打断了。

下一轮的对话历史里，assistant 说的是"今天的天气是晴天，温度 25 度，"——这是用户实际听到的，作为对话上下文输入给 LLM 时，这是一个语义不完整的句子，可能影响 LLM 后续生成的连贯性。

**策略组合**（消融实验对比）：
- **基线（朴素截断）**：直接截断不做任何处理
- **标记法**：在 assistant message 末尾追加标记，告诉 LLM 这是被打断的（如 `interrupted: true` 字段，或省略号）。**零延迟，零额外计算**
- **重写法**：当截断位置在语义不完整处（句子中间被切断而非完整句子结束）时启用重写。**轻量模型并行处理**

**重写效率分析**：
- 重写任务输入 < 50 token、输出 < 60 token，是轻量任务
- 0.5B 级别小模型在主流 GPU 上 300-500ms 可完成
- **架构上重写在用户说话期间并行进行**，对端到端 TTFT 零影响
- 用户从打断到说完新内容通常 > 1 秒，重写延迟完全可被隐藏

---

## 三、技术栈选型（已确定）

| 组件 | 选定方案 | 备注 |
|---|---|---|
| 流式 ASR | 沿用一期的 Whisper streaming | 与一期保持一致便于对比 |
| **软触发分类器** | **TEN Turn Detection**（Qwen 0.5B 微调，Apache 2.0） | 文本侧检测，与 KV prefill 并行，零额外耗时；不做选型消融 |
| 主 LLM | 沿用一期方案（基于 transformers 库），验证 0.5B / 实验 7B | 与一期实验对齐 |
| LLM 推理框架 | transformers 库（沿用一期）| KV 操作显式断言/转换为 `DynamicCache`，用 `crop` 接口 |
| **流式断句** | **stream2sentence** | LLM token stream → 句子/片段 chunk |
| **流式 TTS** | **CosyVoice 2** | 句子级输入，输出端流式合成，首块 ~45ms |
| **重写模型** | **Qwen3-0.6B-Instruct**，直接 prompt 不微调 | 并行运行，延迟隐藏在用户说话期内 |
| 评测 benchmark | HumDial-FDBench（ICASSP 2026）+ 自构造英文打断场景集 | FD-Bench v1.5 作参考 |

**三个 LLM 实例完全独立部署，不复用权重**，模拟真实多服务工程。详见 §3.5。

### 3.1 关于 stream2sentence（核心选型，已确认）

**仓库**：https://github.com/KoljaB/stream2sentence
**安装**：`pip install stream2sentence`
**协议**：MIT

**为什么选它**：
- 作者明确说"main use case 就是 LLM → TTS"，与本论文场景完全一致
- 支持流式 generator 输入（与 LLM 的 token stream 直接对接）
- 提供 `quick_yield_single_sentence_fragment` 实现快速首句吐字（降低 TTFT）
- 支持英文（nltk）和中文（stanza, v0.2.0 已加）
- 是 RealtimeTTS / RealtimeSTT 作者写的，生产场景验证过
- MIT 协议，PyPI 发布，工程成熟

**推荐配置**（二期 pipeline 使用）：
```python
from stream2sentence import generate_sentences

for sentence_chunk in generate_sentences(
    llm_token_generator(),
    quick_yield_single_sentence_fragment=True,  # 第一片段尽快吐
    quick_yield_for_all_sentences=False,        # 后续保持完整以保 TTS 韵律
    minimum_first_fragment_length=10,           # 英文约 2-3 个词
    minimum_sentence_length=20,
    force_first_fragment_after_words=15,        # 长句兜底
    tokenizer="nltk",
    language="en",
    # 后续中文实验：tokenizer="stanza", language="zh"
):
    # 每个 sentence_chunk 同时记录其对应的 LLM token range (start_idx, end_idx)
    # 送入 CosyVoice 2 流式合成
    pass
```

### 3.2 关于 CosyVoice 2（核心选型，已确认）

**仓库**：https://github.com/FunAudioLLM/CosyVoice
**论文**：arXiv:2412.10117

**为什么选它**：
- 输出端流式成熟（chunk-aware causal flow matching）
- 首块延迟 ~45ms（A100，chunk M=5）
- 中英文都好，是中文 TTS 事实标准
- 与 stream2sentence 配合可实现"逼近输入端流式"的端到端 pipeline
- 社区大、文档全、出问题易查

**注意事项**：
- **CosyVoice 2 的"流式"是输出端流式**（已通过 GitHub Issue #1509 确认）—— 输入需要完整句子，输出是 chunked audio。这正好与 stream2sentence 的"句子级输出"匹配
- 需要 GPU（与主 LLM 可共卡也可分卡，看硬件资源）
- 与 LLM 的对接通过 stream2sentence 中转，避免 TTS 收到不完整片段

### 3.3 已经评估过、最终未选用的方案

**TADA（HumeAI/tada）**：
- 1:1 token-audio 对齐很有吸引力
- **但官方不支持流式输出**（`model.generate()` 一次性返回完整 output）
- 需要自己改 generation loop 才能流式化，工作量大且偏离论文核心
- 如有余力可作为对照组实验，证明"1:1 token 对齐能做更细的截断粒度"——但不作为主线

**Aria-TTS**：
- 声称 word-level streaming timestamps 原生支持
- 但截至 2026 年 5 月**未发布**（aria-tts.org 显示 Coming Soon，GitHub 链接为空）
- 不可作为论文主选

**edge-tts**：
- 有 word boundaries
- 但输出端流式要先给完整句子，不适合作为对接 LLM 的主选
- 可作为保底 baseline

### 3.5 软触发的两阈值机制（论文核心图来源）

软触发**不等同于传统端点检测**（YES/NO 硬决策）。它输出连续置信度，配两个阈值：

- **推测阈值**（激进）：超过即触发主 LLM decode 进入推测生成（可被作废）
- **提交阈值**（保守）：超过才允许 TTS 开始播放给用户

> **实现注记（2026-07-29）**：提交阈值在本工作的确定性模拟 harness 中**未启用**——`orchestrator.py:speculative_turn` 仅用单一推测阈值（`spec_threshold`）启动推测，推测的提交（采用）由 ASR 段流终止的真值端点触发（P1 确定性模拟），无需第二阈值门控播放。此为 harness 简化，论文稿（abstract/C1/总结）已据此对齐为"推测阈值"表述；提交阈值作为真实部署的门控设计保留于此。

调整两阈值得到 **"推测浪费率 vs TTFT" trade-off 曲线**（§五"核心 trade-off 曲线"指的就是这条）。

文本侧软触发的推理时间与一期已有的 KV prefill 阶段**并行运行**，挂在 prefill 的延迟阴影里，**实际零额外端到端耗时** —— 这是选 TEN（文本侧）而不是 Smart-Turn（音频侧 20ms）的关键架构依据。

### 3.6 硬件配置与多模型部署架构（已确定）

| 角色 | 硬件 | 用途 |
|---|---|---|
| 设计/写作主机 | 当前机 | 仅论文写作与方案讨论，不运行代码 |
| 验证主机 | 5070 Ti 16GB | 主 LLM 用 0.5B 跑通 pipeline |
| 实验主机 | 3090 24GB × 2（48GB 总） | 主 LLM 用 7B，与一期实验对齐 |

**3090×2 分卡部署（7B fp16 主 LLM）**：

| 卡 | 驻留模型 | 估算显存 |
|---|---|---|
| 卡 0 | 主 LLM(~14GB) + 长 KV(2-4GB) + Whisper-small(~1GB) | 17-19GB |
| 卡 1 | CosyVoice2-0.5B(~2-3GB) + TEN Turn Detection(~1-2GB) + Qwen3-0.6B 重写(~1-2GB) | 5-7GB |

**`src/config.py` 二期需要扩展为按模块指定 device**（main_llm / asr / tts / trigger / rewriter 各一项），一期只支持 asr_device + llm_device 两路。

### 3.4 端到端 Pipeline 模块清单

```
[Whisper streaming]  →  [一期 LLM streaming prefill]  →  [Qwen 0.5B 软触发]
       ↑                          ↓                              ↓
   语音输入                  KV cache 累积              触发 decode
                                  ↓
                          [LLM streaming decode]
                                  ↓
                          [stream2sentence]  ←—— 二期新增
                                  ↓ 句子/片段 + LLM token range
                          [CosyVoice 2 streaming]  ←—— 二期新增
                                  ↓ 音频 chunk + 片段 ID
                          [Player + 播放进度追踪]  ←—— 二期新增
                                  ↓
                          用户听到的内容（"对话历史"的真正来源）
                                  ↑
                          用户打断 → 反向查询当前播放片段
                                  → 截断 KV 到该片段对应的 token 位置
                                  → 重组 prompt（追加 <|im_end|>）
                                  → 继续累积用户新输入
```

---

## 四、领域现状参考（截至 2026 年 5 月）

**级联式架构仍是主流**：ICASSP 2026 HumDial Challenge 中大部分参赛队仍用级联或半级联架构，端到端全双工模型在复杂推理上仍不如级联。

**相关工作（论文要引用并对比的）**：
- **RelayS2S**（2026）：双路并行，speculative prefix + verifier 决定提交
- **LTS-VoiceAgent**（2026）：Listen-Think-Speak 框架，Dynamic Semantic Trigger
- **FireRedChat**（2025）：带 personalized VAD 的 barge-in 抑制
- **LLM-Enhanced Dialogue Management**（Tencent, 2025）：0.5B LLM 微调，输出 4 个控制 token 做 semantic VAD
- **Phoenix-VAD**（厦大+滴滴, 2025）：流式语义端点检测
- **FastTurn**（2025）：声学+流式语义融合的 turn detection
- **Speculative End-Turn Detector**（2025）：推测式端点检测
- **Predictive ASR**（Amazon, 2023）：用部分语音预测完整句子触发推测式 LLM 调用
- **TADA**（Hume AI, 2026）：Text-Acoustic Dual Alignment 的语音语言模型（作为相关工作引用，本项目未使用其作为 TTS）
- **CosyVoice 2**（Alibaba, 2024）：scalable streaming TTS with chunk-aware causal flow matching（本项目所用 TTS）
- **SpeakStream**（Apple, 2025）：interleaved text-speech streaming TTS（可作相关工作对比）

**Thinking Machines 的 Interaction Models**（2026 年 5 月）：
- 论文：https://thinkingmachines.ai/blog/interaction-models/
- 核心思想：micro-turn、time-aligned、连续多模态流，交互做进模型本身而非 harness
- **对二期的启示**：把级联架构的 orchestrator 从硬编码状态机升级为感知用户输出状态的策略系统，让级联架构尽可能"逼近"端到端的交互体验

---

## 五、评测设计

**指标**：

- **延迟指标**：TTFT、mouth-to-ear latency、barge-in 响应延迟
- **一致性指标**：多轮对话连贯性（人工评分 + LLM-as-judge）
- **效率指标**：推测浪费率（生成但被作废的 token 数 / 总生成 token 数）、KV 复用率
- **核心 trade-off 曲线**：通过调整软触发激进度，不同推测浪费率下能达到的 TTFT——**这条曲线是论文的核心图**

**场景**：
- 流畅说完整句子的对话（baseline 场景）
- 带思考停顿的对话
- **频繁打断的对话**（重点测，最能体现方案优势）
- 混合场景

**消融实验**：
- 三种历史处理策略对比（朴素截断 / 标记法 / 重写法）
- 不同软触发激进度的影响
- KV 复用 vs 重新 prefill 的性能对比
- 不同 stream2sentence 配置下的 TTFT 与 TTS 音质 trade-off

---

## 六、需要在 Claude Code 中确认的关键问题

> **状态（2026-05-21）**：Q1-Q8 全部已基于一期源码核实并回答完毕，结论见下方各 Q 的"**答**"块。关键技术决策已写入 `docs/decisions.md`（D-001 至 D-004）。



### Q1：一期用的 transformers KV cache 接口是哪种？
- 老版本 `Tuple[Tuple[Tensor, Tensor]]`
- 新版本 `DynamicCache` 类（有 `crop()`、`update()` 等方法）
- `StaticCache`、`SinkCache`

**关键**：二期要做 KV 截断，用 `DynamicCache.crop()` 是最直接的，需要确认一期是否已经用了这套接口

**答**：**一期对 cache 类型不可知** —— `stream_llm_inference.py:264, 304` 把 `outputs.past_key_values` 当作不透明对象直接塞进自定义 `KVCache` 数据类，从未调 cache 方法。transformers 4.36+ 默认返回 `DynamicCache`，所以 `crop()` 大概率能用。**二期决策（D-001）**：新增 KV 操作模块显式断言/转换为 `DynamicCache`，并在 `KVCache` 数据类新增 `seq_length` 字段（避免靠 `pre_attention_mask.shape[1]` 间接推断）。

### Q2：一期"流式 prefill"是怎么实现的？
- 每个片段单独调用 `model(input_ids=新片段, past_key_values=旧cache)` 让 transformers 自动 append？
- 还是手动管理 `past_key_values` 的拼接？
- 是否处理了 position_ids 的连续性？

**答**：**让 transformers 自动 append，但 attention_mask 与 position_ids 都手动显式给齐**（`stream_llm_inference.py:266-306` `_add_stream_prompt`）。`attention_mask` 用 `torch.cat` 全长拼出，`position_ids` 用 `torch.arange(past_length, past_length+current_length)` 显式算出，`past_length` 取自 `pre_attention_mask.shape[1]`。**对二期的含义**：crop 后只要同步把 `pre_attention_mask` 截短、position_ids 用新 past_length 重算 —— 一期写法可原样复用。**陷阱**：`pre_attention_mask` 是 past_length 的真相来源，crop 时必须同步修，否则 position_ids 全错。

### Q3：chat template 如何处理？
- 一次性完整 `apply_chat_template` 然后切片？
- 还是分段注入，每段加 role 标记？

**这个问题对二期极其重要**——决定截断+追加 `<|im_end|>` KV 的实现方式

**答**：**手工字符串拼接**（`stream_llm_inference.py:124-140, 159-193`）。初始化时用 dummy message 套模板取出 `generation_prompt` 字符串（即 `<|im_end|>\n<|im_start|>assistant\n` 这段），保存。首次 prefill：套模板拿到 system 前缀 + user role 开头，剔除 generation_prompt，拼上真实 prompt；流式追加：每个 final 片段当裸文本直接 prefill，无特殊 token；结束时拼上 `generation_prompt` 关闭 user 并打开 assistant。**对二期 KV 截断 + role 重建的直接含义**：一期已经掌握"裸字符串拼特殊 token"模式，二期"截到 token N → 追加 `<|im_end|>` → 打开 user role"可沿用同一模式 —— 构造 `assistant_close_then_user_open = "<|im_end|>\n<|im_start|>user\n"` 走 `_add_stream_prompt` 注入即可。该字符串需从同一 tokenizer 的 chat_template 推导（写工具函数）。

### Q4：generate（生成回复）用什么接口？
- `model.generate()` 的流式版本（streamer）→ 需要用 stopping_criteria 才能打断
- 手动写的 token-by-token 循环 → 容易加打断逻辑

**答**：**手写 token-by-token 循环**（`stream_llm_inference.py:195-245`）。打断只需消费者停止迭代 + 在 LLM worker 里抛弃 generator，无需 stopping_criteria。**但有 bug**：`generate()` 内部更新了本地 `past_key_values` 但**没有写回 `pre_cache`** —— 一期没把 assistant token 的 KV 累积下来（详见 Q5）。**二期改造**：让 `generate()` 每 yield 一个 token 时同步把 KV / attention_mask / token_id 写回 caller 可见的容器，等价于维护一个"实时增长的 assistant-side KVCache"，这正是被打断时要 crop 的对象。

### Q5：一期有没有处理多轮对话的 KV cache 累积？

**答**：**一期没做**。`generate()` 结束后 caller 拿到的 `kv_cache` 还是 prefill 结束时的快照，assistant 生成的 token 既没进 `past_key_values` 也没进 `pre_input_ids`/`pre_attention_mask`。一期只跑单轮。**对二期的含义**：这是二期必须新建的能力 —— 在 `generate()` 里维护 `assistant_kv: DynamicCache` + `assistant_token_ids` + **每个 token 与 sentence chunk 的归属映射**（反向映射表的源头）。下一轮 user 输入直接 `_add_stream_prompt` 续上，前提是 generate 末尾已盖上 `<|im_end|>\n<|im_start|>user\n` 的 KV。

### Q6：ASR 与 LLM 的衔接粒度？
- ASR 输出是 partial transcript 还是 final transcript？
- 一期"流式 prefill"基于哪种粒度触发？

**答**：**final segment 级**，不是 partial transcript。`faster_whisper_streamer.py:591-624` 只把"既不是前缀也不是后缀"的中间段输出为 final；LLM worker 每收一个 final 片段做一次 `cache_prompt(is_end=False)`，流结束才 `is_end=True` 触发 generate。**对二期的含义**：用户侧 final 粒度与 assistant 侧 stream2sentence chunk 粒度对称，反向映射表两端粒度匹配。但一期 ASR `recognition_threshold=1.0s`、`prefix_segments=1` 意味着 final 片段最小间隔 ~1 秒，可能影响软触发响应度，二期可能需调细 ASR 粒度（牺牲 WER 换响应）。

### Q7：LLM 规模与硬件？
关系到二期重写模型并行运行、以及 CosyVoice 2 共卡部署的可行性

**答**：见 §3.6。验证机 5070 Ti 16GB（主 LLM 0.5B 跑通 pipeline），实验机 3090×2=48GB（主 LLM 7B，与一期对齐），三个 LLM 实例独立部署不复用权重。详细分卡布局见 §3.6。

### Q8：一期有没有接 TTS？
如果没有，二期相当于要从零搭流式 TTS 管线（stream2sentence + CosyVoice 2 + 播放器），需估算工作量

**答**：**一期完全没有**，`src/` 下只有 `asr/` 和 `llm/`，`run_test_simple.py` LLM worker 末尾就是 `print(token)`。二期新增模块工作量估算：

| 模块 | 工作量 | 新代码量 |
|---|---|---|
| `src/tts/` CosyVoice2 流式封装 + 打断停止 | 大 | 300-500 行 |
| `src/tts/sentence_chunker.py` stream2sentence 接入 + LLM token range 标注 | 中 | 150-250 行 |
| `src/player/` 异步播放器 + 实际播放位置回报 | 中 | 200-300 行 |
| `src/dialogue/timeline.py` 反向映射表（token↔fragment↔chunk↔playback） | 大 | 300-500 行 |
| `src/llm/` 改造 KV crop + role 重建 + assistant 累积 | 中 | 100-200 行新增 + 150 行改造 |
| `src/dialogue/trigger.py` 软触发（TEN） | 小 | 100-150 行 |
| `src/dialogue/rewriter.py` 重写（Qwen3-0.6B） | 小 | 100-150 行 |
| `src/dialogue/orchestrator.py` 总编排（替代 run_test_simple） | 大 | 400-600 行 |

**总计 ~1700-2700 行新代码 + stream_llm_inference.py 中等改造**。反向映射表 + KV crop/role 重建是论文工程贡献核心。

---

## 七、下一步行动

在 Claude Code 中按以下顺序推进：

1. ✅ **代码审查**（Claude Code 读 src 目录）：回答 Q1-Q8 —— **2026-05-21 完成**
2. ✅ **架构评估**：基于一期代码现状，评估二期的代码改动范围 —— **2026-05-21 完成**（见 Q8 答）
3. **关键技术验证**（建议先做以下 mini-验证，需在 5070 Ti 验证机上跑）：
   - **stream2sentence 接入测试**：与 LLM 的流式 token 输出对接，验证句子片段输出延迟与正确性
   - **CosyVoice 2 流式合成测试**：测首块延迟、内部 buffer 大小、被打断时如何停止合成
   - **transformers DynamicCache 操作测试**：crop 截断 + 追加新 KV 的端到端可行性
   - **跨模块进度追踪**：建立"LLM token → text fragment → audio chunk → playback position"的反向映射 demo
   - **最小可运行的"打断 → 截断 KV → 重启 prefill"端到端 demo**
4. **完整工程实现**：
   - 跨模块进度时间轴（含 stream2sentence 和 CosyVoice 2 的对接层）
   - 播放感知的 KV 截断 + role 边界重建
   - 软触发推测生成集成
   - 对话历史自然化重写并行模块
5. **评测集构造与实验**

---

## 八、已经否决的方向（不要再走回头路）

- **不要加 SNN（脉冲神经网络）做端点检测**：故事不聚焦，引入非主流方向风险大于收益
- **不要做端点检测的模型创新**：用现成轻量模型即可，不微调
- **不要做打断类型分类**：所有打断都按"用户听到的就是历史"原则处理，无需分类
- **不要让推测生成完整长回复**：限制推测生成的长度（如前 N 个 token 或前一句话），降低作废成本
- **不要追求最低 TTFT 作为唯一目标**：要重新定义延迟——不只是首字延迟，而是交互自然度延迟
- **不要用 TADA 作为主选 TTS**：官方不支持流式输出，改造工作量大且偏离论文核心。可作为对照组
- **不要追求"真正的输入端流式 TTS"**：开源世界目前没有合适方案，业界标准做法就是 stream2sentence 这类句子级 chunking + 输出端流式 TTS
- **KV 截断粒度不必精确到单个 token**：以 stream2sentence 的片段边界为单位即可，这对人类对话感知精度足够，且大幅简化实现

---

## 九、当前状态时间线（里程碑日志）

| 日期 | 里程碑 | 产出 |
|---|---|---|
| 2026-05-21 | 一期代码审查完成 | Q1-Q8 全部回答，明确二期改造点与陷阱 |
| 2026-05-21 | 技术选型完整收口 | 软触发选 TEN Turn Detection、重写选 Qwen3-0.6B、KV 走 DynamicCache.crop |
| 2026-05-21 | 硬件与部署架构确定 | 验证机 5070 Ti / 实验机 3090×2，三 LLM 独立部署，分卡布局见 §3.6 |
| 2026-05-21 | 二期分支建立 | `bargeincache` 已切，与一期 main 隔离 |
| 2026-05-21 | 论文目标与定位确定 | 硕士毕业论文，**一个月内完成编写**；定位为工程/系统贡献（D-005），贡献 3 作可砍缓冲 |
| 2026-05-21 | 启动 novelty 对抗核查 | deep-research（Task wi2gfobgx）核查贡献 2「播放感知 KV 截断」是否已被发表，产出带引用报告 |
| 2026-05-21 | novelty 核查完成 | **结论 (C) 部分重叠**：概念被 OpenAI Realtime/Azure Voice Live/LiveKit 商用系统 pre-empt，但**无学术/开源级联先例**。完整报告见 `docs/research_novelty_check.md`。需按报告重新定位（工程/系统贡献，对齐 D-005） |

| 2026-05-21 | 论文重新定位确定 | **D-006 accepted**：创新点定位为"开源可复现级联实现 + KV 机制 + 可量化对比"，intro 引用商用系统为 prior art；"最强 novelty 杠杆"实验列为锦上添花不进主线 |

| 2026-05-21 | 论文大纲搭建完成 | `paper2/outline.md`（现代 AI 系统论文八章式，非一期模板）；含贡献映射、prior-art 防御结构、实验最小集分级、篇幅与写作顺序建议。**二期论文正文产物统一放 `paper2/` 目录**，与一期 `paper/` 隔离 |

| 2026-05-21 | 标题定稿 + 第二章初稿完成 | 标题：《播放感知的级联式流式语音对话上下文管理》；`paper2/chapter2_related_work.md`（含差异表，遵守 D-006 prior-art 护栏） |

| 2026-05-21 | 实验设计定稿 | `/experiment-agent` plan 模式产出 `paper2/experiment_design.md`（D-007 四项基础决策 + 被测条件 + 指标定义 + 各实验规格 + **instrumentation 埋点清单**）。埋点清单=`src/dialogue/` 编码验收标准 |

| 2026-07-01 | 反向映射表落地 + CPU 验证 | `src/dialogue/timeline.py`（`PlaybackTimeline`，D-008）+ `run_timeline_test.py` smoke **24/24 PASS**（纯 Python，本机 CPU 跑通，不受 GPU/torch 不兼容影响）。修正了片段边界的 count 语义 off-by-one |

| 2026-07-01 | torch 升级 cu128（D-009） | 本机 5070 Ti GPU **已解锁**：torch 2.8.0+cu128，sm_120 matmul 跑通；一期栈回归正常、`DynamicCache.crop` 可用、timeline 测试仍 PASS。同版本兼容 3090 |

| 2026-07-02 | LLM KV 机制落地 + GPU 验证 | `stream_llm_inference.py` 新增 `AccumKVCache` + `generate_accumulating`/`crop_to_token`/`reopen_user_role`/`open_assistant_role`/`prefill_user_text`（一期方法零改动）。`run_kvcrop_test.py` 0.5B GPU **ALL PASS**：累积→crop→role重建→续轮 KV 三长度(seq/mask/DynamicCache)全程一致，裁到 4 token 后仍连贯续生成多轮 |

| 2026-07-02 | 打断→反查→截断 端到端 demo | `src/dialogue/run_bargein_demo.py`：PlaybackTimeline + crop_to_token 拼接，0.5B GPU **ALL PASS**。可视化核心命题：生成16token、播到60%打断→只把听到的12token进KV/历史、作废4token、续轮基于"听到的历史"连贯生成。逐 token-id 校验裁剪前缀正确 |

| 2026-07-02 | stream2sentence 接入 | `src/tts/sentence_chunker.py`：句子→assistant token 区间映射（非空白字符计数对齐），英文 smoke 非空白守恒 PASS |
| 2026-07-02 | 编排闭环端到端跑通 | `src/tts/streaming_tts.py`(接口+Mock)、`src/player/`、`src/dialogue/orchestrator.py`。3 轮对话 demo ALL PASS：打断只留"听到内容"进历史、KV 三长度全程一致、多轮连贯。**论文核心贡献已是一条跑通的闭环**（真实 ASR/TTS/软触发为 swap-in） |

| 2026-07-02 | 指标埋点 + 截断模式开关 | orchestrator 加 TurnMetrics(§6 埋点：浪费率/未听却进历史/TTFT/mouth-to-ear) + truncation_mode(B-ours/B-gen/B-syn)。`run_conditions_demo` 量化 E3 差异：B-ours 历史 unheard=0、B-gen unheard>0（幻觉土壤） |

**HF_HOME 已迁至 `/workspace/hfhome`**（本验证机）；`.env` 已停止 git 跟踪，改用 `.env.example` 模板，每机自维护。

**已跑通并验证的二期核心（本机 0.5B GPU）**：反向映射表 → KV累积/crop/role重建 → 断句+token映射 → 编排闭环(Mock TTS+播放器) → 指标埋点 + B-ours/B-gen 对照。**论文贡献2主链路全部在真模型上验证通过**。
| 2026-07-02 | E3 一致性实验 harness | `src/dialogue/unheard_detector.py`（规则版未听引用检测）+ `experiments/scripts/run_exp3_consistency.py`（MultiWOZ 适配器+fixture / 打断场景 / B-ours vs B-gen / 增量保存）。fixture 18 场景验证：**B-ours 未听引用率 0.0% / B-gen 55.6%**，harness 自检 PASS。E3 决策：数据集 MultiWOZ 派生、检测器规则版先行+LLM-judge 留实验机、已 push origin/paper2 |

| 2026-07-02 | C1 软触发+推测-作废 + E2 harness | trigger.py（替身 AUC~0.80；**D-011：TEN 实测 7.6B 非 0.5B**，实验机同接口换入）+ orchestrator 推测状态机 + E2 曲线：th0.02→waste30.4%/TTFT0.5ms…th≥0.12→0%/43-75ms，拐点清晰 |
| 2026-07-02 | 贡献3 三策略 + A2 harness | rewriter.py（Qwen3-0.6B）+ history_policy(naive/mark/rewrite)；重写 mean~660ms 支撑"可隐藏"论点 |
| 2026-07-02 | A1 微基准 + E1 harness | **barge-in 响应=反查+crop 仅 0.12-0.19ms 近常数**（role 重建不在关键路径）；re-prefill 14→63ms 线性；E1: A TTFT 24.8ms vs B 0ms，建模 m2e 2289 vs 45ms |

| 2026-07-02 | 实验前代码审查 + 修复（D-012） | 两轴审查（Standards/Spec）发现 3 BUG 全修：**E3 指标框架修正**（playback 0%=构造性保证 + 新增 strict 严格 GT 列量化片段粒度误差）、E1 公平性（同 system prompt + m2e 含首片段时间）、chunker 越界钳制。§6 埋点补齐（8 时间戳/ttft_text/KV复用率/映射落盘/boundary 对照）。配置集中化（P2_* config + --model，实验机换 7B 零改码）。A1 计时改 median 抗噪。全部回归 PASS |

| 2026-07-17 | 实验机（3090×2）准备就绪推进 | 环境验证 PASS；修复 HF_ENDPOINT 从未生效 bug（import 顺序）+ .env 尾随空格；MultiWOZ 2.1 派生 100+100 条；模型下载选路实测（ModelScope ~20MB/s 最快）；**TEN 7B 标定 AUC=1.00**（完美可分），建议阈值 0.0052~0.9688+1.1；CosyVoice env/模型/裁判 Mistral 后台就绪中 |

| 2026-07-17 | **§四 实验准备全部完成，正式实验批启动** | CosyVoice2 真机验证 ✓（实测 profile：3175 samples/char@24k、首块 2434ms@3090、RTF 0.513，已回填）；四模型就位（7B/TEN/Qwen3-0.6B/Mistral 裁判）；run_all.sh 顺序跑 A1→E3→E2→E1→A2；**A1@7B 首批数据：crop 恒 ~0.3ms，re-prefill 1k→235ms/2k→459ms（8.6x→15.3x）** |

| 2026-07-17 | **E3 正式数值（7B+真实 MultiWOZ，n=412×2）** | 未听引用率 loose：**B-ours 0.0%（构造性）vs B-gen 51.0%**；strict：**51.0% vs 73.3%**（严格 GT 下仍 22pp 优势，同时量化片段粒度误差）；平均未听 token 0 vs 10.7。A1@7B 完成（crop ~0.3ms 恒定）；E2 进行中 |

| 2026-07-17 | **全部正式实验数据完成（7B+MultiWOZ+TEN）** | A1: crop 恒 0.3ms/re-prefill 至 8k→1863ms（39.7x）；E1: TTFT 27.4→0.6ms（97.9%），建模 m2e 9080→2482ms；**E2 九点曲线**（29.2%/0.5ms→拐点 0.85-0.97→0%/48.5ms）；E3: loose 0%/51%，strict 51%/73.3%，judge 交叉 0%/2.7%（规则=上界/judge=下界，κ≈0.05 系 MultiWOZ 领域词过触发，**方法学发现**）；A2: judge 连贯性 naive 3.76/rewrite 3.62/mark 3.29，重写 P90 937ms 可隐藏。人工校验样本表 37 条已生成（e3_human_validation_sample.md，待人工填写仲裁）。实验机余一未推提交（E2 加密），凭据需人工 |

| 2026-07-21 | **论文收尾：4 图 + 摘要 + 文献 + 统稿完成** | `plot_figures.py` 生成图 6-1~6-4（PDF+PNG，数据直读 JSON 并内置自检，B/W 友好）并织入 ch6；表号重排 6-1~6-4；中英摘要+关键词（`abstract.md`）；全文文献表 `references.md`（[1]-[17]，4 条 † arXiv 编号已联网复核无误）；统稿修正：A1 加速比按 JSON 改 8.6/15.3/25.2×、浪费率符号统一为 ρ（handoff 中 ρ/W 写反）、ch4 推测预算 κ→B 避免与 Cohen's κ 冲突、ch6 定义 3.4→3.3 引用勘误、12→12.1ms；E3 表 6-1 全部计数逐项对 JSON 验证通过 |
| 2026-08-31 | **论文数据审计、形式化修正与完整初稿重写（D-013）** | 不重跑 GPU、保留原始 JSON；新增 `reanalyze_paper2_results.py` 与独立 `paper2_reanalysis.json`。排除 E3 的 3 个 fixture（正式 100 对话/每条件 400）和 E2 的 12 条 fixture；当时的旧 E3 仅作为受独立生成混杂的诊断，A1 分开 crop-only 与 crop+role，A2 因独立生成混杂降格。原始进度不再跨量纲比较，片段内尾部按代码真实语义定义为字符比例—空白边界代理；assistant token 账本保持本轮相对长度。该里程碑随后由 D-014 的固定轨迹 E3 与联合 A1 正文证据升级。 |
| 2026-09-01 | **SCI3/4 补实验验收并修订权威正文（D-014）** | 固定轨迹 E3 成为 RQ1 主证据：片段 n=297 的规则/裁判为 67.0% vs 63.6%、42.8% vs 40.7%；修正 proxy n=380 为 75.3% vs 73.7%、43.9% vs 41.3%；四项 cluster CI 均跨零，点估计均与预设方向相反，不作优效/等效/非劣/伤害主张。playback 400/400 局部完整未播放文本为空单列构造检查；0.5/boundary 片段目标重复；无人工双标，judge v3 为单提示词模型代理。联合 A1（warmup5/repeats50）在 256–8192 token 的中位数 31.054–48.315 ms、IQR 0.635–3.099 ms、相对重新预填充加速 2.254–40.620×，不是完整 barge-in。新旧 campaign CPU 不同但均双 RTX3090，绝对时间不池化。P1 v1 因播放器启动前未同步 `ensure_full()`，使未完成的异步准备工作被 stop 后同步计入 joint path，故协议无效；prepared-state v2 在当时仍 pending，随后由 D-015 完成并接受。权威分章与实验图在本里程碑后继续同步，IEEE 衍生稿仍暂不修改。 |
| 2026-09-01 | **Prepared-state P1 v2 完成并接受（D-015）** | run `sci34_dc52978_20260901_async_prepared_v2` 基于 clean commit `dc52978`，结果入库 commit `ee1dcc7`。9 单元×20=180 条正式记录，120 条片段内、60 条边界；180/180 精确命中软件采样目标且零泄漏。九单元中位数范围：stop ack 0.055–0.062 ms、post-stop sync 0.167–0.176 ms、timeline lookup 0.47–0.50 ms、stop→crop 2.44–2.53 ms、stop→role 78.6–80.8 ms；最大单元 P95 分别约 0.077、0.352、0.94、3.492、86.1 ms。setup 在播放前完成并排除。P1 主机为双路 Xeon Gold 6330、约 756 GiB RAM、Ubuntu 22.04.5、driver 580.105.08、双 RTX3090；作为第三 campaign 单列，不与 A1 或旧实验池化。该结果只支持 headless 软件控制路径，不代表声卡/声学停止或生产端到端 barge-in。 |
| 2026-09-01 | **E1/E2 确认性 campaign 代码实现并与 CLI 对齐（D-016）** | `experiments/sci34_supplement/e1e2_confirmatory/` 已实现 `holdout_builder`、`trigger_cache`、`campaign`、`run_session`、`analyze`、`validate`、`smoke`，全部 argparse 入口已用 `--help` 核对。`campaign` 生成不可变 formal manifest，五个 formal session 强制共享其 SHA-256。accepted E3 排除源固定为 `results/e3/sci34_f11ccba_20260901_e3/manifest.json`。协议仍冻结新 100 条 disjoint MultiWOZ holdout、B@0.92、Qwen2-7B greedy、只读 TEN cache、5 个独立进程（session-index 0..4）与 session→dialogue bootstrap。Raw records 已新增 `last_segment_arrival_ns`、`first_token_ready_ns` 和 arrival-to-ready 差值；analyzer 以其为实际主指标。同步 oracle 的 `endpoint_accept` 不是最后段到达，`TTFT_eff` 只是时延的乐观下界（推测收益的上界）。浪费率主定义为 `wasted/(wasted+final)`。GPU 正式数据尚未执行，论文稿与旧结果保持不动。 |
| 2026-09-02 | **确认性 E1/E2 完成并接受，论文按双口径重写（D-017）** | run `e1e2c_b8c758b_20260901T173306Z`（代码 commit `b8c758b`、结果 commit `62508dc`、manifest `2f4bd76e…`）5000 条 records 全绿：设计侧独立复算与 analysis_v1.json 一致，checksums 72 文件对 git blob 全过，旧三个结果 blob 逐字节不变，holdout 与旧 E1/E2/E3 零交集。结果（n=500）：实际墙钟 arrival→ready 下 C-E1 A 27.70 vs B@0.92 62.38 ms（配对 −34.69，CI [−35.30,−34.11]，B 更慢；机制为最后段后多出的串行 assistant-role 前向）；oracle 口径 A−B +17.44（CI [16.12,18.75]）、never−B +20.80（CI [19.50,22.10]）；B@0.92 waste 2.85%、survival 67.0%、候选领先中位 291 ms；九条件 arrival→ready 平坦 ≈62 ms。旧 E1/E2（48.3→12.1、0.581 ms）定性为探索性旧 campaign 的口径 artifact。摘要与第 1/3/5/6/7/8 章已按双口径重写，图 6-2/6-3 重画（中英），表 6-3/6-4 更新为 C-E2/C-E1，A1/P1 表重编号为 6-5/6-6。设计侧验收记录见 `paper2/e1e2_confirmatory_acceptance_2026-09-02.md`。 |
| 2026-09-02 | **C2 EOT 状态修复与语义等价 campaign 冻结（D-018）** | 二审发现正常 EOS 后可能重复注入 `<|im_end|>`，且既有长度 smoke/A1/P1 不证明 crop 与 clean re-prefill 的模型语义等价。代码已将 assistant 内容 token 与结构 EOT 分离，引入完整 token ledger、`RolePhase`、显式 end reason、token-ID role transition 和 fail-closed crop/role 合同；orchestrator、未来 E1/E2 runtime、A1/P1 fixture 已适配，timeline 增加顺序负向合同。本机 CPU/fake campaign、timeline、speculative、confirmatory smoke 及真实 0.5B EOS/唯一 EOT/crop 多轮回归通过。新增 `c2_equivalence` 正式协议：24 cases、1 session、Qwen2-7B BF16，比较实际 crop/recovery 与 canonical token-ID clean re-prefill；GPU formal evidence 仍 pending。旧 E1/E2/E3/A1/P1 不重跑，论文正文待结果回传后统一修订。 |
| 2026-09-03 | **C2 v1 formal 执行并 rejected；协议 v2 发布（D-019）** | GPU 按 D-018 handoff 完整执行 run `c2eq_563dd22a_20260903T013547Z`（commit `563dd22`/结果 `1a47ac1`）：token/state 层 45/45 checkpoint 100% 等价（token IDs、KV/mask/seq/ledger、内容账本、unique EOT、role phase、top-5≥4/5），但 45/45 超绝对 logit 阈、continuation 30/45 exact、4/10 natural_eos 在 128 cap 内 run-on，判定 rejected 归档；E3 exact 输入同步抢救入库。设计侧独立审计（raw records/NPZ + 本地 0.5B 对照实验）证明失败全部源于协议门槛设计而非实现缺陷：增量 append vs 整段 prefill 的 BF16 核归约差异为环境固有噪声（同形状重复计算差为 0；纯分块 append 零 crop 代码即同量级），top-1 翻转全部为 margin ≤ BF16 ulp 的近并列互换。协议 v2 冻结：噪声对照臂（canonical 按 seam 分块增量 prefill）、2× 相对门槛、margin 感知 top-1/continuation、natural cap 256/重资格化（≥5/10 genuine）、每 checkpoint 三数组 NPZ 恒存并由 validator 独立重算。`src/` 零改动。本地 fake smoke（24 cases/45 sidecars/全 tamper）与 0.5B CUDA 真模型 dry-run 全绿（path/control 比值最差 1.08）。v2 formal 待实验机执行。 |
| 2026-09-03 | **C2 v2 pilot 暴露探针分支缺陷并修复（D-020）** | 实验机 v2 pilot（`899462c`）定位 `_termination_probe` 的 `else:` 误捕 genuine natural_eos（c2_01 实测 genuine 却被强加 max-token 断言）；按规程未启动 formal。设计侧收窄为 `elif max_tokens` 并新增 stub 化四分支路由单元回归（含 validator 交叉校验）。协议 v2 冻结内容不变。pilot 同时证实噪声对照臂在 7B 工作正常（c2_01 control 0.3125 vs path 0.289）。修复后从 §5 pilot 重跑。 |

**全部 6 个既有实验（E1/E2/E3/A1/A2/A3）的 harness 已在验证机 0.5B 跑通并自检 PASS**（A3 与 E2 共享数据）。旧 E1/E2 runner 及其结果现为探索性旧 campaign 审计；RQ2/RQ3 主证据来自 D-017 接受的确认性 campaign。完整状态表 + 实验机待办清单见 `paper2/experiment_design.md` §9'。
**实验机当前唯一无条件待办**：按 `experiments/sci34_supplement/c2_equivalence/GPU_HANDOFF.md`（协议 v2）执行 Qwen2-7B C2 正确性 formal **复跑**。v1 run `c2eq_563dd22a_20260903T013547Z` 已执行并 rejected 归档（D-019）：token/state 层 100% 等价，但 v1 绝对 logit 阈/greedy-exact 门槛对任何正确实现不可达成（BF16 增量 append 固有噪声，设计侧 0.5B 对照实证 path/control≈1），4/10 natural_eos greedy run-on 属 cap×snapshot 组合。v2 以噪声对照臂 + 2× 相对门槛 + margin 规则 + cap 256/重资格化重跑；E3 抢救已在 v1 轮完成入库（`results/e3_exact_rescue/`），本轮只读。既有主 LLM/TEN/CosyVoice/judge campaign 均已完成，不再重复执行。

**环境现状**：本机可运行 0.5B 模型回归；C2 状态机的 EOS、唯一 EOT、非法结构 crop、推测续轮和 A1/P1 fixture 已通过本地验证；协议 v2 已通过本地 fake smoke 与 0.5B CUDA 真模型 dry-run。正式证据仍须目标 Qwen2-7B snapshot、BF16/backend 和 exact clean commit。
**环境坑**：① `.env` 的 HF 路径/令牌是每机私有状态；formal 必须 `HF_TOKEN=` 并严格 offline，显式传本地模型目录；② 当前论文 review 目录若未提交会触发 clean-tree 拒绝，formal 前必须使用包含本轮代码和审稿文件状态的 clean commit；③ C2 是 correctness acceptance，不得用 fake/0.5B smoke 替代 7B formal。

**下一个里程碑（D-019 后）**：先完成并验收 Qwen2-7B C2 v2 formal；结果回传后，再统一处理二审要求的 crossed analysis、E3 weighting/sensitivity、事件命名、贡献层级和论文正文（C2 结果按 v2 主张边界写入方法/实验章）。v1 C2 归档、P1 v2、固定轨迹 E3、联合 A1 和确认性 E1/E2 均不重跑。真实异步音频闭环、固定轨迹 A2 与人工/HCI 仍属于保留较强 Q3 主张时的可选增强，不属于本轮 C2 正确性 campaign。

---

## 附录：可能用到的代码片段参考

### transformers DynamicCache 的 crop 用法（参考）

```python
from transformers import DynamicCache

# 假设 past_key_values 是 DynamicCache 实例
# 截断到第 N 个 token
past_key_values.crop(max_length=N)
```

### 软触发判断的 prompt 设计示例

```
给定当前累积的用户文本片段，判断这是否构成一个完整的、可以回复的语义单元。
只输出 "YES" 或 "NO"。

文本片段："今天天气怎么样"
判断：YES

文本片段："我想问一下"
判断：NO

文本片段："{current_text}"
判断：
```

### 自然化重写的 prompt 设计示例

```
以下是一段被用户打断、未说完的助手回复。请将它改写成一个语义自然完整的版本，
保持原意但让句子在被打断的位置自然结束。只输出改写后的内容。

原文：{truncated_assistant_text}
改写：
```

### stream2sentence 基础用法（参考官方 README）

```python
from stream2sentence import generate_sentences

def llm_token_generator():
    """LLM 的 token stream（来自一期已有的流式 decode）"""
    for token_text in your_llm.streaming_decode(...):
        yield token_text

# 配置见 3.1 节
for sentence_chunk in generate_sentences(
    llm_token_generator(),
    quick_yield_single_sentence_fragment=True,
    minimum_first_fragment_length=10,
    minimum_sentence_length=20,
    force_first_fragment_after_words=15,
    tokenizer="nltk",
    language="en",
):
    # 同时需要追踪：当前 sentence_chunk 对应的 LLM token 范围
    # 这个映射是二期"播放感知"的关键
    send_to_tts(sentence_chunk)
```

### CosyVoice 2 流式合成（参考官方）

```python
from cosyvoice.cli.cosyvoice import CosyVoice2

cosyvoice = CosyVoice2("pretrained_models/CosyVoice2-0.5B")

# 流式合成：stream=True 时返回 generator
for audio_chunk in cosyvoice.inference_zero_shot(
    sentence_chunk,
    "reference text",
    "reference_audio.wav",
    stream=True
):
    # audio_chunk 是音频片段
    # 需要同时记录其对应的源文本片段 ID 用于反向追踪
    player.play(audio_chunk)
```

### 二期论文需要在这个 pipeline 上扩展的关键工作

1. **建立完整的反向映射表**：LLM token range ↔ text fragment (stream2sentence 输出) ↔ audio chunk (CosyVoice 2 输出) ↔ playback position
2. **打断时的快速截断**：从当前播放位置反查 → 找到最后一个完整播放的片段 → 找到该片段对应的 LLM token 末位 → DynamicCache.crop()
3. **role 边界 KV 重建**：截断后追加 `<|im_end|>` token 的 KV，使 prompt 结构合法
4. **推测生成的作废与回滚**：在 TTS 还未开始播放时被打断，整段 LLM 输出作废，KV 回到 user 输入末尾
5. **对话历史自然化重写**：被截断在语义不完整处时，并行调用小模型重写
