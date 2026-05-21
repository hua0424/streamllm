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

**核心原则**（贯穿全篇）：
> **对话历史 = 用户实际听到的内容**

这是符合人类对话的根本原则：LLM 在内部生成了什么、TTS 合成了什么都不算，**只有让用户听到的内容才能进入对话历史**。这个原则直接定义了 KV 缓存的去留边界。

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
| 软触发分类器 | Qwen 3 0.6B 或 Qwen 2.5 0.5B，直接 prompt 不微调 | 仅作辅助模块 |
| 主 LLM | 沿用一期方案（基于 transformers 库） | 与一期保持一致 |
| LLM 推理框架 | transformers 库（沿用一期）| KV 操作主要用 DynamicCache 的 crop 接口 |
| **流式断句** | **stream2sentence** | LLM token stream → 句子/片段 chunk |
| **流式 TTS** | **CosyVoice 2** | 句子级输入，输出端流式合成，首块 ~45ms |
| 重写模型 | Qwen 2.5 0.5B / Qwen 3 0.6B | 直接 prompt，无需微调 |
| 评测 benchmark | HumDial-FDBench（ICASSP 2026）+ 自构造英文打断场景集 | FD-Bench v1.5 作参考 |

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

二期工程实现前必须搞清楚的问题（这些问题决定二期方案的实现路径）：

### Q1：一期用的 transformers KV cache 接口是哪种？
- 老版本 `Tuple[Tuple[Tensor, Tensor]]`
- 新版本 `DynamicCache` 类（有 `crop()`、`update()` 等方法）
- `StaticCache`、`SinkCache`

**关键**：二期要做 KV 截断，用 `DynamicCache.crop()` 是最直接的，需要确认一期是否已经用了这套接口

### Q2：一期"流式 prefill"是怎么实现的？
- 每个片段单独调用 `model(input_ids=新片段, past_key_values=旧cache)` 让 transformers 自动 append？
- 还是手动管理 `past_key_values` 的拼接？
- 是否处理了 position_ids 的连续性？

### Q3：chat template 如何处理？
- 一次性完整 `apply_chat_template` 然后切片？
- 还是分段注入，每段加 role 标记？

**这个问题对二期极其重要**——决定截断+追加 `<|im_end|>` KV 的实现方式

### Q4：generate（生成回复）用什么接口？
- `model.generate()` 的流式版本（streamer）→ 需要用 stopping_criteria 才能打断
- 手动写的 token-by-token 循环 → 容易加打断逻辑

### Q5：一期有没有处理多轮对话的 KV cache 累积？

### Q6：ASR 与 LLM 的衔接粒度？
- ASR 输出是 partial transcript 还是 final transcript？
- 一期"流式 prefill"基于哪种粒度触发？

### Q7：LLM 规模与硬件？
关系到二期重写模型并行运行、以及 CosyVoice 2 共卡部署的可行性

### Q8：一期有没有接 TTS？
如果没有，二期相当于要从零搭流式 TTS 管线（stream2sentence + CosyVoice 2 + 播放器），需估算工作量

---

## 七、下一步行动

在 Claude Code 中按以下顺序推进：

1. **代码审查**（Claude Code 读 src 目录）：回答上面 Q1-Q8
2. **架构评估**：基于一期代码现状，评估二期的代码改动范围
3. **关键技术验证**（建议先做以下 mini-验证）：
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
