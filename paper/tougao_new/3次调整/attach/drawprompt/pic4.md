# 图4 绘图说明（优化定稿版）

**对应论文图题**：图4 上下文感知 ASR 滑动窗口机制示意图 / Fig. 4 Schematic of the context-aware ASR sliding-window mechanism
**对应正文**：thesis.md §3.2.3（基于滑动窗口的上下文感知，line 171-183）

---

## 〇、与原始描述的核对结论（先读）

原始绘图描述**与 §3.2.3 高度吻合**，无图意错位（同图3、区别于图2）。三逻辑区域、临时拼接窗口的"一次性局部全上下文推理"、词级时间戳→稳定区提交、滑窗状态迁移（保留前缀 + 后缀保护）均与正文一致。本定稿只做**保真度微调**：

1. **去掉图内大标题**：与图1/2/3 一致的 IEEE/Nature 惯例，正式图题由 caption 承担，图内不嵌大标题。
2. **Timestamp Alignment 落到正文真实机制**：正文（line 183）= "利用 Whisper 词级时间戳将词按结束时间映射回每个音频段的时间区间，得到段级候选文本"。故该框标注 `map words → audio-segment time intervals`，而非泛化的"对齐"。
3. **稳定区定义精确化**：正文 = "仅输出前缀段与后缀段之间的稳定区域"。故 Stable Region Selection 标注 `commit segments between prefix & suffix only`；**前缀为已冻结/已提交的声学历史，不再重复输出**（line 173 "历史已确定的冻结区域"），图中前缀标注 `frozen / not re-emitted`。
4. **Whisper 非因果**：标注 `one full-context pass over the concatenated window`（line 181 "对临时队列所有音频段拼接后执行一次转录"），明确不是 causal streaming。
5. **状态迁移忠于 line 183**：滑窗"不清空队列，保留最后已输出段之前的 k 个段作为下一轮 prefix，并继续保留尚未输出的后缀保护段"。故 `Slide window` 回环标注 `retain boundary context · protect unstable tail`。
6. **补"前瞻代价"小注**（line 183 关键句）：稳定输出的代价是"至少等待 ≥k 个后续段到达并满足触发阈值 τ"，以小斜体注于后缀区，体现稳定性—延迟权衡。

---

## 〇点五、复杂度取舍（按草稿 4image.png 简化的定稿）

第一版把 §3.2.3 的三个步骤（窗口更新 / 确定性文本提取 / 状态迁移）全画成了多层框 + 流水线 + ghost 窗口 + 图例，**过度复杂、超出图题范围**。caption（line 173）明确图4 只画**"时间步 t 时 ASR 模型的输入构成"**——后续提取/迁移由正文散文承担。故按草稿 4image.png 收敛为**单条时间带**：仅画三区域输入窗口；timestamp alignment / 稳定区提交规则压到**底部一句注**（不画框、但不省略，满足避免错误第6条）。

## 一、设计目标（单条时间带，参考草稿 4image.png）

绘制一张简洁、科学严谨、适合顶刊论文（IEEE / ACM / Nature 方法机制图）风格的 draw.io 矢量机制图，主题为
**"上下文感知 ASR 滑动窗口机制示意图——时间步 t 的输入窗口构成"**。横向时间轴，白底，低饱和学术配色，黑白打印仍可读。约 760×320（≈2.4:1）。

**单条时间带结构（一行三区域，对齐时间轴）：**
- 顶部标题括号 `Total Input Window to Model`（虚线框罩住整带）。
- 三区域横向相邻、等高：`Prefix Context`（灰，frozen history · not re-emitted）｜`Current Segment(s)`（深蓝实心，醒目，active inference → emitted）｜`Suffix Buffer`（浅蓝虚线，future context · protects unstable tail）。区域内以浅竖线示意分段。
- `Transcription Target` 上箭头从 Current 顶部指出；`incoming audio stream` 箭头从右侧指入 Suffix。
- 底部时间轴 `Time t →`，在 Current 两端与中点标 `t_start / t_current / t_end`（虚线引下）。
- 底部一句注（承载 timestamp alignment + 稳定区提交，不画框）。

---

## 二、模块与数据流（单带三区域）

- **Prefix Context（灰）**：历史已确定的冻结区域，提供声学历史，`frozen · not re-emitted`（line 173，亦即草稿的 "Previous Inference Output"）。
- **Current Segment(s)（深蓝醒目）**：本轮待识别且输出的目标区，`active inference → emitted`。
- **Suffix Buffer（浅蓝虚线）**：未来音频缓冲，`future context · protects unstable tail`——提供未来上下文、保护闪烁尾部，不直接提交。
- **时间轴**：`Time t →`，`t_start`（Prefix/Current 边界）、`t_current`（Current 中点）、`t_end`（Current/Suffix 边界）。
- **底部注**：`Whisper transcribes the whole window in one pass; word-level timestamps map words back to segments, and only the timestamp-aligned stable region within Current Segments (between Prefix and Suffix) is emitted.`（一次全上下文推理 + 词级时间戳 + 仅提交稳定区，全部压成一句，避免画成流水线。）

---

## 三、版面与样式

- 单条时间带、单栏宽即可，约 760×320（≈2.4:1）。
- 配色：Prefix 灰 `#d6dce3/#9aa9bd`；Current 深蓝实心 `#2f6db5/#1f4e85`（白字）；Suffix 浅蓝虚线 `#dae8fc/#6c8ebf`；时间轴/标题深蓝 `#1f4e85`；分段浅竖线 `#b8c2cf / #6f9fd0 / #aac4e8`。
- 留白充足，四类要素（标题括号 / 三区域 / 上下箭头 / 时间轴）层次分明。

---

## 四、避免的错误（务必遵守）

- 不画成每个 chunk 独立识别且立即输出全部文本——画一次全上下文推理 + 仅提交稳定区。
- 后缀缓冲不被直接提交——suffix 仅提供未来上下文、保护不稳定尾部。
- Prefix 是**音频上下文片段**，不是文本 prompt。
- 不引入 LLM / KV Cache / 响应生成——本图只聚焦 ASR 侧滑动窗口。
- Whisper 不标 causal streaming——是对临时拼接窗口的局部全上下文推理。
- 不省略 timestamp alignment——它决定哪些文本可稳定提交。
