# 图1 绘图说明（优化定稿版）

**对应论文图题**：图1 Whisper 模型整体架构与特征降采样示意图 / Fig. 1 Overall architecture of the Whisper model and schematic of feature downsampling
**对应正文**：thesis.md §2.1.1（Whisper 模型架构与离线特性）

---

## 一、设计目标

绘制一张简洁、科学严谨、适合顶刊论文（IEEE / Nature / ACM 方法示意图）风格的 draw.io 矢量流程图，主题为
**"Whisper 模型整体架构与特征降采样示意图"**。

- 整体采用**从左到右的流水线结构**：Audio → Log-Mel → Conv Stem → Encoder → Decoder → Transcript。
- 白色背景；浅蓝、浅紫、浅绿等**低饱和学术配色**；线条清晰、字体统一、留白充足、避免复杂装饰。
- 清楚展示 Whisper 从音频输入到文本输出的主要处理路径，以及特征在**时间维度**上的 **2× 下采样**过程。
- 所有标签使用英文，表达简洁、专业。

---

## 二、相对原始描述的优化点（已纳入本定稿）

1. **去掉图内大标题**：论文已有正式图题/caption，图内不再嵌入大标题，避免与 caption 冗余（IEEE/Nature 方法图惯例）。如需，仅保留极小角注。
2. **底部说明条精简为一句**：保留一条简短说明，但不重复图题内容。
3. **新增"离线 / 定长 30 s"角注**：本节标题强调"离线特性"，正文亦指出 Whisper 默认以固定时长片段做离线处理。在 Encoder 处加一行小字 `fixed 30 s window · offline · non-streaming`，呼应正文并衔接 §2.1.2 的流式化挑战。
4. **频谱横轴标注细化**：横轴写 `Time frames (~3000)`，与下游 `3000 → 1500` 下采样箭头视觉对应。
5. **mel 维度按论文正文取 80-dim**：与 thesis.md §2.1.1 文字保持一致（备注：论文实验用 Whisper-Turbo 实际为 128 维，是否修正由作者在正文层面定夺；本图先与正文文字一致）。

---

## 三、模块与数据流（从左到右）

### 1. Input Audio（输入，浅蓝）
- 标注：`16 kHz waveform`、`30 s audio example`
- 用简洁波形/声波图标表示原始语音。

### 2. Log-Mel Feature Extraction（特征提取，浅蓝）
- 标注：`80-dim Log-Mel spectrogram`、`~3000 frames · 10 ms/frame`
- 用二维频谱矩阵图形表示：纵轴 `80 Mel bins`，横轴 `Time frames (~3000)`。

### 3. Conv Stem / Time Downsampling（卷积前端，浅紫）
- 两层一维卷积：
  - `1D Conv, kernel=3`
  - `1D Conv, kernel=3, stride=2`
- 醒目注释（下采样）：
  - `2× temporal downsampling`
  - `~3000 frames → ~1500 frames`
  - `10 ms/frame → 20 ms/frame`
- 用收缩/漏斗形箭头表现时间维被压缩。

### 4. Transformer Encoder（编码器，浅绿分组）
- 标注：`Transformer Encoder`、`full-context self-attention`
- 输入：`~1500 acoustic frames`
- 用堆叠矩形表示多层 Transformer block。
- 角注：`fixed 30 s window · offline · non-streaming`
- **禁止**标注 causal，**禁止**画成流式编码器。

### 5. Transformer Decoder（解码器，浅紫分组，与 Encoder 不同色）
- 标注：`Autoregressive Transformer Decoder`、`cross-attention to encoder states`
- 从 Encoder 输出到 Decoder 画一条清晰的 **cross-attention** 箭头。
- 下方文本 token 输入提示：`previous text tokens`
- 表示逐 token 生成。

### 6. Output Tokens（输出，浅蓝）
- 标注：`transcript tokens`、`timestamp tokens`
- 用小 token 方块表示输出序列，示例：`<|0.00|> hello world <|2.35|>`
- 强调 Whisper 可输出文本 token 与时间戳 token。

---

## 四、布局与样式要求（经典 Transformer 堆叠布局，跨栏宽图）

参考经典 *Attention Is All You Need* 风格，作为**跨双栏（full-width / figure\*）插图**，约 2:1：

- **前端水平流水线**（左侧，沿垂直居中线）：`Raw Audio (16 kHz)` → `Log-Mel (80-dim)` → `Conv Stem (CNN)`。
- 在 Conv Stem → Encoder 处用 `2× downsample` 箭头表示时间下采样；下采样注释框置于 Conv Stem 下方。
- **Encoder 与 Decoder 画成竖直堆叠的多层 Transformer block**（带 self-attention/FFN 小块 + `⋮ ×N layers`），两者**并排**，中间用**水平 Cross-Attention 箭头**连接。
  - Encoder 浅蓝堆叠：`full-context self-attention`、`~1500 acoustic frames`。
  - Decoder 浅绿堆叠：`autoregressive decoding`、`masked self-attn + cross-attn + FFN`。
- Decoder 底部接 `previous text tokens` 输入（向上箭头，体现逐 token 生成）。
- Decoder 顶部向上输出到右上角 `Output Tokens` 框：`transcript & timestamp tokens` + 示例序列 `<|0.00|> hello world <|2.35|>`。
- Encoder 下方加 `fixed 30 s window · offline · non-streaming` 角注。
- 底部一句说明条。

### 相对原 AI 生成图（1image.png）的修正
- 删除重复的第二个 "Transformer Encoder" 标签（原图错误）。
- 补回**时间戳 token**（原图只有 hello world，漏了论文强调的 timestamp token）。
- 补 `previous text tokens` 输入与 `Autoregressive` 标注（原图缺自回归输入）。
- 补 `kernel=3`、`10 ms → 20 ms/frame`、`~1500 acoustic frames` 等论文量化细节。
- 增加 offline 角注。
- 底部一句简短说明条（非图题重复）：
  `The Conv Stem reduces temporal resolution before Transformer encoding; the decoder generates text and timestamp tokens via cross-attention.`
- 风格接近 IEEE / Nature / ACM 方法示意图：干净、对齐、留白充足。

---

## 五、避免的错误（务必遵守）

- 不画 VAD、滑动窗口、KV Cache 或 LLM（不属于本图）。
- 不把 Whisper Encoder 画成 causal encoder。
- 不标注 CTC、RNN-T 或 Conformer。
- 不暗示 Whisper 原生支持实时流式识别。
- 不把 2× 下采样画成频率维度压缩——强调的是**时间维**从约 3000 帧降到约 1500 帧。
