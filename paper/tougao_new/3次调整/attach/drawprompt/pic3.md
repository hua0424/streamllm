# 图3 绘图说明（优化定稿版）

**对应论文图题**：图3 流式并行架构逻辑拓扑图 / Fig. 3 Logical topology of the streaming parallel architecture
**对应正文**：thesis.md §3.1.1（目标与逻辑拓扑）、§3.1.2（关键工程实现）

---

## 〇、与原始描述的核对结论（先读）

原始绘图描述**与 §3.1 高度吻合**，无图意错位（区别于图2）。三大子系统、生产者-消费者队列、KV Cache 增量预填充、"预填充与用户发声重叠"的目标都与正文（line 149-159）一致。本定稿只做**保真度微调**，不改图意：

1. **去掉图内大标题**：与图1/图2 保持一致的 IEEE/Nature 惯例，正式图题由 caption 承担，图内只保留三个子系统分组标题。
2. **补齐第三个队列**：正文 line 157 明确"音频块队列、音频分段队列、文本段队列都由 queue.Queue"。原始描述只画了 Audio Segment Queue 与 Text Chunk Queue，本图补上输入端的 **Audio Chunk Queue**（避免"省略队列"）。
3. **滑动窗口精简并交叉引用图4**：Prefix / Current / Suffix 的完整机制属于 §3.2.3 / 图4。本图只在 ASR 子系统内以**紧凑三格**示意并标注 `(detail in Fig. 4)`，避免与图4 内容重复（与图2/图5 的切分同理）。
4. **线程标注按真实角色**：ASR 实为"收集器 Collector + 转录器 Transcriber"**两个子线程**（line 157），故线程泳道按角色标注，而非平铺 Thread 1/2/3。
5. **状态对象落到论文真实命名**：`StreamState` / `ASRCache` / `KVCache(past_key_values, attention_mask)`（line 159），以虚线小框轻量标注，既增严谨又不堆底层变量。

---

## 一、设计目标（瀑布/阶梯式布局 + 底部并行甘特条）

绘制一张简洁、科学严谨、适合顶刊论文（IEEE / ACM / Nature 方法示意图）风格的 draw.io 矢量逻辑拓扑图，主题为
**"流式并行架构逻辑拓扑图"**。

**布局参考草稿图 3image.png（瀑布/阶梯式）——为压缩过宽的横排、并突出"流水线重叠"：**
- **左侧固定 4 条线程泳道标签**（Thread 1~4：Audio Input / Segmenter / ASR / LLM），上下堆叠。
- **内容阶梯下移右移（waterfall）**：每条泳道主模块相对上一条向右下错位，数据沿对角线"瀑布"流下；队列在相邻泳道间纵向桥接。
- **每条泳道只保留 1 个主模块 + 1 个状态小标签**（大幅简化——作为论文说明图，不画完整子模块链）。
- **底部"Pipeline parallelism"甘特重叠条**：三条时间轴上重叠的色条（Audio+ASR / LLM Prefill / Generation）直观表达"预填充与发声重叠、End-of-Speech 后只剩末段"——本图最有说服力之处。
- 三子系统色：Segmenter 浅蓝、ASR 浅紫、LLM 浅绿；队列灰；终止橙虚线；约 1180×590（≈2:1，较前稿更紧凑）。

### 简化原则（回应"内容是否过于详细"）
- Segmenter 只画 `Silero VAD & Segmentation`（去掉 Audio Buffer / Boundary Detection 子框）。
- ASR 只画 `Streaming ASR (Whisper)` + 副标 `sliding-window context — Fig. 4`（细节交给图4）。
- LLM 保留 2 个子框：`Incremental Prefill & KV-Cache Update` 与 `Response Generation` + KV Cache 立方块 + 橙色终止箭头。
- 状态对象（StreamState / ASRCache）保留为极小虚线标签，KV Cache 保留立方块。

---

## 二、模块与数据流（从左到右）

### 输入
- `Continuous 16 kHz PCM Audio` / `Fixed-size chunks, e.g. 500 ms`
- → **Audio Chunk Queue**（queue.Queue，生产者-消费者）

### 1. Streaming Audio Segmenter（浅蓝，线程：分段）
- 内部：`Audio Buffer` → `Silero VAD` → `Speech Boundary Detection`
- 状态（虚线）：`StreamState — audio buffer · segment id · start/end time`
- 输出：`Speech Segments ⟨segment id, start/end time⟩`
- → **Audio Segment Queue**（producer–consumer）

### 2. Context-Aware ASR Engine（浅紫，线程：Collector + Transcriber 两个子线程）
- `Segment Collector`（收集器子线程，写入等待队列）
- 紧凑滑动窗口示意：`Prefix Context | Current Segment(s) | Suffix Buffer`，标注 `(detail in Fig. 4)`
- `Whisper ASR (transcribe)` → `Timestamp Alignment` → `Stable Text Commit`
- 状态（虚线）：`ASRCache — waiting queue · current window · total_duration`
- 旁注（斜体小字）：`only aligned & stable text is emitted`
- 输出：`Stable Transcript Chunks`
- → **Text Chunk Queue**（thread-safe queue）

### 3. Incremental LLM Inference Service（浅绿，线程：LLM 推理）
- `Incremental Prefill` → `KV Cache Update`
- KV Cache 状态块（立方体）：`KV Cache (VRAM) — past_key_values · attention_mask`（每个稳定片段到达即更新）
- `Response Generation`（复用末步 logits；收到终止标记后启动）
- 终止信号：橙色虚线箭头 `End Marker / End of Speech (from VAD)` → `Response Generation`
- 输出：`First Response Token` + 一排 token 方块 `Generated Response Tokens`

### 主数据流（粗箭头）
`audio chunks → speech segments → stable text chunks → KV cache update → response generation`

---

## 三、并行 / 异步与版面

- 底部浅灰时间条 + 右向箭头：`Streaming execution over time →`；上方按角色标注三条线程泳道。
- 底部一句说明（忠于 line 149-159）：
  `Producer–consumer queues decouple the modules so that ASR transcription and LLM incremental prefilling overlap with the user's ongoing speech; after End-of-Speech only the last text chunk and thread synchronization remain on the critical path.`
- 版面横向，约 1480×640（≈2.3:1，适合跨双栏 full-width 插图）。
- 配色：浅蓝 `#dae8fc/#6c8ebf`；浅紫 `#e1d5e7/#9673a6`；浅绿 `#d5e8d4/#82b366`；队列灰 `#eeeeee/#999999`；终止橙 `#ffe6cc/#d79b00`；说明黄 `#fff2cc/#d6b656`。

---

## 四、避免的错误（务必遵守）

- 不画成传统串行（"整段音频结束才启动 ASR、再启动 LLM"）——必须体现队列解耦与重叠执行。
- ASR 输出不是一次性完整 transcript——画成多个 `Stable Transcript Chunks`。
- 不省略队列——三个队列（audio chunk / audio segment / text chunk）都要画。
- KV Cache 不画成普通数据库——以立方块表示 Transformer 的 past key/value states（标注 VRAM / past_key_values / attention_mask）。
- 不引入图2 的 attention matrix 细节——本图是系统级拓扑，不画单步注意力计算。
- 不画图4 的完整滑动窗口机制——只给紧凑示意并 `(detail in Fig. 4)`。
- 不堆底层代码变量，保持方法图抽象层次。
