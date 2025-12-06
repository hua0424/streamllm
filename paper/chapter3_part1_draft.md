# 第三章 基于流式架构的低延迟语音对话系统设计 (System Design & Methodology)

本章详细阐述“流式语音对话系统” (System B) 的架构设计与核心算法实现。针对传统级联系统 (System A) 在长语音交互中的延迟瓶颈，本设计采用了细粒度的流水线并行策略，通过自适应流式 ASR 上下文管理和 LLM KV Cache 增量预填充技术，实现了“边听边想”的即时响应能力。

## 3.1 系统总体架构设计 (System Architecture)

### 3.1.1 设计目标与逻辑拓扑
本系统的核心设计目标是将串行阻塞式处理转化为流式并行处理，以最小化首字延迟 (TTFT)。系统逻辑拓扑如图 3-1 所示。

[图 3-1: System B 流式架构这一图 (Schematic of Streaming Architecture)]
*(图注：展示从 Audio Stream 输入开始，经过 VAD 分段、ASR 识别、LLM 推理的并行流水线。图中应突出 Ring Buffer 的缓冲作用以及 ASR 与 LLM 之间的异步事件驱动机制。)*

系统由三个核心子系统组成，通过全异步 (Asynchronous) 机制解耦：

1.  **流式音频分段前端 (Streaming Audio Segmenter)**: 基于 Silero VAD 引擎，负责对原始 PCM 音频流进行实时活动检测与切片，消除静音片段，输出有效的语音片段 ($AudioChunk$)。
2.  **上下文感知 ASR 引擎 (Context-Aware ASR Engine)**: 维护一个动态音频环形缓冲区 (Ring Buffer)，采用滑动窗口机制调用 Faster-Whisper 模型，在保证识别准确率的前提下，以“小步快跑”的方式输出确定性文本片段 ($TextSegment$)。
3.  **增量式 LLM 推理服务 (Incremental LLM Inference Service)**: 基于 Transformer 的 KV Cache 机制，接收上游输出的文本片段，实时计算并更新 Attention 键值对，当检测到句意完整或收到结束信号时，立即进行 Token 生成。

### 3.1.2 关键工程实现机制
*   **非阻塞式 I/O (Non-blocking I/O)**: 系统各模块间通过线程安全的队列 (Thread-safe Queue) 通信。ASR 模块无需等待整句语音结束即可开始处理缓冲区内的累积音频。
*   **状态与无状态的权衡**: ASR 模块采用有状态设计 (Stateful) 以保留声学上下文，而 VAD 模块采用流状态对象 (`StreamState`) 跟踪静音计数器与语音起始点，确保了长语音切分的鲁棒性。

## 3.2 自适应流式 ASR 上下文管理 (Adaptive Streaming ASR)

流式 ASR 的核心挑战在于“识别准确率”与“识别延迟”的权衡。过短的音频切片缺乏声学上下文 (Acoustic Context)，导致识别错误；而等待过长则增加了系统延迟。本节提出一种基于滑动窗口的上下文管理策略。

### 3.2.1 动态 VAD 分段策略
系统集成 Silero VAD 模型进行毫秒及语音检测。定义音频流为时间序列 $X = \{x_1, x_2, \dots, x_t\}$。VAD 模块维护一个状态机 $S_{vad}$，包含当前累积音频 $A_{accum}$ 和静音计数器 $C_{silence}$。

分段判定逻辑如下：
当 $Prob(speech|x_t) < \theta_{threshold}$ 时，计数器 $C_{silence}$ 自增。一旦满足以下条件，触发分段截断 (Segment Flushing)：

$$
\text{FlushTrigger}(t) = 
\begin{cases} 
1, & \text{if } C_{silence} > T_{min\_silence} \text{ AND } Len(A_{accum}) > T_{min\_speech} \\
0, & \text{otherwise}
\end{cases}
$$

其中 $T_{min\_silence}$ (默认 300ms) 为最小静音保护阈值，防止句中停顿被错误切断。代码实现详见 `src/asr/streamaudio_segmenter.py` 中的 `process_audio` 方法。

### 3.2.2 基于滑动窗口的上下文感知 (Context-Aware Sliding Window)

为了解决流式识别中的边界截断问题，ASR 引擎引入了包含“前缀上下文” (Prefix Context) 的滑动窗口机制。

[图 3-2: ASR 滑动窗口与重叠切分示意图]
*(图注：展示时间步 t 时，输入窗口包含 [Prefix Segments | Current Segments | Suffix Buffer]。突出显示哪些部分是“Frozen” (已确定)，哪些是“Active” (待识别)。)*

#### 1. 窗口更新方程
设 $Q_t$ 为时刻 $t$ 的音频段队列 (Segment Queue)。新的音频分片 $seg_{new}$ 到达时，队列更新如下：

$$Q_{temp} = Q_{t-1} \oplus seg_{new}$$

ASR 模型$M$ 对组合后的音频进行推断，得到原始文本序列 $T_{raw}$：
$$T_{raw} = M(\text{Concat}(Q_{temp}))$$

#### 2. 确定性文本提取 (Deterministic Text Extraction)
由于 Whisper 模型是基于 Encoder-Decoder 架构，末尾的识别结果通常不稳定。我们引入稳定性约束参数 $N_{prefix}$ (前缀段数) 和 $N_{suffix}$ (后缀段数)。
输出文本 $T_{out}$ 仅来源于窗口中间的稳定区域 (Stable Region)。定义队列索引 $i$ 的取值范围：

$$I_{stable} = [N_{prefix}, \text{Length}(Q_{temp}) - N_{suffix})$$

系统仅输出索引 $i \in I_{stable}$ 对应的文本片段。

#### 3. 缓冲区状态迁移 (Buffer Transition)
在输出 $T_{out}$ 后，窗口进行滑动操作，丢弃已确定的旧数据，但保留部分末尾数据作为下一时刻的前缀上下文，以维持声学连贯性。更新后的队列 $Q_t$ 定义为：

$$Q_t = \{ seg_k \mid k \in Q_{temp} \text{ AND } k \text{ is the last } N_{prefix} \text{ segments of output} \} \cup \{ \text{Remaining Segments} \}$$

具体的实现逻辑对应 `src/asr/faster_whisper_streamer.py` 中的 `transcribe_audio_segment` 方法及其内部的 `_determine_output_segments` 函数。

通过这种 $C_t = C_{t-1}[\text{keep}:] + NewChunk$ 的滑动逻辑，系统在输出延迟仅增加 $OneChunk$ 的代价下，获得了类似整句识别的准确率。
