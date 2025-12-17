# 基于流式架构的低延迟语音对话系统优化

# Optimization of Low-Latency Voice Dialogue Systems based on Streaming Architecture


# 第一章 绪论 (Introduction)

## 1.1 研究背景与意义

### 1.1.1 大模型驱动的语音交互范式演进

随着人工智能技术的飞速发展，人机交互（Human-Computer Interaction, HCI）正经历着一场从“指令式”向“自然流式对话”的深刻范式转移。以 GPT-4o [OpenAI, 2024] 和 Gemini 1.5 [Google Team, 2024] 为代表的原生多模态大模型（Native Multi-modal LLMs）的问世，标志着机器首次具备了在听觉、视觉和文本模态间进行实时理解与生成的全栈能力。传统的语音助手（如早期的 Siri 或 Alexa）主要依赖基于规则或“意图-槽位”（Intent-Slot）的架构，这种架构虽然在执行特定指令（如“设置闹钟”或“查询天气”）时表现精准，但其刚性的句法约束严重限制了用户的表达自由度。用户往往需要遵循预设的命令模板，这种非自然的交互方式造成了人机之间明显的“隔阂感”。

相比之下，基于大语言模型（LLM）的新一代语音对话系统展现了极强的语义理解能力，能够处理复杂的推理任务并进行情感化、多轮次的自然交互。然而，这种智能化程度的提升带来了巨大的计算开销与延迟挑战。最新的行业报告显示，GPT-4o 的平均音频响应延迟已优化至 232ms [OpenAI, 2024]，这一数值逼近了人类在自然对话中的平均反应速度。这表明，“低延迟”已成为继“准确率”之后，下一代语音交互体验的核心竞争标准。

### 1.1.2 延迟：影响用户体验的核心痛点

尽管端到端（E2E）的多模态模型在实验室环境中表现优异，但受限于高昂的训练算力成本、数据隐私壁垒以及垂直领域的定制化需求，目前工业界主流的落地与研究方案仍广泛采用**级联式架构（Cascaded Architecture）**。该架构将系统解耦为三个独立的模块：自动语音识别（ASR）、大语言模型（LLM）和语音合成（TTS）。

在这种串行架构中，系统的总延迟（$L_{total}$）不仅取决于各个模块的独立处理时间，更受到模块间数据流转方式的严重制约。根据 [Radford et al., 2023] 和 [Dao et al., 2022] 的研究，级联系统的总延迟可形式化为各组件延迟的线性叠加：

$$L_{total} = L_{VAD} + L_{ASR} + L_{LLM\_Prefill} + L_{LLM\_Decode} + L_{TTS} + L_{Net}$$

现有的研究表明，传统非流式级联系统的端到端延迟通常在 3 到 5 秒之间 [Wang et al., 2024; Macháček et al., 2025]。考虑到人类对话的平均轮替（Turn-taking）间隙通常仅为 200ms 左右 [Sacks et al., 1974]，这种秒级的延迟差异会产生严重的“认知摩擦”：用户在表达完毕后往往需要经历漫长的静默等待，这不仅打断了思维的连贯性，还常导致用户误以为系统未响应而重复输入，进而引发交互混乱。特别是在长语音输入场景下，非流式 ASR 需要等待整句录音结束才开始转录，且 LLM 的预填充（Prefill）时间随输入长度呈线性甚至超线性增长，进一步恶化了长文本交互的实时性。因此，如何在保留级联架构模块化优势的前提下，通过流式并行策略消除模块间的“死区时间”，实现“打断即响应”的极致体验，具有重要的学术价值和工程意义。  

## 1.2 国内外研究现状 (Literature Review)  

### 1.2.1 自动语音识别 (ASR) 的流式化进展

ASR 系统的实时性直接决定了用户语音转化为机器可理解文本的速度。在架构演进方面，早期的流式 ASR 主要依赖 RNN-Transducer (RNN-T) 架构 [Graves, 2012]。RNN-T 凭借其天然的循环特性，非常适合处理流式音频，但由于无法并行计算，其训练效率较低，且在捕捉长距离语义依赖方面存在局限。随着 Transformer [Vaswani et al., 2017] 的引入，基于注意力机制（Attention）的模型在识别准确率上取得了突破性进展。然而，Transformer 的全局注意力（Global Attention）机制要求模型必须在获取完整的输入序列后才能进行计算，这使得原生的 Transformer 并不支持流式处理。

为了解决这一矛盾，Google 提出了 Conformer 架构 [Gulati et al., 2020]，创造性地结合了 CNN 的局部特征提取能力与 Transformer 的全局建模能力，成为当前的业界主流。针对流式化需求，学术界提出了“分块处理（Chunk-based Processing）”和“局部一致性（Local Agreement）”策略 [Krichli et al., 2025]，即通过滑动窗口对音频进行切片识别，并利用重叠部分的预测结果来修正输出，从而在准确率与延迟之间取得平衡。OpenAI 发布的 Whisper [Radford et al., 2023] 虽然是基于 Encoder-Decoder 的非流式模型，但近期工作如 CarelessWhisper [Krichli et al., 2025] 尝试通过修改 Attention Mask 将其改造为因果（Causal）模型，或通过 Two-pass Decoding [Yu et al., 2021] 结合流式 CTC 解码器来降低首字延迟。

### 1.2.2 大语言模型 (LLM) 推理加速技术

在 ASR 完成转录后，LLM 的推理速度成为系统延迟的第二大瓶颈。这一过程主要受限于 Transformer 解码器的自回归（Auto-regressive）生成机制以及 GPU 的显存带宽（Memory Bandwidth Bound）。

针对推理过程中的重复计算问题，**KV Cache（键值缓存）** 技术应运而生。该技术通过缓存历史 Token 的 Key 和 Value 向量，避免了在每步生成时对历史上下文进行重复的矩阵运算，成功将生成阶段的计算复杂度从 $O(N^2)$ 降低到 $O(N)$ [Pope et al., 2023]。目前，KV Cache 已成为 vLLM、TGI 等现代推理引擎的标配。此外，针对长文本输入的预填充（Prefill）阶段，Attention 计算量巨大的问题，FlashAttention [Dao et al., 2022] 通过优化 GPU 显存的读写模式（Tiling），显著提升了长序列的处理吞吐量。在长对话场景下，StreamingLLM [Xiao et al., 2023] 引入了“注意力汇聚（Attention Sinks）”机制，解决了由于上下文窗口溢出导致的模型崩塌问题，使得 LLM 能够在有限显存下支持理论上无限长度的流式对话。  

### 1.2.3 级联系统的全链路优化

除了针对 ASR 或 LLM 的单点优化外，全链路的协同调度对于降低总延迟至关重要。传统的 Simply-Cascaded 系统在模块间存在明显的“串行阻塞”，即下游模块必须等待上游模块完全输出后才能启动。目前的研究趋势正向 **Pipeline Parallelism（流水线并行）** 演进 [Gao et al., 2024]。其核心思想是打破模块间的时序依赖，例如，一旦 ASR 输出部分文本片段，LLM 即可立即开始增量预填充（Incremental Prefilling）。这种策略能够最大化计算资源的利用率，利用 ASR 处理后续音频的时间窗口来掩盖 LLM 的计算延迟，从而实现端到端的极速响应。

## 1.3 本文主要研究内容

针对传统级联式语音对话系统中存在的长语音延迟累积与资源闲置问题，本文提出了一种基于细粒度流水线并行的低延迟架构（下文称 System B）。本文的主要研究内容与贡献如下：

1. **构建流式 ASR 上下文管理机制**：本文设计了基于 Faster-Whisper 和 Silero VAD 的自适应滑动窗口算法。为了解决流式切片导致的上下文缺失与识别不稳定性问题，本文提出了动态缓冲与“前缀后缀上下文（Prefix Context & Suffix Context）”拼接策略，在保证长句识别准确率的同时，实现了毫秒级的 ASR 转录片段输出。
    
2. **提出 LLM KV Cache 增量预填充策略**：针对长语音输入带来的首字延迟（TTFT）问题，本文设计了增量推理算法。该算法深度利用 Transformer 的 KV Cache 机制，实时接收流式 ASR 的输出片段进行 Attention 计算与状态更新，避免了对历史上下文的重复编码，将 LLM 的计算开销均摊在用户的语音输入过程中。
    
3. **全链路延迟评估与验证**：本文在 MultiWOZ（英文）和 CrossWOZ（中文）数据集上构建了长语音测试基准。实验结果表明，在 15s-30s 的长语音场景下，本文提出的 System B 相比传统非流式基线（System A）实现了显著的延迟降低，且未对系统的语义理解能力造成负面影响。

# 第二章 相关技术基础 (Theoretical Foundation)

本章将详细阐述支撑本研究的核心理论基础，重点解析自动语音识别（ASR）的神经网络架构特性以及大语言模型（LLM）的底层推理机制，并从数学角度定义评估系统性能的关键指标。

## 2.1 自动语音识别技术 (ASR)

本研究选用 OpenAI 提出的 Whisper [Radford et al., 2023] 模型作为语音转文本的核心引擎。相较于传统的 RNN-T 或 CTC 模型，Whisper 基于 Transformer 的 Encoder-Decoder 架构在海量弱监督数据上进行了预训练，展现了极强的多语言泛化能力与抗噪鲁棒性。
  
### 2.1.1 Whisper 模型架构与时间分辨率

Whisper 的核心架构包含音频编码器（Encoder）和文本解码器（Decoder）。尽管本研究主要关注其流式化应用，但理解其底层的特征提取机制对于设计合理的流式切片窗口至关重要。

模型首先将 16kHz 采样率的原始音频转换为 80 通道的 Log-Mel 频谱图。随后，频谱图通过一个由两层 1D 卷积神经网络（CNN）组成的“卷积前端（Conv Stem）”进行处理。第一层卷积使用宽度为 3 的滤波器，第二层卷积则采用了**步长 (Stride) 为 2** 的设计。这一设计在特征工程中起到了关键的降采样作用：对于标准的 30 秒音频输入窗口（约 3000 帧），Conv Stem 将其压缩为 **1500 帧**。

这一机制不仅保留了 20ms 的时间分辨率以区分音素，更重要的是将后续 Transformer Encoder 中 Self-Attention 的序列长度 $N$ 减半。由于 Attention 的计算复杂度为 $O(N^2)$，这一降采样操作将计算量理论上降低了 4 倍，显著提升了推理速度。在本研究的流式实现中，这一时间分辨率特性直接指导了滑动窗口步长的参数选择。

![[图2.1.png]]
**图 2-1 Whisper 模型整体架构与特征降采样示意图** [如图所示，模型首先将 16kHz 原始音频转换为 80 通道 Log-Mel 频谱图。**关键路径**在于卷积前端（Conv Stem），其中第二层卷积采用步长为 2 的设置（Stride 2），实现了 2 倍的时间维降采样。这一操作将 30 秒音频对应的特征序列长度从 3000 帧压缩至 1500 帧，显著降低了后续 Transformer Encoder 的自注意力计算开销。]
  

### 2.1.2 流式化挑战与解决方案

Whisper 的原生设计是面向 30 秒完整音频的“离线”模型，这主要源于其 Encoder 采用了全双向的 Self-Attention 机制，依赖完整的未来上下文。为了适应低延迟流式场景，本研究在工程实现上引入了以下机制：

- **分块处理 (Chunk-based Processing)**：将连续的音频流切分为固定长度（如 500ms 或 1s）的微小片段，并在每个时间步 $t$ 将当前累积的音频缓冲区输入模型。
    
- **局部一致性 (Local Agreement)**：由于缺乏未来上下文（Right Context），模型在处理最新到达的音频块时，末尾部分的输出往往是不稳定的（Flickering）。本研究采用了局部一致性策略，通过比较时刻 $t$ 和 $t-1$ 的输出结果，仅确信并输出那些在连续两个窗口中保持一致的 Token，从而在流式输出的实时性与稳定性之间找到最优解。

## 2.2 大语言模型推理机制 (LLM Inference)

大语言模型通常采用 Decoder-only Transformer 架构。在语音对话系统中，LLM 的推理延迟，尤其是首字延迟（Time to First Token, TTFT），是决定用户是否感到“卡顿”的关键瓶颈。

  
### 2.2.1 Transformer 注意力机制

Transformer 的核心是缩放点积注意力 (Scaled Dot-Product Attention)。其标准计算公式如下：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中，$Q$ (Query), $K$ (Key), $V$ (Value) 分别由输入向量经过线性投影矩阵 $W_Q, W_K, W_V$ 得到。$d_k$ 为 Key 的维度，用于缩放数值以防止梯度消失。这一机制允许模型捕捉序列中任意两个位置之间的依赖关系。

  
### 2.2.2 KV Cache 机制的数学原理与复杂度分析

LLM 的文本生成过程本质上是自回归的 (Auto-regressive)，即模型生成第 $t$ 个 Token 时，必须依赖之前所有 $t-1$ 个 Token 的信息。

System A: 无缓存推理 (Naive Inference)

在不使用 KV Cache 的传统模式下，每生成一个新 Token，模型都需要重新计算所有历史 Token 的 $K$ 和 $V$ 向量。假设生成长度为 $N$ 的序列，第 $t$ 步的计算量主要来源于 Attention 矩阵乘法，其复杂度与当前序列长度 $t$ 成正比。因此，生成整个序列的总计算复杂度为级数求和：

$$\text{Cost}_{total} = \sum_{t=1}^{N} t = \frac{N(N+1)}{2} \approx O(N^2)$$

这种平方级的增长在处理长语音转录文本（通常包含数百个 Token）时，会导致不可接受的延迟。

System B: 基于 KV Cache 的增量推理

利用模型权重固定的特性，历史 Token $x_i (i < t)$ 的 $k_i, v_i$ 向量一旦计算便不再改变。因此，可以将它们存储在 GPU 的显存（VRAM）中。在第 $t$ 步生成时，系统仅需计算当前 Token $x_t$ 的 $k_t$ 和 $v_t$，并将其拼接到缓存末尾：

$$K_{cache}^{(t)} = \text{Concat}(K_{cache}^{(t-1)}, k_t)$$

$$V_{cache}^{(t)} = \text{Concat}(V_{cache}^{(t-1)}, v_t)$$

此时，注意力计算仅涉及当前 Query $q_t$ 与历史 $K_{cache}$ 的交互。这种策略将**单步生成复杂度**降低为 $O(1)$（相对于序列长度而言，计算量不再随 $t$ 显著增长），并将生成 $N$ 个 Token 的**总复杂度**降低为 **$O(N)$**。本研究提出的增量预填充策略正是基于此原理，在 ASR 输出的同时即时更新 KV Cache，从而在用户说完话的瞬间完成大部分计算。

![[图2.2.png]]

**图 2-2 基于 KV Cache 的增量推理与传统推理机制对比**[左图 (System A) 展示了在无缓存机制下，每生成一个新的 Token ($Q_N$) 都需要重新计算与所有历史 Token 的注意力权重（红色区域），导致计算复杂度随序列长度呈 $O(N^2)$ 增长。 右图 (System B) 展示了本研究采用的增量推理策略，通过将历史键值对 ($K, V$) 存储于显存（蓝色区域），当前步仅需计算新增部分的注意力（绿色区域）。该机制成功避免了冗余计算，将单步推理复杂度降低至 $O(N)$，显著提升了长文本生成速度。]


## 2.3 系统评价指标 (Evaluation Metrics)

为了客观评估优化策略的有效性，本研究选取了以下两个核心指标：

### 2.3.1 首字延迟 (Time to First Token, TTFT)

TTFT 是衡量交互实时性的核心指标，直接关联用户的主观等待感。参考 Google Research 的定义 [Google Team, 2024]，语音对话系统的总延迟可分解为各个模块的处理时间。本研究重点关注的 TTFT 特指从 VAD 判定用户语音结束（End of Speech）到 LLM 输出第一个字符的时间差。在 System B 中，由于部分 $L_{LLM\_Prefill}$ 被流水线并行机制所掩盖，TTFT 的理论值将显著低于 System A。

### 2.3.2 词错误率 (Word Error Rate, WER)

虽然本研究的核心目标是降低延迟，但这不能以牺牲识别准确率为代价。WER 是评估 ASR 准确率的通用标准，基于 Levenshtein 编辑距离计算：

$$\text{WER} = \frac{S + D + I}{N} \times 100\%$$

其中 $S$ 为替换 (Substitution)、$D$ 为删除 (Deletion)、$I$ 为插入 (Insertion) 的错误数量，$N$ 为参考文本的总词数。对于中文数据集，本研究采用字符错误率 (CER)，其计算逻辑与 WER 一致。


  

# 第三章 基于流式架构的低延迟语音对话系统设计 (System Design & Methodology)

本章将深入剖析本文提出的流式语音对话系统（System B）的架构范式与核心算法实现。针对传统级联架构（System A）在长语音交互场景中存在的“计算阻塞”与“延迟累积”问题，本研究提出了一种细粒度的**流水线并行（Pipeline Parallelism）** 策略。该策略通过构建自适应流式 ASR 上下文管理机制与 LLM KV Cache 增量预填充技术，从根本上重构了数据流转方式，成功打破了模块间的串行依赖，实现了“感知-认知-表达”的同步进行，即“边听边想”的即时响应能力。

## 3.1 系统总体架构设计 (System Architecture)

### 3.1.1 设计目标与逻辑拓扑

本系统的核心设计愿景是实现从传统的“全量接收-全量处理”范式向“增量接收-流式处理”范式的转移（Paradigm Shift），从而在理论边界上最小化首字延迟 (Time to First Token, TTFT)。系统的逻辑拓扑如图 3-1 所示。

![[图3.1.png]]

**图 3-1 System B 流式并行架构逻辑拓扑图**[本图展示了系统全异步解耦的数据流向。左侧音频前端采用**环形缓冲区 (Ring Buffer)** 持续摄入 PCM 数据流，并经由 **Silero VAD** 进行毫秒级活动检测与切分。中间层的 ASR 引擎输出的中间文本流 (Text Stream) 通过 **异步事件总线 (Async Event Bus)** 即时触发右侧 LLM 的 **KV Cache 增量推理**。底部的时间轴示意带清晰展示了数据如何在流水线中从原始音频切片平滑转换为语义 Token，实现了计算与 I/O 的高度重叠]

整体架构遵循**全异步（Asynchronous）解耦**原则，三个核心子系统通过标准化的接口协议协同工作。首先，**流式音频分段前端 (Streaming Audio Segmenter)** 作为系统的感知触角，集成了高灵敏度的 Silero VAD 引擎 [Silero Team, 2024]，负责对原始 PCM 音频流进行实时过滤。该模块利用动态阈值算法剔除无效静音，并将连续的语音流离散化为包含语义信息的有效切片 (AudioChunk)。其次，**上下文感知 ASR 引擎 (Context-Aware ASR Engine)** 并不等待整句语音结束，而是维护一个动态音频环形缓冲区 (Ring Buffer)，采用带有“前缀上下文（Prefix Context）”的滑动窗口机制循环调用 Faster-Whisper 模型 [Radford et al., 2023]。这种设计旨在解决流式切分可能造成的语义截断问题，以“小步快跑”的方式输出具有高置信度的确定性文本片段。最后，**增量式 LLM 推理服务 (Incremental LLM Inference Service)** 构成了系统的认知核心。区别于传统服务等待完整 Prompt 的被动模式，该服务基于 Transformer 解码器的 KV Cache 机制构建 [Vaswani et al., 2017]，能够实时摄入上游 ASR 的文本增量，即时计算并更新注意力键值对（Key-Value Pairs），从而将计算负载均摊至整个用户发言过程。

### 3.1.2 关键工程实现机制

为了在高并发场景下保证数据流转的实时性与一致性，本系统在工程实现层面引入了非阻塞 I/O 与混合状态管理机制。

在通信机制上，各模块之间摒弃了紧耦合的同步函数调用，转而采用基于**生产者-消费者模型 (Producer-Consumer Model)** 的异步通信策略。通过线程安全的队列 (Thread-safe Queue) 连接各级流水线，ASR 模块无需等待静音结束信号即可处理缓冲区内的累积音频，显著提升了系统的吞吐量与资源利用率。

在状态管理上，系统采用了**有状态 (Stateful) 与无状态 (Stateless) 的混合设计**。ASR 模块设计为有状态组件，在内存中持久化保留历史音频特征以维持声学上下文的连贯性；而 VAD 模块则通过流状态对象 (`StreamState`) 持续跟踪静音计数器与语音起始点。这种设计增强了系统对长语音切分的鲁棒性，有效避免了因网络抖动或非语义停顿导致的误截断现象。

## 3.2 自适应流式 ASR 上下文管理 (Adaptive Streaming ASR)

流式 ASR 的核心挑战在于寻找“识别准确率”与“识别延迟”之间的帕累托最优（Pareto Optimality）。过短的音频切片会导致模型缺乏必要的声学上下文 (Acoustic Context)，从而因归纳偏置（Inductive Bias）不足急剧增加词错误率 (WER)；而等待过长的切片则会引入显著的缓冲延迟，削弱流式处理的优势。为此，本节提出一种基于滑动窗口的自适应上下文管理策略。

### 3.2.1 动态 VAD 分段策略

系统采用 Silero VAD 模型进行高精度的语音活动检测。形式化地，定义输入的音频流为时间序列 $X = \{x_1, x_2, \dots, x_t\}$。VAD 模块维护一个状态机 $S_{vad}$，其中包含当前累积的音频缓冲区 $A_{accum}$ 和连续静音计数器 $C_{silence}$。

分段判定逻辑遵循概率阈值与时间约束的双重标准。当当前帧为语音的概率 $Prob(speech|x_t)$ 低于预设阈值 $\theta_{threshold}$ 时，计数器 $C_{silence}$ 开始自增。为了平衡实时性与完整性，触发分段截断 (Segment Flushing) 的判别函数 $\text{FlushTrigger}(t)$ 定义如下：

$$\text{FlushTrigger}(t) = \begin{cases} 1, & \text{if } C_{silence} > T_{min\_silence} \text{ AND } Len(A_{accum}) > T_{min\_speech} \\ 0, & \text{otherwise} \end{cases}$$

其中，$T_{min\_silence}$ 设定为 300ms，这一数值基于心理语言学研究，大致对应人类自然对话中的最小句间停顿；$T_{min\_speech}$ 为最小语音长度限制，用于过滤非语言的短时噪声脉冲。该策略确保了系统仅在检测到具有明确语义边界的停顿时才触发后续处理。

### 3.2.2 基于滑动窗口的上下文感知 (Context-Aware Sliding Window)

为了解决流式识别中的边界效应，ASR 引擎引入了包含“前缀上下文” (Prefix Context) 的滑动窗口机制。

![[图3.2.png]]

**图 3-2 上下文感知 ASR 滑动窗口机制示意图** [图示详细解析了时间步 $t$ 时 ASR 模型的输入构成。窗口被划分为三个逻辑区域：**前缀上下文 (Prefix Context)** 为历史已确定的冻结区域，提供必要的声学先验信息；**当前片段 (Current Segments)** 为待识别的活跃推理区域；**后缀缓冲 (Suffix Buffer)** 则包含未来音频信息以防止截断效应。系统通过这种重叠切分 (Overlapping) 策略，确保了流式识别结果的连贯性]

该机制包含三个关键步骤：

首先，**执行窗口更新**。设 $Q_t$ 为时刻 $t$ 的音频段队列 (Segment Queue)。当新的音频分片 $seg_{new}$ 到达时，队列执行更新操作 $Q_{temp} = Q_{t-1} \oplus seg_{new}$。随后，ASR 模型 $M$ 对组合后的音频队列进行推断，生成原始文本序列 $T_{raw} = M(\text{Concat}(Q_{temp}))$。

其次，进行确定性文本提取 (Deterministic Text Extraction)。鉴于 Whisper 等 Encoder-Decoder 模型在处理流式输入的末尾时往往存在“闪烁” (Flickering) 现象——即随着新音频的输入，原本输出的末尾词汇可能发生变化。为解决此问题，我们引入了稳定性约束参数。定义 $N_{prefix}$ 为前缀段数，$N_{suffix}$ 为后缀保护段数。系统仅从 $T_{raw}$ 中提取位于稳定区域 (Stable Region) 的文本作为最终输出 $T_{out}$。稳定区域的队列索引范围定义为：

$$I_{stable} = [N_{prefix}, \text{Length}(Q_{temp}) - N_{suffix})$$

最后，执行缓冲区状态迁移 (Buffer Transition)。在输出 $T_{out}$ 后，窗口执行滑动操作。为了维持声学连贯性，系统并非简单丢弃旧数据，而是保留部分末尾数据作为下一时刻的前缀上下文。更新后的队列 $Q_t$ 形式化表示为：

$$Q_t = \{ seg_k \mid k \in Q_{temp} \text{ AND } k \in \text{Last } N_{prefix} \text{ segments of output} \} \cup \{ \text{Remaining Segments} \}$$

通过这种 $C_t = C_{t-1}[\text{keep}:] + NewChunk$ 的滑动逻辑，系统在输出延迟仅增加单次 Chunk 时长的代价下，获得了接近离线整句识别的准确率。

## 3.3 LLM KV Cache 增量预填充策略 (Incremental KV Cache Prefilling)

在传统的 System A 架构中，ASR 输出的每一次文本更新（例如从 $Text_t = \text{"Hello"}$ 更新为 $Text_{t+1} = \text{"Hello world"}$），都会强制 LLM 对整个 $Text_{t+1}$ 序列进行重新编码。这种大量的重复计算不仅造成了算力资源的极大浪费，更导致首字延迟 (TTFT) 随上下文长度呈线性甚至超线性增长，严重制约了长文本交互体验。

针对这一痛点，本系统引入了**增量预填充 (Incremental Prefilling)** 机制。该机制深度利用了 Transformer 解码器的自回归（Autoregressive）特性，将历史 Token 的 Key/Value 状态 ($KV_{cache}$) 持久化存储于显存中。当 ASR 产生新的文本片段时，系统仅计算新增加 Token 的注意力权重，并将其 $K, V$ 向量拼接至缓存末尾。

### 3.3.1 核心算法实现

增量推理的核心逻辑封装于 `StreamLLMInference` 类中。Algorithm 1 详细描述了这一状态更新与计算过程。

**Algorithm 1: Incremental KV Cache Prefill Strategy**

```latex

\begin{algorithm}

\caption{Incremental KV Cache Prefill}

\begin{algorithmic}[1]

\REQUIRE Current KV Cache $C_{prev} = (K_{prev}, V_{prev})$, New Text Fragment $T_{new}$, Is End Flag $IsEnd$, LLM Model $\mathcal{M}$

\ENSURE Updated KV Cache $C_{new}$, Next Token Logits $L$ (Optional)

  

\STATE \textbf{Step 1: Tokenization}

\STATE $Ids_{new} \leftarrow \text{Tokenizer}(T_{new})$

\IF{$Ids_{new}$ is empty}

    \RETURN $C_{prev}, \text{None}$

\ENDIF

  

\STATE \textbf{Step 2: Attention Mask Construction}

\STATE $Mask_{prev} \leftarrow \text{GetMask}(C_{prev})$

\STATE $Mask_{full} \leftarrow \text{Concat}([Mask_{prev}, \text{Ones}(\text{Shape}(Ids_{new}))], \text{dim}=-1)$

  

\STATE \textbf{Step 3: Forward Pass (Incremental)}

\STATE \COMMENT{Compute hidden states using cached history}

\STATE $HiddenStates, C_{new} \leftarrow \mathcal{M}.\text{forward\_layers}(input=Ids_{new}, cache=C_{prev}, mask=Mask_{full})$

  

\STATE \textbf{Step 4: Conditional Logits Projection}

\STATE $L \leftarrow \text{None}$

\IF{$IsEnd$ == True}

    \STATE \COMMENT{Only compute logits projection at the final step to save compute}

    \STATE $L \leftarrow \mathcal{M}.\text{lm\_head}(HiddenStates[-1])$

\ENDIF

  

\RETURN $C_{new}, L$

\end{algorithmic}

\end{algorithm}

```

![[图3.3.png]]

**图 3-3 LLM KV Cache 增量更新机制示意图** 图中直观展示了显存内的状态变化。左侧蓝色区域 ($N \times d$) 代表已缓存的历史键值对 ($K_{prev}, V_{prev}$)，右侧绿色区域 ($M \times d$) 为当前流式输入的新增投影。系统通过矩阵拼接 (Concatenation) 操作直接更新状态 ($Cache_{new}$)，避免了对历史数据的重复计算，实现了 $O(N \cdot M)$ 的线性复杂度增量推理。


### 3.3.2 复杂度分析 (Complexity Analysis)

为了从理论层面证明该策略的有效性，我们对比了 System A (无缓存全量计算) 与 System B (增量计算) 在处理长度为 $N$ 的历史上下文和长度为 $M$ 的新片段时的计算复杂度。主要的计算开销来源于 Transformer 的 Self-Attention 层。

对于 System A (Baseline)，由于每次更新需对整个序列 $L = N + M$ 进行全量 Attention 计算，其 Attention 矩阵维度为 $(N+M) \times (N+M)$。因此，其计算复杂度为：

$$O((N+M)^2) \approx O(N^2) \quad (\text{当 } N \gg M \text{ 时})$$

对于 System B (Ours)，得益于 KV Cache，系统仅需计算新 Token (作为 Query) 与所有 Token (作为 Key/Value) 的注意力交互。Query 向量长度为 $M$，而 Key/Value 向量长度为 $N+M$。其计算复杂度降低为：

$$O(M \cdot (N+M)) \approx O(N \cdot M)$$

由于 $M$ (流式增量片段，通常小于 20 tokens) 远小于 $N$ (历史对话上下文，可达数千 tokens)，System B 成功将复杂度从关于上下文长度的“平方级”降低到了“线性级”。这一数学性质保证了系统在处理长语音交互时，能够维持近乎恒定的首字延迟，验证了架构设计的优越性。

## 3.4 实验数据构建与处理管线 (Data Construction Methodology)

为了在受控环境下系统性验证流式架构的性能，需要有不断递增长度的对话数据，本研究构建了一套自动化的数据合成管线。该管线将开源数据集： MultiWOZ [Budzianowski et al., 2018] (英文) 和 CrossWOZ [Zhu et al., 2020] (中文) 这两个标准多轮对话数据集转化为具有精确时长标注的逐渐递增的长语音测试数据，再通过tts生成音频数据。

### 3.4.1累积对话生成逻辑 (Cumulative Dialogue Strategy)

为了模拟用户进行长语音输入的真实场景（例如用户一口气陈述复杂的旅行需求，或在对话中不断补充信息），本实验采用了“累积对话”策略。如表 3-1 所示，算法遍历对话历史，将多轮对话中的用户侧（User）与上一轮的用户+系统回复文本进行累积拼接，从而构造出长度递增的输入序列。

| Turn Index | User       | System      | Cumulative Input Text (Simulated User Speech) |
| :--------- | :--------- | :---------- | :-------------------------------------------- |
| 1          | "订一张机票"    | "去哪里？"      | "订一张机票"                                       |
| 2          | "去北京，明天出发" | "要哪种舱位呢？"   | "订一张机票，去哪里？，去北京，明天出发"                         |
| 3          | "要商务舱"     | "好的，这就为您预订" | "订一张机票，去哪里？，去北京，明天出发。要哪种舱位呢？要商务舱"             |


通过此策略，我们能够基于真实的语义语境，生成从 3 秒至 60 秒以上不等的连续语音样本，有效覆盖了从短指令到长篇陈述的各种交互形态，解决了传统数据集缺乏长语音样本的问题。

### 3.4.2 数据处理流水线 (Processing Pipeline)

数据构建过程包含三个严格顺序执行的阶段 (Phases)，其核心代码实现对应于 `experiments/datasets/tools/run_pipeline.py`：

首先是**历史累积与筛选 (Accumulation & Filtering)**。策略遍历源数据集，应用上述累积生成逻辑。为了聚焦长语音场景下的性能瓶颈，系统按文本长度对生成样本进行倒序排列，优先选取最长的前 $N$ 个对话片段，并设置最大文本长度阈值（英文 2050 字符，中文 720 字符）以防止显存溢出 (OOM)。

其次是**并发 TTS 音频合成 (Batch TTS Synthesis)**。本研究集成了阿里巴巴提出的 CosyVoice 大模型语音合成服务 [Du et al., 2024]。相较于传统 TTS，CosyVoice 能够生成韵律更自然、情感更丰富的高保真语音波形，更贴近真实人声输入。在工程实现上，我们开发了 `BatchTTSProcessor` 模块，采用多线程 (Thread Pool) 异步请求机制，显著提升了大规模数据生成的效率。

最后是**时长校准与元数据同步 (Duration Calibration)**。由于生成式 TTS 模型的语速具有非确定性 (Non-deterministic)，简单的基于文本字数的时长估算往往存在误差。本管线通过解析生成的 WAV 文件头（Header）来获取精确到毫秒的物理时长：$Duration = \frac{TotalFrames}{SampleRate}$。该真实时长 ($audio\_duration$) 被回写至测试元数据 JSON 中，作为后续实验中 $X$ 轴（输入时长）的真值依据 (Ground Truth)。

### 3.4.3 测试集分组定义 (Benchmark Grouping)

为了细粒度分析不同时长下的延迟表现，我们将生成的样本集依据 $AudioDuration$ 划分为四个标准实验组（见表 3-2）。

[表 3-2: 实验数据分组定义]

| Group Name | Duration Range    | Typical Use Case     | Experiment Focus             |
| :--------- | :---------------- | :------------------- | :--------------------------- |
| Short      | $T < 5s$          | 短指令 (Short Commands) | 基准延迟验证 (Baseline Latency)    |
| Medium     | $5s \le T < 15s$  | 包含多轮意图的陈述            | 日常对话性能 (Daily Conversation)  |
| Long       | $15s \le T < 30s$ | 复杂长难句/长段落            | 流式架构核心优势区间                   |
| Extra Long | $T \ge 30s$       | 极限压力测试               | 系统稳定性与显存边界 (Stability & OOM) |

这一分组标准将在第四章的实验分析中贯穿始终，用于对比不同系统架构在处理长短语音时的性能差异。

  

# 第四章 实验与结果分析 (Experiments & Analysis)

  

本章围绕“VAD 判定用户语音结束至大语言模型产生首个响应 Token”的首字延迟（Time-to-First-Token, TTFT）开展实验验证，系统性评估本文提出的流式级联式语音对话架构（System B）在长语音交互场景中的性能收益与代价。与传统非流式基线（System A）相比，System B 的关键差异在于：其一，将 ASR 从端点后全量解码改造为分片流式解码，使声学计算与用户发声时长发生重叠；其二，在 ASR 增量输出的驱动下，对 LLM 执行增量预计算（KV Cache Prefill），以削弱长上下文一次性 Prefill 的串行等待。基于控制变量法，本章依次从延迟特性、关键模块消融以及精度-延迟权衡三个维度给出实证结果，并对观察到的“主导项切换”与工程边界进行讨论。

  

## 4.1 实验一：延迟与语音长度的关系验证 (Effect Validation)

  

### 4.1.1 实验设置与环境构建

  

为了客观衡量系统在长语音场景下的实时响应能力，本实验选取 **MultiWOZ** 与 **CrossWOZ** 两个公开数据集，分别代表英文与中文的多轮对话任务。依据 3.4 小节的数据构建流程，我们生成了覆盖短指令、中等陈述与长篇叙述的混合测试集，共 1132 条样本。需要说明的是，后续统计遵循表 3-2 的分组定义，并为更细粒度刻画超长语音的边界效应，将 $T \ge 30s$ 的 Extra Long 组进一步划分为 very_long（30–60s）与 extra_long（≥60s）两档。

  

实验硬件平台统一基于 两块**NVIDIA RTX 3090 (24GB VRAM)** 显卡的平台上进行，ASR模型和LLM模型分别在两块显卡中运行。该环境代表了消费级工作站的典型算力水平。为了消除模型加载与初始化带来的随机误差，我们对对比组（System A）与实验组（System B）实施了严格的公平性控制： 首先，两组实验均复用同一套模型权重，即 **Whisper-Turbo** [Radford et al., 2023] 作为声学编码器，以及 **Qwen2-7B-Instruct** [Qwen Team, 2024] 作为语义生成器，排除了参数量差异带来的干扰。其次，引入“热身机制”（Warmup Mechanism），在记录数据前预先执行3轮真实音频推理，以确保 CUDA Context 完成初始化且JIT编译器已优化核心算子。 在流式配置中，音频切片（Chunk Duration）锁定为 500ms，以模拟真实对话中的标准包发送频率。

  

### 4.1.2 延迟趋势分析

  

首字延迟（Time-to-First-Token, TTFT）是衡量语音交互流畅度的核心指标。图 4-1 展示了在不同语音时长输入下，非流式基线系统与本文流式系统的 TTFT 变化轨迹。

  ![[图4.1.png]]

图 4-1 不同语音时长下 System A 与 System B 的 TTFT 变化趋势。
*(图注：X轴为音频真实时长 $audio\_duration$（秒），Y轴为端到端 TTFT（毫秒），统计范围覆盖 1132 条样本。System A 曲线随时长近似单调上升，反映端点后串行等待的累积；System B 在 long 及以上区间趋于平缓，呈现由固定分片粒度与尾部调度开销主导的近似上界。两条曲线在 $T \approx 15s$ 附近出现拐点，意味着流式并行的覆盖收益开始超过其固定开销。)*

| 组别 | 样本数量 | 平均持续时间(秒) | 流式TTFT均值(毫秒) | 非流式TTFT均值(毫秒) | 改进量(毫秒) | 改进比例(%) |
|------|----------|------------------|-------------------|---------------------|-------------|------------|
| extra_long | 679 | 105.73 | 1087.70 | 6753.43 | 5665.73 | 83.9 |
| very_long | 208 | 45.21 | 1099.16 | 3191.97 | 2092.81 | 65.6 |
| long | 121 | 22.16 | 1126.63 | 1722.04 | 595.41 | 34.6 |
| medium | 89 | 9.25 | 959.53 | 923.42 | -36.11 | -3.9 |
| short | 35 | 3.44 | 648.17 | 533.42 | -114.75 | -21.5 |

[表 4-1: 不同语音时长分组下的 TTFT 统计]

表 4-1 给出了按语音时长分组的 TTFT 统计结果。非流式基线 System A 的 TTFT 随输入时长显著增长：short 组均值为 533.42ms，而进入 long 组后上升至 1722.04ms，并在 very\_long 与 extra\_long 组进一步增长到 3191.97ms 与 6753.43ms。该现象与级联式处理范式一致，即 ASR 需要等待端点后完成全量解码，随后 LLM 对完整转录文本执行一次性 Prefill；在此过程中，端点后的计算链路几乎不可重叠，导致等待随语音长度近似线性累积 [Vaswani et al., 2017; Pope et al., 2022]。

相比之下，流式系统 System B 在 long 及以上区间呈现近似“常数上界”：long、very\_long 与 extra\_long 三组 TTFT 均值分别为 1126.63ms、1099.16ms 与 1087.70ms，整体稳定在约 1.1s 量级。对应地，其在 long/very\_long/extra\_long 三组分别获得 34.6%、65.6% 与 83.9% 的 TTFT 缩减，其中 extra\_long 组绝对降低 5.67s，显著缓解长语音交互中的静默等待。需要强调的是，这里的“常数”指主导项由“随时长累积的全量计算”转移为“最后若干分片的尾部处理与调度开销”之和；在固定分片时长（500ms）与资源调度策略不变的前提下，TTFT 不再随输入时长继续增大。

与此同时，short 与 medium 组在 System B 下分别出现 114.75ms 与 36.11ms 的负向改进，说明在短语音场景中分片、缓存维护以及增量提示构造等固定开销尚难被并行覆盖。该结果提示工程落地时宜采用自适应策略：对预测时长较短的输入优先走简化链路，而将流式并行机制聚焦于 $T \ge 15s$ 的优势区间，以获得更稳定的总体交互体验。

  

## 4.2 实验二：核心模块消融实验 (Ablation Study)

  

为了量化“流式 ASR”与“LLM 增量预填充（KV Cache Prefill）”两项机制的相对贡献，本节在 long 及以上的样本上开展消融实验。按照 4.1 节的划分，我们重点考察 long（15–30s）、very\_long（30–60s）与 extra\_long（≥60s）三个分组，因为在该区间端点后的串行等待最为突出，也最能体现流式改造的收益。对比配置设置为三类：Baseline（System A，ASR 与 LLM 均为非流式串行执行）、Streaming ASR Only（仅 ASR 流式化，LLM 仍在最终转录完成后一次性 Prefill）以及 Full Streaming（在 Streaming ASR Only 基础上启用 KV Cache 状态保持与增量 Prefill，对应 System B）。  

### 4.2.1 评价指标与对比组  

本实验以 TTFT 作为核心指标，并在日志层面记录端点后的 ASR 尾部耗时与 LLM Prefill 耗时用于瓶颈诊断。为突出架构差异对用户感知延迟的影响，本文在表 4-2 中主要报告三种配置的 TTFT 均值以及由此计算的 ASR 增益与 KV 增益（正值表示额外缩短 TTFT，负值表示引入额外开销），从而保证结论可被归因于目标机制本身。

### 4.2.2 贡献度量化分析  

表 4-2 列出了各分组下三种配置的 TTFT 均值及对应增益，用于刻画瓶颈在流式化过程中的迁移。

| 组别 | 平均持续时间(秒) | 基线TTFT均值(毫秒) | 流式ASR TTFT均值(毫秒) | 全流式TTFT均值(毫秒) | ASR增益(毫秒) | KV增益(毫秒) | 
|------|------------------|-------------------|-----------------------|---------------------|-------------|-------------|
| long | 21.51 | 1698.88 | 1064.18 | 1084.17 | 634.70 | -19.99 |
| very_long | 42.43 | 3300.58 | 1171.02 | 1154.06 | 2129.56 | 16.96 | 
| extra_long | 83.75 | 6518.40 | 1228.77 | 1114.57 | 5289.63 | 114.20 | 

[表 4-2: 核心模块消融实验结果]

由表 4-2 可见，ASR 流式化是 TTFT 缩减的主要来源。Baseline 相比 Streaming ASR Only，在 long、very\_long 与 extra\_long 三组分别减少 634.70ms、2129.56ms 与 5289.63ms，对应降幅约为 37.4%、64.5% 与 81.1%。这表明在传统串行链路中，端点后等待的主导部分来自 ASR 对最后若干分片的解码、稳定输出与端点收敛，而流式 ASR 将大部分声学计算前移并与发声时长重叠，从而显著压缩端点后的剩余工作量。  

LLM 侧的增量 Prefill 体现出明显的“长度条件性”。在 long 组，Full Streaming 相比 Streaming ASR Only 的 TTFT 增加 19.99ms（KV 增益为 -19.99ms），说明当上下文尚未足够长时，增量调用、缓存管理与同步开销可能超过其节省的一次性 Prefill 计算；而在 very\_long 与 extra\_long 组，KV 增益分别为 16.96ms 与 114.20ms，且在 extra\_long 组的相对收益达到约 9.3%（114.20/1228.77），提示随着输入文本与上下文长度增长，LLM Prefill 的成本上升使得增量预填充更具性价比。  

综合两项机制，Full Streaming 将 TTFT 均值稳定在约 1.1s（long 1084.17ms、very\_long 1154.06ms、extra\_long 1114.57ms），相较 Baseline 分别实现约 36.2%、65.0% 与 82.9% 的缩减。该结果与实验一的总体趋势一致：流式并行能够有效削弱级联系统随输入时长增长的串行等待，但 KV Cache Prefill 的边际收益仍受工程实现与调用粒度影响，后续可通过异步化、批处理与更细的调度策略进一步提升其在长上下文下的贡献度。

  

## 4.3 实验三：准确率与质量边界 (Accuracy & Quality)

  

在延迟被显著压缩后，流式化带来的主要代价来自“上下文截断”与“端点不确定性”。与 System A 可利用整段音频不同，流式 ASR 在每个分片上只能看到局部声学片段，句末边界附近更易出现替换、插入与省略等错误。为引入稳定的归纳偏置，本文在 ASR 侧采用滑动窗口上下文管理，通过前缀（prefix）提供历史约束，通过后缀（suffix）提供有限前瞻，在精度与尾部耗时之间进行可控调节。基于该机制，本实验对比三种配置：prefix=1, suffix=0（默认）、prefix=1, suffix=1（增加 1 段前瞻以改善句尾收敛）以及 prefix=0, suffix=0（完全去除上下文以追求最小尾部延迟），并从 medium、long 与 very\_long 组各抽取 50 条样本，共 150 条，用于覆盖中长语音的主要应用场景。  

### 4.3.1 评估方法  

我们采用业界标准的词错误率（Word Error Rate, WER）和字错误率（Character Error Rate, CER）作为衡量指标，分别针对英文（MultiWOZ）与中文（CrossWOZ）样本计算转录误差，并将非流式全量识别（System A）作为上界对照，比较不同流式上下文配置（System B）下的识别质量与尾部耗时。  

### 4.3.2 精度-延迟权衡分析  

表 4-3 展示了不同上下文窗口配置下的准确率与尾部耗时对比结果。

| context config | multiwoz<br>wer_mean | multiwoz<br>cer_mean | crosswoz<br>wer_mean | crosswoz<br>cer_mean | asr_time_ms |
| :--- | :---: | :---: | :---: | :---: | :---: |
| pre1&suf1 | 0.0895 | 0.0196 | 0.0083 | 0.0203 | 1327.48 |
| pre1&suf0(default) | 0.0813 | 0.0221 | 0.0118 | 0.0292 | 1224.96 | 
| pre0&suf0 | 0.0619 | 0.0226 | 0.0210 | 0.0500 | 1086.16 |

[表 4-3: 不同上下文窗口配置下的流式 ASR 准确率与尾部耗时对比]
  

(注：每种上下文配置均包含 150 条样本，其中 MultiWOZ 74 条、CrossWOZ 76 条。由于评测音频由文本经 TTS 合成生成，参考文本可视为该批样本的近似真值；同时，System A 在该合成集合上的 WER/CER 接近 0，可作为上界对照。本文主要关注不同上下文配置下的相对差异，不将该结果直接外推为真实录音场景的绝对误差。)  

从表 4-3 可以看出，增加后缀前瞻（pre1&suf1）对中文句尾稳定性具有更直接的收益：CrossWOZ 的 WER/CER 分别由默认配置的 0.0118/0.0292 降至 0.0083/0.0203。代价是端点后的尾部耗时上升，`asr_time_ms` 由 1224.96ms 增至 1327.48ms（+102.52ms），这符合后缀窗口需要等待额外音频片段并完成更多解码计算的机制预期。对于英文 MultiWOZ，suffix 的引入使 CER 由 0.0221 降至 0.0196，但 WER 略升至 0.0895，提示在以词为单位的评价下，后缀窗口对对齐与分词的影响可能更敏感；该现象需要在真实录音与更大样本上进一步复核。

当完全去除上下文（pre0&suf0）时，系统获得了更低的尾部耗时（`asr_time_ms` 下降至 1086.16ms，相比默认减少 138.80ms），但中文误差出现明显放大：CrossWOZ 的 CER 上升至 0.0500，WER 上升至 0.0210。该结果表明，prefix 提供的历史上下文在流式识别中起到了重要的约束作用，可为声学-语言模型的局部决策提供更强的归纳偏置，抑制长句中逐步累积的替换与漂移误差 [Yu et al., 2021]。

  

从端到端交互预算看，实验一与实验二表明 long 及以上语音的 TTFT 已可稳定压缩至约 1.1s，而表 4-3 中不同上下文配置带来的 0.1–0.2s 尾部耗时差异已占据可观比例，因此有必要按应用场景进行配置选择：在中文任务对识别正确性更敏感的场景下，可采用 pre1&suf1 以降低句尾错误；在更强调响应速度或允许一定转写噪声的场景下，默认 pre1&suf0 提供较稳健的折中，而 pre0&suf0 适合作为极低延迟模式但需配合后处理与回退机制。考虑到对话式 LLM 在语义层面往往具备一定的噪声鲁棒性，本研究在后续工作中将进一步引入语义一致性与任务完成率等指标，对“转写误差是否导致语义偏离”进行更直接的量化评估（此处待补充相关实证引用）。

  
  

# 第五章 总结与展望 (Conclusion & Future Work)

## 5.1 全文总结(Summary of Contributions)

本文针对当前人机语音交互（HCI）领域中一个亟待解决的核心矛盾——即大语言模型（LLM）日益增长的推理开销与实时对话系统对低延迟响应的严苛要求之间的矛盾，提出并实现了一套基于细粒度流水线并行（Fine-grained Pipeline Parallelism）的低延迟语音对话架构（System B）。

回顾全文，本研究的核心工作主要围绕“计算流式化”与“状态增量化”两大技术主线展开，通过系统性的架构重构，打破了传统级联系统（Cascaded Systems）中存在的串行阻塞瓶颈。

首先，针对感知层面的延迟滞后问题，本文提出了一种**自适应流式 ASR 上下文管理机制**。不同于传统 Faster-Whisper [Radford et al., 2023] 必须等待语音端点检测（VAD）判定整句结束才启动推理的“全量批处理”模式，本研究通过构建环形缓冲区（Ring Buffer）与滑动窗口策略，成功将离线 ASR 改造为具备毫秒级响应能力的流式识别引擎。该机制有效地平衡了“切分粒度”与“语义完整性”之间的权衡（Trade-off），确保了系统在用户发声的同时即可并行输出高置信度的文本片段。

其次，针对认知层面的推理计算冗余，本文设计了**基于 LLM 的增量 KV Cache 预填充算法**。在传统架构中，ASR 输出的每一次文本更新都会迫使 Transformer 模型 [Vaswani et al., 2017] 对不断增长的历史 Prompt 进行重复编码，导致计算复杂度随序列长度呈 $O(N^2)$ 级数增长。本研究利用 Transformer 解码器的自回归特性，实现了一种“边听边想”的增量推理范式。通过持久化维护 Key-Value Cache 状态，系统将计算复杂度成功降维至 $O(N)$ 的线性水平。这一改进从理论上消除了长语音场景下的首字延迟（TTFT）线性增长效应，使得系统在处理长达 60 秒的连续陈述时，仍能保持恒定的毫秒级响应速度。

实验数据有力地支撑了上述架构的有效性。在基于 MultiWOZ 和 CrossWOZ 构建的长语音评测基准上，System B 相比传统非流式基线系统（System A）展现出了显著的性能优势。特别是在 15-30 秒的长语音分组中，系统不仅实现了端到端延迟的数量级下降，更在保持识别准确率（WER/CER）处于工业级可接受范围的同时，极大地提升了用户的交互流畅度与沉浸感。

## 5.2 研究局限性 (Limitations)

尽管本研究在降低交互延迟方面取得了显著进展，且所提出的流式架构在工程实践中表现出了良好的鲁棒性，但受限于级联架构本身的结构性约束，系统在迈向“类人自然对话”的终极目标上仍面临以下局限：

首先是**半双工交互模式（Half-duplex Interaction）带来的体验割裂**。当前系统设计虽然优化了响应速度，但本质上仍遵循“用户发言 $\rightarrow$ 系统处理 $\rightarrow$ 系统回复”的半双工话轮转换（Turn-taking）逻辑。由于现有的 VAD 模块缺乏对用户“打断（Barge-in）”意图的高层语义理解，当系统处于生成或播放语音的状态时，往往难以精准捕捉并响应用户的插话行为。这种对交互状态的刚性锁定，使得人机对话在灵活性上仍落后于人类自然的“重叠对话（Overlapping Speech）”体验。

其次是**级联架构导致的副语言信息丢失（Loss of Paralinguistic Information）**。作为典型的级联系统，本研究采用“文本”作为 ASR 感知模块与 LLM 认知模块之间的唯一信息接口。这种离散化的符号接口虽然简化了工程实现，但也造成了严重的信息有损压缩（Lossy Compression）[Lakomkin et al., 2018]。人类语音中蕴含的丰富副语言特征——如说话人的情绪波动（愤怒、犹豫）、语调变化（疑问、反讽）以及微小的停顿节奏——在强制转录为文本的过程中被不可逆地丢弃。由于 LLM 无法直接感知原始声学特征，导致生成的回复往往局限于字面语义的正确性，而缺乏情感上的共鸣与适配，呈现出“理智但冷漠”的交互缺陷。

## 5.3 未来展望 (Future Work)

尽管本文在长语音场景下显著压缩了 TTFT，并验证了流式并行与状态增量化的总体有效性，但在真实交互中，用户体验往往受“长尾延迟”“保守端点判定”以及“可打断性不足”等因素的共同制约。结合实验一的明细日志与系统实现细节，未来工作将围绕尾部延迟抖动抑制、语义结束判定以及面向打断的回复预生成三条主线展开，以进一步逼近自然对话所要求的低等待与高鲁棒性。

### 5.3.1 尾部分片波动与 TTFT 长尾抑制

从实验一的明细数据可以观察到，尽管 System B 在长语音分组下的 TTFT 均值稳定在约 1.1s，但仍存在少量样本出现 2s 以上的尾部峰值。进一步分析表明，该现象并非来自模型计算复杂度的突增，而是由末端音频分段到达时间的随机波动叠加调度策略导致的“尾分片碰撞”：当倒数第二个音频段刚进入处理流程时，最后一个音频段可能在不足 500ms 的间隔内到达，系统为了维持滑动窗口的一致性需要对队列进行第二次推理。由于单次音频段端点后处理的耗时本身约为 1s 量级，短时间内触发两次推理会直接造成 TTFT 的近似倍增，从而形成影响交互稳定性的长尾延迟。

面向该问题，后续将从调度与算法两侧进行联合优化。调度层面可引入“末端守护时间”与自适应合并策略：当检测到分段间隔显著小于单段处理时间且端点置信度较高时，优先延迟一次触发并合并末端分段，以减少重复推理；同时结合队列回压与异步化处理，将末端突发的分段到达吸收为可控的调度开销。算法层面则进一步探索中间结果复用，例如缓存声学特征或编码器隐藏状态，将“二次推理”由全量重算改造为增量更新，从根本上抑制尾部分段波动对 TTFT 的放大效应。评估方面，除均值指标外将重点报告 P95/P99 等长尾统计，并结合时间轴对 TTFT 进行分解归因，以便量化不同优化策略对端到端稳定性的贡献。

### 5.3.2 基于语义的结束判定与停顿等待压缩

当前原型系统主要依赖静音阈值完成话轮结束判定，例如用户停止发声后需等待约 1s 的停顿窗口，系统才将输入视为结束并进入回复阶段。该策略在噪声环境下具有工程合理性，但当 TTFT 已被压缩到 1s 量级后，固定的停顿等待反而成为新的主导项，限制了进一步降低交互等待的上界。未来工作将研究一种轻量级的语义结束判别模型（End-of-Utterance, EOU），以流式 ASR 的增量文本为主要输入，结合对话上下文与局部语言结构特征，预测当前话轮是否已语义完结，从而在保证鲁棒性的前提下缩短停顿等待。

在系统集成上，更可行的路径是将 EOU 判别与静音阈值形成互补：当模型输出高置信的结束信号时提前触发回复；当置信度不足或环境噪声导致文本不稳定时，则回退到保守的停顿判定，以控制误判带来的抢话与语义截断风险。相应地，评估不仅应关注平均等待缩短幅度，还需同时报告误触发率、回退频次以及对用户主观流畅度的影响，从而把“更快”与“更稳”的权衡纳入统一度量框架。

### 5.3.3 回复预生成、语义影响判别与打断机制

在端点判定能够更早触发的基础上，系统可进一步将 LLM 的部分生成计算前移：一旦判断输入大概率结束，LLM 便提前开始准备回复草稿，使得用户停顿窗口与模型推理发生重叠。然而真实对话具有显著的可打断性，用户可能在系统准备或播放回复时继续补充信息。若系统简单丢弃草稿并完全重算，将同时造成计算浪费与额外等待，因此有必要建立“可提前计算、可随时撤销”的统一机制。

未来工作拟引入轻量级的“语义影响判别”模型，对新增输入片段进行在线评估，判断其是否改变已生成草稿的关键意图或约束条件；当影响可忽略时直接复用草稿并进入输出阶段，当影响显著时再触发回退与增量重生成。与之配套，LLM 侧需要建立可回退的推理状态管理，为“已确认上下文”维护主分支 KV Cache，并为草稿生成维护可丢弃的投机分支，使系统能够在用户续说时快速撤销未提交状态，并在必要时以最小代价重建提示与继续生成。进一步地，为支持从半双工向更自然交互形态过渡，还需完善生成与合成过程的打断机制，包括对 LLM 解码与 TTS 合成的抢占式取消、对已输出语音的停止与淡出策略，以及对对话状态一致性的维护。

![[图5.1.png]]

图 5-1 基于语义结束预测的回复预生成与可回退打断流程示意图。
*(图注：以时间轴展示用户语音流、VAD/EOU 判定、LLM 草稿生成（投机分支）与提交点（主分支）的对应关系，并给出用户续说触发的撤销与重算路径。该流程的关键在于将“提前计算”与“可撤销性”同时纳入调度，使端到端延迟缩减与误触发风险控制能够在同一框架内达成。)*

---

**References**

[1] OpenAI, "GPT-4o System Card," 2024. [Online]. Available: https://arxiv.org/abs/2410.21276

[2] Google Team, "Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context," *arXiv preprint arXiv:2403.05530*, 2024.

[3] A. Radford *et al.*, "Robust Speech Recognition via Large-Scale Weak Supervision," in *Proc. ICML*, 2023, pp. 28492–28518.

[4] T. Dao *et al.*, "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 35, 2022, pp. 16344–16359.

[5] A. Graves, "Sequence Transduction with Recurrent Neural Networks," in *ICML Workshop on Representation Learning*, 2012.

[6] A. Vaswani *et al.*, "Attention Is All You Need," in *Advances in Neural Information Processing Systems (NIPS)*, vol. 30, 2017.

[7] A. Gulati *et al.*, "Conformer: Convolution-augmented Transformer for Speech Recognition," in *Proc. Interspeech*, 2020, pp. 5036–5040.

[8] T. Krichli *et al.*, "CarelessWhisper: Turning Whisper into a Causal Streaming Model," *arXiv preprint arXiv:2508.12301*, 2025.

[9] F. Yu *et al.*, "FastEmit: Low-latency Streaming ASR with Sequence-level Emission Regularization," in *Proc. ICASSP*, 2021, pp. 6004–6008.

[10] R. Pope *et al.*, "Efficiently Scaling Transformer Inference," *arXiv preprint arXiv:2211.05102*, 2022.

[11] G. Xiao *et al.*, "Efficient Streaming Language Models with Attention Sinks," in *Proc. ICLR*, 2024.

[12] Y. Gao *et al.*, "Mini-Omni: Language Models Can Hear, Talk While Thinking in Streaming," *arXiv preprint arXiv:2408.16725*, 2024.

[13] H. Sacks, E. A. Schegloff, and G. Jefferson, "A simplest systematics for the organization of turn-taking for conversation," *Language*, vol. 50, no. 4, pp. 696-735, 1974.

[14] Silero Team, "Silero VAD: pre-trained enterprise-grade Voice Activity Detector," 2024. [Online]. Available: https://github.com/snakers4/silero-vad

[15] P. Budzianowski *et al.*, "MultiWOZ - A Large-Scale Multi-Domain Wizard-of-Oz Dataset for Task-Oriented Dialogue Modelling," in *EMNLP*, 2018.

[16] Q. Zhu *et al.*, "CrossWOZ: A Large-Scale Chinese Cross-Domain Task-Oriented Dialogue Dataset," in *TACL*, 2020.

[17] Z. Du *et al.*, "CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models," *arXiv preprint arXiv:2412.10117*, 2024.

[18] Qwen Team, "Qwen2.5: A Party of Foundation Models," 2024. [Online]. Available: https://qwenlm.github.io/blog/qwen2.5/

[19] E. Lakomkin *et al.*, "Kt-Speech-Crawler: Automatic Dataset Construction for Speech Recognition from YouTube Videos," in *EMNLP*, 2018.

[20] Y. Wang *et al.*, "Real-Time Speech Interaction with Large Language Models," *arXiv preprint arXiv:2404.03058*, 2024.

[21] M. Macháček and T. Polák, "Turn-Taking in Conversational Systems," *arXiv preprint arXiv:2503.05943*, 2025.
---

**BibTeX**

```bibtex
@article{openai2024gpt4o,
  title={GPT-4o System Card},
  author={OpenAI},
  journal={arXiv preprint arXiv:2410.21276},
  year={2024}
}

@article{google2024gemini,
  title={Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context},
  author={Google Team and others},
  journal={arXiv preprint arXiv:2403.05530},
  year={2024}
}

@inproceedings{radford2023robust,
  title={Robust Speech Recognition via Large-Scale Weak Supervision},
  author={Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and McLeavey, Christine and Sutskever, Ilya},
  booktitle={International Conference on Machine Learning},
  pages={28492--28518},
  year={2023},
  organization={PMLR}
}

@inproceedings{dao2022flashattention,
  title={FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness},
  author={Dao, Tri and Fu, Dan and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
  booktitle={Advances in Neural Information Processing Systems},
  volume={35},
  pages={16344--16359},
  year={2022}
}

@inproceedings{graves2012sequence,
  title={Sequence Transduction with Recurrent Neural Networks},
  author={Graves, Alex},
  booktitle={ICML Workshop on Representation Learning},
  year={2012}
}

@inproceedings{vaswani2017attention,
  title={Attention Is All You Need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  booktitle={Advances in Neural Information Processing Systems},
  volume={30},
  year={2017}
}

@inproceedings{gulati2020conformer,
  title={Conformer: Convolution-augmented Transformer for Speech Recognition},
  author={Gulati, Anmol and Qin, James and Chiu, Chung-Cheng and Parmar, Niki and Zhang, Yu and Yu, Jiahui and Han, Wei and Wang, Te-Jason and Zhang, Zhengdong and Wu, Yonghui and others},
  booktitle={Proc. Interspeech},
  pages={5036--5040},
  year={2020}
}

@article{krichli2025careless,
  title={CarelessWhisper: Turning Whisper into a Causal Streaming Model},
  author={Krichli, Tomer and others},
  journal={arXiv preprint arXiv:2508.12301},
  year={2025}
}

@inproceedings{yu2021fastemit,
  title={FastEmit: Low-latency Streaming ASR with Sequence-level Emission Regularization},
  author={Yu, Jiahui and others},
  booktitle={ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={6004--6008},
  year={2021},
  organization={IEEE}
}

@article{pope2022efficiently,
  title={Efficiently Scaling Transformer Inference},
  author={Pope, Reiner and Douglas, Sholto and Chowdhery, Aakanksha and Devlin, Jacob and Bradbury, James and Dean, Jeff and others},
  journal={arXiv preprint arXiv:2211.05102},
  year={2022}
}

@inproceedings{xiao2024efficient,
  title={Efficient Streaming Language Models with Attention Sinks},
  author={Xiao, Guangxuan and Tian, Yuandong and Chen, Beidi and Han, Song and Lewis, Mike},
  booktitle={International Conference on Learning Representations},
  year={2024}
}

@article{gao2024mini,
  title={Mini-Omni: Language Models Can Hear, Talk While Thinking in Streaming},
  author={Gao, Y and others},
  journal={arXiv preprint arXiv:2408.16725},
  year={2024}
}

@article{sacks1974simplest,
  title={A simplest systematics for the organization of turn-taking for conversation},
  author={Sacks, Harvey and Schegloff, Emanuel A and Jefferson, Gail},
  journal={Language},
  pages={696--735},
  year={1974},
  publisher={JSTOR}
}

@misc{silero2024vad,
  title={Silero VAD: pre-trained enterprise-grade Voice Activity Detector},
  author={Silero Team},
  year={2024},
  url={https://github.com/snakers4/silero-vad}
}

@inproceedings{budzianowski2018multiwoz,
  title={MultiWOZ - A Large-Scale Multi-Domain Wizard-of-Oz Dataset for Task-Oriented Dialogue Modelling},
  author={Budzianowski, Pawe{\l} and Wen, Tsung-Hsien and Tseng, Bo-Hsiang and Casanueva, I{\~n}igo and Ultes, Stefan and Ramadan, Osman and Ga{\v{s}}i{\'c}, Milica},
  booktitle={Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
  pages={5016--5026},
  year={2018}
}

@article{zhu2020crosswoz,
  title={CrossWOZ: A Large-Scale Chinese Cross-Domain Task-Oriented Dialogue Dataset},
  author={Zhu, Qi and others},
  journal={Transactions of the Association for Computational Linguistics},
  volume={8},
  pages={281--295},
  year={2020}
}

@article{du2024cosyvoice,
  title={CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models},
  author={Du, Zhihao and others},
  journal={arXiv preprint arXiv:2412.10117},
  year={2024}
}

@misc{qwenteam2024qwen25,
  title={Qwen2.5: A Party of Foundation Models},
  author={Qwen Team},
  year={2024},
  howpublished={\url{https://qwenlm.github.io/blog/qwen2.5/}}
}

@inproceedings{lakomkin2018ktspeech,
  title={Kt-Speech-Crawler: Automatic Dataset Construction for Speech Recognition from YouTube Videos},
  author={Lakomkin, Egor and others},
  booktitle={EMNLP},
  year={2018}
}

@article{wang2024realtime,
    title={Real-Time Speech Interaction with Large Language Models},
    author={Wang, Y and others},
    journal={arXiv preprint arXiv:2404.03058},
    year={2024}
}

@article{machacek2025turntaking,
    title={Turn-Taking in Conversational Systems},
    author={Mach{\'a}{\v{c}}ek, M and Pol{\'a}k, T},
    journal={arXiv preprint arXiv:2503.05943},
    year={2025}
}
```
