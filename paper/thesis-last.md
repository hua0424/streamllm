# 基于流式架构的低延迟语音对话系统优化

# Optimization of Low-Latency Voice Dialogue Systems based on Streaming Architecture

# 第一章 绪论 (Introduction)

## 1.1 研究背景与意义

### 1.1.1 大模型驱动的语音交互范式演进

随着人工智能技术的飞速发展，人机交互正经历着一场从“指令式”向“自然流式对话”的快速转变。 GPT-4o [1] 和 Gemini 1.5 [2] 的出现，代表了原生多模态大模型的问世，使之前只能文字聊天的语言模型，首次具备了在听觉、视觉和文本模态间进行理解与生成的能力。传统的语音助手(如早期的 Siri 或天猫精灵等)主要依赖基于规则或“意图-槽位”(Intent-Slot)的实现方式，其虽然在执行特定指令时表现良好，但其固定的句法十分生硬，严重限制了用户的表达自由度。用户需要遵守厂家预设的命令模板，这种僵硬的对话方式使人机之间的交互充满了机械感。

相比之下，基于大语言模型（LLM）的新一代语音对话系统展现出更强的理解能力，能够处理复杂推理任务，并支持带有情感色彩的多轮自然语言交互，然而这种智能化水平的提升也带来了巨大的计算开销与延迟挑战。公开的系统报告显示，GPT-4o 的端到端语音响应延迟可低至232ms ，平均约320ms [1] ，已接近人类自然对话的反应速度。这意味着 ，“低延迟”正与“准确率”并列，成为下一代语音交互体验的关键竞争指标。

### 1.1.2 延迟：影响用户体验的核心痛点

尽管端到端多模态模型在实验室里成绩亮眼，但训练算力昂贵、数据隐私壁垒高筑，加上垂直场景需深度定制，音频模块，语音合成模块，LLM模块无法及时更新到最新的SOTO，因此工业界的主流落地方案仍倾向级联架构。该架构把系统拆成三个独立模块，并按顺序连接起来：ASR（自动语音识别）、LLM（大语言模型）和 TTS（语音合成）。

在这种串行架构中，系统的总延迟($L_{total}$)不仅取决于各个模块的处理时间，还显著受限于模块间的数据流转方式。为便于分析，级联系统的总延迟可形式化为各组件延迟的线性叠加：

$$L_{total} = L_{VAD} + L_{ASR} + L_{LLM\_Prefill} + L_{LLM\_Decode} + L_{TTS} + L_{Net}$$

在许多传统级联式实现中，用户结束发声到听见系统回应，往往要等待数秒。而人类对话的平均轮替间隙仅大约 200ms [3]，这一数量级的落差带来明显的迟滞感“认知摩擦”：用户完成表达后常被迫陷入长时间的静默等待，思维的连贯性因此被打断性，还常让用户误以为系统无响应而重复输入，交互随之陷入混乱。特别在长语音场景下，传统级联式 ASR 必须等到用户说完整句语音才启动转录，而 LLM 的预填阶段随输入文本变长，耗时同步增加，从而进一步拖慢长文本交互的实时表现。因此，如何在保留级联架构模块化优势的同时下，通过流式并行策略消除模块间的“死区时间”，实现“打断即响应”的交互体验，兼具学术价值与工程意义

## 1.2 国内外研究现状

### 1.2.1 自动语音识别 (ASR) 的流式化进展

ASR 系统的处理速度，决定了用户语音转化为LLM可接受的文本的速度。在架构演进上，早期的流式 ASR 以 RNN-Transducer （RNN-T）架构 [4]为主。RNN-T 凭借其天然的循环特性，非常适合处理流式音频，却因无法并行化，训练效率偏低，导致其长程语义依赖捕捉受限。随着 Transformer [5] 的出现，注意力机制模型将识别精度推至新高。然而，Transformer的全局注意力必须在整段输入就绪后方能计算，原生结构因而无法流式运行。

为了解决这一矛盾，Google 提出 Conformer 架构 [6]，将 CNN 的局部特征提取与 Transformer 的全局建模合二为一，迅速成为业界主流。在实际应用中，流式 ASR 通常采用分块处理与重叠拼接：先用滑动窗口切分音频并识别，再通过重叠区域对输出做一致性校正，在准确率与延迟之间取得平衡。以 Whisper 为例，其原始模型采用 Encoder-Decoder 结构，但主要面向离线转录 [7]；近期 CarelessWhisper [8] 把非因果编码器改成因果编码器，并在轻量微调与推理流程更新后，使 Whisper 可在更小的 chunk 上低延迟在线转录。而 FastEmit [9] 等则是从训练目标入手，通过序列级正则化鼓励token更早发射，以缩短流式识别的输出滞后时长。

### 1.2.2 大语言模型 (LLM) 推理加速技术

在 ASR 完成转录后，LLM 的推理速度成为系统延迟的第二大瓶颈。这一过程主要受限于 Transformer 解码器的自回归生成机制以及 GPU 的显存带宽。

针对推理过程中的重复计算问题，KV Cache(键值缓存)技术应运而生。该技术通过缓存历史 token 的 Key/Value 向量，使生成新 token 时无需对历史上下文重复计算中间表示，从而将解码阶段的单步计算从与序列长度二次相关降低为线性相关 [10]。目前，KV Cache 已成为 vLLM、TGI 等现代推理引擎的标配。针对长文本输入的预填充阶段，attention 计算量随序列长度急剧膨胀的问题，FlashAttention [11] 通过改进 GPU 显存的读写方式，采用分块 Tiling 策略，把长序列的吞吐量推至新高。在长对话场景下，StreamingLLM [12] 引入“注意力汇聚”机制，把早期 token 作为固定锚点，缓解上下文窗口溢出导致的性能衰减，使 LLM 在有限 KV 缓存内仍能维持稳定、持续的流式输出。

### 1.2.3 级联系统的全链路优化

除了单独优化 ASR 或 LLM，全链路的协同调度对降低整体延迟同样关键。传统 Simply-Cascaded 系统在模块间是完全串行处理的：下游模块必须等上游模块完全输出后才能启动，使端点后的计算链路无法重叠。为接近实时交互的延迟，近期研究一方面尝试端到端流式语音对话模型，让系统在语音持续输入的同时边推理边生成 [13]；另一方面，在仍保留级联架构的场景里，流水线并行成为更通用的工程方案：一旦 ASR 输出部分文本片段，LLM便可提前增量预填充并持续刷新 KV Cache，将原本集中在语音输入结束之后的计算前移至用户发声阶段并行完成，最终压缩端点后的等待时长。

## 1.3 本文主要研究内容  

传统级联语音对话系统在处理长语音时，各模块产生的延迟会逐级叠加，同时出现资源空转延迟累积与资源闲置问题，现有研究要么单独优化 ASR 模型，要么把 LLM 扩展为多模态架构直接生成语音，都难以复用已有的 SOTA 模型。本文提出一种细粒度流水线并行方案（下文称 System B），可直接兼容现有非流式 ASR 模型与文本 LLM。本文的主要研究内容与贡献如下：  

构建流式 ASR 上下文管理机制：本文设计了基于 Whisper 和 Silero VAD 的自适应滑动窗口算法。为了解决流式切片导致的上下文缺失与识别不稳定性问题，本文提出了动态缓冲与“前缀后缀上下文(Prefix Context & Suffix Context)”拼接策略，在保证长句识别准确率的同时，实现毫秒级的 ASR 转录片段输出。

提出 LLM KV Cache 增量预填充策略：针对长语音输入带来的首字延迟（TTFT）问题，本文设计了增量推理算法。该算法深度利用 Transformer 的 KV Cache 机制，实时接收流式 ASR 的输出片段进行 Attention 计算与状态更新，避免了对历史上下文的重复编码，将 LLM 的计算开销均摊在用户的语音输入过程中。

全链路延迟评估与验证：本文在 MultiWOZ（英文）和 CrossWOZ（中文）数据集上构建了长语音测试基准。实验结果表明，在 15s 以上的长语音场景下，本文提出的 System B 相比传统非流式基线(System A)实现了显著的延迟降低，且未对系统的语义理解能力造成负面影响。

# 第二章 相关技术基础 (Theoretical Foundation)

本章将详细阐述本研究的相关理论与技术，重点说明 ASR 的神经网络架构特性以及 LLM 的底层推理机制，并以数学形式给出衡量系统性能的关键指标。

## 2.1 自动语音识别技术 (ASR)

本研究采用 OpenAI 在 Whisper 工作中提出的端到端语音识别模型 [7] 作为语音转文本模块的模型。与以 CTC/RNN-T 为代表的神经网络类型不同，Whisper 采用 Transformer 这种编码器—解码器，序列到序列框架，直接以文本 token 为条件生成输出；模型先在大规模弱监督语音–文本数据上预训练，因而在多语言、口音和噪声等复杂场景下仍保持鲁棒，为后续低延迟流式化改造奠定稳定基础。

### 2.1.1 Whisper 模型架构与离线特性

Whisper 的整体结构由音频编码器(Encoder)与文本解码器(Decoder)组成。输入端将 16kHz 音频转换为 80 维 Log-Mel 频谱特征，并通过轻量卷积前端(Conv Stem)完成初步特征提取与时间维下采样后送入 Transformer 编码器；在常见实现中，Conv Stem 由两层一维卷积构成(卷积核宽度为 3)，其中第二层采用 stride=2，从而在时间维实现 2× 下采样。以标准 30 秒音频输入为例，Log-Mel 特征约对应 3000 帧(10ms/帧)，经下采样后压缩至约 1500 帧，对应约 20ms 的帧间隔。解码端采用自回归生成，通过 cross-attention 读取编码器表征并输出文本 token。值得注意的是，Whisper 的词表中包含时间戳 token，可在解码时产生片段级乃至词级的时间对齐信息，这为流式场景中“仅提交已稳定且可对齐的部分输出”提供了直接的实现抓手。总体来看，Whisper的默认接口仍以固定时长片段做离线处理，这一设定及其对上下文的依赖，直接引出了下一节的流式化挑战。

![[图2.1.png]]

**图 2-1 Whisper 模型整体架构与特征降采样示意图** [如图所示，Whisper 先把 16 kHz 音频映射成 80 维 Log-Mel 谱，再由卷积前端在时间轴做 2× 下采样，把 30 s 输入的帧数从约 3000 压到 1500，以减轻后续编码器自注意力的计算量。]

### 2.1.2 流式化挑战与解决方案

由于 Whisper 的编码器自注意力未施加因果掩码，能够同时利用过去与未来帧的上下文；再叠加其默认以固定时长片段进行处理的接口设定，模型天然更偏离线范式。该“全局上下文”在离线场景下有助于提升识别准确率，但在实时对话中会带来两类典型困难：一方面，若将连续音频硬切成短块并独立识别，块边界附近由于上下文不足容易出现漏词、重复或重写；另一方面，最新到达音频对解码结果的影响往往滞后，导致末尾假设频繁变动，使下游模块难以稳定消费。

针对上述问题，流式化改造通常采取“分段输入 + 重叠上下文 + 稳定提交”的组合策略。本文在前端利用 VAD 对连续 PCM 流进行实时检测，将语音区间切分为若干可处理的音频片段，并通过一个音频缓冲区累积近期音频段，除当前目标片段外，还拼接其前后相邻音频作为上下文，以缓解边界处的信息缺失。为避免把尚不稳定的末尾文本提前输出，本文借助 Whisper 的时间戳与词级对齐能力，对输出施加“局部一致性”约束，仅提交时间戳完全落在目标片段内的文本段，从而形成可增量、可对齐的文本流，为下一节 LLM 的增量预填充提供稳定输入。

## 2.2 大语言模型推理机制 (LLM Inference)

大语言模型通常采用 Decoder-only Transformer 架构。在语音对话系统中，LLM 的推理延迟，尤其是首字延迟(Time to First Token, TTFT)，是决定用户是否感到“卡顿”的关键瓶颈。从推理流程看，TTFT 主要由输入 prompt 的预填充(prefill)计算与随后的首个解码步共同构成，而二者都与注意力的序列长度密切相关。

### 2.2.1 Transformer 注意力机制

Transformer 的核心是缩放点积注意力 (Scaled Dot-Product Attention) [5]。其标准计算公式如下：  

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$  

其中，$Q$ (Query), $K$ (Key), $V$ (Value) 分别由输入向量经过线性投影矩阵 $W_Q, W_K, W_V$ 得到。$d_k$ 为 Key 的维度，用于缩放点积的数值范围，避免 $QK^T$ 随维度增大而方差膨胀导致 softmax 饱和、梯度过小。该机制允许模型在一个时间步内显式聚合序列中任意位置的信息，从而捕捉长程依赖关系。
  
### 2.2.2 KV Cache 机制的数学原理与复杂度分析

LLM 的文本生成过程本质上是自回归的 (Auto-regressive)：模型在生成第 $t$ 个 token 时，需要利用此前 $t-1$ 个 token 的上下文信息。在 Self-Attention 中，若每个时间步都对长度为 $t$ 的完整序列重新执行一次前向计算(不复用任何历史中间结果)，则主要开销来自注意力矩阵乘法 $QK^T$，其单步复杂度近似为 $O(t^2)$。因此，生成长度为 $N$ 的序列的总计算复杂度为：

$$\text{Cost}_{total}^{naive} = \sum_{t=1}^{N} t^2 = \frac{N(N+1)(2N+1)}{6} \approx O(N^3)$$  

尽管上述“重复前向”的实现并非现代推理框架的常用做法，但它直观说明了自回归生成中重复计算的来源：每一步都在重新计算历史 token 对应的 Key/Value 以及它们之间的注意力关系。

KV Cache 的基本思想是将历史 token 在各层产生的 Key/Value 缓存并复用。由于模型参数在推理阶段固定，历史 Token $x_i (i < t)$ 的 $k_i, v_i$ 一旦计算便不再改变，因此可将其存储在 GPU 的显存(VRAM)中。在第 $t$ 步生成时，系统仅需计算当前 Token $x_t$ 的 $k_t$ 和 $v_t$，并将其追加到缓存末尾：  

$$K_{cache}^{(t)} = \text{Concat}(K_{cache}^{(t-1)}, k_t)$$  

$$V_{cache}^{(t)} = \text{Concat}(V_{cache}^{(t-1)}, v_t)$$

随后，注意力计算仅涉及当前 Query $q_t$ 与历史 $K_{cache}$ 的交互，即计算 $q_tK_{cache}^{T}$ 并加权 $V_{cache}$。因此，KV Cache 将解码阶段的单步复杂度由 $O(t^2)$ 降低为 $O(t)$(仍会随上下文长度线性增长)，生成 $N$ 个 token 的累计复杂度约为 $\sum_{t=1}^{N} t \approx O(N^2)$；相应代价是缓存的显存占用随序列长度线性增长。

需要强调的是，KV Cache 主要减少的是解码阶段对历史 Key/Value 的重复计算，但并不消除 prompt 预填充阶段的二次方开销：对长度为 $M$ 的输入 prompt，模型仍需至少执行一次全序列前向以构建初始缓存，其复杂度为 $O(M^2)$，并往往构成 TTFT 的主要来源。本文提出的增量预填充策略并非改变复杂度阶，而是利用上游流式 ASR 文本“逐步到达”的时间特性，将 prefill 拆分为多次小批量前向并在语音输入过程中持续更新 KV Cache，从而把原本集中发生在 End-of-Speech 之后的计算前移并与用户发声过程重叠，最终降低语音结束到首 token 生成之间的等待。

![[图2.2.png]]  

**图 2-2 基于 KV Cache 的推理机制示意**[左侧示意“重复前向”做法：在生成新 token 时对长度为 $N$ 的序列重新计算 self-attention，单步开销近似 $O(N^2)$。右侧示意 KV Cache：将历史 $K,V$ 缓存于显存并复用，使当前步仅需计算新 token 的 $K,V$ 并与缓存交互，从而把单步开销降为 $O(N)$；生成 $N$ 个 token 的累计开销约为 $O(N^2)$，同时带来线性增长的缓存显存占用。]

## 2.3 系统评价指标 (Evaluation Metrics)
 
为了客观评估优化策略的有效性，本研究选取了以下两个核心指标：

### 2.3.1 首字延迟 (Time to First Token, TTFT)

TTFT 是衡量交互实时性的核心指标，直接关联用户的主观等待感。本文将 TTFT 定义为从用户语音结束(End of Speech)到 LLM 输出首个响应 token 的时间差，并将其视为端点后关键路径延迟的综合体现。在 System B 中，由于部分 $L_{LLM\_Prefill}$ 可在用户发声过程中被增量预填充策略“前移”并与 ASR 并行，端点后的等待相较 System A 显著降低。

### 2.3.2 词错误率 (Word Error Rate, WER)

虽然本研究的核心目标是降低延迟，但这不能以牺牲识别准确率为代价。WER 是评估 ASR 准确率的通用标准，基于 Levenshtein 编辑距离计算：  

$$\text{WER} = \frac{S + D + I}{N_{ref}}$$    

其中 $S$ 为替换 (Substitution)、$D$ 为删除 (Deletion)、$I$ 为插入 (Insertion) 的错误数量，$N_{ref}$ 为参考文本的总词数。若以百分比形式呈现，可再乘以 $100\%$。对于中文数据集，本研究主要考察字符错误率 (CER)，其计算逻辑与 WER 一致。
  
# 第三章 基于流式架构的低延迟语音对话系统设计 (System Design & Methodology)

本章将深入剖析本文提出的流式语音对话系统(System B)的架构范式与核心算法实现。针对传统级联架构(System A)在长语音交互场景中存在的“计算阻塞”与“延迟累积”问题，本研究提出了一种细粒度的流水线并行(Pipeline Parallelism)策略。该策略通过构建自适应流式 ASR 上下文管理机制与 LLM KV Cache 增量预填充技术，从根本上重构了数据流转方式，成功打破了模块间的串行依赖，实现了“感知-认知-表达”的同步进行，即“边听边想”的即时响应能力。

## 3.1 系统总体架构设计 (System Architecture)

### 3.1.1 设计目标与逻辑拓扑

本系统的核心设计愿景是实现从传统的“全量接收-全量处理”范式向“增量接收-流式处理”范式的转移(Paradigm Shift)，从而在理论边界上最小化首字延迟 (Time to First Token, TTFT)。系统的逻辑拓扑如图 3-1 所示。

![[图3.1 1.png]]

**图 3-1 System B 流式并行架构逻辑拓扑图**[本图展示了基于多线程生产者-消费者队列的流式并行数据流向。音频以固定时长的 PCM 块持续进入分段模块，Silero VAD [14] 在累计缓冲区上进行活动检测并输出语音段；ASR 模块将若干语音段拼接后调用 Whisper [7] 进行转录，并按前缀/后缀上下文策略输出稳定文本片段；文本片段经线程安全队列传递给 LLM 模块，持续更新 KV Cache 并在收到终止标记后启动生成，从而尽可能将 LLM 的预填充计算前移并与音频输入过程重叠。]  

整体架构采用“分段—转录—预填充—生成”的流水线并行方式：当上游模块产生新的音频或文本段时，下游模块即可立即启动计算，无需等待整段输入结束。系统由以下三个子系统组成：负责语音活动检测与分段的 Streaming Audio Segmenter、负责上下文拼接并保证输出稳定的 Context-Aware ASR Engine ，以及负责增量预填充与最终生成的 Incremental LLM Inference Service 。其中 LLM 部分的 KV Cache 属于 Transformer 解码器推理阶段的通用优化思路，本文直接复用推理框架内置的 use_cache/past_key_values 机制完成状态复用，理论背景见[5]。
  
### 3.1.2 关键工程实现机制

为了在单机原型中保持流式链路的实时与可控， 工程实现上把整条链路拆成多线程流水线，并用线程安全队列解耦， 同时借助显式状态对象保存跨片段的上下文。  

在通信机制上，各模块不再使用紧耦合的同步函数调用，而是采用基于生产者-消费者模型的异步通信策略。在代码实现上，音频块队列、音频分段队列与文本段队列均由 queue.Queue 实例承担，消费者侧使用带超时参数的 get 方法循环配合结束事件标记，避免单点阻塞导致整个处理流程卡死。尤其在 ASR 阶段，系统需要承接分段的音频，又要进行音频转录文本的流式处理，因此该系统进一步拆分为“收集器(collector)”与“转录器(transcriber)”两个子线程：前者持续接收上游语音段并写入等待队列，后者在满足触发条件时批量拼接并调用 Whisper 推理，将稳定的文本片段推送给下游 LLM，从而把音频收集与模型推理解耦开来。

在状态管理上，系统采用了有状态 (Stateful) 与无状态 (Stateless) 的混合设计。分段模块通过 StreamState 维护累计音频缓冲、语音段起止时间与段编号等信息；ASR 侧通过 ASRCache 管理“等待加入的段队列”与“当前转录窗口段队列”，并使用 total_duration 与处理标记避免并发竞争；LLM 侧则以 KVCache 持久化 past_key_values、attention_mask 以及预填充阶段产生的最后一步 logits，使得终止标记到达后可以直接进入生成阶段。上述状态对象共同保证了跨片段上下文的一致性，并为后续 3.2 与 3.3 的算法描述提供了可对应的工程落点。

## 3.2 自适应流式 ASR 上下文管理 (Adaptive Streaming ASR)

流式 ASR 的核心挑战在于寻找“识别准确率”与“识别延迟”之间的最优平衡：过短的音频切片会导致模型缺乏必要的声学上下文 (Acoustic Context)，从而因归纳偏置(Inductive Bias)不足急剧增加字词错误率 (WER/CER)；而等待过长的切片则会引入显著的缓冲延迟，削弱流式处理的优势。为此，本节提出一种基于滑动窗口的自适应上下文管理策略。  

### 3.2.1 动态 VAD 分段策略  

系统采用 Silero VAD 模型对连续 PCM 音频流进行在线分段 [14]。形式化地，定义输入音频流为时间序列 $X=\{x_1,x_2,\dots,x_t\}$，分段器维护累计缓冲区 $A_{accum}$，并在每次接收新的音频块 $chunk_t$ 后执行 $A_{accum}\leftarrow A_{accum}\oplus chunk_t$，其中 $\oplus$ 表示拼接，$|A_{accum}|$ 表示缓冲区的采样点数。当累计长度超过最小检测窗口（本文实现取 64ms）后，系统调用 Silero VAD 的时间戳提取函数在 $A_{accum}$ 上返回一组语音区间 $\{(s_i,e_i)\}$，其中 $s_i,e_i$ 为采样点索引。设采样率为 $f_s$，则将第 $i$ 个语音区间记为 $seg_i=(s_i,e_i)$，其有效语音时长为 $Len(seg_i)=1000\cdot\frac{e_i-s_i}{f_s}$（ms）；段后静音时长定义为 $L_{silence}(i)=1000\cdot\frac{|A_{accum}|-e_i}{f_s}$（或相邻区间间隔 $1000\cdot\frac{s_{i+1}-e_i}{f_s}$）。当 $L_{silence}(i)\ge T_{min\_silence}$ 时认为该语音段已“闭合”。为便于下游 ASR 对齐，本实现输出音频片段 $A_{accum}[0:e_i]$（可包含少量前导静音/上下文），并将缓冲裁剪为 $A_{accum}\leftarrow A_{accum}[e_i:]$；同时更新累计的绝对时间起点，以便为下游 ASR 提供可对齐的起止时间。

从实现角度看，分段触发条件可以抽象为静音约束与最小语音长度约束的组合：

$$\text{FlushTrigger}(i)=\mathbb{I}\left[L_{silence}(i)\ge T_{min\_silence}\right]\cdot \mathbb{I}\left[Len(seg_i)\ge T_{min\_speech}\right]$$

其中，$T_{min\_silence}$、$T_{min\_speech}$ 与 VAD 的阈值 $\theta_{threshold}$ 共同决定分段的敏感度：$T_{min\_speech}$ 过小会引入噪声脉冲段，$T_{min\_silence}$ 过小则会将连贯语音过度切碎。本文原型系统的默认配置为 $\theta_{threshold}=0.5$、$T_{min\_speech}=500\text{ms}$、$T_{min\_silence}=300\text{ms}$，以在实时性与稳定性之间取得折中。

### 3.2.2 基于滑动窗口的上下文感知 (Context-Aware Sliding Window)

为了解决流式识别中的边界效应，ASR 引擎引入了包含“前缀上下文” (Prefix Context) 的滑动窗口机制。

![[图3.2.png]]

**图 3-2 上下文感知 ASR 滑动窗口机制示意图** [图示详细解析了时间步 $t$ 时 ASR 模型的输入构成。窗口被划分为三个逻辑区域：**前缀上下文 (Prefix Context)** 为历史已确定的冻结区域，提供必要的声学先验信息；**当前片段 (Current Segments)** 为待识别的活跃推理区域；**后缀缓冲 (Suffix Buffer)** 则包含未来音频信息以防止截断效应。系统通过这种重叠切分 (Overlapping) 策略，确保了流式识别结果的连贯性]

该机制包含三个关键步骤：

首先，执行窗口更新。设 $Q_t$ 为时刻 $t$ 的音频段队列 (Segment Queue)。当新的音频段 $seg_{new}$ 到达时，系统先将其加入等待队列，并在转录线程中批量并入主队列得到 $Q_{temp}$；当队列累计时长满足触发阈值 $T_{asr}$，且队列长度满足 $|Q_{temp}|\ge N_{prefix}+N_{suffix}+1$(或收到最终段)时，ASR 模型 $M$ 对拼接后的音频序列执行一次推断，生成原始转录结果 $T_{raw}=M(\text{Concat}(Q_{temp}))$。在工程实现中，$T_{asr}$ 由 `recognition_threshold` 控制，用于避免过短窗口造成的频繁调用与不稳定输出。

其次，进行确定性文本提取 (Deterministic Text Extraction)。鉴于 Whisper 等 Encoder-Decoder 模型在处理流式输入的末尾时往往存在“闪烁” (Flickering) 现象——即随着新音频的输入，原本输出的末尾词汇可能发生变化——系统在输出侧加入“后缀保护”的稳定性约束。定义 $N_{prefix}$ 为前缀段数，$N_{suffix}$ 为后缀保护段数，系统仅输出稳定区域 (Stable Region) 内的文本。为保证输出与音频段一一对应，本文实现利用词级时间戳将 $T_{raw}$ 中的词按结束时间映射回每个音频段的时间区间，从而得到段级候选文本；随后仅拼接稳定区域内段的文本形成本轮输出 $T_{out}$。稳定区域的队列索引范围定义为：

$$I_{stable}=\{\, i \mid N_{prefix}\le i < |Q_{temp}|-N_{suffix}\,\}$$

其中 $i$ 为队列索引（从 0 开始），$|Q|$ 表示队列长度（段数），稳定区采用半开区间定义以避免边界歧义。在工程实现中，系统还对流式起止边界做了特殊处理：在流式起始轮(首段标记为 `is_start`)时，前缀段尚未输出过，稳定区的起点可从 0 开始；在流式结束轮(`is_final`)时，为避免遗留文本，系统会输出所有剩余段并完成收敛。最后，执行缓冲区状态迁移 (Buffer Transition)。在输出 $T_{out}$ 后，窗口执行滑动以限制累计长度并维持声学连贯性：系统不会将队列全部清空，而是保留最后一个已输出段之前的 $N_{prefix}$ 个段作为下一轮的前缀上下文，同时继续保留尚未输出的后缀保护段。设本轮输出的最后一个段索引为 $i_{last}$，则更新后的队列可写为：

$$Q_t = Q_{temp}[j:],\quad j=\max(0, i_{last}+1-N_{prefix})$$

记 $Q[a:b)$ 为队列 $Q$ 的子序列（含 $a$ 不含 $b$），$Q[a:]$ 为从 $a$ 到末尾的后缀。需要指出的是，稳定输出的代价并非“仅增加一个 chunk”的固定延迟，而是至少需要等待 $N_{suffix}$ 个后续段到达、并满足触发阈值 $T_{asr}$；但该代价换来了输出文本在边界处的可提交性，使下游 LLM 能够稳定地进行增量预填充，从系统层面实现端到端延迟的可控下降。

## 3.3 LLM KV Cache 增量预填充策略 (Incremental KV Cache Prefilling)

在 System A （非流式级联系统）中，LLM 必须等待 ASR 完整转录结束后才能获得完整 prompt，并在 End-of-Speech 之后一次性完成预填充与首步解码，因此 TTFT 会随转录文本长度增长而显著增加。更进一步的说，如果直接将此范式迁移到”边听边算“的增量输入场景且不复用历史状态，每当 ASR 文本出现增量更新，系统往往需对增长后的 prompt 重新执行一次预填充，造成大量重复计算与不必要的等待。

为降低 End-of-Speech 后的集中计算并支撑流水线并行，System B在 LLM 侧引入了增量预填充 ：将 Transformer 解码器在预填充阶段产生的 past_key_values 作为 KV Cache 持久化保存在内存或 GPU 的 VRAM中，并在新文本片段抵达时只对新增 token 做前向计算，并同步维护 attention mask 与 position ids。这样，LLM 的大部分增量预填可以在用户说话期间完成；当终止标记到达时，系统只需处理末尾片段并立即启动生成回复 token 的计算工作，从而压缩语音结束到首token输出的等待。

### 3.3.1 核心算法实现

增量推理的核心逻辑封装在 StreamLLMInference 类中：首次调用时，系统在无缓存状态下一次性执行增量预填充，建立初始 KV Cache；随后，每当上游传来一段新的 ASR 文本片段，系统便触发一次调用 cache_prompt： 在已有 past_key_values 的基础上做增量更新。当收到终止标记时，系统在 prompt 末尾追加生成提示符（常见的比如："assistant:"，用于告诉模型后续输出的是助理的话），并复用预填充阶段最后一步的 logits 直接开始解码。为避免增量 token 的位置编码错位，实现里会显式构造 position_ids，同时将新增片段以 add_special_tokens=False 方式编码，确保缓存拼接可控。Algorithm 1 给出了与该实现对应的增量更新过程。

**Algorithm 1: Incremental KV Cache Prefill Strategy**

```latex

  

\begin{algorithm}

  

\caption{Incremental KV Cache Prefill (StreamLLMInference)}

  

\begin{algorithmic}[1]

  

\REQUIRE Current KV cache $C_{prev}=(P_{prev},Mask_{prev},L_{prev})$, New Text Fragment $T_{new}$, Is End Flag $IsEnd$, Tokenizer $\mathcal{T}$, LLM Model $\mathcal{M}$

  

\ENSURE Updated cache $C_{new}=(P_{new},Mask_{new},L_{new})$

  

  

\STATE \textbf{Step 1: Tokenization (Streaming Fragment)}

  

\STATE $Ids_{new} \leftarrow \mathcal{T}(T_{new}, \texttt{add\_special\_tokens}=False)$

  

\IF{$Ids_{new}$ is empty}

  

\RETURN $C_{prev}$

  

\ENDIF

  

  

\STATE \textbf{Step 2: Attention Mask \& Position IDs}

  

\STATE $Mask_{new} \leftarrow \text{Concat}(Mask_{prev},\mathbf{1}_{|Ids_{new}|})$

  

\STATE $Pos \leftarrow [|Mask_{prev}|, \dots, |Mask_{new}|-1]$ \COMMENT{explicit \texttt{position\_ids}}

  

  

\STATE \textbf{Step 3: Forward Pass (Incremental Update)}

  

\STATE \COMMENT{Forward on new tokens with cached $P_{prev}$}

  

\STATE $Outputs \leftarrow \mathcal{M}(Ids_{new},Mask_{new},Pos,P_{prev},\texttt{use\_cache}=True)$

  

  

\STATE \textbf{Step 4: Cache \& Logits Update}

  

\STATE $P_{new} \leftarrow Outputs.\texttt{past\_key\_values}$

  

\STATE $L_{new} \leftarrow Outputs.\texttt{logits}[:, -1, :]$

  

\STATE \COMMENT{If $IsEnd=True$, decode the first token from $L_{new}$ without extra forward}

  

\STATE $C_{new} \leftarrow (P_{new},Mask_{new},L_{new})$

  

\STATE \COMMENT{End of update}

  

  

\RETURN $C_{new}$

  

\end{algorithmic}

  

\end{algorithm}

  

```

![[图3.3.png]]

**图 3-3 LLM KV Cache 增量更新机制示意图** 图中直观展示了显存内的状态变化。左侧蓝色区域表示历史上下文对应的缓存键值对，右侧绿色区域表示新增片段产生的键值对。增量预填充在新片段到达时复用历史 `past_key_values`，仅对新增 token 执行前向并更新缓存，从而避免在增量场景下反复对全量 prompt 进行 prefill。

### 3.3.2 复杂度分析 (Complexity Analysis)

为了说明增量预填充减少冗余计算的来源，考虑历史上下文长度为 $N$、新增片段长度为 $M$ 的一次更新。若在每次增量到达时都对全序列 $L=N+M$ 重新执行一次 prefill(不复用历史状态)，则 Self-Attention 的主要开销近似为：

$$O((N+M)^2)$$

而在使用 KV Cache 的情况下，历史 token 的 $K,V$ 已经缓存，系统仅需对新增 token 计算 Query 并与长度为 $N+M$ 的 Key/Value 交互，主要开销可写为：

$$O(M\cdot(N+M))$$

当 $M\ll N$ 时，上式近似为 $O(N\cdot M)$，体现为“对新增片段的线性扩展”。需要强调的是，若将所有增量片段累积到最终长度 $L$，增量预填充的累计开销为 $\sum_j O(M_j\cdot L_j)\approx O(L^2)$，与离线一次性 prefill 同阶；本文策略的关键收益不在于改变复杂度阶，而在于两点：其一，避免 naive streaming 中“每次更新都重算全量 prompt”的重复工作；其二，将 prefill 计算前移到用户发声期间，使 End-of-Speech 之后需要完成的计算主要集中在最后一小段增量及线程同步开销，从而有效降低 TTFT。与此同时，本文实现会在 `KVCache` 中保留预填充阶段最后一步 logits，使得终止标记到达后无需额外 forward 即可解码首 token，进一步减少首字阶段的常数开销。

## 3.4 实验数据构建与处理管线 (Data Construction Methodology)

为了在受控环境下系统性验证流式架构的性能，需要有不断递增长度的对话数据，本研究构建了一套自动化的数据合成管线。该管线将开源数据集： MultiWOZ [15] (英文) 和 CrossWOZ [16] (中文) 这两个标准多轮对话数据集转化为具有精确时长标注的逐渐递增的长语音测试数据，再通过tts生成音频数据。

### 3.4.1累积对话生成逻辑 (Cumulative Dialogue Strategy)

为了模拟用户进行长语音输入的真实场景(例如用户一口气陈述复杂的旅行需求，或在对话中不断补充信息)，本实验采用了“累积对话”策略。如表 3-1 所示，算法遍历对话历史，将多轮对话中的用户侧(User)与上一轮的用户+系统回复文本进行累积拼接，从而构造出长度递增的输入序列。

| Turn Index | User       | System      | Cumulative Input Text (Simulated User Speech) |
| :--------- | :--------- | :---------- | :-------------------------------------------|
| 1          | "订一张机票"    | "去哪里？"      | "订一张机票" |
| 2          | "去北京，明天出发" | "要哪种舱位呢？"   | "订一张机票，去哪里？，去北京，明天出发"                         |
| 3          | "要商务舱"     | "好的，这就为您预订" | "订一张机票，去哪里？，去北京，明天出发。要哪种舱位呢？要商务舱"             |

通过此策略，我们能够基于真实的语义语境，生成从 3 秒至 60 秒以上不等的连续语音样本，有效覆盖了从短指令到长篇陈述的各种交互形态，解决了传统数据集缺乏长语音样本的问题。

### 3.4.2 数据处理流水线 (Processing Pipeline)

数据构建过程包含三个严格顺序执行的阶段(Phases)：

首先是历史累积与筛选(Accumulation & Filtering)。策略遍历源数据集，应用上述累积生成逻辑。为了聚焦长语音场景下的性能瓶颈，系统按文本长度对生成样本进行倒序排列，优先选取最长的前 $N$ 个对话片段，并设置最大文本长度阈值(英文 2050 字符，中文 720 字符)以防止显存溢出 (OOM)。

其次是并发 TTS 音频合成 (Batch TTS Synthesis)。本研究集成了阿里巴巴提出的 CosyVoice 大模型语音合成服务 [17]。相较于传统 TTS，CosyVoice 能够生成韵律更自然、情感更丰富的高保真语音波形，更贴近真实人声输入。在工程实现上，我们开发了批量处理模块，采用多线程异步请求机制，显著提升了大规模数据生成的效率。

最后是时长校准与元数据同步 (Duration Calibration)。由于生成式 TTS 模型的语速具有非确定性 (Non-deterministic)，简单的基于文本字数的时长估算往往存在误差。本管线通过解析生成的 WAV 文件头(Header)来获取以秒为单位（可精确到毫秒级）的物理时长：$Duration = \frac{TotalFrames}{SampleRate}$。该真实时长 ($audio\_duration$) 被回写至测试元数据 JSON 中，作为后续实验中 $X$ 轴（输入时长）的真值依据 (Ground Truth)。

### 3.4.3 测试集分组定义 (Benchmark Grouping)

为了细粒度分析不同时长下的延迟表现，我们将生成的样本集依据 $AudioDuration$ 划分为四个标准实验组(见表 3-2)。

[表 3-2: 实验数据分组定义]

| Group Name | Duration Range    | Typical Use Case     | Experiment Focus             |
| :--------- | :---------------- | :------------------- | :---------------------------|
| Short      | $T < 5s$          | 短指令 (Short Commands) | 基准延迟验证 (Baseline Latency)    |
| Medium     | $5s \le T < 15s$  | 包含多轮意图的陈述            | 日常对话性能 (Daily Conversation)  |
| Long       | $15s \le T < 30s$ | 复杂长难句/长段落            | 流式架构核心优势区间                   |
| Extra Long | $T \ge 30s$       | 极限压力测试               | 系统稳定性与显存边界 (Stability & OOM) |

这一分组标准将在第四章的实验分析中贯穿始终，用于对比不同系统架构在处理长短语音时的性能差异。

# 第四章 实验与结果分析 (Experiments & Analysis)

本章围绕“VAD 判定用户语音结束至大语言模型产生首个响应 Token”的首字延迟（Time-to-First-Token, TTFT）开展实验验证，系统性评估本文提出的流式级联式语音对话架构（System B）在长语音交互场景中的性能收益与代价。与传统非流式基线（System A）相比，System B 的关键差异在于：其一，将 ASR 从端点后全量解码改造为分片流式解码，使声学计算与用户发声时长发生重叠；其二，在 ASR 增量输出的推动下，对 LLM 进行增量预计算，以削弱长上下文一次预填充带来的串行等待。通过控制变量，本章依次从延迟特性、关键模块消融以及精度-延迟权衡三方面给出实证结果，并讨论观察到的“主导项切换”现象及其工程边界。

## 4.1 实验一：延迟与语音长度的关系验证 (Effect Validation)

### 4.1.1 实验设置与环境构建

为了客观衡量系统在长语音场景下的实时响应能力，本实验选取 MultiWOZ 与 CrossWOZ 两个公开数据集，分别代表英文与中文的多轮对话任务。按 3.4 小节所述的数据构建流程，我们生成了覆盖短、中、长、很长、超长五组长度的混合测试集，共 1132 条样本。  

实验硬件平台统一基于 两块 NVIDIA RTX 3090 （24GB VRAM）显卡的平台上进行，ASR 模型和 LLM 模型分别在两块显卡中运行。该环境代表了消费级工作站的典型算力水平。为了消除模型加载与初始化带来的随机误差，我们对对比组（System A）与实验组（System B）实施了严格的公平性控制： 首先，两组实验均复用同一套模型权重，即 Whisper-Turbo [7] 作为声学编码器，以及 Qwen2-7B-Instruct [18] 作为语义生成器，排除了参数量差异带来的干扰。其次，引入“热身机制”，在记录数据前预先执行3轮真实音频推理，以确保 CUDA Context 完成初始化且JIT编译器已优化核心算子。 在流式配置中，音频切片（Chunk Duration）锁定为 500ms，以模拟真实对话中的标准包发送频率。

### 4.1.2 延迟趋势分析

首字延迟（Time-to-First-Token, TTFT）是衡量语音交互流畅度的核心指标。图 4-1 展示了在不同语音时长输入下，非流式基线系统与本文流式系统的 TTFT 变化轨迹。

  ![[Pasted image 20251216110152.png]]

图 4-1 不同语音时长下 System A 与 System B 的 TTFT 变化趋势。
*(图注：X轴为音频真实时长 $audio\_duration$(秒)，Y轴为端到端 TTFT（毫秒），统计范围覆盖 1132 条样本。System A 曲线随时长近似单调上升，反映端点后串行等待的累积；System B 在 long 及以上区间趋于平缓，呈现由固定分片粒度与尾部调度开销主导的近似上界。两条曲线在 $T \approx 15s$ 附近出现拐点，意味着流式并行的覆盖收益开始超过其固定开销。)*

| group | sample<br/>count | avg_duration<br/>(s) | streaming_ttft<br/>(mean_ms) | non_streaming_ttft<br/>(mean_ms) | improvement<br/>(ms) | improvement<br/>(ratio_%) |
|------|----------|------------------|-------------------|---------------------|-------------|------------|
| extra_long | 679 | 105.73 | 1087.70 | 6753.43 | 5665.73 | 83.9 |
| very_long | 208 | 45.21 | 1099.16 | 3191.97 | 2092.81 | 65.6 |
| long | 121 | 22.16 | 1126.63 | 1722.04 | 595.41 | 34.6 |
| medium | 89 | 9.25 | 959.53 | 923.42 | -36.11 | -3.9 |
| short | 35 | 3.44 | 648.17 | 533.42 | -114.75 | -21.5 |

[表 4-1: 不同语音时长分组下的 TTFT 统计]

表 4-1 给出了按语音时长分组的 TTFT 统计结果。非流式基线 System A 的 TTFT 随输入时长显著增长：short 组均值为 533.42ms，而进入 long 组后上升至 1722.04ms，并在 very\_long 与 extra\_long 组进一步增长到 3191.97ms 与 6753.43ms。该现象与级联式处理范式一致，即 ASR 需要等待端点后完成全量解码，随后 LLM 对完整转录文本执行一次性 Prefill；在此过程中，端点后的计算链路几乎不可重叠，导致等待随语音长度近似线性累积 [5, 10]。

相比之下，流式系统 System B 在 long 及以上区间表现出近似“常数上界”：long、very_long 与 extra_long 三组的 TTFT 均值依次为： 1126.63ms、1099.16ms 与 1087.70ms，整体稳定在约 1.1s 量级。对应地，其在 long/very_long/extra_long 三组分别把 TTFT 压缩了 34.6%、65.6% 与 83.9%，其中 extra_long 组平均绝对延迟降低了 5.67s，显著缓解长语音交互中的静默等待。需要指出的是，这里的“常数”是指主导项由“随时长累积的全量计算”转为“最后若干分片的尾部处理与调度开销”之和；在固定分片时长 500 ms 与资源调度策略不变的前提下，TTFT 不再随输入时长继续增大。

与此同时，short 与 medium 组在 System B 下分别录得 114.75 ms 与 36.11 ms 的负向改进，表明在短语音场景里，分片、缓存维护及增量提示构造等固定开销仍难被并行收益抵消。该结果提示工程落地时宜采用自适应策略：对预测时长较短的输入优先走传统的一次性处理链路，而将流式并行机制用于处理音频 T ≥ 15s 的优势区间，以换取更稳定的整体交互体验。

## 4.2 实验二：核心模块消融实验 (Ablation Study)

为量化“流式 ASR”与“LLM 增量预填充”两项机制的相对贡献，本节在 long 及以上长度的样本上开展消融实验。按照 4.1 节所描述的分组，我们重点考察 long（15–30s）、very_long（30–60s） 与 extra_long（≥60s)）三个分组，因为在该区间端点后的串行等待最为突出，也最能体现流式改造的收益。对比配置设置为三类：Baseline（System A，ASR 与 LLM 均为非流式串行执行）、Streaming ASR Only（仅 ASR 流式化，LLM 仍在最终转录完成后一次性预填充）以及 Full Streaming（在 Streaming ASR Only 基础上启用 KV Cache 状态保持与增量预填充，对应 System B）。  

### 4.2.1 评价指标与对比组  

本实验以 TTFT 作为核心指标，并在日志层面记录端点后的 ASR 尾部耗时与 LLM Prefill 耗时用于瓶颈诊断。为突出架构差异对用户感知延迟的影响，本文在表 4-2 中主要报告三种配置的 TTFT 均值以及由此计算的 ASR 增益与 KV 增益(正值表示额外缩短 TTFT，负值表示引入额外开销)，从而保证结论可被归因于目标机制本身。

### 4.2.2 贡献度量化分析  

表 4-2 列出了各分组下三种配置的 TTFT 均值及对应增益，用于刻画瓶颈在流式化过程中的迁移。

| 组别 | 平均持续时间(秒) | 基线TTFT均值(毫秒) | 流式ASR TTFT均值(毫秒) | 全流式TTFT均值(毫秒) | ASR增益(毫秒) | KV增益(毫秒) | 
|------|------------------|-------------------|-----------------------|---------------------|-------------|-------------|
| long | 21.51 | 1698.88 | 1064.18 | 1084.17 | 634.70 | -19.99 |
| very_long | 42.43 | 3300.58 | 1171.02 | 1154.06 | 2129.56 | 16.96 | 
| extra_long | 83.75 | 6518.40 | 1228.77 | 1114.57 | 5289.63 | 114.20 | 

[表 4-2: 核心模块消融实验结果]

由表 4-2 可见，在实验平台硬件及所用模型条件下，ASR 流式化是缩短 TTFT 的主因。Baseline 与 Streaming ASR Only 相比，在 long、very_long 与 extra_long 三组分别减少 634.70ms、2129.56ms 与 5289.63ms，对应降幅约为 37.4%、64.5% 与 81.1%。这表明，在传统串行链路中，端点后的主要等待来自 ASR 对末尾分片的解码、稳定输出与端点收敛；而流式 ASR 把大部分声学计算提前并与语音时长重叠，显著压缩了端点后的剩余工作量。 LLM 侧的增量 Prefill 体现出明显的“长度条件性”。在 long 组，Full Streaming 相比 Streaming ASR Only 的 TTFT 增加 19.99ms(KV 增益为 -19.99ms)，说明当上下文尚未足够长时，增量调用、缓存管理与同步开销可能超过其节省的一次性 Prefill 计算；而在 very\_long 与 extra\_long 组，KV 增益分别为 16.96ms 与 114.20ms，且在 extra\_long 组的相对收益达到约 9.3%（114.20/1228.77），提示随着输入文本与上下文长度增长，LLM Prefill 的成本上升使得增量预填充更具性价比。  

综合两项机制，Full Streaming 将 TTFT 均值稳定在约 1.1s（long 1084.17ms、very_long 1154.06ms、extra_long 1114.57ms），相较 Baseline 分别缩减约 36.2 %、65.0 % 与 82.9 %。该结果与实验总体趋势一致：流式并行显著削弱级联系统随输入时长增加的串行等待，但 KV Cache 预填充的边际收益仍受工程实现与调用粒度制约。消融实验显示，在实验平台 RTX 3090 的硬件条件下，运行 Whisper 模型已略显吃力。而 Qwen2-7B-Instruct 在LLM领域可以算是小型模型，在实验平台的硬件条件下可以轻松应对。如果换为主流SOTA模型（100B+），相信流式在 KV Cache Prefill 的收益将会更加明显。另外值得注意的是，音频的长度与ASR后的字符数并非严格符合线性关系，在对实验数据进行进一步的研究后发现KV Cache Prefill 的收益与ASR转录后的文本长度才明显符合线性关系，这说明KV Cache Prefill 的效果基本符合预期，即文本越长，流式效果越明显。

## 4.3 实验三：准确率与质量边界 (Accuracy & Quality)

延迟被大幅压短后，流式化的主要代价落在“上下文截断”与“端点不确定性”上。与 System A 可利用整段音频不同，流式 ASR 在每个输入的语音片段上只能看到整段音频局部的片段，句末边界附近因此更易出现替换、插入或省略。为引入稳定的识别结果，本文在 ASR 侧采用滑动窗口上下文管理，通过前缀（prefix）提供音频，通过后缀（suffix）提供后续音频，在精度与耗时之间进行折中。借助该机制，本实验对比三种配置：prefix=1, suffix=0（默认）、prefix=1, suffix=1（增加 1 段前瞻以改善句尾效果）以及 prefix=0, suffix=0（完全去除上下文以追求最小延迟），并从 medium、long 与 very_long 组各抽取 50 条样本，共 150 条，用于覆盖中长语音的主要应用场景。  

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
  
(注：每种上下文配置均包含 150 条样本，其中 MultiWOZ 74 条、CrossWOZ 76 条。由于评测音频由文本经 TTS 合成生成，参考文本可视为该批样本的近似真值；同时，System A 在该合成集合上的 WER/CER 为 0，作为上界对照。本文主要关注不同上下文配置下的相对差异，不将该结果直接外推为真实录音场景的绝对误差。)  

从表 4-3 可以看出，增加后缀前瞻（pre1&suf1）对中文句尾稳定性具有更直接的收益：CrossWOZ 的 WER/CER 分别由默认配置的 0.0118/0.0292 降至 0.0083/0.0203。代价是端点后的尾部耗时上升，asr_time_ms 由 1224.96ms 增至 1327.48ms（+102.52ms），这符合后缀窗口需要等待额外音频片段并完成更多解码计算的机制预期。对于英文 MultiWOZ，suffix 的引入使 CER 由 0.0221 降至 0.0196，但 WER 略升至 0.0895，提示在以词为单位的评价下，后缀窗口对对齐与分词的影响可能更敏感；该现象需要在真实录音与更大样本上进一步复核。当完全去除上下文（pre0&suf0）时，系统获得了更低的尾部耗时（asr_time_ms 下降至 1086.16ms，相比默认减少 138.80ms），但中文误差出现明显放大：CrossWOZ 的 CER 上升至 0.0500，WER 上升至 0.0210。该结果表明，prefix 提供的历史上下文在流式识别中起到了重要的约束作用，可为声学-语言模型的局部决策提供更强的归纳偏置，抑制长句中逐步累积的替换与漂移误差。  

由于中文是单字结构，而英文是单词组合，因此在中文数据集上我们主要考察的是 CER 也就是字符的错误率，而英文数据集主要考察的是 WER 即词错误率。从中文数据集上看，随着上下文的扩大，CER有明显降低；而在英文数据集上，三个配置的 WER 可以说基本一致，最小的上下文有最低的 WER 可以视为是实验的波动。这主要是因为中文的同音字较多，更多的上下文对于转录出准确的文字有较大帮助；而英文单词的同音词汇就少得多，因此只要发音准确，词级识别就相对稳定，增加的上下文并未显著提升准确率，因此在实际部署时，英文对话可去掉前后缀扩展，把流式架构的延迟优势压到极限。

最后从端到端交互预算看， 实验一与实验二显示 long 及以上语音的 TTFT 已可稳定压缩至约 1.1s，而表 4-3 中不同上下文配置带来的 0.1–0.2s 尾部耗时已占去可观比例，因此需按场景进行取舍：对中文任务中识别正确性更敏感者，可启用 pre1&f1 以降低句尾错误；pre0&suf0 则作为极低延迟模式，或用于英文对话。考虑到对话式 LLM 在语义层面通常对噪声具有一定鲁棒性，转录文本的小误差预计对 LLM 的推理不会造成明显影响。后续可引入语义一致性与任务完成率等指标，直接量化“转写误差是否造成语义漂移”的数值。  

# 第五章 总结与展望 (Conclusion & Future Work)  

## 5.1 全文总结(Summary of Contributions)  

本文聚焦人机语音交互领域一个悬而未决的核心矛盾：大语言模型推理成本持续攀升，而实时对话系统对响应延迟的要求却愈发苛刻。为此，设计并实现了一套基于细粒度流水线并行的低延迟语音对话架构。回顾全文 ，本研究围绕“计算流式化”与“状态增量化”两条技术主线展开，通过系统性的架构重构，消除了传统级联系统(Cascaded Systems)中的串行阻塞瓶颈。  

首先，针对感知层面的延迟，本文提出一种自适应流式 ASR 上下文管理机制。不同于 Whisper 的离线推理流程 [7] 在端点到达后一次性解码整句语音，本研究借助滑动窗口把离线 ASR 封装成可持续输出增量转录的流式引擎，在“切分粒度”与“语义完整性”之间取得平衡，使用户发声的同时即可获得高置信度的文本片段。其次，针对认知层面推理计算的冗余，本文提出一种基于LLM的增量 KV Cache 预填充算法。在传统架构里，ASR每输出一次文本更新，Transformer 模型[5]就得把越来越长的历史prompt重新编码一遍，端点后的预填充开销因此随输入长度线性攀升。本研究利用自回归解码器可复用 Key/Value 缓存的特点，持续维护KV Cache，仅对新增的文本片段做增量更新，从而将原本集中在 End-of-Speech 之后的大规模预填充提前，与用户的发声同步进行。需要指出的是，这一策略并未改变注意力计算的理论二次复杂度，却可削减重复运算，显著缩短端点后的 TTFT，尤其在长语音场景下抑制 TTFT 随输入长度上升的趋势。  

实验结果验证了该架构的有效性。在基于 MultiWOZ 与 CrossWOZ 构建的长语音评测基准上，System B 相比传统非流式基线 SystemA 表现出明显优势。尤其在 15 秒以上的长语音分组中，System B 大幅缩短了端点后等待时间，同时维持了识别准确（WER/CER）处于可接受范围的同时提升了交互流畅度与沉浸感。需要指出的是，在当前实验平台上，long 及以上分组的 TTFT 仍约为 1.1s，与人类对话中约 200ms 的话轮间隙仍有差距；但与传统串行方式相比，System B 在长语音场景下的 TTFT 在达到一定数值后不再随着语音时长增长。未未来可借助更高效的推理硬件、更低延迟的声学模型及进一步调度优化 ，继续压缩这段端点后等待时间。

## 5.2 研究局限性 (Limitations)

尽管本研究在降低交互延迟方面取得了显著进展，且所提出的流式架构在工程实践中表现出了良好的鲁棒性，但受限于级联架构本身的结构性约束，系统在迈向“类人自然对话”的终极目标上仍面临以下局限：  

首先是半双工交互模式(Half-duplex Interaction)带来的体验割裂。当前系统设计虽然优化了响应速度，但本质上仍遵循“用户发言 $\rightarrow$ 系统处理 $\rightarrow$ 系统回复”的半双工话轮转换(Turn-taking)逻辑。由于现有的 VAD 模块缺乏对用户“打断(Barge-in)”意图的高层语义理解，当系统处于生成或播放语音的状态时，往往难以精准捕捉并响应用户的插话行为。这种对交互状态的刚性锁定，使得人机对话在灵活性上仍落后于人类自然的“重叠对话(Overlapping Speech)”体验。其次是级联架构导致的副语言信息丢失(Loss of Paralinguistic Information)。作为典型的级联系统，本研究采用“文本”作为 ASR 感知模块与 LLM 认知模块之间的唯一信息接口。这种离散化的符号接口虽然简化了工程实现，但也造成了不可忽视的信息有损压缩(Lossy Compression)。人类语音中蕴含的丰富副语言特征——如说话人的情绪波动(愤怒、犹豫)、语调变化(疑问、反讽)以及微小的停顿节奏——在强制转录为文本的过程中被不可逆地丢弃。由于 LLM 无法直接感知原始声学特征，生成的回复往往局限于字面语义的正确性，而缺乏情感上的共鸣与适配，呈现出“理智但冷漠”的交互缺陷。  

## 5.3 未来展望 (Future Work)

尽管本文在长语音场景下显著压缩了 TTFT，并验证了流式并行与状态增量化的总体有效性，但在真实交互中，用户体验往往受“长尾延迟”“保守端点判定”以及“可打断性不足”等因素的共同制约。结合实验一的明细日志与系统实现细节，未来工作将围绕尾部延迟抖动抑制、语义结束判定以及面向打断的回复预生成三条主线展开，以进一步逼近自然对话所要求的低等待与高鲁棒性。

### 5.3.1 尾部分片波动与 TTFT 长尾抑制

观察实验一的明细数据可以发现，尽管 System B 在长语音分组下的 TTFT 均值大约是 1.1s，但仍存在少量样本出现 2s 以上的峰值波动。进一步分析表明，该现象并非来自模型计算复杂度的突增，而是末端音频分段的到达时间随机波动与调度策略叠加所致：当倒数第二个音频段刚进入处理流程时，最后一个音频段可能在不足 500ms 的间隔内到达，由于是最后一段音频，因此系统需要对队列进行再一次推理，由于这一过程是串行的，因此需要等待上一次处理结束后才能开始。由于单次音频段处理的耗时本身约为 1s，短时间内触发两次串行推理会直接造成 TTFT 的近似2倍的增长， 由此带来拖慢交互的长尾延迟。

面向该问题，后续可以从调度与算法两侧进行联合优化。调度层面可引入“末端守护时间”与自适应合并策略：当检测到分段间隔显著小于单段处理时间且端点置信度较高时，优先延迟一次触发并合并末端分段，以减少重复推理；同时结合队列回压与异步化处理，将末端突发的分段到达吸收为可控的调度开销。

### 5.3.2 基于语义的结束判定与停顿等待压缩

当前原型系统的测试音频是通过完整音频模拟，因此可以实时知晓音频的结束时间。但是在真实的对话中，一般是通过静音阈值完成话轮结束判定，例如用户停止发声后需等待约 1s 的停顿窗口，系统才将输入视为结束并进入回复阶段。该策略在噪声环境下具有工程合理性，但当 TTFT 已被压缩到 1s 量级后，固定的停顿等待反而成为新的主导项，限制了进一步降低交互等待的上界。未来工作将研究一种轻量级的语义结束判别模型（End-of-Utterance, EOU），以流式 ASR 的增量文本为主要输入，结合对话上下文与局部语言结构特征，预测当前话轮是否已语义完结，从而在保证鲁棒性的前提下缩短停顿等待。

在系统集成层面， 更可行的做法是让 EOU 判别与静音阈值互为补充，一旦模型给出高置信度的结束信号，便提前触发回复； 若置信度偏低或环境噪声使文本摇摆不定， 则退回保守的停顿判定， 借此抑制误判带来的抢话与语义截断风险。

### 5.3.3 回复预生成、语义影响判别与打断机制

在端点判定能够更早触发的基础上，系统可进一步将 LLM 的部分生成计算前移：只要判断输入大概率已结束，模型就抢先起草回复，让用户的停顿间隙与推理过程重叠 ，系统可进一步将LLM 的部分生成计算前移：一旦判断输入大概率结束 ，LLM便提前开始推理产生回复 token ，使得用户停顿窗口与模型推理时间发生重叠。然而真实对话具有显著的可打断性 ，用户可能在系统准备或播放回复时继续补充信息。若系统简单丢弃之前的回复内容并完全重算 ，将同时造成计算浪费与额外等待 ，因此有必要建立“可提前计算、可随时撤销” 的统一机制。

后续计划嵌入轻量级“语义影响判别”模型，新增输入一旦到达便在线评估其是否扰动已有草稿的核心意图或约束；若扰动极小，之前的回复直接送交输出，否则回退并增量重生成。为此，大模型端须构建可回退的推理状态管理：为“已确认上下文”保留主分支 KV Cache，同时为预生成的草稿维护可丢弃的投机分支，使系统具备瞬时切换能力。用户继续说话时，系统立即撤销尚未提交的中间状态，并在必要时以最小开销重建提示，再续写生成。更进一步的话，为支持从半双工向更自然的交互过渡，还需完善生成与合成过程的可打断机制， 包括对 LLM 解码与 TTS 合成的抢占式取消、已播放语音的即时停止与淡出策略， 以及对对话状态一致性的维护。  

---

  

**References**

[1] OpenAI, "GPT-4o System Card," *arXiv preprint arXiv:2410.21276*, 2024. [Online]. Available: https://arxiv.org/abs/2410.21276

[2] Gemini Team, "Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context," *arXiv preprint arXiv:2403.05530*, 2024. [Online]. Available: https://arxiv.org/abs/2403.05530

[3] H. Sacks, E. A. Schegloff, and G. Jefferson, "A Simplest Systematics for the Organization of Turn-Taking for Conversation," *Language*, vol. 50, no. 4, pp. 696–735, 1974, doi:10.2307/412243.

[4] A. Graves, "Sequence Transduction with Recurrent Neural Networks," *arXiv preprint arXiv:1211.3711*, 2012. [Online]. Available: https://arxiv.org/abs/1211.3711

[5] A. Vaswani *et al.*, "Attention Is All You Need," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017. [Online]. Available: https://arxiv.org/abs/1706.03762

[6] A. Gulati *et al.*, "Conformer: Convolution-augmented Transformer for Speech Recognition," in *Proc. Interspeech*, 2020, pp. 5036–5040, doi:10.21437/Interspeech.2020-3015.

[7] A. Radford *et al.*, "Robust Speech Recognition via Large-Scale Weak Supervision," in *Proc. ICML*, 2023, pp. 28492–28518. [Online]. Available: https://arxiv.org/abs/2212.04356

[8] T. Krichli, B. Raj, and J. Keshet, "CarelessWhisper: Turning Whisper into a Causal Streaming Model," *arXiv preprint arXiv:2508.12301*, 2025. [Online]. Available: https://arxiv.org/abs/2508.12301

[9] J. Yu *et al.*, "FastEmit: Low-Latency Streaming ASR with Sequence-Level Emission Regularization," in *ICASSP*, 2021, pp. 6004–6008, doi:10.1109/ICASSP39728.2021.9413803.

[10] R. Pope *et al.*, "Efficiently Scaling Transformer Inference," *arXiv preprint arXiv:2211.05102*, 2022. [Online]. Available: https://arxiv.org/abs/2211.05102

[11] T. Dao *et al.*, "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 35, 2022, pp. 16344–16359. [Online]. Available: https://arxiv.org/abs/2205.14135

[12] G. Xiao *et al.*, "Efficient Streaming Language Models with Attention Sinks," *arXiv preprint arXiv:2309.17453*, 2023. [Online]. Available: https://arxiv.org/abs/2309.17453

[13] Z. Xie and C. Wu, "Mini-Omni: Language Models Can Hear, Talk While Thinking in Streaming," *arXiv preprint arXiv:2408.16725*, 2024. [Online]. Available: https://arxiv.org/abs/2408.16725

[14] Silero Team, "Silero VAD: pre-trained enterprise-grade Voice Activity Detector," 2024. [Online]. Available: https://github.com/snakers4/silero-vad

[15] P. Budzianowski *et al.*, "MultiWOZ: A Large-Scale Multi-Domain Wizard-of-Oz Dataset for Task-Oriented Dialogue Modelling," in *Proc. EMNLP*, 2018, pp. 5016–5026, doi:10.18653/v1/D18-1547. [Online]. Available: https://aclanthology.org/D18-1547/

[16] Q. Zhu *et al.*, "CrossWOZ: A Large-Scale Chinese Cross-Domain Task-Oriented Dialogue Dataset," *Transactions of the Association for Computational Linguistics*, vol. 8, pp. 281–295, 2020, doi:10.1162/tacl_a_00314.

[17] Z. Du *et al.*, "CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models," *arXiv preprint arXiv:2412.10117*, 2024. [Online]. Available: https://arxiv.org/abs/2412.10117

[18] A. Yang *et al.*, "Qwen2 Technical Report," *arXiv preprint arXiv:2407.10671*, 2024. [Online]. Available: https://arxiv.org/abs/2407.10671

---

  

**BibTeX**

```bibtex
@misc{openai2024gpt4o,
  title         = {GPT-4o System Card},
  author        = {{OpenAI}},
  year          = {2024},
  eprint        = {2410.21276},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2410.21276}
}

@misc{gemini2024gemini15,
  title         = {Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context},
  author        = {{Gemini Team}},
  year          = {2024},
  eprint        = {2403.05530},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2403.05530}
}

@article{sacks1974turntaking,
  title   = {A Simplest Systematics for the Organization of Turn-Taking for Conversation},
  author  = {Sacks, Harvey and Schegloff, Emanuel A. and Jefferson, Gail},
  journal = {Language},
  volume  = {50},
  number  = {4},
  pages   = {696--735},
  year    = {1974},
  doi     = {10.2307/412243}
}

@misc{graves2012sequence_transduction,
  title         = {Sequence Transduction with Recurrent Neural Networks},
  author        = {Graves, Alex},
  year          = {2012},
  eprint        = {1211.3711},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/1211.3711}
}

@inproceedings{vaswani2017attention,
  title         = {Attention Is All You Need},
  author        = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N. and Kaiser, Lukasz and Polosukhin, Illia},
  booktitle     = {Advances in Neural Information Processing Systems},
  volume        = {30},
  year          = {2017},
  eprint        = {1706.03762},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/1706.03762}
}

@inproceedings{gulati2020conformer,
  title     = {Conformer: Convolution-augmented Transformer for Speech Recognition},
  author    = {Gulati, Anmol and Qin, James and Chiu, Chung-Cheng and Parmar, Niki and Zhang, Yu and Yu, Jiahui and Han, Wei and Wang, Shibo and Zhang, Zhengdong and Wu, Yonghui and Pang, Ruoming},
  booktitle = {Interspeech 2020},
  pages     = {5036--5040},
  year      = {2020},
  doi       = {10.21437/Interspeech.2020-3015},
  url       = {https://doi.org/10.21437/Interspeech.2020-3015}
}

@inproceedings{radford2023whisper,
  title         = {Robust Speech Recognition via Large-Scale Weak Supervision},
  author        = {Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and McLeavey, Christine and Sutskever, Ilya},
  booktitle     = {Proceedings of the 40th International Conference on Machine Learning},
  pages         = {28492--28518},
  year          = {2023},
  eprint        = {2212.04356},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2212.04356}
}

@misc{krichli2025carelesswhisper,
  title         = {CarelessWhisper: Turning Whisper into a Causal Streaming Model},
  author        = {Krichli, Tomer and Raj, Bhiksha and Keshet, Joseph},
  year          = {2025},
  eprint        = {2508.12301},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2508.12301}
}

@inproceedings{yu2021fastemit,
  title     = {FastEmit: Low-Latency Streaming ASR with Sequence-Level Emission Regularization},
  author    = {Yu, Jiahui and Chiu, Chung-Cheng and Li, Bo and Chang, Shuo-yiin and Sainath, Tara N. and He, Yanzhang and Narayanan, Arun and Han, Wei and Gulati, Anmol and Wu, Yonghui and Pang, Ruoming},
  booktitle = {ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing},
  pages     = {6004--6008},
  year      = {2021},
  doi       = {10.1109/ICASSP39728.2021.9413803},
  url       = {https://doi.org/10.1109/ICASSP39728.2021.9413803}
}

@misc{pope2022transformer_inference,
  title         = {Efficiently Scaling Transformer Inference},
  author        = {Pope, Reiner and Douglas, Sholto and Chowdhery, Aakanksha and Devlin, Jacob and Bradbury, James and Levskaya, Anselm and Heek, Jonathan and Xiao, Kefan and Agrawal, Shivani and Dean, Jeff},
  year          = {2022},
  eprint        = {2211.05102},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2211.05102}
}

@inproceedings{dao2022flashattention,
  title         = {FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness},
  author        = {Dao, Tri and Fu, Daniel Y. and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
  booktitle     = {Advances in Neural Information Processing Systems},
  volume        = {35},
  pages         = {16344--16359},
  year          = {2022},
  eprint        = {2205.14135},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2205.14135}
}

@misc{xiao2023attention_sinks,
  title         = {Efficient Streaming Language Models with Attention Sinks},
  author        = {Xiao, Guangxuan and Tian, Yuandong and Chen, Beidi and Han, Song and Lewis, Mike},
  year          = {2023},
  eprint        = {2309.17453},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2309.17453}
}

@misc{xie2024mini_omni,
  title         = {Mini-Omni: Language Models Can Hear, Talk While Thinking in Streaming},
  author        = {Xie, Zhifei and Wu, Changqiao},
  year          = {2024},
  eprint        = {2408.16725},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2408.16725}
}

@misc{silero2024vad,
  title        = {Silero VAD: pre-trained enterprise-grade Voice Activity Detector},
  author       = {{Silero Team}},
  year         = {2024},
  howpublished = {GitHub repository},
  url          = {https://github.com/snakers4/silero-vad},
  note         = {Accessed: 2025-12-19}
}

@inproceedings{budzianowski2018multiwoz,
  title     = {MultiWOZ: A Large-Scale Multi-Domain Wizard-of-Oz Dataset for Task-Oriented Dialogue Modelling},
  author    = {Budzianowski, Pawel and Wen, Tsung-Hsien and Tseng, Bo-Hsiang and Casanueva, Inigo and Ultes, Stefan and Ramadan, Osman and Gasic, Milica},
  booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
  pages     = {5016--5026},
  year      = {2018},
  doi       = {10.18653/v1/D18-1547},
  url       = {https://aclanthology.org/D18-1547/}
}

@article{zhu2020crosswoz,
  title   = {CrossWOZ: A Large-Scale Chinese Cross-Domain Task-Oriented Dialogue Dataset},
  author  = {Zhu, Qi and Huang, Kaili and Zhang, Zheng and Zhu, Xiaoyan and Huang, Minlie},
  journal = {Transactions of the Association for Computational Linguistics},
  volume  = {8},
  pages   = {281--295},
  year    = {2020},
  doi     = {10.1162/tacl_a_00314},
  url     = {https://doi.org/10.1162/tacl_a_00314}
}

@misc{du2024cosyvoice2,
  title         = {CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models},
  author        = {Du, Zhihao and others},
  year          = {2024},
  eprint        = {2412.10117},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2412.10117}
}

@misc{yang2024qwen2,
  title         = {Qwen2 Technical Report},
  author        = {Yang, An and others},
  year          = {2024},
  eprint        = {2407.10671},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2407.10671}
}
```
