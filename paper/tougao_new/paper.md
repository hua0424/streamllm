**并行流式架构的级联式语音对话系统延迟优化**

摘 要：针对级联式语音对话系统在长语音场景下端点后等待时间随输入时长增长的问题，提出流水线并行的流式优化架构，使语音识别、语言模型推理可重叠执行。前端以语音活动检测实时分段，结合 Whisper 时间戳对齐构建自适应滑窗与动态缓冲，通过前缀-后缀上下文与局部一致性约束提交稳定转录；后端采用键值缓存增量预填充，仅对新增文本更新缓存，降低首token输出时间并限制其不随长度线性增长。长语音数据集实验表明，长语音分组的平均首token输出时间稳定在约1.1秒，相比非流式基线降低34.6%–83.9%，最长分组平均减少5.67秒；转录错误率保持在可接受范围。结果表明该架构可有效降低长语音交互等待。

**关键词**：流式架构；语音对话系统；流水线并行；增量预填充；端到端延迟

文献标志码: A 中图分类号:TP18

**Latency Optimization of Cascaded Voice Dialogue Systems with a Pipeline-Parallel Streaming Architecture**

**Abstract**：A pipeline-parallel streaming architecture is proposed to mitigate post-utterance waiting time in cascaded voice dialogue systems for long-form speech. The architecture overlaps speech recognition and language-model prefilling/inference. Voice activity detection enables online segmentation, and an adaptive sliding window with dynamic buffering leverages Whisper timestamp alignment to commit stable transcripts under a prefix–suffix context and a local-consistency constraint. Incremental prefilling with a key–value cache updates only newly arrived text, reducing the time to first token and preventing it from increasing linearly with input length. Experiments on a long-speech dataset show that mean time to first token remains around 1.1 s in long-utterance groups, achieving 34.6%–83.9% reductions over a non-streaming baseline and a 5.67 s average reduction in the longest group, while keeping transcription error rates within an acceptable range. Results indicate that streaming pipeline parallelism and state incrementalization effectively reduce long-utterance interaction latency in modular cascaded systems.

**Key words**：streaming architecture; voice dialogue system; pipeline parallelism; incremental prefilling; end-to-end latency

随着人工智能技术的飞速发展，人机交互正经历着一场从“指令式”向“自然流式对话”的快速转变。 GPT-4o [1]和Gemini 1.5 [2]的出现，代表了原生多模态大模型的问世，使语言模型首次具备了在听觉、视觉和文本模态间进行理解与生成的能力。传统的语音助手(如早期的Siri或天猫精灵等)主要依赖基于规则的实现方式，其虽然在执行特定指令时表现良好，但其固定的句法十分生硬，严重限制了用户的表达自由度。用户需要遵守厂家预设的命令模板，这种僵硬的对话方式使人机之间的交互充满了机械感。

相比之下，新一代语音对话系统基于大语言模型（Large Language Model, LLM）有更强的理解能力，能处理复杂推理任务，还能支持带有情感色彩的多轮自然语言交互，可这种智能化水平的提高，也带来了巨大的计算开销和延迟挑战。公开的系统报告指出，GPT - 4o端到端语音响应延迟可低至232毫秒，平均320毫秒[1]，这已经接近人类自然对话反应速度。这意味着，“低延迟”与“准确率”并列，是下一代语音交互体验的关键竞争指标。

尽管端到端多模态模型在论文和实验中表现出色，可其训练成本高，数据隐私管控难，垂直场景需深度定制。而级联式系统是把语音活动检测（Voice Activity Detection, VAD）、自动语音识别（Automatic Speech Recognition, ASR）、大语言模型（Large Language Model, LLM）和语音合成（Text-to-Speech, TTS）拆分为独立可替换的模块并按顺序串联，因此可以复用现有的高质量ASR、文本LLM与TTS能力。在需要快速迭代LLM，控制部署成本，降低升级风险的场景中，级联架构仍然具备重要的工程价值。在这种串行架构中，系统的总延迟(![](data:image/x-wmf;base64...))不仅取决于各个模块的处理时间，还显著受限于模块间的数据流转方式。为方便分析，这种串行级联系统的总延迟可表示为各组件延迟的简单线性叠加：

**![](data:image/x-wmf;base64...)** (1)

在大多数常见的传统级联式实现中，从用户说话结束到系统语音回复往往要等待几秒。特别在长语音场景下，必须等到用户说完整句语音才启动ASR，而LLM的Prefill的耗时是随着输入文本同步增加的。因此，如何在保留级联架构模块化优势的基础上，借助流式并行策略，缩短模块间的“死区时间”，达成“打断即响应”的交互体验，且既有学术意义，又具有工程价值。

**1 相关工作**

**1.1 自动语音识别的流式化进展**

自动语音识别ASR系统的处理速度，决定着用户语音被转化到可被文本LLM接受的的文本的速度。早期流式自动语音识别的架构演进以RNN-Transducer（RNN-T）架构[3]为主导。RNN-T凭借其天然的循环神经网络的特性，非常适合处理流式音频，但其没法并行化，因此训练效率低，长程语义捕捉困难。随着 Transformer[4]的出现，让注意力机制模型的识别精度得以提升到新高。但是Transformer的全局注意力要求必须在整段输入完成后才开始计算，因此原生Transformer结构无法支持流式运行。

为了解决这个问题，Google提出了Conformer架构[5]，它把CNN（Convolutional Neural Network）局部特征提取和Transformer全局建模结合起来，很快就成了业界主流。在实际应用中，流式ASR常采用分块处理再加上重叠拼接的方式：先用滑动窗口切分音频并识别，然后通过重叠区域对输出做一致性校正，从而在准确率与延迟之间达成平衡。以Whisper为例，它的原始模型是采用Encoder - Decoder结构的，不过主要面向离线转录[6]；近期的CarelessWhisper[7]则把非因果编码器改成因果编码器，并在轻量微调与推理流程更新后，让Whisper能在更小的chunk上实现较低延迟的在线转录。而FastEmit[8]则是从训练目标入手，借助序列级正则化来激励token更早地发射，从而缩短流式识别的输出滞后时长。

**1.2 大语言模型推理加速技术**

在ASR完成转录之后，LLM的推理速度就成了系统延迟的第二大瓶颈。这一过程主要受Transformer解码器的自回归生成机制以及GPU的显存带宽所限。KV 键值缓存（KV Cache）技术的出现正是为了解决推理过程里的重复计算问题。该技术通过缓存历史token 的Key/Value向量值，这样在生成新token时，就不需要对历史上下文重复计算中间结果了，由此可以将解码阶段的单步计算复杂度从与序列长度二次相关降低为线性相关[9]。现在KV Cache已成为vLLM、TGI等现代推理引擎的标配。在对长文本进行输入的预填充（Prefill）阶段，像FlashAttention[10]则改进了GPU显存的读写方式，采用分块Tiling策略，让长序列的吞吐量提高到新的高度，从而在一定程度上解决attention计算量随序列长度迅速增大这一问题。在长对话场景下，StreamingLLM[11]引入“注意力汇聚”机制，把早期token作为固定锚点，以缓解上下文窗口溢出带来的性能下降，使LLM在有限的KV缓存里还能持续稳定的流式输出。

**1.3 语音对话系统的全链路优化**

除了单独优化ASR或者LLM之外，全链路协同调度对减少整体延迟也很重要。在语音翻译等语音处理任务中，级联方案长期作为常见系统范式存在，其优势在于可复用现有成熟组件、模块替换灵活，但也会带来中间表示损失、误差传播以及模块间延迟累积等问题[12-13]。近期端到端流式语音对话模型，如Mini-Omni、Moshi和LLaMA-Omni，尝试绕过传统ASR-LLM-TTS串行链路，通过语音原生建模和流式生成追求更低交互延迟[14-16]。

**1.4 本文主要研究内容**

传统级联语音对话系统在处理长语音时，各模块产生的延迟会逐级叠加，同时出现资源空转延迟累积与资源闲置问题，现有研究要么单独优化ASR模型，要么把LLM扩展为多模态架构直接生成语音，都难以复用已有的SOTA文本模型。本文提出一种细粒度流水线并行方案（下文称 System B），可直接兼容现有非流式ASR模型与文本LLM。本文的主要研究内容与贡献如下：

构建流式ASR上下文管理机制：本文设计了基于 Whisper和Silero VAD的自适应滑动窗口算法。为了解决流式切片导致的上下文缺失与识别不稳定性问题，本文提出了动态缓冲与“前缀后缀上下文(Prefix Context & Suffix Context)”拼接策略，在保证长句识别准确率的同时，实现毫秒级的ASR转录片段输出。

针对长语音输入引发的首字延迟（TTFT），本文提出了使用LLM的KV Cache技术应对流式转录生成的流式文本进行增量预填充。该算法深度运用 Transformer的KV Cache机制，实时接收流式ASR的输出片段，进行Attention计算与状态更新，不重复编码历史上下文，把LLM的计算开销均摊到用户语音输入过程中。

本文在MultiWOZ（英文）和CrossWOZ（中文）数据集上构建了长语音测试基准，这是对该内容进行全链路延迟评估与验证的工作。实验结果显示，在15秒以上的长语音场景里，本文提出的System B相比传统非流式基线（下文称System A）能显著降低延迟，而且没有对系统语义理解能力产生负面影响。

**2 原理与方法**

**2.1 自动语音识别技术**

**2.1.1 Whisper 模型架构与离线特性**

本研究采用OpenAI的Whisper作为语音转文本模块。Whisper是基于Transformer的编码器-解码器的端到端语音转文本模型，它直接根据语音特征输出文本token。该模型在大规模弱监督语音-文本数据上预训练，在多语言、口音和噪声场景下具有较强鲁棒性。

如图1所示，Whisper的输入端将16 kHz音频转换为80/128维Log-Mel频谱特征，经卷积前端下采样后送入Transformer编码器；解码端通过cross-attention读取编码器表征并自回归输出文本token。其词表包含时间戳token，可产生片段级或词级时间对齐信息，这为流式场景中“仅提交稳定且可对齐的部分输出”提供了实现基础。需要注意，Whisper默认接口仍偏向固定时长片段的离线处理，这也带来了流式化挑战。

**2.1.2 流式化挑战与解决方案**

Whisper的编码器自注意力可利用过去与未来帧信息，离线场景下有助于提升识别准确率，但在实时对话中会带来两类困难：一是短块独立识别时边界附近容易漏词、重复或重写；二是末尾假设会随后续音频到达而改变，下游LLM难以稳定消费。

为解决上述问题，本文采用“分段输入、重叠上下文、稳定提交”的组合策略：前端利用VAD对连续PCM流进行检测并切分语音段；ASR侧在目标片段前后保留上下文以缓解边界信息缺失；输出侧利用 Whisper的词级时间戳，只提交位于稳定区间内的文本片段，从而为LLM增量预填充提供可消费的文本流。

**2.2 大语言模型推理机制**

**2.2.1 Transformer 注意力机制**

大语言模型通常采用Decoder-only Transformer 架构。在语音对话系统中，LLM的推理延迟，尤其是首字延迟(Time to First Token, TTFT)，是决定用户是否感到“卡顿”的关键瓶颈。从推理流程看，TTFT 主要由输入 prompt 的预填充(prefill)计算与随后的首个解码步共同构成，而二者都与注意力的序列长度密切相关。Transformer 的核心是缩放点积注意力 (Scaled Dot-Product Attention)。其标准计算公式如下： ![](data:image/x-wmf;base64...) (2)

![1.drawio](data:image/png;base64...)

**图1 Whisper 模型整体架构与特征降采样示意图**

Fig. 1 Overall architecture of the Whisper model and schematic of feature downsampling

其中， (Query), (Key), (Value) 分别由输入向量经过线性投影矩阵 得到。 为 Key 的维度，用于缩放点积的数值范围，避免 随维度增大而方差膨胀导致 softmax 饱和、梯度过小。该机制允许模型在一个时间步内显式聚合序列中任意位置的信息，从而捕捉长程依赖关系。

**2.2.2 KV Cache机制的数学原理与复杂度分析**

LLM 的文本生成过程本质上是自回归的 (Auto-regressive)：模型在生成第N个 token 时，需要利用此前N-1个token的上下文信息。在 Self-Attention模块中，对长度为N的序列执行一次前向计算，其单步复杂度为![](data:image/x-wmf;base64...)。因此，生成长度为N的序列的总计算复杂度为：

![](data:image/x-wmf;base64...) (3)

尽管现代推理框架通常不会采用完全重复前向的实现，但它直观的说明了自回归生成模式中重复计算的来源：每一步都在重新计算历史token对应的 Key/Value以及它们之间的注意力关系。而KV Cache 的基本思想就是缓存这些历史值并在后续解码中复用。由于模型参数在推理阶段固定，历史Token ![](data:image/x-wmf;base64...)的![](data:image/x-wmf;base64...)一旦计算便不再改变，因此可将其存储在GPU的显存(VRAM)中。在第t步生成时，系统仅需计算当前 Token ![](data:image/x-wmf;base64...)的![](data:image/x-wmf;base64...)和![](data:image/x-wmf;base64...)，并将其追加到缓存末尾：

![](data:image/x-wmf;base64...) (4)

![](data:image/x-wmf;base64...) (5)

随后，注意力计算仅涉及当前 Query ![](data:image/x-wmf;base64...)与历史 ![](data:image/x-wmf;base64...)的交互，即计算![](data:image/x-wmf;base64...)并加权![](data:image/x-wmf;base64...)。因此，KV Cache 将解码阶段的单步复杂度由![](data:image/x-wmf;base64...) 降低为 ![](data:image/x-wmf;base64...),其仍然会随着上下文长度线性增长，生成N个 token的累计复杂度约为![](data:image/x-wmf;base64...),相应代价是缓存的显存占用随序列长度线性增长。

需要强调的是，KV Cache 主要减少的是解码阶段对历史 Key/Value 的重复计算，但并不消除 prompt 预填充阶段的二次方开销：对长度为N的输入，模型仍需至少执行一次全序列前向以构建初始缓存，其复杂度为![](data:image/x-wmf;base64...)，是TTFT的主要来源。本文提出的增量预填充策略并不改变复杂度阶数，而是利用ASR文本逐步到达的时间特性，将prefill拆分为多次小批量前向处理，并在用户发声过程中持续更新KV Cache，把原本集中发生在End-of-Speech之后的大部分计算前移并与用户说话过程在时间上重叠，从用户感知上降低语音结束到首个token生成之间的等待。图2描述了使用KV Cache后的计算复杂度与原来的差异，通过使用KV Cache，将历史K,V缓存并在推理中复用，把单步开销降为![](data:image/x-wmf;base64...)。

![pic2.drawio](data:image/png;base64...)

|  |  |
| --- | --- |
| 1. **System A: 原生推理(No Cache)** | 1. **System B: 增量推理(KV Cache)** |

**图2 基于 KV Cache 的推理机制示意**

Fig. 2 Schematic of the KV Cache-based inference mechanism

**2.3 系统评价指标**

**2.3.1 首字延迟**

TTFT是衡量交互实时性的核心指标，直接关联用户的主观等待感。本文将TTFT定义为从End of Speech到LLM输出首个响应token的时间差，并将其视为端点后关键路径延迟的综合体现。在System B中，由于部分![](data:image/x-wmf;base64...)可在用户发声过程中被增量预填充策略“前移”并与ASR并行处理，因此预计端点后的等待相较System A显著降低。

**2.3.2 词错误率**

虽然本研究的核心目标是降低延迟，但这不能以牺牲识别准确率为代价。WER是评估ASR准确率的通用标准，基于Levenshtein编辑距离计算：

![](data:image/x-wmf;base64...) (6)

其中S为替换(Substitution)、D为删除(Deletion)、 I为插入(Insertion)的错误数量，![](data:image/x-wmf;base64...)为参考文本的总词数。通常以百分比形式呈现，此时可再乘以100%。对于中文数据集，本研究主要考察字符错误率 (CER)，其计算逻辑与WER一致。

**3 系统设计**

**3.1 总体架构设计**

**3.1.1 目标与逻辑拓扑**

本系统的核心设计愿景是实现从传统的“全量接收-全量处理”范式向“增量接收-流式处理”范式的转移(Paradigm Shift)，从而在理论边界上最小化首字延迟(Time to First Token, TTFT)。系统的逻辑拓扑如图 3-1 所示。图中展示了基于多线程生产者-消费者队列的流式并行数据流向。音频以固定时长的PCM块持续进入分段模块，Silero VAD [17]在累计缓冲区上进行活动检测并输出语音段；ASR 模块将若干语音段拼接后调用Whisper进行转录，并按前缀/后缀上下文策略输出稳定文本片段；文本片段经线程安全队列传递给 LLM 模块，持续更新 KV Cache 并在收到终止标记后启动生成，从而尽可能将 LLM 的预填充计算前移并与音频输入过程重叠。

整体架构采用“分段—转录—预填充—生成”的流水线并行方式：当上游模块产生新的音频或文本段时，下游模块即可立即启动计算，无需等待整段输入结束。系统由以下三个子系统组成：负责语音活动检测与分段的Streaming Audio Segmenter、负责上下文拼接并保证输出稳定的Context-Aware ASR Engine ，以及负责增量预填充与最终生成的Incremental LLM Inference Service。其中 LLM 部分的KV Cache属于Transformer解码器推理阶段的通用优化思路，本文直接复用推理框架内置的use\_cache/past\_key\_values机制完成状态复用。

![pic3.drawio](data:image/png;base64...)

**图3 流式并行架构逻辑拓扑图**

Fig. 3 Logical topology of the streaming parallel architecture

**3.1.2 关键工程实现**

为了在单机原型中保持流式链路的实时与可控，工程实现上需要把整条链路拆成多线程流水线，并用线程安全队列解耦， 同时借助状态对象保存跨片段的上下文。

在通信机制上，为了去耦合，各模块不再使用同步函数调用，而是使用生产者-消费者模型进行异步通信。在代码实现方面，音频块队列、音频分段队列以及文本段队列都由queue.Queue的实例承担，消费者端则使用带超时参数的get方法循环，配合指定的结束事件标记生产端的消息生产结束，以避免单点阻塞，防止整个处理流程卡死。在ASR阶段，系统一方面需要承接上游传来的分段的音频，又要进行音频转录文本的流式处理，以将流式文本发往下由，因此该系统拆分为“收集器”与“转录器”两个子线程：前者持续接收上游语音段并写入等待队列，后者在满足触发条件时批量拼接，再调用Whisper进行文本转录，再将转录文本中的稳定的文本片段推送给下游 LLM，从而实现了音频收集与模型推理的解耦。

而状态管理系统则是采用了有状态 (Stateful) 与无状态(Stateless)的混合设计。分段模块通过 StreamState维护累计音频缓冲、语音段起止时间与段编号等信息；ASR侧通过ASRCache管理“等待加入的段队列”与“当前转录窗口段队列”，并使用 total\_duration 与处理标记避免并发竞争；LLM侧则以 KVCache持久化past\_key\_values、attention\_mask以及预填充阶段产生的最后一步logits，使得终止标记到达后可以直接进入生成阶段。上述状态对象共同保证了跨片段上下文的一致性，并为后续 3.2 与 3.3 的算法描述提供了相应的依据。

**3.2 流式ASR上下文管理**

**3.2.1 策略**

流式 ASR 的核心挑战在于寻找“识别准确率”与“识别延迟”之间的最优平衡：过短的音频切片会导致模型缺乏必要的声学上下文，从而因归纳偏置不足会导致字词错误率 (WER/CER)急剧增加；而等待过长的切片则会引入显著的缓冲与处理延迟，削弱流式处理的优势。为此，本节提出一种基于动态滑动窗口的自适应上下文管理策略。

**3.2.2 动态 VAD 分段策略**

系统采用Silero VAD模型对连续PCM音频流进行在线分段。定义输入音频流为时间序列 ![](data:image/x-wmf;base64...)，分段器维护累计缓冲区![](data:image/x-wmf;base64...)，并在每次接收新的音频块![](data:image/x-wmf;base64...)后执行 ![](data:image/x-wmf;base64...)，其中![](data:image/x-wmf;base64...)表示拼接。当累计音频长度超过最小检测窗口后，系统调用 Silero VAD 进行语音活动检测；若检测到足够长的静音，且语音段长度超过最小语音阈值，则认为该语音段已“闭合”，并输出给下游 ASR。本文原型系统的实验配置为音频块长度500 ms、最小语音长度2s、最小静音时长300 ms，以在实时性与稳定性之间取得折中。

**3.2.3 基于滑动窗口的上下文感知**

为了解决流式识别中的边界效应，ASR 引擎引入了包含“前缀上下文”(Prefix Context) 的滑动窗口机制。图4详细展示了时间步t时ASR模型的输入构成。窗口被划分为三个逻辑区域：前缀上下文 (Prefix Context) 为历史已确定的冻结区域，提供必要的声学历史信息；当前片段(Current Segments)为待识别的目标推理区域；后缀缓冲(Suffix Buffer)则包含未来音频信息以防止截断效应。系统通过这种重叠切分策略，确保了流式识别结果的连贯性。

![pic4.drawio](data:image/png;base64...)

**图 4 上下文感知 ASR 滑动窗口机制示意图**

Fig. 4 Schematic of the context-aware ASR sliding-window mechanism

首先，执行窗口更新。设![](data:image/x-wmf;base64...)为时刻t的音频段队列。当新的音频段到达时，系统先将其加入等待队列，并在转录线程中批量并入主队列得到![](data:image/x-wmf;base64...)；当队列累计时长满足触发阈值![](data:image/x-wmf;base64...)，且队列长度满足 ![](data:image/x-wmf;base64...)(或者收到了最后一个语音段时，ASR模型M对临时队列中的所有音频段进行拼接后执行一次文本转录，生成原始转录结果。在工程实现中，往往还会增加一个最短识别阈值进行控制，用于避免过短窗口造成的频繁调用与不稳定输出。

其次，进行确定性文本提取。鉴于Whisper等ASR模型在处理流式输入的末尾时往往存在“闪烁” (Flickering)现象：即随着新音频的输入，原本输出的末尾词汇可能发生变化，因此系统在输出侧加入“后缀保护”的段以增强输出文本的稳定性，系统仅输出前缀段与后缀段之间的稳定区域 (Stable Region) 内的文本。为保证输出与音频段一一对应，本文实现利用Whisper自带的词级时间戳将原始转录结果中的词按结束时间映射回每个音频段的时间区间，从而得到段级候选文本；随后仅拼接稳定区域内段的文本形成本轮输出。在工程实现中，系统还对流式起止边界做了特殊处理：在流式起始轮时，前缀段尚未输出过，稳定区的起点可从0开始；在流式结束轮时，为避免遗留文本，系统会输出所有剩余段并完成收敛。最后，执行缓冲区状态迁移。在输出稳定区域文本后，窗口执行滑动以限制累计长度并维持声学连贯性：系统不会将队列全部清空，而是保留最后一个已输出段之前的![](data:image/x-wmf;base64...)个段作为下一轮的前缀上下文，同时继续保留尚未输出的后缀保护段。需要指出的是，稳定输出的代价并非“仅增加一个chunk”的固定延迟，而是至少需要等待![](data:image/x-wmf;base64...)个后续段到达、并满足触发阈值 ![](data:image/x-wmf;base64...)。但该代价换来了输出文本在边界处的稳定性与准确性，使下游LLM能够稳定地进行增量预填充，从系统层面实现端到端延迟的可控下降的同时保证输入的准确性。

**3.3 LLM KV Cache增量预填充策略**

**3.3.1 策略说明**

在 System A中，LLM需要等ASR完整转录结束后才能拿到完整prompt，并在End-of-Speech之后一次性完成预填充与首个回复token的推理，因此随着转录文本长度越长，TTFT也会显著增加。要是把这种范式直接放到“边听边算”的增量输入场景里，就会白白浪费End-of-Speech之前的宝贵时间。

为减少End-of-Speech后的计算量，支撑流水线并行，System B在LLM部分引入了增量预填充：End-of-Speech之前，使用已经产生的流式文本进行预处理，在Transformer解码器预填充时产生的past\_key\_values作为KV Cache的历史，保存在内存或 GPU的VRAM中，在新文本片段抵达时，利用保存的past\_key\_value，只对新增token做前向计算，再将计算后新的past\_key\_value替换旧值，在另一段新文本到达时重复上述过程。这样LLM的大部分增量预填可以在用户说话期间完成。当终止标记到达时，系统只需处理末尾片段，马上开始生成回复token的计算工作，从而减少语音结束到首token输出的等待时间。

**3.3.2 核心算法实现**

增量推理的核心逻辑：首次调用时，系统在无缓存状态下一次性执行增量预填充，建立初始KV Cache；随后，每当上游传来一段新的ASR文本片段，系统便触发一次调用 cache\_prompt：在已有past\_key\_values的基础上做增量更新。当收到终止标记时，系统在prompt末尾追加生成提示符（常见的比如："assistant:"，用于告诉模型后续输出的首个token即是智能助理回复），并复用预填充阶段最后一步的logits直接开始解码。为避免增量token的位置编码错位，实现里会显式构造position\_ids，同时将新增片段以不增加生成提示符的方式编码，确保缓存拼接可控。Algorithm 1给出了与该实现对应的增量更新过程。图5直观展示了显存内的状态变化。左侧蓝色区域表示历史上下文对应的缓存键值对，右侧绿色区域表示新增片段产生的键值对。增量预填充在新片段到达时复用历史past\_key\_values，仅对新增token执行前向并更新缓存，从而避免在增量场景下反复对全量prompt进行prefill。

![pic5.drawio](data:image/png;base64...)

**图 5 LLM KV Cache 增量更新机制示意图**

Fig. 5 Schematic of the LLM KV Cache incremental update mechanism

**3.3.3 复杂度分析**

为了说明增量预填充减少冗余计算的来源，考虑历史上下文长度为N、新增片段长度为M的一次更新：若在每次增量到达时都对全序列![](data:image/x-wmf;base64...)重新执行一次prefill(不复用历史状态)，则Self-Attention的主要开销近似为![](data:image/x-wmf;base64...)。而在使用KV Cache的情况下，历史token的K,V已经缓存，系统仅需对新增token计算Query并与长度为![](data:image/x-wmf;base64...)的Key/Value交互，主要开销可写为![](data:image/x-wmf;base64...)。当![](data:image/x-wmf;base64...)时，上式近似为![](data:image/x-wmf;base64...)，体现为“对新增片段的线性扩展”。需要强调的是，若将所有增量片段累积到最终长度L，增量预填充的累计开销为 ![](data:image/x-wmf;base64...)，与离线一次性prefill同阶。本文策略的关键收益不在于改变复杂度阶数，而在于将prefill计算前移到用户发声期间，使End-of-Speech之后需要完成的计算主要集中在最后一小段增量及线程同步开销，从而有效降低TTFT。与此同时，本文实现会在K VCache中保留预填充阶段最后一步logits，使得终止标记到达后无需额外forward即可解码首 token，进一步减少首字阶段的常数开销。

**3.4 实验数据构建与处理管线**

**3.4.1 简述**

为了在受控环境下系统性验证流式架构的性能，需要有不断递增长度的对话数据，本研究构建了一套自动化的数据合成管线。该管线将开源数据集： MultiWOZ[18] (英文)和CrossWOZ[19] (中文)这两个标准多轮对话数据集转化为具有精确时长标注的逐渐递增的长语音测试数据，再通过tts生成音频数据。

![Algorithm 1](data:image/png;base64...)

**Algorithm 1 增量prefill的核心算法伪代码**

**3.4.3 累积对话生成逻辑**

**表1 对话数据累积方式**

Table 1 Dialogue Data Accumulation Strategy

|  |  |  |  |
| --- | --- | --- | --- |
| 轮次 | 用户 | 客服 | 数据文本 |
| 1 | 订一张机票 | 去哪里？ | 订一张机票 |
| 2 | 去北京，明天出发 | 要哪种舱位呢？ | 订一张机票，去哪里？，去北京，明天出发 |
| 3 | 要商务舱 | 好的，这就为您预订 | 订一张机票，去哪里？，去北京，明天出发。要哪种舱位呢？要商务舱 |

由于原始数据集中均为单轮对话，一次输入的时长有限。为了模拟用户进行长语音输入的真实场景(例如用户一口气陈述复杂的旅行需求，或在对话中不断补充信息)，本实验采用了“累积对话”策略。如表1 所示，算法遍历对话历史，将多轮对话中的用户侧(User)与上一轮的用户+系统回复文本进行累积拼接，从而构造出长度递增的输入序列。

通过此策略，我们能够基于真实的语义语境，生成从 3 秒至 60 秒以上不等的连续语音样本，有效覆盖了从短指令到长篇陈述的各种交互形态，解决了传统数据集缺乏长语音样本的问题。

**3.4.3 数据处理流水线**

数据构建过程包含三个严格顺序执行的阶段：

首先是历史累积与筛选。策略遍历源数据集，应用上述累积生成逻辑。为了聚焦长语音场景下的性能瓶颈，系统按文本长度对生成样本进行倒序排列，优先选取最长的前N个对话片段，并设置最大文本长度阈值(英文 2050 字符，中文 720 字符)以防止显存溢出，且设置语音长度上限可以防止无意义的增加实验时间。

其次是并发TTS音频合成。本研究集成了阿里巴巴提出的CosyVoice大模型语音合成服务[20]。相较于传统TTS，CosyVoice能够生成韵律更自然、情感更丰富的高保真语音波形，更贴近真实人声输入。在工程实现上，我们开发了批量处理模块，采用多线程异步请求机制，显著提升了大规模数据生成的效率。

最后是时长校准与元数据同步。由于生成式 TTS 模型的语速具有非确定性，简单的基于文本字数的时长估算往往存在误差。本管线通过解析生成的WAV 文件头来获取以秒为单位（可精确到毫秒级）的物理时长。该真实时长被回写至测试元数据JSON中，作为后续实验中X轴（输入时长）的真值依据(Ground Truth)。

**3.4.4 测试集分组定义**

为了细粒度分析不同时长下的延迟表现，我们将生成的样本集依据音频时长划分为五个标准实验组(见表2)。

**表2 音频时长分组**

Table 2 Audio Duration Groups

|  |  |  |  |
| --- | --- | --- | --- |
| 组名 | 时长(s) | 典型场景 | 实验聚焦 |
| Short | T<5 | 短指令 | 基准延迟验证 (Baseline Latency) |
| Medium | 5≤T<15 | 包含多轮意图的陈述 | 日常对话性能 (Daily Conversation) |
| Long | 15≤T<30 | 复杂长难句/长段落 | 流式架构核心优势区间 |
| Very Long | 30≤T<60 | 长段论述说明 | 判断TTFT是否存在稳定上界 |
| Extra Long | T≥60 | 极限压力测试 | 系统稳定性与显存边界 (Stability & OOM) |

这一分组标准将在第四章的实验分析中贯穿始终，用于对比不同系统架构在处理长短语音时的性能差异。

**4 实验与结果分析**

**4.1 实验说明**

本章围绕“VAD 判定用户语音结束至大语言模型产生首个响应Token”的首字延迟（TTFT）开展实验验证，系统性评估本文提出的流式级联式语音对话架构（System B）在长语音交互场景中的性能收益与代价。与传统非流式基线（System A）相比，System B 的关键差异在于：其一，将 ASR 从端点后全量解码改造为分片流式解码，使声学计算与用户发声时长发生重叠；其二，在 ASR 增量输出的推动下，对 LLM 进行增量预计算，以削弱长上下文一次预填充带来的串行等待。通过控制变量，本章依次从延迟特性、关键模块消融以及精度-延迟权衡三方面给出实证结果，并讨论观察到的影响延迟的“主导项切换”现象及其工程边界。

**4.2 实验一：延迟与语音长度的关系验证**

**4.2.1 实验设置与环境构建**

为了客观衡量系统在长语音场景下的实时响应能力，本实验选取MultiWOZ与CrossWOZ两个公开数据集，分别代表英文与中文的多轮对话任务。按上一章节所述的数据构建流程，我们生成了覆盖短、中、长、很长、超长五组长度的混合测试集，共1132条样本。

实验硬件平台统一基于两块NVIDIA RTX 3090 （24GB VRAM）显卡，ASR模型和LLM模型分别在两块显卡中运行。为了消除模型加载与初始化带来的随机误差，对比组（System A）与实验组（System B）实施了严格的公平性控制： 首先，两组实验均复用同一套模型权重，即Whisper-Turbo作为声学编码器，以及Qwen2-7B-Instruct[21]作为语义生成器，排除了参数量差异带来的干扰。其次，引入“热身机制”，在记录数据前预先执行3轮真实音频推理，以确保CUDA Context和核心算子完成初始化。需要说明的是，System A 是常规非流式串行级联实现：完整音频输入结束后进行一次性ASR转录，再将完整文本送入LLM 完成一次性Prefill与首token解码。System B与System A的差异仅在流式分段、流水线调度和KV Cache增量更新方式，而非模型权重或硬件条件。

**4.2.2 延迟趋势分析**

![Pasted image 20251216110152](data:image/png;base64...)

**图 6 System A与System B的TTFT随时长变化趋势**

Fig.6 TTFT trends of System A and System B over input duration

首字延迟（TTFT）是衡量语音交互流畅度的核心指标。图6展示了在不同语音时长输入下，非流式基线系统与本文流式系统的TTFT变化轨迹。

其中X轴为音频真实时长 (秒)，Y轴为端到端 TTFT（毫秒）。System A曲线随时长近似单调上升，反映端点后串行等待的累积；System B在 long 及以上区间趋于平缓，呈现由固定分片粒度与尾部调度开销主导的近似上界。两条曲线在15秒附近出现拐点，意味着流式并行的覆盖收益开始超过其固定开销。

**表3 不同语音时长分组下的 TTFT 统计**

Table 3 TTFT Statistics Across Speech Duration Groups

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| 组名 | 样本数 | 平均时长(s) | 流式TTFT(ms) | 非流式TTFT(ms) | 改进(%) |
| Short | 35 | 3.44 | 648.17 | 533.42 | -21.5 |
| Medium | 89 | 9.25 | 959.53 | 923.42 | -3.9 |
| Long | 121 | 22.16 | 1126.63 | 1722.03 | 34.6 |
| Very Long | 208 | 45.21 | 1099.16 | 3191.97 | 65.6 |
| Extra Long | 679 | 105.73 | 1087.70 | 6753.43 | 83.9 |

表3清晰的给出了按语音时长分组的 TTFT 统计结果。非流式基线System A的TTFT随输入时长有明显增长。其中short 组均值为533.42ms，long组上升到了1722.04ms，very\_long与extra\_long组进一步增长到3191.97ms与6753.43ms。这一现象和级联式处理范式的预期是相符的，因为ASR得等语音生成完成之后才能完成全量解码，之后LLM才会对完整转录文本一次性进行全量Prefill。在此期间，端点之后的计算链路几乎不会重叠，这就使得语音长度与TTFT基本是线性相关的关系。而流式系统 System B 在 long 及以上区间显示出近似常数的“上界”：long、very\_long 与 extra\_long 三组的 TTFT 均值分别为： 1126.63ms、1099.16ms 与 1087.70ms，整体稳定在约 1.1s 左右。相应的TTFT在long、very\_long、extra\_long这三个分组上，相对于基线系统分别压缩了34.6%、65.6%与83.9%，其中extra\_long组平均绝对延迟降低了5.67s，明显减轻了长语音交互时的静默等待。需要说明的是，这里的“常数”是指主导项由“随时长累积的全量计算耗时”转变为“最后若干分片的尾部处理与调度开销”耗时之和。在音频包时长500 ms与计算资源不变的前提下，TTFT 不再随输入时长的增长而继续显著增加。

**表4 核心模块消融实验结果**

Table 4 Ablation Results of Core Modules

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| 组名 | 平均时长(s) | Baseline TTFT(ms) | Streaming ASR Only TTFT(ms) | Full Streaming TTFT(ms) | ASR improvement  (ms) | KV Cache improvement  (ms) |
| Long | 21.51 | 1698.88 | 1064.18 | 1084.17 | 634.70 | -19.99 |
| Very Long | 42.43 | 3300.58 | 1171.02 | 1154.06 | 2129.56 | 16.96 |
| Extra Long | 83.75 | 6518.40 | 1228.77 | 1114.57 | 5289.63 | 114.20 |

值得一提的是short与medium组在System B下分别录得114.75ms与36.11ms的负向优化结果，这表明在短语音场景下，分片、缓存维护以及增量提示构造等流式架构的固定开销难以被并行收益所抵消，而且这么短的语音和文本，并未达到实验硬件的并行处理上限。该结果说明工程落地时应因地制宜进行优化：对预测时长较短的输入优先走传统的一次性处理链路，而将流式并行机制用于处理音频T≥15s的场景，以实现最佳的整体交互体验。

**4.3 实验二：消融实验**

**4.3.1 实验设置**

为了量化“流式 ASR”与“LLM增量预填充”两项机制的相对贡献，我们挑选了long及以上长度的各50个样本开展消融实验。按照前面章节描述的分组，我们选择了long（15–30s）、very\_long（30–60s）与 extra\_long（≥60s）三个分组，因为在该长度区间能比较好地体现流式改造的收益。对比配置设置分为三类：Baseline（System A，ASR 与 LLM 均为非流式串行执行）、Streaming ASR Only（仅 ASR 流式化，LLM在等待全部文本到齐后再启动推理）以及Full Streaming（在 Streaming ASR Only 使用流式ASR和流式LLM，也就是上文中的 System B）。

本实验的核心指标主要就是TTFT，另外还在日志里记录语音生成结束端点后的 ASR 尾部处理耗时以及 LLM Prefill耗时，用来判断处理瓶颈。为了凸显架构差异给用户感知延迟带来的影响，我们在实验结果中统计了三种配置的 TTFT 均值，以及由此计算出的 ASR 增益与 KV Cache 的增益(正值表示缩短 TTFT，负值表示增加耗时)，从而保证结论可被归因于目标机制本身。

**4.3.2 贡献度量化分析**

实验结果如表4所示，在实验平台硬件及所用模型条件下，ASR流式化是缩短TTFT的主因。Baseline与Streaming ASR Only相比，在long、very\_long与extra\_long三组长度下分别减少了634.70ms、2129.56ms与5289.63ms，对应降幅为37.4%、64.5%与81.1%。这表明在传统串行链路中，语音结束端点后的主要耗时来自于ASR对整个音频的处理。而流式ASR把大部分计算提前到与语音生成的时间重叠，因此显著压缩了端点后的计算量。增量Prefill在LLM方面表现出显著的长度的依赖特性。在long组，Full Streaming相比Streaming ASR Only的TTFT反而增加了19.99ms(KV增益为-19.99ms)，这说明当上下文还没足够长的时候，增量调用、缓存管理以及同步这些额外开销或许会超出其节省的一次性Prefill计算。而在very\_long与extra\_long组，KV增益分别为16.96ms与114.20ms，且在extra\_long组，KV Cache相对整体的收益占比达到约9.3%（114.20/1228.77），这表明，随着输入文本与上下文长度增长，LLM Prefill的的收益愈加显著。

综合两项优化机制，Full Streaming将TTFT均值控制在1.1s（long 1084.17ms、very\_long 1154.06ms、extra\_long 1114.57ms）左右，相较于Baseline分别缩减了约36.2%、65.0%与82.9%。该结果与实验总体趋势一致，流式并行能明显降低级联系统在输入时长增加时的等待时间，不过KV Cache预填充的边际收益还是受工程实现和硬件并发与算力的影响。消融实验表明，在实验平台的硬件条件下，运行Whisper模型已显吃力。而Qwen2-7B-Instruct的体量在LLM领域算是小型模型，其推理计算量远小于高性能模型（参数量往往在数百到上千B），因此实验平台的算力应付起来可以说比较轻松。如果换为主流SOTA模型，相信流式在KV Cache Prefill的收益将会更加明显。另外值得注意的是，在三组递增的音频实验结果显示，只使用流式ASR和一次性prefill时，耗时还是逐渐递增的；而在全流式（使用了KV Cache）的结果中，三组音频的处理时长并没有递增的关系。这表示在实验环境下，整个转录文本的平方阶复杂度已经超出了硬件并行计算能力的上限，需要花费更长的时间来计算注意力。而在使用KV Cache的情况下，尾部计算复杂度相对于整个文本接近线性阶，并未达到硬件的并行处理上限，因此即使长度增加了，但是处理时长并未增加，说明KV Cache prefill机制确实能让在长语音下让计算成本显著降低。

**表5 不同上下文窗口配置下的准确率与耗时对比**

Table 5 Accuracy and Processing Time Comparison Under Different Context Window Configurations

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| 上下文配置 | multiwoz  wer\_mean | multiwoz  cer\_mean | crosswoz  wer\_mean | crosswoz  cer\_mean | ASR耗时(ms) |
| 前缀1段+后缀1段 | 0.0895 | 0.0196 | 0.0083 | 0.0203 | 1327.48 |
| 前缀1段+无后缀（默认配置） | 0.0813 | 0.0221 | 0.0118 | 0.0292 | 1224.96 |
| 前缀0段+后缀0段 | 0.0619 | 0.0226 | 0.0210 | 0.0500 | 1086.16 |

**4.4 实验三：准确率与质量边界**

**4.4.1 实验说明与评估方法**

延迟被大幅压短后，流式化的主要代价落在“上下文截断”与“端点不确定性”上。与 System A 可利用整段音频不同，流式 ASR 在每个输入的语音片段上只能看到整段音频局部的片段，句末边界附近因此更易出现替换、插入或省略。为引入稳定的识别结果，本文在 ASR 侧采用滑动窗口上下文管理，通过前缀（prefix）提供音频，通过后缀（suffix）提供后续音频，在精度与耗时之间进行折中。借助该机制，本实验对比三种配置：prefix=1, suffix=0（默认）、prefix=1, suffix=1（增加 1 段前瞻以改善句尾效果）以及 prefix=0, suffix=0（完全去除上下文以追求最小延迟），并从 medium、long 与 very\_long 组各抽取 50 条样本，共 150 条，用于覆盖中长语音的主要应用场景。我们采用业界标准的词错误率（Word Error Rate, WER）和字错误率（Character Error Rate, CER）作为衡量指标，分别针对英文（MultiWOZ）与中文（CrossWOZ）样本计算转录误差，并将非流式全量识别（System A）作为上界对照，比较不同流式上下文配置（System B）下的识别质量与尾部耗时。

**4.4.2 精度-延迟分析**

实验中每种上下文配置均包含150条样本，其中 MultiWOZ 74 条、CrossWOZ 76条。由于评测音频由文本经 TTS 合成生成，参考文本可视为该批样本的近似真值；同时，System A在该合成集合上的 WER/CER为0，作为上界对照。本文主要关注不同上下文配置下的相对差异。

表5展示了不同上下文窗口配置下的准确率与尾部耗时对比结果。可以看出，增加前后缀（pre1&suf1）对中文句尾稳定性的提升更加明显：中文数据集CrossWOZ的WER/CER 分别由默认配置的 0.0118/0.0292 下降到了 0.0083/0.0203。由于处理片段长度增加，也导致了语音结束端点后的尾部耗时上升，使asr\_time\_ms由1224.96ms增加到了1327.48ms（+102.52ms），这与后缀窗口需要等待额外音频片段并完成更多解码计算的机制预期相符。而对于英文数据集的 MultiWOZ，suffix 的增加使 CER 从 0.0221 降低到了 0.0196，不过 WER 稍微升高到了 0.0895，说明在以词为单位的评价标准下，后缀窗口对齐和分词的影响可能更敏感了。这一现象有必要通过真实录音及更大样本数据做进一步验证。当完全去除额外上下文（pre0&suf0）时，系统获得了更低的耗时（asr\_time\_ms 下降至 1086.16ms，相比默认耗时减少 138.80ms），中文误差在预料之内明显变大了：CrossWOZ 的 CER 上升到了 0.0500，WER 也上升到了 0.0210。该结果显示，prefix给出的那些历史上下文在流式识别里发挥了重要的约束作用，能给音频 - 语言模型的局部决策带来更强纠正效果，从而抑制长句里逐步累积的误差。

由于中文是单字结构，而英文是单词组合，因此在中文数据集上我们主要考察的是 CER 也就是字符的错误率，而英文数据集主要考察的是 WER 即词错误率。从中文数据集上看，随着上下文的扩大，CER有明显降低；而在英文数据集上，三个配置的 WER 可以说基本一致，最小的上下文有最低的 WER 可以视为是实验的波动。这主要是因为中文的同音字较多，更多的上下文对于转录出准确的文字有较大帮助；而英文单词的同音词汇就少得多，因此只要发音准确，词级识别就相对稳定，增加的上下文并未显著提升准确率，因此在实际部署时，英文对话可去掉前后缀扩展，把流式架构的延迟优势压到极限。

最后从端到端交互预算看， 实验一与实验二显示 long 及以上语音的 TTFT 已可稳定压缩至约 1.1s，而表 4-3 中不同上下文配置带来的 0.1–0.2s 尾部耗时已占去可观比例，因此需按场景进行取舍：对中文任务中识别正确性更敏感者，可启用 pre1&f1 以降低句尾错误；pre0&suf0 则作为极低延迟模式，或用于英文对话。考虑到对话式 LLM 在语义层面通常对噪声具有一定鲁棒性，转录文本的小误差预计对 LLM 的推理不会造成明显影响。后续可引入语义一致性与任务完成率等指标，直接量化“转写误差是否造成语义漂移”的数值。

**5 总结与展望**

**5.1 全文总结**

本文围绕级联式语音对话系统在长语音输入下端点后等待时间过长的问题，设计并实现了一套基于流水线并行的低延迟语音对话架构。该架构以“流式计算”和“状态增量”为设计主线，将传统串行链路中的ASR全量转录与LLM一次性预填充改造为可重叠执行的流式处理过程：在ASR侧，通过VAD分段、滑动窗口和前缀/后缀上下文，把偏离线的 Whisper 封装为可输出稳定增量文本的流式模块；在 LLM 侧，利用past\_key\_values对新增文本片段进行增量预填充，并复用预填充阶段最后一步logits启动首token解码。需要指出的是，该策略并不改变注意力计算的复杂度阶数，而是把原本集中在End-of-Speech之后的预填充计算提前到用户说话过程中，从而减少端点后的关键路径耗时。

实验结果验证了该架构的有效性。在从 MultiWOZ 与 CrossWOZ 构建的长语音数据集上，本文系统比传统非流式基线在 15 秒以上的长语音分组中具有稳定的优势，其将TTFT保持在了一个稳定的范围内。同时通过滑动窗口上下文维持了语音转录文本的准确性，WER/CER处于可接受范围内。虽然在long及以上分组的TTFT仍然有约为1.1s，与人类对话中约200ms的对话间隙仍有一定差距，但与传统串行方式相比，其最大的优势在于长语音场景下的TTFT在达到一定数值后不再随着语音时长增长。未来借助更高效的推理硬件、更低延迟的ASR模型及进一步调度优化 ，继续压缩这段端点后等待时间，即可实现任意长度语音均保持与人类对话间隙一致的响应延迟。

**5.2 研究局限性**

尽管本文降低了长语音场景下的端点后等待，但系统仍受级联架构约束。首先，当前原型仍遵循“用户发言—系统处理—系统回复”的半双工话轮模式，尚不能处理自然对话中的重叠说话与打断。其次，ASR 与LLM之间仍以文本作为唯一接口，语调、情绪、停顿等副语言信息会在转写过程中丢失。再次，本文测试音频主要由TTS合成，真实噪声、口音和麦克风条件下的鲁棒性仍需进一步验证。

**5.3 未来工作**

未来工作将集中在两方面。第一，抑制尾部延迟抖动：实验日志显示，部分长语音样本的峰值延迟来自末端连续分段触发，后续可通过末端合并、队列回压和自适应调度减少重复 ASR 推理。第二，优化话轮结束判定：真实对话通常依赖静音阈值判断用户是否结束发言，当模型端 TTFT 已压缩到约 1 s 后，固定静音等待会成为新的主导项。后续可引入轻量级语义结束判别模型（End-of-Utterance, EOU），结合流式 ASR 文本和上下文预测话轮是否语义完结，从而进一步降低真实交互等待。

参考文献:

[1] OpenAI. GPT-4o system card[EB/OL]. 2024. https://arxiv.org/abs/2410.21276.

[2] Gemini Team. Gemini 1.5: unlocking multimodal understanding across millions of tokens of context[EB/OL]. 2024. https://arxiv.org/abs/2403.05530.

[3] GRAVES A. Sequence transduction with recurrent neural networks[EB/OL]. 2012. https://arxiv.org/abs/1211.3711.

[4] VASWANI A, SHAZEER N, PARMAR N, et al. Attention is all you need[C]//Advances in Neural Information Processing Systems. Red Hook: Curran Associates, 2017, 30: 5998-6008.

[5] GULATI A, QIN J, CHIU C C, et al. Conformer: convolution-augmented Transformer for speech recognition[C]//Proceedings of Interspeech 2020. Baixas: ISCA, 2020: 5036-5040.

[6] RADFORD A, KIM J W, XU T, et al. Robust speech recognition via large-scale weak supervision[C]//Proceedings of the 40th International Conference on Machine Learning. PMLR, 2023, 202: 28492-28518.

[7] KRICHLI T, RAJ B, KESHET J. CarelessWhisper: turning Whisper into a causal streaming model[EB/OL]. 2025. https://arxiv.org/abs/2508.12301.

[8] YU J, CHIU C C, HAN W, et al. FastEmit: low-latency streaming ASR with sequence-level emission regularization[C]//Proceedings of ICASSP 2021. Piscataway: IEEE, 2021: 6004-6008.

[9] POPE R, DOUGLAS S, CHOWDHERY A, et al. Efficiently scaling Transformer inference[EB/OL]. 2022. https://arxiv.org/abs/2211.05102.

[10] DAO T, FU D Y, ERMON S, et al. FlashAttention: fast and memory-efficient exact attention with IO-awareness[C]//Advances in Neural Information Processing Systems. Red Hook: Curran Associates, 2022, 35: 16344-16359.

[11] XIAO G, TIAN Y, CHEN B, et al. Efficient streaming language models with attention sinks[EB/OL]. 2023. https://arxiv.org/abs/2309.17453.

[12] SPERBER M, PAULIK M. Speech translation and the end-to-end promise: taking stock of where we are[C]//Proceedings of ACL. Stroudsburg: Association for Computational Linguistics, 2020: 7409-7421.

[13] BENTIVOGLI L, CETTOLO M, GAIDO M, et al. Cascade versus direct speech translation: do the differences still make a difference?[C]//Proceedings of ACL-IJCNLP. Stroudsburg: Association for Computational Linguistics, 2021: 2873-2887.

[14] XIE Z, WU C. Mini-Omni: language models can hear, talk while thinking in streaming[EB/OL]. 2024. https://arxiv.org/abs/2408.16725.

[15] DÉFOSSEZ A, MAZARÉ L, ORSINI M, et al. Moshi: a speech-text foundation model for real-time dialogue[EB/OL]. 2024. https://arxiv.org/abs/2410.00037.

[16] FANG Q, GUO S, ZHOU Y, et al. LLaMA-Omni: seamless speech interaction with large language models[C]//Proceedings of the 13th International Conference on Learning Representations. 2025.

[17] Silero Team. Silero VAD: pre-trained enterprise-grade voice activity detector[EB/OL]. 2024. https://github.com/snakers4/silero-vad.

[18] BUDZIANOWSKI P, WEN T H, TSENG B H, et al. MultiWOZ: a large-scale multi-domain Wizard-of-Oz dataset for task-oriented dialogue modelling[C]//Proceedings of EMNLP 2018. Stroudsburg: Association for Computational Linguistics, 2018: 5016-5026.

[19] ZHU Q, HUANG K, ZHANG Z, et al. CrossWOZ: a large-scale Chinese cross-domain task-oriented dialogue dataset[J]. Transactions of the Association for Computational Linguistics, 2020, 8: 281-295.

[20] DU Z, WANG Y, CHEN Q, et al. CosyVoice 2: scalable streaming speech synthesis with large language models[EB/OL]. 2024. https://arxiv.org/abs/2412.10117.

[21] YANG A, YANG B, HUI B, et al. Qwen2 technical report[EB/OL]. 2024. https://arxiv.org/abs/2407.10671.
