**流式架构的级联式语音对话系统延迟优化**

莫海华1

1. 广西大学 计算机与电子信息学院，广西南宁 530004

\+ 通信作者 E-mail: zhyliang@gxu.edu.cn

摘 要：针对级联式语音对话系统在长语音场景下端点后等待时间随输入时长增长的问题，提出一种流水线并行的流式优化架构。该架构保留自动语音识别、大语言模型和语音合成等模块化组件，通过语音活动检测对输入音频实时分段，结合 Whisper 时间戳对齐、前缀/后缀上下文和稳定区提交策略输出增量转录文本；同时利用大语言模型的键值缓存机制对新增文本执行增量预填充，将原本集中在用户语音结束后的计算前移到发声过程中。基于 MultiWOZ 与 CrossWOZ 构建的长语音测试集实验表明，在 15 s 以上长语音分组中，本文方法将首 token 输出时间稳定控制在约 1.1 s，相比同模型、同硬件的传统串行级联基线降低 34.6%~83.9%，最长分组平均减少 5.67 s；消融实验显示主要收益来自 ASR 与 LLM 计算的跨模块重叠，增量预填充在超长文本输入下进一步降低尾部开销。实验结果表明，流式流水线并行与状态增量化可有效缓解级联式语音对话系统的长语音等待问题。

**关键词**：流式架构；语音对话系统；流水线并行；增量预填充；首字延迟

文献标志码: A 中图分类号: TP18

**Latency Optimization of Cascaded Voice Dialogue Systems with a Pipeline-Parallel Streaming Architecture**

MO Haihua1

1. School of Computer, Electronics and Information, Guangxi University, Nanning 530004, China

**Abstract**: To reduce the post-utterance waiting time of cascaded voice dialogue systems in long-speech scenarios, this paper proposes a pipeline-parallel streaming architecture. The proposed method preserves the modularity of automatic speech recognition, large language model inference, and speech synthesis, while overlapping upstream and downstream computation. Voice activity detection segments the input stream online; Whisper timestamp alignment, prefix/suffix context, and stable-region commitment are used to emit reliable incremental transcripts. On the language-model side, key-value cache reuse enables incremental prefilling over newly arrived text fragments, moving most prompt computation from after the end of speech to the speaking period. Experiments on long-speech test sets constructed from MultiWOZ and CrossWOZ show that the proposed system keeps time to first token at about 1.1 s for utterances longer than 15 s, reducing latency by 34.6%--83.9% over a same-model non-streaming cascaded baseline on the same hardware, with an average reduction of 5.67 s in the longest group. Ablation results indicate that most gains come from cross-module overlap between ASR and LLM computation, while incremental prefilling further reduces tail overhead for extra-long inputs.

**Key words**: streaming architecture; voice dialogue system; pipeline parallelism; incremental prefilling; time to first token

## 1 引言

随着大语言模型与多模态模型的发展，语音对话系统正从固定命令式交互转向自然语言交互。GPT-4o 的系统报告显示，端到端语音响应延迟已接近人类对话节奏[1]；Gemini 1.5 等模型也推动了长上下文多模态理解能力的发展[2]。低延迟、强理解和自然轮替成为新一代语音交互系统的重要指标。

现有语音对话系统大体可分为两类。一类是端到端或语音原生模型，直接在音频或语音 token 上建模，能够更好保留语调、情绪和打断等副语言信息。Moshi[13]、LLaMA-Omni[14] 和 Mini-Omni[15] 均围绕实时语音交互进行了探索，其中 Moshi 报告了约 200 ms 的实际交互延迟，LLaMA-Omni 报告了约 226 ms 的响应延迟。另一类是级联式系统，即将语音活动检测（VAD）、自动语音识别（ASR）、文本大语言模型（LLM）和语音合成（TTS）拆分为独立模块。级联式系统虽然存在中间文本瓶颈和模块串行延迟，但具有工程部署简单、可复用成熟 ASR/LLM/TTS、模型升级风险低和隐私控制清晰等优势，仍是许多垂直场景中的可行方案。

本文关注第二类系统。常规级联式实现中，用户音频通常在话轮结束后被完整送入 ASR，得到完整文本后再进入 LLM 预填充和生成阶段。端点后的关键路径可简化表示为：

【公式占位：端点后延迟由 ASR 全量转录、LLM Prefill、首个 Decode 步及系统调度开销组成】

当输入语音变长时，ASR 全量转录时间和 LLM 预填充时间都会增加，导致用户停止说话后仍需等待数秒。人类自然对话的平均轮替间隙通常仅为数百毫秒量级[3]，这种端点后等待会破坏交互连贯性，在长语音输入、复杂任务陈述和连续补充需求等场景中尤其明显。

为缓解上述问题，本文提出一种面向级联式语音对话系统的流水线并行流式架构（System B）。与常规串行基线（System A）相比，System B 不改变底层 ASR 和 LLM 权重，而是改变数据流转方式：音频输入过程中由 VAD 持续分段，ASR 以滑动窗口方式输出稳定文本片段，LLM 在文本片段到达时即时增量预填充并更新 KV Cache。这样，原本集中在 End-of-Speech 后的计算被分摊到用户发声期间，从而降低端点到首 token 的等待时间（Time to First Token, TTFT）。

本文主要贡献如下：

（1）提出一种可复用现有 ASR 与文本 LLM 的流式级联架构，通过多线程队列实现音频输入、VAD 分段、ASR 转录和 LLM 预填充的流水线并行。

（2）设计基于 Whisper 时间戳的流式 ASR 上下文管理方法，采用前缀/后缀上下文和稳定区提交策略，缓解音频分段边界处的识别抖动。

（3）实现面向流式文本输入的 LLM KV Cache 增量预填充，仅对新增 token 执行前向计算，并在预填充阶段保留最后一步 logits 以减少首 token 解码开销。

（4）在 MultiWOZ 与 CrossWOZ 构建的长语音测试集上开展同模型、同硬件对比实验、消融实验和精度-延迟权衡分析，验证该架构在 15 s 以上长语音场景中的延迟收益。

## 2 相关工作

### 2.1 流式自动语音识别

ASR 是级联式语音对话系统的前端模块，其输出速度直接影响下游 LLM 的启动时间。早期流式 ASR 常采用 RNN-Transducer（RNN-T）等结构[4]，能够边接收音频边输出识别结果。Transformer[5] 提升了序列建模能力，但其全局注意力机制天然偏向完整上下文；Conformer[6] 将卷积局部建模与 Transformer 全局建模结合，成为语音识别中的重要结构。

Whisper[7] 采用编码器-解码器结构，在大规模弱监督语音数据上训练，具有较强鲁棒性，但默认接口主要面向离线转录。若直接将长音频切成短块独立识别，边界处容易出现漏词、重复和重写。因此，流式化改造通常需要分段、重叠上下文和稳定提交机制。CarelessWhisper[8] 尝试将 Whisper 改造为因果流式模型，FastEmit[9] 则从训练目标上鼓励更早发射 token。与上述模型级优化不同，本文在不重新训练 Whisper 的前提下，通过 VAD 分段、滑动窗口和时间戳对齐构造工程可用的伪流式输出。

### 2.2 大语言模型推理加速

LLM 推理延迟主要由 prompt 预填充和自回归解码构成。KV Cache 通过缓存历史 token 在各层产生的 Key/Value，避免解码阶段重复计算历史状态，已成为现代推理框架的常用机制[10]。FlashAttention[11] 从显存访问角度提升长序列注意力效率，StreamingLLM[12] 则利用注意力汇聚机制支持长上下文流式推理。

需要指出的是，常规 KV Cache 主要减少解码阶段的重复计算，并不自动消除完整 prompt 在首 token 前的一次性预填充开销。本文利用 ASR 文本逐段到达的特点，将 LLM 预填充拆分为多次小批量增量更新，使大部分 prompt 计算在用户发声过程中提前完成。该策略不改变最终注意力计算的复杂度阶数，而是改变计算发生的时间位置，从用户感知角度降低 End-of-Speech 后等待。

### 2.3 语音对话系统架构

级联式语音系统将 ASR、文本理解/生成和 TTS 作为独立模块组合，长期用于语音翻译和语音交互任务。相关综述指出，级联方案具备模块可替换、可利用单任务成熟模型等优势，但也存在误差传播、中间表示信息损失和模块间延迟累积等问题[16-18]。近年来，端到端语音对话模型试图绕过中间文本瓶颈。Moshi[13] 将语音对话建模为双流语音生成；LLaMA-Omni[14] 将语音编码器、LLM 和流式语音解码器结合；Mini-Omni[15] 探索了开源端到端实时语音交互。

这些先进系统展示了语音原生模型的潜力，但通常需要专门训练、语音响应数据和复杂的跨模态对齐。本文选择另一条工程路径：保留现有高质量 ASR 与文本 LLM，通过流式调度和状态复用降低级联系统延迟。该定位决定了本文实验不直接声称优于端到端先进模型，而是在同一组模型和硬件条件下验证“由串行级联改为流式并行”带来的增量收益。

## 3 系统设计

### 3.1 System A 与 System B 定义

为避免基线含义不清，本文将两种系统定义如下。

System A 是常规非流式串行级联基线。其处理流程为：等待完整音频输入结束，使用同一 Whisper 模型对完整音频一次性转录，随后将完整转录文本作为用户输入送入同一 Qwen2 系列文本 LLM，并执行一次性 prompt 预填充和首 token 解码。System A 不代表某篇单独论文提出的新方法，而代表级联式语音系统中常见的 utterance-wise 串行处理范式。该基线的作用是度量在不改变模型权重时，端点后全量 ASR 与全量 LLM Prefill 造成的等待。

System B 是本文提出的流式流水线并行系统。其处理流程为：输入音频按固定块模拟实时到达；Silero VAD[19] 在累计缓冲区上检测语音段；ASR 模块维护待处理音频段队列，在满足识别阈值、前缀段和后缀段约束后调用 Whisper 转录合并窗口；转录结果依据词级时间戳映射回音频段，仅提交稳定区域文本；LLM 模块收到文本片段后调用 cache_prompt 更新 KV Cache，收到结束标记后追加生成提示符并生成首 token。

两套系统使用相同 ASR 模型、相同 LLM、相同音频输入和相同硬件环境。差异仅在数据流调度方式与是否增量复用中间状态，因此可较公平地评估本文方法带来的工程优化收益。

【图1占位：System A 串行级联流程与 System B 流水线并行流程对比】

### 3.2 多线程流水线

System B 的原型系统采用生产者-消费者队列实现模块解耦。音频生成线程按固定块将 PCM 数据送入音频块队列；分段线程使用 StreamAudioSegmenter 处理音频块并输出语音段；ASR 线程进一步拆分为收集子线程和转录子线程，前者持续接收语音段并写入 ASRCache，后者在满足触发条件时批量拼接窗口并调用 Whisper；LLM 线程消费文本队列，对新增文本执行增量预填充，并在收到结束信号后开始生成。

该设计的关键不是增加模型数量，而是改变模块空转关系。在 System A 中，ASR 必须等音频结束才开始，LLM 又必须等完整转录结束才开始；在 System B 中，ASR 与用户发声重叠，LLM 预填充又与 ASR 增量输出重叠，从而压缩端点后的关键路径。

### 3.3 流式 ASR 上下文管理

Whisper 默认更适合完整片段转录，直接短块识别会导致边界不稳定。本文采用前缀/后缀上下文滑动窗口。设当前 ASR 队列包含若干 VAD 语音段，前缀段提供历史声学上下文，当前稳定区为本轮可提交文本，后缀段用于保护末尾未稳定区域。系统在以下条件满足时触发一次识别：队列累计时长达到识别阈值，且队列长度满足“前缀段数 + 当前段 + 至少后缀段数”的要求；若收到最后一段，则强制触发并输出剩余文本。

【图2占位：前缀上下文、稳定输出区和后缀保护区示意】

实现中，分段器使用 Silero VAD 在 16 kHz 单声道音频上检测语音活动；实验主流程中音频块长度为 500 ms，最小语音段时长为 500 ms，最小静音时长为 300 ms。ASR 调用 Whisper 时开启词级时间戳，随后根据词的结束时间将文本映射回对应音频段。非最终轮次只输出稳定区文本，最终轮次输出剩余所有文本。该策略以少量尾部等待换取文本稳定性，避免 LLM 反复消费会被后续重写的末尾假设。

### 3.4 LLM KV Cache 增量预填充

在 System A 中，LLM 在完整 ASR 文本产生后才执行 prompt 预填充。输入越长，一次性 prefill 越耗时，TTFT 随之增加。System B 则把 ASR 增量文本视为流式 prompt：首次文本到达时构造聊天模板并初始化 KV Cache；后续文本片段到达时不重新编码历史 prompt，而是将新增 token 与历史 past_key_values 一起送入模型，更新缓存和 attention mask；为避免位置编码错位，显式构造 position_ids；当收到结束标记时，追加生成提示符，并复用预填充阶段最后一步 logits 解码首 token。

【图3占位：KV Cache 增量追加与首 token 解码流程】

设历史 token 长度为 N，新增片段长度为 M。若每次新增文本都重新对长度 N+M 的全量 prompt 执行 prefill，则会产生大量重复计算；使用 KV Cache 后，历史 Key/Value 可被复用，新增步骤只需处理新 token 与历史缓存的交互。若从全局累计复杂度看，最终处理完整上下文仍与长序列注意力同阶；但从交互时间线看，System B 将大量计算提前到用户仍在说话的阶段，使 End-of-Speech 后主要剩余最后片段处理、调度和首 token 解码。

【算法1占位：增量 prefill 伪代码】

## 4 实验设计

### 4.1 数据集构建

本文从 MultiWOZ[20] 与 CrossWOZ[21] 构建长语音测试集。原始数据为多轮任务型对话文本，为模拟用户一次性陈述复杂需求或连续补充信息的场景，本文采用历史累积策略：沿对话轮次逐步拼接用户话语及必要的上下文，使样本长度从短指令扩展到长段输入。随后使用 CosyVoice[22] 合成语音，并从 WAV 文件读取实际时长作为实验分组依据。

【表1占位：累积对话构造示例】

样本按音频时长分为 Short（T<5 s）、Medium（5 s≤T<15 s）、Long（15 s≤T<30 s）、Very Long（30 s≤T<60 s）和 Extra Long（T≥60 s）五组。该划分用于观察从短指令到超长输入时，串行级联与流式并行系统的延迟变化。

【表2占位：音频时长分组定义】

### 4.2 实验环境与公平性控制

实验在两块 NVIDIA RTX 3090（24 GB VRAM）显卡的平台上进行，ASR 与 LLM 分别放置在不同显卡上。两组系统复用同一 ASR 和 LLM 权重；主实验中 ASR 为 Whisper large-v3-turbo（下文简称 Whisper-Turbo），LLM 为 Qwen2-7B-Instruct[23]。为降低初始化误差，测试前执行真实音频预热，使 CUDA context 和相关算子完成初始化。流式实验按 500 ms 音频块模拟真实输入节奏。

本文没有将 System B 与 GPT-4o、Moshi、LLaMA-Omni 等直接进行端到端数值对比，原因有三点：第一，先进端到端系统通常直接生成语音或联合建模语音/文本，而本文指标聚焦于级联链路中 End-of-Speech 到 LLM 首 token 的等待；第二，模型参数、训练数据和推理硬件不可控，直接比较容易混入模型能力差异；第三，本文目标是验证在同一模型、同一硬件下，数据流调度与 KV Cache 增量化是否能降低级联系统延迟。因此，本文采用受控基线对比与消融实验作为主要证据，并在相关工作和讨论中说明与先进系统的定位差异。

### 4.3 评价指标

核心延迟指标为 TTFT，定义为从音频输入结束（End-of-Speech）到 LLM 输出首个响应 token 的时间。该指标排除了用户正在说话的时间，更直接反映用户停止说话后的等待感。

精度指标采用英文词错误率（WER）与中文字符错误率（CER）。二者均基于编辑距离计算，用于评估流式切分和稳定提交是否显著损害 ASR 质量。

【公式占位：WER/CER 编辑距离定义】

## 5 实验结果与分析

### 5.1 延迟随语音长度变化

图4展示了 System A 与 System B 的 TTFT 随语音时长变化趋势。System A 随输入时长增长而上升，说明完整 ASR 与完整 prompt prefill 在端点后形成串行瓶颈。System B 在 Long 及以上分组中趋于平缓，表明主要计算已在用户发声期间被流水线覆盖，端点后剩余时间主要由末尾分片、同步调度和首 token 解码决定。

【图4占位：System A 与 System B 的 TTFT 趋势】

【表3占位：不同语音时长分组下的 TTFT 统计】

实验统计显示，System A 的平均 TTFT 从 Short 组的 533.42 ms 增至 Extra Long 组的 6753.43 ms；System B 在 Long、Very Long 和 Extra Long 组分别为 1126.63 ms、1099.16 ms 和 1087.70 ms，基本稳定在约 1.1 s。相对于 System A，System B 在上述三组分别降低 34.6%、65.6% 和 83.9%，Extra Long 组平均绝对延迟减少 5.67 s。

Short 与 Medium 组中，System B 分别出现 114.75 ms 和 36.11 ms 的负向收益。这说明流式分段、队列调度和 KV Cache 维护具有固定开销；当输入很短时，这些开销难以被并行覆盖收益抵消。因此，实际部署可采用混合策略：短输入走一次性串行路径，中长输入启用流式并行路径。

### 5.2 消融实验

为量化不同模块的贡献，本文在 Long、Very Long 与 Extra Long 三组中分别抽样，对比三种配置：Baseline（System A，ASR 与 LLM 均非流式串行）、Streaming ASR Only（仅 ASR 流式化，LLM 等待全部文本后一次性预填充）和 Full Streaming（System B，流式 ASR + LLM 增量预填充）。

【表4占位：核心模块消融实验结果】

消融结果表明，在当前硬件和模型配置下，ASR 流式化是降低 TTFT 的主因。与 Baseline 相比，Streaming ASR Only 在 Long、Very Long 和 Extra Long 三组分别减少 634.70 ms、2129.56 ms 和 5289.63 ms，说明传统串行链路中大量等待来自端点后的完整音频转录。

LLM 增量预填充的收益与文本长度相关。在 Long 组，Full Streaming 相比 Streaming ASR Only 反而增加约 19.99 ms，说明短上下文下增量调用和缓存管理开销可能超过一次性 prefill 节省；在 Very Long 和 Extra Long 组，KV Cache 分别带来 16.96 ms 和 114.20 ms 的进一步收益。该结果符合预期：当输入文本较短时，LLM prefill 不是主要瓶颈；当上下文增长后，增量预填充的价值逐渐显现。

需要谨慎表述的是，本文实验不能据此推断“KV Cache 改变了完整 prefill 的复杂度阶数”。更准确的结论是：KV Cache 增量预填充减少了流式场景中对历史 prompt 的重复处理，并把大部分 prefill 计算前移到说话阶段，因此降低了端点后的尾部等待。

### 5.3 准确率与延迟权衡

流式化的主要代价来自上下文截断与边界不确定性。为评估该代价，本文比较不同 ASR 上下文配置，包括前缀1段+后缀1段、前缀1段+无后缀、前缀0段+后缀0段。实验从 Medium、Long 和 Very Long 组抽样，分别计算 MultiWOZ 的 WER/CER 与 CrossWOZ 的 WER/CER，并记录 ASR 尾部耗时。

【表5占位：不同上下文配置下的 WER/CER 与 ASR 尾部耗时】

总体趋势显示，增加上下文可改善边界稳定性，尤其对中文 CER 更明显。前缀和后缀能够为同音字、句尾词和跨片段语义提供更多约束，但会增加尾部等待；完全去除上下文可降低耗时，却会提高中文错误率。英文样本中，不同上下文配置的 WER 差异较小，可能与英文词级边界更清晰、TTS 音频较干净有关。实际部署时可按场景选择：中文或高准确率任务采用前缀/后缀保护，低延迟优先任务采用较小上下文。

### 5.4 与先进工作的关系

从绝对交互延迟看，System B 的约 1.1 s TTFT 仍高于 GPT-4o、Moshi 和 LLaMA-Omni 等先进语音原生系统报告的数百毫秒延迟[1,13-15]。这说明端到端语音模型在自然对话体验上具有明显潜力。本文的价值不在于替代这些系统，而在于为仍需使用级联架构的场景提供可复现的工程优化路径：在不重新训练语音大模型、不替换文本 LLM 的情况下，通过流水线并行和状态复用显著降低长语音端点后等待。

因此，本文方法的适用边界是：对模型可替换性、部署成本、数据隐私和工程可控性要求较高，且已有成熟 ASR 与文本 LLM 的系统。若目标是全双工、可打断、情绪感知的自然语音对话，则端到端或语音原生模型仍是更有前景的方向。

## 6 结论与展望

本文针对级联式语音对话系统在长语音场景中的端点后等待问题，提出一种流水线并行流式架构。该架构通过 VAD 实时分段、Whisper 滑动窗口稳定提交和 LLM KV Cache 增量预填充，将 ASR 和 LLM 的主要计算从用户停止说话后前移到发声过程中。实验表明，在同模型、同硬件条件下，System B 可将 15 s 以上长语音的 TTFT 稳定控制在约 1.1 s，并在 Extra Long 分组相对串行基线平均减少 5.67 s。

本文也存在局限。首先，系统仍属于半双工级联架构，依赖端点判定后再正式生成回复，尚不能处理自然对话中的重叠说话和打断。其次，ASR 到 LLM 的中间表示仍是文本，语气、情绪和停顿等副语言信息会丢失。再次，当前测试音频由 TTS 合成，噪声、口音和真实麦克风输入条件仍需进一步验证。最后，当前约 1.1 s 的 TTFT 已低于传统长语音串行等待，但距离人类自然对话的数百毫秒轮替间隙仍有差距。

后续工作将从三方面展开：一是优化末端分段调度，减少最后一两个音频段连续触发 ASR 带来的长尾延迟；二是引入语义结束判定模型，在保证鲁棒性的前提下减少固定静音等待；三是在真实录音数据、噪声环境和更大规模 LLM 上验证增量预填充的收益边界。

参考文献:

[1] OpenAI, "GPT-4o System Card," arXiv preprint arXiv:2410.21276, 2024.

[2] Gemini Team, "Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context," arXiv preprint arXiv:2403.05530, 2024.

[3] H. Sacks, E. A. Schegloff, and G. Jefferson, "A Simplest Systematics for the Organization of Turn-Taking for Conversation," Language, vol. 50, no. 4, pp. 696-735, 1974.

[4] A. Graves, "Sequence Transduction with Recurrent Neural Networks," arXiv preprint arXiv:1211.3711, 2012.

[5] A. Vaswani et al., "Attention Is All You Need," in Advances in Neural Information Processing Systems, vol. 30, 2017.

[6] A. Gulati et al., "Conformer: Convolution-augmented Transformer for Speech Recognition," in Proc. Interspeech, pp. 5036-5040, 2020.

[7] A. Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision," in Proc. ICML, pp. 28492-28518, 2023.

[8] T. Krichli, B. Raj, and J. Keshet, "CarelessWhisper: Turning Whisper into a Causal Streaming Model," arXiv preprint arXiv:2508.12301, 2025.

[9] J. Yu et al., "FastEmit: Low-Latency Streaming ASR with Sequence-Level Emission Regularization," in Proc. ICASSP, pp. 6004-6008, 2021.

[10] R. Pope et al., "Efficiently Scaling Transformer Inference," arXiv preprint arXiv:2211.05102, 2022.

[11] T. Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness," in Advances in Neural Information Processing Systems, vol. 35, pp. 16344-16359, 2022.

[12] G. Xiao et al., "Efficient Streaming Language Models with Attention Sinks," arXiv preprint arXiv:2309.17453, 2023.

[13] A. Defossez et al., "Moshi: a speech-text foundation model for real-time dialogue," arXiv preprint arXiv:2410.00037, 2024.

[14] Q. Fang et al., "LLaMA-Omni: Seamless Speech Interaction with Large Language Models," arXiv preprint arXiv:2409.06666, 2024.

[15] Z. Xie and C. Wu, "Mini-Omni: Language Models Can Hear, Talk While Thinking in Streaming," arXiv preprint arXiv:2408.16725, 2024.

[16] M. Sperber and M. Paulik, "Speech Translation and the End-to-End Promise: Taking Stock of Where We Are," in Proc. ACL, pp. 7409-7421, 2020.

[17] L. Bentivogli et al., "Cascade versus Direct Speech Translation: Do the Differences Still Make a Difference?" in Proc. ACL-IJCNLP, pp. 2873-2887, 2021.

[18] C. K. Maurya and N. Sethiya, "End-to-End Speech-to-Text Translation: A Survey," arXiv preprint arXiv:2312.01053, 2023.

[19] Silero Team, "Silero VAD: pre-trained enterprise-grade Voice Activity Detector," 2024. [Online]. Available: https://github.com/snakers4/silero-vad

[20] P. Budzianowski et al., "MultiWOZ: A Large-Scale Multi-Domain Wizard-of-Oz Dataset for Task-Oriented Dialogue Modelling," in Proc. EMNLP, pp. 5016-5026, 2018.

[21] Q. Zhu et al., "CrossWOZ: A Large-Scale Chinese Cross-Domain Task-Oriented Dialogue Dataset," Transactions of the Association for Computational Linguistics, vol. 8, pp. 281-295, 2020.

[22] Z. Du et al., "CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models," arXiv preprint arXiv:2412.10117, 2024.

[23] A. Yang et al., "Qwen2 Technical Report," arXiv preprint arXiv:2407.10671, 2024.
