# 第二章 相关技术基础 (Theoretical Foundation)

本章将详细阐述支撑本研究的核心理论基础，重点分析自动语音识别（ASR）的神经网络架构特性以及大语言模型（LLM）的推理计算机制，并定义评估系统性能的关键数学指标。

## 2.1 自动语音识别技术 (ASR)

本研究采用 OpenAI 提出的 Whisper [1] 模型作为语音转文本的核心引擎。相较于传统的 RNN-T 或 CTC 模型，Whisper 基于 Transformer 的 Encoder-Decoder 架构在弱监督大规模数据下展现了更优的鲁棒性。

### 2.1.1 Whisper 模型架构与降采样机制
Whisper 的核心架构由音频编码器（Encoder）和文本解码器（Decoder）组成。其输入处理模块的设计对于计算效率至关重要。

1.  **特征提取与卷积前端 (Conv Stem)**：
    模型首先将 16kHz 采样率的原始音频转换为 80 通道的 Log-Mel 频谱图。随后，频谱图通过两层 1D 卷积神经网络（CNN）进行特征编码：
    *   **Conv1**: 滤波器宽度为 3，激活函数为 GELU。
    *   **Conv2**: 滤波器宽度为 3，**步长 (Stride) 为 2**。

    **降采样分析**：第二层卷积的 `Stride=2` 起到了关键的降采样作用。对于标准的 30 秒音频输入窗口（采样率为 16kHz，帧移 10ms，共约 3000 帧），Conv Stem 将其压缩为 **1500 帧**。
    这一设计不仅保留了 20ms 的时间分辨率以区分音素，更重要的是将后续 Transformer Encoder 中 Self-Attention 的序列长度 $N$ 减半。由于 Attention 的计算复杂度为 $O(N^2)$，这一操作将计算量降低了 4 倍，显著提升了推理速度。

    [图 2-1: Whisper 模型架构与 Log-Mel 特征提取流程]
    *(图注：展示 Audio Waveform -> Log-Mel Spectrogram -> Conv Stem (Downsampling) -> Transformer Encoder -> Cross-Attention -> Decoder Text Generation 的完整数据流。)*

### 2.1.2 流式化原理：分块与局部一致性
Whisper 原生设计为处理 30 秒完整音频，为了适应低延迟流式场景，本研究引入以下机制：
*   **分块处理 (Chunk-based Processing)**：将连续的音频流切分为固定长度的微小片段（如 500ms），并在每个时间步 $t$ 将当前累积的音频缓冲区输入模型。
*   **局部一致性 (Local Agreement)**：由于缺乏未来上下文（Right Context），模型在处理最新到达的音频块时，输出往往是不稳定的（Flickering）。局部一致性策略通过比较时刻 $t$ 和 $t-1$ 的输出结果，仅通过“交集”确认那些在连续两个窗口中保持一致的 Token，从而保证输出的稳定性。

## 2.2 大语言模型推理机制 (LLM Inference)

大语言模型通常采用 Decoder-only Transformer 架构。在语音对话系统中，LLM 的推理延迟（尤其是首字延迟）是影响体验的关键瓶颈。

### 2.2.1 Transformer 注意力机制
Transformer 的核心是缩放点积注意力 (Scaled Dot-Product Attention)。其标准计算公式如下：

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

其中，$Q$ (Query), $K$ (Key), $V$ (Value) 分别由输入向量经过线性投影矩阵 $W_Q, W_K, W_V$ 得到。$d_k$ 为 Key 的维度，用于缩放数值以稳定梯度。

### 2.2.2 KV Cache 机制的数学原理
LLM 的生成过程是自回归的 (Auto-regressive)。为了生成第 $t$ 个 Token，模型理论上需要依赖之前所有 $t-1$ 个 Token 的信息。

#### System A: 无缓存推理 (Naive Inference)
在不使用 KV Cache 的情况下，每生成一个新 Token，都需要重新计算所有历史 Token 的 $K$ 和 $V$ 向量。
假设生成长度为 $N$ 的序列，第 $t$ 步的计算量主要来源于 Attention 矩阵乘法，其复杂度与序列长度成正比 $O(t)$。
总计算复杂度为级数求和：
$$ \text{Cost}_{total} = \sum_{t=1}^{N} t = \frac{N(N+1)}{2} \approx O(N^2) $$
这种平方级的增长在处理长文本（或长语音转录文本）时会导致不可接受的延迟。

#### System B: 基于 KV Cache 的增量推理
利用模型权重固定的特性，历史 Token $x_i (i < t)$ 的 $k_i, v_i$ 向量一旦计算便不再改变。因此，可以将它们存储在 GPU 显存中。
在第 $t$ 步生成时，仅需计算当前 Token $x_t$ 的 $k_t$ 和 $v_t$，并将其拼接到缓存末尾：
$$ K_{cache}^{(t)} = \text{Concat}(K_{cache}^{(t-1)}, k_t) $$
$$ V_{cache}^{(t)} = \text{Concat}(V_{cache}^{(t-1)}, v_t) $$

此时，注意力计算仅涉及当前 Query $q_t$ 与历史 $K_{cache}$ 的交互。
*   **单步复杂度**：降低为 $O(1)$（矩阵维度为 $1 \times d$ 与 $t \times d$ 的乘法，关于生成步数是常数级的投影，关于上下文是线性的，但避免了 $O(t^2)$ 的全量重算）。
*   **总复杂度**：生成 $N$ 个 Token 的总开销降低为 **$O(N)$**。

[图 2-2: KV Cache 空间换时间原理示意图]
*(图注：左侧展示 System A 每次重新计算整个三角形区域；右侧展示 System B 仅计算对角线上的新增元素，并复用灰色区域的缓存数据。)*

## 2.3 系统评价指标 (Evaluation Metrics)

### 2.3.1 首字延迟 (Time to First Token, TTFT)
TTFT 是衡量交互实时性的核心指标。根据调研报告 [2]，语音对话系统的总延迟形式化分解为：
$$ L_{total} = L_{VAD} + L_{ASR} + L_{LLM\_Prefill} + L_{LLM\_Decode} + L_{TTS} + L_{Net} $$
其中，TTFT 特指从用户语音结束（VAD 判定截断）到 LLM 输出第一个字符的时间差，主要包含 $L_{ASR}$ 和 $L_{LLM\_Prefill}$。

### 2.3.2 词错误率 (Word Error Rate, WER)
WER 是评估 ASR 准确率的标准指标，基于 Levenshtein 编辑距离计算：
$$ \text{WER} = \frac{S + D + I}{N} \times 100\% $$
其中 $S$ 为替换 (Substitution)、$D$ 为删除 (Deletion)、$I$ 为插入 (Insertion) 的错误数量，$N$ 为参考文本的总词数。对于中文，通常使用字符错误率 (CER)，计算逻辑相同。

---
**参考文献**
[1] Radford, A., et al. (2023). Robust Speech Recognition via Large-Scale Weak Supervision.
[2] Google Research. (2024). *Latency Optimization in Spoken Dialogue Systems*.
