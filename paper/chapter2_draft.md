# 第二章 相关技术基础 (Theoretical Foundation)

本章将详细阐述支撑本研究的核心理论基础，包括自动语音识别（ASR）的神经网络架构、大语言模型（LLM）的推理机制以及评估语音对话系统性能的关键指标。

## 2.1 自动语音识别技术 (ASR)

自动语音识别旨在将连续的声学信号序列转换为离散的词（Word）或字（Character）序列。本研究采用 OpenAI 提出的 Whisper [1] 作为核心识别引擎，并结合 Silero VAD 进行端点检测。

### 2.1.1 Whisper 模型架构
Whisper 采用了标准的 Transformer **Encoder-Decoder** 架构，而非流式 ASR 中常见的 RNN-Transducer (RNN-T)。其设计理念是通过在大规模弱监督数据（680,000 小时）上进行训练，学习从声学特征到文本的鲁棒映射。

1.  **特征提取与卷积前端 (Conv Stem)**
    输入音频首先被重采样为 16kHz，并转换为 80 通道的 Log-Mel 频谱图。随后，频谱图通过两层 1D 卷积层进行降采样与特征编码：
    *   **Conv1**: Filter width 3, GeLU activation.
    *   **Conv2**: Filter width 3, **Stride 2**.
    *   **作用**：Stride=2 的设置将时间分辨率从 10ms 压缩至 20ms，即使序列长度减少一半。对于 30s 的标准输入窗口，帧数从 3000 降至 1500，显著降低了后续 Transformer 层的计算负担（Self-Attention 复杂度 $O(N^2)$）。

2.  **编码器 (Encoder) 与解码器 (Decoder)**
    *   **Encoder**: 采用多层 Transformer Block，通过自注意力机制（Self-Attention）建模全局声学上下文。
    *   **Decoder**: 采用自回归方式生成文本 Token。关键在于 **Cross-Attention** 层，它将文本查询（Query）与音频键值（Key/Value）对齐。

3.  **注意力机制 (Scaled Dot-Product Attention)**
    无论是 Self-Attention 还是 Cross-Attention，其核心数学形式均为：

    $$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

    其中：
    *   $Q \in \mathbb{R}^{L_q \times d_k}$ 为查询矩阵。
    *   $K \in \mathbb{R}^{L_k \times d_k}, V \in \mathbb{R}^{L_k \times d_v}$ 为键与值矩阵。
    *   $d_k$ 为缩放因子，用于防止点积数值过大导致 Softmax 梯度消失。

### 2.1.2 语音活动检测 (Voice Activity Detection, VAD)
为了实现流式分段，系统前置了 Silero VAD 模块。它基于轻量级的 LSTM/RNN 结构，输入音频帧 $x_t$，输出该帧为“语音”的后验概率 $P(speech | x_t)$。

$$ y_t = \sigma(W_{out} \cdot h_t + b_{out}) $$

其中 $h_t$ 是 RNN 的隐状态，$\sigma$ 为 Sigmoid 函数。当 $P(speech | x_t) > \theta_{trigger}$ 持续超过 $T_{min}$ 时，判为语音开始；反之判为静音。

## 2.2 大语言模型推理机制 (LLM Inference)

大语言模型通常基于 Transformer Decoder-only 架构。在对话系统中，推理过程分为两个阶段：**Prefill（预填充）** 和 **Decode（解码）**。本节重点分析 Decode 阶段的计算瓶颈与 KV Cache 优化机制。

### 2.2.1 自回归生成与计算复杂度
LLM 的生成是自回归的（Auto-regressive），即第 $t+1$ 个 Token 的生成依赖于序列 $x_{1:t}$。
目标是最大化联合概率：
$$ P(x) = \prod_{t=1}^{T} P(x_t | x_{<t}) $$

在不使用缓存的朴素推理中，为了生成 $x_t$，必须将 $x_{1:t-1}$ 重新输入模型计算 Key 和 Value。
对于长度为 $N$ 的序列，总计算复杂度主要源于 Attention 操作。第 $t$ 步需要计算 $t \times t$ 的注意力矩阵（假设全量计算），总浮点运算量 (FLOPs) 约为：
$$ \text{Cost}_{naive} \approx \sum_{t=1}^{N} O(t^2) \approx O(N^3) $$
*(注：部分文献简化描述为 $O(N^2)$，指每步 $O(t)$ 的情况，但核心在于重复计算导致的非线性增长)*

### 2.2.2 KV Cache 机制推导
**KV Cache (键值缓存)** 利用了模型权重 $W_K, W_V$ 固定的特性。历史 Token $x_i (i < t)$ 的 $k_i, v_i$ 向量一旦计算便不再改变。

1.  **定义**
    设 $W_Q, W_K, W_V$ 为注意力层的投影矩阵。
    在时刻 $t$，输入仅为单一 Token $x_t$。

2.  **增量计算过程**
    *   **投影**：仅计算当前 Token 的 Q, K, V。
        $$ q_t = x_t W_Q, \quad k_{new} = x_t W_K, \quad v_{new} = x_t W_V $$
    
    *   **缓存更新 (Cache Update)**：
        将新计算的键值拼接到历史缓存末尾：
        $$ K_{cache}^{(t)} = \text{Concat}(K_{cache}^{(t-1)}, k_{new}) $$
        $$ V_{cache}^{(t)} = \text{Concat}(V_{cache}^{(t-1)}, v_{new}) $$
        此时，$K_{cache}^{(t)}$ 的维度变为 $(t, d_k)$。

    *   **注意力计算**：
        仅计算当前 $q_t$ 与所有历史 $K_{cache}$ 的相关性：
        $$ \text{Score}_t = \frac{q_t (K_{cache}^{(t)})^T}{\sqrt{d_k}} \in \mathbb{R}^{1 \times t} $$
        $$ \text{Output}_t = \text{softmax}(\text{Score}_t) V_{cache}^{(t)} $$

3.  **复杂度优化**
    引入 KV Cache 后，第 $t$ 步的计算量主要为矩阵乘法 $1 \times t$，复杂度降为 $O(t)$。
    生成 $N$ 个 Token 的总复杂度为：
    $$ \text{Cost}_{cache} \approx \sum_{t=1}^{N} O(t) \approx O(N^2) $$
    相比朴素方法的 $O(N^3)$，实现了显著加速。若仅考虑“重复编码历史”这一项开销，则从 $O(N^2)$ 降为了 $O(N)$（线性）。这一优化对于长上下文对话至关重要，虽然它带来了显存占用（Memory Bound）的增加。

## 2.3 系统评价指标 (Evaluation Metrics)

为了定量评估系统性能，本研究采用延迟与准确率双重指标。

### 2.3.1 自动语音识别准确率 (WER/CER)
使用编辑距离（Levenshtein Distance）衡量识别结果与标注文本的差异。

*   **词错误率 (Word Error Rate, WER)** - 用于英文：
    $$ \text{WER} = \frac{S + D + I}{N} \times 100\% $$
    
*   **字符错误率 (Character Error Rate, CER)** - 用于中文：
    $$ \text{CER} = \frac{S + D + I}{N_{char}} \times 100\% $$

其中：
*   $S$ (Substitution): 替换错误数
*   $D$ (Deletion): 删除错误数
*   $I$ (Insertion): 插入错误数
*   $N$ / $N_{char}$: 參考文本的总词数/字数

### 2.3.2 首字延迟 (Time to First Token, TTFT)
TTFT 是衡量人机交互流畅度的核心指标，定义为从用户停止说话到系统输出第一个文本 Token 的时间差。

$$ \text{TTFT} = T_{first\_token} - T_{audio\_end} $$

*   **$T_{audio\_end}$**: VAD 检测到语音结束或音频流输入的最后时刻。
*   **$T_{first\_token}$**: LLM 解码器产出首个 Token 的时刻。

在流式架构中，理想的 TTFT 应趋近于常数，不随输入语音长度 $L$ 显著增加，即 $\frac{\partial \text{TTFT}}{\partial L} \approx 0$。

---
**参考文献**
[1] Radford, A., et al. (2023). Robust Speech Recognition via Large-Scale Weak Supervision.
[2] Vaswani, A., et al. (2017). Attention Is All You Need.
[3] Pope, R., et al. (2023). Efficiently Scaling Transformer Inference.
