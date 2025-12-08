## 3.3 LLM KV Cache 增量预填充策略 (Incremental KV Cache Prefilling)

在传统架构中，ASR 输出的每次更新（例如 $Text_t = \text{"Hello"}$ 更新为 $Text_{t+1} = \text{"Hello world"}$），都会触发 LLM 对整个 $Text_{t+1}$ 序列的重新编码。这种重复计算不仅浪费算力，更导致首字延迟 (TTFT) 随上下文长度线性甚至超线性增长。

本系统引入增量预填充机制，利用 Transformer 解码器的自回归特性，将历史 Token 的 Key/Value 状态 ($KV_{cache}$) 持久化。当 ASR 产生新文本片段时，系统仅计算新增加 Token 的 Attention，并将其 KV 值拼接至缓存末尾。

### 3.3.1 核心算法实现
核心逻辑封装于 `StreamLLMInference` 类中。算法 1 展示了这一协同过程。

**Algorithm 1: Incremental KV Cache Prefill**
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

[图 3-3: LLM KV Cache 增量更新机制]
*(图注：示意图展示 Transformer 的 K/V 矩阵拼接过程。左侧为时刻 t 的 $Cache_{prev}$ 矩阵块，右侧为新输入的 $Text_{new}$ 经过 Projection 得到的小矩阵块，两者通过 Concat 操作合并为 $Cache_{new}$，避免了对左侧长矩阵的重新计算。)*

### 3.3.2 复杂度分析 (Complexity Analysis)
我们对比 System A (无缓存全量计算) 与 System B (增量计算) 在处理长度为 $N$ 的历史上下文和长度为 $M$ 的新片段时的计算复杂度。主要开销来源于 Transformer 的 Self-Attention 层。

*   **System A (Baseline)**: 需要对整个序列 $L = N + M$ 进行 Attention 计算。
    Attention 矩阵维度为 $(N+M) \times (N+M)$。
    计算复杂度：$O((N+M)^2) \approx O(N^2)$ (当 $N >> M$ 时)。

*   **System B (Ours)**: 仅计算新 Token 与所有 Token (历史 + 新增) 的 Attention。
    Query 向量长度为 $M$，Key/Value 向量长度为 $N+M$。
    计算复杂度：$O(M \cdot (N+M)) \approx O(N \cdot M)$。
## 3.3 LLM KV Cache 增量预填充策略 (Incremental KV Cache Prefilling)

在传统架构中，ASR 输出的每次更新（例如 $Text_t = \text{"Hello"}$ 更新为 $Text_{t+1} = \text{"Hello world"}$），都会触发 LLM 对整个 $Text_{t+1}$ 序列的重新编码。这种重复计算不仅浪费算力，更导致首字延迟 (TTFT) 随上下文长度线性甚至超线性增长。

本系统引入增量预填充机制，利用 Transformer 解码器的自回归特性，将历史 Token 的 Key/Value 状态 ($KV_{cache}$) 持久化。当 ASR 产生新文本片段时，系统仅计算新增加 Token 的 Attention，并将其 KV 值拼接至缓存末尾。

### 3.3.1 核心算法实现
核心逻辑封装于 `StreamLLMInference` 类中。算法 1 展示了这一协同过程。

**Algorithm 1: Incremental KV Cache Prefill**
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

[图 3-3: LLM KV Cache 增量更新机制]
*(图注：示意图展示 Transformer 的 K/V 矩阵拼接过程。左侧为时刻 t 的 $Cache_{prev}$ 矩阵块，右侧为新输入的 $Text_{new}$ 经过 Projection 得到的小矩阵块，两者通过 Concat 操作合并为 $Cache_{new}$，避免了对左侧长矩阵的重新计算。)*

### 3.3.2 复杂度分析 (Complexity Analysis)
我们对比 System A (无缓存全量计算) 与 System B (增量计算) 在处理长度为 $N$ 的历史上下文和长度为 $M$ 的新片段时的计算复杂度。主要开销来源于 Transformer 的 Self-Attention 层。

*   **System A (Baseline)**: 需要对整个序列 $L = N + M$ 进行 Attention 计算。
    Attention 矩阵维度为 $(N+M) \times (N+M)$。
    计算复杂度：$O((N+M)^2) \approx O(N^2)$ (当 $N >> M$ 时)。

*   **System B (Ours)**: 仅计算新 Token 与所有 Token (历史 + 新增) 的 Attention。
    Query 向量长度为 $M$，Key/Value 向量长度为 $N+M$。
    计算复杂度：$O(M \cdot (N+M)) \approx O(N \cdot M)$。

**证明**：
由于 $M$ (流式增量片段，通常 < 10 tokens) 远小于 $N$ (历史对话上下文，可达数千 tokens)，因此 System B 将复杂度从“上下文长度的平方”降低到了“线性”，从而在长语音交互中实现了近乎恒定的首字延迟。

## 3.4 实验数据构建与处理管线 (Data Construction Methodology)

为了在受控环境下系统性验证流式架构的性能，本研究构建了一套自动化数据合成管线。该管线将 MultiWOZ (英文) 和 CrossWOZ (中文) 等标准对话数据集转化为具有精确时长标注的长语音测试基准。

### 3.4.1 累积对话生成逻辑 (Cumulative Dialogue Strategy)

为了模拟用户进行长语音输入的真实场景（例如一口气陈述复杂的旅行需求），本实验采用“累积对话”策略。如表 3-1 所示，我们将多轮对话的历史文本进行拼接，构造出长度递增的输入序列。

[表 3-1: 累积对话生成示例]
| Turn Index | Speaker | Content | Cumulative Input Text (Simulated User Speech) |
| :--- | :--- | :--- | :--- |
| 1 | User | "订一张机票" | "订一张机票" |
| 2 | System | "去哪里？" | (Not used in input) |
| 3 | User | "去北京，明天出发" | "订一张机票 去北京，明天出发" |
| 5 | User | "要商务舱" | "订一张机票... 明天出发 要商务舱" |

通过此策略，我们能够基于真实语义生成 3秒至 60秒+ 的连续语音样本，有效覆盖了短指令到长陈述的各种交互形态。

### 3.4.2 数据处理流水线 (Processing Pipeline)

数据构建过程包含三个严格顺序执行的阶段 (Phases)，代码实现对应 `experiments/datasets/tools/run_pipeline.py`：

1.  **历史累积与筛选 (Accumulation & Filtering)**:
    *   **策略**：遍历对话数据集，应用累积生成逻辑。
    *   **筛选**：为了聚焦长语音性能，系统按文本长度倒序排列，优先选取最长的前 $N$ 个对话。
    *   **截断**：设置最大文本长度阈值（英文 2050 字符，中文 720 字符），防止生成超过 3 分钟的异常样本。

2.  **并发 TTS 音频合成 (Batch TTS Synthesis)**:
    *   **引擎**：集成 CosyVoice 大模型语音合成服务，生成高保真语音波形。
    *   **并发**：实现 `BatchTTSProcessor`，采用多线程 (Thread Pool) 异步请求，显著提升大规模数据生成的效率。
    *   **输出**：统一采样率为 22050Hz 的 WAV 文件。

3.  **时长校准与元数据同步 (Duration Calibration)**:
    *   由于 TTS 生成的语速具有非确定性，简单的文本估算不够准确。本管线通过解析生成的 WAV 文件头（Header）来获取精确到毫秒的物理时长：
        $$ Duration = \frac{TotalFrames}{SampleRate} $$
    *   该真实时长 ($audio\_duration$) 被写入测试元数据 JSON，作为后续实验中 $X$ 轴（输入时长）的 Ground Truth 依据。

### 3.4.3 测试集分组定义 (Benchmark Grouping)

为了细粒度分析不同时长下的延迟表现，我们将生成的样本集依据 $AudioDuration$ 划分为四个标准实验组（见表 3-2）。

[表 3-2: 实验数据分组定义]
| Group Name | Duration Range | Typical Use Case | Experiment Focus |
| :--- | :--- | :--- | :--- |
| **Short** | $T < 5s$ | 短指令 (Short Commands) | 基准延迟验证 |
| **Medium** | $5s \le T < 15s$ | 把持多轮意图的陈述 | 日常对话性能 |
| **Long** | $15s \le T < 30s$ | 复杂长难句/长段落 | **流式架构核心优势区间** |
| **Extra Long**| $T \ge 30s$ | 极限压力测试 | 系统稳定性与显存边界 |

这一分组标准将在第四章的实验分析中贯穿始终。
