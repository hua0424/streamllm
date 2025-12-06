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
\REQUIRE Current KV Cache $C_{prev} = (K_{prev}, V_{prev})$, New Text Fragment $T_{new}$, LLM Model $\mathcal{M}$
\ENSURE Updated KV Cache $C_{new}$, Next Token Logits $L$

\STATE \textbf{Step 1: Tokenization}
\STATE $Ids_{new} \leftarrow \text{Tokenizer}(T_{new})$
\IF{$Ids_{new}$ is empty}
    \RETURN $C_{prev}, \text{None}$
\ENDIF

\STATE \textbf{Step 2: Attention Mask Construction}
\STATE $Mask_{prev} \leftarrow \text{GetMask}(C_{prev})$
\STATE $Mask_{new} \leftarrow \text{Ones}(\text{Shape}(Ids_{new}))$
\STATE $Mask_{full} \leftarrow \text{Concat}([Mask_{prev}, Mask_{new}], \text{dim}=-1)$

\STATE \textbf{Step 3: Position ID Alignment}
\STATE $L_{prev} \leftarrow \text{Length}(K_{prev})$
\STATE $Pos_{new} \leftarrow [L_{prev}, L_{prev} + 1, \dots, L_{prev} + \text{Length}(Ids_{new}) - 1]$

\STATE \textbf{Step 4: Forward Pass (Incremental)}
\STATE \COMMENT{Only compute attention for new tokens using cached history}
\STATE $Out \leftarrow \mathcal{M}.\text{forward}(input\_ids=Ids_{new}, past\_key\_values=C_{prev}, mask=Mask_{full}, pos=Pos_{new})$

\STATE \textbf{Step 5: Cache Update}
\STATE $K_{new} \leftarrow \text{Concat}([K_{prev}, Out.K_{new}], \text{dim}=1)$
\STATE $V_{new} \leftarrow \text{Concat}([V_{prev}, Out.V_{new}], \text{dim}=1)$
\STATE $L \leftarrow Out.logits[-1]$ \COMMENT{Logits for the last token}

\RETURN $(K_{new}, V_{new}), L$
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

为了在受控环境下验证系统对不同时长语音的处理能力，我们构建了一套自动化的数据合成管线，将文本对话数据集转化为具有精确时长的长语音测试用例。这对应 `experiments/datasets/tools/run_pipeline.py` 的实现逻辑。

### 3.4.1 数据处理流程

数据管线包含三个严格顺序执行的阶段 (Phases)：

1.  **历史累积与筛选 (History Accumulation)**:
    *   源数据：MultiWOZ (英文) 和 CrossWOZ (中文)。
    *   处理逻辑：遍历对话的每一轮 $Turn_i$，将 $Turn_0$ 到 $Turn_i$ 的所有用户和系统回复拼接，模拟用户一口气说出长段文字的场景。
    *   筛选策略：按累积文本长度倒序排列，保留 Top-N 个最长对话，确保存储了从 3 秒到 60 秒以上的丰富时长样本。

2.  **并发 TTS 音频生成 (Batch TTS Generation)**:
    *   利用 CosyVoice 大模型 TTS 服务，通过 HTTP API 进行语音合成。
    *   为了提高构建效率，实现了 `BatchTTSProcessor`，采用多线程并发 (Thread Pool) 请求 TTS 服务。
    *   输出：生成采样率为 22050Hz 的 WAV 文件。

3.  **时长校准与元数据同步 (Duration Calibration)**:
    *   由于 TTS 生成的语速具有随机性，必须对 JSON 元数据进行“回填”校准。
    *   **代码实现**：调用 `wave` 标准库读取生成的 WAV 文件头，计算精确时长：
        $$ Duration = \frac{TotalFrames}{SampleRate} $$
    *   系统将此 `audio_duration` 写入任务 JSON，作为后续实验中 $X$ 轴 (Input Duration) 的真值依据，确保了实验数据的严谨性。

[流程图占位: 数据与实验配置生成流程]
*(描述：Raw JSON -> [Accumulator] -> Task JSON (Text only) -> [TTS Engine] -> WAV Files -> [Duration Updater] -> Final Experiment JSON (Text + Audio + Duration))*
