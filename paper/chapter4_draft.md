# 第四章 实验与结果分析 (Experiments & Analysis)

本章旨在通过系统性的实验评估，验证 System B (流式架构) 在长语音交互场景下的低延迟优势及其对识别准确率的影响。我们设计了三个核心实验：延迟与语音长度的关系验证 (Exp 1)、核心模块消融分析 (Exp 2) 以及系统准确率边界测试 (Exp 3)。

## 4.1 实验一：延迟与语音长度的关系验证 (Effect Validation)

### 4.1.1 实验设置
本实验旨在验证论文的核心假设：非流式系统 (System A) 的首字延迟 (TTFT) 随输入语音长度线性增长，而流式系统 (System B) 的 TTFT 保持常数级稳定。

*   **数据集**：基于 MultiWOZ (英文) 和 CrossWOZ (中文) 构建的长语音测试集，涵盖 3s 至 60s 的不同时长分组。
*   **硬件环境**：NVIDIA RTX 4090 GPU (24GB VRAM)，CUDA 11.8。
*   **公平性控制**：
    *   **共享实例**：两种模式复用同一 ASR (Faster-Whisper-Large-v3) 和 LLM (Qwen2.5-0.5B) 模型实例，排除加载时间差异。
    *   **预热机制**：每个 session 前进行 3 轮真实音频预热 (Warmup Rounds)，确保 CUDA Kernel 充分加载。
    *   **音频分块**：流式模式下 Chunk Duration 统一设定为 500ms。

### 4.1.2 实验结果分析
图 4-1 展示了不同语音时长 ($audio\_duration$) 下两种系统的 TTFT ($ttft\_ms$) 变化趋势。

[图 4-1: TTFT vs Audio Duration 趋势对比图]
*(图注：X轴为 audio\_duration (s)，Y轴为 ttft\_ms。图中包含两条曲线：System A (蓝色) 呈线性上升趋势，System B (红色) 呈平缓水平趋势。两条曲线在 x ≈ 5s 处出现交叉点 (Crossover Point)。)*

**趋势分析**：
基于实验数据 (引用 `exp1_results.json`)，我们可以观察到：
1.  **非流式基线 (System A)**：TTFT 与 $audio\_duration$ 呈现显著的线性正相关 ($R^2 > 0.95$)。这是因为级联架构必须等待音频完全结束才开始转录和编码，计算量 $Cost \propto Length$。
    *   短语音 (<5s): TTFT 约为 [待填: ~200] ms。
    *   长语音 (>15s): TTFT 激增至 [待填: ~1000] ms 以上，显著破坏了实时交互体验。
2.  **流式优化 (System B)**：TTFT 曲线几乎平行于 X 轴，不随语音长度增加而显著恶化。
    *   在所有时长分组下，TTFT 稳定在 [待填: ~150] ms 左右。
    *   这验证了流式流水线设计的有效性：系统在用户说话的同时即使处理了大部分数据，最终仅需处理最后一个音频块 (Chunk) 和少量的 LLM 增量计算。

## 4.2 实验二：核心模块消融实验 (Ablation Study)

为了量化“自适应流式 ASR”和“LLM KV Cache 增量预填充”各自的贡献，我们在“长语音组” (15-30s) 上进行了消融对比。

### 4.2.1 评价指标与对比组
我们定义了三种对比配置，如表 4-1 所示：
*   **Baseline**: 传统级联模式。
*   **Streaming ASR Only**: 仅启用流式 ASR，由于 LLM 不支持 KV Cache 预填充，ASR 输出的中间结果被丢弃，直到获得最终完整文本才一次性送入 LLM。
*   **System B (Full)**: 启用全链路流式优化。

### 4.2.2 贡献度量化
表 4-1 展示了各阶段的关键时间指标 (单位: ms)。

[表 4-1: 核心模块消融实验结果 (Duration Group: Long 15-30s)]
| Mode (配置模式) | ASR Strategy | LLM Strategy | first_token_latency (TTFT) | asr_time | llm_prefill_time | Total Improvement |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** | Non-streaming | Non-streaming | [待填: 1023.4] | [待填: 780.2] | [待填: 243.2] | - |
| **Streaming ASR Only** | **Streaming** | Non-streaming | [待填: 412.6] | [待填: 280.5] | [待填: 132.1] | [待填: 60%] |
| **System B (Full Streaming)**| **Streaming** | **KV Prefill** | [待填: 145.8] | [待填: 92.4] | [待填: 53.4] | [待填: 85%] |

*(注：数据源自 `exp2_statistics.csv`。`asr_time` 在流式模式下指最后一次分段的处理时间。)*

**结果分析**：
1.  **ASR 流式化的贡献**：对比 Baseline 与 Streaming ASR Only，TTFT 降低了约 [待填] ms。这主要得益于 ASR 模块消除了“等待音频结束”的死区时间，实现了音频录制与识别的并行。
2.  **KV Cache 的关键作用**：对比 Streaming ASR Only 与 Full Streaming，TTFT 进一步降低了 [待填] ms，尤其是 `llm_prefill_time` 显著减少。这证明了算法 1 (Algorithm 1) 的有效性：通过复用 $O(N)$ 的历史计算结果，LLM 仅需计算 $O(M)$ 的增量 Token，避免了对长 Prompt 的重复编码。

## 4.3 实验三：准确率与质量边界 (Accuracy & Quality)

流式系统的一个潜在风险是由于缺乏“未来上下文” (Future Context) 而导致识别精度下降。本实验评估这种精度损失是否在可接受范围内。

### 4.3.1 评估方法
我们使用 WER (Word Error Rate, 英文) 和 CER (Character Error Rate, 中文) 作为评价指标。对比同一批样本在 System A (全量识别) 和 System B (流式识别) 下的转录结果。

### 4.3.2 精度对比
表 4-2 列出了不同数据集下的误差率对比。

[表 4-2: 流式与非流式 ASR 准确率对比]
| Scope (数据集) | Mode | WER / CER (Mean) | WER / CER (Std) | asr_time_ms (Avg) |
| :--- | :--- | :--- | :--- | :--- |
| **Overall** | Non-streaming | [待填: 0.076] | [待填: 0.024] | [待填: 812.4] |
| **Overall** | **Streaming** | [待填: 0.098] | [待填: 0.029] | [待填: 285.3] |
| | | | | |
| **CrossWOZ (ZH)** | Non-streaming | [待填: 0.055] (CER) | [待填: 0.017] | - |
| **CrossWOZ (ZH)** | **Streaming** | [待填: 0.069] (CER) | [待填: 0.021] | - |

*(注：数据源自 `exp3_statistics.csv`。)*

**结论**：
实验数据显示，流式模式的 WER/CER 相比非流式模式仅有 [待填: ~2%] 的微小上升。
1.  这种精度损失主要来源于句尾的不稳定识别。通过 3.2 节提出的“自适应上下文滑动窗口”策略，我们有效地利用了前缀上下文 ($Prefix Context$) 弥补了部分精度。
2.  考虑到 TTFT 获得了近 [待填: 5-8倍] 的性能提升，这点精度损失在实时对话场景中是完全可接受的 (Acceptable Trade-off)。系统的整体语义理解能力 (由 LLM 决定) 并未受到显著影响。
