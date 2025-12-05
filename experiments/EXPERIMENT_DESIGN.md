# 实验设计方案：级联式语音对话系统的延迟优化

本文档用于规划硕士学位论文《级联式语音对话系统的延迟优化》的实验部分。实验旨在验证流式处理（Streaming）与 KV 缓存预填充（KV Cache Prefill）技术在降低长语音交互延迟方面的有效性。

## 一、 实验目录结构规划

```
experiments/
├── raw_data/                 # [只读] 原始数据集存放位置
│   ├── MultiWOZ/             # 英文多轮对话数据集 (文本)
│   └── CrossWOZ/             # 中文跨域对话数据集 (文本)
├── processed/                # [生成] 实验用数据 (由 tools 生成)
│   ├── audio_pool/           # 所有生成的音频文件 (.wav)
│   ├── metadata.json         # 音频元数据 (ID, 对应文本, 时长, 来源数据集)
│   └── groups/               # 按时长分组的测试列表
│       ├── short/            # 短语音 (< 5s)
│       ├── medium/           # 中等语音 (5s - 15s)
│       └── long/             # 长语音 (> 15s)
├── tools/                    # 数据处理工具代码
│   ├── tts/                  # 文本转语音工具 (已实现)
│   ├── doc/                  # 工具使用文档
│   └── prepare_dataset.py    # [待开发] 数据预处理主脚本：文本->TTS->音频分类
├── scripts/                  # [待开发] 实验运行脚本
│   ├── run_exp_latency.py    # 实验一：延迟与长度关系
│   ├── run_exp_ablation.py   # 实验二：消融实验
│   └── run_exp_quality.py    # 实验三：准确率验证
└── results/                  # [自动生成] 实验结果数据与图表
    ├── latency_logs/
    ├── ablation_tables/
    └── quality_metrics/
```

## 二、 实验环境与基准

### 2.1 待测系统定义

为了控制变量，我们将对比以下两种系统配置：

| 系统标识 | 系统名称 | 配置描述 | 关键技术 |
| :--- | :--- | :--- | :--- |
| **System A** | **Baseline (非流式基线)** | 完整音频录制 -> 完整ASR转录 -> 完整Prompt送入LLM | 传统的级联架构，无流水线并行 |
| **System B** | **Ours (流式优化方案)** | VAD分段 -> 流式ASR (上下文感知) -> 流式LLM (KV缓存增量预填充) | **Streaming ASR + LLM KV Cache Prefill** |

### 2.2 核心评价指标

1.  **TTFT (Time to First Token)**: 首字延迟。定义为从**语音输入结束**到**LLM生成第一个Token**的时间差。这是衡量用户“等待感”的核心指标。
2.  **WER/CER (Word/Character Error Rate)**: 词/字错误率。用于衡量 ASR 转录的准确性。
3.  **Latency Improvement**: 延迟优化率。计算公式：$(TTFT_A - TTFT_B) / TTFT_A \times 100\%$。

---

## 三、 详细实验设计

### 实验一：延迟与语音长度的关系验证 (Effect Validation)

**实验目的**：
验证本项目的核心假设：随着语音输入长度的增加，流式方案 (System B) 的 TTFT 保持相对稳定，而非流式方案 (System A) 的 TTFT 呈线性增长。同时寻找“交叉点” (Crossover Point)，即流式方案开始优于非流式方案的音频时长阈值。

**数据准备**：
1.  从 MultiWOZ (英) 和 CrossWOZ (中) 中随机抽取对话语句。
2.  使用 TTS 工具生成音频，覆盖 1s 到 60s 的时长范围。
3.  将音频按时长分组：
    *   Group 1: 1-5s
    *   Group 2: 5-10s
    *   ...
    *   Group N: >30s

**执行步骤**：
1.  对每个时长组的音频，分别运行 System A 和 System B。
2.  记录每次运行的 `audio_duration` 和 `ttft`。
3.  统计每个组的平均 TTFT 和标准差。

**预期结果**：
- 绘制折线图：X轴为音频时长，Y轴为 TTFT。
- System A 的曲线应随 X 轴线性上升。
- System B 的曲线应趋于平缓（主要取决于最后一段音频的处理时间，而非总时长）。

### 实验二：消融实验 (Ablation Study)

**实验目的**：
量化分解“流式 ASR”和“LLM 流式缓存”两个模块对整体延迟优化的贡献度。

**实验设置**：
选取实验一中效果最明显的“长语音组” (例如 15s-20s) 进行测试。对比以下三种配置：

1.  **Baseline**: 非流式 ASR + 非流式 LLM
2.  **Ablation 1 (Only Streaming ASR)**: 流式 ASR + 非流式 LLM (LLM 等待最终完整文本，不预填充 KV Cache)
3.  **Ours (Full)**: 流式 ASR + 流式 LLM (KV Cache 预填充)

**分析方法**：
- **流式 ASR 贡献**: $TTFT_{Baseline} - TTFT_{Ablation1}$
- **KV 缓存贡献**: $TTFT_{Ablation1} - TTFT_{Ours}$

### 实验三：准确率与质量边界实验 (Accuracy & Quality)

**实验目的**：
验证流式处理是否牺牲了识别准确率。流式 ASR 由于缺乏未来上下文 (Future Context)，理论上准确率略低于整句识别。本实验旨在量化这种差异，证明其在可接受范围内。

**执行步骤**：
1.  选取包含 200 条不同内容的音频测试集。
2.  **ASR 准确性**：
    *   运行 System A (Whisper 完整识别)，记录转录文本 $T_{offline}$。
    *   运行 System B (Whisper 流式识别)，记录转录文本 $T_{streaming}$。
    *   以原始数据集的 Ground Truth 文本为基准，分别计算 System A 和 B 的 WER (英文) 和 CER (中文)。
3.  **LLM 语义一致性 (可选扩展)**：
    *   比较 System A 和 System B 生成的最终回复 $R_A$ 和 $R_B$。
    *   计算 BERTScore 或 语义相似度，确保 ASR 的细微差异没有导致 LLM 回复产生幻觉或逻辑错误。

## 四、 实施路线图

1.  **阶段一：数据准备 (Data Prep)**
    *   编写 `experiments/tools/prepare_dataset.py`。
    *   读取 `raw_data` 中的 JSON 文本。
    *   调用 `experiments/datasets/tools/tts.py` 批量生成 WAV 文件。
    *   生成 `experiments/processed/metadata.json` 索引文件。

2.  **阶段二：脚本开发 (Scripting)**
    *   基于 `src/run_test_simple.py` 改造，编写批量测试脚本。
    *   实现自动记录日志到 CSV/JSON 的功能。

3.  **阶段三：执行与分析 (Run & Analyze)**
    *   在 GPU 环境下运行实验脚本。
    *   使用 Python (Matplotlib/Seaborn) 绘制论文所需的图表。
