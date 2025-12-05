# 实验设计方案：级联式语音对话系统的延迟优化

本文档用于规划硕士学位论文《级联式语音对话系统的延迟优化》的实验部分。实验旨在验证流式处理（Streaming）与 KV 缓存预填充（KV Cache Prefill）技术在降低长语音交互延迟方面的有效性。

## 一、 实验目录结构

```
experiments/
├── datasets/                     # 数据集相关
│   ├── raw_data/                 # [只读] 原始数据集存放位置
│   │   ├── MultiWOZ/             # 英文多轮对话数据集 (文本)
│   │   └── CrossWOZ/             # 中文跨域对话数据集 (文本)
│   ├── processed/                # [生成] 实验用数据 (由 tools 生成)
│   │   ├── json/                 # 任务描述文件 (含文本、时长等元数据)
│   │   │   ├── crosswoz/         # CrossWOZ 处理后的 JSON
│   │   │   └── multiwoz/         # MultiWOZ 处理后的 JSON
│   │   └── audio/                # 生成的音频文件 (.wav)
│   │       ├── crosswoz/
│   │       └── multiwoz/
│   └── tools/                    # 数据处理工具代码
│       ├── data_processor.py     # 数据预处理模块
│       ├── run_pipeline.py       # 数据处理管线主程序
│       ├── tts.py                # TTS 客户端 (支持多并发)
│       ├── doc/                  # 工具使用文档
│       │   ├── PIPELINE_USAGE.md
│       │   └── TTS_USAGE.md
│       └── scripts/              # 运行脚本
│           └── run_data_pipeline.sh
├── scripts/                      # [待开发] 实验运行脚本
│   ├── run_exp_latency.py        # 实验一：延迟与长度关系
│   ├── run_exp_ablation.py       # 实验二：消融实验
│   └── run_exp_quality.py        # 实验三：准确率验证
└── results/                      # [自动生成] 实验结果数据与图表
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

1.  **TTFT (Time to First Token)**: 首字延迟。定义为从**语音输入结束**到**LLM生成第一个Token**的时间差。这是衡量用户"等待感"的核心指标。
2.  **WER/CER (Word/Character Error Rate)**: 词/字错误率。用于衡量 ASR 转录的准确性。
3.  **Latency Improvement**: 延迟优化率。计算公式：$(TTFT_A - TTFT_B) / TTFT_A \times 100\%$。

---

## 三、 数据准备

### 3.1 数据筛选策略

为了获取足够长的语音输入数据用于实验，我们采用以下策略：

1.  **筛选长对话**：计算每个对话的总文本长度（所有轮次累积），按长度降序排序，选取前 100 条对话
2.  **累积对话生成**：对每个对话的每个用户轮次，生成累积的对话文本

### 3.2 累积对话逻辑

| 轮次 | 发言者 | 原始文本 | 累积输出 | 输出时机 |
|------|--------|----------|----------|----------|
| 1 | 用户 (a1) | "你好，推荐一个景点" | "你好，推荐一个景点" | ✓ 生成 turn1 |
| 2 | 系统 (b1) | "推荐颐和园" | - | 累积但不输出 |
| 3 | 用户 (a2) | "门票多少钱" | "你好，推荐一个景点 推荐颐和园 门票多少钱" | ✓ 生成 turn2 |
| 4 | 系统 (b2) | "30元" | - | 累积但不输出 |
| 5 | 用户 (a3) | "怎么去" | "你好... 门票多少钱 30元 怎么去" | ✓ 生成 turn3 |

通过这种方式，随着 turn 增加，输入文本长度递增，生成的音频时长也随之增加，便于分析不同长度输入对系统延迟的影响。

### 3.3 数据处理管线

数据处理分为三个阶段：

**阶段1：数据预处理 (Data Preprocessing)**
- 读取原始数据集 JSON 文件
- 按总文本长度筛选前 N 个对话
- 生成累积对话格式的任务 JSON 文件

**阶段2：TTS 音频生成 (TTS Batch Processing)**
- 调用 CosyVoice TTS 服务
- 多并发处理提高效率（默认 4 并发）
- 生成 WAV 格式音频文件（22050Hz）

**阶段3：更新音频时长 (Update Audio Duration)**
- 读取生成的音频文件
- 获取实际时长，写入 JSON 文件

### 3.4 输出数据格式

每个样本对应一个 JSON 文件：

```json
{
  "sample_id": "crosswoz_391_turn3",
  "dialog_id": "391",
  "turn_index": 3,
  "text": "累积的对话文本...",
  "text_length": 256,
  "audio_file": "crosswoz_391_turn3.wav",
  "audio_duration": 18.52,
  "language": "zh",
  "dataset": "crosswoz"
}
```

### 3.5 数据分组

根据 `audio_duration` 字段，将样本分为以下组别用于实验：

| 分组 | 时长范围 | 典型轮次 | 用途 |
|------|----------|----------|------|
| 短语音 | < 5s | turn1 | 基准测试 |
| 中等语音 | 5-15s | turn2-3 | 主要对比 |
| 长语音 | 15-30s | turn4-6 | 效果验证 |
| 超长语音 | > 30s | turn7+ | 极限测试 |

---

## 四、 详细实验设计

### 实验一：延迟与语音长度的关系验证 (Effect Validation)

**实验目的**：
验证本项目的核心假设：随着语音输入长度的增加，流式方案 (System B) 的 TTFT 保持相对稳定，而非流式方案 (System A) 的 TTFT 呈线性增长。同时寻找"交叉点" (Crossover Point)，即流式方案开始优于非流式方案的音频时长阈值。

**数据准备**：
1.  使用数据处理管线生成的音频样本
2.  根据 `audio_duration` 分组（短/中/长/超长）
3.  每组至少 50 个样本

**执行步骤**：
1.  对每个时长组的音频，分别运行 System A 和 System B
2.  记录每次运行的 `audio_duration` 和 `ttft`
3.  统计每个组的平均 TTFT 和标准差

**预期结果**：
- 绘制折线图：X轴为音频时长，Y轴为 TTFT
- System A 的曲线应随 X 轴线性上升
- System B 的曲线应趋于平缓（主要取决于最后一段音频的处理时间，而非总时长）

### 实验二：消融实验 (Ablation Study)

**实验目的**：
量化分解"流式 ASR"和"LLM 流式缓存"两个模块对整体延迟优化的贡献度。

**实验设置**：
选取实验一中效果最明显的"长语音组" (例如 15s-30s) 进行测试。对比以下三种配置：

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
1.  选取包含 200 条不同内容的音频测试集
2.  **ASR 准确性**：
    *   运行 System A (Whisper 完整识别)，记录转录文本 $T_{offline}$
    *   运行 System B (Whisper 流式识别)，记录转录文本 $T_{streaming}$
    *   以原始数据集的 Ground Truth 文本为基准，分别计算 System A 和 B 的 WER (英文) 和 CER (中文)
3.  **LLM 语义一致性 (可选扩展)**：
    *   比较 System A 和 System B 生成的最终回复 $R_A$ 和 $R_B$
    *   计算 BERTScore 或语义相似度，确保 ASR 的细微差异没有导致 LLM 回复产生幻觉或逻辑错误

---

## 五、 实施路线图

### 5.1 数据准备 ✅ 已完成

- [x] 实现数据预处理模块 (`data_processor.py`)
- [x] 实现数据处理管线 (`run_pipeline.py`)
- [x] 实现 TTS 客户端，支持多并发 (`tts.py`)
- [x] 编写使用文档和运行脚本

**运行方式**：
```bash
conda activate streamllm

# 完整管线（预处理 + TTS + 更新时长）
uv run python -m experiments.datasets.tools.run_pipeline

# 使用 8 并发加速 TTS
uv run python -m experiments.datasets.tools.run_pipeline --tts-workers 8

# 仅预处理（跳过 TTS）
uv run python -m experiments.datasets.tools.run_pipeline --skip-tts
```

### 5.2 实验脚本开发 (待完成)

- [ ] 基于 `src/run_test_simple.py` 改造，编写批量测试脚本
- [ ] 实现自动记录日志到 CSV/JSON 的功能
- [ ] 实现按时长分组运行实验的功能

### 5.3 执行与分析 (待完成)

- [ ] 在 GPU 环境下运行实验脚本
- [ ] 收集实验数据
- [ ] 使用 Python (Matplotlib/Seaborn) 绘制论文所需的图表
