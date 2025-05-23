# 智能语音对话系统低延迟优化研究 (StreamLLM-LowLatency)

## 1. 项目目标

本项目旨在研究和实现一套低延迟智能语音对话系统。核心目标是优化级联式语音对话系统（ASR -> LLM -> TTS）中从用户语音输入开始到大语言模型（LLM）产生首个响应Token的端到端延迟。通过采用流式处理和KV缓存等技术，力求显著降低系统的响应时间，提升用户交互体验。

## 2. 研究方法

### 2.1 自动语音识别 (ASR) 优化

- **核心技术**: 采用 [faster-whisper](https://github.com/SYSTRAN/faster-whisper) 作为ASR引擎，利用其高效的语音转文本能力。
- **流式处理与分段**:
    - 实时接收语音输入流。
    - 根据语音中的停顿（静音段）动态地将语音流分割成较短的语音段。
    - 将分割后的语音段流式输入 `faster-whisper` 进行识别，以减少单次识别的音频长度，从而缩短识别延迟。
- **上下文感知与文本替换**:
    - 为提高识别准确率，采用重叠的语音段输入策略。例如：
        - 时间步1 输入: 语音段1, 语音段2, 语音段3 -> 识别文本T1
        - 时间步2 输入: 语音段2, 语音段3, 语音段4 -> 识别文本T2
        - 时间步3 输入: 语音段3, 语音段4, 语音段5 -> 识别文本T3
    - 利用 `faster-whisper` 的单词级时间戳（word-level timestamps）功能。
    - 记录每个语音段在原始音频流中的绝对起始和结束时间。
    - 根据时间戳信息，将后续时间步识别结果中与前序时间步重叠部分的新识别文本，精确替换掉旧的文本，确保最终拼接文本的准确性和连贯性。例如，最终文本由 (T1中段1的文本) + (T2中段2的新识别部分或确认部分) + (T3中段3、4、5的新识别部分) 构成。

### 2.2 大语言模型 (LLM) 流式响应优化

- **核心技术**: 利用 Transformer 模型的 KV 缓存机制。
- **流式输入与提前处理**:
    - ASR模块产生的文本会以流式方式片段化输入LLM。
    - 在完整的用户语句形成之前，针对已接收到的文本片段，提前进行LLM的KV缓存计算。
    - 当用户语句的某个阶段性文本片段输入完成时（例如，一个完整的句子或意图单元），LLM可以基于已预计算的KV缓存快速生成后续token。



## 3. 实验设计

- **数据集**:
    - 准备或整理适用于语音对话的公开或私有语音数据集。
    - 数据集应包含多样化的语音内容和说话人特征。
    - 将语音对话数据集，按输入语音的长度划分为不同的长度范围，以评估不同的语音输入长度的效果
- **评估指标**:
    - **首个Token延迟 (First Token Latency)**: 从用户语音输入开始，到LLM生成第一个有效token的时间。这是本项目的核心优化指标。按不同的输入语音长度分析效果。
    - **ASR准确率 (Word Error Rate - WER)**: 评估流式ASR方法在保证低延迟的同时，对识别准确率的影响。按不同长度的语音段来测试效果，测试在多少长度的语音段进行ASR时，准确率曲线。该指标只能以较长的语音输入数据集来进行。

- **实验流程**:
    1.  搭建基线系统（非流式ASR + 非KV缓存LLM）。
    2.  逐步集成流式ASR模块，测试其对首个Token延迟及WER的影响。
    3.  集成LLM的KV缓存优化，测试其对首个Token延迟的进一步改善。
    4.  对比不同参数配置（如语音分段策略、KV缓存更新频率等）下的系统性能。
    5.  记录并分析实验数据，撰写研究报告。

## 4. 项目结构 (初步)

```
streamllm-lowlatency/
├── .env.example             # 环境变量配置示例
├── .gitignore               # Git忽略文件配置
├── README.md                # 项目说明 (本文档)
├── requirements.txt         # Python依赖包
│
├── data/                    # 实验数据
│   ├── raw_audio/           # 原始语音数据集
│   ├── processed_audio/     # 预处理后的语音片段
│   └── transcripts/         # 语音对应的标准文本
│
├── notebooks/               # Jupyter Notebooks，用于实验、分析和可视化
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_asr_experiments.ipynb
│   └── 03_llm_kv_cache_experiments.ipynb
│
├── results/                 # 实验结果数据
│   ├── latency_metrics/
│   └── wer_scores/
│
├── src/                     # 核心源代码
│   ├── __init__.py
│   ├── config.py            # 项目配置文件 (模型路径、参数等)
│   ├── asr/                 # ASR相关模块
│   │   ├── __init__.py
│   │   ├── faster_whisper_streamer.py # 流式faster-whisper实现
│   │   └── audio_segmenter.py       # 语音分段逻辑
│   ├── llm/                 # LLM相关模块
│   │   ├── __init__.py
│   │   └── stream_llm_inference.py  # 流式LLM推理与KV缓存管理 (基于stream_input_cache.py)
│   ├── tts/                 # TTS相关模块 (待开发)
│   │   └── __init__.py
│   ├── pipeline/            # ASR-LLM-(TTS) 级联处理流程编排
│   │   ├── __init__.py
│   │   └── streaming_pipeline.py
│   └── utils/               # 通用工具模块
│       ├── __init__.py
│       ├── audio_utils.py   # 音频处理相关工具
│       └── logging_utils.py # 日志记录工具
│
├── scripts/                 # 命令行脚本
│   ├── run_experiment.py    # 运行完整实验流程的脚本
│   └── evaluate_results.py  # 评估实验结果的脚本
│
└── stream_input_cache.py    # 您提供的LLM KV缓存初步实验代码 (可整合或重构到src/llm/)
```

## 5. 安装与运行 (待补充)

(详细的安装步骤、环境配置和运行指南将在后续开发中补充)

## 6. 贡献指南 (待补充)

(如果项目计划开源，此处可以添加贡献指南)
