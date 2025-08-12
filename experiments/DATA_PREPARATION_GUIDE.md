# StreamLLM 数据准备指南 - 三数据集方案

## 📋 数据需求概述

基于《级联式语音对话系统的延迟优化》论文的**5个核心实验**，我们采用**三个知名数据集**快速准备实验数据：

### 🎯 选定数据集
1. **CS-Dialogue** (2024): 中英代码切换对话语音数据集 → 中英混合问题
2. **DailyDialog**: 英文日常交流对话文本数据集 → 英文问题  
3. **LCCC**: 中文对话数据集 → 中文问题

### 🎯 专门的语音对话数据集

#### **1. 多语言语音对话数据集 (2024年推荐)**

**CS-Dialogue: 中英代码切换对话数据集 (2024)**
- **特点**: 104小时中英文代码切换自然对话
- **语言**: 中英混合，真实对话场景
- **规模**: 完整对话录音，全面转录标注
- **获取**: [arXiv:2502.18913](https://arxiv.org/html/2502.18913v1)
- **适用性**: 最新多语言ASR基准，代码切换场景

### 🔄 文本对话数据集(需TTS转换) - 主要数据源

### A. 英文对话数据集

#### **DailyDialog**
- **特点**: 日常英文交流对话
- **规模**: 1.3万个多轮对话
- **场景**: 日常生活场景，自然表达
- **获取**: [GitHub](https://github.com/Deepfakes/DailyDialog)
- **优势**: 贴近真实日常对话

### C. 中文对话数据集
#### **LCCC中文对话数据集**
- **规模**: 1200万轮对话 (选择高质量子集)
- **特点**: 
  - 闲聊对话为主
  - 自然对话表达
  - 多样化话题
- **获取**: [GitHub](https://github.com/thu-coai/CDial-GPT)


## 💾 数据格式要求

### 音频数据格式（基于实际项目结构）
```
experiments/datasets/
├── raw_data/                        # 原始数据集（已存在）
│   ├── CS-Dialogue/                 # 中英混合语音数据集
│   │   ├── data/
│   │   │   ├── long_wav/           # 长语音文件和标注
│   │   │   └── short_wav/          # 短语音文件和标注
│   │   └── README.md
│   ├── dailydialog/                 # 英文对话文本数据集
│   │   ├── train/
│   │   ├── test/
│   │   └── validation/
│   └── lccc/                        # 中文对话文本数据集
│       ├── lccc_base_train.jsonl
│       ├── lccc_base_test.jsonl
│       └── lccc_base_valid.jsonl
├── processed/                       # 处理后的实验数据
│   ├── experiments/                 # 将filter_dataset中的CS-Dialogue和tts_dataset中的dailydialog和lccc数据复制到各个实验的数据目录
│   │   ├── core_comparison/        # 实验一：核心性能质量对比（300个样本）
│   │   │   ├── audio/
│   │   │   │   ├── short/          # 短语音音频分组
│   │   │   │   │   ├── sample_001.wav  # 16kHz, 16bit, mono WAV
│   │   │   │   │   ├── sample_002.wav
│   │   │   │   │   └── ...
│   │   │   │   ├── medium/         # 中等语音音频分组
│   │   │   │   └── long/           # 长语音音频分组
│   │   │   ├── transcripts/        # 对应转录文本
│   │   │   │   ├── short/
│   │   │   │   │   ├── sample_001.json
│   │   │   │   │   └── ...
│   │   │   │   ├── medium/
│   │   │   │   └── long/
│   │   │   └── metadata.json      # 实验元数据
│   │   ├── length_analysis/        # 实验二：长度影响分析（复用core_comparison）
│   │   ├── asr_context/           # 实验三：ASR上下文实验
│   │   │   ├── audio/
│   │   │   │   └── long/          # 仅使用长语音测试上下文效果
│   │   │   ├── transcripts/
│   │   │   │   └── long/
│   │   │   └── context_configs.json # (0,0),(1,0),(0,1),(1,1),(2,2)配置
│   │   ├── ablation_study/        # 实验四：消融研究
│   │   │   ├── audio/
│   │   │   │   └── fixed_samples/ # 固定样本集（代表性样本）
│   │   │   └── transcripts/
│   │   └── case_analysis/         # 实验五：案例分析
│   │       ├── audio/
│   │       │   └── typical_cases/ # 典型案例音频（精选案例）
│   │       ├── transcripts/
│   │       └── case_descriptions.json # 案例描述和预期分析
│   ├── filter_dataset/              # 从原始数据集中过滤出来的数据，只选择第一轮的首个说话人说的话作为长度判断依据，每个数据集都包含短，中，长样本100个，并按下述的转录文本格式生成数据
│   │   ├── CS-Dialogue/
│   │   ├── dailydialog/
│   │   └── lccc/
│   └── tts_dataset/                   # 对filter_dataset中生成的dailydialog，lccc数据进行tts补充音频数据
│       ├── dailydialog/
│       └── lccc/
```

### 音频文件要求
- **格式**: WAV (无压缩)
- **采样率**: 16kHz
- **位深度**: 16bit
- **声道**: 单声道(mono)
- **时长分组**: 
  - `short/`: 短语音（时长待TTS后确定）
  - `medium/`: 中等语音（时长待TTS后确定）
  - `long/`: 长语音（时长待TTS后确定）
- **内容**: 单轮对话问题，包含中文、英文、中英混合

### 转录文本格式样例
```json
{
  "audio_file": "sample_001.wav", 
  "duration": 10.5,
  "text": "播放周杰伦的演唱会视频",
  "language": "zh",
  "word_count": 11,
}
```

### 音频时长预估参考数据

基于当前测试数据的统计分析，提供以下TTS音频时长预估参考：

| 样本文件 | 分组 | 音频时长 | 文本内容 | 字符数 | 每字符时长 |
|---------|------|----------|----------|--------|------------|
| sample_001 | short_5to10s | 2.9s | "请帮我播放一首周杰伦的音乐" | 13个 | 0.227s |
| sample_002 | short_5to10s | 4.7s | "Can you help me set an alarm..." | 36个 | 0.131s |
| sample_001 | medium_10to20s | 9.4s | "我正在准备一个关于机器学习的presentation..." | 70个 | 0.134s |
| sample_002 | medium_10to20s | 5.0s | "Please help me write a Python function..." | 64个 | 0.078s |
| sample_001 | long_20plus | 11.7s | "Could you please explain the concept of transformer..." | 159个 | 0.074s |

**统计结果**:
- **总样本数**: 5个
- **总时长**: 33.7秒
- **总字符数**: 342个
- **平均每字符时长**: 0.099秒

**TTS时长预估建议**:
- **中文文本**: 约0.15-0.23秒/字符 (包含语音停顿)
- **英文文本**: 约0.07-0.13秒/字符 (字母计数)
- **混合文本**: 约0.10-0.15秒/字符 (综合平均)


