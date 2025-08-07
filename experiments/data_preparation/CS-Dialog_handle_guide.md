# CS-Dialogue 数据处理指南

## 数据集概述

**CS-Dialogue** 是一个104小时的中英代码切换语音对话数据集，包含200名说话者的100个对话录音。数据集提供了完整的音频文件和对应的转录文本，是目前最大的自然中英代码切换对话数据集。

## 数据集结构分析

### 原始数据结构
```
CS-Dialogue/
├── data/
│   ├── short_wav/                    # 短音频片段版本
│   │   └── short_wav/
│   │       ├── SCRIPT/               # 文本转录文件
│   │       │   ├── ZH-CN_U0001_S0.txt
│   │       │   ├── ZH-CN_U0002_S0.txt
│   │       │   └── ...
│   │       └── WAVE/                 # 音频文件
│   │           └── C0/               # 对话类别C0
│   │               ├── ZH-CN_U0001_S0/  # 单个对话目录
│   │               │   ├── ZH-CN_U0001_S0_2.wav
│   │               │   ├── ZH-CN_U0001_S0_4.wav
│   │               │   └── ...
│   │               └── ...
│   └── long_wav/                     # 长音频版本(完整对话)
│       └── long_wav/
│           ├── SCRIPT/               # TextGrid标注文件
│           └── WAVE/                 # 完整对话音频
└── index/                            # 数据索引文件
    ├── short_wav/
    │   ├── train/
    │   ├── dev/
    │   └── test/
    └── long_wav/
```

### 文件命名规则
- **对话ID**: `ZH-CN_U0001_S0` 格式
  - `ZH-CN`: 语言标识(中文为主)
  - `U0001`: 用户ID
  - `S0`: 会话ID
- **音频片段**: `ZH-CN_U0001_S0_2.wav`
  - 最后的数字对应文本文件中的行号

### 文本标注格式
```
ZH-CN_U0001_S0_2	<CN> 嗨。
ZH-CN_U0001_S0_4	<MIX> 嗯，我是 Luna ，嗯，你现在。
ZH-CN_U0001_S0_6	<CN> 啊你现在在哪？
```
- **语言标记**:
  - `<CN>`: 纯中文
  - `<EN>`: 纯英文  
  - `<MIX>`: 中英混合

## 处理目标与方案

### 实验需求
根据延迟优化实验需求，需要准备不同时长的单轮对话数据：
- **时长分组**: 5-10s, 10-20s, 20-30s, 30s以上
- **每组样本数**: 90个
- **语言类型**: 中英混合为主
- **数据格式**: 音频-文本对

### 数据处理方案

#### 1. 数据筛选策略
```python
# 筛选原则
duration_groups = {
    "5-10s": [],
    "10-20s": [], 
    "20-30s": [],
    "30s+": []
}

# 语言优先级 (符合实验需求)
language_priority = ["<MIX>", "<CN>", "<EN>"]
```

#### 2. 音频-文本对齐
- 根据文本文件中的ID匹配对应的音频文件
- 验证音频时长与文本内容的一致性
- 过滤掉音频缺失或损坏的条目

#### 3. 质量控制
- **音频质量**: 检查采样率(16kHz)、位深度(16bit)
- **文本质量**: 去除标点符号错误、格式异常
- **时长验证**: 确保音频实际时长符合分组要求
- **内容过滤**: 排除过短(<3字符)或无意义内容

### 输出数据格式

#### 目录结构
```
processed/
├── CS-Dialogue/
│   ├── 5-10s/
│   │   ├── audio/
│   │   │   ├── cs_001.wav
│   │   │   ├── cs_002.wav
│   │   │   └── ...
│   │   ├── transcripts/
│   │   │   ├── cs_001.txt
│   │   │   ├── cs_002.txt
│   │   │   └── ...
│   │   └── metadata.json
│   ├── 10-20s/
│   ├── 20-30s/
│   └── 30s+/
└── README.md
```

#### 数据格式规范

**音频文件**:
- 格式: WAV (16kHz, 16bit, mono)
- 命名: `cs_{duration_group}_{id:03d}.wav`
- 无静音段预处理

**文本文件**:
- 编码: UTF-8
- 格式: 纯文本，每个文件一个单轮对话
- 保留语言标记信息

**元数据文件** (`metadata.json`):
```json
{
  "dataset_info": {
    "source": "CS-Dialogue",
    "processed_date": "2025-01-XX",
    "total_samples": 90,
    "duration_range": "5-10s"
  },
  "samples": [
    {
      "id": "cs_001",
      "audio_file": "audio/cs_001.wav",
      "transcript_file": "transcripts/cs_001.txt",
      "original_id": "ZH-CN_U0001_S0_2",
      "language_tag": "<MIX>",
      "duration": 7.2,
      "text": "嗯，我是 Luna ，嗯，你现在。",
      "speaker_id": "ZH-CN_U0001",
      "quality_score": 0.95
    }
  ]
}
```

## 实现步骤

### 第一阶段：数据扫描与分析
1. 遍历所有文本文件，提取语句信息
2. 匹配对应的音频文件
3. 计算音频时长并分组
4. 统计各组数据分布

### 第二阶段：数据筛选
1. 按时长分组筛选候选数据
2. 优先选择`<MIX>`标记的中英混合语句
3. 质量评估与排序
4. 平衡采样确保每组90个样本

### 第三阶段：数据处理与输出
1. 复制并重命名音频文件
2. 清理文本内容，生成转录文件
3. 生成元数据文件
4. 验证数据完整性

### 第四阶段：质量验证
1. 音频-文本对齐验证
2. 时长分布验证  
3. 语言标记准确性检查
4. 生成处理报告

## 处理脚本设计

### 核心功能模块
```python
class CSDialogueProcessor:
    def __init__(self, raw_data_path, output_path):
        self.raw_data_path = raw_data_path
        self.output_path = output_path
        
    def scan_data(self):
        """扫描原始数据，建立音频-文本映射"""
        
    def filter_by_duration(self, target_duration_range):
        """按时长筛选数据"""
        
    def extract_samples(self, duration_group, sample_count=90):
        """提取指定数量的样本"""
        
    def process_audio(self, src_path, dst_path):
        """处理音频文件(格式转换、质量检查)"""
        
    def process_text(self, text, language_tag):
        """处理文本内容(清理、标准化)"""
        
    def generate_metadata(self, samples, duration_group):
        """生成元数据文件"""
        
    def validate_output(self):
        """验证输出数据质量"""
```

## 实验集成

### 与StreamLLM实验框架集成
1. **数据加载接口**: 提供标准化的数据加载API
2. **时长估算**: 集成TTS时长估算功能  
3. **批处理支持**: 支持实验批量数据读取
4. **格式兼容**: 与现有实验代码数据格式兼容

### 使用示例
```python
# 加载处理后的数据
from cs_dialogue_loader import CSDialogueDataset

# 加载5-10秒的数据用于实验
dataset = CSDialogueDataset(
    duration_group="5-10s",
    data_path="processed/CS-Dialogue/5-10s"
)

# 获取单个样本
sample = dataset[0]
audio_path = sample['audio_file']
text = sample['text']
duration = sample['duration']
```

## 注意事项

1. **版权合规**: CS-Dialogue采用CC BY-NC-SA 4.0许可，仅限学术研究使用
2. **数据完整性**: 部分音频文件可能缺失，需要容错处理
3. **语言平衡**: 优先选择`<MIX>`数据，确保中英混合特性
4. **质量控制**: 严格的质量检查流程，确保实验数据可靠性
5. **存储优化**: 大数据集处理需要考虑存储空间和处理时间

---

**处理完成后，将生成360个高质量的中英混合单轮对话样本，按时长分为4组，每组90个，直接用于StreamLLM延迟优化实验。**