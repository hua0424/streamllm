# 数据过滤处理程序使用说明

## 概述

数据过滤处理程序用于从原始数据集中筛选并过滤符合实验要求的样本，支持CS-Dialogue、DailyDialog和LCCC三种数据集的处理。程序会根据文本时长、语言类型等条件进行智能过滤，生成标准化的实验数据。

## 核心功能

### 1. 多数据集支持
- **CS-Dialogue**: 中英混合客服对话数据集（带音频文件）
- **DailyDialog**: 英文日常对话数据集
- **LCCC**: 中文大规模对话数据集

### 2. 智能过滤
- 时长分组：短(1-3s)、中(3-10s)、长(10-30s)
- 语言过滤：只保留纯中文或纯英文，过滤中英混合
- 质量过滤：移除包含特殊字符(*、<FIL/>)的样本
- 音频验证：CS-Dialogue数据集验证音频文件存在性
- **TTS文本清理**：自动处理标点符号附近的空格，确保TTS生成准确音频

### 3. 双重时长验证
- **实际音频时长**: 使用librosa读取真实音频时长（CS-Dialogue）
- **文本估算时长**: 基于字符数和语言特征估算时长
- **下限验证**: 文本估算时长需满足分组下限要求

## 系统要求

### Python依赖
```bash
pip install librosa json pathlib argparse shutil
```

### 目录结构要求
```
experiments/datasets/
├── raw_data/                    # 原始数据目录
│   ├── CS-Dialogue/            # CS-Dialogue数据集
│   │   ├── data/
│   │   │   ├── index/short_wav/train/
│   │   │   │   ├── text        # 文本索引文件
│   │   │   │   └── wav.scp     # 音频路径映射
│   │   │   └── short_wav/      # 音频文件目录
│   │   ├── dailydialog/        # DailyDialog数据集  
│   │   │   └── train/
│   │   │       └── dialogues_train.txt
│   │   └── lccc/               # LCCC数据集
│   │       └── lccc_base_train.jsonl
└── processed/
    └── filter_dataset/         # 输出目录（自动创建）
```

## 使用方法

### 命令行参数

```bash
python filter_dataset.py [参数选项]
```

#### 参数说明
- `--dataset`: 要处理的数据集类型
  - `cs-dialogue`: 仅处理CS-Dialogue
  - `dailydialog`: 仅处理DailyDialog  
  - `lccc`: 仅处理LCCC
  - `all`: 处理所有数据集（默认）
- `--test`: 测试模式，每组只生成5个样本（默认每组100个）

### 使用示例

#### 1. 处理所有数据集
```bash
python filter_dataset.py
```

#### 2. 仅处理CS-Dialogue数据集
```bash
python filter_dataset.py --dataset cs-dialogue
```

#### 3. 测试模式（快速验证）
```bash
python filter_dataset.py --test
```

#### 4. 处理特定数据集（测试模式）
```bash
python filter_dataset.py --dataset lccc --test
```

## 过滤规则详解

### 1. CS-Dialogue数据集
#### 数据来源
- 只从 `short_wav` 目录抽样
- 要求有对应的音频文件
- 只选择首轮对话（utterance_id以_2结尾）

#### 过滤条件
```python
# 语言过滤
if language == 'mix': continue     # 过滤中英混合

# 特殊字符过滤  
if '*' in text: continue           # 过滤含*号样本
if '<FIL/>' in text: continue      # 过滤含<FIL/>标记样本

# 音频文件验证
if not audio_path.exists(): continue  # 必须有音频文件

# 时长验证
actual_duration = librosa_load(audio_path)    # 实际音频时长
estimated_duration = estimate_text_duration(text)  # 文本估算时长

# 双重验证：实际时长在分组范围内 + 文本时长满足下限
actual_group = get_length_group(actual_duration)
min_duration = length_groups[actual_group][0]
if estimated_duration < min_duration: continue

# TTS文本清理
cleaned_text = clean_text_for_tts(text)  # 处理标点符号附近的空格
```

### 2. DailyDialog数据集
#### 数据来源
- 从 `train/dialogues_train.txt` 读取
- 提取每个对话的第一轮发言

#### 过滤条件
```python
# 长度过滤
if len(utterance) <= 5: continue   # 过滤过短句子

# 语言设定
language = 'en'  # 强制英文标记

# TTS文本清理
cleaned_text = clean_text_for_tts(utterance)  # 处理标点符号附近的空格
```

### 3. LCCC数据集
#### 数据来源  
- 从 `lccc_base_train.jsonl` 读取前50000行
- 提取每个对话的第一轮发言

#### 过滤条件
```python
# 预处理
text = text.replace(' ', '')       # 移除分词空格

# 长度过滤
if len(text) < 2: continue         # 过滤过短文本

# 语言设定
language = 'zh'  # 强制中文标记

# TTS文本清理
cleaned_text = clean_text_for_tts(text)  # 处理标点符号附近的空格
```

## TTS文本清理功能

为确保TTS服务能够准确生成音频，程序会自动清理文本中标点符号附近的多余空格。这一功能可以解决原始文本中格式不标准导致的TTS发音错误问题。

### 清理规则

#### 1. 标点符号前后空格处理
```python
# 处理前：
"Shall we go to cinema this evening ? The new blockbuster starring mel gibson is showing . It ' s supposed to be really good ."

# 处理后：
"Shall we go to cinema this evening?The new blockbuster starring mel gibson is showing.It's supposed to be really good."
```

#### 2. 具体清理操作
- **移除标点符号前的空格**：`word ?` → `word?`
- **规范化标点符号后的空格**：`word ?  ` → `word? `
- **修复英文缩写**：`It ' s` → `It's`、`doesn ' t` → `doesn't`
- **清理引号内空格**：`" hello "` → `"hello"`
- **移除多余连续空格**：`word   word` → `word word`

#### 3. 处理的标点符号
支持的标点符号包括：`? ! . , ; : ' " "`

### 清理效果对比

| 清理前（问题文本） | 清理后（TTS友好） | TTS效果改善 |
|-------------------|------------------|------------|
| `What ' s your name ?` | `What's your name?` | 避免读出撇号和问号 |
| `I don ' t know .` | `I don't know.` | 避免分词错误 |
| `" Hello world " !` | `"Hello world"!` | 规范引号处理 |

### 适用场景

#### DailyDialog数据集
- 原始数据经常有标点符号前后的不规范空格
- 特别是问号、撇号等符号容易被TTS误读
- 清理后大幅提高英文TTS准确性

#### LCCC数据集  
- 中文文本中偶有英文标点符号格式问题
- 清理后确保中英文混合标点的正确处理

#### CS-Dialogue数据集
- 客服对话中标点使用不规范
- 清理后提高TTS生成音频的自然度

### 技术实现

文本清理通过正则表达式实现，处理流程为：
1. 移除标点符号前的空格
2. 规范化标点符号后的空格
3. 修复常见英文缩写格式
4. 清理引号内多余空格
5. 合并多余的连续空格
6. 去除首尾空格

该功能在保存样本数据前自动执行，确保所有输出文本都适合TTS处理。

## 输出格式

### 目录结构
```
processed/filter_dataset/
├── CS-Dialogue/
│   ├── sample_short_001.json
│   ├── sample_medium_001.json  
│   ├── sample_long_001.json
│   └── audio/                   # 音频文件目录
│       ├── short/
│       ├── medium/
│       └── long/
├── dailydialog/
│   ├── sample_short_001.json
│   ├── sample_medium_001.json
│   └── sample_long_001.json
└── lccc/
    ├── sample_short_001.json
    ├── sample_medium_001.json
    └── sample_long_001.json
```

### JSON文件格式
```json
{
  "sample_id": "cs-dialogue_short_001",
  "source_dataset": "cs-dialogue", 
  "text": "您好，请问有什么可以帮到您的吗？",
  "language": "zh",
  "duration": 2.3,
  "audio_file": "sample_short_001.wav",
  "length_group": "short_1to3s", 
  "word_count": 12
}
```

#### 字段说明
- `sample_id`: 唯一样本标识符
- `source_dataset`: 来源数据集名称
- `text`: 文本内容
- `language`: 语言代码（zh/en）
- `duration`: 实际时长（CS-Dialogue）或估算时长
- `audio_file`: 对应音频文件名（仅CS-Dialogue有实际文件）
- `length_group`: 长度分组标识
- `word_count`: 字符/单词数量

## 时长估算算法

### 估算参数
```python
duration_params = {
    'zh': 0.19,    # 中文：约0.19秒/字符
    'en': 0.10,    # 英文：约0.10秒/字符  
    'mix': 0.15    # 混合：约0.15秒/字符
}
```

### 计算公式
```python
def estimate_duration(text, language='zh'):
    char_count = len(text.replace(' ', ''))  # 去空格字符数
    duration_per_char = duration_params[language]
    return char_count * duration_per_char
```

## 长度分组规则

| 分组名称 | 时长范围 | 目标用途 | 默认样本数 |
|----------|----------|----------|------------|
| short | 1-3秒 | 短语音测试 | 100个 |
| medium | 3-10秒 | 中等语音测试 | 100个 |  
| long | 10-30秒 | 长语音测试 | 100个 |

### 分组选择逻辑
1. 计算文本估算时长和音频实际时长
2. 以实际时长确定目标分组（CS-Dialogue）
3. 验证文本估算时长满足分组下限要求
4. 在各分组内按时长排序，选择最适合的样本

## 统计报告

程序运行完成后会生成详细统计报告：

```
=== 数据统计报告 ===

CS-Dialogue:
  总样本数: 300
  音频文件数: 300
  - short: 100 个样本, 时长范围: 1.2s - 2.9s
  - medium: 100 个样本, 时长范围: 3.1s - 9.8s  
  - long: 100 个样本, 时长范围: 10.2s - 28.5s

dailydialog:
  总样本数: 300
  - short: 100 个样本, 时长范围: 1.0s - 3.0s
  - medium: 100 个样本, 时长范围: 3.0s - 9.9s
  - long: 100 个样本, 时长范围: 10.1s - 29.8s

lccc:
  总样本数: 300  
  - short: 100 个样本, 时长范围: 1.1s - 2.8s
  - medium: 100 个样本, 时长范围: 3.2s - 9.7s
  - long: 100 个样本, 时长范围: 10.0s - 29.9s
```

## 性能参数

### 处理能力
- **CS-Dialogue**: 需要读取音频文件，处理较慢
- **DailyDialog**: 纯文本处理，速度较快
- **LCCC**: 大文件读取，中等速度

### 内存占用
- 流式读取JSONL文件，内存占用可控
- 音频文件时长检测使用librosa，内存占用适中

### 输出清理
- 程序启动时自动清空输出目录
- 避免多次运行产生文件混杂

## 常见问题

### 1. 找不到音频文件
```
警告：无法读取音频文件 /path/to/audio.wav: [Error details]
```
**解决方案**: 检查CS-Dialogue数据集的音频文件路径配置

### 2. 某个分组样本不足
```
- long (10-30s): 保存 45 个，可用 45 个
```
**解决方案**: 原始数据中该长度范围样本较少，属于正常现象

### 3. 处理速度慢
**解决方案**: 
- 使用 `--test` 模式快速验证
- CS-Dialogue因需读取音频文件会比较慢

### 4. 中英混合数据过多被过滤
**解决方案**: 这是预期行为，程序只保留纯语言样本以提高ASR准确性

## 高级配置

### 自定义样本数量
修改代码中的 `samples_per_group` 参数：
```python
self.samples_per_group = 150  # 每组150个样本
```

### 自定义时长分组
修改 `length_groups` 参数：
```python  
self.length_groups = {
    'short': (0.5, 2),    # 调整为0.5-2秒
    'medium': (2, 8),     # 调整为2-8秒  
    'long': (8, 25)       # 调整为8-25秒
}
```

### 自定义估算参数
修改 `duration_params`：
```python
self.duration_params = {
    'zh': 0.20,    # 调整中文估算参数
    'en': 0.12,    # 调整英文估算参数
}
```

## 注意事项

1. **数据完整性**: 确保原始数据集完整下载
2. **路径配置**: 检查raw_data目录结构正确
3. **权限问题**: 确保输出目录有写入权限
4. **磁盘空间**: CS-Dialogue会复制音频文件，需要足够空间
5. **依赖版本**: librosa版本需要兼容，建议使用最新版本

## 版本信息

- **支持的数据集版本**: CS-Dialogue v1.0, DailyDialog v1.0, LCCC Base
- **音频格式支持**: WAV（通过librosa）
- **更新日期**: 2025年

## 技术支持

如遇问题，请检查：
1. 原始数据集目录结构是否正确
2. 必要的Python依赖是否安装
3. 磁盘空间是否充足
4. 输出目录权限是否正确