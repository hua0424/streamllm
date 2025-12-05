# 实验数据处理管线使用说明

## 概述

本管线用于将 MultiWOZ (英文) 和 CrossWOZ (中文) 对话数据集处理成实验所需的累积对话格式，并生成对应的音频文件。

### 数据筛选策略

为了获取足够长的语音输入数据，我们采用以下策略：

1. **计算每个对话的总文本长度**（所有轮次累积）
2. **按总文本长度降序排序**，取前 N 条对话（默认100条）
3. **对选中的对话进行累积处理**，生成 turn1, turn2, turn3...
4. **过滤超长文本**：累积文本超过指定长度的样本将被跳过（中英文分开设置）

### 文本长度限制

由于过长的音频数据对实验意义不大，且中英文语速不同，因此支持分别设置文本长度上限：

| 语言 | 参数 | 默认值 | 说明 |
|------|------|--------|------|
| 中文 (CrossWOZ) | `--max-text-length-zh` | 720 | 约对应 150 秒音频 |
| 英文 (MultiWOZ) | `--max-text-length-en` | 2050 | 约对应 150 秒音频 |

**估算依据**：
- 中文语速约 4-5 字/秒（720 字 ÷ 4.8 字/秒 ≈ 150 秒）
- 英文语速约 13-14 字符/秒（2050 字符 ÷ 13.7 字符/秒 ≈ 150 秒）

### 累积对话逻辑

| 轮次 | 发言者 | 原始文本 | 累积输出 | 输出时机 |
|------|--------|----------|----------|----------|
| 1 | 用户 (a1) | "你好，推荐一个景点" | "你好，推荐一个景点" | ✓ 生成样本 |
| 2 | 系统 (b1) | "推荐颐和园" | - | 累积但不输出 |
| 3 | 用户 (a2) | "门票多少钱" | "你好，推荐一个景点 推荐颐和园 门票多少钱" | ✓ 生成样本 |
| 4 | 系统 (b2) | "30元" | - | 累积但不输出 |
| 5 | 用户 (a3) | "怎么去" | "你好，推荐一个景点 推荐颐和园 门票多少钱 30元 怎么去" | ✓ 生成样本 |

这样随着对话轮次增加，输入文本长度递增，便于分析不同长度输入对系统延迟的影响。

## 管线阶段

### 阶段1：数据预处理 (Data Preprocessing)

选取每个数据集中文本最长的对话，转换为累积对话格式的任务 JSON 文件。超过文本长度限制的样本将被跳过。

**输入**：原始数据集 JSON 文件
**输出**：单独的任务 JSON 文件，每个文件对应一个样本

### 阶段2：TTS 音频生成 (TTS Batch Processing)

调用 TTS 服务，将阶段1生成的文本转换为音频文件。

**输入**：阶段1生成的任务 JSON 文件
**输出**：WAV 格式音频文件

### 阶段3：更新音频时长 (Update Audio Duration)

读取生成的音频文件，获取时长信息，更新到 JSON 文件中。

**输入**：JSON 文件 + 音频文件
**输出**：更新后的 JSON 文件（包含 `audio_duration` 字段）

## 目录结构

```
experiments/datasets/
├── raw_data/                    # [只读] 原始数据集
│   ├── CrossWOZ/
│   │   ├── train.json
│   │   ├── val.json
│   │   └── test.json
│   └── MultiWOZ/
│       └── MultiWOZ_2.1/
│           └── data.json
├── processed/                   # [生成] 处理后的数据
│   ├── json/                    # 阶段1输出：任务JSON文件
│   │   ├── crosswoz/
│   │   │   ├── crosswoz_391_turn1.json
│   │   │   ├── crosswoz_391_turn2.json
│   │   │   └── ...
│   │   └── multiwoz/
│   │       ├── multiwoz_SNG01856_turn1.json
│   │       └── ...
│   └── audio/                   # 阶段2输出：音频文件
│       ├── crosswoz/
│       │   ├── crosswoz_391_turn1.wav
│       │   └── ...
│       └── multiwoz/
│           ├── multiwoz_SNG01856_turn1.wav
│           └── ...
└── tools/                       # 处理工具
    ├── data_processor.py        # 数据预处理模块
    ├── run_pipeline.py          # 管线主程序
    ├── tts.py                   # TTS 客户端
    ├── doc/
    │   ├── PIPELINE_USAGE.md    # 本文档
    │   └── TTS_USAGE.md         # TTS 工具文档
    └── scripts/
        └── run_data_pipeline.sh # 运行脚本
```

## 使用方法

### 前置条件

1. **Python 环境**：项目使用 `conda` + `uv` 管理依赖
   - 先激活 conda 环境：`conda activate streamllm`
   - conda 环境中安装了 `uv`，用于运行程序
2. **TTS 服务**：需要运行 TTS 后端服务（阶段2需要）

### 命令行参数

```bash
uv run python -m experiments.datasets.tools.run_pipeline [参数]
```

#### 数据集选择
- `--dataset {crosswoz,multiwoz,all}`：选择要处理的数据集（默认：all）

#### 输入路径
- `--crosswoz-path`：CrossWOZ 数据文件路径
- `--multiwoz-path`：MultiWOZ 数据文件路径

#### 输出路径
- `--output-dir`：输出基础目录（默认：experiments/datasets/processed）

#### 数量控制
- `--top-n-dialogs N`：选取文本最长的前N个对话（默认：100）
- `--max-samples-per-dialog N`：每个对话最多生成的样本数（默认：不限制）

#### 文本长度限制（过滤过长数据）
- `--max-text-length-zh N`：中文累积文本最大字符数，超过则跳过（默认：720，约150秒音频）
- `--max-text-length-en N`：英文累积文本最大字符数，超过则跳过（默认：2050，约150秒音频）

#### TTS 参数
- `--tts-url`：TTS 服务地址（默认：http://host.docker.internal:20401）
- `--tts-speed`：语速系数（默认：0.8）
- `--tts-workers`：TTS 并发数（默认：4）

#### 阶段控制
- `--skip-preprocess`：跳过阶段1，直接使用已有 JSON 文件进行 TTS
- `--skip-tts`：跳过阶段2（TTS生成）

### 使用示例

#### 1. 完整管线（预处理 + TTS + 更新时长）
```bash
conda activate streamllm
uv run python -m experiments.datasets.tools.run_pipeline \
    --tts-url http://localhost:20401
```

#### 2. 仅数据预处理（不生成音频）
```bash
conda activate streamllm
uv run python -m experiments.datasets.tools.run_pipeline --skip-tts
```

#### 3. 使用自定义文本长度限制
```bash
conda activate streamllm
# 默认已启用限制（中文720字符，英文2050字符）
# 如需调整限制值：
uv run python -m experiments.datasets.tools.run_pipeline \
    --max-text-length-zh 300 \
    --max-text-length-en 800 \
    --skip-tts
```

#### 4. 选取文本最长的50个对话
```bash
conda activate streamllm
uv run python -m experiments.datasets.tools.run_pipeline \
    --top-n-dialogs 50 \
    --skip-tts
```

#### 5. 仅处理 CrossWOZ 数据集
```bash
conda activate streamllm
uv run python -m experiments.datasets.tools.run_pipeline \
    --dataset crosswoz \
    --max-text-length-zh 400 \
    --skip-tts
```

#### 6. 跳过预处理，直接进行 TTS
```bash
conda activate streamllm
uv run python -m experiments.datasets.tools.run_pipeline \
    --skip-preprocess \
    --tts-url http://localhost:20401
```

#### 7. 仅更新音频时长（已完成预处理和TTS）
```bash
conda activate streamllm
uv run python -m experiments.datasets.tools.run_pipeline \
    --skip-preprocess \
    --skip-tts
```

#### 8. 使用脚本运行
```bash
# 赋予执行权限
chmod +x experiments/datasets/tools/scripts/run_data_pipeline.sh

# 测试模式（10个对话）
./experiments/datasets/tools/scripts/run_data_pipeline.sh test

# 仅预处理（使用默认文本长度限制：中文720，英文2050）
./experiments/datasets/tools/scripts/run_data_pipeline.sh preprocess

# 完整管线（使用自定义文本长度限制）
./experiments/datasets/tools/scripts/run_data_pipeline.sh full --max-text-length-zh 300 --max-text-length-en 800
```

## 数据集说明

### CrossWOZ (中文)
- **来源**：清华大学
- **语言**：中文
- **格式**：`{dialog_id: {messages: [{content, role}, ...]}}`
- **角色**：`usr`（用户）、`sys`（系统）

### MultiWOZ (英文)
- **来源**：剑桥大学
- **语言**：英文
- **格式**：`{dialog_id: {log: [{text}, ...]}}`
- **角色**：偶数索引为用户，奇数索引为系统

## 输出文件格式

### 任务 JSON 文件
```json
{
  "sample_id": "crosswoz_391_turn2",
  "dialog_id": "391",
  "turn_index": 2,
  "text": "累积的对话文本...",
  "text_length": 128,
  "audio_file": "crosswoz_391_turn2.wav",
  "audio_duration": 15.234,
  "language": "zh",
  "dataset": "crosswoz"
}
```

### 字段说明
| 字段 | 类型 | 说明 |
|------|------|------|
| sample_id | string | 唯一标识符 |
| dialog_id | string | 原始对话 ID |
| turn_index | int | 用户轮次序号（1, 2, 3...） |
| text | string | 累积的对话文本 |
| text_length | int | 文本字符数 |
| audio_file | string | 对应的音频文件名 |
| audio_duration | float | 音频时长（秒），TTS后填充 |
| language | string | 语言代码（zh/en） |
| dataset | string | 数据集名称 |

## 实验数据分组

生成音频后，可根据 `audio_duration` 字段进行分组：

| 分组 | 时长范围 | 典型轮次 |
|------|----------|----------|
| 短语音 | < 5s | turn1 |
| 中等语音 | 5-15s | turn2-3 |
| 长语音 | > 15s | turn4+ |

## 常见问题

### Q: TTS 服务连接失败？
A: 
1. 检查 TTS 服务是否已启动
2. 确认 URL 地址正确
3. 使用 `--skip-tts` 先跳过 TTS 阶段

### Q: 如何断点续传？
A: TTS 阶段会自动跳过已存在的音频文件，直接重新运行即可。

### Q: 如何清理重新生成？
A: 删除 `experiments/datasets/processed/` 目录后重新运行。

### Q: 数据量太大处理太慢？
A: 使用 `--top-n-dialogs` 减少对话数量，或使用 `--max-text-length-*` 过滤过长文本。

### Q: 如何只更新音频时长？
A: 使用 `--skip-preprocess --skip-tts` 参数，只运行阶段3。

### Q: 如何过滤过长的文本数据？
A: 使用 `--max-text-length-zh` 和 `--max-text-length-en` 参数分别设置中英文的文本长度上限。超过该长度的累积对话样本将被跳过。

## 相关文档

- [TTS 工具使用说明](TTS_USAGE.md)
- [实验设计方案](../../EXPERIMENT_DESIGN.md)
- [项目 README](../../../../README.MD)
