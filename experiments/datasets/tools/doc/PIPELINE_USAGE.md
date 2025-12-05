# 实验数据处理管线使用说明

## 概述

本管线用于将 MultiWOZ (英文) 和 CrossWOZ (中文) 对话数据集处理成实验所需的累积对话格式，并生成对应的音频文件。

### 累积对话逻辑

为了实验不同长度的语音输入对延迟的影响，我们采用**累积对话**的方式生成数据：

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

将原始 JSON 数据集转换为累积对话格式的任务 JSON 文件。

**输入**：原始数据集 JSON 文件
**输出**：单独的任务 JSON 文件，每个文件对应一个样本

生成的 JSON 文件格式：
```json
{
  "sample_id": "crosswoz_391_turn2",
  "dialog_id": "391",
  "turn_index": 2,
  "text": "你好，麻烦帮我推荐一个门票免费的景点。 好的，这里有一个免费景点推荐给您。 附近有什么餐厅吗？",
  "audio_file": "crosswoz_391_turn2.wav",
  "language": "zh",
  "dataset": "crosswoz"
}
```

### 阶段2：TTS 音频生成 (TTS Batch Processing)

调用 TTS 服务，将阶段1生成的文本转换为音频文件。

**输入**：阶段1生成的任务 JSON 文件
**输出**：WAV 格式音频文件

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
- `--max-dialogs N`：每个数据集最多处理 N 个对话
- `--max-samples-per-dialog N`：每个对话最多生成 N 个样本

#### TTS 参数
- `--tts-url`：TTS 服务地址（默认：http://host.docker.internal:20401）
- `--tts-speed`：语速系数（默认：1.0）

#### 阶段控制
- `--skip-preprocess`：跳过阶段1，直接使用已有 JSON 文件进行 TTS
- `--skip-tts`：跳过阶段2，仅进行数据预处理

### 使用示例

#### 1. 完整管线（预处理 + TTS）
```bash
uv run python -m experiments.datasets.tools.run_pipeline \
    --tts-url http://localhost:20401
```

#### 2. 仅数据预处理（不生成音频）
```bash
uv run python -m experiments.datasets.tools.run_pipeline --skip-tts
```

#### 3. 测试运行（少量数据）
```bash
uv run python -m experiments.datasets.tools.run_pipeline \
    --max-dialogs 5 \
    --max-samples-per-dialog 3 \
    --skip-tts
```

#### 4. 仅处理 CrossWOZ 数据集
```bash
uv run python -m experiments.datasets.tools.run_pipeline \
    --dataset crosswoz \
    --skip-tts
```

#### 5. 跳过预处理，直接进行 TTS
```bash
uv run python -m experiments.datasets.tools.run_pipeline \
    --skip-preprocess \
    --tts-url http://localhost:20401
```

#### 6. 使用脚本运行
```bash
# 赋予执行权限
chmod +x experiments/datasets/tools/scripts/run_data_pipeline.sh

# 运行脚本
./experiments/datasets/tools/scripts/run_data_pipeline.sh
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
  "audio_file": "crosswoz_391_turn2.wav",
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
| audio_file | string | 对应的音频文件名 |
| language | string | 语言代码（zh/en） |
| dataset | string | 数据集名称 |

## 实验数据分组

生成音频后，可根据音频时长进行分组：

| 分组 | 时长范围 | 典型轮次 |
|------|----------|----------|
| 短语音 | < 5s | turn1 |
| 中等语音 | 5-15s | turn2-3 |
| 长语音 | > 15s | turn4+ |

分组脚本可参考 `experiments/scripts/` 目录（待开发）。

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
A: 使用 `--max-dialogs` 和 `--max-samples-per-dialog` 限制数量进行测试。

## 相关文档

- [TTS 工具使用说明](TTS_USAGE.md)
- [实验设计方案](../../EXPERIMENT_DESIGN.md)
- [项目 README](../../../../README.MD)

