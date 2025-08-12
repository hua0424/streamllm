# TTS Filter Dataset处理工具使用说明

## 概述

TTS Filter Dataset处理工具用于将已过滤的文本数据集（dailydialog和lccc）通过TTS服务生成对应的音频文件，为StreamLLM延迟优化实验提供必要的音频样本。

## 核心功能

### 1. 自动TTS生成
- 处理 `filter_dataset/` 目录下的 `dailydialog` 和 `lccc` 数据集
- 跳过CS-Dialogue数据集（已有音频文件）
- 根据文本语言自动选择合适的说话人

### 2. 智能分组管理
- 按时长分组生成音频：short、medium、long
- 保持原有的JSON元数据文件
- 自动创建完整的目录结构

### 3. 数据完整性保障
- TTS服务连接测试
- 音频文件质量验证
- 生成详细统计报告
- 自动清理空目录

## 系统要求

### 依赖关系
- Python 3.7+
- 现有的 `tts.py` 模块（TTS客户端）
- TTS服务运行中（默认地址：`http://host.docker.internal:20401`）

### 目录结构要求
```
experiments/datasets/
├── processed/
│   ├── filter_dataset/          # 输入：已过滤的数据
│   │   ├── dailydialog/        # DailyDialog文本数据
│   │   │   ├── sample_short_001.json
│   │   │   ├── sample_medium_001.json
│   │   │   └── sample_long_001.json
│   │   └── lccc/               # LCCC文本数据
│   │       ├── sample_short_001.json
│   │       ├── sample_medium_001.json
│   │       └── sample_long_001.json
│   └── tts_dataset/            # 输出：生成的音频数据（自动创建）
└── tools/
    ├── tts.py                  # TTS客户端依赖
    └── tts_filter_dataset.py   # 本工具
```

## 使用方法

### 命令行参数

```bash
python tts_filter_dataset.py [选项]
```

#### 核心参数
- `--dataset`: 要处理的数据集类型
  - `dailydialog`: 仅处理DailyDialog
  - `lccc`: 仅处理LCCC
  - `all`: 处理所有数据集（默认）

#### 连接参数
- `--url`: TTS服务URL（默认：`http://host.docker.internal:20401`）
- `--test-connection`: 仅测试TTS连接，不处理数据

#### 处理参数
- `--speed`: 语速调节系数（默认：1.0）
  - `>1.0`: 加速语音
  - `<1.0`: 减慢语音
  - 示例：`--speed 1.2`（加速20%）

#### 验证参数
- `--validate-only`: 仅验证已生成的数据，不生成新数据
- `--base-path`: 数据集基础路径（默认：实验目录）

#### 重新生成参数
- `--force-regenerate`: 强制重新生成所有音频文件（删除已有文件）

### 使用示例

#### 1. 基础使用 - 处理所有数据集
```bash
# 处理dailydialog和lccc数据集，生成标准语速音频
python tts_filter_dataset.py
```

#### 2. 处理特定数据集
```bash
# 仅处理英文dailydialog数据集
python tts_filter_dataset.py --dataset dailydialog

# 仅处理中文lccc数据集
python tts_filter_dataset.py --dataset lccc
```

#### 3. 调节语速
```bash
# 生成加速20%的音频（提高实验效率）
python tts_filter_dataset.py --speed 1.2

# 生成减速10%的音频（提高清晰度）
python tts_filter_dataset.py --speed 0.9
```

#### 4. 连接测试
```bash
# 测试TTS服务是否正常运行
python tts_filter_dataset.py --test-connection
```

#### 5. 数据验证
```bash
# 验证已生成的音频数据完整性
python tts_filter_dataset.py --validate-only
```

#### 6. 自定义TTS服务
```bash
# 使用自定义TTS服务地址
python tts_filter_dataset.py --url http://custom-tts:8080
```

#### 7. 强制重新生成
```bash
# 强制重新生成所有音频文件（删除已有文件）
python tts_filter_dataset.py --force-regenerate

# 强制重新生成特定数据集
python tts_filter_dataset.py --dataset dailydialog --force-regenerate

# 强制重新生成并调节语速
python tts_filter_dataset.py --force-regenerate --speed 1.1
```

#### 8. 组合参数使用
```bash
# 处理特定数据集，使用自定义语速和服务地址
python tts_filter_dataset.py --dataset dailydialog --speed 1.1 --url http://localhost:20401
```

## 说话人配置

### 自动语言识别
程序根据JSON文件中的language字段自动选择说话人：

| 语言代码 | 说话人 | 数据集 |
|----------|--------|---------|
| `zh` | 晓伊 | LCCC |
| `en` | 英文女 | DailyDialog |

### 修改说话人配置
如需修改说话人，编辑代码中的映射：
```python
self.language_speaker_map = {
    'zh': '晓伊',  # 修改中文说话人
    'en': '英文女'   # 修改英文说话人
}
```

## 输出结构

### 目录结构
```
processed/tts_dataset/
├── dailydialog/                 # 英文对话数据集
│   ├── short/                   # 短语音（1-3秒）
│   │   ├── sample_short_001.wav
│   │   ├── sample_short_001.json
│   │   ├── sample_short_002.wav
│   │   ├── sample_short_002.json
│   │   └── ...
│   ├── medium/                  # 中等语音（3-10秒）
│   │   ├── sample_medium_001.wav
│   │   ├── sample_medium_001.json
│   │   └── ...
│   └── long/                    # 长语音（10-30秒）
│       ├── sample_long_001.wav
│       ├── sample_long_001.json
│       └── ...
└── lccc/                        # 中文对话数据集
    ├── short/
    │   ├── sample_short_001.wav
    │   ├── sample_short_001.json
    │   └── ...
    ├── medium/
    │   └── ...
    └── long/
        └── ...
```

### 文件对应关系
- **音频文件**: `.wav`格式，22050Hz采样率，单声道
- **元数据文件**: 对应的`.json`文件，包含原始文本和元信息
- **一一对应**: 每个音频文件都有对应的JSON元数据文件

## 处理流程

### 1. 初始化阶段
```
验证输入路径 -> 测试TTS连接 -> 创建输出目录结构
```

### 2. 数据处理阶段
```
读取JSON文件 -> 提取文本和语言信息 -> 选择说话人 -> 
调用TTS服务 -> 生成音频文件 -> 复制JSON元数据 -> 验证生成结果
```

### 3. 完成阶段
```
清理空目录 -> 数据完整性验证 -> 生成统计报告
```

## 强制重新生成功能

### 功能说明
`--force-regenerate` 参数允许用户强制重新生成所有音频文件，即使目标目录中已存在对应的音频文件。

### 使用场景

#### 1. TTS服务更新
- TTS模型或服务升级后需要重新生成音频
- 说话人配置发生变化
- 音频质量参数调整

#### 2. 文本数据更新
- `filter_dataset` 中的文本内容发生了修改
- 需要使用最新的清理后文本重新生成音频

#### 3. 实验参数变更
- 语速参数需要调整
- 需要测试不同语速下的音频效果

#### 4. 数据质量问题
- 发现部分音频文件损坏或质量不佳
- 需要重新生成有问题的音频

### 工作原理

1. **文件检测**：程序启动时检查目标目录中的已有文件
2. **批量删除**：删除所有 `.wav` 和 `.json` 文件
3. **重新生成**：重新处理所有JSON文本文件，生成音频
4. **进度显示**：显示删除和重新生成的文件数量

### 安全提示

⚠️ **注意事项**:
- 使用前请确认不需要保留已有的音频文件
- 重新生成过程可能需要较长时间
- 建议先备份重要的音频文件

### 使用示例

```bash
# 基本用法
python tts_filter_dataset.py --force-regenerate

# 输出示例:
# 清理 dailydialog 数据集的已有文件...
# 已删除 30 个文件
# 清理 lccc 数据集的已有文件...  
# 已删除 30 个文件
# ⚠️  强制重新生成模式：将删除所有已有的音频文件
```

### 与其他参数组合

```bash
# 重新生成特定数据集的音频，使用新的语速
python tts_filter_dataset.py --dataset dailydialog --force-regenerate --speed 1.2

# 使用自定义TTS服务重新生成
python tts_filter_dataset.py --force-regenerate --url http://new-tts-server:20401
```

## 质量控制

### 1. 连接验证
- 启动时自动测试TTS服务连接
- 连接失败时提供明确错误信息

### 2. 数据验证
- 检查音频文件大小（>1KB）
- 验证音频-JSON文件对应关系
- 统计生成成功率

### 3. 断点续传
- 自动跳过已存在的音频文件
- 支持中断后继续处理
- 避免重复工作

### 4. 错误处理
- 单个文件失败不影响整体处理
- 详细的错误日志和进度显示
- 最终生成处理结果统计

## 性能参数

### 处理速度
- **音频生成**: 平均2-5秒/样本（取决于文本长度）
- **网络延迟**: 每个样本间隔0.5秒（避免服务器过载）
- **批量处理**: 支持大量样本的自动化处理

### 资源占用
- **内存**: 轻量级，逐个文件处理
- **存储**: 每个短音频约20-50KB，中等约50-150KB，长音频约150-500KB
- **网络**: 依赖TTS服务稳定性

## 统计报告示例

```
==================================================
TTS数据集生成统计报告
==================================================

DAILYDIALOG 数据集:
  short: 15 音频, 15 JSON
    平均文件大小: 35.2 KB
  medium: 15 音频, 15 JSON  
    平均文件大小: 89.4 KB
  long: 15 音频, 15 JSON
    平均文件大小: 234.7 KB
  总计: 45 音频, 45 JSON

LCCC 数据集:
  short: 15 音频, 15 JSON
    平均文件大小: 28.9 KB
  medium: 15 音频, 15 JSON
    平均文件大小: 76.3 KB  
  long: 15 音频, 15 JSON
    平均文件大小: 198.5 KB
  总计: 45 音频, 45 JSON
```

## 常见问题

### 1. TTS服务连接失败
```
❌ TTS服务连接失败，请检查服务状态
```
**解决方案**:
- 检查TTS服务是否启动
- 验证服务URL是否正确
- 确认网络连接正常

### 2. 输入数据不存在
```
错误：filter_dataset路径不存在: /path/to/filter_dataset
```
**解决方案**:
- 先运行 `filter_dataset.py` 生成输入数据
- 检查 `--base-path` 参数是否正确

### 3. 生成的音频文件异常
```
⚠️ 发现以下问题:
- 音频文件过小: /path/to/audio.wav
```
**解决方案**:
- 检查TTS服务运行状态
- 验证输入文本是否有效
- 重新生成有问题的样本

### 4. 磁盘空间不足
**解决方案**:
- 预估空间需求：约50-100MB（100个样本）
- 清理不需要的临时文件
- 分批次处理大量数据

### 5. 处理速度慢
**解决方案**:
- 检查网络延迟
- 减少 `speed_factor` 参数复杂度
- 考虑使用本地TTS服务

## 高级配置

### 1. 修改处理延迟
```python
# 在代码中调整延迟时间（第188行）
time.sleep(0.3)  # 改为0.3秒间隔
```

### 2. 自定义分组逻辑
```python
# 修改分组判断逻辑（第137-141行）
if 'short' in length_group or '1to3s' in length_group:
    group_dir = 'short'
elif 'medium' in length_group or '3to10s' in length_group:
    group_dir = 'medium'
elif 'long' in length_group or '10to30s' in length_group:
    group_dir = 'long'
```

### 3. 批量处理优化
如需处理大量数据，可以考虑：
- 分批次运行，每批处理一个数据集
- 使用 `--validate-only` 检查处理进度
- 定期备份已生成的数据

## 实验集成

### 与实验系统对接
生成的TTS数据可直接用于：
- **实验一**: 核心性能与质量对比实验
- **实验二**: 输入长度对优化效果的影响分析
- **实验三**: 前置与后置音频段对ASR准确性的影响
- **实验四**: 消融研究
- **实验五**: 案例分析

### 数据质量保证
- 音频质量统一：22050Hz，单声道，WAV格式
- 元数据完整：保留原始JSON中的所有实验相关信息
- 分组准确：按原始时长估算进行正确分组

## 维护说明

### 版本信息
- **当前版本**: 基于TTS inference_sft接口
- **依赖版本**: 需要tts.py v2.0+
- **更新日期**: 2025年

### 扩展性
- 支持新增数据集类型
- 可配置多种说话人
- 可调整音频生成参数

### 技术支持
如遇问题，请检查：
1. TTS服务运行状态和版本兼容性
2. 输入数据格式和路径配置
3. 网络连接和磁盘空间
4. Python依赖库版本