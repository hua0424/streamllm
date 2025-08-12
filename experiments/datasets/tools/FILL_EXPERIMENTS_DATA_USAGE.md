# 实验数据填充工具使用说明

**工具名称**: `fill_experiments_data.py`  
**功能**: 根据实验设计要求，将已筛选的数据集复制到experiments目录，并根据实际音频时长重新分组

## 📋 功能概述

### 核心功能
1. **CS-Dialogue数据处理**: 从filter_dataset目录复制CS-Dialogue数据到各实验目录
2. **TTS数据处理**: 从tts_dataset目录复制dailydialog和lccc数据，根据实际音频时长重新分组
3. **长度标准统一**: 按照统一标准（短语音1-3s，中等语音3-10s，长语音10s+）重新分组
4. **智能分配**: 根据各实验的需求自动分配合适数量的样本
5. **详细报告**: 生成完整的数据填充报告

### 处理流程
```
源数据目录                     目标实验目录
├── filter_dataset/            ├── experiments/
│   └── CS-Dialogue/          │   ├── core_comparison/
├── tts_dataset/              │   ├── length_analysis/
│   ├── dailydialog/          │   ├── asr_context/
│   └── lccc/                 │   ├── ablation_study/
                              │   └── case_analysis/
```

## 🎯 实验配置

### 支持的实验类型
| 实验名称 | 描述 | 每组样本数 | 长度分组 | 数据源 |
|---------|------|-----------|----------|--------|
| `core_comparison` | 核心性能与质量对比 | 100 | short/medium/long | CS-Dialogue, DailyDialog, LCCC |
| `length_analysis` | 长度影响分析 | 50 | short/medium/long | CS-Dialogue, DailyDialog, LCCC |
| `asr_context` | ASR上下文准确性 | 30 | long | CS-Dialogue |
| `ablation_study` | 消融研究 | 20 | medium/long | CS-Dialogue, DailyDialog |
| `case_analysis` | 典型案例分析 | 5 | long | CS-Dialogue, DailyDialog, LCCC |

### 长度分组标准
- **短语音 (short)**: 1-3秒
- **中等语音 (medium)**: 3-10秒  
- **长语音 (long)**: 10秒以上

## 🚀 使用方法

### 基本用法

#### 1. 处理所有实验
```bash
# 处理所有实验的数据填充
python fill_experiments_data.py
```

#### 2. 处理特定实验
```bash
# 仅处理核心对比实验
python fill_experiments_data.py --experiment core_comparison

# 处理长度分析实验
python fill_experiments_data.py --experiment length_analysis
```

#### 3. 自定义基础路径
```bash
# 使用自定义数据路径
python fill_experiments_data.py --base-path /path/to/your/datasets
```

#### 4. 预演模式（仅分析不复制）
```bash
# 仅分析数据，不进行实际复制
python fill_experiments_data.py --dry-run
```

### 高级用法

#### 组合参数使用
```bash
# 预演特定实验
python fill_experiments_data.py --experiment core_comparison --dry-run

# 使用自定义路径处理特定实验
python fill_experiments_data.py --experiment asr_context --base-path /custom/path
```

## 📊 输出报告

### 报告内容
工具会生成详细的数据填充报告，包含：

1. **统计摘要**
   - 总处理样本数
   - 重新分组样本数
   - 成功复制样本数
   - 错误数量

2. **长度分组标准说明**
   - 各分组的时长范围
   - 重新分组的统计信息

3. **实验数据统计**
   - 每个实验的样本分布
   - 各长度组的样本数量
   - 是否满足实验需求

4. **目录结构验证**
   - 验证所有实验目录是否正确创建
   - 检查音频和JSON文件的配对情况

5. **错误详情**
   - 处理过程中遇到的错误
   - 失败样本的详细信息

### 报告示例
```
================================================================================
StreamLLM 实验数据填充报告
================================================================================
生成时间: 2025-08-12 15:30:00
基础路径: /usr/local/app/jupyterlab/yanjiu/streamllm/experiments/datasets/processed

统计摘要:
  总处理样本数: 150
  重新分组样本数: 25
  成功复制样本数: 145
  错误数量: 2

长度分组标准:
  短语音 (short): 1.0-3.0秒
  中等语音 (medium): 3.0-10.0秒
  长语音 (long): 10.0秒以上

实验数据统计:
  core_comparison - 核心性能与质量对比实验:
    总样本数: 300
    short: 100 个样本 (需要100个) ✅
    medium: 100 个样本 (需要100个) ✅
    long: 100 个样本 (需要100个) ✅

  length_analysis - 输入长度对优化效果的影响分析:
    总样本数: 150
    short: 50 个样本 (需要50个) ✅
    medium: 50 个样本 (需要50个) ✅
    long: 50 个样本 (需要50个) ✅

目录结构验证:
  ✅ core_comparison
    ✅ short: 100 个配对文件
    ✅ medium: 100 个配对文件
    ✅ long: 100 个配对文件
  ✅ length_analysis
    ✅ short: 50 个配对文件
    ✅ medium: 50 个配对文件
    ✅ long: 50 个配对文件
```

## 🔧 技术细节

### 音频时长检测
- 使用 `librosa` 库精确检测音频实际时长
- 自动处理各种音频格式
- 支持错误恢复和日志记录

### 重新分组逻辑
```python
def determine_length_group(self, duration: float) -> str:
    """根据实际时长确定长度分组"""
    if 1.0 <= duration < 3.0:
        return 'short'
    elif 3.0 <= duration < 10.0:
        return 'medium'
    elif duration >= 10.0:
        return 'long'
    else:
        return 'short'  # 默认
```

### 数据完整性保证
- JSON和音频文件一一对应
- 自动更新元数据中的分组信息
- 保留原始数据信息用于追溯

### 错误处理
- 详细的错误日志记录
- 单个文件失败不影响整体处理
- 提供错误恢复建议

## 📁 输出目录结构

```
experiments/
├── core_comparison/
│   ├── audio/
│   │   ├── short/
│   │   │   ├── sample_001.wav
│   │   │   ├── sample_002.wav
│   │   │   └── ...
│   │   ├── medium/
│   │   └── long/
│   └── transcripts/
│       ├── short/
│       │   ├── sample_001.json
│       │   ├── sample_002.json
│       │   └── ...
│       ├── medium/
│       └── long/
├── length_analysis/
├── asr_context/
├── ablation_study/
└── case_analysis/
```

## ⚠️ 注意事项

### 使用前准备
1. 确保已运行 `filter_dataset.py` 完成CS-Dialogue数据筛选
2. 确保已运行 `tts_filter_dataset.py` 完成TTS音频生成
3. 检查 `librosa` 库是否已安装

### 数据质量检查
- TTS生成的音频可能与预估时长不符，工具会自动重新分组
- 建议先运行 `--dry-run` 模式检查数据情况
- 注意查看生成的报告中的错误和警告信息

### 存储空间要求
- 每个实验大约需要 200-500MB 存储空间
- 音频文件会被复制（非移动），确保有足够磁盘空间

## 🐛 问题排查

### 常见问题

#### 1. 找不到源数据
```
警告：CS-Dialogue路径不存在
```
**解决方案**: 检查是否已运行 `filter_dataset.py` 和 `tts_filter_dataset.py`

#### 2. 音频时长检测失败
```
获取音频时长失败: [Error details]
```
**解决方案**: 检查音频文件是否损坏，确保 `librosa` 正确安装

#### 3. 权限错误
```
复制样本失败: Permission denied
```
**解决方案**: 检查目标目录的写权限

#### 4. 样本数量不足
```
⚠️ short: 10 个样本 (需要100个)
```
**解决方案**: 增加源数据集的样本数量，或调整实验配置

### 调试模式
```bash
# 使用预演模式进行调试
python fill_experiments_data.py --dry-run

# 查看详细输出
python fill_experiments_data.py 2>&1 | tee fill_data.log
```

## 🎓 最佳实践

1. **分步执行**: 先运行预演模式检查数据
2. **逐个实验**: 对重要实验单独执行以便调试
3. **备份数据**: 在处理前备份重要的源数据
4. **验证结果**: 仔细检查生成的报告
5. **测试采样**: 用少量数据测试整个流程

---

**版本信息**: v1.0  
**最后更新**: 2025-08-12  
**维护者**: StreamLLM实验团队

**重要提醒**: 本工具会根据实际音频时长重新分组数据，确保实验的准确性。处理完成后请仔细检查生成的报告。