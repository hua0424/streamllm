# 实验三样本抽样使用指南

## 概述

实验三（准确率与质量验证）现在支持按时长分组均衡抽样，与实验二保持一致。

## 参数说明

### 1. `--max-samples N`
限制**总样本数**（先加载再过滤）

### 2. `--max-samples-per-group N`
限制**每个分组的样本数**（确保各组均衡）

### 3. `--duration-groups g1 g2 ...`
指定要使用的分组，可选值：
- `short` (0-5s)
- `medium` (5-15s)
- `long` (15-30s)
- `very_long` (30-60s)
- `extra_long` (>60s)

**默认**: `medium long very_long`

## 使用示例

### 推荐方式：每组取 100 个样本

```bash
./experiments/scripts/run_exp_quality.sh full \
    --asr-device cuda \
    --duration-groups long very_long extra_long \
    --max-samples-per-group 100
```

**结果**：
- long: 100 个样本
- very_long: 100 个样本
- extra_long: 100 个样本
- **总计**: 300 个样本

### 快速测试：每组取 5 个样本

```bash
./experiments/scripts/run_exp_quality.sh test \
    --duration-groups medium long \
    --max-samples-per-group 5
```

### Python 命令行方式

```bash
uv run python -m experiments.scripts.run_exp_quality \
    --dataset all \
    --duration-groups long very_long extra_long \
    --max-samples-per-group 100 \
    --asr-device cuda \
    --batch-size 10
```

## 与实验二的一致性

实验二和实验三现在使用相同的参数名称和行为：

| 参数 | 实验二 | 实验三 | 说明 |
|------|--------|--------|------|
| `--max-samples-per-group` | ✅ | ✅ | 每组样本上限 |
| `--duration-groups` | ✅ | ✅ | 指定分组 |
| `--max-samples` | ✅ | ✅ | 总样本上限 |

## 日志输出示例

```bash
2025-12-14 10:00:00.000 - __main__ - INFO - 加载了 500 个有效样本
2025-12-14 10:00:00.001 - __main__ - INFO -   long: 从 150 个样本中选取 100 个
2025-12-14 10:00:00.001 - __main__ - INFO -   very_long: 95 个样本
2025-12-14 10:00:00.001 - __main__ - INFO -   extra_long: 从 255 个样本中选取 100 个
2025-12-14 10:00:00.001 - __main__ - INFO - 分层抽样分组 ['long', 'very_long', 'extra_long'], 每组取 100, 最终 295 条
```

## 兼容性说明

旧参数 `--samples-per-group` 仍然可用（映射到 `--max-samples-per-group`），但已标记为弃用：

```bash
# 旧方式（仍可用，但不推荐）
uv run python -m experiments.scripts.run_exp_quality \
    --samples-per-group 100

# 新方式（推荐）
uv run python -m experiments.scripts.run_exp_quality \
    --max-samples-per-group 100
```

## 完整配置示例

### 实验三正式运行配置

```bash
./experiments/scripts/run_exp_quality.sh full \
    --asr-device cuda \
    --asr-model-size base \
    --duration-groups long very_long extra_long \
    --max-samples-per-group 150 \
    --prefix-segments 1 \
    --suffix-segments 1 \
    --batch-size 20
```

### 快速验证配置

```bash
./experiments/scripts/run_exp_quality.sh full \
    --asr-device cuda \
    --duration-groups medium long \
    --max-samples-per-group 10 \
    --batch-size 5
```

## 注意事项

1. **如果某组样本不足**：会使用该组的所有样本
   - 例如：`--max-samples-per-group 100`，但 medium 组只有 48 个样本
   - 结果：使用全部 48 个样本

2. **抽样策略**：按时长从短到长排序，取前 N 个（确保可复现）

3. **断点续传**：改变抽样参数时建议使用 `--no-resume` 从头开始

4. **参数优先级**：
   - 先应用 `--max-samples` 加载样本
   - 再应用 `--duration-groups` 过滤分组
   - 最后应用 `--max-samples-per-group` 限制每组数量

