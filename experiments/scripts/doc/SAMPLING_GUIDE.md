# 实验样本抽样指南

本文档说明实验二中如何控制样本抽样，确保各个时长分组的样本均衡。

## 问题背景

在实验二中，样本按音频时长分为 5 个组：

| 分组 | 时长范围 |
|------|---------|
| short | 0-5s |
| medium | 5-15s |
| long | 15-30s |
| very_long | 30-60s |
| extra_long | >60s |

由于数据集中各分组的样本数量不均（例如 extra_long 可能有几百个，而 long 可能只有几十个），如果直接使用 `--max-samples` 限制总样本数，会导致某些组的样本过多而其他组的样本过少。

## 解决方案：按组限制

### 参数说明

1. **`--max-samples N`**: 限制**总样本数**（先加载再过滤）
2. **`--max-samples-per-group N`**: 限制**每个分组的样本数**（确保各组均衡）
3. **`--duration-groups g1 g2 ...`**: 指定要使用的分组

### 推荐用法

#### 场景1：确保各组均衡（推荐）

每个组取 100 个样本：

```bash
./experiments/scripts/run_exp_ablation.sh full \
    --duration-groups long very_long extra_long \
    --max-samples-per-group 100
```

**结果**：
- long: 100 个样本
- very_long: 100 个样本  
- extra_long: 100 个样本
- **总计**: 300 个样本

#### 场景2：快速测试

每个组取 5 个样本：

```bash
./experiments/scripts/run_exp_ablation.sh full \
    --duration-groups long very_long \
    --max-samples-per-group 5
```

**结果**：
- long: 5 个样本
- very_long: 5 个样本
- **总计**: 10 个样本

#### 场景3：限制总数（不推荐，各组可能不均衡）

```bash
./experiments/scripts/run_exp_ablation.sh full \
    --duration-groups long very_long extra_long \
    --max-samples 300
```

**可能的结果**（取决于数据集）：
- long: 30 个样本
- very_long: 50 个样本
- extra_long: 220 个样本
- **总计**: 300 个样本

⚠️ 注意：这种方式各组不均衡，不推荐用于正式实验。

#### 场景4：组合使用

先限制总数（加快加载），再限制每组：

```bash
./experiments/scripts/run_exp_ablation.sh full \
    --duration-groups long very_long extra_long \
    --max-samples 500 \
    --max-samples-per-group 100
```

这种方式可以加快数据加载（不会加载所有样本），同时保证各组均衡。

## 抽样策略

当某个组的样本数超过 `--max-samples-per-group` 时，系统会：

1. **按时长排序**：从短到长排序
2. **选取前 N 个**：取前 `max-samples-per-group` 个样本

这样做的好处：
- **稳定性**：每次运行选择的样本相同（可复现）
- **效率**：短音频处理更快，便于快速验证
- **覆盖性**：从短到长覆盖该组的时长范围

## 完整示例

### 正式实验配置

```bash
./experiments/scripts/run_exp_ablation.sh full \
    --asr-device cuda:0 \
    --llm-device cuda:1 \
    --duration-groups long very_long extra_long \
    --max-samples-per-group 100 \
    --batch-size 10 \
    --suffix-segments 0
```

### 快速验证配置

```bash
./experiments/scripts/run_exp_ablation.sh full \
    --asr-device cuda:0 \
    --llm-device cuda:1 \
    --duration-groups long very_long \
    --max-samples-per-group 5 \
    --batch-size 5
```

## 查看抽样结果

运行实验时，日志会显示抽样统计：

```
2025-12-12 00:24:26.723 - __main__ - INFO - 样本分组统计:
2025-12-12 00:24:26.723 - __main__ - INFO -   extra_long: 296 个样本
2025-12-12 00:24:26.723 - __main__ - INFO -   long: 57 个样本
2025-12-12 00:24:26.723 - __main__ - INFO -   very_long: 95 个样本
2025-12-12 00:24:26.724 - __main__ - INFO -   long: 57 个样本
2025-12-12 00:24:26.724 - __main__ - INFO -   very_long: 95 个样本
2025-12-12 00:24:26.724 - __main__ - INFO -   extra_long: 从 296 个样本中选取 100 个
2025-12-12 00:24:26.724 - __main__ - INFO - 按分组 ['long', 'very_long', 'extra_long'] 过滤后剩余 252 个样本
```

## 注意事项

1. **如果某个组的样本数不足**：会使用该组的所有样本
   - 例如：`--max-samples-per-group 100`，但 long 组只有 57 个样本
   - 结果：使用全部 57 个样本

2. **参数优先级**：
   - 先应用 `--max-samples` 加载样本
   - 再应用 `--duration-groups` 过滤分组
   - 最后应用 `--max-samples-per-group` 限制每组数量

3. **断点续传兼容**：
   - 如果改变抽样参数，建议使用 `--no-resume` 从头开始
   - 否则可能继续处理旧的样本列表

