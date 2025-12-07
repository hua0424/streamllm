# 实验一：延迟与语音长度关系验证

## 实验目的

验证本项目的核心假设：

1. 随着语音输入长度的增加，**非流式方案 (System A)** 的 TTFT 呈线性增长
2. **流式方案 (System B)** 的 TTFT 保持相对稳定
3. 寻找"交叉点"，即流式方案开始优于非流式方案的音频时长阈值

## 关键设计

### 实验公平性保证

1. **模型预热**：使用真实音频进行多轮预热（默认3轮），确保 CUDA kernel 已加载
2. **共享模型实例**：流式和非流式测试使用同一个 ASR 和 LLM 实例
3. **状态重置**：每次测试前重置模型状态，包括 timing 事件和 GPU 缓存
4. **音频共享**：每个样本的两种测试模式使用相同的音频数据

### 效率优化

1. **模型只加载一次**：所有测试共享 ASR 和 LLM 模型
2. **定期内存清理**：每 5 个样本自动清理 GPU 内存
3. **批量处理**：支持大量样本的连续测试

## 实验原理

### System A：非流式基线
```
完整音频录制 → 完整 ASR 转录 → 完整 Prompt 送入 LLM → 生成回复
```
- TTFT = ASR 时间 + LLM Prefill 时间
- 随音频长度线性增长

### System B：流式优化方案
```
VAD 分段 → 流式 ASR (上下文感知) → 流式 LLM (KV Cache 增量预填充) → 生成回复
```
- TTFT ≈ 最后一段音频的 ASR 时间 + 最后一段文本的 LLM Prefill 时间
- 与总音频长度关系较小

## 核心指标

### TTFT (Time to First Token)

- **定义**：从语音输入结束到 LLM 生成第一个 Token 的时间
- **流式模式**：`first_token_time - audio_end_time`
- **非流式模式**：`first_token_time - audio_load_time`

### Latency Improvement

- **定义**：非流式 TTFT 与流式 TTFT 的差值
- **计算**：`(non_streaming_ttft - streaming_ttft) / non_streaming_ttft × 100%`

## 数据分组

| 分组 | 时长范围 | 典型轮次 |
|------|----------|----------|
| short | < 5s | turn1 |
| medium | 5-15s | turn2-3 |
| long | 15-30s | turn4-6 |
| very_long | 30-60s | turn7+ |
| extra_long | > 60s | turn10+ |

## 使用方法

### 前置条件

1. 完成数据处理管线，生成音频文件
2. 安装项目依赖

### 运行方式

```bash
# 运行完整实验（使用 uv，无需激活 conda）
uv run python -m experiments.scripts.run_exp_latency

# 仅测试 CrossWOZ 数据集
uv run python -m experiments.scripts.run_exp_latency --dataset crosswoz

# 限制样本数进行快速测试
uv run python -m experiments.scripts.run_exp_latency --max-samples 5

# 指定设备和预热轮数
uv run python -m experiments.scripts.run_exp_latency --asr-device cuda --llm-device cuda --warmup-rounds 5

# 调整日志级别查看详细信息
uv run python -m experiments.scripts.run_exp_latency --log-level DEBUG
```

### 命令行参数

#### 数据参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data-dir` | `experiments/datasets/processed` | 处理后的数据目录 |
| `--dataset` | `all` | 数据集选择 (crosswoz/multiwoz/all) |
| `--max-samples` | None | 最大样本数（用于测试） |

#### 设备参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--asr-device` | `auto` | ASR 设备 (auto/cuda/cpu) |
| `--llm-device` | `auto` | LLM 设备 (auto/cuda/cpu) |

#### 模型参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--asr-model-size` | 配置文件 | ASR 模型大小 (tiny/base/small/medium/large) |
| `--llm-model-name` | 配置文件 | LLM 模型名称 |

#### 实验参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--chunk-duration` | 500 | 流式音频块时长 (ms) |
| `--max-tokens` | 50 | LLM 最大生成 token 数 |
| `--warmup-rounds` | 3 | 模型预热轮数 |

#### 输出参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--output-dir` | `experiments/results/exp1_latency` | 结果输出目录 |
| `--log-level` | `INFO` | 日志级别 |

## 输出文件

### 1. 详细结果 (JSON)

`exp1_results_YYYYMMDD_HHMMSS.json`

```json
{
  "config": {
    "asr_model": "tiny",
    "llm_model": "Qwen/Qwen2.5-0.5B-Instruct",
    ...
  },
  "results": [
    {
      "sample_id": "crosswoz_10040_turn5",
      "audio_duration": 18.52,
      "duration_group": "long",
      "mode": "streaming",
      "ttft": 156.23,
      "asr_time": 89.45,
      "llm_prefill_time": 66.78,
      ...
    },
    ...
  ],
  "statistics": [...]
}
```

### 2. CSV 汇总

`exp1_summary_YYYYMMDD_HHMMSS.csv`

| sample_id | audio_duration | duration_group | mode | ttft_ms | asr_time_ms | llm_prefill_time_ms |
|-----------|----------------|----------------|------|---------|-------------|---------------------|
| crosswoz_10040_turn1 | 3.25 | short | streaming | 45.23 | 28.12 | 17.11 |
| crosswoz_10040_turn1 | 3.25 | short | non-streaming | 189.56 | 156.34 | 33.22 |

### 3. 统计结果

`exp1_statistics_YYYYMMDD_HHMMSS.csv`

| group | sample_count | avg_duration_s | streaming_ttft_mean_ms | non_streaming_ttft_mean_ms | improvement_ms | improvement_ratio_% |
|-------|--------------|----------------|------------------------|----------------------------|----------------|---------------------|
| short | 10 | 3.45 | 52.34 | 198.67 | 146.33 | 73.6 |
| medium | 15 | 9.23 | 78.45 | 456.78 | 378.33 | 82.8 |
| long | 8 | 22.56 | 112.34 | 1023.45 | 911.11 | 89.0 |

## 结果分析

### 预期结果

1. **非流式 TTFT**：随音频长度线性增长
   - 短语音 (< 5s): ~200ms
   - 中等语音 (5-15s): ~500ms
   - 长语音 (15-30s): ~1000ms

2. **流式 TTFT**：保持相对稳定
   - 所有分组: ~50-150ms

3. **优化效果**：
   - 短语音：优化率较低 (~30-50%)
   - 长语音：优化率显著 (~80-90%)

### 可视化建议

使用实验结果绘制以下图表：

1. **折线图**：X轴为音频时长，Y轴为 TTFT
   - 绘制两条曲线：流式 vs 非流式
   - 标注交叉点

2. **柱状图**：按分组对比流式和非流式 TTFT

3. **散点图**：所有样本的 audio_duration vs TTFT

## 论文数据支持

本实验为论文提供以下数据支持：

1. **表格数据**：不同时长分组的 TTFT 统计
2. **图表数据**：延迟随语音长度变化的趋势
3. **优化效果量化**：各分组的延迟优化率

## 常见问题

### Q: 实验运行太慢？
A: 
- 使用 `--max-samples` 限制样本数进行测试
- 使用较小的 ASR 模型 (`--asr-model-size tiny`)
- 确保使用 GPU (`--asr-device cuda --llm-device cuda`)

### Q: 内存不足？
A: 
- 减少 `--max-tokens` 参数
- 使用较小的模型
- 分批运行不同数据集

### Q: 结果波动较大？
A: 
- 正常现象，增加样本数可减少波动
- 确保模型已预热
- 避免在系统负载高时运行实验

## 相关文档

- [实验设计方案](../../EXPERIMENT_DESIGN.md)
- [数据处理管线](../../datasets/tools/doc/PIPELINE_USAGE.md)
- [项目 README](../../../README.MD)

