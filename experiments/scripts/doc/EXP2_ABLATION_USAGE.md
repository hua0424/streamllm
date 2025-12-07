# 实验二：消融实验使用说明

## 实验目的

量化两项关键技术对 TTFT 的贡献：

1. **流式 ASR**：减少从音频结束到得到完整转录的时间
2. **LLM KV 预填充**：在收到分段文本时增量构建 KV Cache，加速首个 Token 生成

对比三种配置：

- **Baseline**：非流式 ASR + 非流式 LLM
- **Streaming ASR Only**：流式 ASR + 非流式 LLM（等待完整文本，不做 KV 预填充）
- **Full Streaming**：流式 ASR + 流式 LLM（增量 KV 预填充）

默认仅测试 **长语音组 (15-30s)**，可通过参数调整。

## 关键设计与公平性

- **共享模型实例**：三种配置复用同一组 ASR/LLM 实例，避免加载差异
- **模型预热**：使用真实音频多轮预热（默认 3 轮）
- **状态重置**：每次测试前清理 timing 与 GPU 缓存
- **统一音频**：同一音频样本依次跑三种配置，保证可比性

## 核心指标

- **TTFT**：`first_token_time - audio_end_time`（Baseline 从 `audio_load_time` 计算）
- **ASR 时间**：`last_text_time - audio_end_time`
- **LLM 预填充时间**：`first_token_time - last_text_time`
- **增益拆分**：
  - 流式 ASR 贡献：`TTFT_baseline - TTFT_streaming_asr`
  - KV 预填充贡献：`TTFT_streaming_asr - TTFT_full_streaming`

## 运行方式

前置：完成数据处理管线、安装依赖、准备 GPU（建议）。

```bash
# 运行默认长语音消融实验（使用 uv，无需激活 conda）
uv run python -m experiments.scripts.run_exp_ablation

# 指定数据集与样本数
uv run python -m experiments.scripts.run_exp_ablation --dataset crosswoz --max-samples 10

# 自定义分组（例如长语音与超长语音）
uv run python -m experiments.scripts.run_exp_ablation --duration-groups long very_long

# 指定设备与预热轮数
uv run python -m experiments.scripts.run_exp_ablation --asr-device cuda --llm-device cuda --warmup-rounds 5

# 查看详细日志
uv run python -m experiments.scripts.run_exp_ablation --log-level DEBUG
```

### 命令行参数

#### 数据参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data-dir` | `experiments/datasets/processed` | 处理后的数据目录 |
| `--dataset` | `all` | 数据集选择 (crosswoz/multiwoz/all) |
| `--max-samples` | None | 最大样本数（快速验证用） |
| `--duration-groups` | `long` | 按时长分组筛选，默认为长语音组 |

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
| `--output-dir` | `experiments/results/exp2_ablation` | 结果输出目录 |
| `--log-level` | `INFO` | 日志级别 |

## 输出文件（论文可直接引用的格式）

生成于 `experiments/results/exp2_ablation`，文件均含时间戳后缀：

1) **详细结果 (JSON)** `exp2_results_YYYYMMDD_HHMMSS.json`  
   - `config`：运行参数记录（模型、设备、chunk、分组等）  
   - `results`：逐样本逐配置的原始指标，含 `dataset/language/dialog_id/turn_index/text_length/audio_duration/duration_group/ttft/asr_time/llm_prefill_time` 等  
   - `statistics`：按分组聚合的均值/标准差/增益  
   - `sample_gains`：逐样本增益拆分（见下第4条 CSV）  
   - `overall_statistics`：不分组的总体均值/标准差/总优化率

2) **逐样本汇总 CSV** `exp2_summary_YYYYMMDD_HHMMSS.csv`  
   列：`sample_id,dataset,language,dialog_id,turn_index,text_length,audio_duration,duration_group,mode,ttft_ms,asr_time_ms,llm_prefill_time_ms,error`

3) **分组统计 CSV** `exp2_statistics_YYYYMMDD_HHMMSS.csv`  
   列：`group,sample_count,avg_duration_s,baseline_ttft_mean_ms,streaming_asr_ttft_mean_ms,full_streaming_ttft_mean_ms,asr_gain_ms,kv_gain_ms,total_gain_ms,total_gain_ratio_%`

4) **逐样本增益拆分 CSV** `exp2_gains_YYYYMMDD_HHMMSS.csv`  
   列：`sample_id,dataset,language,dialog_id,turn_index,text_length,duration_group,audio_duration_s,baseline_ttft_ms,streaming_asr_ttft_ms,full_streaming_ttft_ms,asr_gain_ms,kv_gain_ms,total_gain_ms,total_gain_ratio_%`

### 数据样例（虚拟示例，便于论文表格对接）

- `exp2_summary_*.csv`（同一 `sample_id` 会出现三行，分别对应三种配置）
  ```
  sample_id,dataset,language,dialog_id,turn_index,text_length,audio_duration,duration_group,mode,ttft_ms,asr_time_ms,llm_prefill_time_ms,error
  crosswoz_391_turn3,crosswoz,zh,391,3,186,18.5,long,baseline,1023.4,780.2,243.2,
  crosswoz_391_turn3,crosswoz,zh,391,3,186,18.5,long,streaming_asr_only,412.6,280.5,132.1,
  crosswoz_391_turn3,crosswoz,zh,391,3,186,18.5,long,full_streaming,145.8,92.4,53.4,
  ```

- `exp2_gains_*.csv`（逐样本增益，论文表格可直接引用）
  ```
  sample_id,dataset,language,dialog_id,turn_index,text_length,duration_group,audio_duration_s,baseline_ttft_ms,streaming_asr_ttft_ms,full_streaming_ttft_ms,asr_gain_ms,kv_gain_ms,total_gain_ms,total_gain_ratio_%
  crosswoz_391_turn3,crosswoz,zh,391,3,186,long,18.5,1023.4,412.6,145.8,610.8,266.8,877.6,85.8
  ```

- `exp2_statistics_*.csv`（分组均值，适合论文图表）
  ```
  group,sample_count,avg_duration_s,baseline_ttft_mean_ms,streaming_asr_ttft_mean_ms,full_streaming_ttft_mean_ms,asr_gain_ms,kv_gain_ms,total_gain_ms,total_gain_ratio_%
  long,50,21.4,988.2,410.5,152.3,577.7,258.2,835.9,84.6
  ```

### 论文撰写提示

- **方法描述**：引用 `config` 记录的 chunk 长度、模型、设备、预热轮数，确保复现性。  
- **结果呈现**：  
  - “流式 ASR 贡献” = Baseline → Streaming ASR Only 的降幅  
  - “KV 预填充贡献” = Streaming ASR Only → Full Streaming 的降幅  
  - 建议在正文展示分组均值（`statistics`），附录提供逐样本增益（`sample_gains` / `exp2_gains_*`）。  
- **数据充分性**：逐样本层面保留了 `dataset/language/dialog_id/turn_index/text_length/audio_duration`，可用于按语种/轮次/长度做进一步分析；若需更多字段（如 ASR 文本、LLM 预览）可从 JSON 的 `results` 中提取。

## 结果解读

- **ASR 贡献**：Baseline → Streaming ASR Only 的 TTFT 降幅
- **KV 贡献**：Streaming ASR Only → Full Streaming 的 TTFT 降幅
- **总优化率**：Baseline → Full Streaming 的整体 TTFT 降幅
- 重点观察长语音组（默认），可扩展到 very_long/extra_long 检查趋势

## 常见问题

- **样本太少？** 调整 `--duration-groups` 或增加数据处理输出
- **速度慢？** 减小模型尺寸、降低 `--max-samples`、确保 GPU 可用
- **显存不足？** 减小 `--max-tokens` 或使用更小模型
- **结果波动大？** 增加样本数，并确保预热完成

## 相关文档

- [实验设计方案](../../EXPERIMENT_DESIGN.md)
- [实验一使用说明](./EXP1_LATENCY_USAGE.md)
- [项目 README](../../../README.MD)

