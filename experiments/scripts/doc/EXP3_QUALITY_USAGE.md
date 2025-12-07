# 实验三：准确率与质量验证使用说明

## 目标
- 对比 **非流式 ASR** 与 **流式 ASR** 的识别准确率（WER/CER）
- 关注长语音场景，验证流式处理带来的精度影响

## 核心配置
- 代码入口：`experiments/scripts/run_exp_quality.py`
- 输出目录：`experiments/results/exp3_quality`（自动带时间戳）
- 指标：`wer`（词/字级错误率）、`cer`（字符错误率）、`asr_time_ms`
- 模型：`--asr-model-size`（默认来自配置），`--asr-device`（auto/cuda/cpu）

## 运行方式
```bash
# 默认运行（使用 uv，无需激活 conda）
uv run python -m experiments.scripts.run_exp_quality

# 指定数据集与样本数（推荐 50-200 条）
uv run python -m experiments.scripts.run_exp_quality --dataset crosswoz --max-samples 100

# 指定设备、分块与预热
uv run python -m experiments.scripts.run_exp_quality \
    --asr-device cuda --asr-model-size base \
    --chunk-duration 500 --warmup-rounds 3
```

### 参数说明
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data-dir` | `experiments/datasets/processed` | 处理后数据目录 |
| `--dataset` | `all` | 选择 crosswoz/multiwoz/all |
| `--max-samples` | None | 最大样本数（论文建议 50-200） |
| `--asr-device` | `auto` | ASR 设备 (auto/cuda/cpu) |
| `--asr-model-size` | 配置默认 | ASR 模型 (tiny/base/small/medium/large) |
| `--chunk-duration` | 500 | 流式分块时长 ms |
| `--warmup-rounds` | 2 | 预热轮数 |
| `--prefix-segments` | 1 | ASR 前缀段数（影响上下文和延迟） |
| `--suffix-segments` | 1 | ASR 后缀段数（影响准确率和延迟） |
| `--recognition-threshold` | 2.0 | ASR 识别阈值（秒） |
| `--output-dir` | `experiments/results/exp3_quality` | 输出目录 |
| `--log-level` | `INFO` | 日志级别 |

## 输出文件
生成于 `experiments/results/exp3_quality`，均含时间戳：

1) **详细 JSON** `exp3_results_YYYYMMDD_HHMMSS.json`  
   - `config`：运行参数  
   - `results`：逐样本流式/非流式转写、WER/CER、耗时  
   - `statistics`：按 mode、dataset、language、overall 的均值/方差

2) **逐样本汇总 CSV** `exp3_summary_YYYYMMDD_HHMMSS.csv`  
   列：`sample_id,dataset,language,dialog_id,turn_index,text_length,audio_duration,mode,wer,cer,asr_time_ms,error`

3) **统计 CSV** `exp3_statistics_YYYYMMDD_HHMMSS.csv`  
   列：`scope,sample_count,avg_duration_s,wer_mean,wer_std,cer_mean,cer_std`  
   - scope 包含：
     - **mode 维度**（streaming / non-streaming）— 论文核心对比
     - dataset 维度（crosswoz / multiwoz）
     - language 维度（zh / en）
     - overall

### 数据样例（虚拟）
```
# exp3_summary_*.csv（同一 sample_id 有两行：streaming 和 non-streaming）
sample_id,dataset,language,dialog_id,turn_index,text_length,audio_duration,mode,wer,cer,asr_time_ms,error
crosswoz_391_turn3,crosswoz,zh,391,3,186,18.5,non-streaming,0.0720,0.0516,812.4,
crosswoz_391_turn3,crosswoz,zh,391,3,186,18.5,streaming,0.0940,0.0667,285.3,
```
```
# exp3_statistics_*.csv（mode 统计在最前面，论文核心对比）
scope,sample_count,avg_duration_s,wer_mean,wer_std,cer_mean,cer_std
non-streaming,100,17.2,0.076,0.024,0.055,0.017
streaming,100,17.2,0.098,0.029,0.069,0.021
crosswoz,200,17.8,0.081,0.026,0.059,0.018
multiwoz,200,16.5,0.102,0.031,0.071,0.022
overall,400,17.2,0.091,0.030,0.065,0.021
```

## 文本归一化处理

为保证流式与非流式 ASR 的公平比较，WER/CER 计算前会自动进行**文本归一化**：

1. **移除标点符号**：中英文标点均会被移除
   - 中文标点：`，。！？、；：""''（）【】《》—…·`
   - 英文标点：`,.!?;:'"()[]{}<>-_+=*/\@#$%^&|`~`

2. **原因说明**：
   - 流式 ASR 使用词级时间戳匹配，标点符号可能落在音频段边界而丢失
   - 非流式 ASR 直接使用完整转录文本，包含标点符号
   - 为公平比较，统一移除标点后再计算准确率
   - 这是语音识别评估的业界标准做法

3. **影响**：归一化后的 WER/CER 仅反映**内容准确率**，不包含标点预测的评估

## 论文撰写提示
- **核心对比**：直接引用 `statistics` 中 `by_mode` 的 streaming vs non-streaming 均值，展示流式处理的精度影响。
- 按 dataset / language 的统计可用于分析不同语言/数据集的差异。
- 中文推荐使用 CER；WER 通过逐字空格化处理，便于中英文统一对比。
- 逐样本结果可用于绘制误差条/箱线图。
- 若需语义一致性（LLM 级别）扩展，可在得到转写后使用相同 LLM 对比回复相似度，作为后续工作附录。
- **注意**：由于进行了标点归一化，WER/CER 仅反映内容准确率，标点预测不在评估范围内。

