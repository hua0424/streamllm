# LocalAgreement-2-style 基线：方法与排除规则说明（W9 定稿）

对应 PRE-PAPER-AUDIT P1-3 / 复审 §5.1。供论文 Table VII 方法描述与审稿回复引用。

## 命名

论文中须称"**项目内实现的 LocalAgreement-2-style baseline**"（同引擎自实现变体），
不得只写"LocalAgreement-2"暗示 ufal 原版。

## 实现要素（修复后版本，`src/asr/local_agreement_streamer.py`）

1. **绝对音频时间轴**：`buffer_start_abs` / `committed_words` / `committed_end_abs`，
   词级时间戳全部落在绝对时间轴上（修复帧索引错位：修剪后 `n_committed` 不再引用旧帧）；
2. **句界感知裁剪**（sentence-boundary-aware trimming）：优先在句末标点
   （`。！？!?.…`）处裁剪缓冲，避免句末碎片打头导致 Whisper turbo 水印幻觉塌缩；
   缓冲超 `la_max_buffer_s=15.0` 时强制裁剪（与 ufal buffer_trimming_sec 对齐）；
3. **`la_max_buffer_s=15.0`**：上限参数显式记录于运行配置；
4. **标点鲁棒一致性**（punctuation-robust agreement）：归一化比较（`_norm_text` 去边缘标点）、
   纯标点词透明、时间边界提交规则（`end > committed_end_abs + eps`，eps=0.02s）、
   跨骑词排除、去重护栏（区间重叠 + 同文本）、flush 幂等。

## 样本与排除规则（505 → 498）

来源清单：`r3_baseline_la/exp2_ablation_sample_list.json`（与 System A/B 重算同一干净成对子集）。

- 候选 505 条（long/very_long/extra_long 三组）；
- **运行错误成对排除 3 条**：crosswoz_10382_turn3、crosswoz_9365_turn3、multiwoz_MUL2016_turn3；
- **流式挂起成对排除 4 条**（任一流式模式 TTFT>10000ms 判定挂起）：
  crosswoz_10040_turn6、crosswoz_11361_turn6、crosswoz_2228_turn6、crosswoz_9080_turn5；
- 未过滤失败/挂起率：**7/505 = 1.4%**；
- 最终 498 条：long 108 / very_long 150 / extra_long 240；
- 取代旧手工 static-repair.csv（其排除清单不可复现）。

挂起机制注记（2026-08-21 代码核实）：旧实验脚本 `flush()` 返回 None 时 `is_final` 段不入队，
transcriber 永久等待（`run_exp_latency.py:675-678` + `:712-720`），4 条挂起样本即此机制所致；
W1 统一 TTFA 脚本已改用无条件 INPUT_CLOSED sentinel，同类样本将转为正常完成或显式 error。

## 汇总白名单

E3 汇总只读取最终白名单文件：
`r3_baseline_la/system_ab_rerun/exp1_results_20260820_035759.json`（A/B 重跑）与
`r3_baseline_la/la_results_20260821_074150.json`（LA）；
`invalid_dev3_frame_bug/` 为修复前带病中间产物，**任何汇总不得纳入**。

## 表述红线

- 质量对比写"同量级"，**不得写"统计等价"**；
- LA 优势口径统一为"LA-2-style 基线比 System B 慢约 34%"或"System B 比 LA 低约 26%"，
  不得混用（paired 检验见 `stats_inference/paired_inference.csv` 的 table7_b_vs_la 行：
  n=498，差 541.1ms，95% CI [485.3, 599.9]，p=1.2e-70）。
