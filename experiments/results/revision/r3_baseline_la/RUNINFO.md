# RUNINFO — E3 LocalAgreement-2 基线（修复后重跑，R3，意见4）

- 命令: `uv run python -m experiments.scripts.run_exp_baseline_la --dataset all --sample-list experiments/results/revision/r3_baseline_la/exp2_ablation_sample_list.json --asr-device cuda:0 --llm-device cuda:1 --output-dir experiments/results/revision/r3_baseline_la --no-resume`
- 代码 commit: c965240（含 DEV-3 修复 6d74c1c：绝对时间轴提交 + 标点鲁棒一致比较 + 句界裁剪 la_max_buffer_s=15.0）
- 起止时间: 2026-08-20 22:54:39 → 2026-08-21 07:41:53（CST，约 8.8h）
- 样本数: 498，error 数: 0
- 日志: la_run.log；旧无效现场: invalid_dev3_frame_bug/
- 跑后 QA（评审 R2 七项，全过）：
  1. 完成度 498/498 error 0 ✓
  2. config 块 trailing_margin_s=0.0 / la_max_buffer_s=15.0 / cuda:0+cuda:1 / turbo+Qwen2-7B ✓
  3. WER mean=0.130 / CER mean=0.118（修复前 0.545；<0.2 阈值）✓
  4. LA/SysB 转写长度比 mean=0.98 median=0.99，<0.7 占比 0%（修复前 0.52/79%）✓
  5. divergence mean=1.0 max=7，无 >20 样本 ✓
  6. 回放抽查 crosswoz_10296_turn2 WER 0.0185（修复前 0.88）/ multiwoz_MUL0023_turn4 0.162 / crosswoz_9080_turn2 0.068，无跳段无重复 ✓
  7. TTFT：LA mean 2115ms（long/very_long/extra_long=1741/2200/2230ms）vs System B 1574ms（1464/1551/1638ms），LA 因全缓冲重解码系统性偏高，量级可解释 ✓
