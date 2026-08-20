# RUNINFO — E3 System A/B 本机重跑（498 消融清单）

- 命令: `uv run python -m experiments.scripts.run_exp_latency --dataset all --sample-list experiments/results/revision/r3_baseline_la/exp2_ablation_sample_list.json --asr-device cuda:0 --llm-device cuda:1 --suffix-segments 0 --output-dir experiments/results/revision/r3_baseline_la/system_ab_rerun`
- 起止时间: 2026-08-19 18:11:38 → 2026-08-20 03:57:59（CST，约 9.8h）
- 样本数: 498 × 2 模式 = 996 条结果
- error 数: 0
- 日志: run.log
- 备注: 需求方裁决要求的同机重跑（消除 LA 对比的跨机污染）
