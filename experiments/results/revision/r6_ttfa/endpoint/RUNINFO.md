# RUNINFO — E5 端点等待测量（R6.1，意见2）

- 命令: `uv run python -m experiments.scripts.run_exp_latency --dataset all --sample-list experiments/results/revision/r1_stats/repeat_subset_ids.json --asr-device cuda:0 --llm-device cuda:1 --suffix-segments 0 --append-silence-ms 2000 --output-dir experiments/results/revision/r6_ttfa/endpoint --no-resume`
- 起止时间: 2026-08-20 17:15:23 → 18:04:29（CST）
- 样本数: 50 × 2 模式 = 100 条结果
- error 数: 0
- 日志: run.log
