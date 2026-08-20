# RUNINFO — E4 插桩 + 完整回复复跑（R4+R5，意见5）

- 命令: `uv run python -m experiments.scripts.run_exp_latency --dataset all --sample-list experiments/results/revision/r1_stats/repeat_subset_ids.json --asr-device cuda:0 --llm-device cuda:1 --suffix-segments 0 --max-tokens 128 --save-full-response --save-fragments --output-dir experiments/results/revision/r4_commit --no-resume`
- 起止时间: 2026-08-20 16:23:07 → 17:15:23（CST）
- 样本数: 50 × 2 模式 = 100 条结果
- error 数: 0
- 日志: run.log；产物含 commit_log.jsonl（375 commit + 224 correction）
