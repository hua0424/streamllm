# endpoint E2E 冒烟 artifact（R4-P1-2）

- 命令：`HF_TOKEN= HF_HOME=<本地HF缓存> uv run python -m experiments.scripts.run_exp_latency --dataset crosswoz --max-samples 1 --asr-model-size tiny --llm-model-name Qwen/Qwen2.5-0.5B-Instruct --suffix-segments 0 --append-silence-ms 2000 --save-full-response --save-fragments --warmup-rounds 1 --output-dir experiments/results/revision/regression/_tmp_endpoint_smoke --no-resume`
- 运行时间戳：20260818_190557
- 完整输出（含两模式结果/CSV）在临时目录，跑完仅保留本 artifact；自动回归见 test_revision_regressions.py B6（fake 模型生产路径，可复现）
- 关键值：endpoint_detection_wait=0.215s, final_enqueue_wait=2.083s, error=''
