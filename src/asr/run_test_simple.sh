#!/bin/bash

# 简单的流式ASR测试脚本

echo "===== 流式ASR测试（直接运行版） ====="

# 切换到脚本所在目录
cd "$(dirname "$0")"

# 创建结果目录
mkdir -p ../../experiments/results
export LD_LIBRARY_PATH="/usr/local/app/jupyterlab/yanjiu/streamllm/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib/:$LD_LIBRARY_PATH"
# 运行测试
echo "运行流式ASR测试..."
uv run python -m src.asr.run_stream_asr_test \
    --audio "/usr/local/app/jupyterlab/yanjiu/streamllm/experiments/datasets/processed/experiments/length_analysis/audio/long/sample_001.wav" \
    --chunk-duration 100 \
    --model-size large \
    --device cuda \
    --save-results \
    --test-mode streaming \
    --simulate-streaming-delay \
    --results-dir "/usr/local/app/jupyterlab/yanjiu/streamllm/src/asr/test_result" 

echo "测试完成！结果保存在 experiments/results 目录下。"

uv run python -m src.asr.run_stream_asr_test     --audio "/usr/local/app/jupyterlab/yanjiu/streamllm/experiments/datasets/processed/experiments/length_analysis/audio/long/sample_001.wav"     --chunk-duration 100     --model-size medium     --device cuda     --save-results     --test-mode both     --results-dir "/usr/local/app/jupyterlab/yanjiu/streamllm/src/asr/test_result"