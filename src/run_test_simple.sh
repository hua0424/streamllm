#!/bin/bash

# 全链路流式测试脚本 (ASR + LLM)
# 运行方式: ./src/run_test_simple.sh

echo "===== 全链路流式测试 (ASR + LLM) ====="

# 1. 切换到项目根目录
# 获取脚本所在目录的上一级目录 (假设脚本在 src/ 下)
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
echo "工作目录: $(pwd)"

# 2. 设置环境变量
export PYTHONPATH=$PYTHONPATH:.
# 设置 CUDA/CUDNN 库路径 (根据环境调整)
export LD_LIBRARY_PATH="/usr/local/app/jupyterlab/yanjiu/streamllm/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib/:$LD_LIBRARY_PATH"

# 3. 配置运行参数 (可在此处修改默认值)

# 音频文件路径
AUDIO_FILE="data/processed_audio/length10/0_0_d1.wav"

# 运行设备: cuda, cpu, auto
DEVICE="cuda"

# 音频分块时长 (ms)
CHUNK_DURATION=100

# ASR 模型配置
ASR_MODEL="large"          # tiny, base, small, medium, large-v3
ASR_PREFIX_SEGMENTS=1     # 保留的前缀段数
ASR_SUFFIX_SEGMENTS=1     # 保留的后缀段数

# LLM 模型配置
# 默认使用 Qwen/Qwen2.5-7B-Instruct，也可以通过环境变量 LLM_MODEL_NAME 覆盖
LLM_MODEL="Qwen/Qwen2-7B-Instruct"

# 结果保存配置
SAVE_RESULTS=true
RESULTS_DIR="experiments/results/pipeline_test"

# 是否模拟流式延迟 (true/false)
SIMULATE_DELAY=true

# 4. 构建并执行命令
echo "------------------------------------------------"
echo "配置信息:"
echo "  Audio: $AUDIO_FILE"
echo "  Device: $DEVICE"
echo "  ASR Model: $ASR_MODEL"
echo "  LLM Model: $LLM_MODEL"
echo "  Simulate Delay: $SIMULATE_DELAY"
echo "------------------------------------------------"

CMD="uv run python -m src.run_test_simple \
    --audio \"$AUDIO_FILE\" \
    --device \"$DEVICE\" \
    --chunk-duration $CHUNK_DURATION \
    --asr-model-size \"$ASR_MODEL\" \
    --asr-prefix-segments $ASR_PREFIX_SEGMENTS \
    --asr-suffix-segments $ASR_SUFFIX_SEGMENTS \
    --llm-model-name \"$LLM_MODEL\" \
    --results-dir \"$RESULTS_DIR\""

# 对比测试（共享模型 + 预热）
python -m src.run_test_simple --audio data/audio.wav --mode both --save-results
python -m src.run_test_simple --mode both --log-level DEBUG --save-results --asr-model-size large --asr-device cuda --llm-device cuda

# 单独运行流式测试（独立模型，无预热）
python -m src.run_test_simple --audio data/audio.wav --mode streaming

# 单独运行非流式测试（独立模型，无预热）
python -m src.run_test_simple --audio data/audio.wav --mode non-streaming

if [ "$SAVE_RESULTS" = true ]; then
    CMD="$CMD --save-results"
fi

if [ "$SIMULATE_DELAY" = true ]; then
    CMD="$CMD --simulate-delay"
fi

echo "执行命令: $CMD"
echo "------------------------------------------------"

# 执行
eval $CMD

echo "------------------------------------------------"
echo "测试结束"