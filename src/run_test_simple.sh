#!/bin/bash
# ==============================================================================
# 流式语音到LLM响应全链路测试脚本
# 
# 用法:
#   ./src/run_test_simple.sh [MODE] [OPTIONS]
#
# 模式 (MODE):
#   streaming    - 只运行流式测试
#   non-streaming - 只运行非流式测试  
#   both         - 运行两者并对比（默认）
#
# 示例:
#   ./src/run_test_simple.sh both
#   ./src/run_test_simple.sh streaming --log-level DEBUG
#   ./src/run_test_simple.sh both --asr-model-size large --save-results
# ==============================================================================

set -e  # 遇到错误时退出

# 切换到项目根目录
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=================================================="
echo "流式语音LLM全链路测试"
echo "=================================================="
echo "工作目录: $(pwd)"

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:."

# ==============================================================================
# 默认配置（可通过命令行参数覆盖）
# ==============================================================================
MODE="${1:-both}"  # 默认运行对比测试

# 移除第一个位置参数，剩余的传递给 Python 脚本
shift 2>/dev/null || true

# 默认音频文件（如果不存在，使用测试数据）
DEFAULT_AUDIO="experiments/datasets/processed/experiments/length_analysis/audio/long/sample_001.wav"

# ==============================================================================
# 检查音频文件
# ==============================================================================
check_audio_file() {
    local audio_path="$1"
    if [ ! -f "$audio_path" ]; then
        echo "警告: 音频文件不存在: $audio_path"
        echo "请指定有效的音频文件路径，例如:"
        echo "  ./src/run_test_simple.sh both --audio /path/to/audio.wav"
        exit 1
    fi
}

# ==============================================================================
# 运行测试
# ==============================================================================
echo ""
echo "运行模式: $MODE"
echo "额外参数: $@"
echo "=================================================="

# 构建命令
export LD_LIBRARY_PATH="/usr/local/app/jupyterlab/yanjiu/streamllm/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib/:$LD_LIBRARY_PATH"
# uv run python -m src.run_test_simple --mode both --log-level DEBUG --save-results --asr-model-size large --asr-device cuda --llm-device cuda

CMD="python -m src.run_test_simple --mode $MODE"

# 如果没有指定 --audio 参数，添加默认值
if [[ ! " $@ " =~ " --audio " ]]; then
    CMD="$CMD --audio $DEFAULT_AUDIO"
fi

# 添加额外参数
CMD="$CMD $@"

echo "执行命令: $CMD"
echo "=================================================="
echo ""

# 执行
eval $CMD

echo ""
echo "=================================================="
echo "测试完成"
echo "=================================================="
