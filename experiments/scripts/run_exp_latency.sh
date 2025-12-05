#!/bin/bash
# =============================================================================
# 实验一：延迟与语音长度关系验证
#
# 使用方法：
#   ./experiments/scripts/run_exp_latency.sh [模式] [选项]
#
# 示例：
#   ./experiments/scripts/run_exp_latency.sh full
#   ./experiments/scripts/run_exp_latency.sh test
#   ./experiments/scripts/run_exp_latency.sh crosswoz
# =============================================================================

set -e

# 切换到项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 默认配置
CONDA_ENV="${CONDA_ENV:-streamllm}"
ASR_DEVICE="${ASR_DEVICE:-auto}"
LLM_DEVICE="${LLM_DEVICE:-auto}"
MAX_SAMPLES=""
DATASET="all"
LOG_LEVEL="INFO"
WARMUP_ROUNDS="3"

# 打印帮助
print_help() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  实验一：延迟与语音长度关系验证${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    echo "用法: $0 [模式] [选项]"
    echo ""
    echo "模式:"
    echo "  full        运行完整实验（所有数据集）"
    echo "  test        测试模式（5个样本）"
    echo "  crosswoz    仅 CrossWOZ 数据集"
    echo "  multiwoz    仅 MultiWOZ 数据集"
    echo "  help        显示帮助"
    echo ""
    echo "选项:"
    echo "  --max-samples N    最大样本数"
    echo "  --asr-device       ASR 设备 (auto/cuda/cpu)"
    echo "  --llm-device       LLM 设备 (auto/cuda/cpu)"
    echo "  --log-level        日志级别 (DEBUG/INFO/WARNING)"
    echo "  --warmup-rounds    预热轮数 (默认: 3)"
    echo ""
    echo "示例:"
    echo "  $0 full"
    echo "  $0 test"
    echo "  $0 crosswoz --max-samples 10"
}

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 激活 conda 环境
activate_conda() {
    if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    elif [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
        source "/opt/conda/etc/profile.d/conda.sh"
    elif [[ -f "/root/miniconda3/etc/profile.d/conda.sh" ]]; then
        source "/root/miniconda3/etc/profile.d/conda.sh"
    fi
    
    conda activate "$CONDA_ENV" || {
        log_error "无法激活 conda 环境 '$CONDA_ENV'"
        exit 1
    }
    
    log_info "已激活 conda 环境: $CONDA_ENV"
}

# 解析参数
MODE="${1:-help}"
shift || true

while [[ $# -gt 0 ]]; do
    case $1 in
        --max-samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --asr-device)
            ASR_DEVICE="$2"
            shift 2
            ;;
        --llm-device)
            LLM_DEVICE="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --warmup-rounds)
            WARMUP_ROUNDS="$2"
            shift 2
            ;;
        *)
            log_error "未知参数: $1"
            print_help
            exit 1
            ;;
    esac
done

# 构建命令参数
build_args() {
    local args="--dataset $DATASET --asr-device $ASR_DEVICE --llm-device $LLM_DEVICE --log-level $LOG_LEVEL --warmup-rounds $WARMUP_ROUNDS"
    
    if [[ -n "$MAX_SAMPLES" ]]; then
        args="$args --max-samples $MAX_SAMPLES"
    fi
    
    echo "$args"
}

# 运行实验
run_experiment() {
    local args=$(build_args)
    
    log_info "项目根目录: $PROJECT_ROOT"
    log_info "运行实验一..."
    log_info "参数: $args"
    echo ""
    
    activate_conda
    uv run python -m experiments.scripts.run_exp_latency $args
}

# 主逻辑
case $MODE in
    full)
        log_info "模式: 完整实验"
        DATASET="all"
        run_experiment
        ;;
    test)
        log_info "模式: 测试（5个样本）"
        DATASET="all"
        MAX_SAMPLES=5
        run_experiment
        ;;
    crosswoz)
        log_info "模式: CrossWOZ 数据集"
        DATASET="crosswoz"
        run_experiment
        ;;
    multiwoz)
        log_info "模式: MultiWOZ 数据集"
        DATASET="multiwoz"
        run_experiment
        ;;
    help|--help|-h)
        print_help
        exit 0
        ;;
    *)
        log_error "未知模式: $MODE"
        print_help
        exit 1
        ;;
esac

echo ""
log_info "实验完成！"

