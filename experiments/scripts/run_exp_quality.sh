#!/bin/bash
# =============================================================================
# 实验三：准确率与质量验证
#
# 用法：
#   ./experiments/scripts/run_exp_quality.sh [模式] [选项]
#
# 示例：
#   ./experiments/scripts/run_exp_quality.sh full
#   ./experiments/scripts/run_exp_quality.sh test
#   ./experiments/scripts/run_exp_quality.sh crosswoz --max-samples 50
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

CONDA_ENV="${CONDA_ENV:-streamllm}"
ASR_DEVICE="${ASR_DEVICE:-auto}"
ASR_MODEL_SIZE=""
MAX_SAMPLES=""
DATASET="all"
LOG_LEVEL="INFO"
WARMUP_ROUNDS="2"
CHUNK_DURATION="500"
OUTPUT_DIR="experiments/results/exp3_quality"

print_help() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  实验三：准确率与质量验证${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    echo "用法: $0 [模式] [选项]"
    echo ""
    echo "模式:"
    echo "  full        运行完整实验（默认全部样本）"
    echo "  test        测试模式（5 个样本）"
    echo "  crosswoz    仅 CrossWOZ 数据集"
    echo "  multiwoz    仅 MultiWOZ 数据集"
    echo "  help        显示帮助"
    echo ""
    echo "选项:"
    echo "  --max-samples N       最大样本数"
    echo "  --asr-device          ASR 设备 (auto/cuda/cpu)"
    echo "  --asr-model-size      ASR 模型 (tiny/base/small/medium/large)"
    echo "  --chunk-duration      流式分块时长 ms (默认 500)"
    echo "  --warmup-rounds       预热轮数 (默认 2)"
    echo "  --output-dir          输出目录"
    echo "  --log-level           日志级别 (DEBUG/INFO/WARNING/ERROR)"
    echo ""
    echo "示例:"
    echo "  $0 full --asr-device cuda --asr-model-size base"
    echo "  $0 crosswoz --max-samples 100"
}

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

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
    conda activate "$CONDA_ENV" || { log_error "无法激活 conda 环境 '$CONDA_ENV'"; exit 1; }
    log_info "已激活 conda 环境: $CONDA_ENV"
}

MODE="${1:-help}"
shift || true

while [[ $# -gt 0 ]]; do
    case $1 in
        --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
        --asr-device) ASR_DEVICE="$2"; shift 2 ;;
        --asr-model-size) ASR_MODEL_SIZE="$2"; shift 2 ;;
        --chunk-duration) CHUNK_DURATION="$2"; shift 2 ;;
        --warmup-rounds) WARMUP_ROUNDS="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --log-level) LOG_LEVEL="$2"; shift 2 ;;
        *)
            log_error "未知参数: $1"
            print_help
            exit 1 ;;
    esac
done

build_args() {
    local args="--dataset $DATASET --asr-device $ASR_DEVICE --chunk-duration $CHUNK_DURATION --warmup-rounds $WARMUP_ROUNDS --output-dir $OUTPUT_DIR --log-level $LOG_LEVEL"
    if [[ -n "$ASR_MODEL_SIZE" ]]; then
        args="$args --asr-model-size $ASR_MODEL_SIZE"
    fi
    if [[ -n "$MAX_SAMPLES" ]]; then
        args="$args --max-samples $MAX_SAMPLES"
    fi
    echo "$args"
}

run_experiment() {
    local args
    args=$(build_args)
    log_info "项目根目录: $PROJECT_ROOT"
    log_info "运行实验三..."
    log_info "参数: $args"
    echo ""
    activate_conda
    uv run python -m experiments.scripts.run_exp_quality $args
}

case $MODE in
    full)
        log_info "模式: 完整实验"
        DATASET="all"
        run_experiment
        ;;
    test)
        log_info "模式: 测试（5 个样本）"
        DATASET="all"
        MAX_SAMPLES=5
        run_experiment
        ;;
    crosswoz)
        log_info "模式: CrossWOZ"
        DATASET="crosswoz"
        run_experiment
        ;;
    multiwoz)
        log_info "模式: MultiWOZ"
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

