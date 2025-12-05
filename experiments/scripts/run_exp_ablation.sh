#!/bin/bash
# =============================================================================
# 实验二：消融实验 (Ablation)
#
# 用法：
#   ./experiments/scripts/run_exp_ablation.sh [模式] [选项]
#
# 示例：
#   ./experiments/scripts/run_exp_ablation.sh full
#   ./experiments/scripts/run_exp_ablation.sh test
#   ./experiments/scripts/run_exp_ablation.sh crosswoz
#   ./experiments/scripts/run_exp_ablation.sh full --duration-groups long very_long
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
DURATION_GROUPS=("long")   # 默认只跑长语音组
LOG_LEVEL="INFO"
WARMUP_ROUNDS="3"
CHUNK_DURATION="500"
MAX_TOKENS="50"

print_help() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  实验二：消融实验 (Ablation)${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    echo "用法: $0 [模式] [选项]"
    echo ""
    echo "模式:"
    echo "  full        运行完整实验（默认 long 分组）"
    echo "  test        测试模式（5 个样本，long 分组）"
    echo "  crosswoz    仅 CrossWOZ 数据集"
    echo "  multiwoz    仅 MultiWOZ 数据集"
    echo "  help        显示帮助"
    echo ""
    echo "选项:"
    echo "  --max-samples N           最大样本数（快速验证）"
    echo "  --duration-groups g1 g2   指定分组（默认: long；可选: short/medium/long/very_long/extra_long）"
    echo "  --asr-device              ASR 设备 (auto/cuda/cpu)"
    echo "  --llm-device              LLM 设备 (auto/cuda/cpu)"
    echo "  --log-level               日志级别 (DEBUG/INFO/WARNING/ERROR)"
    echo "  --warmup-rounds           预热轮数 (默认: 3)"
    echo "  --chunk-duration          流式分块时长 ms (默认: 500)"
    echo "  --max-tokens              LLM 最大生成 token 数 (默认: 50)"
    echo ""
    echo "示例:"
    echo "  $0 full"
    echo "  $0 test"
    echo "  $0 crosswoz --max-samples 10 --duration-groups long very_long"
    echo "  $0 full --asr-device cuda --llm-device cuda --warmup-rounds 5"
}

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

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
            MAX_SAMPLES="$2"; shift 2 ;;
        --duration-groups)
            shift
            DURATION_GROUPS=()
            while [[ $# -gt 0 && "$1" != --* ]]; do
                DURATION_GROUPS+=("$1"); shift
            done
            ;;
        --asr-device)
            ASR_DEVICE="$2"; shift 2 ;;
        --llm-device)
            LLM_DEVICE="$2"; shift 2 ;;
        --log-level)
            LOG_LEVEL="$2"; shift 2 ;;
        --warmup-rounds)
            WARMUP_ROUNDS="$2"; shift 2 ;;
        --chunk-duration)
            CHUNK_DURATION="$2"; shift 2 ;;
        --max-tokens)
            MAX_TOKENS="$2"; shift 2 ;;
        *)
            log_error "未知参数: $1"
            print_help
            exit 1 ;;
    esac
done

build_args() {
    local args="--dataset $DATASET --asr-device $ASR_DEVICE --llm-device $LLM_DEVICE --log-level $LOG_LEVEL --warmup-rounds $WARMUP_ROUNDS --chunk-duration $CHUNK_DURATION --max-tokens $MAX_TOKENS"

    # duration groups
    if [[ ${#DURATION_GROUPS[@]} -gt 0 ]]; then
        args="$args --duration-groups ${DURATION_GROUPS[*]}"
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
    log_info "运行实验二..."
    log_info "参数: $args"
    echo ""

    activate_conda
    uv run python -m experiments.scripts.run_exp_ablation $args
}

# 主逻辑
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

