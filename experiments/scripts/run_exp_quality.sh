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

ASR_DEVICE="${ASR_DEVICE:-auto}"
ASR_MODEL_SIZE=""
MAX_SAMPLES=""
MAX_SAMPLES_PER_GROUP=""
DURATION_GROUPS=""
DATASET="all"
LOG_LEVEL="INFO"
WARMUP_ROUNDS="2"
CHUNK_DURATION="500"
OUTPUT_DIR="experiments/results/exp3_quality"
PREFIX_SEGMENTS=""
SUFFIX_SEGMENTS=""
RECOGNITION_THRESHOLD=""
BATCH_SIZE=""
NO_RESUME=""

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
    echo "  --max-samples N               最大样本数"
    echo "  --max-samples-per-group N     每个时长分组的最大样本数（确保各组均衡）"
    echo "  --duration-groups g1 g2       指定分组（默认: medium long very_long）"
    echo "  --asr-device                  ASR 设备 (auto/cuda/cuda:0/cuda:1/cpu)"
    echo "  --asr-model-size              ASR 模型 (tiny/base/small/medium/large)"
    echo "  --chunk-duration              流式分块时长 ms (默认 500)"
    echo "  --warmup-rounds               预热轮数 (默认 2)"
    echo "  --output-dir                  输出目录"
    echo "  --log-level                   日志级别 (DEBUG/INFO/WARNING/ERROR)"
    echo "  --prefix-segments             ASR 前缀段数 (默认: 1)"
    echo "  --suffix-segments             ASR 后缀段数 (默认: 1)"
    echo "  --recognition-threshold       ASR 识别阈值秒数 (默认: 2.0)"
    echo "  --batch-size N                每处理 N 个样本保存一次检查点 (默认: 100)"
    echo "  --no-resume                   不从检查点恢复，从头开始运行"
    echo ""
    echo "示例:"
    echo "  $0 full --asr-device cuda --asr-model-size base"
    echo "  $0 crosswoz --max-samples 100"
    echo "  $0 full --duration-groups long very_long extra_long --max-samples-per-group 100"
    echo "  $0 full --batch-size 50        # 更频繁保存检查点"
    echo "  $0 full --no-resume            # 从头开始，忽略已有检查点"
}

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

check_uv() {
    if ! command -v uv &> /dev/null; then
        log_error "uv 未安装，请先安装 uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
    log_info "使用 uv 运行实验"
}

MODE="${1:-help}"
shift || true

while [[ $# -gt 0 ]]; do
    case $1 in
        --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
        --max-samples-per-group) MAX_SAMPLES_PER_GROUP="$2"; shift 2 ;;
        --duration-groups)
            shift
            DURATION_GROUPS_ARRAY=()
            while [[ $# -gt 0 && "$1" != --* ]]; do
                DURATION_GROUPS_ARRAY+=("$1"); shift
            done
            DURATION_GROUPS="${DURATION_GROUPS_ARRAY[*]}"
            ;;
        --asr-device) ASR_DEVICE="$2"; shift 2 ;;
        --asr-model-size) ASR_MODEL_SIZE="$2"; shift 2 ;;
        --chunk-duration) CHUNK_DURATION="$2"; shift 2 ;;
        --warmup-rounds) WARMUP_ROUNDS="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --log-level) LOG_LEVEL="$2"; shift 2 ;;
        --prefix-segments) PREFIX_SEGMENTS="$2"; shift 2 ;;
        --suffix-segments) SUFFIX_SEGMENTS="$2"; shift 2 ;;
        --recognition-threshold) RECOGNITION_THRESHOLD="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --no-resume) NO_RESUME="true"; shift ;;
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
    if [[ -n "$MAX_SAMPLES_PER_GROUP" ]]; then
        args="$args --max-samples-per-group $MAX_SAMPLES_PER_GROUP"
    fi
    if [[ -n "$DURATION_GROUPS" ]]; then
        args="$args --duration-groups $DURATION_GROUPS"
    fi
    if [[ -n "$PREFIX_SEGMENTS" ]]; then
        args="$args --prefix-segments $PREFIX_SEGMENTS"
    fi
    if [[ -n "$SUFFIX_SEGMENTS" ]]; then
        args="$args --suffix-segments $SUFFIX_SEGMENTS"
    fi
    if [[ -n "$RECOGNITION_THRESHOLD" ]]; then
        args="$args --recognition-threshold $RECOGNITION_THRESHOLD"
    fi
    if [[ -n "$BATCH_SIZE" ]]; then
        args="$args --batch-size $BATCH_SIZE"
    fi
    if [[ -n "$NO_RESUME" ]]; then
        args="$args --no-resume"
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
    check_uv
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

