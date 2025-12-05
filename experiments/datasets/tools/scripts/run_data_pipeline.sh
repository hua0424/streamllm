#!/bin/bash
# =============================================================================
# 实验数据处理管线运行脚本
# 
# 功能：
#   1. 数据预处理：将 MultiWOZ/CrossWOZ 转换为累积对话格式
#   2. TTS 生成：批量生成音频文件
#   3. 文本长度过滤：支持分别设置中英文文本长度上限
#
# 使用方法：
#   ./experiments/datasets/tools/scripts/run_data_pipeline.sh [模式] [选项]
#
# 示例：
#   ./experiments/datasets/tools/scripts/run_data_pipeline.sh preprocess
#   ./experiments/datasets/tools/scripts/run_data_pipeline.sh full --top-n-dialogs 50
#   ./experiments/datasets/tools/scripts/run_data_pipeline.sh preprocess --max-text-length-zh 300 --max-text-length-en 800
#
# 环境要求：
#   - conda 环境 streamllm（其中安装了 uv）
# =============================================================================

set -e  # 遇到错误立即退出

# 切换到项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$PROJECT_ROOT"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 默认配置
TTS_URL="${TTS_URL:-http://host.docker.internal:20401}"
DATASET="${DATASET:-all}"
TOP_N_DIALOGS="100"
MAX_SAMPLES=""
TTS_SPEED="0.8"
TTS_WORKERS="4"
CONDA_ENV="${CONDA_ENV:-streamllm}"

# 文本长度限制（默认不限制，可通过参数设置）
MAX_TEXT_LENGTH_ZH=""
MAX_TEXT_LENGTH_EN=""

# 打印帮助信息
print_help() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  实验数据处理管线${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    echo "用法: $0 [模式] [选项]"
    echo ""
    echo "模式:"
    echo "  preprocess    仅数据预处理（生成JSON文件）"
    echo "  tts-only      仅TTS生成（需要已有JSON文件）"
    echo "  full          完整管线（预处理 + TTS）"
    echo "  test          测试模式（少量数据）"
    echo "  help          显示此帮助信息"
    echo ""
    echo "选项:"
    echo "  --dataset             数据集 (crosswoz/multiwoz/all)，默认: all"
    echo "  --top-n-dialogs       选取文本最长的前N个对话，默认: 100"
    echo "  --max-samples         每对话最大样本数"
    echo "  --max-text-length-zh  中文文本最大字符数，默认: 720（约150秒音频）"
    echo "  --max-text-length-en  英文文本最大字符数，默认: 2050（约150秒音频）"
    echo "  --tts-url             TTS服务地址，默认: \$TTS_URL"
    echo "  --tts-speed           TTS语速，默认: 0.8"
    echo "  --tts-workers         TTS并发数，默认: 4"
    echo ""
    echo "环境变量:"
    echo "  TTS_URL        TTS服务地址（可通过环境变量设置）"
    echo "  CONDA_ENV      conda环境名称，默认: streamllm"
    echo ""
    echo "示例:"
    echo "  # 仅预处理数据（使用默认文本长度限制）"
    echo "  $0 preprocess"
    echo ""
    echo "  # 完整管线，选取文本最长的50个对话"
    echo "  $0 full --top-n-dialogs 50"
    echo ""
    echo "  # 使用自定义文本长度限制"
    echo "  $0 preprocess --max-text-length-zh 300 --max-text-length-en 800"
    echo ""
    echo "  # 仅处理CrossWOZ"
    echo "  $0 preprocess --dataset crosswoz"
    echo ""
    echo "  # 测试模式（10个对话）"
    echo "  $0 test"
}

# 打印带颜色的消息
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 解析命令行参数
MODE="${1:-help}"
shift || true

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --top-n-dialogs)
            TOP_N_DIALOGS="$2"
            shift 2
            ;;
        --max-samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --max-text-length-zh)
            MAX_TEXT_LENGTH_ZH="$2"
            shift 2
            ;;
        --max-text-length-en)
            MAX_TEXT_LENGTH_EN="$2"
            shift 2
            ;;
        --tts-url)
            TTS_URL="$2"
            shift 2
            ;;
        --tts-speed)
            TTS_SPEED="$2"
            shift 2
            ;;
        --tts-workers)
            TTS_WORKERS="$2"
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
    local args="--dataset $DATASET --top-n-dialogs $TOP_N_DIALOGS"
    
    if [[ -n "$MAX_SAMPLES" ]]; then
        args="$args --max-samples-per-dialog $MAX_SAMPLES"
    fi
    
    # 文本长度限制
    if [[ -n "$MAX_TEXT_LENGTH_ZH" ]]; then
        args="$args --max-text-length-zh $MAX_TEXT_LENGTH_ZH"
    fi
    
    if [[ -n "$MAX_TEXT_LENGTH_EN" ]]; then
        args="$args --max-text-length-en $MAX_TEXT_LENGTH_EN"
    fi
    
    args="$args --tts-url $TTS_URL --tts-speed $TTS_SPEED --tts-workers $TTS_WORKERS"
    
    echo "$args"
}

# 激活 conda 环境
activate_conda() {
    # 尝试初始化 conda（支持多种安装路径）
    if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    elif [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
        source "/opt/conda/etc/profile.d/conda.sh"
    elif [[ -f "/root/miniconda3/etc/profile.d/conda.sh" ]]; then
        source "/root/miniconda3/etc/profile.d/conda.sh"
    fi
    
    # 激活环境
    conda activate "$CONDA_ENV" || {
        log_error "无法激活 conda 环境 '$CONDA_ENV'"
        log_error "请确保已安装 conda 并创建了 $CONDA_ENV 环境"
        exit 1
    }
    
    log_info "已激活 conda 环境: $CONDA_ENV"
}

# 运行管线
run_pipeline() {
    local extra_args="$1"
    local cmd_args=$(build_args)
    
    log_info "项目根目录: $PROJECT_ROOT"
    log_info "运行数据处理管线..."
    log_info "参数: $cmd_args $extra_args"
    
    # 显示文本长度限制信息
    if [[ -n "$MAX_TEXT_LENGTH_ZH" ]] || [[ -n "$MAX_TEXT_LENGTH_EN" ]]; then
        log_info "文本长度限制:"
        [[ -n "$MAX_TEXT_LENGTH_ZH" ]] && log_info "  中文: $MAX_TEXT_LENGTH_ZH 字符"
        [[ -n "$MAX_TEXT_LENGTH_EN" ]] && log_info "  英文: $MAX_TEXT_LENGTH_EN 字符"
    fi
    echo ""
    
    # 先激活 conda 环境，然后使用 uv run
    activate_conda
    uv run python -m experiments.datasets.tools.run_pipeline $cmd_args $extra_args
}

# 主逻辑
case $MODE in
    preprocess)
        log_info "模式: 仅数据预处理"
        run_pipeline "--skip-tts"
        ;;
    tts-only)
        log_info "模式: 仅TTS生成"
        run_pipeline "--skip-preprocess"
        ;;
    full)
        log_info "模式: 完整管线"
        run_pipeline ""
        ;;
    test)
        log_info "模式: 测试（10个对话）"
        TOP_N_DIALOGS=10
        run_pipeline "--skip-tts"
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
log_info "完成！"
