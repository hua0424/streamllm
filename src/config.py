# src/config.py
"""
项目配置文件
所有配置参数通过环境变量设置，支持 .env 文件
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# ==============================================================================
# 设备配置
# ==============================================================================
def get_default_device():
    """自动检测可用设备"""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"

DEVICE = os.getenv("DEVICE", "auto")
if DEVICE == "auto":
    DEVICE = get_default_device()

# ==============================================================================
# 模型配置
# ==============================================================================
# LLM 模型名称（默认使用 Qwen 系列小模型，适合测试）
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")

# ASR 模型大小 (tiny/base/small/medium/large)
ASR_MODEL_NAME = os.getenv("ASR_MODEL_NAME", "tiny")

# ==============================================================================
# Hugging Face 配置
# ==============================================================================
HF_ENDPOINT = os.getenv("HF_ENDPOINT")
HF_TOKEN = os.getenv("HF_TOKEN")
HF_HOME = os.getenv("HF_HOME")  # 模型缓存路径

# ==============================================================================
# 音频配置
# ==============================================================================
TARGET_SAMPLE_RATE = int(os.getenv("TARGET_SAMPLE_RATE", "16000"))  # 目标采样率 (Hz)
TARGET_CHANNELS = int(os.getenv("TARGET_CHANNELS", "1"))  # 目标通道数 (1=mono)

# ==============================================================================
# ASR 流式处理配置
# ==============================================================================
# VAD (Voice Activity Detection) 参数
VAD_PARAMETERS = {
    "threshold": float(os.getenv("VAD_THRESHOLD", "0.5")),
    "min_speech_duration_ms": int(os.getenv("VAD_MIN_SPEECH_MS", "250")),
    "max_speech_duration_s": float(os.getenv("VAD_MAX_SPEECH_S", "inf")),
    "min_silence_duration_ms": int(os.getenv("VAD_MIN_SILENCE_MS", "1000")),
    "speech_pad_ms": int(os.getenv("VAD_SPEECH_PAD_MS", "200"))
}

# 流式 ASR 块配置
STREAMING_ASR_CHUNK_SECONDS = float(os.getenv("ASR_CHUNK_SECONDS", "10"))
STREAMING_ASR_OVERLAP_SECONDS = float(os.getenv("ASR_OVERLAP_SECONDS", "2"))

# ==============================================================================
# 数据路径配置
# ==============================================================================
DATA_DIR = os.getenv("DATA_DIR", str(PROJECT_ROOT / "data"))
RAW_AUDIO_DIR = os.getenv("RAW_AUDIO_DIR", str(PROJECT_ROOT / "data/raw_audio"))
PROCESSED_AUDIO_DIR = os.getenv("PROCESSED_AUDIO_DIR", str(PROJECT_ROOT / "data/processed_audio"))
TRANSCRIPTS_DIR = os.getenv("TRANSCRIPTS_DIR", str(PROJECT_ROOT / "data/transcripts"))

# ==============================================================================
# 实验结果路径配置
# ==============================================================================
RESULTS_DIR = os.getenv("RESULTS_DIR", str(PROJECT_ROOT / "experiments/results"))
LATENCY_METRICS_DIR = os.getenv("LATENCY_METRICS_DIR", str(PROJECT_ROOT / "experiments/results/latency_metrics"))
WER_SCORES_DIR = os.getenv("WER_SCORES_DIR", str(PROJECT_ROOT / "experiments/results/wer_scores"))

# ==============================================================================
# 日志配置
# ==============================================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE = os.getenv("LOG_FILE", str(PROJECT_ROOT / "logs/experiment.log"))

# ==============================================================================
# 运行模式配置
# ==============================================================================
PRODUCTION_MODE = os.getenv("PRODUCTION_MODE", "false").lower() == "true"

# ==============================================================================
# 初始化
# ==============================================================================
# 确保 HF 缓存目录存在
if HF_HOME and not os.path.exists(HF_HOME):
    os.makedirs(HF_HOME, exist_ok=True)

# 仅在非生产环境下输出配置信息
if not PRODUCTION_MODE and LOG_LEVEL == "DEBUG":
    print(f"[Config] LLM Model: {LLM_MODEL_NAME}")
    print(f"[Config] ASR Model: {ASR_MODEL_NAME}")
    print(f"[Config] Device: {DEVICE}")
    print(f"[Config] HuggingFace Cache: {HF_HOME}")


def validate_config():
    """验证关键配置项，返回警告列表"""
    warnings = []
    
    if not LLM_MODEL_NAME:
        warnings.append("LLM_MODEL_NAME not set")
    if not ASR_MODEL_NAME:
        warnings.append("ASR_MODEL_NAME not set")
    if not HF_HOME:
        warnings.append("HF_HOME not set, using default cache location")
    
    return warnings


def get_config_summary() -> dict:
    """获取配置摘要，用于日志记录"""
    return {
        "device": DEVICE,
        "llm_model": LLM_MODEL_NAME,
        "asr_model": ASR_MODEL_NAME,
        "sample_rate": TARGET_SAMPLE_RATE,
        "log_level": LOG_LEVEL,
        "results_dir": RESULTS_DIR,
    }
