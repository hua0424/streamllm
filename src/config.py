import os
from dotenv import load_dotenv

load_dotenv() # 加载 .env 文件中的环境变量

# --- 模型配置 ---
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")
ASR_MODEL_NAME = os.getenv("ASR_MODEL_NAME") # faster-whisper 模型大小

# Hugging Face 配置 (如果需要从特定端点或使用token)
HF_ENDPOINT = os.getenv("HF_ENDPOINT")
HF_TOKEN = os.getenv("HF_TOKEN")
HF_HOME = os.getenv("HF_HOME") # 默认缓存路径

# 设备配置
DEVICE = "cpu" # 或者 "cpu"

# --- 音频目标格式配置 ---
TARGET_SAMPLE_RATE = int(os.getenv("TARGET_SAMPLE_RATE", "16000")) # 目标采样率 (Hz)
TARGET_CHANNELS = int(os.getenv("TARGET_CHANNELS", "1"))          # 目标通道数 (1 for mono)

# --- ASR 流式处理配置 ---\n# VAD (Voice Activity Detection) 相关参数，用于faster-whisper的vad_filter
VAD_PARAMETERS = {
    "threshold": 0.5, # VAD 阈值
    "min_speech_duration_ms": 250, # 最短语音活动时间 (毫秒)
    "max_speech_duration_s": float('inf'), # 最大语音活动时间 (秒)
    "min_silence_duration_ms": 1000, # 用于分段的最小静音持续时间 (毫秒)
    "speech_pad_ms": 200 # 在检测到的语音片段两端添加的填充 (毫秒)
}

# 流式ASR中，每次送入faster-whisper的音频块时长 (秒) - 需要实验调整
STREAMING_ASR_CHUNK_SECONDS = 10
# 流式ASR中，重叠的音频块时长 (秒) - 需要实验调整
STREAMING_ASR_OVERLAP_SECONDS = 2

# --- LLM 流式处理配置 ---\n# KV缓存更新策略等 (待定)

# --- 实验数据路径 ---
DATA_DIR = os.getenv("DATA_DIR", "data")
RAW_AUDIO_DIR = os.getenv("RAW_AUDIO_DIR", "data/raw_audio")  # 原始音频目录独立配置
PROCESSED_AUDIO_DIR = os.getenv("PROCESSED_AUDIO_DIR", "data/processed_audio")
TRANSCRIPTS_DIR = os.getenv("TRANSCRIPTS_DIR", "data/transcripts")

# --- 实验结果路径 ---
RESULTS_DIR = os.getenv("RESULTS_DIR", "results")
LATENCY_METRICS_DIR = os.getenv("LATENCY_METRICS_DIR", "results/latency_metrics")
WER_SCORES_DIR = os.getenv("WER_SCORES_DIR", "results/wer_scores")

# --- 日志配置 ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO") # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE = os.getenv("LOG_FILE", "logs/experiment.log") # 日志文件路径 (如果需要保存到文件)

# 确保缓存目录存在
if not os.path.exists(HF_HOME):
    os.makedirs(HF_HOME, exist_ok=True)

print(f"LLM Model: {LLM_MODEL_NAME}")
print(f"ASR Model: {ASR_MODEL_NAME}")
print(f"Device: {DEVICE}")
print(f"HuggingFace Cache Home: {HF_HOME}") 
print(f"HF_ENDPOINT: {HF_ENDPOINT}")