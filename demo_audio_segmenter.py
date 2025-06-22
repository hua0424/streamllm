#!/usr/bin/env python3
"""
VAD音频分段器演示脚本
"""

import os
import sys
import numpy as np

# 添加src目录到路径
sys.path.append('src')

def main():
    print("VAD 音频分段器演示程序")
    print("文件已经重写完成，包含以下功能：")
    print()
    print("✅ VAD音频分段 - 使用Silero VAD进行语音活动检测")
    print("✅ 批量分段模式 - 对完整音频文件进行分段")
    print("✅ 流式分段模式 - 对音频流进行实时分段")  
    print("✅ 完整性保证 - 分割后的音频段可以完整重构原始文件")
    print("✅ 自动保存 - 分段结果保存到指定目录")
    print("✅ 完整性验证 - 验证分段结果的连续性")
    print()
    print("使用方法:")
    print("python src/asr/audio_segmenter.py --mode test")
    print("python src/asr/audio_segmenter.py --mode batch --audio your_audio.wav")
    print("python src/asr/audio_segmenter.py --mode streaming --audio your_audio.wav")
    print()
    print("输出目录: /home/project/streamllm/results/wav_segments")

if __name__ == "__main__":
    main()
