"""
流式音频分段器测试程序
测试分段器功能并验证分段后的音频合并是否与原始音频一致
"""

import sys
import numpy as np
import soundfile as sf
from pathlib import Path
import logging
import torch
from streamaudio_segmenter import StreamAudioSegmenter

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def simulate_streaming(audio_data, chunk_size):
    """
    模拟流式音频输入
    
    参数:
        audio_data: 完整的音频数据
        chunk_size: 每次输入的块大小（采样点数）
    
    返回:
        音频块列表
    """
    chunks = []
    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i:i+chunk_size]
        chunks.append(chunk)
    return chunks


def test_segmenter(audio_path, chunk_duration_ms=100):
    """
    测试流式音频分段器
    
    参数:
        audio_path: 音频文件路径
        chunk_duration_ms: 模拟流式输入的块时长（毫秒）
    """
    logger.info(f"Loading audio file: {audio_path}")
    
    # 加载音频文件
    audio_data, sample_rate = sf.read(audio_path, dtype='float32')
    logger.info(f"Audio loaded: shape={audio_data.shape}, sample_rate={sample_rate}, duration={len(audio_data)/sample_rate:.2f}s")
    
    # 如果是立体声，转换为单声道
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
        logger.info("Converted stereo to mono")
    
    # 创建分段器
    segmenter = StreamAudioSegmenter(
        sampling_rate=sample_rate,
        silence_threshold=0.5,
        min_speech_duration_ms=250,
        min_silence_duration_ms=300,
        window_size_ms=64
    )
    
    # 计算块大小
    chunk_size = int(sample_rate * chunk_duration_ms / 1000)
    logger.info(f"Chunk size: {chunk_size} samples ({chunk_duration_ms}ms)")
    
    # 模拟流式输入
    chunks = simulate_streaming(audio_data, chunk_size)
    logger.info(f"Total chunks: {len(chunks)}")
    
    # 处理每个块
    segments = []
    all_accumulated = []
    
    for i, chunk in enumerate(chunks):
        # 处理音频块
        segment, accumulated, metadata = segmenter.process_audio(chunk)
        
        # 记录累积的音频
        all_accumulated.append(accumulated.copy())
        
        # 如果检测到完整段
        if segment is not None:
            segments.append(segment)
            logger.info(f"Chunk {i+1}/{len(chunks)}: Segment detected! Duration: {metadata.get('segment_duration_s', 0):.2f}s")
        else:
            if i % 10 == 0:  # 每10个块打印一次状态
                logger.debug(f"Chunk {i+1}/{len(chunks)}: Processing... "
                           f"Accumulated: {metadata.get('accumulated_duration_s', 0):.2f}s, "
                           f"Speaking: {metadata.get('is_speaking', False)}")
    
    # 处理剩余数据
    remaining = segmenter.flush()
    if remaining is not None and len(remaining) > 0:
        segments.append(remaining)
        logger.info(f"Flushed remaining audio: {len(remaining)/sample_rate:.2f}s")
    
    # 分析结果
    logger.info("\n" + "="*50)
    logger.info(f"Segmentation Results:")
    logger.info(f"Total segments: {len(segments)}")
    
    # 打印每个段的信息
    total_samples = 0
    for i, seg in enumerate(segments):
        duration = len(seg) / sample_rate
        total_samples += len(seg)
        logger.info(f"  Segment {i+1}: {duration:.2f}s ({len(seg)} samples)")
    
    # 合并所有段
    if segments:
        merged_audio = np.concatenate(segments)
        merged_duration = len(merged_audio) / sample_rate
        original_duration = len(audio_data) / sample_rate
        
        logger.info("\n" + "="*50)
        logger.info(f"Validation Results:")
        logger.info(f"Original audio: {original_duration:.3f}s ({len(audio_data)} samples)")
        logger.info(f"Merged segments: {merged_duration:.3f}s ({len(merged_audio)} samples)")
        logger.info(f"Sample difference: {abs(len(audio_data) - len(merged_audio))} samples")
        logger.info(f"Duration difference: {abs(original_duration - merged_duration):.3f}s")
        
        # 计算音频相似度
        min_len = min(len(audio_data), len(merged_audio))
        if min_len > 0:
            # 计算均方误差
            mse = np.mean((audio_data[:min_len] - merged_audio[:min_len]) ** 2)
            # 计算相关系数
            correlation = np.corrcoef(audio_data[:min_len], merged_audio[:min_len])[0, 1]
            
            logger.info(f"MSE (for overlapping part): {mse:.6f}")
            logger.info(f"Correlation: {correlation:.6f}")
            
            # 保存合并后的音频用于对比
            output_path = Path(audio_path).parent / f"{Path(audio_path).stem}_merged.wav"
            sf.write(output_path, merged_audio, sample_rate)
            logger.info(f"\nMerged audio saved to: {output_path}")
            
            # 判断是否一致
            if abs(len(audio_data) - len(merged_audio)) < sample_rate * 0.1:  # 差异小于0.1秒
                if correlation > 0.99:
                    logger.info("\n✅ Test PASSED: Merged audio is highly consistent with original!")
                else:
                    logger.warning(f"\n⚠️ Test WARNING: Sample count similar but correlation is low ({correlation:.4f})")
            else:
                logger.error(f"\n❌ Test FAILED: Significant difference in audio length!")
        else:
            logger.error("Cannot calculate similarity: no overlapping samples")
    else:
        logger.warning("No segments detected!")
    
    return segments


if __name__ == "__main__":
    # 默认测试文件
    default_audio = "/usr/local/app/jupyterlab/yanjiu/streamllm/experiments/datasets/processed/experiments/length_analysis/audio/long/sample_001.wav"
    
    # 从命令行参数获取音频路径
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
    else:
        audio_path = default_audio
        logger.info(f"No audio file specified, using default: {audio_path}")
    
    # 检查文件是否存在
    if not Path(audio_path).exists():
        logger.error(f"Audio file not found: {audio_path}")
        sys.exit(1)
    
    # 设置块时长（毫秒）
    chunk_duration = 100  # 100ms chunks
    if len(sys.argv) > 2:
        chunk_duration = int(sys.argv[2])
    
    # 运行测试
    try:
        segments = test_segmenter(audio_path, chunk_duration)
        logger.info("\nTest completed successfully!")
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        sys.exit(1)
