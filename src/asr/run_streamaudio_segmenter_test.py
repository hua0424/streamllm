"""
流式音频分段器测试程序
测试分段器功能并验证分段后的音频合并是否与原始音频一致
"""

import argparse
import numpy as np
import soundfile as sf
from pathlib import Path
import logging
import torch
from streamaudio_segmenter import StreamAudioSegmenter, StreamState

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


def test_segmenter(audio_path, chunk_duration_ms=100, save_segments=False, save_dir=None):
    """
    测试流式音频分段器
    
    参数:
        audio_path: 音频文件路径
        chunk_duration_ms: 模拟流式输入的块时长（毫秒）
        save_segments: 是否保存分段音频
        save_dir: 保存分段音频的目录
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
    
    # 创建流状态
    state = segmenter.create_state()
    
    for i, chunk in enumerate(chunks):
        # 处理音频块
        segment, state, metadata = segmenter.process_audio(chunk, state)
        
        # 记录累积的音频
        all_accumulated.append(state.accumulated_audio.copy())
        
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
    remaining, state = segmenter.flush(state)
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
    
    # 保存分段音频（如果需要）
    if save_segments and segments:
        # 创建保存目录
        if save_dir is None:
            save_dir = Path(__file__).parent / "test_result"
        else:
            save_dir = Path(save_dir)
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录用于保存当前测试的分段
        audio_stem = Path(audio_path).stem
        segment_dir = save_dir / audio_stem
        segment_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\nSaving segments to: {segment_dir}")
        for i, seg in enumerate(segments):
            segment_path = segment_dir / f"segment_{i+1:03d}.wav"
            sf.write(segment_path, seg, sample_rate)
            logger.info(f"  Saved: {segment_path.name}")
    
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
    # 创建参数解析器
    parser = argparse.ArgumentParser(
        description="流式音频分段器测试程序",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 添加命令行参数
    parser.add_argument(
        "--audio", "-a",
        type=str,
        default="/usr/local/app/jupyterlab/yanjiu/streamllm/experiments/datasets/processed/experiments/length_analysis/audio/long/sample_001.wav",
        help="音频文件路径"
    )
    
    parser.add_argument(
        "--chunk-duration", "-c",
        type=int,
        default=100,
        help="模拟流式输入的块时长（毫秒）"
    )
    
    parser.add_argument(
        "--save-segments", "-s",
        action="store_true",
        help="是否保存分段音频文件"
    )
    
    parser.add_argument(
        "--save-dir", "-d",
        type=str,
        default=None,
        help="保存分段音频的目录路径（默认为程序同级目录下的test_result）"
    )
    
    parser.add_argument(
        "--silence-threshold",
        type=float,
        default=0.5,
        help="VAD静音检测阈值（0-1之间）"
    )
    
    parser.add_argument(
        "--min-speech-duration",
        type=int,
        default=250,
        help="最小语音段时长（毫秒）"
    )
    
    parser.add_argument(
        "--min-silence-duration",
        type=int,
        default=300,
        help="最小静音时长（毫秒）"
    )
    
    # 解析参数
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not Path(args.audio).exists():
        logger.error(f"Audio file not found: {args.audio}")
        parser.print_help()
        exit(1)
    
    # 打印运行参数
    logger.info("="*50)
    logger.info("Test Configuration:")
    logger.info(f"  Audio file: {args.audio}")
    logger.info(f"  Chunk duration: {args.chunk_duration}ms")
    logger.info(f"  Save segments: {args.save_segments}")
    if args.save_segments:
        logger.info(f"  Save directory: {args.save_dir if args.save_dir else 'test_result/'}")
    logger.info(f"  Silence threshold: {args.silence_threshold}")
    logger.info(f"  Min speech duration: {args.min_speech_duration}ms")
    logger.info(f"  Min silence duration: {args.min_silence_duration}ms")
    logger.info("="*50)
    
    # 运行测试
    try:
        segments = test_segmenter(
            audio_path=args.audio,
            chunk_duration_ms=args.chunk_duration,
            save_segments=args.save_segments,
            save_dir=args.save_dir
        )
        logger.info("\nTest completed successfully!")
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        exit(1)
