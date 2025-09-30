"""
测试流式音频分段器的并发处理能力
演示一个分段器实例同时处理多个音频流
"""

import numpy as np
import soundfile as sf
from pathlib import Path
import logging
from streamaudio_segmenter import StreamAudioSegmenter, StreamState
from typing import Dict, List
import time

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def simulate_concurrent_streams(audio_files: List[str], chunk_duration_ms: int = 100):
    """
    模拟并发处理多个音频流
    
    参数:
        audio_files: 音频文件路径列表
        chunk_duration_ms: 每个块的时长（毫秒）
    """
    logger.info(f"Starting concurrent processing of {len(audio_files)} audio streams")
    
    # 创建单个分段器实例
    segmenter = StreamAudioSegmenter(
        sampling_rate=16000,
        silence_threshold=0.5,
        min_speech_duration_ms=250,
        min_silence_duration_ms=300
    )
    
    # 为每个音频流创建独立的状态
    streams: Dict[str, dict] = {}
    
    # 加载所有音频文件
    for audio_path in audio_files:
        logger.info(f"Loading audio: {audio_path}")
        audio_data, sample_rate = sf.read(audio_path, dtype='float32')
        
        # 如果是立体声，转换为单声道
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        
        # 计算块大小
        chunk_size = int(sample_rate * chunk_duration_ms / 1000)
        
        # 将音频分块
        chunks = []
        for i in range(0, len(audio_data), chunk_size):
            chunks.append(audio_data[i:i+chunk_size])
        
        stream_id = Path(audio_path).stem
        streams[stream_id] = {
            'state': segmenter.create_state(),
            'chunks': chunks,
            'current_chunk': 0,
            'segments': [],
            'sample_rate': sample_rate,
            'total_samples': len(audio_data)
        }
        logger.info(f"  Stream '{stream_id}': {len(chunks)} chunks, {len(audio_data)/sample_rate:.2f}s")
    
    # 模拟并发处理
    logger.info("\n" + "="*50)
    logger.info("Starting concurrent processing...")
    
    # 继续处理直到所有流都完成
    all_done = False
    iteration = 0
    
    while not all_done:
        iteration += 1
        active_streams = []
        
        # 处理每个流的下一个块
        for stream_id, stream_data in streams.items():
            if stream_data['current_chunk'] < len(stream_data['chunks']):
                active_streams.append(stream_id)
                
                # 获取当前块
                chunk = stream_data['chunks'][stream_data['current_chunk']]
                
                # 处理音频块
                segment, new_state, metadata = segmenter.process_audio(
                    chunk, 
                    stream_data['state']
                )
                
                # 更新状态
                stream_data['state'] = new_state
                stream_data['current_chunk'] += 1
                
                # 如果检测到语音段
                if segment is not None:
                    stream_data['segments'].append(segment)
                    duration = len(segment) / stream_data['sample_rate']
                    logger.info(f"  [{stream_id}] Segment {len(stream_data['segments'])} detected: {duration:.2f}s")
        
        # 检查是否所有流都处理完成
        if not active_streams:
            all_done = True
        elif iteration % 10 == 0:
            logger.info(f"Iteration {iteration}: Processing {len(active_streams)} active streams")
    
    # Flush所有流的剩余数据
    logger.info("\nFlushing remaining data...")
    for stream_id, stream_data in streams.items():
        remaining, final_state = segmenter.flush(stream_data['state'])
        if remaining is not None and len(remaining) > 0:
            stream_data['segments'].append(remaining)
            duration = len(remaining) / stream_data['sample_rate']
            logger.info(f"  [{stream_id}] Flushed remaining: {duration:.2f}s")
        stream_data['state'] = final_state
    
    # 分析结果
    logger.info("\n" + "="*50)
    logger.info("Concurrent Processing Results:")
    
    for stream_id, stream_data in streams.items():
        total_segments = len(stream_data['segments'])
        
        # 合并所有段
        if stream_data['segments']:
            merged = np.concatenate(stream_data['segments'])
            merged_duration = len(merged) / stream_data['sample_rate']
            original_duration = stream_data['total_samples'] / stream_data['sample_rate']
            sample_diff = abs(stream_data['total_samples'] - len(merged))
            
            logger.info(f"\n[{stream_id}]:")
            logger.info(f"  Total segments: {total_segments}")
            logger.info(f"  Original duration: {original_duration:.3f}s")
            logger.info(f"  Merged duration: {merged_duration:.3f}s")
            logger.info(f"  Sample difference: {sample_diff}")
            
            # 验证准确性
            if sample_diff < stream_data['sample_rate'] * 0.1:  # 差异小于0.1秒
                logger.info(f"  ✅ Stream processed successfully!")
            else:
                logger.warning(f"  ⚠️ Large difference detected!")




if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试并发音频分段处理")
    parser.add_argument(
        "--audio-files", "-a",
        nargs='+',
        help="要并发处理的音频文件路径列表"
    )
    parser.add_argument(
        "--chunk-duration", "-c",
        type=int,
        default=100,
        help="块时长（毫秒）"
    )
    
    args = parser.parse_args()
    
    # 默认测试文件
    default_files = [
        "/usr/local/app/jupyterlab/yanjiu/streamllm/experiments/datasets/processed/experiments/length_analysis/audio/long/sample_001.wav",
    ]
    
    # 如果提供了多个文件，使用第一个文件复制多份来模拟
    if not args.audio_files:
        # 使用同一个文件模拟3个并发流
        if Path(default_files[0]).exists():
            args.audio_files = default_files * 3
            logger.info("Using default file to simulate 3 concurrent streams")
        else:
            logger.error(f"Default file not found: {default_files[0]}")
            exit(1)
    
    # 测试并发处理
    try:
        simulate_concurrent_streams(args.audio_files, args.chunk_duration)
        logger.info("\n✅ All tests completed successfully!")
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        exit(1)