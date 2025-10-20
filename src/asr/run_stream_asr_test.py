#!/usr/bin/env python3
"""
ASR测试程序
支持流式ASR和完整音频ASR两种测试模式，结合StreamAudioSegmenter和FasterWhisperStreamer
实现流式音频分段和转录，以及完整音频的一次性转录
"""

import argparse
import numpy as np
import soundfile as sf
from pathlib import Path
import logging
import time
import json
import os
from typing import List, Dict, Tuple, Optional

# 首先导入配置，确保环境变量在其他模块导入之前设置好
import src.config

# 导入流式音频分段器
from src.asr.streamaudio_segmenter import StreamAudioSegmenter, StreamState, AudioSegment

# 导入流式ASR处理器
from src.asr.faster_whisper_streamer import StreamingASRProcessor, ASRCache, ASRAudioSegment

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def simulate_streaming(audio_data: np.ndarray, chunk_size: int) -> List[np.ndarray]:
    """
    模拟流式音频输入
    
    Args:
        audio_data: 完整的音频数据
        chunk_size: 每次输入的块大小（采样点数）
        
    Returns:
        List[np.ndarray]: 音频块列表
    """
    chunks = []
    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i:i+chunk_size]
        chunks.append(chunk)
    return chunks


def convert_audio_segment(stream_segment: AudioSegment, segment_id: str, 
                         is_start: bool = False, is_final: bool = False) -> ASRAudioSegment:
    """
    将StreamAudioSegmenter的AudioSegment转换为FasterWhisperStreamer的ASRAudioSegment
    
    Args:
        stream_segment: StreamAudioSegmenter的AudioSegment
        segment_id: 段ID
        is_start: 是否为开始段
        is_final: 是否为结束段
        
    Returns:
        ASRAudioSegment: 转换后的音频段
    """
    return ASRAudioSegment(
        id=segment_id,
        audio_data=stream_segment.audio,
        start_time=stream_segment.abs_start_time,
        end_time=stream_segment.abs_end_time,
        duration=stream_segment.segment_duration_s,
        is_start=is_start,
        is_final=is_final
    )


def test_complete_asr(audio_path: str, save_results: bool = True, results_dir: str = "experiments/results",
                     model_size: str = "tiny", device: str = "cuda") -> Dict:
    """
    测试完整音频ASR系统
    
    Args:
        audio_path: 音频文件路径
        save_results: 是否保存结果
        results_dir: 结果保存目录
        model_size: Whisper模型大小
        device: 计算设备 ("cuda" 或 "cpu")
        
    Returns:
        Dict: 测试结果
    """
    
    logger.info(f"Loading audio file: {audio_path}")
    
    # 加载音频文件获取基本信息
    audio_data, sample_rate = sf.read(audio_path, dtype='float32')
    audio_duration = len(audio_data) / sample_rate
    logger.info(f"Audio loaded: shape={audio_data.shape}, sample_rate={sample_rate}, duration={audio_duration:.2f}s")
    
    # 如果是立体声，转换为单声道
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
        logger.info("Converted stereo to mono")
    
    # 创建流式ASR处理器
    # 根据设备选择合适的计算类型
    compute_type = "float16" if device == "cuda" else "int8"
    
    asr_processor = StreamingASRProcessor(
        model_size=model_size,  # 使用指定的模型
        device=device,
        compute_type=compute_type,
        recognition_threshold=2.0,  # 2秒阈值
        prefix_segments=1
    )
    
    logger.info(f"ASR处理器初始化完成: 模型={model_size}, 设备={device}, 计算类型={compute_type}")
    
    # 初始化结果记录
    results = {
        'test_mode': 'complete',
        'audio_file': audio_path,
        'audio_duration': audio_duration,
        'sample_rate': sample_rate,
        'model_size': model_size,
        'device': device,
        'compute_type': compute_type,
        'transcriptions': [],
        'total_processing_time': 0.0,
        'performance_metrics': {}
    }
    
    # 记录开始时间
    start_time = time.time()
    
    # 使用完整音频转录方法
    try:
        complete_result = asr_processor.transcribe_complete_audio(audio_path)
    except Exception as e:
        if "cuDNN" in str(e) and ("NOT_INITIALIZED" in str(e) or "STATUS_NOT_INITIALIZED" in str(e)):
            logger.error(f"处理完整音频时发生 cuDNN 错误: {e}")
            logger.error("程序终止")
            raise e
        else:
            logger.error(f"处理完整音频时发生错误: {e}")
            raise e
    
    # 计算总处理时间
    total_processing_time = time.time() - start_time
    results['total_processing_time'] = total_processing_time
    results['real_time_factor'] = total_processing_time / audio_duration if audio_duration > 0 else 0
    
    # 记录转录结果
    results['full_transcription'] = complete_result['text']
    results['transcription_result'] = complete_result
    
    # 记录每个段的信息
    for i, segment in enumerate(complete_result['segments']):
        results['transcriptions'].append({
            'segment_id': f"seg_{i+1:03d}",
            'text': segment['text'],
            'start_time': segment['start'],
            'end_time': segment['end'],
            'timestamp': segment['start']  # 使用开始时间作为时间戳
        })
    
    # 打印结果摘要
    logger.info("\n" + "="*60)
    logger.info("COMPLETE ASR TEST RESULTS")
    logger.info("="*60)
    logger.info(f"Audio file: {audio_path}")
    logger.info(f"Audio duration: {audio_duration:.2f}s")
    logger.info(f"Model size: {model_size}")
    logger.info(f"Total processing time: {total_processing_time:.2f}s")
    logger.info(f"Real-time factor: {results['real_time_factor']:.2f}")
    logger.info(f"Full transcription: '{results['full_transcription']}'")
    
    # 保存结果
    if save_results:
        save_complete_test_results(results, results_dir, audio_path)
    
    return results


def test_streaming_asr(audio_path: str, chunk_duration_ms: int = 100,
                      save_results: bool = True, results_dir: str = "experiments/results",
                      model_size: str = "tiny", device: str = "cuda") -> Dict:
    """
    测试流式ASR系统
    
    Args:
        audio_path: 音频文件路径
        chunk_duration_ms: 模拟流式输入的块时长（毫秒）
        save_results: 是否保存结果
        results_dir: 结果保存目录
        model_size: Whisper模型大小
        device: 计算设备 ("cuda" 或 "cpu")
        
    Returns:
        Dict: 测试结果
    """
    
    logger.info(f"Loading audio file: {audio_path}")
    
    # 加载音频文件
    audio_data, sample_rate = sf.read(audio_path, dtype='float32')
    audio_duration = len(audio_data) / sample_rate
    logger.info(f"Audio loaded: shape={audio_data.shape}, sample_rate={sample_rate}, duration={audio_duration:.2f}s")
    
    # 如果是立体声，转换为单声道
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
        logger.info("Converted stereo to mono")
    
    # 创建流式音频分段器
    segmenter = StreamAudioSegmenter(
        sampling_rate=sample_rate,
        silence_threshold=0.5,
        min_speech_duration_ms=250,
        min_silence_duration_ms=300,
        window_size_ms=64
    )
    
    # 创建流式ASR处理器
    # 根据设备选择合适的计算类型
    compute_type = "float16" if device == "cuda" else "int8"
    
    asr_processor = StreamingASRProcessor(
        model_size=model_size,  # 使用指定的模型
        device=device,
        compute_type=compute_type,
        recognition_threshold=2.0,  # 2秒阈值
        prefix_segments=1
    )
    
    logger.info(f"ASR处理器初始化完成: 模型={model_size}, 设备={device}, 计算类型={compute_type}")
    
    # 计算块大小
    chunk_size = int(sample_rate * chunk_duration_ms / 1000)
    logger.info(f"Chunk size: {chunk_size} samples ({chunk_duration_ms}ms)")
    
    # 模拟流式输入
    chunks = simulate_streaming(audio_data, chunk_size)
    logger.info(f"Total chunks: {len(chunks)}")
    
    # 创建状态和缓存
    segmenter_state = segmenter.create_state()
    asr_cache = ASRCache()
    
    # 初始化结果记录
    results = {
        'test_mode': 'streaming',
        'audio_file': audio_path,
        'audio_duration': audio_duration,
        'sample_rate': sample_rate,
        'chunk_duration_ms': chunk_duration_ms,
        'model_size': model_size,
        'device': device,
        'compute_type': compute_type,
        'segments': [],
        'transcriptions': [],
        'total_processing_time': 0.0,
        'performance_metrics': {}
    }
    
    # 处理每个音频块
    segment_count = 0
    transcription_text = ""
    start_time = time.time()
    
    for i, chunk in enumerate(chunks):
        logger.debug(f"Processing chunk {i+1}/{len(chunks)}")
        
        # 使用分段器处理音频块
        stream_segment, segmenter_state = segmenter.process_audio(chunk, segmenter_state)
        
        # 如果检测到音频段，转换为ASR音频段并处理
        if stream_segment is not None:
            segment_count += 1
            segment_id = f"seg_{segment_count:03d}"
            
            # 判断是否为开始或结束段
            is_start = (segment_count == 1)
            is_final = (i == len(chunks) - 1 and len(segmenter_state.accumulated_audio) == 0)
            
            # 转换音频段格式
            asr_segment = convert_audio_segment(
                stream_segment, segment_id, is_start, is_final
            )
            
            logger.info(f"Segment {segment_count}: [{asr_segment.start_time:.2f}s-{asr_segment.end_time:.2f}s] "
                       f"Duration: {asr_segment.duration:.2f}s")
            
            # 使用ASR处理器处理音频段
            try:
                asr_cache, output_text = asr_processor.transcribe_audio_segment(asr_cache, asr_segment)
            except Exception as e:
                if "cuDNN" in str(e) and ("NOT_INITIALIZED" in str(e) or "STATUS_NOT_INITIALIZED" in str(e)):
                    logger.error(f"处理音频段 {segment_id} 时发生 cuDNN 错误: {e}")
                    logger.error("程序终止")
                    raise e
                else:
                    logger.error(f"处理音频段 {segment_id} 时发生错误: {e}")
                    raise e
            
            # 如果有输出文本，更新转录结果
            if output_text:
                transcription_text += output_text + " "
                logger.info(f"  Transcription: '{output_text}'")
                
                # 记录转录结果
                results['transcriptions'].append({
                    'segment_id': segment_id,
                    'text': output_text,
                    'start_time': asr_segment.start_time,
                    'end_time': asr_segment.end_time,
                    'timestamp': time.time() - start_time
                })
            
            # 记录段信息
            results['segments'].append({
                'segment_id': segment_id,
                'start_time': asr_segment.start_time,
                'end_time': asr_segment.end_time,
                'duration': asr_segment.duration,
                'is_start': is_start,
                'is_final': is_final,
                'text': output_text
            })
    
    # 处理剩余的音频数据
    logger.info("Processing remaining audio data...")
    remaining_segment, segmenter_state = segmenter.flush(segmenter_state)
    
    if remaining_segment is not None and len(remaining_segment.audio) > 0:
        segment_count += 1
        segment_id = f"seg_{segment_count:03d}"
        
        # 转换音频段格式
        asr_segment = convert_audio_segment(remaining_segment, segment_id, False, True)
        
        logger.info(f"Final segment: [{asr_segment.start_time:.2f}s-{asr_segment.end_time:.2f}s] "
                   f"Duration: {asr_segment.duration:.2f}s")
        
        # 使用ASR处理器处理音频段
        try:
            asr_cache, output_text = asr_processor.transcribe_audio_segment(asr_cache, asr_segment)
        except Exception as e:
            if "cuDNN" in str(e) and ("NOT_INITIALIZED" in str(e) or "STATUS_NOT_INITIALIZED" in str(e)):
                logger.error(f"处理最终音频段 {segment_id} 时发生 cuDNN 错误: {e}")
                logger.error("程序终止")
                raise e
            else:
                logger.error(f"处理最终音频段 {segment_id} 时发生错误: {e}")
                raise e
        
        # 如果有输出文本，更新转录结果
        if output_text:
            transcription_text += output_text
            logger.info(f"  Transcription: '{output_text}'")
            
            # 记录转录结果
            results['transcriptions'].append({
                'segment_id': segment_id,
                'text': output_text,
                'start_time': asr_segment.start_time,
                'end_time': asr_segment.end_time,
                'timestamp': time.time() - start_time
            })
        
        # 记录段信息
        results['segments'].append({
            'segment_id': segment_id,
            'start_time': asr_segment.start_time,
            'end_time': asr_segment.end_time,
            'duration': asr_segment.duration,
            'is_start': False,
            'is_final': True,
            'text': output_text
        })
    
    # 计算总处理时间
    total_processing_time = time.time() - start_time
    results['total_processing_time'] = total_processing_time
    results['real_time_factor'] = total_processing_time / audio_duration if audio_duration > 0 else 0
    
    # 获取性能指标
    performance_metrics = asr_processor.get_performance_metrics()
    results['performance_metrics'] = performance_metrics
    
    # 完整转录文本
    results['full_transcription'] = transcription_text.strip()
    
    # 打印结果摘要
    logger.info("\n" + "="*60)
    logger.info("STREAMING ASR TEST RESULTS")
    logger.info("="*60)
    logger.info(f"Audio file: {audio_path}")
    logger.info(f"Audio duration: {audio_duration:.2f}s")
    logger.info(f"Model size: {model_size}")
    logger.info(f"Total segments: {segment_count}")
    logger.info(f"Total processing time: {total_processing_time:.2f}s")
    logger.info(f"Real-time factor: {results['real_time_factor']:.2f}")
    logger.info(f"Full transcription: '{results['full_transcription']}'")
    
    # 保存结果
    if save_results:
        save_test_results(results, results_dir, audio_path)
    
    return results


def save_complete_test_results(results: Dict, results_dir: str, audio_path: str) -> None:
    """
    保存完整音频ASR测试结果
    
    Args:
        results: 测试结果
        results_dir: 结果保存目录
        audio_path: 音频文件路径
    """
    # 创建结果目录
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    # 创建子目录用于保存当前测试的结果
    audio_stem = Path(audio_path).stem
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    test_dir = results_path / f"complete_asr_test_{audio_stem}_{timestamp}"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving results to: {test_dir}")
    
    # 保存JSON格式的结果
    json_path = test_dir / "results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"  Saved results to: {json_path}")
    
    # 保存转录文本
    transcription_path = test_dir / "transcription.txt"
    with open(transcription_path, 'w', encoding='utf-8') as f:
        f.write(results['full_transcription'])
    logger.info(f"  Saved transcription to: {transcription_path}")
    
    # 保存段信息
    segments_path = test_dir / "segments.csv"
    with open(segments_path, 'w', encoding='utf-8') as f:
        f.write("segment_id,text,start_time,end_time,timestamp\n")
        for transcription in results['transcriptions']:
            f.write(f"{transcription['segment_id']},\"{transcription['text']}\","
                   f"{transcription['start_time']:.3f},{transcription['end_time']:.3f},"
                   f"{transcription['timestamp']:.3f}\n")
    logger.info(f"  Saved segments to: {segments_path}")


def save_test_results(results: Dict, results_dir: str, audio_path: str) -> None:
    """
    保存测试结果
    
    Args:
        results: 测试结果
        results_dir: 结果保存目录
        audio_path: 音频文件路径
    """
    # 创建结果目录
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    # 创建子目录用于保存当前测试的结果
    audio_stem = Path(audio_path).stem
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    test_dir = results_path / f"stream_asr_test_{audio_stem}_{timestamp}"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving results to: {test_dir}")
    
    # 保存JSON格式的结果
    json_path = test_dir / "results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"  Saved results to: {json_path}")
    
    # 保存转录文本
    transcription_path = test_dir / "transcription.txt"
    with open(transcription_path, 'w', encoding='utf-8') as f:
        f.write(results['full_transcription'])
    logger.info(f"  Saved transcription to: {transcription_path}")
    
    # 保存段信息
    segments_path = test_dir / "segments.csv"
    with open(segments_path, 'w', encoding='utf-8') as f:
        f.write("segment_id,start_time,end_time,duration,is_start,is_final,text\n")
        for segment in results['segments']:
            f.write(f"{segment['segment_id']},{segment['start_time']:.3f},"
                   f"{segment['end_time']:.3f},{segment['duration']:.3f},"
                   f"{segment['is_start']},{segment['is_final']},"
                   f"\"{segment['text'] or ''}\"\n")
    logger.info(f"  Saved segments to: {segments_path}")
    
    # 保存转录时间线
    timeline_path = test_dir / "timeline.csv"
    with open(timeline_path, 'w', encoding='utf-8') as f:
        f.write("segment_id,text,start_time,end_time,timestamp\n")
        for transcription in results['transcriptions']:
            segment = next(s for s in results['segments'] if s['segment_id'] == transcription['segment_id'])
            f.write(f"{transcription['segment_id']},\"{transcription['text']}\","
                   f"{segment['start_time']:.3f},{segment['end_time']:.3f},"
                   f"{transcription['timestamp']:.3f}\n")
    logger.info(f"  Saved timeline to: {timeline_path}")


if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser(
        description="ASR测试程序，支持流式ASR和完整音频ASR两种测试模式",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 添加命令行参数
    parser.add_argument(
        "--audio", "-a",
        type=str,
        default="experiments/datasets/processed/experiments/length_analysis/audio/short/sample_001.wav",
        help="音频文件路径"
    )
    
    parser.add_argument(
        "--chunk-duration", "-c",
        type=int,
        default=100,
        help="模拟流式输入的块时长（毫秒）"
    )
    
    parser.add_argument(
        "--save-results", "-s",
        action="store_true",
        default=True,
        help="是否保存测试结果"
    )
    
    parser.add_argument(
        "--results-dir", "-r",
        type=str,
        default="experiments/results",
        help="结果保存目录"
    )
    
    parser.add_argument(
        "--model-size", "-m",
        type=str,
        default="tiny",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper模型大小"
    )
    
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="计算设备 (cuda 或 cpu)"
    )
    
    parser.add_argument(
        "--test-mode", "-t",
        type=str,
        default="streaming",
        choices=["streaming", "complete", "both"],
        help="测试模式: streaming(流式ASR), complete(完整音频ASR), both(两种都测试)"
    )
    
    # 解析参数
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not Path(args.audio).exists():
        logger.error(f"Audio file not found: {args.audio}")
        parser.print_help()
        exit(1)
    
    # 打印运行参数
    logger.info("="*60)
    if args.test_mode == "streaming":
        logger.info("STREAMING ASR TEST CONFIGURATION")
    elif args.test_mode == "complete":
        logger.info("COMPLETE ASR TEST CONFIGURATION")
    else:
        logger.info("ASR TEST CONFIGURATION (STREAMING + COMPLETE)")
    logger.info("="*60)
    logger.info(f"Audio file: {args.audio}")
    logger.info(f"Test mode: {args.test_mode}")
    if args.test_mode in ["streaming", "both"]:
        logger.info(f"Chunk duration: {args.chunk_duration}ms")
    logger.info(f"Model size: {args.model_size}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Save results: {args.save_results}")
    logger.info(f"Results directory: {args.results_dir}")
    logger.info("="*60)
    
    # 运行测试
    try:
        if args.test_mode == "streaming":
            # 只运行流式ASR测试
            results = test_streaming_asr(
                audio_path=args.audio,
                chunk_duration_ms=args.chunk_duration,
                save_results=args.save_results,
                results_dir=args.results_dir,
                model_size=args.model_size,
                device=args.device
            )
            logger.info("\nStreaming ASR test completed successfully!")
            
        elif args.test_mode == "complete":
            # 只运行完整音频ASR测试
            results = test_complete_asr(
                audio_path=args.audio,
                save_results=args.save_results,
                results_dir=args.results_dir,
                model_size=args.model_size,
                device=args.device
            )
            logger.info("\nComplete ASR test completed successfully!")
            
        elif args.test_mode == "both":
            # 运行两种测试并比较结果
            logger.info("Running streaming ASR test...")
            streaming_results = test_streaming_asr(
                audio_path=args.audio,
                chunk_duration_ms=args.chunk_duration,
                save_results=args.save_results,
                results_dir=args.results_dir,
                model_size=args.model_size,
                device=args.device
            )
            
            logger.info("\nRunning complete ASR test...")
            complete_results = test_complete_asr(
                audio_path=args.audio,
                save_results=args.save_results,
                results_dir=args.results_dir,
                model_size=args.model_size,
                device=args.device
            )
            
            # 比较结果
            logger.info("\n" + "="*60)
            logger.info("COMPARISON RESULTS")
            logger.info("="*60)
            logger.info(f"Streaming ASR - First text time: {streaming_results['transcriptions'][0]['timestamp']:.3f}s")
            logger.info(f"Complete ASR - Total processing time: {complete_results['total_processing_time']:.3f}s")
            logger.info(f"Streaming ASR - Total processing time: {streaming_results['total_processing_time']:.3f}s")
            logger.info(f"Complete ASR - Real-time factor: {complete_results['real_time_factor']:.2f}")
            logger.info(f"Streaming ASR - Real-time factor: {streaming_results['real_time_factor']:.2f}")
            
            # 比较转录文本
            logger.info(f"Streaming ASR transcription: '{streaming_results['full_transcription']}'")
            logger.info(f"Complete ASR transcription: '{complete_results['full_transcription']}'")
            
            logger.info("\nBoth tests completed successfully!")
            
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        exit(1)