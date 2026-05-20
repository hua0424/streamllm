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
import threading
import queue
from concurrent.futures import ThreadPoolExecutor

# 首先导入配置，确保环境变量在其他模块导入之前设置好
import src.config

# 导入流式音频分段器
from src.asr.streamaudio_segmenter import StreamAudioSegmenter, StreamState, AudioSegment

# 导入流式ASR处理器
from src.asr.faster_whisper_streamer import StreamingASRProcessor, ASRCache, ASRAudioSegment

# 设置日志
from src.utils.logging_utils import get_logger
logger = get_logger(__name__)


class MultiThreadStreamingASR:
    """
    多线程流式ASR系统
    三个独立线程：音频生成、音频分段、转录处理
    """
    
    def __init__(self, audio_path: str, chunk_duration_ms: int = 100,
                 model_size: str = "tiny", device: str = "cuda",
                 simulate_streaming_delay: bool = False):
        """
        初始化多线程流式ASR系统
        
        Args:
            audio_path: 音频文件路径
            chunk_duration_ms: 音频块时长（毫秒）
            model_size: Whisper模型大小
            device: 计算设备
            simulate_streaming_delay: 是否模拟实际音频产生的延迟
        """
        self.audio_path = audio_path
        self.chunk_duration_ms = chunk_duration_ms
        self.model_size = model_size
        self.device = device
        self.simulate_streaming_delay = simulate_streaming_delay
        
        # 加载音频文件
        logger.info(f"Loading audio file: {audio_path}")
        self.audio_data, self.sample_rate = sf.read(audio_path, dtype='float32')
        self.audio_duration = len(self.audio_data) / self.sample_rate
        
        # 如果是立体声，转换为单声道
        if len(self.audio_data.shape) > 1:
            self.audio_data = self.audio_data.mean(axis=1)
            logger.info("Converted stereo to mono")
        
        logger.info(f"Audio loaded: shape={self.audio_data.shape}, sample_rate={self.sample_rate}, duration={self.audio_duration:.2f}s")
        
        # 计算块大小
        self.chunk_size = int(self.sample_rate * self.chunk_duration_ms / 1000)
        logger.info(f"Chunk size: {self.chunk_size} samples ({self.chunk_duration_ms}ms)")
        
        # 创建队列
        self.audio_chunk_queue = queue.Queue()  # 音频块队列
        self.audio_segment_queue = queue.Queue()  # 音频段队列
        self.transcription_queue = queue.Queue()  # 转录结果队列
        
        # 创建事件
        self.stop_event = threading.Event()
        self.audio_generation_complete = threading.Event()
        self.segmentation_complete = threading.Event()
        
        # 创建组件
        self.segmenter = StreamAudioSegmenter(
            sampling_rate=self.sample_rate,
            silence_threshold=0.5,
            min_speech_duration_ms=500,
            min_silence_duration_ms=300,
            window_size_ms=64
        )
        
        compute_type = "float16" if device == "cuda" else "int8"
        self.asr_processor = StreamingASRProcessor(
            model_size=model_size,
            device=device,
            compute_type=compute_type,
            recognition_threshold=1.0,
            prefix_segments=1
        )
        
        # 创建线程
        self.audio_generation_thread = None
        self.segmentation_thread = None
        self.transcription_thread = None
        
        # 结果
        self.results = {
            'test_mode': 'streaming',
            'audio_file': audio_path,
            'audio_duration': self.audio_duration,
            'model_size': model_size,
            'device': device,
            'full_transcription': "",
            'response_time': 0.0
        }
        
        # 时间记录
        self.start_time = 0
        self.last_chunk_time = 0
        self.transcription_complete_time = 0
    
    def start(self):
        """启动所有线程"""
        self.start_time = time.time()
        
        # 启动音频生成线程
        self.audio_generation_thread = threading.Thread(
            target=self._audio_generation_worker,
            name="AudioGeneration"
        )
        self.audio_generation_thread.start()
        
        # 启动音频分段线程
        self.segmentation_thread = threading.Thread(
            target=self._segmentation_worker,
            name="Segmentation"
        )
        self.segmentation_thread.start()
        
        # 启动转录线程
        self.transcription_thread = threading.Thread(
            target=self._transcription_worker,
            name="Transcription"
        )
        self.transcription_thread.start()
        
        logger.info("All threads started")
    
    def stop(self):
        """停止所有线程"""
        self.stop_event.set()
        
        # 等待所有线程完成
        if self.audio_generation_thread and self.audio_generation_thread.is_alive():
            self.audio_generation_thread.join()
        
        if self.segmentation_thread and self.segmentation_thread.is_alive():
            self.segmentation_thread.join()
        
        if self.transcription_thread and self.transcription_thread.is_alive():
            self.transcription_thread.join()
        
        logger.info("All threads stopped")
    
    def _audio_generation_worker(self):
        """音频生成工作线程"""
        logger.info("Audio generation thread started")
        
        try:
            # 计算总块数
            total_chunks = (len(self.audio_data) + self.chunk_size - 1) // self.chunk_size
            logger.info(f"Total chunks: {total_chunks}")
            
            for i in range(0, len(self.audio_data), self.chunk_size):
                if self.stop_event.is_set():
                    break
                    
                chunk = self.audio_data[i:i+self.chunk_size]
                logger.debug(f"Generated chunk {i//self.chunk_size + 1}/{total_chunks}")
                
                # 将音频块放入队列
                self.audio_chunk_queue.put((i//self.chunk_size, chunk))
                
                # 如果启用了延迟模拟，等待相应时间
                if self.simulate_streaming_delay and i + self.chunk_size < len(self.audio_data):
                    time.sleep(self.chunk_duration_ms / 1000)  # 转换为秒
                
                # 更新最后一个chunk处理完成的时间
                self.last_chunk_time = time.time()
            
            # 音频生成完成
            self.audio_generation_complete.set()
            logger.info("Audio generation completed")
            
        except Exception as e:
            logger.error(f"Error in audio generation thread: {e}")
    
    def _segmentation_worker(self):
        """音频分段工作线程"""
        logger.info("Segmentation thread started")
        
        try:
            # 创建分段器状态
            segmenter_state = self.segmenter.create_state()
            segments_emitted = 0  # 跟踪已输出的段数
            
            while not self.stop_event.is_set():
                try:
                    # 从队列获取音频块，设置超时以避免永久阻塞
                    chunk_id, chunk = self.audio_chunk_queue.get(timeout=0.1)
                    logger.debug(f"Processing chunk {chunk_id + 1} in segmentation thread")
                    
                    # 使用分段器处理音频块
                    stream_segment, segmenter_state = self.segmenter.process_audio(chunk, segmenter_state)
                    
                    # 如果检测到音频段，放入音频段队列
                    if stream_segment is not None:
                        segment_id = f"seg_{stream_segment.segment_id:03d}"
                        
                        # 判断是否为开始或结束段
                        is_start = (segments_emitted == 0)  # 第一个输出的段标记为 is_start
                        is_final = (self.audio_generation_complete.is_set() and
                                  self.audio_chunk_queue.empty() and
                                  len(segmenter_state.accumulated_audio) == 0)
                        
                        # 转换为ASR音频段
                        asr_segment = convert_audio_segment(
                            stream_segment, segment_id, is_start, is_final
                        )
                        
                        logger.debug(f"Generated segment {segment_id}: [{asr_segment.start_time:.2f}s-{asr_segment.end_time:.2f}s]")
                        
                        # 将音频段放入队列
                        self.audio_segment_queue.put(asr_segment)
                        segments_emitted += 1
                    
                    # 标记任务完成
                    self.audio_chunk_queue.task_done()
                    
                except queue.Empty:
                    # 队列为空，检查是否音频生成已完成
                    if self.audio_generation_complete.is_set() and self.audio_chunk_queue.empty():
                        break
                    continue
            
            # 处理剩余的音频数据
            logger.info("Processing remaining audio data...")
            remaining_segment, segmenter_state = self.segmenter.flush(segmenter_state)
            
            if remaining_segment is not None and len(remaining_segment.audio) > 0:
                segment_id = f"seg_{remaining_segment.segment_id:03d}"
                # 如果是第一个段，标记为 is_start
                is_start = (segments_emitted == 0)
                asr_segment = convert_audio_segment(remaining_segment, segment_id, is_start, True)
                
                logger.info(f"Generated final segment {segment_id}: [{asr_segment.start_time:.2f}s-{asr_segment.end_time:.2f}s]")
                
                # 将最终音频段放入队列
                self.audio_segment_queue.put(asr_segment)
            
            # 分段完成
            self.segmentation_complete.set()
            logger.info("Segmentation completed")
            
        except Exception as e:
            logger.error(f"Error in segmentation thread: {e}")
    
    def _transcription_worker(self):
        """转录工作线程"""
        logger.info("Transcription thread started")
        
        try:
            # 创建ASR缓存
            asr_cache = ASRCache()
            transcription_text = ""
            
            while not self.stop_event.is_set():
                try:
                    # 从队列获取音频段，设置超时以避免永久阻塞
                    asr_segment = self.audio_segment_queue.get(timeout=0.1)
                    logger.debug(f"Received segment {asr_segment.id} in transcription thread")
                    
                    # 将音频段添加到缓存
                    asr_cache.add_segment(asr_segment)
                    logger.debug(f"Added segment {asr_segment.id} to cache, queue length: {len(asr_cache.segment_queue)}")
                    
                    # 检查是否应该处理当前队列中的音频段
                    should_process = asr_cache.should_process(
                        self.asr_processor.recognition_threshold,
                        self.asr_processor.prefix_segments,
                        1,  # suffix_segments_atleast 参数
                        asr_segment.is_final
                    )
                    
                    # 如果应该处理且当前没有转录在进行中
                    if should_process:
                        logger.debug(f"Starting transcription for {len(asr_cache.segment_queue)} segments")
                        
                        # 使用ASR处理器处理音频段
                        asr_cache, output_text, _ = self.asr_processor.transcribe_audio_segment(asr_cache)
                        
                        # 如果有输出文本，更新转录结果
                        if output_text:
                            transcription_text += output_text + " "
                            logger.info(f"  Transcription: '{output_text}'")
                            
                            # 将转录结果放入队列
                            self.transcription_queue.put({
                                'segment_id': asr_segment.id,
                                'text': output_text,
                                'timestamp': time.time() - self.start_time
                            })
                        
                        logger.debug(f"Transcription completed, remaining segments: {len(asr_cache.segment_queue)}")
                    
                    # 标记任务完成
                    self.audio_segment_queue.task_done()
                    
                except queue.Empty:
                    # 队列为空，检查是否分段已完成
                    if self.segmentation_complete.is_set() and self.audio_segment_queue.empty():
                        break
                    continue
            
            # 记录转录完成时间
            self.transcription_complete_time = time.time()
            
            # 计算响应时间
            response_time = self.transcription_complete_time - self.last_chunk_time
            
            # 更新结果
            self.results['full_transcription'] = transcription_text.strip()
            self.results['response_time'] = response_time
            
            logger.info("Transcription completed")
            
        except Exception as e:
            logger.error(f"Error in transcription thread: {e}")
    
    def get_results(self) -> Dict:
        """获取测试结果"""
        return self.results


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
    
    # 创建流式ASR处理器
    # 根据设备选择合适的计算类型
    compute_type = "float16" if device == "cuda" else "int8"
    
    asr_processor = StreamingASRProcessor(
        model_size=model_size,  # 使用指定的模型
        device=device,
        compute_type=compute_type,
        recognition_threshold=1.0,  # 2秒阈值
        prefix_segments=1
    )
    
    logger.info(f"ASR处理器初始化完成: 模型={model_size}, 设备={device}, 计算类型={compute_type}")
    
    # 初始化结果记录
    results = {
        'test_mode': 'complete',
        'audio_file': audio_path,
        'model_size': model_size,
        'device': device,
        'full_transcription': "",
        'response_time': 0.0  # 初始化为float类型
    }
    
    # 记录开始时间（音频加载开始）
    start_time = time.time()
    
    # 记录开始时间（音频加载开始）
    start_time = time.time()
    
    logger.info(f"Loading audio file: {audio_path}")
    
    # 加载音频文件获取基本信息
    audio_data, sample_rate = sf.read(audio_path, dtype='float32')
    audio_duration = len(audio_data) / sample_rate
    logger.info(f"Audio loaded: shape={audio_data.shape}, sample_rate={sample_rate}, duration={audio_duration:.2f}s")
    
    # 如果是立体声，转换为单声道
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
        logger.info("Converted stereo to mono")
    
    # 更新结果中的音频时长
    results['audio_duration'] = audio_duration
    
    # 使用StreamingASRProcessor的transcribe_complete_audio方法，传递已加载的音频数据
    logger.info(f"Starting transcription for audio with duration: {audio_duration:.2f}s")
    complete_result = asr_processor.transcribe_complete_audio(audio_path, audio_data, sample_rate)
    
    # 记录转录完成时间
    transcription_complete_time = time.time()
    
    # 计算从音频加载开始到转录文本生成的响应时间
    response_time = transcription_complete_time - start_time
    
    logger.info(f"Complete transcription finished: '{complete_result['text']}', took: {complete_result['timing']['transcription_time']:.2f}s")
    
    # 记录转录结果
    results['full_transcription'] = complete_result['text']
    results['response_time'] = response_time
    
    # 打印结果摘要
    logger.info("\n" + "="*60)
    logger.info("COMPLETE ASR TEST RESULTS")
    logger.info("="*60)
    logger.info(f"Audio file: {audio_path}")
    logger.info(f"Audio duration: {audio_duration:.2f}s")
    logger.info(f"Model size: {model_size}")
    logger.info(f"Response time (load to transcription): {response_time:.3f}s")
    logger.info(f"Full transcription: '{results['full_transcription']}'")
    
    # 保存结果
    if save_results:
        save_complete_test_results(results, results_dir, audio_path)
    
    return results


def test_streaming_asr(audio_path: str, chunk_duration_ms: int = 100,
                      save_results: bool = True, results_dir: str = "experiments/results",
                      model_size: str = "tiny", device: str = "cuda",
                      simulate_streaming_delay: bool = False) -> Dict:
    """
    测试多线程流式ASR系统
    
    Args:
        audio_path: 音频文件路径
        chunk_duration_ms: 模拟流式输入的块时长（毫秒）
        save_results: 是否保存结果
        results_dir: 结果保存目录
        model_size: Whisper模型大小
        device: 计算设备 ("cuda" 或 "cpu")
        simulate_streaming_delay: 是否模拟实际音频产生的延迟
        
    Returns:
        Dict: 测试结果
    """
    
    # 创建多线程流式ASR系统
    logger.info(f"Initializing multi-thread streaming ASR system")
    streaming_asr = MultiThreadStreamingASR(
        audio_path=audio_path,
        chunk_duration_ms=chunk_duration_ms,
        model_size=model_size,
        device=device,
        simulate_streaming_delay=simulate_streaming_delay
    )
    
    logger.info(f"ASR系统初始化完成: 模型={model_size}, 设备={device}")
    
    # 启动多线程处理
    streaming_asr.start()
    
    # 等待处理完成
    logger.info("Waiting for processing to complete...")
    
    # 监控处理进度
    while not streaming_asr.segmentation_complete.is_set() or not streaming_asr.audio_segment_queue.empty():
        time.sleep(0.1)
    
    # 等待转录完成
    while not streaming_asr.transcription_complete_time:
        time.sleep(0.1)
    
    # 停止系统
    streaming_asr.stop()
    
    # 获取结果
    results = streaming_asr.get_results()
    
    # 打印结果摘要
    logger.info("\n" + "="*60)
    logger.info("STREAMING ASR TEST RESULTS")
    logger.info("="*60)
    logger.info(f"Audio file: {audio_path}")
    logger.info(f"Audio duration: {results['audio_duration']:.2f}s")
    logger.info(f"Model size: {model_size}")
    logger.info(f"Response time (last chunk to transcription): {results['response_time']:.3f}s")
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
        default="experiments/datasets/processed/experiments/length_analysis/audio/long/sample_001.wav",
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
    
    parser.add_argument(
        "--simulate-streaming-delay",
        action="store_true",
        default=False,
        help="是否模拟实际音频产生的延迟"
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
        logger.info(f"Simulate streaming delay: {args.simulate_streaming_delay}")
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
                device=args.device,
                simulate_streaming_delay=args.simulate_streaming_delay
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
                device=args.device,
                simulate_streaming_delay=args.simulate_streaming_delay
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
            logger.info(f"Audio duration: {streaming_results['audio_duration']:.2f}s")
            logger.info(f"Streaming ASR - Response time (last chunk to transcription): {streaming_results['response_time']:.3f}s")
            logger.info(f"Complete ASR - Response time (load to transcription): {complete_results['response_time']:.3f}s")
            logger.info(f"Time difference: {streaming_results['response_time'] - complete_results['response_time']:.3f}s")
            
            # 比较转录文本
            logger.info(f"Streaming ASR transcription: '{streaming_results['full_transcription']}'")
            logger.info(f"Complete ASR transcription: '{complete_results['full_transcription']}'")
            
            logger.info("\nBoth tests completed successfully!")
            
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        exit(1)