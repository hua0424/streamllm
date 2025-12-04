#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
流式全链路测试程序
实现：流式语音 -> 语音分段 -> ASR -> 流式文本 -> LLM缓存 -> LLM生成回复
支持流式处理和非流式处理两种模式，用于对比实验

功能：
1. 流式处理：模拟实时音频输入，流式ASR转录，流式LLM预计算
2. 非流式处理：完整音频一次性转录，完整文本一次性输入LLM
3. 支持ASR和LLM分别指定运行设备（CPU/CUDA）
4. 支持不同日志级别输出不同详细程度的信息
5. 对比模式下共享模型实例并进行预热
"""

import time
import queue
import threading
import soundfile as sf
import numpy as np
import os
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# 导入项目模块
from src.utils.logging_utils import get_logger, set_global_log_level
from src.asr.streamaudio_segmenter import StreamAudioSegmenter
from src.asr.faster_whisper_streamer import StreamingASRProcessor, ASRCache, ASRAudioSegment
from src.llm.stream_llm_inference import StreamLLMInference
from src.asr.run_stream_asr_test import convert_audio_segment
from src.config import LLM_MODEL_NAME

# 设置日志
logger = get_logger(__name__)


class SharedModels:
    """共享模型管理类，用于在多种测试模式间共享ASR和LLM模型实例"""
    
    def __init__(self, args):
        self.args = args
        self.asr_processor: Optional[StreamingASRProcessor] = None
        self.llm_inference: Optional[StreamLLMInference] = None
        self._warmed_up = False
    
    def initialize(self):
        """初始化ASR和LLM模型"""
        logger.info("=" * 60)
        logger.info("Initializing shared models...")
        logger.info("=" * 60)
        
        # 初始化ASR
        logger.info(f"Loading ASR model: {self.args.asr_model_size} on {self.args.asr_device}")
        self.asr_processor = StreamingASRProcessor(
            model_size=self.args.asr_model_size,
            device=self.args.asr_device,
            compute_type="auto",
            recognition_threshold=1.0,
            prefix_segments=self.args.asr_prefix_segments,
            suffix_segments_atleast=self.args.asr_suffix_segments
        )
        
        # 初始化LLM
        logger.info(f"Loading LLM model: {self.args.llm_model_name} on {self.args.llm_device}")
        self.llm_inference = StreamLLMInference(
            model_name=self.args.llm_model_name,
            device=self.args.llm_device,
            eval_mode=self.args.eval_mode
        )
        
        logger.info("Shared models initialized successfully.")
    
    def warmup(self, audio_data: np.ndarray, sample_rate: int):
        """
        模型预热：在正式测试前进行一次ASR和LLM推理
        
        Args:
            audio_data: 用于预热的音频数据
            sample_rate: 音频采样率
        """
        if self._warmed_up:
            logger.info("Models already warmed up, skipping...")
            return
        
        logger.info("=" * 60)
        logger.info("Starting model warmup...")
        logger.info("=" * 60)
        
        # 1. ASR预热：使用一小段音频进行转录
        warmup_duration = min(2.0, len(audio_data) / sample_rate)  # 最多使用2秒音频
        warmup_samples = int(warmup_duration * sample_rate)
        warmup_audio = audio_data[:warmup_samples]
        
        logger.info(f"[Warmup] ASR warmup with {warmup_duration:.2f}s audio...")
        warmup_start = time.time()
        
        # 使用完整音频转录方法进行预热
        asr_result = self.asr_processor.transcribe_complete_audio(
            audio_path="warmup",
            audio_data=warmup_audio,
            sample_rate=sample_rate
        )
        
        asr_warmup_time = time.time() - warmup_start
        logger.info(f"[Warmup] ASR warmup completed in {asr_warmup_time:.3f}s, result: '{asr_result['text'][:50]}...'")
        
        # 2. LLM预热：使用简单prompt进行一次推理
        logger.info("[Warmup] LLM warmup...")
        warmup_start = time.time()
        
        warmup_prompt = "你好"
        warmup_response = ""
        for token in self.llm_inference.once_add_and_generate(
            prompt=warmup_prompt,
            max_new_tokens=5  # 只生成少量token用于预热
        ):
            warmup_response += token
        
        llm_warmup_time = time.time() - warmup_start
        logger.info(f"[Warmup] LLM warmup completed in {llm_warmup_time:.3f}s, result: '{warmup_response}'")
        
        self._warmed_up = True
        logger.info("=" * 60)
        logger.info("Model warmup completed!")
        logger.info("=" * 60)


class BasePipelineTest:
    """流水线测试基类"""
    
    def __init__(self, args, shared_models: Optional[SharedModels] = None):
        self.args = args
        self.audio_path = args.audio
        self.full_response = ""
        self.shared_models = shared_models
        
        # 加载音频
        if not os.path.exists(self.audio_path):
            raise FileNotFoundError(f"Audio file not found: {self.audio_path}")
        self.audio_data, self.sample_rate = sf.read(self.audio_path, dtype='float32')
        
        # 转单声道
        if len(self.audio_data.shape) > 1:
            self.audio_data = self.audio_data.mean(axis=1)
        
        self.audio_duration = len(self.audio_data) / self.sample_rate
        logger.info(f"Audio loaded: duration={self.audio_duration:.2f}s, sample_rate={self.sample_rate}")
        
        # 初始化时间记录
        self.timings = {
            "start_time": 0.0,
            "audio_load_time": 0.0,      # 音频加载完成时间（非流式）
            "audio_end_time": 0.0,       # 最后一段音频结束时间（流式）
            "last_text_time": 0.0,       # 最后一段文本生成时间
            "first_token_time": 0.0,     # LLM首个token生成时间
            "llm_end_time": 0.0          # LLM生成完成时间
        }
        
        # 详细时间记录（DEBUG级别使用）
        self.detailed_timings = {
            "audio_chunk_times": [],      # 每个音频块的生成时间
            "text_segment_times": [],     # 每段文本的生成时间
            "token_times": []             # 每个token的生成时间
        }

    def _reset_timings(self):
        """重置时间记录"""
        self.timings = {
            "start_time": 0.0,
            "audio_load_time": 0.0,
            "audio_end_time": 0.0,
            "last_text_time": 0.0,
            "first_token_time": 0.0,
            "llm_end_time": 0.0
        }
        self.detailed_timings = {
            "audio_chunk_times": [],
            "text_segment_times": [],
            "token_times": []
        }
        self.full_response = ""

    def _print_stats(self, mode: str):
        """打印统计信息"""
        t = self.timings
        
        logger.info("\n" + "=" * 60)
        logger.info(f"PERFORMANCE STATISTICS ({mode.upper()} MODE)")
        logger.info("=" * 60)
        logger.info(f"Audio Duration:        {self.audio_duration:.2f}s")
        
        if mode == "streaming":
            logger.info(f"Audio End Time:        {t['audio_end_time'] - t['start_time']:.3f}s (from start)")
            logger.info(f"Last Text Gen Time:    {t['last_text_time'] - t['start_time']:.3f}s (from start)")
        else:
            logger.info(f"Audio Load Time:       {t['audio_load_time'] - t['start_time']:.3f}s (from start)")
            logger.info(f"Text Gen Time:         {t['last_text_time'] - t['start_time']:.3f}s (from start)")
        
        logger.info(f"First Token Time:      {t['first_token_time'] - t['start_time']:.3f}s (from start)")
        logger.info("-" * 30)
        
        # 计算关键延迟指标
        if mode == "streaming":
            audio_to_text = (t['last_text_time'] - t['audio_end_time']) * 1000
            audio_to_token = (t['first_token_time'] - t['audio_end_time']) * 1000
            text_to_token = (t['first_token_time'] - t['last_text_time']) * 1000
            
            logger.info(f"Latency (Audio End -> Last Text):   {audio_to_text:.2f} ms")
            logger.info(f"Latency (Audio End -> First Token): {audio_to_token:.2f} ms")
            logger.info(f"Latency (Last Text -> First Token): {text_to_token:.2f} ms")
        else:
            load_to_text = (t['last_text_time'] - t['audio_load_time']) * 1000
            load_to_token = (t['first_token_time'] - t['audio_load_time']) * 1000
            text_to_token = (t['first_token_time'] - t['last_text_time']) * 1000
            
            logger.info(f"Latency (Audio Load -> Text Gen):    {load_to_text:.2f} ms")
            logger.info(f"Latency (Audio Load -> First Token): {load_to_token:.2f} ms")
            logger.info(f"Latency (Text Gen -> First Token):   {text_to_token:.2f} ms")
        
        logger.info("=" * 60)

    def _get_metrics(self, mode: str) -> Dict[str, Any]:
        """获取性能指标"""
        t = self.timings
        
        metrics = {
            "mode": mode,
            "audio_duration_s": self.audio_duration,
        }
        
        if mode == "streaming":
            metrics.update({
                "audio_end_to_last_text_ms": (t["last_text_time"] - t["audio_end_time"]) * 1000,
                "audio_end_to_first_token_ms": (t["first_token_time"] - t["audio_end_time"]) * 1000,
                "last_text_to_first_token_ms": (t["first_token_time"] - t["last_text_time"]) * 1000,
            })
        else:
            metrics.update({
                "audio_load_to_text_ms": (t["last_text_time"] - t["audio_load_time"]) * 1000,
                "audio_load_to_first_token_ms": (t["first_token_time"] - t["audio_load_time"]) * 1000,
                "text_to_first_token_ms": (t["first_token_time"] - t["last_text_time"]) * 1000,
            })
        
        return metrics

    def _save_results(self, mode: str):
        """保存结果"""
        results_dir = Path(self.args.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_result_{mode}_{timestamp}.json"
        filepath = results_dir / filename
        
        data = {
            "config": {
                "audio": self.args.audio,
                "mode": mode,
                "asr_device": self.args.asr_device,
                "llm_device": self.args.llm_device,
                "asr_model_size": self.args.asr_model_size,
                "llm_model_name": self.args.llm_model_name,
            },
            "timings": self.timings,
            "response": self.full_response,
            "metrics": self._get_metrics(mode)
        }
        
        # DEBUG级别保存详细时间记录
        if self.args.log_level.upper() == "DEBUG":
            data["detailed_timings"] = self.detailed_timings
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Results saved to {filepath}")


class StreamPipelineTest(BasePipelineTest):
    """流式处理流水线测试"""
    
    def __init__(self, args, shared_models: Optional[SharedModels] = None):
        super().__init__(args, shared_models)
        
        self.chunk_duration_ms = args.chunk_duration
        self.chunk_size = int(self.sample_rate * self.chunk_duration_ms / 1000)
        
        # 初始化队列
        self.audio_chunk_queue = queue.Queue()
        self.audio_segment_queue = queue.Queue()
        self.text_queue = queue.Queue()
        
        # 初始化事件
        self.stop_event = threading.Event()
        self.audio_gen_done = threading.Event()
        self.segmentation_done = threading.Event()
        self.asr_done = threading.Event()
        
        # Segmenter（每次都需要新建，因为有状态）
        self.segmenter = StreamAudioSegmenter(
            sampling_rate=self.sample_rate,
            silence_threshold=0.5,
            min_speech_duration_ms=500,
            min_silence_duration_ms=300,
            window_size_ms=64
        )
        
        # 使用共享模型或创建新模型
        if shared_models:
            logger.info("Using shared ASR and LLM models for streaming test")
            self.asr_processor = shared_models.asr_processor
            self.llm_inference = shared_models.llm_inference
        else:
            logger.info("Initializing streaming modules...")
            # ASR - 使用独立的设备参数
            self.asr_processor = StreamingASRProcessor(
                model_size=args.asr_model_size,
                device=args.asr_device,
                compute_type="auto",
                recognition_threshold=1.0,
                prefix_segments=args.asr_prefix_segments,
                suffix_segments_atleast=args.asr_suffix_segments
            )
            
            # LLM - 使用独立的设备参数
            self.llm_inference = StreamLLMInference(
                model_name=args.llm_model_name,
                device=args.llm_device,
                eval_mode=args.eval_mode
            )
            logger.info(f"Streaming modules initialized. ASR device: {args.asr_device}, LLM device: {args.llm_device}")

    def _reset_state(self):
        """重置流水线状态，为新一轮测试做准备"""
        # 清空队列
        while not self.audio_chunk_queue.empty():
            try:
                self.audio_chunk_queue.get_nowait()
            except queue.Empty:
                break
        while not self.audio_segment_queue.empty():
            try:
                self.audio_segment_queue.get_nowait()
            except queue.Empty:
                break
        while not self.text_queue.empty():
            try:
                self.text_queue.get_nowait()
            except queue.Empty:
                break
        
        # 重置事件
        self.stop_event.clear()
        self.audio_gen_done.clear()
        self.segmentation_done.clear()
        self.asr_done.clear()
        
        # 重置时间记录
        self._reset_timings()
        
        # 重新创建Segmenter（有状态）
        self.segmenter = StreamAudioSegmenter(
            sampling_rate=self.sample_rate,
            silence_threshold=0.5,
            min_speech_duration_ms=500,
            min_silence_duration_ms=300,
            window_size_ms=64
        )

    def start(self):
        """启动所有线程"""
        # 重置状态
        self._reset_state()
        
        threads = [
            threading.Thread(target=self._audio_generation_worker, name="AudioGen"),
            threading.Thread(target=self._segmentation_worker, name="Segmenter"),
            threading.Thread(target=self._asr_worker, name="ASR"),
            threading.Thread(target=self._llm_worker, name="LLM")
        ]
        
        self.timings["start_time"] = time.time()
        for t in threads:
            t.start()
            
        for t in threads:
            t.join()
            
        logger.info("All threads finished.")
        self._print_stats("streaming")
        
        if self.args.save_results:
            self._save_results("streaming")
        
        return self._get_metrics("streaming")

    def _audio_generation_worker(self):
        """模拟流式音频输入 - 真实延迟"""
        logger.info("[AudioGen] Started")
        total_chunks = (len(self.audio_data) + self.chunk_size - 1) // self.chunk_size
        
        for i in range(0, len(self.audio_data), self.chunk_size):
            if self.stop_event.is_set():
                break
            
            chunk = self.audio_data[i:i+self.chunk_size]
            chunk_id = i // self.chunk_size
            current_time = time.time()
            
            self.audio_chunk_queue.put((chunk_id, chunk))
            
            # DEBUG级别记录每个音频块的时间
            logger.debug(f"[AudioGen] Chunk {chunk_id+1}/{total_chunks} generated at {current_time - self.timings['start_time']:.3f}s")
            self.detailed_timings["audio_chunk_times"].append({
                "chunk_id": chunk_id,
                "time": current_time - self.timings["start_time"]
            })
            
            # 模拟实时音频产生延迟（关键：30秒音频需要30秒产生）
            time.sleep(self.chunk_duration_ms / 1000)
        
        self.timings["audio_end_time"] = time.time()
        logger.info(f"[AudioGen] Audio finished at {self.timings['audio_end_time'] - self.timings['start_time']:.3f}s")
        
        self.audio_gen_done.set()
        logger.debug("[AudioGen] Finished")

    def _segmentation_worker(self):
        """音频分段"""
        logger.debug("[Segmenter] Started")
        state = self.segmenter.create_state()
        
        while not self.stop_event.is_set():
            try:
                chunk_id, chunk = self.audio_chunk_queue.get(timeout=0.1)
            except queue.Empty:
                if self.audio_gen_done.is_set():
                    break
                continue
            
            stream_segment, state = self.segmenter.process_audio(chunk, state)
            
            if stream_segment:
                segment_id = f"seg_{stream_segment.segment_id:03d}"
                is_start = (stream_segment.segment_id == 1)
                
                asr_segment = convert_audio_segment(stream_segment, segment_id, is_start, False)
                self.audio_segment_queue.put(asr_segment)
                logger.debug(f"[Segmenter] Pushed segment {segment_id}")
        
        # Flush remaining
        remaining_segment, state = self.segmenter.flush(state)
        if remaining_segment and len(remaining_segment.audio) > 0:
            segment_id = f"seg_{remaining_segment.segment_id:03d}"
            asr_segment = convert_audio_segment(remaining_segment, segment_id, False, True)
            self.audio_segment_queue.put(asr_segment)
            logger.info(f"[Segmenter] Pushed final segment {segment_id}")

        self.segmentation_done.set()
        logger.debug("[Segmenter] Finished")

    def _asr_worker(self):
        """
        ASR转录 - 异步版本
        
        使用两个子线程：
        1. collector: 从音频段队列收集音频，加入到 asr_cache
        2. transcriber: 从 asr_cache 中取出音频进行转录
        
        这样音频收集不会被转录处理阻塞，转录可以批量处理多个音频段
        """
        logger.debug("[ASR] Started")
        asr_cache = ASRCache()
        
        # 用于标记是否收到最终段
        final_received = threading.Event()
        # 用于标记转录是否完成
        transcription_done = threading.Event()
        
        def collector_task():
            """收集音频段到 asr_cache"""
            while not self.stop_event.is_set():
                try:
                    asr_segment = self.audio_segment_queue.get(timeout=0.1)
                except queue.Empty:
                    if self.segmentation_done.is_set() and self.audio_segment_queue.empty():
                        break
                    continue
                
                asr_cache.add_segment(asr_segment)
                logger.debug(f"[ASR-Collector] Added segment {asr_segment.id}, is_final={asr_segment.is_final}")
                
                if asr_segment.is_final:
                    final_received.set()
                    break
            
            logger.debug("[ASR-Collector] Finished")
        
        def transcriber_task():
            """从 asr_cache 进行转录"""
            nonlocal asr_cache  # 使用外层的 asr_cache
            is_final = False
            
            while not self.stop_event.is_set() and not is_final:
                # 检查是否有待处理的音频段
                if len(asr_cache.waiting_segment_queue) == 0:
                    # 如果 collector 已经结束且没有待处理的段，退出
                    if final_received.is_set():
                        logger.debug("[ASR-Transcriber] Collector finished but no final segment processed yet, waiting...")
                    time.sleep(0.05)
                    continue
                
                # 检查是否正在处理中（避免并发）
                if asr_cache.is_processing():
                    time.sleep(0.05)
                    continue
                
                # 进行转录（transcribe_audio_segment 内部会调用 set_processing/set_processed）
                asr_cache, output_text, is_final = self.asr_processor.transcribe_audio_segment(asr_cache)
                
                if output_text:
                    current_time = time.time()
                    self.timings["last_text_time"] = current_time
                    
                    logger.debug(f"[ASR-Transcriber] Text at {current_time - self.timings['start_time']:.3f}s: {output_text}")
                    self.detailed_timings["text_segment_times"].append({
                        "text": output_text,
                        "time": current_time - self.timings["start_time"]
                    })
                    
                    logger.info(f"[ASR] Output: {output_text}")
                    self.text_queue.put((output_text, False))
            
            # 发送结束信号
            self.text_queue.put(("", True))
            transcription_done.set()
            logger.debug("[ASR-Transcriber] Finished")
        
        # 启动两个子线程
        collector_thread = threading.Thread(target=collector_task, name="ASR-Collector")
        transcriber_thread = threading.Thread(target=transcriber_task, name="ASR-Transcriber")
        
        collector_thread.start()
        transcriber_thread.start()
        
        # 等待两个线程完成
        collector_thread.join()
        transcriber_thread.join()
        
        # INFO级别输出最后一段文本的时间
        logger.info(f"[ASR] Last text generated at {self.timings['last_text_time'] - self.timings['start_time']:.3f}s")
        
        self.asr_done.set()
        logger.debug("[ASR] Finished")

    def _llm_worker(self):
        """LLM缓存与生成"""
        logger.debug("[LLM] Started")
        kv_cache = None
        self.full_response = ""
        
        while not self.stop_event.is_set():
            try:
                text, is_end = self.text_queue.get(timeout=0.1)
            except queue.Empty:
                if self.asr_done.is_set():
                    break
                continue
            
            # 缓存/预处理
            if text or is_end:
                logger.debug(f"[LLM] Caching prompt: '{text}' (is_end={is_end})")
                kv_cache = self.llm_inference.cache_prompt(text, pre_cache=kv_cache, is_end=is_end)
            
            # 如果结束，开始生成
            if is_end:
                logger.info("=" * 40)
                logger.info("[LLM] Start Generation")
                logger.info("=" * 40)
                
                first_token = True
                
                for token in self.llm_inference.generate(pre_cache=kv_cache):
                    current_time = time.time()
                    
                    if first_token:
                        self.timings["first_token_time"] = current_time
                        logger.info(f"[LLM] First token at {current_time - self.timings['start_time']:.3f}s")
                        first_token = False
                    
                    # DEBUG级别记录每个token的时间
                    logger.debug(f"[LLM] Token '{token}' at {current_time - self.timings['start_time']:.3f}s")
                    self.detailed_timings["token_times"].append({
                        "token": token,
                        "time": current_time - self.timings["start_time"]
                    })
                    
                    print(token, end="", flush=True)
                    self.full_response += token
                
                print()  # 换行
                self.timings["llm_end_time"] = time.time()
                logger.info("=" * 40)
                logger.info(f"[LLM] Generation Finished")
                logger.info(f"[LLM] Full Response: {self.full_response}")
                break
        
        logger.debug("[LLM] Finished")


class NonStreamPipelineTest(BasePipelineTest):
    """非流式处理流水线测试（完整音频一次性处理）"""
    
    def __init__(self, args, shared_models: Optional[SharedModels] = None):
        super().__init__(args, shared_models)
        
        # 使用共享模型或创建新模型
        if shared_models:
            logger.info("Using shared ASR and LLM models for non-streaming test")
            self.asr_processor = shared_models.asr_processor
            self.llm_inference = shared_models.llm_inference
        else:
            logger.info("Initializing non-streaming modules...")
            # ASR - 使用独立的设备参数
            self.asr_processor = StreamingASRProcessor(
                model_size=args.asr_model_size,
                device=args.asr_device,
                compute_type="auto",
                recognition_threshold=1.0,
            )
            
            # LLM - 使用独立的设备参数
            self.llm_inference = StreamLLMInference(
                model_name=args.llm_model_name,
                device=args.llm_device,
                eval_mode=args.eval_mode
            )
            logger.info(f"Non-streaming modules initialized. ASR device: {args.asr_device}, LLM device: {args.llm_device}")

    def start(self):
        """执行非流式处理"""
        # 重置时间记录
        self._reset_timings()
        
        self.timings["start_time"] = time.time()
        
        # 1. 音频已在初始化时加载，记录加载完成时间
        self.timings["audio_load_time"] = time.time()
        logger.info(f"[NonStream] Audio loaded at {self.timings['audio_load_time'] - self.timings['start_time']:.3f}s")
        
        # 2. 一次性ASR转录
        logger.info("[NonStream] Starting complete ASR transcription...")
        asr_result = self.asr_processor.transcribe_complete_audio(
            audio_path=self.audio_path,
            audio_data=self.audio_data,
            sample_rate=self.sample_rate
        )
        
        transcribed_text = asr_result['text']
        self.timings["last_text_time"] = time.time()
        logger.info(f"[NonStream] ASR completed at {self.timings['last_text_time'] - self.timings['start_time']:.3f}s")
        logger.info(f"[NonStream] Transcribed text: {transcribed_text}")
        
        # 3. 一次性LLM生成
        logger.info("=" * 40)
        logger.info("[NonStream] Starting LLM generation...")
        logger.info("=" * 40)
        
        first_token = True
        self.full_response = ""
        
        # 使用once_add_and_generate一次性输入完整prompt并生成
        for token in self.llm_inference.once_add_and_generate(
            prompt=transcribed_text,
            max_new_tokens=50
        ):
            current_time = time.time()
            
            if first_token:
                self.timings["first_token_time"] = current_time
                logger.info(f"[NonStream] First token at {current_time - self.timings['start_time']:.3f}s")
                first_token = False
            
            # DEBUG级别记录每个token的时间
            logger.debug(f"[NonStream] Token '{token}' at {current_time - self.timings['start_time']:.3f}s")
            self.detailed_timings["token_times"].append({
                "token": token,
                "time": current_time - self.timings["start_time"]
            })
            
            print(token, end="", flush=True)
            self.full_response += token
        
        print()  # 换行
        self.timings["llm_end_time"] = time.time()
        
        logger.info("=" * 40)
        logger.info(f"[NonStream] Generation Finished")
        logger.info(f"[NonStream] Full Response: {self.full_response}")
        
        self._print_stats("non-streaming")
        
        if self.args.save_results:
            self._save_results("non-streaming")
        
        return self._get_metrics("non-streaming")


def run_comparison(args):
    """运行流式和非流式对比测试（共享模型实例，带预热）"""
    logger.info("=" * 60)
    logger.info("COMPARISON TEST: STREAMING vs NON-STREAMING")
    logger.info("=" * 60)
    
    # 1. 创建共享模型实例
    shared_models = SharedModels(args)
    shared_models.initialize()
    
    # 2. 加载音频用于预热
    audio_data, sample_rate = sf.read(args.audio, dtype='float32')
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    
    # 3. 模型预热
    shared_models.warmup(audio_data, sample_rate)
    
    # 4. 运行流式测试（使用共享模型）
    logger.info("\n" + "=" * 60)
    logger.info(">>> Running STREAMING test...")
    logger.info("=" * 60)
    stream_pipeline = StreamPipelineTest(args, shared_models)
    stream_metrics = stream_pipeline.start()
    
    # 5. 运行非流式测试（使用共享模型）
    logger.info("\n" + "=" * 60)
    logger.info(">>> Running NON-STREAMING test...")
    logger.info("=" * 60)
    non_stream_pipeline = NonStreamPipelineTest(args, shared_models)
    non_stream_metrics = non_stream_pipeline.start()
    
    # 6. 打印对比结果
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON RESULTS")
    logger.info("=" * 60)
    logger.info(f"Audio Duration: {stream_metrics['audio_duration_s']:.2f}s")
    logger.info("-" * 30)
    
    # 流式模式：音频结束到首个token的延迟
    stream_latency = stream_metrics['audio_end_to_first_token_ms']
    # 非流式模式：音频加载到首个token的延迟
    non_stream_latency = non_stream_metrics['audio_load_to_first_token_ms']
    
    logger.info(f"STREAMING - Audio End to First Token:   {stream_latency:.2f} ms")
    logger.info(f"NON-STREAM - Audio Load to First Token: {non_stream_latency:.2f} ms")
    
    # 计算优化效果
    improvement = non_stream_latency - stream_latency
    logger.info("-" * 30)
    logger.info(f"Latency Improvement: {improvement:.2f} ms")
    if non_stream_latency > 0:
        logger.info(f"Improvement Ratio: {improvement / non_stream_latency * 100:.1f}%")
    
    logger.info("=" * 60)
    
    # 7. 保存对比结果
    if args.save_results:
        results_dir = Path(args.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = results_dir / f"comparison_result_{timestamp}.json"
        
        comparison_data = {
            "config": {
                "audio": args.audio,
                "asr_device": args.asr_device,
                "llm_device": args.llm_device,
                "asr_model_size": args.asr_model_size,
                "llm_model_name": args.llm_model_name,
                "warmup_performed": True,
                "shared_models": True,
            },
            "streaming_metrics": stream_metrics,
            "non_streaming_metrics": non_stream_metrics,
            "comparison": {
                "latency_improvement_ms": improvement,
                "improvement_ratio": improvement / non_stream_latency * 100 if non_stream_latency > 0 else 0
            },
            "streaming_response": stream_pipeline.full_response,
            "non_streaming_response": non_stream_pipeline.full_response
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(comparison_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Comparison results saved to {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="流式全链路测试程序 - 支持流式和非流式处理对比",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 基础参数
    parser.add_argument("--audio", type=str, 
                        default="/usr/local/app/jupyterlab/yanjiu/streamllm/experiments/datasets/processed/experiments/length_analysis/audio/long/sample_001.wav", 
                        help="Path to audio file")
    parser.add_argument("--chunk-duration", type=int, default=500, 
                        help="Chunk duration in ms (for streaming mode)")
    
    # 设备参数 - ASR和LLM分开控制
    parser.add_argument("--asr-device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Device for ASR model (auto/cuda/cpu)")
    parser.add_argument("--llm-device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Device for LLM model (auto/cuda/cpu)")
    
    # ASR参数
    parser.add_argument("--asr-model-size", type=str, default="tiny",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="ASR model size")
    parser.add_argument("--asr-prefix-segments", type=int, default=1, 
                        help="ASR prefix segments")
    parser.add_argument("--asr-suffix-segments", type=int, default=1, 
                        help="ASR suffix segments")
    
    # LLM参数
    parser.add_argument("--llm-model-name", type=str, default=LLM_MODEL_NAME, 
                        help="LLM model name")
    parser.add_argument("--eval-mode", action="store_true", default=False,
                        help="LLM eval mode (only generate one token)")
    
    # 运行模式控制
    parser.add_argument("--mode", type=str, default="streaming",
                        choices=["streaming", "non-streaming", "both"],
                        help="Test mode: streaming, non-streaming, or both for comparison")
    
    # 日志控制
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Log level (DEBUG shows detailed timing, INFO shows key metrics)")
    
    # 结果保存
    parser.add_argument("--save-results", action="store_true", 
                        help="Save results to file")
    parser.add_argument("--results-dir", type=str, default="experiments/results", 
                        help="Directory to save results")
    
    args = parser.parse_args()

    # 设置日志级别
    set_global_log_level(args.log_level)
    
    # 打印配置信息
    logger.info("=" * 60)
    logger.info("STREAM PIPELINE TEST CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Audio file: {args.audio}")
    logger.info(f"Test mode: {args.mode}")
    logger.info(f"ASR device: {args.asr_device}")
    logger.info(f"LLM device: {args.llm_device}")
    logger.info(f"ASR model: {args.asr_model_size}")
    logger.info(f"LLM model: {args.llm_model_name}")
    logger.info(f"Log level: {args.log_level}")
    if args.mode in ["streaming", "both"]:
        logger.info(f"Chunk duration: {args.chunk_duration}ms")
    logger.info("=" * 60)
    
    try:
        if args.mode == "streaming":
            pipeline = StreamPipelineTest(args)
            pipeline.start()
        elif args.mode == "non-streaming":
            pipeline = NonStreamPipelineTest(args)
            pipeline.start()
        elif args.mode == "both":
            run_comparison(args)
            
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
