#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验一：延迟与语音长度的关系验证 (Effect Validation)

实验目的：
验证随着语音输入长度的增加，流式方案 (System B) 的 TTFT 保持相对稳定，
而非流式方案 (System A) 的 TTFT 呈线性增长。

使用方式（在项目根目录下运行）：
    uv run python -m experiments.scripts.run_exp_latency [参数]

主要功能：
1. 扫描实验数据集，按音频时长分组
2. 对每个样本分别运行流式和非流式测试
3. 记录 TTFT 和其他性能指标
4. 生成统计结果和可视化图表

关键设计：
- 模型预热：使用真实音频进行多次预热，确保 CUDA kernel 已加载
- 共享模型：流式和非流式测试共享同一个 ASR 和 LLM 实例
- 状态重置：每次测试前重置所有状态，确保公平性
- 内存管理：定期清理 GPU 缓存
"""

import argparse
import json
import time
import sys
import csv
import wave
import gc
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
import numpy as np
import soundfile as sf

# 获取项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# 导入项目模块
from src.utils.logging_utils import get_logger, set_global_log_level
from src.asr.streamaudio_segmenter import StreamAudioSegmenter
from src.asr.faster_whisper_streamer import StreamingASRProcessor, ASRCache, ASRAudioSegment
from src.llm.stream_llm_inference import StreamLLMInference
from src.asr.run_stream_asr_test import convert_audio_segment
from src.config import LLM_MODEL_NAME, ASR_MODEL_NAME

logger = get_logger(__name__)


# =============================================================================
# 数据结构定义
# =============================================================================

@dataclass
class SampleInfo:
    """样本信息"""
    sample_id: str
    dialog_id: str
    turn_index: int
    text: str
    text_length: int
    audio_file: str
    audio_path: Path
    audio_duration: float
    language: str
    dataset: str
    duration_group: str = ""  # 时长分组


@dataclass
class ExperimentResult:
    """单次实验结果"""
    sample_id: str
    audio_duration: float
    duration_group: str
    mode: str  # streaming / non-streaming
    
    # 核心指标 (ms)
    ttft: float  # Time to First Token
    asr_time: float  # ASR 处理时间
    llm_prefill_time: float  # LLM 预填充时间
    
    # 详细时间戳
    start_time: float = 0.0
    audio_end_time: float = 0.0  # 流式：音频结束时间
    audio_load_time: float = 0.0  # 非流式：音频加载时间
    last_text_time: float = 0.0
    first_token_time: float = 0.0
    
    # 额外信息
    transcribed_text: str = ""
    response_preview: str = ""
    error: str = ""


@dataclass 
class GroupStatistics:
    """分组统计结果"""
    group: str
    sample_count: int
    avg_duration: float
    
    # 流式统计
    streaming_ttft_mean: float
    streaming_ttft_std: float
    streaming_ttft_min: float
    streaming_ttft_max: float
    
    # 非流式统计
    non_streaming_ttft_mean: float
    non_streaming_ttft_std: float
    non_streaming_ttft_min: float
    non_streaming_ttft_max: float
    
    # 优化效果
    improvement_mean: float
    improvement_ratio: float


# =============================================================================
# 时长分组定义
# =============================================================================

DURATION_GROUPS = {
    "short": (0, 5),       # 短语音: < 5s
    "medium": (5, 15),     # 中等语音: 5-15s
    "long": (15, 30),      # 长语音: 15-30s
    "very_long": (30, 60), # 超长语音: 30-60s
    "extra_long": (60, float('inf'))  # 极长语音: > 60s
}


def get_duration_group(duration: float) -> str:
    """根据时长获取分组名称"""
    for group_name, (min_dur, max_dur) in DURATION_GROUPS.items():
        if min_dur <= duration < max_dur:
            return group_name
    return "extra_long"


# =============================================================================
# 内存管理
# =============================================================================

def clear_gpu_memory():
    """清理 GPU 内存"""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


# =============================================================================
# 数据加载
# =============================================================================

def get_audio_duration(audio_path: Path) -> float:
    """获取音频文件时长"""
    try:
        with wave.open(str(audio_path), 'rb') as wav_file:
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
            return frames / sample_rate
    except Exception as e:
        logger.warning(f"无法读取音频时长 {audio_path}: {e}")
        return -1


def load_samples(
    json_dir: Path, 
    audio_dir: Path,
    dataset_filter: Optional[str] = None,
    max_samples: Optional[int] = None
) -> List[SampleInfo]:
    """
    加载实验样本
    
    Args:
        json_dir: JSON 文件目录
        audio_dir: 音频文件目录
        dataset_filter: 数据集过滤 (crosswoz/multiwoz/None=全部)
        max_samples: 最大样本数
    
    Returns:
        样本信息列表
    """
    samples = []
    
    # 遍历数据集目录
    datasets = ["crosswoz", "multiwoz"] if dataset_filter is None else [dataset_filter]
    
    for dataset in datasets:
        dataset_json_dir = json_dir / dataset
        dataset_audio_dir = audio_dir / dataset
        
        if not dataset_json_dir.exists():
            logger.warning(f"数据集目录不存在: {dataset_json_dir}")
            continue
        
        # 遍历 JSON 文件
        json_files = sorted(dataset_json_dir.glob("*.json"))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 构建音频路径
                audio_filename = data.get('audio_file', '')
                audio_path = dataset_audio_dir / audio_filename
                
                # 检查音频文件是否存在
                if not audio_path.exists():
                    logger.debug(f"音频文件不存在，跳过: {audio_path}")
                    continue
                
                # 获取音频时长
                audio_duration = data.get('audio_duration')
                if audio_duration is None or audio_duration <= 0:
                    audio_duration = get_audio_duration(audio_path)
                
                if audio_duration <= 0:
                    logger.warning(f"无效的音频时长，跳过: {audio_path}")
                    continue
                
                # 创建样本信息
                sample = SampleInfo(
                    sample_id=data['sample_id'],
                    dialog_id=data['dialog_id'],
                    turn_index=data['turn_index'],
                    text=data['text'],
                    text_length=data.get('text_length', len(data['text'])),
                    audio_file=audio_filename,
                    audio_path=audio_path,
                    audio_duration=audio_duration,
                    language=data['language'],
                    dataset=data['dataset'],
                    duration_group=get_duration_group(audio_duration)
                )
                
                samples.append(sample)
                
            except Exception as e:
                logger.error(f"加载样本失败 {json_file}: {e}")
                continue
        
        if max_samples and len(samples) >= max_samples:
            samples = samples[:max_samples]
            break
    
    # 按音频时长排序
    samples.sort(key=lambda x: x.audio_duration)
    
    logger.info(f"加载了 {len(samples)} 个有效样本")
    
    # 打印分组统计
    group_counts = {}
    for sample in samples:
        group_counts[sample.duration_group] = group_counts.get(sample.duration_group, 0) + 1
    
    logger.info("样本分组统计:")
    for group, count in sorted(group_counts.items()):
        logger.info(f"  {group}: {count} 个样本")
    
    return samples


# =============================================================================
# 共享模型管理
# =============================================================================

class SharedModels:
    """
    共享模型实例
    
    关键设计：
    1. ASR 和 LLM 模型只加载一次，在所有测试间共享
    2. 提供充分的预热，确保 CUDA kernel 已加载
    3. 每次测试前调用 reset_state() 重置状态
    """
    
    def __init__(self, args):
        self.args = args
        self.asr_processor: Optional[StreamingASRProcessor] = None
        self.llm_inference: Optional[StreamLLMInference] = None
        self._warmed_up = False
        self._warmup_audio = None
        self._warmup_sample_rate = 16000
    
    def initialize(self):
        """初始化模型"""
        logger.info("=" * 60)
        logger.info("初始化共享模型...")
        logger.info("=" * 60)
        
        # ASR
        logger.info(f"加载 ASR 模型: {self.args.asr_model_size} on {self.args.asr_device}")
        self.asr_processor = StreamingASRProcessor(
            model_size=self.args.asr_model_size,
            device=self.args.asr_device,
            compute_type="auto",
            recognition_threshold=1.0,
            prefix_segments=1,
            suffix_segments_atleast=1
        )
        
        # LLM
        logger.info(f"加载 LLM 模型: {self.args.llm_model_name} on {self.args.llm_device}")
        self.llm_inference = StreamLLMInference(
            model_name=self.args.llm_model_name,
            device=self.args.llm_device,
            eval_mode=False
        )
        
        logger.info("模型初始化完成")
    
    def set_warmup_audio(self, audio_data: np.ndarray, sample_rate: int):
        """设置预热用的音频数据（使用第一个样本的音频）"""
        self._warmup_audio = audio_data
        self._warmup_sample_rate = sample_rate
    
    def warmup(self, warmup_rounds: int = 3):
        """
        模型预热
        
        使用真实音频进行多次预热，确保：
        1. CUDA kernel 已加载
        2. 模型权重已缓存到 GPU
        3. 推理速度稳定
        
        Args:
            warmup_rounds: 预热轮数（默认3次）
        """
        if self._warmed_up:
            logger.info("模型已预热，跳过")
            return
        
        logger.info("=" * 60)
        logger.info(f"模型预热中（{warmup_rounds} 轮）...")
        logger.info("=" * 60)
        
        # 如果没有设置预热音频，使用合成的测试音频
        if self._warmup_audio is None:
            # 生成 3 秒的测试音频（正弦波 + 噪声，模拟真实语音）
            duration = 3.0
            sample_rate = 16000
            t = np.linspace(0, duration, int(duration * sample_rate))
            # 合成简单的测试信号
            self._warmup_audio = (
                0.3 * np.sin(2 * np.pi * 440 * t) +  # 基频
                0.1 * np.sin(2 * np.pi * 880 * t) +  # 谐波
                0.05 * np.random.randn(len(t))       # 噪声
            ).astype(np.float32)
            self._warmup_sample_rate = sample_rate
        
        warmup_times = {
            "asr": [],
            "llm_cache": [],
            "llm_generate": []
        }
        
        for round_idx in range(warmup_rounds):
            logger.info(f"  预热轮次 {round_idx + 1}/{warmup_rounds}")
            
            # ASR 预热
            asr_start = time.time()
            asr_result = self.asr_processor.transcribe_complete_audio(
                audio_path=f"warmup_{round_idx}",
                audio_data=self._warmup_audio,
                sample_rate=self._warmup_sample_rate
            )
            asr_time = (time.time() - asr_start) * 1000
            warmup_times["asr"].append(asr_time)
            logger.debug(f"    ASR 预热: {asr_time:.2f}ms, 结果: '{asr_result['text'][:30]}...'")
            
            # LLM 缓存预热
            cache_start = time.time()
            kv_cache = self.llm_inference.cache_prompt("你好，这是一个测试。", is_end=True)
            cache_time = (time.time() - cache_start) * 1000
            warmup_times["llm_cache"].append(cache_time)
            logger.debug(f"    LLM Cache 预热: {cache_time:.2f}ms")
            
            # LLM 生成预热
            gen_start = time.time()
            response = ""
            for token in self.llm_inference.generate(pre_cache=kv_cache, max_new_tokens=10):
                response += token
            gen_time = (time.time() - gen_start) * 1000
            warmup_times["llm_generate"].append(gen_time)
            logger.debug(f"    LLM Generate 预热: {gen_time:.2f}ms, 结果: '{response[:30]}...'")
            
            # 清理本轮预热的缓存
            del kv_cache
            clear_gpu_memory()
        
        # 打印预热统计
        logger.info("-" * 40)
        logger.info("预热统计:")
        for name, times in warmup_times.items():
            avg_time = np.mean(times)
            std_time = np.std(times)
            logger.info(f"  {name}: {avg_time:.2f}ms (±{std_time:.2f}ms)")
        
        self._warmed_up = True
        logger.info("=" * 60)
        logger.info("模型预热完成！")
        logger.info("=" * 60)
    
    def reset_state(self):
        """
        重置模型状态
        
        在每次测试前调用，确保：
        1. LLM timing 事件已清理
        2. ASR timing 事件已清理
        3. GPU 缓存已清理
        """
        # 重置 LLM timing
        if self.llm_inference:
            self.llm_inference.reset_timings()
        
        # 重置 ASR timing
        if self.asr_processor:
            self.asr_processor.timing_events.clear()
        
        # 清理 GPU 缓存
        clear_gpu_memory()


# =============================================================================
# 实验执行器
# =============================================================================

class LatencyExperiment:
    """
    延迟实验执行器
    
    关键设计：
    1. 使用共享的 ASR 和 LLM 模型实例
    2. 每个样本的流式和非流式测试共享同一个音频数据
    3. 每次测试前重置模型状态
    4. 定期清理 GPU 内存
    """
    
    def __init__(self, shared_models: SharedModels, args):
        self.models = shared_models
        self.args = args
        self.results: List[ExperimentResult] = []
    
    def run_single_sample(self, sample: SampleInfo) -> Tuple[ExperimentResult, ExperimentResult]:
        """
        对单个样本运行流式和非流式测试
        
        关键：两种模式使用相同的音频数据和模型实例
        
        Returns:
            (流式结果, 非流式结果)
        """
        # 加载音频（只加载一次，两种模式共用）
        audio_data, sample_rate = sf.read(str(sample.audio_path), dtype='float32')
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        
        # 重采样到 16kHz (如果需要)
        if sample_rate != 16000:
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        
        # ===== 流式测试 =====
        # 重置状态
        self.models.reset_state()
        streaming_result = self._run_streaming_test(sample, audio_data, sample_rate)
        
        # ===== 非流式测试 =====
        # 重置状态（确保公平）
        self.models.reset_state()
        non_streaming_result = self._run_non_streaming_test(sample, audio_data, sample_rate)
        
        return streaming_result, non_streaming_result
    
    def _run_streaming_test(
        self, 
        sample: SampleInfo, 
        audio_data: np.ndarray, 
        sample_rate: int
    ) -> ExperimentResult:
        """运行流式测试"""
        import queue
        import threading
        
        result = ExperimentResult(
            sample_id=sample.sample_id,
            audio_duration=sample.audio_duration,
            duration_group=sample.duration_group,
            mode="streaming",
            ttft=0,
            asr_time=0,
            llm_prefill_time=0
        )
        
        try:
            # 每次测试创建新的 Segmenter（有状态）
            segmenter = StreamAudioSegmenter(
                sampling_rate=sample_rate,
                silence_threshold=0.5,
                min_speech_duration_ms=500,
                min_silence_duration_ms=300,
                window_size_ms=64
            )
            
            chunk_duration_ms = self.args.chunk_duration
            chunk_size = int(sample_rate * chunk_duration_ms / 1000)
            
            # 队列和事件（每次测试新建）
            audio_chunk_queue = queue.Queue()
            audio_segment_queue = queue.Queue()
            text_queue = queue.Queue()
            
            audio_gen_done = threading.Event()
            segmentation_done = threading.Event()
            asr_done = threading.Event()
            
            # 时间记录
            timings = {
                "start_time": 0.0,
                "audio_end_time": 0.0,
                "last_text_time": 0.0,
                "first_token_time": 0.0
            }
            
            full_response = []
            transcribed_text = []
            
            # 音频生成线程
            def audio_gen_worker():
                for i in range(0, len(audio_data), chunk_size):
                    chunk = audio_data[i:i+chunk_size]
                    chunk_id = i // chunk_size
                    audio_chunk_queue.put((chunk_id, chunk))
                    time.sleep(chunk_duration_ms / 1000)  # 模拟实时
                
                timings["audio_end_time"] = time.time()
                audio_gen_done.set()
            
            # 分段线程
            def segmentation_worker():
                state = segmenter.create_state()
                
                while True:
                    try:
                        chunk_id, chunk = audio_chunk_queue.get(timeout=0.1)
                    except queue.Empty:
                        if audio_gen_done.is_set():
                            break
                        continue
                    
                    stream_segment, state = segmenter.process_audio(chunk, state)
                    
                    if stream_segment:
                        segment_id = f"seg_{stream_segment.segment_id:03d}"
                        is_start = (stream_segment.segment_id == 1)
                        asr_segment = convert_audio_segment(stream_segment, segment_id, is_start, False)
                        audio_segment_queue.put(asr_segment)
                
                # Flush
                remaining_segment, state = segmenter.flush(state)
                if remaining_segment and len(remaining_segment.audio) > 0:
                    segment_id = f"seg_{remaining_segment.segment_id:03d}"
                    asr_segment = convert_audio_segment(remaining_segment, segment_id, False, True)
                    audio_segment_queue.put(asr_segment)
                
                segmentation_done.set()
            
            # ASR 线程
            def asr_worker():
                # 每次测试新建 ASRCache
                asr_cache = ASRCache()
                final_received = threading.Event()
                
                def collector():
                    while True:
                        try:
                            asr_segment = audio_segment_queue.get(timeout=0.1)
                        except queue.Empty:
                            if segmentation_done.is_set() and audio_segment_queue.empty():
                                break
                            continue
                        
                        asr_cache.add_segment(asr_segment)
                        if asr_segment.is_final:
                            final_received.set()
                            break
                
                def transcriber():
                    nonlocal asr_cache
                    is_final = False
                    
                    while not is_final:
                        if len(asr_cache.waiting_segment_queue) == 0:
                            if final_received.is_set():
                                time.sleep(0.05)
                                if len(asr_cache.waiting_segment_queue) == 0:
                                    break
                            time.sleep(0.05)
                            continue
                        
                        if asr_cache.is_processing():
                            time.sleep(0.05)
                            continue
                        
                        # 使用共享的 ASR 处理器
                        asr_cache, output_text, is_final = self.models.asr_processor.transcribe_audio_segment(asr_cache)
                        
                        if output_text:
                            timings["last_text_time"] = time.time()
                            transcribed_text.append(output_text)
                            text_queue.put((output_text, False))
                    
                    text_queue.put(("", True))
                
                collector_thread = threading.Thread(target=collector)
                transcriber_thread = threading.Thread(target=transcriber)
                
                collector_thread.start()
                transcriber_thread.start()
                
                collector_thread.join()
                transcriber_thread.join()
                
                asr_done.set()
            
            # LLM 线程
            def llm_worker():
                # 每次测试新建 KV Cache（从 None 开始）
                kv_cache = None
                
                while True:
                    try:
                        text, is_end = text_queue.get(timeout=0.1)
                    except queue.Empty:
                        if asr_done.is_set():
                            break
                        continue
                    
                    if text or is_end:
                        # 使用共享的 LLM 推理器
                        kv_cache = self.models.llm_inference.cache_prompt(text, pre_cache=kv_cache, is_end=is_end)
                    
                    if is_end:
                        first_token = True
                        for token in self.models.llm_inference.generate(pre_cache=kv_cache, max_new_tokens=self.args.max_tokens):
                            if first_token:
                                timings["first_token_time"] = time.time()
                                first_token = False
                            full_response.append(token)
                        break
            
            # 启动线程
            timings["start_time"] = time.time()
            
            threads = [
                threading.Thread(target=audio_gen_worker),
                threading.Thread(target=segmentation_worker),
                threading.Thread(target=asr_worker),
                threading.Thread(target=llm_worker)
            ]
            
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            
            # 计算指标
            result.start_time = timings["start_time"]
            result.audio_end_time = timings["audio_end_time"]
            result.last_text_time = timings["last_text_time"]
            result.first_token_time = timings["first_token_time"]
            
            result.ttft = (timings["first_token_time"] - timings["audio_end_time"]) * 1000
            result.asr_time = (timings["last_text_time"] - timings["audio_end_time"]) * 1000
            result.llm_prefill_time = (timings["first_token_time"] - timings["last_text_time"]) * 1000
            
            result.transcribed_text = " ".join(transcribed_text)
            result.response_preview = "".join(full_response)[:100]
            
        except Exception as e:
            result.error = str(e)
            logger.error(f"流式测试失败 {sample.sample_id}: {e}")
            import traceback
            traceback.print_exc()
        
        return result
    
    def _run_non_streaming_test(
        self, 
        sample: SampleInfo, 
        audio_data: np.ndarray, 
        sample_rate: int
    ) -> ExperimentResult:
        """运行非流式测试"""
        result = ExperimentResult(
            sample_id=sample.sample_id,
            audio_duration=sample.audio_duration,
            duration_group=sample.duration_group,
            mode="non-streaming",
            ttft=0,
            asr_time=0,
            llm_prefill_time=0
        )
        
        try:
            start_time = time.time()
            result.start_time = start_time
            
            # 模拟音频加载完成（非流式需要等待完整音频）
            audio_load_time = time.time()
            result.audio_load_time = audio_load_time
            
            # 一次性 ASR 转录（使用共享的 ASR 处理器）
            asr_result = self.models.asr_processor.transcribe_complete_audio(
                audio_path=str(sample.audio_path),
                audio_data=audio_data,
                sample_rate=sample_rate
            )
            
            transcribed_text = asr_result['text']
            last_text_time = time.time()
            result.last_text_time = last_text_time
            result.transcribed_text = transcribed_text
            
            # 一次性 LLM 生成（使用共享的 LLM 推理器）
            full_response = []
            first_token = True
            first_token_time = 0
            
            for token in self.models.llm_inference.once_add_and_generate(
                prompt=transcribed_text,
                max_new_tokens=self.args.max_tokens
            ):
                if first_token:
                    first_token_time = time.time()
                    result.first_token_time = first_token_time
                    first_token = False
                full_response.append(token)
            
            # 计算指标（非流式：从音频加载完成开始计算）
            result.ttft = (first_token_time - audio_load_time) * 1000
            result.asr_time = (last_text_time - audio_load_time) * 1000
            result.llm_prefill_time = (first_token_time - last_text_time) * 1000
            
            result.response_preview = "".join(full_response)[:100]
            
        except Exception as e:
            result.error = str(e)
            logger.error(f"非流式测试失败 {sample.sample_id}: {e}")
            import traceback
            traceback.print_exc()
        
        return result
    
    def run_all(self, samples: List[SampleInfo]) -> List[ExperimentResult]:
        """
        运行所有样本的实验
        
        Args:
            samples: 样本列表
            
        Returns:
            所有实验结果
        """
        total = len(samples)
        
        for i, sample in enumerate(samples):
            logger.info(f"\n[{i+1}/{total}] 测试样本: {sample.sample_id}")
            logger.info(f"  音频时长: {sample.audio_duration:.2f}s, 分组: {sample.duration_group}")
            
            streaming_result, non_streaming_result = self.run_single_sample(sample)
            
            self.results.append(streaming_result)
            self.results.append(non_streaming_result)
            
            # 打印单次结果
            if not streaming_result.error and not non_streaming_result.error:
                improvement = non_streaming_result.ttft - streaming_result.ttft
                ratio = improvement / non_streaming_result.ttft * 100 if non_streaming_result.ttft > 0 else 0
                logger.info(f"  流式 TTFT: {streaming_result.ttft:.2f} ms")
                logger.info(f"  非流式 TTFT: {non_streaming_result.ttft:.2f} ms")
                logger.info(f"  优化: {improvement:.2f} ms ({ratio:.1f}%)")
            
            # 每 5 个样本清理一次 GPU 内存
            if (i + 1) % 5 == 0:
                clear_gpu_memory()
                logger.debug("已清理 GPU 内存")
        
        return self.results


# =============================================================================
# 结果分析与导出
# =============================================================================

def calculate_group_statistics(results: List[ExperimentResult]) -> List[GroupStatistics]:
    """计算分组统计"""
    # 按分组整理数据
    groups_data = {}
    
    for r in results:
        if r.error:
            continue
        
        group = r.duration_group
        if group not in groups_data:
            groups_data[group] = {
                "durations": [],
                "streaming_ttft": [],
                "non_streaming_ttft": []
            }
        
        groups_data[group]["durations"].append(r.audio_duration)
        
        if r.mode == "streaming":
            groups_data[group]["streaming_ttft"].append(r.ttft)
        else:
            groups_data[group]["non_streaming_ttft"].append(r.ttft)
    
    # 计算统计
    statistics = []
    
    for group in sorted(groups_data.keys()):
        data = groups_data[group]
        
        streaming_ttft = np.array(data["streaming_ttft"])
        non_streaming_ttft = np.array(data["non_streaming_ttft"])
        
        if len(streaming_ttft) == 0 or len(non_streaming_ttft) == 0:
            continue
        
        improvement_mean = np.mean(non_streaming_ttft) - np.mean(streaming_ttft)
        improvement_ratio = improvement_mean / np.mean(non_streaming_ttft) * 100 if np.mean(non_streaming_ttft) > 0 else 0
        
        stat = GroupStatistics(
            group=group,
            sample_count=len(streaming_ttft),
            avg_duration=np.mean(data["durations"]),
            
            streaming_ttft_mean=np.mean(streaming_ttft),
            streaming_ttft_std=np.std(streaming_ttft),
            streaming_ttft_min=np.min(streaming_ttft),
            streaming_ttft_max=np.max(streaming_ttft),
            
            non_streaming_ttft_mean=np.mean(non_streaming_ttft),
            non_streaming_ttft_std=np.std(non_streaming_ttft),
            non_streaming_ttft_min=np.min(non_streaming_ttft),
            non_streaming_ttft_max=np.max(non_streaming_ttft),
            
            improvement_mean=improvement_mean,
            improvement_ratio=improvement_ratio
        )
        
        statistics.append(stat)
    
    return statistics


def save_results(
    results: List[ExperimentResult],
    statistics: List[GroupStatistics],
    output_dir: Path,
    args
):
    """保存实验结果"""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 保存详细结果 (JSON)
    results_file = output_dir / f"exp1_results_{timestamp}.json"
    results_data = {
        "config": {
            "asr_model": args.asr_model_size,
            "llm_model": args.llm_model_name,
            "asr_device": args.asr_device,
            "llm_device": args.llm_device,
            "chunk_duration_ms": args.chunk_duration,
            "max_tokens": args.max_tokens,
            "warmup_rounds": args.warmup_rounds,
            "timestamp": timestamp
        },
        "results": [asdict(r) for r in results],
        "statistics": [asdict(s) for s in statistics]
    }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"详细结果已保存: {results_file}")
    
    # 2. 保存 CSV 汇总
    csv_file = output_dir / f"exp1_summary_{timestamp}.csv"
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 写入表头
        writer.writerow([
            "sample_id", "audio_duration", "duration_group", "mode",
            "ttft_ms", "asr_time_ms", "llm_prefill_time_ms", "error"
        ])
        
        # 写入数据
        for r in results:
            writer.writerow([
                r.sample_id, f"{r.audio_duration:.2f}", r.duration_group, r.mode,
                f"{r.ttft:.2f}", f"{r.asr_time:.2f}", f"{r.llm_prefill_time:.2f}", r.error
            ])
    
    logger.info(f"CSV 汇总已保存: {csv_file}")
    
    # 3. 保存统计结果
    stats_file = output_dir / f"exp1_statistics_{timestamp}.csv"
    
    with open(stats_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        writer.writerow([
            "group", "sample_count", "avg_duration_s",
            "streaming_ttft_mean_ms", "streaming_ttft_std_ms",
            "non_streaming_ttft_mean_ms", "non_streaming_ttft_std_ms",
            "improvement_ms", "improvement_ratio_%"
        ])
        
        for s in statistics:
            writer.writerow([
                s.group, s.sample_count, f"{s.avg_duration:.2f}",
                f"{s.streaming_ttft_mean:.2f}", f"{s.streaming_ttft_std:.2f}",
                f"{s.non_streaming_ttft_mean:.2f}", f"{s.non_streaming_ttft_std:.2f}",
                f"{s.improvement_mean:.2f}", f"{s.improvement_ratio:.1f}"
            ])
    
    logger.info(f"统计结果已保存: {stats_file}")
    
    return results_file, csv_file, stats_file


def print_summary(statistics: List[GroupStatistics]):
    """打印统计摘要"""
    print("\n" + "=" * 80)
    print("实验一结果摘要：延迟与语音长度的关系")
    print("=" * 80)
    
    print(f"\n{'分组':<12} {'样本数':>8} {'平均时长':>10} {'流式TTFT':>12} {'非流式TTFT':>12} {'优化':>12}")
    print("-" * 80)
    
    for s in statistics:
        print(f"{s.group:<12} {s.sample_count:>8} {s.avg_duration:>10.2f}s "
              f"{s.streaming_ttft_mean:>10.2f}ms {s.non_streaming_ttft_mean:>10.2f}ms "
              f"{s.improvement_mean:>8.2f}ms ({s.improvement_ratio:.1f}%)")
    
    print("=" * 80)


# =============================================================================
# 主程序
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="实验一：延迟与语音长度的关系验证",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 运行完整实验
  uv run python -m experiments.scripts.run_exp_latency
  
  # 仅测试 CrossWOZ 数据集
  uv run python -m experiments.scripts.run_exp_latency --dataset crosswoz
  
  # 限制样本数进行测试
  uv run python -m experiments.scripts.run_exp_latency --max-samples 10
  
  # 指定设备和预热轮数
  uv run python -m experiments.scripts.run_exp_latency --asr-device cuda --llm-device cuda --warmup-rounds 5
        """
    )
    
    # 数据参数
    parser.add_argument('--data-dir', type=str, 
                        default='experiments/datasets/processed',
                        help='处理后的数据目录')
    parser.add_argument('--dataset', type=str, choices=['crosswoz', 'multiwoz', 'all'],
                        default='all', help='数据集选择')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='最大样本数（用于测试）')
    
    # 设备参数
    parser.add_argument('--asr-device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'], help='ASR 设备')
    parser.add_argument('--llm-device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'], help='LLM 设备')
    
    # 模型参数
    parser.add_argument('--asr-model-size', type=str, default=ASR_MODEL_NAME,
                        choices=['tiny', 'base', 'small', 'medium', 'large'],
                        help='ASR 模型大小')
    parser.add_argument('--llm-model-name', type=str, default=LLM_MODEL_NAME,
                        help='LLM 模型名称')
    
    # 实验参数
    parser.add_argument('--chunk-duration', type=int, default=500,
                        help='流式音频块时长 (ms)')
    parser.add_argument('--max-tokens', type=int, default=50,
                        help='LLM 最大生成 token 数')
    parser.add_argument('--warmup-rounds', type=int, default=3,
                        help='模型预热轮数（默认3轮）')
    
    # 输出参数
    parser.add_argument('--output-dir', type=str, 
                        default='experiments/results/exp1_latency',
                        help='结果输出目录')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='日志级别')
    
    args = parser.parse_args()
    
    # 设置日志
    set_global_log_level(args.log_level)
    
    # 处理设备参数
    if args.asr_device == 'auto':
        import torch
        args.asr_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.llm_device == 'auto':
        import torch
        args.llm_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 打印配置
    logger.info("=" * 60)
    logger.info("实验一：延迟与语音长度的关系验证")
    logger.info("=" * 60)
    logger.info(f"数据目录: {args.data_dir}")
    logger.info(f"数据集: {args.dataset}")
    logger.info(f"ASR 设备: {args.asr_device}")
    logger.info(f"LLM 设备: {args.llm_device}")
    logger.info(f"ASR 模型: {args.asr_model_size}")
    logger.info(f"LLM 模型: {args.llm_model_name}")
    logger.info(f"预热轮数: {args.warmup_rounds}")
    logger.info("=" * 60)
    
    # 路径设置
    data_dir = PROJECT_ROOT / args.data_dir
    json_dir = data_dir / "json"
    audio_dir = data_dir / "audio"
    output_dir = PROJECT_ROOT / args.output_dir
    
    # 加载样本
    dataset_filter = None if args.dataset == 'all' else args.dataset
    samples = load_samples(json_dir, audio_dir, dataset_filter, args.max_samples)
    
    if not samples:
        logger.error("没有找到有效样本，请先运行数据处理管线")
        sys.exit(1)
    
    # 初始化共享模型
    shared_models = SharedModels(args)
    shared_models.initialize()
    
    # 使用第一个样本的音频进行预热
    first_sample = samples[0]
    audio_data, sample_rate = sf.read(str(first_sample.audio_path), dtype='float32')
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    if sample_rate != 16000:
        import librosa
        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000
    
    shared_models.set_warmup_audio(audio_data, sample_rate)
    shared_models.warmup(warmup_rounds=args.warmup_rounds)
    
    # 运行实验
    experiment = LatencyExperiment(shared_models, args)
    results = experiment.run_all(samples)
    
    # 计算统计
    statistics = calculate_group_statistics(results)
    
    # 打印摘要
    print_summary(statistics)
    
    # 保存结果
    save_results(results, statistics, output_dir, args)
    
    logger.info("\n实验完成！")


if __name__ == "__main__":
    main()
