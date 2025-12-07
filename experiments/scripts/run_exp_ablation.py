#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验二：消融实验 (Ablation Study)

实验目的：
量化“流式 ASR”和“LLM KV 预填充”两个模块各自对 TTFT 的贡献。
对比配置：
1) Baseline：非流式 ASR + 非流式 LLM
2) Streaming ASR Only：流式 ASR + 非流式 LLM（等待完整文本，不做 KV 预填充）
3) Full Streaming：流式 ASR + 流式 LLM（增量 KV 预填充）

使用方式（在项目根目录下运行）：
    uv run python -m experiments.scripts.run_exp_ablation [参数]
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
    dataset: str
    language: str
    dialog_id: str
    turn_index: int
    text_length: int
    audio_duration: float
    duration_group: str
    mode: str  # baseline / streaming_asr_only / full_streaming

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
class AblationStatistics:
    """分组统计结果"""
    group: str
    sample_count: int
    avg_duration: float

    baseline_ttft_mean: float
    baseline_ttft_std: float

    streaming_asr_ttft_mean: float
    streaming_asr_ttft_std: float

    full_streaming_ttft_mean: float
    full_streaming_ttft_std: float

    asr_gain_ms: float
    kv_gain_ms: float
    total_gain_ms: float
    total_gain_ratio: float


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

    datasets = ["crosswoz", "multiwoz"] if dataset_filter is None else [dataset_filter]

    for dataset in datasets:
        dataset_json_dir = json_dir / dataset
        dataset_audio_dir = audio_dir / dataset

        if not dataset_json_dir.exists():
            logger.warning(f"数据集目录不存在: {dataset_json_dir}")
            continue

        json_files = sorted(dataset_json_dir.glob("*.json"))

        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                audio_filename = data.get('audio_file', '')
                audio_path = dataset_audio_dir / audio_filename

                if not audio_path.exists():
                    logger.debug(f"音频文件不存在，跳过: {audio_path}")
                    continue

                audio_duration = data.get('audio_duration')
                if audio_duration is None or audio_duration <= 0:
                    audio_duration = get_audio_duration(audio_path)

                if audio_duration <= 0:
                    logger.warning(f"无效的音频时长，跳过: {audio_path}")
                    continue

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

    samples.sort(key=lambda x: x.audio_duration)

    logger.info(f"加载了 {len(samples)} 个有效样本")

    group_counts: Dict[str, int] = {}
    for sample in samples:
        group_counts[sample.duration_group] = group_counts.get(sample.duration_group, 0) + 1

    logger.info("样本分组统计:")
    for group, count in sorted(group_counts.items()):
        logger.info(f"  {group}: {count} 个样本")

    return samples


def filter_by_groups(samples: List[SampleInfo], target_groups: List[str]) -> List[SampleInfo]:
    """按指定时长分组过滤样本"""
    filtered = [s for s in samples if s.duration_group in target_groups]
    logger.info(f"按分组 {target_groups} 过滤后剩余 {len(filtered)} 个样本")
    return filtered


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

        logger.info(f"加载 ASR 模型: {self.args.asr_model_size} on {self.args.asr_device}")
        logger.info(f"ASR 参数: prefix_segments={self.args.prefix_segments}, suffix_segments={self.args.suffix_segments}, threshold={self.args.recognition_threshold}")
        self.asr_processor = StreamingASRProcessor(
            model_size=self.args.asr_model_size,
            device=self.args.asr_device,
            compute_type="auto",
            recognition_threshold=self.args.recognition_threshold,
            prefix_segments=self.args.prefix_segments,
            suffix_segments_atleast=self.args.suffix_segments
        )

        logger.info(f"加载 LLM 模型: {self.args.llm_model_name} on {self.args.llm_device}")
        self.llm_inference = StreamLLMInference(
            model_name=self.args.llm_model_name,
            device=self.args.llm_device,
            eval_mode=False
        )

        logger.info("模型初始化完成")

    def set_warmup_audio(self, audio_data: np.ndarray, sample_rate: int):
        """设置预热用的音频数据"""
        self._warmup_audio = audio_data
        self._warmup_sample_rate = sample_rate

    def warmup(self, warmup_rounds: int = 3):
        """
        模型预热
        使用真实音频进行多次预热，确保推理速度稳定
        """
        if self._warmed_up:
            logger.info("模型已预热，跳过")
            return

        logger.info("=" * 60)
        logger.info(f"模型预热中（{warmup_rounds} 轮）...")
        logger.info("=" * 60)

        if self._warmup_audio is None:
            duration = 3.0
            sample_rate = 16000
            t = np.linspace(0, duration, int(duration * sample_rate))
            self._warmup_audio = (
                0.3 * np.sin(2 * np.pi * 440 * t) +
                0.1 * np.sin(2 * np.pi * 880 * t) +
                0.05 * np.random.randn(len(t))
            ).astype(np.float32)
            self._warmup_sample_rate = sample_rate

        warmup_times = {"asr": [], "llm_cache": [], "llm_generate": []}

        for round_idx in range(warmup_rounds):
            logger.info(f"  预热轮次 {round_idx + 1}/{warmup_rounds}")

            asr_start = time.time()
            asr_result = self.asr_processor.transcribe_complete_audio(
                audio_path=f"warmup_{round_idx}",
                audio_data=self._warmup_audio,
                sample_rate=self._warmup_sample_rate
            )
            asr_time = (time.time() - asr_start) * 1000
            warmup_times["asr"].append(asr_time)
            logger.debug(f"    ASR 预热: {asr_time:.2f}ms, 结果: '{asr_result['text'][:30]}...'")

            cache_start = time.time()
            kv_cache = self.llm_inference.cache_prompt("你好，这是一个测试。", is_end=True)
            cache_time = (time.time() - cache_start) * 1000
            warmup_times["llm_cache"].append(cache_time)
            logger.debug(f"    LLM Cache 预热: {cache_time:.2f}ms")

            gen_start = time.time()
            response = ""
            for token in self.llm_inference.generate(pre_cache=kv_cache, max_new_tokens=10):
                response += token
            gen_time = (time.time() - gen_start) * 1000
            warmup_times["llm_generate"].append(gen_time)
            logger.debug(f"    LLM Generate 预热: {gen_time:.2f}ms, 结果: '{response[:30]}...'")

            del kv_cache
            clear_gpu_memory()

        logger.info("-" * 40)
        logger.info("预热统计:")
        for name, times in warmup_times.items():
            logger.info(f"  {name}: {np.mean(times):.2f}ms (±{np.std(times):.2f}ms)")

        self._warmed_up = True
        logger.info("=" * 60)
        logger.info("模型预热完成！")
        logger.info("=" * 60)

    def reset_state(self):
        """重置模型状态"""
        if self.llm_inference:
            self.llm_inference.reset_timings()
        if self.asr_processor:
            self.asr_processor.timing_events.clear()
        clear_gpu_memory()


# =============================================================================
# 实验执行器
# =============================================================================


class AblationExperiment:
    """
    消融实验执行器
    """

    def __init__(self, shared_models: SharedModels, args):
        self.models = shared_models
        self.args = args
        self.results: List[ExperimentResult] = []

    def run_single_sample(self, sample: SampleInfo) -> Tuple[ExperimentResult, ExperimentResult, ExperimentResult]:
        """对单个样本运行三种配置"""
        audio_data, sample_rate = sf.read(str(sample.audio_path), dtype='float32')
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        if sample_rate != 16000:
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000

        self.models.reset_state()
        baseline_result = self._run_baseline(sample, audio_data, sample_rate)

        self.models.reset_state()
        streaming_asr_result = self._run_streaming_asr_only(sample, audio_data, sample_rate)

        self.models.reset_state()
        full_streaming_result = self._run_full_streaming(sample, audio_data, sample_rate)

        return baseline_result, streaming_asr_result, full_streaming_result

    # ------------------------------------------------------------------
    # Baseline: 非流式 ASR + 非流式 LLM
    # ------------------------------------------------------------------
    def _run_baseline(self, sample: SampleInfo, audio_data: np.ndarray, sample_rate: int) -> ExperimentResult:
        result = ExperimentResult(
            sample_id=sample.sample_id,
            dataset=sample.dataset,
            language=sample.language,
            dialog_id=sample.dialog_id,
            turn_index=sample.turn_index,
            text_length=sample.text_length,
            audio_duration=sample.audio_duration,
            duration_group=sample.duration_group,
            mode="baseline",
            ttft=0,
            asr_time=0,
            llm_prefill_time=0
        )

        try:
            start_time = time.time()
            result.start_time = start_time

            audio_load_time = time.time()
            result.audio_load_time = audio_load_time

            asr_result = self.models.asr_processor.transcribe_complete_audio(
                audio_path=str(sample.audio_path),
                audio_data=audio_data,
                sample_rate=sample_rate
            )

            transcribed_text = asr_result['text']
            last_text_time = time.time()
            result.last_text_time = last_text_time
            result.transcribed_text = transcribed_text

            full_response = []
            first_token = True
            first_token_time = 0.0

            for token in self.models.llm_inference.once_add_and_generate(
                prompt=transcribed_text,
                max_new_tokens=self.args.max_tokens
            ):
                if first_token:
                    first_token_time = time.time()
                    result.first_token_time = first_token_time
                    first_token = False
                full_response.append(token)

            # 防止 LLM 无输出时 TTFT 计算错误
            if first_token_time == 0.0:
                first_token_time = time.time()
                result.first_token_time = first_token_time
                logger.warning(f"LLM 未生成 token: {sample.sample_id}")

            result.ttft = (first_token_time - audio_load_time) * 1000
            result.asr_time = (last_text_time - audio_load_time) * 1000
            result.llm_prefill_time = (first_token_time - last_text_time) * 1000
            result.response_preview = "".join(full_response)[:100]

        except Exception as e:
            result.error = str(e)
            logger.error(f"Baseline 测试失败 {sample.sample_id}: {e}")
            import traceback
            traceback.print_exc()

        return result

    # ------------------------------------------------------------------
    # Streaming ASR only: 流式 ASR + 非流式 LLM
    # ------------------------------------------------------------------
    def _run_streaming_asr_only(
        self,
        sample: SampleInfo,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> ExperimentResult:
        import queue
        import threading

        result = ExperimentResult(
            sample_id=sample.sample_id,
            dataset=sample.dataset,
            language=sample.language,
            dialog_id=sample.dialog_id,
            turn_index=sample.turn_index,
            text_length=sample.text_length,
            audio_duration=sample.audio_duration,
            duration_group=sample.duration_group,
            mode="streaming_asr_only",
            ttft=0,
            asr_time=0,
            llm_prefill_time=0
        )

        try:
            segmenter = StreamAudioSegmenter(
                sampling_rate=sample_rate,
                silence_threshold=0.5,
                min_speech_duration_ms=500,
                min_silence_duration_ms=300,
                window_size_ms=64
            )

            chunk_duration_ms = self.args.chunk_duration
            chunk_size = int(sample_rate * chunk_duration_ms / 1000)

            audio_chunk_queue: "queue.Queue[Tuple[int, np.ndarray]]" = queue.Queue()
            audio_segment_queue: "queue.Queue[ASRAudioSegment]" = queue.Queue()

            audio_gen_done = threading.Event()
            segmentation_done = threading.Event()
            asr_done = threading.Event()

            timings = {
                "start_time": 0.0,
                "audio_end_time": 0.0,
                "last_text_time": 0.0,
            }

            transcribed_text: List[str] = []

            def audio_gen_worker():
                for i in range(0, len(audio_data), chunk_size):
                    chunk = audio_data[i:i + chunk_size]
                    chunk_id = i // chunk_size
                    audio_chunk_queue.put((chunk_id, chunk))
                    time.sleep(chunk_duration_ms / 1000)
                timings["audio_end_time"] = time.time()
                audio_gen_done.set()

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

                remaining_segment, state = segmenter.flush(state)
                if remaining_segment and len(remaining_segment.audio) > 0:
                    segment_id = f"seg_{remaining_segment.segment_id:03d}"
                    asr_segment = convert_audio_segment(remaining_segment, segment_id, False, True)
                    audio_segment_queue.put(asr_segment)

                segmentation_done.set()

            def asr_worker():
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

                        asr_cache, output_text, is_final = self.models.asr_processor.transcribe_audio_segment(asr_cache)
                        if output_text:
                            timings["last_text_time"] = time.time()
                            transcribed_text.append(output_text)

                collector_thread = threading.Thread(target=collector)
                transcriber_thread = threading.Thread(target=transcriber)
                collector_thread.start()
                transcriber_thread.start()
                collector_thread.join()
                transcriber_thread.join()
                asr_done.set()

            timings["start_time"] = time.time()
            threads = [
                threading.Thread(target=audio_gen_worker),
                threading.Thread(target=segmentation_worker),
                threading.Thread(target=asr_worker),
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # 等待 ASR 完成后再统一送入 LLM（非流式）
            full_text = " ".join(transcribed_text)
            result.transcribed_text = full_text

            first_token = True
            first_token_time = 0.0
            full_response: List[str] = []
            for token in self.models.llm_inference.once_add_and_generate(
                prompt=full_text,
                max_new_tokens=self.args.max_tokens
            ):
                if first_token:
                    first_token_time = time.time()
                    result.first_token_time = first_token_time
                    first_token = False
                full_response.append(token)

            # 防止 LLM 无输出时 TTFT 计算错误
            if first_token_time == 0.0:
                first_token_time = time.time()
                result.first_token_time = first_token_time
                logger.warning(f"LLM 未生成 token: {sample.sample_id}")

            result.audio_end_time = timings["audio_end_time"]
            result.last_text_time = timings["last_text_time"]
            result.ttft = (first_token_time - timings["audio_end_time"]) * 1000
            result.asr_time = (timings["last_text_time"] - timings["audio_end_time"]) * 1000
            result.llm_prefill_time = (first_token_time - timings["last_text_time"]) * 1000
            result.response_preview = "".join(full_response)[:100]

        except Exception as e:
            result.error = str(e)
            logger.error(f"Streaming ASR Only 测试失败 {sample.sample_id}: {e}")
            import traceback
            traceback.print_exc()

        return result

    # ------------------------------------------------------------------
    # Full streaming: 流式 ASR + 流式 LLM (KV Cache)
    # ------------------------------------------------------------------
    def _run_full_streaming(
        self,
        sample: SampleInfo,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> ExperimentResult:
        import queue
        import threading

        result = ExperimentResult(
            sample_id=sample.sample_id,
            dataset=sample.dataset,
            language=sample.language,
            dialog_id=sample.dialog_id,
            turn_index=sample.turn_index,
            text_length=sample.text_length,
            audio_duration=sample.audio_duration,
            duration_group=sample.duration_group,
            mode="full_streaming",
            ttft=0,
            asr_time=0,
            llm_prefill_time=0
        )

        try:
            segmenter = StreamAudioSegmenter(
                sampling_rate=sample_rate,
                silence_threshold=0.5,
                min_speech_duration_ms=500,
                min_silence_duration_ms=300,
                window_size_ms=64
            )

            chunk_duration_ms = self.args.chunk_duration
            chunk_size = int(sample_rate * chunk_duration_ms / 1000)

            audio_chunk_queue = queue.Queue()
            audio_segment_queue = queue.Queue()
            text_queue = queue.Queue()

            audio_gen_done = threading.Event()
            segmentation_done = threading.Event()
            asr_done = threading.Event()

            timings = {
                "start_time": 0.0,
                "audio_end_time": 0.0,
                "last_text_time": 0.0,
                "first_token_time": 0.0
            }

            full_response: List[str] = []
            transcribed_text: List[str] = []

            def audio_gen_worker():
                for i in range(0, len(audio_data), chunk_size):
                    chunk = audio_data[i:i + chunk_size]
                    chunk_id = i // chunk_size
                    audio_chunk_queue.put((chunk_id, chunk))
                    time.sleep(chunk_duration_ms / 1000)
                timings["audio_end_time"] = time.time()
                audio_gen_done.set()

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

                remaining_segment, state = segmenter.flush(state)
                if remaining_segment and len(remaining_segment.audio) > 0:
                    segment_id = f"seg_{remaining_segment.segment_id:03d}"
                    asr_segment = convert_audio_segment(remaining_segment, segment_id, False, True)
                    audio_segment_queue.put(asr_segment)
                segmentation_done.set()

            def asr_worker():
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

            def llm_worker():
                kv_cache = None
                while True:
                    try:
                        text, is_end = text_queue.get(timeout=0.1)
                    except queue.Empty:
                        if asr_done.is_set():
                            break
                        continue
                    if text or is_end:
                        kv_cache = self.models.llm_inference.cache_prompt(text, pre_cache=kv_cache, is_end=is_end)
                    if is_end:
                        first_token = True
                        for token in self.models.llm_inference.generate(pre_cache=kv_cache, max_new_tokens=self.args.max_tokens):
                            if first_token:
                                timings["first_token_time"] = time.time()
                                first_token = False
                            full_response.append(token)
                        break

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

            result.start_time = timings["start_time"]
            result.audio_end_time = timings["audio_end_time"]
            result.last_text_time = timings["last_text_time"]

            # 防止 LLM 无输出时 TTFT 计算错误
            if timings["first_token_time"] == 0.0:
                timings["first_token_time"] = time.time()
                logger.warning(f"LLM 未生成 token: {sample.sample_id}")

            result.first_token_time = timings["first_token_time"]

            result.ttft = (timings["first_token_time"] - timings["audio_end_time"]) * 1000
            result.asr_time = (timings["last_text_time"] - timings["audio_end_time"]) * 1000
            result.llm_prefill_time = (timings["first_token_time"] - timings["last_text_time"]) * 1000

            result.transcribed_text = " ".join(transcribed_text)
            result.response_preview = "".join(full_response)[:100]

        except Exception as e:
            result.error = str(e)
            logger.error(f"Full Streaming 测试失败 {sample.sample_id}: {e}")
            import traceback
            traceback.print_exc()

        return result

    # ------------------------------------------------------------------
    def run_all(self, samples: List[SampleInfo]) -> List[ExperimentResult]:
        total = len(samples)
        for i, sample in enumerate(samples):
            logger.info(f"\n[{i + 1}/{total}] 测试样本: {sample.sample_id}")
            logger.info(f"  音频时长: {sample.audio_duration:.2f}s, 分组: {sample.duration_group}")

            baseline_result, streaming_asr_result, full_streaming_result = self.run_single_sample(sample)

            self.results.extend([baseline_result, streaming_asr_result, full_streaming_result])

            if not baseline_result.error and not streaming_asr_result.error and not full_streaming_result.error:
                logger.info(f"  Baseline TTFT: {baseline_result.ttft:.2f} ms")
                logger.info(f"  Streaming ASR TTFT: {streaming_asr_result.ttft:.2f} ms")
                logger.info(f"  Full Streaming TTFT: {full_streaming_result.ttft:.2f} ms")

            if (i + 1) % 5 == 0:
                clear_gpu_memory()
                logger.debug("已清理 GPU 内存")

        return self.results


# =============================================================================
# 结果分析与导出
# =============================================================================


def calculate_group_statistics(results: List[ExperimentResult]) -> List[AblationStatistics]:
    grouped: Dict[str, Dict[str, List[float]]] = {}

    for r in results:
        if r.error:
            continue
        group = r.duration_group
        grouped.setdefault(group, {
            "durations": [],
            "baseline": [],
            "streaming_asr_only": [],
            "full_streaming": [],
        })
        grouped[group]["durations"].append(r.audio_duration)
        grouped[group][r.mode].append(r.ttft)

    statistics: List[AblationStatistics] = []
    for group in sorted(grouped.keys()):
        data = grouped[group]
        baseline = np.array(data["baseline"])
        streaming_asr = np.array(data["streaming_asr_only"])
        full_streaming = np.array(data["full_streaming"])

        if len(baseline) == 0 or len(streaming_asr) == 0 or len(full_streaming) == 0:
            continue

        asr_gain = np.mean(baseline) - np.mean(streaming_asr)
        kv_gain = np.mean(streaming_asr) - np.mean(full_streaming)
        total_gain = np.mean(baseline) - np.mean(full_streaming)
        total_ratio = total_gain / np.mean(baseline) * 100 if np.mean(baseline) > 0 else 0

        stat = AblationStatistics(
            group=group,
            sample_count=len(baseline),
            avg_duration=np.mean(data["durations"]),
            baseline_ttft_mean=np.mean(baseline),
            baseline_ttft_std=np.std(baseline),
            streaming_asr_ttft_mean=np.mean(streaming_asr),
            streaming_asr_ttft_std=np.std(streaming_asr),
            full_streaming_ttft_mean=np.mean(full_streaming),
            full_streaming_ttft_std=np.std(full_streaming),
            asr_gain_ms=asr_gain,
            kv_gain_ms=kv_gain,
            total_gain_ms=total_gain,
            total_gain_ratio=total_ratio
        )
        statistics.append(stat)

    return statistics


def calculate_overall_statistics(results: List[ExperimentResult]) -> Dict[str, Any]:
    """整体统计（不分组）"""
    buckets = {
        "baseline": [],
        "streaming_asr_only": [],
        "full_streaming": []
    }
    for r in results:
        if r.error:
            continue
        buckets[r.mode].append(r.ttft)

    overall = {}
    for mode, vals in buckets.items():
        if len(vals) == 0:
            continue
        arr = np.array(vals)
        overall[mode] = {
            "mean_ms": float(np.mean(arr)),
            "std_ms": float(np.std(arr)),
            "min_ms": float(np.min(arr)),
            "max_ms": float(np.max(arr)),
            "count": len(arr)
        }

    if overall.get("baseline") and overall.get("full_streaming"):
        total_gain = overall["baseline"]["mean_ms"] - overall["full_streaming"]["mean_ms"]
        overall["total_gain_ms"] = float(total_gain)
        overall["total_gain_ratio_pct"] = float(
            total_gain / overall["baseline"]["mean_ms"] * 100 if overall["baseline"]["mean_ms"] > 0 else 0
        )
    return overall


def compute_sample_gains(results: List[ExperimentResult]) -> List[Dict[str, Any]]:
    """
    逐样本拆分增益，便于论文表格直接引用
    """
    by_sample: Dict[str, Dict[str, ExperimentResult]] = {}
    for r in results:
        if r.error:
            continue
        by_sample.setdefault(r.sample_id, {})[r.mode] = r

    gains: List[Dict[str, Any]] = []
    for sample_id, modes in by_sample.items():
        baseline = modes.get("baseline")
        streaming_asr = modes.get("streaming_asr_only")
        full_streaming = modes.get("full_streaming")
        if not (baseline and streaming_asr and full_streaming):
            continue

        asr_gain = baseline.ttft - streaming_asr.ttft
        kv_gain = streaming_asr.ttft - full_streaming.ttft
        total_gain = baseline.ttft - full_streaming.ttft
        total_ratio = total_gain / baseline.ttft * 100 if baseline.ttft > 0 else 0

        gains.append({
            "sample_id": sample_id,
            "dataset": baseline.dataset,
            "language": baseline.language,
            "dialog_id": baseline.dialog_id,
            "turn_index": baseline.turn_index,
            "text_length": baseline.text_length,
            "duration_group": baseline.duration_group,
            "audio_duration": baseline.audio_duration,
            "baseline_ttft": baseline.ttft,
            "streaming_asr_ttft": streaming_asr.ttft,
            "full_streaming_ttft": full_streaming.ttft,
            "asr_gain": asr_gain,
            "kv_gain": kv_gain,
            "total_gain": total_gain,
            "total_ratio": total_ratio
        })
    return gains


def save_results(
    results: List[ExperimentResult],
    statistics: List[AblationStatistics],
    output_dir: Path,
    args,
    sample_gains: List[Dict[str, Any]],
    overall_stats: Dict[str, Any]
):
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_file = output_dir / f"exp2_results_{timestamp}.json"
    results_data = {
        "config": {
            "asr_model": args.asr_model_size,
            "llm_model": args.llm_model_name,
            "asr_device": args.asr_device,
            "llm_device": args.llm_device,
            "chunk_duration_ms": args.chunk_duration,
            "max_tokens": args.max_tokens,
            "warmup_rounds": args.warmup_rounds,
            "target_groups": args.duration_groups,
            "prefix_segments": args.prefix_segments,
            "suffix_segments": args.suffix_segments,
            "recognition_threshold": args.recognition_threshold,
            "timestamp": timestamp
        },
        "results": [asdict(r) for r in results],
        "statistics": [asdict(s) for s in statistics],
        "sample_gains": sample_gains,
        "overall_statistics": overall_stats
    }
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    logger.info(f"详细结果已保存: {results_file}")

    csv_file = output_dir / f"exp2_summary_{timestamp}.csv"
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_id", "dataset", "language", "dialog_id", "turn_index", "text_length",
            "audio_duration", "duration_group", "mode",
            "ttft_ms", "asr_time_ms", "llm_prefill_time_ms", "error"
        ])
        for r in results:
            writer.writerow([
                r.sample_id, r.dataset, r.language, r.dialog_id, r.turn_index, r.text_length,
                f"{r.audio_duration:.2f}", r.duration_group, r.mode,
                f"{r.ttft:.2f}", f"{r.asr_time:.2f}", f"{r.llm_prefill_time:.2f}", r.error
            ])
    logger.info(f"CSV 汇总已保存: {csv_file}")

    stats_file = output_dir / f"exp2_statistics_{timestamp}.csv"
    with open(stats_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "group", "sample_count", "avg_duration_s",
            "baseline_ttft_mean_ms", "streaming_asr_ttft_mean_ms", "full_streaming_ttft_mean_ms",
            "asr_gain_ms", "kv_gain_ms", "total_gain_ms", "total_gain_ratio_%"
        ])
        for s in statistics:
            writer.writerow([
                s.group, s.sample_count, f"{s.avg_duration:.2f}",
                f"{s.baseline_ttft_mean:.2f}", f"{s.streaming_asr_ttft_mean:.2f}", f"{s.full_streaming_ttft_mean:.2f}",
                f"{s.asr_gain_ms:.2f}", f"{s.kv_gain_ms:.2f}", f"{s.total_gain_ms:.2f}", f"{s.total_gain_ratio:.1f}"
            ])
    logger.info(f"统计结果已保存: {stats_file}")

    gains_file = output_dir / f"exp2_gains_{timestamp}.csv"
    with open(gains_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_id", "dataset", "language", "dialog_id", "turn_index", "text_length",
            "duration_group", "audio_duration_s",
            "baseline_ttft_ms", "streaming_asr_ttft_ms", "full_streaming_ttft_ms",
            "asr_gain_ms", "kv_gain_ms", "total_gain_ms", "total_gain_ratio_%"
        ])
        for g in sample_gains:
            writer.writerow([
                g["sample_id"], g["dataset"], g["language"], g["dialog_id"], g["turn_index"], g["text_length"],
                g["duration_group"], f"{g['audio_duration']:.2f}",
                f"{g['baseline_ttft']:.2f}", f"{g['streaming_asr_ttft']:.2f}", f"{g['full_streaming_ttft']:.2f}",
                f"{g['asr_gain']:.2f}", f"{g['kv_gain']:.2f}", f"{g['total_gain']:.2f}", f"{g['total_ratio']:.1f}"
            ])
    logger.info(f"增益拆分已保存: {gains_file}")

    return results_file, csv_file, stats_file, gains_file


def print_summary(statistics: List[AblationStatistics]):
    print("\n" + "=" * 80)
    print("实验二结果摘要：消融实验")
    print("=" * 80)
    print(f"\n{'分组':<12} {'样本数':>8} {'平均时长':>10} {'Baseline':>12} {'StreamASR':>12} {'FullStream':>12} {'ASR增益':>12} {'KV增益':>10}")
    print("-" * 80)
    for s in statistics:
        print(f"{s.group:<12} {s.sample_count:>8} {s.avg_duration:>10.2f}s "
              f"{s.baseline_ttft_mean:>10.2f}ms {s.streaming_asr_ttft_mean:>10.2f}ms "
              f"{s.full_streaming_ttft_mean:>10.2f}ms {s.asr_gain_ms:>10.2f}ms {s.kv_gain_ms:>10.2f}ms")
    print("=" * 80)


# =============================================================================
# 主程序
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="实验二：消融实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 运行默认（长语音组）消融实验
  uv run python -m experiments.scripts.run_exp_ablation

  # 指定数据集、样本数和分组
  uv run python -m experiments.scripts.run_exp_ablation --dataset crosswoz --max-samples 10 --duration-groups long very_long

  # 指定设备与预热轮数
  uv run python -m experiments.scripts.run_exp_ablation --asr-device cuda --llm-device cuda --warmup-rounds 5
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
    parser.add_argument('--duration-groups', nargs='+', default=['long'],
                        choices=list(DURATION_GROUPS.keys()),
                        help='按时长分组筛选样本，默认只测试 long 组')

    # 设备参数
    parser.add_argument('--asr-device', type=str, default='auto',
                        help='ASR 设备 (auto/cuda/cuda:0/cuda:1/cpu)')
    parser.add_argument('--llm-device', type=str, default='auto',
                        help='LLM 设备 (auto/cuda/cuda:0/cuda:1/cpu)')

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
    
    # ASR 流式参数
    parser.add_argument('--prefix-segments', type=int, default=1,
                        help='ASR 前缀段数（影响上下文和延迟，默认1）')
    parser.add_argument('--suffix-segments', type=int, default=1,
                        help='ASR 后缀段数（影响准确率和延迟，默认1）')
    parser.add_argument('--recognition-threshold', type=float, default=2.0,
                        help='ASR 识别阈值（秒），队列总长度达到此值时开始识别')

    # 输出参数
    parser.add_argument('--output-dir', type=str,
                        default='experiments/results/exp2_ablation',
                        help='结果输出目录')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='日志级别')

    args = parser.parse_args()

    set_global_log_level(args.log_level)

    # 处理设备参数（支持 cuda:0, cuda:1 等具体设备）
    import torch
    if args.asr_device == 'auto':
        args.asr_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.llm_device == 'auto':
        args.llm_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger.info("=" * 60)
    logger.info("实验二：消融实验")
    logger.info("=" * 60)
    logger.info(f"数据目录: {args.data_dir}")
    logger.info(f"数据集: {args.dataset}")
    logger.info(f"目标分组: {args.duration_groups}")
    logger.info(f"ASR 设备: {args.asr_device}")
    logger.info(f"LLM 设备: {args.llm_device}")
    logger.info(f"ASR 模型: {args.asr_model_size}")
    logger.info(f"LLM 模型: {args.llm_model_name}")
    logger.info(f"预热轮数: {args.warmup_rounds}")
    logger.info(f"ASR prefix_segments: {args.prefix_segments}")
    logger.info(f"ASR suffix_segments: {args.suffix_segments}")
    logger.info(f"ASR recognition_threshold: {args.recognition_threshold}s")
    logger.info("=" * 60)

    data_dir = PROJECT_ROOT / args.data_dir
    json_dir = data_dir / "json"
    audio_dir = data_dir / "audio"
    output_dir = PROJECT_ROOT / args.output_dir

    dataset_filter = None if args.dataset == 'all' else args.dataset
    samples = load_samples(json_dir, audio_dir, dataset_filter, args.max_samples)
    if not samples:
        logger.error("没有找到有效样本，请先运行数据处理管线")
        sys.exit(1)

    samples = filter_by_groups(samples, args.duration_groups)
    if not samples:
        logger.error("指定分组无可用样本，请调整 --duration-groups 或生成更多数据")
        sys.exit(1)

    shared_models = SharedModels(args)
    shared_models.initialize()

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

    experiment = AblationExperiment(shared_models, args)
    results = experiment.run_all(samples)

    statistics = calculate_group_statistics(results)
    sample_gains = compute_sample_gains(results)
    overall_stats = calculate_overall_statistics(results)
    print_summary(statistics)
    save_results(results, statistics, output_dir, args, sample_gains, overall_stats)

    logger.info("\n实验完成！")


if __name__ == "__main__":
    main()

