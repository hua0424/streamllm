#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验三：准确率与质量验证 (ASR Accuracy & Quality)

目标：
1) 对比非流式 ASR 与流式 ASR 的识别准确率（WER/CER）
2) 关注长语音场景下流式处理的精度影响

运行示例（项目根目录）：
    uv run python -m experiments.scripts.run_exp_quality --dataset all --max-samples 200

关键设计：
- 增量保存：每处理 N 个样本自动保存检查点，防止中断丢失数据
- 断点续传：支持从上次中断位置继续运行
"""

import argparse
import json
import sys
import time
import wave
import gc
import math
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import soundfile as sf

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# 项目模块
from src.utils.logging_utils import get_logger, set_global_log_level
from src.asr.streamaudio_segmenter import StreamAudioSegmenter
from src.asr.faster_whisper_streamer import StreamingASRProcessor, ASRCache
from src.asr.run_stream_asr_test import convert_audio_segment
from src.config import ASR_MODEL_NAME

logger = get_logger(__name__)

# =============================================================================
# 时长分组
# =============================================================================

DURATION_GROUPS = {
    "short": (0, 5),
    "medium": (5, 15),
    "long": (15, 30),
    "very_long": (30, 60),
    "extra_long": (60, float("inf")),
}


def get_duration_group(duration: float) -> str:
    for name, (lo, hi) in DURATION_GROUPS.items():
        if lo <= duration < hi:
            return name
    return "extra_long"


def _mark_error(result, reason: str) -> None:
    if not reason:
        return
    if not getattr(result, "error", ""):
        result.error = reason
    elif reason not in result.error:
        result.error = f"{result.error}; {reason}"


# =============================================================================
# 工具函数：WER / CER
# =============================================================================


def _levenshtein(seq1: List[str], seq2: List[str]) -> int:
    """简单编辑距离实现"""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # 删除
                dp[i][j - 1] + 1,      # 插入
                dp[i - 1][j - 1] + cost  # 替换
            )
    return dp[m][n]


def normalize_text(text: str, remove_punctuation: bool = True) -> str:
    """
    文本归一化，用于准确率计算前的预处理
    
    Args:
        text: 原始文本
        remove_punctuation: 是否移除标点符号（默认True）
        
    Returns:
        归一化后的文本
    
    Notes:
        - 流式ASR可能缺失标点符号，为公平比较需要统一移除
        - 这是语音识别评估的标准做法
    """
    import re
    import unicodedata
    
    # 先去除首尾空白
    text = text.strip()
    
    if remove_punctuation:
        # 移除中英文标点符号
        # 中文标点
        zh_punctuation = r'[，。！？、；：""''（）【】《》—…·]'
        # 英文标点
        en_punctuation = r'[,.!?;:\'"()\[\]{}<>\-_+=*/\\@#$%^&|`~]'
        
        text = re.sub(zh_punctuation, '', text)
        text = re.sub(en_punctuation, '', text)
    
    # 移除多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def cer(ref: str, hyp: str, normalize: bool = True) -> float:
    """
    字符级错误率 (Character Error Rate)
    
    Args:
        ref: 参考文本
        hyp: 假设文本（识别结果）
        normalize: 是否在计算前进行文本归一化（移除标点）
    """
    if normalize:
        ref = normalize_text(ref)
        hyp = normalize_text(hyp)
    
    ref_chars = list(ref.strip())
    hyp_chars = list(hyp.strip())
    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0
    return _levenshtein(ref_chars, hyp_chars) / len(ref_chars)


def wer(ref: str, hyp: str, normalize: bool = True) -> float:
    """
    词级错误率 (Word Error Rate)
    以空格切分；中文场景可视作每字空格切分后再计算
    
    Args:
        ref: 参考文本
        hyp: 假设文本（识别结果）
        normalize: 是否在计算前进行文本归一化（移除标点）
    """
    if normalize:
        ref = normalize_text(ref)
        hyp = normalize_text(hyp)
    
    ref_words = ref.strip().split()
    hyp_words = hyp.strip().split()
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    return _levenshtein(ref_words, hyp_words) / len(ref_words)


def zh_to_word_seq(text: str, normalize: bool = True) -> str:
    """
    将中文字符串转为空格分隔的"词"(逐字)，便于 WER 统一计算
    
    Args:
        text: 原始中文文本
        normalize: 是否先进行文本归一化（移除标点）
    """
    if normalize:
        text = normalize_text(text)
    return " ".join(list(text.replace(" ", "")))


# =============================================================================
# 数据结构
# =============================================================================


@dataclass
class SampleInfo:
    sample_id: str
    dialog_id: str
    turn_index: int
    text: str  # ground truth
    text_length: int
    audio_file: str
    audio_path: Path
    audio_duration: float
    language: str
    dataset: str
    duration_group: str


@dataclass
class ExperimentResult:
    sample_id: str
    dataset: str
    language: str
    dialog_id: str
    turn_index: int
    text_length: int
    audio_duration: float
    duration_group: str
    mode: str  # streaming / non-streaming

    transcript: str
    wer: float
    cer: float

    asr_time_ms: float = 0.0
    start_time: float = 0.0
    audio_end_time: float = 0.0
    audio_load_time: float = 0.0
    last_text_time: float = 0.0

    error: str = ""


@dataclass
class Statistics:
    scope: str  # overall / dataset name / language
    sample_count: int
    wer_mean: float
    wer_std: float
    cer_mean: float
    cer_std: float
    avg_duration: float


# =============================================================================
# 工具
# =============================================================================


def clear_gpu_memory():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


def get_audio_duration(audio_path: Path) -> float:
    try:
        with wave.open(str(audio_path), 'rb') as wav_file:
            frames = wav_file.getnframes()
            sr = wav_file.getframerate()
            return frames / sr
    except Exception as e:
        logger.warning(f"无法读取音频时长 {audio_path}: {e}")
        return -1


def load_samples(json_dir: Path, audio_dir: Path, dataset_filter: Optional[str], max_samples: Optional[int]) -> List[SampleInfo]:
    samples: List[SampleInfo] = []
    datasets = ["crosswoz", "multiwoz"] if dataset_filter is None else [dataset_filter]
    for dataset in datasets:
        jdir = json_dir / dataset
        adir = audio_dir / dataset
        if not jdir.exists():
            logger.warning(f"数据集目录不存在: {jdir}")
            continue
        json_files = sorted(jdir.glob("*.json"))
        for jf in json_files:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                apath = adir / data.get("audio_file", "")
                if not apath.exists():
                    continue
                audio_duration = data.get("audio_duration") or get_audio_duration(apath)
                if audio_duration <= 0:
                    continue
                samples.append(
                    SampleInfo(
                        sample_id=data["sample_id"],
                        dialog_id=data["dialog_id"],
                        turn_index=data["turn_index"],
                        text=data["text"],
                        text_length=data.get("text_length", len(data["text"])),
                        audio_file=data["audio_file"],
                        audio_path=apath,
                        audio_duration=audio_duration,
                        language=data["language"],
                        dataset=data["dataset"],
                        duration_group=get_duration_group(audio_duration),
                    )
                )
            except Exception as e:
                logger.error(f"加载样本失败 {jf}: {e}")
                continue
            if max_samples and len(samples) >= max_samples:
                return samples[:max_samples]
    samples.sort(key=lambda x: x.audio_duration)
    logger.info(f"加载了 {len(samples)} 个样本")
    return samples


def stratified_sample_by_group(
    samples: List[SampleInfo],
    target_groups: List[str],
    samples_per_group: Optional[int] = None,
    max_total: Optional[int] = None,
) -> List[SampleInfo]:
    """
    按时长分组分层抽样，保障 medium/long/very_long 均衡。
    优先取分组内时长较短的样本以降低运行成本（稳定复现）。
    """
    buckets: Dict[str, List[SampleInfo]] = {g: [] for g in target_groups}
    for s in samples:
        if s.duration_group in buckets:
            buckets[s.duration_group].append(s)

    for g in buckets:
        buckets[g].sort(key=lambda x: x.audio_duration)

    if samples_per_group is None and max_total:
        samples_per_group = math.ceil(max_total / len(target_groups))

    selected: List[SampleInfo] = []
    for g in target_groups:
        grp = buckets.get(g, [])
        take_n = samples_per_group or len(grp)
        selected.extend(grp[:take_n])

    if max_total:
        selected = selected[:max_total]

    logger.info(f"分层抽样分组 {target_groups}, 每组取 {samples_per_group or 'ALL'}, 最终 {len(selected)} 条")
    return selected


# =============================================================================
# 共享 ASR
# =============================================================================


class SharedASR:
    def __init__(self, args):
        self.args = args
        self.asr: Optional[StreamingASRProcessor] = None
        self._warmed = False
        self._warm_audio = None
        self._warm_sr = 16000

    def initialize(self):
        logger.info("加载 ASR 模型...")
        logger.info(f"ASR 参数: prefix_segments={self.args.prefix_segments}, suffix_segments={self.args.suffix_segments}, threshold={self.args.recognition_threshold}")
        self.asr = StreamingASRProcessor(
            model_size=self.args.asr_model_size,
            device=self.args.asr_device,
            compute_type="auto",
            recognition_threshold=self.args.recognition_threshold,
            prefix_segments=self.args.prefix_segments,
            suffix_segments_atleast=self.args.suffix_segments,
        )

    def set_warmup_audio(self, audio: np.ndarray, sr: int):
        self._warm_audio = audio
        self._warm_sr = sr

    def warmup(self, rounds: int = 2):
        if self._warmed:
            return
        if self._warm_audio is None:
            duration = 2.0
            sr = 16000
            t = np.linspace(0, duration, int(duration * sr))
            self._warm_audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
            self._warm_sr = sr
        for i in range(rounds):
            _ = self.asr.transcribe_complete_audio(
                audio_path=f"warmup_{i}",
                audio_data=self._warm_audio,
                sample_rate=self._warm_sr,
            )
            clear_gpu_memory()
        self._warmed = True

    def reset(self):
        if self.asr:
            self.asr.timing_events.clear()
        clear_gpu_memory()


# =============================================================================
# 检查点管理（增量保存与断点续传）
# =============================================================================

def get_checkpoint_path(output_dir: Path) -> Path:
    """获取检查点文件路径"""
    return output_dir / "checkpoint.json"


def load_checkpoint(output_dir: Path) -> Tuple[List[ExperimentResult], set]:
    """
    加载检查点
    
    Returns:
        (已保存的结果列表, 已完成的样本ID集合)
    """
    checkpoint_path = get_checkpoint_path(output_dir)
    
    if not checkpoint_path.exists():
        return [], set()
    
    try:
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = []
        for r in data.get('results', []):
            result = ExperimentResult(**r)
            results.append(result)
        
        completed_ids = set(data.get('completed_sample_ids', []))
        
        logger.info(f"加载检查点: {len(results)} 条结果, {len(completed_ids)} 个已完成样本")
        return results, completed_ids
        
    except Exception as e:
        logger.warning(f"加载检查点失败: {e}, 将从头开始")
        return [], set()


def save_checkpoint(
    results: List[ExperimentResult],
    completed_sample_ids: set,
    output_dir: Path,
    config: Dict[str, Any]
):
    """
    保存检查点
    
    Args:
        results: 当前所有结果
        completed_sample_ids: 已完成的样本ID集合
        output_dir: 输出目录
        config: 实验配置
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = get_checkpoint_path(output_dir)
    
    checkpoint_data = {
        'config': config,
        'results': [asdict(r) for r in results],
        'completed_sample_ids': list(completed_sample_ids),
        'last_update': datetime.now().isoformat()
    }
    
    # 写入临时文件后重命名，确保原子性
    temp_path = checkpoint_path.with_suffix('.tmp')
    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
    
    temp_path.replace(checkpoint_path)
    logger.info(f"检查点已保存: {len(results)} 条结果")


# =============================================================================
# 实验执行
# =============================================================================


class QualityExperiment:
    def __init__(self, shared_asr: SharedASR, args):
        self.models = shared_asr
        self.args = args
        self.results: List[ExperimentResult] = []

    def run_all(
        self, 
        samples: List[SampleInfo],
        output_dir: Path,
        batch_size: int = 100,
        config: Dict[str, Any] = None
    ) -> List[ExperimentResult]:
        """
        运行所有样本的实验（支持增量保存和断点续传）
        
        Args:
            samples: 样本列表
            output_dir: 输出目录（用于保存检查点）
            batch_size: 每处理多少样本保存一次检查点
            config: 实验配置（用于保存到检查点）
            
        Returns:
            所有实验结果
        """
        # 加载检查点
        existing_results, completed_ids = load_checkpoint(output_dir)
        self.results = existing_results
        
        # 过滤已完成的样本
        pending_samples = [s for s in samples if s.sample_id not in completed_ids]
        
        if len(pending_samples) < len(samples):
            logger.info(f"断点续传: 跳过 {len(samples) - len(pending_samples)} 个已完成样本")
        
        total_original = len(samples)
        processed_in_batch = 0
        
        for i, s in enumerate(pending_samples):
            # 显示进度时包含已完成的数量
            done_count = len(completed_ids) + i + 1
            logger.info(f"[{done_count}/{total_original}] 样本 {s.sample_id} ({s.language}) 长度 {s.audio_duration:.2f}s")
            audio, sr = sf.read(str(s.audio_path), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sr = 16000

            self.models.reset()
            non_stream = self._run_non_streaming(s, audio, sr)

            self.models.reset()
            streaming = self._run_streaming(s, audio, sr)

            self.results.extend([non_stream, streaming])
            completed_ids.add(s.sample_id)
            processed_in_batch += 1

            if (i + 1) % 10 == 0:
                clear_gpu_memory()
            
            # 每 batch_size 个样本保存一次检查点
            if processed_in_batch >= batch_size:
                save_checkpoint(self.results, completed_ids, output_dir, config or {})
                processed_in_batch = 0
                logger.info(f"✓ 已保存检查点 ({len(completed_ids)}/{total_original} 完成)")
        
        # 最后保存一次（确保所有结果都被保存）
        if processed_in_batch > 0:
            save_checkpoint(self.results, completed_ids, output_dir, config or {})
            logger.info(f"✓ 最终检查点已保存 ({len(completed_ids)}/{total_original} 完成)")
        
        return self.results

    def _run_non_streaming(self, sample: SampleInfo, audio: np.ndarray, sr: int) -> ExperimentResult:
        res = ExperimentResult(
            sample_id=sample.sample_id,
            dataset=sample.dataset,
            language=sample.language,
            dialog_id=sample.dialog_id,
            turn_index=sample.turn_index,
            text_length=sample.text_length,
            audio_duration=sample.audio_duration,
            duration_group=sample.duration_group,
            mode="non-streaming",
            transcript="",
            wer=0.0,
            cer=0.0,
        )
        try:
            res.start_time = time.time()
            res.audio_load_time = time.time()
            asr_out = self.models.asr.transcribe_complete_audio(
                audio_path=str(sample.audio_path),
                audio_data=audio,
                sample_rate=sr,
            )
            res.last_text_time = time.time()
            res.transcript = asr_out.get("text", "")
            ref_text = sample.text
            if sample.language.lower().startswith("zh"):
                res.wer = wer(zh_to_word_seq(ref_text), zh_to_word_seq(res.transcript))
            else:
                res.wer = wer(ref_text, res.transcript)
            res.cer = cer(ref_text, res.transcript)
            res.asr_time_ms = (res.last_text_time - res.audio_load_time) * 1000
            if not res.transcript:
                _mark_error(res, "asr_no_text")
            if res.asr_time_ms < 0:
                res.asr_time_ms = max(0.0, res.asr_time_ms)
                _mark_error(res, "invalid_timing")
        except Exception as e:
            res.error = str(e)
            logger.error(f"非流式失败 {sample.sample_id}: {e}")
        return res

    def _run_streaming(self, sample: SampleInfo, audio: np.ndarray, sr: int) -> ExperimentResult:
        import queue
        import threading

        res = ExperimentResult(
            sample_id=sample.sample_id,
            dataset=sample.dataset,
            language=sample.language,
            dialog_id=sample.dialog_id,
            turn_index=sample.turn_index,
            text_length=sample.text_length,
            audio_duration=sample.audio_duration,
            duration_group=sample.duration_group,
            mode="streaming",
            transcript="",
            wer=0.0,
            cer=0.0,
        )

        try:
            segmenter = StreamAudioSegmenter(
                sampling_rate=sr,
                silence_threshold=0.5,
                min_speech_duration_ms=500,
                min_silence_duration_ms=300,
                window_size_ms=64,
            )
            chunk_ms = self.args.chunk_duration
            chunk_size = int(sr * chunk_ms / 1000)

            audio_q = queue.Queue()
            seg_q = queue.Queue()
            done_audio = threading.Event()
            done_seg = threading.Event()

            timings = {"start": 0.0, "audio_end": 0.0, "last_text": 0.0}
            texts: List[str] = []

            def audio_worker():
                for i in range(0, len(audio), chunk_size):
                    chunk = audio[i : i + chunk_size]
                    audio_q.put(chunk)
                    time.sleep(chunk_ms / 1000)
                timings["audio_end"] = time.time()
                done_audio.set()

            def seg_worker():
                state = segmenter.create_state()
                while True:
                    try:
                        chunk = audio_q.get(timeout=0.1)
                    except queue.Empty:
                        if done_audio.is_set():
                            break
                        continue
                    stream_seg, state = segmenter.process_audio(chunk, state)
                    if stream_seg:
                        seg = convert_audio_segment(
                            stream_seg,
                            segment_id=f"seg_{stream_seg.segment_id:03d}",
                            is_start=stream_seg.segment_id == 1,
                            is_final=False,
                        )
                        seg_q.put(seg)
                rem, state = segmenter.flush(state)
                if rem and len(rem.audio) > 0:
                    seg = convert_audio_segment(
                        rem,
                        segment_id=f"seg_{rem.segment_id:03d}",
                        is_start=False,
                        is_final=True,
                    )
                    seg_q.put(seg)
                done_seg.set()

            def asr_worker():
                asr_cache = ASRCache()
                final_received = False
                while True:
                    try:
                        seg = seg_q.get(timeout=0.1)
                    except queue.Empty:
                        if done_seg.is_set():
                            break
                        continue
                    asr_cache.add_segment(seg)
                    if seg.is_final:
                        final_received = True
                    if asr_cache.is_processing():
                        continue
                    asr_cache, text, is_final = self.models.asr.transcribe_audio_segment(asr_cache)
                    if text:
                        timings["last_text"] = time.time()
                        texts.append(text)
                    if is_final and final_received:
                        break

            timings["start"] = time.time()
            threads = [
                threading.Thread(target=audio_worker),
                threading.Thread(target=seg_worker),
                threading.Thread(target=asr_worker),
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            res.start_time = timings["start"]
            res.audio_end_time = timings["audio_end"]

            # 防止 ASR 无输出时时间戳计算错误
            if timings["last_text"] == 0.0:
                timings["last_text"] = timings["audio_end"]
                logger.warning(f"ASR 未输出文本: {sample.sample_id}")
                _mark_error(res, "asr_no_text")

            res.last_text_time = timings["last_text"]
            res.asr_time_ms = (timings["last_text"] - timings["audio_end"]) * 1000
            res.transcript = " ".join(texts)

            ref_text = sample.text
            if sample.language.lower().startswith("zh"):
                res.wer = wer(zh_to_word_seq(ref_text), zh_to_word_seq(res.transcript))
            else:
                res.wer = wer(ref_text, res.transcript)
            res.cer = cer(ref_text, res.transcript)
            if res.asr_time_ms < 0:
                res.asr_time_ms = max(0.0, res.asr_time_ms)
                _mark_error(res, "invalid_timing")

        except Exception as e:
            res.error = str(e)
            logger.error(f"流式失败 {sample.sample_id}: {e}")
        return res


# =============================================================================
# 统计
# =============================================================================


def compute_statistics(results: List[ExperimentResult], key: str) -> List[Statistics]:
    stats: List[Statistics] = []
    buckets: Dict[str, List[ExperimentResult]] = {}
    for r in results:
        if r.error:
            continue
        if key == "overall":
            buckets.setdefault("overall", []).append(r)
        else:
            tag = getattr(r, key)
            buckets.setdefault(tag, []).append(r)
    for scope, items in buckets.items():
        wer_arr = np.array([x.wer for x in items])
        cer_arr = np.array([x.cer for x in items])
        dur = np.array([x.audio_duration for x in items])
        stats.append(
            Statistics(
                scope=scope,
                sample_count=len(items),
                wer_mean=float(np.mean(wer_arr)),
                wer_std=float(np.std(wer_arr)),
                cer_mean=float(np.mean(cer_arr)),
                cer_std=float(np.std(cer_arr)),
                avg_duration=float(np.mean(dur)),
            )
        )
    return stats


def compute_mode_statistics(results: List[ExperimentResult]) -> List[Statistics]:
    """按 mode (streaming/non-streaming) 分别统计，论文核心对比数据"""
    stats: List[Statistics] = []
    buckets: Dict[str, List[ExperimentResult]] = {}
    for r in results:
        if r.error:
            continue
        buckets.setdefault(r.mode, []).append(r)
    for mode, items in buckets.items():
        wer_arr = np.array([x.wer for x in items])
        cer_arr = np.array([x.cer for x in items])
        dur = np.array([x.audio_duration for x in items])
        stats.append(
            Statistics(
                scope=mode,
                sample_count=len(items),
                wer_mean=float(np.mean(wer_arr)),
                wer_std=float(np.std(wer_arr)),
                cer_mean=float(np.mean(cer_arr)),
                cer_std=float(np.std(cer_arr)),
                avg_duration=float(np.mean(dur)),
            )
        )
    return stats


def save_results(results: List[ExperimentResult], output_dir: Path, args, stats_dataset, stats_language, stats_overall, stats_mode):
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_file = output_dir / f"exp3_results_{ts}.json"
    payload = {
        "config": {
            "asr_model": args.asr_model_size,
            "asr_device": args.asr_device,
            "chunk_duration_ms": args.chunk_duration,
            "warmup_rounds": args.warmup_rounds,
            "max_samples": args.max_samples,
            "dataset": args.dataset,
            "duration_groups": args.duration_groups,
            "samples_per_group": args.samples_per_group,
            "prefix_segments": args.prefix_segments,
            "suffix_segments": args.suffix_segments,
            "recognition_threshold": args.recognition_threshold,
            "timestamp": ts,
        },
        "results": [asdict(r) for r in results],
        "statistics": {
            "by_mode": [asdict(s) for s in stats_mode],
            "by_dataset": [asdict(s) for s in stats_dataset],
            "by_language": [asdict(s) for s in stats_language],
            "overall": [asdict(s) for s in stats_overall],
        },
    }
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info(f"JSON 结果: {json_file}")

    csv_file = output_dir / f"exp3_summary_{ts}.csv"
    import csv

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "sample_id", "dataset", "language", "dialog_id", "turn_index", "text_length",
            "audio_duration", "duration_group", "mode", "wer", "cer", "asr_time_ms", "error"
        ])
        for r in results:
            w.writerow([
                r.sample_id, r.dataset, r.language, r.dialog_id, r.turn_index, r.text_length,
                f"{r.audio_duration:.2f}", r.duration_group, r.mode, f"{r.wer:.4f}", f"{r.cer:.4f}", f"{r.asr_time_ms:.2f}", r.error
            ])
    logger.info(f"CSV 结果: {csv_file}")

    stats_file = output_dir / f"exp3_statistics_{ts}.csv"
    with open(stats_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scope", "sample_count", "avg_duration_s", "wer_mean", "wer_std", "cer_mean", "cer_std"])
        # 按 mode 统计放在最前面（论文核心对比）
        for s in stats_mode:
            w.writerow([s.scope, s.sample_count, f"{s.avg_duration:.2f}", f"{s.wer_mean:.4f}", f"{s.wer_std:.4f}", f"{s.cer_mean:.4f}", f"{s.cer_std:.4f}"])
        for s in stats_dataset + stats_language + stats_overall:
            w.writerow([s.scope, s.sample_count, f"{s.avg_duration:.2f}", f"{s.wer_mean:.4f}", f"{s.wer_std:.4f}", f"{s.cer_mean:.4f}", f"{s.cer_std:.4f}"])
    logger.info(f"统计结果: {stats_file}")

    return json_file, csv_file, stats_file


# =============================================================================
# 主函数
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="实验三：准确率与质量验证",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  uv run python -m experiments.scripts.run_exp_quality --dataset all --max-samples 50
  uv run python -m experiments.scripts.run_exp_quality --dataset crosswoz --max-samples 20 --asr-device cuda
"""
    )

    parser.add_argument("--data-dir", type=str, default="experiments/datasets/processed", help="处理后的数据目录")
    parser.add_argument("--dataset", type=str, choices=["crosswoz", "multiwoz", "all"], default="all", help="数据集")
    parser.add_argument("--max-samples", type=int, default=None, help="最大样本数")
    parser.add_argument("--duration-groups", nargs="+", default=["medium", "long", "very_long"],
                        choices=list(DURATION_GROUPS.keys()),
                        help="分层抽样的时长分组（默认 medium/long/very_long）")
    parser.add_argument("--samples-per-group", type=int, default=None,
                        help="每个分组抽样数量；未指定时若设置 --max-samples 将自动均分")

    parser.add_argument("--asr-device", type=str, default="auto", help="ASR 设备 (auto/cuda/cuda:0/cuda:1/cpu)")
    parser.add_argument("--asr-model-size", type=str, default=ASR_MODEL_NAME, choices=["tiny", "base", "small", "medium", "large"], help="ASR 模型大小")
    parser.add_argument("--chunk-duration", type=int, default=500, help="流式音频块时长 ms")
    parser.add_argument("--warmup-rounds", type=int, default=2, help="模型预热轮数")
    
    # ASR 流式参数
    parser.add_argument("--prefix-segments", type=int, default=1, help="ASR 前缀段数（影响上下文和延迟，默认1）")
    parser.add_argument("--suffix-segments", type=int, default=1, help="ASR 后缀段数（影响准确率和延迟，默认1）")
    parser.add_argument("--recognition-threshold", type=float, default=2.0, help="ASR 识别阈值（秒）")

    parser.add_argument("--output-dir", type=str, default="experiments/results/exp3_quality", help="输出目录")
    
    # 断点续传参数
    parser.add_argument("--batch-size", type=int, default=100, help="每处理多少样本保存一次检查点（默认100）")
    parser.add_argument("--no-resume", action="store_true", help="不从检查点恢复，从头开始运行")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="日志级别")

    args = parser.parse_args()

    set_global_log_level(args.log_level)

    # 处理设备参数（支持 cuda:0, cuda:1 等具体设备）
    import torch
    if args.asr_device == "auto":
        args.asr_device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("=" * 60)
    logger.info("实验三：准确率与质量验证")
    logger.info("=" * 60)
    logger.info(f"数据集: {args.dataset}, 最大样本: {args.max_samples}")
    logger.info(f"ASR 设备: {args.asr_device}, 模型: {args.asr_model_size}")
    logger.info(f"chunk: {args.chunk_duration} ms, 预热: {args.warmup_rounds}")
    logger.info(f"ASR prefix_segments: {args.prefix_segments}, suffix_segments: {args.suffix_segments}")
    logger.info(f"ASR recognition_threshold: {args.recognition_threshold}s")
    logger.info(f"批次大小（检查点间隔）: {args.batch_size}")
    logger.info(f"分层抽样分组: {args.duration_groups}, 每组数量: {args.samples_per_group or '均分/全部'}")
    logger.info(f"断点续传: {'禁用' if args.no_resume else '启用'}")
    logger.info("=" * 60)

    data_dir = PROJECT_ROOT / args.data_dir
    json_dir = data_dir / "json"
    audio_dir = data_dir / "audio"
    output_dir = PROJECT_ROOT / args.output_dir

    # 如果指定了 --no-resume，删除检查点文件
    if args.no_resume:
        checkpoint_path = get_checkpoint_path(output_dir)
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info("已删除旧的检查点文件，从头开始运行")

    dataset_filter = None if args.dataset == "all" else args.dataset
    raw_samples = load_samples(json_dir, audio_dir, dataset_filter, None)
    if not raw_samples:
        logger.error("没有可用样本，请先运行数据处理管线")
        sys.exit(1)
    samples = stratified_sample_by_group(
        raw_samples,
        target_groups=args.duration_groups,
        samples_per_group=args.samples_per_group,
        max_total=args.max_samples
    )
    if not samples:
        logger.error("分层抽样后无可用样本，请调整分组或增加数据")
        sys.exit(1)

    shared = SharedASR(args)
    shared.initialize()

    # 预热使用首个样本音频
    first = samples[0]
    audio, sr = sf.read(str(first.audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    shared.set_warmup_audio(audio, sr)
    shared.warmup(rounds=args.warmup_rounds)

    # 构建实验配置（用于检查点）
    experiment_config = {
        "asr_model": args.asr_model_size,
        "asr_device": args.asr_device,
        "chunk_duration_ms": args.chunk_duration,
        "warmup_rounds": args.warmup_rounds,
        "max_samples": args.max_samples,
        "dataset": args.dataset,
        "duration_groups": args.duration_groups,
        "samples_per_group": args.samples_per_group,
        "prefix_segments": args.prefix_segments,
        "suffix_segments": args.suffix_segments,
        "recognition_threshold": args.recognition_threshold,
    }

    # 运行实验（支持断点续传）
    exp = QualityExperiment(shared, args)
    results = exp.run_all(
        samples,
        output_dir=output_dir,
        batch_size=args.batch_size,
        config=experiment_config
    )

    # 按 mode 统计（论文核心对比：streaming vs non-streaming）
    stats_mode = compute_mode_statistics(results)
    stats_dataset = compute_statistics(results, "dataset")
    stats_language = compute_statistics(results, "language")
    stats_overall = compute_statistics(results, "overall")

    save_results(results, output_dir, args, stats_dataset, stats_language, stats_overall, stats_mode)

    # 打印核心对比结果
    print("\n" + "=" * 60)
    print("实验三结果摘要：流式 vs 非流式 ASR 准确率对比")
    print("=" * 60)
    for s in stats_mode:
        print(f"  {s.scope:<15} WER: {s.wer_mean:.4f} (±{s.wer_std:.4f})  CER: {s.cer_mean:.4f} (±{s.cer_std:.4f})  样本: {s.sample_count}")
    print("=" * 60)

    logger.info("实验完成！")


if __name__ == "__main__":
    main()

