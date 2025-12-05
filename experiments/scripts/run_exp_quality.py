#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验三：准确率与质量验证 (ASR Accuracy & Quality)

目标：
1) 对比非流式 ASR 与流式 ASR 的识别准确率（WER/CER）
2) 关注长语音场景下流式处理的精度影响

运行示例（项目根目录）：
    uv run python -m experiments.scripts.run_exp_quality --dataset all --max-samples 200
"""

import argparse
import json
import sys
import time
import wave
import gc
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
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


def cer(ref: str, hyp: str) -> float:
    """字符级错误率"""
    ref_chars = list(ref.strip())
    hyp_chars = list(hyp.strip())
    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0
    return _levenshtein(ref_chars, hyp_chars) / len(ref_chars)


def wer(ref: str, hyp: str) -> float:
    """词级错误率（以空格切分；中文场景可视作每字空格切分后再计算）"""
    ref_words = ref.strip().split()
    hyp_words = hyp.strip().split()
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    return _levenshtein(ref_words, hyp_words) / len(ref_words)


def zh_to_word_seq(text: str) -> str:
    """将中文字符串转为空格分隔的“词”(逐字)，便于 WER 统一计算"""
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


@dataclass
class ExperimentResult:
    sample_id: str
    dataset: str
    language: str
    dialog_id: str
    turn_index: int
    text_length: int
    audio_duration: float
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
        self.asr = StreamingASRProcessor(
            model_size=self.args.asr_model_size,
            device=self.args.asr_device,
            compute_type="auto",
            recognition_threshold=1.0,
            prefix_segments=1,
            suffix_segments_atleast=1,
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
# 实验执行
# =============================================================================


class QualityExperiment:
    def __init__(self, shared_asr: SharedASR, args):
        self.models = shared_asr
        self.args = args
        self.results: List[ExperimentResult] = []

    def run_all(self, samples: List[SampleInfo]) -> List[ExperimentResult]:
        total = len(samples)
        for i, s in enumerate(samples):
            logger.info(f"[{i+1}/{total}] 样本 {s.sample_id} ({s.language}) 长度 {s.audio_duration:.2f}s")
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

            if (i + 1) % 10 == 0:
                clear_gpu_memory()
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
            res.last_text_time = timings["last_text"]
            res.asr_time_ms = (timings["last_text"] - timings["audio_end"]) * 1000
            res.transcript = " ".join(texts)

            ref_text = sample.text
            if sample.language.lower().startswith("zh"):
                res.wer = wer(zh_to_word_seq(ref_text), zh_to_word_seq(res.transcript))
            else:
                res.wer = wer(ref_text, res.transcript)
            res.cer = cer(ref_text, res.transcript)

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


def save_results(results: List[ExperimentResult], output_dir: Path, args, stats_dataset, stats_language, stats_overall):
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
            "timestamp": ts,
        },
        "results": [asdict(r) for r in results],
        "statistics": {
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
            "audio_duration", "mode", "wer", "cer", "asr_time_ms", "error"
        ])
        for r in results:
            w.writerow([
                r.sample_id, r.dataset, r.language, r.dialog_id, r.turn_index, r.text_length,
                f"{r.audio_duration:.2f}", r.mode, f"{r.wer:.4f}", f"{r.cer:.4f}", f"{r.asr_time_ms:.2f}", r.error
            ])
    logger.info(f"CSV 结果: {csv_file}")

    stats_file = output_dir / f"exp3_statistics_{ts}.csv"
    with open(stats_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scope", "sample_count", "avg_duration_s", "wer_mean", "wer_std", "cer_mean", "cer_std"])
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

    parser.add_argument("--asr-device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="ASR 设备")
    parser.add_argument("--asr-model-size", type=str, default=ASR_MODEL_NAME, choices=["tiny", "base", "small", "medium", "large"], help="ASR 模型大小")
    parser.add_argument("--chunk-duration", type=int, default=500, help="流式音频块时长 ms")
    parser.add_argument("--warmup-rounds", type=int, default=2, help="模型预热轮数")

    parser.add_argument("--output-dir", type=str, default="experiments/results/exp3_quality", help="输出目录")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="日志级别")

    args = parser.parse_args()

    set_global_log_level(args.log_level)

    if args.asr_device == "auto":
        import torch
        args.asr_device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("=" * 60)
    logger.info("实验三：准确率与质量验证")
    logger.info("=" * 60)
    logger.info(f"数据集: {args.dataset}, 最大样本: {args.max_samples}")
    logger.info(f"ASR 设备: {args.asr_device}, 模型: {args.asr_model_size}")
    logger.info(f"chunk: {args.chunk_duration} ms, 预热: {args.warmup_rounds}")
    logger.info("=" * 60)

    data_dir = PROJECT_ROOT / args.data_dir
    json_dir = data_dir / "json"
    audio_dir = data_dir / "audio"
    output_dir = PROJECT_ROOT / args.output_dir

    dataset_filter = None if args.dataset == "all" else args.dataset
    samples = load_samples(json_dir, audio_dir, dataset_filter, args.max_samples)
    if not samples:
        logger.error("没有可用样本，请先运行数据处理管线")
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

    exp = QualityExperiment(shared, args)
    results = exp.run_all(samples)

    stats_dataset = compute_statistics(results, "dataset")
    stats_language = compute_statistics(results, "language")
    stats_overall = compute_statistics(results, "overall")

    save_results(results, output_dir, args, stats_dataset, stats_language, stats_overall)

    logger.info("实验完成！")


if __name__ == "__main__":
    main()

