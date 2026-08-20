#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R3：LocalAgreement-2 基线实验（DEV-4）

目的：在相同样本、相同模型权重、相同分段器、相同 LLM 增量预填路径下，
仅替换 ASR 上下文/提交策略为 LocalAgreement-2（ufal/whisper_streaming 策略出处），
与 System A / System B（取自 exp2 既有结果）对比 TTFT 与转写质量。

与 run_exp_latency.py 的差异：
- 仅运行 la_streaming 一种模式（System A/B 数字不重跑）
- ASR worker 使用 LocalAgreementStreamer（整缓冲重解码 + 公共前缀提交）
- 输出三件套：la_results_*.json / la_summary_*.csv / la_statistics_*.csv

使用方式：
    uv run python -m experiments.scripts.run_exp_baseline_la \
        --dataset all --sample-list <清单.json> --output-dir <目录>
"""

import argparse
import json
import time
import sys
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_utils import get_logger, set_global_log_level
from src.asr.streamaudio_segmenter import StreamAudioSegmenter
from src.asr.run_stream_asr_test import convert_audio_segment
from src.asr.local_agreement_streamer import LocalAgreementStreamer
from src.llm.stream_llm_inference import StreamLLMInference
from src.config import LLM_MODEL_NAME, ASR_MODEL_NAME

# 复用 run_exp_latency 的通用构件（DEV-1 已含 --sample-list 过滤与数据集扫描）
from experiments.scripts.run_exp_latency import (
    SampleInfo,
    ExperimentResult,
    load_samples,
    clear_gpu_memory,
    load_checkpoint,
    save_checkpoint,
)
# 复用 run_exp_quality 的文本归一化与 WER/CER 计算（与实验三完全同口径）
from experiments.scripts.run_exp_quality import wer as compute_wer, cer as compute_cer, zh_to_word_seq

logger = get_logger(__name__)


class LAExperiment:
    """LocalAgreement 流式实验执行器（每样本仅 la_streaming 一个模式）"""

    def __init__(self, la_streamer: LocalAgreementStreamer, llm_inference: StreamLLMInference, args):
        self.la_streamer = la_streamer
        self.llm = llm_inference
        self.args = args
        self.results: List[ExperimentResult] = []

    def run_single_sample(self, sample: SampleInfo) -> ExperimentResult:
        import queue
        import threading

        result = ExperimentResult(
            sample_id=sample.sample_id,
            audio_duration=sample.audio_duration,
            duration_group=sample.duration_group,
            mode="la_streaming",
            ttft=0, asr_time=0, llm_prefill_time=0,
        )

        try:
            self.la_streamer.reset()  # 每样本必须清空 LA 状态
            self.llm.reset_timings()

            audio_data, sample_rate = sf.read(str(sample.audio_path), dtype='float32')
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            if sample_rate != 16000:
                import librosa
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
                sample_rate = 16000

            segmenter = StreamAudioSegmenter(
                sampling_rate=sample_rate,
                silence_threshold=0.5,
                min_speech_duration_ms=500,
                min_silence_duration_ms=300,
                window_size_ms=64,
            )

            chunk_duration_ms = self.args.chunk_duration
            chunk_size = int(sample_rate * chunk_duration_ms / 1000)

            audio_chunk_queue = queue.Queue()
            audio_segment_queue = queue.Queue()
            text_queue = queue.Queue()
            audio_gen_done = threading.Event()
            segmentation_done = threading.Event()
            asr_done = threading.Event()

            timings = {"start_time": 0.0, "audio_end_time": 0.0,
                       "last_text_time": 0.0, "first_token_time": 0.0}
            full_response: List[str] = []
            committed_fragments: List[str] = []

            def audio_gen_worker():
                for i in range(0, len(audio_data), chunk_size):
                    chunk = audio_data[i:i + chunk_size]
                    audio_chunk_queue.put((i // chunk_size, chunk))
                    time.sleep(chunk_duration_ms / 1000)  # 模拟实时（方法学要求，严禁删除）
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
                        audio_segment_queue.put(
                            convert_audio_segment(stream_segment, segment_id, is_start, False))
                remaining_segment, state = segmenter.flush(state)
                if remaining_segment and len(remaining_segment.audio) > 0:
                    segment_id = f"seg_{remaining_segment.segment_id:03d}"
                    audio_segment_queue.put(
                        convert_audio_segment(remaining_segment, segment_id, False, True))
                segmentation_done.set()

            def asr_worker():
                while True:
                    try:
                        asr_segment = audio_segment_queue.get(timeout=0.1)
                    except queue.Empty:
                        if segmentation_done.is_set() and audio_segment_queue.empty():
                            break
                        continue
                    fragments = self.la_streamer.feed_segment(asr_segment)
                    for frag in fragments:
                        timings["last_text_time"] = time.time()
                        committed_fragments.append(frag)
                        text_queue.put((frag, False))
                # 流结束：提交全部剩余假设并通知 LLM 收尾。
                # 注：tail 为空时发送 ("", True) 是安全的——cache_prompt 在 is_end=True 时
                # 会追加 generation_prompt，不会触发 _add_stream_prompt 的空文本异常。
                tail = self.la_streamer.flush()
                if tail:
                    timings["last_text_time"] = time.time()
                    committed_fragments.append(tail)
                text_queue.put((tail, True))
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
                        kv_cache = self.llm.cache_prompt(text, pre_cache=kv_cache, is_end=is_end)
                    if is_end:
                        first_token = True
                        for token in self.llm.generate(pre_cache=kv_cache,
                                                       max_new_tokens=self.args.max_tokens):
                            if first_token:
                                timings["first_token_time"] = time.time()
                                first_token = False
                            full_response.append(token)
                        break

            timings["start_time"] = time.time()
            threads = [threading.Thread(target=w) for w in
                       (audio_gen_worker, segmentation_worker, asr_worker, llm_worker)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            result.start_time = timings["start_time"]
            result.audio_end_time = timings["audio_end_time"]
            result.last_text_time = timings["last_text_time"]
            result.first_token_time = timings["first_token_time"]
            result.ttft = (timings["first_token_time"] - timings["audio_end_time"]) * 1000
            result.asr_time = (timings["last_text_time"] - timings["audio_end_time"]) * 1000
            result.llm_prefill_time = (timings["first_token_time"] - timings["last_text_time"]) * 1000
            result.transcribed_text = " ".join(committed_fragments)
            result.divergence_count = len(self.la_streamer.divergence_events)

            # P0-1/P0-3：质量指标与空转写标记（与实验三同一归一化口径）
            if not result.transcribed_text.strip():
                result.error = "asr_no_text"
            else:
                ref_text = sample.text
                if sample.language.lower().startswith("zh"):
                    result.wer = compute_wer(zh_to_word_seq(ref_text),
                                             zh_to_word_seq(result.transcribed_text))
                else:
                    result.wer = compute_wer(ref_text, result.transcribed_text)
                result.cer = compute_cer(ref_text, result.transcribed_text)

            result.response_preview = "".join(full_response)[:100]
            if getattr(self.args, 'save_full_response', False):
                result.full_response = "".join(full_response)
            if getattr(self.args, 'save_fragments', False):
                result.committed_fragments = committed_fragments

        except Exception as e:
            result.error = str(e)
            logger.error(f"LA 测试失败 {sample.sample_id}: {e}")
            import traceback
            traceback.print_exc()

        return result

    def run_all(self, samples: List[SampleInfo], output_dir: Path,
                batch_size: int = 100, config: Dict[str, Any] = None) -> List[ExperimentResult]:
        existing_results, completed_ids = load_checkpoint(output_dir)
        self.results = existing_results
        pending = [s for s in samples if s.sample_id not in completed_ids]
        if len(pending) < len(samples):
            logger.info(f"断点续传: 跳过 {len(samples) - len(pending)} 个已完成样本")

        processed_in_batch = 0
        for i, sample in enumerate(pending):
            logger.info(f"\n[{len(completed_ids) + 1}/{len(samples)}] 测试样本: {sample.sample_id} "
                        f"({sample.audio_duration:.1f}s, {sample.duration_group})")
            result = self.run_single_sample(sample)
            self.results.append(result)
            completed_ids.add(sample.sample_id)
            processed_in_batch += 1
            if not result.error:
                logger.info(f"  LA TTFT: {result.ttft:.2f} ms, 提交片段 {len(result.committed_fragments)} 个")
            if (i + 1) % 5 == 0:
                clear_gpu_memory()
            if processed_in_batch >= batch_size:
                save_checkpoint(self.results, completed_ids, output_dir, config or {})
                processed_in_batch = 0
                logger.info(f"✓ 已保存检查点 ({len(completed_ids)}/{len(samples)} 完成)")

        if processed_in_batch > 0:
            save_checkpoint(self.results, completed_ids, output_dir, config or {})
        return self.results


def save_la_results(results: List[ExperimentResult], output_dir: Path, args) -> Tuple[Path, Path, Path]:
    """保存 la_results/la_summary/la_statistics 三件套（含 WER/CER，P0-1）"""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    ok = [r for r in results if not r.error]

    def agg(rows):
        v = np.array([r.ttft for r in rows])
        w = np.array([r.wer for r in rows])
        c = np.array([r.cer for r in rows])
        return {
            "sample_count": len(rows),
            "avg_duration": float(np.mean([r.audio_duration for r in rows])),
            "ttft_mean": float(np.mean(v)), "ttft_std": float(np.std(v)),
            "ttft_min": float(np.min(v)), "ttft_max": float(np.max(v)),
            "wer_mean": float(np.mean(w)), "cer_mean": float(np.mean(c)),
        }

    # 分组统计（时长组）+ 语言统计 + 总体
    stats_rows = []
    for g in sorted({r.duration_group for r in ok}):
        rows = [r for r in ok if r.duration_group == g]
        stats_rows.append({"scope": f"group:{g}", **agg(rows)})
    for lang in sorted({r.sample_id.split('_')[0] for r in ok}):
        rows = [r for r in ok if r.sample_id.startswith(lang)]
        stats_rows.append({"scope": f"dataset:{lang}", **agg(rows)})
    if ok:
        stats_rows.append({"scope": "overall", **agg(ok)})

    results_file = output_dir / f"la_results_{timestamp}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            "config": {
                "mode": "la_streaming",
                "policy": "LocalAgreement-2 (ufal/whisper_streaming 策略, 同引擎自实现)",
                "asr_model": args.asr_model_size,
                "llm_model": args.llm_model_name,
                "asr_device": args.asr_device,
                "llm_device": args.llm_device,
                "chunk_duration_ms": args.chunk_duration,
                "max_tokens": args.max_tokens,
                "warmup_rounds": args.warmup_rounds,
                "decode_trigger_s": args.recognition_threshold,
                "trailing_margin_s": 0.0,
                "la_max_buffer_s": args.la_max_buffer_s,
                "dataset": args.dataset,
                "sample_list": args.sample_list,
                "timestamp": timestamp,
            },
            "results": [r.__dict__ for r in results],
            "statistics": stats_rows,
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"详细结果已保存: {results_file}")

    csv_file = output_dir / f"la_summary_{timestamp}.csv"
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "audio_duration", "duration_group", "mode",
                         "ttft_ms", "asr_time_ms", "llm_prefill_time_ms", "wer", "cer",
                         "divergence_count", "error"])
        for r in results:
            writer.writerow([r.sample_id, f"{r.audio_duration:.2f}", r.duration_group, r.mode,
                             f"{r.ttft:.2f}", f"{r.asr_time:.2f}", f"{r.llm_prefill_time:.2f}",
                             f"{r.wer:.4f}", f"{r.cer:.4f}", r.divergence_count, r.error])

    stats_file = output_dir / f"la_statistics_{timestamp}.csv"
    with open(stats_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["scope", "sample_count", "avg_duration_s",
                         "ttft_mean_ms", "ttft_std_ms", "ttft_min_ms", "ttft_max_ms",
                         "wer_mean", "cer_mean"])
        for s in stats_rows:
            writer.writerow([s["scope"], s["sample_count"], f"{s['avg_duration']:.2f}",
                             f"{s['ttft_mean']:.2f}", f"{s['ttft_std']:.2f}",
                             f"{s['ttft_min']:.2f}", f"{s['ttft_max']:.2f}",
                             f"{s['wer_mean']:.4f}", f"{s['cer_mean']:.4f}"])

    return results_file, csv_file, stats_file


def main():
    parser = argparse.ArgumentParser(
        description="R3：LocalAgreement-2 基线实验（仅 la_streaming 模式）")
    parser.add_argument('--data-dir', type=str, default='experiments/datasets/processed')
    parser.add_argument('--dataset', type=str, default='all',
                        help="数据集目录名，'all' = 扫描 processed/json/ 全部子目录")
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--sample-list', type=str, default=None,
                        help='样本清单 JSON（数组或含 "sample_ids" 的对象）')
    parser.add_argument('--asr-device', type=str, default='auto')
    parser.add_argument('--llm-device', type=str, default='auto')
    parser.add_argument('--asr-model-size', type=str, default=ASR_MODEL_NAME,
                        choices=['tiny', 'base', 'small', 'medium', 'large',
                                 'large-v1', 'large-v2', 'large-v3', 'large-v3-turbo', 'turbo'])
    parser.add_argument('--llm-model-name', type=str, default=LLM_MODEL_NAME)
    parser.add_argument('--chunk-duration', type=int, default=500)
    parser.add_argument('--max-tokens', type=int, default=50)
    parser.add_argument('--warmup-rounds', type=int, default=3)
    parser.add_argument('--recognition-threshold', type=float, default=2.0,
                        help='LA 解码触发新增音频时长（秒），与 System B 锁定值对齐')
    parser.add_argument('--la-max-buffer-s', type=float, default=15.0,
                        help='LA 缓冲长度上限（秒），对齐 ufal whisper_streaming buffer_trimming_sec=15')
    parser.add_argument('--output-dir', type=str, default='experiments/results/revision/r3_baseline_la')
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--no-resume', action='store_true')
    parser.add_argument('--save-full-response', action='store_true')
    parser.add_argument('--save-fragments', action='store_true')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    args = parser.parse_args()

    set_global_log_level(args.log_level)

    import torch
    if args.asr_device == 'auto':
        args.asr_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.llm_device == 'auto':
        args.llm_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_dir = PROJECT_ROOT / args.data_dir
    json_dir = data_dir / "json"
    audio_dir = data_dir / "audio"
    output_dir = PROJECT_ROOT / args.output_dir

    if args.no_resume and (output_dir / "checkpoint.json").exists():
        (output_dir / "checkpoint.json").unlink()
        logger.info("已删除旧的检查点文件，从头开始运行")

    dataset_filter = None if args.dataset == 'all' else args.dataset
    samples = load_samples(json_dir, audio_dir, dataset_filter, args.max_samples)

    if args.sample_list:
        with open(args.sample_list, 'r', encoding='utf-8') as f:
            list_data = json.load(f)
        allow_ids = set(list_data["sample_ids"] if isinstance(list_data, dict) else list_data)
        before = len(samples)
        samples = [s for s in samples if s.sample_id in allow_ids]
        missing = allow_ids - {s.sample_id for s in samples}
        logger.info(f"样本清单过滤: {before} -> {len(samples)}（清单 {len(allow_ids)} 条）")
        if missing:
            # E3-LA 评审要求：清单缺失必须停止而不是静默缩减（否则 498 成对比较失效）
            logger.error(f"清单中 {len(missing)} 条未在数据集中找到: {sorted(missing)[:5]}...")
            sys.exit(1)

    if not samples:
        logger.error("没有找到有效样本")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("R3：LocalAgreement-2 基线实验")
    logger.info(f"样本数: {len(samples)}, ASR: {args.asr_model_size} on {args.asr_device}, "
                f"LLM: {args.llm_model_name} on {args.llm_device}")
    logger.info("=" * 60)

    # ASR：LA 流式器（与 System A/B 同权重/精度/设备）
    la_streamer = LocalAgreementStreamer(
        model_size=args.asr_model_size,
        device=args.asr_device,
        decode_trigger_s=args.recognition_threshold,
        trailing_margin_s=0.0,  # 锁定配置 suffix=0
        max_buffer_s=args.la_max_buffer_s,
    )
    # LLM：与 System B 相同的增量预填路径
    llm_inference = StreamLLMInference(
        model_name=args.llm_model_name,
        device=args.llm_device,
        eval_mode=False,
    )

    # 预热（用第一个样本的音频，流程与 run_exp_latency 一致）
    first = samples[0]
    w_audio, w_sr = sf.read(str(first.audio_path), dtype='float32')
    if len(w_audio.shape) > 1:
        w_audio = w_audio.mean(axis=1)
    if w_sr != 16000:
        import librosa
        w_audio = librosa.resample(w_audio, orig_sr=w_sr, target_sr=16000)
        w_sr = 16000
    from src.asr.faster_whisper_streamer import ASRAudioSegment
    for round_idx in range(args.warmup_rounds):
        logger.info(f"  预热轮次 {round_idx + 1}/{args.warmup_rounds}")
        # 预热直接喂段给共享 la_streamer（避免重复加载模型），结束后重置状态
        la_streamer.reset()
        seg = ASRAudioSegment(id="warmup", audio_data=w_audio[: w_sr * 4].astype(np.float32),
                              start_time=0, end_time=4, duration=4.0, is_final=True)
        la_streamer.feed_segment(seg)
        la_streamer.flush()
        la_streamer.reset()
        kv = llm_inference.cache_prompt("你好，这是一个测试。", is_end=True)
        for _ in llm_inference.generate(pre_cache=kv, max_new_tokens=10):
            pass
        del kv
        clear_gpu_memory()
    logger.info("预热完成")

    experiment = LAExperiment(la_streamer, llm_inference, args)
    config = {
        "mode": "la_streaming",
        "asr_model": args.asr_model_size,
        "llm_model": args.llm_model_name,
        "decode_trigger_s": args.recognition_threshold,
        "trailing_margin_s": 0.0,
        "la_max_buffer_s": args.la_max_buffer_s,
        "chunk_duration_ms": args.chunk_duration,
        "max_tokens": args.max_tokens,
        "sample_list": args.sample_list,
        "dataset": args.dataset,
    }
    results = experiment.run_all(samples, output_dir=output_dir,
                                 batch_size=args.batch_size, config=config)

    save_la_results(results, output_dir, args)
    n_err = sum(1 for r in results if r.error)
    logger.info(f"\n实验完成！共 {len(results)} 条结果，{n_err} 条失败")


if __name__ == "__main__":
    main()
