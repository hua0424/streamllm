#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEV-3 修复验证：单样本 LocalAgreement 回放（真实 Whisper 模型，离线无实时休眠）。

用途：对指定样本逐段回放 LA 提交策略，打印每次提交的片段、最终拼接文本与
WER/CER，用于修复前后对比（如 E3-LA 无效事件的典型样本 crosswoz_10296_turn2）。
分段器与解码参数与 run_exp_baseline_la 生产路径完全一致；仅跳过实时模拟休眠。

用法：
  uv run python -m experiments.scripts.replay_la_sample \
      --sample-id crosswoz_10296_turn2 --asr-model-size turbo --device cuda:0
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_utils import get_logger, set_global_log_level
from src.asr.streamaudio_segmenter import StreamAudioSegmenter
from src.asr.run_stream_asr_test import convert_audio_segment
from experiments.scripts.run_exp_quality import wer as compute_wer, cer as compute_cer, zh_to_word_seq

logger = get_logger(__name__)


def find_sample(data_dir: Path, sample_id: str):
    for js in sorted((data_dir / "json").rglob(f"{sample_id}.json")):
        meta = json.loads(js.read_text(encoding="utf-8"))
        dataset = js.parent.name
        audio = data_dir / "audio" / dataset / meta.get("audio_file", f"{sample_id}.wav")
        if not audio.exists():
            audio = data_dir / "audio" / dataset / f"{sample_id}.wav"
        return meta, audio
    raise SystemExit(f"样本 {sample_id} 未在 {data_dir}/json 下找到")


def main():
    parser = argparse.ArgumentParser(description="DEV-3 修复验证：单样本 LA 回放")
    parser.add_argument("--sample-id", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="experiments/datasets/processed")
    parser.add_argument("--asr-model-size", type=str, default="turbo")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--decode-trigger", type=float, default=2.0)
    parser.add_argument("--trailing-margin", type=float, default=0.0)
    parser.add_argument("--chunk-duration", type=int, default=500)
    parser.add_argument("--output", type=str, default=None, help="可选：回放结果 JSON 输出路径")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    set_global_log_level(args.log_level)

    from src.asr.local_agreement_streamer import LocalAgreementStreamer

    meta, audio_path = find_sample(PROJECT_ROOT / args.data_dir, args.sample_id)
    audio, sr = sf.read(str(audio_path), dtype="float32")
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    duration = len(audio) / sr
    logger.info(f"样本 {args.sample_id}: {duration:.2f}s, 参考 {len(meta['text'])} 字符")

    streamer = LocalAgreementStreamer(
        model_size=args.asr_model_size, device=args.device,
        decode_trigger_s=args.decode_trigger, trailing_margin_s=args.trailing_margin)

    segmenter = StreamAudioSegmenter(
        sampling_rate=sr, silence_threshold=0.5, min_speech_duration_ms=500,
        min_silence_duration_ms=300, window_size_ms=64)
    state = segmenter.create_state()
    chunk_size = int(sr * args.chunk_duration / 1000)

    fragments = []
    n_decode_segments = 0

    def feed(stream_segment, segment_id, is_final):
        nonlocal n_decode_segments
        seg = convert_audio_segment(stream_segment, segment_id, segment_id == 1, is_final)
        frags = streamer.feed_segment(seg)
        n_decode_segments += 1
        for f in frags:
            fragments.append(f)
            print(f"  [提交@{streamer.committed_end_abs:6.2f}s] {f}")

    for i in range(0, len(audio), chunk_size):
        stream_segment, state = segmenter.process_audio(audio[i:i + chunk_size], state)
        if stream_segment:
            feed(stream_segment, f"seg_{stream_segment.segment_id:03d}", False)
    remaining, state = segmenter.flush(state)
    if remaining is not None and len(remaining.audio) > 0:
        feed(remaining, f"seg_{remaining.segment_id:03d}", True)

    tail = streamer.flush()
    if tail:
        fragments.append(tail)
        print(f"  [flush 收尾] {tail}")

    final_text = " ".join(fragments)
    ref = meta["text"]
    is_zh = meta.get("language", "zh").lower().startswith("zh")
    err = compute_wer(zh_to_word_seq(ref), zh_to_word_seq(final_text)) if is_zh \
        else compute_wer(ref.lower(), final_text.lower(), normalize=True)
    cer = compute_cer(ref, final_text)

    print("=" * 70)
    print(f"样本: {args.sample_id} ({duration:.2f}s)")
    print(f"最终文本 ({len(final_text)} 字符): {final_text}")
    print(f"参考文本: {ref}")
    print(f"WER/CER: {err:.4f} / {cer:.4f} | 提交片段 {len(fragments)} 个 | "
          f"失配事件 {len(streamer.divergence_events)} 次 | 已提交词 {len(streamer.committed_words)} 个")
    if streamer.divergence_events:
        for ev in streamer.divergence_events[:5]:
            print(f"  [失配] span={ev['hypothesis_span']} '{ev['hypothesis_text']}' vs 已提交 {ev['committed_texts']}")

    if args.output:
        out = {
            "sample_id": args.sample_id,
            "audio_duration": duration,
            "asr_model": args.asr_model_size,
            "decode_trigger_s": args.decode_trigger,
            "trailing_margin_s": args.trailing_margin,
            "committed_fragments": fragments,
            "final_text": final_text,
            "reference_text": ref,
            "wer": err,
            "cer": cer,
            "divergence_count": len(streamer.divergence_events),
            "committed_word_count": len(streamer.committed_words),
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"回放结果已保存: {args.output}")


if __name__ == "__main__":
    main()
