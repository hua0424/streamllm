#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CosyVoice2 时序画像 benchmark（实验机用；⚠️ 未在真机验证，见 cosyvoice_tts.py 头注）。

目的（D-010）：实测三个 TimingProfile 参数，替换验证机占位值后重跑 E1/E2/E3：
  - samples_per_char       ：每个非空白字符对应的音频采样数（Mock 时长模型）
  - first_chunk_latency_ms ：首块合成延迟（mouth-to-ear 建模项）
  - synth_rtf              ：合成实时率 = 合成耗时/音频时长（E1 System A 的完整合成建模）

运行（实验机，CosyVoice 环境内）：
    uv run python -m experiments.scripts.benchmark_cosyvoice \
        --model-dir pretrained_models/CosyVoice2-0.5B --ref-audio ref.wav [--ref-text "..."]
产出 experiments/results/cosyvoice_profile.json；把三个值分别填回：
  src/tts/streaming_tts.py 的 TimingProfile 默认值 + run_exp1_latency.py 的 SYNTH_RTF。
"""

import argparse
import json
import time
from pathlib import Path
from statistics import median

from src.tts.cosyvoice_tts import CosyVoiceStreamingTTS
from src.utils.logging_utils import get_logger, set_global_log_level
from src.config import RESULTS_DIR

logger = get_logger(__name__)

SENTENCES = [
    "The Great Wall of China is one of the most famous landmarks in the world.",
    "I need a hotel in the north of town with free wifi.",
    "There are five trains arriving before noon on Sunday.",
    "The restaurant serves Italian food and is located near the city centre.",
    "Sure, I have booked two tickets for you, and the reference number is ABC123.",
    "It was built over many centuries to protect the northern borders.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=str, default="pretrained_models/CosyVoice2-0.5B")
    ap.add_argument("--ref-audio", type=str, required=True)
    ap.add_argument("--ref-text", type=str, default="Hello, this is a reference voice sample.")
    ap.add_argument("--out", type=str, default=str(Path(RESULTS_DIR) / "cosyvoice_profile.json"))
    ap.add_argument("--log-level", type=str, default="INFO")
    args = ap.parse_args()

    set_global_log_level(args.log_level)
    tts = CosyVoiceStreamingTTS(args.model_dir, args.ref_text, args.ref_audio)
    sr = tts.sample_rate

    # warmup（首次含模型编译/缓存）
    for _ in tts.synthesize("Warm up sentence for the synthesizer."):
        pass

    rows = []
    for s in SENTENCES:
        t0 = time.perf_counter()
        total, first_ms = 0, None
        for chunk in tts.synthesize(s):
            if first_ms is None:
                first_ms = (time.perf_counter() - t0) * 1000
            total += chunk.n_samples
        wall = time.perf_counter() - t0
        nws = sum(1 for c in s if not c.isspace())
        audio_s = total / sr
        rows.append({"text": s, "nws_chars": nws, "audio_s": round(audio_s, 3),
                     "samples_per_char": round(total / nws, 1),
                     "first_chunk_ms": round(first_ms, 1),
                     "rtf": round(wall / audio_s, 3)})
        logger.info(f"  {nws:>3}ch {audio_s:5.2f}s spc={rows[-1]['samples_per_char']:>7} "
                    f"first={first_ms:6.1f}ms rtf={rows[-1]['rtf']}")

    profile = {
        "sample_rate": sr,
        "samples_per_char": int(median(r["samples_per_char"] for r in rows)),
        "first_chunk_latency_ms": round(median(r["first_chunk_ms"] for r in rows), 1),
        "synth_rtf": round(median(r["rtf"] for r in rows), 3),
    }
    logger.info(f"TimingProfile 实测: {profile}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"profile": profile, "rows": rows}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    logger.info(f"已保存: {out} —— 请回填 streaming_tts.TimingProfile 与 run_exp1_latency.SYNTH_RTF")


if __name__ == "__main__":
    main()
