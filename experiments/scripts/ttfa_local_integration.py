"""W1 本机小模型真实路径集成测试（RTX 3060 6GB，非正式实验数据）。

用途（评审 §6/§9 测试证据补强）：self-test 的 fake 编排之外，用**真实组件**跑通
run_streaming / run_non_streaming 全路径——真实 StreamAudioSegmenter（Silero VAD）、
真实 StreamingASRProcessor（whisper tiny）、真实 ASRCache 状态机、真实
StreamLLMInference.generate_with_meta()（Qwen2-0.5B-Instruct 本地目录）+ 真实
tokenizer 累计重解码 + 请求级 torch.Generator。TTS 用假 PCM HTTP 服务
（格式/计时接口与正式一致，不依赖外部服务）。

仅验证生产代码路径可用性与记录合法性（validate_record 全过），
不产出任何论文数字；正式实验仍在 GPU 实验机按 handoff 执行。

用法：
  uv run python -m experiments.scripts.ttfa_local_integration
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.scripts.run_ttfa_unified import (  # noqa: E402
    _FakeTTSServer, analyze_pse, run_non_streaming, run_streaming, seed_for_pair,
    sha256_text, validate_record, ttfa_ms, _ST_PROBE,
)

QWEN_LOCAL = "C:/Users/hua/.cache/models/qwen2-0.5b-instruct"


def main() -> int:
    import torch
    from src.asr.streamaudio_segmenter import StreamAudioSegmenter
    from src.asr.faster_whisper_streamer import StreamingASRProcessor, ASRCache
    from src.llm.stream_llm_inference import StreamLLMInference
    from src.asr.run_stream_asr_test import convert_audio_segment

    print(f"torch {torch.__version__} cuda={torch.cuda.is_available()}")
    print("加载真实组件（whisper tiny + Qwen2-0.5B 本地 + Silero 缓存）…")
    asr = StreamingASRProcessor(model_size="tiny", device="cuda", compute_type="auto",
                                recognition_threshold=2.0, prefix_segments=1,
                                suffix_segments_atleast=0)
    llm = StreamLLMInference(model_name=QWEN_LOCAL, device="cuda", eval_mode=False)
    segmenter = StreamAudioSegmenter()
    models = {"segmenter": segmenter, "asr": asr, "llm": llm,
              "new_asr_cache": ASRCache, "convert_audio_segment": convert_audio_segment,
              "decode_fn": lambda ids: llm.tokenizer.decode(ids, skip_special_tokens=True)}

    # 合成音频：2s 宽带噪声 + 1s 静音（VAD 可能无语音段 → 正好覆盖 flush=None 路径）
    sr = 16000
    import soundfile as sf
    rng = np.random.default_rng(0)
    audio = np.concatenate([rng.normal(0, 0.15, 2 * sr),
                            np.zeros(sr)]).astype(np.float32)
    wav = Path("experiments/results/revision/r7_ttfa_unified/_integration_audio.wav")
    wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(wav), audio, sr)
    # PSE：真实 Silero（torch.hub 缓存）；若双法之一对纯噪声无 speech → 手工注入能量法结果
    import torch as _t
    _, utils = _t.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad",
                           onnx=False, verbose=False)
    pse = analyze_pse(str(wav), utils[0])
    if pse.get("error"):
        from experiments.scripts.run_ttfa_unified import energy_pse_sample
        e = energy_pse_sample(audio)
        pse = {"wav_sha256": pse.get("wav_sha256", "x"),
               "analysis_waveform_sha256": pse.get("analysis_waveform_sha256", "x"),
               "physical_speech_end_sample": e, "pse_method": "energy_forced",
               "pse_diff_ms": None}
        print(f"注：纯噪声下 PSE 双法未同时命中（{pse and 'expected'}），能量法注入 pse={e}")

    sample = {"sample_id": "integration_noise_1", "language": "zh",
              "duration_group": "long", "audio_path": str(wav), "repeat_idx": 0}
    tts_cfg = {"url": None, "spk_id": "x", "speed": 1.0, "max_tokens": 16,
               "config_hash": "integration", "schedule_hash": "integration",
               "run_id": "integration", "pair_deadline_s": 600.0,
               "tts_total_timeout_s": 30.0}
    fails = 0
    with _FakeTTSServer("normal") as srv:
        tts_cfg["url"] = srv.url
        seed = seed_for_pair(sample["sample_id"], 0)
        print("— streaming（System B 路径，真实 ASR/LLM）…")
        rec_b = run_streaming(sample, audio, sr, models, pse, tts_cfg, dict(_ST_PROBE),
                              seed, threading.Event())
        v = validate_record(rec_b, "integration", "integration")
        ok = rec_b["terminal_state"] == "success" and not v
        print(f"  B: {rec_b['terminal_state']} validate={v or 'OK'} "
              f"TTFA={ttfa_ms(rec_b) if rec_b['terminal_state']=='success' else '-'} "
              f"stop={rec_b['generation_stop_reason']} drain={rec_b['final_drain_triggered']} "
              f"err={rec_b['error'][:120]}")
        fails += 0 if ok else 1
        print("— non-streaming（System A 路径）…")
        rec_a = run_non_streaming(sample, audio, sr, models, pse, tts_cfg, dict(_ST_PROBE),
                                  seed, threading.Event())
        v = validate_record(rec_a, "integration", "integration")
        ok = rec_a["terminal_state"] == "success" and not v
        print(f"  A: {rec_a['terminal_state']} validate={v or 'OK'} "
              f"TTFA={ttfa_ms(rec_a) if rec_a['terminal_state']=='success' else '-'} "
              f"stop={rec_a['generation_stop_reason']} err={rec_a['error'][:120]}")
        fails += 0 if ok else 1
        if rec_a["terminal_state"] == "success":
            print(f"  A asr_start>=feed_end: "
                  f"{rec_a['events']['asr_start_ns'] >= rec_a['events']['feed_end_ns']}")
    print(f"\nintegration {'ALL PASS' if fails == 0 else f'{fails} FAIL'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
