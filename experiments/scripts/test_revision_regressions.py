#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CISR 修改版回归测试（R3-P1-2）—— 纯 CPU，不依赖 GPU/大模型下载。

覆盖：
  A. LocalAgreement mock 三序列：空识别保留状态 / append-only / 失配记录与恢复
  B. endpoint 四场景（StreamAudioSegmenter 层，对应 run_exp_latency 的
     final_speech_segment_commit_time / asr_no_speech 判定依据 contains_speech）：
     1) 正常语音 + 2s 尾静音；2) flush-only 语音（fixture 保证无中途闭段）；
     3) 全静音；4) 超短无效音频
  C. make_exp2_clean_source 动态计数：自定义规模不再因 498/7 断言失败；
     expected-count 参数可选生效

用法：
  uv run python -m experiments.scripts.test_revision_regressions
退出码：0=全部通过，1=有失败。
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

SPEECH_FIXTURE = PROJECT_ROOT / "experiments" / "test_data" / "speech_fixture_4s.wav"
CHUNK_SAMPLES = 8000  # 500ms @ 16k


# =============================================================================
# A. LocalAgreement mock 测试
# =============================================================================

def _make_la():
    from src.asr.local_agreement_streamer import LocalAgreementStreamer
    la = LocalAgreementStreamer.__new__(LocalAgreementStreamer)
    la.sample_rate = 16000
    la.decode_trigger_s = 2.0
    la.trailing_margin_s = 0.0
    la.reset()
    return la


class _FakeSeg:
    def __init__(self, dur):
        self.audio_data = np.zeros(int(16000 * dur), dtype=np.float32)
        self.duration = dur
        self.is_final = False


def _drive(la, seq):
    it = iter(seq)
    la._decode_buffer = lambda: next(it)
    outs = []
    for _ in seq:
        outs.append(la.feed_segment(_FakeSeg(2.0)))
        la.buffer = np.zeros(160000, dtype=np.float32)  # 固定音频长度，解除提交线约束
    outs.append(la.flush())
    return outs


def test_la_empty_decode_recovery():
    """R2-P0-1 序列 1：空识别后不丢文本、不重复提交。"""
    la = _make_la()
    w = lambda *ts: [{"text": t, "start": i, "end": i + 1} for i, t in enumerate(ts)]
    outs = _drive(la, [w("hello", "world"), [], w("hello", "world", "again"),
                       w("hello", "world", "again", "more")])
    assert outs == [[], [], ["helloworld"], ["again"], "more"], outs
    text = "".join(x for c in outs[:-1] for x in c) + outs[-1]
    assert text.count("hello") == 1 and text.count("world") == 1
    return "空识别轮保留状态；helloworld 一次、again/more 不丢"


def test_la_append_only():
    """R2-P0-1 序列 2：append-only 保持（world 从未被提交，无失配事件为正确）。"""
    la = _make_la()
    w = lambda *ts: [{"text": t, "start": i, "end": i + 1} for i, t in enumerate(ts)]
    outs = _drive(la, [w("hello", "world"), w("hello", "word"), w("hello", "word", "again")])
    assert outs == [[], ["hello"], ["word"], "again"], outs
    assert len(la.divergence_events) == 0
    return "append-only 保持；假设词改写不产生已提交失配"


def test_la_divergence_logged_and_recovers():
    """失配触发：已提交词被改写必须记录事件并恢复。"""
    la = _make_la()
    w = lambda *ts: [{"text": t, "start": i, "end": i + 1} for i, t in enumerate(ts)]
    outs = _drive(la, [w("hello", "world"), w("hello", "world", "again"),
                       w("hello", "wurst", "again"), w("hello", "wurst", "again", "more")])
    assert outs == [[], ["helloworld"], [], ["again"], "more"], outs
    assert len(la.divergence_events) == 1, la.divergence_events
    return "world→wurst 记录 1 次失配，公共前缀重延伸恢复，flush 收尾"


# =============================================================================
# B. endpoint 四场景（StreamAudioSegmenter 层）
# =============================================================================

def _new_segmenter():
    from src.asr.streamaudio_segmenter import StreamAudioSegmenter
    return StreamAudioSegmenter(sampling_rate=16000, silence_threshold=0.5,
                                min_speech_duration_ms=500, min_silence_duration_ms=300,
                                window_size_ms=64)


def _run_pipeline(audio):
    """模拟 run_exp_latency.streaming 的分段路径，返回 (中途段列表, flush 段或 None)。"""
    seg = _new_segmenter()
    state = seg.create_state()
    mids = []
    for i in range(0, len(audio), CHUNK_SAMPLES):
        s, state = seg.process_audio(audio[i:i + CHUNK_SAMPLES], state)
        if s:
            mids.append(s)
    rem, _ = seg.flush(state)
    return mids, rem


def _speech_fixture():
    import soundfile as sf
    data, sr = sf.read(str(SPEECH_FIXTURE), dtype="float32")
    assert sr == 16000
    return data


def test_endpoint_normal_speech_with_silence():
    """场景 1：正常语音 + 2s 尾静音 → 中途闭段含语音，flush 残余为静音。"""
    audio = np.concatenate([_speech_fixture(), np.zeros(32000, dtype=np.float32)])
    mids, rem = _run_pipeline(audio)
    assert any(m.contains_speech for m in mids), "应有中途语音闭段"
    assert all(m.contains_speech for m in mids), "VAD 闭段 contains_speech 必须恒 True"
    assert rem is not None and not rem.contains_speech, "尾静音残余不得判为语音段"
    return f"中途闭段 {len(mids)} 个（含语音），flush 残余静音不伪造语音段时间"


def test_endpoint_flush_only_speech():
    """场景 2（R3-P0-1）：语音未在中途闭合、由 flush 输出 → 不得误判 asr_no_speech。"""
    audio = _speech_fixture()  # fixture 已验证：无中途闭段，语音全在 flush
    mids, rem = _run_pipeline(audio)
    assert rem is not None and rem.contains_speech, \
        "flush-only 语音段必须 contains_speech=True（否则被误判 asr_no_speech）"
    return "flush 段含语音被正确识别，不会误标 asr_no_speech"


def test_endpoint_all_silence():
    """场景 3：全静音 → 无任何含语音段（对应 asr_no_speech）。"""
    audio = np.zeros(96000, dtype=np.float32)  # 6s
    mids, rem = _run_pipeline(audio)
    assert not any(m.contains_speech for m in mids)
    assert rem is not None and not rem.contains_speech
    return "全静音无语音段（final_speech_segment_commit_time 应为 0）"


def test_endpoint_ultra_short():
    """场景 4：低于 VAD 最小语音长度的音频 → 无语音段。"""
    t = np.linspace(0, 0.3, 4800)
    audio = (0.2 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    mids, rem = _run_pipeline(audio)
    assert not any(m.contains_speech for m in mids)
    assert rem is None or not rem.contains_speech
    return "超短无效音频无语音段"


# =============================================================================
# C. make_exp2_clean_source 动态计数
# =============================================================================

def _mini_dataset(tmp: Path):
    """5 样本 × 3 模式：1 个 runtime error + 1 个挂起（>10s）→ clean 3 / excl 2。"""
    import csv
    keep_ids, err_id, hang_id = [], "mini_err_1", "mini_hang_1"
    all_ids = ["mini_a", "mini_b", "mini_c", err_id, hang_id]
    results = []
    for sid in all_ids:
        for mode in ["baseline", "streaming_asr_only", "full_streaming"]:
            ttft = {"baseline": 2000.0, "streaming_asr_only": 900.0,
                    "full_streaming": 880.0}[mode]
            error = ""
            if sid == err_id and mode == "streaming_asr_only":
                error, ttft = "HTTP Error 504", 0.0
            if sid == hang_id and mode == "full_streaming":
                ttft = 20000.0
            results.append({
                "sample_id": sid, "dataset": "mini", "language": "zh", "dialog_id": "1",
                "turn_index": 1, "text_length": 10, "duration_group": "long",
                "audio_duration": 20.0, "mode": mode, "ttft": ttft, "error": error,
            })
    keep_ids = [s for s in all_ids if s not in (err_id, hang_id)]
    (tmp / "results.json").write_text(json.dumps({"results": results}), encoding="utf-8")
    (tmp / "list.json").write_text(json.dumps({"sample_ids": keep_ids}), encoding="utf-8")
    # 原 gains CSV：error 样本不在内（4 行 = 3 保留 + 1 挂起），2dp 与 results 一致
    with open(tmp / "gains.csv", "w", newline="", encoding="utf-8") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["sample_id", "dataset", "language", "dialog_id", "turn_index",
                       "text_length", "duration_group", "audio_duration_s",
                       "baseline_ttft_ms", "streaming_asr_ttft_ms", "full_streaming_ttft_ms",
                       "asr_gain_ms", "kv_gain_ms", "total_gain_ms", "total_gain_ratio_%"])
        for sid in [s for s in all_ids if s != err_id]:
            wcsv.writerow([sid, "mini", "zh", "1", 1, 10, "long", "20.00",
                           "2000.00", "900.00", "880.00", "1100.00", "20.00",
                           "1120.00", "56.0"])
    return tmp / "results.json", tmp / "list.json", tmp / "gains.csv"


def test_clean_source_dynamic_counts():
    """R3-P1-1：自定义规模（3/2）不再因 498/7 断言失败；expected 参数可选生效。"""
    import argparse
    from experiments.scripts import make_exp2_clean_source as mcs
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        rj, lst, gcsv = _mini_dataset(tmp)
        out_c, out_e = tmp / "clean.csv", tmp / "excl.csv"
        args = argparse.Namespace(results_json=rj, gains_csv=gcsv, sample_list=lst,
                                  output_clean=out_c, output_exclusions=out_e,
                                  expected_clean_count=3, expected_exclusion_count=2)
        mcs.main(args)  # 不抛异常即通过（含三重验证）
        meta = json.loads((tmp / "clean.meta.json").read_text(encoding="utf-8"))
        assert meta["counts"]["kept"] == 3 and meta["counts"]["excluded"] == 2, meta
        assert meta["counts"]["expected_clean"] == 3 and meta["counts"]["sample_list"] == 3
        # 错误的期望值必须失败（审计断言生效）
        bad = argparse.Namespace(**{**vars(args), "expected_clean_count": 5})
        try:
            mcs.main(bad)
            raise AssertionError("expected-clean-count=5 应当触发断言失败")
        except AssertionError as e:
            if "应当触发" in str(e):
                raise
        return "迷你集（3 保留/2 排除）动态通过；错误 expected 值触发断言"


# =============================================================================
# 主入口
# =============================================================================

def main():
    tests = [
        ("A1 LA 空识别恢复", test_la_empty_decode_recovery),
        ("A2 LA append-only", test_la_append_only),
        ("A3 LA 失配记录恢复", test_la_divergence_logged_and_recovers),
        ("B1 endpoint 正常语音+尾静音", test_endpoint_normal_speech_with_silence),
        ("B2 endpoint flush-only 语音", test_endpoint_flush_only_speech),
        ("B3 endpoint 全静音", test_endpoint_all_silence),
        ("B4 endpoint 超短无效", test_endpoint_ultra_short),
        ("C1 clean source 动态计数", test_clean_source_dynamic_counts),
    ]
    failures = 0
    for name, fn in tests:
        try:
            note = fn()
            print(f"[PASS] {name}: {note}")
        except Exception as e:
            failures += 1
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
