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
    """场景 2（R3-P0-1 / R4-P2-1）：语音未在中途闭合、由 flush 输出 → 不得误判 asr_no_speech。
    fixture 选取时已验证量化 round-trip 后无中途闭段，此处显式断言 flush-only 性质。"""
    audio = _speech_fixture()
    mids, rem = _run_pipeline(audio)
    assert mids == [], f"fixture 不是 flush-only，检测到 {len(mids)} 个中途段"
    assert rem is not None and rem.contains_speech, \
        "flush-only 语音段必须 contains_speech=True（否则被误判 asr_no_speech）"
    return "无中途闭段 + flush 段含语音，不会误标 asr_no_speech"


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
# B2. 生产端点字段（run_exp_latency 闭包 + 纯函数，R4-P1-1）
# =============================================================================

def test_classify_endpoint_times_pure():
    """classify_endpoint_times 纯函数四用例（R4-P1-1 修改建议）。"""
    from experiments.scripts.run_exp_latency import classify_endpoint_times as cet
    # 1) 正常语音 + 尾静音：两 final 时间非零、无错误
    err, dw, ew = cet(100.0, 100.5, 102.5)
    assert err == "" and abs(dw - 0.5) < 1e-9 and abs(ew - 2.5) < 1e-9
    # 2) flush-only 语音：语音提交与 final enqueue 均非零
    err, dw, ew = cet(100.0, 102.5, 102.5)
    assert err == "" and dw is not None and ew is not None
    # 3) 全静音：语音提交时间为 0 → asr_no_speech；detection_wait 为 None
    err, dw, ew = cet(100.0, 0.0, 102.5)
    assert err == "asr_no_speech" and dw is None and abs(ew - 2.5) < 1e-9
    # 4) 时间顺序非法：语音段晚于 final 段 → endpoint_timing_invalid
    err, _, _ = cet(100.0, 105.0, 102.5)
    assert err == "endpoint_timing_invalid"
    return "四用例通过（正常/flush-only/全静音/顺序非法）"


def _fake_models():
    """fake ASR/LLM：满足 run_exp_latency 流式/非流式闭包的调用协议，无模型依赖。"""
    from types import SimpleNamespace

    class FakeASRProcessor:
        timing_events = {}

        def reset_commit_tracking(self, sample_id=""):
            pass

        def transcribe_audio_segment(self, cache):
            cache.add_to_asr_segments()
            if cache.segment_queue and cache.segment_queue[-1].is_final:
                return cache, "文本身", True
            return cache, "", False

        def transcribe_complete_audio(self, **kw):
            return {"text": "文本身"}

    class FakeLLM:
        def reset_timings(self):
            pass

        def cache_prompt(self, text, pre_cache=None, is_end=False):
            return object()

        def generate(self, pre_cache=None, max_new_tokens=50):
            yield "好"

        def once_add_and_generate(self, prompt, **kw):
            yield "好"

    return SimpleNamespace(asr_processor=FakeASRProcessor(), llm_inference=FakeLLM(),
                           reset_state=lambda: None)


def _run_production_sample(audio: np.ndarray, append_silence_ms: int = 0):
    """走完整生产路径 run_single_sample（含 --append-silence-ms 拼接），fake 模型（R4-P1-1）。"""
    from types import SimpleNamespace
    import soundfile as sf
    import tempfile
    from experiments.scripts.run_exp_latency import LatencyExperiment, SampleInfo
    args = SimpleNamespace(chunk_duration=500, max_tokens=5,
                           append_silence_ms=append_silence_ms,
                           save_full_response=False, save_fragments=False)
    exp = LatencyExperiment(_fake_models(), args)
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "stub.wav"
        sf.write(str(wav), audio, 16000, subtype="PCM_16")
        sample = SampleInfo(sample_id="stub_sample", dialog_id="0", turn_index=1, text="参考文本",
                            text_length=4, audio_file="stub.wav", audio_path=wav,
                            audio_duration=len(audio) / 16000, language="zh", dataset="stub",
                            duration_group="medium")
        streaming, _ = exp.run_single_sample(sample)
    return streaming


def test_production_endpoint_bookkeeping():
    """R4-P1-1 核心：完整生产路径（真实闭包 + 真实分段器 + 尾静音经 --append-silence-ms），
    断言写入结果 JSON 的三个时间字段及其关系。"""
    # 场景 1：正常语音 + 2s 尾静音（经生产 append 路径拼接）
    r1 = _run_production_sample(_speech_fixture(), append_silence_ms=2000)
    se, fs, fe = r1.speech_end_time, r1.final_speech_segment_commit_time, r1.final_is_final_segment_enqueue_time
    assert se > 0 and fs > 0 and fe > 0, (se, fs, fe)
    assert fs <= fe, "语音段提交不得晚于 final 段入队"
    assert fs > se, "VAD 闭段不得早于真实语音结束"
    assert r1.error == "", r1.error
    wait = fs - se
    assert 0 <= wait < 2.0, f"detection_wait={wait} 异常"
    print(f"\n    [B6-1] speech_end={se:.3f} speech_commit={fs:.3f} final_enqueue={fe:.3f} "
          f"detection_wait={wait:.3f}s enqueue_wait={fe-se:.3f}s")
    # 场景 2：flush-only 语音（无尾静音）
    r2 = _run_production_sample(_speech_fixture())
    assert r2.final_speech_segment_commit_time > 0 and r2.final_is_final_segment_enqueue_time > 0
    assert r2.error == "", r2.error
    print(f"    [B6-2] speech_commit={r2.final_speech_segment_commit_time:.3f} "
          f"final_enqueue={r2.final_is_final_segment_enqueue_time:.3f}")
    # 场景 3：全静音 → asr_no_speech，final enqueue 仍记录
    r3 = _run_production_sample(np.zeros(64000, dtype=np.float32))
    assert r3.final_speech_segment_commit_time == 0.0
    assert r3.final_is_final_segment_enqueue_time > 0
    assert r3.error == "asr_no_speech", r3.error
    print(f"    [B6-3] speech_commit=0 final_enqueue={r3.final_is_final_segment_enqueue_time:.3f} "
          f"error={r3.error}")
    return "生产路径三场景：字段非零/排序/非负等待/错误标记全部符合不变量"




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
        ("B5 endpoint 纯函数判定", test_classify_endpoint_times_pure),
        ("B6 endpoint 生产闭包 bookkeeping", test_production_endpoint_bookkeeping),
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
