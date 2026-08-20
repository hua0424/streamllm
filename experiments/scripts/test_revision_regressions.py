#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CISR 修改版回归测试（R3-P1-2）—— 纯 CPU，不依赖 GPU/大模型下载。

覆盖：
  A. LocalAgreement 脚本化词序列（绝对时间轴驱动，真实喂缓冲/真实裁剪）：
     A1 空识别恢复（R2-P0-1 语义）；A2 未提交改写 append-only 无失配；
     A3 已提交区域改写记录失配并恢复
  D. LocalAgreement 错帧修复验收（DEV-3 2026-08-20 修复，E3-LA 无效事件）：
     D1 跨帧跳段（bug 复现序列：提交→裁剪→新假设前缀不同→再提交，中段不得丢失）；
     D2 多裁剪周期无重复；D3 裁剪边界 ±0.1s 不重复不跳过；
     D4 flush 裁剪后收尾且幂等；D5 生产路径 run_single_sample 全链路无缺口
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
# A/D. LocalAgreement 脚本化词序列测试（绝对时间轴驱动）
# =============================================================================
# 驱动器约定：每轮真实喂入 seg_dur 秒静音推进缓冲（含真实 _trim_buffer 裁剪），
# _decode_buffer 被替换为返回"脚本词"（绝对音频时间轴，驱动器内部转为 buffer 相对轴）。
# 脚本词必须满足：abs_end <= 已累计喂入时长（提交线 = 缓冲末端）。

def _make_la(max_buffer_s=15.0):
    from src.asr.local_agreement_streamer import LocalAgreementStreamer
    la = LocalAgreementStreamer.__new__(LocalAgreementStreamer)
    la.sample_rate = 16000
    la.decode_trigger_s = 2.0
    la.trailing_margin_s = 0.0
    la.max_buffer_s = max_buffer_s
    la.reset()
    return la


class _FakeSeg:
    def __init__(self, dur, is_final=False):
        self.audio_data = np.zeros(int(16000 * dur), dtype=np.float32)
        self.duration = dur
        self.is_final = is_final


def _drive_scripted(la, rounds, seg_dur=2.0):
    """rounds: 每轮脚本词 [(text, abs_start, abs_end), ...]（绝对音频时间轴）。
    返回 (逐轮提交片段列表, flush 收尾文本)。"""
    it = iter(rounds)

    def fake_decode():
        base = la.buffer_start_abs
        return [{"text": t, "start": s - base, "end": e - base} for t, s, e in next(it)]

    la._decode_buffer = fake_decode
    outs = [la.feed_segment(_FakeSeg(seg_dur)) for _ in rounds]
    return outs, la.flush()


def _committed_texts(la):
    return [w["text"] for w in la.committed_words]


def test_la_empty_decode_recovery():
    """A1 / R2-P0-1：空识别轮保留状态，恢复后不丢文本、不重复提交。"""
    la = _make_la()
    outs, tail = _drive_scripted(la, [
        [("a", 0, 1), ("b", 1, 2)],              # 首轮仅建立基线
        [],                                       # 空识别：保留上一轮假设
        [("a", 0, 1), ("b", 1, 2), ("c", 2, 3)],  # 恢复：提交 a,b → 裁剪至 1.9s
        [("c", 2, 3), ("d", 3, 4)],               # 裁剪后新帧：c 重现即提交
    ])
    assert outs == [[], [], ["ab"], ["c"]], outs
    assert tail == "d", tail
    assert _committed_texts(la) == ["a", "b", "c", "d"]
    assert len(la.divergence_events) == 0
    return "空识别轮保留状态；恢复后 abcd 完整各一次（含跨裁剪帧的 c/d）"


def test_la_append_only():
    """A2 / R2-P0-1：未提交区域改写（b→x）属正常 LA 行为，不产生失配事件。"""
    la = _make_la()
    outs, tail = _drive_scripted(la, [
        [("a", 0, 1), ("b", 1, 2)],
        [("a", 0, 1), ("x", 1, 2)],               # b→x 改写在未提交区域
        [("x", 1, 2), ("c", 2, 3)],
    ])
    assert outs == [[], ["a"], ["x"]], outs
    assert tail == "c", tail
    assert _committed_texts(la) == ["a", "x", "c"]
    assert len(la.divergence_events) == 0, la.divergence_events
    return "append-only 保持；未提交词改写不产生失配事件"


def test_la_divergence_logged_and_recovers():
    """A3：新假设改写已提交区域（b→bx）必须记录失配事件；假设停滞一轮后恢复。"""
    la = _make_la()
    outs, tail = _drive_scripted(la, [
        [("a", 0, 1), ("b", 1, 2)],
        [("a", 0, 1), ("b", 1, 2), ("c", 2, 3)],  # 提交 a,b → 裁剪至 1.9s
        [("bx", 1.9, 2.05), ("c", 2, 3)],         # 保留区内 b 被改写为 bx → 失配
        [("c", 2, 3), ("d", 3, 4)],               # bx 不再出现：基线错位一轮
        [("c", 2, 3), ("d", 3, 4), ("e", 4, 5)],  # 基线重建后恢复提交
    ])
    assert outs == [[], ["ab"], [], [], ["cd"]], outs
    assert tail == "e", tail
    assert _committed_texts(la) == ["a", "b", "c", "d", "e"]
    assert len(la.divergence_events) == 1, la.divergence_events
    return "b→bx 记录 1 次失配；公共前缀重建后恢复，abcde 完整"


# =============================================================================
# D. 错帧修复验收（DEV-3 2026-08-20，E3-LA 无效事件）
# =============================================================================

def test_la_cross_frame_no_gap():
    """D1：bug 复现序列——提交→句界裁剪→新假设前缀不同→再提交，中段词必须提交。

    修复前：n_committed 停在旧帧下标，裁剪后新帧的前若干个未提交词被当作"已提交"跳过，
    flush 丢尾，中段文本静默丢失（E3-LA 无效事件机制）。修复后：中段 m1m2 在第 5 轮
    经正常提交产出（非 flush 兜底），全序列无缺无重。"""
    la = _make_la()
    outs, tail = _drive_scripted(la, [
        [("w1", 0, 1), ("w2。", 1, 2)],
        [("w1", 0, 1), ("w2。", 1, 2), ("w3", 2, 3), ("w4", 3, 4)],  # 提交 w1w2。→ 句界裁剪至 2.0s
        [("w3", 2, 3), ("w4", 3, 4), ("w5", 4, 5)],   # 裁剪后新帧：未提交尾重现后继续
        [("w5", 4, 5), ("m1", 5, 6), ("m2", 6, 7)],
        [("m1", 5, 6), ("m2", 6, 7), ("m3。", 7, 8)],  # 中段 m1m2 本轮提交
    ])
    assert outs == [[], ["w1w2。"], ["w3w4"], ["w5"], ["m1m2"]], outs
    assert tail == "m3。", tail
    full = "".join(x for frags in outs for x in frags) + tail
    assert full == "w1w2。w3w4w5m1m2m3。", full
    assert _committed_texts(la) == ["w1", "w2。", "w3", "w4", "w5", "m1", "m2", "m3。"]
    return "中段 m1m2 经正常提交（非 flush）；句界裁剪触发；全序列无缺无重"


def test_la_no_duplicate_multi_cycle():
    """D2：8 轮连续裁剪周期（max_buffer_s=5 触发强制裁剪路径），16 个稳定词各恰好提交一次。"""
    la = _make_la(max_buffer_s=5.0)  # 无句界词 → 缓冲超限强制裁剪，覆盖强制裁剪路径
    rounds = []
    for r in range(8):  # 第 r 轮（喂后总长 2(r+1)s）：发出窗口 [max(0,2r-2), 2r+2) 的词
        lo = max(0, 2 * r - 2)
        hi = 2 * r + 2
        rounds.append([(f"w{k}", k, k + 1) for k in range(lo, hi)])
    outs, tail = _drive_scripted(la, rounds)
    texts = _committed_texts(la)
    assert texts == [f"w{k}" for k in range(16)], texts
    assert len(texts) == len(set(texts)), "存在重复提交"
    ends = [w["end"] for w in la.committed_words]
    assert all(b >= a for a, b in zip(ends, ends[1:])), "提交时间轴非单调"
    assert la.buffer_start_abs > 0, "强制裁剪路径未被触发"
    return "8 轮强制裁剪周期 16 词各一次、顺序与时间轴单调"


def test_la_boundary_overlap_no_dup_no_skip():
    """D3：裁剪边界 ±0.1s——重识别残留（同文本区间重叠，end 越界 +0.06s）不得重复；
    真实后继词（start 与前一提交词重叠 0.03s）不得跳过。"""
    la = _make_la()
    outs, tail = _drive_scripted(la, [
        [("a", 0, 1), ("b", 1, 2)],
        [("a", 0, 1), ("b", 1, 2), ("c", 2, 3)],           # 提交 a,b → 裁剪至 1.9s
        [("b", 1.9, 2.06), ("c", 2, 3), ("d", 3, 4)],      # b 的重识别残留（同文本）
        [("b", 1.9, 2.06), ("c", 2, 3), ("d", 3, 4), ("e", 4, 5)],  # 残留被去重守卫跳过
        [("e", 4, 5), ("f", 4.97, 5.6)],                   # f 与 e 时间重叠 0.03s：须提交
    ])
    full = "".join(x for frags in outs for x in frags) + tail
    assert full == "abcdef", full
    assert full.count("b") == 1, full
    assert _committed_texts(la) == ["a", "b", "c", "d", "e", "f"]
    assert len(la.divergence_events) == 0  # 同文本重识别不算失配
    return "边界残留 b 未重复；重叠 0.03s 的真实后继词 f 未跳过；abcdef 完整"


def test_la_flush_after_trim_idempotent():
    """D4：发生裁剪后 flush 提交全部剩余词且不重复已提交文本；二次 flush 返回空。
    另验证无任何提交时 flush 直接提交当前假设。"""
    la = _make_la()
    outs, tail = _drive_scripted(la, [
        [("a", 0, 1), ("b", 1, 2)],
        [("a", 0, 1), ("b", 1, 2), ("c", 2, 3)],  # 提交 a,b → 裁剪
    ])
    assert outs == [[], ["ab"]], outs
    assert tail == "c", tail
    assert la.flush() == "", "flush 必须幂等"
    assert _committed_texts(la) == ["a", "b", "c"]

    la2 = _make_la()
    outs2, tail2 = _drive_scripted(la2, [[("x", 0, 1), ("y", 1, 2)]])
    assert outs2 == [[]] and tail2 == "xy", (outs2, tail2)
    return "裁剪后 flush 提交剩余词一次且幂等；无提交时 flush 提交当前假设"


def test_la_punctuation_flap_no_stall():
    """D6：标点附着/纯标点词在解码轮间不稳定（'b,'→'b'、悬置'，'）不得卡死提交。

    真实样本实证：同一音频在不同缓冲长度下 '宿。'/'宿'/'，' 渲染不稳，
    逐字比较会让公共前缀永远停在原地。修复后规范化比较 + 纯标点词透明。"""
    la = _make_la()
    outs, tail = _drive_scripted(la, [
        [("a", 0, 1), ("b,", 1, 2)],                 # 首轮基线：'b,'
        [("a", 0, 1), ("b", 1, 2), ("c", 2, 3)],     # 'b,'→'b' 标点抖动：仍应一致并提交
        [("，", 1.98, 2.1), ("c", 2, 3), ("d", 3, 4)],  # 悬置纯标点词不得作锚点卡死
        [("d", 3, 4), ("e", 4, 5)],
    ])
    assert outs[1] == ["ab"], outs  # 标点抖动未阻断提交
    assert outs[2], "悬置纯标点词导致提交卡死"
    lexical = [t for t in _committed_texts(la) if t.strip("，。！？,.!?")]
    assert lexical == ["a", "b", "c", "d", "e"], lexical
    assert _committed_texts(la).count("，") <= 1
    return "标点抖动/悬置标点不卡死；a..e 完整无缺无重"


def test_la_production_path_no_gap():
    """D5：生产路径 run_exp_baseline_la.run_single_sample 全链路（真实分段器+线程队列+
    LLM 收尾），parrot 解码（每秒一词）→ 提交文本必须逐秒连续无缺无重，
    且 LLM 收到的文本 == 提交文本。"""
    import re
    import tempfile
    from types import SimpleNamespace
    import soundfile as sf
    from experiments.scripts.run_exp_baseline_la import LAExperiment
    from experiments.scripts.run_exp_latency import SampleInfo
    from src.asr.local_agreement_streamer import LocalAgreementStreamer

    la = LocalAgreementStreamer.__new__(LocalAgreementStreamer)
    la.sample_rate = 16000
    la.decode_trigger_s = 2.0
    la.trailing_margin_s = 0.0
    la.max_buffer_s = 15.0
    la.reset()

    def parrot_decode():
        """每秒一格的确定性词序列（绝对网格），覆盖当前缓冲全部完整秒；
        每第 4 词带句末标点，让句界裁剪路径在生产链路中被真实触发。"""
        base = la.buffer_start_abs
        end_abs = base + len(la.buffer) / la.sample_rate
        words = []
        k = max(0, int(np.floor(base)))
        while k + 1 <= end_abs + 1e-9:
            if k + 1 > base + 1e-9:
                text = f"w{k}。" if k % 4 == 3 else f"w{k}"
                words.append({"text": text, "start": k - base, "end": k + 1 - base})
            k += 1
        return words

    la._decode_buffer = parrot_decode

    class FakeLLM:
        def __init__(self):
            self.texts = []

        def reset_timings(self):
            pass

        def cache_prompt(self, text, pre_cache=None, is_end=False):
            if text:
                self.texts.append(text)
            return object()

        def generate(self, pre_cache=None, max_new_tokens=50):
            yield "好"

    fake_llm = FakeLLM()
    args = SimpleNamespace(chunk_duration=500, max_tokens=5,
                           save_full_response=False, save_fragments=True)
    exp = LAExperiment(la, fake_llm, args)

    audio = np.concatenate([_speech_fixture(), np.zeros(32000, dtype=np.float32)])  # 4s 语音 + 2s 静音
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "la_stub.wav"
        sf.write(str(wav), audio, 16000, subtype="PCM_16")
        sample = SampleInfo(sample_id="la_stub", dialog_id="0", turn_index=1, text="参考文本",
                            text_length=4, audio_file="la_stub.wav", audio_path=wav,
                            audio_duration=len(audio) / 16000, language="zh", dataset="stub",
                            duration_group="medium")
        result = exp.run_single_sample(sample)

    assert not result.error, result.error
    assert result.first_token_time > 0, "LLM 未产生首 token（收尾失败）"
    ids = [int(m) for m in re.findall(r"w(\d+)", result.transcribed_text)]
    assert ids == list(range(ids[0], ids[-1] + 1)), f"提交序列有缺口/重复: {ids}"
    assert ids[-1] >= 4, f"覆盖时长不足: {ids}"
    llm_text = "".join(fake_llm.texts)
    assert llm_text == result.transcribed_text.replace(" ", ""), \
        "LLM 收到的文本与 ASR 提交文本不一致"
    assert result.divergence_count == 0, result.divergence_count
    return (f"生产路径提交 w{ids[0]}..w{ids[-1]} 连续无缺无重；"
            f"LLM 收到的文本与提交文本一致；error 空")


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
        ("D1 LA 跨帧跳段（错帧 bug 复现序列）", test_la_cross_frame_no_gap),
        ("D2 LA 多裁剪周期无重复", test_la_no_duplicate_multi_cycle),
        ("D3 LA 裁剪边界不重复不跳过", test_la_boundary_overlap_no_dup_no_skip),
        ("D4 LA flush 裁剪后收尾幂等", test_la_flush_after_trim_idempotent),
        ("D5 LA 生产路径全链路无缺口", test_la_production_path_no_gap),
        ("D6 LA 标点抖动不卡死提交", test_la_punctuation_flap_no_stall),
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
