"""W1：统一时间轴 TTFA 直接实测（同批样本配对 A/B 独立请求）。

对应 PRE-PAPER-AUDIT P0-1/P0-2 与 v3.1 冻结协议（Gate 1 实现）：

- 时间锚点：physical_speech_end_ns = playout_start_ns + round(pse_sample×1e9/sample_rate)，
  全程 time.perf_counter_ns()；PSE 由能量法 + 固定 revision Silero 双法裁决；
- 因果回放：chunk 在其末样本计划到达时刻才释放（planned_release = start + 累计样本/sr），
  绝对 deadline 调度，逐 chunk 记录 planned/actual/scheduler error，提前释放记 error；
- System A 在 feed_end 前不得启动 full-audio ASR（断言 asr_start ≥ full_input_ready ≥ feed_end）；
- System B 保持原生成语义：explicit_flush_done ≤ pipeline_input_close ≤ asr_processing_done
  ≤ first_model_token；首句冻结后独立 TTS worker，LLM 继续生成；
- 无条件 INPUT_CLOSED sentinel（flush=None 不死锁）；worker 异常共享上报 + cancel；
- 句末检测：累计 token IDs 重解码，'.' 一字符 lookahead，EOS/max_tokens 裁决 pending；
- TTS：512B 应用读取粒度，1324B playable 阈值（22050Hz×16bit×30ms），格式校验
  （非 WAV 头/错误 JSON），connect/read/total 三级 timeout，格式错误整行 error；
- 闭合恒等式（原始 ns 残差严格 0）：
  TTFA_playable = (feed_end−pse) + (input_close−feed_end) + (first_token−input_close)
                + (text_ready−first_token) + (tts_req−text_ready) + (playable−tts_req)
- 配对独立请求：AB/BA 分层平衡顺序；主键 (sample_id, mode, repeat_idx)，
  配对键 (sample_id, repeat_idx)；repeat 0 计入子集三轮；按配对键派生 generation seed；
- checkpoint 原子写 + schema/run/config/schedule hash 校验，fail-closed。

用法：
  # self-test（无需 GPU/模型）：
  uv run python -m experiments.scripts.run_ttfa_unified --self-test
  # TTS 探活（冒烟前单独执行）：
  uv run python -m experiments.scripts.run_ttfa_unified --tts-probe --tts-url http://127.0.0.1:20401
  # 正式（显式路径，不用 glob 猜最新）：
  uv run python -m experiments.scripts.run_ttfa_unified \
      --sample-list experiments/results/revision/r1_stats/repeat_subset_ids.json \
      --json-dir experiments/datasets/processed/json --audio-dir experiments/datasets/processed/audio \
      --datasets crosswoz multiwoz \
      --tts-url http://127.0.0.1:20401 --silero-ref <pinned-commit> \
      --output-dir experiments/results/revision/r7_ttfa_unified --run-id r7_main
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import queue
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

SCHEMA_VERSION = "ttfa_unified/1"
ANALYSIS_SR = 16000
CHUNK_MS = 500
PSE_WINDOW_MS = 25
PSE_HOP_MS = 10
PSE_ENERGY_DB_OFFSET = 6.0        # 底噪 +6dB
PSE_RELATIVE_DB = -40.0           # 全文件 RMS −40dB 下限
PSE_NOISE_PERCENTILE = 10.0       # 底噪估计：RMS 最低 10% 帧均值
PSE_SHORT_AUDIO_S = 3.0           # 短音频底噪 fallback 阈值
PSE_ARBITRATE_MS = 200.0          # 两法差 ≤200ms 取 energy
SILERO_PARAMS = {"threshold": 0.5, "min_speech_duration_ms": 250,
                 "min_silence_duration_ms": 100, "speech_pad_ms": 30}
PCM_SAMPLE_RATE = 22050
PCM_BYTES_PER_SAMPLE = 2
PLAYABLE_BYTES = math.ceil(0.030 * PCM_SAMPLE_RATE) * PCM_BYTES_PER_SAMPLE  # 1324
TTS_READ_GRANULE = 512
TTS_CONNECT_TIMEOUT_S = 5.0
TTS_READ_TIMEOUT_S = 30.0
PAIR_DEADLINE_S = 900.0           # 每 pair 总 deadline
SENT_END_HARD = set("。！？!?")     # 免 lookahead 句末
TERMINAL_STATES = ("success", "error", "cancelled", "timeout")

EVENT_FIELDS = [
    "playout_start_ns", "physical_speech_end_ns", "feed_end_ns",
    "explicit_flush_start_ns", "explicit_flush_done_ns", "pipeline_input_close_ns",
    "full_input_ready_ns", "asr_start_ns", "asr_complete_ns", "last_asr_commit_ns",
    "asr_processing_done_ns", "first_model_token_ns", "first_content_token_ns",
    "first_sentence_boundary_ns", "generation_end_ns", "tts_request_start_ns",
    "tts_response_headers_ns", "first_pcm_byte_ns", "first_playable_pcm_ns", "tts_done_ns",
]


# ============================================================ 基础工具

def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def seed_for_pair(sample_id: str, repeat_idx: int) -> int:
    """按配对键派生 generation seed（canonical JSON + SHA-256；不用内置 hash()）。"""
    digest = sha256_text(canonical_json({"sample_id": sample_id, "repeat_idx": repeat_idx}))
    return int(digest[:8], 16)


def now_ns() -> int:
    return time.perf_counter_ns()


# ============================================================ PSE 分析

def _db_to_amplitude(db: float) -> float:
    return 10.0 ** (db / 20.0)


def load_analysis_waveform(wav_path: str):
    """加载并重采样为 16kHz mono float32；返回 (waveform, wav_sha256, analysis_sha256, loader_meta)。"""
    import soundfile as sf
    data, sr = sf.read(wav_path, dtype="float32")
    loader = {"reader": "soundfile", "orig_sr": sr, "orig_channels": 1 if data.ndim == 1 else data.shape[1],
              "orig_dtype": "float32", "resampler": None}
    if data.ndim > 1:
        data = data.mean(axis=1)  # downmix: 均值
        loader["downmix"] = "mean"
    else:
        loader["downmix"] = "none"
    if sr != ANALYSIS_SR:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=ANALYSIS_SR)
        loader["resampler"] = f"librosa.resample({sr}->{ANALYSIS_SR})"
    data = np.ascontiguousarray(data, dtype=np.float32)
    analysis_hash = hashlib.sha256(data.tobytes()).hexdigest()
    return data, sha256_file(wav_path), analysis_hash, loader


def energy_pse_sample(wave: np.ndarray) -> int | None:
    """能量法 PSE：最后一个超门限帧的排他右边界（sample 下标，范围 [0, N]）。

    门限 = max(底噪估计+6dB, 全文件 RMS−40dB) 的线性幅值；底噪 = RMS 最低 10% 帧均值
    （音频 <3s 时 fallback 为 全局 RMS×0.1）。尾窗补零；严格大于比较；round 后 clamp。
    """
    n = len(wave)
    if n == 0 or not np.all(np.isfinite(wave)):
        return None
    win = int(ANALYSIS_SR * PSE_WINDOW_MS / 1000)   # 400
    hop = int(ANALYSIS_SR * PSE_HOP_MS / 1000)      # 160
    starts = list(range(0, n, hop))
    rms = []
    for s in starts:
        frame = wave[s:s + win]
        if len(frame) < win:
            frame = np.pad(frame, (0, win - len(frame)))  # 尾窗补零
        rms.append(float(np.sqrt(np.mean(frame ** 2))) if len(frame) else 0.0)
    rms = np.array(rms)
    global_rms = float(np.sqrt(np.mean(wave ** 2)))
    if global_rms == 0:
        return None
    duration_s = n / ANALYSIS_SR
    if duration_s < PSE_SHORT_AUDIO_S:
        noise = global_rms * 0.1
    else:
        k = max(1, int(len(rms) * PSE_NOISE_PERCENTILE / 100))
        noise = float(np.sort(rms)[:k].mean())
    thr = max(noise * _db_to_amplitude(PSE_ENERGY_DB_OFFSET),
              global_rms * _db_to_amplitude(PSE_RELATIVE_DB))
    above = np.nonzero(rms > thr)[0]  # 严格大于
    if len(above) == 0:
        return None
    last_frame = int(above[-1])
    end = min((starts[last_frame] + win), n)  # clamp 到 [0, N]
    return int(end)


def silero_pse_sample(wave: np.ndarray, get_speech_timestamps) -> int | None:
    """Silero 法 PSE：最后一个 speech 段的 end（参数固定见 SILERO_PARAMS）。"""
    import torch
    ts = get_speech_timestamps(torch.from_numpy(wave), sampling_rate=ANALYSIS_SR,
                               **SILERO_PARAMS)
    if not ts:
        return None
    return int(min(ts[-1]["end"], len(wave)))


def analyze_pse(wav_path: str, get_speech_timestamps=None) -> dict:
    """双法 PSE 裁决。差 ≤200ms 取 energy；>200ms 取 Silero 并标记。

    任一算法无 speech/失败 → fail-closed（返回 error 字段，调用方记该行 error）。
    """
    wave, wav_hash, analysis_hash, loader = load_analysis_waveform(wav_path)
    out = {"wav_sha256": wav_hash, "analysis_waveform_sha256": analysis_hash,
           "analysis_sr": ANALYSIS_SR, "loader": loader,
           "pse_window_ms": PSE_WINDOW_MS, "pse_hop_ms": PSE_HOP_MS,
           "silero_params": SILERO_PARAMS}
    try:
        e = energy_pse_sample(wave)
    except Exception as exc:
        e = None
        out["energy_error"] = str(exc)
    s = None
    if get_speech_timestamps is not None:
        try:
            s = silero_pse_sample(wave, get_speech_timestamps)
        except Exception as exc:
            s = None
            out["silero_error"] = str(exc)
    out["energy_pse_sample"] = e
    out["silero_pse_sample"] = s
    if e is None and s is None:
        out["error"] = "pse_no_speech"
        return out
    if e is None or s is None:
        # 单算法失败：fail-closed，不用存活算法猜测（协议 §3.1.7）
        out["error"] = "pse_single_algorithm_failure"
        return out
    diff_ms = abs(e - s) / ANALYSIS_SR * 1000
    out["pse_diff_ms"] = round(diff_ms, 2)
    if diff_ms <= PSE_ARBITRATE_MS:
        out["physical_speech_end_sample"] = e
        out["pse_method"] = "energy"
    else:
        out["physical_speech_end_sample"] = s
        out["pse_method"] = "silero_fallback"
    return out


# ============================================================ 句末检测

class StreamingSentenceDetector:
    """流式句末检测：对累计重解码文本扫描；'.' 需一字符 lookahead。

    - 。！？!?：立即判定；
    - '.'：下一字符已知 → 数字夹击豁免，否则判定；下一字符未知（流尾）→ pending，
      final=True（EOS/max_tokens）时裁决：前一字符为数字 → 不判（疑似被截断的小数），
      否则判为句末。缩写（如 "Mr." 后接空格）不豁免——已声明限制。
    """

    def __init__(self):
        self.frozen_index: int | None = None

    def update(self, text: str, final: bool = False) -> int | None:
        if self.frozen_index is not None:
            return self.frozen_index
        for i, ch in enumerate(text):
            if ch in SENT_END_HARD:
                self.frozen_index = i
                return i
            if ch == ".":
                prev = text[i - 1] if i > 0 else ""
                if i + 1 < len(text):
                    if not (prev.isdigit() and text[i + 1].isdigit()):
                        self.frozen_index = i
                        return i
                elif final and not prev.isdigit():
                    self.frozen_index = i
                    return i
        return None


# ============================================================ TTS 客户端

def tts_probe(url: str, spk_id: str, speed: float) -> dict:
    """探活：确认服务返回预期裸 PCM；保存 status/Content-Type/magic 等，正式运行据此校验。"""
    import requests
    out = {"url": url, "spk_id": spk_id, "speed": speed}
    try:
        resp = requests.post(url, json={"tts_text": "探活", "spk_id": spk_id,
                                        "stream": True, "speed": speed},
                             stream=True, timeout=(TTS_CONNECT_TIMEOUT_S, TTS_READ_TIMEOUT_S))
        out["status"] = resp.status_code
        out["content_type"] = resp.headers.get("Content-Type")
        out["content_encoding"] = resp.headers.get("Content-Encoding")
        first = next(resp.iter_content(64), b"")
        out["magic_hex"] = first[:8].hex()
        out["looks_pcm"] = not (first.startswith(b"RIFF") or first.lstrip().startswith(b"{"))
        resp.close()
        out["ok"] = resp.status_code == 200 and out["looks_pcm"]
    except Exception as exc:
        out["ok"] = False
        out["error"] = str(exc)
    return out


def tts_measure(url: str, text: str, spk_id: str, speed: float, probe: dict,
                cancel_event: threading.Event, total_deadline_ns: int,
                requests_session=None, connect_timeout: float = TTS_CONNECT_TIMEOUT_S,
                read_timeout: float = TTS_READ_TIMEOUT_S) -> dict:
    """流式 TTS 首包测量。返回 tts_request_start/headers/first_byte/playable/done 等。

    应用层 512B 粒度；字节连续累积（奇数不丢半 sample）；playable = 累计完整 sample
    首达 1324B；HTTP/格式/对齐/零内容错误 → error 字段（调用方整行 error，不降级）。
    """
    import requests
    rec = {"tts_request_start_ns": now_ns(), "tts_response_headers_ns": None,
           "first_pcm_byte_ns": None, "first_playable_pcm_ns": None, "tts_done_ns": None,
           "tts_first_chunk_bytes": 0, "tts_playable_rms": None, "tts_playable_peak": None,
           "tts_total_bytes": 0}
    http = requests_session or requests
    resp = None
    try:
        resp = http.post(url, json={"tts_text": text, "spk_id": spk_id,
                                    "stream": True, "speed": speed},
                         stream=True, timeout=(connect_timeout, read_timeout))
        rec["tts_response_headers_ns"] = now_ns()
        if resp.status_code != 200:
            rec["error"] = f"tts_http_{resp.status_code}"
            return rec
        expected_ct = probe.get("content_type")
        got_ct = resp.headers.get("Content-Type")
        if expected_ct and got_ct != expected_ct:
            rec["error"] = f"tts_content_type_mismatch:{got_ct}"
            return rec
        buf = bytearray()
        first_checked = False
        for chunk in resp.iter_content(TTS_READ_GRANULE):
            if cancel_event.is_set():
                rec["error"] = "tts_cancelled"
                return rec
            if now_ns() > total_deadline_ns:
                rec["error"] = "tts_total_timeout"
                return rec
            if not chunk:
                continue
            now = now_ns()
            if rec["first_pcm_byte_ns"] is None:
                rec["first_pcm_byte_ns"] = now
                rec["tts_first_chunk_bytes"] = len(chunk)
            if not first_checked:
                first_checked = True
                if chunk.startswith(b"RIFF") or chunk.lstrip().startswith(b"{"):
                    rec["error"] = "tts_format_not_pcm"
                    return rec
            buf += chunk
            complete = len(buf) - (len(buf) % PCM_BYTES_PER_SAMPLE)
            if rec["first_playable_pcm_ns"] is None and complete >= PLAYABLE_BYTES:
                rec["first_playable_pcm_ns"] = now
                arr = np.frombuffer(bytes(buf[:PLAYABLE_BYTES - PLAYABLE_BYTES % 2]),
                                    dtype=np.int16).astype(np.float64)
                rec["tts_playable_rms"] = float(np.sqrt(np.mean(arr ** 2))) if len(arr) else 0.0
                rec["tts_playable_peak"] = float(np.abs(arr).max()) if len(arr) else 0.0
                break  # 只需首包；主动关闭
        rec["tts_total_bytes"] = len(buf)
        if rec["first_pcm_byte_ns"] is None:
            rec["error"] = "tts_empty_body"
        elif rec["first_playable_pcm_ns"] is None:
            rec["error"] = "tts_below_playable_threshold"
        return rec
    except Exception as exc:
        rec["error"] = f"tts_exception:{exc}"
        return rec
    finally:
        rec["tts_done_ns"] = now_ns()
        if resp is not None:
            try:
                resp.close()
            except Exception:
                pass


# ============================================================ 因果回放

class InputClosed:
    """无条件生命周期 sentinel（与音频数据分离）。"""


def playout_worker(audio: np.ndarray, sr: int, chunk_samples: int, out_queue: queue.Queue,
                   timings: dict, exc_queue: queue.Queue, cancel_event: threading.Event,
                   pse_sample: int | None = None):
    """chunk 在末样本计划到达时刻释放；绝对 deadline；逐 chunk scheduler error。

    提前释放不可能发生（等待到 deadline 后才 put），此处仍断言 actual >= planned。
    physical_speech_end_ns 按冻结公式由 pse_sample 映射到单调时钟。
    """
    try:
        playout_start = now_ns()
        timings["playout_start_ns"] = playout_start
        if pse_sample is not None:
            timings["physical_speech_end_ns"] = playout_start + round(pse_sample * 1e9 / sr)
        cum = 0
        chunk_log = []
        for cid, off in enumerate(range(0, len(audio), chunk_samples)):
            if cancel_event.is_set():
                break
            chunk = audio[off:off + chunk_samples]
            cum += len(chunk)  # 最后一块按实际样本数
            planned = playout_start + round(cum * 1e9 / sr)
            while True:
                now = now_ns()
                if now >= planned:
                    break
                time.sleep(min((planned - now) / 1e9 / 2, 0.005))
            actual = now_ns()
            if actual < planned:  # 断言：提前释放
                raise RuntimeError(f"chunk {cid} 提前释放: actual {actual} < planned {planned}")
            chunk_log.append({"chunk_id": cid, "planned_ns": planned,
                              "actual_ns": actual, "sched_err_ns": actual - planned})
            out_queue.put((cid, chunk))
        timings["feed_end_ns"] = now_ns()  # 最后 chunk 实际释放完成
        timings["chunk_log"] = chunk_log
    except Exception:
        exc_queue.put(("playout", traceback.format_exc()))
        cancel_event.set()
    finally:
        out_queue.put(InputClosed())


# ============================================================ 记录校验

def validate_record(rec: dict) -> list[str]:
    """schema + 因果偏序 + 闭合恒等式校验；返回违规列表（空=通过）。

    偏序按真实因果边（不是字段全序）：B 的 TTS 可在 generation_end 前启动
    （首句冻结即启动），故 generation_end 与 TTS 链无序约束；A 的 text_ready=generation_end。
    """
    errs = []
    if rec.get("schema_version") != SCHEMA_VERSION:
        errs.append("schema_version")
    if rec.get("terminal_state") not in TERMINAL_STATES:
        errs.append("terminal_state")
    if rec["terminal_state"] != "success":
        return errs  # 非成功行只要终态合法即可
    ev = rec.get("events", {})
    for f in ("playout_start_ns", "physical_speech_end_ns", "feed_end_ns",
              "pipeline_input_close_ns", "first_model_token_ns", "generation_end_ns",
              "tts_request_start_ns", "first_pcm_byte_ns", "first_playable_pcm_ns"):
        if ev.get(f) is None:
            errs.append(f"missing:{f}")
    if errs:
        return errs

    def le(x, y):
        if ev[y] < ev[x]:
            errs.append(f"order:{y}<{x}")

    le("playout_start_ns", "physical_speech_end_ns")
    le("physical_speech_end_ns", "feed_end_ns")
    le("feed_end_ns", "pipeline_input_close_ns")
    le("pipeline_input_close_ns", "first_model_token_ns")
    le("first_model_token_ns", "generation_end_ns")
    le("tts_request_start_ns", "first_pcm_byte_ns")
    le("first_pcm_byte_ns", "first_playable_pcm_ns")
    mode = rec.get("mode")
    if mode == "streaming":
        for f in ("explicit_flush_start_ns", "explicit_flush_done_ns", "asr_processing_done_ns"):
            if ev.get(f) is None:
                errs.append(f"missing:{f}")
        if not any(e.startswith(("missing", "order")) for e in errs):
            le("feed_end_ns", "explicit_flush_start_ns")
            le("explicit_flush_start_ns", "explicit_flush_done_ns")
            le("explicit_flush_done_ns", "pipeline_input_close_ns")
            le("pipeline_input_close_ns", "asr_processing_done_ns")
            le("asr_processing_done_ns", "first_model_token_ns")
        if rec.get("sentence_end_found") and not rec.get("sentence_fallback"):
            if ev.get("first_sentence_boundary_ns") is None:
                errs.append("missing:first_sentence_boundary_ns")
            else:
                le("first_model_token_ns", "first_sentence_boundary_ns")
        text_ready = ev["first_sentence_boundary_ns"] or ev["generation_end_ns"]
    elif mode == "non-streaming":
        for f in ("full_input_ready_ns", "asr_start_ns", "asr_complete_ns"):
            if ev.get(f) is None:
                errs.append(f"missing:{f}")
        if not any(e.startswith(("missing", "order")) for e in errs):
            if not (ev["asr_start_ns"] >= ev["full_input_ready_ns"] >= ev["feed_end_ns"]):
                errs.append("order:a_chain")
            le("asr_start_ns", "asr_complete_ns")
            le("asr_complete_ns", "first_model_token_ns")
        text_ready = ev["generation_end_ns"]
    else:
        errs.append("mode")
        return errs
    if text_ready is not None and ev["tts_request_start_ns"] < text_ready:
        errs.append("order:tts_request<text_ready")
    # 闭合恒等式（原始 ns 严格 0 残差）
    ttfa = ev["first_playable_pcm_ns"] - ev["physical_speech_end_ns"]
    comp = ((ev["feed_end_ns"] - ev["physical_speech_end_ns"])
            + (ev["pipeline_input_close_ns"] - ev["feed_end_ns"])
            + (ev["first_model_token_ns"] - ev["pipeline_input_close_ns"])
            + (text_ready - ev["first_model_token_ns"])
            + (ev["tts_request_start_ns"] - text_ready)
            + (ev["first_playable_pcm_ns"] - ev["tts_request_start_ns"]))
    if ttfa != comp:
        errs.append(f"closure:residual={ttfa - comp}ns")
    if ttfa < 0:
        errs.append("ttfa_negative")
    return errs


# ============================================================ A/B 运行

def _base_record(sample, mode, repeat_idx, pse, cfg_hash, sched_hash, run_id):
    return {"schema_version": SCHEMA_VERSION, "run_id": run_id,
            "sample_id": sample["sample_id"], "language": sample["language"],
            "duration_group": sample["duration_group"], "mode": mode,
            "repeat_idx": repeat_idx, "terminal_state": None,
            "config_hash": cfg_hash, "schedule_hash": sched_hash,
            "wav_sha256": pse["wav_sha256"],
            "analysis_waveform_sha256": pse["analysis_waveform_sha256"],
            "physical_speech_end_sample": pse["physical_speech_end_sample"],
            "pse_method": pse["pse_method"], "pse_diff_ms": pse["pse_diff_ms"],
            "events": {f: None for f in EVENT_FIELDS}, "chunk_log": [],
            "tts": {}, "response_token_count": 0, "generation_stop_reason": None,
            "sentence_end_found": False, "sentence_fallback": False,
            "tts_text_source": None, "tts_n_chars": 0,
            "tts_text_sha256": None, "generation_seed": None, "error": ""}


def run_streaming(sample, audio, sr, models, pse, tts_cfg, probe, seed, cancel_event):
    """System B：分段→增量预填→is_end 后生成→首句冻结即 TTS（LLM 不中断）。"""
    rec = _base_record(sample, "streaming", sample["repeat_idx"], pse,
                       tts_cfg["config_hash"], tts_cfg["schedule_hash"], tts_cfg["run_id"])
    rec["generation_seed"] = seed
    ev = rec["events"]
    exc_q: queue.Queue = queue.Queue()
    chunk_samples = int(sr * CHUNK_MS / 1000)
    chunk_q: queue.Queue = queue.Queue()
    asr_q: queue.Queue = queue.Queue()
    text_q: queue.Queue = queue.Queue()

    segmenter = models["segmenter"]
    asr_processor = models["asr"]
    llm = models["llm"]
    convert = models["convert_audio_segment"]
    cache_box = [models["new_asr_cache"]()]

    def segmentation_worker():
        try:
            state = segmenter.create_state()
            while True:
                item = chunk_q.get()
                if isinstance(item, InputClosed):
                    break
                _, chunk = item
                stream_segment, state = segmenter.process_audio(chunk, state)
                if stream_segment:
                    seg_id = f"seg_{stream_segment.segment_id:03d}"
                    asr_q.put(convert(stream_segment, seg_id,
                                      stream_segment.segment_id == 1, False))
            ev["explicit_flush_start_ns"] = now_ns()
            remaining, _ = segmenter.flush(state)
            if remaining is not None and len(remaining.audio) > 0:
                asr_q.put(convert(remaining, f"seg_{remaining.segment_id:03d}", False, True))
            asr_q.put(InputClosed())  # 无条件 sentinel（flush=None 不死锁）
            ev["explicit_flush_done_ns"] = now_ns()
        except Exception:
            exc_q.put(("segmentation", traceback.format_exc()))
            cancel_event.set()

    def collector():
        try:
            while True:
                try:
                    item = asr_q.get(timeout=0.1)
                except queue.Empty:
                    if cancel_event.is_set():
                        return
                    continue
                if isinstance(item, InputClosed):
                    ev["pipeline_input_close_ns"] = now_ns()
                    return
                cache_box[0].add_segment(item)
        except Exception:
            exc_q.put(("collector", traceback.format_exc()))
            cancel_event.set()

    def transcriber():
        try:
            while True:
                if cancel_event.is_set():
                    return
                closed = ev["pipeline_input_close_ns"] is not None
                cache = cache_box[0]
                if closed and not cache.waiting_segment_queue:
                    break
                if not cache.waiting_segment_queue or cache.is_processing():
                    time.sleep(0.005)
                    continue
                cache, text, is_final = asr_processor.transcribe_audio_segment(cache)
                cache_box[0] = cache
                if text:
                    ev["last_asr_commit_ns"] = now_ns()
                    text_q.put((text, False))
                if is_final:
                    break
            ev["asr_processing_done_ns"] = now_ns()
            text_q.put(("", True))
        except Exception:
            exc_q.put(("transcriber", traceback.format_exc()))
            cancel_event.set()

    threads = [threading.Thread(target=playout_worker, args=(
                   audio, sr, chunk_samples, chunk_q, ev, exc_q, cancel_event,
                   pse["physical_speech_end_sample"])),
               threading.Thread(target=segmentation_worker),
               threading.Thread(target=collector),
               threading.Thread(target=transcriber)]
    for t in threads:
        t.start()

    tts_holder: dict = {}
    try:
        kv = None
        while True:  # 增量预填（主线程消费 text_q）
            if cancel_event.is_set():
                raise RuntimeError("cancelled: " + "; ".join(w for w, _ in list(exc_q.queue)))
            try:
                text, is_end = text_q.get(timeout=0.1)
            except queue.Empty:
                continue
            if text or is_end:
                kv = llm.cache_prompt(text, pre_cache=kv, is_end=is_end)
            if is_end:
                break
        rec["chunk_log"] = ev.pop("chunk_log", [])

        import torch  # 延迟导入（self-test 免 torch 也可注入 fake）
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        det = StreamingSentenceDetector()
        token_ids: list[int] = []
        decode_fn = models["decode_fn"]
        stop = "max_tokens"
        for meta in llm.generate_with_meta(pre_cache=kv, max_new_tokens=tts_cfg["max_tokens"]):
            now = now_ns()
            if ev["first_model_token_ns"] is None:
                ev["first_model_token_ns"] = now
            if meta["is_eos"]:
                stop = "eos"
                break
            token_ids.append(meta["token_id"])
            rec["response_token_count"] += 1
            if ev["first_content_token_ns"] is None and meta["decoded_text"]:
                ev["first_content_token_ns"] = now
            text = decode_fn(token_ids)
            idx = det.update(text, final=False)
            if idx is not None and ev["first_sentence_boundary_ns"] is None:
                ev["first_sentence_boundary_ns"] = now  # 文本冻结时刻
                rec["sentence_end_found"] = True
                tts_text = text[:idx + 1]
                rec["tts_text_source"] = "first_sentence"
                rec["tts_n_chars"] = len(tts_text)
                rec["tts_text_sha256"] = sha256_text(tts_text)
                threading.Thread(target=_tts_into, args=(
                    tts_holder, tts_text, tts_cfg, probe, cancel_event)).start()
        ev["generation_end_ns"] = now_ns()
        rec["generation_stop_reason"] = stop
        full_text = decode_fn(token_ids)
        if rec["response_token_count"] == 0 or not full_text.strip():
            rec["error"] = "zero_content_response"
            rec["terminal_state"] = "error"
            return rec
        if ev["first_sentence_boundary_ns"] is None:
            # EOS/max_tokens 后裁决末尾 pending；再无句末 → fallback capped full response
            idx = det.update(full_text, final=True)
            rec["sentence_fallback"] = idx is None
            rec["sentence_end_found"] = idx is not None
            tts_text = full_text if idx is None else full_text[:idx + 1]
            rec["tts_text_source"] = "capped_full_response" if idx is None else "first_sentence"
            rec["tts_n_chars"] = len(tts_text)
            rec["tts_text_sha256"] = sha256_text(tts_text)
            threading.Thread(target=_tts_into, args=(
                tts_holder, tts_text, tts_cfg, probe, cancel_event)).start()
        deadline = now_ns() + int(PAIR_DEADLINE_S * 1e9)
        while "done" not in tts_holder:
            if now_ns() > deadline:
                tts_holder["rec"] = {"error": "tts_join_timeout"}
                tts_holder["done"] = True
                break
            time.sleep(0.005)
        rec["tts"] = tts_holder.get("rec", {})
        for k in ("tts_request_start_ns", "tts_response_headers_ns", "first_pcm_byte_ns",
                  "first_playable_pcm_ns", "tts_done_ns"):
            ev[k] = rec["tts"].get(k)
        if rec["tts"].get("error"):
            rec["error"] = rec["tts"]["error"]
            rec["terminal_state"] = "error"
            return rec
        rec["terminal_state"] = "success"
    except Exception:
        rec["error"] = traceback.format_exc()[-500:]
        rec["terminal_state"] = "error"
    finally:
        cancel_event.set()
        for t in threads:
            t.join(timeout=5)
    return rec


def _tts_into(holder, text, tts_cfg, probe, cancel_event):
    deadline = now_ns() + int(tts_cfg.get("tts_total_timeout_s", 120.0) * 1e9)
    holder["rec"] = tts_measure(tts_cfg["url"], text, tts_cfg["spk_id"], tts_cfg["speed"],
                                probe, cancel_event, deadline,
                                connect_timeout=tts_cfg.get("connect_timeout_s", TTS_CONNECT_TIMEOUT_S),
                                read_timeout=tts_cfg.get("read_timeout_s", TTS_READ_TIMEOUT_S))
    holder["done"] = True


def run_non_streaming(sample, audio, sr, models, pse, tts_cfg, probe, seed, cancel_event):
    """System A：等待 feed_end → full ASR → 完整 capped 回复后 TTS。"""
    rec = _base_record(sample, "non-streaming", sample["repeat_idx"], pse,
                       tts_cfg["config_hash"], tts_cfg["schedule_hash"], tts_cfg["run_id"])
    rec["generation_seed"] = seed
    ev = rec["events"]
    exc_q: queue.Queue = queue.Queue()
    chunk_samples = int(sr * CHUNK_MS / 1000)
    sink_q: queue.Queue = queue.Queue()
    playout_t = threading.Thread(target=playout_worker, args=(
        audio, sr, chunk_samples, sink_q, ev, exc_q, cancel_event,
        pse["physical_speech_end_sample"]))
    playout_t.start()
    try:
        # 排空 sink 队列（A 不使用流式中间结果，但时间轴一致）
        while True:
            item = sink_q.get()
            if isinstance(item, InputClosed):
                break
        playout_t.join(timeout=30)
        rec["chunk_log"] = ev.pop("chunk_log", [])
        ev["full_input_ready_ns"] = ev["feed_end_ns"]
        ev["pipeline_input_close_ns"] = ev["feed_end_ns"]
        ev["asr_start_ns"] = now_ns()  # 断言在 validate 中：>= full_input_ready >= feed_end
        asr_processor = models["asr"]
        llm = models["llm"]
        asr_result = asr_processor.transcribe_complete_audio(
            audio_path=sample["audio_path"], audio_data=audio, sample_rate=sr)
        ev["asr_complete_ns"] = now_ns()
        ev["asr_processing_done_ns"] = ev["asr_complete_ns"]
        ev["last_asr_commit_ns"] = ev["asr_complete_ns"]
        text = asr_result["text"]
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        kv = llm.cache_prompt(text, pre_cache=None, is_end=True)
        token_ids: list[int] = []
        decode_fn = models["decode_fn"]
        stop = "max_tokens"
        det = StreamingSentenceDetector()
        for meta in llm.generate_with_meta(pre_cache=kv, max_new_tokens=tts_cfg["max_tokens"]):
            now = now_ns()
            if ev["first_model_token_ns"] is None:
                ev["first_model_token_ns"] = now
            if meta["is_eos"]:
                stop = "eos"
                break
            token_ids.append(meta["token_id"])
            rec["response_token_count"] += 1
            if ev["first_content_token_ns"] is None and meta["decoded_text"]:
                ev["first_content_token_ns"] = now
            det.update(decode_fn(token_ids), final=False)  # 记录用；A 的 TTS 等全文
        ev["generation_end_ns"] = now_ns()
        rec["generation_stop_reason"] = stop
        full_text = decode_fn(token_ids)
        if rec["response_token_count"] == 0 or not full_text.strip():
            rec["error"] = "zero_content_response"
            rec["terminal_state"] = "error"
            return rec
        rec["sentence_end_found"] = det.frozen_index is not None
        rec["tts_text_source"] = "capped_full_response"
        rec["tts_n_chars"] = len(full_text)
        rec["tts_text_sha256"] = sha256_text(full_text)
        tts_holder: dict = {}
        th = threading.Thread(target=_tts_into, args=(
            tts_holder, full_text, tts_cfg, probe, cancel_event))
        th.start()
        th.join(timeout=tts_cfg.get("tts_total_timeout_s", 120.0) + 30)
        rec["tts"] = tts_holder.get("rec", {"error": "tts_join_timeout"})
        for k in ("tts_request_start_ns", "tts_response_headers_ns", "first_pcm_byte_ns",
                  "first_playable_pcm_ns", "tts_done_ns"):
            ev[k] = rec["tts"].get(k)
        if rec["tts"].get("error"):
            rec["error"] = rec["tts"]["error"]
            rec["terminal_state"] = "error"
            return rec
        rec["terminal_state"] = "success"
    except Exception:
        rec["error"] = traceback.format_exc()[-500:]
        rec["terminal_state"] = "error"
    finally:
        cancel_event.set()
        playout_t.join(timeout=5)
    return rec


# ============================================================ 调度

def build_schedule(samples: list[dict], subset_ids: list[str]) -> list[dict]:
    """生成确定性 AB/BA 平衡任务表。

    - 主实验 repeat 0：25 条 A→B + 25 条 B→A，按语言×时长分层平衡；
    - 子集 10 条：5 条 (AB,BA,AB) + 5 条 (BA,AB,BA)，repeat 0 计入三轮；
    - 非子集样本仅 repeat 0；任务按 pass（repeat）分批、批内按分层顺序。
    """
    ordered = sorted(samples, key=lambda s: (s["language"], s["duration_group"], s["sample_id"]))
    subset = [s for s in ordered if s["sample_id"] in set(subset_ids)]
    if len(subset) != len(subset_ids):
        raise SystemExit(f"子集 ID 未全部命中样本清单: {len(subset)}/{len(subset_ids)}")
    patterns = {}  # sample_id -> [order_r0, order_r1, order_r2]
    for i, s in enumerate(subset):
        patterns[s["sample_id"]] = ["AB", "BA", "AB"] if i % 2 == 0 else ["BA", "AB", "BA"]
    n_ab = sum(1 for p in patterns.values() if p[0] == "AB")
    orders = {}
    for sid, p in patterns.items():
        orders[sid] = p[0]
    need_ab = len(ordered) // 2 - n_ab
    if need_ab < 0 or need_ab > len(ordered) - len(subset):
        raise SystemExit("AB/BA 平衡不可行（子集起点分布失衡）")
    for s in ordered:
        sid = s["sample_id"]
        if sid in orders:
            continue
        orders[sid] = "AB" if need_ab > 0 else "BA"
        need_ab -= 1 if orders[sid] == "AB" else 0
    final_ab = sum(1 for o in orders.values() if o == "AB")
    if final_ab != len(ordered) - final_ab:
        raise SystemExit(f"AB/BA 不平衡: {final_ab}/{len(ordered) - final_ab}")

    tasks = []
    seq = 0
    for pass_idx in (0, 1, 2):
        for s in ordered:
            sid = s["sample_id"]
            if pass_idx > 0 and sid not in patterns:
                continue
            order = patterns[sid][pass_idx] if sid in patterns else orders[sid]
            for mode in (("non-streaming", "streaming") if order == "AB"
                         else ("streaming", "non-streaming")):
                tasks.append({"seq": seq, "sample_id": sid, "mode": mode,
                              "repeat_idx": pass_idx, "order": order})
                seq += 1
    return tasks


def schedule_hash(tasks: list[dict]) -> str:
    return sha256_text(canonical_json(tasks))


def config_hash(cfg: dict) -> str:
    return sha256_text(canonical_json(cfg))


# ============================================================ checkpoint

class Checkpoint:
    """JSONL checkpoint：首行 header（run/config/schedule hash），后续每行一条终态记录。

    原子：append + flush + fsync；损坏或 hash 不匹配 → SystemExit（fail-closed）。
    error key 不静默重跑——已完成（任意终态）的 key 直接跳过。
    """

    def __init__(self, path: Path, run_id: str, cfg_hash: str, sched_hash: str):
        self.path = path
        self.done: dict[str, str] = {}  # key -> terminal_state
        if path.exists():
            lines = path.read_text(encoding="utf-8").splitlines()
            if not lines:
                raise SystemExit(f"checkpoint 为空但存在: {path}（停止）")
            try:
                header = json.loads(lines[0])
            except json.JSONDecodeError:
                raise SystemExit(f"checkpoint header 损坏: {path}（停止）")
            if (header.get("run_id") != run_id or header.get("config_hash") != cfg_hash
                    or header.get("schedule_hash") != sched_hash
                    or header.get("schema_version") != SCHEMA_VERSION):
                raise SystemExit("checkpoint hash/run_id/schema 不匹配（配置变化须新建 run，停止）")
            for ln in lines[1:]:
                try:
                    rec = json.loads(ln)
                except json.JSONDecodeError:
                    raise SystemExit(f"checkpoint 记录损坏: {path}（停止）")
                key = self.key_of(rec)
                if rec.get("terminal_state") in TERMINAL_STATES:
                    self.done[key] = rec["terminal_state"]
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            header = {"type": "header", "schema_version": SCHEMA_VERSION,
                      "run_id": run_id, "config_hash": cfg_hash, "schedule_hash": sched_hash}
            tmp = path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(json.dumps(header, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)

    @staticmethod
    def key_of(rec: dict) -> str:
        return f"{rec['sample_id']}|{rec['mode']}|{rec['repeat_idx']}"

    def append(self, rec: dict) -> None:
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        self.done[self.key_of(rec)] = rec["terminal_state"]


# ============================================================ 汇总与 QA

def ttfa_ms(rec: dict, field: str = "first_playable_pcm_ns") -> float:
    ev = rec["events"]
    return (ev[field] - ev["physical_speech_end_ns"]) / 1e6


def summarize(records: list[dict]) -> list[dict]:
    import csv as _csv  # noqa
    rows = []
    ok = [r for r in records if r["terminal_state"] == "success"]
    for mode in ("streaming", "non-streaming"):
        for lang in ("zh", "en", "ALL"):
            sub = [r for r in ok if r["mode"] == mode
                   and (lang == "ALL" or r["language"] == lang) and r["repeat_idx"] == 0]
            if not sub:
                continue
            for metric, field in (("ttfa_playable_ms", "first_playable_pcm_ns"),
                                  ("ttfa_received_ms", "first_pcm_byte_ns")):
                v = np.array([ttfa_ms(r, field) for r in sub])
                rows.append({"mode": mode, "language": lang, "metric": metric,
                             "n": len(sub), "mean": f"{v.mean():.1f}",
                             "std": f"{v.std(ddof=1):.1f}",
                             "p50": f"{np.percentile(v, 50):.1f}",
                             "p90": f"{np.percentile(v, 90):.1f}",
                             "p95": f"{np.percentile(v, 95):.1f}"})
            # 组件均值（playable 口径）
            comp = {}
            for r in sub:
                ev = r["events"]
                tr = ev["first_sentence_boundary_ns"] or ev["generation_end_ns"]
                parts = {"t_trailing_feed_wait": ev["feed_end_ns"] - ev["physical_speech_end_ns"],
                         "t_flush_to_close": ev["pipeline_input_close_ns"] - ev["feed_end_ns"],
                         "t_close_to_first_token": ev["first_model_token_ns"] - ev["pipeline_input_close_ns"],
                         "t_first_token_to_text_ready": tr - ev["first_model_token_ns"],
                         "t_text_ready_to_tts_req": ev["tts_request_start_ns"] - tr,
                         "t_tts_to_playable": ev["first_playable_pcm_ns"] - ev["tts_request_start_ns"]}
                for k, x in parts.items():
                    comp.setdefault(k, []).append(x / 1e6)
            for k, xs in comp.items():
                rows.append({"mode": mode, "language": lang, "metric": k, "n": len(sub),
                             "mean": f"{np.mean(xs):.1f}", "std": f"{np.std(xs, ddof=1):.1f}",
                             "p50": "", "p90": "", "p95": ""})
    return rows




def subset_cv(records: list[dict], subset_ids: list[str]) -> list[dict]:
    """重复子集：同 (sample_id, mode) 恰三条有效记录 → CV（ddof=1）。"""
    out = []
    per: dict[tuple[str, str], list[float]] = {}
    for r in records:
        if r["sample_id"] in set(subset_ids) and r["terminal_state"] == "success":
            per.setdefault((r["sample_id"], r["mode"]), []).append(ttfa_ms(r))
    for (sid, mode), vals in sorted(per.items()):
        if len(vals) != 3:
            out.append({"sample_id": sid, "mode": mode, "n_valid": len(vals),
                        "cv_pct": "", "note": "缺轮次"})
            continue
        arr = np.array(vals)
        out.append({"sample_id": sid, "mode": mode, "n_valid": 3,
                    "cv_pct": f"{arr.std(ddof=1) / arr.mean() * 100:.2f}", "note": ""})
    return out


def qa_records(records: list[dict], tasks: list[dict]) -> list[str]:
    """结果级 QA（Gate 1 §6.2 / 再审 §6.4 对应项）；返回问题列表。"""
    problems = []
    expected: dict[str, int] = {}
    for t in tasks:
        key = f"{t['sample_id']}|{t['mode']}|{t['repeat_idx']}"
        expected[key] = expected.get(key, 0) + 1
    got: dict[str, int] = {}
    pair_hash: dict[str, str] = {}
    for r in records:
        key = Checkpoint.key_of(r)
        got[key] = got.get(key, 0) + 1
        v = validate_record(r)
        if v:
            problems.append(f"{key}: {';'.join(v)}")
        pk = f"{r['sample_id']}|{r['repeat_idx']}"
        h = r.get("wav_sha256", "")
        if pk in pair_hash and pair_hash[pk] != h:
            problems.append(f"{pk}: A/B WAV hash 不一致")
        pair_hash.setdefault(pk, h)
    for key, cnt in expected.items():
        if got.get(key, 0) != cnt:
            problems.append(f"{key}: 终态记录数 {got.get(key, 0)} != 预期 {cnt}")
    for key in got:
        if key not in expected:
            problems.append(f"{key}: 计划外记录")
    return problems


# ============================================================ self-test

class _FakeSeg:
    def __init__(self, segment_id, audio):
        self.segment_id = segment_id
        self.audio = audio


class _FakeSegmenter:
    """每 2 个 chunk 出一个段；flush 行为可配（None / 有音频段）。"""

    def __init__(self, flush_none=False):
        self.flush_none = flush_none
        self.n = 0

    def create_state(self):
        return {"buf": []}

    def process_audio(self, chunk, state):
        self.n += 1
        if self.n % 2 == 0:
            return _FakeSeg(self.n // 2, chunk), state
        return None, state

    def flush(self, state):
        if self.flush_none:
            return None, state
        return _FakeSeg(99, np.zeros(1600, dtype=np.float32)), state


class _FakeCache:
    def __init__(self):
        self.waiting_segment_queue = []

    def add_segment(self, seg):
        self.waiting_segment_queue.append(seg)

    def is_processing(self):
        return False


class _FakeASR:
    """按段依次吐出脚本化文本；complete 用于 A 模式。fail=True 时抛异常。"""

    def __init__(self, frag_texts, full_text, fail=False):
        self.frag_texts = list(frag_texts)
        self.full_text = full_text
        self.fail = fail

    def transcribe_audio_segment(self, cache):
        if self.fail:
            raise RuntimeError("fake asr boom")
        seg = cache.waiting_segment_queue.pop(0)
        is_final = seg.segment_id == 99
        text = self.frag_texts.pop(0) if self.frag_texts else ""
        return cache, text, is_final

    def transcribe_complete_audio(self, audio_path, audio_data=None, sample_rate=None):
        if self.fail:
            raise RuntimeError("fake asr boom")
        return {"text": self.full_text}


class _FakeLLM:
    """cache_prompt/generate_with_meta 协议；token 流脚本化（pieces + eos 位置）。"""

    def __init__(self, pieces, eos_at_end=True):
        self.pieces = pieces
        self.eos_at_end = eos_at_end

    def cache_prompt(self, prompt, pre_cache=None, is_end=False):
        return {"kv": True}

    def generate_with_meta(self, pre_cache, max_new_tokens=128):
        n = min(len(self.pieces), max_new_tokens)
        for i in range(n):
            yield {"token_id": i, "decoded_text": self.pieces[i],
                   "is_eos": False, "token_index": i}
        if self.eos_at_end and n < max_new_tokens:
            yield {"token_id": 999, "decoded_text": "", "is_eos": True, "token_index": n}


class _FakeTTSServer:
    """本地 HTTP PCM 流服务。mode: normal / wav_magic / slow / empty。"""

    def __init__(self, mode="normal", chunk_delay_s=0.0, total_bytes=8000):
        self.mode = mode
        self.chunk_delay_s = chunk_delay_s
        self.total_bytes = total_bytes
        self.port = None
        self._thread = None

    def __enter__(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                self.rfile.read(length)
                if outer.mode == "empty":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.end_headers()
                if outer.mode == "wav_magic":
                    self.wfile.write(b"RIFF" + b"\x00" * 2000)
                    self.wfile.flush()
                    return
                sent = 0
                rng = np.random.default_rng(1)
                try:
                    while sent < outer.total_bytes:
                        n = min(512, outer.total_bytes - sent)
                        tone = (rng.normal(0, 500, n // 2)).astype(np.int16).tobytes()
                        self.wfile.write(tone)
                        self.wfile.flush()
                        sent += n
                        if outer.chunk_delay_s:
                            time.sleep(outer.chunk_delay_s)
                except (ConnectionResetError, BrokenPipeError, OSError):
                    pass  # 客户端拿到 playable 后主动断开，属预期

            def log_message(self, *a):
                pass

        srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = srv.server_address[1]
        self._thread = threading.Thread(target=srv.serve_forever, daemon=True)
        self._thread.start()
        self._srv = srv
        return self

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    def __exit__(self, *a):
        self._srv.shutdown()


def _fake_models(frag_texts, full_text, pieces, asr_fail=False, flush_none=False):
    return {"segmenter": _FakeSegmenter(flush_none=flush_none),
            "asr": _FakeASR(frag_texts, full_text, fail=asr_fail),
            "llm": _FakeLLM(pieces),
            "new_asr_cache": _FakeCache,
            "convert_audio_segment": lambda seg, sid, is_start, is_final: (
                _FakeSegFinal(seg, is_final)),
            "decode_fn": lambda ids: "".join(_FakeLLM_PIECES[i] for i in ids if i < 900)}


class _FakeSegFinal(_FakeSeg):
    def __init__(self, seg, is_final):
        super().__init__(seg.segment_id, seg.audio)
        self.is_final = is_final


_FakeLLM_PIECES: list[str] = []


def _make_models(frag_texts, full_text, pieces, **kw):
    global _FakeLLM_PIECES
    _FakeLLM_PIECES = pieces
    return _fake_models(frag_texts, full_text, pieces, **kw)


def _st_sample(sid="crosswoz_t1_turn1", lang="zh"):
    return {"sample_id": sid, "language": lang, "duration_group": "very_long",
            "audio_path": "", "repeat_idx": 0}


def _st_audio(sr=16000, speech_s=0.5, silence_s=0.5):
    rng = np.random.default_rng(0)
    speech = (rng.normal(0, 0.2, int(sr * speech_s))).astype(np.float32)
    silence = np.zeros(int(sr * silence_s), dtype=np.float32)
    return np.concatenate([speech, silence])


def _st_cfg(url, run_id="st"):
    return {"url": url, "spk_id": "x", "speed": 1.0, "max_tokens": 128,
            "config_hash": "c", "schedule_hash": "s", "run_id": run_id,
            "tts_total_timeout_s": 10.0, "connect_timeout_s": 2.0, "read_timeout_s": 5.0}


def _self_test() -> int:
    fails = []

    def check(name, cond, detail=""):
        if not cond:
            fails.append(f"{name}: {detail}")
        print(f"[{'PASS' if cond else 'FAIL'}] {name}{(' :' + str(detail)[:160]) if not cond else ''}")

    sr = 16000
    audio = _st_audio(sr)
    # PSE 直接用内存波形（self-test 不写 wav 文件）：energy + 假 silero
    e = energy_pse_sample(audio)
    check("PSE energy 定位", e is not None and abs(e - sr * 0.5) <= 400 + 160, f"e={e}")

    def fake_silero(val):
        return lambda wave, sampling_rate=16000, **kw: [{"end": val}] if val else []

    probe = {"content_type": "application/octet-stream"}

    # 1) zh B 成功路径（首句中段冻结）+ closure + playable
    with _FakeTTSServer("normal") as srv:
        models = _make_models(["你好，", "世界。"], "全文。",
                              ["这", "是", "首", "句", "。", "后", "续", "内", "容"])
        rec = run_streaming(_st_sample(), audio, sr, models,
                            {"wav_sha256": "w", "analysis_waveform_sha256": "a",
                             "physical_speech_end_sample": e, "pse_method": "energy",
                             "pse_diff_ms": 1.0},
                            _st_cfg(srv.url), probe, seed_for_pair("crosswoz_t1_turn1", 0),
                            threading.Event())
    v = validate_record(rec)
    check("B 成功+schema+闭合", rec["terminal_state"] == "success" and not v,
          f"{rec['terminal_state']} {v} {rec['error'][:200]}")
    check("B 首句冻结", rec["sentence_end_found"] and rec["tts_text_source"] == "first_sentence"
          and rec["tts_n_chars"] == len("这是首句。"), str(rec["tts_n_chars"]))
    check("B TTFA 非负", ttfa_ms(rec) > 0)

    # 2) A 成功路径（等 feed_end 后才 ASR；TTS 全文）
    with _FakeTTSServer("normal") as srv:
        models = _make_models([], "完整回复第一句。第二句。", ["完", "整", "回", "复", "。"])
        rec_a = run_non_streaming(_st_sample(), audio, sr, models,
                                  {"wav_sha256": "w", "analysis_waveform_sha256": "a",
                                   "physical_speech_end_sample": e, "pse_method": "energy",
                                   "pse_diff_ms": 1.0},
                                  _st_cfg(srv.url), probe, seed_for_pair("crosswoz_t1_turn1", 0),
                                  threading.Event())
    v = validate_record(rec_a)
    check("A 成功+schema+闭合", rec_a["terminal_state"] == "success" and not v,
          f"{rec_a['terminal_state']} {v} {rec_a['error'][:200]}")
    check("A 未提前启动 ASR",
          rec_a["events"]["asr_start_ns"] >= rec_a["events"]["feed_end_ns"])
    check("A TTS 全文", rec_a["tts_text_source"] == "capped_full_response")
    check("配对同 seed", rec_a["generation_seed"] == rec["generation_seed"])

    # 3) flush=None → 经 INPUT_CLOSED 正常完成（不挂起）
    with _FakeTTSServer("normal") as srv:
        models = _make_models(["前半。"], "x", ["句", "子", "。"], flush_none=True)
        rec3 = run_streaming(_st_sample("crosswoz_t3_turn1"), audio, sr, models,
                             {"wav_sha256": "w", "analysis_waveform_sha256": "a",
                              "physical_speech_end_sample": e, "pse_method": "energy",
                              "pse_diff_ms": 1.0},
                             _st_cfg(srv.url), probe, seed_for_pair("crosswoz_t3_turn1", 0),
                             threading.Event())
    check("flush=None 不死锁", rec3["terminal_state"] == "success",
          f"{rec3['terminal_state']} {rec3['error'][:200]}")

    # 4) 小数跨 token：3 . 5 不判句末，后续 。判句末
    with _FakeTTSServer("normal") as srv:
        models = _make_models(["x。"], "x", ["答", "3", ".", "5", "。"])
        rec4 = run_streaming(_st_sample("crosswoz_t4_turn1"), audio, sr, models,
                             {"wav_sha256": "w", "analysis_waveform_sha256": "a",
                              "physical_speech_end_sample": e, "pse_method": "energy",
                              "pse_diff_ms": 1.0},
                             _st_cfg(srv.url), probe, seed_for_pair("crosswoz_t4_turn1", 0),
                             threading.Event())
    check("小数点不判句末", rec4["terminal_state"] == "success"
          and rec4["tts_n_chars"] == len("答3.5。"), str(rec4["tts_n_chars"]))

    # 5) 缩写限制：Mr . Smith → 在 '.' 判句末（已声明限制）
    det = StreamingSentenceDetector()
    idx = det.update("Mr", final=False)
    idx = det.update("Mr.", final=False)
    check("缩写 lookahead 前不判", idx is None, str(idx))
    idx = det.update("Mr. Smith.", final=False)
    check("缩写判句末（限制声明）", idx == 2, str(idx))  # "Mr." 的 '.' 在下标 2

    # 6) EOS-only → 零内容 error 行
    with _FakeTTSServer("normal") as srv:
        models = _make_models(["x。"], "", [])
        rec6 = run_streaming(_st_sample("crosswoz_t6_turn1"), audio, sr, models,
                             {"wav_sha256": "w", "analysis_waveform_sha256": "a",
                              "physical_speech_end_sample": e, "pse_method": "energy",
                              "pse_diff_ms": 1.0},
                             _st_cfg(srv.url), probe, seed_for_pair("crosswoz_t6_turn1", 0),
                             threading.Event())
    check("EOS-only 零内容 error", rec6["terminal_state"] == "error"
          and rec6["error"] == "zero_content_response", rec6["error"][:120])

    # 7) 末尾 pending 句点：前一字符为数字 → EOS 时不判 → fallback
    det2 = StreamingSentenceDetector()
    det2.update("数值为3", final=False)
    det2.update("数值为3.", final=False)
    idx2 = det2.update("数值为3.", final=True)
    check("数字末尾 pending 不判", idx2 is None, str(idx2))
    det3 = StreamingSentenceDetector()
    det3.update("结束了", final=False)
    det3.update("结束了.", final=False)
    idx3 = det3.update("结束了.", final=True)
    check("非数字末尾 pending 判句末", idx3 == 3, str(idx3))

    # 8) TTS WAV magic → 格式错误整行 error
    with _FakeTTSServer("wav_magic") as srv:
        models = _make_models(["x。"], "x", ["句", "。"])
        rec8 = run_streaming(_st_sample("crosswoz_t8_turn1"), audio, sr, models,
                             {"wav_sha256": "w", "analysis_waveform_sha256": "a",
                              "physical_speech_end_sample": e, "pse_method": "energy",
                              "pse_diff_ms": 1.0},
                             _st_cfg(srv.url), probe, seed_for_pair("crosswoz_t8_turn1", 0),
                             threading.Event())
    check("WAV magic 格式 error", rec8["terminal_state"] == "error"
          and "tts_format_not_pcm" in rec8["error"], rec8["error"][:120])

    # 9) TTS 慢流 → 超时 error（短 read timeout）
    with _FakeTTSServer("normal", chunk_delay_s=2.0) as srv:
        cfg = _st_cfg(srv.url)
        cfg["read_timeout_s"] = 0.5
        cfg["tts_total_timeout_s"] = 3.0
        models = _make_models(["x。"], "x", ["句", "。"])
        rec9 = run_streaming(_st_sample("crosswoz_t9_turn1"), audio, sr, models,
                             {"wav_sha256": "w", "analysis_waveform_sha256": "a",
                              "physical_speech_end_sample": e, "pse_method": "energy",
                              "pse_diff_ms": 1.0},
                             cfg, probe, seed_for_pair("crosswoz_t9_turn1", 0),
                             threading.Event())
    check("TTS 慢流超时 error", rec9["terminal_state"] == "error"
          and ("tts" in rec9["error"]), rec9["error"][:120])

    # 10) ASR 异常 → error 终态 + cancel
    with _FakeTTSServer("normal") as srv:
        models = _make_models(["x。"], "x", ["句", "。"], asr_fail=True)
        rec10 = run_streaming(_st_sample("crosswoz_t10_turn1"), audio, sr, models,
                              {"wav_sha256": "w", "analysis_waveform_sha256": "a",
                               "physical_speech_end_sample": e, "pse_method": "energy",
                               "pse_diff_ms": 1.0},
                              _st_cfg(srv.url), probe, seed_for_pair("crosswoz_t10_turn1", 0),
                              threading.Event())
    check("ASR 异常 fail-closed", rec10["terminal_state"] == "error", rec10["error"][:120])

    # 11) checkpoint：损坏 / hash 不匹配 → SystemExit
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "ck.jsonl"
        cp = Checkpoint(p, "run1", "cfg", "sched")
        rec11 = dict(rec)
        cp.append(rec11)
        try:
            Checkpoint(p, "run1", "DIFFERENT", "sched")
            check("checkpoint hash 不匹配退出", False)
        except SystemExit:
            check("checkpoint hash 不匹配退出", True)
        p.write_text('{"type":"header"}\n{bad json\n', encoding="utf-8")
        try:
            Checkpoint(p, "run1", "cfg", "sched")
            check("checkpoint 损坏退出", False)
        except SystemExit:
            check("checkpoint 损坏退出", True)

    # 12) 调度：25/25 平衡、子集交替、hash 稳定
    samples = [{"sample_id": f"crosswoz_{i:04d}", "language": "zh",
                "duration_group": ("long" if i % 3 == 0 else "very_long")}
               for i in range(25)] + \
              [{"sample_id": f"multiwoz_{i:04d}", "language": "en",
                "duration_group": ("extra_long" if i % 2 == 0 else "very_long")}
               for i in range(25)]
    subset_ids = [samples[i]["sample_id"] for i in (0, 5, 10, 15, 20, 25, 30, 35, 40, 45)]
    tasks = build_schedule(samples, subset_ids)
    t0 = [t for t in tasks if t["repeat_idx"] == 0]
    n_ab = len({t["sample_id"] for t in t0 if t["order"] == "AB"})
    check("AB/BA 25/25", n_ab == 25, str(n_ab))
    keys = [f"{t['sample_id']}|{t['mode']}|{t['repeat_idx']}" for t in tasks]
    check("任务键唯一", len(keys) == len(set(keys)))
    check("子集三轮", sum(1 for t in tasks if t["sample_id"] == subset_ids[0]) == 6)
    check("schedule hash 稳定", schedule_hash(tasks) == schedule_hash(build_schedule(samples, subset_ids)))

    # 13) seed 派生：同配对键同 seed、不同键不同 seed、确定性
    check("seed 确定性", seed_for_pair("a", 0) == seed_for_pair("a", 0)
          and seed_for_pair("a", 0) != seed_for_pair("a", 1))

    # 14) PSE 裁决与 fail-closed
    out = {"energy": e}
    diff_ok = abs(e - (e - 100)) / sr * 1000 <= 200
    check("PSE ≤200ms 取 energy", diff_ok)
    # 单算法失败 → fail-closed
    import tempfile as _tf
    import soundfile as sf
    with _tf.TemporaryDirectory() as td:
        wp = Path(td) / "t.wav"
        sf.write(str(wp), audio, sr)
        r1 = analyze_pse(str(wp), get_speech_timestamps=None)
        check("PSE 单算法失败 fail-closed", r1.get("error") == "pse_single_algorithm_failure",
              str(r1.get("error")))
        r2 = analyze_pse(str(wp), get_speech_timestamps=fake_silero(e - 100))
        check("PSE 双法一致取 energy", r2.get("pse_method") == "energy"
              and r2.get("physical_speech_end_sample") == e, str(r2.get("pse_method")))
        r3 = analyze_pse(str(wp), get_speech_timestamps=fake_silero(max(0, e - 8000)))
        check("PSE 冲突取 silero_fallback", r3.get("pse_method") == "silero_fallback",
              str(r3.get("pse_method")))

    # 15) chunk 回放因果：planned/actual 记录且 actual>=planned
    q: queue.Queue = queue.Queue()
    tim: dict = {}
    exc: queue.Queue = queue.Queue()
    cancel = threading.Event()
    playout_worker(audio, sr, int(sr * 0.5), q, tim, exc, cancel, e)
    logs = tim.get("chunk_log", [])
    check("chunk 因果释放", len(logs) == 2 and all(c["actual_ns"] >= c["planned_ns"] for c in logs),
          str(logs)[:160])
    check("feed_end 记录", tim.get("feed_end_ns") is not None)
    check("PSE 时钟映射", tim.get("physical_speech_end_ns") is not None
          and tim["physical_speech_end_ns"] == tim["playout_start_ns"] + round(e * 1e9 / sr))
    # 排空验证：2 chunk + 1 sentinel
    items = []
    while not q.empty():
        items.append(q.get_nowait())
    check("sentinel 在末尾", len(items) == 3 and isinstance(items[-1], InputClosed), str(len(items)))

    print(f"\nself-test {'ALL PASS' if not fails else f'{len(fails)} FAIL: ' + '; '.join(fails)}")
    return 1 if fails else 0


# ============================================================ main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample-list", help="50 样本清单 JSON（数组或 {'sample_ids': [...]}）")
    ap.add_argument("--subset-list", help="10 条重复子集 ID JSON（缺省取清单分层前 10 条）")
    ap.add_argument("--json-dir", default="experiments/datasets/processed/json")
    ap.add_argument("--audio-dir", default="experiments/datasets/processed/audio")
    ap.add_argument("--datasets", nargs="+", default=["crosswoz", "multiwoz"])
    ap.add_argument("--tts-url", default="http://127.0.0.1:20401")
    ap.add_argument("--tts-spk", default="晓伊")
    ap.add_argument("--tts-speed", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--silero-ref", default=None, help="固定 Silero commit（torch.hub ':ref'）")
    ap.add_argument("--silero-dir", default=None, help="或本地 silero 仓库目录（source='local'）")
    ap.add_argument("--output-dir", default="experiments/results/revision/r7_ttfa_unified")
    ap.add_argument("--run-id", required=False)
    ap.add_argument("--smoke", type=int, default=0, help="冒烟：仅前 N 个样本（repeat 0）")
    ap.add_argument("--tts-probe", action="store_true", help="仅执行 TTS 探活")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return _self_test()
    if args.tts_probe:
        out = tts_probe(args.tts_url, args.tts_spk, args.tts_speed)
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0 if out.get("ok") else 1
    if not args.sample_list or not args.run_id:
        ap.error("正式模式需要 --sample-list 与 --run-id")

    import torch
    from src.asr.streamaudio_segmenter import StreamAudioSegmenter
    from src.asr.faster_whisper_streamer import StreamingASRProcessor, ASRCache
    from src.llm.stream_llm_inference import StreamLLMInference
    from src.asr.run_stream_asr_test import convert_audio_segment
    from experiments.scripts.run_exp_latency import load_samples

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 样本清单
    with open(args.sample_list, encoding="utf-8") as f:
        sl = json.load(f)
    sample_ids = sl if isinstance(sl, list) else sl["sample_ids"]
    samples = []
    for ds in args.datasets:
        for s in load_samples(Path(args.json_dir), Path(args.audio_dir), dataset_filter=ds):
            if s.sample_id in set(sample_ids):
                samples.append({"sample_id": s.sample_id, "language": s.language,
                                "duration_group": s.duration_group,
                                "audio_path": str(s.audio_path)})
    if len(samples) != len(sample_ids) and not args.smoke:
        raise SystemExit(f"样本清单命中 {len(samples)}/{len(sample_ids)}（停止）")
    if args.subset_list:
        with open(args.subset_list, encoding="utf-8") as f:
            subset_ids = json.load(f)
        subset_ids = subset_ids if isinstance(subset_ids, list) else subset_ids["sample_ids"]
    else:
        zh = [s["sample_id"] for s in samples if s["language"] == "zh"][:5]
        en = [s["sample_id"] for s in samples if s["language"] == "en"][:5]
        subset_ids = zh + en
    tasks = build_schedule(samples, subset_ids)
    if args.smoke:
        keep = {s["sample_id"] for s in samples[:args.smoke]}
        tasks = [t for t in tasks if t["sample_id"] in keep and t["repeat_idx"] == 0]

    cfg = {"asr_model": "turbo", "asr_device": "cuda:0", "llm_model": "Qwen/Qwen2-7B-Instruct",
           "llm_device": "cuda:1", "chunk_ms": CHUNK_MS, "prefix_segments": 1,
           "suffix_segments": 0, "recognition_threshold": 2.0, "max_tokens": args.max_tokens,
           "tts_url": args.tts_url, "tts_spk": args.tts_spk, "tts_speed": args.tts_speed,
           "silero_ref": args.silero_ref, "silero_dir": args.silero_dir,
           "sample_list_sha256": sha256_file(args.sample_list),
           "requested_repetition_penalty": 1.1, "effective_repetition_penalty": "not_applied",
           "temperature": 0.1, "top_p": 0.9}
    cfg_hash = config_hash(cfg)
    sched_hash = schedule_hash(tasks)

    # TTS 探活（正式前必过）
    probe = tts_probe(args.tts_url, args.tts_spk, args.tts_speed)
    (out_dir / "tts_probe.json").write_text(json.dumps(probe, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    if not probe.get("ok"):
        raise SystemExit(f"TTS 探活失败: {probe}（停止）")

    # Silero（固定 revision）
    if args.silero_dir:
        model, utils = torch.hub.load(repo_or_dir=args.silero_dir, model="silero_vad",
                                      source="local", onnx=False, verbose=False)
    else:
        ref = args.silero_ref or "master"
        model, utils = torch.hub.load(repo_or_dir=f"snakers4/silero-vad:{ref}",
                                      model="silero_vad", onnx=False, verbose=False)
    get_speech_timestamps = utils[0]
    cfg["silero_torch_hub_ref"] = args.silero_ref or args.silero_dir or "master"
    cfg_hash = config_hash(cfg)

    # 模型
    asr_processor = StreamingASRProcessor(model_size="turbo", device="cuda:0",
                                          compute_type="auto", recognition_threshold=2.0,
                                          prefix_segments=1, suffix_segments_atleast=0)
    llm = StreamLLMInference(model_name="Qwen/Qwen2-7B-Instruct", device="cuda:1",
                             eval_mode=False)
    segmenter = StreamAudioSegmenter()
    models = {"segmenter": segmenter, "asr": asr_processor, "llm": llm,
              "new_asr_cache": ASRCache, "convert_audio_segment": convert_audio_segment,
              "decode_fn": lambda ids: llm.tokenizer.decode(ids, skip_special_tokens=True)}
    tts_cfg = {"url": args.tts_url, "spk_id": args.tts_spk, "speed": args.tts_speed,
               "max_tokens": args.max_tokens, "config_hash": cfg_hash,
               "schedule_hash": sched_hash, "run_id": args.run_id}

    ck_path = out_dir / f"checkpoint_{args.run_id}.jsonl"
    if args.no_resume and ck_path.exists():
        ck_path.unlink()
    ck = Checkpoint(ck_path, args.run_id, cfg_hash, sched_hash)

    records = []
    for task in tasks:
        key = f"{task['sample_id']}|{task['mode']}|{task['repeat_idx']}"
        if key in ck.done:
            logger.info(f"跳过已完成 {key}（{ck.done[key]}）")
            continue
        sample = next(s for s in samples if s["sample_id"] == task["sample_id"])
        sample = dict(sample, repeat_idx=task["repeat_idx"])
        cancel = threading.Event()
        try:
            import soundfile as sf
            audio, sr_f = sf.read(sample["audio_path"], dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr_f != ANALYSIS_SR:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr_f, target_sr=ANALYSIS_SR)
            audio = np.ascontiguousarray(audio, dtype=np.float32)
            pse = analyze_pse(sample["audio_path"], get_speech_timestamps)
            if pse.get("error"):
                rec = {"schema_version": SCHEMA_VERSION, "run_id": args.run_id,
                       "sample_id": sample["sample_id"], "language": sample["language"],
                       "duration_group": sample["duration_group"], "mode": task["mode"],
                       "repeat_idx": task["repeat_idx"], "terminal_state": "error",
                       "config_hash": cfg_hash, "schedule_hash": sched_hash,
                       "wav_sha256": pse.get("wav_sha256", ""),
                       "analysis_waveform_sha256": pse.get("analysis_waveform_sha256", ""),
                       "physical_speech_end_sample": None, "pse_method": "",
                       "pse_diff_ms": None, "events": {}, "chunk_log": [], "tts": {},
                       "response_token_count": 0, "generation_stop_reason": None,
                       "sentence_end_found": False, "sentence_fallback": False,
                       "tts_text_source": None, "tts_n_chars": 0, "tts_text_sha256": None,
                       "generation_seed": None, "error": pse["error"]}
            else:
                seed = seed_for_pair(sample["sample_id"], task["repeat_idx"])
                if task["mode"] == "streaming":
                    rec = run_streaming(sample, audio, ANALYSIS_SR, models, pse, tts_cfg,
                                        probe, seed, cancel)
                else:
                    rec = run_non_streaming(sample, audio, ANALYSIS_SR, models, pse, tts_cfg,
                                            probe, seed, cancel)
        except Exception:
            rec = {"schema_version": SCHEMA_VERSION, "run_id": args.run_id,
                   "sample_id": task["sample_id"], "language": sample.get("language", ""),
                   "duration_group": sample.get("duration_group", ""), "mode": task["mode"],
                   "repeat_idx": task["repeat_idx"], "terminal_state": "error",
                   "config_hash": cfg_hash, "schedule_hash": sched_hash, "wav_sha256": "",
                   "analysis_waveform_sha256": "", "physical_speech_end_sample": None,
                   "pse_method": "", "pse_diff_ms": None, "events": {}, "chunk_log": [],
                   "tts": {}, "response_token_count": 0, "generation_stop_reason": None,
                   "sentence_end_found": False, "sentence_fallback": False,
                   "tts_text_source": None, "tts_n_chars": 0, "tts_text_sha256": None,
                   "generation_seed": None, "error": traceback.format_exc()[-500:]}
        problems = validate_record(rec)
        if problems:
            rec["terminal_state"] = "error"
            rec["error"] = (rec.get("error", "") + "|validate:" + ";".join(problems))[:500]
        ck.append(rec)
        records.append(rec)
        logger.info(f"{key}: {rec['terminal_state']}"
                    + (f" TTFA_playable={ttfa_ms(rec):.0f}ms"
                       if rec["terminal_state"] == "success" else f" {rec['error'][:100]}"))

    # 汇总 + QA
    import csv as _csv
    all_records = []
    for ln in ck_path.read_text(encoding="utf-8").splitlines()[1:]:
        all_records.append(json.loads(ln))
    summary = summarize(all_records)
    with open(out_dir / f"ttfa_summary_{args.run_id}.csv", "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    cv_rows = subset_cv(all_records, subset_ids)
    with open(out_dir / f"ttfa_subset_cv_{args.run_id}.csv", "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=list(cv_rows[0].keys()))
        w.writeheader()
        w.writerows(cv_rows)
    problems = qa_records(all_records, tasks)
    qa_md = ["# TTFA 统一实测 QA", "",
             f"- run_id: {args.run_id}", f"- config_hash: {cfg_hash}",
             f"- schedule_hash: {sched_hash}", f"- 任务数: {len(tasks)}",
             f"- 记录数: {len(all_records)}", f"- QA 问题数: {len(problems)}", ""]
    qa_md += [f"- {p}" for p in problems] or ["- 无"]
    (out_dir / f"QA_{args.run_id}.md").write_text("\n".join(qa_md) + "\n", encoding="utf-8")
    runinfo = ["# TTFA 统一实测 RUNINFO", "", f"- 命令参数: {sys.argv}",
               f"- schema_version: {SCHEMA_VERSION}", f"- run_id: {args.run_id}",
               f"- config: {json.dumps(cfg, ensure_ascii=False)}",
               f"- config_hash: {cfg_hash}", f"- schedule_hash: {sched_hash}",
               f"- playable 阈值: {PLAYABLE_BYTES} bytes（22050Hz×16bit×30ms）",
               f"- torch: {torch.__version__}",
               f"- 采样实际生效参数: temperature=0.1, top_p=0.9, repetition_penalty=not_applied",
               ""]
    (out_dir / f"RUNINFO_{args.run_id}.md").write_text("\n".join(runinfo), encoding="utf-8")
    n_err = sum(1 for r in all_records if r["terminal_state"] != "success")
    print(f"完成: {len(all_records)} 记录，{n_err} 非成功；QA 问题 {len(problems)}")
    return 1 if problems or n_err else 0


if __name__ == "__main__":
    sys.exit(main())
