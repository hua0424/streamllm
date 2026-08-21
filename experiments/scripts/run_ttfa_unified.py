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


class PairTimeout(Exception):
    """pair 总 deadline 超时（覆盖 playout/ASR/LLM/TTS 全阶段）。"""


def remaining_s(deadline_ns: int) -> float:
    return (deadline_ns - now_ns()) / 1e9


def classify_payload(prefix: bytes) -> str:
    """TTS 响应格式判定（探活与正式请求共用同一校验器）。

    调用方须先累积足够前缀（≥16 字节或流结束）再判定，防跨 read 分片绕过；
    前导空白剥离后识别 JSON/HTML/XML 错误响应与 WAV 头。
    """
    p = prefix.lstrip()
    if not p:
        return "empty"
    if p.startswith(b"RIFF"):
        return "wav"
    if p[:1] in (b"{", b"["):
        return "json"
    if p.startswith(b"<"):
        return "html"
    return "pcm"

EVENT_FIELDS = [
    "playout_start_ns", "physical_speech_end_ns", "last_input_sample_ns", "feed_end_ns",
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
    """探活：确认服务返回预期裸 PCM；固定允许策略（Content-Type/Encoding 取值），
    正式请求据此逐项校验，不得临时放宽。"""
    import requests
    out = {"url": url, "spk_id": spk_id, "speed": speed}
    try:
        resp = requests.post(url, json={"tts_text": "探活", "spk_id": spk_id,
                                        "stream": True, "speed": speed},
                             stream=True, timeout=(TTS_CONNECT_TIMEOUT_S, TTS_READ_TIMEOUT_S))
        out["status"] = resp.status_code
        out["content_type"] = resp.headers.get("Content-Type")
        out["content_encoding"] = resp.headers.get("Content-Encoding")
        # 固定允许策略：探活时的取值即正式允许值（None 也作为固定策略记录下来）
        out["allow_content_type"] = out["content_type"]
        out["allow_content_encoding"] = out["content_encoding"]
        buf = bytearray()
        for chunk in resp.iter_content(64):
            buf += chunk
            if len(buf) >= 16:
                break
        kind = classify_payload(bytes(buf))
        out["payload_class"] = kind
        out["magic_hex"] = bytes(buf[:8]).hex()
        resp.close()
        out["ok"] = resp.status_code == 200 and kind == "pcm"
    except Exception as exc:
        out["ok"] = False
        out["error"] = str(exc)
    return out


def tts_measure(url: str, text: str, spk_id: str, speed: float, probe: dict,
                cancel_event: threading.Event, total_deadline_ns: int,
                requests_session=None, connect_timeout: float = TTS_CONNECT_TIMEOUT_S,
                read_timeout: float = TTS_READ_TIMEOUT_S,
                resp_holder: dict | None = None) -> dict:
    """流式 TTS 首包测量。返回 tts_request_start/headers/first_byte/playable/done 等。

    - 应用层把任意 read 重切为 ≤512B granule（iter_content 不保证块大小）；
    - 字节连续累积，不丢奇数字节；只在完整 sample 边界推进 playable 计数；
    - 先累积 ≥16 字节前缀再判定格式（与探活同一 classify_payload）；
    - resp_holder（r2 P1-2）：response 句柄写入 holder["resp"]，外层超时/取消可主动
      close() 打断阻塞中的 HTTP read；
    - 读到自然结束（不提前断连），结束时总字节非 sample-width 倍数 → 整行 error；
    - HTTP/格式/对齐/零内容错误 → error 字段（调用方整行 error，不降级）。
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
        if resp_holder is not None:
            resp_holder["resp"] = resp  # 外层可主动 close() 打断阻塞 read
        rec["tts_response_headers_ns"] = now_ns()
        if resp.status_code != 200:
            rec["error"] = f"tts_http_{resp.status_code}"
            return rec
        # 探活固定的允许策略逐项比对（含 None 取值）
        if resp.headers.get("Content-Type") != probe.get("allow_content_type"):
            rec["error"] = f"tts_content_type_mismatch:{resp.headers.get('Content-Type')}"
            return rec
        if resp.headers.get("Content-Encoding") != probe.get("allow_content_encoding"):
            rec["error"] = f"tts_content_encoding_mismatch:{resp.headers.get('Content-Encoding')}"
            return rec
        buf = bytearray()
        format_checked = False
        for raw in resp.iter_content(TTS_READ_GRANULE):
            if cancel_event.is_set():
                rec["error"] = "tts_cancelled"
                return rec
            if now_ns() > total_deadline_ns:
                rec["error"] = "tts_total_timeout"
                return rec
            if not raw:
                continue
            now = now_ns()
            if rec["first_pcm_byte_ns"] is None:
                rec["first_pcm_byte_ns"] = now
                rec["tts_first_chunk_bytes"] = min(len(raw), TTS_READ_GRANULE)
            # 应用层重切 ≤512B granule（防御 iter_content 返回更大块）
            for off in range(0, len(raw), TTS_READ_GRANULE):
                buf += raw[off:off + TTS_READ_GRANULE]
            if not format_checked and len(buf) >= 16:
                format_checked = True
                kind = classify_payload(bytes(buf[:64]))
                if kind != "pcm":
                    rec["error"] = f"tts_format_not_pcm:{kind}"
                    return rec
            complete = len(buf) - (len(buf) % PCM_BYTES_PER_SAMPLE)
            if rec["first_playable_pcm_ns"] is None and complete >= PLAYABLE_BYTES:
                rec["first_playable_pcm_ns"] = now
                arr = np.frombuffer(bytes(buf[:PLAYABLE_BYTES]), dtype=np.int16).astype(np.float64)
                rec["tts_playable_rms"] = float(np.sqrt(np.mean(arr ** 2))) if len(arr) else 0.0
                rec["tts_playable_peak"] = float(np.abs(arr).max()) if len(arr) else 0.0
        # 自然结束后的收尾校验（顺序：空 body → 格式 → 对齐 → 阈值）
        rec["tts_total_bytes"] = len(buf)
        if rec["first_pcm_byte_ns"] is None:
            rec["error"] = "tts_empty_body"
        elif not format_checked and classify_payload(bytes(buf[:64])) != "pcm":
            rec["error"] = f"tts_format_not_pcm:{classify_payload(bytes(buf[:64]))}"
        elif len(buf) % PCM_BYTES_PER_SAMPLE != 0:
            rec["error"] = "tts_misaligned_bytes"
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
        # 最后一个输入样本的计划到达时刻（单调时钟映射）
        timings["last_input_sample_ns"] = playout_start + round(len(audio) * 1e9 / sr)
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

def validate_record(rec: dict, expected_config_hash: str | None = None,
                    expected_schedule_hash: str | None = None) -> list[str]:
    """schema + 因果偏序 + 闭合恒等式校验；返回违规列表（空=通过）。

    偏序按真实因果边（不是字段全序）：B 的 TTS 可在 generation_end 前启动
    （首句冻结即启动），故 generation_end 与 TTS 链无序约束；A 的 text_ready=generation_end。
    error 行必须有非空 error 诊断字段。
    """
    errs = []
    if rec.get("schema_version") != SCHEMA_VERSION:
        errs.append("schema_version")
    if rec.get("clock_type") != "perf_counter_ns":
        errs.append("clock_type")
    if rec.get("endpoint_mode") not in ("explicit_flush", "full_input"):
        errs.append("endpoint_mode")
    if rec.get("terminal_state") not in TERMINAL_STATES:
        errs.append("terminal_state")
    if expected_config_hash and rec.get("config_hash") != expected_config_hash:
        errs.append("config_hash_mismatch")
    if expected_schedule_hash and rec.get("schedule_hash") != expected_schedule_hash:
        errs.append("schedule_hash_mismatch")
    if rec.get("terminal_state") is None:
        errs.append("terminal_state")
        return errs
    if rec["terminal_state"] != "success":
        if not rec.get("error"):
            errs.append("error_row_missing_diagnostic")
        return errs  # 非成功行：终态合法 + 诊断字段即可
    ev = rec.get("events", {})
    for f in ("playout_start_ns", "physical_speech_end_ns", "last_input_sample_ns",
              "feed_end_ns", "pipeline_input_close_ns", "first_model_token_ns",
              "generation_end_ns", "tts_request_start_ns", "first_pcm_byte_ns",
              "first_playable_pcm_ns"):
        if ev.get(f) is None:
            errs.append(f"missing:{f}")
    if errs:
        return errs

    def le(x, y):
        if ev[y] < ev[x]:
            errs.append(f"order:{y}<{x}")

    le("playout_start_ns", "physical_speech_end_ns")
    le("physical_speech_end_ns", "last_input_sample_ns")
    le("last_input_sample_ns", "feed_end_ns")
    le("feed_end_ns", "pipeline_input_close_ns")
    le("pipeline_input_close_ns", "first_model_token_ns")
    le("first_model_token_ns", "generation_end_ns")
    if ev.get("first_content_token_ns") is not None:
        le("first_model_token_ns", "first_content_token_ns")
        le("first_content_token_ns", "generation_end_ns")
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
    # chunk 调度误差非负
    for c in rec.get("chunk_log", []):
        if c.get("sched_err_ns", -1) < 0:
            errs.append("chunk_sched_err_negative")
            break
    # TTS 字段完整性
    tts = rec.get("tts", {})
    if tts.get("tts_total_bytes", 0) < PLAYABLE_BYTES:
        errs.append("tts_total_bytes<playable")
    for f in ("tts_playable_rms", "tts_playable_peak"):
        v = tts.get(f)
        if v is None or not np.isfinite(v):
            errs.append(f"tts_{f}_invalid")
    if not rec.get("tts_text_sha256") or rec.get("tts_n_chars", 0) <= 0:
        errs.append("tts_text_missing")
    if rec.get("tts_n_bytes_utf8", 0) <= 0:
        errs.append("tts_utf8_bytes_missing")
    # r2 P1-1：TTS 文本派生字段一致性（长度/UTF-8 字节/哈希/来源与 mode·fallback 一致）
    tts_text = rec.get("tts_text")
    if tts_text is not None:
        if rec.get("tts_n_chars") != len(tts_text):
            errs.append("tts_n_chars_mismatch")
        if rec.get("tts_n_bytes_utf8") != len(tts_text.encode("utf-8")):
            errs.append("tts_utf8_mismatch")
        if rec.get("tts_text_sha256") != sha256_text(tts_text):
            errs.append("tts_sha_mismatch")
    if mode == "non-streaming":
        expect_src = "capped_full_response"
    else:
        expect_src = ("first_sentence"
                      if rec.get("sentence_end_found") and not rec.get("sentence_fallback")
                      else "capped_full_response")
    if rec.get("tts_text_source") != expect_src:
        errs.append(f"tts_text_source_inconsistent({rec.get('tts_text_source')}!={expect_src})")
    if rec.get("tts_seeded") is not False:
        errs.append("tts_seeded_flag")  # 本 TTS 服务不可控随机性，必须显式标 False
    if rec.get("generation_seed") is None:
        errs.append("generation_seed_missing")
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
            "clock_type": "perf_counter_ns",
            "endpoint_mode": "explicit_flush" if mode == "streaming" else "full_input",
            "sample_id": sample["sample_id"], "language": sample["language"],
            "duration_group": sample["duration_group"], "mode": mode,
            "repeat_idx": repeat_idx, "terminal_state": None, "fatal": False,
            "config_hash": cfg_hash, "schedule_hash": sched_hash,
            "wav_sha256": pse["wav_sha256"],
            "analysis_waveform_sha256": pse["analysis_waveform_sha256"],
            "physical_speech_end_sample": pse["physical_speech_end_sample"],
            "pse_method": pse["pse_method"], "pse_diff_ms": pse["pse_diff_ms"],
            "events": {f: None for f in EVENT_FIELDS}, "chunk_log": [],
            "tts": {}, "response_token_count": 0, "generation_stop_reason": None,
            "sentence_end_found": False, "sentence_fallback": False,
            "final_drain_triggered": False, "final_drain_empty": False,
            "tts_text": None, "tts_text_source": None, "tts_n_chars": 0,
            "tts_n_bytes_utf8": 0, "tts_seeded": False,
            "tts_text_sha256": None, "generation_seed": None, "error": ""}


def _join_all(rec, threads, deadline_ns, exc_q):
    """join 全部线程并确认退出；遗留线程 → timeout + fatal（fail-stop run）。"""
    for t in threads:
        t.join(timeout=max(remaining_s(deadline_ns), 0.05))
    alive = [t.name for t in threads if t.is_alive()]
    if alive:
        rec["terminal_state"] = "timeout"
        rec["fatal"] = True
        rec["error"] = (rec.get("error", "") + f"|thread_leak:{alive}")[:500]


def run_streaming(sample, audio, sr, models, pse, tts_cfg, probe, seed, cancel_event):
    """System B：分段→增量预填→is_end 后生成→首句冻结即 TTS（LLM 不中断）。

    P0-1：InputClosed 到达且无 is_final 段时，显式把尾部段标记 is_final 触发 final
    drain；drain 后仍无 final 输出 → error（不静默截断）。
    P0-2：pair 绝对 deadline 覆盖全阶段；worker 异常/线程遗留 → fatal fail-stop。
    """
    rec = _base_record(sample, "streaming", sample["repeat_idx"], pse,
                       tts_cfg["config_hash"], tts_cfg["schedule_hash"], tts_cfg["run_id"])
    rec["generation_seed"] = seed
    ev = rec["events"]
    pair_deadline = now_ns() + int(tts_cfg.get("pair_deadline_s", PAIR_DEADLINE_S) * 1e9)
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
    drain = {"pending_empty_wq": False}
    state_lock = threading.Lock()  # close 发布/队列 final 化/drain 标志的原子保护（r2 P0-1）

    def segmentation_worker():
        try:
            state = segmenter.create_state()
            while True:
                if cancel_event.is_set():
                    return
                try:
                    item = chunk_q.get(timeout=0.1)
                except queue.Empty:
                    continue
                if isinstance(item, InputClosed):
                    break
                _, chunk = item
                stream_segment, state = segmenter.process_audio(chunk, state)
                if stream_segment:
                    seg_id = f"seg_{stream_segment.segment_id:03d}"
                    asr_q.put(convert(stream_segment, seg_id,
                                      stream_segment.segment_id == 1, False))
            ev["explicit_flush_start_ns"] = now_ns()
            remaining_seg, _ = segmenter.flush(state)
            if remaining_seg is not None and len(remaining_seg.audio) > 0:
                asr_q.put(convert(remaining_seg, f"seg_{remaining_seg.segment_id:03d}",
                                  False, True))
            asr_q.put(InputClosed())  # 无条件 sentinel（flush=None 不死锁）
            ev["explicit_flush_done_ns"] = now_ns()
        except Exception:
            exc_q.put(("segmentation", traceback.format_exc()))
            cancel_event.set()

    def collector():
        try:
            saw_final = False
            while True:
                if cancel_event.is_set():
                    return
                if remaining_s(pair_deadline) <= 0:
                    raise PairTimeout("collector")
                try:
                    item = asr_q.get(timeout=min(0.1, max(remaining_s(pair_deadline), 0.01)))
                except queue.Empty:
                    continue
                if isinstance(item, InputClosed):
                    # P0-1（r2）：final 化与 drain 标志在锁内**先于** close 发布完成，
                    # transcriber 一旦看到 closed 即可安全依据 must_call 判定退出
                    with state_lock:
                        if not saw_final:
                            cache = cache_box[0]
                            if cache.waiting_segment_queue:
                                cache.waiting_segment_queue[-1].is_final = True
                                rec["final_drain_triggered"] = True
                            elif cache.segment_queue:
                                cache.segment_queue[-1].is_final = True
                                rec["final_drain_triggered"] = True
                                drain["pending_empty_wq"] = True
                        ev["pipeline_input_close_ns"] = now_ns()
                    return
                saw_final = saw_final or bool(getattr(item, "is_final", False))
                cache_box[0].add_segment(item)
        except Exception:
            exc_q.put(("collector", traceback.format_exc()))
            cancel_event.set()

    def transcriber():
        try:
            got_final = False
            while True:
                if cancel_event.is_set():
                    return
                if remaining_s(pair_deadline) <= 0:
                    raise PairTimeout("transcriber")
                with state_lock:
                    closed = ev["pipeline_input_close_ns"] is not None
                    cache = cache_box[0]
                    must_call = bool(cache.waiting_segment_queue) or drain["pending_empty_wq"]
                    exit_now = closed and not must_call
                if exit_now:
                    break
                if not must_call:
                    time.sleep(0.005)
                    continue
                if cache.is_processing():
                    time.sleep(0.005)
                    continue
                drain["pending_empty_wq"] = False
                cache, text, is_final = asr_processor.transcribe_audio_segment(cache)
                cache_box[0] = cache
                if text:
                    ev["last_asr_commit_ns"] = now_ns()
                    text_q.put((text, False))
                if is_final:
                    got_final = True
                    break
            if rec["final_drain_triggered"] and not got_final:
                # drain 已触发但无 final 输出（如空识别早退）→ 显式 error，不静默截断
                rec["final_drain_empty"] = True
                rec["error"] = "asr_final_drain_no_output"
                rec["terminal_state"] = "error"
                cancel_event.set()
                return
            ev["asr_processing_done_ns"] = now_ns()
            text_q.put(("", True))
        except Exception:
            exc_q.put(("transcriber", traceback.format_exc()))
            cancel_event.set()

    threads = [threading.Thread(target=playout_worker, args=(
                   audio, sr, chunk_samples, chunk_q, ev, exc_q, cancel_event,
                   pse["physical_speech_end_sample"]), name="playout", daemon=True),
               threading.Thread(target=segmentation_worker, name="segmentation", daemon=True),
               threading.Thread(target=collector, name="collector", daemon=True),
               threading.Thread(target=transcriber, name="transcriber", daemon=True)]
    for t in threads:
        t.start()

    tts_holder: dict = {}
    tts_thread: list[threading.Thread] = []
    try:
        kv = None
        while True:  # 增量预填（主线程消费 text_q）
            if cancel_event.is_set():
                if rec.get("error"):
                    return rec  # transcriber 已写诊断（如 drain 失败）
                raise RuntimeError("cancelled: " + "; ".join(w for w, _ in list(exc_q.queue)))
            if remaining_s(pair_deadline) <= 0:
                raise PairTimeout("prefill")
            try:
                text, is_end = text_q.get(timeout=min(0.1, max(remaining_s(pair_deadline), 0.01)))
            except queue.Empty:
                continue
            if text or is_end:
                kv = llm.cache_prompt(text, pre_cache=kv, is_end=is_end)
            if is_end:
                break
        rec["chunk_log"] = ev.pop("chunk_log", [])

        import torch  # 延迟导入（self-test 免 torch 也可注入 fake）
        gen = torch.Generator(device=getattr(llm, "device", "cpu"))
        gen.manual_seed(seed)  # 请求级独立随机流（不消费全局 RNG）
        det = StreamingSentenceDetector()
        token_ids: list[int] = []
        decode_fn = models["decode_fn"]
        stop = "max_tokens"
        for meta in llm.generate_with_meta(pre_cache=kv, max_new_tokens=tts_cfg["max_tokens"],
                                           generator=gen):
            now = now_ns()
            if remaining_s(pair_deadline) <= 0:
                raise PairTimeout("generate")
            if meta["is_eos"]:
                stop = "eos"
                break  # EOS 不计入 first_model_token / token 数
            if ev["first_model_token_ns"] is None:
                ev["first_model_token_ns"] = now  # 首个非 EOS 模型 token
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
                rec["tts_text"] = tts_text
                rec["tts_text_source"] = "first_sentence"
                rec["tts_n_chars"] = len(tts_text)
                rec["tts_n_bytes_utf8"] = len(tts_text.encode("utf-8"))
                rec["tts_text_sha256"] = sha256_text(tts_text)
                th = threading.Thread(target=_tts_into, args=(
                    tts_holder, tts_text, tts_cfg, probe, cancel_event, pair_deadline),
                    name="tts", daemon=True)
                th.start()
                tts_thread.append(th)
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
            rec["tts_text"] = tts_text
            rec["tts_text_source"] = "capped_full_response" if idx is None else "first_sentence"
            rec["tts_n_chars"] = len(tts_text)
            rec["tts_n_bytes_utf8"] = len(tts_text.encode("utf-8"))
            rec["tts_text_sha256"] = sha256_text(tts_text)
            th = threading.Thread(target=_tts_into, args=(
                tts_holder, tts_text, tts_cfg, probe, cancel_event, pair_deadline),
                name="tts", daemon=True)
            th.start()
            tts_thread.append(th)
        while "done" not in tts_holder:
            if remaining_s(pair_deadline) <= 0:
                cancel_event.set()
                _close_resp(tts_holder)  # r2 P1-2：打断阻塞中的 HTTP read
                raise PairTimeout("tts_join")
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
    except PairTimeout as e:
        rec["error"] = f"pair_timeout:{e}"
        rec["terminal_state"] = "timeout"
        rec["fatal"] = True
    except Exception:
        # r2 P0-2：ASR/LLM/模型状态异常（主线程或 worker）一律 fail-stop；
        # 可恢复错误（TTS/输入类）在到达此处前已提前 return，不会落入本分支
        if not rec.get("error"):
            rec["error"] = traceback.format_exc()[-500:]
        rec["terminal_state"] = "error"
        rec["fatal"] = True
    finally:
        cancel_event.set()
        if rec["terminal_state"] is None:
            rec["terminal_state"] = "error"
            rec["error"] = rec.get("error") or "unknown"
        _join_all(rec, threads + tts_thread, pair_deadline, exc_q)
    return rec


def _close_resp(holder: dict) -> None:
    """主动关闭 TTS response（外层超时/取消时打断阻塞 read，r2 P1-2）。"""
    resp = (holder.get("resp_holder") or {}).get("resp")
    if resp is not None:
        try:
            resp.close()
        except Exception:
            pass


def _tts_into(holder, text, tts_cfg, probe, cancel_event, pair_deadline_ns=None):
    deadline = now_ns() + int(tts_cfg.get("tts_total_timeout_s", 120.0) * 1e9)
    holder["resp_holder"] = {}
    read_timeout = tts_cfg.get("read_timeout_s", TTS_READ_TIMEOUT_S)
    if pair_deadline_ns is not None:
        # r2 P1-2：read timeout 动态收紧为配置值与 pair 剩余时间的较小者
        read_timeout = max(min(read_timeout, remaining_s(pair_deadline_ns)), 0.1)
    holder["rec"] = tts_measure(tts_cfg["url"], text, tts_cfg["spk_id"], tts_cfg["speed"],
                                probe, cancel_event, deadline,
                                connect_timeout=tts_cfg.get("connect_timeout_s", TTS_CONNECT_TIMEOUT_S),
                                read_timeout=read_timeout,
                                resp_holder=holder["resp_holder"])
    holder["done"] = True


def run_non_streaming(sample, audio, sr, models, pse, tts_cfg, probe, seed, cancel_event):
    """System A：等待 feed_end → full ASR → 完整 capped 回复后 TTS。"""
    rec = _base_record(sample, "non-streaming", sample["repeat_idx"], pse,
                       tts_cfg["config_hash"], tts_cfg["schedule_hash"], tts_cfg["run_id"])
    rec["generation_seed"] = seed
    ev = rec["events"]
    pair_deadline = now_ns() + int(tts_cfg.get("pair_deadline_s", PAIR_DEADLINE_S) * 1e9)
    exc_q: queue.Queue = queue.Queue()
    chunk_samples = int(sr * CHUNK_MS / 1000)
    sink_q: queue.Queue = queue.Queue()
    playout_t = threading.Thread(target=playout_worker, args=(
        audio, sr, chunk_samples, sink_q, ev, exc_q, cancel_event,
        pse["physical_speech_end_sample"]), name="playout", daemon=True)
    playout_t.start()
    tts_thread: list[threading.Thread] = []
    try:
        while True:  # 排空 sink 队列（A 不用流式中间结果，但时间轴一致）
            if cancel_event.is_set():
                raise RuntimeError("cancelled: " + "; ".join(w for w, _ in list(exc_q.queue)))
            if remaining_s(pair_deadline) <= 0:
                raise PairTimeout("playout_drain")
            try:
                item = sink_q.get(timeout=min(0.1, max(remaining_s(pair_deadline), 0.01)))
            except queue.Empty:
                continue
            if isinstance(item, InputClosed):
                break
        playout_t.join(timeout=max(remaining_s(pair_deadline), 0.05))
        if playout_t.is_alive():
            raise PairTimeout("playout_join")
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
        if remaining_s(pair_deadline) <= 0:
            raise PairTimeout("after_asr")
        import torch
        gen = torch.Generator(device=getattr(llm, "device", "cpu"))
        gen.manual_seed(seed)
        kv = llm.cache_prompt(text, pre_cache=None, is_end=True)
        token_ids: list[int] = []
        decode_fn = models["decode_fn"]
        stop = "max_tokens"
        det = StreamingSentenceDetector()
        for meta in llm.generate_with_meta(pre_cache=kv, max_new_tokens=tts_cfg["max_tokens"],
                                           generator=gen):
            now = now_ns()
            if remaining_s(pair_deadline) <= 0:
                raise PairTimeout("generate")
            if meta["is_eos"]:
                stop = "eos"
                break  # EOS 不计入 first_model_token / token 数
            if ev["first_model_token_ns"] is None:
                ev["first_model_token_ns"] = now
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
        rec["tts_text"] = full_text
        rec["tts_text_source"] = "capped_full_response"
        rec["tts_n_chars"] = len(full_text)
        rec["tts_n_bytes_utf8"] = len(full_text.encode("utf-8"))
        rec["tts_text_sha256"] = sha256_text(full_text)
        tts_holder: dict = {}
        th = threading.Thread(target=_tts_into, args=(
            tts_holder, full_text, tts_cfg, probe, cancel_event, pair_deadline),
            name="tts", daemon=True)
        th.start()
        tts_thread.append(th)
        while "done" not in tts_holder:
            if remaining_s(pair_deadline) <= 0:
                cancel_event.set()
                _close_resp(tts_holder)  # r2 P1-2：打断阻塞中的 HTTP read
                raise PairTimeout("tts_join")
            time.sleep(0.005)
        rec["tts"] = tts_holder.get("rec", {"error": "tts_join_timeout"})
        for k in ("tts_request_start_ns", "tts_response_headers_ns", "first_pcm_byte_ns",
                  "first_playable_pcm_ns", "tts_done_ns"):
            ev[k] = rec["tts"].get(k)
        if rec["tts"].get("error"):
            rec["error"] = rec["tts"]["error"]
            rec["terminal_state"] = "error"
            return rec
        rec["terminal_state"] = "success"
    except PairTimeout as e:
        rec["error"] = f"pair_timeout:{e}"
        rec["terminal_state"] = "timeout"
        rec["fatal"] = True
    except Exception:
        # r2 P0-2：System A 主线程 full ASR / cache_prompt / generate 异常一律 fatal
        if not rec.get("error"):
            rec["error"] = traceback.format_exc()[-500:]
        rec["terminal_state"] = "error"
        rec["fatal"] = True
    finally:
        cancel_event.set()
        if rec["terminal_state"] is None:
            rec["terminal_state"] = "error"
            rec["error"] = rec.get("error") or "unknown"
        _join_all(rec, [playout_t] + tts_thread, pair_deadline, exc_q)
    return rec


# ============================================================ 调度

def build_schedule(samples: list[dict], subset_ids: list[str]) -> list[dict]:
    """生成确定性 AB/BA 平衡任务表（语言×时长分层，P1-3）。

    - 每个 (language, duration_group) stratum 内 |AB−BA|≤1；奇数 stratum 的多数方向
      在相邻 stratum 间交替，使全局恰 25/25；
    - 子集 10 条：三轮序列 (AB,BA,AB)/(BA,AB,BA) 按语言分层交替分配，repeat 0 计入三轮；
    - 非子集样本仅 repeat 0；任务按 pass（repeat）分批、批内按分层顺序。
    """
    from collections import defaultdict
    strata: dict[tuple, list[str]] = defaultdict(list)
    info: dict[str, dict] = {}
    for s in samples:
        strata[(s["language"], s["duration_group"])].append(s["sample_id"])
        info[s["sample_id"]] = s
    for k in strata:
        strata[k].sort()

    subset_set = set(subset_ids)
    missing = [sid for sid in subset_ids if sid not in info]
    if missing:
        raise SystemExit(f"子集 ID 未命中样本清单: {missing}（停止）")
    subset_sorted = sorted(subset_set, key=lambda sid: (info[sid]["language"],
                                                      info[sid]["duration_group"], sid))
    # 子集三轮序列：按语言分层交替 P1/P2，整体 5/5
    patterns: dict[str, list[str]] = {}
    lang_seen: dict[str, int] = defaultdict(int)
    for sid in subset_sorted:
        lang = info[sid]["language"]
        idx = lang_seen[lang]
        lang_seen[lang] += 1
        start = idx % 2 if lang == "zh" else (idx + 1) % 2  # 两语种起始方向相反
        patterns[sid] = ["AB", "BA", "AB"] if start == 0 else ["BA", "AB", "BA"]

    # repeat 0 方向：stratum 内交替，stratum 多数方向与全局累计失衡相反
    orders: dict[str, str] = {sid: p[0] for sid, p in patterns.items()}
    global_diff = sum(1 for o in orders.values() if o == "AB") - \
        sum(1 for o in orders.values() if o == "BA")
    for key in sorted(strata):
        ids = strata[key]
        pre = [sid for sid in ids if sid in orders]
        rest = [sid for sid in ids if sid not in orders]
        stratum_diff = sum(1 for sid in pre if orders[sid] == "AB") - \
            sum(1 for sid in pre if orders[sid] == "BA")
        # 起始方向：让 stratum 的多数方向抵消全局失衡
        start = "BA" if global_diff + stratum_diff > 0 else "AB"
        for i, sid in enumerate(rest):
            orders[sid] = start if i % 2 == 0 else ("BA" if start == "AB" else "AB")
        d = sum(1 for sid in ids if orders[sid] == "AB") - \
            sum(1 for sid in ids if orders[sid] == "BA")
        if abs(d) > 1:
            raise SystemExit(f"stratum {key} 失衡: {d}（停止）")
        global_diff += d
    n_ab = sum(1 for o in orders.values() if o == "AB")
    if len(samples) % 2 == 0 and n_ab != len(samples) - n_ab:
        raise SystemExit(f"AB/BA 全局不平衡: {n_ab}/{len(samples) - n_ab}（停止）")

    ordered_ids = [sid for key in sorted(strata) for sid in strata[key]]
    tasks = []
    seq = 0
    for pass_idx in (0, 1, 2):
        for sid in ordered_ids:
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
    """原子快照 checkpoint（P0-3）：header（run/binding）+ 每行一条终态记录。

    - 每次 append 后整文件 tmp+fsync+replace 重写（记录级原子，崩溃不会产生截断行）；
    - 加载时：header 损坏/记录截断/重复主键/binding 任一项不匹配/同目录混入其他 run
      → SystemExit（fail-closed，不得复用目录）；
    - binding 含 config/schedule hash、git commit/dirty、环境版本、模型与 Silero
      revision/hash、TTS 配置、样本清单/子集/音频映射 hash、downmix/resampler 版本；
    - error/cancelled/timeout 均为终态，已终态 key 直接跳过（不静默重跑）；
    - 配置变化或重跑必须用新 run_id（binding 不匹配即拒）。
    """

    BINDING_KEYS = ("schema_version", "run_id", "config_hash", "schedule_hash",
                    "git_commit", "git_dirty", "env_versions", "asr_model", "llm_model",
                    "silero_meta", "tts_config", "sample_list_sha256", "subset_sha256",
                    "audio_map_sha256")

    def __init__(self, path: Path, run_id: str, binding: dict):
        self.path = path
        self.records: list[dict] = []
        self.done: dict[str, str] = {}
        # 同目录其他 run 的 checkpoint 混入检查
        for other in sorted(path.parent.glob("checkpoint_*.jsonl")):
            if other != path:
                raise SystemExit(f"目录中存在其他 run 的 checkpoint: {other}（不得复用目录，停止）")
        if path.exists():
            lines = path.read_text(encoding="utf-8").splitlines()
            if not lines:
                raise SystemExit(f"checkpoint 为空但存在: {path}（停止）")
            try:
                header = json.loads(lines[0])
            except json.JSONDecodeError:
                raise SystemExit(f"checkpoint header 损坏: {path}（停止）")
            for k in self.BINDING_KEYS:
                want = binding.get(k)
                if header.get(k) != want:
                    raise SystemExit(f"checkpoint binding 不匹配: {k}（配置变化须新建 run，停止）")
            for ln in lines[1:]:
                try:
                    rec = json.loads(ln)
                except json.JSONDecodeError:
                    raise SystemExit(f"checkpoint 记录截断/损坏: {path}（停止）")
                key = self.key_of(rec)
                if key in self.done:
                    raise SystemExit(f"checkpoint 重复主键: {key}（停止）")
                if rec.get("terminal_state") not in TERMINAL_STATES:
                    raise SystemExit(f"checkpoint 记录终态非法: {key}（停止）")
                self.done[key] = rec["terminal_state"]
                self.records.append(rec)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            header = {"type": "header", **{k: binding.get(k) for k in self.BINDING_KEYS}}
            self._write_all(header)
        # r2 P0-3：恢复历史 fatal 记录 → run 级 fail-stop（main 据此只为剩余任务补
        # cancelled 终态，不再执行任何 GPU 任务）
        self.fatal_seen = any(bool(r.get("fatal")) for r in self.records)

    @staticmethod
    def key_of(rec: dict) -> str:
        return f"{rec['sample_id']}|{rec['mode']}|{rec['repeat_idx']}"

    def _write_all(self, header: dict | None = None) -> None:
        if header is None:
            header = json.loads(self.path.read_text(encoding="utf-8").splitlines()[0])
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(json.dumps(header, ensure_ascii=False) + "\n")
            for rec in self.records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)

    def append(self, rec: dict) -> None:
        key = self.key_of(rec)
        if key in self.done:
            raise SystemExit(f"重复追加主键: {key}（停止）")
        self.records.append(rec)
        self.done[key] = rec["terminal_state"]
        self._write_all()


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


def qa_records(records: list[dict], tasks: list[dict],
               expected_config_hash: str | None = None,
               expected_schedule_hash: str | None = None) -> list[str]:
    """结果级 QA（Gate 1 §6.2 / 再审 §6.4 对应项）；返回问题列表。"""
    problems = []
    expected: dict[str, int] = {}
    for t in tasks:
        key = f"{t['sample_id']}|{t['mode']}|{t['repeat_idx']}"
        expected[key] = expected.get(key, 0) + 1
    got: dict[str, int] = {}
    pair_hash: dict[str, str] = {}
    pair_seed: dict[str, int] = {}
    pair_analysis: dict[str, str] = {}
    for r in records:
        key = Checkpoint.key_of(r)
        got[key] = got.get(key, 0) + 1
        v = validate_record(r, expected_config_hash, expected_schedule_hash)
        if v:
            problems.append(f"{key}: {';'.join(v)}")
        pk = f"{r['sample_id']}|{r['repeat_idx']}"
        for store, field, label in ((pair_hash, "wav_sha256", "WAV hash"),
                                    (pair_analysis, "analysis_waveform_sha256", "分析波形 hash"),
                                    (pair_seed, "generation_seed", "generation seed")):
            val = r.get(field)
            if val is None:
                continue
            if pk in store and store[pk] != val:
                problems.append(f"{pk}: A/B {label} 不一致")
            store.setdefault(pk, val)
    for key, cnt in expected.items():
        if got.get(key, 0) != cnt:
            problems.append(f"{key}: 终态记录数 {got.get(key, 0)} != 预期 {cnt}")
    for key in got:
        if key not in expected:
            problems.append(f"{key}: 计划外记录")
    return problems


# ============================================================ self-test

class _FakeSeg:
    def __init__(self, segment_id, audio, payload=""):
        self.segment_id = segment_id
        self.audio = audio
        self.duration = len(audio) / 16000
        self.is_final = False
        self.payload = payload  # 协议桩用：该段应产出的文本


class _FakeSegmenter:
    """每 2 个 chunk 出一个段；flush 行为可配（None / 有音频段）。block=True 时永久阻塞。"""

    def __init__(self, flush_none=False, block=False):
        self.flush_none = flush_none
        self.block = block
        self.n = 0

    def create_state(self):
        return {"buf": []}

    def process_audio(self, chunk, state):
        if self.block:
            while True:  # 永久阻塞（P0-2 线程遗留测试；daemon 线程不影响进程退出）
                time.sleep(0.05)
        self.n += 1
        if self.n % 2 == 0:
            return _FakeSeg(self.n // 2, chunk, payload=f"段{self.n // 2}"), state
        return None, state

    def flush(self, state):
        if self.flush_none:
            return None, state
        return _FakeSeg(99, np.zeros(1600, dtype=np.float32), payload="尾段"), state


class _FakeCache:
    def __init__(self):
        self.waiting_segment_queue = []
        self.segment_queue = []

    def add_segment(self, seg):
        self.waiting_segment_queue.append(seg)

    def is_processing(self):
        return False


class _FakeASR:
    """按段依次吐出脚本化文本；is_final 由段属性判定（支持 drain 翻转）。

    fail=True 时抛异常。only_final=True 模拟 should_process 语义：非 final 段不出文本。
    """

    def __init__(self, frag_texts, full_text, fail=False, only_final=False):
        self.frag_texts = list(frag_texts)
        self.full_text = full_text
        self.fail = fail
        self.only_final = only_final

    def transcribe_audio_segment(self, cache):
        if self.fail:
            raise RuntimeError("fake asr boom")
        seg = cache.waiting_segment_queue.pop(0)
        is_final = bool(getattr(seg, "is_final", False)) or seg.segment_id == 99
        if self.only_final and not is_final:
            cache.segment_queue.append(seg)  # 未达阈值：段留在 segment_queue 未提交
            return cache, None, False
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
        self.prompts = []

    def cache_prompt(self, prompt, pre_cache=None, is_end=False):
        self.prompts.append(prompt)
        return {"kv": True}

    def generate_with_meta(self, pre_cache, max_new_tokens=128, generator=None):
        n = min(len(self.pieces), max_new_tokens)
        for i in range(n):
            yield {"token_id": i, "decoded_text": self.pieces[i],
                   "is_eos": False, "token_index": i}
        if self.eos_at_end and n < max_new_tokens:
            yield {"token_id": 999, "decoded_text": "", "is_eos": True, "token_index": n}


class _FakeResp:
    """脚本化 HTTP 响应（跨 read 分片/大 read/奇数字节测试用）。"""

    def __init__(self, chunks, status=200, headers=None):
        self._chunks = chunks
        self.status_code = status
        self.headers = headers or {"Content-Type": "application/octet-stream"}
        self.closed = False

    def iter_content(self, n):
        yield from self._chunks

    def close(self):
        self.closed = True


class _FakeSession:
    def __init__(self, resp):
        self._resp = resp

    def post(self, *a, **kw):
        return self._resp


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
                if outer.mode == "headers_only":
                    time.sleep(10.0)  # 发 headers 后停发 body（r2 P1-2 慢流测试）
                    return
                if outer.mode == "wav_magic":
                    try:
                        self.wfile.write(b"RIFF" + b"\x00" * 2000)
                        self.wfile.flush()
                    except (ConnectionResetError, BrokenPipeError, OSError):
                        pass
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
                    pass  # 客户端提前断开，属预期

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


def _fake_models(frag_texts, full_text, pieces, asr_fail=False, flush_none=False,
                 only_final=False, block=False):
    asr = _FakeASR(frag_texts, full_text, fail=asr_fail, only_final=only_final)
    llm = _FakeLLM(pieces)
    return {"segmenter": _FakeSegmenter(flush_none=flush_none, block=block),
            "asr": asr, "llm": llm,
            "new_asr_cache": _FakeCache,
            "convert_audio_segment": lambda seg, sid, is_start, is_final: (
                _FakeSegFinal(seg, is_final)),
            "decode_fn": lambda ids: "".join(_FakeLLM_PIECES[i] for i in ids if i < 900)}


class _FakeSegFinal(_FakeSeg):
    def __init__(self, seg, is_final):
        super().__init__(seg.segment_id, seg.audio, getattr(seg, "payload", ""))
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
            "pair_deadline_s": PAIR_DEADLINE_S,
            "tts_total_timeout_s": 10.0, "connect_timeout_s": 2.0, "read_timeout_s": 5.0}


_ST_PROBE = {"allow_content_type": "application/octet-stream", "allow_content_encoding": None}


def _st_pse(e):
    return {"wav_sha256": "w", "analysis_waveform_sha256": "a",
            "physical_speech_end_sample": e, "pse_method": "energy", "pse_diff_ms": 1.0}


def _self_test() -> int:
    fails = []

    n_checks = [0]

    def check(name, cond, detail=""):
        n_checks[0] += 1
        if not cond:
            fails.append(f"{name}: {detail}")
        print(f"[{'PASS' if cond else 'FAIL'}] {name}{(' :' + str(detail)[:160]) if not cond else ''}")

    sr = 16000
    audio = _st_audio(sr)
    e = energy_pse_sample(audio)
    check("PSE energy 定位", e is not None and abs(e - sr * 0.5) <= 400 + 160, f"e={e}")

    def fake_silero(val):
        return lambda wave, sampling_rate=16000, **kw: [{"end": val}] if val else []

    probe = dict(_ST_PROBE)

    # 1) zh B 成功路径（首句中段冻结）+ closure + playable
    with _FakeTTSServer("normal") as srv:
        models = _make_models(["你好，", "世界。"], "全文。",
                              ["这", "是", "首", "句", "。", "后", "续", "内", "容"])
        rec = run_streaming(_st_sample(), audio, sr, models, _st_pse(e),
                            _st_cfg(srv.url), probe, seed_for_pair("crosswoz_t1_turn1", 0),
                            threading.Event())
    v = validate_record(rec, "c", "s")
    check("B 成功+schema+闭合", rec["terminal_state"] == "success" and not v,
          f"{rec['terminal_state']} {v} {rec['error'][:200]}")
    check("B 首句冻结", rec["sentence_end_found"] and rec["tts_text_source"] == "first_sentence"
          and rec["tts_n_chars"] == len("这是首句。"), str(rec["tts_n_chars"]))
    check("B TTFA 非负", ttfa_ms(rec) > 0)
    check("B TTS 文本落盘", rec["tts_text"] == "这是首句。"
          and rec["tts_n_bytes_utf8"] == len("这是首句。".encode("utf-8")))

    # 2) A 成功路径（等 feed_end 后才 ASR；TTS 全文）
    with _FakeTTSServer("normal") as srv:
        models = _make_models([], "完整回复第一句。第二句。", ["完", "整", "回", "复", "。"])
        rec_a = run_non_streaming(_st_sample(), audio, sr, models, _st_pse(e),
                                  _st_cfg(srv.url), probe, seed_for_pair("crosswoz_t1_turn1", 0),
                                  threading.Event())
    v = validate_record(rec_a, "c", "s")
    check("A 成功+schema+闭合", rec_a["terminal_state"] == "success" and not v,
          f"{rec_a['terminal_state']} {v} {rec_a['error'][:200]}")
    check("A 未提前启动 ASR",
          rec_a["events"]["asr_start_ns"] >= rec_a["events"]["feed_end_ns"])
    check("A TTS 全文", rec_a["tts_text_source"] == "capped_full_response")
    check("配对同 seed", rec_a["generation_seed"] == rec["generation_seed"])

    # 3) flush=None → drain 触发且尾文本入 LLM（P0-1）；2s 音频产 2 段，第 2 段为尾部
    audio3 = _st_audio(sr, 1.5, 0.5)
    e3 = energy_pse_sample(audio3)
    with _FakeTTSServer("normal") as srv:
        models = _make_models(["前半。", "尾段文本"], "x", ["句", "子", "。"], flush_none=True)
        rec3 = run_streaming(_st_sample("crosswoz_t3_turn1"), audio3, sr, models, _st_pse(e3),
                             _st_cfg(srv.url), probe, seed_for_pair("crosswoz_t3_turn1", 0),
                             threading.Event())
    check("flush=None 不死锁", rec3["terminal_state"] == "success",
          f"{rec3['terminal_state']} {rec3['error'][:200]}")
    check("flush=None drain 触发", rec3["final_drain_triggered"] is True)
    check("flush=None 尾文本不丢",
          any("尾段文本" in p for p in models["llm"].prompts), str(models["llm"].prompts))

    # 3b) drain 触发但 ASR 无 final 输出 → 显式 error（不静默截断）
    with _FakeTTSServer("normal") as srv:
        models = _make_models([], "x", ["句", "。"], flush_none=True)
        # only_final + 段全部非 final：drain 翻转后 fake 仍不产出（模拟空识别早退）
        models["asr"] = _FakeASR([], "x", only_final=False)
        orig = models["asr"].transcribe_audio_segment

        def empty_recognition(cache):  # 模拟真实空识别早退：返回 is_final=False
            cache.waiting_segment_queue.pop(0)
            return cache, None, False
        models["asr"].transcribe_audio_segment = empty_recognition
        rec3b = run_streaming(_st_sample("crosswoz_t3b_turn1"), audio, sr, models, _st_pse(e),
                              _st_cfg(srv.url), probe, seed_for_pair("crosswoz_t3b_turn1", 0),
                              threading.Event())
    check("drain 无输出记 error", rec3b["terminal_state"] == "error"
          and "asr_final_drain_no_output" in rec3b["error"],
          f"{rec3b['terminal_state']} {rec3b['error'][:120]}")

    # 3c) 真实 ASRCache 协议集成（P0-1 要求：真实 cache 状态机 + stub processor）
    from src.asr.faster_whisper_streamer import ASRCache as _RealCache

    class _ProtocolASR:
        """实现真实 StreamingASRProcessor 调用协议的桩（不含模型权重）。"""

        recognition_threshold = 2.0
        prefix_segments = 1
        suffix_segments_atleast = 0

        def transcribe_audio_segment(self, cache):
            if not cache.set_processing():
                return cache, None, False
            try:
                cache.add_to_asr_segments()
                seg = cache.segment_queue[-1]
                if not cache.should_process(self.recognition_threshold,
                                            self.prefix_segments,
                                            self.suffix_segments_atleast, seg.is_final):
                    return cache, None, False  # 未达阈值：段留存未提交
                text = "".join(getattr(s, "payload", "") for s in cache.segment_queue)
                cache.segment_queue.clear()
                return cache, text, seg.is_final
            finally:
                cache.set_processed()

        def transcribe_complete_audio(self, **kw):
            return {"text": "完整。"}

    with _FakeTTSServer("normal") as srv:
        models = _make_models([], "x", ["句", "。"], flush_none=True)
        models["asr"] = _ProtocolASR()
        models["new_asr_cache"] = _RealCache
        rec3c = run_streaming(_st_sample("crosswoz_t3c_turn1"), audio, sr, models, _st_pse(e),
                              _st_cfg(srv.url), probe, seed_for_pair("crosswoz_t3c_turn1", 0),
                              threading.Event())
    check("真实 ASRCache drain 不丢尾文本",
          rec3c["terminal_state"] == "success" and rec3c["final_drain_triggered"],
          f"{rec3c['terminal_state']} {rec3c['error'][:200]}")

    # 4) 小数跨 token：3 . 5 不判句末，后续 。判句末
    with _FakeTTSServer("normal") as srv:
        models = _make_models(["x。"], "x", ["答", "3", ".", "5", "。"])
        rec4 = run_streaming(_st_sample("crosswoz_t4_turn1"), audio, sr, models, _st_pse(e),
                             _st_cfg(srv.url), probe, seed_for_pair("crosswoz_t4_turn1", 0),
                             threading.Event())
    check("小数点不判句末", rec4["terminal_state"] == "success"
          and rec4["tts_n_chars"] == len("答3.5。"), str(rec4["tts_n_chars"]))

    # 5) 缩写限制：Mr . Smith → 在 '.' 判句末（已声明限制）
    det = StreamingSentenceDetector()
    det.update("Mr", final=False)
    idx = det.update("Mr.", final=False)
    check("缩写 lookahead 前不判", idx is None, str(idx))
    idx = det.update("Mr. Smith.", final=False)
    check("缩写判句末（限制声明）", idx == 2, str(idx))  # "Mr." 的 '.' 在下标 2

    # 6) EOS-only → 零内容 error 行；first_model_token 不记录（P1-1）
    with _FakeTTSServer("normal") as srv:
        models = _make_models(["x。"], "", [])
        rec6 = run_streaming(_st_sample("crosswoz_t6_turn1"), audio, sr, models, _st_pse(e),
                             _st_cfg(srv.url), probe, seed_for_pair("crosswoz_t6_turn1", 0),
                             threading.Event())
    check("EOS-only 零内容 error", rec6["terminal_state"] == "error"
          and rec6["error"] == "zero_content_response", rec6["error"][:120])
    check("EOS-only 无 first_model_token", rec6["events"]["first_model_token_ns"] is None)

    # 6b) 真实 StreamLLMInference.generate_with_meta 方法级测试（P1-1/P1-2）
    import torch
    from src.llm.stream_llm_inference import StreamLLMInference as _SLI

    class _StubOut:
        def __init__(self, logits):
            self.past_key_values = None
            self.logits = logits

    class _StubModel(torch.nn.Module):
        def __init__(self, script):
            super().__init__()
            self.script = list(script)

        def forward(self, **kw):
            logits = torch.zeros(1, 1, 100)
            logits[0, 0, self.script.pop(0)] = 10.0
            return _StubOut(logits)

    class _StubTok:
        eos_token_id = 99

        def decode(self, ids, skip_special_tokens=True):
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            return {5: "你", 7: "好", 99: ""}.get(ids[0] if isinstance(ids, list) else ids, "")

    def _make_sli(script, first_logits_id):
        inst = _SLI.__new__(_SLI)
        inst.tokenizer = _StubTok()
        inst.model = _StubModel(script)
        inst.device = "cpu"
        inst.eval_mode = False
        inst.timing_events = {}
        kv = _SLI.KVCache(None, None, None)
        kv.next_token_logits = torch.zeros(1, 100)
        kv.next_token_logits[0, first_logits_id] = 10.0
        kv.past_key_values = None
        kv.pre_attention_mask = torch.ones(1, 1, dtype=torch.long)
        return inst, kv

    # 正常：5→7→EOS
    inst, kv = _make_sli([7, 99], 5)
    metas = list(inst.generate_with_meta(pre_cache=kv, max_new_tokens=10,
                                         generator=torch.Generator().manual_seed(1)))
    check("真实接口 token 序列", [m["token_id"] for m in metas] == [5, 7, 99]
          and metas[-1]["is_eos"] and inst.last_stop_reason == "eos",
          str([m["token_id"] for m in metas]) + "/" + str(inst.last_stop_reason))
    # EOS-only
    inst, kv = _make_sli([], 99)
    metas = list(inst.generate_with_meta(pre_cache=kv, max_new_tokens=10))
    check("真实接口 EOS-only", len(metas) == 1 and metas[0]["is_eos"]
          and inst.last_stop_reason == "eos")
    # max_tokens
    inst, kv = _make_sli([7, 5], 5)
    metas = list(inst.generate_with_meta(pre_cache=kv, max_new_tokens=2))
    check("真实接口 max_tokens", len(metas) == 2 and inst.last_stop_reason == "max_tokens",
          inst.last_stop_reason)
    # RNG 请求级隔离：同 seed 同序列；且与全局 RNG 状态无关
    logits = torch.zeros(1, 100)
    for i in range(100):
        logits[0, i] = float((i * 37) % 10)
    torch.manual_seed(12345)
    g1 = torch.Generator().manual_seed(42)
    inst, kv = _make_sli([7, 99], 5)
    inst.model = _StubModel([99])
    seq1 = [m["token_id"] for m in inst.generate_with_meta(pre_cache=kv, max_new_tokens=3, generator=g1)]
    torch.manual_seed(99999)  # 改变全局 RNG
    g2 = torch.Generator().manual_seed(42)
    inst2, kv2 = _make_sli([99], 5)
    seq2 = [m["token_id"] for m in inst2.generate_with_meta(pre_cache=kv2, max_new_tokens=3, generator=g2)]
    check("请求级 RNG 同 seed 同序列", seq1 == seq2, f"{seq1} vs {seq2}")

    # 7) 末尾 pending 句点裁决
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

    # 8) TTS 格式：跨 read 分片 RIFF / 前导空白 JSON / HTML / 奇数字节 / 大 read（P1-4/P1-5）
    def run_tts(chunks):
        resp = _FakeResp(chunks)
        return tts_measure("http://x", "t", "x", 1.0, probe, threading.Event(),
                           now_ns() + int(30e9), requests_session=_FakeSession(resp))

    pcm_head = np.zeros(2048, dtype=np.int16).tobytes()
    r = run_tts([b"R", b"IFF" + b"\x00" * 3000])
    check("跨 read RIFF 判格式错误", r.get("error", "").startswith("tts_format_not_pcm"), r.get("error"))
    r = run_tts([b'  {"error": "x"}' + b" " * 2000])
    check("前导空白 JSON 判格式错误", r.get("error", "").startswith("tts_format_not_pcm"), r.get("error"))
    r = run_tts([b"<html><body>err</body></html>" + b" " * 2000])
    check("HTML 判格式错误", r.get("error", "").startswith("tts_format_not_pcm"), r.get("error"))
    r = run_tts([pcm_head[:1325]])  # 奇数总字节
    check("奇数字节对齐错误", r.get("error") == "tts_misaligned_bytes", r.get("error"))
    r = run_tts([pcm_head])  # 单大 read（4096B>512）重切后正常
    check("大 read 重切正常", not r.get("error") and r["first_playable_pcm_ns"] is not None
          and r["tts_total_bytes"] == 4096, r.get("error", ""))
    r = run_tts([pcm_head[:1000]])
    check("低于 playable 阈值", r.get("error") == "tts_below_playable_threshold", r.get("error"))
    r = run_tts([])
    check("空 body", r.get("error") == "tts_empty_body", r.get("error"))

    # 9) TTS 慢流 → 超时 error（短 read timeout）
    with _FakeTTSServer("normal", chunk_delay_s=2.0) as srv:
        cfg = _st_cfg(srv.url)
        cfg["read_timeout_s"] = 0.5
        cfg["tts_total_timeout_s"] = 3.0
        models = _make_models(["x。"], "x", ["句", "。"])
        rec9 = run_streaming(_st_sample("crosswoz_t9_turn1"), audio, sr, models, _st_pse(e),
                             cfg, probe, seed_for_pair("crosswoz_t9_turn1", 0),
                             threading.Event())
    check("TTS 慢流超时 error", rec9["terminal_state"] == "error"
          and ("tts" in rec9["error"]), rec9["error"][:120])

    # 10) ASR 异常 → error 终态 + fatal（fail-stop）
    with _FakeTTSServer("normal") as srv:
        models = _make_models(["x。"], "x", ["句", "。"], asr_fail=True)
        rec10 = run_streaming(_st_sample("crosswoz_t10_turn1"), audio, sr, models, _st_pse(e),
                              _st_cfg(srv.url), probe, seed_for_pair("crosswoz_t10_turn1", 0),
                              threading.Event())
    check("ASR 异常 fail-closed+fatal", rec10["terminal_state"] == "error"
          and rec10["fatal"] is True, f"{rec10['terminal_state']} {rec10['fatal']}")

    # 10b) worker 永久阻塞 → pair deadline 超时 + 线程遗留 → timeout+fatal（P0-2）
    with _FakeTTSServer("normal") as srv:
        cfg = _st_cfg(srv.url)
        cfg["pair_deadline_s"] = 2.5  # 音频 1s，阻塞在 segmenter
        models = _make_models(["x。"], "x", ["句", "。"], block=True)
        rec10b = run_streaming(_st_sample("crosswoz_t10b_turn1"), audio, sr, models, _st_pse(e),
                               cfg, probe, seed_for_pair("crosswoz_t10b_turn1", 0),
                               threading.Event())
    check("阻塞 worker → timeout+fatal", rec10b["terminal_state"] == "timeout"
          and rec10b["fatal"] is True and "thread_leak" in rec10b["error"],
          f"{rec10b['terminal_state']} {rec10b['error'][:160]}")

    # 11) checkpoint：截断 / 重复主键 / hash 不匹配 / 目录混入旧 run（P0-3）
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "checkpoint_run1.jsonl"
        binding = {k: "v" for k in Checkpoint.BINDING_KEYS}
        cp = Checkpoint(p, "run1", binding)
        cp.append(dict(rec))
        try:
            Checkpoint(p, "run1", {**binding, "config_hash": "DIFFERENT"})
            check("checkpoint binding 不匹配退出", False)
        except SystemExit:
            check("checkpoint binding 不匹配退出", True)
        # 截断记录
        lines = p.read_text(encoding="utf-8").splitlines()
        p.write_text("\n".join(lines) + "\n{\"sample_id\": \"x\", \n", encoding="utf-8")
        try:
            Checkpoint(p, "run1", binding)
            check("checkpoint 截断退出", False)
        except SystemExit:
            check("checkpoint 截断退出", True)
        # 重复主键
        p.write_text("\n".join(lines) + "\n" + lines[1] + "\n", encoding="utf-8")
        try:
            Checkpoint(p, "run1", binding)
            check("checkpoint 重复主键退出", False)
        except SystemExit:
            check("checkpoint 重复主键退出", True)
        # 目录混入旧 run
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        (Path(td) / "checkpoint_other.jsonl").write_text("{}\n", encoding="utf-8")
        try:
            Checkpoint(p, "run1", binding)
            check("目录混入旧 run 退出", False)
        except SystemExit:
            check("目录混入旧 run 退出", True)

    # 12) 调度：全局 25/25 + 逐 stratum ≤1 + 子集交替 + hash 稳定（P1-3）
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
    from collections import defaultdict
    strata_orders: dict[tuple, list[str]] = defaultdict(list)
    for t in t0:
        if t["mode"] != "non-streaming":
            continue
        s = next(x for x in samples if x["sample_id"] == t["sample_id"])
        strata_orders[(s["language"], s["duration_group"])].append(t["order"])
    strata_ok = all(abs(o.count("AB") - o.count("BA")) <= 1 for o in strata_orders.values())
    check("逐 stratum 差值≤1", strata_ok,
          str({k: (o.count("AB"), o.count("BA")) for k, o in strata_orders.items()}))
    keys = [f"{t['sample_id']}|{t['mode']}|{t['repeat_idx']}" for t in tasks]
    check("任务键唯一", len(keys) == len(set(keys)))
    check("子集三轮", sum(1 for t in tasks if t["sample_id"] == subset_ids[0]) == 6)
    sub_patterns = []
    for sid in subset_ids:
        orders = [t["order"] for t in tasks if t["sample_id"] == sid and t["mode"] == "non-streaming"]
        sub_patterns.append(tuple(orders))
    check("子集交替序列 5/5",
          sub_patterns.count(("AB", "BA", "AB")) == 5 and sub_patterns.count(("BA", "AB", "BA")) == 5,
          str(sub_patterns))
    check("schedule hash 稳定", schedule_hash(tasks) == schedule_hash(build_schedule(samples, subset_ids)))

    # 13) seed 派生：同配对键同 seed、不同键不同 seed、确定性
    check("seed 确定性", seed_for_pair("a", 0) == seed_for_pair("a", 0)
          and seed_for_pair("a", 0) != seed_for_pair("a", 1))

    # 14) PSE 裁决与 fail-closed
    diff_ok = abs(e - (e - 100)) / sr * 1000 <= 200
    check("PSE ≤200ms 取 energy", diff_ok)
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
    items = []
    while not q.empty():
        items.append(q.get_nowait())
    check("sentinel 在末尾", len(items) == 3 and isinstance(items[-1], InputClosed), str(len(items)))

    # 16) error 行缺诊断字段 → validate 拦截（P1-6）
    bad = _error_record({"sample_id": "x", "language": "zh", "duration_group": "long"},
                        {"sample_id": "x", "mode": "streaming", "repeat_idx": 0},
                        "c", "s", "r", "")
    check("error 行缺诊断被拦截", "error_row_missing_diagnostic" in validate_record(bad, "c", "s"))

    # 17) TTS 派生字段一致性负向（r2 P1-1）：篡改字节数/哈希/来源 → 拦截
    bad2 = json.loads(json.dumps(rec))
    bad2["tts_n_bytes_utf8"] += 1
    check("TTS 字节数篡改被拦截", "tts_utf8_mismatch" in validate_record(bad2, "c", "s"))
    bad3 = json.loads(json.dumps(rec))
    bad3["tts_text_sha256"] = "deadbeef"
    check("TTS 哈希篡改被拦截", "tts_sha_mismatch" in validate_record(bad3, "c", "s"))
    bad4 = json.loads(json.dumps(rec))
    bad4["tts_text_source"] = "capped_full_response"  # B 首句成功路径应为 first_sentence
    v4 = validate_record(bad4, "c", "s")
    check("TTS 来源不一致被拦截",
          any(e.startswith("tts_text_source_inconsistent") for e in v4), str(v4))

    # 18) final-drain 竞态回归（r2 P0-1）：真实 ASRCache + 首次转写被放慢，
    # close 发布与 final 化的交错窗口内 transcriber 不得提前退出（锁+先 final 化后发布）
    from src.asr.faster_whisper_streamer import ASRCache as _RealCacheRace

    class _SlowProtocolASR:
        recognition_threshold = 2.0
        prefix_segments = 1
        suffix_segments_atleast = 0

        def __init__(self):
            self.slowed = False

        def transcribe_audio_segment(self, cache):
            if not self.slowed:
                self.slowed = True
                time.sleep(0.4)  # 强制与 close 发布交错
            if not cache.set_processing():
                return cache, None, False
            try:
                cache.add_to_asr_segments()
                seg = cache.segment_queue[-1]
                if not cache.should_process(self.recognition_threshold,
                                            self.prefix_segments,
                                            self.suffix_segments_atleast, seg.is_final):
                    return cache, None, False
                text = "".join(getattr(s, "payload", "") for s in cache.segment_queue)
                cache.segment_queue.clear()
                return cache, text, seg.is_final
            finally:
                cache.set_processed()

        def transcribe_complete_audio(self, **kw):
            return {"text": "完整。"}

    for trial in range(3):
        with _FakeTTSServer("normal") as srv:
            models = _make_models([], "x", ["句", "。"], flush_none=True)
            models["asr"] = _SlowProtocolASR()
            models["new_asr_cache"] = _RealCacheRace
            recr = run_streaming(_st_sample(f"crosswoz_race{trial}_turn1"), audio3, sr,
                                 models, _st_pse(e3), _st_cfg(srv.url), probe,
                                 seed_for_pair(f"crosswoz_race{trial}_turn1", 0),
                                 threading.Event())
            # 尾段 = segmenter 产出的"段2"（drain 翻转 is_final 后经真实 cache 输出）
            ok = (recr["terminal_state"] == "success" and recr["final_drain_triggered"]
                  and any("段2" in p for p in models["llm"].prompts))
            if not ok:
                break
    check("final-drain 竞态回归×3（真实 ASRCache）", ok,
          f"trial {trial}: {recr['terminal_state']} {recr['error'][:120]} "
          f"{models['llm'].prompts}")

    # 19) checkpoint 恢复 fatal-stop + 回填（r2 P0-3）
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "checkpoint_runF.jsonl"
        binding = {k: "v2" for k in Checkpoint.BINDING_KEYS}
        ckf = Checkpoint(p, "runF", binding)
        fatal_rec = json.loads(json.dumps(rec10))
        fatal_rec["fatal"] = True
        ckf.append(fatal_rec)  # 模拟：写入 fatal 后进程崩溃，后续任务未补
        ck2 = Checkpoint(p, "runF", binding)
        check("checkpoint 恢复 fatal_seen", ck2.fatal_seen is True)
        tasks_left = [{"sample_id": fatal_rec["sample_id"], "mode": "non-streaming",
                       "repeat_idx": 0},
                      {"sample_id": fatal_rec["sample_id"], "mode": "streaming",
                       "repeat_idx": 1}]
        n_back = _backfill_cancelled(ck2, tasks_left, "runF", "v2", "v2")
        check("fatal 回填 cancelled", n_back == 2 and all(
            ck2.done[f"{t['sample_id']}|{t['mode']}|{t['repeat_idx']}"] == "cancelled"
            for t in tasks_left), str(n_back))

    # 20) System A 主线程 ASR/LLM 异常 → 无条件 fatal（r2 P0-2）
    with _FakeTTSServer("normal") as srv:
        models = _make_models([], "完整。", ["完", "。"], asr_fail=True)
        recA1 = run_non_streaming(_st_sample("crosswoz_t20a_turn1"), audio, sr, models,
                                  _st_pse(e), _st_cfg(srv.url), probe,
                                  seed_for_pair("crosswoz_t20a_turn1", 0), threading.Event())
    check("A ASR 异常 fatal", recA1["terminal_state"] == "error" and recA1["fatal"] is True,
          f"{recA1['terminal_state']} {recA1['fatal']}")
    with _FakeTTSServer("normal") as srv:
        models = _make_models([], "完整。", ["完", "。"])

        class _BoomLLM(_FakeLLM):
            def cache_prompt(self, prompt, pre_cache=None, is_end=False):
                raise RuntimeError("llm cache boom")
        models["llm"] = _BoomLLM(["完", "。"])
        recA2 = run_non_streaming(_st_sample("crosswoz_t20b_turn1"), audio, sr, models,
                                  _st_pse(e), _st_cfg(srv.url), probe,
                                  seed_for_pair("crosswoz_t20b_turn1", 0), threading.Event())
    check("A LLM cache 异常 fatal", recA2["terminal_state"] == "error"
          and recA2["fatal"] is True, f"{recA2['terminal_state']} {recA2['fatal']}")

    # 21) headers 后停发 body → read timeout 打断（r2 P1-2）
    with _FakeTTSServer("headers_only") as srv:
        cfg = _st_cfg(srv.url)
        cfg["read_timeout_s"] = 0.5
        cfg["tts_total_timeout_s"] = 3.0
        models = _make_models(["x。"], "x", ["句", "。"])
        t0 = time.time()
        rec21 = run_streaming(_st_sample("crosswoz_t21_turn1"), audio, sr, models,
                              _st_pse(e), cfg, probe, seed_for_pair("crosswoz_t21_turn1", 0),
                              threading.Event())
        dur = time.time() - t0
    check("headers-only 慢流被 read timeout 打断",
          rec21["terminal_state"] == "error" and "tts" in rec21["error"] and dur < 8.0,
          f"{rec21['terminal_state']} {rec21['error'][:100]} dur={dur:.1f}s")

    # 22) 零命中/少命中 smoke 拒绝（r2 P0-5）
    smoke_samples = [{"sample_id": f"s{i}", "language": "zh", "duration_group": "long"}
                     for i in range(3)]
    smoke_tasks = [{"seq": 2 * i + k, "sample_id": f"s{i}", "mode": m, "repeat_idx": 0,
                    "order": "AB"} for i in range(3) for k, m in
                   enumerate(("non-streaming", "streaming"))]
    try:
        _select_smoke(smoke_samples, smoke_tasks, 2)
        check("单语种 smoke≥2 拒绝", False)  # 3 样本全 zh，smoke=2 无法覆盖两语种
    except SystemExit:
        check("单语种 smoke≥2 拒绝", True)
    bi_samples = smoke_samples + [{"sample_id": "e1", "language": "en",
                                   "duration_group": "long"}]
    bi_tasks = smoke_tasks + [{"seq": 99, "sample_id": "e1", "mode": m, "repeat_idx": 0,
                               "order": "AB"} for m in ("non-streaming", "streaming")]
    s_ok, t_ok = _select_smoke(bi_samples, bi_tasks, 2)
    check("双语种 smoke 命中", len(s_ok) == 2 and len(t_ok) == 4
          and {x["language"] for x in s_ok} == {"zh", "en"},
          f"{[x['sample_id'] for x in s_ok]}")
    ghost_samples = [dict(bi_samples[0], sample_id="ghost")]
    try:
        _select_smoke(ghost_samples, [], 1)
        check("零命中 smoke 拒绝", False)
    except SystemExit:
        check("零命中 smoke 拒绝", True)

    # 23) segmenter 注入固定 Silero（r2 P0-4）：注入路径不得二次访问 torch.hub
    from src.asr.streamaudio_segmenter import StreamAudioSegmenter as _Seg
    import torch as _tt
    _orig_hub = _tt.hub.load

    def _no_hub(*a, **kw):
        raise AssertionError("segmenter 注入路径不得访问 torch.hub")
    _tt.hub.load = _no_hub
    try:
        seg = _Seg(silero_model=object(), silero_utils=(lambda *a, **kw: [], None, None, None, None))
        check("segmenter 注入不触发 hub", seg.silero_injected is True
              and callable(seg.get_speech_timestamps))
    except AssertionError as exc:
        check("segmenter 注入不触发 hub", False, str(exc))
    finally:
        _tt.hub.load = _orig_hub
    try:
        _Seg(silero_model=object())
        check("注入缺 utils 拒绝", False)
    except ValueError:
        check("注入缺 utils 拒绝", True)

    print(f"\nself-test {n_checks[0]} PASS / {len(fails)} FAIL"
          + ("" if not fails else ": " + "; ".join(fails)))
    return 1 if fails else 0

# ============================================================ main

def _git_info() -> dict:
    import subprocess
    out = {"git_commit": "unknown", "git_dirty": "unknown"}
    try:
        out["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
        out["git_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True).strip())
    except Exception:
        pass
    return out


def _env_versions() -> dict:
    import platform
    v = {"python": platform.python_version()}
    for mod in ("torch", "numpy", "soundfile", "librosa", "requests", "scipy"):
        try:
            m = __import__(mod)
            v[mod] = getattr(m, "__version__", "unknown")
        except ImportError:
            v[mod] = "absent"
    return v


def _silero_artifact_meta(ref: str | None, silero_dir: str | None) -> dict:
    """固定 Silero 来源并取模型 artifact hash（P0-4）。正式模式找不到 artifact 即拒。"""
    import subprocess
    meta = {"ref": ref, "dir": silero_dir, "repo_commit": None, "repo_dirty": None,
            "artifact_path": None, "artifact_sha256": None}
    candidates: list[Path] = []
    if silero_dir:
        d = Path(silero_dir)
        try:
            meta["repo_commit"] = subprocess.check_output(
                ["git", "-C", str(d), "rev-parse", "HEAD"], text=True).strip()
            meta["repo_dirty"] = bool(subprocess.check_output(
                ["git", "-C", str(d), "status", "--porcelain"], text=True).strip())
        except Exception:
            pass
        candidates = sorted(d.rglob("*.jit")) + sorted(d.rglob("*.pt")) + sorted(d.rglob("*.onnx"))
    else:
        import torch
        hub = Path(torch.hub.get_dir())
        repo_dirs = sorted(hub.glob(f"snakers4_silero-vad*{ref or ''}*")) or \
            sorted(hub.glob("snakers4_silero-vad*"))
        meta["hub_repo_dirs"] = [str(p.name) for p in repo_dirs]
        ck = hub / "checkpoints"
        if ck.exists():
            candidates = sorted(ck.glob("silero_vad*"))
    for c in candidates:
        if c.is_file() and c.stat().st_size > 100_000:
            meta["artifact_path"] = str(c)
            meta["artifact_sha256"] = sha256_file(c)
            break
    return meta


class _FaultASRProxy:
    """仅限冒烟的可控故障注入（P1-7）：armed 时 ASR 调用抛异常。"""

    def __init__(self, inner):
        self._inner = inner
        self.armed = False

    def transcribe_audio_segment(self, cache):
        if self.armed:
            raise RuntimeError("fault_injection:asr_error")
        return self._inner.transcribe_audio_segment(cache)

    def transcribe_complete_audio(self, **kw):
        if self.armed:
            raise RuntimeError("fault_injection:asr_error")
        return self._inner.transcribe_complete_audio(**kw)

    def __getattr__(self, k):
        return getattr(self._inner, k)


def _error_record(sample: dict, task: dict, cfg_hash: str, sched_hash: str,
                  run_id: str, error: str, terminal: str = "error",
                  pse: dict | None = None) -> dict:
    pse = pse or {}
    return {"schema_version": SCHEMA_VERSION, "run_id": run_id,
            "clock_type": "perf_counter_ns",
            "endpoint_mode": "explicit_flush" if task["mode"] == "streaming" else "full_input",
            "sample_id": sample.get("sample_id", task["sample_id"]),
            "language": sample.get("language", ""),
            "duration_group": sample.get("duration_group", ""),
            "mode": task["mode"], "repeat_idx": task["repeat_idx"],
            "terminal_state": terminal, "fatal": False,
            "config_hash": cfg_hash, "schedule_hash": sched_hash,
            "wav_sha256": pse.get("wav_sha256", ""),
            "analysis_waveform_sha256": pse.get("analysis_waveform_sha256", ""),
            "physical_speech_end_sample": pse.get("physical_speech_end_sample"),
            "pse_method": pse.get("pse_method", ""), "pse_diff_ms": pse.get("pse_diff_ms"),
            "events": {}, "chunk_log": [], "tts": {},
            "response_token_count": 0, "generation_stop_reason": None,
            "sentence_end_found": False, "sentence_fallback": False,
            "final_drain_triggered": False, "final_drain_empty": False,
            "tts_text": None, "tts_text_source": None, "tts_n_chars": 0,
            "tts_n_bytes_utf8": 0, "tts_seeded": False,
            "tts_text_sha256": None, "generation_seed": None, "error": error}


def _select_smoke(samples: list[dict], tasks: list[dict], n_smoke: int):
    """冒烟分层选取 + 精确命中校验（r2 P0-5，独立可测）。返回 (samples, tasks)。"""
    zh_s = [s for s in samples if s["language"] == "zh"]
    en_s = [s for s in samples if s["language"] == "en"]
    keep, i = [], 0
    while len(keep) < n_smoke and (i < len(zh_s) or i < len(en_s)):
        if i < len(zh_s):
            keep.append(zh_s[i]["sample_id"])
        if len(keep) < n_smoke and i < len(en_s):
            keep.append(en_s[i]["sample_id"])
        i += 1
    keep_set = set(keep[:n_smoke])
    if len(keep_set) != n_smoke:
        raise SystemExit(f"smoke 命中 {len(keep_set)}/{n_smoke} 个样本（停止）")
    if n_smoke >= 2 and len({s["language"] for s in samples if s["sample_id"] in keep_set}) < 2:
        raise SystemExit("smoke 未覆盖中英两语种（停止）")
    new_tasks = [t for t in tasks if t["sample_id"] in keep_set and t["repeat_idx"] == 0]
    if len(new_tasks) != n_smoke * 2:
        raise SystemExit(f"smoke 任务数 {len(new_tasks)} != {n_smoke}*2（停止）")
    new_samples = [s for s in samples if s["sample_id"] in keep_set]
    return new_samples, new_tasks


def _backfill_cancelled(ck: "Checkpoint", tasks: list[dict], run_id: str,
                        cfg_hash: str, sched_hash: str) -> int:
    """fatal 后为全部剩余任务补写 cancelled 终态（r2 P0-2/P0-3 可测函数）。"""
    n = 0
    for task in tasks:
        key = f"{task['sample_id']}|{task['mode']}|{task['repeat_idx']}"
        if key in ck.done:
            continue
        ck.append(_error_record({}, task, cfg_hash, sched_hash, run_id,
                                "cancelled_after_fatal", terminal="cancelled"))
        n += 1
    return n


def main() -> int:
    from src.config import ASR_MODEL_NAME, LLM_MODEL_NAME
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample-list", help="50 样本清单 JSON（数组或 {'sample_ids': [...]}）")
    ap.add_argument("--subset-list", help="10 条重复子集 ID JSON（缺省取清单分层前 10 条）")
    ap.add_argument("--json-dir", default="experiments/datasets/processed/json")
    ap.add_argument("--audio-dir", default="experiments/datasets/processed/audio")
    ap.add_argument("--datasets", nargs="+", default=["crosswoz", "multiwoz"])
    ap.add_argument("--asr-model", default=ASR_MODEL_NAME)
    ap.add_argument("--asr-device", default="cuda:0")
    ap.add_argument("--llm-model", default=LLM_MODEL_NAME)
    ap.add_argument("--llm-device", default="cuda:1")
    ap.add_argument("--tts-url", default="http://127.0.0.1:20401")
    ap.add_argument("--tts-spk", default="晓伊")
    ap.add_argument("--tts-speed", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--pair-deadline-s", type=float, default=PAIR_DEADLINE_S)
    ap.add_argument("--silero-ref", default=None, help="固定 Silero commit（正式模式必填其一）")
    ap.add_argument("--silero-dir", default=None, help="本地 silero 仓库目录（source='local'）")
    ap.add_argument("--output-dir", default="experiments/results/revision/r7_ttfa_unified")
    ap.add_argument("--run-id", required=False)
    ap.add_argument("--smoke", type=int, default=0, help="冒烟：分层选取 N 个样本（repeat 0）")
    ap.add_argument("--inject-fault", choices=["none", "asr_error"], default="none",
                    help="可控故障注入（仅限 --smoke；正式模式禁止）")
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
    # P0-4：正式/冒烟模式必须固定 Silero 来源
    if not args.silero_ref and not args.silero_dir:
        ap.error("正式/冒烟模式必须 --silero-ref 或 --silero-dir（禁止浮动 master）")
    if args.inject_fault != "none" and not args.smoke:
        ap.error("--inject-fault 仅限 --smoke 冒烟模式")

    import torch
    from src.asr.streamaudio_segmenter import StreamAudioSegmenter
    from src.asr.faster_whisper_streamer import StreamingASRProcessor, ASRCache
    from src.llm.stream_llm_inference import StreamLLMInference
    from src.asr.run_stream_asr_test import convert_audio_segment
    from experiments.scripts.run_exp_latency import load_samples

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
        # 分层选取 + 精确命中校验（r2 P0-5，零命中/少命中立即非零退出）
        samples, tasks = _select_smoke(samples, tasks, args.smoke)

    git = _git_info()
    env = _env_versions()
    cfg = {"asr_model": args.asr_model, "asr_device": args.asr_device,
           "llm_model": args.llm_model, "llm_device": args.llm_device,
           "chunk_ms": CHUNK_MS, "prefix_segments": 1, "suffix_segments": 0,
           "recognition_threshold": 2.0, "max_tokens": args.max_tokens,
           "pair_deadline_s": args.pair_deadline_s,
           "tts_url": args.tts_url, "tts_spk": args.tts_spk, "tts_speed": args.tts_speed,
           "sample_list_sha256": sha256_file(args.sample_list),
           "requested_repetition_penalty": 1.1, "effective_repetition_penalty": "not_applied",
           "temperature": 0.1, "top_p": 0.9}
    sched_hash = schedule_hash(tasks)

    # TTS 探活（正式前必过；允许策略由探活固定）
    probe = tts_probe(args.tts_url, args.tts_spk, args.tts_speed)
    (out_dir / "tts_probe.json").write_text(json.dumps(probe, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    if not probe.get("ok"):
        raise SystemExit(f"TTS 探活失败: {probe}（停止）")

    # Silero（固定 revision + artifact hash，P0-4）
    silero_meta = _silero_artifact_meta(args.silero_ref, args.silero_dir)
    if not silero_meta.get("artifact_sha256"):
        raise SystemExit(f"Silero artifact 未找到/未哈希: {silero_meta}（停止）")
    cfg["silero_meta"] = silero_meta
    cfg_hash = config_hash(cfg)

    if args.silero_dir:
        silero_model, utils = torch.hub.load(repo_or_dir=args.silero_dir, model="silero_vad",
                                             source="local", onnx=False, verbose=False)
    else:
        silero_model, utils = torch.hub.load(repo_or_dir=f"snakers4/silero-vad:{args.silero_ref}",
                                             model="silero_vad", onnx=False, verbose=False)
    get_speech_timestamps = utils[0]

    # PSE 预扫描（fail fast；音频 hash 映射入 binding）
    logger.info("PSE 预扫描…")
    pse_by_id: dict[str, dict] = {}
    for s in samples:
        pse = analyze_pse(s["audio_path"], get_speech_timestamps)
        if pse.get("error"):
            raise SystemExit(f"PSE 失败 {s['sample_id']}: {pse['error']}（停止）")
        pse_by_id[s["sample_id"]] = pse
    audio_map_hash = sha256_text(canonical_json(
        {sid: pse_by_id[sid]["wav_sha256"] for sid in sorted(pse_by_id)}))

    asr_processor = StreamingASRProcessor(
        model_size=args.asr_model, device=args.asr_device, compute_type="auto",
        recognition_threshold=2.0, prefix_segments=1, suffix_segments_atleast=0)
    llm = StreamLLMInference(model_name=args.llm_model, device=args.llm_device,
                             eval_mode=False)
    if args.inject_fault == "asr_error":
        asr_processor = _FaultASRProxy(asr_processor)
    # r2 P0-4：正式分段器注入与 PSE 同一固定 Silero 实例（不再二次浮动 hub 加载）
    segmenter = StreamAudioSegmenter(silero_model=silero_model, silero_utils=utils)
    if not getattr(segmenter, "silero_injected", False):
        raise SystemExit("segmenter 未使用注入的固定 Silero（停止）")
    segmenter_meta = {"pse_and_segmenter_same_artifact": True,
                      "artifact_sha256": silero_meta.get("artifact_sha256"),
                      "segmenter_silero_injected": segmenter.silero_injected}
    models = {"segmenter": segmenter, "asr": asr_processor, "llm": llm,
              "new_asr_cache": ASRCache, "convert_audio_segment": convert_audio_segment,
              "decode_fn": lambda ids: llm.tokenizer.decode(ids, skip_special_tokens=True)}
    tts_cfg = {"url": args.tts_url, "spk_id": args.tts_spk, "speed": args.tts_speed,
               "max_tokens": args.max_tokens, "config_hash": cfg_hash,
               "schedule_hash": sched_hash, "run_id": args.run_id,
               "pair_deadline_s": args.pair_deadline_s}

    binding = {"schema_version": SCHEMA_VERSION, "run_id": args.run_id,
               "config_hash": cfg_hash, "schedule_hash": sched_hash,
               "git_commit": git["git_commit"], "git_dirty": git["git_dirty"],
               "env_versions": env, "asr_model": args.asr_model, "llm_model": args.llm_model,
               "silero_meta": silero_meta,
               "tts_config": {"url": args.tts_url, "spk_id": args.tts_spk,
                              "speed": args.tts_speed, "probe": probe},
               "sample_list_sha256": cfg["sample_list_sha256"],
               "subset_sha256": sha256_text(canonical_json(sorted(subset_ids))),
               "audio_map_sha256": audio_map_hash}
    ck_path = out_dir / f"checkpoint_{args.run_id}.jsonl"
    if args.no_resume and ck_path.exists():
        ck_path.unlink()
    ck = Checkpoint(ck_path, args.run_id, binding)
    # r2 P0-3：恢复 fatal-stop（历史 fatal 记录 → 剩余任务只补 cancelled，不执行）
    fatal_stop = bool(getattr(ck, "fatal_seen", False))
    if fatal_stop:
        logger.warning("checkpoint 含 fatal 记录：本 run 恢复 fail-stop，剩余任务补 cancelled")
    fault_task_key = tasks[-1] and f"{tasks[-1]['sample_id']}|{tasks[-1]['mode']}|{tasks[-1]['repeat_idx']}" \
        if args.inject_fault != "none" else None
    for task in tasks:
        key = f"{task['sample_id']}|{task['mode']}|{task['repeat_idx']}"
        if key in ck.done:
            logger.info(f"跳过已完成 {key}（{ck.done[key]}）")
            continue
        if fatal_stop:
            continue  # 循环外统一 _backfill_cancelled 补写 cancelled 终态
        sample = dict(next(s for s in samples if s["sample_id"] == task["sample_id"]),
                      repeat_idx=task["repeat_idx"])
        cancel = threading.Event()
        if fault_task_key == key and hasattr(models["asr"], "armed"):
            models["asr"].armed = True
        try:
            import soundfile as sf
            audio, sr_f = sf.read(sample["audio_path"], dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr_f != ANALYSIS_SR:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr_f, target_sr=ANALYSIS_SR)
            audio = np.ascontiguousarray(audio, dtype=np.float32)
            pse = pse_by_id[sample["sample_id"]]
            seed = seed_for_pair(sample["sample_id"], task["repeat_idx"])
            if task["mode"] == "streaming":
                rec = run_streaming(sample, audio, ANALYSIS_SR, models, pse, tts_cfg,
                                    probe, seed, cancel)
            else:
                rec = run_non_streaming(sample, audio, ANALYSIS_SR, models, pse, tts_cfg,
                                        probe, seed, cancel)
        except Exception:
            rec = _error_record(sample, task, cfg_hash, sched_hash, args.run_id,
                                traceback.format_exc()[-500:])
        finally:
            if hasattr(models["asr"], "armed"):
                models["asr"].armed = False
        problems = validate_record(rec, cfg_hash, sched_hash)
        if problems:
            rec["terminal_state"] = "error"
            rec["error"] = (rec.get("error", "") + "|validate:" + ";".join(problems))[:500]
        ck.append(rec)
        if rec.get("fatal"):
            fatal_stop = True
            logger.error(f"{key} 发生 fatal（{rec['error'][:120]}），本 run 后续任务记 cancelled")
        logger.info(f"{key}: {rec['terminal_state']}"
                    + (f" TTFA_playable={ttfa_ms(rec):.0f}ms"
                       if rec["terminal_state"] == "success" else f" {rec['error'][:100]}"))
    # r2 P0-2/P0-3：fatal（本次或 checkpoint 恢复）→ 剩余任务统一补 cancelled
    if fatal_stop:
        n_cancel = _backfill_cancelled(ck, tasks, args.run_id, cfg_hash, sched_hash)
        logger.warning(f"fail-stop：补写 {n_cancel} 条 cancelled 终态")

    # 汇总 + QA
    import csv as _csv
    all_records = list(ck.records)
    summary = summarize(all_records)
    sum_fields = ["mode", "language", "metric", "n", "mean", "std", "p50", "p90", "p95"]
    with open(out_dir / f"ttfa_summary_{args.run_id}.csv", "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=sum_fields)
        w.writeheader()
        w.writerows(summary)  # 无成功行时仅表头（中危项 3 修复）
    cv_rows = subset_cv(all_records, subset_ids)
    cv_fields = ["sample_id", "mode", "n_valid", "cv_pct", "note"]
    with open(out_dir / f"ttfa_subset_cv_{args.run_id}.csv", "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=cv_fields)
        w.writeheader()
        w.writerows(cv_rows)
    problems = qa_records(all_records, tasks, cfg_hash, sched_hash)
    n_success = sum(1 for r in all_records if r["terminal_state"] == "success")
    n_err = sum(1 for r in all_records if r["terminal_state"] != "success")
    # r2 P0-5：QA 断言成功路径与故障路径均实际执行（不只看计数）
    if args.smoke and args.inject_fault == "none" and n_success == 0:
        problems.append("smoke 无任何成功路径记录（冒烟无效）")
    if args.smoke and args.inject_fault == "asr_error":
        fault_ok = any("fault_injection" in (r.get("error") or "")
                       for r in all_records)
        if not fault_ok:
            problems.append("故障注入未产生预期 error 终态（冒烟无效）")
        if n_success == 0:
            problems.append("故障冒烟无成功路径记录（冒烟无效）")
    qa_md = ["# TTFA 统一实测 QA", "",
             f"- run_id: {args.run_id}", f"- config_hash: {cfg_hash}",
             f"- schedule_hash: {sched_hash}", f"- 任务数: {len(tasks)}",
             f"- 记录数: {len(all_records)}（success {n_success} / 非成功 {n_err}）",
             f"- QA 问题数: {len(problems)}", ""]
    qa_md += [f"- {p}" for p in problems] or ["- 无"]
    (out_dir / f"QA_{args.run_id}.md").write_text("\n".join(qa_md) + "\n", encoding="utf-8")
    runinfo = ["# TTFA 统一实测 RUNINFO", "", f"- 命令参数: {sys.argv}",
               f"- schema_version: {SCHEMA_VERSION}", f"- run_id: {args.run_id}",
               f"- config: {json.dumps(cfg, ensure_ascii=False)}",
               f"- config_hash: {cfg_hash}", f"- schedule_hash: {sched_hash}",
               f"- git: {git}", f"- env_versions: {json.dumps(env)}",
               f"- silero_meta: {json.dumps(silero_meta, ensure_ascii=False)}",
               f"- segmenter_meta: {json.dumps(segmenter_meta, ensure_ascii=False)}"
               "（PSE 与流式分段器同一固定 artifact，已断言一致）",
               f"- subset_sha256: {binding['subset_sha256']}",
               f"- audio_map_sha256: {audio_map_hash}",
               f"- playable 阈值: {PLAYABLE_BYTES} bytes（22050Hz×16bit×30ms）",
               f"- 采样实际生效参数: temperature=0.1, top_p=0.9, repetition_penalty=not_applied",
               f"- 故障注入: {args.inject_fault}（仅冒烟）", ""]
    (out_dir / f"RUNINFO_{args.run_id}.md").write_text("\n".join(runinfo), encoding="utf-8")
    print(f"完成: {len(all_records)} 记录，success {n_success} / 非成功 {n_err}；QA 问题 {len(problems)}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
