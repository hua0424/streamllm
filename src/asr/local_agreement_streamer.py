# src/asr/local_agreement_streamer.py
"""
LocalAgreement-2 流式 ASR 提交策略（DEV-3）

策略出处：ufal/whisper_streaming（论文修改稿中作为基线策略引用），此处为同引擎自实现，
用于 R3 基线对比：与 System B 唯一的实验变量是 ASR 上下文/提交策略。

硬性约束（与 System A/B 对齐）：
- 模型加载复用 faster_whisper_streamer._load_whisper_model_offline_first（同权重/精度/设备）
- 转录参数与 _transcribe_segments 完全一致
- 分段沿用同一 StreamAudioSegmenter（由调用方保证）

算法（LA-2，2026-08-20 错帧修复 + 裁剪幻听修复后语义）：
1. 维护音频缓冲 buffer、缓冲起点绝对时间 buffer_start_abs、上一轮假设中
   未提交区域的词序列 prev_words（绝对时间轴）、已提交词簿记 committed_words
   与已提交边界 committed_end_abs（均为绝对音频时间轴）
2. feed_segment：音频入 buffer；new_audio 达到 decode_trigger_s 或 is_final 时解码整个 buffer，
   词时间戳立即换算为绝对轴
3. 未提交区域词 = end > committed_end_abs + eps 且 start >= committed_end_abs - eps；
   骑跨边界的词若与重叠已提交词的规范化文本相同，视为已提交内容的重渲染而排除
4. 公共前缀比较只看"实质词"（滤掉纯标点词），词文本按去首尾标点后的规范化文本比较
   （Whisper 对同一音频的标点附着/分词在不同解码轮间不稳定，逐字比较会卡死提交）；
   提交时把 cur_new 中到最后一个达成一致词为止的全部词（含中间纯标点词）一并提交，
   且不超过 缓冲末端 - trailing_margin 提交线；与已提交词区间重叠且文本相同的
   重识别残留跳过（防重复）
5. 缓冲裁剪（与提交解耦，仅控成本，对齐 ufal buffer_trimming="segment" + 15s 上限）：
   优先裁到最后一个句末已提交词（。！？!?.… 结尾）的 end，无回退——
   实测 Whisper turbo 对以句末残片开头的缓冲会坍缩为训练集水印幻听
   （'请不吝点赞订阅转发…'），句中词边界切开则正常；无句界锚点且缓冲
   超过 max_buffer_s 时强制裁到 committed_end_abs
6. flush()：按同一时间下界提交 prev_words 中全部剩余词（幂等）

修复背景（E3-LA 2026-08-20 结果无效事件）：原实现用"n_committed 词数下标 + buffer 相对轴"
标记已提交位置，_trim_buffer() 裁剪音频后下一轮假设是尾段帧的词序列，下标错帧导致
提交跳过新假设前 n_committed 个未提交词（中段文本静默丢失）。逐帧重放另发现裁剪在
句中制造句末残片开头缓冲触发 Whisper 幻听坍缩。修复要点：绝对时间轴 + 时间下界提交 +
标点鲁棒的一致比较 + 句界裁剪，见 experiments/results/revision/REVISION_CHANGELOG.md。
"""

import time
from typing import List, Dict

import numpy as np

from src.asr.faster_whisper_streamer import (
    ASRAudioSegment,
    _load_whisper_model_offline_first,
    _normalize_device,
    DEFAULT_BEAM_SIZE,
    DEFAULT_TEMPERATURE,
    DEFAULT_COMPRESSION_RATIO_THRESHOLD,
    DEFAULT_LOG_PROB_THRESHOLD,
    DEFAULT_NO_SPEECH_THRESHOLD,
)
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

COMMIT_EPS_S = 0.02  # 提交边界时间容差（秒）：防止边界词的时间戳抖动造成重复/跳过
SENTENCE_END_CHARS = "。！？!?.…"  # 句末标点（裁剪锚点判定）
_PUNCT_STRIP = "，。！？,.!?、；;：:…—~～·\"'（）()「」『』【】 "  # 规范化时剥离的首尾字符


def _norm_text(t: str) -> str:
    """词文本规范化：去首尾标点/空白（一致比较用；提交仍用原文）"""
    return t.strip(_PUNCT_STRIP)


def _same_word_text(a: str, b: str) -> bool:
    """判定两词文本是否同一内容：原文相同，或规范化后非空且相同（容忍标点附着差异）"""
    if a == b:
        return True
    na, nb = _norm_text(a), _norm_text(b)
    return bool(na) and na == nb


class LocalAgreementStreamer:
    """LocalAgreement-2 提交策略的流式 ASR 处理器"""

    def __init__(
        self,
        model_size: str,
        device: str,
        sample_rate: int = 16000,
        decode_trigger_s: float = 2.0,
        trailing_margin_s: float = 0.0,
        max_buffer_s: float = 15.0,
    ):
        """
        Args:
            model_size: whisper 模型大小（与 System A/B 相同）
            device: 推理设备
            sample_rate: 音频采样率
            decode_trigger_s: 解码触发新增音频时长，与 System B recognition_threshold 对齐（2.0s）
            trailing_margin_s: 尾随保护余量（秒）。锁定配置 suffix=0 对应 0.0
            max_buffer_s: 缓冲长度上限（秒），对齐 ufal whisper_streaming buffer_trimming_sec=15；
                无句界锚点时超过该长度强制裁剪到已提交边界
        """
        self.sample_rate = sample_rate
        self.decode_trigger_s = decode_trigger_s
        self.trailing_margin_s = trailing_margin_s
        self.max_buffer_s = max_buffer_s
        self._device = _normalize_device(device)
        self._model_size = model_size
        self.model = _load_whisper_model_offline_first(model_size, self._device)

        self.reset()
        logger.info(f"LocalAgreementStreamer 初始化完成: model={model_size}, device={self._device}, "
                    f"trigger={decode_trigger_s}s, trailing_margin={trailing_margin_s}s, "
                    f"max_buffer={max_buffer_s}s")

    def reset(self) -> None:
        """清空全部流状态（每个样本测试前必须调用）"""
        self.buffer = np.array([], dtype=np.float32)
        self.buffer_start_abs: float = 0.0   # buffer[0] 对应的绝对音频时间（秒）
        self.prev_words: List[Dict] = []     # 上一轮假设的未提交区域词（绝对时间轴）
        self.committed_words: List[Dict] = []  # 已提交词簿记（text/start/end，绝对时间轴，append-only）
        self.committed_end_abs: float = 0.0  # 最后提交词的 end（绝对音频时间轴）
        self.new_audio: float = 0.0
        # P1-3 观测：新假设改写已提交区域内容的事件序列
        self.divergence_events: List[Dict] = []

    def feed_segment(self, segment: ASRAudioSegment) -> List[str]:
        """
        喂入一个 VAD 音频段，返回本轮新提交的文本片段列表（可为空）。

        Args:
            segment: 分段器输出的音频段（is_final 标记流结束）
        """
        audio = segment.audio_data
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        self.buffer = np.concatenate([self.buffer, audio])
        self.new_audio += segment.duration

        if self.new_audio < self.decode_trigger_s and not segment.is_final:
            return []

        cur_rel = self._decode_buffer()
        self.new_audio = 0.0
        if not cur_rel:
            # 空识别（如纯静音段）：保留上一轮假设（append-only 状态不可回退）。
            # 等待下一轮重新识别即可自然恢复（R2-P0-1）。
            logger.debug("LA 本轮识别为空，保留上一轮假设")
            return []

        # 词时间轴换算：buffer 相对轴 -> 绝对音频时间轴（此后全部判定在绝对轴上进行）
        base = self.buffer_start_abs
        cur_words = [{"text": w["text"], "start": base + w["start"], "end": base + w["end"]}
                     for w in cur_rel]

        # P1-3 观测：新假设改写已提交区域内容 = 失配事件（诊断用，不触发回滚）
        self._check_divergence(cur_words)

        prev_new = [w for w in self.prev_words if self._in_uncommitted_region(w)]
        cur_new = [w for w in cur_words if self._in_uncommitted_region(w)]

        # 一致比较只看实质词（滤掉纯标点词），按规范化文本求最长公共前缀。
        # 纯标点词不参与比较：其附着在解码轮间不稳定，作锚点会卡死提交。
        prev_cmp = [w for w in prev_new if _norm_text(w["text"])]
        cur_cmp = [w for w in cur_new if _norm_text(w["text"])]
        agreed: List[Dict] = []
        for pw, cw in zip(prev_cmp, cur_cmp):
            if _same_word_text(pw["text"], cw["text"]):
                agreed.append(cw)
            else:
                break
        logger.debug(
            f"LA 比对: agreed={len(agreed)} "
            f"prev_cmp[{len(prev_cmp)}]={[w['text'] for w in prev_cmp[:6]]} "
            f"cur_cmp[{len(cur_cmp)}]={[w['text'] for w in cur_cmp[:6]]} "
            f"boundary={self.committed_end_abs:.2f}")

        # 提交 cur_new 中到最后一个达成一致词为止的全部词（含中间纯标点词），
        # 不超过提交线；裁剪保留区内已提交词的重识别残留跳过（防重复）
        buffer_end_abs = base + len(self.buffer) / self.sample_rate
        commit_upto_end = buffer_end_abs - self.trailing_margin_s
        new_committed: List[Dict] = []
        if agreed:
            last_idx = next(i for i, w in enumerate(cur_new) if w is agreed[-1])
            for w in cur_new[:last_idx + 1]:
                if w["end"] > commit_upto_end:
                    break
                if self._is_duplicate_of_committed(w):
                    continue
                new_committed.append(w)

        fragments: List[str] = []
        if new_committed:
            self.committed_words.extend(
                {"text": w["text"], "start": w["start"], "end": w["end"]} for w in new_committed)
            self.committed_end_abs = new_committed[-1]["end"]
            text = "".join(w["text"] for w in new_committed).strip()
            if text:
                fragments.append(text)
            logger.debug(f"LA 提交 {len(new_committed)} 词: '{text}'")

        # 帧基线 = 本轮假设的未提交区域词（绝对轴）。裁剪后无需平移；
        # 未提交词必须在下一轮假设中重新出现才会被提交（两轮确认不削弱）
        self.prev_words = cur_new
        self._trim_buffer()
        return fragments

    def flush(self) -> str:
        """流结束：提交当前假设中全部未提交词，返回文本（可为空串）。幂等：二次调用返回空串"""
        remaining = [w for w in self.prev_words
                     if self._in_uncommitted_region(w)
                     and not self._is_duplicate_of_committed(w)]
        self.prev_words = []
        if not remaining:
            return ""
        self.committed_words.extend(
            {"text": w["text"], "start": w["start"], "end": w["end"]} for w in remaining)
        self.committed_end_abs = remaining[-1]["end"]
        text = "".join(w["text"] for w in remaining).strip()
        logger.debug(f"LA flush 提交 {len(remaining)} 词: '{text}'")
        return text

    def _in_uncommitted_region(self, w: Dict) -> bool:
        """w（绝对轴）是否属于未提交区域：end 越过已提交边界，且不是已提交内容的重渲染。

        骑跨边界（start < boundary - eps）且与重叠已提交词规范化文本相同的词视为
        重渲染而排除（标点附着/分词在边界附近不稳定，纳入比较会锚定失配卡死提交）；
        骑跨但不匹配的按新词保留（宁可承担极小重复风险，不跳过真实内容）。
        """
        eps = COMMIT_EPS_S
        if w["end"] <= self.committed_end_abs + eps:
            return False
        if w["start"] >= self.committed_end_abs - eps:
            return True
        for cw in reversed(self.committed_words):
            if cw["end"] <= self.buffer_start_abs + eps:
                break  # 只检查仍在缓冲覆盖范围内的已提交词
            if w["start"] < cw["end"] and w["end"] > cw["start"]:
                if _same_word_text(w["text"], cw["text"]):
                    return False
        return True

    def _is_duplicate_of_committed(self, w: Dict) -> bool:
        """w（绝对轴）是否为缓冲保留区内已提交词的重识别残留：时间区间重叠且文本相同"""
        for cw in reversed(self.committed_words):
            if cw["end"] <= self.buffer_start_abs + COMMIT_EPS_S:
                break  # 只检查仍被缓冲覆盖的已提交词
            if w["start"] < cw["end"] and w["end"] > cw["start"] \
                    and _same_word_text(w["text"], cw["text"]):
                return True
        return False

    def _check_divergence(self, cur_words: List[Dict]) -> None:
        """新假设改写已提交区域内容则记录失配事件（每轮最多一次；append-only 不回滚）"""
        if not self.committed_words:
            return
        eps = COMMIT_EPS_S
        # 仍被缓冲覆盖的已提交词才可能被新假设重新识别
        zone = [cw for cw in self.committed_words
                if cw["end"] > self.buffer_start_abs + eps]
        if not zone:
            return
        for w in cur_words:
            if w["start"] >= self.committed_end_abs + eps:
                break  # 词按时间有序，其后必在未提交区域
            mid = 0.5 * (w["start"] + w["end"])
            if mid >= self.committed_end_abs:
                continue  # 骑跨边界的新词（中点在边界以右）不算改写已提交内容
            overlap = [cw for cw in zone if cw["start"] < w["end"] and w["start"] < cw["end"]]
            if not overlap or all(not _same_word_text(cw["text"], w["text"]) for cw in overlap):
                event = {
                    "t": time.time(),
                    "committed_end": self.committed_end_abs,
                    "hypothesis_span": [w["start"], w["end"]],
                    "hypothesis_text": w["text"],
                    "committed_texts": [cw["text"] for cw in overlap],
                }
                self.divergence_events.append(event)
                logger.warning(f"LA 假设改写已提交区域: '{w['text']}' vs "
                               f"已提交 {[cw['text'] for cw in overlap]}")
                return

    def _decode_buffer(self) -> List[Dict]:
        """对整个 buffer 转录，返回扁平化词序列 [{text,start,end}]（buffer 相对轴）"""
        buffer_duration = len(self.buffer) / self.sample_rate
        start = time.perf_counter()
        result_obj = self.model.transcribe(
            self.buffer,
            beam_size=DEFAULT_BEAM_SIZE,
            word_timestamps=True,
            temperature=DEFAULT_TEMPERATURE,
            compression_ratio_threshold=DEFAULT_COMPRESSION_RATIO_THRESHOLD,
            logprob_threshold=DEFAULT_LOG_PROB_THRESHOLD,
            no_speech_threshold=DEFAULT_NO_SPEECH_THRESHOLD,
            condition_on_previous_text=False,
        )
        elapsed = time.perf_counter() - start
        words: List[Dict] = []
        for seg in result_obj.get("segments", []):
            for w in seg.get("words", []) or []:
                words.append({"text": w["word"], "start": w["start"], "end": w["end"]})
        logger.debug(f"LA 解码 buffer {buffer_duration:.2f}s -> {len(words)} 词, 耗时 {elapsed:.3f}s")
        return words

    def _trim_buffer(self) -> None:
        """裁剪缓冲控制解码成本（与提交解耦；词簿记为绝对时间轴，无需平移）。

        锚点选择（实测依据见模块 docstring）：优先最后一个句末已提交词的 end
        （缓冲从句首开始，避免 Whisper 对句末残片开头的缓冲产生幻听坍缩）；
        无句界锚点且缓冲超过 max_buffer_s 时强制裁到已提交边界；否则不裁。
        裁剪不做回退：回退会把句末词尾部残片留在缓冲开头，同样触发幻听。
        """
        anchor = None
        for w in reversed(self.committed_words):
            if w["end"] <= self.buffer_start_abs + COMMIT_EPS_S:
                break  # 已裁出缓冲的区域
            if w["text"].rstrip().endswith(tuple(SENTENCE_END_CHARS)):
                anchor = w["end"]
                break
        if anchor is None:
            buffer_dur = len(self.buffer) / self.sample_rate
            if buffer_dur <= self.max_buffer_s:
                return
            if self.committed_end_abs <= self.buffer_start_abs + COMMIT_EPS_S:
                return  # 无可裁的已提交内容
            anchor = self.committed_end_abs  # 超限强制裁剪（对齐 ufal buffer_trimming_sec）
        trim_s = anchor - self.buffer_start_abs
        if trim_s <= 0.0:
            return
        trim_samples = int(trim_s * self.sample_rate)
        if trim_samples <= 0 or trim_samples >= len(self.buffer):
            return
        self.buffer = self.buffer[trim_samples:]
        self.buffer_start_abs += trim_s
        logger.debug(f"LA 缓冲裁剪 {trim_s:.2f}s, 剩余 {len(self.buffer) / self.sample_rate:.2f}s")
