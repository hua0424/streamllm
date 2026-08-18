# src/asr/local_agreement_streamer.py
"""
LocalAgreement-2 流式 ASR 提交策略（DEV-3）

策略出处：ufal/whisper_streaming（论文修改稿中作为基线策略引用），此处为同引擎自实现，
用于 R3 基线对比：与 System B 唯一的实验变量是 ASR 上下文/提交策略。

硬性约束（与 System A/B 对齐）：
- 模型加载复用 faster_whisper_streamer._load_whisper_model_offline_first（同权重/精度/设备）
- 转录参数与 _transcribe_segments 完全一致
- 分段沿用同一 StreamAudioSegmenter（由调用方保证）

算法（LA-2）：
1. 维护音频缓冲 buffer、上一轮假设词序列 prev_words、已提交词数 n_committed、
   自上次解码以来新增音频时长 new_audio
2. feed_segment：音频入 buffer；new_audio 达到 decode_trigger_s 或 is_final 时解码整个 buffer
3. 计算 prev_words 与当前假设的最长公共前缀（按词文本），提交其中
   word.end <= 当前音频总时长 - trailing_margin 的未提交词
4. 缓冲裁剪：丢弃最后提交词 end-0.1s 之前的音频，控制解码成本
5. flush()：提交全部剩余词
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

TRIM_BACK_SECONDS = 0.1  # 缓冲裁剪回退余量（秒）


class LocalAgreementStreamer:
    """LocalAgreement-2 提交策略的流式 ASR 处理器"""

    def __init__(
        self,
        model_size: str,
        device: str,
        sample_rate: int = 16000,
        decode_trigger_s: float = 2.0,
        trailing_margin_s: float = 0.0,
    ):
        """
        Args:
            model_size: whisper 模型大小（与 System A/B 相同）
            device: 推理设备
            sample_rate: 音频采样率
            decode_trigger_s: 解码触发新增音频时长，与 System B recognition_threshold 对齐（2.0s）
            trailing_margin_s: 尾随保护余量（秒）。锁定配置 suffix=0 对应 0.0
        """
        self.sample_rate = sample_rate
        self.decode_trigger_s = decode_trigger_s
        self.trailing_margin_s = trailing_margin_s
        self._device = _normalize_device(device)
        self._model_size = model_size
        self.model = _load_whisper_model_offline_first(model_size, self._device)

        self.buffer = np.array([], dtype=np.float32)
        self.prev_words: List[Dict] = []  # 上一轮假设（text/start/end，时间轴相对 buffer 起点）
        self.n_committed: int = 0         # 已提交词数（在当前假设序列中的位置）
        self.new_audio: float = 0.0
        self.last_committed_end: float = 0.0  # 最后提交词的 end（buffer 相对轴）
        # P1-3 观测：新假设在已提交位置之前发生分歧的事件序列
        self.divergence_events: List[Dict] = []
        logger.info(f"LocalAgreementStreamer 初始化完成: model={model_size}, device={self._device}, "
                    f"trigger={decode_trigger_s}s, trailing_margin={trailing_margin_s}s")

    def reset(self) -> None:
        """清空全部流状态（每个样本测试前必须调用）"""
        self.buffer = np.array([], dtype=np.float32)
        self.prev_words = []
        self.n_committed = 0
        self.new_audio = 0.0
        self.last_committed_end = 0.0
        self.divergence_events = []

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

        cur_words = self._decode_buffer()
        self.new_audio = 0.0
        if not cur_words:
            self.prev_words = []
            return []

        # 最长公共前缀（按词文本；prev 仅取文本，时间戳不参与比较）
        agreed: List[Dict] = []
        for pw, cw in zip(self.prev_words, cur_words):
            if pw["text"] == cw["text"]:
                agreed.append(cw)
            else:
                break

        # P1-3 观测：公共前缀短于已提交位置 = 新假设改动了已提交内容。
        # 已提交文本不可撤销（append-only），不回退 n_committed；
        # 后续轮次公共前缀可自然重新延伸，不会永久卡死；flush 仍提交剩余。
        if len(agreed) < self.n_committed:
            event = {
                "t": time.time(),
                "n_committed": self.n_committed,
                "agreed_len": len(agreed),
                "committed_head": [w["text"] for w in self.prev_words[:self.n_committed]][-3:],
                "hypothesis_head": [w["text"] for w in cur_words][:5],
            }
            self.divergence_events.append(event)
            logger.warning(f"LA 假设在已提交位置前分歧: agreed={len(agreed)} < committed={self.n_committed}")

        # 提交 agreed 中未提交且越过尾随保护线的词
        buffer_duration = len(self.buffer) / self.sample_rate
        commit_upto_end = buffer_duration - self.trailing_margin_s
        new_committed: List[Dict] = []
        for w in agreed[self.n_committed:]:
            if w["end"] <= commit_upto_end:
                new_committed.append(w)
            else:
                break

        fragments: List[str] = []
        if new_committed:
            self.n_committed += len(new_committed)
            self.last_committed_end = new_committed[-1]["end"]
            text = "".join(w["text"] for w in new_committed).strip()
            if text:
                fragments.append(text)
            logger.debug(f"LA 提交 {len(new_committed)} 词: '{text}'")

        self.prev_words = cur_words
        self._trim_buffer()
        return fragments

    def flush(self) -> str:
        """流结束：提交当前假设中全部未提交词，返回文本（可为空串）"""
        remaining = self.prev_words[self.n_committed:]
        if not remaining:
            return ""
        self.n_committed = len(self.prev_words)
        text = "".join(w["text"] for w in remaining).strip()
        logger.debug(f"LA flush 提交 {len(remaining)} 词: '{text}'")
        return text

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
        """丢弃最后提交词 end-0.1s 之前的音频，并前移词时间戳保持相对轴一致"""
        trim_s = max(0.0, self.last_committed_end - TRIM_BACK_SECONDS)
        if trim_s <= 0.0:
            return
        trim_samples = int(trim_s * self.sample_rate)
        if trim_samples <= 0 or trim_samples >= len(self.buffer):
            return
        self.buffer = self.buffer[trim_samples:]
        self.prev_words = [
            {"text": w["text"], "start": w["start"] - trim_s, "end": w["end"] - trim_s}
            for w in self.prev_words
        ]
        self.last_committed_end -= trim_s
        logger.debug(f"LA 缓冲裁剪 {trim_s:.2f}s, 剩余 {len(self.buffer) / self.sample_rate:.2f}s")
