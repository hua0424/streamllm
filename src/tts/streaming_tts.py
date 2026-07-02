# src/tts/streaming_tts.py
"""
流式 TTS 接口 + 时长 profile + Mock 实现。

见 docs/decisions.md D-010：验证机用 Mock（真实测得的时长 profile 驱动，与真机时序等价），
real CosyVoice2 在实验机上实现同一接口 swap-in。所有实验指标是时序/文本类，
Mock 只要在"每片段音频时长 + 首块延迟"上与真机一致即可。

约定：音频以采样数(n_samples)度量（16kHz），Mock 不产真波形（samples=None），
只给时长；real CosyVoice2 实现会带真实波形供播放/存盘。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generator, Optional

import numpy as np

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_SAMPLE_RATE = 16000


@dataclass
class AudioChunk:
    """一个流式音频块。samples 为 None 时表示只有时长（Mock）。"""
    n_samples: int
    samples: Optional[np.ndarray] = None
    sample_rate: int = DEFAULT_SAMPLE_RATE

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.sample_rate


@dataclass
class TimingProfile:
    """
    TTS 时序画像。占位初值取自 CosyVoice2 公开指标（英文），**上实验机后用真实 benchmark 替换**。
    - samples_per_char：每个非空白字符对应的音频采样数（决定片段时长；~16 chars/s @16k ≈ 1000）
    - first_chunk_latency_ms：首块合成延迟（mouth-to-ear 用；CosyVoice2 A100 chunk M=5 ~45ms）
    - chunk_samples：单个流式音频块大小（决定 chunk 粒度；~0.5s）
    """
    samples_per_char: int = 1000
    first_chunk_latency_ms: float = 45.0
    chunk_samples: int = 8000
    sample_rate: int = DEFAULT_SAMPLE_RATE

    def n_samples_for_text(self, text: str) -> int:
        nws = sum(1 for c in text if not c.isspace())
        return max(self.chunk_samples // 4, nws * self.samples_per_char)  # 最短兜底


class StreamingTTS(ABC):
    """流式 TTS 接口：输入完整句子片段，流式产出音频块。CosyVoice2 / Mock 都实现它。"""

    @abstractmethod
    def synthesize(self, text: str) -> Generator[AudioChunk, None, None]:
        ...

    @property
    @abstractmethod
    def first_chunk_latency_ms(self) -> float:
        ...

    @property
    def sample_rate(self) -> int:
        """输出音频采样率（下游换算 samples↔秒 统一从这里取，勿硬编码）。"""
        return DEFAULT_SAMPLE_RATE


class MockStreamingTTS(StreamingTTS):
    """时长 profile 驱动的 Mock：不产真波形，只按 profile 给出与真机等价的时长/分块。"""

    def __init__(self, profile: Optional[TimingProfile] = None):
        self.profile = profile or TimingProfile()

    @property
    def first_chunk_latency_ms(self) -> float:
        return self.profile.first_chunk_latency_ms

    @property
    def sample_rate(self) -> int:
        return self.profile.sample_rate

    def synthesize(self, text: str) -> Generator[AudioChunk, None, None]:
        total = self.profile.n_samples_for_text(text)
        emitted = 0
        while emitted < total:
            n = min(self.profile.chunk_samples, total - emitted)
            emitted += n
            yield AudioChunk(n_samples=n, samples=None, sample_rate=self.profile.sample_rate)
        logger.debug(f"[mock-tts] '{text[:20]}' → {total} samples "
                     f"({total / self.profile.sample_rate:.2f}s)")
