# src/player/player.py
"""
SimulatedPlayer —— 确定性播放器（实验用）。

职责：接收 TTS 产出的音频块，向 PlaybackTimeline 登记每块时长（推进 sample 轴），
并回报"播放到哪了"。见 experiment_design.md P1（确定性程序注入）：实验不需实时交互，
用程序化定位播放位置即可复现。

真实时播放器（实验机 / 定性 demo 用，实时输出音频 + 按墙钟回报位置）留待后续；
它与本类共享"向 timeline 回报 played_samples"的契约。
"""

from src.dialogue.timeline import PlaybackTimeline
from src.tts.streaming_tts import AudioChunk
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class SimulatedPlayer:
    def __init__(self, timeline: PlaybackTimeline):
        self.timeline = timeline
        self._total_samples = 0
        self._chunk_counter = 0

    def enqueue(self, fragment_id: int, chunk: AudioChunk) -> int:
        """把一个音频块登记到 timeline（推进 sample 轴），返回其 chunk_id。"""
        cid = self._chunk_counter
        self._chunk_counter += 1
        self.timeline.attach_chunk(fragment_id, chunk_id=cid, n_samples=chunk.n_samples)
        self._total_samples += chunk.n_samples
        return cid

    @property
    def total_samples(self) -> int:
        return self._total_samples

    def seek_fraction(self, fraction: float) -> int:
        """程序化把播放位置定到总时长的 fraction 处，回报给 timeline，返回采样位置。"""
        fraction = max(0.0, min(1.0, fraction))
        pos = int(self._total_samples * fraction)
        self.timeline.set_played(pos)
        logger.debug(f"[player] seek {fraction:.2f} → {pos}/{self._total_samples} samples")
        return pos

    def seek_samples(self, pos: int) -> int:
        pos = max(0, min(self._total_samples, int(pos)))
        self.timeline.set_played(pos)
        return pos
