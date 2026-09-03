# src/dialogue/timeline.py
"""
PlaybackTimeline —— 二期核心数据结构：播放感知的反向映射表

建立 LLM token ↔ 文本片段(fragment) ↔ 音频 chunk ↔ 播放进度(samples) 的四向映射，
支持"打断时按用户实际听到的播放位置反查、并给出 KV 截断点"。

设计要点（见 docs/decisions.md D-008）：
- 主干 = 按生成顺序排列的 FragmentRecord 列表（片段是截断单位）。
- 并发：一把锁罩整个 timeline；played_samples 游标原子写（Python int 赋值本身原子）。
- 打断语义（选 A）：mid-fragment 打断时，被打断的片段算"已听到"，
  截断到其 token_end（物理仍为片段边界）；若被部分播放则置 partial=True，供贡献3重写。

关键定义 —— played_samples 是"已播放的采样总数"（count 语义）：
播放到 pos 个采样，意味着采样索引 [0, pos) 已被听到。因此"当前听到位置"落在
满足 `sample_start < pos <= sample_end` 的片段里；pos == sample_end 表示该片段被完整听完
（干净边界，非 partial）。这一点对"恰好在片段边界打断"的正确性至关重要。

本模块为纯 Python 逻辑，不依赖 torch/GPU，可在任意机器上 smoke 验证。
片段数（每轮回复）通常很小，反查用线性扫描即可（避免维护增量有序缓存带来的一致性坑）。
"""

import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class FragmentStatus(Enum):
    """片段在流水线中的生命周期状态。"""
    SPECULATIVE = auto()   # LLM 已生成 token，尚未确认要不要说（推测生成，可作废）
    SYNTHESIZING = auto()  # 已送入 TTS，正在合成音频
    ENQUEUED = auto()      # 音频已合成，进入播放队列，但尚未播放
    PLAYING = auto()       # 正在播放
    PLAYED = auto()        # 已完整播放（用户完整听到）
    DISCARDED = auto()     # 被作废（打断时游标之后、或推测作废）


@dataclass
class FragmentRecord:
    """一个文本片段（stream2sentence 的一次输出）及其跨模块映射。"""
    fragment_id: int
    text: str
    token_start: int                       # 该片段覆盖的 LLM token 区间 [start, end)
    token_end: int
    status: FragmentStatus = FragmentStatus.SPECULATIVE
    chunk_ids: List[int] = field(default_factory=list)
    # 在累积播放时间轴上的采样区间 [sample_start, sample_end)；
    # 未 attach 任何 chunk 前为 None（尚不知道时长）。
    sample_start: Optional[int] = None
    sample_end: Optional[int] = None

    @property
    def has_audio(self) -> bool:
        return self.sample_start is not None and self.sample_end is not None


@dataclass
class BargeInResult:
    """打断时反查结果，供 KV 截断与重写触发使用。"""
    interrupted_fragment_id: Optional[int]  # 打断时听到位置所在片段；None=打断时啥都没播
    crop_token_end: int                     # KV 应 crop 到的 token 数（= 保留 [0, crop_token_end)）
    heard_fragment_ids: List[int]           # 计入历史（用户听到）的片段
    discarded_fragment_ids: List[int]       # 被作废的片段（游标之后 / 未播放）
    partial: bool                           # 被打断片段是否只播了一半（→ 可能触发重写）


class PlaybackTimeline:
    """
    单轮 assistant 回复的播放感知时间轴。

    线程分工（对应四线程流水线）：
    - LLM decode 线程：add_fragment()  写 token↔fragment
    - TTS 线程：       attach_chunk()  写 fragment↔chunk 并推进 sample 轴
    - 播放线程：       set_played() / mark_status()  推进播放游标与状态
    - 主/打断线程：    barge_in()  反查

    所有写操作与 barge_in 的读取都在同一把锁下；played_samples 单值原子更新可无锁读。
    """

    def __init__(self, turn_id: int = 0):
        self.turn_id = turn_id
        self._lock = threading.Lock()
        self._fragments: List[FragmentRecord] = []
        self._by_id: Dict[int, FragmentRecord] = {}
        self._next_fragment_id = 0
        self._total_samples = 0            # 已 attach 音频的累积采样数（sample 轴末端）
        self._played_samples = 0           # 播放游标：已播放的累积采样数
        self._chunk_ids = set()            # 单轮 timeline 内全局唯一
        self._last_audio_fragment_id: Optional[int] = None

    # mark_status() 只允许按生命周期向前走一步；终态不可复活。
    _STATUS_TRANSITIONS = {
        FragmentStatus.SPECULATIVE: {FragmentStatus.SYNTHESIZING, FragmentStatus.DISCARDED},
        FragmentStatus.SYNTHESIZING: {FragmentStatus.ENQUEUED, FragmentStatus.DISCARDED},
        FragmentStatus.ENQUEUED: {FragmentStatus.PLAYING, FragmentStatus.DISCARDED},
        FragmentStatus.PLAYING: {FragmentStatus.PLAYED, FragmentStatus.DISCARDED},
        FragmentStatus.PLAYED: set(),
        FragmentStatus.DISCARDED: set(),
    }

    @staticmethod
    def _require_status(status: FragmentStatus) -> None:
        if not isinstance(status, FragmentStatus):
            raise TypeError(f"status must be FragmentStatus, got {status!r}")

    def _mark_status_locked(self, rec: FragmentRecord, status: FragmentStatus) -> None:
        self._require_status(status)
        if status == rec.status:            # 幂等上报合法
            return
        if status not in self._STATUS_TRANSITIONS[rec.status]:
            raise ValueError(f"illegal status transition {rec.status.name} -> {status.name} "
                             f"for fragment {rec.fragment_id}")
        rec.status = status

    # ----------------------------------------------------------------- 写入侧
    def add_fragment(self, text: str, token_start: int, token_end: int,
                     status: FragmentStatus = FragmentStatus.SPECULATIVE) -> int:
        """LLM decode 侧：登记连续、非空的 token 区间，返回 fragment_id。"""
        self._require_status(status)
        if status != FragmentStatus.SPECULATIVE:
            raise ValueError(f"new fragment must start as SPECULATIVE, got {status.name}")
        with self._lock:
            expected_start = self._fragments[-1].token_end if self._fragments else 0
            if token_end <= token_start:
                raise ValueError(f"token span must be non-empty, got [{token_start},{token_end})")
            if token_start != expected_start:
                raise ValueError(f"token span must be contiguous: expected start {expected_start}, "
                                 f"got {token_start}")
            fid = self._next_fragment_id
            self._next_fragment_id += 1
            rec = FragmentRecord(
                fragment_id=fid, text=text,
                token_start=token_start, token_end=token_end, status=status,
            )
            self._fragments.append(rec)
            self._by_id[fid] = rec
            logger.debug(f"[timeline] +fragment {fid} tokens[{token_start},{token_end}) '{text[:20]}'")
            return fid

    def attach_chunk(self, fragment_id: int, chunk_id: int, n_samples: int) -> None:
        """TTS 侧：按 fragment 顺序追加全局唯一 chunk，并连续推进 sample 轴。"""
        if n_samples <= 0:
            raise ValueError(f"n_samples must be > 0, got {n_samples}")
        with self._lock:
            rec = self._by_id.get(fragment_id)
            if rec is None:
                raise KeyError(f"unknown fragment_id {fragment_id}")
            if chunk_id in self._chunk_ids:
                raise ValueError(f"duplicate chunk_id {chunk_id}")
            if (self._last_audio_fragment_id is not None
                    and fragment_id < self._last_audio_fragment_id):
                raise ValueError(f"cannot attach fragment {fragment_id} after audio for later fragment "
                                 f"{self._last_audio_fragment_id} has started")
            if rec.status not in (FragmentStatus.SPECULATIVE, FragmentStatus.SYNTHESIZING):
                raise ValueError(f"cannot attach audio to fragment {fragment_id} in terminal/advanced "
                                 f"status {rec.status.name}")

            # 所有校验先完成再写，异常时 timeline 保持原样（fail closed）。
            if rec.sample_start is None:
                sample_start = self._total_samples
                sample_end = self._total_samples
            else:
                if rec.sample_end != self._total_samples:
                    raise RuntimeError(f"non-contiguous sample range for fragment {fragment_id}: "
                                       f"end={rec.sample_end}, timeline_end={self._total_samples}")
                sample_start = rec.sample_start
                sample_end = rec.sample_end

            rec.sample_start = sample_start
            rec.sample_end = sample_end + n_samples
            rec.chunk_ids.append(chunk_id)
            self._chunk_ids.add(chunk_id)
            self._total_samples = rec.sample_end
            self._last_audio_fragment_id = fragment_id
            if rec.status == FragmentStatus.SPECULATIVE:
                rec.status = FragmentStatus.SYNTHESIZING
            logger.debug(f"[timeline] fragment {fragment_id} +chunk {chunk_id} "
                         f"(+{n_samples} samples) → [{rec.sample_start},{rec.sample_end})")

    def mark_status(self, fragment_id: int, status: FragmentStatus) -> None:
        """推进片段生命周期；拒绝跳步、回退和终态复活。"""
        with self._lock:
            rec = self._by_id.get(fragment_id)
            if rec is None:
                raise KeyError(f"unknown fragment_id {fragment_id}")
            self._mark_status_locked(rec, status)

    def set_played(self, played_samples: int) -> None:
        """单调推进播放游标；超出已 attach 音频的部分钳制到 sample 轴末端。"""
        if not isinstance(played_samples, int):
            raise TypeError(f"played_samples must be int, got {played_samples!r}")
        if played_samples < 0:
            raise ValueError(f"played_samples must be >= 0, got {played_samples}")
        with self._lock:
            clamped = min(played_samples, self._total_samples)
            if clamped < self._played_samples:
                raise ValueError(f"played_samples cannot move backward: "
                                 f"{self._played_samples} -> {clamped}")
            self._played_samples = clamped

    @property
    def played_samples(self) -> int:
        return self._played_samples

    @property
    def total_samples(self) -> int:
        return self._total_samples

    # ----------------------------------------------------------------- 反查侧
    def _fragment_heard_up_to_locked(self, pos: int) -> Optional[FragmentRecord]:
        """
        count 语义：播放了 pos 个采样后，用户"当前听到位置"所在的片段。
        规则 `sample_start < pos <= sample_end`：
        - pos <= 0：还没听到任何内容 → None
        - pos 落在某片段区间内：命中该片段（pos==sample_end 表示恰好听完，仍归该片段但非 partial）
        - pos > 全部音频末端（超播）：归到最后一个有音频的片段
        片段数很小，线性扫描即可。
        """
        if pos <= 0:
            return None
        last_audio: Optional[FragmentRecord] = None
        for f in self._fragments:
            if not f.has_audio:
                continue
            last_audio = f
            if f.sample_start < pos <= f.sample_end:
                return f
        # pos 超过所有已合成音频 → 归到最后一个有音频的片段（None 表示无任何音频）
        return last_audio

    def fragment_at_sample(self, sample_pos: int) -> Optional[FragmentRecord]:
        """反查：给定已播放采样数，返回"当前听到位置"所在片段（count 语义）。"""
        with self._lock:
            return self._fragment_heard_up_to_locked(sample_pos)

    def _resolve_barge_in_locked(self, pos: int) -> BargeInResult:
        """在锁内计算打断反查结果（不修改 status）。"""
        frag = self._fragment_heard_up_to_locked(pos)
        if frag is None:
            # 打断时还没有任何音频被听到 → 整段推测作废，KV 回滚到 assistant 起点
            discarded = [f.fragment_id for f in self._fragments]
            return BargeInResult(None, 0, [], discarded, False)
        crop_token_end = frag.token_end                      # 选 A：含被打断片段
        partial = frag.sample_end is not None and pos < frag.sample_end
        heard = [f.fragment_id for f in self._fragments if f.token_end <= crop_token_end]
        discarded = [f.fragment_id for f in self._fragments if f.token_end > crop_token_end]
        return BargeInResult(frag.fragment_id, crop_token_end, heard, discarded, partial)

    def barge_in(self, playback_samples: Optional[int] = None) -> BargeInResult:
        """
        打断反查（选 A 语义）并落实片段 status。
        以 playback_samples（缺省用当前游标）为"用户听到位置"：
        - 命中片段 F 算已听到，crop 到 F.token_end；F 之后片段全部 DISCARDED；
        - pos < F.sample_end → F 只播了一半，partial=True（供贡献3重写）。
        无任何已听到音频时：crop_token_end=0（整段推测回滚）。
        """
        pos = self._played_samples if playback_samples is None else playback_samples
        with self._lock:
            res = self._resolve_barge_in_locked(pos)
            heard = set(res.heard_fragment_ids)
            targets = {}
            for f in self._fragments:
                if f.fragment_id in heard:
                    target = (FragmentStatus.PLAYING
                              if f.fragment_id == res.interrupted_fragment_id and res.partial
                              else FragmentStatus.PLAYED)
                else:
                    target = FragmentStatus.DISCARDED
                targets[f.fragment_id] = target

            # barge_in 可按实际播放位置从合成/排队态直接结算，但不能复活 DISCARDED、
            # 也不能把用户已听到的 PLAYED 内容改成未听。先全量校验再统一写入。
            allowed_sources = {
                FragmentStatus.PLAYING: {
                    FragmentStatus.SYNTHESIZING, FragmentStatus.ENQUEUED,
                    FragmentStatus.PLAYING,
                },
                FragmentStatus.PLAYED: {
                    FragmentStatus.SYNTHESIZING, FragmentStatus.ENQUEUED,
                    FragmentStatus.PLAYING, FragmentStatus.PLAYED,
                },
                FragmentStatus.DISCARDED: {
                    FragmentStatus.SPECULATIVE, FragmentStatus.SYNTHESIZING,
                    FragmentStatus.ENQUEUED, FragmentStatus.PLAYING,
                    FragmentStatus.DISCARDED,
                },
            }
            for f in self._fragments:
                target = targets[f.fragment_id]
                if f.status not in allowed_sources[target]:
                    raise ValueError(f"illegal barge_in status transition {f.status.name} -> "
                                     f"{target.name} for fragment {f.fragment_id}")
            for f in self._fragments:
                f.status = targets[f.fragment_id]
            if res.interrupted_fragment_id is None:
                logger.info("[timeline] barge_in with nothing heard → full rollback (crop_token_end=0)")
            else:
                logger.info(f"[timeline] barge_in @sample {pos}: interrupted frag "
                            f"{res.interrupted_fragment_id}, crop_token_end={res.crop_token_end}, "
                            f"partial={res.partial}, heard={res.heard_fragment_ids}, "
                            f"discarded={res.discarded_fragment_ids}")
            return res

    def barge_in_readonly(self, playback_samples: Optional[int] = None) -> BargeInResult:
        """barge_in 的只读版本：只反查、不改任何片段 status（供离线核对/埋点用）。"""
        pos = self._played_samples if playback_samples is None else playback_samples
        with self._lock:
            return self._resolve_barge_in_locked(pos)

    # ----------------------------------------------------------------- 辅助
    def heard_text(self, playback_samples: Optional[int] = None) -> str:
        """返回"用户实际听到"的文本（含被打断片段的完整文本，选 A 语义）。"""
        with self._lock:
            pos = self._played_samples if playback_samples is None else playback_samples
            res = self._resolve_barge_in_locked(pos)
            return "".join(self._by_id[fid].text for fid in res.heard_fragment_ids)

    def get_fragment(self, fragment_id: int) -> FragmentRecord:
        """按 id 取片段记录（供严格 ground-truth 切分、边界注入等反查用）。"""
        with self._lock:
            rec = self._by_id.get(fragment_id)
            if rec is None:
                raise KeyError(f"unknown fragment_id {fragment_id}")
            return rec

    def snapshot(self) -> List[FragmentRecord]:
        """返回当前所有片段记录的浅拷贝列表（供落盘埋点，见 experiment_design.md §6）。"""
        with self._lock:
            return list(self._fragments)
