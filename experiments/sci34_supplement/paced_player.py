"""Headless wall-clock paced sample consumer used by the P1 microbenchmark."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from src.dialogue.timeline import PlaybackTimeline


@dataclass(frozen=True)
class StopResult:
    request_ns: int
    acknowledged_ns: int
    played_at_request: int
    played_at_ack: int

    @property
    def latency_ms(self) -> float:
        return (self.acknowledged_ns - self.request_ns) / 1_000_000

    @property
    def leaked_samples(self) -> int:
        return self.played_at_ack - self.played_at_request


class PacedSamplePlayer:
    """Advance ``played_samples`` at real or accelerated wall-clock pace.

    TTS chunks remain mapping units.  Playback is advanced in smaller blocks so
    stop behavior is not quantized by the default 500 ms Mock-TTS chunk.
    """

    def __init__(
        self,
        timeline: PlaybackTimeline,
        *,
        total_samples: int,
        sample_rate: int,
        block_ms: float = 20.0,
        time_scale: float = 1.0,
    ):
        if total_samples <= 0 or sample_rate <= 0 or block_ms <= 0 or time_scale <= 0:
            raise ValueError("Player parameters must be positive")
        self.timeline = timeline
        self.total_samples = int(total_samples)
        self.sample_rate = int(sample_rate)
        self.block_samples = max(1, round(sample_rate * block_ms / 1000))
        self.time_scale = float(time_scale)
        self._stop = threading.Event()
        self._started = threading.Event()
        self._stopped = threading.Event()
        self._progress = threading.Condition()
        self._thread: threading.Thread | None = None
        self.start_ns: int | None = None
        self.stop_ns: int | None = None
        self.wakeup_error_ms: list[float] = []

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            raise RuntimeError("Player already running")
        self._stop.clear()
        self._started.clear()
        self._stopped.clear()
        self._thread = threading.Thread(target=self._run, name="paced-sample-player", daemon=True)
        self._thread.start()
        if not self._started.wait(timeout=5):
            raise TimeoutError("Player did not start")

    def _run(self) -> None:
        self.start_ns = time.perf_counter_ns()
        self._started.set()
        played = 0
        while played < self.total_samples and not self._stop.is_set():
            next_played = min(self.total_samples, played + self.block_samples)
            elapsed_s = next_played / self.sample_rate / self.time_scale
            deadline_ns = self.start_ns + int(elapsed_s * 1_000_000_000)
            remaining_s = max(0.0, (deadline_ns - time.perf_counter_ns()) / 1_000_000_000)
            if self._stop.wait(timeout=remaining_s):
                break
            actual_ns = time.perf_counter_ns()
            self.wakeup_error_ms.append((actual_ns - deadline_ns) / 1_000_000)
            played = next_played
            self.timeline.set_played(played)
            with self._progress:
                self._progress.notify_all()
        self.stop_ns = time.perf_counter_ns()
        self._stopped.set()
        with self._progress:
            self._progress.notify_all()

    def wait_until(self, target_samples: int, timeout: float = 30.0) -> None:
        deadline = time.perf_counter() + timeout
        with self._progress:
            while self.timeline.played_samples < target_samples:
                if self._stopped.is_set():
                    break
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Playback did not reach {target_samples}; current={self.timeline.played_samples}"
                    )
                self._progress.wait(timeout=remaining)

    def stop(self) -> StopResult:
        request_ns = time.perf_counter_ns()
        played_at_request = self.timeline.played_samples
        self._stop.set()
        if not self._stopped.wait(timeout=5):
            raise TimeoutError("Player did not acknowledge stop")
        assert self.stop_ns is not None
        return StopResult(
            request_ns=request_ns,
            acknowledged_ns=self.stop_ns,
            played_at_request=played_at_request,
            played_at_ack=self.timeline.played_samples,
        )

    def join(self) -> None:
        if self._thread:
            self._thread.join(timeout=5)

    def verify_stable_after_stop(self) -> None:
        before = self.timeline.played_samples
        time.sleep(self.block_samples / self.sample_rate / self.time_scale * 2)
        after = self.timeline.played_samples
        if after != before:
            raise AssertionError(f"played_samples advanced after stop: {before} -> {after}")
