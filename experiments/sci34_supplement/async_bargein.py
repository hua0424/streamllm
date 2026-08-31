"""P1 headless asynchronous playback and state-repair microbenchmark."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from experiments.sci34_supplement.common import (
    DEFAULT_RESULTS_ROOT,
    append_jsonl,
    atomic_write_json,
    build_manifest,
    completed_keys,
    load_jsonl,
    prepare_run_directory,
)
from experiments.sci34_supplement.kv_runtime import make_kv_fixture, timed_ms
from experiments.sci34_supplement.paced_player import PacedSamplePlayer
from src.dialogue.timeline import PlaybackTimeline


LOGGER = logging.getLogger(__name__)


def load_llm(runtime: str, model: str | None, device: str):
    if runtime == "fake":
        return None
    if not model:
        raise ValueError("--model is required for transformers runtime")
    from src.llm.stream_llm_inference import StreamLLMInference

    return StreamLLMInference(model_name=model, device=device, eval_mode=False)


def build_timeline(
    *,
    token_count: int,
    sample_rate: int,
    duration_s: float,
    fragments: int,
) -> PlaybackTimeline:
    timeline = PlaybackTimeline()
    sample_total = round(sample_rate * duration_s)
    token_cursor = sample_cursor = 0
    for index in range(fragments):
        token_end = token_count if index == fragments - 1 else round(token_count * (index + 1) / fragments)
        sample_end = sample_total if index == fragments - 1 else round(sample_total * (index + 1) / fragments)
        fragment_id = timeline.add_fragment(
            f"fragment-{index}", token_cursor, token_end
        )
        timeline.attach_chunk(fragment_id, index, sample_end - sample_cursor)
        token_cursor = token_end
        sample_cursor = sample_end
    return timeline


def run_trial(args: argparse.Namespace, fixture, context_length: int, fraction: float, repeat: int) -> dict:
    token_count = max(4, fixture.actual_length - fixture.assistant_start)
    timeline = build_timeline(
        token_count=token_count,
        sample_rate=args.sample_rate,
        duration_s=args.duration_s,
        fragments=args.fragments,
    )
    target_samples = max(1, int(timeline.total_samples * fraction))
    player = PacedSamplePlayer(
        timeline,
        total_samples=timeline.total_samples,
        sample_rate=args.sample_rate,
        block_ms=args.block_ms,
        time_scale=args.time_scale,
    )
    fixture.ensure_full()
    player.start()
    player.wait_until(target_samples)
    stop = player.stop()
    player.join()

    fixture.synchronize()
    lookup_start = time.perf_counter_ns()
    boundary = timeline.barge_in()
    lookup_done = time.perf_counter_ns()
    keep_length = min(
        fixture.actual_length,
        fixture.assistant_start + boundary.crop_token_end,
    )
    fixture.crop(keep_length)
    fixture.synchronize()
    crop_done = time.perf_counter_ns()
    fixture.recover_role()
    fixture.synchronize()
    role_done = time.perf_counter_ns()
    player.verify_stable_after_stop()

    fixture.crop(keep_length)
    fixture.ensure_full()
    crop_ms = timed_ms(fixture, lambda: fixture.crop(keep_length))
    role_ms = timed_ms(fixture, fixture.recover_role)
    fixture.crop(keep_length)
    return {
        "context_length_target": context_length,
        "context_length_actual": fixture.actual_length,
        "assistant_start": fixture.assistant_start,
        "fraction": fraction,
        "repeat": repeat,
        "sample_rate": args.sample_rate,
        "duration_s": args.duration_s,
        "block_ms": args.block_ms,
        "time_scale": args.time_scale,
        "target_samples": target_samples,
        "played_at_request": stop.played_at_request,
        "played_at_ack": stop.played_at_ack,
        "stop_ack_ms": stop.latency_ms,
        "leaked_samples": stop.leaked_samples,
        "leaked_ms": stop.leaked_samples / args.sample_rate * 1000,
        "lookup_ms": (lookup_done - lookup_start) / 1_000_000,
        "crop_only_ms": crop_ms,
        "role_recovery_only_ms": role_ms,
        "joint_crop_ms": (crop_done - lookup_done) / 1_000_000,
        "joint_role_recovery_ms": (role_done - crop_done) / 1_000_000,
        "stop_to_crop_done_ms": (crop_done - stop.request_ns) / 1_000_000,
        "stop_to_role_done_ms": (role_done - stop.request_ns) / 1_000_000,
        "interrupted_fragment_id": boundary.interrupted_fragment_id,
        "crop_token_end": boundary.crop_token_end,
        "keep_length": keep_length,
        "partial": boundary.partial,
        "fragment_statuses": [fragment.status.name for fragment in timeline.snapshot()],
        "max_wakeup_error_ms": max(player.wakeup_error_ms, default=0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT / "async_bargein")
    parser.add_argument("--runtime", choices=("fake", "transformers"), default="transformers")
    parser.add_argument("--model")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lengths", nargs="+", type=int, default=[512, 2048, 8192])
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.25, 0.5, 0.75])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--duration-s", type=float, default=0.8)
    parser.add_argument("--fragments", type=int, default=4)
    parser.add_argument("--block-ms", type=float, default=20.0)
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    if args.runtime == "transformers":
        from experiments.sci34_supplement.common import enforce_offline_mode, require_clean_tree

        require_clean_tree(allow_dirty=args.allow_dirty)
        enforce_offline_mode()

    config = {
        "runtime": args.runtime,
        "model": args.model,
        "device": args.device,
        "lengths": args.lengths,
        "fractions": args.fractions,
        "repeats": args.repeats,
        "sample_rate": args.sample_rate,
        "duration_s": args.duration_s,
        "fragments": args.fragments,
        "block_ms": args.block_ms,
        "time_scale": args.time_scale,
        "player": "headless-wall-clock-paced",
        "model_identity": __import__(
            "experiments.sci34_supplement.common", fromlist=["model_identity"]
        ).model_identity(args.model),
    }
    manifest = build_manifest(
        experiment="async_bargein_control_path",
        run_id=args.run_id,
        config=config,
        sample_ids=[
            f"{length}|{fraction}|{repeat}"
            for length in args.lengths
            for fraction in args.fractions
            for repeat in range(args.repeats)
        ],
        extra={
            "excluded": [
                "sound-card hardware buffer",
                "online TTS cancellation",
                "ASR/LLM/TTS concurrency",
                "acoustic user-heard latency",
            ]
        },
    )
    run_dir = prepare_run_directory(
        results_root=args.results_root,
        run_id=args.run_id,
        manifest=manifest,
        resume=args.resume,
    )
    records_path = run_dir / "records.jsonl"
    completed = completed_keys(records_path, ("context_length_target", "fraction", "repeat"))
    llm = load_llm(args.runtime, args.model, args.device)
    for length in args.lengths:
        fixture = make_kv_fixture(args.runtime, target_length=length, llm=llm)
        for fraction in args.fractions:
            for repeat in range(args.repeats):
                key = (str(length), str(fraction), str(repeat))
                if key in completed:
                    continue
                record = run_trial(args, fixture, length, fraction, repeat)
                append_jsonl(records_path, record)
                completed.add(key)
                LOGGER.info("async length=%s fraction=%s repeat=%s", length, fraction, repeat)
    records = load_jsonl(records_path)
    expected = len(args.lengths) * len(args.fractions) * args.repeats
    if len(records) != expected:
        raise AssertionError(f"Expected {expected} records, found {len(records)}")
    atomic_write_json(
        run_dir / "run_summary.json",
        {
            "records": len(records),
            "scope_note": (
                "Headless wall-clock-paced software playback. It excludes sound-card buffers, "
                "online TTS cancellation, and production end-to-end barge-in latency."
            ),
        },
    )
    print(run_dir)


if __name__ == "__main__":
    main()
