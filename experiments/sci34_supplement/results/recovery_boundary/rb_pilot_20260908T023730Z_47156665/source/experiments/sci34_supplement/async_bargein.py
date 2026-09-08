"""P1 prepared-state headless asynchronous playback microbenchmark."""

from __future__ import annotations

import argparse
import logging
import math
import time
from pathlib import Path
from typing import Any

from experiments.sci34_supplement.common import (
    DEFAULT_RESULTS_ROOT,
    append_jsonl,
    atomic_write_json,
    build_manifest,
    load_jsonl,
    prepare_run_directory,
)
from experiments.sci34_supplement.kv_runtime import make_kv_fixture, timed_ms
from experiments.sci34_supplement.paced_player import PacedSamplePlayer
from src.dialogue.timeline import PlaybackTimeline


LOGGER = logging.getLogger(__name__)
PROTOCOL = "async_prepared_v2"
PREPARED_STATE = "full_kv_synchronized_before_playback"
FORMAL_PARTIAL_EXPECTATIONS = {0.25: True, 0.5: False, 0.75: True}


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


def expected_partial(fraction: float) -> bool | None:
    for expected_fraction, expected in FORMAL_PARTIAL_EXPECTATIONS.items():
        if math.isclose(fraction, expected_fraction, rel_tol=0.0, abs_tol=1e-9):
            return expected
    return None


def _validate_args(args: argparse.Namespace) -> None:
    if not args.run_id.endswith(PROTOCOL):
        raise ValueError(
            f"Prepared-state P1 run IDs must end with {PROTOCOL!r}; got {args.run_id!r}. "
            "Use a new run ID so the original async run is never overwritten."
        )
    if args.warmups < 0:
        raise ValueError("--warmups must be non-negative")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.fragments <= 0:
        raise ValueError("--fragments must be positive")
    if any(not 0.0 < fraction < 1.0 for fraction in args.fractions):
        raise ValueError("--fractions values must lie strictly between 0 and 1")
    if args.fragments != 6 and any(expected_partial(fraction) is not None for fraction in args.fractions):
        raise ValueError(
            "The prepared-state P1 0.25/0.5/0.75 protocol requires --fragments 6 so "
            "0.25 and 0.75 are partial while 0.5 is a clean boundary."
        )


def _validate_partial(record: dict[str, Any]) -> None:
    expected = record["partial_expected"]
    if expected is not None and record["partial"] is not expected:
        raise AssertionError(
            "P1 injection geometry violated: "
            f"fraction={record['fraction']} expected partial={expected}, "
            f"observed partial={record['partial']} at played_samples={record['played_at_ack']}"
        )


def run_trial(
    args: argparse.Namespace,
    fixture,
    context_length: int,
    fraction: float,
    repeat: int,
    *,
    trial_kind: str = "formal",
) -> dict[str, Any]:
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

    # Prepared-state barrier: state restoration and all queued device work must
    # finish before playback starts.  setup_ms is deliberately outside every
    # stop/crop latency interval.
    setup_start = time.perf_counter_ns()
    fixture.ensure_full()
    fixture.synchronize()
    setup_done = time.perf_counter_ns()

    player.start()
    player.wait_until(target_samples)
    stop = player.stop()
    player.join()

    if target_samples % player.block_samples:
        raise AssertionError(
            f"P1 target {target_samples} is not aligned to playback block {player.block_samples}"
        )
    if stop.played_at_request != target_samples or stop.played_at_ack != target_samples:
        raise AssertionError(
            "P1 injection request/ack missed the requested sample target: "
            f"target={target_samples}, request={stop.played_at_request}, ack={stop.played_at_ack}. "
            "Use protocol geometry with injection targets aligned to playback blocks."
        )
    if stop.leaked_samples:
        raise AssertionError(f"P1 leaked {stop.leaked_samples} samples after stop request")
    post_stop_sync_start = time.perf_counter_ns()
    fixture.synchronize()
    post_stop_sync_done = time.perf_counter_ns()
    lookup_start = post_stop_sync_done
    boundary = timeline.barge_in(playback_samples=stop.played_at_request)
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

    partial_expected = expected_partial(fraction)
    record = {
        "protocol": PROTOCOL,
        "prepared_state": PREPARED_STATE,
        "prepared_state_synchronized": True,
        "trial_kind": trial_kind,
        "setup_ms": (setup_done - setup_start) / 1_000_000,
        "post_stop_sync_ms": (post_stop_sync_done - post_stop_sync_start) / 1_000_000,
        "stop_to_sync_done_ms": (post_stop_sync_done - stop.request_ns) / 1_000_000,
        "path_kind": "mid_fragment" if boundary.partial else "fragment_boundary",
        "block_samples": player.block_samples,
        "context_length_target": context_length,
        "context_length_actual": fixture.actual_length,
        "assistant_start": fixture.assistant_start,
        "fraction": fraction,
        "repeat": repeat,
        "sample_rate": args.sample_rate,
        "duration_s": args.duration_s,
        "fragments": args.fragments,
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
        "partial_expected": partial_expected,
        "partial_expectation_met": partial_expected is None or boundary.partial is partial_expected,
        "fragment_statuses": [fragment.status.name for fragment in timeline.snapshot()],
        "max_wakeup_error_ms": max(player.wakeup_error_ms, default=0.0),
    }
    _validate_partial(record)
    return record


def _completed_formal_keys(records: list[dict[str, Any]]) -> set[tuple[str, str, str]]:
    completed: set[tuple[str, str, str]] = set()
    for record in records:
        if record.get("protocol") != PROTOCOL:
            raise ValueError("Existing records do not use the prepared-state v2 protocol")
        if record.get("trial_kind") != "formal":
            raise ValueError("Warmup or unknown trial found in formal records.jsonl")
        _validate_partial(record)
        key = tuple(
            str(record[field])
            for field in ("context_length_target", "fraction", "repeat")
        )
        if key in completed:
            raise ValueError(f"Duplicate formal P1 record key: {key}")
        completed.add(key)
    return completed


def run_experiment(args: argparse.Namespace, llm=None) -> Path:
    _validate_args(args)
    config = {
        "protocol": PROTOCOL,
        "prepared_state": PREPARED_STATE,
        "runtime": args.runtime,
        "model": args.model,
        "device": args.device,
        "lengths": args.lengths,
        "fractions": args.fractions,
        "warmups_per_incomplete_cell": args.warmups,
        "warmups_persisted": False,
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
        experiment="async_bargein_control_path_prepared",
        run_id=args.run_id,
        config=config,
        sample_ids=[
            f"{length}|{fraction}|{repeat}"
            for length in args.lengths
            for fraction in args.fractions
            for repeat in range(args.repeats)
        ],
        extra={
            "protocol": PROTOCOL,
            "prepared_state": PREPARED_STATE,
            "formal_partial_expectations": {
                str(fraction): expected
                for fraction, expected in FORMAL_PARTIAL_EXPECTATIONS.items()
            },
            "excluded": [
                "sound-card hardware buffer",
                "online TTS cancellation",
                "ASR/LLM/TTS concurrency",
                "acoustic user-heard latency",
            ],
        },
    )
    run_dir = prepare_run_directory(
        results_root=args.results_root,
        run_id=args.run_id,
        manifest=manifest,
        resume=args.resume,
    )
    records_path = run_dir / "records.jsonl"
    existing_records = load_jsonl(records_path)
    completed = _completed_formal_keys(existing_records)

    if llm is None:
        llm = load_llm(args.runtime, args.model, args.device)
    for length in args.lengths:
        fixture = make_kv_fixture(args.runtime, target_length=length, llm=llm)
        for fraction in args.fractions:
            missing_repeats = [
                repeat
                for repeat in range(args.repeats)
                if (str(length), str(fraction), str(repeat)) not in completed
            ]
            if not missing_repeats:
                continue

            # Resume intentionally warms only cells that will emit at least one
            # missing formal record.  Warmups exercise the complete prepared
            # control path but are never appended to records.jsonl.
            LOGGER.info(
                "warming async cell length=%s fraction=%s warmups=%s missing=%s",
                length,
                fraction,
                args.warmups,
                len(missing_repeats),
            )
            for warmup in range(args.warmups):
                run_trial(
                    args,
                    fixture,
                    length,
                    fraction,
                    repeat=-(warmup + 1),
                    trial_kind="warmup",
                )

            for repeat in missing_repeats:
                record = run_trial(
                    args,
                    fixture,
                    length,
                    fraction,
                    repeat,
                    trial_kind="formal",
                )
                append_jsonl(records_path, record)
                completed.add((str(length), str(fraction), str(repeat)))
                LOGGER.info("async length=%s fraction=%s repeat=%s", length, fraction, repeat)

    records = load_jsonl(records_path)
    completed = _completed_formal_keys(records)
    expected = len(args.lengths) * len(args.fractions) * args.repeats
    if len(records) != expected or len(completed) != expected:
        raise AssertionError(f"Expected {expected} unique formal records, found {len(records)}")
    atomic_write_json(
        run_dir / "run_summary.json",
        {
            "protocol": PROTOCOL,
            "prepared_state": PREPARED_STATE,
            "records": len(records),
            "warmups_per_incomplete_cell": args.warmups,
            "warmups_persisted": False,
            "partial_expectations_validated": all(
                record["partial_expectation_met"] for record in records
            ),
            "prepared_state_validated": all(
                record["prepared_state_synchronized"] for record in records
            ),
            "zero_leakage_validated": all(record["leaked_samples"] == 0 for record in records),
            "exact_target_validated": all(
                record["played_at_request"] == record["target_samples"]
                and record["played_at_ack"] == record["target_samples"]
                for record in records
            ),
            "records_by_path_kind": {
                path_kind: sum(record["path_kind"] == path_kind for record in records)
                for path_kind in ("mid_fragment", "fragment_boundary")
            },
            "scope_note": (
                "Headless wall-clock-paced software playback. It excludes sound-card buffers, "
                "online TTS cancellation, and production end-to-end barge-in latency."
            ),
        },
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT / "async_bargein")
    parser.add_argument("--runtime", choices=("fake", "transformers"), default="transformers")
    parser.add_argument("--model")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lengths", nargs="+", type=int, default=[512, 2048, 8192])
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.25, 0.5, 0.75])
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--duration-s", type=float, default=0.8)
    parser.add_argument("--fragments", type=int, default=6)
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

    run_dir = run_experiment(args)
    print(run_dir)


if __name__ == "__main__":
    main()
