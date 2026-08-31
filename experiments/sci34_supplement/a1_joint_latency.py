"""A1: raw and jointly timed KV crop + role-recovery microbenchmark."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from experiments.sci34_supplement.common import (
    DEFAULT_RESULTS_ROOT,
    append_jsonl,
    atomic_write_json,
    build_manifest,
    completed_keys,
    describe,
    prepare_run_directory,
)
from experiments.sci34_supplement.kv_runtime import make_kv_fixture, timed_ms


LOGGER = logging.getLogger(__name__)


def load_llm(runtime: str, model: str | None, device: str):
    if runtime == "fake":
        return None
    if not model:
        raise ValueError("--model is required for transformers runtime")
    from src.llm.stream_llm_inference import StreamLLMInference

    return StreamLLMInference(model_name=model, device=device, eval_mode=False)


def measure_length(args: argparse.Namespace, llm, target_length: int) -> dict:
    fixture = make_kv_fixture(args.runtime, target_length=target_length, llm=llm)
    keep_length = fixture.actual_length - args.crop_tokens
    if keep_length <= fixture.assistant_start:
        raise ValueError(
            f"Context {fixture.actual_length} is too short for crop_tokens={args.crop_tokens}"
        )

    def reset_to_crop_point() -> None:
        fixture.ensure_full()
        fixture.crop(keep_length)

    for _ in range(args.warmup):
        fixture.ensure_full()
        timed_ms(fixture, lambda: fixture.crop(keep_length))
        fixture.ensure_full()
        timed_ms(
            fixture,
            lambda: (fixture.crop(keep_length), fixture.recover_role()),
        )
        fixture.crop(keep_length)
        fixture.ensure_full()
        timed_ms(fixture, lambda: fixture.reprefill(keep_length))

    crop_only: list[float] = []
    role_only: list[float] = []
    joint: list[float] = []
    reprefill: list[float] = []
    for _ in range(args.repeats):
        fixture.ensure_full()
        crop_only.append(timed_ms(fixture, lambda: fixture.crop(keep_length)))

        reset_to_crop_point()
        role_only.append(timed_ms(fixture, fixture.recover_role))

        fixture.crop(keep_length)
        fixture.ensure_full()
        joint.append(
            timed_ms(
                fixture,
                lambda: (fixture.crop(keep_length), fixture.recover_role()),
            )
        )

        fixture.crop(keep_length)
        fixture.ensure_full()
        reprefill.append(timed_ms(fixture, lambda: fixture.reprefill(keep_length)))

    joint_summary = describe(joint)
    reprefill_summary = describe(reprefill)
    return {
        "target_length": target_length,
        "actual_length": fixture.actual_length,
        "assistant_start": fixture.assistant_start,
        "keep_length": keep_length,
        "crop_tokens": args.crop_tokens,
        "reprefill_target": "retained prefix plus the same assistant-to-user role switch",
        "warmup": args.warmup,
        "repeats": args.repeats,
        "raw": {
            "crop_only_ms": crop_only,
            "role_recovery_only_ms": role_only,
            "crop_role_joint_ms": joint,
            "reprefill_ms": reprefill,
        },
        "statistics": {
            "crop_only_ms": describe(crop_only),
            "role_recovery_only_ms": describe(role_only),
            "crop_role_joint_ms": joint_summary,
            "reprefill_ms": reprefill_summary,
        },
        "speedup_reprefill_over_joint_median": (
            reprefill_summary["median"] / joint_summary["median"]
            if joint_summary["median"]
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT / "a1")
    parser.add_argument("--runtime", choices=("fake", "transformers"), default="transformers")
    parser.add_argument("--model")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lengths", nargs="+", type=int, default=[256, 512, 1024, 2048, 4096, 8192])
    parser.add_argument("--crop-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
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
        "crop_tokens": args.crop_tokens,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "timing": "perf_counter_ns with pre/post device synchronization",
        "model_identity": __import__(
            "experiments.sci34_supplement.common", fromlist=["model_identity"]
        ).model_identity(args.model),
    }
    manifest = build_manifest(
        experiment="a1_joint_latency",
        run_id=args.run_id,
        config=config,
        sample_ids=[str(length) for length in args.lengths],
    )
    run_dir = prepare_run_directory(
        results_root=args.results_root,
        run_id=args.run_id,
        manifest=manifest,
        resume=args.resume,
    )
    records_path = run_dir / "records.jsonl"
    completed = completed_keys(records_path, ("target_length",))
    llm = load_llm(args.runtime, args.model, args.device)
    for target_length in args.lengths:
        if (str(target_length),) in completed:
            continue
        record = measure_length(args, llm, target_length)
        append_jsonl(records_path, record)
        completed.add((str(target_length),))
        LOGGER.info("A1 complete: target=%s actual=%s", target_length, record["actual_length"])
    records = __import__(
        "experiments.sci34_supplement.common", fromlist=["load_jsonl"]
    ).load_jsonl(records_path)
    if any(len(row["raw"]["crop_role_joint_ms"]) != args.repeats for row in records):
        raise AssertionError("A1 raw repeat count mismatch")
    atomic_write_json(
        run_dir / "summary.json",
        {
            "rows": [
                {
                    "target_length": row["target_length"],
                    "actual_length": row["actual_length"],
                    "keep_length": row["keep_length"],
                    "statistics": row["statistics"],
                    "speedup_reprefill_over_joint_median": row[
                        "speedup_reprefill_over_joint_median"
                    ],
                }
                for row in records
            ],
            "scope_note": (
                "The primary denominator is the median of a jointly timed crop + role-recovery "
                "path. This is a model-side microbenchmark, not complete barge-in latency."
            ),
        },
    )
    print(run_dir)


if __name__ == "__main__":
    main()
