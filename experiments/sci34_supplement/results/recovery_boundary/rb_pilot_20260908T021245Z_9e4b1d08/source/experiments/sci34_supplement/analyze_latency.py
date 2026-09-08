"""Descriptive analysis for A1 and headless asynchronous latency runs."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from experiments.sci34_supplement.common import atomic_write_json, describe, load_jsonl


def analyze_a1(records: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for record in sorted(records, key=lambda row: row["actual_length"]):
        raw = record["raw"]
        computed = {name: describe(values) for name, values in raw.items()}
        if computed != record["statistics"]:
            raise ValueError(f"Stored A1 statistics do not match raw values at {record['actual_length']}")
        rows.append(
            {
                "target_length": record["target_length"],
                "actual_length": record["actual_length"],
                "keep_length": record["keep_length"],
                "statistics": computed,
                "speedup_reprefill_over_joint_median": (
                    computed["reprefill_ms"]["median"]
                    / computed["crop_role_joint_ms"]["median"]
                ),
            }
        )
    return {
        "experiment": "a1_joint_latency",
        "rows": rows,
        "scope_note": "Synchronized model-side wall-clock microbenchmark; no playback stop is included.",
    }


def analyze_async(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[int, float], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["context_length_target"], record["fraction"])].append(record)
    metrics = (
        "setup_ms",
        "stop_ack_ms",
        "post_stop_sync_ms",
        "stop_to_sync_done_ms",
        "leaked_ms",
        "lookup_ms",
        "joint_crop_ms",
        "joint_role_recovery_ms",
        "crop_only_ms",
        "role_recovery_only_ms",
        "stop_to_crop_done_ms",
        "stop_to_role_done_ms",
        "max_wakeup_error_ms",
    )
    cell_sizes = {len(subset) for subset in grouped.values()}
    if len(cell_sizes) != 1:
        raise ValueError(f"Async cells have inconsistent record counts: {sorted(cell_sizes)}")
    rows = []
    for (length, fraction), subset in sorted(grouped.items()):
        protocols = {record.get("protocol") for record in subset}
        path_kinds = {record.get("path_kind") for record in subset}
        if protocols != {"async_prepared_v2"}:
            raise ValueError(f"Unexpected async protocol at {length}/{fraction}: {protocols}")
        if len(path_kinds) != 1 or None in path_kinds:
            raise ValueError(f"Mixed or missing path kind at {length}/{fraction}: {path_kinds}")
        if any(not record.get("prepared_state_synchronized") for record in subset):
            raise ValueError(f"Unprepared state found at {length}/{fraction}")
        if any(record["leaked_samples"] != 0 for record in subset):
            raise ValueError(f"Sample leakage found at {length}/{fraction}")
        if any(
            record["played_at_request"] != record["target_samples"]
            or record["played_at_ack"] != record["target_samples"]
            for record in subset
        ):
            raise ValueError(f"Playback target mismatch at {length}/{fraction}")
        rows.append(
            {
                "context_length_target": length,
                "fraction": fraction,
                "path_kind": next(iter(path_kinds)),
                "n": len(subset),
                "statistics": {
                    metric: describe([record[metric] for record in subset])
                    for metric in metrics
                },
                "partial_rate": sum(bool(record["partial"]) for record in subset) / len(subset),
                "expected_partial_rate": sum(bool(record["partial_expected"]) for record in subset)
                / len(subset),
                "crop_token_ends": sorted({record["crop_token_end"] for record in subset}),
            }
        )
    return {
        "experiment": "async_bargein_control_path_prepared",
        "protocol": "async_prepared_v2",
        "records_per_cell": next(iter(cell_sizes)),
        "rows": rows,
        "scope_note": (
            "Headless wall-clock-paced software playback. Values are not acoustic stop latency "
            "and do not include online TTS cancellation or ASR/LLM/TTS competition."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--kind", choices=("a1", "async"), required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    records = load_jsonl(args.run_dir / "records.jsonl")
    if not records:
        raise SystemExit(f"No records in {args.run_dir}")
    result = analyze_a1(records) if args.kind == "a1" else analyze_async(records)
    output = args.out or args.run_dir / "analysis.json"
    atomic_write_json(output, result)
    print(output)


if __name__ == "__main__":
    main()
