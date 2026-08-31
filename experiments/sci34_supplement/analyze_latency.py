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
        "stop_ack_ms",
        "leaked_ms",
        "lookup_ms",
        "crop_only_ms",
        "role_recovery_only_ms",
        "stop_to_crop_done_ms",
        "stop_to_role_done_ms",
        "max_wakeup_error_ms",
    )
    rows = []
    for (length, fraction), subset in sorted(grouped.items()):
        rows.append(
            {
                "context_length_target": length,
                "fraction": fraction,
                "n": len(subset),
                "statistics": {
                    metric: describe([record[metric] for record in subset])
                    for metric in metrics
                },
                "partial_rate": sum(bool(record["partial"]) for record in subset) / len(subset),
                "crop_token_ends": sorted({record["crop_token_end"] for record in subset}),
            }
        )
    return {
        "experiment": "async_bargein_control_path",
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
