"""Versioned descriptive analysis for the deterministic C2 correctness campaign."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping

from experiments.sci34_supplement.common import atomic_write_json, load_jsonl, sha256_file, utc_now
from experiments.sci34_supplement.c2_equivalence.validate import validate_campaign


ANALYSIS_SCHEMA_VERSION = 1


def _group_summary(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    checkpoints = [checkpoint for row in rows for checkpoint in row.get("checkpoints", [])]
    max_values = [float(item["logit_diff_float32"]["max_abs"]) for item in checkpoints]
    mean_values = [float(item["logit_diff_float32"]["mean_abs"]) for item in checkpoints]
    rms_values = [float(item["logit_diff_float32"]["rms"]) for item in checkpoints]
    overlaps = [int(item["next_token"]["top5_overlap"]) for item in checkpoints]
    control_max = [float(item["noise_control"]["max_abs"]) for item in checkpoints]
    control_mean = [float(item["noise_control"]["mean_abs"]) for item in checkpoints]
    ratios = [
        float(item["logit_diff_float32"]["max_abs"]) / max(control, 1e-12)
        for item, control in zip(checkpoints, control_max)
    ]
    flips = [item for item in checkpoints if item["next_token"].get("top1_flip_near_tie")]
    return {
        "cases": len(rows),
        "checkpoints": len(checkpoints),
        "case_passes": sum(bool(row.get("passed")) for row in rows),
        "checkpoint_passes": sum(bool(item.get("passed")) for item in checkpoints),
        "token_exact_rate": (
            sum(bool(item.get("token_ids_exact")) for item in checkpoints) / len(checkpoints)
            if checkpoints else None
        ),
        "top1_exact_rate": (
            sum(bool(item.get("next_token", {}).get("top1_exact")) for item in checkpoints)
            / len(checkpoints) if checkpoints else None
        ),
        "top1_near_tie_flips": len(flips),
        "continuation_exact_rate": (
            sum(bool(item.get("continuation", {}).get("exact")) for item in checkpoints)
            / len(checkpoints) if checkpoints else None
        ),
        "top5_overlap": {
            "min": min(overlaps) if overlaps else None,
            "mean": mean(overlaps) if overlaps else None,
            "max": max(overlaps) if overlaps else None,
        },
        "float32_logit_difference": {
            "max_abs_worst": max(max_values) if max_values else None,
            "mean_abs_worst": max(mean_values) if mean_values else None,
            "mean_abs_across_checkpoints": mean(mean_values) if mean_values else None,
            "rms_worst": max(rms_values) if rms_values else None,
        },
        "noise_control": {
            "max_abs_worst": max(control_max) if control_max else None,
            "max_abs_median": sorted(control_max)[len(control_max) // 2] if control_max else None,
            "mean_abs_worst": max(control_mean) if control_mean else None,
            "path_over_control_ratio_worst": max(ratios) if ratios else None,
        },
    }


def _group(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record[field])].append(record)
    return {key: _group_summary(rows) for key, rows in sorted(grouped.items())}


def build_analysis(campaign_dir: Path, *, formal: bool = True) -> dict[str, Any]:
    validation = validate_campaign(campaign_dir, formal=formal)
    if not validation["ok"]:
        raise ValueError("C2 validation failed: " + "; ".join(validation["errors"][:10]))
    records_path = campaign_dir / "records.jsonl"
    records = load_jsonl(records_path)
    checkpoint_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    worst: list[dict[str, Any]] = []
    failed_indexes: list[dict[str, Any]] = []
    for record in records:
        if not record.get("passed"):
            failed_indexes.append({"case_id": record.get("case_id"), "errors": record.get("errors", [])})
        for checkpoint in record.get("checkpoints", []):
            checkpoint_groups[str(checkpoint["checkpoint"])].append(
                {**record, "checkpoints": [checkpoint]}
            )
            worst.append(
                {
                    "case_id": record["case_id"],
                    "checkpoint": checkpoint["checkpoint"],
                    "context_class": record["context_class"],
                    "scenario": record["scenario"],
                    "termination": record["termination"],
                    "max_abs": float(checkpoint["logit_diff_float32"]["max_abs"]),
                    "mean_abs": float(checkpoint["logit_diff_float32"]["mean_abs"]),
                    "rms": float(checkpoint["logit_diff_float32"]["rms"]),
                    "control_max_abs": float(checkpoint["noise_control"]["max_abs"]),
                    "control_mean_abs": float(checkpoint["noise_control"]["mean_abs"]),
                    "noise_max_limit": float(checkpoint["logit_gates"]["noise_max_limit"]),
                    "near_tie_margin_limit": float(checkpoint["logit_gates"]["near_tie_margin_limit"]),
                    "canonical_top1_top2_margin": float(
                        checkpoint["next_token"]["canonical_top1_top2_margin"]
                    ),
                    "top1_flip_near_tie": bool(checkpoint["next_token"].get("top1_flip_near_tie")),
                    "top5_overlap": int(checkpoint["next_token"]["top5_overlap"]),
                    "first_token_mismatch": checkpoint.get("first_token_mismatch"),
                    "first_continuation_divergence": checkpoint.get("continuation", {}).get("first_divergence"),
                    "passed": bool(checkpoint.get("passed")),
                }
            )
    worst.sort(key=lambda item: (item["max_abs"], item["mean_abs"], -item["top5_overlap"]), reverse=True)
    termination_probes = [record["termination_probe"] for record in records]
    termination_summary: dict[str, Any] = {}
    for label in sorted({str(record["termination"]) for record in records}):
        grouped = [
            record["termination_probe"]
            for record in records
            if record["termination"] == label
        ]
        termination_summary[label] = {
            "cases": len(grouped),
            "qualified": sum(bool(probe.get("passed")) for probe in grouped),
            "genuine_eos": sum(bool(probe.get("genuine_eos")) for probe in grouped),
            "requalified": sum(bool(probe.get("requalified")) for probe in grouped),
            "observed_end_reasons": dict(sorted({
                reason: sum(probe.get("observed_end_reason") == reason for probe in grouped)
                for reason in {str(probe.get("observed_end_reason")) for probe in grouped}
            }.items())),
            "modes": dict(sorted({
                mode: sum(probe.get("mode") == mode for probe in grouped)
                for mode in {str(probe.get("mode")) for probe in grouped}
            }.items())),
            "caps": sorted({int(probe["cap"]) for probe in grouped}),
            "content_token_counts": [int(probe["content_token_count"]) for probe in grouped],
            "eos_steps": [probe.get("eos_step") for probe in grouped],
            "eot_in_kv_count": sum(bool(probe.get("eot_in_kv")) for probe in grouped),
            "eot_in_content_ledger_count": sum(
                bool(probe.get("eot_in_content_ledger")) for probe in grouped
            ),
        }
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "experiment": "c2_equivalence",
        "protocol_version": 2,
        "created_at_utc": utc_now(),
        "design": {
            "sessions": 1,
            "statistical_repeats": 0,
            "descriptive_only": True,
            "bootstrap": None,
            "comparison": "independent termination qualification plus retained-token crop/recovery versus canonical token-ID clean re-prefill",
            "noise_control_arm": (
                "canonical sequence re-prefilled incrementally at structural seams versus single-shot; "
                "crop-path deviation is gated relative to this measured intrinsic BF16 noise"
            ),
        },
        "acceptance": {
            "passed": True,
            "failed_case_count": 0,
            "failed_checkpoint_count": 0,
            "failed_termination_probe_count": 0,
            "failed_indexes": failed_indexes,
        },
        "noise_control": {
            "arm": "canonical_ids_boundary_seam_chunked_prefill",
            "note": (
                "path/control ratio near 1 with token/state exactness indicates the crop/recovery "
                "path adds no numerical deviation beyond intrinsic incremental-append BF16 noise"
            ),
            **(_group_summary(records)["noise_control"] if records else {}),
        },
        "termination_probes": {
            "cases": len(termination_probes),
            "qualified": sum(bool(probe.get("passed")) for probe in termination_probes),
            "by_declared_label": termination_summary,
        },
        "scenario_execution": {
            "cases": len(records),
            "applicable": sum(
                bool(record.get("scenario_execution", {}).get("applies"))
                for record in records
            ),
            "qualified": sum(
                bool(record.get("scenario_execution", {}).get("passed"))
                for record in records
            ),
            "crop_pending_eot": sum(
                record.get("scenario") == "crop_pending_eot"
                and record.get("scenario_execution", {}).get("pending_cleared_by_crop") is True
                for record in records
            ),
            "reply_tail_noop": sum(
                record.get("scenario") == "reply_tail_noop"
                and record.get("scenario_execution", {}).get("no_op_preserved_pending") is True
                for record in records
            ),
        },
        "overall": _group_summary(records),
        "by_context": _group(records, "context_class"),
        "by_scenario": _group(records, "scenario"),
        "by_termination": _group(records, "termination"),
        "by_checkpoint": {
            key: _group_summary(rows) for key, rows in sorted(checkpoint_groups.items())
        },
        "worst_cases": worst[:10],
        "all_failure_indexes": validation["failed_indexes"],
        "validation": validation,
        "provenance": {
            "records": {"path": str(records_path.resolve()), "sha256": sha256_file(records_path)},
            "validation_logic": {
                "path": str((Path(__file__).resolve().parent / "validate.py").resolve()),
                "sha256": sha256_file(Path(__file__).resolve().parent / "validate.py"),
            },
            "analyzer": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        },
        "claim_boundary": (
            "Evidence is limited to deterministic Qwen2-7B BF16 model-state equivalence under the frozen "
            "cases and implementation: token/state ledgers are exact, and logit distributions agree within "
            "the environment's intrinsic incremental-append BF16 noise as measured by the frozen control arm "
            "(near-tie top-1 flips and greedy-rollout divergences are allowed only at bounded margins). It is "
            "not a latency, quality, cross-model, or production claim."
        ),
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create immutable descriptive C2 analysis_v1.json.")
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--non-formal", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    output = args.out or args.campaign_dir / "analysis_v1.json"
    if output.exists():
        raise FileExistsError(f"Analysis is immutable and already exists: {output}")
    result = build_analysis(args.campaign_dir, formal=not args.non_formal)
    atomic_write_json(output, result)
    print(output)


if __name__ == "__main__":
    main()
