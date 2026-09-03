"""Descriptive analysis for accepted C2 v3 crop-integrity evidence."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.sci34_supplement.common import atomic_write_json, load_jsonl, sha256_file, utc_now
from experiments.sci34_supplement.c2_crop_integrity.protocol import PRIOR_V2_RUN_ID, PROTOCOL_VERSION
from experiments.sci34_supplement.c2_crop_integrity.validate import validate_campaign


def build_analysis(campaign_dir: Path, *, formal: bool = True) -> dict[str, Any]:
    validation = validate_campaign(campaign_dir, formal=formal)
    if not validation["ok"]:
        raise ValueError("C2 v3 validation failed: " + "; ".join(validation["errors"][:10]))
    records_path = campaign_dir / "records.jsonl"
    records = load_jsonl(records_path)
    events = [event for record in records for event in record["crop_events"]]
    contexts = Counter(record["context_class"] for record in records)
    scenarios = Counter(record["scenario"] for record in records)
    return {
        "schema_version": 1,
        "experiment": "c2_crop_integrity",
        "protocol_version": PROTOCOL_VERSION,
        "created_at_utc": utc_now(),
        "design": {
            "sessions": 1,
            "statistical_repeats": 0,
            "descriptive_only": True,
            "cases": len(records),
            "crop_events": len(events),
            "termination_probe_rerun": False,
            "prior_v2_run_id": PRIOR_V2_RUN_ID,
        },
        "acceptance": {
            "passed": True,
            "failed_cases": 0,
            "failed_crop_events": 0,
            "all_exact_gates": all(validation["exact_gates"].values()),
        },
        "overall": {
            "cases": len(records),
            "case_passes": sum(record["passed"] is True for record in records),
            "crop_events": len(events),
            "crop_event_passes": sum(event["passed"] is True for event in events),
            "no_op_crop_events": sum(event["no_op"] is True for event in events),
            "recovery_steps": sum(len(event["recovery_checks"]) for event in events),
            "second_crop_events": sum(event["event_id"] == "crop_2" for event in events),
            "fixture_tokens": sum(len(record["fixture"]["assistant_token_ids"]) for record in records),
            "tokenwise_prefill_calls": sum(record["fixture"]["prefill_ids_p2_calls"] for record in records),
        },
        "by_context": dict(sorted(contexts.items())),
        "by_scenario": dict(sorted(scenarios.items())),
        "exact_gates": validation["exact_gates"],
        "negative_control": records[0]["formal_negative_control"] if records else None,
        "validation": validation,
        "provenance": {
            "records": {"path": str(records_path.resolve()), "sha256": sha256_file(records_path)},
            "validator": {"path": str((Path(__file__).parent / "validate.py").resolve()), "sha256": sha256_file(Path(__file__).parent / "validate.py")},
            "analyzer": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        },
        "claim_boundary": (
            "This addendum directly proves crop/truncation integrity and matched recovery determinism "
            "for the frozen Qwen2-7B snapshot, dtype, model implementation, and backend: retained K/V "
            "prefixes and every matched recovery step are bitwise equal. It does not establish numerical "
            "equivalence to a clean re-prefill, cross-model correctness, or online ASR/TTS/player correctness."
        ),
        "rejected_descriptive_evidence": (
            "Any v2 clean-prefill numerical comparison remains descriptive prior evidence and is not a v3 gate."
        ),
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create immutable C2 v3 analysis_v1.json")
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--non-formal", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    output = args.out or args.campaign_dir / "analysis_v1.json"
    if output.exists():
        raise FileExistsError(f"Analysis is immutable: {output}")
    atomic_write_json(output, build_analysis(args.campaign_dir, formal=not args.non_formal))
    print(output)


if __name__ == "__main__":
    main()
