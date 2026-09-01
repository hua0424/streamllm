"""Paired and dialogue-clustered analysis for fixed-trajectory E3."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from scipy.stats import binomtest

from experiments.sci34_supplement.common import (
    atomic_write_json,
    config_hash,
    load_jsonl,
    sha256_file,
    utc_now,
)


FRACTIONS = (0.25, 0.5, 0.75, "boundary")
CONDITIONS = ("playback", "generation")
TARGETS = {
    "fragment": {
        "target_field": "unheard_text",
        "rule_field": "referenced_unheard",
    },
    "proxy": {
        "target_field": "strict_unheard_text",
        "rule_field": "referenced_unheard_strict",
    },
}
ANALYSIS_VERSION = "metric-specific-eligibility-v1"


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if not total:
        return [math.nan, math.nan]
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, centre - half), min(1.0, centre + half)]


def exact_mcnemar(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    pairs: dict[tuple[str, str], dict[str, bool]] = defaultdict(dict)
    for record in records:
        pairs[(record["id"], str(record["fraction"]))][record["condition"]] = bool(record[field])
    playback_only = generation_only = 0
    for pair in pairs.values():
        if set(pair) != {"playback", "generation"}:
            raise ValueError(f"Incomplete pair: {pair}")
        playback_only += pair["playback"] and not pair["generation"]
        generation_only += pair["generation"] and not pair["playback"]
    discordant = playback_only + generation_only
    p_value = (
        float(binomtest(min(playback_only, generation_only), discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    return {
        "pairs": len(pairs),
        "playback_only": playback_only,
        "generation_only": generation_only,
        "exact_p": p_value,
    }


def cluster_bootstrap(
    records: list[dict[str, Any]], field: str, *, repeats: int, seed: int
) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[bool]]] = defaultdict(
        lambda: {"playback": [], "generation": []}
    )
    for record in records:
        grouped[record["id"]][record["condition"]].append(bool(record[field]))
    dialogue_ids = sorted(grouped)
    playback = np.asarray([mean(grouped[item]["playback"]) for item in dialogue_ids])
    generation = np.asarray([mean(grouped[item]["generation"]) for item in dialogue_ids])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(dialogue_ids), size=(repeats, len(dialogue_ids)))
    difference = generation[indices].mean(axis=1) - playback[indices].mean(axis=1)
    low, high = np.quantile(difference, [0.025, 0.975])
    return {
        "clusters": len(dialogue_ids),
        "generation_minus_playback": float((generation - playback).mean()),
        "difference_95_ci": [float(low), float(high)],
        "bootstrap_repeats": repeats,
        "seed": seed,
    }


def attach_judgments(
    records: list[dict[str, Any]], judge_path: Path | None
) -> list[dict[str, Any]]:
    if not judge_path:
        return []
    judged = load_jsonl(judge_path)
    if any(not row.get("parse_success") for row in judged):
        raise ValueError("Judge records contain parse failures")
    lookup = {
        (row["id"], str(row["fraction"]), row["condition"], row["target_kind"]): row
        for row in judged
    }
    if len(lookup) != len(judged):
        raise ValueError("Judge records contain duplicate keys")
    expected = {
        (record["id"], str(record["fraction"]), record["condition"], target_kind)
        for record in records
        for target_kind in TARGETS
    }
    if set(lookup) != expected:
        missing = sorted(expected - set(lookup))
        extra = sorted(set(lookup) - expected)
        raise ValueError(f"Judge key mismatch: missing={missing[:3]}, extra={extra[:3]}")
    for record in records:
        for target_kind in TARGETS:
            key = (record["id"], str(record["fraction"]), record["condition"], target_kind)
            record[f"judge_{target_kind}"] = bool(lookup[key]["verdict"])
    return judged


def summarize_metric(
    records: list[dict[str, Any]], field: str, *, repeats: int, seed: int
) -> dict[str, Any]:
    conditions = {}
    for condition in CONDITIONS:
        subset = [record for record in records if record["condition"] == condition]
        successes = sum(bool(record[field]) for record in subset)
        conditions[condition] = {
            "n": len(subset),
            "positive": successes,
            "rate": successes / len(subset) if subset else 0.0,
            "wilson_95_ci": wilson(successes, len(subset)),
        }
    return {
        "conditions": conditions,
        "mcnemar": exact_mcnemar(records, field),
        "dialogue_cluster_bootstrap": cluster_bootstrap(
            records, field, repeats=repeats, seed=seed
        ),
    }


def pair_records(
    records: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, dict[str, Any]]]:
    pairs: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        key = (record["id"], str(record["fraction"]))
        condition = record["condition"]
        if condition not in CONDITIONS:
            raise ValueError(f"Unknown condition: {condition}")
        if condition in pairs[key]:
            raise ValueError(f"Duplicate pair condition: {key + (condition,)}")
        pairs[key][condition] = record
    for key, pair in pairs.items():
        if set(pair) != set(CONDITIONS):
            raise ValueError(f"Incomplete pair {key}: {sorted(pair)}")
    return dict(pairs)


def validate_design(records: list[dict[str, Any]]) -> dict[str, int]:
    if not records:
        raise ValueError("No E3 records")
    by_dialogue: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if str(record["id"]).lower().startswith("fx"):
            raise ValueError("Formal analysis contains fixture records")
        by_dialogue[record["id"]].append(record)
    pairs = pair_records(records)
    for dialogue_id, rows in by_dialogue.items():
        if len({row["trajectory_id"] for row in rows}) != 1:
            raise ValueError(f"{dialogue_id}: conditions do not share a trajectory")
        if len({row["history_key"] for row in rows if row["condition"] == "generation"}) != 1:
            raise ValueError(f"{dialogue_id}: generation history differs by fraction")
    for key, pair in pairs.items():
        for target_kind, spec in TARGETS.items():
            target_field = spec["target_field"]
            playback_target = pair["playback"][target_field]
            generation_target = pair["generation"][target_field]
            if playback_target != generation_target:
                raise ValueError(
                    f"Pair conditions disagree on exact {target_kind} target "
                    f"{target_field}: {key}"
                )
    return {
        "dialogues": len(by_dialogue),
        "records": len(records),
        "pairs": len(pairs),
    }


def eligible_records_by_target(
    records: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    pairs = pair_records(records)
    eligible: dict[str, list[dict[str, Any]]] = {}
    eligibility: dict[str, dict[str, Any]] = {}
    for target_kind, spec in TARGETS.items():
        target_field = spec["target_field"]
        eligible_keys = {
            key
            for key, pair in pairs.items()
            if pair["playback"][target_field].strip()
        }
        eligible[target_kind] = [
            record
            for record in records
            if (record["id"], str(record["fraction"])) in eligible_keys
        ]
        by_fraction = {
            str(fraction): {
                "total_pairs": sum(key[1] == str(fraction) for key in pairs),
                "eligible_pairs": sum(key[1] == str(fraction) for key in eligible_keys),
            }
            for fraction in FRACTIONS
        }
        for counts in by_fraction.values():
            counts["empty_target_pairs"] = counts["total_pairs"] - counts["eligible_pairs"]
        eligibility[target_kind] = {
            "target_field": target_field,
            "definition": f"nonblank {target_field}",
            "total_pairs": len(pairs),
            "eligible_pairs": len(eligible_keys),
            "empty_target_pairs": len(pairs) - len(eligible_keys),
            "by_fraction": by_fraction,
        }
    return eligible, eligibility


def load_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_provenance(
    *,
    e3_run_dir: Path,
    records_path: Path,
    judge_path: Path | None,
    judge_records: list[dict[str, Any]],
    bootstrap_repeats: int,
    seed: int,
) -> dict[str, Any]:
    analyzer_path = Path(__file__).resolve()
    e3_manifest_path = e3_run_dir / "manifest.json"
    judge_manifest_path = judge_path.parent / "manifest.json" if judge_path else None
    e3_manifest = load_manifest(e3_manifest_path)
    judge_manifest = load_manifest(judge_manifest_path) if judge_manifest_path else None
    parameters = {
        "analysis_version": ANALYSIS_VERSION,
        "bootstrap_repeats": bootstrap_repeats,
        "seed": seed,
        "targets": {
            target_kind: {
                "target_field": spec["target_field"],
                "eligibility": f"nonblank {spec['target_field']}",
                "rule_field": spec["rule_field"],
                "judge_field": f"judge_{target_kind}" if judge_path else None,
            }
            for target_kind, spec in TARGETS.items()
        },
    }
    return {
        "analysis_version": ANALYSIS_VERSION,
        "generated_at_utc": utc_now(),
        "parameters": parameters,
        "parameters_sha256": config_hash(parameters),
        "analyzer": {
            "path": str(analyzer_path),
            "sha256": sha256_file(analyzer_path),
        },
        "e3_source": {
            "run_id": e3_manifest.get("run_id") if e3_manifest else e3_run_dir.name,
            "records_path": str(records_path.resolve()),
            "records_sha256": sha256_file(records_path),
            "dialogue_input_sha256": (
                e3_manifest.get("input", {}).get("sha256") if e3_manifest else None
            ),
            "manifest_sha256": sha256_file(e3_manifest_path) if e3_manifest else None,
            "source_git_commit": (
                e3_manifest.get("git", {}).get("commit") if e3_manifest else None
            ),
            "source_config_hash": e3_manifest.get("config_hash") if e3_manifest else None,
        },
        "judge_source": (
            {
                "run_id": (
                    judge_manifest.get("run_id") if judge_manifest else judge_path.parent.name
                ),
                "records_path": str(judge_path.resolve()),
                "records_sha256": sha256_file(judge_path),
                "record_count": len(judge_records),
                "manifest_sha256": (
                    sha256_file(judge_manifest_path) if judge_manifest else None
                ),
                "source_git_commit": (
                    judge_manifest.get("git", {}).get("commit") if judge_manifest else None
                ),
                "source_config_hash": (
                    judge_manifest.get("config_hash") if judge_manifest else None
                ),
                "prompt_version": (
                    judge_manifest.get("config", {}).get("prompt_version")
                    if judge_manifest
                    else None
                ),
                "model_identity_hash": (
                    judge_manifest.get("config", {})
                    .get("model_identity", {})
                    .get("identity_hash")
                    if judge_manifest
                    else None
                ),
            }
            if judge_path
            else None
        ),
    }


def analyze_records(
    records: list[dict[str, Any]],
    *,
    repeats: int,
    seed: int,
    include_judge: bool,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    design = validate_design(records)
    eligible_by_target, eligibility = eligible_records_by_target(records)
    fragment_eligibility = eligibility["fragment"]
    design.update(
        {
            "total_pairs": fragment_eligibility["total_pairs"],
            "eligible_pairs": fragment_eligibility["eligible_pairs"],
            "empty_target_pairs": fragment_eligibility["empty_target_pairs"],
            "eligibility_alias_note": (
                "Legacy design eligible fields alias fragment eligibility "
                "(nonblank unheard_text)."
            ),
            "eligibility_by_target": eligibility,
        }
    )
    metrics: dict[str, dict[str, str]] = {}
    for target_kind, spec in TARGETS.items():
        metrics[f"rule_{target_kind}"] = {
            "target_kind": target_kind,
            "field": spec["rule_field"],
        }
        if include_judge:
            metrics[f"judge_{target_kind}"] = {
                "target_kind": target_kind,
                "field": f"judge_{target_kind}",
            }
    local_playback = [
        record for record in records if record["condition"] == "playback"
    ]
    construction_checks = {
        "playback_local_unheard_empty": all(
            not record["local_unheard_in_history_text"].strip()
            for record in local_playback
        ),
        "playback_local_fragment_reference_positive": sum(
            bool(record["local_referenced_unheard"])
            for record in local_playback
        ),
        "n": len(local_playback),
    }
    if not construction_checks["playback_local_unheard_empty"]:
        raise ValueError("Playback construction check failed: local unheard text is non-empty")
    summary: dict[str, Any] = {
        "design": design,
        "construction_checks": construction_checks,
        "metrics": {
            name: summarize_metric(
                eligible_by_target[metric["target_kind"]],
                metric["field"],
                repeats=repeats,
                seed=seed,
            )
            for name, metric in metrics.items()
            if eligible_by_target[metric["target_kind"]]
        },
        "by_fraction": {},
        "scope_note": (
            "All conditions within a dialogue share one interrupted-turn trajectory and "
            "exact target text. Fragment metrics include pairs with nonblank unheard_text; "
            "proxy metrics independently include pairs with nonblank strict_unheard_text. "
            "Playback fragment-level zero remains a construction check."
        ),
    }
    if provenance is not None:
        summary["provenance"] = provenance
    for fraction in FRACTIONS:
        fraction_key = str(fraction)
        target_subsets = {
            target_kind: [
                record
                for record in eligible_records
                if str(record["fraction"]) == fraction_key
            ]
            for target_kind, eligible_records in eligible_by_target.items()
        }
        summary["by_fraction"][fraction_key] = {
            "eligible_pairs": eligibility["fragment"]["by_fraction"][fraction_key][
                "eligible_pairs"
            ],
            "eligibility_alias_note": "eligible_pairs aliases fragment eligibility.",
            "eligibility_by_target": {
                target_kind: target_eligibility["by_fraction"][fraction_key]
                for target_kind, target_eligibility in eligibility.items()
            },
            "metrics": {
                name: summarize_metric(
                    target_subsets[metric["target_kind"]],
                    metric["field"],
                    repeats=repeats,
                    seed=seed,
                )
                for name, metric in metrics.items()
                if target_subsets[metric["target_kind"]]
            },
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e3-run-dir", type=Path, required=True)
    parser.add_argument("--judge-records", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--bootstrap-repeats", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()

    records_path = args.e3_run_dir / "records.jsonl"
    records = load_jsonl(records_path)
    judged = attach_judgments(records, args.judge_records)
    provenance = build_provenance(
        e3_run_dir=args.e3_run_dir,
        records_path=records_path,
        judge_path=args.judge_records,
        judge_records=judged,
        bootstrap_repeats=args.bootstrap_repeats,
        seed=args.seed,
    )
    summary = analyze_records(
        records,
        repeats=args.bootstrap_repeats,
        seed=args.seed,
        include_judge=bool(args.judge_records),
        provenance=provenance,
    )
    output = args.out or args.e3_run_dir / "summary.json"
    atomic_write_json(output, summary)
    print(output)


if __name__ == "__main__":
    main()
