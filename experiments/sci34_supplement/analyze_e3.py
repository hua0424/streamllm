"""Paired and dialogue-clustered analysis for fixed-trajectory E3."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from scipy.stats import binomtest

from experiments.sci34_supplement.common import atomic_write_json, load_jsonl


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


def attach_judgments(records: list[dict[str, Any]], judge_path: Path | None) -> None:
    if not judge_path:
        return
    judged = load_jsonl(judge_path)
    if any(not row.get("parse_success") for row in judged):
        raise ValueError("Judge records contain parse failures")
    lookup = {
        (row["id"], str(row["fraction"]), row["condition"], row["target_kind"]): row
        for row in judged
    }
    for record in records:
        for target_kind in ("fragment", "proxy"):
            key = (record["id"], str(record["fraction"]), record["condition"], target_kind)
            if key not in lookup:
                raise ValueError(f"Missing judgment: {key}")
            record[f"judge_{target_kind}"] = bool(lookup[key]["verdict"])


def summarize_metric(
    records: list[dict[str, Any]], field: str, *, repeats: int, seed: int
) -> dict[str, Any]:
    conditions = {}
    for condition in ("playback", "generation"):
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


def validate_design(records: list[dict[str, Any]]) -> dict[str, int]:
    by_dialogue: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if str(record["id"]).lower().startswith("fx"):
            raise ValueError("Formal analysis contains fixture records")
        by_dialogue[record["id"]].append(record)
    for dialogue_id, rows in by_dialogue.items():
        if len({row["trajectory_id"] for row in rows}) != 1:
            raise ValueError(f"{dialogue_id}: conditions do not share a trajectory")
        if len({row["history_key"] for row in rows if row["condition"] == "generation"}) != 1:
            raise ValueError(f"{dialogue_id}: generation history differs by fraction")
    return {
        "dialogues": len(by_dialogue),
        "records": len(records),
        "pairs": len(records) // 2,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e3-run-dir", type=Path, required=True)
    parser.add_argument("--judge-records", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--bootstrap-repeats", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()

    records = load_jsonl(args.e3_run_dir / "records.jsonl")
    design = validate_design(records)
    attach_judgments(records, args.judge_records)
    pair_eligibility: dict[tuple[str, str], set[bool]] = defaultdict(set)
    for record in records:
        pair_eligibility[(record["id"], str(record["fraction"]))].add(
            bool(record["eligible_delta"])
        )
    if any(len(values) != 1 for values in pair_eligibility.values()):
        raise ValueError("Pair conditions disagree on eligible_delta")
    eligible_pairs = {key for key, values in pair_eligibility.items() if True in values}
    eligible_records = [
        record
        for record in records
        if (record["id"], str(record["fraction"])) in eligible_pairs
    ]
    design.update(
        {
            "total_pairs": len(pair_eligibility),
            "eligible_pairs": len(eligible_pairs),
            "empty_target_pairs": len(pair_eligibility) - len(eligible_pairs),
        }
    )
    if not eligible_records:
        raise ValueError("No eligible shared-delta pairs")
    metrics = {
        "rule_fragment": "referenced_unheard",
        "rule_proxy": "referenced_unheard_strict",
    }
    if args.judge_records:
        metrics.update(
            {
                "judge_fragment": "judge_fragment",
                "judge_proxy": "judge_proxy",
            }
        )
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
    summary = {
        "design": design,
        "construction_checks": construction_checks,
        "metrics": {
            name: summarize_metric(
                eligible_records,
                field,
                repeats=args.bootstrap_repeats,
                seed=args.seed,
            )
            for name, field in metrics.items()
        },
        "by_fraction": {},
        "scope_note": (
            "All conditions within a dialogue share one interrupted-turn trajectory. "
            "Playback fragment-level zero remains a construction check; paired effects "
            "are evaluated on the shared fixed trajectory."
        ),
    }
    for fraction in (0.25, 0.5, 0.75, "boundary"):
        subset = [
            record
            for record in eligible_records
            if str(record["fraction"]) == str(fraction)
        ]
        summary["by_fraction"][str(fraction)] = {
            "eligible_pairs": len(subset) // 2,
            "metrics": (
                {
                    name: summarize_metric(
                        subset,
                        field,
                        repeats=args.bootstrap_repeats,
                        seed=args.seed,
                    )
                    for name, field in metrics.items()
                }
                if subset
                else {}
            ),
        }
    output = args.out or args.e3_run_dir / "summary.json"
    atomic_write_json(output, summary)
    print(output)


if __name__ == "__main__":
    main()
