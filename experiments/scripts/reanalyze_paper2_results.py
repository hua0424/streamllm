#!/usr/bin/env python3
"""Offline integrity checks and statistics for the paper-2 result files.

This script never rewrites GPU-produced result JSON.  It excludes built-in
``fx*`` fixtures, recomputes descriptive statistics, and writes a separate
analysis artifact for use when drafting the thesis.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, pstdev

import numpy as np
from scipy.stats import binomtest

from src.config import RESULTS_DIR


ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path(RESULTS_DIR)
FORMAL_E2_THRESHOLDS = (0.0052, 0.1979, 0.3906, 0.5833, 0.776, 0.85, 0.92, 0.9688, 1.1)
E3_FIELDS = {
    "loose_rule": "referenced_unheard",
    "strict_rule": "referenced_unheard_strict",
    "loose_judge": "judge_referenced_unheard",
    "strict_judge": "judge_referenced_unheard_strict",
}


def _load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def _is_fixture(record_id: str) -> bool:
    return record_id.lower().startswith("fx")


def _percentile_interval(values: np.ndarray, alpha: float = 0.05) -> list[float]:
    low, high = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return [round(float(low), 6), round(float(high), 6)]


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [math.nan, math.nan]
    probability = successes / total
    denominator = 1 + z * z / total
    centre = (probability + z * z / (2 * total)) / denominator
    half_width = (
        z
        * math.sqrt(probability * (1 - probability) / total + z * z / (4 * total * total))
        / denominator
    )
    return [round(max(0.0, centre - half_width), 6), round(min(1.0, centre + half_width), 6)]


def _cohen_kappa(left: list[bool], right: list[bool]) -> float:
    if len(left) != len(right) or not left:
        return math.nan
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    left_yes = sum(left) / len(left)
    right_yes = sum(right) / len(right)
    expected = left_yes * right_yes + (1 - left_yes) * (1 - right_yes)
    if expected == 1:
        return 1.0
    return (observed - expected) / (1 - expected)


def analyze_e1(rng: np.random.Generator, bootstrap_repeats: int) -> dict:
    records = _load("exp1_latency.json")["records"]
    a = np.asarray([record["system_a"]["ttft_ms"] for record in records], dtype=float)
    b = np.asarray([record["system_b"]["ttft_ms"] for record in records], dtype=float)
    differences = a - b
    boot_indices = rng.integers(0, len(records), size=(bootstrap_repeats, len(records)))
    boot_differences = differences[boot_indices].mean(axis=1)
    survived = [bool(record["system_b"]["spec_survived"]) for record in records]
    return {
        "n": len(records),
        "system_a_ttft_ms": {
            "mean": round(float(a.mean()), 3),
            "median": round(float(np.median(a)), 3),
            "sd_population": round(float(a.std()), 3),
            "min": round(float(a.min()), 3),
            "max": round(float(a.max()), 3),
        },
        "system_b_ttft_eff_ms": {
            "mean": round(float(b.mean()), 3),
            "median": round(float(np.median(b)), 3),
            "sd_population": round(float(b.std()), 3),
            "min": round(float(b.min()), 3),
            "max": round(float(b.max()), 3),
        },
        "paired_improvement_ms": {
            "mean": round(float(differences.mean()), 3),
            "bootstrap_95_ci": _percentile_interval(boot_differences),
            "system_b_slower_count": int(np.sum(differences < 0)),
        },
        "speculation_survived": {
            "count": sum(survived),
            "rate": round(sum(survived) / len(survived), 6),
        },
        "scope_note": (
            "Text-segment-driven harness; the saved records do not contain input duration, "
            "so an input-length slope cannot be reconstructed."
        ),
    }


def analyze_e2() -> dict:
    raw = _load("exp2_tradeoff.json")["records"]
    clean = [
        record
        for record in raw
        if not _is_fixture(record["id"])
        and any(math.isclose(record["threshold"], threshold) for threshold in FORMAL_E2_THRESHOLDS)
    ]
    curve = []
    for threshold in FORMAL_E2_THRESHOLDS:
        records = [record for record in clean if math.isclose(record["threshold"], threshold)]
        wasted = sum(record["spec_wasted"] for record in records)
        generated = sum(record["n_generated"] for record in records)
        curve.append(
            {
                "threshold": threshold,
                "n": len(records),
                "spec_waste_rate": round(wasted / (wasted + generated), 6),
                "ttft_eff_ms": round(mean(record["ttft_eff_ms"] for record in records), 3),
                "survived_rate": round(mean(bool(record["survived"]) for record in records), 6),
                "avg_ready_tokens": round(mean(record["ready"] for record in records), 3),
            }
        )

    dominated = []
    for point in curve:
        for other in curve:
            if other is point:
                continue
            no_worse = (
                other["spec_waste_rate"] <= point["spec_waste_rate"]
                and other["ttft_eff_ms"] <= point["ttft_eff_ms"]
            )
            strictly_better = (
                other["spec_waste_rate"] < point["spec_waste_rate"]
                or other["ttft_eff_ms"] < point["ttft_eff_ms"]
            )
            if no_worse and strictly_better:
                dominated.append(
                    {"threshold": point["threshold"], "dominated_by": other["threshold"]}
                )
                break

    return {
        "raw_record_count": len(raw),
        "clean_record_count": len(clean),
        "excluded_fixture_records": sum(_is_fixture(record["id"]) for record in raw),
        "excluded_nonformal_threshold_records": sum(
            not _is_fixture(record["id"])
            and not any(
                math.isclose(record["threshold"], threshold)
                for threshold in FORMAL_E2_THRESHOLDS
            )
            for record in raw
        ),
        "curve": curve,
        "dominated_points": dominated,
        "scope_note": (
            "Nine discrete points from a synchronous speculation harness; the curve is not "
            "strictly monotonic at the aggressive end."
        ),
    }


def _e3_bootstrap(
    records: list[dict], field: str, rng: np.random.Generator, bootstrap_repeats: int
) -> dict:
    by_dialogue: dict[str, dict[str, list[bool]]] = defaultdict(
        lambda: {"playback": [], "generation": []}
    )
    for record in records:
        by_dialogue[record["id"]][record["condition"]].append(bool(record[field]))
    ids = sorted(by_dialogue)
    playback = np.asarray(
        [mean(by_dialogue[dialogue_id]["playback"]) for dialogue_id in ids], dtype=float
    )
    generation = np.asarray(
        [mean(by_dialogue[dialogue_id]["generation"]) for dialogue_id in ids], dtype=float
    )
    boot_indices = rng.integers(0, len(ids), size=(bootstrap_repeats, len(ids)))
    boot_playback = playback[boot_indices].mean(axis=1)
    boot_generation = generation[boot_indices].mean(axis=1)
    return {
        "playback_rate": round(float(playback.mean()), 6),
        "playback_cluster_bootstrap_95_ci": _percentile_interval(boot_playback),
        "generation_rate": round(float(generation.mean()), 6),
        "generation_cluster_bootstrap_95_ci": _percentile_interval(boot_generation),
        "generation_minus_playback": round(float((generation - playback).mean()), 6),
        "difference_cluster_bootstrap_95_ci": _percentile_interval(
            boot_generation - boot_playback
        ),
    }


def _mcnemar(records: list[dict], field: str) -> dict:
    paired: dict[tuple[str, str], dict[str, bool]] = defaultdict(dict)
    for record in records:
        paired[(record["id"], str(record["fraction"]))][record["condition"]] = bool(
            record[field]
        )
    playback_only = 0
    generation_only = 0
    for pair in paired.values():
        playback = pair["playback"]
        generation = pair["generation"]
        playback_only += playback and not generation
        generation_only += generation and not playback
    discordant = playback_only + generation_only
    p_value = (
        float(binomtest(min(playback_only, generation_only), discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    return {
        "paired_records": len(paired),
        "playback_only": playback_only,
        "generation_only": generation_only,
        "exact_p": p_value,
    }


def analyze_e3(rng: np.random.Generator, bootstrap_repeats: int) -> dict:
    raw = _load("exp3_consistency_judged.json")["records"]
    clean = [record for record in raw if not _is_fixture(record["id"])]
    by_condition = {
        condition: [record for record in clean if record["condition"] == condition]
        for condition in ("playback", "generation")
    }
    metrics = {}
    for name, field in E3_FIELDS.items():
        condition_results = {}
        for condition, records in by_condition.items():
            successes = sum(bool(record[field]) for record in records)
            condition_results[condition] = {
                "count": successes,
                "n": len(records),
                "rate": round(successes / len(records), 6),
                "wilson_95_ci": _wilson(successes, len(records)),
            }
        metrics[name] = {
            "conditions": condition_results,
            "paired_mcnemar": _mcnemar(clean, field),
            "dialogue_cluster_bootstrap": _e3_bootstrap(
                clean, field, rng, bootstrap_repeats
            ),
        }

    generation_loose_by_fraction = {}
    for fraction in (0.25, 0.5, 0.75, "boundary"):
        records = [
            record
            for record in by_condition["generation"]
            if str(record["fraction"]) == str(fraction)
        ]
        rule_count = sum(bool(record["referenced_unheard"]) for record in records)
        judge_count = sum(bool(record["judge_referenced_unheard"]) for record in records)
        generation_loose_by_fraction[str(fraction)] = {
            "n": len(records),
            "rule_count": rule_count,
            "rule_rate": round(rule_count / len(records), 6),
            "judge_count": judge_count,
            "judge_rate": round(judge_count / len(records), 6),
        }

    clean_boundary = [
        record
        for record in by_condition["playback"]
        if str(record["fraction"]) == "boundary"
    ]
    return {
        "raw_record_count": len(raw),
        "clean_record_count": len(clean),
        "dialogue_count": len({record["id"] for record in clean}),
        "excluded_fixture_records": len(raw) - len(clean),
        "metrics": metrics,
        "generation_loose_by_fraction": generation_loose_by_fraction,
        "playback_clean_boundary_zero_strict_residual": {
            "count": sum(record["strict_unheard_chars"] == 0 for record in clean_boundary),
            "n": len(clean_boundary),
        },
        "detector_kappa": {
            scope: round(
                _cohen_kappa(
                    [bool(record[E3_FIELDS[f"{scope}_rule"]]) for record in clean],
                    [bool(record[E3_FIELDS[f"{scope}_judge"]]) for record in clean],
                ),
                6,
            )
            for scope in ("loose", "strict")
        },
        "scope_note": (
            "Results come from a synchronous full-generation simulation with a 40-token cap; "
            "loose playback zero is guaranteed by the metric construction."
        ),
    }


def analyze_human_sample() -> dict:
    text = (RESULTS / "e3_human_validation_sample.md").read_text(encoding="utf-8")
    pattern = re.compile(
        r"^### #\d+\s+\[(?P<id>\S+) f=(?P<fraction>\S+) "
        r"(?P<condition>\S+) 列=(?P<scope>\S+)\]\s+"
        r"规则=(?P<rule>[YN]) / judge=(?P<judge>[YN]) / 人判=(?P<human>[YN])$",
        re.MULTILINE,
    )
    rows = [match.groupdict() for match in pattern.finditer(text)]
    clean = [row for row in rows if not _is_fixture(row["id"])]

    def summarize(selected: list[dict]) -> dict:
        human = [row["human"] == "Y" for row in selected]
        rule = [row["rule"] == "Y" for row in selected]
        judge = [row["judge"] == "Y" for row in selected]
        return {
            "n": len(selected),
            "human_positive": sum(human),
            "human_rule_agreement": round(mean(a == b for a, b in zip(human, rule)), 6)
            if selected
            else math.nan,
            "human_rule_kappa": round(_cohen_kappa(human, rule), 6),
            "human_judge_agreement": round(mean(a == b for a, b in zip(human, judge)), 6)
            if selected
            else math.nan,
            "human_judge_kappa": round(_cohen_kappa(human, judge), 6),
        }

    groups = {
        "all": clean,
        "loose_generation": [
            row for row in clean if row["scope"] == "loose" and row["condition"] == "generation"
        ],
        "strict_playback": [
            row for row in clean if row["scope"] == "strict" and row["condition"] == "playback"
        ],
        "strict_generation": [
            row for row in clean if row["scope"] == "strict" and row["condition"] == "generation"
        ],
    }
    result = {name: summarize(selected) for name, selected in groups.items()}
    strict_playback = result["strict_playback"]
    strict_playback["human_positive_wilson_95_ci"] = _wilson(
        strict_playback["human_positive"], strict_playback["n"]
    )
    return {
        "raw_sample_count": len(rows),
        "clean_sample_count": len(clean),
        "excluded_fixture_samples": len(rows) - len(clean),
        "groups": result,
        "scope_note": (
            "The sample was stratified on detector outputs and the labels were visible during "
            "annotation; its kappa values are descriptive, not population estimates."
        ),
    }


def analyze_a1() -> dict:
    rows = _load("exp_a1_kvreuse.json")["results"]
    return {
        "results": [
            {
                **row,
                "reprefill_over_crop_only": round(row["reprefill_ms"] / row["crop_only_ms"], 1),
                "reprefill_over_full_recovery": round(
                    row["reprefill_ms"] / row["crop_role_ms"], 1
                ),
            }
            for row in rows
        ],
        "scope_note": (
            "crop_only_ms times DynamicCache.crop only; timeline lookup, playback stopping, "
            "thread scheduling, and concurrent load are outside this microbenchmark."
        ),
    }


def analyze_a2() -> dict:
    records = _load("exp_a2_history_judged.json")["records"]
    by_id: dict[str, dict[str, dict]] = defaultdict(dict)
    for record in records:
        by_id[record["id"]][record["policy"]] = record
    complete = [policies for policies in by_id.values() if len(policies) == 3]
    heard_all_equal = sum(
        len({policies[policy]["heard_text"] for policy in ("naive", "mark", "rewrite")}) == 1
        for policies in complete
    )
    pair_equal = {
        "naive_mark": sum(
            policies["naive"]["heard_text"] == policies["mark"]["heard_text"]
            for policies in complete
        ),
        "naive_rewrite": sum(
            policies["naive"]["heard_text"] == policies["rewrite"]["heard_text"]
            for policies in complete
        ),
        "mark_rewrite": sum(
            policies["mark"]["heard_text"] == policies["rewrite"]["heard_text"]
            for policies in complete
        ),
    }
    rewrite_records = [record for record in records if record["policy"] == "rewrite"]
    rewrite_times = [record["rewrite_ms"] for record in rewrite_records]
    return {
        "dialogues_with_all_policies": len(complete),
        "all_three_heard_text_equal": heard_all_equal,
        "pairwise_heard_text_equal": pair_equal,
        "rewrite_changed_retained_history": sum(
            record["history_text"] != record["heard_text"] for record in rewrite_records
        ),
        "rewrite_latency_ms": {
            "mean": round(mean(rewrite_times), 3),
            "median": round(median(rewrite_times), 3),
            "p90_linear": round(float(np.quantile(rewrite_times, 0.9, method="linear")), 3),
            "max": round(max(rewrite_times), 3),
        },
        "score_descriptives": {
            policy: {
                "n": len(scores),
                "mean": round(mean(scores), 3),
                "median": round(median(scores), 3),
                "sd_population": round(pstdev(scores), 3),
                "counts": {
                    str(score): scores.count(score) for score in sorted(set(scores))
                },
            }
            for policy in ("naive", "mark", "rewrite")
            for scores in [
                [record["judge_coherence"] for record in records if record["policy"] == policy]
            ]
        },
        "scope_note": (
            "Policies independently regenerated the interrupted reply and the next reply; the "
            "reported score differences do not isolate a causal history-policy effect."
        ),
    }


def run(bootstrap_repeats: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "analysis": {
            "kind": "offline_reanalysis_without_gpu",
            "bootstrap_repeats": bootstrap_repeats,
            "seed": seed,
            "source_results_directory": str(RESULTS),
            "original_gpu_result_files_modified": False,
        },
        "e1": analyze_e1(rng, bootstrap_repeats),
        "e2": analyze_e2(),
        "e3": analyze_e3(rng, bootstrap_repeats),
        "e3_human_sample": analyze_human_sample(),
        "a1": analyze_a1(),
        "a2": analyze_a2(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=RESULTS / "paper2_reanalysis.json",
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()

    result = run(args.bootstrap_repeats, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote offline reanalysis: {args.out}")


if __name__ == "__main__":
    main()
