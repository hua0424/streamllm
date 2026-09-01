"""Aggregate confirmatory E1/E2 records with hierarchical bootstrap CIs."""

from __future__ import annotations

import argparse
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable, Mapping, Sequence

from experiments.sci34_supplement.common import (
    atomic_write_json,
    canonical_json,
    describe,
    sha256_bytes,
    sha256_file,
    utc_now,
)
from experiments.sci34_supplement.e1e2_confirmatory.protocol import (
    B_CONDITIONS,
    CONDITIONS,
    CONFIRMATORY_CONDITION,
    NEVER_SPECULATE,
    SYSTEM_A,
)
from experiments.sci34_supplement.e1e2_confirmatory.validate import (
    load_campaign_records,
    validate_campaign,
)


ANALYSIS_SCHEMA_VERSION = 1


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("Cannot compute a quantile of an empty sequence")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def _ci(values: Sequence[float], level: float = 0.95) -> dict[str, float]:
    alpha = (1.0 - level) / 2.0
    return {
        "level": level,
        "lower": _quantile(values, alpha),
        "upper": _quantile(values, 1.0 - alpha),
    }


def _ns_to_ms(value: int | float) -> float:
    return float(value) / 1_000_000.0


def _describe_ms(records: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    values = [_ns_to_ms(row[field]) for row in records]
    result = describe(values)
    result["mean"] = round(mean(values), 6)
    return result


def _optional_describe_ms(
    records: Sequence[Mapping[str, Any]], field: str
) -> dict[str, Any] | None:
    values = [_ns_to_ms(row[field]) for row in records if row.get(field) is not None]
    if not values:
        return None
    result = describe(values)
    result["mean"] = round(mean(values), 6)
    return result


def _utterance_waste(row: Mapping[str, Any]) -> float:
    denominator = int(row.get("waste_denominator_tokens", 0))
    return int(row.get("wasted_tokens", 0)) / denominator if denominator else 0.0


def summarize_condition(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot summarize an empty condition")
    wasted = sum(int(row.get("wasted_tokens", 0)) for row in records)
    waste_denominator = sum(int(row.get("waste_denominator_tokens", 0)) for row in records)
    speculative = sum(int(row.get("speculative_tokens", 0)) for row in records)
    survived = [bool(row.get("survived")) for row in records]
    ready = [int(row.get("ready_tokens", 0)) for row in records]
    invalidations = [int(row.get("n_invalidated", 0)) for row in records]
    waste_values = [_utterance_waste(row) for row in records]
    return {
        "n": len(records),
        "arrival_to_first_token_ready_ms_primary": _describe_ms(
            records, "arrival_to_first_token_ready_ns"
        ),
        "ttft_eff_ms_oracle_latency_lower_bound": _describe_ms(records, "ttft_eff_ns"),
        "oracle_preaccept_processing_ms": _describe_ms(
            records, "oracle_preaccept_processing_ns"
        ),
        "consumer_delivery_ms_diagnostic": _describe_ms(
            records, "consumer_delivery_latency_ns"
        ),
        "pooled_token_waste_ratio": wasted / waste_denominator if waste_denominator else 0.0,
        "pooled_wasted_tokens": wasted,
        "pooled_waste_denominator_tokens": waste_denominator,
        "waste_estimand": "sum(wasted_tokens) / sum(wasted_tokens + final_tokens)",
        "pooled_speculative_tokens_diagnostic": speculative,
        "utterance_waste": {
            **describe(waste_values),
            "mean": round(mean(waste_values), 6),
        },
        "survival_rate": sum(survived) / len(survived),
        "ready_tokens": {
            **describe(ready),
            "mean": round(mean(ready), 6),
        },
        "invalidations": {
            **describe(invalidations),
            "mean": round(mean(invalidations), 6),
        },
        "candidate_lead_ms": _optional_describe_ms(records, "candidate_lead_ns"),
        "on_demand_ttft_ms": _optional_describe_ms(records, "on_demand_ttft_ns"),
        "eos_rate": sum(bool(row.get("eos")) for row in records) / len(records),
        "max_tokens_hit_rate": sum(bool(row.get("max_tokens_hit")) for row in records)
        / len(records),
    }


def _by_key(records: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    result: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in records:
        key = (str(row["session_id"]), str(row["dialogue_id"]), str(row["condition"]))
        if key in result:
            raise ValueError(f"Duplicate record in analysis input: {key}")
        result[key] = row
    return result


def paired_differences(
    records: Sequence[Mapping[str, Any]],
    condition_a: str,
    condition_b: str,
    *,
    metric_field: str = "arrival_to_first_token_ready_ns",
) -> list[dict[str, Any]]:
    indexed = _by_key(records)
    base_keys = sorted(
        (session, dialogue)
        for session, dialogue, condition in indexed
        if condition == condition_a
    )
    pairs: list[dict[str, Any]] = []
    for session, dialogue in base_keys:
        left = indexed.get((session, dialogue, condition_a))
        right = indexed.get((session, dialogue, condition_b))
        if right is None:
            raise ValueError(f"Missing paired record for {session}/{dialogue}/{condition_b}")
        left_ms = _ns_to_ms(left[metric_field])
        right_ms = _ns_to_ms(right[metric_field])
        pairs.append(
            {
                "session_id": session,
                "dialogue_id": dialogue,
                "a_ms": left_ms,
                "b_ms": right_ms,
                "absolute_difference_ms": left_ms - right_ms,
                "relative_difference_over_a": (
                    (left_ms - right_ms) / left_ms if left_ms > 0 else None
                ),
            }
        )
    return pairs


def summarize_pairs(
    pairs: Sequence[Mapping[str, Any]], *, metric_label: str
) -> dict[str, Any]:
    absolute = [float(pair["absolute_difference_ms"]) for pair in pairs]
    relative = [
        float(pair["relative_difference_over_a"])
        for pair in pairs
        if pair["relative_difference_over_a"] is not None
    ]
    return {
        "n": len(pairs),
        "absolute_difference_ms_a_minus_b": {
            **describe(absolute),
            "mean": round(mean(absolute), 6),
        },
        "relative_difference_over_a": (
            {**describe(relative), "mean": round(mean(relative), 6)}
            if relative
            else None
        ),
        "metric": metric_label,
        "relative_denominator": (
            f"condition_a {metric_label}; zero-denominator pairs are excluded"
        ),
        "zero_denominator_pairs": len(pairs) - len(relative),
    }


def hierarchical_resample(
    records: Sequence[Mapping[str, Any]], rng: random.Random
) -> list[Mapping[str, Any]]:
    """Resample session, then dialogue within each drawn session.

    Draw identifiers are written onto shallow record copies so repeated source
    sessions/dialogues remain distinct paired clusters in downstream estimands.
    All ten conditions for a selected dialogue are retained together.
    """
    sessions: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in records:
        sessions[str(row["session_id"])][str(row["dialogue_id"])].append(row)
    session_ids = sorted(sessions)
    sampled: list[Mapping[str, Any]] = []
    for session_draw in range(len(session_ids)):
        selected_session = rng.choice(session_ids)
        dialogue_map = sessions[selected_session]
        dialogue_ids = sorted(dialogue_map)
        for dialogue_draw in range(len(dialogue_ids)):
            selected_dialogue = rng.choice(dialogue_ids)
            for source in dialogue_map[selected_dialogue]:
                copy = dict(source)
                copy["source_session_id"] = selected_session
                copy["source_dialogue_id"] = selected_dialogue
                copy["session_id"] = f"bootstrap-session-{session_draw}"
                copy["dialogue_id"] = f"bootstrap-dialogue-{dialogue_draw}"
                sampled.append(copy)
    return sampled


def hierarchical_bootstrap(
    records: Sequence[Mapping[str, Any]],
    estimands: Mapping[str, Callable[[Sequence[Mapping[str, Any]]], float]],
    *,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    if repeats <= 0:
        raise ValueError("bootstrap repeats must be positive")
    rng = random.Random(seed)
    draws = {name: [] for name in estimands}
    for _ in range(repeats):
        sample = hierarchical_resample(records, rng)
        for name, function in estimands.items():
            value = float(function(sample))
            if not math.isfinite(value):
                raise ValueError(f"Non-finite bootstrap estimand {name}: {value}")
            draws[name].append(value)
    return {
        "method": "percentile two-level bootstrap: session then dialogue within drawn session",
        "seed": seed,
        "repeats": repeats,
        "ci": {name: _ci(values) for name, values in draws.items()},
    }


def _condition_rows(records: Sequence[Mapping[str, Any]], condition: str):
    return [row for row in records if row["condition"] == condition]


def _paired_mean_difference(
    records: Sequence[Mapping[str, Any]],
    condition_a: str,
    condition_b: str,
    metric_field: str = "arrival_to_first_token_ready_ns",
) -> float:
    return mean(
        pair["absolute_difference_ms"]
        for pair in paired_differences(
            records, condition_a, condition_b, metric_field=metric_field
        )
    )


def _condition_mean_metric(
    records: Sequence[Mapping[str, Any]], condition: str, metric_field: str
) -> float:
    return mean(
        _ns_to_ms(row[metric_field]) for row in _condition_rows(records, condition)
    )


def _pooled_waste(records: Sequence[Mapping[str, Any]], condition: str) -> float:
    subset = _condition_rows(records, condition)
    denominator = sum(int(row.get("waste_denominator_tokens", 0)) for row in subset)
    numerator = sum(int(row.get("wasted_tokens", 0)) for row in subset)
    return numerator / denominator if denominator else 0.0


def _survival(records: Sequence[Mapping[str, Any]], condition: str) -> float:
    subset = _condition_rows(records, condition)
    return sum(bool(row.get("survived")) for row in subset) / len(subset)


def build_analysis(
    campaign_dir: Path,
    *,
    bootstrap_repeats: int = 10_000,
    bootstrap_seed: int = 20260901,
    expected_sessions: int = 5,
    expected_dialogues: int = 100,
    formal: bool = True,
) -> dict[str, Any]:
    if formal and (
        expected_sessions != 5 or expected_dialogues != 100
    ):
        raise ValueError("Formal analyzer dimensions are frozen at 5 sessions × 100 dialogues")
    validation = validate_campaign(
        campaign_dir,
        expected_sessions=expected_sessions,
        expected_dialogues=expected_dialogues,
        formal=formal,
    )
    if not validation["ok"]:
        raise ValueError("Campaign validation failed: " + "; ".join(validation["errors"][:10]))
    records, record_paths = load_campaign_records(campaign_dir)
    by_condition = {
        condition: summarize_condition(_condition_rows(records, condition))
        for condition in CONDITIONS
    }
    e1_pairs = paired_differences(records, SYSTEM_A, CONFIRMATORY_CONDITION)
    e2_pairs = paired_differences(records, NEVER_SPECULATE, CONFIRMATORY_CONDITION)
    e1_oracle_pairs = paired_differences(
        records, SYSTEM_A, CONFIRMATORY_CONDITION, metric_field="ttft_eff_ns"
    )
    e2_oracle_pairs = paired_differences(
        records, NEVER_SPECULATE, CONFIRMATORY_CONDITION, metric_field="ttft_eff_ns"
    )
    estimands: dict[str, Callable[[Sequence[Mapping[str, Any]]], float]] = {
        "e1_primary_mean_arrival_to_ready_difference_ms_system_a_minus_b092": lambda sample: _paired_mean_difference(
            sample, SYSTEM_A, CONFIRMATORY_CONDITION
        ),
        "e2_primary_mean_arrival_to_ready_difference_ms_never_minus_b092": lambda sample: _paired_mean_difference(
            sample, NEVER_SPECULATE, CONFIRMATORY_CONDITION
        ),
        "e1_oracle_mean_ttft_eff_difference_ms_system_a_minus_b092": lambda sample: _paired_mean_difference(
            sample, SYSTEM_A, CONFIRMATORY_CONDITION, "ttft_eff_ns"
        ),
        "e2_oracle_mean_ttft_eff_difference_ms_never_minus_b092": lambda sample: _paired_mean_difference(
            sample, NEVER_SPECULATE, CONFIRMATORY_CONDITION, "ttft_eff_ns"
        ),
    }
    for condition in B_CONDITIONS:
        estimands[f"{condition}.primary_mean_arrival_to_ready_ms"] = (
            lambda sample, target=condition: _condition_mean_metric(
                sample, target, "arrival_to_first_token_ready_ns"
            )
        )
        estimands[f"{condition}.oracle_mean_ttft_eff_ms"] = (
            lambda sample, target=condition: _condition_mean_metric(
                sample, target, "ttft_eff_ns"
            )
        )
        estimands[f"{condition}.pooled_token_waste_ratio"] = (
            lambda sample, target=condition: _pooled_waste(sample, target)
        )
        estimands[f"{condition}.survival_rate"] = (
            lambda sample, target=condition: _survival(sample, target)
        )
    bootstrap = hierarchical_bootstrap(
        records,
        estimands,
        repeats=bootstrap_repeats,
        seed=bootstrap_seed,
    )
    sources = {
        "campaign_dir": str(campaign_dir.resolve()),
        "session_records": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in record_paths
        ],
        "session_manifests": [
            {
                "path": str((path.parent / "manifest.json").resolve()),
                "sha256": sha256_file(path.parent / "manifest.json"),
            }
            for path in record_paths
        ],
        "analyzer": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "validator": {
            "path": str((Path(__file__).resolve().parent / "validate.py").resolve()),
            "sha256": sha256_file(Path(__file__).resolve().parent / "validate.py"),
        },
    }
    sources["provenance_hash"] = sha256_bytes(canonical_json(sources).encode("utf-8"))
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "experiment": "e1e2_confirmatory",
        "created_at_utc": utc_now(),
        "scope_note": (
            "Controlled synchronous text segments with oracle endpoint acceptance; not real ASR, "
            "online TTS, acoustic playback, or production end-to-end latency."
        ),
        "design": {
            "sessions": expected_sessions,
            "dialogues_per_session": expected_dialogues,
            "conditions": list(CONDITIONS),
            "confirmatory_condition": CONFIRMATORY_CONDITION,
            "primary_latency_metric": "last_segment_arrival_ns to first_token_ready_ns",
            "oracle_latency_lower_bound_metric": (
                "TTFT_eff from endpoint_accept_ns; survived candidates are set to zero"
            ),
            "endpoint_note": (
                "For System B endpoint acceptance occurs after final-segment prefill/trigger/"
                "candidate processing, not immediately at final segment arrival."
            ),
            "e1_pair": [SYSTEM_A, CONFIRMATORY_CONDITION],
            "e2_pair": [CONFIRMATORY_CONDITION, NEVER_SPECULATE],
            "outlier_trimming": None,
            "campaign_identity_hashes": validation["provenance"]["campaign_identity_hashes"],
            "input_sha256": validation["provenance"]["input_sha256"],
            "trigger_cache_sha256": validation["provenance"]["trigger_cache_sha256"],
        },
        "validation": validation,
        "condition_summaries": by_condition,
        "e1": {
            "primary_comparison": (
                "System A minus B@0.92 arrival-to-first-token-ready, paired within "
                "session and dialogue"
            ),
            "primary_paired": summarize_pairs(
                e1_pairs, metric_label="arrival_to_first_token_ready"
            ),
            "oracle_ttft_eff_latency_lower_bound_paired": summarize_pairs(
                e1_oracle_pairs, metric_label="oracle_ttft_eff_latency_lower_bound"
            ),
            "b092_diagnostics": by_condition[CONFIRMATORY_CONDITION],
        },
        "e2": {
            "primary_comparison": (
                "never_speculate minus B@0.92 arrival-to-first-token-ready, paired "
                "within session and dialogue"
            ),
            "primary_paired": summarize_pairs(
                e2_pairs, metric_label="arrival_to_first_token_ready"
            ),
            "oracle_ttft_eff_latency_lower_bound_paired": summarize_pairs(
                e2_oracle_pairs, metric_label="oracle_ttft_eff_latency_lower_bound"
            ),
            "discrete_working_points": {
                condition: by_condition[condition] for condition in B_CONDITIONS
            },
        },
        "bootstrap": bootstrap,
        "provenance": sources,
        "excluded_records": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--bootstrap-repeats", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260901)
    parser.add_argument("--expected-sessions", type=int, default=5)
    parser.add_argument("--expected-dialogues", type=int, default=100)
    parser.add_argument("--non-formal", action="store_true")
    args = parser.parse_args()
    output = args.out or args.campaign_dir / "analysis_v1.json"
    if output.exists():
        raise FileExistsError(
            f"Analysis is immutable and already exists: {output}; choose a new versioned path"
        )
    result = build_analysis(
        args.campaign_dir,
        bootstrap_repeats=args.bootstrap_repeats,
        bootstrap_seed=args.bootstrap_seed,
        expected_sessions=args.expected_sessions,
        expected_dialogues=args.expected_dialogues,
        formal=not args.non_formal,
    )
    atomic_write_json(output, result)
    print(output)


if __name__ == "__main__":
    main()
