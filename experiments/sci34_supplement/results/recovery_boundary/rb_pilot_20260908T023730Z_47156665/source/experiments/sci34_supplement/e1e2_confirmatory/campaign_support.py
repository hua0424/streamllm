"""Model-free scheduling, persistence, and analysis primitives for E1/E2.

The core runner/analyzer can call these helpers through their public data
contracts.  Keeping them independent also permits smoke coverage before those
entry points are available.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable, Mapping, Sequence

from experiments.sci34_supplement.common import (
    append_jsonl,
    config_hash,
    load_jsonl,
    sha256_file,
)
from experiments.sci34_supplement.e1e2_confirmatory.protocol import (
    CONDITIONS,
    CONFIRMATORY_CONDITION,
    NEVER_SPECULATE,
    SYSTEM_A,
    WARMUP_PATHS,
    balanced_condition_order as protocol_condition_order,
)


def balanced_condition_order(
    *, session_index: int, dialogue_index: int, seed: int, conditions: Sequence[str] = CONDITIONS
) -> list[str]:
    """Use the frozen public protocol order, with an optional test subset."""
    full_order = protocol_condition_order(
        session_index=session_index, dialogue_index=dialogue_index, seed=seed
    )
    if tuple(conditions) == tuple(CONDITIONS):
        return full_order
    if not conditions or len(set(conditions)) != len(conditions):
        raise ValueError("Conditions must be non-empty and unique")
    requested = set(conditions)
    if not requested.issubset(CONDITIONS):
        raise ValueError("Unknown condition requested")
    return [condition for condition in full_order if condition in requested]


def assert_balanced_orders(orders: Sequence[Sequence[str]]) -> None:
    if not orders:
        raise ValueError("No condition orders supplied")
    expected = set(orders[0])
    position_counts: dict[str, Counter[int]] = defaultdict(Counter)
    for order in orders:
        if len(order) != len(expected) or set(order) != expected:
            raise ValueError("Condition order is not a complete grid permutation")
        for position, condition in enumerate(order):
            position_counts[condition][position] += 1
    counts = [
        position_counts[condition][position]
        for condition in expected
        for position in range(len(expected))
    ]
    if max(counts) - min(counts) > 1:
        raise ValueError("Condition-position allocation is not balanced")


def validate_record_file(
    path: Path, *, key_fields: Sequence[str] = ("session_id", "id", "condition")
) -> list[dict[str, Any]]:
    """Reject truncated JSONL, missing keys, and duplicate formal keys."""
    records = load_jsonl(path)
    seen: set[tuple[str, ...]] = set()
    for record in records:
        if record.get("trial_kind", "formal") != "formal":
            raise ValueError("Warmup/non-formal record leaked into records.jsonl")
        try:
            key = tuple(str(record[field]) for field in key_fields)
        except KeyError as error:
            raise ValueError(f"Record missing key field: {error.args[0]}") from error
        if key in seen:
            raise ValueError(f"Duplicate formal record key: {key}")
        seen.add(key)
    return records


def run_missing_grid(
    *,
    rows: Sequence[Mapping[str, Any]],
    session_id: str,
    session_index: int,
    seed: int,
    records_path: Path,
    warmup: Callable[[str], None],
    run_cell: Callable[[Mapping[str, Any], str, int], Mapping[str, Any]],
    conditions: Sequence[str] = CONDITIONS,
) -> list[dict[str, Any]]:
    """Warm and append only missing grid cells; never persist warmup calls."""
    existing = validate_record_file(records_path) if records_path.exists() else []
    completed = {(str(row["id"]), str(row["condition"])) for row in existing}
    missing_by_dialogue: list[tuple[int, Mapping[str, Any], list[tuple[str, int]]]] = []
    for dialogue_index, row in enumerate(rows):
        order = balanced_condition_order(
            session_index=session_index,
            dialogue_index=dialogue_index,
            seed=seed,
            conditions=conditions,
        )
        missing = [
            (condition, position)
            for position, condition in enumerate(order)
            if (str(row["id"]), condition) not in completed
        ]
        if missing:
            missing_by_dialogue.append((dialogue_index, row, missing))
    if not missing_by_dialogue:
        return existing

    for path_kind in WARMUP_PATHS:
        warmup(path_kind)
    for _, row, missing in missing_by_dialogue:
        for condition, position in missing:
            record = dict(run_cell(row, condition, position))
            record.update(
                {
                    "trial_kind": "formal",
                    "session_id": session_id,
                    "id": str(row["id"]),
                    "condition": condition,
                    "condition_order": position,
                }
            )
            append_jsonl(records_path, record)
    return validate_record_file(records_path)


def validate_ttft_record(record: Mapping[str, Any]) -> None:
    endpoint = int(record["endpoint_accept_ns"])
    deliverable = int(record["first_deliverable_token_ns"])
    ttft = int(record["ttft_eff_ns"])
    if deliverable < endpoint or ttft != deliverable - endpoint or ttft < 0:
        raise ValueError("Invalid TTFT_eff timestamps")
    if record.get("survived") and int(record.get("ready_tokens", 0)) > 0 and ttft != 0:
        raise ValueError("Survived candidate with ready tokens must have TTFT_eff=0")


def pooled_waste(records: Iterable[Mapping[str, Any]]) -> dict[str, float | int | None]:
    rows = list(records)
    wasted = sum(int(row.get("wasted_tokens", 0)) for row in rows)
    speculative = sum(
        int(row.get("speculative_tokens", row.get("wasted_tokens", 0) + row.get("ready_tokens", 0)))
        for row in rows
    )
    return {
        "wasted_tokens": wasted,
        "speculative_tokens": speculative,
        "ratio": wasted / speculative if speculative else None,
    }


def paired_e1_difference(
    records: Iterable[Mapping[str, Any]],
    *,
    a_condition: str = SYSTEM_A,
    b_condition: str = CONFIRMATORY_CONDITION,
) -> dict[str, Any]:
    cells: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in records:
        condition = str(row["condition"])
        if condition in (a_condition, b_condition):
            cells[(str(row["session_id"]), str(row["id"]))][condition] = (
                float(row["ttft_eff_ns"]) / 1_000_000
            )
    differences = [
        values[a_condition] - values[b_condition]
        for values in cells.values()
        if a_condition in values and b_condition in values
    ]
    if not differences:
        raise ValueError("No complete E1 pairs")
    return {"n_pairs": len(differences), "mean_a_minus_b_ms": mean(differences), "raw_ms": differences}


def _estimand_from_clusters(
    clusters: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Any]]]],
    *,
    a_condition: str,
    b_condition: str,
) -> float:
    differences: list[float] = []
    for dialogues in clusters.values():
        for conditions in dialogues.values():
            if a_condition in conditions and b_condition in conditions:
                differences.append(
                    (
                        float(conditions[a_condition]["ttft_eff_ns"])
                        - float(conditions[b_condition]["ttft_eff_ns"])
                    )
                    / 1_000_000
                )
    if not differences:
        raise ValueError("Bootstrap sample has no complete condition pairs")
    return mean(differences)


def hierarchical_bootstrap(
    records: Sequence[Mapping[str, Any]],
    *,
    repeats: int,
    seed: int,
    a_condition: str = SYSTEM_A,
    b_condition: str = CONFIRMATORY_CONDITION,
) -> dict[str, Any]:
    """Resample sessions, then dialogues within each sampled session."""
    if repeats < 1:
        raise ValueError("Bootstrap repeats must be positive")
    clusters: dict[str, dict[str, dict[str, Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for record in records:
        clusters[str(record["session_id"])][str(record["id"])][str(record["condition"])] = record
    session_ids = sorted(clusters)
    if not session_ids:
        raise ValueError("No sessions to bootstrap")
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(repeats):
        sampled: dict[str, dict[str, dict[str, Mapping[str, Any]]]] = {}
        for session_draw in range(len(session_ids)):
            source_session = rng.choice(session_ids)
            dialogue_ids = sorted(clusters[source_session])
            if not dialogue_ids:
                raise ValueError(f"Session {source_session} has no dialogues")
            sampled_dialogues: dict[str, dict[str, Mapping[str, Any]]] = {}
            for dialogue_draw in range(len(dialogue_ids)):
                source_dialogue = rng.choice(dialogue_ids)
                sampled_dialogues[f"{source_dialogue}#{dialogue_draw}"] = clusters[source_session][source_dialogue]
            sampled[f"{source_session}#{session_draw}"] = sampled_dialogues
        samples.append(
            _estimand_from_clusters(sampled, a_condition=a_condition, b_condition=b_condition)
        )
    ordered = sorted(samples)

    def percentile(probability: float) -> float:
        return ordered[min(len(ordered) - 1, max(0, round((len(ordered) - 1) * probability)))]

    point = _estimand_from_clusters(clusters, a_condition=a_condition, b_condition=b_condition)
    return {
        "point_estimate_ms": point,
        "ci95_ms": [percentile(0.025), percentile(0.975)],
        "provenance": {
            "method": "two-level nonparametric bootstrap: sessions then dialogues",
            "session_count": len(session_ids),
            "dialogues_per_session": {key: len(value) for key, value in clusters.items()},
            "repeats": repeats,
            "seed": seed,
            "estimand": f"mean paired TTFT_eff({a_condition} - {b_condition}) in ms",
            "conditions_retained_together": True,
        },
        "samples": samples,
    }


def analysis_provenance(
    *,
    records_paths: Sequence[Path],
    manifest_paths: Sequence[Path],
    trigger_cache_path: Path,
    input_path: Path,
    analyzer_path: Path,
    bootstrap: Mapping[str, Any],
    excluded_records: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    sources = {
        "records": [{"path": str(path.resolve()), "sha256": sha256_file(path)} for path in records_paths],
        "manifests": [{"path": str(path.resolve()), "sha256": sha256_file(path)} for path in manifest_paths],
        "trigger_cache": {"path": str(trigger_cache_path.resolve()), "sha256": sha256_file(trigger_cache_path)},
        "input": {"path": str(input_path.resolve()), "sha256": sha256_file(input_path)},
        "analyzer": {"path": str(analyzer_path.resolve()), "sha256": sha256_file(analyzer_path)},
    }
    return {
        "sources": sources,
        "bootstrap": dict(bootstrap),
        "excluded_records": list(excluded_records),
        "provenance_hash": config_hash({"sources": sources, "bootstrap": bootstrap, "excluded_records": excluded_records}),
    }
