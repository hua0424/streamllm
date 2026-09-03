"""Post-review E3 weighting and exact-semantic-deduplication analysis.

The formal path is fail-closed over the accepted E3 and judge artifacts. It reads
those sources without rewriting them and creates one versioned JSON analysis plus
an SHA-256 sidecar in the accepted E3 run directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from experiments.sci34_supplement.common import (
    ROOT,
    atomic_write_json,
    atomic_write_text,
    canonical_json,
    config_hash,
    utc_now,
)


SCHEMA_VERSION = "e3-analysis/2.0"
ANALYSIS_VERSION = "e3-weighting-dedup-v2"
POST_REVIEW_SENSITIVITY = True
FORMAL_REPEATS = 10_000
FORMAL_SEED = 20260831
FORMAL_E3_RUN_ID = "sci34_f11ccba_20260901_e3"
FORMAL_JUDGE_RUN_ID = "sci34_f11ccba_20260901_judge_v2"
FORMAL_OUTPUT_NAME = "analysis_weighting_dedup_v2.json"
FORMAL_SIDECAR_NAME = "analysis_weighting_dedup_v2.sha256"
CONDITIONS = ("playback", "generation")
FRACTIONS = ("0.25", "0.5", "0.75", "boundary")
TARGETS = {
    "fragment": {
        "target_field": "unheard_text",
        "rule_field": "referenced_unheard",
        "dedup_key": (
            "id",
            "trajectory_id",
            "playback_history_key",
            "generation_history_key",
            "heard_token_end",
            "sha256_exact_target",
        ),
    },
    "proxy": {
        "target_field": "strict_unheard_text",
        "rule_field": "referenced_unheard_strict",
        "dedup_key": (
            "id",
            "trajectory_id",
            "playback_history_key",
            "generation_history_key",
            "sha256_exact_target",
        ),
    },
}
ESTIMANDS = (
    "label_weighted",
    "dialogue_weighted",
    "unique_semantic_group_weighted",
    "unique_dialogue_weighted",
)

# LF-normalized identities and corresponding Git blob IDs for accepted sources.
FORMAL_SOURCE_IDENTITIES = {
    "e3_records": {
        "relative_path": (
            "experiments/sci34_supplement/results/e3/"
            "sci34_f11ccba_20260901_e3/records.jsonl"
        ),
        "lf_sha256": "9406ea42d1112b5ad97c94e7e27856946acc79c14dd4ebe5762d0b702fe458e9",
        "git_blob_sha1": "117837340e030afca03aa2cf2ae0b5fc6199aaee",
    },
    "e3_manifest": {
        "relative_path": (
            "experiments/sci34_supplement/results/e3/"
            "sci34_f11ccba_20260901_e3/manifest.json"
        ),
        "lf_sha256": "7690f1003109a37c6f216b674ff6df2b71a4bfac98f6992c1eb23b37f98967a4",
        "git_blob_sha1": "bbe87c6a1a1a610deeb7bbdd889122748b585de5",
    },
    "judge_records": {
        "relative_path": (
            "experiments/sci34_supplement/results/judge/"
            "sci34_f11ccba_20260901_judge_v2/judge_records.jsonl"
        ),
        "lf_sha256": "eeb2db2c4ed76ed3d01a0bbce4a4ec8ab8491aa8c09178bd3e4cfb5811945d44",
        "git_blob_sha1": "957a8d67544aa506e23c924c7cb9b10b543539d3",
    },
    "judge_manifest": {
        "relative_path": (
            "experiments/sci34_supplement/results/judge/"
            "sci34_f11ccba_20260901_judge_v2/manifest.json"
        ),
        "lf_sha256": "f103963a300dc8f668161d1c1147d9c8e0d0f488a8d66ff76f59dee55cbc91c4",
        "git_blob_sha1": "abc892e6f472e27d3ebae16a1cb8aa075a027980",
    },
    "superseded_v1": {
        "relative_path": (
            "experiments/sci34_supplement/results/e3/"
            "sci34_f11ccba_20260901_e3/analysis_metric_specific_eligibility_v1.json"
        ),
        "lf_sha256": "b280d4292a0dbb692f71ed34d65b5268f0340bb439063d266b8267fbf4f5afd3",
        "git_blob_sha1": "9f6108c62dfc9313b243074095b2218217ab8b72",
    },
}

FORMAL_EXPECTED = {
    "raw_records": 800,
    "paired_labels": 400,
    "judge_records": 1600,
    "raw_dialogues": 100,
    "playback_records": 400,
    "playback_local_unheard_empty": 400,
    "playback_local_reference_positives": 0,
    "fragment": {
        "eligible_labels": 297,
        "eligible_dialogues": 96,
        "semantic_groups": 169,
        "rule": {"playback_positive": 199, "generation_positive": 189, "positive_difference": -10},
        "judge": {"playback_positive": 127, "generation_positive": 121, "positive_difference": -6},
    },
    "proxy": {
        "eligible_labels": 380,
        "eligible_dialogues": 100,
        "semantic_groups": 379,
        "rule": {"playback_positive": 286, "generation_positive": 280, "positive_difference": -6},
        "judge": {"playback_positive": 167, "generation_positive": 157, "positive_difference": -10},
    },
}


class E3V2ValidationError(ValueError):
    """Raised when a v2 fail-closed validation gate rejects an input."""


def _lf_bytes(value: bytes) -> bytes:
    return value.replace(b"\r\n", b"\n")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git_blob_sha1(value: bytes) -> str:
    header = f"blob {len(value)}\0".encode("ascii")
    return hashlib.sha1(header + value).hexdigest()


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def artifact_identity(
    path: Path,
    *,
    expected_lf_sha256: str | None = None,
    expected_git_blob_sha1: str | None = None,
) -> dict[str, Any]:
    if not path.is_file():
        raise E3V2ValidationError(f"Missing source artifact: {path}")
    raw = path.read_bytes()
    normalized = _lf_bytes(raw)
    identity = {
        "path": _repo_relative(path),
        "local_sha256": _sha256(raw),
        "lf_normalized_sha256": _sha256(normalized),
        "git_blob_sha1": _git_blob_sha1(normalized),
        "identity_bytes": "LF-normalized bytes",
        "line_ending_normalization_applied": raw != normalized,
    }
    if expected_lf_sha256 and identity["lf_normalized_sha256"] != expected_lf_sha256:
        raise E3V2ValidationError(
            f"LF-normalized SHA-256 mismatch for {identity['path']}: "
            f"{identity['lf_normalized_sha256']} != {expected_lf_sha256}"
        )
    if expected_git_blob_sha1 and identity["git_blob_sha1"] != expected_git_blob_sha1:
        raise E3V2ValidationError(
            f"Git blob identity mismatch for {identity['path']}: "
            f"{identity['git_blob_sha1']} != {expected_git_blob_sha1}"
        )
    return identity


def _load_json(path: Path) -> Any:
    try:
        return json.loads(_lf_bytes(path.read_bytes()).decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise E3V2ValidationError(f"Malformed JSON source {path}: {error}") from error


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = _lf_bytes(path.read_bytes()).decode("utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise E3V2ValidationError(f"Malformed UTF-8 JSONL source {path}: {error}") from error
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise E3V2ValidationError(
                f"Malformed JSONL source {path}:{line_number}: {error}"
            ) from error
        if not isinstance(row, dict):
            raise E3V2ValidationError(f"JSONL row is not an object: {path}:{line_number}")
        records.append(row)
    return records


def normalize_fraction(value: Any) -> str:
    if isinstance(value, bool):
        raise E3V2ValidationError("fraction must not be boolean")
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped == "boundary":
            return "boundary"
        try:
            numeric = float(stripped)
        except ValueError as error:
            raise E3V2ValidationError(f"Unknown fraction: {value!r}") from error
    elif isinstance(value, (int, float)):
        numeric = float(value)
    else:
        raise E3V2ValidationError(f"Unknown fraction type: {type(value).__name__}")
    if not math.isfinite(numeric):
        raise E3V2ValidationError(f"Non-finite fraction: {value!r}")
    for expected in FRACTIONS[:-1]:
        if numeric == float(expected):
            return expected
    raise E3V2ValidationError(f"Unknown fraction: {value!r}")


def _require_string(row: Mapping[str, Any], field: str, label: str, *, nonblank: bool = False) -> str:
    value = row.get(field)
    if not isinstance(value, str) or (nonblank and not value.strip()):
        qualifier = "nonblank string" if nonblank else "string"
        raise E3V2ValidationError(f"{label}.{field} must be a {qualifier}")
    return value


def _require_bool(row: Mapping[str, Any], field: str, label: str) -> bool:
    value = row.get(field)
    if not isinstance(value, bool):
        raise E3V2ValidationError(f"{label}.{field} must be boolean")
    return value


def _validate_e3_payloads(records: Sequence[Mapping[str, Any]]) -> None:
    required_strings = (
        "id",
        "trajectory_id",
        "condition",
        "history_key",
        "unheard_text",
        "strict_unheard_text",
        "local_unheard_in_history_text",
    )
    required_bools = (
        "referenced_unheard",
        "referenced_unheard_strict",
        "local_referenced_unheard",
    )
    for index, row in enumerate(records):
        label = f"e3[{index}]"
        for field in required_strings:
            _require_string(
                row,
                field,
                label,
                nonblank=field in {"id", "trajectory_id", "condition", "history_key"},
            )
        for field in required_bools:
            _require_bool(row, field, label)
        if row["condition"] not in CONDITIONS:
            raise E3V2ValidationError(f"{label}.condition is unknown: {row['condition']!r}")
        normalize_fraction(row.get("fraction"))
        heard = row.get("heard_token_end")
        if isinstance(heard, bool) or not isinstance(heard, int) or heard < 0:
            raise E3V2ValidationError(f"{label}.heard_token_end must be a nonnegative integer")
        replies = row.get("probe_replies")
        if not isinstance(replies, list) or any(not isinstance(reply, str) for reply in replies):
            raise E3V2ValidationError(f"{label}.probe_replies must be a string list")


def _validate_judge_payloads(records: Sequence[Mapping[str, Any]]) -> None:
    for index, row in enumerate(records):
        label = f"judge[{index}]"
        for field in ("id", "trajectory_id", "condition", "target_kind", "prompt_version"):
            _require_string(row, field, label, nonblank=True)
        if row["condition"] not in CONDITIONS:
            raise E3V2ValidationError(f"{label}.condition is unknown: {row['condition']!r}")
        if row["target_kind"] not in TARGETS:
            raise E3V2ValidationError(f"{label}.target_kind is unknown: {row['target_kind']!r}")
        normalize_fraction(row.get("fraction"))
        _require_bool(row, "verdict", label)
        if _require_bool(row, "parse_success", label) is not True:
            raise E3V2ValidationError(f"{label} has parse_success=false")


def index_pairs(
    records: Sequence[dict[str, Any]],
) -> dict[tuple[str, str, str], dict[str, dict[str, Any]]]:
    _validate_e3_payloads(records)
    trajectories_by_id: dict[str, set[str]] = defaultdict(set)
    pairs: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in records:
        dialogue_id = row["id"]
        trajectory_id = row["trajectory_id"]
        fraction = normalize_fraction(row["fraction"])
        trajectories_by_id[dialogue_id].add(trajectory_id)
        key = (dialogue_id, trajectory_id, fraction)
        condition = row["condition"]
        if condition in pairs[key]:
            raise E3V2ValidationError(f"Duplicate E3 condition for {key + (condition,)}")
        pairs[key][condition] = row
    inconsistent = {
        dialogue_id: sorted(trajectory_ids)
        for dialogue_id, trajectory_ids in trajectories_by_id.items()
        if len(trajectory_ids) != 1
    }
    if inconsistent:
        raise E3V2ValidationError(f"Dialogue has multiple trajectory IDs: {inconsistent}")
    for key, pair in pairs.items():
        if set(pair) != set(CONDITIONS):
            raise E3V2ValidationError(f"Incomplete E3 pair {key}: {sorted(pair)}")
        for target_kind, spec in TARGETS.items():
            target_field = spec["target_field"]
            if pair["playback"][target_field] != pair["generation"][target_field]:
                raise E3V2ValidationError(
                    f"Pair conditions disagree on exact {target_kind} target: {key}"
                )
        if pair["playback"]["heard_token_end"] != pair["generation"]["heard_token_end"]:
            raise E3V2ValidationError(f"Pair conditions disagree on heard_token_end: {key}")
    return dict(pairs)


def attach_judgments(
    pairs: Mapping[tuple[str, str, str], Mapping[str, dict[str, Any]]],
    judge_records: Sequence[Mapping[str, Any]],
) -> None:
    _validate_judge_payloads(judge_records)
    lookup: dict[tuple[str, str, str, str, str], Mapping[str, Any]] = {}
    for row in judge_records:
        key = (
            row["id"],
            row["trajectory_id"],
            normalize_fraction(row["fraction"]),
            row["condition"],
            row["target_kind"],
        )
        if key in lookup:
            raise E3V2ValidationError(f"Duplicate judge key: {key}")
        lookup[key] = row
    expected = {
        (*pair_key, condition, target_kind)
        for pair_key in pairs
        for condition in CONDITIONS
        for target_kind in TARGETS
    }
    if set(lookup) != expected:
        missing = sorted(expected - set(lookup))
        extra = sorted(set(lookup) - expected)
        raise E3V2ValidationError(
            f"Judge key mismatch with trajectory-aware join: missing={missing[:3]}, extra={extra[:3]}"
        )
    for pair_key, pair in pairs.items():
        for condition in CONDITIONS:
            pair[condition]["_judge"] = {
                target_kind: dict(lookup[(*pair_key, condition, target_kind)])
                for target_kind in TARGETS
            }


def eligible_pairs(
    pairs: Mapping[tuple[str, str, str], Mapping[str, dict[str, Any]]], target_kind: str
) -> list[dict[str, Any]]:
    target_field = TARGETS[target_kind]["target_field"]
    result = []
    for (dialogue_id, trajectory_id, fraction), pair in sorted(pairs.items()):
        target = pair["playback"][target_field]
        if target.strip():
            result.append(
                {
                    "id": dialogue_id,
                    "trajectory_id": trajectory_id,
                    "dialogue_key": (dialogue_id, trajectory_id),
                    "fraction": fraction,
                    "target": target,
                    "rows": pair,
                }
            )
    return result


def semantic_group_key(label: Mapping[str, Any], target_kind: str) -> tuple[Any, ...]:
    pair = label["rows"]
    common: tuple[Any, ...] = (
        label["id"],
        label["trajectory_id"],
        pair["playback"]["history_key"],
        pair["generation"]["history_key"],
    )
    if target_kind == "fragment":
        common += (pair["playback"]["heard_token_end"],)
    return common + (_sha256(label["target"].encode("utf-8")),)


def _group_invariant(label: Mapping[str, Any], target_kind: str) -> tuple[Any, ...]:
    spec = TARGETS[target_kind]
    pair = label["rows"]
    by_condition = []
    for condition in CONDITIONS:
        row = pair[condition]
        judge = row["_judge"][target_kind]
        by_condition.append(
            (
                condition,
                tuple(row["probe_replies"]),
                row[spec["rule_field"]],
                judge["verdict"],
                judge["prompt_version"],
                judge["parse_success"],
            )
        )
    return (label["target"], tuple(by_condition))


def build_semantic_groups(
    labels: Sequence[dict[str, Any]], target_kind: str
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for label in labels:
        grouped[semantic_group_key(label, target_kind)].append(label)
    result: list[dict[str, Any]] = []
    for key in sorted(grouped, key=repr):
        members = grouped[key]
        invariants = {_group_invariant(member, target_kind) for member in members}
        if len(invariants) != 1:
            fractions = sorted(member["fraction"] for member in members)
            raise E3V2ValidationError(
                "Semantic group violates exact target/probe replies/rule/judge/prompt/parse "
                f"invariant: key={key!r}, fractions={fractions}"
            )
        representative = members[0]
        result.append(
            {
                "group_key": key,
                "dialogue_key": representative["dialogue_key"],
                "fractions": sorted(member["fraction"] for member in members),
                "members": members,
                "representative": representative,
            }
        )
    return result


def _outcome(label: Mapping[str, Any], target_kind: str, metric: str, condition: str) -> float:
    row = label["rows"][condition]
    if metric == "rule":
        return float(row[TARGETS[target_kind]["rule_field"]])
    if metric == "judge":
        return float(row["_judge"][target_kind]["verdict"])
    raise E3V2ValidationError(f"Unknown metric: {metric}")


def _dialogue_components(
    labels: Sequence[Mapping[str, Any]],
    groups: Sequence[Mapping[str, Any]],
    target_kind: str,
    metric: str,
) -> dict[str, Any]:
    dialogue_keys = sorted({tuple(label["dialogue_key"]) for label in labels})
    label_by_dialogue: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    group_by_dialogue: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for label in labels:
        label_by_dialogue[tuple(label["dialogue_key"])].append(label)
    for group in groups:
        group_by_dialogue[tuple(group["dialogue_key"])].append(group)

    label_counts = np.asarray([len(label_by_dialogue[key]) for key in dialogue_keys], dtype=float)
    group_counts = np.asarray([len(group_by_dialogue[key]) for key in dialogue_keys], dtype=float)
    label_sums: dict[str, np.ndarray] = {}
    group_sums: dict[str, np.ndarray] = {}
    for condition in CONDITIONS:
        label_sums[condition] = np.asarray(
            [
                sum(_outcome(label, target_kind, metric, condition) for label in label_by_dialogue[key])
                for key in dialogue_keys
            ],
            dtype=float,
        )
        group_sums[condition] = np.asarray(
            [
                sum(
                    _outcome(group["representative"], target_kind, metric, condition)
                    for group in group_by_dialogue[key]
                )
                for key in dialogue_keys
            ],
            dtype=float,
        )
    if np.any(label_counts <= 0) or np.any(group_counts <= 0):
        raise E3V2ValidationError("Eligible dialogue has no label or semantic group")
    return {
        "dialogue_keys": dialogue_keys,
        "label_counts": label_counts,
        "group_counts": group_counts,
        "label_sums": label_sums,
        "group_sums": group_sums,
    }


def _estimate_from_weights(
    components: Mapping[str, Any], estimand: str, dialogue_weights: np.ndarray
) -> tuple[float, float]:
    label_counts = components["label_counts"]
    group_counts = components["group_counts"]
    label_sums = components["label_sums"]
    group_sums = components["group_sums"]
    if estimand == "label_weighted":
        denominator = float(np.dot(dialogue_weights, label_counts))
        values = tuple(float(np.dot(dialogue_weights, label_sums[c]) / denominator) for c in CONDITIONS)
    elif estimand == "dialogue_weighted":
        denominator = float(dialogue_weights.sum())
        values = tuple(
            float(np.dot(dialogue_weights, label_sums[c] / label_counts) / denominator)
            for c in CONDITIONS
        )
    elif estimand == "unique_semantic_group_weighted":
        denominator = float(np.dot(dialogue_weights, group_counts))
        values = tuple(float(np.dot(dialogue_weights, group_sums[c]) / denominator) for c in CONDITIONS)
    elif estimand == "unique_dialogue_weighted":
        denominator = float(dialogue_weights.sum())
        values = tuple(
            float(np.dot(dialogue_weights, group_sums[c] / group_counts) / denominator)
            for c in CONDITIONS
        )
    else:
        raise E3V2ValidationError(f"Unknown estimand: {estimand}")
    return values[0], values[1]


def bootstrap_estimands(
    labels: Sequence[Mapping[str, Any]],
    groups: Sequence[Mapping[str, Any]],
    target_kind: str,
    metric: str,
    *,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    if repeats <= 0:
        raise E3V2ValidationError("bootstrap repeats must be positive")
    components = _dialogue_components(labels, groups, target_kind, metric)
    dialogue_count = len(components["dialogue_keys"])
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, dialogue_count, size=(repeats, dialogue_count))
    multiplicities = np.apply_along_axis(
        lambda indices: np.bincount(indices, minlength=dialogue_count), 1, sampled
    ).astype(float)
    unit_weights = np.ones(dialogue_count, dtype=float)
    result: dict[str, Any] = {}
    for estimand in ESTIMANDS:
        playback, generation = _estimate_from_weights(components, estimand, unit_weights)
        playback_draws = np.empty(repeats, dtype=float)
        generation_draws = np.empty(repeats, dtype=float)
        for index, weights in enumerate(multiplicities):
            playback_draws[index], generation_draws[index] = _estimate_from_weights(
                components, estimand, weights
            )
        effect_draws = generation_draws - playback_draws
        if not np.all(np.isfinite(effect_draws)):
            raise E3V2ValidationError(f"Non-finite bootstrap draw for {target_kind}/{metric}/{estimand}")
        result[estimand] = {
            "conditions": {
                "playback": {"rate": playback},
                "generation": {"rate": generation},
            },
            "generation_minus_playback": generation - playback,
            "difference_95_ci": [
                float(value) for value in np.quantile(effect_draws, [0.025, 0.975])
            ],
            "bootstrap_condition_rate_95_ci": {
                "playback": [
                    float(value) for value in np.quantile(playback_draws, [0.025, 0.975])
                ],
                "generation": [
                    float(value) for value in np.quantile(generation_draws, [0.025, 0.975])
                ],
            },
        }
    return result


def confusion_table(observations: Sequence[tuple[bool, bool]]) -> dict[str, Any]:
    counts = {
        "rule_positive_judge_positive": 0,
        "rule_positive_judge_negative": 0,
        "rule_negative_judge_positive": 0,
        "rule_negative_judge_negative": 0,
    }
    for rule, judge in observations:
        if rule and judge:
            counts["rule_positive_judge_positive"] += 1
        elif rule:
            counts["rule_positive_judge_negative"] += 1
        elif judge:
            counts["rule_negative_judge_positive"] += 1
        else:
            counts["rule_negative_judge_negative"] += 1
    total = len(observations)
    agreement = counts["rule_positive_judge_positive"] + counts["rule_negative_judge_negative"]
    return {"n": total, **counts, "agreement": agreement, "agreement_rate": agreement / total if total else None}


def confusion_by_condition(
    units: Sequence[Mapping[str, Any]], target_kind: str, *, representative: bool
) -> dict[str, Any]:
    observations: dict[str, list[tuple[bool, bool]]] = {condition: [] for condition in CONDITIONS}
    for unit in units:
        label = unit["representative"] if representative else unit
        for condition in CONDITIONS:
            row = label["rows"][condition]
            observations[condition].append(
                (
                    bool(row[TARGETS[target_kind]["rule_field"]]),
                    bool(row["_judge"][target_kind]["verdict"]),
                )
            )
    pooled = [item for condition in CONDITIONS for item in observations[condition]]
    return {
        "playback": confusion_table(observations["playback"]),
        "generation": confusion_table(observations["generation"]),
        "pooled": confusion_table(pooled),
    }


def _count_summary(
    labels: Sequence[Mapping[str, Any]], groups: Sequence[Mapping[str, Any]], target_kind: str, metric: str
) -> dict[str, int]:
    playback = int(sum(_outcome(label, target_kind, metric, "playback") for label in labels))
    generation = int(sum(_outcome(label, target_kind, metric, "generation") for label in labels))
    return {
        "eligible_labels_per_condition": len(labels),
        "playback_positive": playback,
        "generation_positive": generation,
        "positive_difference": generation - playback,
        "eligible_dialogues": len({tuple(label["dialogue_key"]) for label in labels}),
        "semantic_groups": len(groups),
    }


def _assert_equal(actual: Any, expected: Any, name: str, checks: dict[str, Any]) -> None:
    checks[name] = {"actual": actual, "expected": expected, "passed": actual == expected}
    if actual != expected:
        raise E3V2ValidationError(f"Formal hard gate failed for {name}: {actual!r} != {expected!r}")


def formal_hard_gates(
    records: Sequence[Mapping[str, Any]],
    pairs: Mapping[tuple[str, str, str], Mapping[str, Mapping[str, Any]]],
    judge_records: Sequence[Mapping[str, Any]],
    target_data: Mapping[str, Mapping[str, Any]],
    construction: Mapping[str, Any],
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    _assert_equal(len(records), FORMAL_EXPECTED["raw_records"], "raw_records", checks)
    _assert_equal(len(pairs), FORMAL_EXPECTED["paired_labels"], "paired_labels", checks)
    _assert_equal(len(judge_records), FORMAL_EXPECTED["judge_records"], "judge_records", checks)
    _assert_equal(
        len({(row["id"], row["trajectory_id"]) for row in records}),
        FORMAL_EXPECTED["raw_dialogues"],
        "raw_dialogues",
        checks,
    )
    for field in (
        "playback_records",
        "playback_local_unheard_empty",
        "playback_local_reference_positives",
    ):
        _assert_equal(construction[field], FORMAL_EXPECTED[field], field, checks)
    for target_kind in TARGETS:
        expected = FORMAL_EXPECTED[target_kind]
        labels = target_data[target_kind]["labels"]
        groups = target_data[target_kind]["groups"]
        _assert_equal(len(labels), expected["eligible_labels"], f"{target_kind}.eligible_labels", checks)
        _assert_equal(
            len({tuple(label["dialogue_key"]) for label in labels}),
            expected["eligible_dialogues"],
            f"{target_kind}.eligible_dialogues",
            checks,
        )
        _assert_equal(len(groups), expected["semantic_groups"], f"{target_kind}.semantic_groups", checks)
        for metric in ("rule", "judge"):
            counts = _count_summary(labels, groups, target_kind, metric)
            for field in ("playback_positive", "generation_positive", "positive_difference"):
                _assert_equal(
                    counts[field], expected[metric][field], f"{target_kind}.{metric}.{field}", checks
                )
    return {"passed": True, "assertions": checks}


def _source_artifacts(
    e3_run_dir: Path, judge_path: Path, *, formal: bool
) -> dict[str, dict[str, Any]]:
    paths = {
        "e3_records": e3_run_dir / "records.jsonl",
        "e3_manifest": e3_run_dir / "manifest.json",
        "judge_records": judge_path,
        "judge_manifest": judge_path.parent / "manifest.json",
        "superseded_v1": e3_run_dir / "analysis_metric_specific_eligibility_v1.json",
    }
    identities: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        expected = FORMAL_SOURCE_IDENTITIES[name] if formal else {}
        if formal and _repo_relative(path) != expected["relative_path"]:
            raise E3V2ValidationError(
                f"Formal {name} path mismatch: {_repo_relative(path)} != {expected['relative_path']}"
            )
        identities[name] = artifact_identity(
            path,
            expected_lf_sha256=expected.get("lf_sha256"),
            expected_git_blob_sha1=expected.get("git_blob_sha1"),
        )
    return identities


def _validate_manifests(e3_run_dir: Path, judge_path: Path, identities: Mapping[str, Any]) -> dict[str, Any]:
    e3_manifest = _load_json(e3_run_dir / "manifest.json")
    judge_manifest = _load_json(judge_path.parent / "manifest.json")
    if e3_manifest.get("run_id") != FORMAL_E3_RUN_ID:
        raise E3V2ValidationError(f"Unexpected E3 run ID: {e3_manifest.get('run_id')!r}")
    if judge_manifest.get("run_id") != FORMAL_JUDGE_RUN_ID:
        raise E3V2ValidationError(f"Unexpected judge run ID: {judge_manifest.get('run_id')!r}")
    source_hash = judge_manifest.get("config", {}).get("source_sha256")
    if source_hash != identities["e3_records"]["lf_normalized_sha256"]:
        raise E3V2ValidationError(
            "Judge manifest source hash does not match LF-normalized E3 records identity"
        )
    return {
        "e3_run_id": e3_manifest["run_id"],
        "e3_source_git_commit": e3_manifest.get("git", {}).get("commit"),
        "e3_source_config_hash": e3_manifest.get("config_hash"),
        "judge_run_id": judge_manifest["run_id"],
        "judge_source_git_commit": judge_manifest.get("git", {}).get("commit"),
        "judge_source_config_hash": judge_manifest.get("config_hash"),
        "judge_prompt_version": judge_manifest.get("config", {}).get("prompt_version"),
        "judge_model_identity_hash": (
            judge_manifest.get("config", {}).get("model_identity", {}).get("identity_hash")
        ),
    }


def build_analysis(
    e3_run_dir: Path,
    judge_path: Path,
    *,
    repeats: int,
    seed: int,
    formal: bool,
) -> dict[str, Any]:
    if formal and (repeats != FORMAL_REPEATS or seed != FORMAL_SEED):
        raise E3V2ValidationError(
            f"Formal bootstrap is frozen at repeats={FORMAL_REPEATS}, seed={FORMAL_SEED}"
        )
    source_identities = _source_artifacts(e3_run_dir, judge_path, formal=formal)
    manifest_summary = _validate_manifests(e3_run_dir, judge_path, source_identities) if formal else {}
    records = _load_jsonl(e3_run_dir / "records.jsonl")
    judge_records = _load_jsonl(judge_path)
    pairs = index_pairs(records)
    attach_judgments(pairs, judge_records)

    playback_records = [row for row in records if row["condition"] == "playback"]
    construction = {
        "playback_records": len(playback_records),
        "playback_local_unheard_empty": sum(
            not row["local_unheard_in_history_text"].strip() for row in playback_records
        ),
        "playback_local_reference_positives": sum(
            bool(row["local_referenced_unheard"]) for row in playback_records
        ),
    }
    if construction["playback_local_unheard_empty"] != construction["playback_records"]:
        raise E3V2ValidationError("Playback construction has nonempty local unheard text")
    if construction["playback_local_reference_positives"] != 0:
        raise E3V2ValidationError("Playback construction has local fragment reference positives")

    target_data: dict[str, dict[str, Any]] = {}
    for target_kind in TARGETS:
        labels = eligible_pairs(pairs, target_kind)
        groups = build_semantic_groups(labels, target_kind)
        target_data[target_kind] = {"labels": labels, "groups": groups}

    hard_gates = (
        formal_hard_gates(records, pairs, judge_records, target_data, construction)
        if formal
        else {"passed": None, "assertions": {}, "mode": "non-formal"}
    )
    targets_output: dict[str, Any] = {}
    confusion_output: dict[str, Any] = {
        "label_level": {},
        "unique_semantic_group_level": {},
    }
    for target_kind, spec in TARGETS.items():
        labels = target_data[target_kind]["labels"]
        groups = target_data[target_kind]["groups"]
        by_fraction = {
            fraction: sum(label["fraction"] == fraction for label in labels) for fraction in FRACTIONS
        }
        metrics = {}
        for metric in ("rule", "judge"):
            metrics[metric] = {
                "legacy_label_counts": _count_summary(labels, groups, target_kind, metric),
                "estimands": bootstrap_estimands(
                    labels,
                    groups,
                    target_kind,
                    metric,
                    repeats=repeats,
                    seed=seed,
                ),
            }
        targets_output[target_kind] = {
            "target_field": spec["target_field"],
            "eligibility": f"nonblank {spec['target_field']}",
            "eligible_labels": len(labels),
            "empty_target_labels": len(pairs) - len(labels),
            "eligible_dialogues": len({tuple(label["dialogue_key"]) for label in labels}),
            "eligible_labels_by_fraction": by_fraction,
            "semantic_groups": len(groups),
            "duplicate_labels_removed": len(labels) - len(groups),
            "semantic_group_size_distribution": {
                str(size): sum(len(group["members"]) == size for group in groups)
                for size in sorted({len(group["members"]) for group in groups})
            },
            "dedup_key_fields": list(spec["dedup_key"]),
            "group_invariant_fields": [
                "exact target",
                "probe_replies by condition",
                "rule verdict by condition",
                "judge verdict by condition",
                "judge prompt_version by condition",
                "judge parse_success by condition",
            ],
            "metrics": metrics,
        }
        confusion_output["label_level"][target_kind] = confusion_by_condition(
            labels, target_kind, representative=False
        )
        confusion_output["unique_semantic_group_level"][target_kind] = confusion_by_condition(
            groups, target_kind, representative=True
        )

    parameters = {
        "analysis_version": ANALYSIS_VERSION,
        "bootstrap_repeats": repeats,
        "seed": seed,
        "effect_direction": "generation minus playback",
        "random_generator": "numpy.random.default_rng",
        "interval": "percentile 95%",
        "targets": {
            target: {
                "eligibility": f"nonblank {spec['target_field']}",
                "dedup_key_fields": list(spec["dedup_key"]),
            }
            for target, spec in TARGETS.items()
        },
    }
    analyzer_identity = artifact_identity(Path(__file__).resolve())
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_version": ANALYSIS_VERSION,
        "post_review_sensitivity": POST_REVIEW_SENSITIVITY,
        "generated_at_utc": utc_now(),
        "method": {
            "effect_direction": "generation minus playback",
            "estimands": {
                "label_weighted": "Each eligible fraction label receives equal weight.",
                "dialogue_weighted": "Each eligible dialogue receives equal weight after averaging its eligible labels.",
                "unique_semantic_group_weighted": "Each exact target-specific deduplicated semantic group receives equal weight.",
                "unique_dialogue_weighted": "Each eligible dialogue receives equal weight after averaging its unique semantic groups.",
            },
            "bootstrap": {
                "method": (
                    "Paired dialogue-cluster percentile bootstrap. Sort (id, trajectory_id) "
                    "clusters, draw the same number with replacement using "
                    "numpy.random.default_rng(seed), retain playback/generation pairing and "
                    "all within-dialogue labels/groups, then recompute each estimand."
                ),
                "random_generator": "numpy.random.default_rng",
                "seed": seed,
                "repeats": repeats,
                "interval": "percentile 95%",
                "numpy_quantile_method": "linear (NumPy default)",
            },
        },
        "design": {
            "raw_records": len(records),
            "paired_labels": len(pairs),
            "raw_dialogues": len({(row["id"], row["trajectory_id"]) for row in records}),
            "judge_records": len(judge_records),
            "conditions": list(CONDITIONS),
            "fractions": list(FRACTIONS),
        },
        "construction_checks": construction,
        "targets": targets_output,
        "confusion_tables_rule_vs_judge": confusion_output,
        "formal_hard_gates": hard_gates,
        "provenance": {
            "source_access": "read-only; sources were not rewritten",
            "parameters": parameters,
            "parameters_sha256": config_hash(parameters),
            "analyzer": analyzer_identity,
            "source_artifacts": source_identities,
            "manifest_summary": manifest_summary,
            "supersedes": {
                "analysis_version": "metric-specific-eligibility-v1",
                "artifact": source_identities["superseded_v1"],
                "reason": "Adds explicit weighting estimands and exact target-specific semantic deduplication.",
            },
        },
        "excluded_records": [],
    }


def _synthetic_sources() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    judges: list[dict[str, Any]] = []
    specifications = {
        "d0": [
            ("0.25", "target-a", "strict-a", True, True),
            ("0.5", "target-a", "strict-a", True, True),
            ("0.75", "target-b", "", True, False),
        ],
        "d1": [("boundary", "target-c", "strict-c", False, False)],
    }
    for dialogue_id, labels in specifications.items():
        trajectory = f"trajectory-{dialogue_id}"
        for fraction, fragment, proxy, playback_rule, generation_rule in labels:
            for condition, rule in (("playback", playback_rule), ("generation", generation_rule)):
                row = {
                    "id": dialogue_id,
                    "trajectory_id": trajectory,
                    "fraction": fraction,
                    "condition": condition,
                    "history_key": f"{dialogue_id}-{condition}-history",
                    "heard_token_end": 2 if fragment == "target-a" else 3,
                    "unheard_text": fragment,
                    "strict_unheard_text": proxy,
                    "probe_replies": [f"reply-{dialogue_id}-{condition}-{fragment}"],
                    "referenced_unheard": rule,
                    "referenced_unheard_strict": rule,
                    "local_unheard_in_history_text": "",
                    "local_referenced_unheard": False,
                }
                records.append(row)
                for target_kind in TARGETS:
                    judges.append(
                        {
                            "id": dialogue_id,
                            "trajectory_id": trajectory,
                            "fraction": fraction,
                            "condition": condition,
                            "target_kind": target_kind,
                            "verdict": rule,
                            "prompt_version": "test-prompt",
                            "parse_success": True,
                        }
                    )
    return records, judges


def _expect_failure(function: Callable[[], Any], label: str) -> None:
    try:
        function()
    except (E3V2ValidationError, ValueError):
        return
    raise AssertionError(f"Self-test expected failure: {label}")


def run_self_test() -> None:
    assert normalize_fraction(0.25) == "0.25"
    assert normalize_fraction(" 0.2500 ") == "0.25"
    assert normalize_fraction("BOUNDARY") == "boundary"
    _expect_failure(lambda: normalize_fraction(True), "boolean fraction")
    _expect_failure(lambda: normalize_fraction("0.2"), "unknown fraction")

    records, judges = _synthetic_sources()
    pairs = index_pairs(records)
    attach_judgments(pairs, judges)
    fragment_labels = eligible_pairs(pairs, "fragment")
    proxy_labels = eligible_pairs(pairs, "proxy")
    fragment_groups = build_semantic_groups(fragment_labels, "fragment")
    proxy_groups = build_semantic_groups(proxy_labels, "proxy")
    assert len(fragment_labels) == 4 and len(fragment_groups) == 3
    assert len(proxy_labels) == 3 and len(proxy_groups) == 2

    # Three positive labels in d0 and one negative in d1 make label and dialogue
    # weighting differ; the duplicate target-a labels also exercise exact dedup.
    first = bootstrap_estimands(
        fragment_labels, fragment_groups, "fragment", "rule", repeats=100, seed=17
    )
    second = bootstrap_estimands(
        fragment_labels, fragment_groups, "fragment", "rule", repeats=100, seed=17
    )
    assert first == second
    assert first["label_weighted"]["conditions"]["playback"]["rate"] == 0.75
    assert first["dialogue_weighted"]["conditions"]["playback"]["rate"] == 0.5
    assert math.isclose(
        first["unique_semantic_group_weighted"]["conditions"]["playback"]["rate"], 2 / 3
    )
    assert first["unique_dialogue_weighted"]["conditions"]["playback"]["rate"] == 0.5

    malformed_judges = [dict(row) for row in judges]
    malformed_judges[0]["parse_success"] = False
    clean_pairs = index_pairs(records)
    _expect_failure(
        lambda: attach_judgments(clean_pairs, malformed_judges), "malformed judge parse"
    )

    wrong_trajectory = [dict(row) for row in judges]
    wrong_trajectory[0]["trajectory_id"] = "wrong-trajectory"
    clean_pairs = index_pairs(records)
    _expect_failure(
        lambda: attach_judgments(clean_pairs, wrong_trajectory), "trajectory-aware judge join"
    )

    inconsistent_records = [dict(row) for row in records]
    inconsistent_records[0]["trajectory_id"] = "other-trajectory"
    _expect_failure(lambda: index_pairs(inconsistent_records), "raw trajectory inconsistency")

    # Keep a duplicate group key but alter one exact invariant field.
    group_bad_records = [dict(row) for row in records]
    duplicate_index = next(
        index
        for index, row in enumerate(group_bad_records)
        if row["id"] == "d0" and normalize_fraction(row["fraction"]) == "0.5" and row["condition"] == "playback"
    )
    group_bad_records[duplicate_index]["probe_replies"] = ["inconsistent reply"]
    bad_pairs = index_pairs(group_bad_records)
    attach_judgments(bad_pairs, judges)
    _expect_failure(
        lambda: build_semantic_groups(eligible_pairs(bad_pairs, "fragment"), "fragment"),
        "semantic group invariant",
    )

    with tempfile.TemporaryDirectory() as temporary:
        lf_path = Path(temporary) / "source.jsonl"
        crlf_path = Path(temporary) / "source-crlf.jsonl"
        lf = b'{"x":1}\n{"x":2}\n'
        lf_path.write_bytes(lf)
        crlf_path.write_bytes(lf.replace(b"\n", b"\r\n"))
        lf_identity = artifact_identity(lf_path)
        crlf_identity = artifact_identity(crlf_path)
        assert lf_identity["lf_normalized_sha256"] == crlf_identity["lf_normalized_sha256"]
        assert lf_identity["git_blob_sha1"] == crlf_identity["git_blob_sha1"]
        assert not lf_identity["line_ending_normalization_applied"]
        assert crlf_identity["line_ending_normalization_applied"]

    print("E3 analyze_e3_v2 self-test PASS")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute E3 v2 weighting and exact-semantic-deduplication sensitivity outputs."
    )
    parser.add_argument("--e3-run-dir", type=Path)
    parser.add_argument("--judge-records", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--bootstrap-repeats", type=int, default=FORMAL_REPEATS)
    parser.add_argument("--seed", type=int, default=FORMAL_SEED)
    parser.add_argument("--non-formal", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.e3_run_dir is None or args.judge_records is None:
        raise SystemExit("--e3-run-dir and --judge-records are required unless --self-test is used")
    e3_run_dir = args.e3_run_dir.resolve()
    judge_path = args.judge_records.resolve()
    output = (args.out or e3_run_dir / FORMAL_OUTPUT_NAME).resolve()
    sidecar = output.with_suffix(".sha256")
    formal = not args.non_formal
    if formal:
        expected_output = e3_run_dir / FORMAL_OUTPUT_NAME
        expected_sidecar = e3_run_dir / FORMAL_SIDECAR_NAME
        if output != expected_output or sidecar != expected_sidecar:
            raise E3V2ValidationError(
                f"Formal outputs are frozen at {expected_output} and {expected_sidecar}"
            )
    for path in (output, sidecar):
        if path.exists():
            raise FileExistsError(f"Versioned analysis artifact is immutable and exists: {path}")
    result = build_analysis(
        e3_run_dir,
        judge_path,
        repeats=args.bootstrap_repeats,
        seed=args.seed,
        formal=formal,
    )
    atomic_write_json(output, result)
    output_hash = _sha256(output.read_bytes())
    atomic_write_text(sidecar, f"{output_hash}  {output.name}\n")
    print(output)
    print(sidecar)
    print(output_hash)


if __name__ == "__main__":
    main()
