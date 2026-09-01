"""Frozen protocol and model-free validation helpers for confirmatory E1/E2.

This module is the single source of truth for condition identifiers and formal
session dimensions.  It intentionally imports only the standard library and
shared supplement helpers.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.sci34_supplement.common import canonical_json, config_hash, sha256_file


SCHEMA_VERSION = 1
EXPERIMENT = "e1e2_confirmatory"
FORMAL_SESSION_COUNT = 5
FORMAL_DIALOGUE_COUNT = 100
SYSTEM_PROMPT = "You are a helpful assistant. Reply in English."
THRESHOLDS: tuple[float, ...] = (
    0.0052,
    0.1979,
    0.3906,
    0.5833,
    0.7760,
    0.8500,
    0.9200,
    0.9688,
)
CONFIRMATORY_THRESHOLD = 0.9200
SYSTEM_A = "system_a_full_prefill"
NEVER_SPECULATE = "b_never_speculate"


def threshold_condition(threshold: float) -> str:
    return f"b_threshold_{threshold:.4f}"


THRESHOLD_CONDITIONS: tuple[str, ...] = tuple(
    threshold_condition(value) for value in THRESHOLDS
)
CONDITIONS: tuple[str, ...] = (SYSTEM_A, *THRESHOLD_CONDITIONS, NEVER_SPECULATE)
CONFIRMATORY_CONDITION = threshold_condition(CONFIRMATORY_THRESHOLD)
B_CONDITIONS: tuple[str, ...] = (*THRESHOLD_CONDITIONS, NEVER_SPECULATE)
WARMUP_PATHS: tuple[str, ...] = (
    "full_prefill",
    "survived_speculation",
    "invalidated_crop",
    "never_speculate",
    "sentence_chunker",
)


@dataclass(frozen=True)
class ProtocolConfig:
    max_new_tokens: int = 32
    spec_chunk: int = 12
    warmup_repeats: int = 3
    decode: str = "greedy"
    batch_size: int = 1
    system_prompt: str = SYSTEM_PROMPT
    thresholds: tuple[float, ...] = THRESHOLDS
    confirmatory_threshold: float = CONFIRMATORY_THRESHOLD
    condition_order_seed: int = 20260901

    def validate(self) -> None:
        if self.max_new_tokens != 32:
            raise ValueError("Confirmatory protocol freezes max_new_tokens=32")
        if self.spec_chunk != 12:
            raise ValueError("Confirmatory protocol freezes spec_chunk=12")
        if self.decode != "greedy" or self.batch_size != 1:
            raise ValueError("Confirmatory protocol requires greedy batch-size-one decoding")
        if tuple(self.thresholds) != THRESHOLDS:
            raise ValueError("Confirmatory threshold grid is frozen")
        if self.confirmatory_threshold != CONFIRMATORY_THRESHOLD:
            raise ValueError("Confirmatory threshold is frozen at 0.92")
        if self.warmup_repeats < 3:
            raise ValueError("At least three warmups per path are required")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["thresholds"] = list(self.thresholds)
        payload["conditions"] = list(CONDITIONS)
        payload["confirmatory_condition"] = CONFIRMATORY_CONDITION
        payload["warmup_paths"] = list(WARMUP_PATHS)
        payload["endpoint_semantics"] = (
            "controlled synchronous oracle acceptance after final-segment processing; "
            "not immediate at final segment arrival"
        )
        payload["primary_latency_semantics"] = (
            "last_segment_arrival_ns to actual first_token_ready_ns"
        )
        payload["ttft_eff_semantics"] = (
            "oracle-ready policy upper bound after endpoint acceptance; zero iff a valid "
            "ready candidate survives at acceptance"
        )
        return payload


@dataclass(frozen=True)
class InputRow:
    id: str
    full_text: str
    segments: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "full_text": self.full_text, "segments": list(self.segments)}


def validate_input_rows(
    value: Any,
    *,
    formal: bool,
    expected_count: int | None = None,
) -> list[InputRow]:
    if not isinstance(value, list) or not value:
        raise ValueError("Input must be a non-empty JSON list")
    if expected_count is not None and len(value) != expected_count:
        raise ValueError(f"Expected {expected_count} inputs, found {len(value)}")
    rows: list[InputRow] = []
    seen: set[str] = set()
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise ValueError(f"Input row {index} is not an object")
        sample_id = str(raw.get("id", "")).strip()
        full_text = raw.get("full_text")
        segments = raw.get("segments")
        if not sample_id or sample_id in seen:
            raise ValueError(f"Missing or duplicate input id: {sample_id!r}")
        if formal and sample_id.lower().startswith(("fx", "fixture", "smoke")):
            raise ValueError(f"Formal input contains fixture-like id: {sample_id}")
        if not isinstance(full_text, str) or not full_text.strip():
            raise ValueError(f"Input {sample_id} has empty full_text")
        if not isinstance(segments, list) or len(segments) < 2 or not all(
            isinstance(segment, str) and bool(segment) for segment in segments
        ):
            raise ValueError(f"Input {sample_id} must have at least two non-empty segments")
        if "".join(segments) != full_text:
            raise ValueError(f"Input {sample_id} segments are not lossless")
        seen.add(sample_id)
        rows.append(InputRow(sample_id, full_text, tuple(segments)))
    return rows


def load_input_rows(path: Path, *, formal: bool) -> list[InputRow]:
    if not path.exists():
        raise FileNotFoundError(path)
    return validate_input_rows(
        json.loads(path.read_text(encoding="utf-8")),
        formal=formal,
        expected_count=FORMAL_DIALOGUE_COUNT if formal else None,
    )


def threshold_for_condition(condition: str) -> float | None:
    if condition in (SYSTEM_A, NEVER_SPECULATE):
        return None
    if condition not in THRESHOLD_CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    return THRESHOLDS[THRESHOLD_CONDITIONS.index(condition)]


def balanced_condition_order(
    *, session_index: int, dialogue_index: int, seed: int = 20260901
) -> list[str]:
    """Return a deterministic cyclic Latin order.

    Every block of ten dialogues places every condition once in every ordinal
    position.  The session offset prevents the first dialogue from sharing the
    same first condition across all five independently launched sessions.
    """
    if not 0 <= session_index < FORMAL_SESSION_COUNT:
        raise ValueError(f"session_index must be in [0, {FORMAL_SESSION_COUNT})")
    if dialogue_index < 0:
        raise ValueError("dialogue_index must be non-negative")
    keyed = sorted(
        CONDITIONS,
        key=lambda condition: config_hash({"seed": seed, "condition": condition}),
    )
    shift = (dialogue_index + 2 * session_index) % len(keyed)
    return keyed[shift:] + keyed[:shift]


def condition_order_balance(
    *, dialogue_count: int, session_index: int, seed: int = 20260901
) -> dict[str, list[int]]:
    counts = {condition: [0] * len(CONDITIONS) for condition in CONDITIONS}
    for dialogue_index in range(dialogue_count):
        for ordinal, condition in enumerate(
            balanced_condition_order(
                session_index=session_index, dialogue_index=dialogue_index, seed=seed
            )
        ):
            counts[condition][ordinal] += 1
    return counts


def artifact_identity(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def campaign_identity_payload(
    *,
    protocol: ProtocolConfig,
    input_path: Path,
    trigger_cache_path: Path,
    model_identity: Mapping[str, Any],
    runtime_kind: str,
    device: str,
    trigger_model_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    protocol.validate()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "protocol": protocol.to_dict(),
        "input": artifact_identity(input_path),
        "trigger_cache": artifact_identity(trigger_cache_path),
        "model_identity": dict(model_identity),
        "trigger_model_identity": dict(trigger_model_identity or {}),
        "runtime": runtime_kind,
        "device": device,
    }
    payload["identity_hash"] = config_hash(payload)
    return payload


def assert_same_identity(existing: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    old_hash = existing.get("identity_hash")
    new_hash = expected.get("identity_hash")
    if old_hash != new_hash:
        raise ValueError(f"{label} identity mismatch: {old_hash} != {new_hash}")
    if canonical_json(existing) != canonical_json(expected):
        raise ValueError(f"{label} payload differs despite matching identity_hash")
