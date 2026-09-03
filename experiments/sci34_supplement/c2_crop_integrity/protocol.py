"""Frozen protocol for the independent C2 v3 crop-integrity addendum."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.sci34_supplement.common import canonical_json, config_hash, sha256_file
from experiments.sci34_supplement.c2_equivalence.protocol import (
    CONTEXT_CLASSES,
    CONTEXT_TARGETS,
    EXPECTED_DTYPE,
    EXPECTED_MODEL_ARCHITECTURE,
    EXPECTED_MODEL_ARTIFACT_HASH,
    EXPECTED_MODEL_TYPE,
    SCENARIOS,
    SYSTEM_PROMPT,
    TERMINATIONS,
    CaseSpec,
    validate_cases as validate_v2_cases,
)


SCHEMA_VERSION = 1
PROTOCOL_VERSION = 3
EXPERIMENT = "c2_crop_integrity"
FORMAL_CASE_COUNT = 24
FORMAL_CROP_EVENT_COUNT = 27
EXPECTED_SECOND_CROP_CASES = (
    "c2_08_second_short_cap",
    "c2_16_second_medium_eos",
    "c2_24_second_long_max",
)
SOURCE_CASES_RELATIVE = "../c2_equivalence/cases.json"
EXPECTED_CASES_SHA256 = "acda9afb83ff393d222fc06eb433c1a307dba1eb213fcbbbedfdb860f7500696"
PRIOR_V2_RUN_ID = "c2eq_5c56b014_20260903T040829Z"
PRIOR_V2_EVIDENCE_ROLE = (
    "immutable prior termination/EOT and v2 equivalence evidence; provenance only, "
    "not read or required by the v3 runtime"
)


@dataclass(frozen=True)
class ProtocolConfig:
    sessions: int = 1
    statistical_repeats: int = 0
    cases: int = FORMAL_CASE_COUNT
    crop_events: int = FORMAL_CROP_EVENT_COUNT
    fixture_append: str = "generate_accumulating with controlled non-EOT token IDs"
    production_crop: str = "StreamLLMInference.crop_to_token"
    oracle_crop: str = "clone pre-crop retained-prefix K/V tensors without crop_to_token"
    recovery: str = "production role APIs versus identical direct token-ID model forwards"
    gates: str = "bitwise/exact only"
    clean_reprefill_gate: bool = False
    termination_probe_rerun: bool = False
    dtype: str = "bfloat16"
    system_prompt: str = SYSTEM_PROMPT

    def validate(self) -> None:
        if self != ProtocolConfig():
            raise ValueError("C2 v3 formal protocol is frozen")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.update(
            {
                "protocol_version": PROTOCOL_VERSION,
                "expected_model_artifact_hash": EXPECTED_MODEL_ARTIFACT_HASH,
                "expected_model_type": EXPECTED_MODEL_TYPE,
                "expected_model_architecture": EXPECTED_MODEL_ARCHITECTURE,
                "expected_dtype": EXPECTED_DTYPE,
                "source_cases_relative": SOURCE_CASES_RELATIVE,
                "expected_cases_sha256": EXPECTED_CASES_SHA256,
                "prior_v2_run": {
                    "run_id": PRIOR_V2_RUN_ID,
                    "role": PRIOR_V2_EVIDENCE_ROLE,
                    "runtime_dependency": False,
                },
                "context_targets": list(CONTEXT_TARGETS),
                "context_classes": list(CONTEXT_CLASSES),
                "scenarios": list(SCENARIOS),
                "terminations": list(TERMINATIONS),
                "second_crop_case_ids": list(EXPECTED_SECOND_CROP_CASES),
                "claim": (
                    "crop/truncation integrity and matched recovery determinism for the frozen "
                    "Qwen2-7B snapshot/backend"
                ),
                "excluded_claims": [
                    "clean-reprefill numerical equivalence",
                    "cross-model or cross-backend correctness",
                    "online ASR/TTS/player correctness",
                ],
            }
        )
        return value


def expected_event_grid(cases: Sequence[CaseSpec]) -> list[tuple[str, str]]:
    events: list[tuple[str, str]] = []
    for case in cases:
        events.append((case.id, "crop_1"))
        if case.second_crop_fraction is not None:
            events.append((case.id, "crop_2"))
    return events


def validate_cases(value: Any, *, formal: bool) -> list[CaseSpec]:
    cases = validate_v2_cases(value, formal=formal, expected_count=FORMAL_CASE_COUNT if formal else None)
    second = tuple(case.id for case in cases if case.second_crop_fraction is not None)
    if formal and second != EXPECTED_SECOND_CROP_CASES:
        raise ValueError(f"Second-crop grid differs: {second!r}")
    if formal and len(expected_event_grid(cases)) != FORMAL_CROP_EVENT_COUNT:
        raise ValueError("Formal crop-event grid must contain exactly 27 events")
    return cases


def load_cases(path: Path, *, formal: bool) -> list[CaseSpec]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if formal and sha256_file(path) != EXPECTED_CASES_SHA256:
        raise ValueError("Campaign cases.json is not the exact frozen 24-case copy")
    return validate_cases(json.loads(path.read_text(encoding="utf-8")), formal=formal)


def protocol_identity(cases_path: Path) -> dict[str, Any]:
    protocol = ProtocolConfig()
    protocol.validate()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "protocol": protocol.to_dict(),
        "cases": {
            "path": str(cases_path.resolve()),
            "sha256": sha256_file(cases_path),
            "expected_sha256": EXPECTED_CASES_SHA256,
            "source_relative": SOURCE_CASES_RELATIVE,
        },
    }
    payload["identity_hash"] = config_hash(payload)
    return payload


def assert_exact_identity(existing: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    if existing.get("identity_hash") != expected.get("identity_hash"):
        raise ValueError(f"{label} identity hash mismatch")
    if canonical_json(existing) != canonical_json(expected):
        raise ValueError(f"{label} payload differs despite matching identity hash")
