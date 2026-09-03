"""Versioned crossed/product-bootstrap reanalysis of accepted confirmatory E1/E2.

The formal path is intentionally bound to the accepted five-session campaign.
It reads accepted artifacts without modifying them and creates only analysis_v2.json
and analysis_v2.sha256 in the campaign root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Mapping, Sequence

from experiments.sci34_supplement.common import (
    ROOT,
    atomic_write_json,
    atomic_write_text,
    canonical_json,
    describe,
    sha256_bytes,
    utc_now,
)
from experiments.sci34_supplement.e1e2_confirmatory.analyze import _ci
from experiments.sci34_supplement.e1e2_confirmatory.protocol import (
    B_CONDITIONS,
    CONDITIONS,
    CONFIRMATORY_CONDITION,
    FORMAL_DIALOGUE_COUNT,
    FORMAL_SESSION_COUNT,
    NEVER_SPECULATE,
    SYSTEM_A,
)
from experiments.sci34_supplement.e1e2_confirmatory.validate import validate_grid


SCHEMA_VERSION = 2
ANALYSIS_VERSION = "crossed-product-bootstrap-v2"
FORMAL_REPEATS = 10_000
FORMAL_SEED = 20260901
FORMAL_CAMPAIGN_ID = "e1e2c_b8c758b_20260901T173306Z"
FORMAL_METHOD = (
    "Percentile 95% crossed/product bootstrap. Sort the 5 session IDs and the 100 "
    "global dialogue IDs. For each replicate, one random.Random(20260901) stream "
    "draws 5 sessions with replacement and then 100 dialogues with replacement. "
    "Every original session-by-dialogue cell receives Cartesian-product multiplicity "
    "m_s*n_d, and all conditions in that cell retain the same weight (pairing). Point "
    "estimates use the unweighted complete 5x100 grid. Percentiles use the same linear "
    "interpolation as analyze.py: position=(n-1)*p between adjacent sorted draws."
)

# Frozen LF/Git identities from the accepted seal. The second mapping records the
# known CRLF checkout identities supplied for Windows auditability.
FORMAL_NORMALIZED_SHA256 = {
    "campaign_manifest.json": "2f4bd76e759945e62a5536b6b4399ad129c47a0b76c967bb653e22ffcf0f4ed8",
    "validation.json": "72fd0e94d637bb09b370f6053b82733574bd0ff214c3196f17ed7b4496973376",
    "analysis_v1.json": "2c08939dd041497701eab2ff6841178e8dd506252f2250b4ce7743ddb39c16a1",
    "sessions/s01/manifest.json": "b5d407ba2c55a5dce9fed85a90a7a5017c915546eb9e2a524a0d3e8a2431a69f",
    "sessions/s02/manifest.json": "4d203f06ca712442f41fb282ae54d291e5fd4c07e079cb9474d90a323dac6dd3",
    "sessions/s03/manifest.json": "347f43e40c483b34f292342a7309c92a467e1e7c259e136bb57d844127be0c05",
    "sessions/s04/manifest.json": "a7e677de3c4dc21052aa2eb7db9b150107ef5aba6b6b5a6ef14b523a74ff6906",
    "sessions/s05/manifest.json": "18ae01fb1dae06b229057fd5fea7e08e0a2475c99df171b5e4a56d934f8f1dcd",
    "sessions/s01/records.jsonl": "acb7436d4ac23c8bc6900b95eeb43187dc6ff64f7924157fe33d5ec0b8e00a6a",
    "sessions/s02/records.jsonl": "072059714b65eebe812d508aaf3369c018cf80bf1f38a792d034d312be9e54a3",
    "sessions/s03/records.jsonl": "49a9b96dadb512771b411f4dcc7580b63993a2b7e8cbb372f1fc548628a036e6",
    "sessions/s04/records.jsonl": "2c4f26ff22777b28b2069d2c41b4e4786daf6234b1d3056821123802c062b9ee",
    "sessions/s05/records.jsonl": "4a22e86885e8d3b55ec39805f4d233a1624dfbc2da430f373133ef891d930f63",
}
FORMAL_WINDOWS_CRLF_RECORD_SHA256 = {
    "sessions/s01/records.jsonl": "dbdd13f9133ed503eb8f4b9726a6e3b1faf8f5c1e21cce00cb07f121cbc4d1c0",
    "sessions/s02/records.jsonl": "72d329abe743c0070b2a7873c4a0ffe8d0d740cdb4938f7339a2ac20fa3f0a0c",
    "sessions/s03/records.jsonl": "9452e7abaae8f2bed0c2d73e00bee0796185db09ecd54a4edaf010613ab52231",
    "sessions/s04/records.jsonl": "9d4bbaf2fdc1eb72360cd77251d03a854b7ef5e31b83267a0e5dbcad89b36ebc",
    "sessions/s05/records.jsonl": "ae803e7f6a63a97318af7f47b7ed5b76209af060921e900d7223456dbd01db31",
}


class V2ValidationError(ValueError):
    """Raised when a v2 fail-closed gate rejects the source data."""


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as error:
        raise V2ValidationError(f"Source is outside repository: {path}") from error


def _lf_bytes(value: bytes) -> bytes:
    return value.replace(b"\r\n", b"\n")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def artifact_identity(
    path: Path, *, expected_normalized_sha256: str | None = None,
    expected_windows_sha256: str | None = None,
) -> dict[str, Any]:
    if not path.is_file():
        raise V2ValidationError(f"Missing source artifact: {path}")
    raw = path.read_bytes()
    normalized = _lf_bytes(raw)
    local_hash = _sha256(raw)
    normalized_hash = _sha256(normalized)
    if expected_normalized_sha256 and normalized_hash != expected_normalized_sha256:
        raise V2ValidationError(
            f"Normalized LF SHA-256 mismatch for {_repo_relative(path)}: "
            f"{normalized_hash} != {expected_normalized_sha256}"
        )
    if expected_windows_sha256 and local_hash not in {
        expected_windows_sha256,
        expected_normalized_sha256,
    }:
        raise V2ValidationError(
            f"Unexpected local SHA-256 for {_repo_relative(path)}: {local_hash}"
        )
    return {
        "path": _repo_relative(path),
        "identity_sha256": normalized_hash,
        "identity_bytes": "normalized_lf",
        "local_sha256": local_hash,
        "line_ending_normalization_applied": raw != normalized,
        "expected_normalized_lf_sha256": expected_normalized_sha256,
        "expected_windows_crlf_sha256": expected_windows_sha256,
    }


def _load_json_lf(path: Path) -> Any:
    try:
        return json.loads(_lf_bytes(path.read_bytes()).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V2ValidationError(f"Malformed JSON source {path}: {error}") from error


def load_records_with_identities(
    campaign_dir: Path, *, formal: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    sessions_root = campaign_dir / "sessions"
    session_dirs = sorted(path for path in sessions_root.iterdir() if path.is_dir()) \
        if sessions_root.is_dir() else []
    for session_dir in session_dirs:
        relative = f"sessions/{session_dir.name}/records.jsonl"
        path = session_dir / "records.jsonl"
        identity = artifact_identity(
            path,
            expected_normalized_sha256=(FORMAL_NORMALIZED_SHA256.get(relative) if formal else None),
            expected_windows_sha256=(
                FORMAL_WINDOWS_CRLF_RECORD_SHA256.get(relative) if formal else None
            ),
        )
        identities.append(identity)
        try:
            lines = _lf_bytes(path.read_bytes()).decode("utf-8").splitlines()
        except UnicodeDecodeError as error:
            raise V2ValidationError(f"Malformed UTF-8 JSONL source {path}: {error}") from error
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise V2ValidationError(
                    f"Malformed JSONL source {path}:{line_number}: {error}"
                ) from error
            if not isinstance(row, dict):
                raise V2ValidationError(f"JSONL row is not an object: {path}:{line_number}")
            records.append(row)
    return records, identities


def _as_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise V2ValidationError(f"{label} must be an integer")
    return value


def validate_record_payloads(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    timing_identities = (
        "oracle_preaccept_processing_ns=endpoint_accept_ns-last_segment_arrival_ns",
        "arrival_to_first_token_ready_ns=first_token_ready_ns-last_segment_arrival_ns",
        "consumer_delivery_from_arrival_ns=consumer_delivery_ns-last_segment_arrival_ns",
        "consumer_delivery_latency_ns=consumer_delivery_ns-endpoint_accept_ns",
        "generation_total_ns=generation_done_ns-endpoint_accept_ns",
        "candidate_lead_ns=endpoint_accept_ns-candidate_first_token_ns when candidate exists",
        "on_demand_ttft_ns=first_deliverable_token_ns-endpoint_accept_ns when on-demand exists",
        "prefill_to_first_token_ns=first_deliverable_token_ns-endpoint_accept_ns when stored (accepted raw alias includes full prefill)",
        "ttft_eff_ns=0 for a surviving ready candidate, otherwise endpoint-to-first-deliverable",
    )
    for row in records:
        label = f"{row.get('session_id')}/{row.get('dialogue_id')}/{row.get('condition')}"
        try:
            token_ids = row["output_token_ids"]
            if not isinstance(token_ids, list) or any(
                isinstance(token, bool) or not isinstance(token, int) for token in token_ids
            ):
                raise V2ValidationError("output_token_ids must be an integer list")
            if _as_int(row["final_tokens"], "final_tokens") != len(token_ids):
                raise V2ValidationError("final_tokens differs from output_token_ids length")
            arrival = _as_int(row["last_segment_arrival_ns"], "last_segment_arrival_ns")
            endpoint = _as_int(row["endpoint_accept_ns"], "endpoint_accept_ns")
            ready = _as_int(row["first_token_ready_ns"], "first_token_ready_ns")
            first = _as_int(row["first_deliverable_token_ns"], "first_deliverable_token_ns")
            consumer = _as_int(row["consumer_delivery_ns"], "consumer_delivery_ns")
            done = _as_int(row["generation_done_ns"], "generation_done_ns")
            prefill_done = _as_int(row["prefill_done_ns"], "prefill_done_ns")
            if not (0 <= arrival <= endpoint <= first <= consumer <= done):
                raise V2ValidationError("illegal arrival/endpoint/deliverable/consumer/done order")
            if not arrival <= ready <= done:
                raise V2ValidationError("illegal first-token readiness order")
            if not arrival <= prefill_done <= first:
                raise V2ValidationError("illegal prefill_done order")
            expected = {
                "oracle_preaccept_processing_ns": endpoint - arrival,
                "arrival_to_first_token_ready_ns": ready - arrival,
                "consumer_delivery_from_arrival_ns": consumer - arrival,
                "consumer_delivery_latency_ns": consumer - endpoint,
                "generation_total_ns": done - endpoint,
            }
            for field, expected_value in expected.items():
                if _as_int(row[field], field) != expected_value:
                    raise V2ValidationError(f"{field} identity mismatch")
            candidate_values = (
                row.get("candidate_started_ns"),
                row.get("candidate_first_token_ns"),
                row.get("candidate_lead_ns"),
            )
            if any(value is not None for value in candidate_values):
                if not all(value is not None for value in candidate_values):
                    raise V2ValidationError("partial candidate timing tuple")
                candidate_started, candidate_first, candidate_lead = (
                    _as_int(value, "candidate timing") for value in candidate_values
                )
                if not arrival <= candidate_started <= candidate_first <= endpoint:
                    raise V2ValidationError("illegal candidate timing order")
                if candidate_lead != endpoint - candidate_first:
                    raise V2ValidationError("candidate_lead_ns identity mismatch")
            on_demand = row.get("on_demand_ttft_ns")
            if on_demand is not None and _as_int(on_demand, "on_demand_ttft_ns") != first - endpoint:
                raise V2ValidationError("on_demand_ttft_ns identity mismatch")
            prefill_to_first = row.get("prefill_to_first_token_ns")
            if prefill_to_first is not None and _as_int(
                prefill_to_first, "prefill_to_first_token_ns"
            ) != first - endpoint:
                raise V2ValidationError("prefill_to_first_token_ns accepted-raw identity mismatch")
            survived = row.get("survived")
            if not isinstance(survived, bool):
                raise V2ValidationError("survived must be boolean")
            ready_tokens = _as_int(row["ready_tokens"], "ready_tokens")
            expected_ttft = 0 if survived and ready_tokens > 0 else first - endpoint
            if _as_int(row["ttft_eff_ns"], "ttft_eff_ns") != expected_ttft:
                raise V2ValidationError("ttft_eff_ns identity mismatch")
            expected_ready = row.get("candidate_first_token_ns") if survived else first
            if ready != expected_ready:
                raise V2ValidationError("first_token_ready_ns source mismatch")
            waste = _as_int(row["wasted_tokens"], "wasted_tokens")
            denominator = _as_int(row["waste_denominator_tokens"], "waste denominator")
            if denominator != waste + len(token_ids):
                raise V2ValidationError("waste denominator identity mismatch")
            if not isinstance(row.get("output_text"), str):
                raise V2ValidationError("output_text must be a string")
            for field in ("eos", "max_tokens_hit"):
                if not isinstance(row.get(field), bool):
                    raise V2ValidationError(f"{field} must be boolean")
        except (KeyError, V2ValidationError) as error:
            errors.append(f"{label}: {error}")
    return {
        "ok": not errors,
        "errors": errors,
        "records_checked": len(records),
        "timing_identities_checked": list(timing_identities),
        "output_token_ids_contract": "list[int], bool excluded",
        "final_tokens_contract": "final_tokens == len(output_token_ids)",
    }


def _index_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    indexed: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in records:
        try:
            key = (str(row["session_id"]), str(row["dialogue_id"]), str(row["condition"]))
        except KeyError as error:
            raise V2ValidationError(f"Record missing grid key: {error}") from error
        if key in indexed:
            raise V2ValidationError(f"Duplicate record key: {key}")
        indexed[key] = row
    return indexed


def validate_pair_counts(
    records: Sequence[Mapping[str, Any]],
    pairs: Sequence[tuple[str, str]],
    *, expected_pairs: int,
) -> dict[str, int]:
    indexed = _index_records(records)
    result: dict[str, int] = {}
    for left, right in pairs:
        left_keys = {(s, d) for s, d, c in indexed if c == left}
        right_keys = {(s, d) for s, d, c in indexed if c == right}
        if left_keys != right_keys:
            missing_right = sorted(left_keys - right_keys)[:3]
            missing_left = sorted(right_keys - left_keys)[:3]
            raise V2ValidationError(
                f"Pairing failure {left} vs {right}; missing right={missing_right}, "
                f"missing left={missing_left}"
            )
        if len(left_keys) != expected_pairs:
            raise V2ValidationError(
                f"Pair count {left} vs {right} is {len(left_keys)}, expected {expected_pairs}"
            )
        result[f"{left}_vs_{right}"] = len(left_keys)
    return result


def validate_formal_sources(campaign_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if campaign_dir.name != FORMAL_CAMPAIGN_ID:
        raise V2ValidationError(
            f"Formal v2 is bound to {FORMAL_CAMPAIGN_ID}, got {campaign_dir.name}"
        )
    records, record_sources = load_records_with_identities(campaign_dir, formal=True)
    source_artifacts: list[dict[str, Any]] = []
    for relative in (
        "campaign_manifest.json",
        "validation.json",
        "analysis_v1.json",
        *[f"sessions/s0{i}/manifest.json" for i in range(1, 6)],
    ):
        source_artifacts.append(
            artifact_identity(
                campaign_dir / relative,
                expected_normalized_sha256=FORMAL_NORMALIZED_SHA256[relative],
            )
        )
    campaign_manifest = _load_json_lf(campaign_dir / "campaign_manifest.json")
    accepted_validation = _load_json_lf(campaign_dir / "validation.json")
    if campaign_manifest.get("run_id") != FORMAL_CAMPAIGN_ID:
        raise V2ValidationError("Campaign manifest run_id mismatch")
    config = campaign_manifest.get("config", {})
    if not config.get("formal") or config.get("expected_sessions") != 5 \
            or config.get("expected_dialogues") != 100:
        raise V2ValidationError("Campaign manifest does not freeze formal 5x100 design")
    if accepted_validation.get("ok") is not True or accepted_validation.get("errors") != []:
        raise V2ValidationError("Accepted validation.json is not clean")
    manifest_identity = campaign_manifest.get("config", {}).get("campaign_identity", {}).get(
        "identity_hash"
    )
    for session_index in range(1, 6):
        manifest = _load_json_lf(campaign_dir / f"sessions/s0{session_index}/manifest.json")
        identity = manifest.get("config", {}).get("campaign_identity", {}).get("identity_hash")
        if identity != manifest_identity:
            raise V2ValidationError(f"s0{session_index} campaign identity mismatch")
    return records, {
        "record_sources": record_sources,
        "supporting_sources": source_artifacts,
        "formal_identity_note": (
            "identity_sha256 is computed after CRLF-to-LF normalization and is the formal "
            "provenance binding; local_sha256 is retained only for checkout diagnostics"
        ),
    }


def validate_analysis_input(
    records: Sequence[Mapping[str, Any]], *, expected_sessions: int,
    expected_dialogues: int, formal: bool,
) -> dict[str, Any]:
    grid = validate_grid(
        records,
        expected_sessions=expected_sessions,
        expected_dialogues=expected_dialogues,
        formal=formal,
    )
    payloads = validate_record_payloads(records)
    pair_counts: dict[str, int] = {}
    errors = [*grid["errors"], *payloads["errors"]]
    if not errors:
        try:
            pair_counts = validate_pair_counts(
                records,
                ((SYSTEM_A, CONFIRMATORY_CONDITION),
                 (CONFIRMATORY_CONDITION, NEVER_SPECULATE)),
                expected_pairs=expected_sessions * expected_dialogues,
            )
        except V2ValidationError as error:
            errors.append(str(error))
    return {
        "ok": not errors,
        "errors": errors,
        "grid": grid,
        "payloads": payloads,
        "pair_counts": pair_counts,
    }


def _ns_to_ms(value: int | float) -> float:
    return float(value) / 1_000_000.0


def _describe_values(values: Sequence[float]) -> dict[str, Any]:
    result = describe(values)
    result["mean"] = round(mean(values), 6)
    return result


def _describe_ms(records: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    return _describe_values([_ns_to_ms(int(row[field])) for row in records])


def derived_event_summaries(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    definitions: tuple[tuple[str, str, Callable[[Mapping[str, Any]], int]], ...] = (
        (
            "arrival_to_candidate_selection_compute_readiness_ms_raw_alias",
            "Candidate selection/compute readiness; raw alias arrival_to_first_token_ready_ns",
            lambda row: int(row["arrival_to_first_token_ready_ns"]),
        ),
        (
            "arrival_to_endpoint_accept_ms",
            "Last controlled segment arrival to synchronous oracle endpoint acceptance",
            lambda row: int(row["endpoint_accept_ns"]) - int(row["last_segment_arrival_ns"]),
        ),
        (
            "arrival_to_first_deliverable_event_ms",
            "Last controlled segment arrival to first_deliverable_token event",
            lambda row: int(row["first_deliverable_token_ns"])
            - int(row["last_segment_arrival_ns"]),
        ),
        (
            "endpoint_to_first_deliverable_event_ms",
            "Synchronous oracle endpoint acceptance to first_deliverable_token event",
            lambda row: int(row["first_deliverable_token_ns"]) - int(row["endpoint_accept_ns"]),
        ),
        (
            "arrival_to_consumer_marker_ms_harness_diagnostic",
            "Harness consumer/yield marker diagnostic; not production deliverability",
            lambda row: int(row["consumer_delivery_ns"])
            - int(row["last_segment_arrival_ns"]),
        ),
        (
            "endpoint_to_consumer_marker_ms_harness_diagnostic",
            "Harness consumer/yield marker diagnostic; not production deliverability",
            lambda row: int(row["consumer_delivery_ns"]) - int(row["endpoint_accept_ns"]),
        ),
    )
    return {
        name: {
            "definition": definition,
            "summary": _describe_values([_ns_to_ms(function(row)) for row in records]),
        }
        for name, definition, function in definitions
    }


def summarize_condition(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise V2ValidationError("Cannot summarize an empty condition")
    wasted = sum(int(row["wasted_tokens"]) for row in records)
    denominator = sum(int(row["waste_denominator_tokens"]) for row in records)
    return {
        "n": len(records),
        "candidate_selection_compute_readiness_ms": {
            "raw_field": "arrival_to_first_token_ready_ns",
            "terminology": "internal candidate selection/compute readiness, not generator deliverability",
            "summary": _describe_ms(records, "arrival_to_first_token_ready_ns"),
        },
        "ttft_eff_ms_oracle_latency_lower_bound": _describe_ms(records, "ttft_eff_ns"),
        "pooled_token_waste_ratio": wasted / denominator if denominator else 0.0,
        "pooled_wasted_tokens": wasted,
        "pooled_waste_denominator_tokens": denominator,
        "waste_estimand": "sum(wasted_tokens) / sum(wasted_tokens + final_tokens)",
        "survival_rate": sum(bool(row["survived"]) for row in records) / len(records),
        "derived_event_summaries": derived_event_summaries(records),
        "eos_rate": sum(bool(row["eos"]) for row in records) / len(records),
        "max_tokens_hit_rate": sum(bool(row["max_tokens_hit"]) for row in records) / len(records),
    }


def _common_prefix(left: Sequence[int], right: Sequence[int]) -> int:
    length = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        length += 1
    return length


def output_identity_diagnostics(
    records: Sequence[Mapping[str, Any]], condition_left: str, condition_right: str,
) -> dict[str, Any]:
    indexed = _index_records(records)
    keys = sorted((s, d) for s, d, c in indexed if c == condition_left)
    sessions = sorted({session for session, _ in keys})
    dialogues = sorted({dialogue for _, dialogue in keys})
    counts: Counter[str] = Counter()
    mismatch_by_session: Counter[str] = Counter()
    mismatch_dialogues: set[str] = set()
    common_prefixes: list[float] = []
    signatures: dict[str, list[tuple[Any, ...]]] = defaultdict(list)
    left_outputs: dict[str, list[tuple[Any, ...]]] = defaultdict(list)
    right_outputs: dict[str, list[tuple[Any, ...]]] = defaultdict(list)
    for session, dialogue in keys:
        left = indexed[(session, dialogue, condition_left)]
        right = indexed.get((session, dialogue, condition_right))
        if right is None:
            raise V2ValidationError(f"Missing output pair {session}/{dialogue}/{condition_right}")
        left_tokens = list(left["output_token_ids"])
        right_tokens = list(right["output_token_ids"])
        left_empty, right_empty = not left_tokens, not right_tokens
        if left_empty and right_empty:
            counts["both_empty"] += 1
        elif left_empty:
            counts["left_empty_only"] += 1
        elif right_empty:
            counts["right_empty_only"] += 1
        else:
            counts["first_token_denominator_nonempty_both"] += 1
            if left_tokens[0] == right_tokens[0]:
                counts["first_token_exact"] += 1
        full_exact = left_tokens == right_tokens
        length_equal = len(left_tokens) == len(right_tokens)
        eos_equal = left["eos"] == right["eos"]
        max_equal = left["max_tokens_hit"] == right["max_tokens_hit"]
        text_equal = left["output_text"] == right["output_text"]
        common_prefix = _common_prefix(left_tokens, right_tokens)
        flags = (full_exact, length_equal, eos_equal, max_equal, text_equal)
        counts["full_output_token_ids_exact"] += int(full_exact)
        counts["length_equal"] += int(length_equal)
        counts["eos_agreement"] += int(eos_equal)
        counts["max_token_agreement"] += int(max_equal)
        counts["eos_and_max_token_joint_agreement"] += int(eos_equal and max_equal)
        counts["text_equality"] += int(text_equal)
        common_prefixes.append(float(common_prefix))
        if not all(flags):
            mismatch_by_session[session] += 1
            mismatch_dialogues.add(dialogue)
        signatures[dialogue].append(
            (*flags, (not left_empty and not right_empty and left_tokens[0] == right_tokens[0]),
             common_prefix, len(left_tokens), len(right_tokens))
        )
        left_outputs[dialogue].append(
            (tuple(left_tokens), left["output_text"], left["eos"], left["max_tokens_hit"])
        )
        right_outputs[dialogue].append(
            (tuple(right_tokens), right["output_text"], right["eos"], right["max_tokens_hit"])
        )
    total = len(keys)
    first_denominator = counts["first_token_denominator_nonempty_both"]
    return {
        "conditions": [condition_left, condition_right],
        "paired_records": total,
        "full_output_token_ids_exact": {
            "numerator": counts["full_output_token_ids_exact"], "denominator": total,
            "rate": counts["full_output_token_ids_exact"] / total,
        },
        "first_token_exact": {
            "numerator": counts["first_token_exact"], "denominator": first_denominator,
            "rate": counts["first_token_exact"] / first_denominator if first_denominator else None,
            "empty_cases": {
                "both_empty": counts["both_empty"],
                "left_empty_only": counts["left_empty_only"],
                "right_empty_only": counts["right_empty_only"],
                "policy": "first-token equality excludes any pair lacking a first token",
            },
        },
        "length_equality": {
            "numerator": counts["length_equal"], "denominator": total,
            "rate": counts["length_equal"] / total,
        },
        "eos_agreement": {
            "numerator": counts["eos_agreement"], "denominator": total,
            "rate": counts["eos_agreement"] / total,
        },
        "max_token_agreement": {
            "numerator": counts["max_token_agreement"], "denominator": total,
            "rate": counts["max_token_agreement"] / total,
        },
        "eos_max_token_joint_agreement": {
            "numerator": counts["eos_and_max_token_joint_agreement"], "denominator": total,
            "rate": counts["eos_and_max_token_joint_agreement"] / total,
        },
        "text_equality": {
            "numerator": counts["text_equality"], "denominator": total,
            "rate": counts["text_equality"] / total,
        },
        "unique_dialogues_with_any_mismatch": {
            "numerator": len(mismatch_dialogues), "denominator": len(dialogues),
            "dialogue_ids": sorted(mismatch_dialogues),
        },
        "mismatches_by_session": {
            session: {"mismatches": mismatch_by_session[session], "dialogues": len(dialogues)}
            for session in sessions
        },
        "invariance_across_sessions": {
            "comparison_diagnostic_signature": {
                "invariant_dialogues": sum(len(set(values)) == 1 for values in signatures.values()),
                "denominator": len(dialogues),
            },
            "left_condition_output": {
                "invariant_dialogues": sum(len(set(values)) == 1 for values in left_outputs.values()),
                "denominator": len(dialogues),
            },
            "right_condition_output": {
                "invariant_dialogues": sum(len(set(values)) == 1 for values in right_outputs.values()),
                "denominator": len(dialogues),
            },
        },
        "common_prefix_token_count": _describe_values(common_prefixes),
        "primary_latency_filtering": "none; output identity never filters a latency estimand",
    }


def assert_formal_output_sanity(e1: Mapping[str, Any], e2: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "a_vs_b092_full_output_exact": (
            e1["full_output_token_ids_exact"]["numerator"], 280
        ),
        "a_vs_b092_first_token_exact": (e1["first_token_exact"]["numerator"], 465),
        "a_vs_b092_unique_divergent_dialogues": (
            e1["unique_dialogues_with_any_mismatch"]["numerator"], 44
        ),
        "b092_vs_never_full_output_exact": (
            e2["full_output_token_ids_exact"]["numerator"], 500
        ),
    }
    failures = [f"{name}: {actual} != {expected}" for name, (actual, expected) in checks.items()
                if actual != expected]
    if failures:
        raise V2ValidationError("Formal output sanity failed: " + "; ".join(failures))
    if e1["paired_records"] != 500 or e1["first_token_exact"]["denominator"] != 500 \
            or e2["paired_records"] != 500:
        raise V2ValidationError("Formal output sanity denominators are not 500")
    return {
        "ok": True,
        "assertions": {
            name: {"actual": actual, "expected": expected}
            for name, (actual, expected) in checks.items()
        },
    }


def _pair_values(
    indexed: Mapping[tuple[str, str, str], Mapping[str, Any]],
    sessions: Sequence[str], dialogues: Sequence[str], left: str, right: str, field: str,
) -> list[list[float]]:
    return [[
        _ns_to_ms(int(indexed[(session, dialogue, left)][field]))
        - _ns_to_ms(int(indexed[(session, dialogue, right)][field]))
        for dialogue in dialogues
    ] for session in sessions]


def _condition_values(
    indexed: Mapping[tuple[str, str, str], Mapping[str, Any]],
    sessions: Sequence[str], dialogues: Sequence[str], condition: str, field: str,
) -> list[list[float]]:
    return [[float(indexed[(session, dialogue, condition)][field]) for dialogue in dialogues]
            for session in sessions]


def _weighted_sum(matrix: Sequence[Sequence[float]], session_weights: Sequence[int],
                  dialogue_weights: Sequence[int]) -> float:
    return sum(
        session_weight * sum(value * dialogue_weight for value, dialogue_weight in zip(row, dialogue_weights))
        for row, session_weight in zip(matrix, session_weights)
        if session_weight
    )


def product_draw_weights(
    session_ids: Sequence[str], dialogue_ids: Sequence[str], rng: random.Random,
) -> tuple[list[int], list[int]]:
    sorted_sessions = sorted(session_ids)
    sorted_dialogues = sorted(dialogue_ids)
    session_counts = Counter(rng.choice(sorted_sessions) for _ in sorted_sessions)
    dialogue_counts = Counter(rng.choice(sorted_dialogues) for _ in sorted_dialogues)
    return (
        [session_counts[session] for session in sorted_sessions],
        [dialogue_counts[dialogue] for dialogue in sorted_dialogues],
    )


def product_bootstrap(
    records: Sequence[Mapping[str, Any]], *, repeats: int, seed: int,
) -> dict[str, Any]:
    if repeats <= 0:
        raise V2ValidationError("bootstrap repeats must be positive")
    indexed = _index_records(records)
    sessions = sorted({session for session, _, _ in indexed})
    dialogues = sorted({dialogue for _, dialogue, _ in indexed})
    pair_specs = {
        "e1_candidate_compute_readiness_mean_difference_ms_system_a_minus_b092": (
            SYSTEM_A, CONFIRMATORY_CONDITION, "arrival_to_first_token_ready_ns"
        ),
        "e2_candidate_compute_readiness_mean_difference_ms_never_minus_b092": (
            NEVER_SPECULATE, CONFIRMATORY_CONDITION, "arrival_to_first_token_ready_ns"
        ),
        "e1_ttft_eff_oracle_lower_bound_mean_difference_ms_system_a_minus_b092": (
            SYSTEM_A, CONFIRMATORY_CONDITION, "ttft_eff_ns"
        ),
        "e2_ttft_eff_oracle_lower_bound_mean_difference_ms_never_minus_b092": (
            NEVER_SPECULATE, CONFIRMATORY_CONDITION, "ttft_eff_ns"
        ),
    }
    pair_matrices = {
        name: _pair_values(indexed, sessions, dialogues, *spec)
        for name, spec in pair_specs.items()
    }
    waste_matrices = {
        condition: _condition_values(indexed, sessions, dialogues, condition, "wasted_tokens")
        for condition in B_CONDITIONS
    }
    denominator_matrices = {
        condition: _condition_values(
            indexed, sessions, dialogues, condition, "waste_denominator_tokens"
        ) for condition in B_CONDITIONS
    }
    survival_matrices = {
        condition: _condition_values(indexed, sessions, dialogues, condition, "survived")
        for condition in B_CONDITIONS
    }
    draws: dict[str, list[float]] = {name: [] for name in pair_specs}
    for condition in B_CONDITIONS:
        draws[f"{condition}.pooled_token_waste_ratio"] = []
        draws[f"{condition}.survival_rate"] = []
    rng = random.Random(seed)
    expected_weight = len(sessions) * len(dialogues)
    for _ in range(repeats):
        session_weights, dialogue_weights = product_draw_weights(sessions, dialogues, rng)
        product_weight = sum(session_weights) * sum(dialogue_weights)
        if product_weight != expected_weight:
            raise V2ValidationError(
                f"Product bootstrap weight is {product_weight}, expected {expected_weight}"
            )
        for name, matrix in pair_matrices.items():
            draws[name].append(
                _weighted_sum(matrix, session_weights, dialogue_weights) / product_weight
            )
        for condition in B_CONDITIONS:
            numerator = _weighted_sum(waste_matrices[condition], session_weights, dialogue_weights)
            denominator = _weighted_sum(
                denominator_matrices[condition], session_weights, dialogue_weights
            )
            draws[f"{condition}.pooled_token_waste_ratio"].append(
                numerator / denominator if denominator else 0.0
            )
            draws[f"{condition}.survival_rate"].append(
                _weighted_sum(survival_matrices[condition], session_weights, dialogue_weights)
                / product_weight
            )
    for name, values in draws.items():
        if not all(math.isfinite(value) for value in values):
            raise V2ValidationError(f"Non-finite bootstrap draw in {name}")
    return {
        "method": FORMAL_METHOD,
        "sampling_unit_order": {
            "session_ids": sessions,
            "global_dialogue_ids": dialogues,
        },
        "session_draws_per_replicate": len(sessions),
        "dialogue_draws_per_replicate": len(dialogues),
        "cartesian_product_weight_sum_per_replicate": expected_weight,
        "condition_pairing": "all conditions retain each sampled session-dialogue cell weight m_s*n_d",
        "random_generator": "random.Random",
        "seed": seed,
        "repeats": repeats,
        "interval": "percentile 95%",
        "quantile_interpolation": "linear; position=(n-1)*p",
        "ci": {name: _ci(values, 0.95) for name, values in draws.items()},
    }


def _pair_effect(
    records: Sequence[Mapping[str, Any]], left: str, right: str, field: str,
) -> float:
    indexed = _index_records(records)
    keys = sorted((s, d) for s, d, c in indexed if c == left)
    return mean(
        _ns_to_ms(int(indexed[(s, d, left)][field]))
        - _ns_to_ms(int(indexed[(s, d, right)][field]))
        for s, d in keys
    )


def _pair_distribution(
    records: Sequence[Mapping[str, Any]], left: str, right: str, field: str,
) -> dict[str, Any]:
    indexed = _index_records(records)
    keys = sorted((s, d) for s, d, c in indexed if c == left)
    values = [
        _ns_to_ms(int(indexed[(s, d, left)][field]))
        - _ns_to_ms(int(indexed[(s, d, right)][field]))
        for s, d in keys
    ]
    return {"n": len(values), "summary": _describe_values(values)}


def session_sensitivity(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    sessions = sorted({str(row["session_id"]) for row in records})
    specifications = {
        "e1_candidate_compute_readiness_system_a_minus_b092": (
            SYSTEM_A, CONFIRMATORY_CONDITION, "arrival_to_first_token_ready_ns"
        ),
        "e2_candidate_compute_readiness_never_minus_b092": (
            NEVER_SPECULATE, CONFIRMATORY_CONDITION, "arrival_to_first_token_ready_ns"
        ),
        "e1_ttft_eff_oracle_lower_bound_system_a_minus_b092": (
            SYSTEM_A, CONFIRMATORY_CONDITION, "ttft_eff_ns"
        ),
        "e2_ttft_eff_oracle_lower_bound_never_minus_b092": (
            NEVER_SPECULATE, CONFIRMATORY_CONDITION, "ttft_eff_ns"
        ),
    }
    result: dict[str, Any] = {}
    for name, (left, right, field) in specifications.items():
        per_session = {
            session: _pair_effect(
                [row for row in records if row["session_id"] == session], left, right, field
            ) for session in sessions
        }
        leave_one_out = {
            session: _pair_effect(
                [row for row in records if row["session_id"] != session], left, right, field
            ) for session in sessions
        }
        result[name] = {
            "unit": "ms",
            "full_grid_mean": _pair_effect(records, left, right, field),
            "per_session_mean_effect": per_session,
            "leave_one_session_out_mean_effect": leave_one_out,
            "leave_one_session_out_range": {
                "min": min(leave_one_out.values()), "max": max(leave_one_out.values())
            },
        }
    return result


def point_estimate_compatibility(
    campaign_dir: Path, condition_summaries: Mapping[str, Any],
    pair_effects: Mapping[str, float],
) -> dict[str, Any]:
    v1 = _load_json_lf(campaign_dir / "analysis_v1.json")
    comparisons: dict[str, dict[str, float]] = {}
    v1_pairs = {
        "e1_candidate_compute_readiness_system_a_minus_b092": v1["e1"]["primary_paired"]
        ["absolute_difference_ms_a_minus_b"]["mean"],
        "e2_candidate_compute_readiness_never_minus_b092": v1["e2"]["primary_paired"]
        ["absolute_difference_ms_a_minus_b"]["mean"],
        "e1_ttft_eff_oracle_lower_bound_system_a_minus_b092": v1["e1"]
        ["oracle_ttft_eff_latency_lower_bound_paired"]["absolute_difference_ms_a_minus_b"]["mean"],
        "e2_ttft_eff_oracle_lower_bound_never_minus_b092": v1["e2"]
        ["oracle_ttft_eff_latency_lower_bound_paired"]["absolute_difference_ms_a_minus_b"]["mean"],
    }
    for name, v1_value in v1_pairs.items():
        v2_value = round(pair_effects[name], 6)
        comparisons[name] = {"v1": v1_value, "v2": v2_value, "absolute_difference": abs(v2_value-v1_value)}
    for condition in CONDITIONS:
        metrics = {
            "candidate_compute_readiness_mean_ms": (
                v1["condition_summaries"][condition]
                ["arrival_to_first_token_ready_ms_primary"]["mean"],
                condition_summaries[condition]["candidate_selection_compute_readiness_ms"]
                ["summary"]["mean"],
            ),
            "ttft_eff_mean_ms": (
                v1["condition_summaries"][condition]
                ["ttft_eff_ms_oracle_latency_lower_bound"]["mean"],
                condition_summaries[condition]["ttft_eff_ms_oracle_latency_lower_bound"]["mean"],
            ),
            "pooled_waste_ratio": (
                v1["condition_summaries"][condition]["pooled_token_waste_ratio"],
                condition_summaries[condition]["pooled_token_waste_ratio"],
            ),
            "survival_rate": (
                v1["condition_summaries"][condition]["survival_rate"],
                condition_summaries[condition]["survival_rate"],
            ),
        }
        for metric, (v1_value, v2_value) in metrics.items():
            comparisons[f"{condition}.{metric}"] = {
                "v1": v1_value, "v2": v2_value,
                "absolute_difference": abs(v2_value-v1_value),
            }
    maximum = max(item["absolute_difference"] for item in comparisons.values())
    if maximum > 1e-12:
        raise V2ValidationError(f"v2 point estimate differs from v1: max abs={maximum}")
    return {
        "compatible": True,
        "tolerance": 1e-12,
        "maximum_absolute_difference": maximum,
        "comparisons": comparisons,
    }


def _formal_defaults_guard(
    *, expected_sessions: int, expected_dialogues: int, repeats: int, seed: int,
) -> None:
    actual = (expected_sessions, expected_dialogues, repeats, seed)
    expected = (FORMAL_SESSION_COUNT, FORMAL_DIALOGUE_COUNT, FORMAL_REPEATS, FORMAL_SEED)
    if actual != expected:
        raise V2ValidationError(
            "Formal defaults are fixed at sessions/dialogues/repeats/seed="
            f"{expected}, got {actual}"
        )


def build_analysis(
    campaign_dir: Path, *, bootstrap_repeats: int = FORMAL_REPEATS,
    bootstrap_seed: int = FORMAL_SEED,
    expected_sessions: int = FORMAL_SESSION_COUNT,
    expected_dialogues: int = FORMAL_DIALOGUE_COUNT,
    formal: bool = True,
) -> dict[str, Any]:
    campaign_dir = campaign_dir.resolve()
    if formal:
        _formal_defaults_guard(
            expected_sessions=expected_sessions, expected_dialogues=expected_dialogues,
            repeats=bootstrap_repeats, seed=bootstrap_seed,
        )
        records, source_provenance = validate_formal_sources(campaign_dir)
    else:
        records, record_sources = load_records_with_identities(campaign_dir, formal=False)
        source_provenance = {"record_sources": record_sources, "supporting_sources": []}
    validation = validate_analysis_input(
        records, expected_sessions=expected_sessions,
        expected_dialogues=expected_dialogues, formal=formal,
    )
    if not validation["ok"]:
        raise V2ValidationError("v2 input validation failed: " + "; ".join(validation["errors"][:10]))
    indexed = _index_records(records)
    by_condition = {
        condition: summarize_condition(
            [indexed[key] for key in sorted(indexed) if key[2] == condition]
        ) for condition in CONDITIONS
    }
    pair_effects = {
        "e1_candidate_compute_readiness_system_a_minus_b092": _pair_effect(
            records, SYSTEM_A, CONFIRMATORY_CONDITION, "arrival_to_first_token_ready_ns"
        ),
        "e2_candidate_compute_readiness_never_minus_b092": _pair_effect(
            records, NEVER_SPECULATE, CONFIRMATORY_CONDITION, "arrival_to_first_token_ready_ns"
        ),
        "e1_ttft_eff_oracle_lower_bound_system_a_minus_b092": _pair_effect(
            records, SYSTEM_A, CONFIRMATORY_CONDITION, "ttft_eff_ns"
        ),
        "e2_ttft_eff_oracle_lower_bound_never_minus_b092": _pair_effect(
            records, NEVER_SPECULATE, CONFIRMATORY_CONDITION, "ttft_eff_ns"
        ),
    }
    identity_e1 = output_identity_diagnostics(records, SYSTEM_A, CONFIRMATORY_CONDITION)
    identity_e2 = output_identity_diagnostics(records, CONFIRMATORY_CONDITION, NEVER_SPECULATE)
    sanity = assert_formal_output_sanity(identity_e1, identity_e2) if formal else None
    compatibility = point_estimate_compatibility(
        campaign_dir, by_condition, pair_effects
    ) if formal else None
    bootstrap = product_bootstrap(records, repeats=bootstrap_repeats, seed=bootstrap_seed)
    analyzer_path = Path(__file__).resolve()
    analyzer_identity = artifact_identity(analyzer_path)
    superseded_identity = artifact_identity(
        campaign_dir / "analysis_v1.json",
        expected_normalized_sha256=(FORMAL_NORMALIZED_SHA256["analysis_v1.json"] if formal else None),
    )
    provenance = {
        **source_provenance,
        "analyzer": analyzer_identity,
        "provenance_identity_basis": "normalized LF SHA-256 and repository-relative paths",
    }
    provenance["provenance_hash"] = sha256_bytes(canonical_json(provenance).encode("utf-8"))
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_version": ANALYSIS_VERSION,
        "experiment": "e1e2_confirmatory",
        "created_at_utc": utc_now(),
        "scope_note": (
            "Controlled synchronous text-segment, model-side harness. No real ASR, endpoint "
            "detector, online TTS, player, sound card, acoustic hearing, or production end-to-end "
            "deliverability is measured."
        ),
        "supersedes": {
            "path": superseded_identity["path"],
            "sha256": superseded_identity["identity_sha256"],
            "reason": (
                "analysis_v1 used session-then-dialogue-within-session nested resampling; v2 "
                "uses the frozen crossed session x global-dialogue product bootstrap and adds "
                "event/identity/sensitivity diagnostics without changing point estimates"
            ),
        },
        "design": {
            "sessions": expected_sessions,
            "global_dialogues": expected_dialogues,
            "conditions": list(CONDITIONS),
            "full_grid_records": len(records),
            "point_estimate_grid": "complete unweighted 5 session x 100 dialogue grid",
            "confirmatory_condition": CONFIRMATORY_CONDITION,
            "e1_estimand": "System A minus B@0.92",
            "e2_estimand": "never_speculate minus B@0.92",
            "candidate_compute_readiness_raw_field": "arrival_to_first_token_ready_ns",
            "candidate_compute_readiness_note": (
                "renamed internal candidate selection/compute readiness; not generator or "
                "production deliverability"
            ),
            "ttft_eff_note": "synchronous oracle latency lower bound/speculation-benefit upper bound",
            "outlier_trimming": None,
            "primary_latency_filtering_by_output_identity": None,
        },
        "validation": validation,
        "formal_raw_sanity": sanity,
        "condition_summaries": by_condition,
        "paired_estimands": {
            "e1": {
                "candidate_compute_readiness_system_a_minus_b092_ms": {
                    "point_estimate_mean": pair_effects[
                        "e1_candidate_compute_readiness_system_a_minus_b092"
                    ],
                    "paired_distribution": _pair_distribution(
                        records, SYSTEM_A, CONFIRMATORY_CONDITION,
                        "arrival_to_first_token_ready_ns"
                    ),
                    "bootstrap_ci": bootstrap["ci"][
                        "e1_candidate_compute_readiness_mean_difference_ms_system_a_minus_b092"
                    ],
                },
                "ttft_eff_oracle_lower_bound_system_a_minus_b092_ms": {
                    "point_estimate_mean": pair_effects[
                        "e1_ttft_eff_oracle_lower_bound_system_a_minus_b092"
                    ],
                    "paired_distribution": _pair_distribution(
                        records, SYSTEM_A, CONFIRMATORY_CONDITION, "ttft_eff_ns"
                    ),
                    "bootstrap_ci": bootstrap["ci"][
                        "e1_ttft_eff_oracle_lower_bound_mean_difference_ms_system_a_minus_b092"
                    ],
                },
            },
            "e2": {
                "candidate_compute_readiness_never_minus_b092_ms": {
                    "point_estimate_mean": pair_effects[
                        "e2_candidate_compute_readiness_never_minus_b092"
                    ],
                    "paired_distribution": _pair_distribution(
                        records, NEVER_SPECULATE, CONFIRMATORY_CONDITION,
                        "arrival_to_first_token_ready_ns"
                    ),
                    "bootstrap_ci": bootstrap["ci"][
                        "e2_candidate_compute_readiness_mean_difference_ms_never_minus_b092"
                    ],
                },
                "ttft_eff_oracle_lower_bound_never_minus_b092_ms": {
                    "point_estimate_mean": pair_effects[
                        "e2_ttft_eff_oracle_lower_bound_never_minus_b092"
                    ],
                    "paired_distribution": _pair_distribution(
                        records, NEVER_SPECULATE, CONFIRMATORY_CONDITION, "ttft_eff_ns"
                    ),
                    "bootstrap_ci": bootstrap["ci"][
                        "e2_ttft_eff_oracle_lower_bound_mean_difference_ms_never_minus_b092"
                    ],
                },
            },
        },
        "b_condition_bootstrap_estimands": {
            condition: {
                "pooled_token_waste_ratio": {
                    "point_estimate": by_condition[condition]["pooled_token_waste_ratio"],
                    "bootstrap_ci": bootstrap["ci"][f"{condition}.pooled_token_waste_ratio"],
                    "estimand": "ratio of weighted sums in every replicate",
                },
                "survival_rate": {
                    "point_estimate": by_condition[condition]["survival_rate"],
                    "bootstrap_ci": bootstrap["ci"][f"{condition}.survival_rate"],
                },
            } for condition in B_CONDITIONS
        },
        "per_session_and_leave_one_session_out_sensitivity": session_sensitivity(records),
        "output_identity_diagnostics": {
            "system_a_vs_b092": identity_e1,
            "b092_vs_never_speculate": identity_e2,
            "interpretation": (
                "Implementation-path output identity is diagnostic only. Primary latency uses "
                "all 500 pairs and is never filtered by output equality."
            ),
        },
        "point_estimate_compatibility_with_v1": compatibility,
        "bootstrap": bootstrap,
        "provenance": provenance,
        "excluded_records": [],
    }


def _synthetic_record(session: str, dialogue: str, condition: str, value: int) -> dict[str, Any]:
    arrival = 1_000_000
    endpoint = arrival + 100
    first = endpoint + value
    consumer = first + 3
    done = consumer + 5
    token_ids = [] if value == 0 else [value]
    return {
        "session_id": session, "session_index": int(session[1:]) - 1,
        "dialogue_id": dialogue, "dialogue_index": int(dialogue[1:]),
        "condition": condition, "condition_ordinal": CONDITIONS.index(condition),
        "condition_order_seed": FORMAL_SEED, "condition_order": list(CONDITIONS),
        "process_start_id": f"process-{session}", "campaign_identity_hash": "synthetic",
        "last_segment_arrival_ns": arrival, "endpoint_accept_ns": endpoint,
        "first_token_ready_ns": first, "first_deliverable_token_ns": first,
        "consumer_delivery_ns": consumer, "generation_done_ns": done,
        "prefill_done_ns": endpoint,
        "oracle_preaccept_processing_ns": endpoint-arrival,
        "arrival_to_first_token_ready_ns": first-arrival,
        "consumer_delivery_from_arrival_ns": consumer-arrival,
        "consumer_delivery_latency_ns": consumer-endpoint,
        "generation_total_ns": done-endpoint,
        "candidate_started_ns": None, "candidate_first_token_ns": None,
        "candidate_lead_ns": None, "on_demand_ttft_ns": first-endpoint,
        "prefill_to_first_token_ns": first-endpoint,
        "ttft_eff_ns": first-endpoint,
        "survived": False, "ready_tokens": 0,
        "wasted_tokens": value, "waste_denominator_tokens": value + len(token_ids),
        "final_tokens": len(token_ids), "output_token_ids": token_ids,
        "output_text": "" if not token_ids else str(value), "eos": False,
        "max_tokens_hit": True, "n_speculations": 0, "n_invalidated": 0,
        "speculative_tokens": value,
    }


def _expect_failure(function: Callable[[], Any], label: str) -> None:
    try:
        function()
    except (V2ValidationError, ValueError):
        return
    raise AssertionError(f"Self-test expected failure: {label}")


def run_self_test() -> None:
    sessions = ["s01", "s02"]
    dialogues = ["d0", "d1"]
    rng = random.Random(8)
    session_weights, dialogue_weights = product_draw_weights(sessions, dialogues, rng)
    assert sum(session_weights) * sum(dialogue_weights) == 4
    matrix = [[0.0, 0.0], [0.0, 100.0]]
    crossed = _weighted_sum(matrix, session_weights, dialogue_weights) / 4
    nested_rng = random.Random(8)
    nested_values = []
    for _ in sessions:
        chosen_session = nested_rng.choice(range(2))
        for _ in dialogues:
            chosen_dialogue = nested_rng.choice(range(2))
            nested_values.append(matrix[chosen_session][chosen_dialogue])
    nested = mean(nested_values)
    assert crossed != nested, (crossed, nested)

    # Pairing: one common product-weight matrix is applied to both arms.
    left = [[10.0, 20.0], [30.0, 40.0]]
    right = [[1.0, 2.0], [3.0, 4.0]]
    paired = [[left[i][j]-right[i][j] for j in range(2)] for i in range(2)]
    assert math.isclose(
        _weighted_sum(paired, session_weights, dialogue_weights),
        _weighted_sum(left, session_weights, dialogue_weights)
        - _weighted_sum(right, session_weights, dialogue_weights),
    )
    numerators = [[1.0, 3.0], [5.0, 7.0]]
    denominators = [[2.0, 10.0], [10.0, 14.0]]
    ratio = _weighted_sum(numerators, session_weights, dialogue_weights) / _weighted_sum(
        denominators, session_weights, dialogue_weights
    )
    assert ratio != mean(
        numerators[i][j] / denominators[i][j] for i in range(2) for j in range(2)
    )

    empty_rows = [
        {"session_id": "s", "dialogue_id": "d0", "condition": "left",
         "output_token_ids": [], "output_text": "", "eos": True, "max_tokens_hit": False},
        {"session_id": "s", "dialogue_id": "d0", "condition": "right",
         "output_token_ids": [], "output_text": "", "eos": True, "max_tokens_hit": False},
        {"session_id": "s", "dialogue_id": "d1", "condition": "left",
         "output_token_ids": [1], "output_text": "a", "eos": False, "max_tokens_hit": True},
        {"session_id": "s", "dialogue_id": "d1", "condition": "right",
         "output_token_ids": [], "output_text": "", "eos": False, "max_tokens_hit": True},
        {"session_id": "s", "dialogue_id": "d2", "condition": "left",
         "output_token_ids": [2], "output_text": "b", "eos": False, "max_tokens_hit": True},
        {"session_id": "s", "dialogue_id": "d2", "condition": "right",
         "output_token_ids": [2], "output_text": "b", "eos": False, "max_tokens_hit": True},
    ]
    diagnostics = output_identity_diagnostics(empty_rows, "left", "right")
    assert diagnostics["first_token_exact"]["numerator"] == 1
    assert diagnostics["first_token_exact"]["denominator"] == 1
    assert diagnostics["first_token_exact"]["empty_cases"] == {
        "both_empty": 1, "left_empty_only": 0, "right_empty_only": 1,
        "policy": "first-token equality excludes any pair lacking a first token",
    }

    synthetic = [
        _synthetic_record(session, dialogue, condition, 1 + i + j)
        for i, session in enumerate(sessions)
        for j, dialogue in enumerate(dialogues)
        for condition in CONDITIONS
    ]
    first_bootstrap = product_bootstrap(synthetic, repeats=25, seed=17)
    second_bootstrap = product_bootstrap(synthetic, repeats=25, seed=17)
    assert first_bootstrap["ci"] == second_bootstrap["ci"]
    assert validate_pair_counts(
        synthetic, ((SYSTEM_A, CONFIRMATORY_CONDITION),), expected_pairs=4
    )

    missing = synthetic[:-1]
    _expect_failure(
        lambda: validate_pair_counts(
            missing, ((CONFIRMATORY_CONDITION, NEVER_SPECULATE),), expected_pairs=4
        ), "missing pair",
    )
    duplicate = [*synthetic, dict(synthetic[0])]
    _expect_failure(lambda: _index_records(duplicate), "duplicate record")
    malformed = [dict(synthetic[0])]
    malformed[0]["output_token_ids"] = [True]
    assert not validate_record_payloads(malformed)["ok"]
    bad_timing = [dict(synthetic[0])]
    bad_timing[0]["consumer_delivery_from_arrival_ns"] += 1
    assert not validate_record_payloads(bad_timing)["ok"]
    with tempfile.TemporaryDirectory() as temporary:
        malformed_path = Path(temporary) / "bad.json"
        malformed_path.write_text("{bad", encoding="utf-8")
        _expect_failure(lambda: _load_json_lf(malformed_path), "malformed JSON")
    print("E1/E2 analyze_v2 self-test PASS")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--bootstrap-repeats", type=int, default=FORMAL_REPEATS)
    parser.add_argument("--bootstrap-seed", type=int, default=FORMAL_SEED)
    parser.add_argument("--expected-sessions", type=int, default=FORMAL_SESSION_COUNT)
    parser.add_argument("--expected-dialogues", type=int, default=FORMAL_DIALOGUE_COUNT)
    parser.add_argument("--non-formal", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.campaign_dir is None:
        raise SystemExit("--campaign-dir is required unless --self-test is used")
    campaign_dir = args.campaign_dir.resolve()
    output = (args.out or campaign_dir / "analysis_v2.json").resolve()
    checksum_path = campaign_dir / "analysis_v2.sha256"
    if not args.non_formal and output != campaign_dir / "analysis_v2.json":
        raise V2ValidationError("Formal output path is frozen at campaign-root analysis_v2.json")
    for path in (output, checksum_path):
        if path.exists():
            raise FileExistsError(f"Versioned analysis artifact is immutable and exists: {path}")
    result = build_analysis(
        campaign_dir,
        bootstrap_repeats=args.bootstrap_repeats,
        bootstrap_seed=args.bootstrap_seed,
        expected_sessions=args.expected_sessions,
        expected_dialogues=args.expected_dialogues,
        formal=not args.non_formal,
    )
    atomic_write_json(output, result)
    output_hash = _sha256(output.read_bytes())
    atomic_write_text(checksum_path, f"{output_hash}  {output.name}\n")
    print(output)
    print(checksum_path)
    print(output_hash)


if __name__ == "__main__":
    main()
