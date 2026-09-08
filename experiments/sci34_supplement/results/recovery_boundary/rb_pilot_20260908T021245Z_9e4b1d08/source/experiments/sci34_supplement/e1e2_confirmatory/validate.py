"""Strict grid, timing, balance, and provenance validation for E1/E2."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from experiments.sci34_supplement.common import atomic_write_json, load_jsonl, sha256_file, utc_now
from experiments.sci34_supplement.e1e2_confirmatory.campaign import load_campaign_manifest
from experiments.sci34_supplement.e1e2_confirmatory.protocol import (
    CONDITIONS,
    FORMAL_DIALOGUE_COUNT,
    FORMAL_SESSION_COUNT,
    NEVER_SPECULATE,
    SYSTEM_A,
    THRESHOLD_CONDITIONS,
    balanced_condition_order,
)


class ValidationError(ValueError):
    pass


def discover_sessions(campaign_dir: Path) -> list[Path]:
    sessions_root = campaign_dir / "sessions"
    if not sessions_root.exists():
        raise ValidationError(f"Missing sessions directory: {sessions_root}")
    return sorted(path for path in sessions_root.iterdir() if path.is_dir())


def load_campaign_records(campaign_dir: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    paths: list[Path] = []
    records: list[dict[str, Any]] = []
    for session_dir in discover_sessions(campaign_dir):
        path = session_dir / "records.jsonl"
        paths.append(path)
        records.extend(load_jsonl(path))
    return records, paths


def _assert(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def validate_grid(
    records: Iterable[Mapping[str, Any]],
    *,
    expected_sessions: int = FORMAL_SESSION_COUNT,
    expected_dialogues: int = FORMAL_DIALOGUE_COUNT,
    formal: bool = True,
) -> dict[str, Any]:
    rows = [dict(record) for record in records]
    errors: list[str] = []
    duplicate_keys: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in rows:
        key = (
            str(record.get("session_id")),
            str(record.get("dialogue_id")),
            str(record.get("condition")),
        )
        if key in seen:
            duplicate_keys.append(key)
        seen.add(key)
        by_session[key[0]].append(record)
    _assert(not duplicate_keys, f"Duplicate record keys: {duplicate_keys[:5]}", errors)
    _assert(
        len(by_session) == expected_sessions,
        f"Expected {expected_sessions} sessions, found {len(by_session)}",
        errors,
    )
    if formal:
        _assert(
            expected_sessions == FORMAL_SESSION_COUNT
            and expected_dialogues == FORMAL_DIALOGUE_COUNT,
            "Formal validation dimensions are frozen at 5 sessions × 100 dialogues",
            errors,
        )
    global_dialogues: set[str] | None = None
    process_ids: dict[str, set[str]] = {}
    identity_hashes: dict[str, set[str]] = {}
    balance_audit: dict[str, Any] = {}
    session_indices: set[int] = set()
    for session_id, subset in sorted(by_session.items()):
        raw_indices = {int(row.get("session_index", -1)) for row in subset}
        _assert(len(raw_indices) == 1, f"{session_id} mixes session_index values", errors)
        session_index = next(iter(raw_indices)) if raw_indices else -1
        session_indices.add(session_index)
        dialogues = {str(row["dialogue_id"]) for row in subset}
        if global_dialogues is None:
            global_dialogues = dialogues
        else:
            _assert(dialogues == global_dialogues, f"{session_id} dialogue set differs", errors)
        _assert(
            len(dialogues) == expected_dialogues,
            f"{session_id} has {len(dialogues)} dialogues, expected {expected_dialogues}",
            errors,
        )
        expected_keys = {(dialogue, condition) for dialogue in dialogues for condition in CONDITIONS}
        actual_keys = {(str(row["dialogue_id"]), str(row["condition"])) for row in subset}
        missing = expected_keys - actual_keys
        extra = actual_keys - expected_keys
        _assert(not missing, f"{session_id} missing {len(missing)} cells", errors)
        _assert(not extra, f"{session_id} has {len(extra)} unexpected cells", errors)
        process_ids[session_id] = {str(row.get("process_start_id")) for row in subset}
        identity_hashes[session_id] = {
            str(row.get("campaign_identity_hash")) for row in subset
        }
        _assert(
            len(process_ids[session_id]) == 1,
            f"{session_id} mixes process restart identities",
            errors,
        )
        _assert(
            len(identity_hashes[session_id]) == 1,
            f"{session_id} mixes campaign identities",
            errors,
        )
        ordinal_counts = {
            condition: [0] * len(CONDITIONS) for condition in CONDITIONS
        }
        order_seeds = {
            int(row.get("condition_order_seed", 20260901)) for row in subset
        }
        _assert(len(order_seeds) == 1, f"{session_id} mixes order seeds", errors)
        order_seed = next(iter(order_seeds)) if order_seeds else 20260901
        for row in subset:
            condition = str(row["condition"])
            ordinal = int(row["condition_ordinal"])
            expected_order = balanced_condition_order(
                session_index=session_index,
                dialogue_index=int(row["dialogue_index"]),
                seed=order_seed,
            )
            if row.get("condition_order") != expected_order:
                errors.append(
                    f"{session_id}/{row.get('dialogue_id')} stores non-frozen condition order"
                )
            elif expected_order[ordinal] != condition:
                errors.append(
                    f"{session_id}/{row.get('dialogue_id')} ordinal does not match condition"
                )
            if condition not in ordinal_counts or not 0 <= ordinal < len(CONDITIONS):
                errors.append(f"{session_id} invalid condition/ordinal: {condition}/{ordinal}")
            else:
                ordinal_counts[condition][ordinal] += 1
        maximum_spread = max(
            max(counts) - min(counts) for counts in ordinal_counts.values()
        )
        _assert(
            maximum_spread <= 1,
            f"{session_id} condition-order imbalance spread={maximum_spread}",
            errors,
        )
        balance_audit[session_id] = {
            "ordinal_counts": ordinal_counts,
            "maximum_spread": maximum_spread,
        }
    if formal:
        _assert(
            session_indices == set(range(FORMAL_SESSION_COUNT)),
            f"Formal session_index set must be {{0,1,2,3,4}}, found {sorted(session_indices)}",
            errors,
        )
    all_identity_hashes = set().union(*identity_hashes.values()) if identity_hashes else set()
    _assert(len(all_identity_hashes) == 1, "Sessions do not share one campaign identity", errors)
    all_processes = [next(iter(values)) for values in process_ids.values() if values]
    _assert(
        len(all_processes) == len(set(all_processes)),
        "Formal sessions are not independent process identities",
        errors,
    )
    expected_total = expected_sessions * expected_dialogues * len(CONDITIONS)
    _assert(len(rows) == expected_total, f"Expected {expected_total} rows, found {len(rows)}", errors)
    if formal:
        fixture_ids = sorted(
            {
                str(row["dialogue_id"])
                for row in rows
                if str(row["dialogue_id"]).lower().startswith(("fx", "fixture", "smoke"))
            }
        )
        _assert(not fixture_ids, f"Fixture-like formal IDs: {fixture_ids}", errors)
    return {
        "ok": not errors,
        "errors": errors,
        "records": len(rows),
        "sessions": len(by_session),
        "dialogues": len(global_dialogues or set()),
        "conditions": list(CONDITIONS),
        "expected_records": expected_total,
        "duplicate_keys": [list(key) for key in duplicate_keys],
        "process_ids": {key: sorted(value) for key, value in process_ids.items()},
        "campaign_identity_hashes": sorted(all_identity_hashes),
        "balance": balance_audit,
    }


def validate_timing(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    counters: Counter[str] = Counter()
    for row in records:
        label = f"{row.get('session_id')}/{row.get('dialogue_id')}/{row.get('condition')}"
        try:
            arrival = int(row["last_segment_arrival_ns"])
            endpoint = int(row["endpoint_accept_ns"])
            ready_ns = int(row["first_token_ready_ns"])
            first = int(row["first_deliverable_token_ns"])
            consumer = int(row["consumer_delivery_ns"])
            done = int(row["generation_done_ns"])
            ttft = int(row["ttft_eff_ns"])
            ready = int(row["ready_tokens"])
            survived = bool(row["survived"])
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"{label}: malformed timing field ({error})")
            continue
        if not (0 <= arrival <= endpoint <= first <= consumer <= done):
            errors.append(f"{label}: illegal timestamp ordering")
        if ready_ns < arrival:
            errors.append(f"{label}: first token ready precedes final segment arrival")
        if int(row.get("oracle_preaccept_processing_ns", -1)) != endpoint - arrival:
            errors.append(f"{label}: oracle preaccept duration mismatch")
        if int(row.get("arrival_to_first_token_ready_ns", -1)) != ready_ns - arrival:
            errors.append(f"{label}: arrival-to-ready duration mismatch")
        expected_ready = row.get("candidate_first_token_ns") if bool(row.get("survived")) else first
        if expected_ready is not None and ready_ns != int(expected_ready):
            errors.append(f"{label}: first_token_ready source mismatch")
        expected = 0 if survived and ready > 0 else first - endpoint
        if ttft != expected:
            errors.append(f"{label}: ttft_eff {ttft} != {expected}")
        if int(row.get("consumer_delivery_latency_ns", -1)) != consumer - endpoint:
            errors.append(f"{label}: consumer delivery latency mismatch")
        condition = str(row.get("condition"))
        if condition == NEVER_SPECULATE and int(row.get("n_speculations", -1)) != 0:
            errors.append(f"{label}: never_speculate has speculation")
        if condition == SYSTEM_A and int(row.get("n_speculations", -1)) != 0:
            errors.append(f"{label}: System A has speculation")
        if survived:
            counters["survived"] += 1
            if ready <= 0 or ttft != 0:
                errors.append(f"{label}: survived record lacks ready zero-TTFT candidate")
        if int(row.get("waste_denominator_tokens", -1)) != int(
            row.get("wasted_tokens", 0)
        ) + int(row.get("final_tokens", 0)):
            errors.append(f"{label}: waste denominator mismatch")
        if condition in THRESHOLD_CONDITIONS:
            counters["threshold_records"] += 1
        for count_field in (
            "ready_tokens",
            "n_speculations",
            "n_invalidated",
            "wasted_tokens",
            "speculative_tokens",
            "waste_denominator_tokens",
            "final_tokens",
        ):
            if int(row.get(count_field, -1)) < 0:
                errors.append(f"{label}: negative {count_field}")
    return {"ok": not errors, "errors": errors, "counters": dict(counters)}


def validate_manifests(
    campaign_dir: Path, record_paths: list[Path], *, formal: bool
) -> dict[str, Any]:
    errors: list[str] = []
    sources: list[dict[str, Any]] = []
    campaign_manifest_path = campaign_dir / "campaign_manifest.json"
    campaign_manifest: dict[str, Any] | None = None
    campaign_manifest_sha256: str | None = None
    if formal:
        try:
            campaign_manifest = load_campaign_manifest(campaign_manifest_path)
            campaign_manifest_sha256 = sha256_file(campaign_manifest_path)
            if not campaign_manifest.get("config", {}).get("formal"):
                errors.append("Top-level campaign manifest is not formal")
            if campaign_manifest.get("config", {}).get("runtime") != "transformers":
                errors.append("Top-level campaign manifest is not a transformers campaign")
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"Invalid top-level campaign manifest: {error}")
    manifest_identities: set[str] = set()
    trigger_hashes: set[str] = set()
    input_hashes: set[str] = set()
    campaign_manifest_hashes: set[str] = set()
    runtime_metadata_values: set[str] = set()
    for records_path in record_paths:
        manifest_path = records_path.parent / "manifest.json"
        if not manifest_path.exists():
            errors.append(f"Missing session manifest: {manifest_path}")
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        config = manifest.get("config", {})
        identity = config.get("campaign_identity", {})
        manifest_identities.add(str(identity.get("identity_hash")))
        if formal and not config.get("formal"):
            errors.append(f"Session manifest is not formal: {manifest_path}")
        if campaign_manifest is not None:
            if identity != campaign_manifest.get("campaign_identity"):
                errors.append(f"Session identity differs from top-level campaign manifest: {manifest_path}")
            if config.get("runtime") != campaign_manifest.get("config", {}).get("runtime"):
                errors.append(f"Session runtime differs from top-level campaign manifest: {manifest_path}")
        if config.get("formal"):
            if not config.get("strict_offline"):
                errors.append(f"Formal manifest lacks strict_offline: {manifest_path}")
            if config.get("hf_hub_offline") != "1" or config.get("transformers_offline") != "1":
                errors.append(f"Formal manifest lacks offline environment flags: {manifest_path}")
            if not config.get("hf_token_empty"):
                errors.append(f"Formal manifest recorded a non-empty HF_TOKEN: {manifest_path}")
            if not config.get("hugging_face_hub_token_empty"):
                errors.append(
                    f"Formal manifest recorded a non-empty HUGGING_FACE_HUB_TOKEN: {manifest_path}"
                )
            if manifest.get("git", {}).get("dirty"):
                errors.append(f"Formal manifest records a dirty tree: {manifest_path}")
        trigger_hashes.add(str(identity.get("trigger_cache", {}).get("sha256")))
        input_hashes.add(str(identity.get("input", {}).get("sha256")))
        session_campaign_manifest_sha = manifest.get("extra", {}).get(
            "campaign_manifest_sha256"
        )
        campaign_manifest_hashes.add(str(session_campaign_manifest_sha))
        if formal and session_campaign_manifest_sha != campaign_manifest_sha256:
            errors.append(
                f"Session manifest does not reference the top-level campaign manifest: {manifest_path}"
            )
        runtime_metadata_values.add(json.dumps(config.get("runtime_metadata", {}), sort_keys=True))
        rows = load_jsonl(records_path)
        record_trigger = {str(row.get("trigger_cache_sha256")) for row in rows}
        record_input = {str(row.get("input_sha256")) for row in rows}
        record_identity = {str(row.get("campaign_identity_hash")) for row in rows}
        record_campaign_manifest = {
            str(row.get("campaign_manifest_sha256")) for row in rows
        }
        record_runtime_metadata = {
            json.dumps(
                {
                    "resolved_dtype": row.get("resolved_dtype"),
                    "attention_backend": row.get("attention_backend"),
                },
                sort_keys=True,
            )
            for row in rows
        }
        if record_trigger != {identity.get("trigger_cache", {}).get("sha256")}:
            errors.append(f"Trigger hash mismatch in {records_path}")
        if record_input != {identity.get("input", {}).get("sha256")}:
            errors.append(f"Input hash mismatch in {records_path}")
        if record_identity != {identity.get("identity_hash")}:
            errors.append(f"Campaign identity mismatch in {records_path}")
        expected_campaign_manifest = str(
            manifest.get("extra", {}).get("campaign_manifest_sha256")
        )
        if record_campaign_manifest != {expected_campaign_manifest}:
            errors.append(f"Campaign manifest hash mismatch in {records_path}")
        expected_runtime_metadata = json.dumps(config.get("runtime_metadata", {}), sort_keys=True)
        if record_runtime_metadata != {expected_runtime_metadata}:
            errors.append(f"Runtime metadata mismatch in {records_path}")
        sources.append(
            {
                "session_id": manifest.get("run_id"),
                "manifest_path": str(manifest_path.resolve()),
                "manifest_sha256": sha256_file(manifest_path),
                "records_path": str(records_path.resolve()),
                "records_sha256": sha256_file(records_path),
                "records": len(rows),
            }
        )
    if len(manifest_identities) != 1:
        errors.append("Session manifests contain multiple campaign identities")
    if len(trigger_hashes) != 1:
        errors.append("Session manifests contain multiple trigger hashes")
    if len(input_hashes) != 1:
        errors.append("Session manifests contain multiple input hashes")
    if len(campaign_manifest_hashes) != 1:
        errors.append("Session manifests contain multiple campaign manifest hashes")
    if formal and campaign_manifest_sha256 is not None and campaign_manifest_hashes != {
        campaign_manifest_sha256
    }:
        errors.append("Session manifests do not share the real top-level campaign manifest SHA-256")
    if len(runtime_metadata_values) != 1:
        errors.append("Sessions differ in resolved dtype or attention backend")
    return {
        "ok": not errors,
        "errors": errors,
        "campaign_identity_hashes": sorted(manifest_identities),
        "trigger_cache_sha256": sorted(trigger_hashes),
        "input_sha256": sorted(input_hashes),
        "campaign_manifest_sha256": sorted(campaign_manifest_hashes),
        "top_level_campaign_manifest": (
            {
                "path": str(campaign_manifest_path.resolve()),
                "sha256": campaign_manifest_sha256,
                "content_hash": campaign_manifest.get("manifest_content_hash"),
                "identity_hash": campaign_manifest.get("identity_hash"),
            }
            if campaign_manifest is not None
            else None
        ),
        "runtime_metadata": [json.loads(value) for value in sorted(runtime_metadata_values)],
        "sources": sources,
    }


def validate_campaign(
    campaign_dir: Path,
    *,
    expected_sessions: int = FORMAL_SESSION_COUNT,
    expected_dialogues: int = FORMAL_DIALOGUE_COUNT,
    formal: bool = True,
) -> dict[str, Any]:
    if formal and (
        expected_sessions != FORMAL_SESSION_COUNT
        or expected_dialogues != FORMAL_DIALOGUE_COUNT
    ):
        raise ValidationError(
            "Formal validator dimensions are frozen at 5 sessions × 100 dialogues"
        )
    records, paths = load_campaign_records(campaign_dir)
    grid = validate_grid(
        records,
        expected_sessions=expected_sessions,
        expected_dialogues=expected_dialogues,
        formal=formal,
    )
    timing = validate_timing(records)
    manifests = validate_manifests(campaign_dir, paths, formal=formal)
    errors = [*grid["errors"], *timing["errors"], *manifests["errors"]]
    return {
        "schema_version": 1,
        "experiment": "e1e2_confirmatory",
        "validated_at_utc": utc_now(),
        "campaign_dir": str(campaign_dir.resolve()),
        "ok": not errors,
        "errors": errors,
        "grid": grid,
        "timing": timing,
        "provenance": manifests,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--expected-sessions", type=int, default=FORMAL_SESSION_COUNT)
    parser.add_argument("--expected-dialogues", type=int, default=FORMAL_DIALOGUE_COUNT)
    parser.add_argument("--non-formal", action="store_true")
    args = parser.parse_args()
    result = validate_campaign(
        args.campaign_dir,
        expected_sessions=args.expected_sessions,
        expected_dialogues=args.expected_dialogues,
        formal=not args.non_formal,
    )
    output = args.out or args.campaign_dir / "validation.json"
    if output.exists():
        raise FileExistsError(f"Validation output already exists: {output}")
    atomic_write_json(output, result)
    if not result["ok"]:
        raise SystemExit("Validation failed: " + "; ".join(result["errors"][:10]))
    print(output)


if __name__ == "__main__":
    main()
