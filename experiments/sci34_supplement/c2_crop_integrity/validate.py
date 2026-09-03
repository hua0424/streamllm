"""Independent fail-closed validator for C2 v3 raw crop-integrity records."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

from experiments.sci34_supplement.common import atomic_write_json, load_jsonl, sha256_file, utc_now
from experiments.sci34_supplement.c2_crop_integrity.campaign import code_identity, load_campaign_manifest
from experiments.sci34_supplement.c2_crop_integrity.integrity import (
    EXACT_BOOLEAN_FIELDS,
    manifest_aggregate,
    manifests_equal,
    record_content_hash,
    validate_ledger,
    validate_manifest,
)
from experiments.sci34_supplement.c2_crop_integrity.canonical_chat import token_ids_hash
from experiments.sci34_supplement.c2_crop_integrity.protocol import (
    EXPECTED_CASES_SHA256,
    EXPECTED_DTYPE,
    EXPECTED_MODEL_ARCHITECTURE,
    EXPECTED_MODEL_ARTIFACT_HASH,
    EXPECTED_MODEL_TYPE,
    FORMAL_CASE_COUNT,
    FORMAL_CROP_EVENT_COUNT,
    PRIOR_V2_RUN_ID,
    PROTOCOL_VERSION,
    expected_event_grid,
    load_cases,
)


class ValidationError(ValueError):
    pass


def _check(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def _validate_fixture(record: Mapping[str, Any], *, eot_token_id: int) -> list[str]:
    label = f"{record.get('case_id')}/fixture"
    fixture = record.get("fixture", {})
    errors: list[str] = []
    ids = fixture.get("assistant_token_ids")
    _check(errors, isinstance(ids, list) and bool(ids) and all(isinstance(value, int) for value in ids or []), f"{label}: token IDs malformed")
    if isinstance(ids, list):
        _check(errors, fixture.get("assistant_token_hash") == token_ids_hash(ids), f"{label}: token hash differs")
        _check(errors, eot_token_id not in ids and fixture.get("all_non_eot") is True, f"{label}: fixture contains EOT")
        _check(errors, fixture.get("prefill_ids_p2_calls") == len(ids), f"{label}: _prefill_ids_p2 call count differs")
    _check(errors, fixture.get("generate_api") == "StreamLLMInference.generate_accumulating", f"{label}: wrong generate API")
    _check(errors, fixture.get("one_token_per_forward") is True, f"{label}: not one token per forward")
    second_ids = fixture.get("second_assistant_token_ids")
    if second_ids is not None:
        _check(errors, isinstance(second_ids, list) and bool(second_ids) and eot_token_id not in second_ids, f"{label}: second fixture malformed/EOT")
        if isinstance(second_ids, list):
            _check(errors, fixture.get("second_assistant_token_hash") == token_ids_hash(second_ids), f"{label}: second fixture hash differs")
    ledger = fixture.get("event_ledger")
    if isinstance(ids, list):
        expected = [{"operation": "generate_accumulating_token", "token_ids": [value]} for value in ids]
        initial = int(record.get("context_tokens_actual", -1))
        errors.extend(validate_ledger(ledger, arm="production", initial_length=initial, expected_chunks=expected, eot_token_id=eot_token_id))

    second_ids = fixture.get("second_assistant_token_ids")
    second_ledger = fixture.get("second_assistant_event_ledger")
    events = record.get("crop_events")
    if isinstance(second_ids, list):
        _check(errors, bool(second_ids), f"{label}: second assistant IDs are empty")
        _check(errors, eot_token_id not in second_ids, f"{label}: second assistant fixture contains EOT")
        _check(
            errors,
            fixture.get("second_assistant_token_hash") == token_ids_hash(second_ids),
            f"{label}: second assistant token hash differs",
        )
        crop_1 = events[0] if isinstance(events, list) and events else {}
        first_final = crop_1.get("final_token_ids") if isinstance(crop_1, dict) else None
        second_initial = len(first_final) if isinstance(first_final, list) else -1
        second_expected = [
            {"operation": "generate_accumulating_token", "token_ids": [value]}
            for value in second_ids
        ]
        errors.extend(
            validate_ledger(
                second_ledger,
                arm="production",
                initial_length=second_initial,
                expected_chunks=second_expected,
                eot_token_id=eot_token_id,
            )
        )
    else:
        _check(
            errors,
            second_ledger is None and fixture.get("second_assistant_token_hash") is None,
            f"{label}: unexpected second assistant fixture/ledger",
        )
    return errors


def _validate_recovery_check(
    check: Mapping[str, Any],
    *,
    label: str,
    expected_chunk: Mapping[str, Any],
    original_prefix: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    _check(errors, check.get("operation") == expected_chunk.get("operation"), f"{label}: operation differs")
    ids = check.get("token_ids")
    _check(errors, ids == expected_chunk.get("token_ids"), f"{label}: token chunk differs")
    if isinstance(ids, list):
        _check(errors, check.get("token_hash") == token_ids_hash(ids), f"{label}: token hash differs")
    for name in ("production_manifest", "oracle_manifest", "production_prefix_manifest", "oracle_prefix_manifest"):
        manifest = check.get(name)
        if not isinstance(manifest, dict):
            errors.append(f"{label}: missing {name}")
        else:
            errors.extend(validate_manifest(manifest, label=f"{label}/{name}"))
    production = check.get("production_manifest", {})
    oracle = check.get("oracle_manifest", {})
    production_prefix = check.get("production_prefix_manifest", {})
    oracle_prefix = check.get("oracle_prefix_manifest", {})
    _check(errors, manifests_equal(production, oracle), f"{label}: production/oracle K/V manifests differ")
    _check(errors, manifests_equal(production_prefix, original_prefix), f"{label}: production retained prefix changed")
    _check(errors, manifests_equal(oracle_prefix, original_prefix), f"{label}: oracle retained prefix changed")
    _check(errors, check.get("production_logits_sha256") == check.get("oracle_logits_sha256"), f"{label}: logits hashes differ")
    for field in ("kv_exact", "logits_exact", "masks_exact", "token_ids_exact", "retained_prefix_hash_exact", "production_state_exact", "passed"):
        _check(errors, check.get(field) is True, f"{label}: {field} is not exactly true")
    operation = str(expected_chunk.get("operation"))
    after_length = None
    state = check.get("production_state")
    if isinstance(state, dict):
        after_length = state.get("seq_length")
    if operation in {"reopen_user_role", "prefill_user_text"}:
        expected_state = {
            "role_phase": "USER_OPEN",
            "generation_end_reason": "NONE",
            "assistant_content_start": None,
            "assistant_token_count": 0,
            "seq_length": after_length,
            "mask_length": after_length,
            "kv_length": after_length,
            "token_ledger_length": after_length,
        }
    elif operation == "open_assistant_role":
        expected_state = {
            "role_phase": "ASSISTANT_OPEN",
            "generation_end_reason": "NONE",
            "assistant_content_start": after_length,
            "assistant_token_count": 0,
            "seq_length": after_length,
            "mask_length": after_length,
            "kv_length": after_length,
            "token_ledger_length": after_length,
        }
    else:
        expected_state = None
        errors.append(f"{label}: unsupported recovery operation")
    _check(errors, isinstance(state, dict) and state == expected_state, f"{label}: production role/end/content state differs")
    _check(errors, check.get("expected_state") == expected_state, f"{label}: stored expected state differs")
    _check(errors, isinstance(check.get("errors"), list) and not check.get("errors"), f"{label}: runtime errors present")
    return errors


def _validate_event(
    event: Mapping[str, Any],
    *,
    case: Any,
    expected_id: str,
    event_index: int,
    eot_token_id: int,
    token_plan: Mapping[str, Any],
) -> list[str]:
    label = f"{case.id}/{expected_id}"
    errors: list[str] = []
    _check(errors, event.get("event_id") == expected_id and event.get("event_index") == event_index, f"{label}: event identity/order differs")
    pre_len = event.get("pre_crop_length")
    keep = event.get("keep_length")
    pre_ids = event.get("pre_crop_token_ids")
    retained = event.get("retained_token_ids")
    _check(errors, isinstance(pre_len, int) and isinstance(keep, int) and 0 <= keep <= pre_len, f"{label}: keep/pre length malformed")
    _check(errors, isinstance(pre_ids, list) and len(pre_ids) == pre_len, f"{label}: pre-crop IDs/length differ")
    if isinstance(pre_ids, list):
        _check(errors, event.get("pre_crop_token_hash") == token_ids_hash(pre_ids), f"{label}: pre-crop token hash differs")
        _check(errors, retained == pre_ids[:keep], f"{label}: retained IDs are not exact prefix")
    if isinstance(retained, list):
        _check(errors, len(retained) == keep and event.get("retained_token_hash") == token_ids_hash(retained), f"{label}: retained token hash/length differs")
    _check(errors, event.get("no_op") is (keep == pre_len), f"{label}: no-op flag differs")
    content_start = event.get("assistant_content_start")
    role_start = event.get("assistant_role_start")
    if expected_id == "crop_1":
        fragment_ids = event.get("fragment_token_ids")
        fixture_ids = token_plan.get("assistant_token_ids")
        _check(errors, event.get("retain_fragment_count") == case.retain_fragment_count, f"{label}: retain_fragment_count differs")
        _check(errors, isinstance(fragment_ids, list) and len(fragment_ids) == len(case.fragments) and all(isinstance(ids, list) and ids for ids in fragment_ids or []), f"{label}: fragment token partition malformed/empty")
        if isinstance(fragment_ids, list):
            _check(errors, [value for ids in fragment_ids for value in ids] == fixture_ids, f"{label}: fragment token partition does not concatenate to frozen fixture IDs")
        if case.scenario == "reply_tail_noop":
            independently_expected_keep = pre_len
            expected_semantics = "reply_tail_noop"
        elif case.scenario == "speculation_full_invalidation":
            independently_expected_keep = role_start
            expected_semantics = "speculation_full_invalidation"
        elif case.retain_fragment_count == 0:
            independently_expected_keep = content_start
            expected_semantics = "empty_assistant_turn_p0"
        else:
            independently_expected_keep = content_start + sum(len(ids) for ids in fragment_ids[: case.retain_fragment_count]) if isinstance(content_start, int) and isinstance(fragment_ids, list) else None
            expected_semantics = "retained_fragment_prefix"
        _check(errors, event.get("second_assistant_token_ids") is None and event.get("second_crop_fraction") is None, f"{label}: second-crop metadata populated")
    else:
        second_ids = event.get("second_assistant_token_ids")
        fraction = event.get("second_crop_fraction")
        _check(errors, event.get("fragment_token_ids") is None and event.get("retain_fragment_count") is None, f"{label}: first-crop fragment metadata populated")
        _check(errors, isinstance(second_ids, list) and bool(second_ids), f"{label}: second assistant IDs malformed")
        _check(errors, second_ids == token_plan.get("second_assistant_token_ids"), f"{label}: second assistant IDs differ from frozen token plan")
        _check(errors, fraction == case.second_crop_fraction, f"{label}: second crop fraction differs")
        retained_second = max(1, min(len(second_ids), math.floor(len(second_ids) * float(case.second_crop_fraction)))) if isinstance(second_ids, list) and second_ids else None
        independently_expected_keep = content_start + retained_second if isinstance(content_start, int) and isinstance(retained_second, int) else None
        expected_semantics = "second_fraction_floor_clamp"
    _check(errors, isinstance(role_start, int) and isinstance(content_start, int) and role_start < content_start <= pre_len, f"{label}: assistant role/content boundaries malformed")
    _check(errors, keep == independently_expected_keep, f"{label}: keep length differs from independent case/partition derivation")
    _check(errors, event.get("crop_target_semantics") == expected_semantics, f"{label}: crop target semantics differs")
    manifests: dict[str, Mapping[str, Any]] = {}
    for name in ("pre_prefix_manifest", "post_production_manifest", "oracle_manifest"):
        value = event.get(name)
        if not isinstance(value, dict):
            errors.append(f"{label}: missing {name}")
            continue
        manifests[name] = value
        errors.extend(validate_manifest(value, label=f"{label}/{name}"))
    if len(manifests) == 3:
        pre = manifests["pre_prefix_manifest"]
        post = manifests["post_production_manifest"]
        oracle = manifests["oracle_manifest"]
        _check(errors, manifests_equal(pre, post), f"{label}: production crop differs from pre-crop retained prefix")
        _check(errors, manifests_equal(pre, oracle), f"{label}: oracle clone differs from pre-crop retained prefix")
        _check(errors, manifests_equal(post, oracle), f"{label}: production crop differs from oracle")
        _check(errors, pre.get("aggregate_sha256") == post.get("aggregate_sha256") == oracle.get("aggregate_sha256"), f"{label}: aggregate hashes differ")
        for manifest_name, manifest in manifests.items():
            for layer in manifest.get("layers", []):
                for side in ("key", "value"):
                    shape = layer.get(side, {}).get("shape")
                    _check(
                        errors,
                        isinstance(shape, list) and len(shape) >= 2 and shape[-2] == keep,
                        f"{label}/{manifest_name}: layer {layer.get('layer')} {side} sequence dimension differs from keep",
                    )
    for field in EXACT_BOOLEAN_FIELDS:
        _check(errors, event.get(field) is True, f"{label}: {field} is not exactly true")
    post_crop_state = event.get("post_crop_state")
    oracle_crop_state = event.get("post_crop_oracle_state")
    _check(errors, event.get("post_crop_lengths_exact") is True, f"{label}: post-crop length exact gate failed")
    _check(errors, event.get("post_crop_mask_exact") is True, f"{label}: post-crop mask exact gate failed")
    _check(errors, event.get("post_crop_token_ids_exact") is True, f"{label}: post-crop token ledger exact gate failed")
    expected_crop_lengths = {
        "seq_length": keep,
        "mask_length": keep,
        "kv_length": keep,
        "token_ledger_length": keep,
    }
    if isinstance(post_crop_state, dict):
        _check(
            errors,
            {key: post_crop_state.get(key) for key in expected_crop_lengths} == expected_crop_lengths,
            f"{label}: production post-crop lengths differ",
        )
        expected_phase = "USER_OPEN" if expected_semantics == "speculation_full_invalidation" else "ASSISTANT_OPEN"
        expected_content_start = None if expected_phase == "USER_OPEN" else content_start
        expected_assistant_count = 0 if expected_phase == "USER_OPEN" else keep - content_start
        expected_end = "CROPPED" if keep < pre_len else "MAX_TOKENS"
        _check(errors, post_crop_state.get("role_phase") == expected_phase, f"{label}: post-crop role phase differs")
        _check(errors, post_crop_state.get("generation_end_reason") == expected_end, f"{label}: post-crop end reason differs")
        _check(errors, post_crop_state.get("assistant_content_start") == expected_content_start and post_crop_state.get("assistant_token_count") == expected_assistant_count, f"{label}: post-crop assistant state differs")
    else:
        errors.append(f"{label}: post-crop production state missing")
    _check(errors, oracle_crop_state == expected_crop_lengths, f"{label}: oracle post-crop lengths differ")
    chunks = event.get("expected_recovery_chunks")
    production_ledger = event.get("production_event_ledger")
    oracle_ledger = event.get("oracle_event_ledger")
    _check(errors, isinstance(chunks, list), f"{label}: recovery chunks malformed")
    if isinstance(chunks, list) and isinstance(keep, int):
        errors.extend(validate_ledger(production_ledger, arm="production", initial_length=keep, expected_chunks=chunks, eot_token_id=eot_token_id))
        errors.extend(validate_ledger(oracle_ledger, arm="oracle", initial_length=keep, expected_chunks=chunks, eot_token_id=eot_token_id))
        if isinstance(production_ledger, list) and isinstance(oracle_ledger, list):
            _check(
                errors,
                [entry.get("token_ids") for entry in production_ledger] == [entry.get("token_ids") for entry in oracle_ledger],
                f"{label}: production/oracle recovery chunks differ",
            )
    checks = event.get("recovery_checks")
    _check(errors, isinstance(checks, list) and isinstance(chunks, list) and len(checks) == len(chunks), f"{label}: recovery check count differs")
    if isinstance(checks, list) and isinstance(chunks, list) and "pre_prefix_manifest" in manifests:
        for ordinal, (check, chunk) in enumerate(zip(checks, chunks)):
            if not isinstance(check, dict):
                errors.append(f"{label}/recovery/{ordinal}: malformed")
            else:
                _check(errors, check.get("ordinal") == ordinal, f"{label}/recovery/{ordinal}: ordinal differs")
                errors.extend(_validate_recovery_check(check, label=f"{label}/recovery/{ordinal}", expected_chunk=chunk, original_prefix=manifests["pre_prefix_manifest"]))
    final_ids = event.get("final_token_ids")
    canonical = event.get("canonical_ledger", {})
    _check(errors, isinstance(final_ids, list) and event.get("final_token_hash") == token_ids_hash(final_ids or []), f"{label}: final token hash differs")
    canonical_ids = canonical.get("token_ids")
    _check(errors, isinstance(canonical_ids, list) and canonical.get("token_hash") == token_ids_hash(canonical_ids or []), f"{label}: canonical token hash differs")
    _check(errors, final_ids == canonical_ids, f"{label}: final and independently canonical IDs differ")
    expected_boundaries = sum(chunk.get("operation") == "reopen_user_role" for chunk in chunks or [])
    _check(errors, canonical.get("assistant_boundaries") == expected_boundaries, f"{label}: assistant boundary count differs")
    eot_positions = canonical.get("eot_positions")
    expected_eot_positions: list[int] = []
    cursor = keep if isinstance(keep, int) else 0
    reopen_chunks_valid = True
    for chunk in chunks or []:
        chunk_ids = chunk.get("token_ids", [])
        if chunk.get("operation") == "reopen_user_role":
            offsets = [offset for offset, value in enumerate(chunk_ids) if value == eot_token_id]
            reopen_chunks_valid = reopen_chunks_valid and len(offsets) == 1
            expected_eot_positions.extend(cursor + offset for offset in offsets)
        cursor += len(chunk_ids)
    _check(
        errors,
        reopen_chunks_valid and eot_positions == expected_eot_positions and len(expected_eot_positions) == expected_boundaries,
        f"{label}: unique EOT positions differ",
    )
    _check(errors, canonical.get("unique_eot") is True and canonical.get("role_boundary_exact") is True, f"{label}: role/EOT canonical gates failed")
    _check(errors, isinstance(event.get("errors"), list) and not event.get("errors"), f"{label}: runtime errors present")
    _check(errors, event.get("passed") is True, f"{label}: runtime verdict failed")
    return errors


def validate_campaign(campaign_dir: Path, *, formal: bool = True, expected_cases: int | None = None) -> dict[str, Any]:
    errors: list[str] = []
    failed_indexes: list[dict[str, Any]] = []
    manifest_path = campaign_dir / "campaign_manifest.json"
    records_path = campaign_dir / "records.jsonl"
    cases_path = campaign_dir / "cases.json"
    try:
        manifest = load_campaign_manifest(manifest_path)
    except Exception as error:
        return {"ok": False, "acceptance_eligible": False, "errors": [f"manifest: {error}"], "failed_indexes": []}
    manifest_formal = bool(manifest.get("config", {}).get("formal"))
    if formal and not manifest_formal:
        errors.append("formal validation requires a formal campaign manifest")
    cases = load_cases(cases_path, formal=formal)
    records = load_jsonl(records_path)
    expected_count = expected_cases if expected_cases is not None else (FORMAL_CASE_COUNT if formal else len(cases))
    expected_cases_slice = cases[:expected_count]
    expected_ids = [case.id for case in expected_cases_slice]
    actual_ids = [record.get("case_id") for record in records]
    _check(errors, len(records) == expected_count, f"record count {len(records)} != {expected_count}")
    _check(errors, actual_ids == expected_ids, "record case order/grid differs")
    _check(errors, len(set(actual_ids)) == len(actual_ids), "duplicate case record")
    _check(errors, sha256_file(cases_path) == EXPECTED_CASES_SHA256 == manifest.get("input", {}).get("sha256"), "cases hash/provenance differs")
    _check(errors, code_identity() == manifest.get("config", {}).get("code_identity"), "code identity differs")
    _check(errors, manifest.get("config", {}).get("protocol", {}).get("protocol_version") == PROTOCOL_VERSION, "protocol version differs")
    prior = manifest.get("config", {}).get("prior_v2_evidence", {})
    _check(errors, prior.get("run_id") == PRIOR_V2_RUN_ID and prior.get("runtime_dependency") is False and prior.get("artifact_path") is None, "immutable v2 provenance differs or became a runtime dependency")
    negative = manifest.get("config", {}).get("negative_control", {})
    _check(errors, negative.get("detected") is True and negative.get("positive_control_metadata", {}).get("wrong_crop_length_would_be_detected") is True, "formal wrong-length negative control missing")
    token_plans = manifest.get("config", {}).get("case_token_plans")
    _check(errors, isinstance(token_plans, list) and len(token_plans) == len(cases), "immutable case token plans missing or incomplete")
    token_plan_by_id = {
        str(plan.get("case_id")): plan for plan in token_plans or [] if isinstance(plan, dict)
    }
    runtime = manifest.get("config", {}).get("runtime_metadata", {})
    if formal:
        identity = manifest.get("config", {}).get("model_identity", {})
        _check(errors, manifest.get("config", {}).get("runtime") == "transformers", "formal runtime is not Transformers")
        _check(errors, runtime.get("execution") == "transformers_model", "formal execution metadata differs")
        _check(errors, runtime.get("accepted_model_artifact_hash") == EXPECTED_MODEL_ARTIFACT_HASH, "model artifact hash differs")
        _check(errors, runtime.get("model_type") == EXPECTED_MODEL_TYPE and EXPECTED_MODEL_ARCHITECTURE in runtime.get("architectures", []), "model type/architecture differs")
        _check(errors, runtime.get("resolved_dtype") == EXPECTED_DTYPE, "dtype differs")
        _check(errors, bool(identity), "strong model identity missing")
        config = manifest.get("config", {})
        _check(errors, config.get("strict_offline") is True and config.get("hf_hub_offline") == "1" and config.get("transformers_offline") == "1", "formal offline metadata differs")
        _check(errors, config.get("hf_token_empty") is True and config.get("hugging_face_hub_token_empty") is True, "formal token metadata differs")

    manifest_sha = sha256_file(manifest_path)
    event_grid: list[tuple[str, str]] = []
    eot = int(runtime.get("eot_token_id", 2))
    for index, (record, case) in enumerate(zip(records, expected_cases_slice)):
        row_errors: list[str] = []
        _check(row_errors, record.get("record_content_hash") == record_content_hash(record), f"{case.id}: record JSON content hash differs")
        _check(row_errors, record.get("protocol_version") == PROTOCOL_VERSION and record.get("experiment") == "c2_crop_integrity", f"{case.id}: record identity differs")
        _check(row_errors, record.get("case_index") == index and record.get("case_id") == case.id, f"{case.id}: case order differs")
        _check(row_errors, record.get("campaign_identity_hash") == manifest.get("identity_hash") and record.get("campaign_manifest_sha256") == manifest_sha, f"{case.id}: campaign provenance differs")
        _check(row_errors, record.get("cases_sha256") == EXPECTED_CASES_SHA256, f"{case.id}: cases hash differs")
        _check(row_errors, record.get("prior_v2_run_id") == PRIOR_V2_RUN_ID and record.get("prior_v2_probe_reused") is True and record.get("termination_probe_rerun") is False, f"{case.id}: v2 probe provenance differs")
        _check(row_errors, record.get("formal_negative_control") == negative, f"{case.id}: negative control differs")
        token_plan = token_plan_by_id.get(case.id, {})
        _check(row_errors, token_plan.get("assistant_token_ids") == record.get("fixture", {}).get("assistant_token_ids"), f"{case.id}: fixture differs from immutable token plan")
        _check(row_errors, token_plan.get("assistant_token_hash") == record.get("fixture", {}).get("assistant_token_hash"), f"{case.id}: fixture hash differs from immutable token plan")
        _check(row_errors, token_plan.get("second_assistant_token_ids") == record.get("fixture", {}).get("second_assistant_token_ids"), f"{case.id}: second fixture differs from immutable token plan")
        _check(row_errors, token_plan.get("second_assistant_token_hash") == record.get("fixture", {}).get("second_assistant_token_hash"), f"{case.id}: second fixture hash differs from immutable token plan")
        _check(row_errors, record.get("token_plan_hash") == token_plan.get("plan_hash"), f"{case.id}: token plan hash differs")
        row_errors.extend(_validate_fixture(record, eot_token_id=eot))
        events = record.get("crop_events")
        expected_names = ["crop_1"] + (["crop_2"] if case.second_crop_fraction is not None else [])
        _check(row_errors, isinstance(events, list) and len(events) == len(expected_names), f"{case.id}: crop event count differs")
        if isinstance(events, list):
            for event_index, (event, name) in enumerate(zip(events, expected_names)):
                event_grid.append((case.id, name))
                if not isinstance(event, dict):
                    row_errors.append(f"{case.id}/{name}: malformed event")
                else:
                    row_errors.extend(
                        _validate_event(
                            event,
                            case=case,
                            expected_id=name,
                            event_index=event_index,
                            eot_token_id=eot,
                            token_plan=token_plan,
                        )
                    )
        _check(row_errors, isinstance(record.get("errors"), list) and not record.get("errors"), f"{case.id}: runtime errors present")
        _check(row_errors, record.get("passed") is True, f"{case.id}: runtime verdict failed")
        if row_errors:
            failed_indexes.append({"case_id": case.id, "errors": row_errors})
            errors.extend(row_errors)
    expected_grid = expected_event_grid(expected_cases_slice)
    _check(errors, event_grid == expected_grid, "crop-event grid/order differs")
    if formal:
        _check(errors, len(event_grid) == FORMAL_CROP_EVENT_COUNT, f"formal event count {len(event_grid)} != {FORMAL_CROP_EVENT_COUNT}")
    summary_path = campaign_dir / "summary.json"
    if not summary_path.is_file():
        errors.append("summary.json missing")
    else:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        _check(errors, summary.get("cases") == len(records) and summary.get("crop_events") == len(event_grid), "summary counts differ")
        _check(errors, summary.get("termination_probe_reruns") == 0, "summary reports termination probe rerun")
        _check(errors, summary.get("status") == "PASS" and summary.get("acceptance_eligible") is True, "summary is not acceptance eligible")
    ok = not errors
    return {
        "schema_version": 1,
        "experiment": "c2_crop_integrity",
        "protocol_version": PROTOCOL_VERSION,
        "campaign_dir": str(campaign_dir.resolve()),
        "formal": formal,
        "ok": ok,
        "acceptance_eligible": ok,
        "errors": errors,
        "failed_indexes": failed_indexes,
        "grid": {
            "cases": len(records),
            "expected_cases": expected_count,
            "crop_events": len(event_grid),
            "expected_crop_events": FORMAL_CROP_EVENT_COUNT if formal else len(expected_grid),
            "case_order_exact": actual_ids == expected_ids,
            "event_order_exact": event_grid == expected_grid,
        },
        "exact_gates": {field: all(event.get(field) is True for row in records for event in row.get("crop_events", [])) for field in EXACT_BOOLEAN_FIELDS},
        "provenance": {
            "manifest_sha256": manifest_sha,
            "cases_sha256": sha256_file(cases_path),
            "records_sha256": sha256_file(records_path) if records_path.is_file() else None,
            "prior_v2_run_id": PRIOR_V2_RUN_ID,
            "termination_probe_rerun": False,
        },
        "validated_at_utc": utc_now(),
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate C2 v3 crop-integrity artifacts")
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--non-formal", action="store_true")
    parser.add_argument("--expected-cases", type=int)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    result = validate_campaign(args.campaign_dir, formal=not args.non_formal, expected_cases=args.expected_cases)
    output = args.out or args.campaign_dir / "validation.json"
    atomic_write_json(output, result)
    print(json.dumps({"ok": result["ok"], "errors": result["errors"][:10]}, sort_keys=True))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
