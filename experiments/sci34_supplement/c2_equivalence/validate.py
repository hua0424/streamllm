"""Independent, fail-closed validation of raw C2 correctness records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from experiments.sci34_supplement.common import atomic_write_json, load_jsonl, sha256_file, utc_now
from experiments.sci34_supplement.c2_equivalence.campaign import (
    code_identity,
    load_campaign_manifest,
)
from experiments.sci34_supplement.c2_equivalence.canonical_chat import first_mismatch, token_ids_hash
from experiments.sci34_supplement.c2_equivalence.protocol import (
    BF16_MAX_ABS_THRESHOLD,
    BF16_MEAN_ABS_THRESHOLD,
    CONTINUATION_TOKENS,
    EOS_AT_CAP_MAX_NEW_TOKENS,
    EXPECTED_DTYPE,
    EXPECTED_MODEL_ARCHITECTURE,
    EXPECTED_MODEL_ARTIFACT_HASH,
    EXPECTED_MODEL_TYPE,
    FORMAL_CASE_COUNT,
    MAX_TOKENS_PROBE_BUDGET,
    NATURAL_EOS_MAX_NEW_TOKENS,
    TERMINATION_PROBE_SCHEMA_VERSION,
    TOP_K_MIN_OVERLAP,
    CaseSpec,
    load_cases,
)


class ValidationError(ValueError):
    pass


def _error(errors: list[str], label: str, condition: bool, detail: str) -> None:
    if not condition:
        errors.append(f"{label}: {detail}")


def _validate_termination_probe(
    probe: Mapping[str, Any], *, case: CaseSpec, formal: bool
) -> list[str]:
    label = f"{case.id}/termination_probe"
    errors: list[str] = []
    expected_cap = {
        "natural_eos": NATURAL_EOS_MAX_NEW_TOKENS,
        "eos_at_cap": EOS_AT_CAP_MAX_NEW_TOKENS,
        "max_tokens": MAX_TOKENS_PROBE_BUDGET,
    }[case.termination]
    _error(errors, label, probe.get("schema_version") == TERMINATION_PROBE_SCHEMA_VERSION, "schema differs")
    _error(errors, label, probe.get("declared") == case.termination, "declared label differs from case")
    _error(errors, label, probe.get("generate_api") == "StreamLLMInference.generate_accumulating", "wrong generation API")
    if formal:
        _error(errors, label, probe.get("execution") == "transformers_model", "formal probe is not Transformers execution")
    _error(errors, label, probe.get("temperature") == 0.0, "probe is not greedy")
    _error(errors, label, probe.get("top_p") == 1.0, "top_p differs")
    _error(errors, label, probe.get("repetition_penalty") == 1.0, "repetition penalty differs")
    _error(errors, label, probe.get("cap") == expected_cap, "frozen cap differs")
    ids = probe.get("content_token_ids")
    selected_ids = probe.get("selected_token_ids")
    _error(errors, label, isinstance(ids, list) and all(isinstance(item, int) for item in ids or []), "content token IDs malformed")
    _error(errors, label, isinstance(selected_ids, list) and all(isinstance(item, int) for item in selected_ids or []), "selected token IDs malformed")
    if isinstance(ids, list):
        _error(errors, label, probe.get("content_token_count") == len(ids), "content token count differs")
        _error(errors, label, probe.get("content_token_hash") == token_ids_hash(ids), "content token hash differs")
        _error(errors, label, probe.get("eot_token_id") not in ids, "EOT appears in content IDs")
    if isinstance(selected_ids, list):
        _error(errors, label, probe.get("selected_token_count") == len(selected_ids), "selected token count differs")
        _error(errors, label, probe.get("selected_token_hash") == token_ids_hash(selected_ids), "selected token hash differs")
    _error(errors, label, probe.get("eot_in_kv") is False, "pending EOT entered KV")
    _error(errors, label, probe.get("eot_in_full_ledger") is False, "pending EOT entered full ledger")
    _error(errors, label, probe.get("eot_in_content_ledger") is False, "pending EOT entered content ledger")
    prefix = probe.get("prefix_seq_length")
    post = probe.get("post_seq_length")
    count = probe.get("content_token_count")
    _error(errors, label, all(isinstance(value, int) for value in (prefix, post, count)), "probe lengths malformed")
    if all(isinstance(value, int) for value in (prefix, post, count)):
        _error(errors, label, post == prefix + count, "post length does not equal prefix plus content")
    observed = probe.get("observed_end_reason")
    eos_step = probe.get("eos_step")
    if isinstance(ids, list) and isinstance(selected_ids, list):
        expected_selected = [*ids, probe.get("eot_token_id")] if observed == "EOS" else ids
        _error(errors, label, selected_ids == expected_selected, "selected/content/EOT sequence differs")
    if case.termination == "natural_eos":
        _error(errors, label, probe.get("mode") == "real_greedy" and probe.get("controlled") is False, "natural EOS is not real greedy")
        _error(errors, label, observed == "EOS", f"observed end reason is {observed!r}")
        _error(errors, label, isinstance(eos_step, int) and 1 <= eos_step <= expected_cap, "EOS not reached within cap")
        _error(errors, label, probe.get("role_phase") == "ASSISTANT_EOT_PENDING", "EOS phase differs")
    elif case.termination == "eos_at_cap":
        _error(errors, label, case.controlled_fixture, "case is not marked controlled")
        _error(errors, label, probe.get("mode") == "controlled_logits_fixture" and probe.get("controlled") is True, "cap probe is not controlled")
        _error(errors, label, observed == "EOS", f"observed end reason is {observed!r}")
        _error(errors, label, eos_step == expected_cap and probe.get("eos_at_cap") is True, "EOT is not exactly at cap")
        fixture_ids = probe.get("fixture_token_ids")
        _error(errors, label, isinstance(fixture_ids, list) and len(fixture_ids) == expected_cap, "controlled fixture length differs")
        if isinstance(fixture_ids, list) and isinstance(ids, list):
            _error(errors, label, fixture_ids[:-1] == ids, "controlled content differs from fixture")
            _error(errors, label, fixture_ids[-1:] == [probe.get("eot_token_id")], "fixture does not end in EOT")
        _error(errors, label, bool(probe.get("fixture_description")), "controlled fixture is not described")
        _error(errors, label, probe.get("role_phase") == "ASSISTANT_EOT_PENDING", "EOS phase differs")
    else:
        _error(errors, label, probe.get("mode") == "real_greedy" and probe.get("controlled") is False, "max-token probe is not real greedy")
        _error(errors, label, observed == "MAX_TOKENS", f"observed end reason is {observed!r}")
        _error(errors, label, eos_step is None and probe.get("eos_at_cap") is False, "max-token probe recorded EOS")
        _error(errors, label, isinstance(ids, list) and len(ids) == expected_cap, "max-token content count differs")
        _error(errors, label, probe.get("role_phase") == "ASSISTANT_OPEN", "max-token phase differs")
    stored_errors = probe.get("errors")
    _error(errors, label, isinstance(stored_errors, list), "stored errors are malformed")
    if isinstance(stored_errors, list):
        _error(errors, label, not stored_errors, "runtime probe reported errors")
    _error(errors, label, bool(probe.get("passed")) == (not errors), "stored probe verdict differs")
    return errors


def _validate_scenario_execution(
    probe: Mapping[str, Any], *, case: CaseSpec, formal: bool
) -> list[str]:
    label = f"{case.id}/scenario_execution"
    errors: list[str] = []
    applies = case.scenario in {"crop_pending_eot", "reply_tail_noop"}
    _error(errors, label, probe.get("schema_version") == 1, "schema differs")
    _error(errors, label, probe.get("scenario") == case.scenario, "scenario differs")
    _error(errors, label, probe.get("applies") is applies, "applicability differs")
    if formal:
        _error(errors, label, probe.get("execution") == "transformers_model", "formal scenario probe is not Transformers execution")
    if applies:
        _error(errors, label, probe.get("generate_api") == "StreamLLMInference.generate_accumulating", "wrong generation API")
        _error(errors, label, probe.get("forced_decode_token") == "EOT", "forced token is not EOT")
        _error(errors, label, probe.get("generate_max_new_tokens") == 1, "generation budget differs")
        _error(errors, label, probe.get("pending_before_crop") is True, "EOT pending was not observed before crop")
        _error(errors, label, probe.get("eot_in_full_ledger_before_crop") is False, "pending EOT entered full ledger")
        _error(errors, label, probe.get("eot_in_content_ledger_before_crop") is False, "pending EOT entered content ledger")
        _error(errors, label, probe.get("reopen_called") is True, "reopen_user_role was not called")
        if formal:
            _error(errors, label, probe.get("seq_unchanged_by_pending_eot") is True, "pending EOT changed seq length")
            _error(errors, label, probe.get("full_ledger_unchanged_by_pending_eot") is True, "pending EOT changed full ledger")
            _error(errors, label, probe.get("content_ledger_unchanged_by_pending_eot") is True, "pending EOT changed content ledger")
        if case.scenario == "crop_pending_eot":
            _error(errors, label, probe.get("crop_target") == "retained_boundary", "wrong crop target")
            _error(errors, label, probe.get("crop_was_noop") is False, "crop unexpectedly no-op")
            _error(errors, label, probe.get("pending_after_crop") is False, "pending state survived truncating crop")
            _error(errors, label, probe.get("pending_cleared_by_crop") is True, "crop did not clear pending state")
            _error(errors, label, probe.get("no_op_preserved_pending") is False, "crop marked as pending-preserving no-op")
        else:
            _error(errors, label, probe.get("crop_target") == "current_seq", "wrong no-op target")
            _error(errors, label, probe.get("crop_was_noop") is True, "tail crop was not no-op")
            _error(errors, label, probe.get("pending_after_crop") is True, "no-op did not preserve pending")
            _error(errors, label, probe.get("no_op_preserved_pending") is True, "pending preservation not recorded")
            _error(errors, label, probe.get("pending_cleared_by_crop") is False, "no-op incorrectly cleared pending")
    else:
        for field in (
            "generate_api", "forced_decode_token", "generate_max_new_tokens",
            "pending_before_crop", "crop_target", "reopen_called",
        ):
            expected = False if field == "reopen_called" else None
            _error(errors, label, probe.get(field) == expected, f"non-applicable field {field} is populated")
    stored = probe.get("errors")
    _error(errors, label, isinstance(stored, list), "stored errors malformed")
    if isinstance(stored, list):
        _error(errors, label, not stored, "runtime scenario probe reported errors")
    _error(errors, label, bool(probe.get("passed")) == (not errors), "stored scenario verdict differs")
    return errors


def _validate_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    case: CaseSpec,
    record_probe: Mapping[str, Any],
    record_scenario_execution: Mapping[str, Any],
    formal: bool,
    failure_indexes: list[dict[str, Any]],
) -> list[str]:
    case_id = case.id
    name = str(checkpoint.get("checkpoint"))
    label = f"{case_id}/{name}"
    errors: list[str] = []
    checkpoint_probe = checkpoint.get("termination_probe")
    _error(errors, label, isinstance(checkpoint_probe, dict), "checkpoint lacks termination probe")
    if isinstance(checkpoint_probe, dict):
        _error(errors, label, checkpoint_probe == record_probe, "checkpoint termination probe differs from record")
        errors.extend(_validate_termination_probe(checkpoint_probe, case=case, formal=formal))
    checkpoint_scenario = checkpoint.get("scenario_execution")
    _error(errors, label, isinstance(checkpoint_scenario, dict), "checkpoint lacks scenario execution")
    if isinstance(checkpoint_scenario, dict):
        _error(errors, label, checkpoint_scenario == record_scenario_execution, "checkpoint scenario execution differs from record")
        errors.extend(_validate_scenario_execution(checkpoint_scenario, case=case, formal=formal))
    canonical = checkpoint.get("canonical", {})
    boundaries_meta = canonical.get("boundaries", {})
    zero_semantics = boundaries_meta.get("zero_retain_semantics")
    if case.retain_fragment_count == 0:
        _error(errors, label, zero_semantics == case.scenario, "zero-retain semantics differs from scenario")
        expected_boundaries = 0 if case.scenario == "speculation_full_invalidation" else 1
    else:
        _error(errors, label, zero_semantics is None, "nonzero-retain checkpoint has zero-retain semantics")
        expected_boundaries = None
    path = checkpoint.get("path", {})
    canonical_ids = canonical.get("token_ids")
    path_ids = path.get("token_ids")
    _error(errors, label, isinstance(canonical_ids, list), "missing canonical token IDs")
    _error(errors, label, isinstance(path_ids, list), "missing path token IDs")
    if isinstance(canonical_ids, list) and isinstance(path_ids, list):
        mismatch = first_mismatch(path_ids, canonical_ids)
        _error(errors, label, mismatch is None, f"token mismatch at {mismatch}")
        _error(
            errors,
            label,
            canonical.get("token_hash") == token_ids_hash(canonical_ids),
            "canonical token hash mismatch",
        )
        _error(
            errors,
            label,
            path.get("token_hash") == token_ids_hash(path_ids),
            "path token hash mismatch",
        )
        _error(errors, label, checkpoint.get("first_token_mismatch") == mismatch, "stored mismatch index differs")
        _error(errors, label, bool(checkpoint.get("token_ids_exact")) == (mismatch is None), "token_ids_exact differs")
    state = checkpoint.get("state", {})
    path_state = state.get("path", {})
    lengths = [
        path_state.get("seq_length"),
        path_state.get("mask_length"),
        path_state.get("kv_length"),
        path_state.get("ledger_length"),
    ]
    _error(errors, label, all(isinstance(value, int) for value in lengths), "malformed state lengths")
    if all(isinstance(value, int) for value in lengths):
        _error(errors, label, len(set(lengths)) == 1, f"state lengths differ: {lengths}")
        if isinstance(path_ids, list):
            _error(errors, label, lengths[0] == len(path_ids), "state length differs from path tokens")
    _error(errors, label, bool(path_state.get("lengths_exact")), "runtime lengths_exact is false")
    _error(errors, label, bool(path_state.get("assistant_content_span_exact")), "assistant ledger span differs")
    _error(errors, label, bool(path_state.get("role_phase_exact")), "role phase differs")
    unique = checkpoint.get("unique_eot", {})
    _error(errors, label, bool(unique.get("ok")), "EOT positions differ")
    _error(
        errors,
        label,
        unique.get("path_positions") == unique.get("canonical_positions"),
        "stored EOT positions are not exact",
    )
    boundaries = int(unique.get("assistant_boundaries", -1))
    if expected_boundaries is not None:
        _error(
            errors,
            label,
            boundaries == expected_boundaries,
            f"zero-retain assistant boundary count should be {expected_boundaries}",
        )
    positions = unique.get("path_positions", [])
    _error(errors, label, isinstance(positions, list), "EOT positions must be a list")
    if isinstance(positions, list):
        _error(errors, label, len(positions) == boundaries, f"expected {boundaries} structural EOT, found {len(positions)}")
    next_token = checkpoint.get("next_token", {})
    _error(errors, label, bool(next_token.get("top1_exact")), "next-token top1 differs")
    _error(
        errors,
        label,
        next_token.get("path_top1") == next_token.get("canonical_top1"),
        "stored top1 equality differs",
    )
    _error(
        errors,
        label,
        int(next_token.get("top5_overlap", -1)) >= TOP_K_MIN_OVERLAP,
        f"top5 overlap below {TOP_K_MIN_OVERLAP}/5",
    )
    logit = checkpoint.get("logit_diff_float32", {})
    try:
        max_abs = float(logit["max_abs"])
        mean_abs = float(logit["mean_abs"])
        rms = float(logit["rms"])
        _error(errors, label, 0.0 <= max_abs <= BF16_MAX_ABS_THRESHOLD, f"max_abs={max_abs}")
        _error(errors, label, 0.0 <= mean_abs <= BF16_MEAN_ABS_THRESHOLD, f"mean_abs={mean_abs}")
        _error(errors, label, 0.0 <= rms <= max_abs + 1e-12, f"invalid RMS={rms}")
    except (KeyError, TypeError, ValueError):
        errors.append(f"{label}: malformed float32 logit differences")
    continuation = checkpoint.get("continuation", {})
    _error(
        errors,
        label,
        continuation.get("continuation_source") == "actual_crop_cache",
        "path continuation was not generated from the actual crop/recovery cache",
    )
    _error(
        errors,
        label,
        continuation.get("canonical_source") == "clean_prefill_cache",
        "canonical continuation source differs",
    )
    _error(
        errors,
        label,
        continuation.get("checkpoint_state_captured_before_mutation") is True,
        "checkpoint state/logits were not captured before continuation mutation",
    )
    left = continuation.get("path_token_ids")
    right = continuation.get("canonical_token_ids")
    _error(errors, label, continuation.get("tokens") == CONTINUATION_TOKENS, "continuation length protocol differs")
    _error(errors, label, isinstance(left, list) and len(left) == CONTINUATION_TOKENS, "path continuation malformed")
    _error(errors, label, isinstance(right, list) and len(right) == CONTINUATION_TOKENS, "canonical continuation malformed")
    if isinstance(left, list) and isinstance(right, list):
        divergence = first_mismatch(left, right)
        _error(errors, label, divergence is None, f"continuation diverges at {divergence}")
        _error(errors, label, continuation.get("first_divergence") == divergence, "stored continuation divergence differs")
        _error(errors, label, continuation.get("path_hash") == token_ids_hash(left), "path continuation hash mismatch")
        _error(errors, label, continuation.get("canonical_hash") == token_ids_hash(right), "canonical continuation hash mismatch")
    stored_checkpoint_errors = checkpoint.get("errors")
    _error(errors, label, isinstance(stored_checkpoint_errors, list), "stored checkpoint errors malformed")
    if isinstance(stored_checkpoint_errors, list):
        _error(errors, label, not stored_checkpoint_errors, "runtime checkpoint reported errors")
    _error(errors, label, bool(checkpoint.get("passed")) == (not errors), "stored checkpoint verdict differs")
    if errors:
        failure_indexes.append({"case_id": case_id, "checkpoint": name, "errors": list(errors)})
    return errors


def validate_campaign(
    campaign_dir: Path, *, formal: bool = True, expected_cases: int | None = None
) -> dict[str, Any]:
    manifest_path = campaign_dir / "campaign_manifest.json"
    cases_path = campaign_dir / "cases.json"
    records_path = campaign_dir / "records.jsonl"
    manifest = load_campaign_manifest(manifest_path)
    cases = load_cases(cases_path, formal=formal)
    if expected_cases is not None:
        if formal:
            raise ValidationError("Formal C2 does not permit an expected-case override")
        if expected_cases <= 0 or expected_cases > len(cases):
            raise ValidationError("expected_cases is out of range")
        cases = cases[:expected_cases]
    records = load_jsonl(records_path)
    errors: list[str] = []
    failures: list[dict[str, Any]] = []
    if formal:
        _error(errors, "campaign", bool(manifest.get("config", {}).get("formal")), "manifest is non-formal")
        _error(errors, "campaign", len(cases) == FORMAL_CASE_COUNT, f"formal case count must be {FORMAL_CASE_COUNT}")
        _error(errors, "campaign", len(records) == len(cases), f"expected {len(cases)} records, found {len(records)}")
        _error(errors, "campaign", manifest.get("config", {}).get("session_count") == 1, "formal session count is not one")
        _error(errors, "campaign", manifest.get("config", {}).get("statistical_repeats") == 0, "statistical repeats are not zero")
        _error(errors, "campaign", manifest.get("git", {}).get("dirty") is False, "formal manifest records dirty tree")
        config = manifest.get("config", {})
        runtime_metadata = config.get("runtime_metadata", {})
        _error(errors, "campaign", runtime_metadata.get("resolved_dtype") == EXPECTED_DTYPE, "formal dtype is not BF16")
        _error(errors, "campaign", runtime_metadata.get("model_type") == EXPECTED_MODEL_TYPE, "formal model_type is not qwen2")
        _error(errors, "campaign", EXPECTED_MODEL_ARCHITECTURE in runtime_metadata.get("architectures", []), "formal model architecture differs")
        _error(errors, "campaign", runtime_metadata.get("accepted_model_artifact_hash") == EXPECTED_MODEL_ARTIFACT_HASH, "formal model snapshot differs from accepted D-017 artifact")
        _error(errors, "campaign", config.get("strict_offline") is True, "strict_offline missing")
        _error(errors, "campaign", config.get("hf_hub_offline") == "1", "HF_HUB_OFFLINE was not 1")
        _error(errors, "campaign", config.get("transformers_offline") == "1", "TRANSFORMERS_OFFLINE was not 1")
        _error(errors, "campaign", config.get("hf_token_empty") is True, "HF_TOKEN was non-empty")
        _error(errors, "campaign", config.get("hugging_face_hub_token_empty") is True, "HUGGING_FACE_HUB_TOKEN was non-empty")
    _error(errors, "campaign", sha256_file(cases_path) == manifest.get("input", {}).get("sha256"), "cases SHA differs from manifest")
    _error(errors, "campaign", code_identity() == manifest.get("config", {}).get("code_identity"), "C2 code identity changed")
    expected_ids = [case.id for case in cases]
    record_ids = [str(record.get("case_id")) for record in records]
    _error(errors, "grid", record_ids == expected_ids, "records are missing, duplicated, or out of frozen case order")
    manifest_sha = sha256_file(manifest_path)
    process_ids: set[str] = set()
    for index, record in enumerate(records):
        case_id = str(record.get("case_id"))
        label = f"record[{index}]/{case_id}"
        _error(errors, label, record.get("session_id") == "s01", "session_id differs")
        _error(errors, label, record.get("session_index") == 0, "session_index differs")
        _error(errors, label, record.get("statistical_repeat") is None, "statistical repeat present")
        _error(errors, label, record.get("campaign_identity_hash") == manifest.get("identity_hash"), "identity differs")
        _error(errors, label, record.get("campaign_manifest_sha256") == manifest_sha, "manifest SHA differs")
        _error(errors, label, record.get("cases_sha256") == sha256_file(cases_path), "cases SHA differs")
        process_ids.add(str(record.get("process_start_id")))
        case_errors: list[str] = []
        case = cases[index] if index < len(cases) else None
        probe = record.get("termination_probe")
        scenario_execution = record.get("scenario_execution")
        _error(case_errors, label, case is not None, "record has no corresponding frozen case")
        _error(case_errors, label, isinstance(probe, dict), "record lacks termination probe")
        _error(case_errors, label, isinstance(scenario_execution, dict), "record lacks scenario execution")
        if case is not None and isinstance(probe, dict):
            case_errors.extend(_validate_termination_probe(probe, case=case, formal=formal))
            _error(case_errors, label, record.get("termination") == case.termination, "record termination label differs")
            _error(case_errors, label, record.get("controlled_fixture") is case.controlled_fixture, "controlled fixture flag differs")
        if case is not None and isinstance(scenario_execution, dict):
            case_errors.extend(
                _validate_scenario_execution(
                    scenario_execution, case=case, formal=formal
                )
            )
        checkpoints = record.get("checkpoints")
        if not isinstance(checkpoints, list) or not checkpoints:
            case_errors.append(f"{label}: no checkpoints")
        else:
            expected_checkpoints = list(cases[index].checkpoints) if index < len(cases) else []
            actual_checkpoints = [str(item.get("checkpoint")) for item in checkpoints]
            _error(case_errors, label, actual_checkpoints == expected_checkpoints, "checkpoint grid differs")
            if (
                case is not None
                and isinstance(probe, dict)
                and isinstance(scenario_execution, dict)
            ):
                for checkpoint in checkpoints:
                    case_errors.extend(
                        _validate_checkpoint(
                            checkpoint,
                            case=case,
                            record_probe=probe,
                            record_scenario_execution=scenario_execution,
                            formal=formal,
                            failure_indexes=failures,
                        )
                    )
        stored_case_errors = record.get("errors")
        _error(case_errors, label, isinstance(stored_case_errors, list), "stored case errors malformed")
        if isinstance(stored_case_errors, list):
            _error(case_errors, label, not stored_case_errors, "runtime case reported errors")
        _error(case_errors, label, bool(record.get("passed")) == (not case_errors), "stored case verdict differs")
        if case_errors:
            failures.append({"case_id": case_id, "checkpoint": None, "errors": case_errors})
        errors.extend(case_errors)
    _error(errors, "campaign", bool(process_ids), "records lack process identities")
    summary_path = campaign_dir / "summary.json"
    if formal:
        _error(errors, "summary", summary_path.is_file(), "formal campaign requires summary.json")
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary_probes = summary.get("termination_probes", {})
        failed_ids = sorted(
            str(record.get("case_id")) for record in records if not record.get("passed")
        )
        expected_checkpoint_count = sum(
            len(record.get("checkpoints", [])) for record in records
        )
        _error(errors, "summary", summary.get("cases") == len(cases), "case count differs")
        _error(errors, "summary", summary.get("checkpoint_count") == expected_checkpoint_count, "checkpoint count differs")
        _error(errors, "summary", summary.get("failed_cases") == failed_ids, "failed case list differs")
        _error(errors, "summary", summary.get("status") == ("FAIL" if failed_ids else "PASS"), "status differs")
        _error(errors, "summary", summary.get("sessions") == 1, "session count differs")
        _error(errors, "summary", summary.get("statistical_repeats") == 0, "repeat count differs")
        _error(errors, "summary", summary.get("logical_session_id") == "s01", "logical session differs")
        _error(errors, "summary", summary.get("process_identities") == sorted(process_ids), "process identities differ")
        _error(errors, "summary", summary.get("resume_process_count") == len(process_ids), "resume count differs")
        _error(errors, "summary", summary.get("campaign_identity_hash") == manifest.get("identity_hash"), "campaign identity differs")
        _error(errors, "summary", summary.get("campaign_manifest_sha256") == manifest_sha, "manifest SHA differs")
        _error(errors, "summary", summary_probes.get("required") == len(cases), "termination required count differs")
        _error(errors, "summary", summary_probes.get("observed") == len(records), "termination observed count differs")
        _error(errors, "summary", summary_probes.get("runner_qualified") == len(cases), "runner qualification count differs")
        _error(errors, "summary", bool(summary.get("acceptance_eligible")) == (not failed_ids), "acceptance verdict differs")
    result = {
        "schema_version": 1,
        "experiment": "c2_equivalence",
        "validated_at_utc": utc_now(),
        "campaign_dir": str(campaign_dir.resolve()),
        "formal": formal,
        "ok": not errors,
        "acceptance_eligible": not errors,
        "errors": errors,
        "failed_indexes": failures,
        "grid": {
            "sessions": 1,
            "statistical_repeats": 0,
            "cases_expected": len(cases),
            "cases_observed": len(records),
            "process_identities": sorted(process_ids),
            "resume_process_count": len(process_ids),
        },
        "termination_probes": {
            "required": len(cases),
            "observed": sum(isinstance(record.get("termination_probe"), dict) for record in records),
            "qualified": sum(
                isinstance(record.get("termination_probe"), dict)
                and not _validate_termination_probe(
                    record["termination_probe"], case=cases[index], formal=formal
                )
                for index, record in enumerate(records)
                if index < len(cases)
            ),
        },
        "thresholds": {
            "max_abs_logit_diff": BF16_MAX_ABS_THRESHOLD,
            "mean_abs_logit_diff": BF16_MEAN_ABS_THRESHOLD,
            "top5_min_overlap": TOP_K_MIN_OVERLAP,
            "continuation_tokens": CONTINUATION_TOKENS,
        },
        "provenance": {
            "campaign_manifest": {"path": str(manifest_path.resolve()), "sha256": manifest_sha},
            "cases": {"path": str(cases_path.resolve()), "sha256": sha256_file(cases_path)},
            "records": {"path": str(records_path.resolve()), "sha256": sha256_file(records_path)},
        },
    }
    return result


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate C2 raw records independently and fail closed.")
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--non-formal", action="store_true", help="Validate a pilot/smoke campaign")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    result = validate_campaign(args.campaign_dir, formal=not args.non_formal)
    output = args.out or args.campaign_dir / "validation.json"
    if output.exists():
        raise FileExistsError(f"Validation output already exists: {output}")
    atomic_write_json(output, result)
    if not result["ok"]:
        raise SystemExit("C2 validation failed; artifacts retained: " + "; ".join(result["errors"][:10]))
    print(output)


if __name__ == "__main__":
    main()
