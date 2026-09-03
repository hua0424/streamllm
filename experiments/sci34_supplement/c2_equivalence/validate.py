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
    CONTINUATION_TOKENS,
    EOS_AT_CAP_MAX_NEW_TOKENS,
    EXPECTED_DTYPE,
    EXPECTED_MODEL_ARCHITECTURE,
    EXPECTED_MODEL_ARTIFACT_HASH,
    EXPECTED_MODEL_TYPE,
    FORMAL_CASE_COUNT,
    LOGIT_MAX_ABS_BACKSTOP,
    LOGIT_MEAN_ABS_BACKSTOP,
    MAX_TOKENS_PROBE_BUDGET,
    NATURAL_EOS_MAX_NEW_TOKENS,
    NATURAL_EOS_MIN_GENUINE,
    NEAR_TIE_ABS_MARGIN_LIMIT,
    NEAR_TIE_MARGIN_FLOOR,
    NOISE_CONTROL_MAX_ABS_FLOOR,
    NOISE_CONTROL_MEAN_ABS_FLOOR,
    NOISE_RATIO_LIMIT,
    PROTOCOL_VERSION,
    TERMINATION_PROBE_SCHEMA_VERSION,
    TOP_K_MIN_OVERLAP,
    CaseSpec,
    load_cases,
    near_tie_margin_limit,
    noise_limits,
)

# NPZ arrays are float32; the runtime computed stats on float32 tensors, the
# validator recomputes in float64. Elementwise float32 rounding differences are
# bounded by ~1e-6 at these magnitudes, so 1e-4 is a strict but safe tolerance.
STATS_TOLERANCE = 1e-4


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
        genuine = probe.get("genuine_eos")
        requalified = probe.get("requalified")
        if genuine is True and requalified is False:
            _error(errors, label, observed == "EOS", f"observed end reason is {observed!r}")
            _error(errors, label, isinstance(eos_step, int) and 1 <= eos_step <= expected_cap, "EOS not reached within cap")
            _error(errors, label, probe.get("role_phase") == "ASSISTANT_EOT_PENDING", "EOS phase differs")
        elif requalified is True and genuine is False:
            # v2 deterministic requalification of a greedy run-on.
            _error(errors, label, observed == "MAX_TOKENS", f"observed end reason is {observed!r}")
            _error(errors, label, eos_step is None and probe.get("eos_at_cap") is False, "requalified probe recorded EOS")
            _error(errors, label, isinstance(ids, list) and len(ids) == expected_cap, "requalified content count differs from cap")
            _error(errors, label, probe.get("role_phase") == "ASSISTANT_OPEN", "requalified phase differs")
        else:
            errors.append(f"{label}: natural probe must be exclusively genuine or requalified")
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


def _load_checkpoint_arrays(
    campaign_dir: Path, case_id: str, attempt: int, name: str
) -> dict[str, Any]:
    import numpy as np

    path = campaign_dir / "checkpoints" / f"{case_id}.attempt{attempt}.{name}.npz"
    if not path.is_file():
        raise ValidationError(f"missing checkpoint logits sidecar: {path.name}")
    with np.load(path, allow_pickle=False) as data:
        arrays: dict[str, Any] = {}
        for key in ("path", "canonical", "control"):
            if key not in data.files:
                raise ValidationError(f"{path.name} lacks the '{key}' logits array")
            arrays[key] = np.asarray(data[key], dtype=np.float64).reshape(-1)
    if len({array.size for array in arrays.values()}) != 1:
        raise ValidationError(f"{path.name} logits arrays disagree on vocabulary size")
    return arrays


def _top_k_facts(values: Any) -> dict[str, Any]:
    """Stable ordering plus exact-tie sets for independent top-1/top-5 checks."""
    import numpy as np

    order = np.argsort(-values, kind="stable")
    top1 = int(order[0])
    top2 = int(order[1])
    top1_ties = [int(index) for index in np.nonzero(values == values[order[0]])[0]]
    top5 = [int(index) for index in order[:5]]
    boundary_tie = bool(values[order[4]] == values[order[5]])
    return {
        "top1": top1,
        "top2": top2,
        "margin": float(values[order[0]] - values[order[1]]),
        "top1_ties": top1_ties,
        "top5": top5,
        "boundary_tie": boundary_tie,
    }


def _diff_stats(left: Any, right: Any) -> dict[str, float]:
    import numpy as np

    difference = np.abs(left - right)
    return {
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "rms": float(np.sqrt(np.mean(difference * difference))),
    }


def _close(left: float, right: float) -> bool:
    return abs(float(left) - float(right)) <= STATS_TOLERANCE


def _validate_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    case: CaseSpec,
    record_probe: Mapping[str, Any],
    record_scenario_execution: Mapping[str, Any],
    formal: bool,
    failure_indexes: list[dict[str, Any]],
    campaign_dir: Path,
    attempt: int,
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
    continuation = checkpoint.get("continuation", {})

    # v2: reload the frozen logits sidecar and recompute every numeric fact from it.
    arrays_error: str | None = None
    arrays: dict[str, Any] | None = None
    try:
        arrays = _load_checkpoint_arrays(campaign_dir, case_id, attempt, name)
    except ValidationError as error:
        arrays_error = str(error)
        errors.append(f"{label}: {arrays_error}")
    recomputed_stats: dict[str, float] = {}
    control_stats: dict[str, float] = {}
    if arrays is not None:
        recomputed_stats = _diff_stats(arrays["path"], arrays["canonical"])
        control_stats = _diff_stats(arrays["control"], arrays["canonical"])
        logit = checkpoint.get("logit_diff_float32", {})
        noise = checkpoint.get("noise_control", {})
        for stats, recorded, stats_label in (
            (recomputed_stats, logit, "logit_diff_float32"),
            (control_stats, noise, "noise_control"),
        ):
            try:
                for key in ("max_abs", "mean_abs", "rms"):
                    _error(
                        errors,
                        label,
                        _close(recorded[key], stats[key]),
                        f"{stats_label}.{key} {recorded[key]} differs from sidecar recompute {stats[key]}",
                    )
            except (KeyError, TypeError):
                errors.append(f"{label}: malformed {stats_label}")
        if isinstance(canonical_ids, list):
            _error(
                errors,
                label,
                isinstance(noise.get("seam_positions"), list)
                and all(
                    isinstance(item, int) and not isinstance(item, bool) and 0 < item < len(canonical_ids)
                    for item in (noise.get("seam_positions") or [])
                )
                and list(noise.get("seam_positions") or []) == sorted(set(noise.get("seam_positions") or [])),
                "noise-control seams are malformed",
            )
            _error(
                errors,
                label,
                isinstance(noise.get("chunk_count"), int) and noise.get("chunk_count") >= 1,
                "noise-control chunk count is malformed",
            )
        canonical_facts = _top_k_facts(arrays["canonical"])
        path_facts = _top_k_facts(arrays["path"])
        _error(
            errors,
            label,
            next_token.get("canonical_top1") in canonical_facts["top1_ties"],
            "recorded canonical top1 differs from sidecar recompute",
        )
        _error(
            errors,
            label,
            next_token.get("path_top1") in path_facts["top1_ties"],
            "recorded path top1 differs from sidecar recompute",
        )
        _error(
            errors,
            label,
            _close(next_token.get("canonical_top1_top2_margin", float("nan")), canonical_facts["margin"]),
            "recorded canonical margin differs from sidecar recompute",
        )
        _error(
            errors,
            label,
            _close(next_token.get("path_top1_top2_margin", float("nan")), path_facts["margin"]),
            "recorded path margin differs from sidecar recompute",
        )
        overlap = len(set(canonical_facts["top5"]) & set(path_facts["top5"]))
        recorded_overlap = int(next_token.get("top5_overlap", -1))
        effective_overlap = recorded_overlap if (
            canonical_facts["boundary_tie"] or path_facts["boundary_tie"]
        ) else overlap
        if not (canonical_facts["boundary_tie"] or path_facts["boundary_tie"]):
            _error(
                errors,
                label,
                recorded_overlap == overlap,
                f"recorded top5 overlap {recorded_overlap} differs from recompute {overlap}",
            )
        _error(
            errors,
            label,
            effective_overlap >= TOP_K_MIN_OVERLAP,
            f"top5 overlap below {TOP_K_MIN_OVERLAP}/5",
        )
        # v2 gates recomputed from the sidecar alone.
        limit_max, limit_mean = noise_limits(control_stats["max_abs"], control_stats["mean_abs"])
        margin_limit = near_tie_margin_limit(control_stats["max_abs"])
        _error(
            errors,
            label,
            recomputed_stats["max_abs"] <= limit_max,
            f"max_abs={recomputed_stats['max_abs']:.6g} exceeds noise-relative limit {limit_max:.6g}",
        )
        _error(
            errors,
            label,
            recomputed_stats["mean_abs"] <= limit_mean,
            f"mean_abs={recomputed_stats['mean_abs']:.6g} exceeds noise-relative limit {limit_mean:.6g}",
        )
        _error(
            errors,
            label,
            recomputed_stats["max_abs"] <= LOGIT_MAX_ABS_BACKSTOP,
            f"max_abs={recomputed_stats['max_abs']:.6g} exceeds absolute backstop {LOGIT_MAX_ABS_BACKSTOP}",
        )
        _error(
            errors,
            label,
            recomputed_stats["mean_abs"] <= LOGIT_MEAN_ABS_BACKSTOP,
            f"mean_abs={recomputed_stats['mean_abs']:.6g} exceeds absolute backstop {LOGIT_MEAN_ABS_BACKSTOP}",
        )
        top1_exact = next_token.get("path_top1") == next_token.get("canonical_top1")
        near_tie_flip = (not top1_exact) and canonical_facts["margin"] <= margin_limit
        _error(
            errors,
            label,
            top1_exact or near_tie_flip,
            f"next-token top1 differs at canonical margin {canonical_facts['margin']:.6g} above near-tie limit {margin_limit:.6g}",
        )
        _error(
            errors,
            label,
            bool(next_token.get("top1_flip_near_tie")) is near_tie_flip,
            "recorded top1_flip_near_tie differs from recompute",
        )
        gates = checkpoint.get("logit_gates", {})
        _error(
            errors,
            label,
            _close(gates.get("noise_max_limit", float("nan")), limit_max)
            and _close(gates.get("noise_mean_limit", float("nan")), limit_mean)
            and _close(gates.get("near_tie_margin_limit", float("nan")), margin_limit),
            "recorded gate limits differ from recompute",
        )

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
    canonical_steps = continuation.get("canonical_steps")
    path_steps = continuation.get("path_steps")
    _error(
        errors,
        label,
        isinstance(canonical_steps, list) and len(canonical_steps) == CONTINUATION_TOKENS,
        "canonical continuation steps malformed",
    )
    _error(
        errors,
        label,
        isinstance(path_steps, list) and len(path_steps) == CONTINUATION_TOKENS,
        "path continuation steps malformed",
    )
    if isinstance(right, list) and isinstance(canonical_steps, list):
        _error(
            errors,
            label,
            all(
                isinstance(step, dict)
                and step.get("top1") == right[index]
                and isinstance(step.get("margin"), (int, float))
                and float(step.get("margin", -1)) >= 0.0
                for index, step in enumerate(canonical_steps)
            ),
            "canonical continuation steps disagree with recorded tokens",
        )
    if isinstance(left, list) and isinstance(path_steps, list):
        _error(
            errors,
            label,
            all(
                isinstance(step, dict) and step.get("top1") == left[index]
                for index, step in enumerate(path_steps)
            ),
            "path continuation steps disagree with recorded tokens",
        )
    if isinstance(canonical_steps, list) and canonical_steps:
        _error(
            errors,
            label,
            canonical_steps[0].get("top1") == next_token.get("canonical_top1"),
            "continuation first step disagrees with checkpoint next-token top1",
        )
    if isinstance(left, list) and isinstance(right, list):
        divergence = first_mismatch(left, right)
        _error(
            errors,
            label,
            continuation.get("first_divergence") == divergence,
            "stored continuation divergence differs",
        )
        _error(errors, label, continuation.get("path_hash") == token_ids_hash(left), "path continuation hash mismatch")
        _error(errors, label, continuation.get("canonical_hash") == token_ids_hash(right), "canonical continuation hash mismatch")
        if divergence is not None and isinstance(canonical_steps, list) and arrays is not None:
            margin_limit = near_tie_margin_limit(control_stats["max_abs"])
            divergence_margin = float(canonical_steps[divergence].get("margin", float("nan")))
            _error(
                errors,
                label,
                divergence_margin <= margin_limit,
                f"continuation diverges at {divergence} with canonical margin {divergence_margin:.6g} above near-tie limit {margin_limit:.6g}",
            )
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
        _error(
            errors,
            "campaign",
            manifest.get("config", {}).get("protocol", {}).get("protocol_version") == PROTOCOL_VERSION,
            "manifest protocol version differs from v2",
        )
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
        _error(errors, label, record.get("schema_version") == 2, "record schema differs from v2")
        _error(errors, label, record.get("protocol_version") == PROTOCOL_VERSION, "record protocol version differs")
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
            attempt = int(record.get("attempt", 0))
            expected_sidecars = [
                f"checkpoints/{case_id}.attempt{attempt}.{name}.npz"
                for name in actual_checkpoints
            ]
            _error(
                case_errors,
                label,
                record.get("checkpoint_logits") == expected_sidecars,
                "checkpoint logits sidecar list differs",
            )
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
                            campaign_dir=campaign_dir,
                            attempt=attempt,
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
    natural_declared = [
        record for record in records if record.get("termination") == "natural_eos"
    ]
    natural_genuine = sum(
        bool(record.get("termination_probe", {}).get("genuine_eos"))
        for record in natural_declared
    )
    natural_gate_passed = natural_genuine >= NATURAL_EOS_MIN_GENUINE
    if formal:
        _error(
            errors,
            "campaign",
            natural_gate_passed,
            f"natural-EOS genuine coverage {natural_genuine}/{len(natural_declared)} below {NATURAL_EOS_MIN_GENUINE}",
        )
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
        _error(errors, "summary", summary.get("protocol_version") == PROTOCOL_VERSION, "protocol version differs")
        _error(errors, "summary", summary.get("cases") == len(cases), "case count differs")
        _error(errors, "summary", summary.get("checkpoint_count") == expected_checkpoint_count, "checkpoint count differs")
        _error(errors, "summary", summary.get("failed_cases") == failed_ids, "failed case list differs")
        expected_status = "FAIL" if (failed_ids or (formal and not natural_gate_passed)) else "PASS"
        _error(errors, "summary", summary.get("status") == expected_status, "status differs")
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
        expected_gate = {
            "declared": len(natural_declared),
            "genuine": natural_genuine,
            "requalified": len(natural_declared) - natural_genuine,
            "min_genuine": NATURAL_EOS_MIN_GENUINE,
            "applies": formal,
            "passed": natural_gate_passed if formal else True,
        }
        _error(
            errors,
            "summary",
            summary.get("natural_eos_gate") == expected_gate,
            "natural-EOS gate summary differs",
        )
        expected_eligible = not failed_ids and (natural_gate_passed or not formal)
        _error(errors, "summary", bool(summary.get("acceptance_eligible")) == expected_eligible, "acceptance verdict differs")
    result = {
        "schema_version": 2,
        "experiment": "c2_equivalence",
        "protocol_version": PROTOCOL_VERSION,
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
            "natural_eos": {
                "declared": len(natural_declared),
                "genuine": natural_genuine,
                "requalified": len(natural_declared) - natural_genuine,
                "min_genuine": NATURAL_EOS_MIN_GENUINE,
                "gate_passed": natural_gate_passed if formal else True,
            },
        },
        "thresholds": {
            "protocol_version": PROTOCOL_VERSION,
            "noise_control": "canonical_ids_boundary_seam_chunked_prefill",
            "noise_ratio_limit": NOISE_RATIO_LIMIT,
            "noise_control_max_abs_floor": NOISE_CONTROL_MAX_ABS_FLOOR,
            "noise_control_mean_abs_floor": NOISE_CONTROL_MEAN_ABS_FLOOR,
            "max_abs_backstop": LOGIT_MAX_ABS_BACKSTOP,
            "mean_abs_backstop": LOGIT_MEAN_ABS_BACKSTOP,
            "near_tie_margin_floor": NEAR_TIE_MARGIN_FLOOR,
            "near_tie_abs_margin_limit": NEAR_TIE_ABS_MARGIN_LIMIT,
            "top5_min_overlap": TOP_K_MIN_OVERLAP,
            "continuation_tokens": CONTINUATION_TOKENS,
            "natural_eos_min_genuine": NATURAL_EOS_MIN_GENUINE,
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
