"""Run one deterministic C2 session with case-atomic persistence and resume."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

from experiments.sci34_supplement.common import (
    atomic_write_json,
    enforce_offline_mode,
    load_jsonl,
    require_clean_tree,
    seed_everything,
    sha256_file,
    utc_now,
)
from experiments.sci34_supplement.c2_equivalence.campaign import (
    code_identity,
    load_campaign_manifest,
)
from experiments.sci34_supplement.c2_equivalence.canonical_chat import token_ids_hash
from experiments.sci34_supplement.c2_equivalence.protocol import (
    EOS_AT_CAP_MAX_NEW_TOKENS,
    EXPERIMENT,
    FORMAL_CASE_COUNT,
    MAX_TOKENS_PROBE_BUDGET,
    NATURAL_EOS_MAX_NEW_TOKENS,
    ProtocolConfig,
    load_cases,
)
from experiments.sci34_supplement.c2_equivalence.runtime import EquivalenceBackend, make_backend


PROCESS_ID_ENV = "C2_PROCESS_START_ID"


def _process_identity() -> str:
    return os.environ.setdefault(
        PROCESS_ID_ENV, f"pid-{os.getpid()}-{time.time_ns()}-{uuid.uuid4().hex}"
    )


def _atomic_append_record(path: Path, record: Mapping[str, Any]) -> None:
    """Rewrite complete JSONL atomically so a case appears whole or not at all."""
    rows = load_jsonl(path)
    rows.append(dict(record))
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    from experiments.sci34_supplement.common import atomic_write_text

    atomic_write_text(path, text)


def _load_existing(
    records_path: Path,
    *,
    campaign_identity_hash: str,
    campaign_manifest_sha256: str,
) -> tuple[list[dict[str, Any]], set[str], dict[str, int]]:
    records = load_jsonl(records_path)
    completed: set[str] = set()
    attempts: dict[str, int] = {}
    for line_no, record in enumerate(records, start=1):
        case_id = str(record.get("case_id", ""))
        if not case_id or case_id in completed:
            raise ValueError(f"Duplicate or empty case at records line {line_no}: {case_id!r}")
        if record.get("campaign_identity_hash") != campaign_identity_hash:
            raise ValueError(f"Resume identity mismatch at records line {line_no}")
        if record.get("campaign_manifest_sha256") != campaign_manifest_sha256:
            raise ValueError(f"Resume manifest hash mismatch at records line {line_no}")
        completed.add(case_id)
        attempts[case_id] = int(record.get("attempt", 0))
    return records, completed, attempts


def _write_failure_sidecars(
    campaign_dir: Path,
    case_id: str,
    attempt: int,
    measurement: dict[str, Any],
) -> list[str]:
    saved: list[str] = []
    for checkpoint in measurement.get("checkpoints", []):
        arrays = checkpoint.pop("_failure_logits", None)
        if arrays is None:
            continue
        try:
            import numpy as np
        except ImportError as error:
            raise RuntimeError("Failure logits require numpy for compressed sidecars") from error
        path = campaign_dir / "failures" / f"{case_id}.attempt{attempt}.{checkpoint['checkpoint']}.npz"
        np.savez_compressed(path, **arrays)
        saved.append(str(path.relative_to(campaign_dir)))
    return saved


def _assert_probe_qualified(probe: Mapping[str, Any], *, termination: str) -> None:
    """Runner-side hard gate; independent validator repeats these checks."""
    cap = {
        "natural_eos": NATURAL_EOS_MAX_NEW_TOKENS,
        "eos_at_cap": EOS_AT_CAP_MAX_NEW_TOKENS,
        "max_tokens": MAX_TOKENS_PROBE_BUDGET,
    }[termination]
    ids = probe.get("content_token_ids")
    prefix = probe.get("prefix_seq_length")
    post = probe.get("post_seq_length")
    common = (
        isinstance(ids, list)
        and all(isinstance(token, int) for token in ids)
        and isinstance(prefix, int)
        and isinstance(post, int)
        and probe.get("content_token_count") == len(ids)
        and probe.get("content_token_hash") == token_ids_hash(ids)
        and probe.get("cap") == cap
        and probe.get("eot_token_id") not in ids
        and probe.get("eot_in_kv") is False
        and probe.get("eot_in_full_ledger") is False
        and probe.get("eot_in_content_ledger") is False
        and post == prefix + len(ids)
        and not probe.get("errors")
        and probe.get("passed") is True
    )
    if not common:
        raise RuntimeError(f"termination probe common invariants failed for {termination}")
    if termination == "natural_eos":
        qualified = (
            probe.get("mode") == "real_greedy"
            and probe.get("controlled") is False
            and probe.get("observed_end_reason") == "EOS"
            and isinstance(probe.get("eos_step"), int)
            and 1 <= probe["eos_step"] <= cap
            and probe.get("role_phase") == "ASSISTANT_EOT_PENDING"
        )
    elif termination == "eos_at_cap":
        fixture = probe.get("fixture_token_ids")
        qualified = (
            probe.get("mode") == "controlled_logits_fixture"
            and probe.get("controlled") is True
            and probe.get("observed_end_reason") == "EOS"
            and probe.get("eos_step") == cap
            and probe.get("eos_at_cap") is True
            and isinstance(fixture, list)
            and fixture[:-1] == ids
            and fixture[-1:] == [probe.get("eot_token_id")]
            and bool(probe.get("fixture_description"))
            and probe.get("role_phase") == "ASSISTANT_EOT_PENDING"
        )
    else:
        qualified = (
            probe.get("mode") == "real_greedy"
            and probe.get("controlled") is False
            and probe.get("observed_end_reason") == "MAX_TOKENS"
            and probe.get("eos_step") is None
            and probe.get("eos_at_cap") is False
            and len(ids) == cap
            and probe.get("role_phase") == "ASSISTANT_OPEN"
        )
    if not qualified:
        raise RuntimeError(f"termination probe label qualification failed for {termination}")


def run_campaign(
    *,
    campaign_dir: Path,
    runtime_kind: str,
    model_path: str | None,
    device: str,
    seed: int,
    resume: bool,
    limit: int | None,
    backend: EquivalenceBackend | None = None,
) -> Path:
    manifest_path = campaign_dir / "campaign_manifest.json"
    manifest = load_campaign_manifest(manifest_path)
    formal = bool(manifest.get("config", {}).get("formal"))
    if formal and limit is not None:
        raise ValueError("Formal C2 forbids --limit")
    if formal and runtime_kind != "transformers":
        raise ValueError("Formal C2 requires transformers runtime")
    if formal and backend is not None and backend.__class__.__name__ != "TransformersBackend":
        raise ValueError("Formal C2 forbids an injected non-Transformers backend")
    cases_path = campaign_dir / "cases.json"
    cases = load_cases(cases_path, formal=formal)
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        cases = cases[:limit]
    expected_runtime = manifest.get("config", {}).get("runtime")
    if expected_runtime != runtime_kind:
        raise ValueError(f"Runtime differs from manifest: {runtime_kind} != {expected_runtime}")
    if sha256_file(cases_path) != manifest.get("input", {}).get("sha256"):
        raise ValueError("Campaign-local cases.json hash differs from manifest")
    if code_identity() != manifest.get("config", {}).get("code_identity"):
        raise ValueError("C2 implementation changed after campaign manifest creation")
    ProtocolConfig().validate()
    seed_everything(seed)
    backend = backend or make_backend(
        runtime_kind, model_path=model_path, device=device, seed=seed
    )
    if dict(backend.identity) != manifest.get("config", {}).get("model_identity"):
        raise ValueError("Runtime model identity differs from campaign manifest")
    if dict(backend.runtime_metadata) != manifest.get("config", {}).get("runtime_metadata"):
        raise ValueError("Runtime metadata differs from campaign manifest")

    records_path = campaign_dir / "records.jsonl"
    if records_path.exists() and not resume:
        raise FileExistsError("records.jsonl exists; pass --resume for case-atomic continuation")
    manifest_sha = sha256_file(manifest_path)
    records, completed, prior_attempts = _load_existing(
        records_path,
        campaign_identity_hash=manifest["identity_hash"],
        campaign_manifest_sha256=manifest_sha,
    )
    attempts_path = campaign_dir / "attempts.jsonl"
    for attempt_record in load_jsonl(attempts_path):
        case_id = str(attempt_record.get("case_id", ""))
        if case_id:
            prior_attempts[case_id] = max(
                prior_attempts.get(case_id, 0), int(attempt_record.get("attempt", 0))
            )
    process_id = _process_identity()
    for case in cases:
        if case.id in completed:
            continue
        attempt = prior_attempts.get(case.id, 0) + 1
        started = time.perf_counter_ns()
        try:
            measurement = backend.run_case(case)
            probe = measurement.get("termination_probe")
            if not isinstance(probe, dict):
                raise RuntimeError(f"{case.id}: backend omitted termination_probe")
            runner_probe_errors: list[str] = []
            if probe.get("declared") != case.termination:
                runner_probe_errors.append("termination probe label differs")
            try:
                _assert_probe_qualified(probe, termination=case.termination)
            except RuntimeError as error:
                runner_probe_errors.append(str(error))
            if formal and probe.get("execution") != "transformers_model":
                runner_probe_errors.append("formal termination probe is not Transformers execution")
            checkpoint_probes = [
                checkpoint.get("termination_probe")
                for checkpoint in measurement.get("checkpoints", [])
            ]
            if not checkpoint_probes or any(item != probe for item in checkpoint_probes):
                runner_probe_errors.append("record/checkpoint termination probe differs")
            scenario_execution = measurement.get("scenario_execution")
            if not isinstance(scenario_execution, dict):
                runner_probe_errors.append("backend omitted scenario_execution")
            else:
                checkpoint_scenarios = [
                    checkpoint.get("scenario_execution")
                    for checkpoint in measurement.get("checkpoints", [])
                ]
                if not checkpoint_scenarios or any(
                    item != scenario_execution for item in checkpoint_scenarios
                ):
                    runner_probe_errors.append(
                        "record/checkpoint scenario execution differs"
                    )
                if not scenario_execution.get("passed"):
                    runner_probe_errors.extend(
                        f"scenario_execution: {message}"
                        for message in scenario_execution.get("errors", [])
                    )
                if formal and scenario_execution.get("execution") != "transformers_model":
                    runner_probe_errors.append(
                        "formal scenario execution is not Transformers execution"
                    )
            if runner_probe_errors:
                measurement.setdefault("errors", []).extend(
                    f"termination_probe: {message}" for message in runner_probe_errors
                )
                measurement["passed"] = False
            sidecars = _write_failure_sidecars(campaign_dir, case.id, attempt, measurement)
            record = {
                "schema_version": 1,
                "experiment": EXPERIMENT,
                "run_id": manifest["run_id"],
                "session_id": "s01",
                "session_index": 0,
                "statistical_repeat": None,
                "case_id": case.id,
                "case_index": [item.id for item in cases].index(case.id),
                "attempt": attempt,
                "process_start_id": process_id,
                "pid": os.getpid(),
                "python_executable": sys.executable,
                "started_ns": started,
                "completed_ns": time.perf_counter_ns(),
                "campaign_identity_hash": manifest["identity_hash"],
                "campaign_manifest_sha256": manifest_sha,
                "cases_sha256": sha256_file(cases_path),
                "failure_sidecars": sidecars,
                **measurement,
            }
            _atomic_append_record(records_path, record)
            records.append(record)
            completed.add(case.id)
            prior_attempts[case.id] = attempt
            _atomic_append_record(
                attempts_path,
                {
                    "case_id": case.id,
                    "attempt": attempt,
                    "process_start_id": process_id,
                    "pid": os.getpid(),
                    "started_ns": started,
                    "completed_ns": record["completed_ns"],
                    "status": "completed",
                    "passed": bool(record.get("passed")),
                },
            )
        except Exception as error:
            _atomic_append_record(
                attempts_path,
                {
                    "case_id": case.id,
                    "attempt": attempt,
                    "process_start_id": process_id,
                    "pid": os.getpid(),
                    "started_ns": started,
                    "failed_ns": time.perf_counter_ns(),
                    "status": "exception",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            atomic_write_json(
                campaign_dir / "progress.json",
                {
                    "updated_at_utc": utc_now(),
                    "status": "failed-preserved",
                    "completed_cases": len(completed),
                    "expected_cases": len(cases),
                    "failed_cases": sorted(
                        case_id for case_id in completed
                        if not next(row for row in records if row["case_id"] == case_id).get("passed")
                    ),
                    "exception_case": case.id,
                    "exception": str(error),
                },
            )
            raise
        failed = sorted(row["case_id"] for row in records if not row.get("passed"))
        atomic_write_json(
            campaign_dir / "progress.json",
            {
                "updated_at_utc": utc_now(),
                "status": "running" if len(completed) < len(cases) else ("failed-preserved" if failed else "complete"),
                "completed_cases": len(completed),
                "expected_cases": len(cases),
                "failed_cases": failed,
                "last_case": case.id,
                "process_start_id": process_id,
            },
        )

    if len(completed) != len(cases):
        raise AssertionError(f"Incomplete C2 grid: {len(completed)} != {len(cases)}")
    failed = sorted(record["case_id"] for record in records if not record.get("passed"))
    qualified_probes = sum(
        bool(record.get("termination_probe", {}).get("passed"))
        and record.get("passed")
        for record in records
    )
    process_ids = sorted({str(record.get("process_start_id")) for record in records})
    atomic_write_json(
        campaign_dir / "summary.json",
        {
            "completed_at_utc": utc_now(),
            "status": "FAIL" if failed else "PASS",
            "acceptance_eligible": not failed,
            "sessions": 1,
            "statistical_repeats": 0,
            "cases": len(cases),
            "expected_formal_cases": FORMAL_CASE_COUNT,
            "failed_cases": failed,
            "checkpoint_count": sum(len(row.get("checkpoints", [])) for row in records),
            "termination_probes": {
                "required": len(cases),
                "observed": sum(isinstance(row.get("termination_probe"), dict) for row in records),
                "runner_qualified": qualified_probes,
            },
            "logical_session_id": "s01",
            "process_identities": process_ids,
            "resume_process_count": len(process_ids),
            "campaign_identity_hash": manifest["identity_hash"],
            "campaign_manifest_sha256": manifest_sha,
        },
    )
    if failed:
        raise RuntimeError(
            f"C2 correctness failed for {len(failed)} cases; artifacts retained: {failed}"
        )
    return campaign_dir


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the one-session C2 equivalence grid with case-atomic resume."
    )
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--runtime", choices=("transformers", "fake"), default="transformers")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, help="Pilot-only case prefix")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    manifest = load_campaign_manifest(args.campaign_dir / "campaign_manifest.json")
    formal = bool(manifest.get("config", {}).get("formal"))
    if formal:
        if args.runtime != "transformers":
            raise SystemExit("Formal C2 requires --runtime transformers")
        if args.model is None or not args.model.exists() or not args.model.is_dir():
            raise SystemExit("Formal C2 requires an explicit local --model directory")
        if args.limit is not None:
            raise SystemExit("Formal C2 forbids --limit")
        require_clean_tree(allow_dirty=False)
        enforce_offline_mode()
        if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
            raise SystemExit("Formal C2 requires empty Hugging Face tokens")
    output = run_campaign(
        campaign_dir=args.campaign_dir,
        runtime_kind=args.runtime,
        model_path=str(args.model) if args.model else None,
        device=args.device,
        seed=args.seed,
        resume=args.resume,
        limit=args.limit,
    )
    print(output)


if __name__ == "__main__":
    main()
