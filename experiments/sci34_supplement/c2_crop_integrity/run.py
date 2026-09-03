"""Run C2 v3 with case-atomic JSONL persistence and resume."""

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
    atomic_write_text,
    enforce_offline_mode,
    load_jsonl,
    require_clean_tree,
    seed_everything,
    sha256_file,
    utc_now,
)
from experiments.sci34_supplement.c2_crop_integrity.campaign import code_identity, load_campaign_manifest
from experiments.sci34_supplement.c2_crop_integrity.integrity import record_content_hash
from experiments.sci34_supplement.c2_crop_integrity.protocol import (
    EXPERIMENT,
    FORMAL_CASE_COUNT,
    FORMAL_CROP_EVENT_COUNT,
    PRIOR_V2_RUN_ID,
    PROTOCOL_VERSION,
    ProtocolConfig,
    load_cases,
)
from experiments.sci34_supplement.c2_crop_integrity.runtime import CropIntegrityBackend, make_backend


PROCESS_ID_ENV = "C2_V3_PROCESS_START_ID"


def _process_identity() -> str:
    return os.environ.setdefault(PROCESS_ID_ENV, f"pid-{os.getpid()}-{time.time_ns()}-{uuid.uuid4().hex}")


def _atomic_append(path: Path, record: Mapping[str, Any]) -> None:
    rows = load_jsonl(path)
    rows.append(dict(record))
    atomic_write_text(
        path,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
    )


def _load_existing(path: Path, *, identity: str, manifest_sha: str) -> tuple[list[dict[str, Any]], set[str], dict[str, int]]:
    records = load_jsonl(path)
    completed: set[str] = set()
    attempts: dict[str, int] = {}
    for line_no, record in enumerate(records, start=1):
        case_id = record.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in completed:
            raise ValueError(f"Duplicate/empty case at records line {line_no}")
        if record.get("campaign_identity_hash") != identity or record.get("campaign_manifest_sha256") != manifest_sha:
            raise ValueError(f"Resume identity mismatch at records line {line_no}")
        if record.get("record_content_hash") != record_content_hash(record):
            raise ValueError(f"Resume record hash mismatch at records line {line_no}")
        completed.add(case_id)
        attempts[case_id] = int(record.get("attempt", 0))
    return records, completed, attempts


def run_campaign(
    *,
    campaign_dir: Path,
    runtime_kind: str,
    model_path: str | None,
    device: str,
    seed: int,
    resume: bool,
    limit: int | None,
    backend: CropIntegrityBackend | None = None,
) -> Path:
    manifest_path = campaign_dir / "campaign_manifest.json"
    manifest = load_campaign_manifest(manifest_path)
    formal = bool(manifest.get("config", {}).get("formal"))
    if formal and (runtime_kind != "transformers" or limit is not None):
        raise ValueError("Formal C2 v3 requires Transformers and forbids --limit")
    if formal and backend is not None and backend.__class__.__name__ != "TransformersBackend":
        raise ValueError("Formal C2 v3 forbids injected fake backends")
    cases_path = campaign_dir / "cases.json"
    cases = load_cases(cases_path, formal=formal)
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        cases = cases[:limit]
    if manifest.get("config", {}).get("runtime") != runtime_kind:
        raise ValueError("Runtime differs from immutable campaign manifest")
    if sha256_file(cases_path) != manifest.get("input", {}).get("sha256"):
        raise ValueError("Campaign-local cases hash differs from manifest")
    if code_identity() != manifest.get("config", {}).get("code_identity"):
        raise ValueError("C2 v3 implementation changed after manifest creation")
    ProtocolConfig().validate()
    seed_everything(seed)
    backend = backend or make_backend(runtime_kind, model_path=model_path, device=device, seed=seed)
    if dict(backend.identity) != manifest.get("config", {}).get("model_identity"):
        raise ValueError("Runtime model identity differs from manifest")
    if dict(backend.runtime_metadata) != manifest.get("config", {}).get("runtime_metadata"):
        raise ValueError("Runtime metadata differs from manifest")
    negative = backend.negative_control()
    if negative != manifest.get("config", {}).get("negative_control") or negative.get("detected") is not True:
        raise RuntimeError("Formal deterministic negative control failed or changed")

    records_path = campaign_dir / "records.jsonl"
    if records_path.exists() and not resume:
        raise FileExistsError("records.jsonl exists; use --resume")
    manifest_sha = sha256_file(manifest_path)
    records, completed, attempts = _load_existing(
        records_path, identity=manifest["identity_hash"], manifest_sha=manifest_sha
    )
    for attempt_row in load_jsonl(campaign_dir / "attempts.jsonl"):
        case_id = attempt_row.get("case_id")
        if isinstance(case_id, str):
            attempts[case_id] = max(attempts.get(case_id, 0), int(attempt_row.get("attempt", 0)))
    process_id = _process_identity()
    all_case_ids = [case.id for case in cases]
    for case in cases:
        if case.id in completed:
            continue
        attempt = attempts.get(case.id, 0) + 1
        started = time.perf_counter_ns()
        try:
            measurement = backend.run_case(case)
            expected_events = 1 + (case.second_crop_fraction is not None)
            if len(measurement.get("crop_events", [])) != expected_events:
                raise RuntimeError(f"{case.id}: backend omitted crop events")
            record = {
                "schema_version": 1,
                "protocol_version": PROTOCOL_VERSION,
                "experiment": EXPERIMENT,
                "run_id": manifest["run_id"],
                "session_id": "s01",
                "session_index": 0,
                "statistical_repeat": None,
                "case_id": case.id,
                "case_index": all_case_ids.index(case.id),
                "attempt": attempt,
                "process_start_id": process_id,
                "pid": os.getpid(),
                "python_executable": sys.executable,
                "started_ns": started,
                "completed_ns": time.perf_counter_ns(),
                "campaign_identity_hash": manifest["identity_hash"],
                "campaign_manifest_sha256": manifest_sha,
                "cases_sha256": sha256_file(cases_path),
                "prior_v2_run_id": PRIOR_V2_RUN_ID,
                "formal_negative_control": negative,
                **measurement,
            }
            record["record_content_hash"] = record_content_hash(record)
            _atomic_append(records_path, record)
            records.append(record)
            completed.add(case.id)
            attempts[case.id] = attempt
            _atomic_append(
                campaign_dir / "attempts.jsonl",
                {
                    "case_id": case.id,
                    "attempt": attempt,
                    "process_start_id": process_id,
                    "started_ns": started,
                    "completed_ns": record["completed_ns"],
                    "status": "completed",
                    "passed": bool(record.get("passed")),
                    "record_content_hash": record["record_content_hash"],
                },
            )
        except Exception as error:
            _atomic_append(
                campaign_dir / "attempts.jsonl",
                {
                    "case_id": case.id,
                    "attempt": attempt,
                    "process_start_id": process_id,
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
                    "completed_crop_events": sum(len(row.get("crop_events", [])) for row in records),
                    "exception_case": case.id,
                    "exception": str(error),
                },
            )
            raise
        failed = [row["case_id"] for row in records if not row.get("passed")]
        event_count = sum(len(row.get("crop_events", [])) for row in records)
        atomic_write_json(
            campaign_dir / "progress.json",
            {
                "updated_at_utc": utc_now(),
                "status": "running" if len(completed) < len(cases) else ("failed-preserved" if failed else "complete"),
                "completed_cases": len(completed),
                "expected_cases": len(cases),
                "completed_crop_events": event_count,
                "expected_crop_events": sum(1 + (case.second_crop_fraction is not None) for case in cases),
                "failed_cases": failed,
                "last_case": case.id,
                "process_start_id": process_id,
            },
        )

    failed = [row["case_id"] for row in records if not row.get("passed")]
    event_count = sum(len(row.get("crop_events", [])) for row in records)
    expected_event_count = sum(1 + (case.second_crop_fraction is not None) for case in cases)
    complete = len(completed) == len(cases) and event_count == expected_event_count
    formal_grid = not formal or (len(records) == FORMAL_CASE_COUNT and event_count == FORMAL_CROP_EVENT_COUNT)
    eligible = complete and formal_grid and not failed
    atomic_write_json(
        campaign_dir / "summary.json",
        {
            "completed_at_utc": utc_now(),
            "protocol_version": PROTOCOL_VERSION,
            "status": "PASS" if eligible else "FAIL",
            "acceptance_eligible": eligible,
            "sessions": 1,
            "statistical_repeats": 0,
            "cases": len(records),
            "expected_cases": len(cases),
            "crop_events": event_count,
            "expected_crop_events": expected_event_count,
            "expected_formal_cases": FORMAL_CASE_COUNT,
            "expected_formal_crop_events": FORMAL_CROP_EVENT_COUNT,
            "failed_cases": failed,
            "termination_probe_reruns": sum(bool(row.get("termination_probe_rerun")) for row in records),
            "prior_v2_run_id": PRIOR_V2_RUN_ID,
            "negative_control": negative,
            "process_identities": sorted({str(row.get("process_start_id")) for row in records}),
            "campaign_identity_hash": manifest["identity_hash"],
            "campaign_manifest_sha256": manifest_sha,
        },
    )
    if not eligible:
        raise RuntimeError("C2 v3 crop-integrity campaign failed; artifacts retained")
    return campaign_dir


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run C2 v3 crop-integrity cases with atomic resume")
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--runtime", choices=("transformers", "fake"), default="transformers")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    manifest = load_campaign_manifest(args.campaign_dir / "campaign_manifest.json")
    formal = bool(manifest.get("config", {}).get("formal"))
    if formal:
        if args.runtime != "transformers" or args.model is None or not args.model.is_dir() or args.limit is not None:
            raise SystemExit("Formal C2 v3 requires local Transformers model and no --limit")
        require_clean_tree(allow_dirty=False)
        enforce_offline_mode()
        if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
            raise SystemExit("Formal C2 v3 requires empty Hugging Face tokens")
    run_campaign(
        campaign_dir=args.campaign_dir,
        runtime_kind=args.runtime,
        model_path=str(args.model) if args.model else None,
        device=args.device,
        seed=args.seed,
        resume=args.resume,
        limit=args.limit,
    )
    print(args.campaign_dir)


if __name__ == "__main__":
    main()
