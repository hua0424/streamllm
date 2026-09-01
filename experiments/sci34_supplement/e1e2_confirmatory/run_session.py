"""Run one independently launched confirmatory E1/E2 session.

A formal invocation owns exactly one process identity and writes append-only,
fsync-backed JSONL.  Resume is allowed only while the recorded process start
identity still matches, preventing records from process restarts being pooled as
one wall-clock session.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

from experiments.sci34_supplement.common import (
    append_jsonl,
    atomic_write_json,
    build_manifest,
    collect_environment,
    enforce_offline_mode,
    load_jsonl,
    require_clean_tree,
    seed_everything,
    sha256_file,
    stable_seed,
    utc_now,
)
from experiments.sci34_supplement.e1e2_confirmatory.protocol import (
    CONDITIONS,
    EXPERIMENT,
    FORMAL_DIALOGUE_COUNT,
    ProtocolConfig,
    WARMUP_PATHS,
    balanced_condition_order,
    campaign_identity_payload,
    load_input_rows,
    threshold_for_condition,
)
from experiments.sci34_supplement.e1e2_confirmatory.campaign import load_campaign_manifest
from experiments.sci34_supplement.e1e2_confirmatory.runtime import SessionBackend, make_backend
from experiments.sci34_supplement.e1e2_confirmatory.trigger_cache import ReplayTrigger


LOGGER = logging.getLogger(__name__)
PROCESS_ID_ENV = "E1E2_PROCESS_START_ID"


def _process_start_identity() -> str:
    return os.environ.setdefault(
        PROCESS_ID_ENV,
        f"pid-{os.getpid()}-{time.time_ns()}-{uuid.uuid4().hex}",
    )


def _rows_as_dicts(rows) -> list[dict[str, Any]]:
    return [row.to_dict() for row in rows]


def _load_confidences(trigger: ReplayTrigger, row: Mapping[str, Any]) -> list[float]:
    accumulated = ""
    values: list[float] = []
    for prefix_index, segment in enumerate(row["segments"], start=1):
        accumulated += str(segment)
        values.append(
            trigger.confidence_for(str(row["id"]), prefix_index, accumulated)
        )
    return values


def _validate_existing_jsonl(
    path: Path,
    *,
    expected_identity: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], set[tuple[str, str]]]:
    records = load_jsonl(path)
    keys: set[tuple[str, str]] = set()
    for line_no, record in enumerate(records, start=1):
        key = (str(record.get("dialogue_id")), str(record.get("condition")))
        if key in keys:
            raise ValueError(f"Duplicate formal record key at {path}:{line_no}: {key}")
        keys.add(key)
        for field, expected in expected_identity.items():
            if record.get(field) != expected:
                raise ValueError(
                    f"Resume refused: {path}:{line_no} {field} mismatch "
                    f"({record.get(field)!r} != {expected!r})"
                )
    return records, keys


def _normalize_measurement(
    measurement: Mapping[str, Any],
    *,
    condition: str,
    threshold: float | None,
) -> dict[str, Any]:
    value = dict(measurement)
    required = (
        "last_segment_arrival_ns",
        "endpoint_accept_ns",
        "first_token_ready_ns",
        "first_deliverable_token_ns",
        "oracle_preaccept_processing_ns",
        "arrival_to_first_token_ready_ns",
        "ttft_eff_ns",
        "consumer_delivery_ns",
        "generation_done_ns",
        "survived",
        "ready_tokens",
        "n_speculations",
        "n_invalidated",
        "wasted_tokens",
        "final_tokens",
        "eos",
        "output_text",
    )
    missing = [field for field in required if field not in value]
    if missing:
        raise ValueError(f"Backend omitted fields for {condition}: {missing}")
    for field in (
        "last_segment_arrival_ns",
        "endpoint_accept_ns",
        "first_token_ready_ns",
        "first_deliverable_token_ns",
        "oracle_preaccept_processing_ns",
        "arrival_to_first_token_ready_ns",
        "ttft_eff_ns",
        "consumer_delivery_ns",
        "generation_done_ns",
        "ready_tokens",
        "n_speculations",
        "n_invalidated",
        "wasted_tokens",
        "final_tokens",
    ):
        value[field] = int(value[field])
    value["speculative_tokens"] = int(
        value.get("speculative_tokens", value["wasted_tokens"] + value["ready_tokens"])
    )
    value["waste_denominator_tokens"] = value["wasted_tokens"] + value["final_tokens"]
    arrival = value["last_segment_arrival_ns"]
    endpoint = value["endpoint_accept_ns"]
    ready_ns = value["first_token_ready_ns"]
    first = value["first_deliverable_token_ns"]
    consumer = value["consumer_delivery_ns"]
    survived = bool(value["survived"])
    ready = value["ready_tokens"]
    if min(arrival, endpoint, ready_ns, first, consumer, value["generation_done_ns"]) < 0:
        raise ValueError("Monotonic timestamps must be non-negative")
    if endpoint < arrival:
        raise ValueError("endpoint acceptance precedes final segment arrival")
    if ready_ns < arrival:
        raise ValueError("first token readiness precedes final segment arrival")
    if first < endpoint:
        raise ValueError("first_deliverable_token_ns precedes endpoint acceptance")
    if value["oracle_preaccept_processing_ns"] != endpoint - arrival:
        raise ValueError("oracle preaccept processing duration mismatch")
    if value["arrival_to_first_token_ready_ns"] != ready_ns - arrival:
        raise ValueError("arrival-to-ready duration mismatch")
    expected_ttft = 0 if survived and ready > 0 else first - endpoint
    if value["ttft_eff_ns"] != expected_ttft:
        raise ValueError(
            f"Corrected TTFT mismatch for {condition}: {value['ttft_eff_ns']} != {expected_ttft}"
        )
    if consumer < first:
        raise ValueError("consumer delivery precedes token deliverability")
    if value["generation_done_ns"] < consumer:
        raise ValueError("generation completion precedes consumer delivery")
    if condition.endswith("never_speculate") and value["n_speculations"] != 0:
        raise ValueError("never_speculate launched a speculation")
    if condition.startswith("system_a") and threshold is not None:
        raise ValueError("System A must not carry a threshold")
    value["consumer_delivery_latency_ns"] = consumer - endpoint
    value["consumer_delivery_from_arrival_ns"] = consumer - arrival
    value["condition_threshold"] = threshold
    value["survived"] = survived
    return value


def _session_manifest(
    *,
    args: argparse.Namespace,
    protocol: ProtocolConfig,
    identity: Mapping[str, Any],
    rows: list[dict[str, Any]],
    backend: SessionBackend,
    process_start_id: str,
    trigger_identity_hash: str,
) -> dict[str, Any]:
    config = {
        "session_id": args.session_id,
        "session_index": args.session_index,
        "session_seed": stable_seed(args.seed, args.session_id),
        "formal": args.formal,
        "runtime": args.runtime,
        "model": args.model,
        "device": args.device,
        "protocol": protocol.to_dict(),
        "campaign_identity": dict(identity),
        "process_start_id": process_start_id,
        "process_resume_contract": "same process_start_id only",
        "model_identity": dict(backend.identity),
        "runtime_metadata": dict(getattr(backend, "runtime_metadata", {})),
        "strict_offline": bool(args.formal),
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
        "transformers_offline": os.environ.get("TRANSFORMERS_OFFLINE"),
        "hf_token_empty": not bool(os.environ.get("HF_TOKEN")),
        "hugging_face_hub_token_empty": not bool(
            os.environ.get("HUGGING_FACE_HUB_TOKEN")
        ),
    }
    return build_manifest(
        experiment=EXPERIMENT,
        run_id=args.session_id,
        config=config,
        input_path=args.input,
        sample_ids=[row["id"] for row in rows],
        extra={
            "session_id": args.session_id,
            "session_index": args.session_index,
            "pid": os.getpid(),
            "process_start_id": process_start_id,
            "python_executable": sys.executable,
            "argv": sys.argv,
            "campaign_manifest_path": (
                str(args.campaign_manifest.resolve()) if args.campaign_manifest else None
            ),
            "campaign_manifest_sha256": (
                sha256_file(args.campaign_manifest) if args.campaign_manifest else None
            ),
            "trigger_cache_sha256": sha256_file(args.trigger_cache),
            "trigger_identity_hash": trigger_identity_hash,
        },
    )


def prepare_session_directory(
    session_dir: Path,
    *,
    manifest: Mapping[str, Any],
    resume: bool,
    process_start_id: str,
) -> None:
    manifest_path = session_dir / "manifest.json"
    if session_dir.exists() and not resume:
        raise FileExistsError(f"Session already exists: {session_dir}")
    session_dir.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("config_hash") != manifest.get("config_hash"):
            raise ValueError("Resume refused: session config/model/input/trigger identity changed")
        old_process = existing.get("extra", {}).get("process_start_id")
        if old_process != process_start_id:
            raise ValueError(
                "Resume refused after process restart; use a new session ID and rerun from scratch"
            )
    else:
        atomic_write_json(manifest_path, dict(manifest))


def run_session(args: argparse.Namespace, backend: SessionBackend | None = None) -> Path:
    protocol = ProtocolConfig(
        max_new_tokens=args.max_new_tokens,
        spec_chunk=args.spec_chunk,
        warmup_repeats=args.warmup_repeats,
        condition_order_seed=args.order_seed,
    )
    protocol.validate()
    limit = getattr(args, "limit", None)
    if args.formal and limit is not None:
        raise ValueError("--limit is pilot-only and forbidden in formal sessions")
    rows = _rows_as_dicts(load_input_rows(args.input, formal=args.formal))
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        rows = rows[:limit]
    session_seed = stable_seed(args.seed, args.session_id)
    seed_everything(session_seed)
    trigger = ReplayTrigger(
        args.trigger_cache,
        expected_input_sha256=sha256_file(args.input),
        expected_identity_hash=args.trigger_identity_hash,
    )
    if backend is None:
        backend = make_backend(
            args.runtime,
            model_name=args.model,
            device=args.device,
            seed=session_seed,
            max_new_tokens=protocol.max_new_tokens,
            spec_chunk=protocol.spec_chunk,
        )
    campaign_manifest = (
        load_campaign_manifest(args.campaign_manifest) if args.campaign_manifest else None
    )
    if args.formal and campaign_manifest is None:
        raise ValueError("Formal sessions require --campaign-manifest")
    if campaign_manifest and not args.formal:
        raise ValueError("Campaign manifests are reserved for formal sessions")
    trigger_model_identity = trigger.model_identity
    identity = campaign_identity_payload(
        protocol=protocol,
        input_path=args.input,
        trigger_cache_path=args.trigger_cache,
        model_identity=backend.identity,
        runtime_kind=args.runtime,
        device=args.device,
        trigger_model_identity=trigger_model_identity,
    )
    if campaign_manifest:
        if campaign_manifest.get("run_id") != args.campaign_id:
            raise ValueError("Campaign manifest run_id differs from --campaign-id")
        manifest_identity = campaign_manifest.get("campaign_identity", {})
        if manifest_identity != identity:
            raise ValueError("Campaign manifest payload does not exactly match this invocation")
        manifest_runtime = campaign_manifest.get("config", {}).get("runtime_metadata", {})
        actual_runtime = dict(getattr(backend, "runtime_metadata", {}))
        if manifest_runtime.get("resolved_dtype") not in (None, actual_runtime.get("resolved_dtype")):
            raise ValueError("Campaign manifest resolved dtype differs from runtime")
        if manifest_runtime.get("attention_backend") not in (
            None,
            actual_runtime.get("attention_backend"),
        ):
            raise ValueError("Campaign manifest attention backend differs from runtime")
        manifest_input = campaign_manifest.get("input", {})
        if manifest_input.get("sha256") != sha256_file(args.input):
            raise ValueError("Campaign manifest input hash mismatch")
        if campaign_manifest.get("extra", {}).get("trigger_cache_sha256") != trigger.cache_sha256:
            raise ValueError("Campaign manifest trigger cache hash mismatch")
    process_start_id = _process_start_identity()
    session_dir = args.results_root / args.campaign_id / "sessions" / args.session_id
    manifest = _session_manifest(
        args=args,
        protocol=protocol,
        identity=identity,
        rows=rows,
        backend=backend,
        process_start_id=process_start_id,
        trigger_identity_hash=trigger.identity_hash,
    )
    manifest["extra"]["campaign_manifest_sha256"] = (
        sha256_file(args.campaign_manifest) if args.campaign_manifest else None
    )
    prepare_session_directory(
        session_dir,
        manifest=manifest,
        resume=args.resume,
        process_start_id=process_start_id,
    )
    records_path = session_dir / "records.jsonl"
    record_identity = {
        "campaign_id": args.campaign_id,
        "session_id": args.session_id,
        "session_index": args.session_index,
        "process_start_id": process_start_id,
        "trigger_cache_sha256": trigger.cache_sha256,
        "trigger_identity_hash": trigger.identity_hash,
        "input_sha256": sha256_file(args.input),
        "campaign_identity_hash": identity["identity_hash"],
        "campaign_manifest_sha256": (
            sha256_file(args.campaign_manifest) if args.campaign_manifest else None
        ),
    }
    _, completed = _validate_existing_jsonl(
        records_path, expected_identity=record_identity
    )

    warmup_log = session_dir / "warmups.jsonl"
    if not completed and not warmup_log.exists():
        for path_kind in WARMUP_PATHS:
            for repeat in range(protocol.warmup_repeats):
                started = time.perf_counter_ns()
                backend.warmup(path_kind)
                append_jsonl(
                    warmup_log,
                    {
                        "path_kind": path_kind,
                        "repeat": repeat,
                        "started_ns": started,
                        "done_ns": time.perf_counter_ns(),
                    },
                )

    for dialogue_index, row in enumerate(rows):
        confidences = _load_confidences(trigger, row)
        order = balanced_condition_order(
            session_index=args.session_index,
            dialogue_index=dialogue_index,
            seed=protocol.condition_order_seed,
        )
        for condition_ordinal, condition in enumerate(order):
            key = (row["id"], condition)
            if key in completed:
                continue
            threshold = threshold_for_condition(condition)
            measurement = backend.run_condition(
                row,
                condition=condition,
                threshold=threshold,
                confidences=confidences,
                session_id=args.session_id,
            )
            normalized = _normalize_measurement(
                measurement, condition=condition, threshold=threshold
            )
            append_jsonl(
                records_path,
                {
                    "schema_version": 1,
                    "experiment": EXPERIMENT,
                    "campaign_id": args.campaign_id,
                    "session_id": args.session_id,
                    "session_index": args.session_index,
                    "pid": os.getpid(),
                    "process_start_id": process_start_id,
                    "dialogue_id": row["id"],
                    "dialogue_index": dialogue_index,
                    "condition": condition,
                    "condition_ordinal": condition_ordinal,
                    "condition_order": order,
                    "condition_order_seed": protocol.condition_order_seed,
                    "recorded_at_ns": time.perf_counter_ns(),
                    "trigger_confidences": confidences,
                    "trigger_cache_sha256": trigger.cache_sha256,
                    "trigger_identity_hash": trigger.identity_hash,
                    "input_sha256": sha256_file(args.input),
                    "campaign_identity_hash": identity["identity_hash"],
                    "campaign_manifest_sha256": record_identity[
                        "campaign_manifest_sha256"
                    ],
                    "resolved_dtype": getattr(backend, "runtime_metadata", {}).get(
                        "resolved_dtype"
                    ),
                    "attention_backend": getattr(backend, "runtime_metadata", {}).get(
                        "attention_backend"
                    ),
                    **normalized,
                },
            )
            completed.add(key)
        atomic_write_json(
            session_dir / "progress.json",
            {
                "updated_at_utc": utc_now(),
                "completed_records": len(completed),
                "expected_records": len(rows) * len(CONDITIONS),
                "completed_dialogues": dialogue_index + 1,
                "expected_dialogues": len(rows),
            },
        )
    expected = len(rows) * len(CONDITIONS)
    if len(completed) != expected:
        raise AssertionError(f"Session grid incomplete: {len(completed)} != {expected}")
    atomic_write_json(
        session_dir / "summary.json",
        {
            "completed_at_utc": utc_now(),
            "records": expected,
            "dialogues": len(rows),
            "conditions": list(CONDITIONS),
            "warmup_records": len(load_jsonl(warmup_log)),
            "campaign_identity_hash": identity["identity_hash"],
            "environment": collect_environment(),
        },
    )
    return session_dir


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--session-index", required=True, type=int)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--trigger-cache", required=True, type=Path)
    parser.add_argument("--trigger-identity-hash")
    parser.add_argument("--campaign-manifest", type=Path)
    parser.add_argument("--results-root", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "e1e2_confirmatory")
    parser.add_argument("--runtime", choices=("fake", "transformers"), default="transformers")
    parser.add_argument("--model")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--order-seed", type=int, default=20260901)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--spec-chunk", type=int, default=12)
    parser.add_argument("--warmup-repeats", type=int, default=3)
    parser.add_argument("--limit", type=int, help="Non-formal pilot subset only")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    if args.formal:
        if args.allow_dirty:
            raise SystemExit("Formal sessions forbid --allow-dirty")
        if not args.campaign_manifest:
            raise SystemExit("Formal sessions require --campaign-manifest")
        if args.limit is not None:
            raise SystemExit("Formal sessions forbid --limit")
        if args.runtime != "transformers":
            raise SystemExit("Formal sessions require --runtime transformers")
        if not args.model or not Path(args.model).exists():
            raise SystemExit("Formal sessions require an explicit local --model path")
        require_clean_tree(allow_dirty=False)
        enforce_offline_mode()
        if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
            raise SystemExit("Formal sessions require Hugging Face tokens to be empty")
    output = run_session(args)
    print(output)


if __name__ == "__main__":
    main()
