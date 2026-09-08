"""Create and verify an immutable C2 v3 crop-integrity campaign."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping

from experiments.sci34_supplement.common import (
    atomic_write_json,
    build_manifest,
    canonical_json,
    config_hash,
    enforce_offline_mode,
    require_clean_tree,
    sha256_file,
)
from experiments.sci34_supplement.c2_crop_integrity.protocol import (
    EXPECTED_CASES_SHA256,
    EXPERIMENT,
    PRIOR_V2_EVIDENCE_ROLE,
    PRIOR_V2_RUN_ID,
    ProtocolConfig,
    load_cases,
    protocol_identity,
)
from experiments.sci34_supplement.c2_crop_integrity.runtime import make_backend


CAMPAIGN_MANIFEST_SCHEMA_VERSION = 1
PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[2]
DEFAULT_CASES = PACKAGE_ROOT / "cases.json"
SOURCE_CASES = PACKAGE_ROOT.parent / "c2_equivalence" / "cases.json"
CODE_FILES = (
    "experiments/sci34_supplement/c2_crop_integrity/protocol.py",
    "experiments/sci34_supplement/c2_crop_integrity/canonical_chat.py",
    "experiments/sci34_supplement/c2_crop_integrity/integrity.py",
    "experiments/sci34_supplement/c2_crop_integrity/runtime.py",
    "experiments/sci34_supplement/c2_crop_integrity/campaign.py",
    "experiments/sci34_supplement/c2_crop_integrity/run.py",
    "experiments/sci34_supplement/c2_crop_integrity/validate.py",
    "experiments/sci34_supplement/c2_crop_integrity/analyze.py",
    "experiments/sci34_supplement/c2_crop_integrity/seal.py",
    "experiments/sci34_supplement/c2_crop_integrity/smoke.py",
    "experiments/sci34_supplement/common.py",
    "experiments/sci34_supplement/e1e2_confirmatory/strong_identity.py",
    "src/llm/stream_llm_inference.py",
)


def code_identity() -> dict[str, Any]:
    files = []
    for name in CODE_FILES:
        path = REPO_ROOT / name
        if not path.is_file():
            raise FileNotFoundError(f"C2 v3 code identity is incomplete: {path}")
        files.append({"path": name, "sha256": sha256_file(path), "size": path.stat().st_size})
    payload = {"files": files}
    payload["identity_hash"] = config_hash(payload)
    return payload


def _assert_frozen_cases(cases_path: Path) -> None:
    source_hash = sha256_file(SOURCE_CASES)
    local_hash = sha256_file(cases_path)
    if source_hash != EXPECTED_CASES_SHA256 or local_hash != EXPECTED_CASES_SHA256:
        raise ValueError(
            "C2 v3 cases must be an exact byte copy of ../c2_equivalence/cases.json "
            f"({source_hash=}, {local_hash=})"
        )


def build_campaign_manifest(
    *,
    run_id: str,
    cases_path: Path,
    runtime_kind: str,
    model_path: Path | None,
    device: str,
    seed: int,
    formal: bool,
    backend=None,
) -> dict[str, Any]:
    ProtocolConfig().validate()
    _assert_frozen_cases(cases_path)
    cases = load_cases(cases_path, formal=formal)
    backend = backend or make_backend(
        runtime_kind,
        model_path=str(model_path) if model_path is not None else None,
        device=device,
        seed=seed,
    )
    negative_control = backend.negative_control()
    if negative_control.get("detected") is not True:
        raise RuntimeError("Campaign negative control did not detect the wrong crop length")
    config = {
        "run_id": run_id,
        "formal": formal,
        "runtime": runtime_kind,
        "device": device,
        "seed": seed,
        "session_count": 1,
        "statistical_repeats": 0,
        "protocol": ProtocolConfig().to_dict(),
        "protocol_identity": protocol_identity(cases_path),
        "code_identity": code_identity(),
        "model_identity": dict(backend.identity),
        "runtime_metadata": dict(backend.runtime_metadata),
        "strict_offline": bool(formal),
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
        "transformers_offline": os.environ.get("TRANSFORMERS_OFFLINE"),
        "hf_token_empty": not bool(os.environ.get("HF_TOKEN")),
        "hugging_face_hub_token_empty": not bool(os.environ.get("HUGGING_FACE_HUB_TOKEN")),
        "negative_control": negative_control,
        "prior_v2_evidence": {
            "run_id": PRIOR_V2_RUN_ID,
            "role": PRIOR_V2_EVIDENCE_ROLE,
            "runtime_dependency": False,
            "artifact_path": None,
        },
        "cases_provenance": {
            "source_relative": "../c2_equivalence/cases.json",
            "source_sha256": sha256_file(SOURCE_CASES),
            "package_copy_sha256": sha256_file(cases_path),
        },
        "case_token_plans": [backend.case_token_plan(case) for case in cases],
    }
    identity = {
        "schema_version": CAMPAIGN_MANIFEST_SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "runtime": runtime_kind,
        "device": device,
        "protocol_identity": config["protocol_identity"],
        "code_identity": config["code_identity"],
        "model_identity": config["model_identity"],
        "runtime_metadata": config["runtime_metadata"],
    }
    identity["identity_hash"] = config_hash(identity)
    config["campaign_identity"] = identity
    manifest = build_manifest(
        experiment=EXPERIMENT,
        run_id=run_id,
        config=config,
        input_path=cases_path,
        sample_ids=[case.id for case in cases],
        extra={
            "manifest_kind": "immutable_c2_v3_crop_integrity",
            "cases_count": len(cases),
            "crop_event_count": sum(1 + (case.second_crop_fraction is not None) for case in cases),
            "pilot_or_formal": "formal" if formal else "pilot",
        },
    )
    manifest["campaign_manifest_schema_version"] = CAMPAIGN_MANIFEST_SCHEMA_VERSION
    manifest["campaign_identity"] = identity
    manifest["identity_hash"] = identity["identity_hash"]
    manifest["manifest_content_hash"] = config_hash(
        {key: value for key, value in manifest.items() if key != "manifest_content_hash"}
    )
    return manifest


def validate_campaign_manifest_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("campaign_manifest_schema_version") != CAMPAIGN_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported C2 v3 campaign manifest schema")
    expected = config_hash(
        {key: value for key, value in payload.items() if key != "manifest_content_hash"}
    )
    if payload.get("manifest_content_hash") != expected:
        raise ValueError("Campaign manifest content hash mismatch")
    identity = payload.get("campaign_identity", {})
    if payload.get("identity_hash") != identity.get("identity_hash"):
        raise ValueError("Campaign identity hash is inconsistent")


def load_campaign_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_campaign_manifest_payload(payload)
    payload["_artifact_path"] = str(path.resolve())
    payload["_artifact_sha256"] = sha256_file(path)
    return payload


def prepare_campaign_directory(*, output_dir: Path, manifest: Mapping[str, Any], cases_path: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Campaign directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    try:
        atomic_write_json(output_dir / "campaign_manifest.json", dict(manifest))
        shutil.copyfile(cases_path, output_dir / "cases.json")
        if sha256_file(output_dir / "cases.json") != EXPECTED_CASES_SHA256:
            raise RuntimeError("Campaign-local cases copy changed")
        atomic_write_json(
            output_dir / "progress.json",
            {
                "status": "prepared",
                "completed_cases": 0,
                "expected_cases": manifest.get("extra", {}).get("cases_count"),
                "completed_crop_events": 0,
                "expected_crop_events": manifest.get("extra", {}).get("crop_event_count"),
                "failed_cases": [],
            },
        )
        (output_dir / "logs").mkdir()
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create an immutable C2 v3 crop-integrity campaign")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--runtime", choices=("transformers", "fake"), default="transformers")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--non-formal", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    formal = not args.non_formal
    if formal:
        if args.runtime != "transformers":
            raise SystemExit("Formal C2 v3 requires --runtime transformers")
        if args.model is None or not args.model.is_dir():
            raise SystemExit("Formal C2 v3 requires an explicit local --model directory")
        require_clean_tree(allow_dirty=False)
        enforce_offline_mode()
        if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
            raise SystemExit("Formal C2 v3 requires empty Hugging Face tokens")
    manifest = build_campaign_manifest(
        run_id=args.run_id,
        cases_path=args.cases,
        runtime_kind=args.runtime,
        model_path=args.model,
        device=args.device,
        seed=args.seed,
        formal=formal,
    )
    prepare_campaign_directory(output_dir=args.output_dir, manifest=manifest, cases_path=args.cases)
    print(
        canonical_json(
            {
                "campaign_dir": str(args.output_dir.resolve()),
                "manifest_sha256": sha256_file(args.output_dir / "campaign_manifest.json"),
                "identity_hash": manifest["identity_hash"],
            }
        )
    )


if __name__ == "__main__":
    main()
