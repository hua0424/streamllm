"""Create and verify the immutable C2 campaign manifest."""

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
from experiments.sci34_supplement.c2_equivalence.protocol import (
    EXPERIMENT,
    ProtocolConfig,
    load_cases,
    protocol_identity,
)
from experiments.sci34_supplement.c2_equivalence.runtime import make_backend


CAMPAIGN_MANIFEST_SCHEMA_VERSION = 1
PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[2]
DEFAULT_CASES = PACKAGE_ROOT / "cases.json"
CODE_FILES = (
    "experiments/sci34_supplement/c2_equivalence/protocol.py",
    "experiments/sci34_supplement/c2_equivalence/canonical_chat.py",
    "experiments/sci34_supplement/c2_equivalence/runtime.py",
    "experiments/sci34_supplement/c2_equivalence/campaign.py",
    "experiments/sci34_supplement/c2_equivalence/run.py",
    "experiments/sci34_supplement/c2_equivalence/validate.py",
    "experiments/sci34_supplement/c2_equivalence/analyze.py",
    "experiments/sci34_supplement/c2_equivalence/seal.py",
    "experiments/sci34_supplement/c2_equivalence/smoke.py",
    "experiments/sci34_supplement/common.py",
    "experiments/sci34_supplement/e1e2_confirmatory/strong_identity.py",
    "src/llm/stream_llm_inference.py",
    "src/dialogue/orchestrator.py",
    "src/dialogue/timeline.py",
)


def code_identity() -> dict[str, Any]:
    files = []
    for name in CODE_FILES:
        path = REPO_ROOT / name
        if not path.exists():
            raise FileNotFoundError(f"C2 code identity is incomplete: {path}")
        files.append({"path": name, "sha256": sha256_file(path), "size": path.stat().st_size})
    payload = {"files": files}
    payload["identity_hash"] = config_hash(payload)
    return payload


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
    protocol = ProtocolConfig()
    protocol.validate()
    cases = load_cases(cases_path, formal=formal)
    if backend is None:
        backend = make_backend(
            runtime_kind,
            model_path=str(model_path) if model_path is not None else None,
            device=device,
            seed=seed,
        )
    config = {
        "run_id": run_id,
        "formal": formal,
        "session_count": 1,
        "statistical_repeats": 0,
        "runtime": runtime_kind,
        "device": device,
        "seed": seed,
        "protocol": protocol.to_dict(),
        "protocol_identity": protocol_identity(cases_path),
        "code_identity": code_identity(),
        "model_identity": dict(backend.identity),
        "runtime_metadata": dict(backend.runtime_metadata),
        "strict_offline": bool(formal),
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
        "transformers_offline": os.environ.get("TRANSFORMERS_OFFLINE"),
        "hf_token_empty": not bool(os.environ.get("HF_TOKEN")),
        "hugging_face_hub_token_empty": not bool(os.environ.get("HUGGING_FACE_HUB_TOKEN")),
    }
    identity_payload = {
        "schema_version": CAMPAIGN_MANIFEST_SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "runtime": runtime_kind,
        "device": device,
        "protocol_identity": config["protocol_identity"],
        "code_identity": config["code_identity"],
        "model_identity": config["model_identity"],
        "runtime_metadata": config["runtime_metadata"],
    }
    identity_payload["identity_hash"] = config_hash(identity_payload)
    config["campaign_identity"] = identity_payload
    manifest = build_manifest(
        experiment=EXPERIMENT,
        run_id=run_id,
        config=config,
        input_path=cases_path,
        sample_ids=[case.id for case in cases],
        extra={
            "manifest_kind": "immutable_c2_campaign",
            "cases_count": len(cases),
            "pilot_or_formal": "formal" if formal else "pilot",
        },
    )
    manifest["campaign_manifest_schema_version"] = CAMPAIGN_MANIFEST_SCHEMA_VERSION
    manifest["campaign_identity"] = identity_payload
    manifest["identity_hash"] = identity_payload["identity_hash"]
    manifest["manifest_content_hash"] = config_hash(
        {key: value for key, value in manifest.items() if key != "manifest_content_hash"}
    )
    return manifest


def validate_campaign_manifest_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("campaign_manifest_schema_version") != CAMPAIGN_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported C2 campaign manifest schema")
    stored = payload.get("manifest_content_hash")
    expected = config_hash(
        {key: value for key, value in payload.items() if key != "manifest_content_hash"}
    )
    if stored != expected:
        raise ValueError(f"Campaign manifest content hash mismatch: {stored} != {expected}")
    identity = payload.get("campaign_identity", {})
    if payload.get("identity_hash") != identity.get("identity_hash"):
        raise ValueError("Campaign manifest identity hash is inconsistent")


def load_campaign_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_campaign_manifest_payload(payload)
    payload["_artifact_path"] = str(path.resolve())
    payload["_artifact_sha256"] = sha256_file(path)
    return payload


def prepare_campaign_directory(
    *,
    output_dir: Path,
    manifest: Mapping[str, Any],
    cases_path: Path,
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Campaign directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    try:
        atomic_write_json(output_dir / "campaign_manifest.json", dict(manifest))
        shutil.copyfile(cases_path, output_dir / "cases.json")
        atomic_write_json(
            output_dir / "progress.json",
            {
                "status": "prepared",
                "completed_cases": 0,
                "expected_cases": manifest.get("extra", {}).get("cases_count"),
                "failed_cases": [],
            },
        )
        (output_dir / "failures").mkdir()
        (output_dir / "logs").mkdir()
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create an immutable one-session C2 crop/re-prefill equivalence campaign."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--runtime", choices=("transformers", "fake"), default="transformers")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--non-formal", action="store_true", help="Pilot/smoke only")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    formal = not args.non_formal
    if formal:
        if args.runtime != "transformers":
            raise SystemExit("Formal C2 requires --runtime transformers")
        if args.model is None or not args.model.exists() or not args.model.is_dir():
            raise SystemExit("Formal C2 requires an explicit local --model directory")
        require_clean_tree(allow_dirty=False)
        enforce_offline_mode()
        if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
            raise SystemExit("Formal C2 requires empty Hugging Face tokens")
    manifest = build_campaign_manifest(
        run_id=args.run_id,
        cases_path=args.cases,
        runtime_kind=args.runtime,
        model_path=args.model,
        device=args.device,
        seed=args.seed,
        formal=formal,
    )
    prepare_campaign_directory(
        output_dir=args.output_dir,
        manifest=manifest,
        cases_path=args.cases,
    )
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
