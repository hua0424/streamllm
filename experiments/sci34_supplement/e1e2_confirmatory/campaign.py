"""Prepare the immutable campaign manifest consumed by formal sessions."""

from __future__ import annotations

import argparse
import json
import os
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
from experiments.sci34_supplement.e1e2_confirmatory.protocol import (
    EXPERIMENT,
    FORMAL_DIALOGUE_COUNT,
    FORMAL_SESSION_COUNT,
    ProtocolConfig,
    campaign_identity_payload,
    load_input_rows,
)
from experiments.sci34_supplement.e1e2_confirmatory.strong_identity import strong_model_identity
from experiments.sci34_supplement.e1e2_confirmatory.trigger_cache import ReplayTrigger


CAMPAIGN_MANIFEST_SCHEMA_VERSION = 1


def build_campaign_manifest(
    *,
    campaign_id: str,
    input_path: Path,
    trigger_cache_path: Path,
    main_model_path: Path,
    device: str,
    protocol: ProtocolConfig,
    formal: bool,
) -> dict[str, Any]:
    protocol.validate()
    rows = load_input_rows(input_path, formal=formal)
    trigger = ReplayTrigger(trigger_cache_path, expected_input_sha256=sha256_file(input_path))
    main_identity = strong_model_identity(main_model_path)
    runtime_kind = "transformers" if formal else "fake"
    trigger_weak_identity = trigger.model_identity
    if formal and not (
        trigger_weak_identity.get("content_identity_hash")
        or trigger_weak_identity.get("content_sha256")
    ):
        raise ValueError("Formal TEN cache lacks a strong content identity")
    trigger_strong_identity = dict(trigger_weak_identity)
    campaign_identity = campaign_identity_payload(
        protocol=protocol,
        input_path=input_path,
        trigger_cache_path=trigger_cache_path,
        model_identity=main_identity,
        runtime_kind=runtime_kind,
        device=device,
        trigger_model_identity=trigger_strong_identity,
    )
    config = {
        "campaign_id": campaign_id,
        "formal": formal,
        "expected_sessions": FORMAL_SESSION_COUNT if formal else None,
        "expected_dialogues": FORMAL_DIALOGUE_COUNT if formal else len(rows),
        "runtime": runtime_kind,
        "device": device,
        "protocol": protocol.to_dict(),
        "campaign_identity": campaign_identity,
        "main_model_identity": main_identity,
        "runtime_metadata": {
            "resolved_dtype": None,
            "attention_backend": None,
            "capture_stage": "session runtime after model load",
        },
        "trigger_model_identity": trigger_strong_identity,
        "trigger_cache_identity_hash": trigger.identity_hash,
    }
    manifest = build_manifest(
        experiment=EXPERIMENT,
        run_id=campaign_id,
        config=config,
        input_path=input_path,
        sample_ids=[row.id for row in rows],
        extra={
            "manifest_kind": "immutable_campaign",
            "trigger_cache_path": str(trigger_cache_path.resolve()),
            "trigger_cache_sha256": trigger.cache_sha256,
        },
    )
    manifest["campaign_manifest_schema_version"] = CAMPAIGN_MANIFEST_SCHEMA_VERSION
    manifest["campaign_identity"] = campaign_identity
    manifest["identity_hash"] = campaign_identity["identity_hash"]
    manifest["manifest_content_hash"] = config_hash(
        {key: value for key, value in manifest.items() if key != "manifest_content_hash"}
    )
    return manifest


def validate_campaign_manifest_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("campaign_manifest_schema_version") != CAMPAIGN_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported campaign manifest schema")
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
    payload["_artifact_sha256"] = sha256_file(path)
    payload["_artifact_path"] = str(path.resolve())
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--trigger-cache", type=Path, required=True)
    parser.add_argument("--main-model", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--non-formal", action="store_true")
    args = parser.parse_args()
    formal = not args.non_formal
    if formal:
        require_clean_tree(allow_dirty=False)
        enforce_offline_mode()
        if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
            raise SystemExit("Formal campaign manifest requires Hugging Face tokens to be empty")
    if args.output.exists():
        raise FileExistsError(f"Campaign manifest already exists: {args.output}")
    manifest = build_campaign_manifest(
        campaign_id=args.campaign_id,
        input_path=args.input,
        trigger_cache_path=args.trigger_cache,
        main_model_path=args.main_model,
        device=args.device,
        protocol=ProtocolConfig(),
        formal=formal,
    )
    atomic_write_json(args.output, manifest)
    print(canonical_json({"path": str(args.output.resolve()), "sha256": sha256_file(args.output)}))


if __name__ == "__main__":
    main()
