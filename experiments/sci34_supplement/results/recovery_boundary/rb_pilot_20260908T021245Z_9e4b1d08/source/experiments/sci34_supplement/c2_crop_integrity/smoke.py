"""Pure-CPU full-workflow smoke and tamper tests for C2 v3."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

from experiments.sci34_supplement.common import atomic_write_json, atomic_write_text, load_jsonl, sha256_file
from experiments.sci34_supplement.c2_crop_integrity.analyze import build_analysis
from experiments.sci34_supplement.c2_crop_integrity.campaign import DEFAULT_CASES, build_campaign_manifest, prepare_campaign_directory
from experiments.sci34_supplement.c2_crop_integrity.integrity import record_content_hash
from experiments.sci34_supplement.c2_crop_integrity.protocol import FORMAL_CASE_COUNT, FORMAL_CROP_EVENT_COUNT, PROTOCOL_VERSION, load_cases
from experiments.sci34_supplement.c2_crop_integrity.run import run_campaign
from experiments.sci34_supplement.c2_crop_integrity.runtime import FakeBackend
from experiments.sci34_supplement.c2_crop_integrity.seal import create_seal, verify_seal
from experiments.sci34_supplement.c2_crop_integrity.validate import validate_campaign


PACKAGE_ROOT = Path(__file__).resolve().parent


def _rewrite_records(path: Path, records: list[dict]) -> None:
    atomic_write_text(path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records))


def _tampered_copy(source: Path, root: Path, name: str) -> Path:
    target = root / name
    shutil.copytree(source, target)
    for artifact in ("validation.json", "analysis_v1.json", "ACCEPTANCE.md", "checksums.sha256"):
        (target / artifact).unlink(missing_ok=True)
    return target


def _assert_detected(path: Path, needle: str | None = None) -> None:
    result = validate_campaign(path, formal=False, expected_cases=FORMAL_CASE_COUNT)
    assert not result["ok"], "tampering was not detected"
    if needle is not None:
        assert any(needle in error for error in result["errors"]), result["errors"][:10]


def main() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_TOKEN"] = ""
    os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
    cases = load_cases(DEFAULT_CASES, formal=True)
    assert len(cases) == FORMAL_CASE_COUNT
    root = Path(tempfile.mkdtemp(prefix="c2-crop-integrity-smoke-"))
    try:
        campaign_dir = root / "campaign"
        backend = FakeBackend(seed=17)
        manifest = build_campaign_manifest(
            run_id="c2-v3-smoke",
            cases_path=DEFAULT_CASES,
            runtime_kind="fake",
            model_path=None,
            device="cpu",
            seed=17,
            formal=False,
            backend=backend,
        )
        prepare_campaign_directory(output_dir=campaign_dir, manifest=manifest, cases_path=DEFAULT_CASES)
        run_campaign(
            campaign_dir=campaign_dir,
            runtime_kind="fake",
            model_path=None,
            device="cpu",
            seed=17,
            resume=False,
            limit=None,
            backend=backend,
        )
        (campaign_dir / "logs" / "smoke.log").write_text("pure CPU fake workflow\n", encoding="utf-8")
        for side in ("before", "after"):
            snapshot = campaign_dir / "snapshots" / side
            snapshot.mkdir(parents=True)
            (snapshot / "state.txt").write_text(f"{side}\n", encoding="utf-8")
        validation = validate_campaign(campaign_dir, formal=False, expected_cases=FORMAL_CASE_COUNT)
        assert validation["ok"], validation["errors"][:10]
        assert validation["grid"]["crop_events"] == FORMAL_CROP_EVENT_COUNT
        analysis = build_analysis(campaign_dir, formal=False)
        assert analysis["overall"]["cases"] == FORMAL_CASE_COUNT
        assert analysis["overall"]["crop_events"] == FORMAL_CROP_EVENT_COUNT
        atomic_write_json(campaign_dir / "validation.json", validation)
        atomic_write_json(campaign_dir / "analysis_v1.json", analysis)
        (campaign_dir / "ACCEPTANCE.md").write_text("# Smoke acceptance\n\nStatus: accepted\n", encoding="utf-8")
        create_seal(campaign_dir, formal=False)
        assert verify_seal(campaign_dir)["ok"]

        # Wrong keep length, with record hash recomputed so structural validation must catch it.
        tampered = _tampered_copy(campaign_dir, root, "wrong-keep")
        records = load_jsonl(tampered / "records.jsonl")
        event = records[0]["crop_events"][0]
        event["keep_length"] += 1
        event["retained_token_ids"].append(event["pre_crop_token_ids"][event["keep_length"] - 1])
        from experiments.sci34_supplement.c2_crop_integrity.canonical_chat import token_ids_hash
        event["retained_token_hash"] = token_ids_hash(event["retained_token_ids"])
        records[0]["record_content_hash"] = record_content_hash(records[0])
        _rewrite_records(tampered / "records.jsonl", records)
        _assert_detected(tampered, "independent case/partition derivation")

        # Alter one layer hash and recompute record JSON hash: aggregate and cross-equality must fail.
        tampered = _tampered_copy(campaign_dir, root, "layer-hash")
        records = load_jsonl(tampered / "records.jsonl")
        records[0]["crop_events"][0]["post_production_manifest"]["layers"][0]["key"]["sha256"] = "0" * 64
        records[0]["record_content_hash"] = record_content_hash(records[0])
        _rewrite_records(tampered / "records.jsonl", records)
        _assert_detected(tampered, "aggregate hash")

        # Duplicate EOT/ledger event.
        tampered = _tampered_copy(campaign_dir, root, "duplicate-eot")
        records = load_jsonl(tampered / "records.jsonl")
        event = records[0]["crop_events"][0]
        if event["expected_recovery_chunks"]:
            event["expected_recovery_chunks"][0]["token_ids"].append(2)
        else:
            event["expected_recovery_chunks"].append({"operation": "reopen_user_role", "token_ids": [2, 2]})
        records[0]["record_content_hash"] = record_content_hash(records[0])
        _rewrite_records(tampered / "records.jsonl", records)
        _assert_detected(tampered, "duplicate or missing EOT")

        # Missing crop event.
        tampered = _tampered_copy(campaign_dir, root, "missing-event")
        records = load_jsonl(tampered / "records.jsonl")
        records[-1]["crop_events"].pop()
        records[-1]["record_content_hash"] = record_content_hash(records[-1])
        _rewrite_records(tampered / "records.jsonl", records)
        _assert_detected(tampered, "crop event count differs")

        # A validation-pass directory without environment snapshots still cannot seal.
        no_snapshot = _tampered_copy(campaign_dir, root, "missing-snapshot")
        shutil.rmtree(no_snapshot / "snapshots")
        atomic_write_json(
            no_snapshot / "validation.json",
            validate_campaign(no_snapshot, formal=False, expected_cases=FORMAL_CASE_COUNT),
        )
        atomic_write_json(no_snapshot / "analysis_v1.json", build_analysis(no_snapshot, formal=False))
        (no_snapshot / "ACCEPTANCE.md").write_text("Status: accepted\n", encoding="utf-8")
        try:
            create_seal(no_snapshot, formal=False)
        except ValueError as error:
            assert "snapshots" in str(error)
        else:
            raise AssertionError("seal accepted missing snapshots")

        negative = manifest["config"]["negative_control"]
        assert negative["detected"] is True
        assert negative["positive_control_metadata"]["wrong_crop_length_would_be_detected"] is True

        # Existing C2 v2 summary regression: probe qualification is independent of case verdict.
        from experiments.sci34_supplement.c2_equivalence.run import _count_runner_qualified_probes
        assert _count_runner_qualified_probes([
            {"passed": False, "termination_probe": {"passed": True}},
            {"passed": True, "termination_probe": {"passed": False}},
        ]) == 1

        print(
            json.dumps(
                {
                    "status": "PASS",
                    "protocol_version": PROTOCOL_VERSION,
                    "cases": FORMAL_CASE_COUNT,
                    "crop_events": FORMAL_CROP_EVENT_COUNT,
                    "tamper_tests": ["wrong_keep", "layer_hash", "duplicate_eot_ledger", "missing_event"],
                    "negative_control": True,
                },
                sort_keys=True,
            )
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
