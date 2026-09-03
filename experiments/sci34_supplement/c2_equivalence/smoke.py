"""Pure-CPU fake smoke for the full C2 artifact workflow."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

from experiments.sci34_supplement.common import atomic_write_json, atomic_write_text, load_jsonl, sha256_file
from experiments.sci34_supplement.c2_equivalence.analyze import build_analysis
from experiments.sci34_supplement.c2_equivalence.campaign import (
    DEFAULT_CASES,
    build_campaign_manifest,
    prepare_campaign_directory,
)
from experiments.sci34_supplement.c2_equivalence.protocol import load_cases
from experiments.sci34_supplement.c2_equivalence.run import run_campaign
from experiments.sci34_supplement.c2_equivalence.runtime import FakeBackend
from experiments.sci34_supplement.c2_equivalence.seal import create_seal, verify_seal
from experiments.sci34_supplement.c2_equivalence.validate import validate_campaign


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[2]
GUARDED_RESULTS = (
    REPO_ROOT / "experiments" / "results" / "exp1_latency.json",
    REPO_ROOT / "experiments" / "results" / "exp2_tradeoff.json",
    REPO_ROOT / "experiments" / "results" / "paper2_reanalysis.json",
    REPO_ROOT / "experiments" / "sci34_supplement" / "results" / "e3" / "sci34_f11ccba_20260901_e3" / "manifest.json",
)


def _expect(error_type, function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except error_type:
        return
    raise AssertionError(f"Expected {error_type.__name__} from {function.__name__}")


def main() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_TOKEN"] = ""
    os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
    before = {path: sha256_file(path) for path in GUARDED_RESULTS if path.exists()}
    root = Path(tempfile.mkdtemp(prefix="c2-equivalence-smoke-"))
    try:
        cases = load_cases(DEFAULT_CASES, formal=True)
        assert 20 <= len(cases) <= 24
        assert {case.context_tokens for case in cases} == {512, 2048, 8192}
        assert len({case.id for case in cases}) == len(cases)

        # Pilot and formal-like fake directories stay distinct.
        pilot_dir = root / "pilot"
        pilot_backend = FakeBackend(seed=7)
        pilot_manifest = build_campaign_manifest(
            run_id="c2-smoke-pilot",
            cases_path=DEFAULT_CASES,
            runtime_kind="fake",
            model_path=None,
            device="cpu",
            seed=7,
            formal=False,
            backend=pilot_backend,
        )
        prepare_campaign_directory(
            output_dir=pilot_dir,
            manifest=pilot_manifest,
            cases_path=DEFAULT_CASES,
        )
        run_campaign(
            campaign_dir=pilot_dir,
            runtime_kind="fake",
            model_path=None,
            device="cpu",
            seed=7,
            resume=False,
            limit=3,
            backend=pilot_backend,
        )
        pilot_validation = validate_campaign(pilot_dir, formal=False, expected_cases=3)
        assert pilot_validation["ok"], pilot_validation["errors"]

        campaign_dir = root / "formal-fake"
        backend = FakeBackend(seed=11)
        manifest = build_campaign_manifest(
            run_id="c2-smoke-formal-like",
            cases_path=DEFAULT_CASES,
            runtime_kind="fake",
            model_path=None,
            device="cpu",
            seed=11,
            formal=False,
            backend=backend,
        )
        prepare_campaign_directory(
            output_dir=campaign_dir,
            manifest=manifest,
            cases_path=DEFAULT_CASES,
        )
        run_campaign(
            campaign_dir=campaign_dir,
            runtime_kind="fake",
            model_path=None,
            device="cpu",
            seed=11,
            resume=False,
            limit=None,
            backend=backend,
        )
        first_records = load_jsonl(campaign_dir / "records.jsonl")
        assert len(first_records) == len(cases)
        assert all(record["passed"] for record in first_records)
        assert {record["termination_probe"]["observed_end_reason"] for record in first_records} == {"EOS", "MAX_TOKENS"}
        assert all(
            checkpoint["termination_probe"] == record["termination_probe"]
            for record in first_records
            for checkpoint in record["checkpoints"]
        )
        assert all(
            checkpoint["scenario_execution"] == record["scenario_execution"]
            for record in first_records
            for checkpoint in record["checkpoints"]
        )
        pending_records = [
            record for record in first_records
            if record["scenario"] == "crop_pending_eot"
        ]
        tail_records = [
            record for record in first_records
            if record["scenario"] == "reply_tail_noop"
        ]
        assert all(
            record["scenario_execution"]["pending_before_crop"] is True
            and record["scenario_execution"]["eot_in_full_ledger_before_crop"] is False
            and record["scenario_execution"]["eot_in_content_ledger_before_crop"] is False
            and record["scenario_execution"]["pending_cleared_by_crop"] is True
            and record["scenario_execution"]["pending_after_crop"] is False
            for record in pending_records
        )
        assert all(
            record["scenario_execution"]["pending_before_crop"] is True
            and record["scenario_execution"]["crop_was_noop"] is True
            and record["scenario_execution"]["no_op_preserved_pending"] is True
            and record["scenario_execution"]["pending_after_crop"] is True
            for record in tail_records
        )
        assert all(
            checkpoint["continuation"]["continuation_source"] == "actual_crop_cache"
            and checkpoint["continuation"]["canonical_source"] == "clean_prefill_cache"
            and checkpoint["continuation"]["checkpoint_state_captured_before_mutation"] is True
            for record in first_records
            for checkpoint in record["checkpoints"]
        )
        runtime_source = (PACKAGE_ROOT / "runtime.py").read_text(encoding="utf-8")
        assert "self._continue(path_cache, CONTINUATION_TOKENS)" in runtime_source
        assert "_clone_cache_for_continuation" not in runtime_source
        assert all(
            record["termination_probe"]["eos_step"] == record["termination_probe"]["cap"]
            for record in first_records
            if record["termination"] == "eos_at_cap"
        )
        assert all(
            record["termination_probe"]["observed_end_reason"] == "MAX_TOKENS"
            and record["termination_probe"]["content_token_count"]
            == record["termination_probe"]["cap"]
            for record in first_records
            if record["termination"] == "max_tokens"
        )
        p0_records = [
            record for record in first_records
            if record["scenario"] == "full_rollback_p0"
        ]
        invalidation_records = [
            record for record in first_records
            if record["scenario"] == "speculation_full_invalidation"
        ]
        assert all(
            checkpoint["canonical"]["boundaries"]["zero_retain_semantics"]
            == "full_rollback_p0"
            and checkpoint["unique_eot"]["assistant_boundaries"] == 1
            for record in p0_records
            for checkpoint in record["checkpoints"]
        )
        assert all(
            checkpoint["canonical"]["boundaries"]["zero_retain_semantics"]
            == "speculation_full_invalidation"
            and checkpoint["unique_eot"]["assistant_boundaries"] == 0
            for record in invalidation_records
            for checkpoint in record["checkpoints"]
        )
        assert all(
            p0["checkpoints"][0]["canonical"]["token_ids"]
            != invalidated["checkpoints"][0]["canonical"]["token_ids"]
            for p0, invalidated in zip(p0_records, invalidation_records)
        )
        assert 'prefill_user_text(cache, "\\n" + case.next_user)' not in runtime_source
        canonical_source = (PACKAGE_ROOT / "canonical_chat.py").read_text(encoding="utf-8")
        assert "if user_end != role_start:" in canonical_source
        assert "role_start + len(parts.user_to_assistant)" in canonical_source

        # v2: noise-control arm, margins, and frozen per-checkpoint logits sidecars.
        import numpy as np
        from experiments.sci34_supplement.c2_equivalence.protocol import PROTOCOL_VERSION
        from experiments.sci34_supplement.c2_equivalence.validate import _validate_termination_probe
        from experiments.sci34_supplement.c2_equivalence.canonical_chat import token_ids_hash as _hash

        checkpoint_total = sum(len(record["checkpoints"]) for record in first_records)
        npz_files = sorted((campaign_dir / "checkpoints").glob("*.npz"))
        assert len(npz_files) == checkpoint_total
        assert all(
            record["protocol_version"] == PROTOCOL_VERSION
            and record["checkpoint_logits"] == [
                f"checkpoints/{record['case_id']}.attempt{record['attempt']}.{cp['checkpoint']}.npz"
                for cp in record["checkpoints"]
            ]
            for record in first_records
        )
        assert all(
            isinstance(cp["noise_control"], dict)
            and cp["noise_control"]["chunk_count"] >= 1
            and cp["noise_control"]["max_abs"] >= 0.0
            and cp["logit_gates"]["all_ok"] is True
            and cp["next_token"]["canonical_top1_top2_margin"] >= 0.0
            and cp["next_token"]["top1_flip_near_tie"] is False
            and cp["continuation"]["canonical_steps"][0]["top1"]
            == cp["next_token"]["canonical_top1"]
            for record in first_records
            for cp in record["checkpoints"]
        )
        assert all(
            record["termination_probe"]["genuine_eos"] is True
            for record in first_records
            if record["termination"] == "natural_eos"
        )

        # v2 requalification semantics exercised directly on the probe validator.
        natural_record = next(
            record for record in first_records if record["termination"] == "natural_eos"
        )
        natural_case = next(
            case for case in cases if case.id == natural_record["case_id"]
        )
        genuine_probe = natural_record["termination_probe"]
        assert not _validate_termination_probe(genuine_probe, case=natural_case, formal=False)
        requalified_probe = json.loads(json.dumps(genuine_probe))
        requalified_probe.update(
            {
                "observed_end_reason": "MAX_TOKENS",
                "eos_step": None,
                "eos_at_cap": False,
                "genuine_eos": False,
                "requalified": True,
                "role_phase": "ASSISTANT_OPEN",
                "content_token_ids": [7001] * requalified_probe["cap"],
            }
        )
        requalified_probe["content_token_count"] = len(requalified_probe["content_token_ids"])
        requalified_probe["content_token_hash"] = _hash(requalified_probe["content_token_ids"])
        requalified_probe["selected_token_ids"] = list(requalified_probe["content_token_ids"])
        requalified_probe["selected_token_count"] = len(requalified_probe["selected_token_ids"])
        requalified_probe["selected_token_hash"] = _hash(requalified_probe["selected_token_ids"])
        requalified_probe["post_seq_length"] = (
            requalified_probe["prefix_seq_length"] + requalified_probe["content_token_count"]
        )
        assert not _validate_termination_probe(
            requalified_probe, case=natural_case, formal=False
        ), "self-consistent requalified probe must qualify"
        broken_requalified = json.loads(json.dumps(requalified_probe))
        broken_requalified["role_phase"] = "ASSISTANT_EOT_PENDING"
        assert _validate_termination_probe(
            broken_requalified, case=natural_case, formal=False
        ), "inconsistent requalified probe must fail closed"
        mixed_probe = json.loads(json.dumps(genuine_probe))
        mixed_probe["genuine_eos"] = None
        mixed_probe["requalified"] = None
        assert _validate_termination_probe(
            mixed_probe, case=natural_case, formal=False
        ), "probe that is neither genuine nor requalified must fail closed"

        # Complete resume is a no-op and preserves record bytes.
        records_sha = sha256_file(campaign_dir / "records.jsonl")
        run_campaign(
            campaign_dir=campaign_dir,
            runtime_kind="fake",
            model_path=None,
            device="cpu",
            seed=11,
            resume=True,
            limit=None,
            backend=backend,
        )
        assert records_sha == sha256_file(campaign_dir / "records.jsonl")

        validation = validate_campaign(campaign_dir, formal=False)
        assert validation["ok"], validation["errors"]
        atomic_write_json(campaign_dir / "validation.json", validation)
        analysis = build_analysis(campaign_dir, formal=False)
        assert analysis["acceptance"]["passed"]
        assert analysis["design"]["bootstrap"] is None
        assert analysis["termination_probes"]["cases"] == len(cases)
        assert analysis["termination_probes"]["qualified"] == len(cases)
        assert set(analysis["termination_probes"]["by_declared_label"]) == {
            "natural_eos", "eos_at_cap", "max_tokens"
        }
        atomic_write_json(campaign_dir / "analysis_v1.json", analysis)
        atomic_write_text(
            campaign_dir / "ACCEPTANCE.md",
            "# Smoke acceptance\n\nStatus: accepted\n",
        )
        (campaign_dir / "snapshots" / "before").mkdir(parents=True)
        (campaign_dir / "snapshots" / "after").mkdir(parents=True)
        atomic_write_text(campaign_dir / "logs" / "smoke.log", "smoke\n")
        atomic_write_text(
            campaign_dir / "snapshots" / "before" / "state.txt", "before\n"
        )
        atomic_write_text(
            campaign_dir / "snapshots" / "after" / "state.txt", "after\n"
        )
        create_seal(campaign_dir, formal=False)
        assert verify_seal(campaign_dir)["ok"]

        # Tampering and a failed case both fail acceptance while retaining artifacts.
        tamper_dir = root / "tamper"
        shutil.copytree(campaign_dir, tamper_dir)
        (tamper_dir / "checksums.sha256").unlink()
        records = load_jsonl(tamper_dir / "records.jsonl")
        records[0]["checkpoints"][0]["next_token"]["path_top1"] += 1
        atomic_write_text(
            tamper_dir / "records.jsonl",
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        )
        tampered = validate_campaign(tamper_dir, formal=False)
        assert not tampered["ok"]
        _expect(ValueError, build_analysis, tamper_dir, formal=False)

        source_tamper_dir = root / "continuation-source-tamper"
        shutil.copytree(campaign_dir, source_tamper_dir)
        (source_tamper_dir / "checksums.sha256").unlink()
        source_records = load_jsonl(source_tamper_dir / "records.jsonl")
        source_records[0]["checkpoints"][0]["continuation"][
            "continuation_source"
        ] = "clean_prefill_clone"
        atomic_write_text(
            source_tamper_dir / "records.jsonl",
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in source_records),
        )
        source_tampered = validate_campaign(source_tamper_dir, formal=False)
        assert not source_tampered["ok"]
        assert any("actual crop/recovery cache" in error for error in source_tampered["errors"])

        npz_tamper_dir = root / "npz-tamper"
        shutil.copytree(campaign_dir, npz_tamper_dir)
        (npz_tamper_dir / "checksums.sha256").unlink()
        first_npz = sorted((npz_tamper_dir / "checkpoints").glob("*.npz"))[0]
        with np.load(first_npz, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        arrays["path"] = np.asarray(arrays["path"], dtype=np.float32) + np.float32(5.0)
        np.savez_compressed(first_npz, **arrays)
        npz_tampered = validate_campaign(npz_tamper_dir, formal=False)
        assert not npz_tampered["ok"]
        assert any(
            "differs from sidecar recompute" in error for error in npz_tampered["errors"]
        )

        control_tamper_dir = root / "control-tamper"
        shutil.copytree(campaign_dir, control_tamper_dir)
        (control_tamper_dir / "checksums.sha256").unlink()
        control_records = load_jsonl(control_tamper_dir / "records.jsonl")
        control_records[0]["checkpoints"][0]["noise_control"]["max_abs"] = 9.9
        atomic_write_text(
            control_tamper_dir / "records.jsonl",
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in control_records),
        )
        control_tampered = validate_campaign(control_tamper_dir, formal=False)
        assert not control_tampered["ok"]
        assert any(
            "noise_control.max_abs" in error for error in control_tampered["errors"]
        )

        steps_tamper_dir = root / "steps-tamper"
        shutil.copytree(campaign_dir, steps_tamper_dir)
        (steps_tamper_dir / "checksums.sha256").unlink()
        steps_records = load_jsonl(steps_tamper_dir / "records.jsonl")
        steps_records[0]["checkpoints"][0]["continuation"]["canonical_steps"][0]["top1"] = 12345
        atomic_write_text(
            steps_tamper_dir / "records.jsonl",
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in steps_records),
        )
        steps_tampered = validate_campaign(steps_tamper_dir, formal=False)
        assert not steps_tampered["ok"]
        assert any(
            "continuation steps disagree" in error for error in steps_tampered["errors"]
        )

        scenario_tamper_dir = root / "scenario-tamper"
        shutil.copytree(campaign_dir, scenario_tamper_dir)
        (scenario_tamper_dir / "checksums.sha256").unlink()
        scenario_records = load_jsonl(scenario_tamper_dir / "records.jsonl")
        pending_record = next(
            row for row in scenario_records if row["scenario"] == "crop_pending_eot"
        )
        pending_record["scenario_execution"]["pending_cleared_by_crop"] = False
        for checkpoint in pending_record["checkpoints"]:
            checkpoint["scenario_execution"]["pending_cleared_by_crop"] = False
        atomic_write_text(
            scenario_tamper_dir / "records.jsonl",
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in scenario_records),
        )
        scenario_tampered = validate_campaign(scenario_tamper_dir, formal=False)
        assert not scenario_tampered["ok"]
        assert any("scenario_execution" in error for error in scenario_tampered["errors"])

        probe_tamper_dir = root / "probe-tamper"
        shutil.copytree(campaign_dir, probe_tamper_dir)
        (probe_tamper_dir / "checksums.sha256").unlink()
        probe_records = load_jsonl(probe_tamper_dir / "records.jsonl")
        max_record = next(row for row in probe_records if row["termination"] == "max_tokens")
        max_record["termination_probe"]["observed_end_reason"] = "EOS"
        for checkpoint in max_record["checkpoints"]:
            checkpoint["termination_probe"]["observed_end_reason"] = "EOS"
        atomic_write_text(
            probe_tamper_dir / "records.jsonl",
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in probe_records),
        )
        probe_tampered = validate_campaign(probe_tamper_dir, formal=False)
        assert not probe_tampered["ok"]
        assert any("termination_probe" in error for error in probe_tampered["errors"])
        _expect(ValueError, build_analysis, probe_tamper_dir, formal=False)

        verdict_tamper_dir = root / "probe-verdict-tamper"
        shutil.copytree(campaign_dir, verdict_tamper_dir)
        (verdict_tamper_dir / "checksums.sha256").unlink()
        verdict_records = load_jsonl(verdict_tamper_dir / "records.jsonl")
        verdict_record = next(row for row in verdict_records if row["termination"] == "natural_eos")
        verdict_record["termination_probe"]["errors"] = ["injected probe failure"]
        verdict_record["termination_probe"]["passed"] = False
        verdict_record["errors"] = ["termination_probe: injected probe failure"]
        verdict_record["passed"] = False
        for checkpoint in verdict_record["checkpoints"]:
            checkpoint["termination_probe"] = dict(verdict_record["termination_probe"])
        atomic_write_text(
            verdict_tamper_dir / "records.jsonl",
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in verdict_records),
        )
        verdict_tampered = validate_campaign(verdict_tamper_dir, formal=False)
        assert not verdict_tampered["ok"], "self-consistent failed verdict must not qualify"

        incomplete_seal_dir = root / "incomplete-seal"
        shutil.copytree(campaign_dir, incomplete_seal_dir)
        (incomplete_seal_dir / "checksums.sha256").unlink()
        shutil.rmtree(incomplete_seal_dir / "snapshots")
        _expect(ValueError, create_seal, incomplete_seal_dir, formal=False)

        derived_tamper_dir = root / "derived-tamper"
        shutil.copytree(campaign_dir, derived_tamper_dir)
        (derived_tamper_dir / "checksums.sha256").unlink()
        derived = json.loads(
            (derived_tamper_dir / "validation.json").read_text(encoding="utf-8")
        )
        derived["termination_probes"]["qualified"] -= 1
        atomic_write_json(derived_tamper_dir / "validation.json", derived)
        _expect(ValueError, create_seal, derived_tamper_dir, formal=False)

        missing_summary_dir = root / "missing-summary"
        shutil.copytree(campaign_dir, missing_summary_dir)
        (missing_summary_dir / "checksums.sha256").unlink()
        (missing_summary_dir / "summary.json").unlink()
        _expect(ValueError, create_seal, missing_summary_dir, formal=False)

        after = {path: sha256_file(path) for path in before}
        assert before == after, "C2 smoke modified guarded legacy results"
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "protocol_version": PROTOCOL_VERSION,
                    "models_loaded": False,
                    "network_used": False,
                    "cases": len(cases),
                    "pilot_cases": 3,
                    "formal_like_cases": len(first_records),
                    "checkpoint_sidecars": len(npz_files),
                    "root": str(root),
                },
                ensure_ascii=False,
            )
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
