"""No-model, no-network smoke suite for the SCI supplementary campaign."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

from experiments.sci34_supplement import a1_joint_latency, async_bargein
from experiments.sci34_supplement.analyze_e3 import analyze_records, validate_design
from experiments.sci34_supplement.analyze_latency import analyze_a1, analyze_async
from experiments.sci34_supplement.common import (
    build_manifest,
    config_hash,
    load_jsonl,
    prepare_run_directory,
)
from experiments.sci34_supplement.e3_fixed_trajectory import run_experiment, validate_records
from experiments.sci34_supplement.model_runtime import FakeChatRuntime


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mini_dialogues.json"


def smoke_manifest(root: Path) -> None:
    config = {"seed": 7, "threshold": 0.92}
    first = build_manifest(
        experiment="smoke",
        run_id="manifest",
        config=config,
        input_path=FIXTURE,
        sample_ids=["a", "b"],
    )
    run_dir = prepare_run_directory(
        results_root=root,
        run_id="manifest",
        manifest=first,
        resume=False,
    )
    second = build_manifest(
        experiment="smoke",
        run_id="manifest",
        config=config,
        input_path=FIXTURE,
        sample_ids=["a", "b"],
    )
    assert first["config_hash"] == second["config_hash"] == config_hash(config)
    prepare_run_directory(
        results_root=root,
        run_id="manifest",
        manifest=second,
        resume=True,
    )
    changed = dict(second)
    changed["config_hash"] = config_hash({"seed": 8})
    try:
        prepare_run_directory(
            results_root=root,
            run_id="manifest",
            manifest=changed,
            resume=True,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Manifest mismatch was not rejected")
    assert (run_dir / "manifest.json").exists()


def smoke_e3(root: Path) -> None:
    args = SimpleNamespace(
        dialogues=FIXTURE,
        run_id="e3",
        results_root=root,
        runtime="fake",
        model=None,
        device="cpu",
        seed=20260831,
        formal=False,
        limit=None,
        max_first_tokens=40,
        max_probe_tokens=20,
        sample_rate=1000,
        samples_per_char=15,
        chunker_tokenizer="nltk",
        resume=False,
    )
    runtime = FakeChatRuntime(seed=args.seed)
    run_dir = run_experiment(args, runtime)
    records = load_jsonl(run_dir / "records.jsonl")
    validate_records(run_dir / "records.jsonl")
    assert len(records) == 2 * 4 * 2
    assert len({row["trajectory_id"] for row in records if row["id"] == "smoke_hotel"}) == 1
    assert all(not row["local_unheard_in_history_text"] for row in records if row["condition"] == "playback")
    calls_before_resume = len(runtime.calls)
    args.resume = True
    run_experiment(args, runtime)
    assert len(runtime.calls) == calls_before_resume, "Resume unexpectedly called the backend"


def _e3_analysis_record(
    *,
    condition: str,
    unheard_text: str,
    strict_unheard_text: str,
) -> dict[str, object]:
    return {
        "id": "smoke_analysis",
        "fraction": 0.25,
        "condition": condition,
        "trajectory_id": "shared-trajectory",
        "history_key": "generation-history" if condition == "generation" else "playback-history",
        "unheard_text": unheard_text,
        "strict_unheard_text": strict_unheard_text,
        "referenced_unheard": False,
        "referenced_unheard_strict": condition == "generation",
        "judge_fragment": False,
        "judge_proxy": condition == "generation",
        "local_unheard_in_history_text": "",
        "local_referenced_unheard": False,
    }


def smoke_e3_analyzer() -> None:
    records = [
        _e3_analysis_record(
            condition=condition,
            unheard_text="   ",
            strict_unheard_text="proxy tail",
        )
        for condition in ("playback", "generation")
    ]
    result = analyze_records(
        records,
        repeats=20,
        seed=20260831,
        include_judge=True,
    )
    eligibility = result["design"]["eligibility_by_target"]
    assert result["design"]["eligible_pairs"] == 0
    assert eligibility["fragment"]["eligible_pairs"] == 0
    assert eligibility["proxy"]["eligible_pairs"] == 1
    assert "rule_fragment" not in result["metrics"]
    assert "rule_fragment" not in result["by_fraction"]["0.25"]["metrics"]
    assert result["metrics"]["rule_proxy"]["conditions"]["generation"]["n"] == 1
    assert result["metrics"]["rule_proxy"]["conditions"]["generation"]["positive"] == 1
    assert result["metrics"]["judge_proxy"]["conditions"]["generation"]["positive"] == 1

    for target_field, expected_error in (
        ("unheard_text", "exact fragment target"),
        ("strict_unheard_text", "exact proxy target"),
    ):
        mismatched = [dict(record) for record in records]
        mismatched[1][target_field] = "different target"
        try:
            validate_design(mismatched)
        except ValueError as error:
            assert expected_error in str(error)
        else:
            raise AssertionError(f"Mismatched {target_field} was not rejected")


def smoke_a1(root: Path) -> Path:
    args = argparse.Namespace(
        runtime="fake",
        model=None,
        device="cpu",
        lengths=[256, 512],
        crop_tokens=32,
        warmup=1,
        repeats=4,
        run_id="a1",
        results_root=root,
        resume=False,
        log_level="INFO",
    )
    config = {
        "runtime": args.runtime,
        "lengths": args.lengths,
        "crop_tokens": args.crop_tokens,
        "warmup": args.warmup,
        "repeats": args.repeats,
    }
    manifest = build_manifest(
        experiment="a1_joint_latency",
        run_id=args.run_id,
        config=config,
        sample_ids=[str(value) for value in args.lengths],
    )
    run_dir = prepare_run_directory(
        results_root=root,
        run_id=args.run_id,
        manifest=manifest,
        resume=False,
    )
    records_path = run_dir / "records.jsonl"
    from experiments.sci34_supplement.common import append_jsonl

    for length in args.lengths:
        append_jsonl(records_path, a1_joint_latency.measure_length(args, None, length))
    records = load_jsonl(records_path)
    result = analyze_a1(records)
    assert all(row["statistics"]["crop_role_joint_ms"]["n"] == 4 for row in result["rows"])
    return run_dir


def smoke_async(root: Path) -> Path:
    args = argparse.Namespace(
        runtime="fake",
        model=None,
        device="cpu",
        lengths=[512],
        fractions=[0.25, 0.5, 0.75],
        warmups=1,
        repeats=2,
        run_id="smoke_async_prepared_v2",
        results_root=root,
        sample_rate=1000,
        duration_s=1.2,
        fragments=6,
        block_ms=20.0,
        time_scale=1.0,
        resume=False,
    )
    calls: list[tuple[str, int, float, int]] = []
    original_run_trial = async_bargein.run_trial

    def tracked_run_trial(args, fixture, length, fraction, repeat, *, trial_kind="formal"):
        calls.append((trial_kind, length, fraction, repeat))
        return original_run_trial(
            args,
            fixture,
            length,
            fraction,
            repeat,
            trial_kind=trial_kind,
        )

    async_bargein.run_trial = tracked_run_trial
    try:
        run_dir = async_bargein.run_experiment(args)
        records = load_jsonl(run_dir / "records.jsonl")
        assert len(records) == 6
        assert len([call for call in calls if call[0] == "warmup"]) == 3
        assert all(record["trial_kind"] == "formal" for record in records)
        assert all(record["protocol"] == async_bargein.PROTOCOL for record in records)
        assert all(record["prepared_state_synchronized"] for record in records)
        assert all(record["setup_ms"] >= 0 for record in records)
        assert all(record["post_stop_sync_ms"] >= 0 for record in records)
        assert all(record["stop_to_sync_done_ms"] >= record["stop_ack_ms"] for record in records)
        assert all(record["played_at_request"] == record["target_samples"] for record in records)
        assert all(record["played_at_ack"] == record["target_samples"] for record in records)
        assert all(record["leaked_samples"] == 0 for record in records)
        assert {
            record["fraction"]: record["partial"] for record in records
        } == {0.25: True, 0.5: False, 0.75: True}
        assert all(record["partial_expectation_met"] for record in records)
        assert all(record["leaked_samples"] >= 0 for record in records)
        result = analyze_async(records)
        assert result["protocol"] == async_bargein.PROTOCOL
        assert result["records_per_cell"] == 2
        assert len(result["rows"]) == 3
        assert {row["path_kind"] for row in result["rows"]} == {
            "mid_fragment",
            "fragment_boundary",
        }
        assert all(
            row["partial_rate"] == row["expected_partial_rate"]
            for row in result["rows"]
        )

        # Simulate an interrupted run.  Resume must warm and refill only the
        # incomplete (length, fraction) cell, not already complete cells.
        retained = [
            record
            for record in records
            if not (record["fraction"] == 0.75 and record["repeat"] == 1)
        ]
        (run_dir / "records.jsonl").write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in retained),
            encoding="utf-8",
        )
        calls.clear()
        args.resume = True

        # Exercise resume scheduling without adding wall-clock jitter to this
        # bookkeeping assertion; the first run above already covered real
        # partial/boundary trials through the fake runtime.
        saved_missing_record = next(
            record
            for record in records
            if record["fraction"] == 0.75 and record["repeat"] == 1
        )

        def deterministic_resume_trial(
            args, fixture, length, fraction, repeat, *, trial_kind="formal"
        ):
            calls.append((trial_kind, length, fraction, repeat))
            if trial_kind == "formal":
                return dict(saved_missing_record)
            return None

        async_bargein.run_trial = deterministic_resume_trial
        async_bargein.run_experiment(args)
        assert calls == [
            ("warmup", 512, 0.75, -1),
            ("formal", 512, 0.75, 1),
        ]
        assert len(load_jsonl(run_dir / "records.jsonl")) == 6

        calls.clear()
        async_bargein.run_experiment(args)
        assert not calls, "Complete resume unexpectedly ran warmups or formal trials"
        return run_dir
    finally:
        async_bargein.run_trial = original_run_trial


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()
    root = Path(tempfile.mkdtemp(prefix="sci34-smoke-"))
    try:
        smoke_manifest(root / "manifest")
        smoke_e3(root / "e3")
        smoke_e3_analyzer()
        smoke_a1(root / "a1")
        smoke_async(root / "async")
        print(json.dumps({"status": "PASS", "root": str(root)}, ensure_ascii=False))
    finally:
        if not args.keep:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
