"""No-model, no-network smoke suite for the SCI supplementary campaign."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

from experiments.sci34_supplement import a1_joint_latency, async_bargein
from experiments.sci34_supplement.analyze_e3 import main as _unused_analyze_main
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
        sample_rate=1000,
        duration_s=0.04,
        fragments=4,
        block_ms=2.0,
        time_scale=1.0,
    )
    fixture = async_bargein.make_kv_fixture("fake", target_length=512, llm=None)
    records = [
        async_bargein.run_trial(args, fixture, 512, fraction, repeat)
        for fraction in (0.25, 0.5, 0.75)
        for repeat in range(2)
    ]
    assert all(record["leaked_samples"] >= 0 for record in records)
    result = analyze_async(records)
    assert len(result["rows"]) == 3
    run_dir = root / "async"
    run_dir.mkdir(parents=True)
    from experiments.sci34_supplement.common import append_jsonl

    for record in records:
        append_jsonl(run_dir / "records.jsonl", record)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()
    root = Path(tempfile.mkdtemp(prefix="sci34-smoke-"))
    try:
        smoke_manifest(root / "manifest")
        smoke_e3(root / "e3")
        smoke_a1(root / "a1")
        smoke_async(root / "async")
        print(json.dumps({"status": "PASS", "root": str(root)}, ensure_ascii=False))
    finally:
        if not args.keep:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
