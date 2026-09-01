"""No-model, no-network smoke suite for confirmatory E1/E2 helpers."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

from experiments.sci34_supplement.common import (
    atomic_write_json,
    build_manifest,
    prepare_run_directory,
    sha256_file,
)
from experiments.sci34_supplement.e1e2_confirmatory.campaign_support import (
    CONDITIONS,
    analysis_provenance,
    assert_balanced_orders,
    balanced_condition_order,
    hierarchical_bootstrap,
    paired_e1_difference,
    pooled_waste,
    run_missing_grid,
    validate_record_file,
    validate_ttft_record,
)
from experiments.sci34_supplement.e1e2_confirmatory.fake_backend import FakeBackend
from experiments.sci34_supplement.e1e2_confirmatory.protocol import (
    CONFIRMATORY_CONDITION,
    NEVER_SPECULATE,
    SYSTEM_A,
    threshold_for_condition,
)
from experiments.sci34_supplement.e1e2_confirmatory.holdout_builder import (
    DEFAULT_EXCLUSION_SOURCES,
    ExclusionSource,
    build_holdout,
    collect_excluded_ids,
    derive_holdout,
    exclusion_sources_with_defaults,
    validate_holdout,
    validate_new_output_path,
)
from experiments.sci34_supplement.e1e2_confirmatory.analyze import summarize_condition
from experiments.sci34_supplement.e1e2_confirmatory.run_session import (
    _validate_existing_jsonl,
    run_session,
)
from experiments.sci34_supplement.e1e2_confirmatory.strong_identity import strong_model_identity
from experiments.sci34_supplement.e1e2_confirmatory.validate import (
    ValidationError,
    validate_campaign,
    validate_grid,
    validate_timing,
)
from experiments.sci34_supplement.e1e2_confirmatory.trigger_cache import (
    ReplayTrigger,
    assert_formal_cache_prerequisites,
    build_trigger_cache_payload,
    capture_trigger_run_identity,
    write_trigger_cache,
)

PACKAGE_ROOT = Path(__file__).resolve().parent
FIXTURE = PACKAGE_ROOT / "fixtures" / "mini_multiwoz.json"
REPO_ROOT = PACKAGE_ROOT.parents[2]
OLD_RESULTS = (
    REPO_ROOT / "experiments" / "results" / "exp1_latency.json",
    REPO_ROOT / "experiments" / "results" / "exp2_tradeoff.json",
    REPO_ROOT / "experiments" / "results" / "paper2_reanalysis.json",
)


class FakeTEN:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def confidence(self, accumulated_text: str) -> float:
        self.calls.append(accumulated_text)
        # Full precision deterministic value; final prefixes survive at 0.92.
        if accumulated_text.rstrip().endswith((".", "?")):
            return 0.973456789012345
        return 0.421234567890123


def _expect_raises(error_type, function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except error_type:
        return
    raise AssertionError(f"Expected {error_type.__name__} from {function.__name__}")


def smoke_holdout_source_gates(root: Path) -> None:
    valid_sources = []
    source_payloads = (
        ("old_e1_records", {"records": [{"id": "e1.json#u0"}]}),
        ("old_e2_records", {"records": [{"id": "e2.json#u1"}]}),
        ("fixed_e3_manifest", {"input": {"sample_ids": ["e3.json"]}}),
    )
    for index, (kind, payload) in enumerate(source_payloads):
        path = root / f"source-{index}.json"
        atomic_write_json(path, payload)
        valid_sources.append(ExclusionSource(path, kind))
    excluded, provenance = collect_excluded_ids(valid_sources)
    assert {"e1.json", "e2.json", "e3.json"}.issubset(excluded)
    assert [source["kind"] for source in provenance["sources"]] == [
        kind for kind, _ in source_payloads
    ]

    malformed_payloads = (
        ("old_e1_records", {"records": []}),
        ("old_e2_records", {"records": [{"id": "  "}]}),
        ("fixed_e3_manifest", {"input": {"sample_ids": []}}),
    )
    for index, (kind, payload) in enumerate(malformed_payloads):
        path = root / f"malformed-{index}.json"
        atomic_write_json(path, payload)
        _expect_raises(
            ValueError,
            collect_excluded_ids,
            [ExclusionSource(path, kind)],
        )
    _expect_raises(ValueError, ExclusionSource, root / "missing-kind.json", "")

    custom = root / "custom-exclusion.json"
    atomic_write_json(custom, [{"id": "custom.json#u0"}])
    combined = exclusion_sources_with_defaults([custom])
    assert tuple(combined[:3]) == DEFAULT_EXCLUSION_SOURCES
    assert len(combined) == len(DEFAULT_EXCLUSION_SOURCES) + 1
    assert combined[-1].kind == "custom_auto"

    existing = root / "existing-formal-output.json"
    atomic_write_json(existing, {"sentinel": True})
    _expect_raises(
        FileExistsError,
        validate_new_output_path,
        existing,
        formal=True,
    )
    for protected in OLD_RESULTS:
        _expect_raises(
            ValueError,
            validate_new_output_path,
            protected,
            formal=True,
        )


def smoke_holdout(root: Path) -> list[dict[str, object]]:
    exclusions = root / "excluded.json"
    atomic_write_json(exclusions, [{"id": "fx_old.json#u0"}])
    rows = derive_holdout(
        FIXTURE,
        excluded_ids={"fx_old.json#u0", "fx_old.json"},
        count=3,
        seed=17,
        formal=False,
    )
    assert len(rows) == 3
    assert all(not str(row["id"]).startswith("fx_old") for row in rows)
    assert all("".join(row["segments"]) == row["full_text"] for row in rows)
    output = root / "holdout.json"
    provenance_path = root / "holdout.provenance.json"
    provenance = build_holdout(
        source=FIXTURE,
        output=output,
        provenance_output=provenance_path,
        exclusion_paths=[exclusions],
        count=3,
        seed=17,
        formal=False,
    )
    assert provenance["source"]["sha256"] == sha256_file(FIXTURE)
    assert provenance["output"]["sha256"] == sha256_file(output)
    _expect_raises(
        ValueError,
        derive_holdout,
        FIXTURE,
        excluded_ids=set(),
        count=3,
        seed=17,
        formal=True,
    )
    bad = list(rows)
    bad[0] = dict(bad[0], segments=[bad[0]["segments"][0], ""])
    _expect_raises(
        ValueError,
        validate_holdout,
        bad,
        expected_count=3,
        excluded_ids=set(),
        formal=False,
    )
    return rows


def smoke_trigger_cache_gates(root: Path) -> None:
    model_dir = root / "local-ten"
    model_dir.mkdir(parents=True)
    atomic_write_json(model_dir / "config.json", {"model_type": "fake-ten"})
    output = root / "formal-trigger-cache.json"
    clean_offline = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_TOKEN": "",
        "HUGGING_FACE_HUB_TOKEN": "",
    }
    assert assert_formal_cache_prerequisites(
        model_path=model_dir,
        output_path=output,
        git_status="",
        environment=clean_offline,
    ) == model_dir.resolve()
    _expect_raises(
        RuntimeError,
        assert_formal_cache_prerequisites,
        model_path=model_dir,
        output_path=output,
        git_status=" M trigger_cache.py",
        environment=clean_offline,
    )
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        environment = dict(clean_offline, **{name: "0"})
        _expect_raises(
            RuntimeError,
            assert_formal_cache_prerequisites,
            model_path=model_dir,
            output_path=output,
            git_status="",
            environment=environment,
        )
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        environment = dict(clean_offline, **{name: "secret"})
        _expect_raises(
            RuntimeError,
            assert_formal_cache_prerequisites,
            model_path=model_dir,
            output_path=output,
            git_status="",
            environment=environment,
        )
    _expect_raises(
        FileNotFoundError,
        assert_formal_cache_prerequisites,
        model_path=root / "missing-model",
        output_path=output,
        git_status="",
        environment=clean_offline,
    )
    atomic_write_json(output, {"sentinel": True})
    _expect_raises(
        FileExistsError,
        assert_formal_cache_prerequisites,
        model_path=model_dir,
        output_path=output,
        git_status="",
        environment=clean_offline,
    )
    for protected in OLD_RESULTS:
        _expect_raises(
            ValueError,
            assert_formal_cache_prerequisites,
            model_path=model_dir,
            output_path=protected,
            git_status="",
            environment=clean_offline,
        )

    class FakeProperties:
        name = "fake-gpu"
        total_memory = 24 * 1024**3
        major = 8
        minor = 6

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def current_device():
            return 0

        @staticmethod
        def get_device_properties(_index):
            return FakeProperties()

    class FakeCudnn:
        @staticmethod
        def version():
            return 8900

    class FakeDevice:
        index = 0

    class FakeTorch:
        __version__ = "2.fake"
        version = SimpleNamespace(cuda="12.fake")
        cuda = FakeCuda()
        backends = SimpleNamespace(cudnn=FakeCudnn())

        @staticmethod
        def device(_value):
            return FakeDevice()

    fake_trigger = SimpleNamespace(
        model=SimpleNamespace(dtype="torch.bfloat16"),
        device="cuda:0",
    )
    fake_config = SimpleNamespace(
        model_name=str(model_dir),
        system_prompt=None,
        user_template="{text}",
        positive_words=["finished"],
        negative_words=["unfinished", "wait"],
        device="cuda:0",
    )
    run_identity = capture_trigger_run_identity(
        trigger=fake_trigger,
        config=fake_config,
        model_path=model_dir,
        torch_module=FakeTorch,
        transformers_module=SimpleNamespace(__version__="4.fake"),
        git_status="",
    )
    assert run_identity["config"]["actual_model_dtype"] == "torch.bfloat16"
    assert run_identity["environment"]["torch"] == "2.fake"
    assert run_identity["environment"]["transformers"] == "4.fake"
    assert run_identity["device"]["actual"] == "cuda:0"
    assert run_identity["model"]["file_count"] == 1
    assert run_identity["model"]["files"][0]["sha256"] == sha256_file(
        model_dir / "config.json"
    )
    assert run_identity["git"]["commit"]


def smoke_trigger_cache(root: Path, rows: list[dict[str, object]]) -> tuple[Path, ReplayTrigger]:
    ten = FakeTEN()
    input_path = root / "holdout.json"
    input_sha = sha256_file(input_path)
    payload = build_trigger_cache_payload(
        rows,
        ten,
        trigger_identity={"requested": "fake-ten", "identity_hash": "ten-v1"},
        trigger_template="{text}",
        positive_words=["finished"],
        negative_words=["unfinished", "wait"],
        input_sha256=input_sha,
        created_at_utc="2026-09-01T00:00:00+00:00",
    )
    expected_calls = sum(len(row["segments"]) for row in rows)
    assert len(ten.calls) == expected_calls == payload["entry_count"]
    assert any(entry["confidence"] == 0.973456789012345 for entry in payload["entries"])
    cache_path = root / "trigger-cache.json"
    artifact = write_trigger_cache(cache_path, payload)
    assert artifact["sha256"] == sha256_file(cache_path)
    replay = ReplayTrigger(
        cache_path,
        expected_input_sha256=input_sha,
        expected_identity_hash=payload["identity_hash"],
    )
    first = rows[0]
    accumulated = ""
    replay.start(str(first["id"]))
    for index, segment in enumerate(first["segments"], start=1):
        accumulated += str(segment)
        assert replay.confidence(accumulated) == replay.confidence_for(
            str(first["id"]), index, accumulated
        )
    _expect_raises(
        ValueError,
        replay.confidence_for,
        str(first["id"]),
        1,
        "tampered text",
    )
    _expect_raises(
        ValueError,
        ReplayTrigger,
        cache_path,
        expected_input_sha256="0" * 64,
    )
    return cache_path, replay


def smoke_resume_hashes(root: Path, input_path: Path, cache_path: Path) -> None:
    config = {
        "seed": 9,
        "model_identity_hash": "model-a",
        "trigger_cache_sha256": sha256_file(cache_path),
    }
    manifest = build_manifest(
        experiment="e1e2-confirmatory",
        run_id="session",
        config=config,
        input_path=input_path,
    )
    prepare_run_directory(
        results_root=root, run_id="session", manifest=manifest, resume=False
    )
    prepare_run_directory(
        results_root=root, run_id="session", manifest=manifest, resume=True
    )
    changed_model = dict(manifest)
    changed_model["config_hash"] = "1" * 64
    _expect_raises(
        ValueError,
        prepare_run_directory,
        results_root=root,
        run_id="session",
        manifest=changed_model,
        resume=True,
    )
    changed_input = dict(manifest)
    changed_input["input"] = dict(manifest["input"], sha256="2" * 64)
    _expect_raises(
        ValueError,
        prepare_run_directory,
        results_root=root,
        run_id="session",
        manifest=changed_input,
        resume=True,
    )


def _condition_parts(condition: str) -> tuple[str, float | None]:
    if condition == SYSTEM_A:
        return SYSTEM_A, None
    if condition == NEVER_SPECULATE:
        return NEVER_SPECULATE, None
    return condition, threshold_for_condition(condition)


def smoke_grid_and_analysis(
    root: Path,
    rows: list[dict[str, object]],
    cache_path: Path,
    replay: ReplayTrigger,
) -> None:
    orders = [
        balanced_condition_order(session_index=0, dialogue_index=index, seed=23)
        for index in range(len(CONDITIONS))
    ]
    assert_balanced_orders(orders)

    backend = FakeBackend(seed=23)
    records_path = root / "records.jsonl"

    def run_cell(row, condition, position):
        accumulated = ""
        confidences = []
        for prefix_index, segment in enumerate(row["segments"], start=1):
            accumulated += str(segment)
            confidences.append(
                replay.confidence_for(str(row["id"]), prefix_index, accumulated)
            )
        backend_condition, threshold = _condition_parts(condition)
        record = backend.run_condition(
            row,
            condition=backend_condition,
            threshold=threshold,
            confidences=confidences,
            session_id="s01",
        )
        record["speculative_tokens"] = record["wasted_tokens"] + record["ready_tokens"]
        validate_ttft_record(record)
        return record

    first_records = run_missing_grid(
        rows=rows,
        session_id="s01",
        session_index=0,
        seed=23,
        records_path=records_path,
        warmup=backend.warmup,
        run_cell=run_cell,
    )
    assert len(first_records) == len(rows) * len(CONDITIONS)
    assert len([call for call in backend.calls if call["kind"] == "warmup"]) == 5
    assert all(record["trial_kind"] == "formal" for record in first_records)
    assert any(
        record["condition"] == CONFIRMATORY_CONDITION
        and record["survived"]
        and record["ttft_eff_ns"] == 0
        for record in first_records
    )

    # Complete resume is a no-op, including warmups.
    calls_before = len(backend.calls)
    run_missing_grid(
        rows=rows,
        session_id="s01",
        session_index=0,
        seed=23,
        records_path=records_path,
        warmup=backend.warmup,
        run_cell=run_cell,
    )
    assert len(backend.calls) == calls_before

    # Remove one grid cell; resume warms once then adds only that cell.
    missing = first_records[-1]
    retained = first_records[:-1]
    records_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in retained),
        encoding="utf-8",
    )
    calls_before = len(backend.calls)
    resumed = run_missing_grid(
        rows=rows,
        session_id="s01",
        session_index=0,
        seed=23,
        records_path=records_path,
        warmup=backend.warmup,
        run_cell=run_cell,
    )
    new_calls = backend.calls[calls_before:]
    assert len(resumed) == len(first_records)
    assert len([call for call in new_calls if call["kind"] == "warmup"]) == 5
    formal_calls = [call for call in new_calls if call["kind"] == "formal"]
    assert len(formal_calls) == 1
    assert formal_calls[0]["id"] == missing["id"]

    # Duplicate and truncated JSONL fail closed.
    duplicate_path = root / "duplicate.jsonl"
    duplicate_path.write_text(
        json.dumps(resumed[0]) + "\n" + json.dumps(resumed[0]) + "\n",
        encoding="utf-8",
    )
    _expect_raises(ValueError, validate_record_file, duplicate_path)
    truncated_path = root / "truncated.jsonl"
    truncated_path.write_text('{"session_id":"s01"', encoding="utf-8")
    _expect_raises(ValueError, validate_record_file, truncated_path)

    waste = pooled_waste(
        [
            {"wasted_tokens": 3, "speculative_tokens": 10},
            {"wasted_tokens": 1, "speculative_tokens": 10},
        ]
    )
    assert waste["ratio"] == 0.2
    paired = paired_e1_difference(resumed)
    assert paired["n_pairs"] == len(rows)

    # Duplicate this model-free session under independent IDs for a two-level bootstrap.
    bootstrap_records = []
    for session_id in ("s01", "s02", "s03"):
        for record in resumed:
            copied = dict(record, session_id=session_id)
            bootstrap_records.append(copied)
    bootstrap = hierarchical_bootstrap(
        bootstrap_records, repeats=50, seed=1234
    )
    provenance = analysis_provenance(
        records_paths=[records_path],
        manifest_paths=[root / "manifest.json"],
        trigger_cache_path=cache_path,
        input_path=root / "holdout.json",
        analyzer_path=Path(__file__),
        bootstrap=bootstrap["provenance"],
    )
    assert bootstrap["provenance"]["method"].startswith("two-level")
    assert bootstrap["provenance"]["session_count"] == 3
    assert provenance["sources"]["records"][0]["sha256"] == sha256_file(records_path)
    assert provenance["provenance_hash"]


def smoke_review_contracts(root: Path) -> None:
    model_dir = root / "strong-model"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text('{"v":1}\n', encoding="utf-8")
    (model_dir / "weights.bin").write_bytes(b"weights-v1")
    identity_v1 = strong_model_identity(model_dir)
    (model_dir / "weights.bin").write_bytes(b"weights-v2")
    identity_v2 = strong_model_identity(model_dir)
    assert identity_v1["content_identity_hash"] != identity_v2["content_identity_hash"]

    backend = FakeBackend(seed=77)
    row = {
        "id": "review",
        "full_text": "Tell me more please.",
        "segments": ["Tell me", " more please."],
    }
    survived = backend.run_condition(
        row,
        condition=CONFIRMATORY_CONDITION,
        threshold=0.92,
        confidences=[0.1, 0.99],
        session_id="s01",
    )
    assert survived["ttft_eff_ns"] == 0
    assert survived["first_token_ready_ns"] == survived["candidate_first_token_ns"]
    assert survived["arrival_to_first_token_ready_ns"] > 0
    assert survived["oracle_preaccept_processing_ns"] > 0
    no_candidate = backend.run_condition(
        row,
        condition=NEVER_SPECULATE,
        threshold=None,
        confidences=[0.1, 0.99],
        session_id="s01",
    )
    assert no_candidate["first_token_ready_ns"] == no_candidate["first_deliverable_token_ns"]

    normalized = []
    for condition, source in (
        (CONFIRMATORY_CONDITION, survived),
        (NEVER_SPECULATE, no_candidate),
    ):
        record = dict(source)
        record.update(
            {
                "session_id": "s01",
                "session_index": 0,
                "dialogue_id": "review",
                "dialogue_index": 0,
                "condition": condition,
                "condition_ordinal": 0,
                "condition_order": list(CONDITIONS),
                "condition_order_seed": 20260901,
                "consumer_delivery_latency_ns": source["consumer_delivery_ns"]
                - source["endpoint_accept_ns"],
                "speculative_tokens": source["wasted_tokens"] + source["ready_tokens"],
                "waste_denominator_tokens": source["wasted_tokens"] + source["final_tokens"],
            }
        )
        normalized.append(record)
    timing = validate_timing(normalized)
    assert timing["ok"], timing["errors"]
    summary = summarize_condition(normalized)
    assert "arrival_to_first_token_ready_ms_primary" in summary
    assert "ttft_eff_ms_oracle_latency_lower_bound" in summary
    assert summary["pooled_token_waste_ratio"] == (
        sum(row["wasted_tokens"] for row in normalized)
        / sum(row["wasted_tokens"] + row["final_tokens"] for row in normalized)
    )
    _expect_raises(
        ValidationError,
        validate_campaign,
        root / "missing-campaign",
        expected_sessions=1,
        expected_dialogues=1,
        formal=True,
    )

    records_path = root / "resume-records.jsonl"
    expected = {
        "campaign_id": "c",
        "session_id": "s",
        "session_index": 0,
        "process_start_id": "p",
    }
    records_path.write_text(
        json.dumps(
            {
                **expected,
                "dialogue_id": "d",
                "condition": SYSTEM_A,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _validate_existing_jsonl(records_path, expected_identity=expected)
    tampered = dict(expected, process_start_id="other")
    _expect_raises(
        ValueError,
        _validate_existing_jsonl,
        records_path,
        expected_identity=tampered,
    )


def smoke_public_runner(root: Path, input_path: Path, cache_path: Path) -> None:
    backend = FakeBackend(seed=31)
    args = SimpleNamespace(
        campaign_id="smoke-campaign",
        session_id="s01",
        session_index=0,
        input=input_path,
        trigger_cache=cache_path,
        trigger_identity_hash=None,
        campaign_manifest=None,
        results_root=root,
        runtime="fake",
        model=None,
        device="cpu",
        seed=31,
        order_seed=31,
        max_new_tokens=32,
        spec_chunk=12,
        limit=None,
        warmup_repeats=3,
        formal=False,
        resume=False,
    )
    session_dir = run_session(args, backend)
    records = validate_record_file(
        session_dir / "records.jsonl",
        key_fields=("session_id", "dialogue_id", "condition"),
    )
    assert len(records) == 3 * len(CONDITIONS)
    assert len(json.loads((session_dir / "summary.json").read_text(encoding="utf-8"))["conditions"]) == len(CONDITIONS)
    assert len([call for call in backend.calls if call["kind"] == "warmup"]) == 15
    assert not any("trial_kind" in record and record["trial_kind"] != "formal" for record in records)
    assert any(
        record["condition"] == CONFIRMATORY_CONDITION
        and record["survived"]
        and record["ttft_eff_ns"] == 0
        for record in records
    )


def main() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    before = {path: sha256_file(path) for path in OLD_RESULTS if path.exists()}
    root = Path(tempfile.mkdtemp(prefix="e1e2-confirmatory-smoke-"))
    try:
        smoke_holdout_source_gates(root / "holdout-gates")
        rows = smoke_holdout(root)
        smoke_trigger_cache_gates(root / "trigger-gates")
        cache_path, replay = smoke_trigger_cache(root, rows)
        input_path = root / "holdout.json"
        smoke_resume_hashes(root / "resume", input_path, cache_path)
        smoke_review_contracts(root / "review-contracts")
        smoke_public_runner(root / "runner", input_path, cache_path)
        # Provenance needs a manifest path independent of resume directory.
        atomic_write_json(root / "manifest.json", {"smoke": True})
        smoke_grid_and_analysis(root, rows, cache_path, replay)
        after = {path: sha256_file(path) for path in before}
        assert before == after, "Archived E1/E2 results changed during smoke"
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "models_loaded": False,
                    "network_used": False,
                    "root": str(root),
                },
                ensure_ascii=False,
            )
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
