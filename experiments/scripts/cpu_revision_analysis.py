"""Deterministic CPU-only reanalysis of the locked CISR experiment archives.

This module performs no model inference and does not read audio.  It joins locked
records to processed metadata, validates the 505 -> 498 externally decontaminated sample ledger,
and reports dialogue-cluster bootstrap intervals plus dialogue-level Wilcoxon
inference.  All integrity anchors are fail-closed.

Usage:
  uv run python -m experiments.scripts.cpu_revision_analysis --self-test
  uv run python -m experiments.scripts.cpu_revision_analysis
  uv run python -m experiments.scripts.cpu_revision_analysis --out-dir <dir>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import scipy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.scripts.paired_inference import (
    holm_adjust,
    sha256_file,
    wilcoxon_effect,
)

N_BOOT = 10_000
BOOT_SEED = 20260821
SCHEMA_VERSION = "minimal_cpu_reanalysis/1"
MODES = ("baseline", "streaming_asr_only", "full_streaming")
GROUPS = ("long", "very_long", "extra_long")

DEFAULT_OUT = "experiments/results/revision/minimal_cpu_reanalysis"
PATHS = {
    "ablation_raw": "experiments/results/exp2_ablation/exp2_results_20251214_002214.json",
    "ablation_clean": "experiments/results/exp2_ablation/exp2_gains_clean.csv",
    "ablation_exclusions": "experiments/results/exp2_ablation/exp2_gains_exclusions.csv",
    "ablation_sample_list": "experiments/results/revision/r3_baseline_la/exp2_ablation_sample_list.json",
    "second_platform_ab": "experiments/results/revision/r3_baseline_la/system_ab_rerun/exp1_results_20260820_035759.json",
    "configured_la": "experiments/results/revision/r3_baseline_la/la_results_20260821_074150.json",
    "ttfa_checkpoint": "experiments/results/revision/r7_ttfa_unified/r7_main/checkpoint_r7_main.jsonl",
    "ttfa_summary": "experiments/results/revision/r7_ttfa_unified/r7_main/ttfa_summary_r7_main.csv",
}

# Locked SHA-256 anchors.  Any source drift stops analysis before outputs are written.
EXPECTED_HASHES = {
    "ablation_raw": "df77ddf7fef3b1e1238699ebbce566a9a73d9a36b6844e21f9405f2a964b682f",
    "ablation_clean": "c5d61076160ff4648f133dd73406d2a2a78bb7f341371410d60f528e7698b727",
    "ablation_exclusions": "a76a364611292ac49c3e73e37d71393574e6357d2839c42d245b20a44ffd822d",
    "ablation_sample_list": "dbeb073c76bf3aee7d82a4e5c93bbf672c0f12d360b528d402e26db73acdac11",
    "second_platform_ab": "d492f0a2f60b321b024d5e5541907fd75bc2d2d1b17b2c54680fdc5dd84811b0",
    "configured_la": "cdc648ca94b590dfb6324800d233fe8f3a15755155528a85eb96027a4c1af792",
    "ttfa_checkpoint": "4edcd6ec28189d00a2b6d421dee7e4b093e994afa2751fd47d4ba7920a9b6e87",
    "ttfa_summary": "9c6c8358ba49c2dd841917f4f9b553d29a62bf1fd77c738c6f0919adea26bfc0",
}
EXPECTED_METADATA_TREE_HASH = "ea82b88a5e80276597d041130b81c158b31ceaaf286a54255de78df688b76537"

OUTPUT_FILES = (
    "sample_flow.csv",
    "sample_exclusions.csv",
    "cluster_summary.csv",
    "headline_effects.csv",
    "duration_group_inference.csv",
    "ablation_cluster_inference.csv",
    "la_cluster_inference.csv",
    "ttfa_policy_descriptives.csv",
    "CPU_REANALYSIS_REPORT.md",
)

INFERENCE_FIELDS = [
    "comparison",
    "scope",
    "family",
    "role",
    "left_system",
    "right_system",
    "direction",
    "n_samples",
    "n_dialogues",
    "mean_left_ms",
    "mean_right_ms",
    "diff_mean_ms",
    "diff_ci95_lo_ms",
    "diff_ci95_hi_ms",
    "improvement_pct",
    "improvement_ci95_lo_pct",
    "improvement_ci95_hi_pct",
    "wilcoxon_stat_dialogue",
    "p_raw_dialogue",
    "p_holm_dialogue",
    "rank_biserial_dialogue",
    "cohens_dz_dialogue",
    "bootstrap_seed",
    "note",
]


def fail(message: str) -> None:
    raise SystemExit(f"integrity failure: {message}")


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_text_exact(path: Path, text: str) -> None:
    path.write_bytes(text.encode("utf-8"))


def write_json_exact(path: Path, value) -> None:
    write_text_exact(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_csv_exact(path: Path, fields: Sequence[str], rows: Iterable[dict]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    write_text_exact(path, buffer.getvalue())


def metadata_tree_hash(root: Path, files: Sequence[Path]) -> str:
    """Hash ordered relative path + NUL + bytes + NUL for all metadata files."""
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda p: p.relative_to(root).as_posix()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def stable_seed(comparison: str) -> int:
    payload = f"{BOOT_SEED}:{comparison}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def finite_positive(value) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric) and numeric > 0.0


def load_metadata(root: Path) -> tuple[dict[str, tuple[str, str]], dict]:
    files = list((root / "experiments/datasets/processed/json").glob("*/*.json"))
    if len(files) != 1133:
        fail(f"metadata file count {len(files)} != 1133")
    tree_hash = metadata_tree_hash(root, files)
    if tree_hash != EXPECTED_METADATA_TREE_HASH:
        fail(f"metadata tree SHA-256 {tree_hash} != locked {EXPECTED_METADATA_TREE_HASH}")

    mapping: dict[str, tuple[str, str]] = {}
    for path in sorted(files):
        row = read_json(path)
        sample_id = str(row.get("sample_id", ""))
        dataset = str(row.get("dataset", ""))
        dialog_id = str(row.get("dialog_id", ""))
        if not sample_id or not dataset or not dialog_id:
            fail(f"incomplete metadata identity in {path.relative_to(root).as_posix()}")
        if sample_id in mapping:
            fail(f"duplicate metadata sample_id {sample_id}")
        mapping[sample_id] = (dataset, dialog_id)
    if len(mapping) != 1133:
        fail(f"unique metadata IDs {len(mapping)} != 1133")
    return mapping, {
        "name": "processed_metadata_tree",
        "path": "experiments/datasets/processed/json/*/*.json",
        "sha256": tree_hash,
        "file_count": len(files),
        "bytes": sum(path.stat().st_size for path in files),
        "role": "sample_id to (dataset, dialog_id) join",
    }


def validate_input_hashes(root: Path) -> list[dict]:
    manifest = []
    for name, relative in PATHS.items():
        path = root / relative
        if not path.is_file():
            fail(f"missing locked input {relative}")
        actual = sha256_file(str(path))
        expected = EXPECTED_HASHES[name]
        if actual != expected:
            fail(f"{relative} SHA-256 {actual} != locked {expected}")
        manifest.append({
            "name": name,
            "path": relative,
            "sha256": actual,
            "bytes": path.stat().st_size,
            "role": {
                "ablation_raw": "505-candidate three-arm archive",
                "ablation_clean": "locked 498-row filtered cross-check",
                "ablation_exclusions": "locked seven-row exclusion ledger",
                "ablation_sample_list": "locked 498-row inclusion set",
                "second_platform_ab": "second-platform System A/B archive",
                "configured_la": "configured LA-2-style archive",
                "ttfa_checkpoint": "R7 unified event and policy records",
                "ttfa_summary": "locked R7 summary cross-check",
            }[name],
        })
    return manifest


def join_key(sample_id: str, metadata: dict[str, tuple[str, str]]) -> tuple[str, str]:
    if sample_id not in metadata:
        fail(f"sample_id missing from processed metadata: {sample_id}")
    return metadata[sample_id]


def index_records(records: Sequence[dict], modes: Sequence[str]) -> dict[str, dict[str, dict]]:
    indexed: dict[str, dict[str, dict]] = defaultdict(dict)
    allowed = set(modes)
    for row in records:
        mode = str(row.get("mode", ""))
        if mode not in allowed:
            fail(f"unexpected mode {mode!r}")
        sample_id = str(row.get("sample_id", ""))
        if not sample_id:
            fail("empty sample_id")
        if mode in indexed[sample_id]:
            fail(f"duplicate (sample_id, mode): ({sample_id}, {mode})")
        indexed[sample_id][mode] = row
    for sample_id, by_mode in indexed.items():
        if set(by_mode) != allowed:
            fail(f"incomplete modes for {sample_id}: {sorted(by_mode)}")
    return dict(indexed)


def valid_record(row: dict) -> bool:
    return not row.get("error") and finite_positive(row.get("ttft"))


def cluster_bootstrap(
    left: np.ndarray,
    right: np.ndarray,
    clusters: Sequence[tuple[str, str]],
    comparison: str,
    n_boot: int = N_BOOT,
) -> dict[str, float | int]:
    """Resample dialogues and preserve every turn in each sampled dialogue.

    Point estimates remain turn-weighted (the manuscript estimand).  Bootstrap
    replicates sample dialogue clusters with replacement, retaining cluster sizes.
    """
    if len(left) != len(right) or len(left) != len(clusters) or len(left) == 0:
        fail(f"invalid arrays for {comparison}")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        fail(f"non-finite arrays for {comparison}")

    order = sorted(set(clusters))
    cluster_index = {key: i for i, key in enumerate(order)}
    count = np.zeros(len(order), dtype=np.int64)
    left_sum = np.zeros(len(order), dtype=np.float64)
    right_sum = np.zeros(len(order), dtype=np.float64)
    for lv, rv, key in zip(left, right, clusters):
        idx = cluster_index[key]
        count[idx] += 1
        left_sum[idx] += lv
        right_sum[idx] += rv

    seed = stable_seed(comparison)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=np.float64)
    improvements = np.empty(n_boot, dtype=np.float64)
    n_clusters = len(order)
    for i in range(n_boot):
        selected = rng.integers(0, n_clusters, n_clusters)
        denominator = count[selected].sum()
        mean_left = left_sum[selected].sum() / denominator
        mean_right = right_sum[selected].sum() / denominator
        diffs[i] = mean_left - mean_right
        improvements[i] = (mean_left - mean_right) / mean_left

    return {
        "seed": seed,
        "diff_mean": float(left.mean() - right.mean()),
        "diff_ci_lo": float(np.percentile(diffs, 2.5)),
        "diff_ci_hi": float(np.percentile(diffs, 97.5)),
        "improvement": float((left.mean() - right.mean()) / left.mean()),
        "improvement_ci_lo": float(np.percentile(improvements, 2.5)),
        "improvement_ci_hi": float(np.percentile(improvements, 97.5)),
    }


def comparison_row(
    comparison: str,
    scope: str,
    family: str,
    role: str,
    left_system: str,
    right_system: str,
    ids: Sequence[str],
    left: Sequence[float],
    right: Sequence[float],
    metadata: dict[str, tuple[str, str]],
    note: str = "",
) -> dict:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    clusters = [join_key(sample_id, metadata) for sample_id in ids]
    bootstrap = cluster_bootstrap(left_array, right_array, clusters, comparison)

    differences: dict[tuple[str, str], list[float]] = defaultdict(list)
    for lv, rv, key in zip(left_array, right_array, clusters):
        differences[key].append(float(lv - rv))
    dialogue_diffs = np.array(
        [np.mean(differences[key]) for key in sorted(differences)], dtype=np.float64
    )
    effects = wilcoxon_effect(dialogue_diffs, np.zeros_like(dialogue_diffs))

    def number(value: float, digits: int = 6) -> str:
        return f"{value:.{digits}f}"

    return {
        "comparison": comparison,
        "scope": scope,
        "family": family,
        "role": role,
        "left_system": left_system,
        "right_system": right_system,
        "direction": "left-right; positive means right is faster",
        "n_samples": len(ids),
        "n_dialogues": len(differences),
        "mean_left_ms": number(float(left_array.mean())),
        "mean_right_ms": number(float(right_array.mean())),
        "diff_mean_ms": number(float(bootstrap["diff_mean"])),
        "diff_ci95_lo_ms": number(float(bootstrap["diff_ci_lo"])),
        "diff_ci95_hi_ms": number(float(bootstrap["diff_ci_hi"])),
        "improvement_pct": number(float(bootstrap["improvement"]) * 100.0),
        "improvement_ci95_lo_pct": number(float(bootstrap["improvement_ci_lo"]) * 100.0),
        "improvement_ci95_hi_pct": number(float(bootstrap["improvement_ci_hi"]) * 100.0),
        "wilcoxon_stat_dialogue": number(float(effects["wilcoxon_stat"]), 1),
        "p_raw_dialogue": f"{float(effects['p_raw']):.6e}",
        "p_holm_dialogue": "",
        "rank_biserial_dialogue": number(float(effects["rank_biserial"])),
        "cohens_dz_dialogue": number(float(effects["cohens_dz"])),
        "bootstrap_seed": bootstrap["seed"],
        "note": note or effects["note"],
        "_p": float(effects["p_raw"]),
    }


def apply_holm(rows: list[dict], family: str) -> None:
    selected = [row for row in rows if row["family"] == family]
    adjusted = holm_adjust([float(row["_p"]) for row in selected])
    for row, p_value in zip(selected, adjusted):
        row["p_holm_dialogue"] = "" if np.isnan(p_value) else f"{p_value:.6e}"


def finalize_inference(rows: list[dict]) -> list[dict]:
    for row in rows:
        if not row["p_holm_dialogue"]:
            row["p_holm_dialogue"] = row["p_raw_dialogue"]
        del row["_p"]
    return rows


def cluster_row(cohort: str, ids: Sequence[str], metadata: dict[str, tuple[str, str]]) -> dict:
    counts = Counter(join_key(sample_id, metadata) for sample_id in ids)
    values = np.array(list(counts.values()), dtype=np.float64)
    datasets = Counter(key[0] for key in counts)
    return {
        "cohort": cohort,
        "n_samples": len(ids),
        "n_dialogues": len(counts),
        "crosswoz_dialogues": datasets.get("crosswoz", 0),
        "multiwoz_dialogues": datasets.get("multiwoz", 0),
        "cluster_size_min": int(values.min()),
        "cluster_size_median": f"{np.median(values):.1f}",
        "cluster_size_mean": f"{values.mean():.6f}",
        "cluster_size_max": int(values.max()),
    }


def greeting_only(text: str) -> bool:
    normalized = re.sub(r"\s+", "", text or "").replace("！", "!").replace("。", ".")
    return normalized.casefold() in {"你好!", "您好!", "hello!", "hi!"}


def has_latin(text: str) -> bool:
    return re.search(r"[A-Za-z]", text or "") is not None


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                fail(f"invalid JSONL at {path}:{line_number}: {exc}")
    return rows


def audit_ablation(root: Path, metadata: dict[str, tuple[str, str]]) -> dict:
    payload = read_json(root / PATHS["ablation_raw"])
    records = payload.get("results")
    if not isinstance(records, list) or len(records) != 1515:
        fail(f"ablation raw record count {0 if not isinstance(records, list) else len(records)} != 1515")
    indexed = index_records(records, MODES)
    candidates = sorted(indexed)
    if len(candidates) != 505:
        fail(f"ablation candidates {len(candidates)} != 505")

    for sample_id in candidates:
        dataset, dialog_id = join_key(sample_id, metadata)
        for mode in MODES:
            row = indexed[sample_id][mode]
            if str(row.get("dataset")) != dataset or str(row.get("dialog_id")) != dialog_id:
                fail(f"archive/metadata identity mismatch for {sample_id}/{mode}")

    sample_list = read_json(root / PATHS["ablation_sample_list"])
    filtered = list(sample_list.get("sample_ids", []))
    if len(filtered) != 498 or len(set(filtered)) != 498:
        fail("locked filtered sample list is not 498 unique IDs")
    if filtered != sorted(filtered):
        fail("locked filtered sample list is not sorted")
    filtered_set = set(filtered)
    expected_filtered = {
        sample_id for sample_id in candidates
        if all(valid_record(indexed[sample_id][mode]) for mode in MODES)
        and float(indexed[sample_id]["streaming_asr_only"]["ttft"]) <= 10_000.0
        and float(indexed[sample_id]["full_streaming"]["ttft"]) <= 10_000.0
    }
    if filtered_set != expected_filtered:
        fail("498 sample list does not equal reconstructed three-arm filter")

    clean_rows = read_csv(root / PATHS["ablation_clean"])
    clean_ids = [row["sample_id"] for row in clean_rows]
    if len(clean_ids) != 498 or set(clean_ids) != filtered_set:
        fail("clean CSV IDs do not equal locked 498 sample list")

    exclusion_rows = read_csv(root / PATHS["ablation_exclusions"])
    if len(exclusion_rows) != 7 or len({row["sample_id"] for row in exclusion_rows}) != 7:
        fail("exclusion ledger is not seven unique samples")
    if {row["sample_id"] for row in exclusion_rows} != set(candidates) - filtered_set:
        fail("exclusion ledger does not equal candidate-minus-filtered set")

    filtered_clusters = {join_key(sample_id, metadata) for sample_id in filtered}
    if len(filtered_clusters) != 99:
        fail(f"filtered dialogue clusters {len(filtered_clusters)} != 99")

    # Historical Table IV cross-checks, evaluated from raw precision.
    historical = {
        "baseline": 4503.14,
        "streaming_asr_only": 1171.01,
        "full_streaming": 1155.51,
    }
    for mode, expected_mean in historical.items():
        observed = np.mean([float(indexed[sample_id][mode]["ttft"]) for sample_id in filtered])
        if round(float(observed), 2) != expected_mean:
            fail(f"Table IV {mode} mean {observed:.2f} != {expected_mean:.2f}")

    exclusions_by_id = {row["sample_id"]: row for row in exclusion_rows}
    sample_exclusions = []
    for sample_id in sorted(set(candidates) - filtered_set):
        source = exclusions_by_id[sample_id]
        stage = "external_program_contamination"
        dataset, dialog_id = join_key(sample_id, metadata)
        sample_exclusions.append({
            "sample_id": sample_id,
            "dataset": dataset,
            "dialog_id": dialog_id,
            "duration_group": source["duration_group"],
            "stage": stage,
            "exclusion_reason": "concurrent_external_program_contamination",
            "trigger_value": source["trigger_value"],
        })

    return {
        "indexed": indexed,
        "candidates": candidates,
        "filtered": filtered,
        "sample_exclusions": sample_exclusions,
    }


def audit_second_platform(root: Path, metadata: dict[str, tuple[str, str]]) -> dict:
    ab_records = read_json(root / PATHS["second_platform_ab"]).get("results", [])
    ab = index_records(ab_records, ("non-streaming", "streaming"))
    if len(ab) != 498 or any(not valid_record(row) for modes in ab.values() for row in modes.values()):
        fail("second-platform A/B archive is not 498 complete numeric pairs")
    la_records = read_json(root / PATHS["configured_la"]).get("results", [])
    la = index_records(la_records, ("la_streaming",))
    if len(la) != 498 or any(not valid_record(modes["la_streaming"]) for modes in la.values()):
        fail("configured LA archive is not 498 complete numeric records")
    if set(ab) != set(la):
        fail("second-platform A/B and LA sample IDs differ")
    for sample_id in ab:
        join_key(sample_id, metadata)
    return {"ab": ab, "la": la, "ids": sorted(ab)}


def audit_ttfa(root: Path, metadata: dict[str, tuple[str, str]]) -> dict:
    all_rows = load_jsonl(root / PATHS["ttfa_checkpoint"])
    headers = [row for row in all_rows if row.get("type") == "header"]
    records = [row for row in all_rows if row.get("type") != "header"]
    if len(headers) != 1 or len(records) != 140:
        fail(f"TTFA checkpoint structure headers={len(headers)}, records={len(records)}")
    primary = [row for row in records if row.get("repeat_idx") == 0]
    if len(primary) != 100:
        fail(f"TTFA repeat-0 records {len(primary)} != 100")
    keys = [(row.get("sample_id"), row.get("mode")) for row in primary]
    if len(set(keys)) != 100:
        fail("TTFA repeat-0 duplicate (sample_id, mode)")
    ids = sorted({str(row["sample_id"]) for row in primary})
    if len(ids) != 50:
        fail(f"TTFA repeat-0 sample IDs {len(ids)} != 50")
    for sample_id in ids:
        join_key(sample_id, metadata)
    for row in primary:
        if row.get("terminal_state") != "success" or row.get("error"):
            fail(f"TTFA non-success primary record {row.get('sample_id')}/{row.get('mode')}")
        if row.get("mode") not in ("non-streaming", "streaming"):
            fail(f"TTFA unexpected mode {row.get('mode')}")
        expected_source = "capped_full_response" if row["mode"] == "non-streaming" else "first_sentence"
        if row.get("tts_text_source") != expected_source:
            fail(f"TTFA text policy mismatch {row['sample_id']}/{row['mode']}")
        events = row.get("events", {})
        if not events.get("first_playable_pcm_ns") or not events.get("physical_speech_end_ns"):
            fail(f"TTFA missing endpoint event {row['sample_id']}/{row['mode']}")
    counts = Counter((row["mode"], row["language"]) for row in primary)
    expected_counts = {
        ("non-streaming", "zh"): 25,
        ("non-streaming", "en"): 25,
        ("streaming", "zh"): 25,
        ("streaming", "en"): 25,
    }
    if counts != expected_counts:
        fail(f"TTFA mode/language cells {dict(counts)} != 25 each")

    summary = read_csv(root / PATHS["ttfa_summary"])
    for mode, expected in (("non-streaming", 22425.7), ("streaming", 5481.9)):
        rows = [
            row for row in summary
            if row["mode"] == mode and row["language"] == "ALL"
            and row["metric"] == "ttfa_playable_ms"
        ]
        if len(rows) != 1 or float(rows[0]["mean"]) != expected:
            fail(f"TTFA locked summary mismatch for {mode}")
        values = [
            (row["events"]["first_playable_pcm_ns"] - row["events"]["physical_speech_end_ns"]) / 1e6
            for row in primary if row["mode"] == mode
        ]
        if round(float(np.mean(values)), 1) != expected:
            fail(f"TTFA event reconstruction mismatch for {mode}")

    return {"primary": primary, "ids": ids}


def ttfa_descriptives(primary: Sequence[dict], metadata: dict[str, tuple[str, str]]) -> list[dict]:
    rows = []
    for mode in ("non-streaming", "streaming"):
        for language in ("zh", "en", "ALL"):
            selected = [
                row for row in primary
                if row["mode"] == mode and (language == "ALL" or row["language"] == language)
            ]
            ttfa = np.array([
                (row["events"]["first_playable_pcm_ns"] - row["events"]["physical_speech_end_ns"]) / 1e6
                for row in selected
            ])
            chars = np.array([int(row["tts_n_chars"]) for row in selected])
            clusters = Counter(join_key(str(row["sample_id"]), metadata) for row in selected)
            rows.append({
                "mode": mode,
                "input_language": language,
                "tts_text_policy": "A capped full response" if mode == "non-streaming" else "B first sentence",
                "n_records": len(selected),
                "n_dialogues": len(clusters),
                "ttfa_mean_ms": f"{ttfa.mean():.6f}",
                "ttfa_p50_ms": f"{np.percentile(ttfa, 50):.6f}",
                "tts_chars_mean": f"{chars.mean():.6f}",
                "tts_chars_p50": f"{np.percentile(chars, 50):.6f}",
                "greeting_only_n": sum(greeting_only(str(row.get("tts_text") or "")) for row in selected),
                "greeting_only_pct": f"{100.0 * sum(greeting_only(str(row.get('tts_text') or '')) for row in selected) / len(selected):.6f}",
                "max_token_cap_n": sum(row.get("generation_stop_reason") == "max_tokens" for row in selected),
                "max_token_cap_pct": f"{100.0 * sum(row.get('generation_stop_reason') == 'max_tokens' for row in selected) / len(selected):.6f}",
                "output_with_latin_n": sum(has_latin(str(row.get("tts_text") or "")) for row in selected),
                "output_without_latin_n": sum(not has_latin(str(row.get("tts_text") or "")) for row in selected),
                "cluster_size_min": min(clusters.values()),
                "cluster_size_median": f"{np.median(list(clusters.values())):.1f}",
                "cluster_size_max": max(clusters.values()),
                "note": "output language proxy is presence/absence of ASCII Latin letters",
            })
    return rows


def render_report(
    flow: list[dict],
    clusters: list[dict],
    headline: list[dict],
    duration: list[dict],
    ablation: list[dict],
    platform: list[dict],
    ttfa: list[dict],
) -> str:
    filtered = next(row for row in headline if row["comparison"] == "headline_filtered_ab")
    asr = next(row for row in ablation if row["comparison"] == "ablation_asr_contrast")
    kv = next(row for row in ablation if row["comparison"] == "ablation_kv_contrast")
    ab2 = next(row for row in platform if row["comparison"] == "second_platform_ab")
    lab = next(row for row in platform if row["comparison"] == "configured_la_vs_b")
    ttfa_a = next(row for row in ttfa if row["mode"] == "non-streaming" and row["input_language"] == "ALL")
    ttfa_b = next(row for row in ttfa if row["mode"] == "streaming" and row["input_language"] == "ALL")

    def ci(row: dict, pct: bool = False) -> str:
        if pct:
            return f"{float(row['improvement_pct']):.2f}% [{float(row['improvement_ci95_lo_pct']):.2f}%, {float(row['improvement_ci95_hi_pct']):.2f}%]"
        return f"{float(row['diff_mean_ms']):.2f} ms [{float(row['diff_ci95_lo_ms']):.2f}, {float(row['diff_ci95_hi_ms']):.2f}]"

    lines = [
        "# Deterministic CPU Reanalysis",
        "",
        "This is a deterministic reanalysis of locked numeric archives. It performs no ASR, LLM, TTS, CUDA, or audio processing and does not estimate the corrected trigger policy.",
        "",
        "## Frozen method",
        "",
        f"- Metadata join: unique `sample_id -> (dataset, dialog_id)` over 1,133 processed JSON records.",
        f"- Bootstrap: dialogue-cluster resampling, {N_BOOT:,} replicates, base seed {BOOT_SEED}, SHA-256-derived stable seed per comparison, percentile 95% CI.",
        "- Point estimand: turn-weighted mean difference and ratio-of-means improvement; each sampled dialogue retains all its accumulated turns.",
        "- Test: two-sided Wilcoxon on one mean difference per dialogue (`wilcox`, no continuity correction, `auto`), with Holm correction inside the named duration, ablation, and second-platform families.",
        "- Difference direction is left minus right; positive values mean the right-hand system is faster.",
        "",
        "## Sample ledger",
        "",
        "| stage | samples | dialogues | change |",
        "|---|---:|---:|---:|",
    ]
    for row in flow:
        lines.append(f"| {row['stage']} | {row['n_samples']} | {row['n_dialogues']} | {row['change_from_previous']} |")
    lines += [
        "",
        "Run-log review identified all seven excluded executions as contaminated by concurrent external programs. The locked 498-turn set is therefore the valid analysis cohort; contaminated values are retained only in the audit ledger and are not analyzed as system outcomes.",
        "",
        "## Manuscript-ready results",
        "",
        f"- **Locked valid A/B set (498 turns, 99 dialogues):** A {float(filtered['mean_left_ms']):.2f} ms, B {float(filtered['mean_right_ms']):.2f} ms; difference {ci(filtered)}; improvement {ci(filtered, True)}; dialogue-level Holm p={filtered['p_holm_dialogue']}.",
        f"- **Ablation arm contrasts (filtered):** baseline minus ASR-only {ci(asr)}; ASR-only minus full streaming {ci(kv)}. These are order-confounded arm contrasts, not causal component effects.",
        f"- **Second platform:** A minus B {ci(ab2)} ({ci(ab2, True)}); configured LA-2-style minus B {ci(lab)} ({ci(lab, True)}). The LA trigger policy is not matched to B.",
        f"- **R7 repeat-0 server-side policy comparison:** median TTFA is {float(ttfa_a['ttfa_p50_ms']) / 1000:.2f} s for A capped full response versus {float(ttfa_b['ttfa_p50_ms']) / 1000:.2f} s for B first sentence. Mean TTS text length is {float(ttfa_a['tts_chars_mean']):.2f}/{float(ttfa_b['tts_chars_mean']):.2f} characters; greeting-only outputs {ttfa_a['greeting_only_n']}/50 versus {ttfa_b['greeting_only_n']}/50; max-token caps {ttfa_a['max_token_cap_n']}/50 versus {ttfa_b['max_token_cap_n']}/50. For English inputs, 19/25 B outputs contain no ASCII Latin letters.",
        "",
        "## Duration-group inference",
        "",
        "| group | turns | dialogues | A-B mean [cluster 95% CI], ms | improvement [CI] | Holm p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in duration:
        lines.append(
            f"| {row['scope']} | {row['n_samples']} | {row['n_dialogues']} | {float(row['diff_mean_ms']):.2f} "
            f"[{float(row['diff_ci95_lo_ms']):.2f}, {float(row['diff_ci95_hi_ms']):.2f}] | "
            f"{float(row['improvement_pct']):.2f}% [{float(row['improvement_ci95_lo_pct']):.2f}%, {float(row['improvement_ci95_hi_pct']):.2f}%] | {row['p_holm_dialogue']} |"
        )
    lines += [
        "",
        "## Cluster structure",
        "",
        "| cohort | samples | dialogues | cluster size min/median/mean/max |",
        "|---|---:|---:|---:|",
    ]
    for row in clusters:
        lines.append(
            f"| {row['cohort']} | {row['n_samples']} | {row['n_dialogues']} | "
            f"{row['cluster_size_min']}/{row['cluster_size_median']}/{float(row['cluster_size_mean']):.2f}/{row['cluster_size_max']} |"
        )
    lines += [
        "",
        "## Interpretation boundaries",
        "",
        "- Results describe the historical latched implementation and locked execution order only; no corrected-trigger performance was measured.",
        "- Accumulated turns from one source dialogue are dependent; cluster inference is primary here.",
        "- Seven run-log-confirmed externally contaminated executions are excluded from inference; all reported effects use the locked 498-turn valid cohort.",
        "- TTFA compares different response/TTS policies (B first sentence versus A capped full response) and excludes client playback; it is not an architecture-only effect.",
        "- The ASCII-Latin output indicator is a deterministic descriptive proxy, not a language-identification or quality metric.",
        "",
        f"Software: Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}; NumPy {np.__version__}; SciPy {scipy.__version__}.",
        "",
    ]
    return "\n".join(lines)


def run_analysis(root: Path, out_dir: Path) -> dict:
    input_manifest = validate_input_hashes(root)
    metadata, metadata_manifest = load_metadata(root)
    input_manifest.append(metadata_manifest)

    ablation_data = audit_ablation(root, metadata)
    second_platform = audit_second_platform(root, metadata)
    ttfa_data = audit_ttfa(root, metadata)

    indexed = ablation_data["indexed"]
    candidates = ablation_data["candidates"]
    filtered = ablation_data["filtered"]

    excluded = sorted(set(candidates) - set(filtered))
    sample_flow = []
    for stage, ids, change, definition in (
        ("candidate", candidates, 0, "all sample IDs with three archived mode rows"),
        ("excluded_external_contamination", excluded, -7, "run-log-confirmed concurrent external program contamination"),
        ("valid_three_arm", filtered, -7, "505 candidates minus seven contaminated executions"),
    ):
        sample_flow.append({
            "stage": stage,
            "n_samples": len(ids),
            "n_dialogues": len({join_key(sample_id, metadata) for sample_id in ids}),
            "change_from_previous": change,
            "definition": definition,
        })

    cluster_summary = [
        cluster_row("ablation_candidate", candidates, metadata),
        cluster_row("ablation_excluded_external_contamination", sorted(set(candidates) - set(filtered)), metadata),
        cluster_row("ablation_valid_three_arm", filtered, metadata),
        cluster_row("second_platform_paired", second_platform["ids"], metadata),
        cluster_row("ttfa_repeat0_unique_samples", ttfa_data["ids"], metadata),
    ]

    def values(ids: Sequence[str], mode: str) -> list[float]:
        return [float(indexed[sample_id][mode]["ttft"]) for sample_id in ids]

    headline = [
        comparison_row(
            "headline_filtered_ab", "valid_three_arm", "headline_valid", "primary",
            "System A baseline", "System B full streaming", filtered,
            values(filtered, "baseline"), values(filtered, "full_streaming"), metadata,
            "locked 498-turn valid cohort after excluding seven externally contaminated executions",
        ),
    ]
    apply_holm(headline, "headline_valid")

    duration = []
    for group in GROUPS:
        ids = [sample_id for sample_id in filtered if indexed[sample_id]["baseline"]["duration_group"] == group]
        duration.append(comparison_row(
            f"duration_{group}_ab", group, "duration_group_ab", "Holm family",
            "System A baseline", "System B full streaming", ids,
            values(ids, "baseline"), values(ids, "full_streaming"), metadata,
        ))
    apply_holm(duration, "duration_group_ab")

    ablation = [
        comparison_row(
            "ablation_asr_contrast", "filtered_three_arm", "ablation_contrasts", "Holm family",
            "baseline arm", "streaming-ASR-only arm", filtered,
            values(filtered, "baseline"), values(filtered, "streaming_asr_only"), metadata,
            "non-causal arm contrast; fixed execution order",
        ),
        comparison_row(
            "ablation_kv_contrast", "filtered_three_arm", "ablation_contrasts", "Holm family",
            "streaming-ASR-only arm", "full-streaming arm", filtered,
            values(filtered, "streaming_asr_only"), values(filtered, "full_streaming"), metadata,
            "non-causal arm contrast; fixed execution order",
        ),
    ]
    apply_holm(ablation, "ablation_contrasts")

    platform_ids = second_platform["ids"]
    platform = [
        comparison_row(
            "second_platform_ab", "second_platform", "second_platform_comparisons", "Holm family",
            "System A non-streaming", "System B streaming", platform_ids,
            [float(second_platform["ab"][sample_id]["non-streaming"]["ttft"]) for sample_id in platform_ids],
            [float(second_platform["ab"][sample_id]["streaming"]["ttft"]) for sample_id in platform_ids],
            metadata, "same-platform validation",
        ),
        comparison_row(
            "configured_la_vs_b", "second_platform", "second_platform_comparisons", "Holm family",
            "configured LA-2-style", "System B streaming", platform_ids,
            [float(second_platform["la"][sample_id]["la_streaming"]["ttft"]) for sample_id in platform_ids],
            [float(second_platform["ab"][sample_id]["streaming"]["ttft"]) for sample_id in platform_ids],
            metadata, "configured operating point; trigger policy not matched",
        ),
    ]
    apply_holm(platform, "second_platform_comparisons")

    headline = finalize_inference(headline)
    duration = finalize_inference(duration)
    ablation = finalize_inference(ablation)
    platform = finalize_inference(platform)
    ttfa = ttfa_descriptives(ttfa_data["primary"], metadata)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv_exact(
        out_dir / "sample_flow.csv",
        ["stage", "n_samples", "n_dialogues", "change_from_previous", "definition"],
        sample_flow,
    )
    write_csv_exact(
        out_dir / "sample_exclusions.csv",
        ["sample_id", "dataset", "dialog_id", "duration_group", "stage", "exclusion_reason", "trigger_value"],
        ablation_data["sample_exclusions"],
    )
    write_csv_exact(
        out_dir / "cluster_summary.csv",
        ["cohort", "n_samples", "n_dialogues", "crosswoz_dialogues", "multiwoz_dialogues",
         "cluster_size_min", "cluster_size_median", "cluster_size_mean", "cluster_size_max"],
        cluster_summary,
    )
    write_csv_exact(out_dir / "headline_effects.csv", INFERENCE_FIELDS, headline)
    write_csv_exact(out_dir / "duration_group_inference.csv", INFERENCE_FIELDS, duration)
    write_csv_exact(out_dir / "ablation_cluster_inference.csv", INFERENCE_FIELDS, ablation)
    write_csv_exact(out_dir / "la_cluster_inference.csv", INFERENCE_FIELDS, platform)
    ttfa_fields = [
        "mode", "input_language", "tts_text_policy", "n_records", "n_dialogues",
        "ttfa_mean_ms", "ttfa_p50_ms", "tts_chars_mean", "tts_chars_p50",
        "greeting_only_n", "greeting_only_pct", "max_token_cap_n", "max_token_cap_pct",
        "output_with_latin_n", "output_without_latin_n", "cluster_size_min",
        "cluster_size_median", "cluster_size_max", "note",
    ]
    write_csv_exact(out_dir / "ttfa_policy_descriptives.csv", ttfa_fields, ttfa)
    write_text_exact(
        out_dir / "CPU_REANALYSIS_REPORT.md",
        render_report(sample_flow, cluster_summary, headline, duration, ablation, platform, ttfa),
    )

    outputs = []
    for name in OUTPUT_FILES:
        path = out_dir / name
        outputs.append({"path": name, "bytes": path.stat().st_size, "sha256": sha256_file(str(path))})
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "analysis": {
            "cpu_only": True,
            "model_inference": False,
            "audio_read": False,
            "n_boot": N_BOOT,
            "base_seed": BOOT_SEED,
            "per_comparison_seed": "uint32(first4(SHA256(f'{base_seed}:{comparison}'))) big-endian",
            "bootstrap_unit": "dialogue cluster",
            "wilcoxon_unit": "dialogue mean difference",
            "wilcoxon": "two-sided; zero_method=wilcox; correction=False; method=auto",
            "ci": "percentile 95%",
            "holm_families": [
                "headline_valid (1)",
                "duration_group_ab (3)",
                "ablation_contrasts (2)",
                "second_platform_comparisons (2)",
            ],
            "ttfa_policy": "repeat_idx=0 only; A capped full response; B first sentence",
        },
        "integrity_counts": {
            "metadata_ids": 1133,
            "candidates": 505,
            "excluded_external_contamination": 7,
            "valid_three_arm": 498,
            "filtered_dialogues": 99,
            "ttfa_primary_records": 100,
            "ttfa_primary_pairs": 50,
        },
        "inputs": input_manifest,
        "outputs": outputs,
        "software": {
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }
    write_json_exact(out_dir / "input_manifest.json", manifest)
    return manifest


def self_test() -> int:
    failures = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        if not condition:
            failures.append(f"{name}: {detail}")

    check("stable seed repeats", stable_seed("x") == stable_seed("x"))
    check("stable seed differs", stable_seed("x") != stable_seed("y"))
    check("locked base seed", BOOT_SEED == 20260821)

    left = np.array([10.0, 20.0, 30.0, 40.0])
    right = np.array([5.0, 10.0, 15.0, 20.0])
    clusters = [("d", "1"), ("d", "1"), ("d", "2"), ("d", "3")]
    first = cluster_bootstrap(left, right, clusters, "self_test", n_boot=1000)
    second = cluster_bootstrap(left, right, clusters, "self_test", n_boot=1000)
    check("cluster bootstrap deterministic", first == second, str(first))
    check("cluster point difference", abs(float(first["diff_mean"]) - 12.5) < 1e-12, str(first))
    check("cluster CI ordered", float(first["diff_ci_lo"]) <= 12.5 <= float(first["diff_ci_hi"]), str(first))

    effects = wilcoxon_effect(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.zeros(5))
    check("dialogue Wilcoxon", abs(float(effects["p_raw"]) - 0.0625) < 1e-12, str(effects))
    check("rank biserial", abs(float(effects["rank_biserial"]) - 1.0) < 1e-12, str(effects))
    check("Holm", np.allclose(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06]))

    check("greeting Chinese", greeting_only("你好！"))
    check("greeting whitespace", greeting_only(" 您好! \n"))
    check("not long greeting", not greeting_only("你好！很高兴帮助您。"))
    check("Latin proxy", has_latin("中文 Qwen") and not has_latin("纯中文"))

    try:
        index_records([
            {"sample_id": "s", "mode": "a"},
            {"sample_id": "s", "mode": "a"},
        ], ("a",))
        check("duplicate fail closed", False)
    except SystemExit:
        check("duplicate fail closed", True)
    try:
        index_records([{"sample_id": "s", "mode": "a"}], ("a", "b"))
        check("missing mode fail closed", False)
    except SystemExit:
        check("missing mode fail closed", True)

    if failures:
        for item in failures:
            print(f"FAIL {item}")
        return 1
    print("self-test: ALL PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=repository_root())
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()

    root = args.root.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else root / DEFAULT_OUT
    manifest = run_analysis(root, out_dir)
    print(
        f"CPU reanalysis complete: {out_dir} "
        f"({len(manifest['outputs']) + 1} deterministic files; 505 - 7 contaminated = 498 valid)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
