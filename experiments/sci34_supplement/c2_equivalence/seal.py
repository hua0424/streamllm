"""Create or verify a sorted relative-path SHA-256 seal for an accepted C2 campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.sci34_supplement.common import atomic_write_text, canonical_json, sha256_file


SEAL_NAME = "checksums.sha256"
EXCLUDED_SUFFIXES = (".tar.gz", ".tar.gz.sha256")


def _files(campaign_dir: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in campaign_dir.rglob("*")
            if path.is_file()
            and path.name != SEAL_NAME
            and not any(path.name.endswith(suffix) for suffix in EXCLUDED_SUFFIXES)
        ),
        key=lambda path: path.relative_to(campaign_dir).as_posix(),
    )


def seal_lines(campaign_dir: Path) -> list[str]:
    return [
        f"{sha256_file(path)}  {path.relative_to(campaign_dir).as_posix()}"
        for path in _files(campaign_dir)
    ]


def verify_seal(campaign_dir: Path) -> dict[str, object]:
    seal_path = campaign_dir / SEAL_NAME
    if not seal_path.exists():
        raise FileNotFoundError(seal_path)
    actual = [line for line in seal_path.read_text(encoding="utf-8").splitlines() if line]
    expected = seal_lines(campaign_dir)
    if actual != expected:
        common = min(len(actual), len(expected))
        mismatch = next((index for index in range(common) if actual[index] != expected[index]), common)
        raise ValueError(f"C2 seal mismatch at line {mismatch + 1}")
    return {"ok": True, "files": len(expected), "seal_sha256": sha256_file(seal_path)}


def _validation_core(payload: dict[str, object]) -> dict[str, object]:
    keys = (
        "schema_version", "experiment", "protocol_version", "campaign_dir", "formal", "ok",
        "acceptance_eligible", "errors", "failed_indexes", "grid",
        "termination_probes", "thresholds", "provenance",
    )
    return {key: payload.get(key) for key in keys}


def _analysis_core(payload: dict[str, object]) -> dict[str, object]:
    keys = (
        "schema_version", "experiment", "protocol_version", "design", "acceptance",
        "noise_control", "termination_probes", "scenario_execution", "overall", "by_context", "by_scenario",
        "by_termination", "by_checkpoint", "worst_cases",
        "all_failure_indexes", "provenance", "claim_boundary",
    )
    return {key: payload.get(key) for key in keys}


def create_seal(campaign_dir: Path, *, formal: bool = True) -> Path:
    from experiments.sci34_supplement.c2_equivalence.analyze import build_analysis
    from experiments.sci34_supplement.c2_equivalence.validate import validate_campaign

    validation_path = campaign_dir / "validation.json"
    analysis_path = campaign_dir / "analysis_v1.json"
    acceptance_path = campaign_dir / "ACCEPTANCE.md"
    required_files = (
        campaign_dir / "campaign_manifest.json",
        campaign_dir / "cases.json",
        campaign_dir / "records.jsonl",
        campaign_dir / "attempts.jsonl",
        campaign_dir / "progress.json",
        campaign_dir / "summary.json",
        validation_path,
        analysis_path,
        acceptance_path,
    )
    required_dirs = (
        campaign_dir / "logs",
        campaign_dir / "snapshots",
        campaign_dir / "snapshots" / "before",
        campaign_dir / "snapshots" / "after",
        campaign_dir / "checkpoints",
    )
    missing = [str(path) for path in required_files if not path.is_file()]
    missing.extend(str(path) for path in required_dirs if not path.is_dir())
    if missing:
        raise ValueError(f"Refusing to seal incomplete C2 artifacts: {missing}")
    if not any((campaign_dir / "logs").iterdir()):
        raise ValueError("Refusing to seal an empty logs directory")
    if not any((campaign_dir / "checkpoints").iterdir()):
        raise ValueError("Refusing to seal without frozen checkpoint logits sidecars")
    if not any((campaign_dir / "snapshots" / "before").iterdir()):
        raise ValueError("Refusing to seal an empty snapshots/before directory")
    if not any((campaign_dir / "snapshots" / "after").iterdir()):
        raise ValueError("Refusing to seal an empty snapshots/after directory")

    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    recomputed_validation = validate_campaign(campaign_dir, formal=formal)
    recomputed_analysis = build_analysis(campaign_dir, formal=formal)
    validation_core = _validation_core(validation)
    recomputed_validation_core = _validation_core(recomputed_validation)
    if canonical_json(validation_core) != canonical_json(recomputed_validation_core):
        raise ValueError("Stored validation differs from direct raw-artifact recomputation")
    analysis_core = _analysis_core(analysis)
    recomputed_analysis_core = _analysis_core(recomputed_analysis)
    if canonical_json(analysis_core) != canonical_json(recomputed_analysis_core):
        raise ValueError("Stored analysis differs from direct raw-artifact recomputation")
    acceptance = acceptance_path.read_text(encoding="utf-8")
    if not recomputed_validation.get("ok") or not recomputed_validation.get("acceptance_eligible"):
        raise ValueError("Refusing to seal a failed C2 validation")
    if not recomputed_analysis.get("acceptance", {}).get("passed"):
        raise ValueError("Refusing to seal a failed C2 analysis")
    accepted_lines = {
        line.strip() for line in acceptance.splitlines()
        if not line.lstrip().startswith("<!--")
    }
    if "Status: accepted" not in accepted_lines and "状态：accepted" not in accepted_lines:
        raise ValueError("ACCEPTANCE.md must contain a standalone accepted status line")
    seal_path = campaign_dir / SEAL_NAME
    if seal_path.exists():
        raise FileExistsError(f"Seal already exists: {seal_path}")
    atomic_write_text(seal_path, "\n".join(seal_lines(campaign_dir)) + "\n")
    verify_seal(campaign_dir)
    return seal_path


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create or verify the immutable C2 checksum seal.")
    parser.add_argument("--campaign-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--create", action="store_true")
    mode.add_argument("--verify", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.create:
        print(create_seal(args.campaign_dir, formal=True))
    else:
        print(json.dumps(verify_seal(args.campaign_dir), sort_keys=True))


if __name__ == "__main__":
    main()
