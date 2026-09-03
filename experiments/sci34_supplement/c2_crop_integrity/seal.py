"""Create or verify the immutable SHA-256 seal for accepted C2 v3 evidence."""

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
            path for path in campaign_dir.rglob("*")
            if path.is_file() and path.name != SEAL_NAME
            and not any(path.name.endswith(suffix) for suffix in EXCLUDED_SUFFIXES)
        ),
        key=lambda path: path.relative_to(campaign_dir).as_posix(),
    )


def seal_lines(campaign_dir: Path) -> list[str]:
    return [f"{sha256_file(path)}  {path.relative_to(campaign_dir).as_posix()}" for path in _files(campaign_dir)]


def verify_seal(campaign_dir: Path) -> dict[str, object]:
    path = campaign_dir / SEAL_NAME
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    expected = seal_lines(campaign_dir)
    if actual != expected:
        common = min(len(actual), len(expected))
        mismatch = next((index for index in range(common) if actual[index] != expected[index]), common)
        raise ValueError(f"C2 v3 seal mismatch at line {mismatch + 1}")
    return {"ok": True, "files": len(expected), "seal_sha256": sha256_file(path)}


def _validation_core(value: dict[str, object]) -> dict[str, object]:
    return {key: value.get(key) for key in (
        "schema_version", "experiment", "protocol_version", "campaign_dir", "formal", "ok",
        "acceptance_eligible", "errors", "failed_indexes", "grid", "exact_gates", "provenance",
    )}


def _analysis_core(value: dict[str, object]) -> dict[str, object]:
    return {key: value.get(key) for key in (
        "schema_version", "experiment", "protocol_version", "design", "acceptance", "overall",
        "by_context", "by_scenario", "exact_gates", "negative_control", "provenance",
        "claim_boundary", "rejected_descriptive_evidence",
    )}


def create_seal(campaign_dir: Path, *, formal: bool = True) -> Path:
    from experiments.sci34_supplement.c2_crop_integrity.analyze import build_analysis
    from experiments.sci34_supplement.c2_crop_integrity.validate import validate_campaign

    required = [
        campaign_dir / name for name in (
            "campaign_manifest.json", "cases.json", "records.jsonl", "attempts.jsonl",
            "progress.json", "summary.json", "validation.json", "analysis_v1.json", "ACCEPTANCE.md",
        )
    ]
    required_dirs = (
        campaign_dir / "logs",
        campaign_dir / "snapshots" / "before",
        campaign_dir / "snapshots" / "after",
    )
    missing = [str(path) for path in required if not path.is_file()]
    missing.extend(str(path) for path in required_dirs if not path.is_dir())
    if missing:
        raise ValueError(f"Refusing to seal incomplete C2 v3 artifacts: {missing}")
    if not any((campaign_dir / "logs").iterdir()):
        raise ValueError("Refusing to seal an empty C2 v3 logs directory")
    if not any((campaign_dir / "snapshots" / "before").iterdir()):
        raise ValueError("Refusing to seal without a before snapshot")
    if not any((campaign_dir / "snapshots" / "after").iterdir()):
        raise ValueError("Refusing to seal without an after snapshot")
    stored_validation = json.loads((campaign_dir / "validation.json").read_text(encoding="utf-8"))
    stored_analysis = json.loads((campaign_dir / "analysis_v1.json").read_text(encoding="utf-8"))
    validation = validate_campaign(campaign_dir, formal=formal)
    analysis = build_analysis(campaign_dir, formal=formal)
    if canonical_json(_validation_core(stored_validation)) != canonical_json(_validation_core(validation)):
        raise ValueError("Stored validation differs from raw-artifact recomputation")
    if canonical_json(_analysis_core(stored_analysis)) != canonical_json(_analysis_core(analysis)):
        raise ValueError("Stored analysis differs from raw-artifact recomputation")
    if not validation["ok"] or not analysis["acceptance"]["passed"]:
        raise ValueError("Formal validation failure blocks seal")
    accepted = {
        line.strip() for line in (campaign_dir / "ACCEPTANCE.md").read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("<!--")
    }
    if "Status: accepted" not in accepted and "状态：accepted" not in accepted:
        raise ValueError("ACCEPTANCE.md lacks standalone accepted status")
    seal_path = campaign_dir / SEAL_NAME
    if seal_path.exists():
        raise FileExistsError(seal_path)
    atomic_write_text(seal_path, "\n".join(seal_lines(campaign_dir)) + "\n")
    verify_seal(campaign_dir)
    return seal_path


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create or verify C2 v3 campaign seal")
    parser.add_argument("--campaign-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--create", action="store_true")
    mode.add_argument("--verify", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    print(create_seal(args.campaign_dir) if args.create else json.dumps(verify_seal(args.campaign_dir), sort_keys=True))


if __name__ == "__main__":
    main()
