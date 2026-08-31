"""LLM-judge wrapper for fixed-trajectory E3 with resumable JSONL output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.sci34_supplement.common import (
    append_jsonl,
    atomic_write_json,
    build_manifest,
    completed_keys,
    load_jsonl,
    prepare_run_directory,
    sha256_file,
)


PROMPT_VERSION = "specific-reference-v2"
SYSTEM = (
    "You are a strict evaluator. Decide whether the REPLY makes use of, repeats, "
    "or refers to specific information that appears in TARGET. Generic topical "
    "overlap does not count. Answer exactly YES or NO."
)


class JudgeRuntime:
    def __init__(self, model_name: str, device: str):
        from experiments.sci34_supplement.common import enforce_offline_mode

        enforce_offline_mode()
        from experiments.scripts.run_llm_judge import Judge

        self.model_name = model_name
        self._judge = Judge(model_name, device)

    def evaluate(self, target: str, replies: list[str]) -> tuple[bool, str, bool]:
        if not target.strip():
            return False, "NO", True
        user = f"TARGET:\n{target}\n\nREPLY:\n" + "\n---\n".join(replies)
        raw = self._judge._generate(SYSTEM, user, 6).strip()
        head = raw.upper()
        parsed = head.startswith("YES") or head.startswith("NO")
        return head.startswith("YES"), raw, parsed


class FakeJudgeRuntime:
    model_name = "fake-judge"

    def evaluate(self, target: str, replies: list[str]) -> tuple[bool, str, bool]:
        from src.dialogue.unheard_detector import references_unheard

        verdict = references_unheard(target, " ".join(replies))
        return verdict, "YES" if verdict else "NO", True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e3-run-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--judge-model")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not args.fake:
        from experiments.sci34_supplement.common import enforce_offline_mode, require_clean_tree

        require_clean_tree(allow_dirty=args.allow_dirty)
        enforce_offline_mode()
    source = args.e3_run_dir / "records.jsonl"
    if not source.exists():
        raise SystemExit(f"Missing E3 records: {source}")
    records = load_jsonl(source)
    results_root = args.results_root or args.e3_run_dir.parent / "judge"
    config = {
        "source_sha256": sha256_file(source),
        "judge_model": "fake-judge" if args.fake else args.judge_model,
        "device": args.device,
        "prompt_version": PROMPT_VERSION,
        "target_fields": ["unheard_text", "strict_unheard_text"],
        "model_identity": __import__(
            "experiments.sci34_supplement.common", fromlist=["model_identity"]
        ).model_identity(None if args.fake else args.judge_model),
    }
    manifest = build_manifest(
        experiment="e3_fixed_trajectory_judge",
        run_id=args.run_id,
        config=config,
        input_path=source,
        sample_ids=[
            f"{record['id']}|{record['fraction']}|{record['condition']}"
            for record in records
        ],
    )
    run_dir = prepare_run_directory(
        results_root=results_root,
        run_id=args.run_id,
        manifest=manifest,
        resume=args.resume,
    )
    output = run_dir / "judge_records.jsonl"
    completed = completed_keys(output, ("id", "fraction", "condition", "target_kind"))
    runtime = FakeJudgeRuntime() if args.fake else JudgeRuntime(args.judge_model, args.device)
    for record in records:
        for target_kind, field in (
            ("fragment", "unheard_text"),
            ("proxy", "strict_unheard_text"),
        ):
            key = (
                str(record["id"]),
                str(record["fraction"]),
                str(record["condition"]),
                target_kind,
            )
            if key in completed:
                continue
            verdict, raw, parsed = runtime.evaluate(record[field], record["probe_replies"])
            if not parsed:
                raise RuntimeError(
                    f"Judge parse failure for {key}: {raw!r}. Use a new judge run ID after fixing it."
                )
            append_jsonl(
                output,
                {
                    "id": record["id"],
                    "fraction": record["fraction"],
                    "condition": record["condition"],
                    "trajectory_id": record["trajectory_id"],
                    "target_kind": target_kind,
                    "verdict": verdict,
                    "raw_output": raw,
                    "parse_success": True,
                    "prompt_version": PROMPT_VERSION,
                },
            )
            completed.add(key)
    judged = load_jsonl(output)
    summary: dict[str, dict[str, float | int]] = {}
    for target_kind in ("fragment", "proxy"):
        for condition in ("playback", "generation"):
            subset = [
                row
                for row in judged
                if row["target_kind"] == target_kind and row["condition"] == condition
            ]
            summary[f"{target_kind}_{condition}"] = {
                "n": len(subset),
                "positive": sum(bool(row["verdict"]) for row in subset),
                "rate": (
                    sum(bool(row["verdict"]) for row in subset) / len(subset)
                    if subset
                    else 0.0
                ),
                "parse_failures": sum(not row["parse_success"] for row in subset),
            }
    atomic_write_json(run_dir / "summary.json", summary)
    print(run_dir)


if __name__ == "__main__":
    main()
