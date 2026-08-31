"""Fixed-trajectory E3 for playback-aware history consistency.

Each dialogue generates the interrupted assistant turn exactly once.  The same
trajectory is then used to derive every playback/generation condition.  Probe
chains are generated once per unique retained history and reused across
fractions, eliminating the principal confound in the original E3 harness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from experiments.sci34_supplement.common import (
    DEFAULT_RESULTS_ROOT,
    append_jsonl,
    atomic_write_json,
    build_manifest,
    canonical_json,
    completed_keys,
    load_dialogues,
    prepare_run_directory,
    utc_now,
)
from experiments.sci34_supplement.model_runtime import (
    SYSTEM_PROMPT,
    ChatRuntime,
    GeneratedTurn,
    make_runtime,
)
from src.dialogue.unheard_detector import matched_cues, references_unheard


LOGGER = logging.getLogger(__name__)
FRACTIONS: tuple[float | str, ...] = (0.25, 0.5, 0.75, "boundary")
CONDITIONS = ("playback", "generation")


@dataclass(frozen=True)
class Fragment:
    fragment_id: int
    text: str
    token_start: int
    token_end: int
    sample_start: int
    sample_end: int


@dataclass(frozen=True)
class Trajectory:
    dialogue_id: str
    trajectory_id: str
    text: str
    token_ids: list[int]
    token_texts: list[str]
    fragments: list[Fragment]
    total_samples: int


@dataclass(frozen=True)
class DerivedBoundary:
    fraction: float | str
    played_samples: int
    interrupted_fragment_id: int
    heard_token_end: int
    partial: bool
    proxy_tail: str


def _nonblank_count(text: str) -> int:
    return sum(not char.isspace() for char in text)


def _fallback_fragments(turn: GeneratedTurn) -> list[tuple[str, int, int]]:
    if not turn.token_ids:
        return []
    text = turn.text.strip()
    if not text:
        return [(turn.text, 0, len(turn.token_ids))]
    return [(turn.text, 0, len(turn.token_ids))]


def split_fragments(
    turn: GeneratedTurn, *, tokenizer: str = "nltk", allow_fallback: bool
) -> tuple[list[tuple[str, int, int]], bool]:
    """Run the production chunker; formal runs fail instead of silently degrading."""
    try:
        from src.tts.sentence_chunker import chunk_llm_tokens

        fragments = list(
            chunk_llm_tokens(
                ((text, index) for index, text in enumerate(turn.token_texts)),
                language="en",
                tokenizer=tokenizer,
            )
        )
        if fragments:
            return [
                (fragment.text, fragment.token_start, fragment.token_end)
                for fragment in fragments
            ], False
        raise RuntimeError("Sentence chunker returned no fragments")
    except Exception as error:
        if not allow_fallback:
            raise RuntimeError(
                "Formal E3 requires a working sentence chunker; fallback is forbidden"
            ) from error
        LOGGER.warning("Sentence chunker unavailable; using one fragment: %s", error)
        return _fallback_fragments(turn), True


def capture_trajectory(
    runtime: ChatRuntime,
    dialogue_id: str,
    first_user_turn: str,
    *,
    max_new_tokens: int,
    sample_rate: int,
    samples_per_char: int,
    chunker_tokenizer: str,
    allow_chunker_fallback: bool,
) -> Trajectory:
    turn = runtime.generate(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": first_user_turn},
        ],
        max_new_tokens=max_new_tokens,
    )
    raw_fragments, fallback_used = split_fragments(
        turn,
        tokenizer=chunker_tokenizer,
        allow_fallback=allow_chunker_fallback,
    )
    if fallback_used and not allow_chunker_fallback:
        raise AssertionError("Formal chunker fallback was unexpectedly used")
    fragments: list[Fragment] = []
    sample_cursor = 0
    for fragment_id, (text, token_start, token_end) in enumerate(raw_fragments):
        n_samples = max(sample_rate // 8, _nonblank_count(text) * samples_per_char)
        fragments.append(
            Fragment(
                fragment_id=fragment_id,
                text=text,
                token_start=token_start,
                token_end=token_end,
                sample_start=sample_cursor,
                sample_end=sample_cursor + n_samples,
            )
        )
        sample_cursor += n_samples
    if not fragments:
        raise RuntimeError(f"Trajectory {dialogue_id} produced no fragments")
    payload = {
        "dialogue_id": dialogue_id,
        "token_ids": turn.token_ids,
        "fragments": [asdict(fragment) for fragment in fragments],
        "model": runtime.model_name,
        "revision": runtime.revision,
    }
    trajectory_id = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return Trajectory(
        dialogue_id=dialogue_id,
        trajectory_id=trajectory_id,
        text=turn.text,
        token_ids=turn.token_ids,
        token_texts=turn.token_texts,
        fragments=fragments,
        total_samples=sample_cursor,
    )


def derive_boundary(trajectory: Trajectory, fraction: float | str) -> DerivedBoundary:
    if fraction == "boundary":
        target = trajectory.total_samples // 2
        hit = next(
            (fragment for fragment in trajectory.fragments if fragment.sample_start < target <= fragment.sample_end),
            trajectory.fragments[-1],
        )
        played_samples = hit.sample_end
    else:
        played_samples = max(1, int(trajectory.total_samples * float(fraction)))
        hit = next(
            (fragment for fragment in trajectory.fragments if fragment.sample_start < played_samples <= fragment.sample_end),
            trajectory.fragments[-1],
        )
    partial = played_samples < hit.sample_end
    proxy_tail = ""
    if partial and hit.sample_end > hit.sample_start:
        ratio = (played_samples - hit.sample_start) / (hit.sample_end - hit.sample_start)
        cut = int(round(ratio * len(hit.text)))
        while cut > 0 and not hit.text[cut - 1].isspace():
            cut -= 1
        proxy_tail = hit.text[cut:]
    return DerivedBoundary(
        fraction=fraction,
        played_samples=played_samples,
        interrupted_fragment_id=hit.fragment_id,
        heard_token_end=hit.token_end,
        partial=partial,
        proxy_tail=proxy_tail,
    )


def build_probe_chain(
    runtime: ChatRuntime,
    *,
    first_user_turn: str,
    retained_assistant_text: str,
    probes: Sequence[str],
    max_probe_tokens: int,
) -> list[str]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": first_user_turn},
        {"role": "assistant", "content": retained_assistant_text},
    ]
    replies: list[str] = []
    for probe in probes:
        messages.append({"role": "user", "content": probe})
        reply = runtime.generate(messages, max_new_tokens=max_probe_tokens)
        replies.append(reply.text)
        messages.append({"role": "assistant", "content": reply.text})
    return replies


def record_key(dialogue_id: str, fraction: float | str, condition: str) -> tuple[str, str, str]:
    return dialogue_id, str(fraction), condition


def run_experiment(args: argparse.Namespace, runtime: ChatRuntime) -> Path:
    dialogues = load_dialogues(
        args.dialogues,
        formal=args.formal,
        limit=args.limit,
    )
    config = {
        "runtime": args.runtime,
        "model": args.model,
        "device": args.device,
        "seed": args.seed,
        "formal": args.formal,
        "limit": args.limit,
        "max_first_tokens": args.max_first_tokens,
        "max_probe_tokens": args.max_probe_tokens,
        "sample_rate": args.sample_rate,
        "samples_per_char": args.samples_per_char,
        "chunker_tokenizer": args.chunker_tokenizer,
        "chunker_fallback_allowed": not args.formal,
        "fractions": [str(value) for value in FRACTIONS],
        "conditions": list(CONDITIONS),
        "target_semantics": "shared generation-minus-playback delta evaluated in both conditions",
        "local_semantics": "condition-local unheard-in-history retained only as a construction check",
        "decode": "greedy" if args.runtime == "transformers" else "fake-deterministic",
        "model_identity": __import__(
            "experiments.sci34_supplement.common", fromlist=["model_identity"]
        ).model_identity(args.model, runtime.revision),
    }
    manifest = build_manifest(
        experiment="e3_fixed_trajectory",
        run_id=args.run_id,
        config=config,
        input_path=args.dialogues,
        sample_ids=[dialogue["id"] for dialogue in dialogues],
        extra={"model_revision": runtime.revision},
    )
    run_dir = prepare_run_directory(
        results_root=args.results_root,
        run_id=args.run_id,
        manifest=manifest,
        resume=args.resume,
    )
    trajectory_path = run_dir / "trajectories.jsonl"
    records_path = run_dir / "records.jsonl"
    existing_records = __import__(
        "experiments.sci34_supplement.common", fromlist=["load_jsonl"]
    ).load_jsonl(records_path)
    completed = completed_keys(records_path, ("id", "fraction", "condition"))
    probe_cache: dict[tuple[str, str], list[str]] = {
        (record["id"], record["history_key"]): record["probe_replies"]
        for record in existing_records
    }
    saved_trajectories = {
        row["id"]: row
        for row in __import__(
            "experiments.sci34_supplement.common", fromlist=["load_jsonl"]
        ).load_jsonl(trajectory_path)
    }

    for dialogue_index, dialogue in enumerate(dialogues, start=1):
        dialogue_id = dialogue["id"]
        if dialogue_id in saved_trajectories:
            saved = saved_trajectories[dialogue_id]
            trajectory = Trajectory(
                dialogue_id=dialogue_id,
                trajectory_id=saved["trajectory_id"],
                text=saved["text"],
                token_ids=saved["token_ids"],
                token_texts=saved["token_texts"],
                fragments=[Fragment(**fragment) for fragment in saved["fragments"]],
                total_samples=saved["total_samples"],
            )
            restore_tokens = getattr(runtime, "restore_tokens", None)
            if restore_tokens:
                restore_tokens(trajectory.token_ids, trajectory.token_texts)
        else:
            trajectory = capture_trajectory(
                runtime,
                dialogue_id,
                dialogue["turns"][0],
                max_new_tokens=args.max_first_tokens,
                sample_rate=args.sample_rate,
                samples_per_char=args.samples_per_char,
                chunker_tokenizer=args.chunker_tokenizer,
                allow_chunker_fallback=not args.formal,
            )
            saved = {
                "id": dialogue_id,
                "trajectory_id": trajectory.trajectory_id,
                "text": trajectory.text,
                "token_ids": trajectory.token_ids,
                "token_texts": trajectory.token_texts,
                "fragments": [asdict(fragment) for fragment in trajectory.fragments],
                "total_samples": trajectory.total_samples,
            }
            append_jsonl(trajectory_path, saved)
            saved_trajectories[dialogue_id] = saved
        generation_history = runtime.decode(trajectory.token_ids)
        for fraction in FRACTIONS:
            boundary = derive_boundary(trajectory, fraction)
            for condition in CONDITIONS:
                key = record_key(dialogue_id, fraction, condition)
                if key in completed:
                    continue
                keep_end = (
                    boundary.heard_token_end
                    if condition == "playback"
                    else len(trajectory.token_ids)
                )
                retained_ids = trajectory.token_ids[:keep_end]
                retained_text = runtime.decode(retained_ids)
                history_key = hashlib.sha256(
                    canonical_json([dialogue_id, retained_ids]).encode("utf-8")
                ).hexdigest()
                probe_key = (dialogue_id, history_key)
                if probe_key not in probe_cache:
                    probe_cache[probe_key] = build_probe_chain(
                        runtime,
                        first_user_turn=dialogue["turns"][0],
                        retained_assistant_text=retained_text,
                        probes=dialogue["turns"][1:],
                        max_probe_tokens=args.max_probe_tokens,
                    )
                probe_replies = probe_cache[probe_key]
                delta_text = runtime.decode(
                    trajectory.token_ids[boundary.heard_token_end:]
                )
                strict_target_text = (
                    (boundary.proxy_tail + " " + delta_text).strip()
                    if boundary.proxy_tail
                    else delta_text
                )
                local_unheard = runtime.decode(
                    trajectory.token_ids[boundary.heard_token_end:keep_end]
                )
                local_proxy_unheard = (
                    (boundary.proxy_tail + " " + local_unheard).strip()
                    if boundary.proxy_tail
                    else local_unheard
                )
                joined = " ".join(probe_replies)
                record = {
                    "id": dialogue_id,
                    "fraction": fraction,
                    "condition": condition,
                    "trajectory_id": trajectory.trajectory_id,
                    "shared_trajectory": True,
                    "history_key": history_key,
                    "assistant_token_count": len(trajectory.token_ids),
                    "heard_token_end": boundary.heard_token_end,
                    "history_token_end": keep_end,
                    "played_samples": boundary.played_samples,
                    "interrupted_fragment_id": boundary.interrupted_fragment_id,
                    "partial": boundary.partial,
                    "history_text": retained_text,
                    "unheard_text": delta_text,
                    "strict_unheard_text": strict_target_text,
                    "local_unheard_in_history_text": local_unheard,
                    "local_proxy_unheard_in_history_text": local_proxy_unheard,
                    "eligible_delta": bool(delta_text.strip()),
                    "proxy_definition": "character-proportional-whitespace-snapped",
                    "probe_replies": probe_replies,
                    "referenced_unheard": references_unheard(delta_text, joined),
                    "referenced_unheard_strict": references_unheard(strict_target_text, joined),
                    "local_referenced_unheard": references_unheard(local_unheard, joined),
                    "local_referenced_unheard_strict": references_unheard(local_proxy_unheard, joined),
                    "matched_cues": matched_cues(delta_text, joined),
                    "matched_cues_strict": matched_cues(strict_target_text, joined),
                }
                if condition == "playback":
                    assert retained_ids == trajectory.token_ids[: boundary.heard_token_end]
                    assert not local_unheard
                append_jsonl(records_path, record)
                completed.add(key)
        atomic_write_json(
            run_dir / "progress.json",
            {
                "updated_at_utc": utc_now(),
                "completed_dialogues": dialogue_index,
                "total_dialogues": len(dialogues),
                "record_count": len(completed),
            },
        )
        LOGGER.info("Completed %s/%s: %s", dialogue_index, len(dialogues), dialogue_id)
    validate_records(records_path)
    atomic_write_json(
        run_dir / "run_summary.json",
        {
            "dialogues": len(dialogues),
            "records": len(completed),
            "unique_probe_histories": len(probe_cache),
            "completed_at_utc": utc_now(),
        },
    )
    return run_dir


def validate_records(records_path: Path) -> None:
    from experiments.sci34_supplement.common import load_jsonl

    records = load_jsonl(records_path)
    by_id: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_id.setdefault(record["id"], []).append(record)
    for dialogue_id, rows in by_id.items():
        trajectories = {row["trajectory_id"] for row in rows}
        if len(trajectories) != 1:
            raise AssertionError(f"{dialogue_id} does not share one trajectory")
        generation_histories = {
            row["history_key"] for row in rows if row["condition"] == "generation"
        }
        if len(generation_histories) != 1:
            raise AssertionError(f"{dialogue_id} generation histories differ across fractions")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dialogues", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT / "e3")
    parser.add_argument("--runtime", choices=("fake", "transformers"), default="transformers")
    parser.add_argument("--model")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-first-tokens", type=int, default=40)
    parser.add_argument("--max-probe-tokens", type=int, default=40)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--samples-per-char", type=int, default=3175)
    parser.add_argument("--chunker-tokenizer", default="nltk")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    if args.formal and args.runtime == "fake":
        raise SystemExit("Formal runs require --runtime transformers")
    if args.formal:
        from experiments.sci34_supplement.common import enforce_offline_mode, require_clean_tree

        require_clean_tree(allow_dirty=args.allow_dirty)
        enforce_offline_mode()
    runtime = make_runtime(
        args.runtime,
        model_name=args.model,
        device=args.device,
        seed=args.seed,
    )
    run_dir = run_experiment(args, runtime)
    print(run_dir)


if __name__ == "__main__":
    main()
