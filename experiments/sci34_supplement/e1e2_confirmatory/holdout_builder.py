"""Build a deterministic, disjoint E1/E2 holdout from local MultiWOZ.

The builder deliberately has no download path. Formal mode fails closed on a
missing or malformed exclusion source, fixture-like input/IDs, insufficient
eligible utterances, unsafe output paths, or any overlap with archived data.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from experiments.sci34_supplement.common import (
    atomic_write_json,
    canonical_json,
    sha256_bytes,
    sha256_file,
)
from experiments.scripts.prepare_multiwoz_data import (
    MIN_UTT_WORDS,
    load_multiwoz,
    split_segments,
)

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[2]
DEFAULT_OLD_E1 = REPO_ROOT / "experiments" / "results" / "exp1_latency.json"
DEFAULT_OLD_E2 = REPO_ROOT / "experiments" / "results" / "exp2_tradeoff.json"
DEFAULT_OLD_REANALYSIS = (
    REPO_ROOT / "experiments" / "results" / "paper2_reanalysis.json"
)
DEFAULT_E3_MANIFEST = (
    REPO_ROOT
    / "experiments"
    / "sci34_supplement"
    / "results"
    / "e3"
    / "sci34_f11ccba_20260901_e3"
    / "manifest.json"
)
BUILDER_SCHEMA_VERSION = 2

OLD_E1_RECORDS = "old_e1_records"
OLD_E2_RECORDS = "old_e2_records"
FIXED_E3_MANIFEST = "fixed_e3_manifest"
CUSTOM_AUTO = "custom_auto"
_SUPPORTED_SOURCE_KINDS = {
    OLD_E1_RECORDS,
    OLD_E2_RECORDS,
    FIXED_E3_MANIFEST,
    CUSTOM_AUTO,
}


@dataclass(frozen=True)
class ExclusionSource:
    path: Path
    kind: str

    def __post_init__(self) -> None:
        normalized_kind = str(self.kind).strip()
        if not normalized_kind:
            raise ValueError("Exclusion source kind must be non-empty")
        if normalized_kind not in _SUPPORTED_SOURCE_KINDS:
            raise ValueError(f"Unsupported exclusion source kind: {normalized_kind}")
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "kind", normalized_kind)


DEFAULT_EXCLUSION_SOURCES = (
    ExclusionSource(DEFAULT_OLD_E1, OLD_E1_RECORDS),
    ExclusionSource(DEFAULT_OLD_E2, OLD_E2_RECORDS),
    ExclusionSource(DEFAULT_E3_MANIFEST, FIXED_E3_MANIFEST),
)

# Protect both the three archived result JSONs and the accepted E3 manifest.
# The latter is also a mandatory exclusion source and must never become output.
PROTECTED_OLD_RESULTS = (
    DEFAULT_OLD_E1,
    DEFAULT_OLD_E2,
    DEFAULT_OLD_REANALYSIS,
    DEFAULT_E3_MANIFEST,
)


def _normalize_dialogue_id(sample_id: str) -> str:
    return str(sample_id).split("#", 1)[0].strip()


def _fixture_like(path: Path) -> bool:
    lowered = str(path).replace("\\", "/").lower()
    return "fixture" in lowered or "mini_" in path.name.lower()


def _non_empty_ids(values: Any, *, path: Path, schema: str) -> set[str]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{path} {schema} must contain a non-empty ID list")
    ids: set[str] = set()
    for index, value in enumerate(values):
        sample_id = str(value).strip() if value is not None else ""
        if not sample_id:
            raise ValueError(f"{path} {schema} has an empty ID at index {index}")
        ids.add(sample_id)
    if not ids:
        raise ValueError(f"{path} {schema} produced no IDs")
    return ids


def _record_ids(records: Any, *, path: Path, schema: str) -> set[str]:
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path} {schema} must contain non-empty records")
    values: list[Any] = []
    for index, row in enumerate(records):
        if not isinstance(row, dict):
            raise ValueError(f"{path} {schema} record {index} is not an object")
        values.append(row.get("id"))
    return _non_empty_ids(values, path=path, schema=schema)


def _ids_from_json(source: ExclusionSource) -> tuple[set[str], str]:
    """Read and strictly validate IDs according to the declared source kind."""
    path = source.path
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    if source.kind in {OLD_E1_RECORDS, OLD_E2_RECORDS}:
        if not isinstance(payload, dict) or "records" not in payload:
            raise ValueError(f"{path} does not match {source.kind} schema")
        return _record_ids(
            payload["records"], path=path, schema=source.kind
        ), "records[].id"

    if source.kind == FIXED_E3_MANIFEST:
        input_payload = payload.get("input") if isinstance(payload, dict) else None
        if not isinstance(input_payload, dict) or "sample_ids" not in input_payload:
            raise ValueError(f"{path} does not match {source.kind} schema")
        return _non_empty_ids(
            input_payload["sample_ids"], path=path, schema=source.kind
        ), "input.sample_ids[]"

    # Custom exclusions retain the path-only CLI. Their concrete schema is
    # detected, named in provenance, and then validated just as strictly.
    if isinstance(payload, list):
        return _record_ids(
            payload, path=path, schema="custom_rows"
        ), "custom_rows[].id"
    if isinstance(payload, dict) and "records" in payload:
        return _record_ids(
            payload["records"], path=path, schema="custom_records"
        ), "custom_records[].id"
    input_payload = payload.get("input") if isinstance(payload, dict) else None
    if isinstance(input_payload, dict) and "sample_ids" in input_payload:
        return _non_empty_ids(
            input_payload["sample_ids"], path=path, schema="custom_manifest"
        ), "custom_manifest.input.sample_ids[]"
    raise ValueError(f"{path} does not match a supported custom exclusion schema")


def exclusion_sources_with_defaults(
    custom_paths: Iterable[Path | ExclusionSource],
) -> list[ExclusionSource]:
    """Return mandatory old E1/E2/E3 sources plus additive custom sources."""
    sources = list(DEFAULT_EXCLUSION_SOURCES)
    seen = {source.path.resolve() for source in sources}
    for value in custom_paths:
        source = (
            value
            if isinstance(value, ExclusionSource)
            else ExclusionSource(Path(value), CUSTOM_AUTO)
        )
        resolved = source.path.resolve()
        if resolved not in seen:
            sources.append(source)
            seen.add(resolved)
    return sources


def _as_sources(paths: Iterable[Path | ExclusionSource]) -> list[ExclusionSource]:
    return [
        value
        if isinstance(value, ExclusionSource)
        else ExclusionSource(Path(value), CUSTOM_AUTO)
        for value in paths
    ]


def collect_excluded_ids(
    paths: Iterable[Path | ExclusionSource],
) -> tuple[set[str], dict[str, Any]]:
    excluded: set[str] = set()
    sources: list[dict[str, Any]] = []
    for source in _as_sources(paths):
        ids, schema = _ids_from_json(source)
        dialogue_ids = {_normalize_dialogue_id(value) for value in ids}
        if not dialogue_ids or "" in dialogue_ids:
            raise ValueError(f"Exclusion source yielded an empty dialogue ID: {source.path}")
        excluded.update(ids)
        excluded.update(dialogue_ids)
        sources.append(
            {
                "path": str(source.path.resolve()),
                "kind": source.kind,
                "schema": schema,
                "sha256": sha256_file(source.path),
                "raw_id_count": len(ids),
                "dialogue_id_count": len(dialogue_ids),
            }
        )
    return excluded, {"sources": sources, "excluded_ids": sorted(excluded)}


def validate_new_output_path(
    path: Path,
    *,
    formal: bool,
    protected_paths: Iterable[Path] = PROTECTED_OLD_RESULTS,
) -> None:
    resolved = path.resolve()
    protected = {candidate.resolve() for candidate in protected_paths}
    if resolved in protected:
        raise ValueError(f"Output path points to a protected archived result: {resolved}")
    if formal and path.exists():
        raise FileExistsError(f"Formal output already exists and will not be overwritten: {path}")


def validate_holdout(
    rows: Any,
    *,
    expected_count: int,
    excluded_ids: set[str],
    formal: bool,
) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise ValueError(f"Holdout must contain exactly {expected_count} rows")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Holdout row {index} is not an object")
        sample_id = str(row.get("id", "")).strip()
        full_text = row.get("full_text")
        segments = row.get("segments")
        if not sample_id or sample_id in seen:
            raise ValueError(f"Missing or duplicate holdout id: {sample_id!r}")
        if formal and sample_id.lower().startswith(("fx", "smoke", "fixture")):
            raise ValueError(f"Formal holdout contains fixture-like id: {sample_id}")
        if sample_id in excluded_ids or _normalize_dialogue_id(sample_id) in excluded_ids:
            raise ValueError(f"Excluded id leaked into holdout: {sample_id}")
        if not isinstance(full_text, str) or not full_text.strip():
            raise ValueError(f"Holdout {sample_id} has empty full_text")
        if not isinstance(segments, list) or len(segments) < 2 or not all(
            isinstance(segment, str) and segment for segment in segments
        ):
            raise ValueError(f"Holdout {sample_id} has invalid segments")
        if "".join(segments) != full_text:
            raise ValueError(f"Holdout {sample_id} segments are not lossless")
        seen.add(sample_id)
        normalized.append({"id": sample_id, "full_text": full_text, "segments": segments})
    return normalized


def derive_holdout(
    source: Path,
    *,
    excluded_ids: set[str],
    count: int = 100,
    seed: int = 20260901,
    formal: bool = True,
) -> list[dict[str, Any]]:
    if not source.exists():
        raise FileNotFoundError(source)
    if formal and _fixture_like(source):
        raise ValueError("Formal holdout refuses fixture-like input paths")

    dialogues = load_multiwoz(source)
    random.Random(seed).shuffle(dialogues)  # identical policy to prepare_multiwoz_data
    rows: list[dict[str, Any]] = []
    for dialogue_id, users in dialogues:
        if not str(dialogue_id).strip():
            raise ValueError("MultiWOZ source contains an empty dialogue ID")
        if dialogue_id in excluded_ids:
            continue
        for user_index, utterance in enumerate(users):
            sample_id = f"{dialogue_id}#u{user_index}"
            if sample_id in excluded_ids or len(utterance.split()) < MIN_UTT_WORDS:
                continue
            segments = split_segments(utterance)
            if segments:
                rows.append(
                    {"id": sample_id, "full_text": utterance, "segments": segments}
                )
                break
        if len(rows) == count:
            break
    if len(rows) != count:
        raise ValueError(
            f"Only {len(rows)} eligible disjoint utterances found; formal holdout requires {count}"
        )
    return validate_holdout(
        rows, expected_count=count, excluded_ids=excluded_ids, formal=formal
    )


def build_holdout(
    *,
    source: Path,
    output: Path,
    provenance_output: Path,
    exclusion_paths: list[Path | ExclusionSource],
    count: int = 100,
    seed: int = 20260901,
    formal: bool = True,
) -> dict[str, Any]:
    if output.resolve() == provenance_output.resolve():
        raise ValueError("Holdout and provenance outputs must be different files")

    sources = (
        exclusion_sources_with_defaults(exclusion_paths)
        if formal
        else _as_sources(exclusion_paths)
    )
    source_paths = [item.path for item in sources]
    protected = [*PROTECTED_OLD_RESULTS, source, *source_paths]
    validate_new_output_path(output, formal=formal, protected_paths=protected)
    validate_new_output_path(
        provenance_output, formal=formal, protected_paths=protected
    )

    excluded, exclusion_provenance = collect_excluded_ids(sources)
    rows = derive_holdout(
        source, excluded_ids=excluded, count=count, seed=seed, formal=formal
    )
    atomic_write_json(output, rows)
    payload_hash = sha256_bytes(canonical_json(rows).encode("utf-8"))
    provenance = {
        "schema_version": BUILDER_SCHEMA_VERSION,
        "builder": "e1e2_confirmatory.holdout_builder",
        "formal": formal,
        "seed": seed,
        "requested_count": count,
        "selection": {
            "shuffle": "random.Random(seed).shuffle(dialogues)",
            "utterance": "first eligible user utterance per shuffled dialogue",
            "min_utterance_words": MIN_UTT_WORDS,
            "splitter": "experiments.scripts.prepare_multiwoz_data.split_segments",
            "requires_two_or_more_segments": True,
        },
        "source": {
            "path": str(source.resolve()),
            "sha256": sha256_file(source),
        },
        "exclusions": exclusion_provenance,
        "output": {
            "path": str(output.resolve()),
            "sha256": sha256_file(output),
            "canonical_payload_sha256": payload_hash,
            "count": len(rows),
            "sample_ids": [row["id"] for row in rows],
        },
    }
    atomic_write_json(provenance_output, provenance)
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument(
        "--exclude",
        type=Path,
        action="append",
        default=[],
        help="Add a custom exclusion source; mandatory old E1/E2/E3 sources remain active",
    )
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--non-formal", action="store_true")
    args = parser.parse_args()
    provenance = build_holdout(
        source=args.input,
        output=args.output,
        provenance_output=args.provenance,
        exclusion_paths=args.exclude,
        count=args.count,
        seed=args.seed,
        formal=not args.non_formal,
    )
    print(json.dumps(provenance["output"], ensure_ascii=False))


if __name__ == "__main__":
    main()
