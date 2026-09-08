"""Shared persistence, reproducibility, and statistics helpers.

Only the Python standard library is imported at module load time so the local
smoke suite never initializes torch or attempts to download model weights.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Iterator, Sequence


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = PACKAGE_ROOT / "results"
SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_hash(config: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(config).encode("utf-8"))


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = canonical_json([int(base_seed), *[str(part) for part in parts]])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big") % (2**31)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", delete=False, dir=path.parent
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSONL at {path}:{line_no}: {error}") from error


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path)) if path.exists() else []


def _git(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, encoding="utf-8", errors="replace",
            stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _git_bytes(*args: str) -> bytes:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.SubprocessError):
        return b""


def model_identity(model_name: str | None, revision: str | None = None) -> dict[str, Any]:
    if not model_name:
        return {"requested": None, "revision": revision, "files": {}}
    path = Path(model_name)
    files: dict[str, str] = {}
    weight_inventory: list[dict[str, Any]] = []
    if path.exists() and path.is_dir():
        candidates = [
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        ]
        files = {
            name: sha256_file(path / name)
            for name in candidates
            if (path / name).exists()
        }
        for pattern in ("*.safetensors", "*.bin"):
            for weight in sorted(path.glob(pattern)):
                stat = weight.stat()
                weight_inventory.append(
                    {
                        "name": weight.name,
                        "size": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                    }
                )
    identity = {
        "model": model_name,
        "revision": revision,
        "files": files,
        "weight_inventory": weight_inventory,
    }
    return {
        "requested": model_name,
        "resolved_path": str(path.resolve()) if path.exists() else None,
        "revision": revision,
        "metadata_files_sha256": files,
        "weight_inventory": weight_inventory,
        "identity_hash": config_hash(identity),
    }


def require_clean_tree(*, allow_dirty: bool) -> None:
    status = _git("status", "--porcelain") or ""
    if status and not allow_dirty:
        raise RuntimeError(
            "Formal run requires a clean git working tree. Commit the experiment code or pass "
            "--allow-dirty only for a non-public diagnostic run."
        )


def enforce_offline_mode() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def collect_environment() -> dict[str, Any]:
    environment: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "perf_counter": vars(__import__("time").get_clock_info("perf_counter")),
    }
    try:
        import numpy as np

        environment["numpy"] = np.__version__
    except ImportError:
        environment["numpy"] = None
    try:
        import scipy

        environment["scipy"] = scipy.__version__
    except ImportError:
        environment["scipy"] = None
    try:
        import torch

        environment.update(
            {
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "cuda_available": torch.cuda.is_available(),
                "gpus": [
                    torch.cuda.get_device_name(index)
                    for index in range(torch.cuda.device_count())
                ],
            }
        )
    except ImportError:
        environment["torch"] = None
    try:
        import transformers

        environment["transformers"] = transformers.__version__
    except ImportError:
        environment["transformers"] = None
    return environment


def build_manifest(
    *,
    experiment: str,
    run_id: str,
    config: dict[str, Any],
    input_path: Path | None = None,
    sample_ids: Sequence[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    lock_file = ROOT / "uv.lock"
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": experiment,
        "run_id": run_id,
        "created_at_utc": utc_now(),
        "git": {
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("branch", "--show-current"),
            "dirty": bool(_git("status", "--porcelain")),
            "status_paths": (_git("status", "--porcelain") or "").splitlines(),
            "diff_sha256": sha256_bytes(_git_bytes("diff", "--binary", "HEAD")),
        },
        "config_hash": config_hash(config),
        "config": config,
        "input": {
            "path": str(input_path.resolve()) if input_path else None,
            "sha256": sha256_file(input_path) if input_path and input_path.exists() else None,
            "sample_count": len(sample_ids) if sample_ids is not None else None,
            "sample_ids": list(sample_ids) if sample_ids is not None else None,
        },
        "environment": collect_environment(),
        "uv_lock_sha256": sha256_file(lock_file) if lock_file.exists() else None,
        "extra": extra or {},
    }


def prepare_run_directory(
    *,
    results_root: Path,
    run_id: str,
    manifest: dict[str, Any],
    resume: bool,
) -> Path:
    run_dir = results_root / run_id
    manifest_path = run_dir / "manifest.json"
    if run_dir.exists() and not resume:
        raise FileExistsError(f"Run already exists: {run_dir}. Pass --resume to continue it.")
    run_dir.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("config_hash") != manifest.get("config_hash"):
            raise ValueError(
                "Resume refused: config hash differs from the existing run "
                f"({existing.get('config_hash')} != {manifest.get('config_hash')})."
            )
        if existing.get("input", {}).get("sha256") != manifest.get("input", {}).get("sha256"):
            raise ValueError("Resume refused: input SHA-256 differs from the existing run.")
    else:
        atomic_write_json(manifest_path, manifest)
    return run_dir


def completed_keys(path: Path, key_fields: Sequence[str]) -> set[tuple[str, ...]]:
    return {
        tuple(str(record[field]) for field in key_fields)
        for record in load_jsonl(path)
    }


def describe(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("Cannot describe an empty sequence")
    ordered = sorted(float(value) for value in values)

    def quantile(probability: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = (len(ordered) - 1) * probability
        low = int(position)
        high = min(low + 1, len(ordered) - 1)
        fraction = position - low
        return ordered[low] * (1 - fraction) + ordered[high] * fraction

    q1, q3 = quantile(0.25), quantile(0.75)
    return {
        "n": len(ordered),
        "min": round(ordered[0], 6),
        "q1": round(q1, 6),
        "median": round(median(ordered), 6),
        "q3": round(q3, 6),
        "iqr": round(q3 - q1, 6),
        "p90": round(quantile(0.90), 6),
        "p95": round(quantile(0.95), 6),
        "max": round(ordered[-1], 6),
    }


def validate_formal_dialogues(dialogues: Any) -> list[dict[str, Any]]:
    if not isinstance(dialogues, list) or not dialogues:
        raise ValueError("Dialogues must be a non-empty JSON list")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, dialogue in enumerate(dialogues):
        if not isinstance(dialogue, dict):
            raise ValueError(f"Dialogue {index} is not an object")
        dialogue_id = str(dialogue.get("id", "")).strip()
        turns = dialogue.get("turns")
        if not dialogue_id or dialogue_id.lower().startswith("fx"):
            raise ValueError(f"Formal data contains a missing or fixture id: {dialogue_id!r}")
        if dialogue_id in seen:
            raise ValueError(f"Duplicate dialogue id: {dialogue_id}")
        if not isinstance(turns, list) or len(turns) < 3 or not all(
            isinstance(turn, str) and turn.strip() for turn in turns
        ):
            raise ValueError(f"Dialogue {dialogue_id} must contain at least three non-empty turns")
        seen.add(dialogue_id)
        normalized.append({"id": dialogue_id, "turns": turns})
    return normalized


def load_dialogues(path: Path, *, formal: bool, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if formal:
        dialogues = validate_formal_dialogues(data)
    else:
        if not isinstance(data, list) or not data:
            raise ValueError("Fixture data must be a non-empty list")
        dialogues = data
    return dialogues[:limit] if limit is not None else dialogues
