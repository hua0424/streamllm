"""Content-addressed local model identities for formal experiment provenance."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from experiments.sci34_supplement.common import config_hash, sha256_file


IDENTITY_SCHEMA_VERSION = 1
_IGNORED_NAMES = {".DS_Store"}
_IGNORED_SUFFIXES = {".lock", ".tmp", ".partial"}


def _model_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file() or path.name in _IGNORED_NAMES:
            continue
        if path.suffix.lower() in _IGNORED_SUFFIXES:
            continue
        yield path


def strong_model_identity(model_path: str | Path, revision: str | None = None) -> dict[str, Any]:
    """Hash every stable file in an explicit local model directory by content."""
    root = Path(model_path)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Local model directory not found: {root}")
    resolved = root.resolve()
    files: list[dict[str, Any]] = []
    for path in _model_files(resolved):
        stat = path.stat()
        files.append(
            {
                "path": path.relative_to(resolved).as_posix(),
                "size": stat.st_size,
                "sha256": sha256_file(path),
            }
        )
    if not files:
        raise ValueError(f"Local model directory contains no hashable files: {resolved}")
    payload = {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "resolved_path": str(resolved),
        "revision": revision,
        "file_count": len(files),
        "total_bytes": sum(item["size"] for item in files),
        "files": files,
    }
    payload["content_identity_hash"] = config_hash(payload)
    return payload


def model_path_from_weak_identity(identity: dict[str, Any]) -> Path:
    for field in ("resolved_path", "model", "requested"):
        value = identity.get(field)
        if value and Path(str(value)).exists() and Path(str(value)).is_dir():
            return Path(str(value)).resolve()
    raise ValueError("Trigger cache does not identify an existing local TEN model directory")
