"""Build and replay a one-pass TEN confidence cache.

This module is model-free at import time. The formal TEN implementation and
heavy runtime libraries are imported only after all formal preflight gates pass.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from experiments.sci34_supplement.common import (
    ROOT as REPO_ROOT,
    atomic_write_json,
    canonical_json,
    config_hash,
    sha256_bytes,
    sha256_file,
    utc_now,
)
from experiments.sci34_supplement.e1e2_confirmatory.strong_identity import (
    strong_model_identity,
)

CACHE_SCHEMA_VERSION = 2
_OFFLINE_ENVIRONMENT = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
_TOKEN_ENVIRONMENT = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
_PROTECTED_OLD_RESULTS = (
    REPO_ROOT / "experiments" / "results" / "exp1_latency.json",
    REPO_ROOT / "experiments" / "results" / "exp2_tradeoff.json",
    REPO_ROOT / "experiments" / "results" / "paper2_reanalysis.json",
)


class ConfidenceTrigger(Protocol):
    def confidence(self, accumulated_text: str) -> float: ...


@dataclass(frozen=True)
class TriggerCacheEntry:
    id: str
    prefix_index: int
    accumulated_text_sha256: str
    confidence: float


def text_sha256(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def _git_output(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=REPO_ROOT,
            text=True,
            encoding="utf-8",
            errors="replace",
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError("Formal trigger cache requires an inspectable git repository") from error


def _git_bytes(*args: str) -> bytes:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError("Formal trigger cache could not fingerprint the git diff") from error


def _validate_new_cache_path(path: Path) -> None:
    resolved = path.resolve()
    if resolved in {candidate.resolve() for candidate in _PROTECTED_OLD_RESULTS}:
        raise ValueError(f"Trigger cache output points to a protected old result: {resolved}")
    if path.exists():
        raise FileExistsError(f"Trigger cache output already exists: {path}")


def assert_formal_cache_prerequisites(
    *,
    model_path: str | Path,
    output_path: Path,
    git_status: str | None = None,
    environment: Mapping[str, str] | None = None,
) -> Path:
    """Fail before importing torch/transformers or loading model content."""
    env = os.environ if environment is None else environment
    status = _git_output("status", "--porcelain") if git_status is None else git_status
    if status.strip():
        raise RuntimeError("Formal trigger cache requires a clean git working tree")
    for name in _OFFLINE_ENVIRONMENT:
        if env.get(name) != "1":
            raise RuntimeError(f"Formal trigger cache requires {name}=1")
    for name in _TOKEN_ENVIRONMENT:
        if env.get(name, ""):
            raise RuntimeError(f"Formal trigger cache requires {name} to be empty")

    local_model = Path(model_path)
    if not local_model.exists() or not local_model.is_dir():
        raise FileNotFoundError(f"Local TEN model directory not found: {local_model}")
    _validate_new_cache_path(output_path)
    return local_model.resolve()


def validate_holdout_rows(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or not rows:
        raise ValueError("Trigger cache input must be a non-empty JSON list")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        sample_id = str(row.get("id", "")).strip() if isinstance(row, dict) else ""
        segments = row.get("segments") if isinstance(row, dict) else None
        full_text = row.get("full_text") if isinstance(row, dict) else None
        if not sample_id or sample_id in seen:
            raise ValueError(f"Missing or duplicate trigger-cache id: {sample_id!r}")
        if not isinstance(segments, list) or not segments or not all(
            isinstance(segment, str) and segment for segment in segments
        ):
            raise ValueError(f"Invalid segments for {sample_id}")
        if full_text is not None and "".join(segments) != full_text:
            raise ValueError(f"Non-lossless segments for {sample_id}")
        seen.add(sample_id)
        normalized.append({"id": sample_id, "segments": segments})
    return normalized


def _config_payload(config: Any) -> dict[str, Any]:
    fields = (
        "model_name",
        "system_prompt",
        "user_template",
        "positive_words",
        "negative_words",
        "device",
    )
    payload = {field: getattr(config, field) for field in fields}
    payload["positive_words"] = list(payload["positive_words"])
    payload["negative_words"] = list(payload["negative_words"])
    return payload


def _actual_model_dtype(trigger: Any) -> str:
    model = getattr(trigger, "model", None)
    dtype = getattr(model, "dtype", None)
    if dtype is not None:
        return str(dtype)
    parameters: Callable[[], Any] | None = getattr(model, "parameters", None)
    if callable(parameters):
        try:
            return str(next(parameters()).dtype)
        except (StopIteration, TypeError):
            pass
    raise RuntimeError("Could not determine the actual trigger model dtype")


def _device_payload(trigger: Any, requested_device: str, torch_module: Any) -> dict[str, Any]:
    actual_device = str(getattr(trigger, "device", ""))
    if not actual_device:
        raise RuntimeError("Could not determine the actual trigger device")
    payload: dict[str, Any] = {
        "requested": requested_device,
        "actual": actual_device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_available": bool(torch_module.cuda.is_available()),
        "cuda_runtime": getattr(getattr(torch_module, "version", None), "cuda", None),
    }
    if actual_device.startswith("cuda"):
        index = torch_module.device(actual_device).index
        if index is None:
            index = torch_module.cuda.current_device()
        properties = torch_module.cuda.get_device_properties(index)
        payload["cuda"] = {
            "index": index,
            "name": properties.name,
            "total_memory": properties.total_memory,
            "major": properties.major,
            "minor": properties.minor,
        }
    payload["identity_hash"] = config_hash(payload)
    return payload


def capture_trigger_run_identity(
    *,
    trigger: Any,
    config: Any,
    model_path: Path,
    torch_module: Any,
    transformers_module: Any,
    git_status: str | None = None,
) -> dict[str, Any]:
    """Capture the actual loaded trigger and reproducibility identities."""
    status = _git_output("status", "--porcelain") if git_status is None else git_status
    git_payload = {
        "commit": _git_output("rev-parse", "HEAD"),
        "branch": _git_output("branch", "--show-current"),
        "dirty": bool(status.strip()),
        "status_paths": status.splitlines(),
        "diff_sha256": (
            sha256_bytes(b"")
            if git_status is not None
            else sha256_bytes(_git_bytes("diff", "--binary", "HEAD"))
        ),
    }
    git_payload["identity_hash"] = config_hash(git_payload)

    lock_files = {}
    for name in ("pyproject.toml", "uv.lock"):
        path = REPO_ROOT / name
        lock_files[name] = sha256_file(path) if path.exists() else None
    environment_payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch": str(torch_module.__version__),
        "transformers": str(transformers_module.__version__),
        "cuda_runtime": getattr(getattr(torch_module, "version", None), "cuda", None),
        "cudnn": torch_module.backends.cudnn.version(),
        "strict_offline": all(os.environ.get(name) == "1" for name in _OFFLINE_ENVIRONMENT),
        "hf_token_empty": all(not os.environ.get(name, "") for name in _TOKEN_ENVIRONMENT),
        "lock_files_sha256": lock_files,
    }
    environment_payload["identity_hash"] = config_hash(environment_payload)

    config_payload = _config_payload(config)
    config_payload["actual_model_dtype"] = _actual_model_dtype(trigger)
    config_payload["config_hash"] = config_hash(config_payload)
    device_payload = _device_payload(
        trigger, str(config_payload["device"]), torch_module
    )
    return {
        "git": git_payload,
        "environment": environment_payload,
        "device": device_payload,
        "model": strong_model_identity(model_path),
        "config": config_payload,
    }


def build_trigger_cache_payload(
    rows: Sequence[Mapping[str, Any]],
    trigger: ConfidenceTrigger,
    *,
    trigger_identity: Mapping[str, Any],
    trigger_template: str,
    positive_words: Sequence[str],
    negative_words: Sequence[str],
    input_sha256: str,
    created_at_utc: str | None = None,
    run_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for row in rows:
        accumulated = ""
        for prefix_index, segment in enumerate(row["segments"], start=1):
            accumulated += str(segment)
            confidence = float(trigger.confidence(accumulated))
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise ValueError(
                    f"Invalid confidence for {row['id']} prefix {prefix_index}: {confidence}"
                )
            entries.append(
                asdict(
                    TriggerCacheEntry(
                        id=str(row["id"]),
                        prefix_index=prefix_index,
                        accumulated_text_sha256=text_sha256(accumulated),
                        confidence=confidence,
                    )
                )
            )
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "created_at_utc": created_at_utc or utc_now(),
        "input_sha256": input_sha256,
        "trigger": {
            "template": trigger_template,
            "template_sha256": text_sha256(trigger_template),
            "positive_words": list(positive_words),
            "negative_words": list(negative_words),
            "positive_token_ids": list(getattr(trigger, "_pos_ids", [])),
            "negative_token_ids": list(getattr(trigger, "_neg_ids", [])),
            "aggregation": "logsumexp(positive logits) vs logsumexp(negative logits), then sigmoid difference",
            "model_identity": dict(trigger_identity),
            "run_identity": dict(run_identity or {}),
        },
        "entry_count": len(entries),
        "entries": entries,
    }
    payload["identity_hash"] = config_hash(
        {
            "schema_version": payload["schema_version"],
            "input_sha256": input_sha256,
            "trigger": payload["trigger"],
            "entries": entries,
        }
    )
    return payload


def write_trigger_cache(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    _validate_new_cache_path(path)
    atomic_write_json(path, payload)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "identity_hash": payload["identity_hash"],
        "entry_count": payload["entry_count"],
    }


def load_ten_trigger(*, model_path: str, device: str | None = None):
    """Load production TEN from an already-validated local directory."""
    local_model = Path(model_path).resolve()
    from src.dialogue.trigger import LLMSoftTrigger, TEN_CONFIG, TriggerConfig

    config = TriggerConfig(
        model_name=str(local_model),
        system_prompt=TEN_CONFIG.system_prompt,
        user_template=TEN_CONFIG.user_template,
        positive_words=list(TEN_CONFIG.positive_words),
        negative_words=list(TEN_CONFIG.negative_words),
        device=device or TEN_CONFIG.device,
    )
    try:
        trigger = LLMSoftTrigger(config=config)
    except Exception as error:
        raise RuntimeError(
            "Failed to load TEN from the explicit local directory in strict offline mode"
        ) from error
    return trigger, config


class ReplayTrigger:
    """Read-only adapter exposing the production ``confidence(text)`` interface."""

    def __init__(
        self,
        cache_path: Path,
        *,
        expected_input_sha256: str | None = None,
        expected_identity_hash: str | None = None,
    ) -> None:
        self.cache_path = cache_path
        self.cache_sha256 = sha256_file(cache_path)
        self.payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if self.payload.get("schema_version") != CACHE_SCHEMA_VERSION:
            raise ValueError("Unsupported trigger-cache schema")
        if expected_input_sha256 and self.payload.get("input_sha256") != expected_input_sha256:
            raise ValueError("Trigger cache input SHA-256 mismatch")
        if expected_identity_hash and self.payload.get("identity_hash") != expected_identity_hash:
            raise ValueError("Trigger cache identity mismatch")
        self.identity_hash = str(self.payload.get("identity_hash", ""))
        self._entries: dict[tuple[str, int], TriggerCacheEntry] = {}
        for raw in self.payload.get("entries", []):
            entry = TriggerCacheEntry(
                id=str(raw["id"]),
                prefix_index=int(raw["prefix_index"]),
                accumulated_text_sha256=str(raw["accumulated_text_sha256"]),
                confidence=float(raw["confidence"]),
            )
            key = (entry.id, entry.prefix_index)
            if key in self._entries:
                raise ValueError(f"Duplicate trigger-cache entry: {key}")
            self._entries[key] = entry
        if len(self._entries) != self.payload.get("entry_count"):
            raise ValueError("Trigger-cache entry_count mismatch")
        self._active_id: str | None = None
        self._next_prefix = 1

    @property
    def model_identity(self) -> dict[str, Any]:
        return dict(self.payload["trigger"]["model_identity"])

    def start(self, sample_id: str) -> "ReplayTrigger":
        self._active_id = str(sample_id)
        self._next_prefix = 1
        return self

    def confidence_for(self, sample_id: str, prefix_index: int, accumulated_text: str) -> float:
        key = (str(sample_id), int(prefix_index))
        if key not in self._entries:
            raise KeyError(f"No cached confidence for {key}")
        entry = self._entries[key]
        actual_hash = text_sha256(accumulated_text)
        if actual_hash != entry.accumulated_text_sha256:
            raise ValueError(
                f"Trigger replay text hash mismatch for {key}: "
                f"{actual_hash} != {entry.accumulated_text_sha256}"
            )
        return entry.confidence

    def confidence(self, accumulated_text: str) -> float:
        if self._active_id is None:
            raise RuntimeError("Call ReplayTrigger.start(sample_id) before confidence(text)")
        value = self.confidence_for(self._active_id, self._next_prefix, accumulated_text)
        self._next_prefix += 1
        return value


def build_formal_cache(
    *, input_path: Path, output_path: Path, model_path: str, device: str | None
) -> dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Trigger cache output must differ from its holdout input")
    local_model = assert_formal_cache_prerequisites(
        model_path=model_path,
        output_path=output_path,
    )
    rows = validate_holdout_rows(json.loads(input_path.read_text(encoding="utf-8")))

    trigger, config = load_ten_trigger(model_path=str(local_model), device=device)
    import torch
    import transformers

    run_identity = capture_trigger_run_identity(
        trigger=trigger,
        config=config,
        model_path=local_model,
        torch_module=torch,
        transformers_module=transformers,
    )
    config_payload = run_identity["config"]
    payload = build_trigger_cache_payload(
        rows,
        trigger,
        trigger_identity=run_identity["model"],
        trigger_template=config_payload["user_template"],
        positive_words=config_payload["positive_words"],
        negative_words=config_payload["negative_words"],
        input_sha256=sha256_file(input_path),
        run_identity=run_identity,
    )
    return write_trigger_cache(output_path, payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Explicit local TEN model directory")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    result = build_formal_cache(
        input_path=args.input,
        output_path=args.output,
        model_path=args.model,
        device=args.device,
    )
    print(canonical_json(result))


if __name__ == "__main__":
    main()
