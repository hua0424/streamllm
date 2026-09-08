"""Tensor-free hash-manifest and exact-record integrity helpers."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from experiments.sci34_supplement.common import canonical_json, config_hash, sha256_bytes
from experiments.sci34_supplement.c2_equivalence.canonical_chat import token_ids_hash


HASH_FIELDS = ("key_sha256", "value_sha256")
EXACT_BOOLEAN_FIELDS = (
    "keep_length_exact",
    "pre_prefix_equals_oracle",
    "post_equals_pre_prefix",
    "post_equals_oracle",
    "shapes_exact",
    "dtypes_exact",
    "devices_exact",
    "mask_exact",
    "token_ids_exact",
    "logits_exact",
    "retained_prefix_hash_exact",
    "negative_control_detected",
)


def tensor_sha256(tensor: Any) -> str:
    """Hash a tensor's logical bytes after a deterministic CPU/contiguous copy."""
    array = tensor.detach().contiguous().cpu().view(-1)
    try:
        payload = array.numpy().tobytes(order="C")
    except TypeError:
        # NumPy cannot expose BF16 on some versions. Reinterpret without conversion.
        payload = array.view(__import__("torch").uint8).numpy().tobytes(order="C")
    return sha256_bytes(payload)


def layer_manifest(cache: Any, *, limit: int | None = None) -> dict[str, Any]:
    layers = []
    legacy = cache.to_legacy_cache() if hasattr(cache, "to_legacy_cache") else tuple(cache)
    for index, layer in enumerate(legacy):
        key, value = layer[:2]
        if limit is not None:
            key = key[..., :limit, :]
            value = value[..., :limit, :]
        layers.append(
            {
                "layer": index,
                "key": {
                    "shape": list(key.shape),
                    "dtype": str(key.dtype),
                    "device": str(key.device),
                    "sha256": tensor_sha256(key),
                },
                "value": {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "device": str(value.device),
                    "sha256": tensor_sha256(value),
                },
            }
        )
    payload = {"layer_count": len(layers), "layers": layers}
    payload["aggregate_sha256"] = manifest_aggregate(payload)
    return payload


def manifest_aggregate(manifest: Mapping[str, Any]) -> str:
    normalized = {
        "layer_count": manifest.get("layer_count"),
        "layers": manifest.get("layers"),
    }
    return config_hash(normalized)


def validate_manifest(manifest: Mapping[str, Any], *, label: str) -> list[str]:
    errors: list[str] = []
    layers = manifest.get("layers")
    if not isinstance(layers, list) or manifest.get("layer_count") != len(layers):
        return [f"{label}: malformed layer list/count"]
    for index, layer in enumerate(layers):
        if not isinstance(layer, dict) or layer.get("layer") != index:
            errors.append(f"{label}: layer {index} index differs")
            continue
        for side in ("key", "value"):
            tensor = layer.get(side)
            if not isinstance(tensor, dict):
                errors.append(f"{label}: layer {index} missing {side}")
                continue
            if not isinstance(tensor.get("shape"), list) or not all(
                isinstance(value, int) and value >= 0 for value in tensor.get("shape", [])
            ):
                errors.append(f"{label}: layer {index} {side} shape malformed")
            if not isinstance(tensor.get("dtype"), str) or not tensor.get("dtype"):
                errors.append(f"{label}: layer {index} {side} dtype malformed")
            if not isinstance(tensor.get("device"), str) or not tensor.get("device"):
                errors.append(f"{label}: layer {index} {side} device malformed")
            digest = tensor.get("sha256")
            if not isinstance(digest, str) or len(digest) != 64:
                errors.append(f"{label}: layer {index} {side} hash malformed")
    try:
        expected = manifest_aggregate(manifest)
    except (TypeError, ValueError):
        errors.append(f"{label}: aggregate cannot be recomputed")
    else:
        if manifest.get("aggregate_sha256") != expected:
            errors.append(f"{label}: aggregate hash differs from per-layer manifest")
    return errors


def manifests_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return canonical_json(left.get("layers")) == canonical_json(right.get("layers"))


def ledger_entry(
    *,
    ordinal: int,
    arm: str,
    operation: str,
    token_ids: Sequence[int],
    before_length: int,
    after_length: int,
    api: str,
) -> dict[str, Any]:
    ids = [int(value) for value in token_ids]
    return {
        "ordinal": ordinal,
        "arm": arm,
        "operation": operation,
        "api": api,
        "token_ids": ids,
        "token_count": len(ids),
        "token_hash": token_ids_hash(ids),
        "before_length": int(before_length),
        "after_length": int(after_length),
    }


def validate_ledger(
    ledger: Any,
    *,
    arm: str,
    initial_length: int,
    expected_chunks: Sequence[Mapping[str, Any]],
    eot_token_id: int,
) -> list[str]:
    label = f"{arm}_event_ledger"
    errors: list[str] = []
    if not isinstance(ledger, list) or len(ledger) != len(expected_chunks):
        return [f"{label}: event count differs"]
    position = int(initial_length)
    for ordinal, (entry, expected) in enumerate(zip(ledger, expected_chunks)):
        if not isinstance(entry, dict):
            errors.append(f"{label}: event {ordinal} malformed")
            continue
        ids = entry.get("token_ids")
        if not isinstance(ids, list) or not all(isinstance(value, int) for value in ids):
            errors.append(f"{label}: event {ordinal} token IDs malformed")
            continue
        if entry.get("ordinal") != ordinal or entry.get("arm") != arm:
            errors.append(f"{label}: event {ordinal} identity differs")
        if entry.get("operation") != expected.get("operation") or ids != expected.get("token_ids"):
            errors.append(f"{label}: event {ordinal} chunk differs")
        if entry.get("token_count") != len(ids) or entry.get("token_hash") != token_ids_hash(ids):
            errors.append(f"{label}: event {ordinal} token hash/count differs")
        if entry.get("before_length") != position or entry.get("after_length") != position + len(ids):
            errors.append(f"{label}: event {ordinal} length chain differs")
        position += len(ids)
    flattened = [value for entry in ledger for value in entry.get("token_ids", [])]
    expected_eots = sum(
        1 for expected in expected_chunks for value in expected.get("token_ids", []) if value == eot_token_id
    )
    if flattened.count(eot_token_id) != expected_eots:
        errors.append(f"{label}: duplicate or missing EOT")
    return errors


def record_content_hash(record: Mapping[str, Any]) -> str:
    return config_hash({key: value for key, value in record.items() if key != "record_content_hash"})
