"""Fake and Transformers runtimes for the C2 v3 crop-integrity addendum."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Mapping, Protocol, Sequence

from experiments.sci34_supplement.common import canonical_json, config_hash
from experiments.sci34_supplement.c2_crop_integrity.canonical_chat import (
    ChatTemplateParts,
    apply_chat_ids,
    token_ids_hash,
)
from experiments.sci34_supplement.c2_crop_integrity.integrity import (
    layer_manifest,
    ledger_entry,
    manifest_aggregate,
    manifests_equal,
)
from experiments.sci34_supplement.c2_crop_integrity.protocol import (
    EXPECTED_DTYPE,
    EXPECTED_MODEL_ARCHITECTURE,
    EXPECTED_MODEL_ARTIFACT_HASH,
    EXPECTED_MODEL_TYPE,
    SYSTEM_PROMPT,
    CaseSpec,
)
from experiments.sci34_supplement.e1e2_confirmatory.strong_identity import strong_model_identity


class CropIntegrityBackend(Protocol):
    @property
    def identity(self) -> Mapping[str, Any]: ...

    @property
    def runtime_metadata(self) -> Mapping[str, Any]: ...

    def negative_control(self) -> dict[str, Any]: ...

    def case_token_plan(self, case: CaseSpec) -> dict[str, Any]: ...

    def run_case(self, case: CaseSpec) -> dict[str, Any]: ...


def _digest(*parts: object) -> str:
    return hashlib.sha256(canonical_json([str(value) for value in parts]).encode()).hexdigest()


def _fake_manifest(label: str, keep: int, *, layers: int = 3) -> dict[str, Any]:
    rows = []
    for index in range(layers):
        shape = [1, 2, keep, 4]
        rows.append(
            {
                "layer": index,
                "key": {
                    "shape": shape,
                    "dtype": "torch.float32",
                    "device": "cpu",
                    "sha256": _digest(label, index, "key", keep),
                },
                "value": {
                    "shape": shape,
                    "dtype": "torch.float32",
                    "device": "cpu",
                    "sha256": _digest(label, index, "value", keep),
                },
            }
        )
    value = {"layer_count": layers, "layers": rows}
    value["aggregate_sha256"] = manifest_aggregate(value)
    return value


def _clone_json(value: Any) -> Any:
    import json

    return json.loads(json.dumps(value))


@dataclass
class FakeBackend:
    """Pure CPU/no-model backend exercising persistence, validation, and tamper gates."""

    seed: int = 20260903

    @property
    def identity(self) -> dict[str, Any]:
        payload = {"kind": "fake-crop-integrity", "seed": self.seed, "version": 1}
        payload["content_identity_hash"] = config_hash(payload)
        return payload

    @property
    def runtime_metadata(self) -> dict[str, Any]:
        return {
            "execution": "synthetic_cpu",
            "resolved_dtype": "fake-float32",
            "attention_backend": "fake",
            "tokenizer_class": "FakeTokenizer",
            "chat_template_sha256": "f" * 64,
            "eot_token_id": 2,
        }

    def case_token_plan(self, case: CaseSpec) -> dict[str, Any]:
        assistant_ids = [500 + index for index, _ in enumerate(case.assistant_text.split(), start=1)]
        second_ids = (
            [800 + index for index, _ in enumerate((case.second_assistant_text or "").split(), start=1)]
            if case.second_crop_fraction is not None else None
        )
        payload = {
            "case_id": case.id,
            "assistant_token_ids": assistant_ids,
            "assistant_token_hash": token_ids_hash(assistant_ids),
            "second_assistant_token_ids": second_ids,
            "second_assistant_token_hash": token_ids_hash(second_ids) if second_ids else None,
            "all_non_eot": True,
        }
        payload["plan_hash"] = config_hash(payload)
        return payload

    def negative_control(self) -> dict[str, Any]:
        correct = _fake_manifest("negative-control", 4)
        wrong = _fake_manifest("negative-control-wrong", 3)
        detected = not manifests_equal(correct, wrong)
        return {
            "kind": "wrong_keep_length_manifest",
            "disposable": True,
            "correct_keep_length": 4,
            "wrong_keep_length": 3,
            "detected": detected,
            "positive_control_metadata": {
                "wrong_crop_length_would_be_detected": detected,
                "mechanism": "shape and aggregate manifest mismatch",
            },
        }

    @staticmethod
    def _chunks(case: CaseSpec, *, second: bool) -> list[dict[str, Any]]:
        if second:
            return [{"operation": "reopen_user_role", "token_ids": [2, 31, 32]}]
        chunks = []
        if case.scenario != "speculation_full_invalidation":
            chunks.append({"operation": "reopen_user_role", "token_ids": [2, 31, 32]})
        if case.next_user is not None:
            chunks.append({"operation": "prefill_user_text", "token_ids": [401, 402]})
            chunks.append({"operation": "open_assistant_role", "token_ids": [41, 42]})
        return chunks

    def _event(self, case: CaseSpec, *, ordinal: int, second: bool) -> dict[str, Any]:
        event_id = "crop_2" if second else "crop_1"
        if second:
            second_ids = self.case_token_plan(case)["second_assistant_token_ids"] or []
            assistant_content_start = case.context_tokens + 10
            pre_len = assistant_content_start + len(second_ids)
            retained_second = max(1, min(len(second_ids), math.floor(len(second_ids) * float(case.second_crop_fraction))))
            keep = assistant_content_start + retained_second
            crop_semantics = "second_fraction_floor_clamp"
            fragment_ids: list[list[int]] = []
            assistant_role_start = assistant_content_start - 2
        else:
            assistant_ids = self.case_token_plan(case)["assistant_token_ids"]
            fragment_ids = []
            cursor = 0
            word_counts = [max(1, len(fragment.split())) for fragment in case.fragments]
            for index, count in enumerate(word_counts):
                remaining_fragments = len(word_counts) - index - 1
                remaining_ids = len(assistant_ids) - cursor
                take = remaining_ids if remaining_fragments == 0 else max(1, min(count, remaining_ids - remaining_fragments))
                fragment_ids.append(list(assistant_ids[cursor : cursor + take]))
                cursor += take
            if cursor != len(assistant_ids) or any(not ids for ids in fragment_ids):
                raise RuntimeError("Fake fragment token partition failed")
            assistant_content_start = case.context_tokens
            assistant_role_start = assistant_content_start - 2
            pre_len = assistant_content_start + len(assistant_ids)
            if case.scenario == "reply_tail_noop":
                keep, crop_semantics = pre_len, "reply_tail_noop"
            elif case.scenario == "speculation_full_invalidation":
                keep, crop_semantics = assistant_role_start, "speculation_full_invalidation"
            elif case.retain_fragment_count == 0:
                keep, crop_semantics = assistant_content_start, "empty_assistant_turn_p0"
            else:
                keep = assistant_content_start + sum(len(ids) for ids in fragment_ids[: case.retain_fragment_count])
                crop_semantics = "retained_fragment_prefix"
        pre = _fake_manifest(case.id + event_id, keep)
        post = _clone_json(pre)
        oracle = _clone_json(pre)
        chunks = self._chunks(case, second=second)
        production_ledger = []
        oracle_ledger = []
        position = keep
        recovery = []
        for chunk_index, chunk in enumerate(chunks):
            ids = list(chunk["token_ids"])
            production_ledger.append(
                ledger_entry(
                    ordinal=chunk_index,
                    arm="production",
                    operation=chunk["operation"],
                    token_ids=ids,
                    before_length=position,
                    after_length=position + len(ids),
                    api=f"StreamLLMInference.{chunk['operation']}",
                )
            )
            oracle_ledger.append(
                ledger_entry(
                    ordinal=chunk_index,
                    arm="oracle",
                    operation=chunk["operation"],
                    token_ids=ids,
                    before_length=position,
                    after_length=position + len(ids),
                    api="direct_model_forward",
                )
            )
            position += len(ids)
            recovery_manifest = _fake_manifest(case.id + event_id + f"r{chunk_index}", position)
            prefix = _clone_json(pre)
            expected_phase = "ASSISTANT_OPEN" if chunk["operation"] == "open_assistant_role" else "USER_OPEN"
            expected_state = {
                "role_phase": expected_phase,
                "generation_end_reason": "NONE",
                "assistant_content_start": position if expected_phase == "ASSISTANT_OPEN" else None,
                "assistant_token_count": 0,
                "seq_length": position,
                "mask_length": position,
                "kv_length": position,
                "token_ledger_length": position,
            }
            recovery.append(
                {
                    "ordinal": chunk_index,
                    "operation": chunk["operation"],
                    "token_ids": ids,
                    "token_hash": token_ids_hash(ids),
                    "production_manifest": recovery_manifest,
                    "oracle_manifest": _clone_json(recovery_manifest),
                    "production_prefix_manifest": prefix,
                    "oracle_prefix_manifest": _clone_json(prefix),
                    "production_logits_sha256": _digest(case.id, event_id, chunk_index, "logits"),
                    "oracle_logits_sha256": _digest(case.id, event_id, chunk_index, "logits"),
                    "production_state": expected_state,
                    "expected_state": dict(expected_state),
                    "production_state_exact": True,
                    "kv_exact": True,
                    "logits_exact": True,
                    "masks_exact": True,
                    "token_ids_exact": True,
                    "retained_prefix_hash_exact": True,
                    "passed": True,
                    "errors": [],
                }
            )
        retained_ids = list(range(100, 100 + keep))
        final_ids = [*retained_ids, *[value for chunk in chunks for value in chunk["token_ids"]]]
        return {
            "event_id": event_id,
            "event_index": ordinal,
            "pre_crop_length": pre_len,
            "keep_length": keep,
            "no_op": keep == pre_len,
            "crop_target_semantics": crop_semantics,
            "assistant_role_start": assistant_role_start,
            "assistant_content_start": assistant_content_start,
            "retain_fragment_count": case.retain_fragment_count if not second else None,
            "fragment_token_ids": fragment_ids if not second else None,
            "second_assistant_token_ids": second_ids if second else None,
            "second_crop_fraction": case.second_crop_fraction if second else None,
            "pre_crop_token_ids": list(range(100, 100 + pre_len)),
            "pre_crop_token_hash": token_ids_hash(range(100, 100 + pre_len)),
            "retained_token_ids": retained_ids,
            "retained_token_hash": token_ids_hash(retained_ids),
            "pre_prefix_manifest": pre,
            "post_production_manifest": post,
            "oracle_manifest": oracle,
            "post_crop_state": {
                "role_phase": "USER_OPEN" if crop_semantics == "speculation_full_invalidation" else "ASSISTANT_OPEN",
                "generation_end_reason": "CROPPED" if keep < pre_len else "MAX_TOKENS",
                "assistant_content_start": None if crop_semantics == "speculation_full_invalidation" else assistant_content_start,
                "assistant_token_count": 0 if crop_semantics == "speculation_full_invalidation" else max(0, keep - assistant_content_start),
                "seq_length": keep,
                "mask_length": keep,
                "kv_length": keep,
                "token_ledger_length": keep,
            },
            "post_crop_oracle_state": {
                "seq_length": keep,
                "mask_length": keep,
                "kv_length": keep,
                "token_ledger_length": keep,
            },
            "post_crop_lengths_exact": True,
            "post_crop_mask_exact": True,
            "post_crop_token_ids_exact": True,
            "production_event_ledger": production_ledger,
            "oracle_event_ledger": oracle_ledger,
            "expected_recovery_chunks": chunks,
            "recovery_checks": recovery,
            "final_token_ids": final_ids,
            "final_token_hash": token_ids_hash(final_ids),
            "canonical_ledger": {
                "token_ids": final_ids,
                "token_hash": token_ids_hash(final_ids),
                "eot_positions": [keep] if chunks and chunks[0]["operation"] == "reopen_user_role" else [],
                "assistant_boundaries": 1 if chunks and chunks[0]["operation"] == "reopen_user_role" else 0,
                "unique_eot": True,
                "role_boundary_exact": True,
            },
            "keep_length_exact": True,
            "pre_prefix_equals_oracle": True,
            "post_equals_pre_prefix": True,
            "post_equals_oracle": True,
            "shapes_exact": True,
            "dtypes_exact": True,
            "devices_exact": True,
            "mask_exact": True,
            "token_ids_exact": True,
            "logits_exact": True,
            "retained_prefix_hash_exact": True,
            "negative_control_detected": True,
            "passed": True,
            "errors": [],
        }

    def run_case(self, case: CaseSpec) -> dict[str, Any]:
        token_plan = self.case_token_plan(case)
        assistant_ids = token_plan["assistant_token_ids"]
        events = [self._event(case, ordinal=0, second=False)]
        if case.second_crop_fraction is not None:
            events.append(self._event(case, ordinal=1, second=True))
        fixture_ledger = [
            ledger_entry(
                ordinal=index,
                arm="production",
                operation="generate_accumulating_token",
                token_ids=[token],
                before_length=case.context_tokens + index,
                after_length=case.context_tokens + index + 1,
                api="StreamLLMInference.generate_accumulating->_prefill_ids_p2",
            )
            for index, token in enumerate(assistant_ids)
        ]
        second_ids = token_plan["second_assistant_token_ids"]
        second_initial = len(events[0]["final_token_ids"])
        second_fixture_ledger = (
            [
                ledger_entry(
                    ordinal=index,
                    arm="production",
                    operation="generate_accumulating_token",
                    token_ids=[token],
                    before_length=second_initial + index,
                    after_length=second_initial + index + 1,
                    api="StreamLLMInference.generate_accumulating->_prefill_ids_p2",
                )
                for index, token in enumerate(second_ids)
            ]
            if isinstance(second_ids, list)
            else None
        )
        return {
            "case_id": case.id,
            "context_tokens_target": case.context_tokens,
            "context_tokens_actual": case.context_tokens,
            "context_class": case.context_class,
            "scenario": case.scenario,
            "termination": case.termination,
            "prior_v2_probe_reused": True,
            "termination_probe_rerun": False,
            "token_plan_hash": token_plan["plan_hash"],
            "fixture": {
                "assistant_token_ids": assistant_ids,
                "assistant_token_hash": token_ids_hash(assistant_ids),
                "second_assistant_token_ids": token_plan["second_assistant_token_ids"],
                "second_assistant_token_hash": token_plan["second_assistant_token_hash"],
                "all_non_eot": token_plan["all_non_eot"],
                "generate_api": "StreamLLMInference.generate_accumulating",
                "prefill_ids_p2_calls": len(assistant_ids),
                "one_token_per_forward": True,
                "event_ledger": fixture_ledger,
                "second_assistant_event_ledger": second_fixture_ledger,
            },
            "crop_events": events,
            "passed": all(event["passed"] for event in events),
            "errors": [],
        }


class TransformersBackend:
    """Bitwise crop/oracle comparison on the frozen Qwen2-7B Transformers backend."""

    def __init__(self, model_path: str, *, device: str = "cuda:0", system_prompt: str = SYSTEM_PROMPT) -> None:
        import torch
        from transformers import DynamicCache
        from src.llm.stream_llm_inference import StreamLLMInference

        self.torch = torch
        self.DynamicCache = DynamicCache
        self.device = device
        self.system_prompt = system_prompt
        self.llm = StreamLLMInference(model_name=model_path, device=device, eval_mode=False)
        self.model = self.llm.model
        self.tokenizer = self.llm.tokenizer
        self.parts = ChatTemplateParts.from_tokenizer(self.tokenizer, system_prompt)
        self._identity = strong_model_identity(model_path)
        artifact_payload = {
            key: self._identity[key]
            for key in ("schema_version", "file_count", "total_bytes", "files")
        }
        artifact_hash = config_hash(artifact_payload)
        model_type = str(getattr(self.model.config, "model_type", ""))
        architectures = list(getattr(self.model.config, "architectures", None) or [])
        first_parameter = next(self.model.parameters())
        resolved_dtype = str(first_parameter.dtype)
        if artifact_hash != EXPECTED_MODEL_ARTIFACT_HASH:
            raise RuntimeError(f"Wrong accepted model artifact: {artifact_hash}")
        if model_type != EXPECTED_MODEL_TYPE or EXPECTED_MODEL_ARCHITECTURE not in architectures:
            raise RuntimeError("C2 v3 requires the frozen Qwen2ForCausalLM snapshot")
        if resolved_dtype != EXPECTED_DTYPE:
            raise RuntimeError(f"C2 v3 requires {EXPECTED_DTYPE}, got {resolved_dtype}")
        self._runtime_metadata = {
            "execution": "transformers_model",
            "resolved_dtype": resolved_dtype,
            "model_type": model_type,
            "architectures": architectures,
            "accepted_model_artifact_hash": artifact_hash,
            "attention_backend": str(
                getattr(self.model.config, "_attn_implementation", None)
                or getattr(self.model.config, "attn_implementation", None)
                or "eager/default"
            ),
            "tokenizer_class": type(self.tokenizer).__name__,
            "chat_template_sha256": self.parts.template_hash,
            "eot_token_id": self.parts.eot_token_id,
        }

    @property
    def identity(self) -> dict[str, Any]:
        return dict(self._identity)

    @property
    def runtime_metadata(self) -> dict[str, Any]:
        return dict(self._runtime_metadata)

    def case_token_plan(self, case: CaseSpec) -> dict[str, Any]:
        assistant_ids = self._encode(case.assistant_text)
        second_ids = self._encode(case.second_assistant_text or "") if case.second_crop_fraction is not None else None
        payload = {
            "case_id": case.id,
            "assistant_token_ids": assistant_ids,
            "assistant_token_hash": token_ids_hash(assistant_ids),
            "second_assistant_token_ids": second_ids,
            "second_assistant_token_hash": token_ids_hash(second_ids) if second_ids else None,
            "all_non_eot": self.parts.eot_token_id not in assistant_ids
            and (second_ids is None or self.parts.eot_token_id not in second_ids),
        }
        if not payload["all_non_eot"]:
            raise RuntimeError(f"{case.id}: frozen assistant fixture contains structural EOT")
        payload["plan_hash"] = config_hash(payload)
        return payload

    def negative_control(self) -> dict[str, Any]:
        torch = self.torch
        correct_tensor = torch.arange(32, dtype=torch.float32).reshape(1, 2, 4, 4)
        wrong_tensor = correct_tensor[..., :3, :].clone()
        correct_cache = self.DynamicCache.from_legacy_cache(((correct_tensor, correct_tensor.clone()),))
        wrong_cache = self.DynamicCache.from_legacy_cache(((wrong_tensor, wrong_tensor.clone()),))
        correct = layer_manifest(correct_cache)
        wrong = layer_manifest(wrong_cache)
        detected = not manifests_equal(correct, wrong)
        if not detected:
            raise RuntimeError("Disposable wrong-length negative control was not detected")
        return {
            "kind": "wrong_keep_length_manifest",
            "disposable": True,
            "correct_keep_length": 4,
            "wrong_keep_length": 3,
            "detected": True,
            "positive_control_metadata": {
                "wrong_crop_length_would_be_detected": True,
                "mechanism": "shape and aggregate manifest mismatch",
            },
        }

    def _encode(self, text: str) -> list[int]:
        return [int(value) for value in self.tokenizer.encode(text, add_special_tokens=False)]

    def _cache_ids(self, cache: Any) -> list[int]:
        return [int(value) for value in cache.token_ids]

    def _context_user_text(self, case: CaseSpec) -> str:
        seeds = (" context", " reference", " detail", " x", "\nContext item.")
        for seed in seeds:
            seed_ids = self._encode(seed)
            estimate = max(0, (case.context_tokens - len(self._encode(case.user_prompt))) // max(len(seed_ids), 1))
            for repeats in range(max(0, estimate - 40), estimate + 41):
                text = seed * repeats + "\n" + case.user_prompt
                ids = apply_chat_ids(
                    self.tokenizer,
                    [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": text}],
                    add_generation_prompt=True,
                )
                if len(ids) == case.context_tokens:
                    return text
        raise RuntimeError(f"Cannot build exact {case.context_tokens}-token context for {case.id}")

    def _initial_cache(self, user_text: str) -> Any:
        pre = self.llm.cache_prompt(user_text, is_end=True, system_prompt=self.system_prompt)
        cache = self.llm.to_accum_cache(pre)
        canonical = apply_chat_ids(
            self.tokenizer,
            [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": user_text}],
            add_generation_prompt=True,
        )
        if self._cache_ids(cache) != canonical:
            raise RuntimeError("Initial production cache differs from canonical chat token IDs")
        return cache

    def _append_fixture_tokenwise(self, cache: Any, token_ids: Sequence[int]) -> list[dict[str, Any]]:
        ids = [int(value) for value in token_ids]
        if not ids or self.parts.eot_token_id in ids:
            raise ValueError("Assistant fixture must contain non-EOT token IDs")
        original_decode = self.llm._decode_logits
        original_prefill = self.llm._prefill_ids_p2
        selected = iter(ids)
        calls: list[list[int]] = []
        decoded: list[int] = []

        def controlled_decode(logits, temperature, top_p, repetition_penalty):
            del logits, temperature, top_p, repetition_penalty
            return self.torch.tensor([[next(selected)]], dtype=self.torch.long, device=self.device)

        def counted_prefill(target, chunk):
            values = [int(value) for value in chunk]
            calls.append(values)
            return original_prefill(target, values)

        before = cache.seq_length
        self.llm._decode_logits = controlled_decode
        self.llm._prefill_ids_p2 = counted_prefill
        try:
            for _ in self.llm.generate_accumulating(
                cache,
                max_new_tokens=len(ids),
                temperature=0.0,
                top_p=1.0,
                repetition_penalty=1.0,
                on_token_decoded=lambda text, index, token_id: decoded.append(int(token_id)),
            ):
                pass
        finally:
            self.llm._decode_logits = original_decode
            self.llm._prefill_ids_p2 = original_prefill
        if decoded != ids or calls != [[value] for value in ids]:
            raise RuntimeError("Fixture did not execute one _prefill_ids_p2 forward per selected token")
        return [
            ledger_entry(
                ordinal=index,
                arm="production",
                operation="generate_accumulating_token",
                token_ids=[token],
                before_length=before + index,
                after_length=before + index + 1,
                api="StreamLLMInference.generate_accumulating->_prefill_ids_p2",
            )
            for index, token in enumerate(ids)
        ]

    def _clone_prefix_cache(self, source: Any, keep: int) -> Any:
        layers = []
        for layer in source.to_legacy_cache():
            key, value = layer[:2]
            layers.append((key[..., :keep, :].clone(), value[..., :keep, :].clone()))
        return self.DynamicCache.from_legacy_cache(tuple(layers))

    def _oracle(self, source: Any, keep: int, token_ids: Sequence[int], mask: Any) -> Any:
        return SimpleNamespace(
            past_key_values=self._clone_prefix_cache(source, keep),
            attention_mask=mask[:, :keep].clone(),
            token_ids=[int(value) for value in token_ids[:keep]],
            seq_length=keep,
            next_token_logits=None,
        )

    def _direct_append(self, oracle: Any, token_ids: Sequence[int]) -> None:
        ids_list = [int(value) for value in token_ids]
        ids = self.torch.tensor([ids_list], dtype=self.torch.long, device=self.device)
        attention = self.torch.cat(
            [oracle.attention_mask, self.torch.ones(ids.shape, dtype=oracle.attention_mask.dtype, device=self.device)],
            dim=-1,
        )
        positions = self.torch.arange(
            oracle.seq_length, oracle.seq_length + len(ids_list), dtype=self.torch.long, device=self.device
        ).unsqueeze(0)
        with self.torch.no_grad():
            output = self.model(
                input_ids=ids,
                attention_mask=attention,
                position_ids=positions,
                past_key_values=oracle.past_key_values,
                use_cache=True,
                return_dict=True,
            )
        oracle.past_key_values = output.past_key_values
        oracle.attention_mask = attention
        oracle.token_ids.extend(ids_list)
        oracle.seq_length += len(ids_list)
        oracle.next_token_logits = output.logits[:, -1, :]

    @staticmethod
    def _enum_name(value: Any) -> str | None:
        if value is None:
            return None
        return str(getattr(value, "name", getattr(value, "value", value))).upper()

    def _production_state(self, cache: Any) -> dict[str, Any]:
        return {
            "role_phase": self._enum_name(cache.role_phase),
            "generation_end_reason": self._enum_name(cache.generation_end_reason),
            "assistant_content_start": cache.assistant_content_start,
            "assistant_token_count": len(cache.assistant_token_ids),
            "seq_length": int(cache.seq_length),
            "mask_length": int(cache.attention_mask.shape[1]),
            "kv_length": int(cache.past_key_values.get_seq_length()),
            "token_ledger_length": len(cache.token_ids),
        }

    @staticmethod
    def _expected_state(operation: str, after_length: int) -> dict[str, Any]:
        if operation in {"reopen_user_role", "prefill_user_text"}:
            phase = "USER_OPEN"
            content_start = None
        elif operation == "open_assistant_role":
            phase = "ASSISTANT_OPEN"
            content_start = after_length
        else:
            raise ValueError(operation)
        return {
            "role_phase": phase,
            "generation_end_reason": "NONE",
            "assistant_content_start": content_start,
            "assistant_token_count": 0,
            "seq_length": after_length,
            "mask_length": after_length,
            "kv_length": after_length,
            "token_ledger_length": after_length,
        }

    def _compare_cache(self, production: Any, oracle: Any, keep: int, original_prefix: Mapping[str, Any]) -> dict[str, Any]:
        production_manifest = layer_manifest(production.past_key_values)
        oracle_manifest = layer_manifest(oracle.past_key_values)
        production_prefix = layer_manifest(production.past_key_values, limit=keep)
        oracle_prefix = layer_manifest(oracle.past_key_values, limit=keep)
        production_layers = production.past_key_values.to_legacy_cache()
        oracle_layers = oracle.past_key_values.to_legacy_cache()
        kv_exact = len(production_layers) == len(oracle_layers) and all(
            self.torch.equal(left, right)
            for production_layer, oracle_layer in zip(production_layers, oracle_layers)
            for left, right in zip(production_layer[:2], oracle_layer[:2])
        )
        logits_exact = (
            production.next_token_logits is not None
            and oracle.next_token_logits is not None
            and self.torch.equal(production.next_token_logits, oracle.next_token_logits)
        )
        masks_exact = self.torch.equal(production.attention_mask, oracle.attention_mask)
        token_exact = self._cache_ids(production) == oracle.token_ids
        retained_exact = (
            manifests_equal(production_prefix, original_prefix)
            and manifests_equal(oracle_prefix, original_prefix)
        )
        return {
            "production_manifest": production_manifest,
            "oracle_manifest": oracle_manifest,
            "production_prefix_manifest": production_prefix,
            "oracle_prefix_manifest": oracle_prefix,
            "production_logits_sha256": (
                hashlib.sha256(production.next_token_logits.detach().contiguous().cpu().view(self.torch.uint8).numpy().tobytes()).hexdigest()
                if production.next_token_logits is not None else None
            ),
            "oracle_logits_sha256": (
                hashlib.sha256(oracle.next_token_logits.detach().contiguous().cpu().view(self.torch.uint8).numpy().tobytes()).hexdigest()
                if oracle.next_token_logits is not None else None
            ),
            "kv_exact": kv_exact,
            "logits_exact": logits_exact,
            "masks_exact": masks_exact,
            "token_ids_exact": token_exact,
            "retained_prefix_hash_exact": retained_exact,
            "passed": all((kv_exact, logits_exact, masks_exact, token_exact, retained_exact)),
            "errors": [],
        }

    def _recovery_chunks(self, case: CaseSpec, *, second: bool) -> list[dict[str, Any]]:
        if second:
            return [{"operation": "reopen_user_role", "token_ids": list(self.parts.assistant_eot) + list(self.parts.assistant_to_user)}]
        chunks: list[dict[str, Any]] = []
        if case.scenario != "speculation_full_invalidation":
            chunks.append({"operation": "reopen_user_role", "token_ids": list(self.parts.assistant_eot) + list(self.parts.assistant_to_user)})
        if case.next_user is not None:
            chunks.append({"operation": "prefill_user_text", "token_ids": self._encode(case.next_user)})
            chunks.append({"operation": "open_assistant_role", "token_ids": list(self.parts.user_to_assistant)})
        return chunks

    def _production_event(self, cache: Any, operation: str, case: CaseSpec) -> None:
        if operation == "reopen_user_role":
            self.llm.reopen_user_role(cache)
        elif operation == "prefill_user_text":
            self.llm.prefill_user_text(cache, case.next_user or "")
        elif operation == "open_assistant_role":
            self.llm.open_assistant_role(cache)
        else:
            raise ValueError(operation)

    def _crop_event(
        self,
        case: CaseSpec,
        cache: Any,
        keep: int,
        *,
        event_index: int,
        second: bool,
        crop_metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        pre_len = cache.seq_length
        pre_ids = self._cache_ids(cache)
        pre_mask = cache.attention_mask.clone()
        pre_prefix = layer_manifest(cache.past_key_values, limit=keep)
        pre_layers = cache.past_key_values.to_legacy_cache()
        oracle = self._oracle(cache.past_key_values, keep, pre_ids, pre_mask)
        oracle_layers_before = oracle.past_key_values.to_legacy_cache()
        pre_oracle_torch_equal = len(pre_layers) == len(oracle_layers_before) and all(
            self.torch.equal(source[..., :keep, :], cloned)
            for source_layer, oracle_layer in zip(pre_layers, oracle_layers_before)
            for source, cloned in zip(source_layer[:2], oracle_layer[:2])
        )
        oracle_manifest = layer_manifest(oracle.past_key_values)
        self.llm.crop_to_token(cache, keep)
        post_layers = cache.past_key_values.to_legacy_cache()
        post_oracle_torch_equal = len(post_layers) == len(oracle_layers_before) and all(
            self.torch.equal(production, cloned)
            for production_layer, oracle_layer in zip(post_layers, oracle_layers_before)
            for production, cloned in zip(production_layer[:2], oracle_layer[:2])
        )
        post = layer_manifest(cache.past_key_values)
        post_crop_state = self._production_state(cache)
        post_crop_oracle_state = {
            "seq_length": oracle.seq_length,
            "mask_length": int(oracle.attention_mask.shape[1]),
            "kv_length": int(oracle.past_key_values.get_seq_length()),
            "token_ledger_length": len(oracle.token_ids),
        }
        post_crop_lengths_exact = (
            post_crop_state["seq_length"]
            == post_crop_state["mask_length"]
            == post_crop_state["kv_length"]
            == post_crop_state["token_ledger_length"]
            == post_crop_oracle_state["seq_length"]
            == post_crop_oracle_state["mask_length"]
            == post_crop_oracle_state["kv_length"]
            == post_crop_oracle_state["token_ledger_length"]
            == keep
        )
        post_crop_mask_exact = self.torch.equal(cache.attention_mask, oracle.attention_mask)
        post_crop_token_ids_exact = self._cache_ids(cache) == oracle.token_ids == pre_ids[:keep]
        chunks = self._recovery_chunks(case, second=second)
        production_ledger = []
        oracle_ledger = []
        checks = []
        for ordinal, chunk in enumerate(chunks):
            ids = list(chunk["token_ids"])
            before = cache.seq_length
            self._production_event(cache, chunk["operation"], case)
            self._direct_append(oracle, ids)
            production_ledger.append(
                ledger_entry(
                    ordinal=ordinal, arm="production", operation=chunk["operation"], token_ids=ids,
                    before_length=before, after_length=cache.seq_length,
                    api=f"StreamLLMInference.{chunk['operation']}",
                )
            )
            oracle_ledger.append(
                ledger_entry(
                    ordinal=ordinal, arm="oracle", operation=chunk["operation"], token_ids=ids,
                    before_length=before, after_length=oracle.seq_length, api="direct_model_forward",
                )
            )
            check = self._compare_cache(cache, oracle, keep, pre_prefix)
            production_state = self._production_state(cache)
            expected_state = self._expected_state(chunk["operation"], cache.seq_length)
            state_exact = production_state == expected_state
            check.update(
                {
                    "ordinal": ordinal,
                    "operation": chunk["operation"],
                    "token_ids": ids,
                    "token_hash": token_ids_hash(ids),
                    "production_state": production_state,
                    "expected_state": expected_state,
                    "production_state_exact": state_exact,
                }
            )
            check["passed"] = check["passed"] and state_exact
            if not check["passed"]:
                check["errors"].append("production/oracle recovery or role/end state differs")
            checks.append(check)
        shapes_exact = all(
            left[side]["shape"] == right[side]["shape"]
            for left, right in zip(post["layers"], oracle_manifest["layers"])
            for side in ("key", "value")
        )
        dtypes_exact = all(
            left[side]["dtype"] == right[side]["dtype"]
            for left, right in zip(post["layers"], oracle_manifest["layers"])
            for side in ("key", "value")
        )
        devices_exact = all(
            left[side]["device"] == right[side]["device"]
            for left, right in zip(post["layers"], oracle_manifest["layers"])
            for side in ("key", "value")
        )
        pre_oracle_equal = manifests_equal(pre_prefix, oracle_manifest) and pre_oracle_torch_equal
        post_pre_equal = manifests_equal(post, pre_prefix) and post_oracle_torch_equal
        post_oracle_equal = manifests_equal(post, oracle_manifest) and post_oracle_torch_equal
        final_ids = self._cache_ids(cache)
        canonical_ids = list(pre_ids[:keep])
        assistant_eot_positions: list[int] = []
        reopen_chunks_valid = True
        for chunk in chunks:
            chunk_ids = [int(value) for value in chunk["token_ids"]]
            if chunk["operation"] == "reopen_user_role":
                offsets = [
                    offset for offset, value in enumerate(chunk_ids)
                    if value == self.parts.eot_token_id
                ]
                reopen_chunks_valid = reopen_chunks_valid and len(offsets) == 1
                assistant_eot_positions.extend(len(canonical_ids) + offset for offset in offsets)
            canonical_ids.extend(chunk_ids)
        assistant_boundaries = sum(chunk["operation"] == "reopen_user_role" for chunk in chunks)
        canonical = {
            "token_ids": canonical_ids,
            "eot_positions": assistant_eot_positions,
            "assistant_boundaries": assistant_boundaries,
        }
        canonical["token_hash"] = token_ids_hash(canonical["token_ids"])
        canonical["unique_eot"] = reopen_chunks_valid and len(assistant_eot_positions) == assistant_boundaries
        canonical["role_boundary_exact"] = final_ids == canonical["token_ids"]
        errors = []
        exact_values = (
            cache.seq_length == keep + sum(len(chunk["token_ids"]) for chunk in chunks),
            pre_oracle_equal,
            post_pre_equal,
            post_oracle_equal,
            shapes_exact,
            dtypes_exact,
            devices_exact,
            self.torch.equal(cache.attention_mask, oracle.attention_mask),
            final_ids == oracle.token_ids == canonical["token_ids"],
            all(check["logits_exact"] for check in checks) if checks else True,
            all(check["retained_prefix_hash_exact"] for check in checks) if checks else True,
            True,
            canonical["unique_eot"],
            canonical["role_boundary_exact"],
            post_crop_lengths_exact,
            post_crop_mask_exact,
            post_crop_token_ids_exact,
            all(check["passed"] for check in checks),
        )
        if not all(exact_values):
            errors.append("one or more exact crop/recovery gates failed")
        return {
            "event_id": "crop_2" if second else "crop_1",
            "event_index": event_index,
            "pre_crop_length": pre_len,
            "keep_length": keep,
            "no_op": keep == pre_len,
            **dict(crop_metadata),
            "pre_crop_token_ids": pre_ids,
            "pre_crop_token_hash": token_ids_hash(pre_ids),
            "retained_token_ids": pre_ids[:keep],
            "retained_token_hash": token_ids_hash(pre_ids[:keep]),
            "pre_prefix_manifest": pre_prefix,
            "post_production_manifest": post,
            "oracle_manifest": oracle_manifest,
            "post_crop_state": post_crop_state,
            "post_crop_oracle_state": post_crop_oracle_state,
            "post_crop_lengths_exact": post_crop_lengths_exact,
            "post_crop_mask_exact": post_crop_mask_exact,
            "post_crop_token_ids_exact": post_crop_token_ids_exact,
            "production_event_ledger": production_ledger,
            "oracle_event_ledger": oracle_ledger,
            "expected_recovery_chunks": chunks,
            "recovery_checks": checks,
            "final_token_ids": final_ids,
            "final_token_hash": token_ids_hash(final_ids),
            "canonical_ledger": canonical,
            "keep_length_exact": exact_values[0],
            "pre_prefix_equals_oracle": pre_oracle_equal,
            "post_equals_pre_prefix": post_pre_equal,
            "post_equals_oracle": post_oracle_equal,
            "shapes_exact": shapes_exact,
            "dtypes_exact": dtypes_exact,
            "devices_exact": devices_exact,
            "mask_exact": exact_values[7],
            "token_ids_exact": exact_values[8],
            "logits_exact": exact_values[9],
            "retained_prefix_hash_exact": exact_values[10],
            "negative_control_detected": True,
            "passed": not errors,
            "errors": errors,
        }

    def _fragment_partition(self, case: CaseSpec, assistant_ids: Sequence[int]) -> list[list[int]]:
        partition: list[list[int]] = []
        previous = 0
        accumulated = ""
        for fragment in case.fragments:
            accumulated += fragment
            end = len(self._encode(accumulated))
            ids = [int(value) for value in assistant_ids[previous:end]]
            if not ids:
                raise RuntimeError(f"{case.id}: empty fragment token partition")
            partition.append(ids)
            previous = end
        if previous != len(assistant_ids) or [value for ids in partition for value in ids] != list(assistant_ids):
            raise RuntimeError(f"{case.id}: fragment token partition differs from assistant fixture")
        return partition

    def _first_crop_metadata(self, case: CaseSpec, cache: Any, assistant_ids: Sequence[int]) -> tuple[int, dict[str, Any]]:
        content_start = int(cache.assistant_content_start)
        role_start = int(cache.assistant_role_start)
        fragments = self._fragment_partition(case, assistant_ids)
        if case.scenario == "reply_tail_noop":
            keep, semantics = cache.seq_length, "reply_tail_noop"
        elif case.scenario == "speculation_full_invalidation":
            keep, semantics = role_start, "speculation_full_invalidation"
        elif case.retain_fragment_count == 0:
            keep, semantics = content_start, "empty_assistant_turn_p0"
        else:
            keep = content_start + sum(len(ids) for ids in fragments[: case.retain_fragment_count])
            semantics = "retained_fragment_prefix"
        return keep, {
            "crop_target_semantics": semantics,
            "assistant_role_start": role_start,
            "assistant_content_start": content_start,
            "retain_fragment_count": case.retain_fragment_count,
            "fragment_token_ids": fragments,
            "second_assistant_token_ids": None,
            "second_crop_fraction": None,
        }

    def run_case(self, case: CaseSpec) -> dict[str, Any]:
        user_text = self._context_user_text(case)
        token_plan = self.case_token_plan(case)
        cache = self._initial_cache(user_text)
        assistant_ids = token_plan["assistant_token_ids"]
        fixture_ledger = self._append_fixture_tokenwise(cache, assistant_ids)
        first_keep, first_metadata = self._first_crop_metadata(case, cache, assistant_ids)
        events = [
            self._crop_event(
                case,
                cache,
                first_keep,
                event_index=0,
                second=False,
                crop_metadata=first_metadata,
            )
        ]
        second_fixture = None
        if case.second_crop_fraction is not None:
            second_ids = token_plan["second_assistant_token_ids"] or []
            second_fixture = self._append_fixture_tokenwise(cache, second_ids)
            second_content_start = int(cache.assistant_content_start)
            retained = max(1, min(len(second_ids), math.floor(len(second_ids) * case.second_crop_fraction)))
            second_keep = second_content_start + retained
            events.append(
                self._crop_event(
                    case,
                    cache,
                    second_keep,
                    event_index=1,
                    second=True,
                    crop_metadata={
                        "crop_target_semantics": "second_fraction_floor_clamp",
                        "assistant_role_start": int(cache.assistant_role_start),
                        "assistant_content_start": second_content_start,
                        "retain_fragment_count": None,
                        "fragment_token_ids": None,
                        "second_assistant_token_ids": second_ids,
                        "second_crop_fraction": case.second_crop_fraction,
                    },
                )
            )
        errors = [
            f"{event['event_id']}: {message}"
            for event in events
            for message in event["errors"]
        ]
        return {
            "case_id": case.id,
            "context_tokens_target": case.context_tokens,
            "context_tokens_actual": case.context_tokens,
            "context_class": case.context_class,
            "scenario": case.scenario,
            "termination": case.termination,
            "prior_v2_probe_reused": True,
            "termination_probe_rerun": False,
            "token_plan_hash": token_plan["plan_hash"],
            "fixture": {
                "assistant_token_ids": assistant_ids,
                "assistant_token_hash": token_ids_hash(assistant_ids),
                "second_assistant_token_ids": token_plan["second_assistant_token_ids"],
                "second_assistant_token_hash": token_plan["second_assistant_token_hash"],
                "all_non_eot": token_plan["all_non_eot"],
                "generate_api": "StreamLLMInference.generate_accumulating",
                "prefill_ids_p2_calls": len(fixture_ledger),
                "one_token_per_forward": all(entry["token_count"] == 1 for entry in fixture_ledger),
                "event_ledger": fixture_ledger,
                "second_assistant_event_ledger": second_fixture,
            },
            "crop_events": events,
            "passed": not errors and all(event["passed"] for event in events),
            "errors": errors,
        }


def make_backend(kind: str, *, model_path: str | None, device: str, seed: int) -> CropIntegrityBackend:
    if kind == "fake":
        return FakeBackend(seed=seed)
    if kind == "transformers":
        if not model_path:
            raise ValueError("Transformers C2 v3 runtime requires --model")
        return TransformersBackend(model_path, device=device)
    raise ValueError(f"Unknown runtime: {kind}")
