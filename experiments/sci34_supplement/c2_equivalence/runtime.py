"""Fake and Transformers runtimes for crop/recovery versus clean re-prefill C2."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Mapping, Protocol, Sequence

from experiments.sci34_supplement.common import canonical_json, config_hash
from experiments.sci34_supplement.c2_equivalence.canonical_chat import (
    CanonicalSequence,
    ChatTemplateParts,
    append_assistant_ids,
    apply_chat_ids,
    build_initial_open_sequence,
    first_mismatch,
    token_ids_hash,
)
from experiments.sci34_supplement.c2_equivalence.protocol import (
    CONTINUATION_TOKENS,
    EOS_AT_CAP_MAX_NEW_TOKENS,
    EXPECTED_DTYPE,
    EXPECTED_MODEL_ARCHITECTURE,
    EXPECTED_MODEL_ARTIFACT_HASH,
    EXPECTED_MODEL_TYPE,
    MAX_TOKENS_PROBE_BUDGET,
    NATURAL_EOS_MAX_NEW_TOKENS,
    SYSTEM_PROMPT,
    TERMINATION_PROBE_SCHEMA_VERSION,
    TOP_K,
    CaseSpec,
)
from experiments.sci34_supplement.e1e2_confirmatory.strong_identity import strong_model_identity


class EquivalenceBackend(Protocol):
    @property
    def identity(self) -> Mapping[str, Any]: ...

    @property
    def runtime_metadata(self) -> Mapping[str, Any]: ...

    def run_case(self, case: CaseSpec) -> dict[str, Any]: ...


def _unit(*parts: object) -> float:
    digest = hashlib.sha256(canonical_json([str(value) for value in parts]).encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


@dataclass
class FakeBackend:
    """Pure-CPU deterministic state model used only by the smoke workflow."""

    seed: int = 20260902

    @property
    def identity(self) -> dict[str, Any]:
        payload = {"kind": "fake", "seed": self.seed, "version": 1}
        payload["content_identity_hash"] = hashlib.sha256(
            canonical_json(payload).encode()
        ).hexdigest()
        return payload

    @property
    def runtime_metadata(self) -> dict[str, Any]:
        return {
            "resolved_dtype": "fake-float32",
            "attention_backend": "fake-eager",
            "tokenizer_class": "FakeTokenizer",
            "chat_template_sha256": "f" * 64,
            "eos_token_id": 2,
            "eot_token_id": 2,
        }

    @staticmethod
    def _fake_scenario_execution(case: CaseSpec) -> dict[str, Any]:
        applies = case.scenario in {"crop_pending_eot", "reply_tail_noop"}
        no_op = case.scenario == "reply_tail_noop"
        return {
            "schema_version": 1,
            "scenario": case.scenario,
            "applies": applies,
            "execution": "synthetic_state_machine",
            "generate_api": (
                "StreamLLMInference.generate_accumulating" if applies else None
            ),
            "forced_decode_token": "EOT" if applies else None,
            "generate_max_new_tokens": 1 if applies else None,
            "pending_before_crop": True if applies else None,
            "eot_in_full_ledger_before_crop": False if applies else None,
            "eot_in_content_ledger_before_crop": False if applies else None,
            "crop_target": (
                "current_seq" if no_op else "retained_boundary" if applies else None
            ),
            "crop_was_noop": no_op if applies else None,
            "no_op_preserved_pending": no_op if applies else None,
            "pending_after_crop": (True if no_op else False) if applies else None,
            "pending_cleared_by_crop": (False if no_op else True) if applies else None,
            "reopen_called": applies,
            "passed": True,
            "errors": [],
        }

    def run_case(self, case: CaseSpec) -> dict[str, Any]:
        cap = {
            "natural_eos": NATURAL_EOS_MAX_NEW_TOKENS,
            "eos_at_cap": EOS_AT_CAP_MAX_NEW_TOKENS,
            "max_tokens": MAX_TOKENS_PROBE_BUDGET,
        }[case.termination]
        if case.termination == "natural_eos":
            content = [7001, 7002, 7003]
            observed = "EOS"
            eos_step = len(content) + 1
            role_phase = "ASSISTANT_EOT_PENDING"
            controlled = False
        elif case.termination == "eos_at_cap":
            content = list(range(7101, 7101 + cap - 1))
            observed = "EOS"
            eos_step = cap
            role_phase = "ASSISTANT_EOT_PENDING"
            controlled = True
        else:
            content = list(range(7201, 7201 + cap))
            observed = "MAX_TOKENS"
            eos_step = None
            role_phase = "ASSISTANT_OPEN"
            controlled = False
        probe_errors: list[str] = []
        scenario_execution = self._fake_scenario_execution(case)
        termination_probe = {
            "schema_version": TERMINATION_PROBE_SCHEMA_VERSION,
            "declared": case.termination,
            "mode": "controlled_logits_fixture" if controlled else "real_greedy",
            "controlled": controlled,
            "generate_api": "StreamLLMInference.generate_accumulating",
            "execution": "synthetic_state_machine",
            "temperature": 0.0,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
            "cap": cap,
            "observed_end_reason": observed,
            "selected_token_ids": [*content, 2] if observed == "EOS" else list(content),
            "selected_token_hash": token_ids_hash([*content, 2] if observed == "EOS" else content),
            "selected_token_count": len(content) + (1 if observed == "EOS" else 0),
            "content_token_ids": content,
            "content_token_hash": token_ids_hash(content),
            "content_token_count": len(content),
            "eos_step": eos_step,
            "eos_at_cap": eos_step == cap,
            "eot_token_id": 2,
            "eot_in_kv": False,
            "eot_in_full_ledger": False,
            "eot_in_content_ledger": False,
            "prefix_seq_length": case.context_tokens,
            "post_seq_length": case.context_tokens + len(content),
            "role_phase": role_phase,
            "fixture_token_ids": [*content, 2] if controlled else None,
            "fixture_description": (
                "deterministic fake decode-token fixture mirroring generate_accumulating EOS semantics"
                if controlled else None
            ),
            "passed": not probe_errors,
            "errors": probe_errors,
        }
        checkpoints: list[dict[str, Any]] = []
        for ordinal, checkpoint_name in enumerate(case.checkpoints):
            length = case.context_tokens + 7 + ordinal * 11
            zero_retain_semantics = (
                case.scenario if case.retain_fragment_count == 0 else None
            )
            assistant_boundaries = (
                0 if case.scenario == "speculation_full_invalidation" else 1
            )
            tokens = (
                [101, *range(1000, 1000 + length - 3), 2, 202]
                if assistant_boundaries
                else [101, *range(1000, 1000 + length - 2), 202]
            )
            logits = [
                float(_unit(self.seed, case.id, checkpoint_name, index))
                for index in range(16)
            ]
            top5 = sorted(range(len(logits)), key=lambda index: logits[index], reverse=True)[:5]
            continuation = [top5[0]] * CONTINUATION_TOKENS
            phase = "ASSISTANT_OPEN" if checkpoint_name == "next_assistant" else "USER_OPEN"
            state = {
                "seq_length": len(tokens),
                "mask_length": len(tokens),
                "kv_length": len(tokens),
                "ledger_length": len(tokens),
                "assistant_content_length": 0,
                "assistant_content_span_exact": True,
                "role_phase": phase,
                "expected_role_phase": phase,
                "role_phase_exact": True,
                "end_reason": "none",
                "lengths_exact": True,
            }
            checkpoints.append(
                {
                    "checkpoint": checkpoint_name,
                    "canonical": {
                        "token_ids": tokens,
                        "token_hash": token_ids_hash(tokens),
                        "token_count": len(tokens),
                        "eot_positions": (
                            [len(tokens) - 2] if assistant_boundaries else []
                        ),
                        "boundaries": {
                            "zero_retain_semantics": zero_retain_semantics,
                            "assistant_eot_positions": (
                                [len(tokens) - 2] if assistant_boundaries else []
                            ),
                        },
                    },
                    "path": {
                        "token_ids": list(tokens),
                        "token_hash": token_ids_hash(tokens),
                        "token_count": len(tokens),
                        "eot_positions": (
                            [len(tokens) - 2] if assistant_boundaries else []
                        ),
                    },
                    "token_ids_exact": True,
                    "first_token_mismatch": None,
                    "state": {"path": state, "canonical": dict(state), "exact": True},
                    "unique_eot": {
                        "ok": True,
                        "path_positions": (
                            [len(tokens) - 2] if assistant_boundaries else []
                        ),
                        "canonical_positions": (
                            [len(tokens) - 2] if assistant_boundaries else []
                        ),
                        "all_path_positions": (
                            [len(tokens) - 2] if assistant_boundaries else []
                        ),
                        "all_canonical_positions": (
                            [len(tokens) - 2] if assistant_boundaries else []
                        ),
                        "assistant_boundaries": assistant_boundaries,
                    },
                    "next_token": {
                        "path_top1": top5[0],
                        "canonical_top1": top5[0],
                        "top1_exact": True,
                        "path_top5": top5,
                        "canonical_top5": top5,
                        "top5_overlap": 5,
                    },
                    "logit_diff_float32": {
                        "max_abs": 0.0,
                        "mean_abs": 0.0,
                        "rms": 0.0,
                    },
                    "continuation": {
                        "tokens": CONTINUATION_TOKENS,
                        "continuation_source": "actual_crop_cache",
                        "canonical_source": "clean_prefill_cache",
                        "checkpoint_state_captured_before_mutation": True,
                        "path_token_ids": continuation,
                        "canonical_token_ids": list(continuation),
                        "path_hash": token_ids_hash(continuation),
                        "canonical_hash": token_ids_hash(continuation),
                        "exact": True,
                        "first_divergence": None,
                    },
                    "termination_probe": dict(termination_probe),
                    "scenario_execution": dict(scenario_execution),
                    "passed": True,
                    "errors": [],
                }
            )
        return {
            "case_id": case.id,
            "context_tokens_target": case.context_tokens,
            "context_tokens_actual": case.context_tokens,
            "context_class": case.context_class,
            "scenario": case.scenario,
            "termination": case.termination,
            "controlled_fixture": case.controlled_fixture,
            "source": case.source,
            "termination_probe": termination_probe,
            "scenario_execution": scenario_execution,
            "checkpoints": checkpoints,
            "passed": termination_probe["passed"] and scenario_execution["passed"],
            "errors": [
                *termination_probe["errors"],
                *scenario_execution["errors"],
            ],
        }


class TransformersBackend:
    """Exercises StreamLLM crop/recovery and an independent token-ID clean oracle."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda:0",
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        import torch
        from src.llm.stream_llm_inference import StreamLLMInference

        self.torch = torch
        self.device = device
        self.system_prompt = system_prompt
        self.llm = StreamLLMInference(model_name=model_path, device=device, eval_mode=False)
        self.model = self.llm.model
        self.tokenizer = self.llm.tokenizer
        self.parts = ChatTemplateParts.from_tokenizer(self.tokenizer, system_prompt)
        self._identity = strong_model_identity(model_path)
        artifact_payload = {
            "schema_version": self._identity["schema_version"],
            "file_count": self._identity["file_count"],
            "total_bytes": self._identity["total_bytes"],
            "files": self._identity["files"],
        }
        artifact_hash = config_hash(artifact_payload)
        model_type = str(getattr(self.model.config, "model_type", ""))
        architectures = list(getattr(self.model.config, "architectures", None) or [])
        first_parameter = next(self.model.parameters(), None)
        resolved_dtype = str(first_parameter.dtype) if first_parameter is not None else None
        if artifact_hash != EXPECTED_MODEL_ARTIFACT_HASH:
            raise RuntimeError(
                "Formal C2 model snapshot differs from the D-017 accepted Qwen2-7B artifact: "
                f"{artifact_hash} != {EXPECTED_MODEL_ARTIFACT_HASH}"
            )
        if model_type != EXPECTED_MODEL_TYPE or EXPECTED_MODEL_ARCHITECTURE not in architectures:
            raise RuntimeError(
                "C2 requires Qwen2ForCausalLM/Qwen2 model identity; "
                f"got model_type={model_type!r}, architectures={architectures!r}"
            )
        if resolved_dtype != EXPECTED_DTYPE:
            raise RuntimeError(f"C2 requires {EXPECTED_DTYPE}, got {resolved_dtype}")
        self._runtime_metadata = {
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
            "eos_token_id": self.parts.eos_token_id,
            "eot_token_id": self.parts.eot_token_id,
            "pad_token_id": self.parts.pad_token_id,
            "model_max_position_embeddings": getattr(
                self.model.config, "max_position_embeddings", None
            ),
        }
        required = (
            "crop_to_token",
            "generate_accumulating",
            "reopen_user_role",
            "open_assistant_role",
            "prefill_user_text",
        )
        missing = [name for name in required if not hasattr(self.llm, name)]
        if missing:
            raise RuntimeError(f"StreamLLMInference lacks C2-required APIs: {missing}")
        if not hasattr(self.llm, "prefill_assistant_text"):
            raise RuntimeError("C2 requires the token-ledger-aware prefill_assistant_text API")

    @property
    def identity(self) -> dict[str, Any]:
        return dict(self._identity)

    @property
    def runtime_metadata(self) -> dict[str, Any]:
        return dict(self._runtime_metadata)

    def _encode(self, text: str) -> list[int]:
        return [int(value) for value in self.tokenizer.encode(text, add_special_tokens=False)]

    def _context_user_text(self, case: CaseSpec) -> str:
        target_content = (
            case.context_tokens
            - len(self.parts.prefix_to_user_content)
            - len(self.parts.user_to_assistant)
        )
        if target_content <= 0:
            raise ValueError(f"Context target is too short for the chat structure: {case.id}")
        seeds = (
            " context",
            " reference",
            " detail",
            " x",
            "\nContext item.",
        )
        for seed in seeds:
            seed_ids = self._encode(seed)
            if not seed_ids:
                continue
            estimate = max(0, (target_content - len(self._encode(case.user_prompt))) // len(seed_ids))
            for repeats in range(max(0, estimate - 32), estimate + 33):
                text = seed * repeats + "\n" + case.user_prompt
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": text},
                ]
                if len(apply_chat_ids(self.tokenizer, messages, add_generation_prompt=True)) == case.context_tokens:
                    return text
        raise ValueError(
            f"Could not construct an exactly {case.context_tokens}-token canonical context for {case.id}"
        )

    def _cache_token_ids(self, cache: Any) -> list[int]:
        for name in ("token_ids", "full_token_ids", "token_ledger"):
            value = getattr(cache, name, None)
            if value is not None:
                if hasattr(value, "tolist"):
                    value = value.tolist()
                if value and isinstance(value[0], list):
                    value = value[0]
                return [int(item) for item in value]
        raise RuntimeError("AccumKVCache does not expose its complete token ledger")

    @staticmethod
    def _enum_value(value: Any) -> str | None:
        if value is None:
            return None
        return str(getattr(value, "name", getattr(value, "value", value)))

    def _seq_length(self, cache: Any) -> int:
        return int(getattr(cache, "seq_length", cache.attention_mask.shape[1]))

    def _assistant_content_ids(self, cache: Any) -> list[int]:
        value = getattr(cache, "assistant_token_ids", [])
        if hasattr(value, "tolist"):
            value = value.tolist()
        return [int(item) for item in value]

    def _state(
        self,
        cache: Any,
        expected_phase: str,
        expected_content_spans: Sequence[Sequence[int]],
    ) -> dict[str, Any]:
        token_ids = self._cache_token_ids(cache)
        seq = self._seq_length(cache)
        mask = int(cache.attention_mask.shape[1])
        kv = int(cache.past_key_values.get_seq_length())
        assistant_ids = self._assistant_content_ids(cache)
        raw_assistant_start = getattr(
            cache, "assistant_content_start", getattr(cache, "assistant_start", seq)
        )
        assistant_start = seq if raw_assistant_start is None else int(raw_assistant_start)
        assistant_end = assistant_start + len(assistant_ids)
        expected_spans = [
            [int(span[0]), int(span[1])] for span in expected_content_spans
        ]
        ledger_content = [
            token
            for start, end in expected_spans
            for token in token_ids[start:end]
        ]
        spans_valid = all(
            0 <= start <= end <= len(token_ids) for start, end in expected_spans
        )
        if expected_phase == "ASSISTANT_OPEN":
            current_expected = ledger_content[-len(assistant_ids):] if assistant_ids else []
            span_exact = spans_valid and assistant_ids == current_expected
        else:
            span_exact = spans_valid and not assistant_ids
        phase = self._enum_value(getattr(cache, "role_phase", None))
        end_reason = self._enum_value(
            getattr(cache, "generation_end_reason", getattr(cache, "end_reason", None))
        )
        lengths_exact = len(token_ids) == seq == mask == kv
        phase_ok = phase is not None and expected_phase.upper() in phase.upper()
        return {
            "seq_length": seq,
            "mask_length": mask,
            "kv_length": kv,
            "ledger_length": len(token_ids),
            "assistant_content_start": assistant_start,
            "assistant_content_end": assistant_end,
            "assistant_content_spans": expected_spans,
            "assistant_content_length": len(ledger_content),
            "assistant_content_span_exact": span_exact,
            "role_phase": phase,
            "expected_role_phase": expected_phase,
            "role_phase_exact": phase_ok,
            "end_reason": end_reason,
            "lengths_exact": lengths_exact,
        }

    def _initial_path(self, user_text: str) -> tuple[Any, CanonicalSequence]:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_text},
        ]
        canonical_ids = apply_chat_ids(self.tokenizer, messages, add_generation_prompt=True)
        pre = self.llm.cache_prompt(user_text, is_end=True, system_prompt=self.system_prompt)
        cache = self.llm.to_accum_cache(pre)
        actual_ids = self._cache_token_ids(cache)
        mismatch = first_mismatch(actual_ids, canonical_ids)
        if mismatch is not None:
            raise RuntimeError(f"Initial StreamLLM token ledger differs from apply_chat_template at {mismatch}")
        canonical = build_initial_open_sequence(
            self.tokenizer,
            self.parts,
            user_prompt=user_text,
        )
        if list(canonical.token_ids) != canonical_ids:
            raise RuntimeError("Canonical builder differs from apply_chat_template initial sequence")
        expected_role_start = len(self.parts.prefix_to_user_content) + len(self._encode(user_text))
        expected_content_start = expected_role_start + len(self.parts.user_to_assistant)
        if canonical.boundaries["assistant_role_start"] != expected_role_start:
            raise RuntimeError("Canonical assistant role start is not prefix plus raw user IDs")
        if canonical.boundaries["assistant_content_start"] != expected_content_start:
            raise RuntimeError("Canonical assistant content start differs from role transition end")
        cache_role_start = getattr(cache, "assistant_role_start", None)
        cache_content_start = getattr(cache, "assistant_content_start", None)
        if cache_role_start != expected_role_start or cache_content_start != expected_content_start:
            raise RuntimeError(
                "StreamLLM assistant role/content boundaries differ from canonical token boundaries"
            )
        return cache, canonical

    def _termination_probe(self, case: CaseSpec, user_text: str) -> dict[str, Any]:
        """Exercise generate_accumulating independently of retained-ID equivalence."""
        cache, _ = self._initial_path(user_text)
        before_ids = self._cache_token_ids(cache)
        before_seq = self._seq_length(cache)
        controlled = case.termination == "eos_at_cap"
        cap = {
            "natural_eos": NATURAL_EOS_MAX_NEW_TOKENS,
            "eos_at_cap": EOS_AT_CAP_MAX_NEW_TOKENS,
            "max_tokens": MAX_TOKENS_PROBE_BUDGET,
        }[case.termination]
        decoded: list[tuple[str, int]] = []
        original_decode = self.llm._decode_logits
        fixture_content_ids: list[int] = []
        fixture_steps: list[int] = []

        if controlled:
            candidates = self._encode(" controlled cap fixture")
            fixture_content_ids = [
                token for token in candidates if token != self.parts.eot_token_id
            ][: cap - 1]
            if len(fixture_content_ids) != cap - 1:
                raise RuntimeError("Could not construct non-EOT controlled cap fixture tokens")
            fixture_steps = [*fixture_content_ids, self.parts.eot_token_id]
            step = 0

            def controlled_decode(logits, temperature, top_p, repetition_penalty):
                del logits, temperature, top_p, repetition_penalty
                nonlocal step
                if step >= len(fixture_steps):
                    raise RuntimeError("Controlled EOS-at-cap fixture exhausted")
                token_id = fixture_steps[step]
                step += 1
                return self.torch.tensor([[token_id]], dtype=self.torch.long, device=self.device)

            # Only token selection is controlled. Content tokens still take the
            # production generate_accumulating -> _prefill_ids_p2 KV append path.
            self.llm._decode_logits = controlled_decode

        errors: list[str] = []
        try:
            generator = self.llm.generate_accumulating(
                cache,
                max_new_tokens=cap,
                temperature=0.0,
                top_p=1.0,
                repetition_penalty=1.0,
                on_token_decoded=lambda text, index, token_id: decoded.append(
                    (text, int(token_id))
                ),
            )
            for _ in generator:
                pass
        finally:
            self.llm._decode_logits = original_decode

        content_ids = self._assistant_content_ids(cache)
        after_ids = self._cache_token_ids(cache)
        observed = self._enum_value(cache.generation_end_reason)
        role_phase = self._enum_value(cache.role_phase)
        eot_in_kv = self.parts.eot_token_id in after_ids[len(before_ids) :]
        eot_in_ledger = self.parts.eot_token_id in content_ids
        eos_step = len(content_ids) + 1 if observed == "EOS" else None
        if after_ids != [*before_ids, *content_ids]:
            errors.append("probe full ledger does not equal prefix plus content tokens")
        if self._seq_length(cache) != before_seq + len(content_ids):
            errors.append("probe sequence length does not exclude pending EOT")
        if eot_in_kv or eot_in_ledger:
            errors.append("predicted EOT entered KV or assistant content ledger")
        selected_ids = [token_id for _, token_id in decoded]
        if observed == "EOS":
            if selected_ids[:-1] != content_ids or selected_ids[-1:] != [self.parts.eot_token_id]:
                errors.append("selection callback must contain content IDs followed by one EOT")
        elif selected_ids != content_ids:
            errors.append("selection callback IDs differ from assistant content ledger")
        if case.termination == "natural_eos":
            if observed != "EOS" or eos_step is None or eos_step > cap:
                errors.append("natural greedy generation did not reach EOS within frozen cap")
            if role_phase != "ASSISTANT_EOT_PENDING":
                errors.append("natural EOS did not leave ASSISTANT_EOT_PENDING")
        elif case.termination == "eos_at_cap":
            if observed != "EOS" or eos_step != cap:
                errors.append("controlled EOT was not observed at the final cap step")
            if content_ids != fixture_content_ids:
                errors.append("controlled content tokens differ from fixture")
            if role_phase != "ASSISTANT_EOT_PENDING":
                errors.append("controlled EOS did not leave ASSISTANT_EOT_PENDING")
        else:
            if observed != "MAX_TOKENS":
                errors.append("small-budget greedy generation did not report MAX_TOKENS")
            if len(content_ids) != cap:
                errors.append("max-token probe did not retain exactly the frozen budget")
            if role_phase != "ASSISTANT_OPEN":
                errors.append("max-token probe did not remain ASSISTANT_OPEN")
        return {
            "schema_version": TERMINATION_PROBE_SCHEMA_VERSION,
            "declared": case.termination,
            "mode": "controlled_logits_fixture" if controlled else "real_greedy",
            "controlled": controlled,
            "generate_api": "StreamLLMInference.generate_accumulating",
            "execution": "transformers_model",
            "temperature": 0.0,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
            "cap": cap,
            "observed_end_reason": observed,
            "selected_token_ids": selected_ids,
            "selected_token_hash": token_ids_hash(selected_ids),
            "selected_token_count": len(selected_ids),
            "content_token_ids": content_ids,
            "content_token_hash": token_ids_hash(content_ids),
            "content_token_count": len(content_ids),
            "eos_step": eos_step,
            "eos_at_cap": eos_step == cap,
            "eot_token_id": self.parts.eot_token_id,
            "eot_in_kv": eot_in_kv,
            "eot_in_full_ledger": eot_in_kv,
            "eot_in_content_ledger": eot_in_ledger,
            "prefix_seq_length": before_seq,
            "post_seq_length": self._seq_length(cache),
            "role_phase": role_phase,
            "fixture_token_ids": fixture_steps if controlled else None,
            "fixture_description": (
                "deterministic decode-token fixture; non-EOT content uses production KV prefill and final EOT uses generate_accumulating EOS branch"
                if controlled
                else None
            ),
            "passed": not errors,
            "errors": errors,
        }

    def _force_pending_eot(self, cache: Any) -> dict[str, Any]:
        """Drive the real generation EOS branch without appending EOT to either ledger."""
        before_ids = self._cache_token_ids(cache)
        before_content = self._assistant_content_ids(cache)
        before_seq = self._seq_length(cache)
        original_decode = self.llm._decode_logits

        def force_eot(logits, temperature, top_p, repetition_penalty):
            del logits, temperature, top_p, repetition_penalty
            return self.torch.tensor(
                [[self.parts.eot_token_id]], dtype=self.torch.long, device=self.device
            )

        self.llm._decode_logits = force_eot
        try:
            for _ in self.llm.generate_accumulating(
                cache,
                max_new_tokens=1,
                temperature=0.0,
                top_p=1.0,
                repetition_penalty=1.0,
            ):
                raise AssertionError("Forced EOT unexpectedly yielded content")
        finally:
            self.llm._decode_logits = original_decode
        after_ids = self._cache_token_ids(cache)
        after_content = self._assistant_content_ids(cache)
        return {
            "pending": self._enum_value(cache.role_phase) == "ASSISTANT_EOT_PENDING",
            "end_reason": self._enum_value(cache.generation_end_reason),
            "eot_in_full_ledger": self.parts.eot_token_id in after_ids[len(before_ids) :],
            "eot_in_content_ledger": self.parts.eot_token_id in after_content[len(before_content) :],
            "seq_unchanged": self._seq_length(cache) == before_seq,
            "full_ledger_unchanged": after_ids == before_ids,
            "content_ledger_unchanged": after_content == before_content,
        }

    def _prefill_assistant(self, cache: Any, text: str) -> list[int]:
        expected = self._encode(text)
        before = self._cache_token_ids(cache)
        self.llm.prefill_assistant_text(cache, text)
        after = self._cache_token_ids(cache)
        appended = after[len(before) :]
        if appended != expected:
            raise RuntimeError("prefill_assistant_text did not preserve canonical assistant token IDs")
        if self._assistant_content_ids(cache)[-len(expected) :] != expected:
            raise RuntimeError("Assistant content ledger differs from teacher-forced token IDs")
        return expected

    def _fragment_token_ends(self, case: CaseSpec, full_ids: Sequence[int]) -> list[int]:
        del full_ids
        ends: list[int] = []
        accumulated = ""
        previous = 0
        for fragment in case.fragments:
            accumulated += fragment
            end = len(self._encode(accumulated))
            if end <= previous:
                raise ValueError(f"{case.id}: fragment has an empty/non-monotonic token span")
            ends.append(end)
            previous = end
        if ends[-1] != len(self._encode(case.assistant_text)):
            raise ValueError(f"{case.id}: fragment token partition does not cover assistant text")
        return ends

    def _clean_cache(self, token_ids: Sequence[int]) -> Any:
        torch = self.torch
        ids_list = [int(value) for value in token_ids]
        ids = torch.tensor([ids_list], dtype=torch.long, device=self.device)
        mask = torch.ones_like(ids)
        position_ids = torch.arange(len(ids_list), dtype=torch.long, device=self.device).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(
                input_ids=ids,
                attention_mask=mask,
                position_ids=position_ids,
                use_cache=True,
                return_dict=True,
            )
        return SimpleNamespace(
            past_key_values=outputs.past_key_values,
            attention_mask=mask,
            seq_length=len(ids_list),
            next_token_logits=outputs.logits[:, -1, :],
            token_ids=ids_list,
        )

    def _continue(self, cache: Any, count: int) -> list[int]:
        torch = self.torch
        result: list[int] = []
        logits = cache.next_token_logits
        if logits is None:
            raise RuntimeError("Checkpoint lacks next-token logits")
        for _ in range(count):
            token_id = int(torch.argmax(logits.float(), dim=-1).item())
            result.append(token_id)
            ids = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
            mask = torch.cat(
                [
                    cache.attention_mask,
                    torch.ones((1, 1), dtype=cache.attention_mask.dtype, device=self.device),
                ],
                dim=-1,
            )
            position = torch.tensor([[self._seq_length(cache)]], dtype=torch.long, device=self.device)
            with torch.no_grad():
                outputs = self.model(
                    input_ids=ids,
                    attention_mask=mask,
                    position_ids=position,
                    past_key_values=cache.past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
            cache.past_key_values = outputs.past_key_values
            cache.attention_mask = mask
            cache.seq_length = self._seq_length(cache) + 1
            cache.next_token_logits = outputs.logits[:, -1, :]
            logits = cache.next_token_logits
        return result

    def _checkpoint(
        self,
        *,
        name: str,
        path_cache: Any,
        canonical: CanonicalSequence,
        expected_phase: str,
        assistant_boundaries: int,
    ) -> dict[str, Any]:
        torch = self.torch
        clean_cache = self._clean_cache(canonical.token_ids)
        path_ids = self._cache_token_ids(path_cache)
        mismatch = first_mismatch(path_ids, canonical.token_ids)
        # Crop invalidates cached next-token logits. Recompute the path logits from its
        # actual cropped KV by feeding the final token once against a cache cropped by
        # one; this remains independent of the clean-from-empty oracle.
        if path_cache.next_token_logits is None:
            if not path_ids:
                raise RuntimeError("Cannot recover next-token logits for an empty path")
            base_len = self._seq_length(path_cache)
            path_cache.past_key_values.crop(base_len - 1)
            path_cache.attention_mask = path_cache.attention_mask[:, : base_len - 1]
            last_id = self.torch.tensor([[path_ids[-1]]], dtype=self.torch.long, device=self.device)
            full_mask = self.torch.ones((1, base_len), dtype=path_cache.attention_mask.dtype, device=self.device)
            position = self.torch.tensor([[base_len - 1]], dtype=self.torch.long, device=self.device)
            with self.torch.no_grad():
                refreshed = self.model(
                    input_ids=last_id,
                    attention_mask=full_mask,
                    position_ids=position,
                    past_key_values=path_cache.past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
            path_cache.past_key_values = refreshed.past_key_values
            path_cache.attention_mask = full_mask
            path_cache.next_token_logits = refreshed.logits[:, -1, :]
        # Capture every checkpoint fact before continuation mutates either cache.
        path_logits = path_cache.next_token_logits.float()
        clean_logits = clean_cache.next_token_logits.float()
        difference = (path_logits - clean_logits).abs()
        path_top = torch.topk(path_logits, k=TOP_K, dim=-1).indices[0].tolist()
        clean_top = torch.topk(clean_logits, k=TOP_K, dim=-1).indices[0].tolist()
        all_path_eots = [
            index for index, token in enumerate(path_ids) if token == self.parts.eot_token_id
        ]
        all_canonical_eots = list(canonical.eot_positions)
        canonical_eots = [
            int(value) for value in canonical.boundaries.get("assistant_eot_positions", [])
        ]
        path_eots = [
            index for index in canonical_eots
            if index < len(path_ids) and path_ids[index] == self.parts.eot_token_id
        ]
        state_path = self._state(
            path_cache,
            expected_phase,
            canonical.boundaries.get("assistant_content_spans", []),
        )
        state_canonical = {
            "seq_length": len(canonical.token_ids),
            "mask_length": len(canonical.token_ids),
            "kv_length": len(canonical.token_ids),
            "ledger_length": len(canonical.token_ids),
            "role_phase": expected_phase,
            "lengths_exact": True,
        }
        top_overlap = len(set(path_top) & set(clean_top))
        unique_eot_ok = path_eots == canonical_eots
        max_abs = float(difference.max().item())
        mean_abs = float(difference.mean().item())
        rms = float(torch.sqrt(torch.mean(difference * difference)).item())
        failure_logits = {
            "path": path_logits.detach().cpu().numpy(),
            "canonical": clean_logits.detach().cpu().numpy(),
        }

        # The path continuation must consume the refreshed production crop/recovery
        # cache itself. The clean side alone starts from an empty-cache re-prefill.
        path_continuation = self._continue(path_cache, CONTINUATION_TOKENS)
        clean_continuation = self._continue(clean_cache, CONTINUATION_TOKENS)
        continuation_divergence = first_mismatch(path_continuation, clean_continuation)
        errors: list[str] = []
        if mismatch is not None:
            errors.append(f"token mismatch at {mismatch}")
        if not state_path["lengths_exact"] or not state_path["assistant_content_span_exact"]:
            errors.append("path state invariant failed")
        if not state_path["role_phase_exact"]:
            errors.append("role phase mismatch")
        if not unique_eot_ok:
            errors.append("assistant EOT positions differ")
        if int(path_top[0]) != int(clean_top[0]):
            errors.append("next-token top1 differs")
        if top_overlap < 4:
            errors.append(f"top5 overlap {top_overlap}/5")
        if max_abs > 0.1:
            errors.append(f"max_abs logit diff {max_abs:.8g} exceeds 0.1")
        if mean_abs > 0.01:
            errors.append(f"mean_abs logit diff {mean_abs:.8g} exceeds 0.01")
        if continuation_divergence is not None:
            errors.append(f"continuation diverges at {continuation_divergence}")
        return {
            "checkpoint": name,
            "canonical": canonical.to_dict(include_ids=True),
            "path": {
                "token_ids": path_ids,
                "token_hash": token_ids_hash(path_ids),
                "token_count": len(path_ids),
                "eot_positions": all_path_eots,
            },
            "token_ids_exact": mismatch is None,
            "first_token_mismatch": mismatch,
            "state": {
                "path": state_path,
                "canonical": state_canonical,
                "exact": state_path["lengths_exact"]
                and state_path["role_phase_exact"]
                and len(path_ids) == len(canonical.token_ids),
            },
            "unique_eot": {
                "ok": unique_eot_ok,
                "path_positions": path_eots,
                "canonical_positions": canonical_eots,
                "all_path_positions": all_path_eots,
                "all_canonical_positions": all_canonical_eots,
                "assistant_boundaries": assistant_boundaries,
            },
            "next_token": {
                "path_top1": int(path_top[0]),
                "canonical_top1": int(clean_top[0]),
                "top1_exact": int(path_top[0]) == int(clean_top[0]),
                "path_top5": [int(value) for value in path_top],
                "canonical_top5": [int(value) for value in clean_top],
                "top5_overlap": top_overlap,
            },
            "logit_diff_float32": {
                "max_abs": max_abs,
                "mean_abs": mean_abs,
                "rms": rms,
            },
            "continuation": {
                "tokens": CONTINUATION_TOKENS,
                "continuation_source": "actual_crop_cache",
                "canonical_source": "clean_prefill_cache",
                "checkpoint_state_captured_before_mutation": True,
                "path_token_ids": path_continuation,
                "canonical_token_ids": clean_continuation,
                "path_hash": token_ids_hash(path_continuation),
                "canonical_hash": token_ids_hash(clean_continuation),
                "exact": continuation_divergence is None,
                "first_divergence": continuation_divergence,
            },
            "passed": not errors,
            "errors": errors,
            "_failure_logits": failure_logits if errors else None,
        }

    def _build_recovery(
        self, case: CaseSpec, user_text: str
    ) -> tuple[Any, CanonicalSequence, list[int], int, dict[str, Any]]:
        cache, initial = self._initial_path(user_text)
        assistant_ids = self._prefill_assistant(cache, case.assistant_text)
        applies = case.scenario in {"crop_pending_eot", "reply_tail_noop"}
        scenario_execution = {
            "schema_version": 1,
            "scenario": case.scenario,
            "applies": applies,
            "execution": "transformers_model",
            "generate_api": (
                "StreamLLMInference.generate_accumulating" if applies else None
            ),
            "forced_decode_token": "EOT" if applies else None,
            "generate_max_new_tokens": 1 if applies else None,
            "pending_before_crop": None,
            "eot_in_full_ledger_before_crop": None,
            "eot_in_content_ledger_before_crop": None,
            "crop_target": None,
            "crop_was_noop": None,
            "no_op_preserved_pending": None,
            "pending_after_crop": None,
            "pending_cleared_by_crop": None,
            "reopen_called": False,
            "passed": True,
            "errors": [],
        }
        if applies:
            forced = self._force_pending_eot(cache)
            scenario_execution.update(
                {
                    "pending_before_crop": forced["pending"],
                    "forced_end_reason": forced["end_reason"],
                    "eot_in_full_ledger_before_crop": forced["eot_in_full_ledger"],
                    "eot_in_content_ledger_before_crop": forced["eot_in_content_ledger"],
                    "seq_unchanged_by_pending_eot": forced["seq_unchanged"],
                    "full_ledger_unchanged_by_pending_eot": forced["full_ledger_unchanged"],
                    "content_ledger_unchanged_by_pending_eot": forced["content_ledger_unchanged"],
                }
            )
            if not all(
                (
                    forced["pending"],
                    forced["end_reason"] == "EOS",
                    not forced["eot_in_full_ledger"],
                    not forced["eot_in_content_ledger"],
                    forced["seq_unchanged"],
                    forced["full_ledger_unchanged"],
                    forced["content_ledger_unchanged"],
                )
            ):
                scenario_execution["errors"].append(
                    "forced generate_accumulating EOT did not produce clean pending state"
                )
        fragment_ends = self._fragment_token_ends(case, assistant_ids)
        retained = fragment_ends[case.retain_fragment_count - 1] if case.retain_fragment_count else 0
        if case.scenario == "reply_tail_noop":
            keep = self._seq_length(cache)
            scenario_execution["crop_target"] = "current_seq"
            self.llm.crop_to_token(cache, keep)
            scenario_execution["crop_was_noop"] = self._seq_length(cache) == keep
            scenario_execution["pending_after_crop"] = (
                self._enum_value(cache.role_phase) == "ASSISTANT_EOT_PENDING"
            )
            scenario_execution["no_op_preserved_pending"] = (
                scenario_execution["pending_after_crop"]
                and self._enum_value(cache.generation_end_reason) == "EOS"
            )
            scenario_execution["pending_cleared_by_crop"] = False
            self.llm.reopen_user_role(cache)
            scenario_execution["reopen_called"] = True
            canonical = append_assistant_ids(
                initial,
                assistant_ids,
                self.parts,
                close_assistant=True,
                open_user=True,
            )
            boundaries = 1
        elif retained == 0 and case.scenario == "speculation_full_invalidation":
            # Speculation never became an assistant turn: remove user-close plus the
            # assistant header and continue the still-open original user content.
            keep = int(initial.boundaries["assistant_role_start"])
            self.llm.crop_to_token(cache, keep)
            canonical_tokens = initial.token_ids[:keep]
            canonical = CanonicalSequence(
                token_ids=tuple(canonical_tokens),
                boundaries={
                    "user_content_end": keep,
                    "assistant_content_spans": [],
                    "assistant_eot_positions": [],
                    "zero_retain_semantics": "speculation_full_invalidation",
                    "crop_token": keep,
                },
                eot_positions=tuple(
                    index for index, token in enumerate(canonical_tokens)
                    if token == self.parts.eot_token_id
                ),
                special_token_ids=self.parts.special_tokens(),
            )
            boundaries = 0
        elif retained == 0:
            # Playback p=0 is an empty-but-real assistant turn: retain its header,
            # commit exactly one assistant EOT, then open the next user role.
            keep = int(initial.boundaries["assistant_content_start"])
            self.llm.crop_to_token(cache, keep)
            self.llm.reopen_user_role(cache)
            canonical = append_assistant_ids(
                initial,
                [],
                self.parts,
                close_assistant=True,
                open_user=True,
            )
            canonical.boundaries["zero_retain_semantics"] = "full_rollback_p0"
            canonical.boundaries["crop_token"] = keep
            boundaries = 1
        else:
            keep = int(initial.boundaries["assistant_content_start"]) + retained
            if case.scenario == "crop_pending_eot":
                scenario_execution["crop_target"] = "retained_boundary"
            self.llm.crop_to_token(cache, keep)
            if case.scenario == "crop_pending_eot":
                scenario_execution["crop_was_noop"] = False
                scenario_execution["pending_after_crop"] = (
                    self._enum_value(cache.role_phase) == "ASSISTANT_EOT_PENDING"
                )
                scenario_execution["pending_cleared_by_crop"] = (
                    not scenario_execution["pending_after_crop"]
                    and self._enum_value(cache.generation_end_reason) == "CROPPED"
                )
                scenario_execution["no_op_preserved_pending"] = False
            self.llm.reopen_user_role(cache)
            if case.scenario == "crop_pending_eot":
                scenario_execution["reopen_called"] = True
            canonical = append_assistant_ids(
                initial,
                assistant_ids[:retained],
                self.parts,
                close_assistant=True,
                open_user=True,
            )
            boundaries = 1
        if applies:
            required = (
                scenario_execution["pending_before_crop"] is True
                and scenario_execution["eot_in_full_ledger_before_crop"] is False
                and scenario_execution["eot_in_content_ledger_before_crop"] is False
                and scenario_execution["reopen_called"] is True
            )
            if case.scenario == "crop_pending_eot":
                required = required and (
                    scenario_execution["crop_target"] == "retained_boundary"
                    and scenario_execution["crop_was_noop"] is False
                    and scenario_execution["pending_after_crop"] is False
                    and scenario_execution["pending_cleared_by_crop"] is True
                )
            else:
                required = required and (
                    scenario_execution["crop_target"] == "current_seq"
                    and scenario_execution["crop_was_noop"] is True
                    and scenario_execution["pending_after_crop"] is True
                    and scenario_execution["no_op_preserved_pending"] is True
                )
            if not required:
                scenario_execution["errors"].append("scenario pending-EOT transition failed")
        scenario_execution["passed"] = not scenario_execution["errors"]
        if case.next_user is not None:
            # Raw next-user content is appended identically on both paths. Structural
            # separation is owned by the role transition, never by an injected newline.
            self.llm.prefill_user_text(cache, case.next_user)
            next_user_ids = self._encode(case.next_user)
            canonical_tokens = [*canonical.token_ids, *next_user_ids]
            canonical = CanonicalSequence(
                token_ids=tuple(canonical_tokens),
                boundaries={
                    **canonical.boundaries,
                    "next_user_content_end": len(canonical_tokens),
                },
                eot_positions=tuple(
                    index for index, token in enumerate(canonical_tokens)
                    if token == self.parts.eot_token_id
                ),
                special_token_ids=self.parts.special_tokens(),
            )
        return cache, canonical, assistant_ids, boundaries, scenario_execution

    def run_case(self, case: CaseSpec) -> dict[str, Any]:
        user_text = self._context_user_text(case)
        checkpoints: list[dict[str, Any]] = []
        termination_probe = self._termination_probe(case, user_text)
        errors: list[str] = [
            f"termination_probe: {message}" for message in termination_probe["errors"]
        ]

        (
            recovery_cache,
            recovery_canonical,
            _,
            first_boundaries,
            scenario_execution,
        ) = self._build_recovery(case, user_text)
        errors.extend(
            f"scenario_execution: {message}"
            for message in scenario_execution["errors"]
        )
        checkpoints.append(
            self._checkpoint(
                name="post_recovery",
                path_cache=recovery_cache,
                canonical=recovery_canonical,
                expected_phase="USER_OPEN",
                assistant_boundaries=first_boundaries,
            )
        )

        if case.next_user is not None:
            next_cache, next_canonical, _, first_boundaries, next_scenario = (
                self._build_recovery(case, user_text)
            )
            if next_scenario != scenario_execution:
                errors.append("scenario_execution: repeated recovery evidence differs")
            self.llm.open_assistant_role(next_cache)
            next_canonical = CanonicalSequence(
                token_ids=tuple([*next_canonical.token_ids, *self.parts.user_to_assistant]),
                boundaries={
                    **next_canonical.boundaries,
                    "next_user_content_end": len(next_canonical.token_ids),
                    "next_assistant_content_start": (
                        len(next_canonical.token_ids) + len(self.parts.user_to_assistant)
                    ),
                },
                eot_positions=tuple(
                    index
                    for index, token in enumerate(
                        [*next_canonical.token_ids, *self.parts.user_to_assistant]
                    )
                    if token == self.parts.eot_token_id
                ),
                special_token_ids=self.parts.special_tokens(),
            )
            checkpoints.append(
                self._checkpoint(
                    name="next_assistant",
                    path_cache=next_cache,
                    canonical=next_canonical,
                    expected_phase="ASSISTANT_OPEN",
                    assistant_boundaries=first_boundaries,
                )
            )

        if case.second_crop_fraction is not None:
            second_cache, second_canonical, _, first_boundaries, second_scenario = (
                self._build_recovery(case, user_text)
            )
            if second_scenario != scenario_execution:
                errors.append("scenario_execution: second recovery evidence differs")
            self.llm.open_assistant_role(second_cache)
            second_open_tokens = [*second_canonical.token_ids, *self.parts.user_to_assistant]
            second_open = CanonicalSequence(
                token_ids=tuple(second_open_tokens),
                boundaries={
                    **second_canonical.boundaries,
                    "assistant_content_start": len(second_open_tokens),
                },
                eot_positions=tuple(
                    index for index, token in enumerate(second_open_tokens)
                    if token == self.parts.eot_token_id
                ),
                special_token_ids=self.parts.special_tokens(),
            )
            second_ids = self._prefill_assistant(second_cache, case.second_assistant_text or "")
            retained_second = max(1, min(len(second_ids), math.floor(len(second_ids) * case.second_crop_fraction)))
            keep = len(second_open_tokens) + retained_second
            self.llm.crop_to_token(second_cache, keep)
            self.llm.reopen_user_role(second_cache)
            second_canonical = append_assistant_ids(
                second_open,
                second_ids[:retained_second],
                self.parts,
                close_assistant=True,
                open_user=True,
            )
            checkpoints.append(
                self._checkpoint(
                    name="post_second_recovery",
                    path_cache=second_cache,
                    canonical=second_canonical,
                    expected_phase="USER_OPEN",
                    assistant_boundaries=first_boundaries + 1,
                )
            )

        for checkpoint in checkpoints:
            checkpoint["termination_probe"] = dict(termination_probe)
            checkpoint["scenario_execution"] = dict(scenario_execution)
            errors.extend(f"{checkpoint['checkpoint']}: {message}" for message in checkpoint["errors"])
        return {
            "case_id": case.id,
            "context_tokens_target": case.context_tokens,
            "context_tokens_actual": len(
                apply_chat_ids(
                    self.tokenizer,
                    [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_text},
                    ],
                    add_generation_prompt=True,
                )
            ),
            "context_class": case.context_class,
            "scenario": case.scenario,
            "termination": case.termination,
            "controlled_fixture": case.controlled_fixture,
            "source": case.source,
            "termination_probe": termination_probe,
            "scenario_execution": scenario_execution,
            "checkpoints": checkpoints,
            "passed": not errors,
            "errors": errors,
        }


def make_backend(kind: str, *, model_path: str | None, device: str, seed: int) -> EquivalenceBackend:
    if kind == "fake":
        return FakeBackend(seed=seed)
    if kind == "transformers":
        if not model_path:
            raise ValueError("Transformers C2 runtime requires --model")
        return TransformersBackend(model_path, device=device)
    raise ValueError(f"Unknown runtime: {kind}")
