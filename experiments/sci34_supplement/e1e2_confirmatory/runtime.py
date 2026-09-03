"""Runtime adapters for confirmatory E1/E2 measurements.

The import surface is model-free.  ``TransformersBackend`` lazily loads the
project StreamLLM implementation only after the formal runner has enabled strict
offline mode and validated all input/cache identities.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from experiments.sci34_supplement.common import canonical_json
from experiments.sci34_supplement.e1e2_confirmatory.strong_identity import strong_model_identity
from experiments.sci34_supplement.e1e2_confirmatory.protocol import (
    NEVER_SPECULATE,
    SYSTEM_A,
    SYSTEM_PROMPT,
    threshold_for_condition,
)


class SessionBackend(Protocol):
    model_name: str
    revision: str | None

    @property
    def identity(self) -> Mapping[str, Any]: ...

    def warmup(self, path_kind: str) -> None: ...

    def run_condition(
        self,
        row: Mapping[str, Any],
        *,
        condition: str,
        threshold: float | None,
        confidences: Sequence[float],
        session_id: str,
    ) -> dict[str, Any]: ...


def _synchronize(device: str) -> None:
    try:
        import torch

        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(device)
    except ImportError:
        return


def _cache_entry_hashes(
    sample_id: str, segments: Sequence[str], confidences: Sequence[float]
) -> list[str]:
    accumulated = ""
    hashes: list[str] = []
    for prefix_index, (segment, confidence) in enumerate(zip(segments, confidences), start=1):
        accumulated += segment
        hashes.append(
            hashlib.sha256(
                canonical_json(
                    {
                        "id": sample_id,
                        "prefix_index": prefix_index,
                        "text_sha256": hashlib.sha256(accumulated.encode("utf-8")).hexdigest(),
                        "confidence": float(confidence),
                    }
                ).encode("utf-8")
            ).hexdigest()
        )
    return hashes


@dataclass
class _Candidate:
    base_seq_len: int
    started_ns: int
    first_token_ns: int | None
    tokens: list[tuple[str, int]]
    end_reason: Any


class TransformersBackend:
    """Greedy adapter over StreamLLMInference with corrected acceptance timing."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cuda:0",
        max_new_tokens: int = 32,
        spec_chunk: int = 12,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        from src.llm.stream_llm_inference import StreamLLMInference

        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.spec_chunk = spec_chunk
        self.system_prompt = system_prompt
        self._llm = StreamLLMInference(model_name=model_name, device=device, eval_mode=False)
        self.revision = getattr(self._llm.model.config, "_commit_hash", None)
        self._identity = strong_model_identity(model_name)
        parameters = list(self._llm.model.parameters())
        self.resolved_dtype = str(parameters[0].dtype) if parameters else None
        config = self._llm.model.config
        self.attention_backend = str(
            getattr(config, "_attn_implementation", None)
            or getattr(config, "attn_implementation", None)
            or "eager/default"
        )

    @property
    def identity(self) -> dict[str, Any]:
        return dict(self._identity)

    @property
    def runtime_metadata(self) -> dict[str, Any]:
        return {
            "resolved_dtype": self.resolved_dtype,
            "attention_backend": self.attention_backend,
        }

    def _greedy_tokens(self, cache, limit: int, *, on_token_decoded=None):
        return self._llm.generate_accumulating(
            cache,
            max_new_tokens=limit,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.0,
            on_token_decoded=on_token_decoded,
        )

    def _consume(
        self,
        cache,
        token_iter,
        *,
        initial_tokens: list[tuple[str, int]] | None = None,
        mark_delivery: bool,
    ) -> tuple[list[tuple[str, int]], int | None, int, bool]:
        tokens = list(initial_tokens or [])
        first_delivery_ns: int | None = None
        before = len(cache.assistant_token_ids)
        for token in token_iter:
            if mark_delivery and first_delivery_ns is None:
                first_delivery_ns = time.perf_counter_ns()
            tokens.append(token)
        generated_now = len(cache.assistant_token_ids) - before
        eos = cache.generation_end_reason == self._llm.GenerationEndReason.EOS
        return tokens, first_delivery_ns, generated_now, eos

    def warmup(self, path_kind: str) -> None:
        if path_kind == "sentence_chunker":
            from src.tts.sentence_chunker import chunk_llm_tokens

            list(
                chunk_llm_tokens(
                    iter((("Warm", 0), (" up.", 1))),
                    language="en",
                    tokenizer="nltk",
                )
            )
            return
        if path_kind == "full_prefill":
            self.run_condition(
                {"id": "warmup-full", "full_text": "Please give a short answer.", "segments": ["Please give", " a short answer."]},
                condition=SYSTEM_A,
                threshold=None,
                confidences=(0.0, 1.0),
                session_id="warmup",
            )
            return
        confidence_sets = {
            "survived_speculation": (0.0, 1.0),
            "invalidated_crop": (1.0, 0.0),
            "never_speculate": (0.0, 0.0),
        }
        confidences = confidence_sets[path_kind]
        condition = NEVER_SPECULATE if path_kind == "never_speculate" else "warmup_threshold"
        self.run_condition(
            {"id": f"warmup-{path_kind}", "full_text": "Please give a short answer.", "segments": ["Please give", " a short answer."]},
            condition=condition,
            threshold=None if condition == NEVER_SPECULATE else 0.5,
            confidences=confidences,
            session_id="warmup",
        )

    def run_condition(
        self,
        row: Mapping[str, Any],
        *,
        condition: str,
        threshold: float | None,
        confidences: Sequence[float],
        session_id: str,
    ) -> dict[str, Any]:
        del session_id
        segments = [str(value) for value in row["segments"]]
        if len(segments) != len(confidences):
            raise ValueError(f"Trigger confidence count mismatch for {row['id']}")
        if condition == SYSTEM_A:
            return self._run_system_a(row)
        return self._run_system_b(row, condition, threshold, confidences)

    def _run_system_a(self, row: Mapping[str, Any]) -> dict[str, Any]:
        _synchronize(self.device)
        last_arrival_ns = time.perf_counter_ns()
        endpoint_ns = last_arrival_ns
        cache = self._llm.cache_prompt(
            str(row["full_text"]), is_end=True, system_prompt=self.system_prompt
        )
        acc = self._llm.to_accum_cache(cache)
        prefill_done_ns = time.perf_counter_ns()
        first_decoded: list[int] = []
        tokens, consumed_first_ns, _, eos = self._consume(
            acc,
            self._greedy_tokens(
                acc,
                self.max_new_tokens,
                on_token_decoded=lambda *_: first_decoded.append(time.perf_counter_ns())
                if not first_decoded else None,
            ),
            mark_delivery=True,
        )
        first_delivery_ns = first_decoded[0] if first_decoded else consumed_first_ns
        consumer_delivery_ns = consumed_first_ns or first_delivery_ns
        _synchronize(self.device)
        generation_done_ns = time.perf_counter_ns()
        if first_delivery_ns is None:
            first_delivery_ns = generation_done_ns
        output_ids = list(acc.assistant_token_ids)
        return {
            "last_segment_arrival_ns": last_arrival_ns,
            "endpoint_accept_ns": endpoint_ns,
            "prefill_done_ns": prefill_done_ns,
            "first_token_ready_ns": first_delivery_ns,
            "first_deliverable_token_ns": first_delivery_ns,
            "oracle_preaccept_processing_ns": 0,
            "arrival_to_first_token_ready_ns": first_delivery_ns - last_arrival_ns,
            "consumer_delivery_ns": consumer_delivery_ns,
            "generation_done_ns": generation_done_ns,
            "ttft_eff_ns": first_delivery_ns - endpoint_ns,
            "prefill_to_first_token_ns": first_delivery_ns - endpoint_ns,
            "generation_total_ns": generation_done_ns - endpoint_ns,
            "survived": False,
            "ready_tokens": 0,
            "candidate_started_ns": None,
            "candidate_first_token_ns": None,
            "candidate_lead_ns": None,
            "on_demand_ttft_ns": first_delivery_ns - endpoint_ns,
            "n_speculations": 0,
            "n_invalidated": 0,
            "wasted_tokens": 0,
            "speculative_tokens": 0,
            "final_tokens": len(output_ids),
            "eos": eos,
            "max_tokens_hit": len(output_ids) >= self.max_new_tokens and not eos,
            "output_token_ids": output_ids,
            "output_text": self._llm.tokenizer.decode(output_ids, skip_special_tokens=True),
        }

    def _run_system_b(
        self,
        row: Mapping[str, Any],
        condition: str,
        threshold: float | None,
        confidences: Sequence[float],
    ) -> dict[str, Any]:
        never = condition == NEVER_SPECULATE
        if not never and threshold is None:
            threshold = threshold_for_condition(condition)
        active: _Candidate | None = None
        acc = None
        accumulated = ""
        n_speculations = 0
        n_invalidated = 0
        wasted_tokens = 0
        speculative_tokens = 0
        trigger_prefixes: list[dict[str, Any]] = []
        segment_count = len(row["segments"])
        last_arrival_ns: int | None = None
        for prefix_index, (segment, confidence) in enumerate(
            zip(row["segments"], confidences), start=1
        ):
            if prefix_index == segment_count:
                _synchronize(self.device)
                last_arrival_ns = time.perf_counter_ns()
            if active is not None:
                n_invalidated += 1
                wasted_tokens += len(active.tokens)
                self._llm.crop_to_token(acc, active.base_seq_len)
                if acc.role_phase != self._llm.RolePhase.USER_OPEN:
                    raise AssertionError("invalidated candidate did not restore USER_OPEN")
                active = None
            if acc is None:
                pre = self._llm.cache_prompt(
                    str(segment), is_end=False, system_prompt=self.system_prompt
                )
                acc = self._llm.to_accum_cache(pre)
                if acc.role_phase != self._llm.RolePhase.USER_OPEN:
                    raise AssertionError("streaming prefill must remain USER_OPEN")
            else:
                self._llm.prefill_user_text(acc, str(segment))
            accumulated += str(segment)
            fired = not never and float(confidence) >= float(threshold)
            trigger_prefixes.append(
                {
                    "prefix_index": prefix_index,
                    "confidence": float(confidence),
                    "fired": fired,
                }
            )
            if fired:
                started_ns = time.perf_counter_ns()
                self._llm.open_assistant_role(acc)
                base = acc.assistant_role_start
                first_decoded: list[int] = []
                candidate_tokens: list[tuple[str, int]] = []
                for token in self._greedy_tokens(
                    acc,
                    self.spec_chunk,
                    on_token_decoded=lambda *_: first_decoded.append(time.perf_counter_ns())
                    if not first_decoded else None,
                ):
                    candidate_tokens.append(token)
                first_ns = first_decoded[0] if first_decoded else None
                n_speculations += 1
                speculative_tokens += len(candidate_tokens)
                end_reason = acc.generation_end_reason
                active = _Candidate(
                    base, started_ns, first_ns, candidate_tokens, end_reason
                )

        _synchronize(self.device)
        endpoint_ns = time.perf_counter_ns()
        if last_arrival_ns is None:
            raise AssertionError("System B did not record the final segment arrival")
        eos_only_candidate = (
            active is not None
            and active.end_reason == self._llm.GenerationEndReason.EOS
            and not active.tokens
        )
        survived = active is not None and (bool(active.tokens) or eos_only_candidate)
        if survived:
            ready = len(active.tokens)
            first_deliverable_ns = endpoint_ns
            consumer_delivery_ns = time.perf_counter_ns()
            remaining = (
                0 if active.end_reason == self._llm.GenerationEndReason.EOS
                else max(0, self.max_new_tokens - ready)
            )
            _, _, _, eos = self._consume(
                acc,
                self._greedy_tokens(acc, remaining) if remaining > 0 else iter(()),
                initial_tokens=active.tokens,
                mark_delivery=False,
            )
            eos = eos or eos_only_candidate
            candidate_started_ns = active.started_ns
            candidate_first_ns = active.first_token_ns
            candidate_lead_ns = endpoint_ns - active.first_token_ns if active.first_token_ns else None
            on_demand_ns = None
        else:
            ready = 0
            self._llm.open_assistant_role(acc)
            first_decoded = []
            _, consumed_first_ns, _, eos = self._consume(
                acc,
                self._greedy_tokens(
                    acc,
                    self.max_new_tokens,
                    on_token_decoded=lambda *_: first_decoded.append(time.perf_counter_ns())
                    if not first_decoded else None,
                ),
                mark_delivery=True,
            )
            first_deliverable_ns = (
                first_decoded[0] if first_decoded else consumed_first_ns or time.perf_counter_ns()
            )
            consumer_delivery_ns = consumed_first_ns or first_deliverable_ns
            candidate_started_ns = None
            candidate_first_ns = None
            candidate_lead_ns = None
            on_demand_ns = first_deliverable_ns - endpoint_ns
        _synchronize(self.device)
        generation_done_ns = time.perf_counter_ns()
        output_ids = list(acc.assistant_token_ids)
        ttft_eff_ns = 0 if survived else first_deliverable_ns - endpoint_ns
        first_token_ready_ns = candidate_first_ns if survived else first_deliverable_ns
        return {
            "last_segment_arrival_ns": last_arrival_ns,
            "endpoint_accept_ns": endpoint_ns,
            "prefill_done_ns": endpoint_ns,
            "first_token_ready_ns": first_token_ready_ns,
            "first_deliverable_token_ns": first_deliverable_ns,
            "oracle_preaccept_processing_ns": endpoint_ns - last_arrival_ns,
            "arrival_to_first_token_ready_ns": first_token_ready_ns - last_arrival_ns,
            "consumer_delivery_ns": consumer_delivery_ns,
            "generation_done_ns": generation_done_ns,
            "ttft_eff_ns": ttft_eff_ns,
            "prefill_to_first_token_ns": None,
            "generation_total_ns": generation_done_ns - endpoint_ns,
            "survived": survived,
            "ready_tokens": ready,
            "candidate_started_ns": candidate_started_ns,
            "candidate_first_token_ns": candidate_first_ns,
            "candidate_lead_ns": candidate_lead_ns,
            "on_demand_ttft_ns": on_demand_ns,
            "n_speculations": n_speculations,
            "n_invalidated": n_invalidated,
            "wasted_tokens": wasted_tokens,
            "speculative_tokens": speculative_tokens,
            "final_tokens": len(output_ids),
            "eos": eos,
            "max_tokens_hit": len(output_ids) >= self.max_new_tokens and not eos,
            "output_token_ids": output_ids,
            "output_text": self._llm.tokenizer.decode(output_ids, skip_special_tokens=True),
            "trigger_prefixes": trigger_prefixes,
            "trigger_entry_hashes": _cache_entry_hashes(
                str(row["id"]), row["segments"], confidences
            ),
        }


def make_backend(
    kind: str,
    *,
    model_name: str | None,
    device: str,
    seed: int,
    max_new_tokens: int,
    spec_chunk: int,
) -> SessionBackend:
    if kind == "fake":
        from experiments.sci34_supplement.e1e2_confirmatory.fake_backend import FakeBackend

        return FakeBackend(seed=seed)
    if kind == "transformers":
        if not model_name:
            raise ValueError("--model is required for transformers runtime")
        return TransformersBackend(
            model_name,
            device=device,
            max_new_tokens=max_new_tokens,
            spec_chunk=spec_chunk,
        )
    raise ValueError(f"Unknown runtime kind: {kind}")
