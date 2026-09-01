"""Deterministic model-free backend for confirmatory runner contract tests."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from experiments.sci34_supplement.common import canonical_json
from experiments.sci34_supplement.e1e2_confirmatory.protocol import NEVER_SPECULATE, SYSTEM_A


@dataclass
class FakeBackend:
    """Small deterministic stand-in for generation and warmup paths.

    The methods intentionally expose semantic measurements rather than sleeping,
    which makes smoke assertions stable on slow CI hosts.
    """

    seed: int = 20260901
    model_name: str = "fake://e1e2-confirmatory"
    revision: str = "v1"
    calls: list[dict[str, Any]] = field(default_factory=list)

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "requested": self.model_name,
            "revision": self.revision,
            "identity_hash": hashlib.sha256(
                canonical_json([self.model_name, self.revision, self.seed]).encode("utf-8")
            ).hexdigest(),
            "resolved_dtype": "fake-float32",
            "attention_backend": "fake-eager",
        }

    @property
    def runtime_metadata(self) -> dict[str, Any]:
        return {
            "resolved_dtype": "fake-float32",
            "attention_backend": "fake-eager",
        }

    def _unit(self, *parts: object) -> float:
        digest = hashlib.sha256(
            canonical_json([self.seed, *[str(part) for part in parts]]).encode("utf-8")
        ).digest()
        return int.from_bytes(digest[:8], "big") / float(2**64)

    def warmup(self, path_kind: str) -> None:
        self.calls.append({"kind": "warmup", "path_kind": path_kind})

    def run_condition(
        self,
        row: Mapping[str, Any],
        *,
        condition: str,
        threshold: float | None,
        confidences: Sequence[float],
        session_id: str,
    ) -> dict[str, Any]:
        sample_id = str(row["id"])
        self.calls.append(
            {"kind": "formal", "id": sample_id, "condition": condition, "session_id": session_id}
        )
        endpoint_ns = 10_000_000_000 + int(self._unit(sample_id, session_id) * 1_000_000)
        if condition in ("system_a", SYSTEM_A):
            last_arrival_ns = endpoint_ns
            ttft_ns = 40_000_000 + int(self._unit(sample_id, "a", session_id) * 10_000_000)
            first_ns = endpoint_ns + ttft_ns
            consumer_ns = first_ns + 100_000
            return {
                "last_segment_arrival_ns": last_arrival_ns,
                "endpoint_accept_ns": endpoint_ns,
                "first_token_ready_ns": first_ns,
                "first_deliverable_token_ns": first_ns,
                "oracle_preaccept_processing_ns": endpoint_ns - last_arrival_ns,
                "arrival_to_first_token_ready_ns": first_ns - last_arrival_ns,
                "ttft_eff_ns": ttft_ns,
                "consumer_delivery_ns": consumer_ns,
                "generation_done_ns": consumer_ns + 20_000_000,
                "survived": False,
                "ready_tokens": 0,
                "candidate_started_ns": None,
            "candidate_first_token_ns": None,
            "candidate_lead_ns": None,
                "on_demand_ttft_ns": ttft_ns,
                "n_speculations": 0,
                "n_invalidated": 0,
                "wasted_tokens": 0,
                "final_tokens": 32,
                "eos": False,
                "output_text": f"fake response for {sample_id}",
            }

        never = condition in ("never_speculate", NEVER_SPECULATE)
        threshold_value = math.inf if never else float(threshold)
        trigger_indices = [
            index for index, confidence in enumerate(confidences, start=1)
            if confidence >= threshold_value
        ]
        final_prefix = len(confidences)
        survived = final_prefix in trigger_indices
        invalidated = len([index for index in trigger_indices if index != final_prefix])
        ready_tokens = 12 if survived else 0
        on_demand_ns = 45_000_000 + int(
            self._unit(sample_id, condition, session_id) * 12_000_000
        )
        ttft_ns = 0 if survived else on_demand_ns
        wasted = invalidated * 12
        oracle_preaccept_ns = 6_000_000
        last_arrival_ns = endpoint_ns - oracle_preaccept_ns
        candidate_first_ns = last_arrival_ns + 2_000_000 if survived else None
        first_ready_ns = candidate_first_ns if survived else endpoint_ns + on_demand_ns
        first_ns = endpoint_ns + ttft_ns
        consumer_ns = first_ns + 150_000
        return {
            "last_segment_arrival_ns": last_arrival_ns,
            "endpoint_accept_ns": endpoint_ns,
            "first_token_ready_ns": first_ready_ns,
            "first_deliverable_token_ns": first_ns,
            "oracle_preaccept_processing_ns": oracle_preaccept_ns,
            "arrival_to_first_token_ready_ns": first_ready_ns - last_arrival_ns,
            "ttft_eff_ns": ttft_ns,
            "consumer_delivery_ns": consumer_ns,
            "generation_done_ns": consumer_ns + 20_000_000,
            "survived": survived,
            "ready_tokens": ready_tokens,
            "candidate_started_ns": last_arrival_ns + 1_000_000 if survived else None,
            "candidate_first_token_ns": candidate_first_ns,
            "candidate_lead_ns": endpoint_ns - candidate_first_ns if survived else None,
            "on_demand_ttft_ns": None if survived else on_demand_ns,
            "n_speculations": len(trigger_indices),
            "n_invalidated": invalidated,
            "wasted_tokens": wasted,
            "final_tokens": 32,
            "eos": False,
            "output_text": f"fake response for {sample_id}",
        }
