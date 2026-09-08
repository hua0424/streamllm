"""State fixtures used by the A1 and asynchronous control-path benchmarks."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol


FILLER = (
    "The city has museums, parks, restaurants, railway stations, hotels, and "
    "historical sites that attract visitors throughout the year. "
)


class KVFixture(Protocol):
    actual_length: int
    assistant_start: int

    def ensure_full(self) -> None:
        ...

    def crop(self, keep_length: int) -> None:
        ...

    def recover_role(self) -> None:
        ...

    def reprefill(self, keep_length: int) -> None:
        ...

    def synchronize(self) -> None:
        ...


@dataclass
class FakeKVFixture:
    actual_length: int
    assistant_start: int = 64
    crop_delay_s: float = 0.00008
    role_delay_s: float = 0.00025
    reprefill_scale_s: float = 0.0000002

    def ensure_full(self) -> None:
        return None

    def crop(self, keep_length: int) -> None:
        if not 0 <= keep_length <= self.actual_length:
            raise ValueError(keep_length)
        time.sleep(self.crop_delay_s)

    def recover_role(self) -> None:
        time.sleep(self.role_delay_s)

    def reprefill(self, keep_length: int) -> None:
        time.sleep(max(0.00005, keep_length * self.reprefill_scale_s))

    def synchronize(self) -> None:
        return None


class TransformersKVFixture:
    """Mutable real-model fixture restored outside every measured interval."""

    def __init__(self, llm, target_length: int):
        import torch

        self.llm = llm
        self.torch = torch
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Reply in English."},
            {"role": "user", "content": "Tell me about the city."},
        ]
        base_ids = llm._token_list(llm.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        ))
        filler_ids = llm.tokenizer(
            FILLER, return_tensors="pt", add_special_tokens=False
        ).input_ids[0].tolist()
        if not filler_ids:
            raise RuntimeError("Filler text produced no tokens")
        if target_length <= len(base_ids):
            raise ValueError(
                f"target_length={target_length} must exceed base prompt length={len(base_ids)}"
            )
        repeated = filler_ids * (((target_length - len(base_ids)) // len(filler_ids)) + 2)
        self._full_ids = (base_ids + repeated)[:target_length]
        self.actual_length = len(self._full_ids)
        self.assistant_start = len(base_ids)
        ids = torch.tensor([self._full_ids], dtype=torch.long, device=llm.device)
        mask = torch.ones_like(ids)
        with torch.no_grad():
            outputs = llm.model(
                input_ids=ids,
                attention_mask=mask,
                use_cache=True,
                return_dict=True,
            )
        self.acc = llm.AccumKVCache(
            past_key_values=llm._as_dynamic_cache(outputs.past_key_values),
            attention_mask=mask,
            next_token_logits=outputs.logits[:, -1, :],
            seq_length=self.actual_length,
            token_ids=self._full_ids,
            role_phase=llm.RolePhase.ASSISTANT_OPEN,
            assistant_role_start=self.assistant_start - len(llm._user_to_assistant_ids),
            assistant_content_start=self.assistant_start,
            assistant_token_ids=self._full_ids[self.assistant_start :],
            generation_end_reason=llm.GenerationEndReason.MAX_TOKENS,
        )
        llm._assert_accum_consistent(self.acc)
        self._role_ids = list(llm._assistant_to_user_ids)
        self._reprefill_ids = self._full_ids

    @property
    def device(self) -> str:
        return str(self.llm.device)

    def synchronize(self) -> None:
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize()

    def ensure_full(self) -> None:
        if self.acc.seq_length > self.actual_length:
            self.llm.crop_to_token(self.acc, self.actual_length)
        if self.acc.seq_length < self.actual_length:
            missing = self._full_ids[self.acc.seq_length : self.actual_length]
            ids = self.torch.tensor([missing], dtype=self.torch.long, device=self.llm.device)
            mask = self.torch.cat(
                [self.acc.attention_mask, self.torch.ones_like(ids)], dim=-1
            )
            positions = self.torch.arange(
                self.acc.seq_length,
                self.acc.seq_length + ids.shape[1],
                dtype=self.torch.long,
                device=self.llm.device,
            ).unsqueeze(0)
            with self.torch.no_grad():
                outputs = self.llm.model(
                    input_ids=ids,
                    attention_mask=mask,
                    position_ids=positions,
                    past_key_values=self.acc.past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
            self.acc.past_key_values = self.llm._as_dynamic_cache(outputs.past_key_values)
            self.acc.attention_mask = mask
            self.acc.token_ids.extend(missing)
            self.acc.seq_length += ids.shape[1]
            self.acc.next_token_logits = outputs.logits[:, -1, :]
        self.acc.token_ids = list(self._full_ids)
        self.acc.role_phase = self.llm.RolePhase.ASSISTANT_OPEN
        self.acc.assistant_role_start = self.assistant_start - len(self.llm._user_to_assistant_ids)
        self.acc.assistant_content_start = self.assistant_start
        self.acc.assistant_content_end = None
        self.acc.assistant_role_end = None
        self.acc.assistant_token_ids = self._full_ids[
            self.assistant_start : self.actual_length
        ]
        self.acc.generation_end_reason = self.llm.GenerationEndReason.MAX_TOKENS
        self.llm._assert_accum_consistent(self.acc)

    def crop(self, keep_length: int) -> None:
        self.llm.crop_to_token(self.acc, keep_length)

    def recover_role(self) -> None:
        self.llm.reopen_user_role(self.acc)

    def reprefill(self, keep_length: int) -> None:
        comparable_ids = self._reprefill_ids[:keep_length] + self._role_ids
        ids = self.torch.tensor(
            [comparable_ids], dtype=self.torch.long, device=self.llm.device
        )
        mask = self.torch.ones_like(ids)
        with self.torch.no_grad():
            self.llm.model(
                input_ids=ids,
                attention_mask=mask,
                use_cache=True,
                return_dict=True,
            )


def make_kv_fixture(
    runtime: str,
    *,
    target_length: int,
    llm=None,
) -> KVFixture:
    if runtime == "fake":
        return FakeKVFixture(actual_length=target_length)
    if runtime == "transformers":
        if llm is None:
            raise ValueError("A loaded StreamLLMInference instance is required")
        return TransformersKVFixture(llm, target_length)
    raise ValueError(runtime)


def timed_ms(fixture: KVFixture, callback) -> float:
    fixture.synchronize()
    start = time.perf_counter_ns()
    callback()
    fixture.synchronize()
    return (time.perf_counter_ns() - start) / 1_000_000
