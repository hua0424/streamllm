"""Greedy chat runtimes for the fixed-trajectory E3 experiment.

The fake runtime is deterministic and model-free.  It validates orchestration,
persistence, deduplication, and statistics locally without importing torch.
The transformers runtime loads the project's ``StreamLLMInference`` lazily and
is used only on the GPU host.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from experiments.sci34_supplement.common import stable_seed


SYSTEM_PROMPT = "You are a helpful assistant. Reply in English."


@dataclass(frozen=True)
class GeneratedTurn:
    text: str
    token_ids: list[int]
    token_texts: list[str]


class ChatRuntime(Protocol):
    model_name: str
    revision: str | None

    def generate(self, messages: Sequence[dict[str, str]], *, max_new_tokens: int) -> GeneratedTurn:
        ...

    def decode(self, token_ids: Sequence[int]) -> str:
        ...


class FakeChatRuntime:
    """Deterministic word-token runtime used by the no-model smoke suite."""

    model_name = "fake-word-runtime"
    revision = "smoke-v1"

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.calls: list[str] = []
        self._vocabulary: dict[str, int] = {}

    def _tokenize(self, text: str) -> tuple[list[int], list[str]]:
        pieces = re.findall(r"\s+|[^\s]+", text)
        ids: list[int] = []
        for piece in pieces:
            if piece not in self._vocabulary:
                self._vocabulary[piece] = len(self._vocabulary) + 1
            ids.append(self._vocabulary[piece])
        return ids, pieces

    def restore_tokens(self, token_ids: Sequence[int], token_texts: Sequence[str]) -> None:
        for token_id, piece in zip(token_ids, token_texts):
            existing = self._vocabulary.get(piece)
            if existing is not None and existing != int(token_id):
                raise ValueError(f"Conflicting fake token mapping for {piece!r}")
            self._vocabulary[piece] = int(token_id)

    def decode(self, token_ids: Sequence[int]) -> str:
        reverse = {token_id: piece for piece, token_id in self._vocabulary.items()}
        return "".join(reverse[int(token_id)] for token_id in token_ids)

    def generate(self, messages: Sequence[dict[str, str]], *, max_new_tokens: int) -> GeneratedTurn:
        signature = "|".join(f"{message['role']}:{message['content']}" for message in messages)
        self.calls.append(signature)
        last_user = next(
            (message["content"] for message in reversed(messages) if message["role"] == "user"),
            "the request",
        )
        digest = hashlib.sha256(
            f"{self.seed}|{signature}".encode("utf-8")
        ).hexdigest()[:8]
        if len(messages) <= 2:
            text = (
                f"I can help with {last_user.strip()}. First, note reference {digest}. "
                "Second, keep the generated trajectory fixed. Third, compare only the retained history."
            )
        else:
            assistant_history = " ".join(
                message["content"] for message in messages if message["role"] == "assistant"
            )
            tail = assistant_history.split()[-5:]
            text = f"For {last_user.strip()}, the retained context is {' '.join(tail)} ({digest})."
        ids, token_texts = self._tokenize(text)
        ids = ids[:max_new_tokens]
        token_texts = token_texts[:max_new_tokens]
        return GeneratedTurn("".join(token_texts), ids, token_texts)


class TransformersChatRuntime:
    """Greedy chat runtime backed by the project's 7B model wrapper."""

    def __init__(self, model_name: str, *, device: str = "cuda", seed: int = 20260831):
        from experiments.sci34_supplement.common import enforce_offline_mode

        enforce_offline_mode()
        from src.llm.stream_llm_inference import StreamLLMInference

        self.model_name = model_name
        self.seed = seed
        self._llm = StreamLLMInference(model_name=model_name, device=device, eval_mode=False)
        self.revision = getattr(self._llm.model.config, "_commit_hash", None)

    @property
    def tokenizer(self):
        return self._llm.tokenizer

    def decode(self, token_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(token_ids), skip_special_tokens=True)

    def generate(self, messages: Sequence[dict[str, str]], *, max_new_tokens: int) -> GeneratedTurn:
        import torch

        item_seed = stable_seed(
            self.seed,
            *[f"{message['role']}:{message['content']}" for message in messages],
        )
        torch.manual_seed(item_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(item_seed)
        prompt = self.tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True
        )
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self._llm.device)
        with torch.no_grad():
            output = self._llm.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        new_ids = output[0, encoded.input_ids.shape[1] :].tolist()
        token_texts = [
            self.tokenizer.decode([token_id], skip_special_tokens=True)
            for token_id in new_ids
        ]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return GeneratedTurn(text=text, token_ids=new_ids, token_texts=token_texts)


def make_runtime(kind: str, *, model_name: str | None, device: str, seed: int) -> ChatRuntime:
    if kind == "fake":
        return FakeChatRuntime(seed=seed)
    if kind == "transformers":
        if not model_name:
            raise ValueError("--model is required for the transformers runtime")
        return TransformersChatRuntime(model_name, device=device, seed=seed)
    raise ValueError(f"Unknown runtime: {kind}")
