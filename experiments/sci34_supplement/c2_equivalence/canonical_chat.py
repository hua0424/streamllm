"""Canonical token-ID chat construction without assistant decode/re-encode round trips."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from experiments.sci34_supplement.common import canonical_json, sha256_bytes


@dataclass(frozen=True)
class CanonicalSequence:
    token_ids: tuple[int, ...]
    boundaries: dict[str, Any]
    eot_positions: tuple[int, ...]
    special_token_ids: dict[str, int]

    @property
    def token_hash(self) -> str:
        return token_ids_hash(self.token_ids)

    def to_dict(self, *, include_ids: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "token_hash": self.token_hash,
            "token_count": len(self.token_ids),
            "boundaries": dict(self.boundaries),
            "eot_positions": list(self.eot_positions),
            "special_token_ids": dict(self.special_token_ids),
        }
        if include_ids:
            payload["token_ids"] = list(self.token_ids)
        return payload


def token_ids_hash(token_ids: Iterable[int]) -> str:
    normalized = [int(value) for value in token_ids]
    return sha256_bytes(canonical_json(normalized).encode("utf-8"))


def first_mismatch(left: Sequence[int], right: Sequence[int]) -> int | None:
    common = min(len(left), len(right))
    for index in range(common):
        if int(left[index]) != int(right[index]):
            return index
    return None if len(left) == len(right) else common


def _flatten_template_ids(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("C2 supports batch-size-one chat templates only")
        value = value[0]
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        raise TypeError("apply_chat_template(tokenize=True) did not return integer token IDs")
    return [int(item) for item in value]


def apply_chat_ids(tokenizer: Any, messages: Sequence[Mapping[str, str]], *, add_generation_prompt: bool) -> list[int]:
    return _flatten_template_ids(
        tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )
    )


def _prefix_delta(prefix: Sequence[int], full: Sequence[int], label: str) -> list[int]:
    if list(full[: len(prefix)]) != list(prefix):
        raise ValueError(f"Chat template is not prefix-decomposable at {label}")
    delta = list(full[len(prefix) :])
    if not delta:
        raise ValueError(f"Chat template produced an empty structural delta at {label}")
    return delta


@dataclass(frozen=True)
class ChatTemplateParts:
    prefix_to_user_content: tuple[int, ...]
    user_to_assistant: tuple[int, ...]
    assistant_to_user: tuple[int, ...]
    assistant_eot: tuple[int, ...]
    assistant_open: tuple[int, ...]
    eot_token_id: int
    assistant_close_eot_offset: int
    eos_token_id: int
    pad_token_id: int | None
    template_hash: str

    @classmethod
    def from_tokenizer(cls, tokenizer: Any, system_prompt: str) -> "ChatTemplateParts":
        user_marker = "C2_USER_MARKER_718263"
        assistant_marker = "C2_ASSISTANT_MARKER_918273"
        next_user_marker = "C2_NEXT_USER_MARKER_192837"
        system = {"role": "system", "content": system_prompt}
        user = {"role": "user", "content": user_marker}
        assistant = {"role": "assistant", "content": assistant_marker}
        next_user = {"role": "user", "content": next_user_marker}
        user_content_ids = tokenizer.encode(user_marker, add_special_tokens=False)
        assistant_content_ids = tokenizer.encode(assistant_marker, add_special_tokens=False)
        next_user_content_ids = tokenizer.encode(next_user_marker, add_special_tokens=False)
        if not user_content_ids or not assistant_content_ids or not next_user_content_ids:
            raise ValueError("Marker text did not tokenize")

        assistant_open_full = apply_chat_ids(tokenizer, [system, user], add_generation_prompt=True)
        user_at = len(assistant_open_full) - len(user_content_ids)
        while user_at >= 0 and assistant_open_full[user_at:user_at + len(user_content_ids)] != list(user_content_ids):
            user_at -= 1
        if user_at < 0:
            raise ValueError("Cannot locate user content in canonical chat template")
        user_full = assistant_open_full[: user_at + len(user_content_ids)]
        user_open = user_full[:user_at]
        user_to_assistant = _prefix_delta(user_full, assistant_open_full, "user_to_assistant")
        assistant_closed = apply_chat_ids(tokenizer, [system, user, assistant], add_generation_prompt=False)
        if assistant_closed[: len(assistant_open_full)] != assistant_open_full:
            raise ValueError("Assistant template prefix differs from generation prompt")
        content_start = len(assistant_open_full)
        if assistant_closed[content_start:content_start + len(assistant_content_ids)] != list(assistant_content_ids):
            raise ValueError("Assistant content is not embedded as raw canonical token IDs")
        eos_token_id = int(tokenizer.eos_token_id)
        eot_candidates = [
            index for index in range(content_start + len(assistant_content_ids), len(assistant_closed))
            if assistant_closed[index] == eos_token_id
        ]
        if len(eot_candidates) != 1:
            raise ValueError("C2 requires exactly one structural assistant EOT token")
        content_end = eot_candidates[0]
        assistant_eot = assistant_closed[content_end:]
        if assistant_eot.count(eos_token_id) != 1:
            raise ValueError("Assistant close transition does not contain exactly one EOT")

        next_user_full = apply_chat_ids(
            tokenizer,
            [system, user, assistant, next_user],
            add_generation_prompt=True,
        )
        next_user_at = len(next_user_full) - len(next_user_content_ids)
        while next_user_at >= len(assistant_closed) and next_user_full[next_user_at:next_user_at + len(next_user_content_ids)] != list(next_user_content_ids):
            next_user_at -= 1
        if next_user_at < len(assistant_closed):
            raise ValueError("Cannot locate next-user content in canonical chat template")
        next_user_open_full = next_user_full[:next_user_at]
        assistant_to_user = _prefix_delta(assistant_closed, next_user_open_full, "assistant_to_user")
        if assistant_to_user[0] == assistant_eot[0]:
            raise ValueError("assistant_to_user unexpectedly contains a duplicate EOT")

        eot_token_id = eos_token_id
        if assistant_eot.count(eot_token_id) != 1:
            raise ValueError("Qwen C2 requires exactly one EOS/EOT in assistant close transition")
        eot_offset = assistant_eot.index(eot_token_id)
        template_text = getattr(tokenizer, "chat_template", None)
        if not isinstance(template_text, str) or not template_text:
            raise ValueError("Tokenizer lacks an explicit chat_template")
        return cls(
            prefix_to_user_content=tuple(user_open),
            user_to_assistant=tuple(user_to_assistant),
            assistant_to_user=tuple(assistant_to_user),
            assistant_eot=tuple(assistant_eot),
            assistant_open=tuple(assistant_open_full[len(user_full) :]),
            eot_token_id=eot_token_id,
            assistant_close_eot_offset=eot_offset,
            eos_token_id=eos_token_id,
            pad_token_id=(int(tokenizer.pad_token_id) if tokenizer.pad_token_id is not None else None),
            template_hash=sha256_bytes(template_text.encode("utf-8")),
        )

    def special_tokens(self) -> dict[str, int]:
        values = {"eot": self.eot_token_id, "eos": self.eos_token_id}
        if self.pad_token_id is not None:
            values["pad"] = self.pad_token_id
        return values


def _position_of_all(token_ids: Sequence[int], target: int) -> tuple[int, ...]:
    return tuple(index for index, token in enumerate(token_ids) if int(token) == int(target))


def validate_initial_open_boundaries(
    sequence: CanonicalSequence,
    parts: ChatTemplateParts,
    user_ids: Sequence[int],
) -> None:
    """Fail closed if a rollback boundary lands inside the role transition."""
    boundaries = sequence.boundaries
    user_start = int(boundaries["user_content_start"])
    user_end = int(boundaries["user_content_end"])
    role_start = int(boundaries["assistant_role_start"])
    content_start = int(boundaries["assistant_content_start"])
    if user_end != role_start:
        raise ValueError("assistant_role_start must equal raw user content end")
    if content_start != role_start + len(parts.user_to_assistant):
        raise ValueError("assistant_content_start must follow the complete role transition")
    if list(sequence.token_ids[user_start:user_end]) != [int(value) for value in user_ids]:
        raise ValueError("canonical raw user content span differs")
    if list(sequence.token_ids[role_start:content_start]) != list(parts.user_to_assistant):
        raise ValueError("canonical assistant transition span differs")


def build_initial_open_sequence(
    tokenizer: Any,
    parts: ChatTemplateParts,
    *,
    user_prompt: str,
) -> CanonicalSequence:
    user_ids = [int(value) for value in tokenizer.encode(user_prompt, add_special_tokens=False)]
    if not user_ids:
        raise ValueError("User prompt tokenized to an empty sequence")
    token_ids = [*parts.prefix_to_user_content, *user_ids, *parts.user_to_assistant]
    boundaries = {
        "user_content_start": len(parts.prefix_to_user_content),
        "user_content_end": len(parts.prefix_to_user_content) + len(user_ids),
        "assistant_role_start": len(parts.prefix_to_user_content) + len(user_ids),
        "assistant_content_start": len(token_ids),
        "assistant_eot_positions": [],
    }
    sequence = CanonicalSequence(
        token_ids=tuple(token_ids),
        boundaries=boundaries,
        eot_positions=_position_of_all(token_ids, parts.eot_token_id),
        special_token_ids=parts.special_tokens(),
    )
    validate_initial_open_boundaries(sequence, parts, user_ids)
    return sequence


def append_assistant_ids(
    sequence: CanonicalSequence,
    assistant_ids: Sequence[int],
    parts: ChatTemplateParts,
    *,
    close_assistant: bool,
    open_user: bool,
    user_ids: Sequence[int] = (),
    open_assistant: bool = False,
) -> CanonicalSequence:
    if open_user and not close_assistant:
        raise ValueError("A user role cannot open before the assistant is closed")
    if user_ids and not open_user:
        raise ValueError("User token IDs require an open user role")
    tokens = list(sequence.token_ids)
    boundaries = dict(sequence.boundaries)
    boundaries["assistant_content_start"] = len(tokens)
    tokens.extend(int(value) for value in assistant_ids)
    boundaries["assistant_content_end"] = len(tokens)
    boundaries["assistant_close_end"] = len(tokens)
    spans = list(boundaries.get("assistant_content_spans", []))
    spans.append([boundaries["assistant_content_start"], boundaries["assistant_content_end"]])
    boundaries["assistant_content_spans"] = spans
    if close_assistant:
        boundaries["assistant_eot"] = len(tokens) + parts.assistant_close_eot_offset
        assistant_eot_positions = list(boundaries.get("assistant_eot_positions", []))
        assistant_eot_positions.append(boundaries["assistant_eot"])
        boundaries["assistant_eot_positions"] = assistant_eot_positions
        tokens.extend(parts.assistant_eot)
        boundaries["assistant_close_end"] = len(tokens)
        if open_user:
            tokens.extend(parts.assistant_to_user)
    if open_user:
        boundaries["user_role_start"] = boundaries["assistant_close_end"]
        if not close_assistant:
            tokens.extend(parts.assistant_to_user)
        boundaries["next_user_content_start"] = len(tokens)
        tokens.extend(int(value) for value in user_ids)
        boundaries["next_user_content_end"] = len(tokens)
    if open_assistant:
        boundaries["next_assistant_role_start"] = len(tokens)
        tokens.extend(parts.user_to_assistant)
        boundaries["next_assistant_content_start"] = len(tokens)
    return CanonicalSequence(
        token_ids=tuple(tokens),
        boundaries=boundaries,
        eot_positions=_position_of_all(tokens, parts.eot_token_id),
        special_token_ids=parts.special_tokens(),
    )
