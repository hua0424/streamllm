"""Frozen protocol and case-schema validation for the C2 equivalence campaign."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.sci34_supplement.common import canonical_json, config_hash, sha256_file


SCHEMA_VERSION = 3
EXPERIMENT = "c2_equivalence"
# v2（D-019）：v1 formal run c2eq_563dd22a_20260903T013547Z 在 token/state 层 100% 等价，
# 但冻结的 BF16 绝对 logit 阈（max_abs<=0.1 / mean_abs<=0.01）与 32-token greedy exact
# 门槛对任何正确实现都不可达成：增量 append 与整段 prefill 的 kernel 归约顺序不同
# （同形状重复计算差为 0，分块 append 在 0.5B/3060 与 7B/3090 上均给出同量级差异）。
# v2 改为"噪声对照臂 + 相对门槛"：同一 canonical 序列按结构 seam 分块增量预填充
# （不含任何 crop/生产恢复代码）测得本环境固有噪声，crop 路径偏差不得超过其 2 倍；
# top-1 翻转与 continuation 发散仅在近并列（margin 受限）处允许。全部常数先验冻结。
PROTOCOL_VERSION = 2
PRIOR_REJECTED_RUN_ID = "c2eq_563dd22a_20260903T013547Z"
PRIOR_REJECTED_COMMIT = "1a47ac1"
FORMAL_SESSION_COUNT = 1
FORMAL_CASE_COUNT = 24
# 与 D-017 已接受的 Qwen2-7B-Instruct snapshot 相同；忽略本地绝对路径，
# 对 file_count/total_bytes/逐文件内容哈希的规范 payload 再取 SHA-256。
EXPECTED_MODEL_ARTIFACT_HASH = "fae2ece10b76512237cf28957f98e7b0d2c609455a173031e3bd16b3dff7c5ab"
EXPECTED_MODEL_TYPE = "qwen2"
EXPECTED_MODEL_ARCHITECTURE = "Qwen2ForCausalLM"
EXPECTED_DTYPE = "torch.bfloat16"
CONTINUATION_TOKENS = 32
# v2：natural_eos cap 128→256（v1 中 4/10 greedy 在 128 内 run-on，属 cap×snapshot 组合）；
# 未在 cap 内 EOS 的 natural_eos case 重资格化为 max_tokens 语义继续做等价比较，
# campaign 级门槛要求 10 个 natural_eos 中至少 5 个真实命中 EOS（真实 EOS 分支覆盖）。
NATURAL_EOS_MAX_NEW_TOKENS = 256
NATURAL_EOS_MIN_GENUINE = 5
EOS_AT_CAP_MAX_NEW_TOKENS = 4
MAX_TOKENS_PROBE_BUDGET = 2
TERMINATION_PROBE_SCHEMA_VERSION = 1
TOP_K = 5
TOP_K_MIN_OVERLAP = 4
# v2 相对门槛常数（先验冻结，不得在看到 formal 结果后调整）：
NOISE_CONTROL_MAX_ABS_FLOOR = 0.05
NOISE_CONTROL_MEAN_ABS_FLOOR = 0.01
NOISE_RATIO_LIMIT = 2.0
LOGIT_MAX_ABS_BACKSTOP = 2.0
LOGIT_MEAN_ABS_BACKSTOP = 0.5
NEAR_TIE_MARGIN_FLOOR = 0.125  # BF16 在 |logit|∈[16,32) 的 ulp
NEAR_TIE_ABS_MARGIN_LIMIT = 0.5
SYSTEM_PROMPT = "You are a helpful assistant. Reply in English."
CONTEXT_TARGETS = (512, 2048, 8192)
CONTEXT_CLASSES = ("short_512", "medium_2048", "long_8192")
SCENARIOS = (
    "full_rollback_p0",
    "clean_fragment_boundary",
    "mid_fragment_snap_end",
    "reply_tail_noop",
    "crop_pending_eot",
    "speculation_full_invalidation",
    "next_user_next_assistant",
    "second_crop_later_turn",
)
TERMINATIONS = ("natural_eos", "eos_at_cap", "max_tokens")
CHECKPOINTS = ("post_recovery", "next_assistant", "post_second_recovery")


@dataclass(frozen=True)
class ProtocolConfig:
    sessions: int = FORMAL_SESSION_COUNT
    continuation_tokens: int = CONTINUATION_TOKENS
    top_k: int = TOP_K
    top_k_min_overlap: int = TOP_K_MIN_OVERLAP
    noise_control: str = "canonical_ids_boundary_seam_chunked_prefill"
    noise_ratio_limit: float = NOISE_RATIO_LIMIT
    max_abs_backstop: float = LOGIT_MAX_ABS_BACKSTOP
    mean_abs_backstop: float = LOGIT_MEAN_ABS_BACKSTOP
    near_tie_margin_floor: float = NEAR_TIE_MARGIN_FLOOR
    near_tie_abs_margin_limit: float = NEAR_TIE_ABS_MARGIN_LIMIT
    natural_eos_max_new_tokens: int = NATURAL_EOS_MAX_NEW_TOKENS
    natural_eos_min_genuine: int = NATURAL_EOS_MIN_GENUINE
    decode: str = "greedy"
    batch_size: int = 1
    dtype: str = "bfloat16"
    system_prompt: str = SYSTEM_PROMPT

    def validate(self) -> None:
        expected = ProtocolConfig()
        if self != expected:
            raise ValueError("C2 formal protocol is frozen and does not permit parameter overrides")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "protocol_version": PROTOCOL_VERSION,
                "prior_rejected_run": {
                    "run_id": PRIOR_REJECTED_RUN_ID,
                    "commit": PRIOR_REJECTED_COMMIT,
                    "note": "v1 absolute BF16 gates were unattainable for any correct implementation; see D-019",
                },
                "formal_case_count": FORMAL_CASE_COUNT,
                "expected_model_artifact_hash": EXPECTED_MODEL_ARTIFACT_HASH,
                "expected_model_type": EXPECTED_MODEL_TYPE,
                "expected_model_architecture": EXPECTED_MODEL_ARCHITECTURE,
                "expected_dtype": EXPECTED_DTYPE,
                "context_targets": list(CONTEXT_TARGETS),
                "context_classes": list(CONTEXT_CLASSES),
                "scenarios": list(SCENARIOS),
                "terminations": list(TERMINATIONS),
                "termination_probe": {
                    "schema_version": TERMINATION_PROBE_SCHEMA_VERSION,
                    "natural_eos_max_new_tokens": NATURAL_EOS_MAX_NEW_TOKENS,
                    "eos_at_cap_max_new_tokens": EOS_AT_CAP_MAX_NEW_TOKENS,
                    "max_tokens_budget": MAX_TOKENS_PROBE_BUDGET,
                    "natural_eos": (
                        "real greedy generate_accumulating; genuine EOS within frozen cap counts toward "
                        f"NATURAL_EOS_MIN_GENUINE={NATURAL_EOS_MIN_GENUINE}; a non-terminating greedy is "
                        "deterministically requalified to max_tokens semantics and still runs the full "
                        "equivalence comparison"
                    ),
                    "eos_at_cap": "controlled deterministic token-state fixture; EOT required at final cap step through generate_accumulating EOS branch",
                    "max_tokens": "real greedy generate_accumulating; no EOS in frozen small budget",
                },
                "checkpoints": list(CHECKPOINTS),
                "comparison": (
                    "independent termination probe plus production crop/recovery versus canonical token-ID clean re-prefill using shared retained IDs"
                ),
                "noise_control_arm": (
                    "the canonical checkpoint sequence is re-prefilled incrementally in chunks split at its "
                    "structural boundary seams (mirroring the path's append structure, ending with the same "
                    "one-token refresh forward); its FP32 logit difference versus the single-shot canonical "
                    "prefill measures the environment's intrinsic incremental-append BF16 noise"
                ),
                "v2_gates": {
                    "relative": (
                        f"path max_abs <= {NOISE_RATIO_LIMIT} * max(control max_abs, {NOISE_CONTROL_MAX_ABS_FLOOR}) "
                        f"and path mean_abs <= {NOISE_RATIO_LIMIT} * max(control mean_abs, {NOISE_CONTROL_MEAN_ABS_FLOOR})"
                    ),
                    "absolute_backstop": (
                        f"path max_abs <= {LOGIT_MAX_ABS_BACKSTOP} and path mean_abs <= {LOGIT_MEAN_ABS_BACKSTOP}"
                    ),
                    "top1": (
                        "top-1 must be exact, or the canonical top1-top2 margin must be within the near-tie "
                        f"limit min(max({NOISE_RATIO_LIMIT} * max(control max_abs, {NOISE_CONTROL_MAX_ABS_FLOOR}), "
                        f"{NEAR_TIE_MARGIN_FLOOR}), {NEAR_TIE_ABS_MARGIN_LIMIT})"
                    ),
                    "continuation": (
                        "the 32-token greedy continuation must be exact, or the canonical margin at the first "
                        "divergence step must be within the same near-tie limit"
                    ),
                },
                "statistics": "one deterministic session; no statistical repetition or bootstrap",
            }
        )
        return payload


def noise_limits(control_max_abs: float, control_mean_abs: float) -> tuple[float, float]:
    """v2 relative logit-difference limits derived from the measured control noise."""
    limit_max = NOISE_RATIO_LIMIT * max(float(control_max_abs), NOISE_CONTROL_MAX_ABS_FLOOR)
    limit_mean = NOISE_RATIO_LIMIT * max(float(control_mean_abs), NOISE_CONTROL_MEAN_ABS_FLOOR)
    return limit_max, limit_mean


def near_tie_margin_limit(control_max_abs: float) -> float:
    """Margin below which a top-1 flip / continuation divergence is numerically expected."""
    relative = NOISE_RATIO_LIMIT * max(float(control_max_abs), NOISE_CONTROL_MAX_ABS_FLOOR)
    return min(max(relative, NEAR_TIE_MARGIN_FLOOR), NEAR_TIE_ABS_MARGIN_LIMIT)


@dataclass(frozen=True)
class CaseSpec:
    id: str
    context_tokens: int
    context_class: str
    scenario: str
    termination: str
    user_prompt: str
    assistant_text: str
    fragments: tuple[str, ...]
    retain_fragment_count: int
    next_user: str | None
    second_assistant_text: str | None
    second_crop_fraction: float | None
    controlled_fixture: bool
    source: str

    @property
    def checkpoints(self) -> tuple[str, ...]:
        values = ["post_recovery"]
        if self.next_user is not None:
            values.append("next_assistant")
        if self.second_crop_fraction is not None:
            values.append("post_second_recovery")
        return tuple(values)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["fragments"] = list(self.fragments)
        payload["checkpoints"] = list(self.checkpoints)
        return payload


def _as_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def validate_cases(
    value: Any,
    *,
    formal: bool,
    expected_count: int | None = None,
) -> list[CaseSpec]:
    if not isinstance(value, list) or not value:
        raise ValueError("cases.json must contain a non-empty JSON list")
    count = len(value)
    if expected_count is not None and count != expected_count:
        raise ValueError(f"Expected {expected_count} cases, found {count}")
    if formal and count != FORMAL_CASE_COUNT:
        raise ValueError(
            f"Formal C2 requires exactly {FORMAL_CASE_COUNT} cases, found {count}"
        )

    seen: set[str] = set()
    cases: list[CaseSpec] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise ValueError(f"Case {index} is not an object")
        case_id = _as_nonempty_string(raw.get("id"), f"case {index} id")
        if case_id in seen:
            raise ValueError(f"Duplicate case id: {case_id}")
        context_tokens = int(raw.get("context_tokens", 0))
        context_class = _as_nonempty_string(raw.get("context_class"), f"{case_id}.context_class")
        scenario = _as_nonempty_string(raw.get("scenario"), f"{case_id}.scenario")
        termination = _as_nonempty_string(raw.get("termination"), f"{case_id}.termination")
        fragments_raw = raw.get("fragments")
        if not isinstance(fragments_raw, list) or not fragments_raw or not all(
            isinstance(fragment, str) and bool(fragment) for fragment in fragments_raw
        ):
            raise ValueError(f"{case_id}.fragments must contain non-empty strings")
        fragments = tuple(fragments_raw)
        assistant_text = _as_nonempty_string(raw.get("assistant_text"), f"{case_id}.assistant_text")
        if "".join(fragments) != assistant_text:
            raise ValueError(f"{case_id}: fragments are not a lossless assistant_text partition")
        retain = int(raw.get("retain_fragment_count", -1))
        if not 0 <= retain <= len(fragments):
            raise ValueError(f"{case_id}: retain_fragment_count is out of range")
        next_user = raw.get("next_user")
        second_assistant = raw.get("second_assistant_text")
        second_fraction = raw.get("second_crop_fraction")
        if next_user is not None:
            next_user = _as_nonempty_string(next_user, f"{case_id}.next_user")
            second_assistant = _as_nonempty_string(
                second_assistant, f"{case_id}.second_assistant_text"
            )
        elif second_assistant is not None or second_fraction is not None:
            raise ValueError(f"{case_id}: second-turn fields require next_user")
        if second_fraction is not None:
            second_fraction = float(second_fraction)
            if not 0.0 <= second_fraction <= 1.0:
                raise ValueError(f"{case_id}: second_crop_fraction must be in [0,1]")
        if context_tokens not in CONTEXT_TARGETS:
            raise ValueError(f"{case_id}: unsupported context_tokens={context_tokens}")
        if context_class not in CONTEXT_CLASSES:
            raise ValueError(f"{case_id}: unsupported context_class={context_class}")
        expected_class = CONTEXT_CLASSES[CONTEXT_TARGETS.index(context_tokens)]
        if context_class != expected_class:
            raise ValueError(f"{case_id}: context class does not match context target")
        if scenario not in SCENARIOS:
            raise ValueError(f"{case_id}: unsupported scenario={scenario}")
        if termination not in TERMINATIONS:
            raise ValueError(f"{case_id}: unsupported termination={termination}")
        controlled = bool(raw.get("controlled_fixture"))
        if termination == "eos_at_cap" and not controlled:
            raise ValueError(
                f"{case_id}: eos_at_cap requires controlled_fixture=true because token selection is deterministic"
            )
        if formal and scenario in {
            "full_rollback_p0",
            "crop_pending_eot",
            "speculation_full_invalidation",
            "second_crop_later_turn",
        } and not controlled:
            raise ValueError(f"{case_id}: controlled state scenario must be explicitly marked")
        seen.add(case_id)
        cases.append(
            CaseSpec(
                id=case_id,
                context_tokens=context_tokens,
                context_class=context_class,
                scenario=scenario,
                termination=termination,
                user_prompt=_as_nonempty_string(raw.get("user_prompt"), f"{case_id}.user_prompt"),
                assistant_text=assistant_text,
                fragments=fragments,
                retain_fragment_count=retain,
                next_user=next_user,
                second_assistant_text=second_assistant,
                second_crop_fraction=second_fraction,
                controlled_fixture=controlled,
                source=_as_nonempty_string(raw.get("source"), f"{case_id}.source"),
            )
        )

    if formal:
        present_contexts = {case.context_tokens for case in cases}
        present_scenarios = {case.scenario for case in cases}
        present_terminations = {case.termination for case in cases}
        missing = {
            "contexts": sorted(set(CONTEXT_TARGETS) - present_contexts),
            "scenarios": sorted(set(SCENARIOS) - present_scenarios),
            "terminations": sorted(set(TERMINATIONS) - present_terminations),
        }
        if any(missing.values()):
            raise ValueError(f"Formal case coverage incomplete: {missing}")
    return cases


def load_cases(path: Path, *, formal: bool) -> list[CaseSpec]:
    if not path.exists():
        raise FileNotFoundError(path)
    return validate_cases(json.loads(path.read_text(encoding="utf-8")), formal=formal)


def protocol_identity(cases_path: Path) -> dict[str, Any]:
    protocol = ProtocolConfig()
    protocol.validate()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "protocol": protocol.to_dict(),
        "cases": {"path": str(cases_path.resolve()), "sha256": sha256_file(cases_path)},
    }
    payload["identity_hash"] = config_hash(payload)
    return payload


def assert_exact_identity(existing: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    if existing.get("identity_hash") != expected.get("identity_hash"):
        raise ValueError(f"{label} identity hash mismatch")
    if canonical_json(existing) != canonical_json(expected):
        raise ValueError(f"{label} payload differs despite matching identity hash")


def expected_record_keys(cases: Sequence[CaseSpec]) -> set[str]:
    return {case.id for case in cases}
