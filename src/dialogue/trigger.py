# src/dialogue/trigger.py
"""
软触发（Soft Trigger）—— turn 完成度的连续置信度判断器（贡献1 的判断内核）。

与传统端点检测（YES/NO 硬决策）不同（见 paper2_context.md §3.5）：本模块输出
p(用户话语已构成可回复的语义单元) ∈ [0,1]，由编排层配两个阈值使用：
  - 推测阈值（低/激进）：conf >= spec_th → 触发推测生成（可作废）
  - 提交阈值（高/保守）：conf >= commit_th → 允许 TTS 播放

实现（D-011）：统一 LLMSoftTrigger，用因果 LM 的**首生成 token 在类别词上的
logits softmax** 作为连续置信度——
  - 验证机开发替身：prompted Qwen2.5-0.5B，类别词 YES/NO
  - 实验机正式：TEN Turn Detection 7B，类别词 finished/unfinished/wait
两者同一代码路径，仅 TriggerConfig 不同。软触发推理与 KV prefill 并行（挂在
prefill 延迟阴影里，D-003）；本模块只管单次评估，调度在编排层。
"""

from dataclasses import dataclass, field
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import HF_HOME, P2_DEVICE, P2_TRIGGER_MODEL_NAME
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class TriggerConfig:
    model_name: str
    system_prompt: Optional[str]
    user_template: str              # {text} 处填入累积的用户文本
    positive_words: List[str]       # 判"已完成"的类别词（取首 token）
    negative_words: List[str]       # 判"未完成"的类别词
    device: str = P2_DEVICE         # 集中于 src/config.py（.env 可覆盖），勿硬编码


# 验证机开发替身（D-011）：prompted Qwen2.5-0.5B（模型名走 config，实验机 .env 覆盖）
QWEN05B_DEV_CONFIG = TriggerConfig(
    model_name=P2_TRIGGER_MODEL_NAME,
    system_prompt=(
        "Task: decide if the speaker has FINISHED their sentence.\n"
        "A FINISHED utterance is a grammatically complete question or request.\n"
        "An UNFINISHED utterance stops mid-sentence (ends in an article, preposition, "
        "conjunction, or an incomplete clause) — the speaker will keep talking.\n"
        "Reply with exactly one word: YES (finished) or NO (unfinished)."
    ),
    user_template=(
        "\"Where is the nearest train station?\" -> YES\n"
        "\"Can you recommend a\" -> NO\n"
        "\"Please turn off the lights in the living room.\" -> YES\n"
        "\"I was wondering if you could\" -> NO\n"
        "\"Do we have any meetings on\" -> NO\n"
        "\"Play some jazz music.\" -> YES\n"
        "\"{text}\" ->"
    ),
    positive_words=["YES", " YES", "Yes", " Yes"],
    negative_words=["NO", " NO", "No", " No"],
)

# 实验机正式（D-003/D-011）：TEN Turn Detection 7B（同一代码路径，换 config 即可）
TEN_CONFIG = TriggerConfig(
    model_name="TEN-framework/TEN_Turn_Detection",
    system_prompt=None,                 # TEN 用自身 chat template，无需额外 system
    user_template="{text}",
    positive_words=["finished", " finished"],
    negative_words=["unfinished", " unfinished", "wait", " wait"],
)


class LLMSoftTrigger:
    """用因果 LM 首 token 类别词概率作为 turn 完成度置信度。"""

    def __init__(self, config: TriggerConfig = QWEN05B_DEV_CONFIG, hf_home: str = HF_HOME):
        self.config = config
        device = config.device
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        logger.info(f"Loading soft-trigger model {config.model_name} on {device}")
        kw = dict(cache_dir=hf_home, trust_remote_code=True)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, local_files_only=True, **kw)
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_name, local_files_only=True, dtype="auto", **kw)
        except Exception:
            logger.warning("soft-trigger 本地缓存未命中，尝试在线下载…")
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, local_files_only=False, **kw)
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_name, local_files_only=False, dtype="auto", **kw)
        self.model.to(device).eval()

        self._pos_ids = self._first_token_ids(config.positive_words)
        self._neg_ids = self._first_token_ids(config.negative_words)
        overlap = set(self._pos_ids) & set(self._neg_ids)
        if overlap:
            raise ValueError(f"正/负类别词首 token 冲突: {overlap}")
        logger.info(f"soft-trigger ready (pos_ids={self._pos_ids}, neg_ids={self._neg_ids})")

    def _first_token_ids(self, words: List[str]) -> List[int]:
        ids = []
        for w in words:
            toks = self.tokenizer.encode(w, add_special_tokens=False)
            if toks and toks[0] not in ids:
                ids.append(toks[0])
        return ids

    @torch.no_grad()
    def confidence(self, accumulated_text: str) -> float:
        """返回 p(已完成) ∈ [0,1]：正类首 token 概率 / (正类+负类)。"""
        text = self.config.user_template.format(text=accumulated_text.strip())
        messages = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.append({"role": "user", "content": text})
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        logits = self.model(**inputs).logits[0, -1, :]          # 首生成 token 的 logits
        pos = torch.logsumexp(logits[self._pos_ids], dim=0)
        neg = torch.logsumexp(logits[self._neg_ids], dim=0)
        p = torch.sigmoid(pos - neg).item()                     # = e^pos/(e^pos+e^neg)
        return p
