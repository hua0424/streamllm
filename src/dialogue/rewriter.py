# src/dialogue/rewriter.py
"""
对话历史自然化重写器（贡献3，扩展贡献）。

场景（paper2_context.md §2.3 贡献3）：用户在句中打断，播放感知截断后历史里的
assistant 内容语义不完整（"…温度25度，"）。重写法用轻量模型把它改写成
**在同一信息量上自然收尾**的版本，避免半句话影响后续轮次生成的连贯性。

关键约束：**重写不得新增信息**——历史只能包含用户真正听到的信息，
否则比 B-gen 还糟（说了没说过的话）。prompt 里强制。

模型：Qwen3-0.6B（D-004），独立实例、独立 device，直接 prompt 不微调。
架构上重写与用户说话并行（延迟被隐藏，见 §2.3 重写效率分析）；
本模块只做单次同步调用并回报耗时，编排层记录 rewrite_ms 供论文验证可隐藏性。
"""

import time
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import HF_HOME
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

REWRITE_SYSTEM = (
    "You fix an assistant reply that was cut off mid-sentence when the user interrupted. "
    "Rewrite it so it ends naturally and grammatically at the same point. "
    "You MUST NOT add any new information, facts, or continuation beyond what is already "
    "stated. Prefer trimming the trailing incomplete clause. Output ONLY the rewritten reply."
)


class HistoryRewriter:
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B", device: str = "cuda",
                 hf_home: str = HF_HOME, max_new_tokens: int = 80):
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.max_new_tokens = max_new_tokens
        logger.info(f"Loading rewriter model {model_name} on {device}")
        kw = dict(cache_dir=hf_home, trust_remote_code=True)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True, **kw)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, local_files_only=True, dtype="auto", **kw)
        except Exception:
            logger.warning("rewriter 本地缓存未命中，尝试在线下载…")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=False, **kw)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, local_files_only=False, dtype="auto", **kw)
        self.model.to(device).eval()
        logger.info("rewriter ready")

    @torch.no_grad()
    def rewrite(self, truncated_text: str) -> Tuple[str, float]:
        """把被打断的 assistant 文本改写为自然收尾版。返回 (重写文本, 耗时 ms)。"""
        t0 = time.perf_counter()
        messages = [
            {"role": "system", "content": REWRITE_SYSTEM},
            {"role": "user", "content": truncated_text},
        ]
        # Qwen3 关闭 thinking 模式（重写是轻量任务，不需要推理链）
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id)
        text = self.tokenizer.decode(out[0][inputs.input_ids.shape[1]:],
                                     skip_special_tokens=True).strip()
        ms = (time.perf_counter() - t0) * 1000
        # 兜底：重写失败/为空 → 退化为标记法（原文+省略号），保证历史永不为空
        if not text:
            text = truncated_text.rstrip() + " …"
        logger.debug(f"[rewriter] {ms:.0f}ms: {truncated_text[:30]!r} → {text[:30]!r}")
        return text, ms
