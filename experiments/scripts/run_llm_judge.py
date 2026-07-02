#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM-as-judge：E3 未听引用交叉验证 + A2 连贯性评分（handoff §四.4）。

两种模式：
  e3：读 exp3_consistency.json（须含 unheard_text/strict_unheard_text/probe_replies，
      2026-07-02 后的 harness 产出即含），judge 判"回复是否引用了未听内容"，
      输出 judge 版 loose/strict 引用率 + 与规则检测器的 Cohen's κ（P3 要求）。
  a2：读 exp_a2_history.json，judge 给"历史+下一轮回复"的连贯性打 1-5 分，
      填 judge_coherence 输出到 *_judged.json（原文件不动）。

裁判模型要求（experiment_design.md §9.3）：与主 LLM **不同家族**的更强模型。
验证机可用 --judge-model Qwen/Qwen2.5-0.5B-Instruct 当替身验证脚本逻辑
（正式数值必须换强裁判）；另需人工小样本 ~50 条验证裁判可靠性（P3）。

运行：
    HF_TOKEN= uv run python -m experiments.scripts.run_llm_judge e3 \
        --results experiments/results/exp3_consistency.json --judge-model <强模型>
    HF_TOKEN= uv run python -m experiments.scripts.run_llm_judge a2 \
        --results experiments/results/exp_a2_history.json --judge-model <强模型>
"""

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import HF_HOME
from src.utils.logging_utils import get_logger, set_global_log_level

logger = get_logger(__name__)

REF_SYSTEM = (
    "You are a strict evaluator. Decide whether the REPLY makes use of, repeats, or refers to "
    "any specific information that appears ONLY in the UNHEARD text (content the user never "
    "heard). Generic topical overlap does not count; specific facts, numbers, or phrasings do. "
    "Answer with exactly one word: YES or NO."
)
COH_SYSTEM = (
    "You are a strict evaluator of dialogue coherence. Given the assistant's previous (possibly "
    "interrupted) utterance HISTORY and its NEXT reply to the user's follow-up, rate how "
    "coherent and natural the NEXT reply is as a continuation, on a scale of 1 (incoherent) "
    "to 5 (fully coherent). Answer with exactly one digit 1-5."
)


class Judge:
    def __init__(self, model_name: str, device: str = "cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        logger.info(f"Loading judge model {model_name} on {device}")
        kw = dict(cache_dir=HF_HOME, trust_remote_code=True)
        try:
            self.tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True, **kw)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True,
                                                              dtype="auto", **kw)
        except Exception:
            self.tok = AutoTokenizer.from_pretrained(model_name, local_files_only=False, **kw)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=False,
                                                              dtype="auto", **kw)
        self.model.to(device).eval()
        self._yes = [self.tok.encode(w, add_special_tokens=False)[0] for w in ("YES", " YES", "Yes")]
        self._no = [self.tok.encode(w, add_special_tokens=False)[0] for w in ("NO", " NO", "No")]

    @torch.no_grad()
    def _logits(self, system: str, user: str):
        prompt = self.tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True)
        inputs = self.tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
        return self.model(**inputs).logits[0, -1, :], inputs

    def yes_no(self, system: str, user: str) -> bool:
        logits, _ = self._logits(system, user)
        pos = torch.logsumexp(logits[self._yes], dim=0)
        neg = torch.logsumexp(logits[self._no], dim=0)
        return bool(pos > neg)

    @torch.no_grad()
    def rate_1_5(self, system: str, user: str):
        logits, _ = self._logits(system, user)
        digit_ids = [self.tok.encode(str(d), add_special_tokens=False)[0] for d in range(1, 6)]
        probs = torch.softmax(logits[digit_ids], dim=0)
        return int(torch.argmax(probs).item()) + 1


def cohen_kappa(a, b):
    """两组布尔判定的 Cohen's κ（手工实现，免额外依赖）。"""
    assert len(a) == len(b) and a
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa1, pb1 = sum(a) / n, sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return 1.0 if pe == 1.0 else (po - pe) / (1 - pe)


def judge_e3(records, judge: Judge):
    """对每条 record：judge 判 loose/strict 引用；返回聚合 + κ。"""
    out = {"loose": {"rule": [], "judge": []}, "strict": {"rule": [], "judge": []}}
    for r in records:
        replies = "\n".join(r.get("probe_replies", []))
        for col, text_key, rule_key in [("loose", "unheard_text", "referenced_unheard"),
                                        ("strict", "strict_unheard_text", "referenced_unheard_strict")]:
            unheard = (r.get(text_key) or "").strip()
            if not unheard:
                verdict = False           # 无未听内容 → 不可能引用（与规则版语义一致）
            else:
                verdict = judge.yes_no(REF_SYSTEM, f"UNHEARD:\n{unheard}\n\nREPLY:\n{replies}")
            r[f"judge_{rule_key}"] = verdict
            out[col]["rule"].append(bool(r.get(rule_key, False)))
            out[col]["judge"].append(verdict)
    summary = {}
    for col in ("loose", "strict"):
        rule, jd = out[col]["rule"], out[col]["judge"]
        summary[col] = {
            "n": len(jd),
            "rule_rate": round(sum(rule) / len(rule), 3),
            "judge_rate": round(sum(jd) / len(jd), 3),
            "cohen_kappa": round(cohen_kappa(rule, jd), 3),
        }
    return summary


def judge_a2(records, judge: Judge):
    for r in records:
        r["judge_coherence"] = judge.rate_1_5(
            COH_SYSTEM, f"HISTORY (assistant, possibly interrupted):\n{r['history_text']}\n\n"
                        f"NEXT reply:\n{r['next_reply']}")
    by_policy = {}
    for r in records:
        by_policy.setdefault(r["policy"], []).append(r["judge_coherence"])
    return {p: {"n": len(v), "mean_coherence": round(sum(v) / len(v), 2)}
            for p, v in sorted(by_policy.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["e3", "a2"])
    ap.add_argument("--results", type=str, required=True)
    ap.add_argument("--judge-model", type=str, required=True,
                    help="裁判模型（正式：与主 LLM 不同家族的强模型；替身仅验证脚本）")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--log-level", type=str, default="INFO")
    args = ap.parse_args()

    set_global_log_level(args.log_level)
    src = Path(args.results)
    data = json.loads(src.read_text(encoding="utf-8"))
    records = data["records"]
    judge = Judge(args.judge_model, args.device)

    if args.mode == "e3":
        missing = sum(1 for r in records if "probe_replies" not in r)
        if missing:
            logger.warning(f"{missing}/{len(records)} 条记录缺文本字段（旧 schema）——请重跑 E3 后再 judge")
        summary = judge_e3(records, judge)
        for col, s in summary.items():
            logger.info(f"[{col}] n={s['n']} rule={s['rule_rate']:.1%} judge={s['judge_rate']:.1%} "
                        f"κ={s['cohen_kappa']}")
    else:
        summary = judge_a2(records, judge)
        for p, s in summary.items():
            logger.info(f"[{p}] n={s['n']} 平均连贯性={s['mean_coherence']}")

    out = src.with_name(src.stem + "_judged.json")
    out.write_text(json.dumps({"judge_model": args.judge_model, "judge_summary": summary,
                               "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"已保存: {out}")
    logger.info("ALL DONE ✓")


if __name__ == "__main__":
    main()
