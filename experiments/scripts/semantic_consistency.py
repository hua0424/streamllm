#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R5 §6.2：下游语义一致性双轨评估（意见5 质量部分）—— 离线。

输入：E4 结果（`r4_commit/exp1_results_*.json`）逐样本 full_response：
System A（non-streaming）回复 vs System B（streaming）回复，用户输入取 streaming 的
transcribed_text（LLM 实际收到的输入）。

轨道 A：BAAI/bge-m3 嵌入余弦相似度（本机可跑，~2.2GB 模型，HF 下载）。
轨道 B：LLM-as-a-Judge（DeepSeek，OpenAI 兼容接口）：
- 从环境变量读 DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL / DEEPSEEK_MODEL（key 不入库）；
- 该网关拦截 python 默认 UA（403），请求带 curl UA（2026-08-21 实测）；
- 推理型模型，max_tokens 须给足（默认 1024，reasoning 先烧预算）；
- A/B 顺序逐样本确定性随机交换（sum(ord(c)) 种子），结果记录 order 并映射回；
- 逐样本 JSON 落盘（可断点续跑）；解析失败/HTTP 错误重试 3 次后记 error 继续。

输出：`r5_semantic/semantic_consistency.csv`（逐样本 cosine + judge 分）+ `.summary.txt`。
用法：
  uv run python -m experiments.scripts.semantic_consistency --track A      # bge-m3
  DEEPSEEK_API_KEY=... uv run python -m experiments.scripts.semantic_consistency --track B
  uv run python -m experiments.scripts.semantic_consistency --self-test    # 无模型/无网络
"""

import argparse
import glob
import json
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_utils import get_logger, set_global_log_level

logger = get_logger(__name__)

JUDGE_PROMPT = """你是语音助手系统的评审专家。给定用户输入和两份系统回复（甲/乙，顺序已随机化），
评估两份回复在**下游语义上的等价性**：
5=语义完全等价（可互换）；4=细微差异不影响用户意图满足；3=部分信息差异但不冲突；
2=明显信息差异或一方缺失关键信息；1=语义冲突或答非所问。
用户输入：{user}
回复甲：{resp_a}
回复乙：{resp_b}
只输出一行 JSON：{{"score": <1-5整数>, "equivalent": <true/false>, "rationale": "<≤50字>"}}"""

# 轨道 B2：独立盲评（2026-08-21 增设）——成对等价会混入采样发散（两模式各自采样，
# 推荐内容本就不同）与 128 token 截断，B2 对每份回复独立评意图满足度，
# 比较 A/B 分布才可分离"管线退化"与"采样噪声"。
JUDGE_PROMPT_SOLO = """你是语音助手系统的评审专家。给定用户输入和一份系统回复，
评估该回复对**用户意图的满足度**：
5=完整准确地回应了用户全部需求；4=基本满足，细微遗漏；3=部分满足，有信息缺失；
2=仅回应了少量需求或含明显错误；1=未回应用户需求或答非所问。
注意：回复长度受系统固定上限约束，在已有内容范围内评估。
用户输入：{user}
回复：{resp}
只输出一行 JSON：{{"score": <1-5整数>, "rationale": "<≤50字>"}}"""


def lang_of(sample_id: str) -> str:
    return {"crosswoz": "zh", "aishell1": "zh", "multiwoz": "en",
            "librispeech": "en"}.get(sample_id.split("_")[0], "zh")


def load_e4_pairs(e4_results_glob: str) -> list:
    files = sorted(glob.glob(e4_results_glob))
    if not files:
        raise SystemExit(f"E4 结果未找到: {e4_results_glob}")
    path = files[-1]
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    by_sample = {}
    for r in data["results"]:
        if r.get("error"):
            continue
        by_sample.setdefault(r["sample_id"], {})[r["mode"]] = r
    pairs = []
    for sid, modes in sorted(by_sample.items()):
        a = (modes.get("non-streaming") or {}).get("full_response", "").strip()
        b = (modes.get("streaming") or {}).get("full_response", "").strip()
        user = (modes.get("streaming") or {}).get("transcribed_text", "").strip()
        if not a or not b:
            raise SystemExit(f"样本 {sid} 缺少 full_response（{path}）")
        pairs.append({"sample_id": sid, "language": lang_of(sid),
                      "user": user, "resp_a": a, "resp_b": b})
    logger.info(f"输入: {path}（{len(pairs)} 对回复）")
    return pairs


# ---------------------------------------------------------------- 轨道 A：bge-m3

def track_a(pairs: list, model_name: str = "BAAI/bge-m3") -> dict:
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    sims = {}
    with torch.no_grad():
        for p in pairs:
            enc = tok([p["resp_a"], p["resp_b"]], padding=True, truncation=True,
                      max_length=512, return_tensors="pt").to(device)
            hidden = model(**enc).last_hidden_state[:, 0]  # bge-m3 dense: CLS
            hidden = torch.nn.functional.normalize(hidden, p=2, dim=-1)
            sims[p["sample_id"]] = float((hidden[0] * hidden[1]).sum())
    return sims


# ---------------------------------------------------------------- 轨道 B：LLM judge

def judge_call(base_url: str, api_key: str, model: str, user: str, ra: str, rb: str = None,
               max_tokens: int = 1024, timeout: int = 180) -> dict:
    prompt = (JUDGE_PROMPT.format(user=user, resp_a=ra, resp_b=rb) if rb is not None
              else JUDGE_PROMPT_SOLO.format(user=user, resp=ra))
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions", data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
                 "User-Agent": "curl/8.0"})  # 该网关拦截 python 默认 UA（403）
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode("utf-8"))
    choice = d["choices"][0]
    if choice.get("finish_reason") == "length":
        raise RuntimeError("token 预算被 reasoning 耗尽（finish_reason=length）")
    content = (choice["message"].get("content") or "").strip()
    return parse_judge_content(content)


def parse_judge_content(content: str) -> dict:
    """从回复中解析 JSON 判定（容忍代码围栏/前后噪声）。"""
    import re
    m = re.search(r"\{[^{}]*\"score\"[^{}]*\}", content, re.S)
    if not m:
        raise ValueError(f"判定回复中未找到 JSON: {content[:120]!r}")
    obj = json.loads(m.group(0))
    score = int(obj["score"])
    if not 1 <= score <= 5:
        raise ValueError(f"score 越界: {score}")
    return {"score": score, "equivalent": bool(obj.get("equivalent")),
            "rationale": str(obj.get("rationale", ""))[:200]}


def track_b(pairs: list, out_dir: Path, base_url: str, api_key: str, model: str,
            sleep_s: float = 0.5, max_retries: int = 3, max_tokens: int = 1024) -> dict:
    judge_dir = out_dir / "judge"
    judge_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for i, p in enumerate(pairs):
        out_file = judge_dir / f"{p['sample_id']}.json"
        if out_file.exists():  # 断点续跑
            cached = json.loads(out_file.read_text(encoding="utf-8"))
            if not cached.get("error"):
                results[p["sample_id"]] = cached
                continue  # error 记录不缓存生效，重试
        # 确定性随机顺序（隐藏 A/B 来源，防顺序偏置）
        swap = sum(ord(c) for c in p["sample_id"]) % 2 == 1
        ra, rb = (p["resp_b"], p["resp_a"]) if swap else (p["resp_a"], p["resp_b"])
        rec = None
        last_err = ""
        for attempt in range(max_retries):
            try:
                rec = judge_call(base_url, api_key, model, p["user"], ra, rb,
                                 max_tokens=max_tokens)
                break
            except Exception as e:
                last_err = str(e)
                logger.warning(f"  {p['sample_id']} 第 {attempt + 1} 次调用失败: {e}")
                time.sleep(2 * (attempt + 1))
        if rec is None:
            rec = {"score": None, "equivalent": None, "rationale": "", "error": last_err}
        rec.update({"sample_id": p["sample_id"], "order": "BA" if swap else "AB"})
        out_file.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        results[p["sample_id"]] = rec
        logger.info(f"[{i + 1}/{len(pairs)}] {p['sample_id']}: score={rec['score']} "
                    f"order={rec['order']}")
        time.sleep(sleep_s)
    return results


def track_b2(pairs: list, out_dir: Path, base_url: str, api_key: str, model: str,
             sleep_s: float = 0.5, max_retries: int = 3, max_tokens: int = 1024) -> dict:
    """轨道 B2：每份回复独立盲评意图满足度（无甲/乙标签，单向调用）。"""
    judge_dir = out_dir / "judge_solo"
    judge_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for i, p in enumerate(pairs):
        rec = {"sample_id": p["sample_id"]}
        for side in ("a", "b"):
            out_file = judge_dir / f"{p['sample_id']}_{side}.json"
            if out_file.exists():
                cached = json.loads(out_file.read_text(encoding="utf-8"))
                if not cached.get("error"):
                    rec[f"score_{side}"] = cached.get("score")
                    continue
            resp = p["resp_a"] if side == "a" else p["resp_b"]
            one = None
            last_err = ""
            for attempt in range(max_retries):
                try:
                    one = judge_call(base_url, api_key, model, p["user"], resp,
                                     max_tokens=max_tokens)
                    break
                except Exception as e:
                    last_err = str(e)
                    time.sleep(2 * (attempt + 1))
            if one is None:
                one = {"score": None, "rationale": "", "error": last_err}
            out_file.write_text(json.dumps(one, ensure_ascii=False, indent=2),
                                encoding="utf-8")
            rec[f"score_{side}"] = one.get("score")
            time.sleep(sleep_s)
        results[p["sample_id"]] = rec
        logger.info(f"[{i + 1}/{len(pairs)}] {p['sample_id']}: "
                    f"A={rec.get('score_a')} B={rec.get('score_b')}")
    return results


# ---------------------------------------------------------------- 汇总

def write_outputs(pairs: list, sims: dict, judges: dict, out_csv: Path,
                  solo: dict = None) -> str:
    import csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    solo = solo or {}
    rows = []
    for p in pairs:
        sid = p["sample_id"]
        j = judges.get(sid) or {}
        s2 = solo.get(sid) or {}
        rows.append({"sample_id": sid, "language": p["language"],
                     "cosine": f"{sims[sid]:.4f}" if sid in sims else "",
                     "judge_score": j.get("score") if j.get("score") is not None else "",
                     "judge_equivalent": j.get("equivalent") if j else "",
                     "judge_order": j.get("order", ""),
                     "judge_error": j.get("error", "") if j else "",
                     "solo_score_A": s2.get("score_a") if s2.get("score_a") is not None else "",
                     "solo_score_B": s2.get("score_b") if s2.get("score_b") is not None else ""})
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [f"pairs={len(pairs)}"]
    if sims:
        v = np.array(list(sims.values()))
        lines.append(f"轨道A cosine: n={len(v)} mean={v.mean():.4f} std={v.std():.4f} "
                     f"min={v.min():.4f} p10={np.percentile(v, 10):.4f}")
    scored = [j["score"] for j in judges.values() if j.get("score") is not None]
    if scored:
        v = np.array(scored)
        lines.append(f"轨道B judge: n={len(v)} mean={v.mean():.2f}/5 "
                     f">=4分占比={np.mean(v >= 4) * 100:.1f}% 最低={v.min()}")
        n_err = sum(1 for j in judges.values() if j.get("error"))
        if n_err:
            lines.append(f"judge error: {n_err} 条（逐样本 JSON 可查）")
    sa = [s["score_a"] for s in solo.values() if s.get("score_a") is not None]
    sb = [s["score_b"] for s in solo.values() if s.get("score_b") is not None]
    if sa and sb:
        va, vb = np.array(sa), np.array(sb)
        lines.append(f"轨道B2 独立意图满足: A mean={va.mean():.2f}/5 (n={len(va)}), "
                     f"B mean={vb.mean():.2f}/5 (n={len(vb)}), "
                     f"A-B 差={float((va - vb).mean()):+.2f}, "
                     f"B>=4分占比={np.mean(vb >= 4) * 100:.1f}% vs A {np.mean(va >= 4) * 100:.1f}%")
    summary = "\n".join(lines)
    out_csv.with_suffix(".summary.txt").write_text(summary + "\n", encoding="utf-8")
    return summary


def self_test() -> int:
    failures = 0

    def check(name, cond, detail=""):
        nonlocal failures
        print(f"[{'PASS' if cond else 'FAIL'}] {name}{(': ' + detail) if detail else ''}")
        if not cond:
            failures += 1

    # 判定解析：干净 JSON / 代码围栏 / 噪声 / 越界
    check("干净 JSON", parse_judge_content('{"score": 5, "equivalent": true, "rationale": "x"}')["score"] == 5)
    check("围栏 JSON", parse_judge_content('```json\n{"score": 4, "equivalent": true}\n```')["score"] == 4)
    check("噪声 JSON", parse_judge_content('分析过程……最终：{"score": 3, "equivalent": false} 以上')["score"] == 3)
    try:
        parse_judge_content("没有JSON")
        check("无 JSON 报错", False)
    except ValueError:
        check("无 JSON 报错", True)
    try:
        parse_judge_content('{"score": 9}')
        check("越界报错", False)
    except ValueError:
        check("越界报错", True)

    # 汇总链路（假数据）
    pairs = [{"sample_id": "crosswoz_a", "language": "zh", "user": "u",
              "resp_a": "x", "resp_b": "y"}]
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "semantic_consistency.csv"
        s = write_outputs(pairs, {"crosswoz_a": 0.91},
                          {"crosswoz_a": {"score": 4, "equivalent": True,
                                          "order": "AB", "error": ""}}, out)
        check("汇总双轨", "mean=0.9100" in s and "mean=4.00/5" in s and ">=4分占比=100.0%" in s, s)

    print(f"\nself-test {'全部通过' if failures == 0 else f'{failures} 项失败'}")
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description="R5 语义一致性双轨评估（离线）")
    parser.add_argument("--e4-results", type=str,
                        default=str(PROJECT_ROOT / "experiments/results/revision/r4_commit/exp1_results_*.json"))
    parser.add_argument("--track", type=str, default="both",
                        choices=["A", "B", "B2", "both"],
                        help="A=bge-m3 嵌入；B=成对等价 judge；B2=独立意图满足盲评；both=A+B+B2")
    parser.add_argument("--judge-max-tokens", type=int, default=2048,
                        help="judge 输出预算（推理型模型，reasoning 先烧预算，需给足）")
    parser.add_argument("--embedding-model", type=str, default="BAAI/bge-m3")
    parser.add_argument("--output", type=str,
                        default="experiments/results/revision/r5_semantic/semantic_consistency.csv")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    set_global_log_level(args.log_level)

    if args.self_test:
        sys.exit(self_test())

    pairs = load_e4_pairs(args.e4_results)
    out_csv = PROJECT_ROOT / args.output

    sims = {}
    if args.track in ("A", "both"):
        logger.info(f"轨道 A：{args.embedding_model} 嵌入相似度")
        sims = track_a(pairs, args.embedding_model)

    judges = {}
    solo = {}
    if args.track in ("B", "B2", "both"):
        import os
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            raise SystemExit("缺少 DEEPSEEK_API_KEY 环境变量（key 不入库，走环境变量）")
        base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.commandcode.ai/provider/v1")
        model = os.environ.get("DEEPSEEK_MODEL", "deepseek/deepseek-v4-flash")
        if args.track in ("B", "both"):
            logger.info(f"轨道 B：成对等价 judge（{model} @ {base_url}）")
            judges = track_b(pairs, out_csv.parent, base_url, api_key, model,
                             max_tokens=args.judge_max_tokens)
        if args.track in ("B2", "both"):
            logger.info("轨道 B2：独立意图满足盲评")
            solo = track_b2(pairs, out_csv.parent, base_url, api_key, model,
                            max_tokens=args.judge_max_tokens)

    summary = write_outputs(pairs, sims, judges, out_csv, solo)
    print(f"\n{summary}")
    logger.info(f"已保存: {out_csv}")


if __name__ == "__main__":
    main()
