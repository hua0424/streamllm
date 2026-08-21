#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R6 §7.3：TTFA 预算表装配（Table VIII）—— 纯离线。

组成（逐样本对齐相加；2026-08-21 审查 P0 两轮裁决，最终=方案 (a)）：
  TTFA = T_endpoint（E5：speech_end → 最后语音段入队，mean 53.1ms）
       + T_post_endpoint（**E4 streaming TTFT**（同 50 样本，mean 1422.9ms）——
         "端点触发 flush"尾延迟的直接实测。说明：E5 的 2s 追加静音窗并非纯空转，
         窗内 ~410ms 是端点时积压队列的真实排空（E4 TTFT − E5 post-flush 逐样本
         50/50 为正，mean 410.5ms）；取 final 段入队→首 token（1012.5ms）会把这部分
         真实工作一并剔除、低于直接实测，故用 E4 口径）
       + T_decode_to_first_sentence（2026-08-21 补测：首 token → 首句末 token，mean 389.0ms）
       + T_TTS_first_chunk（E6：TTS 首包）
System B 四项全为实测；System A 的 decode/TTFC 项为估计值（source 列标注）：
  - A 的 pipeline 项 = E5 non-streaming 实测（ttft = audio_end → 首 token）+ 同样本端点等待；
  - A 的 decode 项 = B 的同语种均值代理（同模型同解码速率假设）；
  - A 的 TTFC 项 = 0.09s/字符 × A 回复均长（E6 实测 TTFC-长度关系；B 为实测值）。

输入：r6_ttfa/endpoint/exp1_results_*.json、decode_to_first_sentence.csv、
      tts_first_chunk.csv、r4_commit/exp1_results_*.json（A 回复长度）。
输出：r6_ttfa/ttfa_budget.csv（system × language 的各分项 mean/std 与 TTFA 合计）。
用法：
  uv run python -m experiments.scripts.assemble_ttfa_budget
  uv run python -m experiments.scripts.assemble_ttfa_budget --self-test
"""

import argparse
import csv
import glob
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_utils import get_logger, set_global_log_level

logger = get_logger(__name__)

TTFC_PER_CHAR_S = 0.09  # E6 实测 TTFC-长度关系（~0.09s/字符）
COMPONENTS = ["t_endpoint_ms", "t_post_endpoint_ms", "t_decode_first_sentence_ms",
              "t_tts_first_chunk_ms", "ttfa_total_ms"]


def lang_of(sample_id: str) -> str:
    return {"crosswoz": "zh", "aishell1": "zh", "multiwoz": "en",
            "librispeech": "en"}.get(sample_id.split("_")[0], "zh")


def load_inputs(endpoint_glob, decode_csv, tts_csv, e4_glob):
    ep_files = sorted(glob.glob(endpoint_glob))
    if not ep_files:
        raise SystemExit(f"E5 结果未找到: {endpoint_glob}")
    ep = json.loads(Path(ep_files[-1]).read_text(encoding="utf-8"))
    pipeline = {}  # sid -> {mode: {...}}
    for r in ep["results"]:
        if r.get("error"):
            continue
        sid = r["sample_id"]
        if r["mode"] == "streaming":
            endpoint_ms = (r["final_speech_segment_commit_time"] - r["speech_end_time"]) * 1000
            # 方案2：post 从 final is_final 段入队起算，剔除 2s 追加静音的装置等待
            post_ms = (r["first_token_time"] - r["final_is_final_segment_enqueue_time"]) * 1000
            pipeline.setdefault(sid, {})["streaming"] = {
                "endpoint": endpoint_ms, "post": post_ms,
                "total": endpoint_ms + post_ms}
        else:
            # non-streaming 的端点时间戳为哨兵 0（不跟踪）；pipeline 项直接取 ttft 字段
            pipeline.setdefault(sid, {})["non-streaming"] = {"post": r["ttft"]}

    decode = {r["sample_id"]: float(r["decode_to_first_sentence_ms"])
              for r in csv.DictReader(open(decode_csv, encoding="utf-8")) if not r["error"]}
    tts = {r["sample_id"]: float(r["ttfc_ms"])
           for r in csv.DictReader(open(tts_csv, encoding="utf-8")) if not r["error"]}

    e4_files = sorted(glob.glob(e4_glob))
    e4 = json.loads(Path(e4_files[-1]).read_text(encoding="utf-8"))
    a_chars = {}  # lang -> [chars]
    b_ttft = {}   # sid -> E4 streaming ttft（方案 (a)：B 行 post 分项的直接实测）
    for r in e4["results"]:
        if r.get("error"):
            continue
        if r["mode"] == "non-streaming":
            a_chars.setdefault(lang_of(r["sample_id"]), []).append(
                len((r.get("full_response") or "").strip()))
        elif r["mode"] == "streaming":
            b_ttft[r["sample_id"]] = float(r["ttft"])
    return pipeline, decode, tts, a_chars, b_ttft


def assemble(pipeline: dict, decode: dict, tts: dict, a_chars: dict, b_ttft: dict) -> list:
    rows = []
    sids_b = sorted(s for s, m in pipeline.items()
                    if "streaming" in m and s in decode and s in tts and s in b_ttft)
    if not sids_b:
        raise SystemExit("无四源齐全的 System B 样本（endpoint/decode/tts/E4 ttft）")
    a_char_mean = {lg: float(np.mean(v)) for lg, v in a_chars.items()}

    for system in ("streaming", "non-streaming"):
        for lang in ("zh", "en", "ALL"):
            sids = [s for s in sids_b if lang == "ALL" or lang_of(s) == lang]
            if not sids:
                continue
            comp = {c: [] for c in COMPONENTS}
            for sid in sids:
                m = pipeline[sid]
                if system == "streaming":
                    endpoint = m["streaming"]["endpoint"]
                    post = b_ttft[sid]  # 方案 (a)：E4 streaming TTFT（端点触发 flush 的直接实测）
                    dec = decode[sid]
                    ttfc = tts[sid]
                else:
                    # A：端点等待同样本同值（VAD 属性）；post = A 实测 ttft（audio_end→首 token）
                    if "non-streaming" not in m:
                        continue
                    endpoint = m["streaming"]["endpoint"]
                    post = m["non-streaming"]["post"]
                    dec = float(np.mean([decode[s] for s in sids_b
                                         if lang_of(s) == lang_of(sid)]))  # 代理（估计）
                    ttfc = TTFC_PER_CHAR_S * 1000 * a_char_mean.get(lang_of(sid), 200)  # 估计
                comp["t_endpoint_ms"].append(endpoint)
                comp["t_post_endpoint_ms"].append(post)
                comp["t_decode_first_sentence_ms"].append(dec)
                comp["t_tts_first_chunk_ms"].append(ttfc)
                comp["ttfa_total_ms"].append(endpoint + post + dec + ttfc)
            if not comp["t_endpoint_ms"]:
                continue
            row = {"system": system, "language": lang, "n": len(comp["t_endpoint_ms"]),
                   "source": "全实测" if system == "streaming"
                             else "pipeline 实测；decode/TTFC 估计（见脚本 docstring）"}
            for c in COMPONENTS:
                v = np.array(comp[c])
                row[f"{c}_mean"] = f"{v.mean():.1f}"
                row[f"{c}_std"] = f"{v.std():.1f}"
            rows.append(row)
    return rows


def self_test() -> int:
    failures = 0

    def check(name, cond, detail=""):
        nonlocal failures
        print(f"[{'PASS' if cond else 'FAIL'}] {name}{(': ' + detail) if detail else ''}")
        if not cond:
            failures += 1

    with tempfile.TemporaryDirectory() as td:
        tp = Path(td)
        # E5 风格结果：2 样本 × 2 模式（streaming：speech_end=100，commit=100.05，
        # final_enqueue=102.0（方案2 post=1000ms），first_token=103；
        # non-streaming 端点字段为哨兵 0，取 ttft=4950ms）
        results = []
        for i, sid in enumerate(["crosswoz_a", "multiwoz_b"]):
            results.append({"sample_id": sid, "mode": "streaming", "error": "",
                            "speech_end_time": 100.0,
                            "final_speech_segment_commit_time": 100.05,
                            "final_is_final_segment_enqueue_time": 102.0,
                            "first_token_time": 103.0})
            results.append({"sample_id": sid, "mode": "non-streaming", "error": "",
                            "speech_end_time": 0.0,
                            "final_speech_segment_commit_time": 0.0,
                            "first_token_time": 105.0, "ttft": 4950.0})
        ep = tp / "exp1_results_e5.json"
        ep.write_text(json.dumps({"results": results}), encoding="utf-8")
        dec = tp / "decode.csv"
        dec.write_text("sample_id,decode_to_first_sentence_ms,error\n"
                       "crosswoz_a,400,\nmultiwoz_b,600,\n", encoding="utf-8")
        tts = tp / "tts.csv"
        tts.write_text("sample_id,ttfc_ms,error\ncrosswoz_a,10000,\nmultiwoz_b,8000,\n",
                       encoding="utf-8")
        e4 = tp / "exp1_results_e4.json"
        e4.write_text(json.dumps({"results": [
            {"sample_id": "crosswoz_a", "mode": "streaming", "error": "", "ttft": 3000.0},
            {"sample_id": "multiwoz_b", "mode": "streaming", "error": "", "ttft": 3000.0},
            {"sample_id": "crosswoz_a", "mode": "non-streaming", "error": "",
             "full_response": "好" * 100},
            {"sample_id": "multiwoz_b", "mode": "non-streaming", "error": "",
             "full_response": "x" * 200}]}), encoding="utf-8")

        pipeline, decode, tts_d, a_chars, b_ttft = load_inputs(str(ep), str(dec), str(tts), str(e4))
        rows = assemble(pipeline, decode, tts_d, a_chars, b_ttft)
        b_all = [r for r in rows if r["system"] == "streaming" and r["language"] == "ALL"]
        check("B 行存在", len(b_all) == 1 and b_all[0]["n"] == 2)
        # B: endpoint=50ms, post=E4 ttft=3000ms；zh: dec400+tts10000；en: dec600+tts8000
        b = b_all[0]
        check("B 分量", b["t_endpoint_ms_mean"] == "50.0" and b["t_post_endpoint_ms_mean"] == "3000.0",
              f"{b['t_endpoint_ms_mean']}/{b['t_post_endpoint_ms_mean']}")
        check("B 合计", b["ttfa_total_ms_mean"] == "12550.0",
              b["ttfa_total_ms_mean"] + "（50+3000+500+9000=12550）")
        a = [r for r in rows if r["system"] == "non-streaming" and r["language"] == "zh"]
        # A zh: endpoint50 + post(5000-50=4950) + dec代理400 + ttfc 0.09*1000*100=9000 → 14400
        check("A 估计链", len(a) == 1 and a[0]["ttfa_total_ms_mean"] == "14400.0",
              a[0]["ttfa_total_ms_mean"] if a else "缺行")
        check("source 标注", a and "估计" in a[0]["source"])

    print(f"\nself-test {'全部通过' if failures == 0 else f'{failures} 项失败'}")
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description="R6 §7.3 TTFA 预算表装配（Table VIII）")
    parser.add_argument("--endpoint-results", type=str,
                        default=str(PROJECT_ROOT / "experiments/results/revision/r6_ttfa/endpoint/exp1_results_*.json"))
    parser.add_argument("--decode-csv", type=str,
                        default="experiments/results/revision/r6_ttfa/decode_to_first_sentence.csv")
    parser.add_argument("--tts-csv", type=str,
                        default="experiments/results/revision/r6_ttfa/tts_first_chunk.csv")
    parser.add_argument("--e4-results", type=str,
                        default=str(PROJECT_ROOT / "experiments/results/revision/r4_commit/exp1_results_*.json"))
    parser.add_argument("--output", type=str,
                        default="experiments/results/revision/r6_ttfa/ttfa_budget.csv")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    set_global_log_level(args.log_level)

    if args.self_test:
        sys.exit(self_test())

    pipeline, decode, tts, a_chars, b_ttft = load_inputs(args.endpoint_results, args.decode_csv,
                                                         args.tts_csv, args.e4_results)
    rows = assemble(pipeline, decode, tts, a_chars, b_ttft)
    out = PROJECT_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(f"{r['system']:>14} {r['language']:>3}: TTFA={r['ttfa_total_ms_mean']}ms "
              f"(endpoint {r['t_endpoint_ms_mean']} + post {r['t_post_endpoint_ms_mean']} "
              f"+ decode {r['t_decode_first_sentence_ms_mean']} + ttfc {r['t_tts_first_chunk_ms_mean']}) "
              f"[{r['source']}]")
    logger.info(f"已保存: {out}")


if __name__ == "__main__":
    main()
