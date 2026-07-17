#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 E3 人工校验样本表（P3：~50 条，仲裁规则检测器与 LLM-judge 的分歧）。

分层抽样（loose 列）：按 (condition, rule判定, judge判定) 分格，
分歧格（rule=True/judge=False 为主要争议）多抽，一致格少量作对照。
产出 markdown 标注表：人工在"人判"列填 Y/N（是否引用了未听内容——
标准同 REF_SYSTEM：特定事实/数字/措辞才算，泛泛话题重叠不算）。

运行（本机）：
    uv run python -m experiments.scripts.make_human_validation_sample
"""

import argparse
import json
import random
from pathlib import Path

from src.config import RESULTS_DIR
from src.utils.logging_utils import get_logger, set_global_log_level

logger = get_logger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=str,
                    default=str(Path(RESULTS_DIR) / "exp3_consistency_judged.json"))
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str,
                    default=str(Path(RESULTS_DIR) / "e3_human_validation_sample.md"))
    ap.add_argument("--log-level", type=str, default="INFO")
    args = ap.parse_args()

    set_global_log_level(args.log_level)
    records = json.loads(Path(args.results).read_text(encoding="utf-8"))["records"]
    rng = random.Random(args.seed)

    # 双列分格（loose + strict；只取对应 unheard 非空的——空则双方必然 No，无仲裁价值）。
    # 注意：loose 列 playback 恒为空（构造性），playback 的仲裁全在 strict 列（片段尾部量化误差）。
    cells = {}
    for r in records:
        for col, tkey, rkey, jkey in (
                ("loose", "unheard_text", "referenced_unheard", "judge_referenced_unheard"),
                ("strict", "strict_unheard_text", "referenced_unheard_strict",
                 "judge_referenced_unheard_strict")):
            if not (r.get(tkey) or "").strip():
                continue
            key = (col, r["condition"], bool(r.get(rkey)), bool(r.get(jkey)))
            cells.setdefault(key, []).append((col, r))

    # 配额：分歧格各 12，双 True 格各 6，双 False 格各 4（不足取全部），凑 ~n
    quota = {(True, False): 6, (False, True): 6, (True, True): 4, (False, False): 2}
    picked = []
    for key, rs in sorted(cells.items(), key=lambda kv: str(kv[0])):
        q = quota.get((key[2], key[3]), 2)
        rng.shuffle(rs)
        picked += rs[:q]
    rng.shuffle(picked)
    picked = picked[:args.n]

    lines = [
        "# E3 人工校验样本（P3 仲裁：规则检测器 vs LLM-judge）",
        "",
        f"> 判定标准（与 judge 同）：下一轮回复是否**使用/复述/指涉**了 UNHEARD 文本中的",
        f"> **特定信息**（事实、数字、措辞）——泛泛的话题重叠**不算**。在【人判】填 Y 或 N。",
        f"> 共 {len(picked)} 条（seed={args.seed}），分层覆盖各判定组合。",
        "",
    ]
    for i, (col, r) in enumerate(picked, 1):
        tkey = "unheard_text" if col == "loose" else "strict_unheard_text"
        rkey = "referenced_unheard" if col == "loose" else "referenced_unheard_strict"
        jkey = "judge_" + rkey
        ckey = "matched_cues" if col == "loose" else "matched_cues_strict"
        lines += [
            f"---",
            f"### #{i}  [{r['id']} f={r['fraction']} {r['condition']} 列={col}]  "
            f"规则={'Y' if r.get(rkey) else 'N'} / "
            f"judge={'Y' if r.get(jkey) else 'N'} / 人判=____",
            f"**UNHEARD（用户没听到的内容）**：{r[tkey].strip()}",
            f"**下一轮回复**：{' / '.join(x.strip() for x in r.get('probe_replies', []))[:500]}",
            f"**规则命中词**：{', '.join(r.get(ckey, [])) or '（无）'}",
            "",
        ]
    out = Path(args.out)
    out.write_text("\n".join(lines), encoding="utf-8")
    dist = {}
    for col, r in picked:
        dist[col] = dist.get(col, 0) + 1
    logger.info(f"样本分布 (condition, rule, judge)→n: { {str(k): v for k, v in sorted(dist.items(), key=str)} }")
    logger.info(f"已生成: {out}")


if __name__ == "__main__":
    main()
