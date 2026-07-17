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

    # 分格（loose 列；只取 unheard 非空的——空 unheard 双方必然一致为 No，无仲裁价值）
    cells = {}
    for r in records:
        if not (r.get("unheard_text") or "").strip():
            continue
        key = (r["condition"], bool(r.get("referenced_unheard")),
               bool(r.get("judge_referenced_unheard")))
        cells.setdefault(key, []).append(r)

    # 配额：分歧格各 12，双 True 格各 6，双 False 格各 4（不足取全部），凑 ~n
    quota = {(True, False): 12, (False, True): 12, (True, True): 6, (False, False): 4}
    picked = []
    for key, rs in sorted(cells.items(), key=lambda kv: str(kv[0])):
        q = quota.get((key[1], key[2]), 4)
        rng.shuffle(rs)
        picked += [(key, r) for r in rs[:q]]
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
    for i, (key, r) in enumerate(picked, 1):
        lines += [
            f"---",
            f"### #{i}  [{r['id']} f={r['fraction']} {r['condition']}]  "
            f"规则={'Y' if r.get('referenced_unheard') else 'N'} / "
            f"judge={'Y' if r.get('judge_referenced_unheard') else 'N'} / 人判=____",
            f"**UNHEARD（用户没听到的内容）**：{r['unheard_text'].strip()}",
            f"**下一轮回复**：{' / '.join(x.strip() for x in r.get('probe_replies', []))[:500]}",
            f"**规则命中词**：{', '.join(r.get('matched_cues', [])) or '（无）'}",
            "",
        ]
    out = Path(args.out)
    out.write_text("\n".join(lines), encoding="utf-8")
    dist = {}
    for key, _ in picked:
        dist[key] = dist.get(key, 0) + 1
    logger.info(f"样本分布 (condition, rule, judge)→n: { {str(k): v for k, v in sorted(dist.items(), key=str)} }")
    logger.info(f"已生成: {out}")


if __name__ == "__main__":
    main()
