#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R4 §5.1/5.3：提交分歧（commit-divergence）完整统计 —— 纯离线。

输入：
- `r4_commit/commit_log.jsonl`（E4 插桩：commit / correction 事件）；
- E4 结果 JSON（streaming 与 non-streaming 的 transcribed_text，作外部一致性参照）。

统计（计划 §5.1）：
1. correction 事件总数、涉及段比例、涉及样本比例；
2. 每事件字符级编辑距离（normalize_text 去标点后）与归一化比率（dist/max(len)）的
   mean/median/p90/max——correction 漂移幅度分布（注意：2026-08-21 抽查发现 correction
   并非全是同音字/标点级，存在实词级漂移，必须以实测分布为准）；
3. 外部一致性：逐样本 WER/CER（streaming 已提交拼接串 vs non-streaming 全量转写），
   口径复用 score_wer_offline.score_pair（中文 zh_to_word_seq / 英文大小写折叠）。

输出：`r4_commit/commit_divergence.json` + 控制台汇总。
用法：
  uv run python -m experiments.scripts.analyze_commit_divergence
  uv run python -m experiments.scripts.analyze_commit_divergence --self-test
"""

import argparse
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


def char_dist(a: str, b: str) -> int:
    from experiments.scripts.run_exp_quality import _levenshtein, normalize_text
    return _levenshtein(list(normalize_text(a)), list(normalize_text(b)))


def analyze(commit_log_path: Path, e4_results_path: Path) -> dict:
    commits = []
    corrections = []
    for line in open(commit_log_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        ev = json.loads(line)
        (commits if ev["type"] == "commit" else corrections).append(ev)

    # correction 漂移幅度
    for ev in corrections:
        d = char_dist(ev["old"], ev["new"])
        ev["_dist"] = d
        ev["_ratio"] = d / max(len(ev["old"]), len(ev["new"]), 1)

    dists = np.array([e["_dist"] for e in corrections]) if corrections else np.array([0])
    ratios = np.array([e["_ratio"] for e in corrections]) if corrections else np.array([0.0])
    n_committed_segments = sum(len(c.get("segment_ids", [])) for c in commits)
    corrected_segments = {(e["sample_id"], e["segment_id"]) for e in corrections}
    samples_with_corr = sorted({e["sample_id"] for e in corrections})

    # 外部一致性：streaming 拼接串 vs non-streaming 全量转写（同样本）
    from experiments.scripts.score_wer_offline import score_pair
    data = json.loads(Path(e4_results_path).read_text(encoding="utf-8"))
    by_sample = {}
    for r in data["results"]:
        if r.get("error"):
            continue
        by_sample.setdefault(r["sample_id"], {})[r["mode"]] = (r.get("transcribed_text") or "").strip()
    ext = []
    for sid, modes in by_sample.items():
        if "streaming" not in modes or "non-streaming" not in modes:
            continue
        lang = "zh" if sid.split("_")[0] in ("crosswoz", "aishell1") else "en"
        w, c = score_pair(modes["non-streaming"], modes["streaming"], lang)
        ext.append({"sample_id": sid, "wer": w, "cer": c})
    ext_w = np.array([e["wer"] for e in ext])
    ext_c = np.array([e["cer"] for e in ext])

    top_events = sorted(corrections, key=lambda e: -e["_ratio"])[:5]
    return {
        "inputs": {"commit_log": str(commit_log_path), "e4_results": str(e4_results_path)},
        "commit_events": len(commits),
        "correction_events": len(corrections),
        "committed_segments": n_committed_segments,
        "corrected_segments": len(corrected_segments),
        "corrected_segment_ratio": len(corrected_segments) / max(n_committed_segments, 1),
        "samples_with_correction": len(samples_with_corr),
        "rollback_events": 0,  # 构造保证 append-only，无回滚下发
        "correction_edit_distance": {
            "mean": float(dists.mean()), "median": float(np.median(dists)),
            "p90": float(np.percentile(dists, 90)), "max": int(dists.max()),
        },
        "correction_edit_ratio": {
            "mean": float(ratios.mean()), "median": float(np.median(ratios)),
            "p90": float(np.percentile(ratios, 90)), "max": float(ratios.max()),
        },
        "external_consistency": {
            "n_samples": len(ext),
            "wer_mean": float(ext_w.mean()), "wer_std": float(ext_w.std()),
            "cer_mean": float(ext_c.mean()), "cer_std": float(ext_c.std()),
            "wer_max": float(ext_w.max()), "cer_max": float(ext_c.max()),
        },
        "top_drift_examples": [
            {"sample_id": e["sample_id"], "segment_id": e["segment_id"],
             "edit_ratio": round(e["_ratio"], 3), "old": e["old"], "new": e["new"]}
            for e in top_events
        ],
    }


def self_test() -> int:
    failures = 0

    def check(name, cond, detail=""):
        nonlocal failures
        print(f"[{'PASS' if cond else 'FAIL'}] {name}{(': ' + detail) if detail else ''}")
        if not cond:
            failures += 1

    with tempfile.TemporaryDirectory() as td:
        tp = Path(td)
        log = tp / "commit_log.jsonl"
        events = [
            {"sample_id": "crosswoz_a", "type": "commit", "round": 0,
             "text": "你好世界", "segment_ids": ["s1", "s2"]},
            {"sample_id": "crosswoz_a", "type": "correction", "segment_id": "s2",
             "old": "世界", "new": "世届"},
            {"sample_id": "multiwoz_b", "type": "commit", "round": 0,
             "text": "hello", "segment_ids": ["s1"]},
        ]
        log.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in events),
                       encoding="utf-8")
        res = tp / "exp1_results_x.json"
        res.write_text(json.dumps({"results": [
            {"sample_id": "crosswoz_a", "mode": "streaming", "error": "",
             "transcribed_text": "你好世界"},
            {"sample_id": "crosswoz_a", "mode": "non-streaming", "error": "",
             "transcribed_text": "你好世界"},
            {"sample_id": "multiwoz_b", "mode": "streaming", "error": "",
             "transcribed_text": "hello world"},
            {"sample_id": "multiwoz_b", "mode": "non-streaming", "error": "",
             "transcribed_text": "hello word"},
        ]}), encoding="utf-8")
        out = analyze(log, res)
        check("事件计数", out["commit_events"] == 2 and out["correction_events"] == 1)
        check("段比例", out["committed_segments"] == 3 and out["corrected_segments"] == 1
              and abs(out["corrected_segment_ratio"] - 1 / 3) < 1e-9)
        check("编辑距离", out["correction_edit_distance"]["mean"] == 1.0)
        check("外部一致性样本数", out["external_consistency"]["n_samples"] == 2)
        check("外部一致性数值", out["external_consistency"]["wer_mean"] == 0.25,
              f"wer_mean={out['external_consistency']['wer_mean']}（zh 0 + en 1/2 → 0.25）")
        check("回滚恒零", out["rollback_events"] == 0)

    print(f"\nself-test {'全部通过' if failures == 0 else f'{failures} 项失败'}")
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description="R4 提交分歧完整统计（离线）")
    parser.add_argument("--commit-log", type=str,
                        default="experiments/results/revision/r4_commit/commit_log.jsonl")
    parser.add_argument("--e4-results", type=str,
                        default=str(PROJECT_ROOT / "experiments/results/revision/r4_commit/exp1_results_*.json"))
    parser.add_argument("--output", type=str,
                        default="experiments/results/revision/r4_commit/commit_divergence.json")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    set_global_log_level(args.log_level)

    if args.self_test:
        sys.exit(self_test())

    files = sorted(glob.glob(args.e4_results))
    if not files:
        raise SystemExit(f"E4 结果未找到: {args.e4_results}")
    out = analyze(PROJECT_ROOT / args.commit_log, Path(files[-1]))
    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    ext = out["external_consistency"]
    print(f"commit={out['commit_events']} correction={out['correction_events']} "
          f"涉及段 {out['corrected_segments']}/{out['committed_segments']} "
          f"({out['corrected_segment_ratio'] * 100:.1f}%) 涉及样本 {out['samples_with_correction']}")
    print(f"correction 编辑距离 mean={out['correction_edit_distance']['mean']:.1f} "
          f"p90={out['correction_edit_distance']['p90']:.1f} max={out['correction_edit_distance']['max']} | "
          f"归一化比率 mean={out['correction_edit_ratio']['mean']:.3f} "
          f"max={out['correction_edit_ratio']['max']:.3f}")
    print(f"外部一致性（streaming 拼接 vs System A）n={ext['n_samples']} "
          f"WER mean={ext['wer_mean']:.4f} max={ext['wer_max']:.4f} | "
          f"CER mean={ext['cer_mean']:.4f}")
    logger.info(f"已保存: {out_path}")


if __name__ == "__main__":
    main()
