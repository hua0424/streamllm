"""从权威分章 Markdown 确定性生成学位论文合并稿。"""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT / "paper2"
OUTPUT = PAPER_DIR / "thesis_draft.md"
SOURCES = [
    "abstract.md",
    "chapter1_introduction.md",
    "chapter2_related_work.md",
    "chapter3_formulation.md",
    "chapter4_method.md",
    "chapter5_implementation.md",
    "chapter6_experiments.md",
    "chapter7_discussion.md",
    "chapter8_conclusion.md",
    "references.md",
]
HEADER = """# 级联式语音对话中软件播放游标与 TTS 片段驱动的 KV 状态修正

> 全文合并草稿（自动生成，勿直接编辑；请修改分章 Markdown 后重新合并）。
> 本文件已于 2026-09-03 依据 crossed E1/E2 analysis v2、E3 weighting/dedup analysis v2、accepted C2 v3 及二审意见统一更新。
"""
SEPARATOR = "\n---\n\n"


def render() -> str:
    sections = [
        (PAPER_DIR / name).read_text(encoding="utf-8").strip()
        for name in SOURCES
    ]
    return HEADER.rstrip() + "\n\n" + SEPARATOR.join(sections) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="仅检查 thesis_draft.md 是否与权威分章一致",
    )
    args = parser.parse_args()
    content = render()
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text(encoding="utf-8") != content:
            raise SystemExit(f"{OUTPUT} 不是最新合并稿")
        print(f"[ok] {OUTPUT} 与 {len(SOURCES)} 个权威源文件一致")
        return
    OUTPUT.write_text(content, encoding="utf-8")
    print(f"[saved] {OUTPUT}（{len(SOURCES)} 个权威源文件）")


if __name__ == "__main__":
    main()
