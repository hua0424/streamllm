#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MultiWOZ → 二期实验输入数据派生脚本（D-007 P4：英文为主）。

从 MultiWOZ 原始数据抽取 user 轮，产出两种实验输入（格式见 docs/handoff.md §三）：
  1. turns 格式（E3/A2）：{"id", "turns": [user轮1(被打断轮), probe轮2, probe轮3...]}
  2. segments 格式（E2/E1）：{"id", "segments": [" 子句1", " 子句2", ...]}
     —— 在子句边界切分单条 user 话语，模拟 ASR final 片段流；首段常呈"句法近似完整"，
        天然构成假停顿（E2 推测浪费的来源）。

兼容两种 MultiWOZ 格式：
  - 2.0/2.1 data.json：{dialogue_id: {"log": [{"text":..., "metadata":{}}, ...]}}
    （偶数下标=user、奇数=system；metadata 空=user 轮，双保险判断）
  - 2.2 风格：[{"dialogue_id":..., "turns":[{"speaker":"USER","utterance":...}]}]

运行（项目根目录，实验机）：
    uv run python -m experiments.scripts.prepare_multiwoz_data \
        --input experiments/datasets/raw_data/MultiWOZ/data.json \
        --out-turns experiments/datasets/processed/p2_turns.json \
        --out-segments experiments/datasets/processed/p2_segments.json \
        --max-dialogues 100 --seed 42
本机已用迷你合成样本验证脚本逻辑；实验机对全量数据执行即可。
"""

import argparse
import json
import random
import re
from pathlib import Path

from src.utils.logging_utils import get_logger, set_global_log_level

logger = get_logger(__name__)

# 子句切分点：逗号 / 并列连词 / 分号（保守，保证每段仍是自然语流）
_SPLIT_PAT = re.compile(r"(,\s+|;\s+|\s+and\s+|\s+but\s+|\s+because\s+)", re.IGNORECASE)
MIN_SEG_WORDS = 3       # 每段最少词数（太碎的合并进前段）
MIN_UTT_WORDS = 8       # 参与 segments 派生的话语最少词数


def load_multiwoz(path: Path):
    """返回 [(dialogue_id, [user_utterances...])]，兼容 2.0/2.1 与 2.2 风格。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    if isinstance(data, dict):          # 2.0/2.1: {id: {"log": [...]}}
        for did, d in data.items():
            log = d.get("log", [])
            users = [t["text"].strip() for i, t in enumerate(log)
                     if (i % 2 == 0) or not t.get("metadata")]
            # 双保险去重相邻（偶数位判定与 metadata 判定重叠时）
            users = [u for i, u in enumerate(users) if i == 0 or u != users[i - 1]]
            if users:
                out.append((did, users))
    elif isinstance(data, list):        # 2.2 风格
        for d in data:
            did = d.get("dialogue_id", d.get("id", f"dlg{len(out)}"))
            users = [t.get("utterance", "").strip() for t in d.get("turns", [])
                     if str(t.get("speaker", "")).upper() == "USER"]
            users = [u for u in users if u]
            if users:
                out.append((did, users))
    else:
        raise ValueError("无法识别的 MultiWOZ JSON 结构")
    logger.info(f"读入 {len(out)} 条对话（{path}）")
    return out


def split_segments(utt: str):
    """
    把一条话语切成 ASR final 片段流；返回 None 表示不适合（切不出 ≥2 段）。
    分隔符（", "/" and "等）粘到**前段尾部**——自然携带空格，
    保证严格无损：''.join(segs) == utt（编排层是直接拼接）。
    """
    parts = _SPLIT_PAT.split(utt)
    raw = []
    for p in parts:
        if not p:
            continue
        if _SPLIT_PAT.fullmatch(p) and raw:
            raw[-1] += p                    # 分隔符归前段尾
        else:
            raw.append(p)
    # 过短段向前合并
    segs = []
    for s in raw:
        if segs and len(s.split()) < MIN_SEG_WORDS:
            segs[-1] += s
        else:
            segs.append(s)
    return segs if len(segs) >= 2 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help="MultiWOZ data.json（2.0/2.1 或 2.2）")
    ap.add_argument("--out-turns", type=str, required=True)
    ap.add_argument("--out-segments", type=str, required=True)
    ap.add_argument("--max-dialogues", type=int, default=100, help="每种输出的最大条数")
    ap.add_argument("--min-user-turns", type=int, default=3, help="turns 格式要求的最少 user 轮数（§4 ≥3）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-level", type=str, default="INFO")
    args = ap.parse_args()

    set_global_log_level(args.log_level)
    dialogues = load_multiwoz(Path(args.input))
    random.Random(args.seed).shuffle(dialogues)

    # ---- turns 格式（E3/A2）：取前 min_user_turns 轮 user 话语 ----
    turns_out = []
    for did, users in dialogues:
        if len(users) < args.min_user_turns:
            continue
        turns_out.append({"id": did, "turns": users[:args.min_user_turns]})
        if len(turns_out) >= args.max_dialogues:
            break

    # ---- segments 格式（E2/E1）：取每条对话第一条可切分的长话语 ----
    seg_out = []
    for did, users in dialogues:
        for k, u in enumerate(users):
            if len(u.split()) < MIN_UTT_WORDS:
                continue
            segs = split_segments(u)
            if segs:
                # 严格无损校验：拼回 == 原文
                assert "".join(segs) == u, f"切分不无损: {did}"
                seg_out.append({"id": f"{did}#u{k}", "segments": segs})
                break
        if len(seg_out) >= args.max_dialogues:
            break

    for path, data, tag in [(args.out_turns, turns_out, "turns"),
                            (args.out_segments, seg_out, "segments")]:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"[{tag}] {len(data)} 条 → {p}")

    assert turns_out and seg_out, "输出为空——检查输入格式或放宽过滤参数"
    logger.info("ALL DONE ✓")


if __name__ == "__main__":
    main()
