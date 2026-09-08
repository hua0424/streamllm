# src/dialogue/unheard_detector.py
"""
未听到内容引用检测器（E3 主指标的规则版代理）。

判断"下一轮 assistant 回复"是否引用了"上一轮生成但用户没听到"的内容。
这是 E3「未听到内容引用率」的可自动化代理（见 experiment_design.md §5 E3 / §9.3）：
- 规则版（本文件）：从未听文本抽显著线索（数字、专有名词、显著内容词），
  查下一轮回复是否复述 → 本机可跑、客观、零依赖，用于 harness 验证与快速迭代。
- LLM-judge 版（实验机）：更强模型判"是否引用了未听内容"，作最终数值与规则版交叉验证。

规则版会略保守/略激进，论文中作为代理指标报告，并说明与 LLM-judge 的一致性（Cohen's κ）。
"""

import re
from typing import List, Set

# 常见停用/功能词，不作为"显著线索"
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "of", "to", "in",
    "on", "at", "and", "or", "but", "it", "its", "this", "that", "these", "those", "for",
    "with", "as", "by", "from", "one", "most", "some", "such", "also", "which", "who",
    "you", "your", "i", "we", "they", "he", "she", "his", "her", "their", "our", "about",
    "there", "here", "over", "between", "into", "than", "then", "so", "very", "can", "will",
}


def extract_cues(unheard_text: str) -> Set[str]:
    """
    从未听文本抽显著线索：
    - 数字（含年份/量词，如 1406、25）
    - 专有名词候选（首字母大写、长度>=3 的词，排除句首噪声由停用词兜底）
    - 显著内容词（长度>=5 的非停用词，覆盖名词/动词等实义词）
    统一小写返回，供子串/词匹配。
    """
    cues: Set[str] = set()
    # 数字
    for m in re.findall(r"\d+(?:[.,]\d+)?", unheard_text):
        cues.add(m.lower())
    # 词
    for w in re.findall(r"[A-Za-z][A-Za-z\-']+", unheard_text):
        wl = w.lower()
        if wl in _STOPWORDS:
            continue
        if w[0].isupper() and len(w) >= 3:      # 专有名词候选
            cues.add(wl)
        elif len(w) >= 5:                        # 显著内容词
            cues.add(wl)
    return cues


def references_unheard(unheard_text: str, response_text: str) -> bool:
    """下一轮回复是否复述了未听内容的显著线索（任一命中即算引用）。"""
    if not unheard_text.strip():
        return False
    cues = extract_cues(unheard_text)
    if not cues:
        return False
    resp = response_text.lower()
    # 词边界匹配，避免子串误命中（如 'in' 命中 'china'）
    resp_words = set(re.findall(r"[a-z0-9\-']+", resp))
    return any((cue in resp_words) or (len(cue) >= 6 and cue in resp) for cue in cues)


def matched_cues(unheard_text: str, response_text: str) -> List[str]:
    """返回命中的线索列表（调试/报告用）。"""
    cues = extract_cues(unheard_text)
    resp = response_text.lower()
    resp_words = set(re.findall(r"[a-z0-9\-']+", resp))
    return sorted(c for c in cues if (c in resp_words) or (len(c) >= 6 and c in resp))
