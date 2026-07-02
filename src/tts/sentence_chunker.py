# src/tts/sentence_chunker.py
"""
句子分块器：把 LLM 的 token 流经 stream2sentence 断句，并把每个句子片段
映射回它覆盖的 assistant token 区间 [token_start, token_end)。

这个 token-range 映射是二期"播放感知"的关键桥梁（见 docs/handoff.md 方向1 step1）：
stream2sentence 只吐句子字符串、不给 token 索引，需要我们自己对齐。

对齐算法 —— 非空白字符计数（对 whitespace 归一化免疫，比子串匹配鲁棒）：
- 喂给 stream2sentence 的每个 token 文本，累计其"非空白字符数" nws_end[i]。
- stream2sentence（关闭 cleanup）只重组文本、可能改动空白，但不增删非空白字符，
  故非空白字符序列严格保持。
- 每吐一个句子，累加其非空白字符数得到游标 cursor，则该片段末 token =
  第一个满足 nws_end[i] >= cursor 的 token i（nws_end 单调不减，二分定位）。

产出 SentenceFragment(text, token_start, token_end)，token 用 assistant 相对下标，
与 StreamLLMInference.generate_accumulating 的 rel_idx 对齐，可直接喂 PlaybackTimeline.add_fragment。
"""

import bisect
from dataclasses import dataclass
from typing import Generator, Iterable, Tuple

from stream2sentence import generate_sentences

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class SentenceFragment:
    text: str            # stream2sentence 吐出的句子/片段文本
    token_start: int     # 覆盖的 assistant token 区间起始（含）
    token_end: int       # 结束（不含）


def _nws(s: str) -> int:
    """非空白字符数。"""
    return sum(1 for c in s if not c.isspace())


# paper2_context.md §3.1 推荐配置（英文为主）
DEFAULT_S2S_KWARGS = dict(
    quick_yield_single_sentence_fragment=True,   # 首片段尽快吐，降 TTFT
    quick_yield_for_all_sentences=False,         # 后续保持完整句以保 TTS 韵律
    minimum_first_fragment_length=10,
    minimum_sentence_length=20,
    cleanup_text_links=False,                    # 关闭清洗，保持文本不增删非空白字符
    cleanup_text_emojis=False,
)


def chunk_llm_tokens(
    token_iter: Iterable[Tuple[str, int]],
    *,
    language: str = "en",
    tokenizer: str = "nltk",
    **s2s_overrides,
) -> Generator[SentenceFragment, None, None]:
    """
    token_iter：产出 (token_text, assistant_rel_idx) 的可迭代（如 generate_accumulating）。
    惰性：stream2sentence 按需拉取 token，符合真实流水线时序。
    yield：SentenceFragment，token 区间为 assistant 相对下标。
    """
    token_texts = []
    nws_end = []   # nws_end[i] = 累计到第 i 个 token（含）的非空白字符数，单调不减

    def _feeder():
        for text, _idx in token_iter:
            token_texts.append(text)
            prev = nws_end[-1] if nws_end else 0
            nws_end.append(prev + _nws(text))
            yield text

    kwargs = dict(DEFAULT_S2S_KWARGS)
    kwargs.update(s2s_overrides)

    cursor = 0        # 已吐句子累计的非空白字符数
    prev_end = 0      # 上一片段的 token_end（下一片段的 token_start）

    for sentence in generate_sentences(_feeder(), language=language, tokenizer=tokenizer, **kwargs):
        cursor += _nws(sentence)
        # 末 token = 第一个 nws_end[i] >= cursor 的 i
        end_idx = bisect.bisect_left(nws_end, cursor)
        if end_idx >= len(nws_end):
            end_idx = len(nws_end) - 1
        token_end = end_idx + 1
        if token_end <= prev_end:          # 兜底：保证区间前进（纯空白句等边缘情况）
            token_end = prev_end + 1
        frag = SentenceFragment(text=sentence, token_start=prev_end, token_end=token_end)
        logger.debug(f"[chunker] frag tokens[{frag.token_start},{frag.token_end}) '{sentence[:30]}'")
        yield frag
        prev_end = token_end
