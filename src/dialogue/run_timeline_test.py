# src/dialogue/run_timeline_test.py
"""
PlaybackTimeline smoke test（纯 Python，无需 GPU/torch）。

运行（项目根目录）：
    uv run python -m src.dialogue.run_timeline_test

覆盖场景：
- S1 mid-fragment 打断（选 A：被打断片段算已听到、partial=True、后续作废）
- S2 片段边界打断（partial=False，干净截断）
- S3 打断时尚无音频播放（整段推测回滚，crop_token_end=0）
- S4 反查一致性（fragment_at_sample / heard_text）
"""

from src.dialogue.timeline import PlaybackTimeline, FragmentStatus
from src.utils.logging_utils import get_logger, set_global_log_level

logger = get_logger(__name__)

# 采样率仅用于把"秒"换算成 samples 便于阅读
SR = 16000


def _build_reply() -> PlaybackTimeline:
    """
    构造一段 assistant 回复：3 个片段。
    frag0 tokens[0,5)   "今天天气晴，"   1.0s 音频 -> samples [0, 16000)
    frag1 tokens[5,11)  "温度25度，"     1.0s 音频 -> samples [16000, 32000)
    frag2 tokens[11,18) "适合出门散步。" 1.0s 音频 -> samples [32000, 48000)
    """
    tl = PlaybackTimeline(turn_id=1)
    f0 = tl.add_fragment("今天天气晴，", 0, 5)
    f1 = tl.add_fragment("温度25度，", 5, 11)
    f2 = tl.add_fragment("适合出门散步。", 11, 18)
    tl.attach_chunk(f0, chunk_id=0, n_samples=SR)      # frag0 [0,16000)
    tl.attach_chunk(f1, chunk_id=1, n_samples=SR)      # frag1 [16000,32000)
    tl.attach_chunk(f2, chunk_id=2, n_samples=SR)      # frag2 [32000,48000)
    return tl


def _check(name: str, cond: bool):
    status = "PASS" if cond else "FAIL"
    logger.info(f"  [{status}] {name}")
    if not cond:
        raise AssertionError(name)


def test_mid_fragment():
    """S1：在 frag1 中间（播放到 1.5s = 24000 samples）打断。"""
    logger.info("S1 mid-fragment 打断 @1.5s")
    tl = _build_reply()
    tl.set_played(int(1.5 * SR))          # 播放到 frag1 中段
    res = tl.barge_in()
    _check("被打断片段=frag1", res.interrupted_fragment_id == 1)
    _check("crop 到 frag1.token_end=11", res.crop_token_end == 11)   # 选 A：含被打断片段
    _check("partial=True（半截）", res.partial is True)
    _check("heard=[0,1]", res.heard_fragment_ids == [0, 1])
    _check("discarded=[2]", res.discarded_fragment_ids == [2])
    _check("frag2 状态=DISCARDED", tl._by_id[2].status == FragmentStatus.DISCARDED)


def test_boundary():
    """S2：恰好在 frag1 结束边界（2.0s = 32000 samples）打断 —— 干净截断。
    count 语义：播放 32000 采样 = 听完 [0,32000) = 完整听完 frag0+frag1、frag2 一个采样都没听到。
    故命中 frag1、crop 到 frag1.token_end=11、partial=False（干净边界）、frag2 作废。"""
    logger.info("S2 片段边界打断 @2.0s（干净截断）")
    tl = _build_reply()
    tl.set_played(int(2.0 * SR))          # 恰好听完 frag1
    res = tl.barge_in()
    _check("被打断片段=frag1（听完 frag1）", res.interrupted_fragment_id == 1)
    _check("crop 到 frag1.token_end=11", res.crop_token_end == 11)
    _check("partial=False（干净边界，非半截）", res.partial is False)
    _check("heard=[0,1]", res.heard_fragment_ids == [0, 1])
    _check("discarded=[2]", res.discarded_fragment_ids == [2])


def test_clean_boundary_frag0():
    """S2b：听完 frag0（1.0s = 16000）时打断 —— 只保留 frag0，干净截断。"""
    logger.info("S2b 听完 frag0 @1.0s（干净截断）")
    tl = _build_reply()
    res = tl.barge_in_readonly(int(1.0 * SR))   # 只读，不改状态
    _check("命中 frag0（听完 frag0）", res.interrupted_fragment_id == 0)
    _check("crop 到 frag0.token_end=5", res.crop_token_end == 5)
    _check("partial=False（frag0 完整听完）", res.partial is False)
    _check("heard=[0]", res.heard_fragment_ids == [0])
    _check("discarded=[1,2]", res.discarded_fragment_ids == [1, 2])
    _check("只读不改状态：frag2 仍 SYNTHESIZING", tl._by_id[2].status == FragmentStatus.SYNTHESIZING)


def test_full_rollback():
    """S3：打断时还没有任何音频播放（played=0，且无片段命中开头之前）。"""
    logger.info("S3 打断时尚未播放任何音频")
    tl = PlaybackTimeline(turn_id=2)
    tl.add_fragment("推测的内容", 0, 4)   # 只有 SPECULATIVE，还没 attach 音频
    res = tl.barge_in()                    # played=0，无 audio 片段 → 整段回滚
    _check("interrupted=None", res.interrupted_fragment_id is None)
    _check("crop_token_end=0（回滚到 assistant 起点）", res.crop_token_end == 0)
    _check("heard 为空", res.heard_fragment_ids == [])
    _check("全部 discarded", res.discarded_fragment_ids == [0])


def test_reverse_and_heard_text():
    """S4：反查与 heard_text 一致性。"""
    logger.info("S4 反查 / heard_text")
    tl = _build_reply()
    _check("sample 24000 命中 frag1", tl.fragment_at_sample(24000).fragment_id == 1)
    _check("sample 40000 命中 frag2", tl.fragment_at_sample(40000).fragment_id == 2)
    _check("sample 越界(99999) 命中最后片段 frag2", tl.fragment_at_sample(99999).fragment_id == 2)
    heard = tl.heard_text(int(1.5 * SR))   # 听到 frag0+frag1
    _check("heard_text='今天天气晴，温度25度，'", heard == "今天天气晴，温度25度，")


def main():
    set_global_log_level("INFO")
    logger.info("=" * 56)
    logger.info("PlaybackTimeline smoke test")
    logger.info("=" * 56)
    test_mid_fragment()
    test_boundary()
    test_clean_boundary_frag0()
    test_full_rollback()
    test_reverse_and_heard_text()
    logger.info("=" * 56)
    logger.info("ALL PASS ✓")
    logger.info("=" * 56)


if __name__ == "__main__":
    main()
