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
- S5-S9 有限顺序契约负向测试（token/chunk/fragment/sample/played/status）
"""

from src.dialogue.timeline import PlaybackTimeline, FragmentStatus
from src.utils.check_utils import make_check
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


_check = make_check(logger)


def _check_raises(name, exception_type, action, message_part=None):
    """负向 smoke 断言：指定调用必须以预期异常失败。"""
    try:
        action()
    except exception_type as exc:
        if message_part is not None:
            _check(f"{name}（错误信息）", message_part in str(exc))
        _check(name, True)
    else:
        _check(name, False)


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


def test_token_span_contract():
    """S5：token span 必须从 0 开始、连续且非空；失败不污染下一次合法写入。"""
    logger.info("S5 token span 连续且非空")
    tl = PlaybackTimeline()
    _check_raises("首片段不能跳过 token", ValueError,
                  lambda: tl.add_fragment("bad", 1, 2), "expected start 0")
    _check_raises("新片段拒绝非初始状态", ValueError,
                  lambda: tl.add_fragment("bad", 0, 1, FragmentStatus.PLAYED),
                  "must start as SPECULATIVE")
    _check_raises("空 token span 被拒绝", ValueError,
                  lambda: tl.add_fragment("bad", 0, 0), "non-empty")
    f0 = tl.add_fragment("ok", 0, 2)
    _check("失败后首个合法 fragment_id 仍为 0", f0 == 0)
    _check_raises("后续 token gap 被拒绝", ValueError,
                  lambda: tl.add_fragment("gap", 3, 4), "expected start 2")
    _check_raises("后续 token overlap 被拒绝", ValueError,
                  lambda: tl.add_fragment("overlap", 1, 3), "expected start 2")
    f1 = tl.add_fragment("ok2", 2, 4)
    _check("失败后可追加连续 span", f1 == 1 and len(tl.snapshot()) == 2)


def test_chunk_order_contract():
    """S6：chunk ID 全局唯一，且后续片段开始后不能回头 attach。"""
    logger.info("S6 chunk 唯一与 fragment attach 顺序")
    tl = PlaybackTimeline()
    f0 = tl.add_fragment("a", 0, 1)
    f1 = tl.add_fragment("b", 1, 2)
    tl.attach_chunk(f0, 10, 100)
    _check_raises("重复 chunk_id 被拒绝", ValueError,
                  lambda: tl.attach_chunk(f0, 10, 50), "duplicate chunk_id")
    _check("重复 chunk 失败后 sample 轴不变", tl.total_samples == 100)
    tl.attach_chunk(f1, 11, 200)
    _check_raises("禁止回头给旧 fragment attach", ValueError,
                  lambda: tl.attach_chunk(f0, 12, 50), "cannot attach fragment 0")
    _check("回头 attach 失败后 sample ranges 连续不重叠",
           tl.get_fragment(f0).sample_start == 0
           and tl.get_fragment(f0).sample_end == 100
           and tl.get_fragment(f1).sample_start == 100
           and tl.get_fragment(f1).sample_end == 300
           and tl.total_samples == 300)


def test_sample_range_fail_closed():
    """S7：检测内部 sample 轴不连续时拒绝写入，并保持记录原样。"""
    logger.info("S7 sample ranges 连续且 fail closed")
    tl = PlaybackTimeline()
    f0 = tl.add_fragment("a", 0, 1)
    tl.attach_chunk(f0, 20, 100)
    rec = tl.get_fragment(f0)
    rec.sample_end = 90                  # 模拟并发外部误改/损坏输入状态
    _check_raises("不连续 sample range 被拒绝", RuntimeError,
                  lambda: tl.attach_chunk(f0, 21, 10), "non-contiguous sample range")
    _check("失败不追加 chunk 或推进 total_samples",
           rec.chunk_ids == [20] and rec.sample_end == 90 and tl.total_samples == 100)


def test_played_cursor_contract():
    """S8：set_played 拒绝负数和回退，合法超播钳制；只读查询仍保留越界语义。"""
    logger.info("S8 played 游标单调与钳制")
    tl = PlaybackTimeline()
    f0 = tl.add_fragment("a", 0, 1)
    tl.attach_chunk(f0, 30, 100)
    _check_raises("非整数播放游标被拒绝", TypeError,
                  lambda: tl.set_played(1.5), "must be int")
    _check_raises("负播放游标被拒绝", ValueError,
                  lambda: tl.set_played(-1), "must be >= 0")
    tl.set_played(60)
    _check_raises("播放游标回退被拒绝", ValueError,
                  lambda: tl.set_played(59), "cannot move backward")
    _check("回退失败后游标不变", tl.played_samples == 60)
    tl.set_played(999)
    _check("合法超播钳制到 total_samples", tl.played_samples == 100)
    _check("越界只读查询仍命中最后片段", tl.fragment_at_sample(999).fragment_id == f0)
    _check("负数只读查询仍表示尚未听到", tl.fragment_at_sample(-1) is None)


def test_status_contract():
    """S9：状态只按合法生命周期推进，非法类型/跳步/回退/终态复活均 fail closed。"""
    logger.info("S9 status 转换 fail closed")
    tl = PlaybackTimeline()
    f0 = tl.add_fragment("a", 0, 1)
    _check_raises("状态拒绝非法类型", TypeError,
                  lambda: tl.mark_status(f0, "PLAYED"), "FragmentStatus")
    _check_raises("状态拒绝跳步", ValueError,
                  lambda: tl.mark_status(f0, FragmentStatus.PLAYING), "SPECULATIVE -> PLAYING")
    _check("非法状态转换后保持 SPECULATIVE",
           tl.get_fragment(f0).status == FragmentStatus.SPECULATIVE)
    tl.mark_status(f0, FragmentStatus.SYNTHESIZING)
    tl.mark_status(f0, FragmentStatus.ENQUEUED)
    tl.mark_status(f0, FragmentStatus.PLAYING)
    tl.mark_status(f0, FragmentStatus.PLAYED)
    tl.mark_status(f0, FragmentStatus.PLAYED)  # 幂等合法
    _check_raises("PLAYED 终态不能回退", ValueError,
                  lambda: tl.mark_status(f0, FragmentStatus.PLAYING), "PLAYED -> PLAYING")

    f1 = tl.add_fragment("b", 1, 2)
    tl.mark_status(f1, FragmentStatus.DISCARDED)
    _check_raises("DISCARDED 终态不能复活", ValueError,
                  lambda: tl.mark_status(f1, FragmentStatus.SYNTHESIZING),
                  "DISCARDED -> SYNTHESIZING")
    _check_raises("DISCARDED 片段不能 attach 音频", ValueError,
                  lambda: tl.attach_chunk(f1, 31, 100), "status DISCARDED")
    _check("状态失败后保持终态且 sample 轴未推进",
           tl.get_fragment(f1).status == FragmentStatus.DISCARDED and tl.total_samples == 0)

    tl2 = PlaybackTimeline()
    g0 = tl2.add_fragment("heard", 0, 1)
    g1 = tl2.add_fragment("discarded", 1, 2)
    tl2.attach_chunk(g0, 40, 100)
    tl2.attach_chunk(g1, 41, 100)
    tl2.set_played(50)
    tl2.barge_in()
    statuses_before = [f.status for f in tl2.snapshot()]
    _check_raises("重复 barge_in 不能复活已丢弃片段", ValueError,
                  lambda: tl2.barge_in(150), "DISCARDED -> PLAYING")
    _check("barge_in 状态失败时全量 fail closed",
           [f.status for f in tl2.snapshot()] == statuses_before)


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
    test_token_span_contract()
    test_chunk_order_contract()
    test_sample_range_fail_closed()
    test_played_cursor_contract()
    test_status_contract()
    logger.info("=" * 56)
    logger.info("ALL PASS ✓")
    logger.info("=" * 56)


if __name__ == "__main__":
    main()
