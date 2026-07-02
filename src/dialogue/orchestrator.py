# src/dialogue/orchestrator.py
"""
DialogueOrchestrator —— 二期编排闭环 + 实验指标埋点 + 截断模式开关 + 推测-作废（C1）。

闭环（见 docs/paper2_context.md §2.2）：
  用户输入(可增量) → [软触发: conf>=spec_th → 推测生成(可作废)] → stream2sentence 断句
    → TTS 流式合成(时长) → 播放器登记进度 → PlaybackTimeline 建映射
    → [打断: 反查听到位置 → 按 truncation_mode 决定进历史边界 → crop → 重建 role]
    → 累积下一轮 → 循环

两条入口：
  - user_turn(text, barge_in_fraction)      一次性全文（无推测；E3 等既有 harness 用）
  - speculative_turn(segments, ...)          增量段输入 + 软触发推测-作废（E2/A3 用）
    确定性模拟（P1）：段序列即 ASR final 片段流（Q6 粒度），最后一段喂完 = 用户真实说完。
    推测状态机：无推测 & conf>=spec_th → 注入 generation_prompt 并预生成 spec_chunk 个 token；
    新段到来 & 有活跃推测 → 作废（KV crop 回推测起点，token 计入浪费）；
    说完时推测存活 → 直接复用（TTFT≈0），否则现场生成。

截断模式（experiment_design.md §2）：playback(B-ours) / generation(B-gen) / synthesis(B-syn)。
指标（§6）：TurnMetrics 含打断浪费 + 推测浪费(spec_*) + TTFT_effective。
"""

import time
from dataclasses import dataclass, field
from itertools import chain
from typing import List, Optional

from src.dialogue.timeline import PlaybackTimeline
from src.llm.stream_llm_inference import StreamLLMInference
from src.player.player import SimulatedPlayer
from src.tts.sentence_chunker import SentenceFragment, chunk_llm_tokens
from src.tts.streaming_tts import StreamingTTS
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

TRUNCATION_MODES = ("playback", "generation", "synthesis")


@dataclass
class TurnMetrics:
    turn_id: int
    truncation_mode: str
    n_generated: int            # 本轮最终生成 token 数（不含被作废推测）
    n_heard: int                # 用户实际听到（播放位置对齐）
    n_in_history: int           # 进入历史/KV 的 token 数（依 truncation_mode）
    n_wasted: int               # 打断浪费：生成但未进历史 = n_generated - n_in_history
    n_unheard_in_history: int   # 进了历史但用户没听到（E3 幻觉面）
    waste_rate: float           # 打断浪费率 n_wasted / n_generated
    gen_wall_ms: float          # 生成+断句+TTS 循环墙钟
    first_token_ms: float       # 首 token 相对参考时刻（user_turn: 生成开始；spec: 用户说完）
    mouth_to_ear_ms: float      # 建模：first_token_ms + TTS 首块延迟
    total_audio_s: float
    # ---- 历史处理策略（贡献3 / A2）----
    history_policy: str = "naive"       # naive / mark / rewrite
    rewrite_ms: float = 0.0             # 重写耗时（架构上被用户说话隐藏，记录供验证）
    # ---- 推测（C1 / E2）指标；user_turn 路径下为 0/None ----
    n_speculations: int = 0             # 本轮启动过几次推测
    n_invalidated: int = 0              # 被作废几次
    spec_wasted_tokens: int = 0         # 被作废推测生成的 token 总数
    spec_survived: bool = False         # 说完时是否有存活推测
    ready_tokens_at_user_end: int = 0   # 用户说完瞬间已就绪的 token 数
    spec_waste_rate: float = 0.0        # spec_wasted / (spec_wasted + n_generated)
    trigger_confs: List[float] = field(default_factory=list)


@dataclass
class TurnResult:
    turn_id: int
    user_text: str
    full_assistant_text: str        # 生成的完整（capped）回复
    history_text: str               # 进入历史的文本（依 truncation_mode）
    heard_text: str                 # 用户实际听到
    unheard_in_history_text: str    # 进了历史但没听到（B-gen 才非空；E3 关注）
    interrupted: bool
    partial: bool
    metrics: TurnMetrics = None
    fragments: List[SentenceFragment] = field(default_factory=list)


@dataclass
class _ActiveSpec:
    base_seq_len: int                       # 推测起点（user 内容末尾，crop 回滚点）
    tokens: List[tuple] = field(default_factory=list)   # [(text, rel_idx)]
    eos_hit: bool = False
    t_first_token: Optional[float] = None   # 推测首 token 墙钟


class DialogueOrchestrator:
    def __init__(self, llm: StreamLLMInference, tts: StreamingTTS, *,
                 system_prompt: str = "You are a helpful assistant. Reply in English.",
                 language: str = "en", tokenizer: str = "nltk",
                 max_speculative_tokens: int = 48,
                 truncation_mode: str = "playback",
                 trigger=None, spec_threshold: float = 0.5, spec_chunk: int = 16,
                 history_policy: str = "naive", rewriter=None, mark_text: str = " …"):
        if truncation_mode not in TRUNCATION_MODES:
            raise ValueError(f"truncation_mode 须为 {TRUNCATION_MODES}")
        if history_policy not in ("naive", "mark", "rewrite"):
            raise ValueError("history_policy 须为 naive/mark/rewrite")
        if history_policy == "rewrite" and rewriter is None:
            raise ValueError("history_policy=rewrite 需要传入 rewriter（HistoryRewriter）")
        self.llm = llm
        self.tts = tts
        self.system_prompt = system_prompt
        self.language = language
        self.tokenizer = tokenizer
        self.max_spec = max_speculative_tokens
        self.truncation_mode = truncation_mode
        self.trigger = trigger                   # LLMSoftTrigger 或 None（不推测）
        self.spec_threshold = spec_threshold     # 推测阈值（激进度，E2 扫描对象）
        self.spec_chunk = spec_chunk             # 每次推测预生成 token 上限（§八：限制推测长度）
        self.history_policy = history_policy     # 贡献3：naive / mark / rewrite
        self.rewriter = rewriter                 # HistoryRewriter（rewrite 时必需）
        self.mark_text = mark_text               # 标记法追加的打断标记（零模型成本）
        self.acc: Optional[StreamLLMInference.AccumKVCache] = None
        self.turn_id = 0
        self._started = False

    # ------------------------------------------------------------- 内部工具
    def _timed_tokens(self, gen):
        """包裹 token 迭代器以记录首 token 墙钟时刻。"""
        self._t_first = None
        for tok in gen:
            if self._t_first is None:
                self._t_first = time.perf_counter()
            yield tok

    def _finish_assistant(self, token_iter, barge_in_fraction: Optional[float],
                          t_ref: float, spec_stats: dict) -> TurnResult:
        """
        后半段（生成消费→断句→TTS→timeline→打断→crop→reopen→metrics）。
        token_iter: (text, rel_idx) 迭代器；t_ref: first_token_ms 的参考时刻。
        """
        timeline = PlaybackTimeline(turn_id=self.turn_id)
        player = SimulatedPlayer(timeline)
        fragments: List[SentenceFragment] = []

        t_start = time.perf_counter()
        for frag in chunk_llm_tokens(
            self._timed_tokens(token_iter),
            language=self.language, tokenizer=self.tokenizer,
        ):
            fid = timeline.add_fragment(frag.text, frag.token_start, frag.token_end)
            for chunk in self.tts.synthesize(frag.text):
                player.enqueue(fid, chunk)
            fragments.append(frag)
        gen_wall_ms = (time.perf_counter() - t_start) * 1000
        first_token_ms = ((self._t_first - t_ref) * 1000) if self._t_first else 0.0

        full_ids = list(self.acc.assistant_token_ids)      # crop 前完整生成
        n_gen = len(full_ids)

        if barge_in_fraction is not None and player.total_samples > 0:
            player.seek_fraction(barge_in_fraction)
            res = timeline.barge_in_readonly()
            heard_rel, partial, interrupted = res.crop_token_end, res.partial, True
        else:
            heard_rel, partial, interrupted = n_gen, False, False

        keep_rel = heard_rel if self.truncation_mode == "playback" else n_gen
        keep = self.acc.assistant_start + keep_rel
        self.llm.crop_to_token(self.acc, keep)             # keep==seq_length 时等价 no-op

        heard_text = self.llm.tokenizer.decode(full_ids[:heard_rel], skip_special_tokens=True)
        history_text = self.llm.tokenizer.decode(full_ids[:keep_rel], skip_special_tokens=True)
        unheard_in_hist = self.llm.tokenizer.decode(full_ids[heard_rel:keep_rel], skip_special_tokens=True)

        # ---- 贡献3：被打断轮的历史处理策略（assistant role 尚未关闭，追加/替换发生在此）----
        rewrite_ms = 0.0
        truly_truncated = interrupted and keep_rel < n_gen
        if truly_truncated and self.history_policy == "mark":
            # 标记法：在被截断的 assistant 内容尾部追加打断标记（零延迟零模型成本）。
            # _prefill_text_p2 是"向当前打开的 role 追加裸文本"，此时打开的是 assistant。
            self.llm.prefill_user_text(self.acc, self.mark_text)
            history_text = history_text + self.mark_text
        elif truly_truncated and self.history_policy == "rewrite" and partial and history_text.strip():
            # 重写法：仅截断落在语义不完整处（partial）时启用。重写不新增信息；
            # KV 层面替换被打断的 assistant 段（crop 回 assistant_start + prefill 重写文本）。
            # 架构上重写与用户说话并行（延迟被隐藏）；此处同步执行并记录耗时供验证。
            rewritten, rewrite_ms = self.rewriter.rewrite(history_text)
            self.llm.crop_to_token(self.acc, self.acc.assistant_start)
            self.llm.prefill_user_text(self.acc, rewritten)
            history_text = rewritten

        self.llm.reopen_user_role(self.acc)                # 关闭 assistant、打开 user

        spec_wasted = spec_stats.get("wasted", 0)
        metrics = TurnMetrics(
            turn_id=self.turn_id, truncation_mode=self.truncation_mode,
            n_generated=n_gen, n_heard=heard_rel, n_in_history=keep_rel,
            n_wasted=n_gen - keep_rel, n_unheard_in_history=keep_rel - heard_rel,
            waste_rate=(n_gen - keep_rel) / n_gen if n_gen else 0.0,
            gen_wall_ms=gen_wall_ms, first_token_ms=first_token_ms,
            mouth_to_ear_ms=first_token_ms + self.tts.first_chunk_latency_ms,
            total_audio_s=player.total_samples / 16000.0,
            n_speculations=spec_stats.get("n_spec", 0),
            n_invalidated=spec_stats.get("n_inval", 0),
            spec_wasted_tokens=spec_wasted,
            spec_survived=spec_stats.get("survived", False),
            ready_tokens_at_user_end=spec_stats.get("ready", 0),
            spec_waste_rate=spec_wasted / (spec_wasted + n_gen) if (spec_wasted + n_gen) else 0.0,
            trigger_confs=spec_stats.get("confs", []),
            history_policy=self.history_policy, rewrite_ms=rewrite_ms,
        )
        return TurnResult(
            turn_id=self.turn_id, user_text="",
            full_assistant_text=self.llm.tokenizer.decode(full_ids, skip_special_tokens=True),
            history_text=history_text, heard_text=heard_text,
            unheard_in_history_text=unheard_in_hist,
            interrupted=interrupted, partial=partial, metrics=metrics, fragments=fragments,
        )

    # ------------------------------------------------------------- 入口 1：一次性全文
    def user_turn(self, user_text: str, barge_in_fraction: Optional[float] = None) -> TurnResult:
        """一次性全文输入（无推测）。E3 等既有 harness 的入口，行为与旧版一致。"""
        self.turn_id += 1
        if not self._started:
            kv = self.llm.cache_prompt(user_text, is_end=True, system_prompt=self.system_prompt)
            self.acc = self.llm.to_accum_cache(kv)
            self._started = True
        else:
            self.llm.prefill_user_text(self.acc, user_text)
            self.llm.open_assistant_role(self.acc)

        t_ref = time.perf_counter()
        r = self._finish_assistant(
            self.llm.generate_accumulating(self.acc, max_new_tokens=self.max_spec),
            barge_in_fraction, t_ref, spec_stats={},
        )
        r.user_text = user_text
        m = r.metrics
        logger.info(f"[turn {self.turn_id}/{self.truncation_mode}] interrupted={r.interrupted} "
                    f"gen={m.n_generated} heard={m.n_heard} hist={m.n_in_history} "
                    f"waste={m.waste_rate:.0%} unheard_in_hist={m.n_unheard_in_history} "
                    f"TTFT~{m.first_token_ms:.0f}ms")
        return r

    # ------------------------------------------------------------- 入口 2：增量段 + 推测
    def _start_speculation(self) -> _ActiveSpec:
        spec = _ActiveSpec(base_seq_len=self.acc.seq_length)
        self.llm.open_assistant_role(self.acc)     # 注入 generation_prompt，设 assistant_start
        for text, idx in self.llm.generate_accumulating(self.acc, max_new_tokens=self.spec_chunk):
            if spec.t_first_token is None:
                spec.t_first_token = time.perf_counter()
            spec.tokens.append((text, idx))
        spec.eos_hit = len(spec.tokens) < self.spec_chunk
        return spec

    def _invalidate_speculation(self, spec: _ActiveSpec) -> int:
        """作废推测：KV crop 回推测起点（user role 回到打开状态），返回浪费 token 数。"""
        wasted = len(spec.tokens)
        self.llm.crop_to_token(self.acc, spec.base_seq_len)
        self.acc.assistant_start = self.acc.seq_length      # 同步占位，避免悬空
        self.acc.assistant_token_ids = []
        return wasted

    def speculative_turn(self, segments: List[str],
                         barge_in_fraction: Optional[float] = None) -> TurnResult:
        """
        增量段输入 + 软触发推测-作废（确定性模拟）。segments 为 ASR final 片段流，
        最后一段喂完 = 用户真实说完（ground truth 端点，P1）。
        需要 self.trigger 非 None，否则退化为无推测（仅增量 prefill）。
        """
        self.turn_id += 1
        accum_text = ""
        spec: Optional[_ActiveSpec] = None
        stats = {"n_spec": 0, "n_inval": 0, "wasted": 0, "confs": []}

        for seg in segments:
            # 新段到来：活跃推测一律作废（用户还在说 → 早触发错误）
            if spec is not None:
                stats["n_inval"] += 1
                stats["wasted"] += self._invalidate_speculation(spec)
                spec = None
            # prefill 本段
            if not self._started:
                kv = self.llm.cache_prompt(seg, is_end=False, system_prompt=self.system_prompt)
                self.acc = self.llm.to_accum_cache(kv)
                self.acc.assistant_start = self.acc.seq_length   # user 未结束，占位
                self._started = True
            else:
                self.llm.prefill_user_text(self.acc, seg)
            accum_text += seg
            # 软触发评估（含最后一段——真实系统中"存活的推测"正是最后一个 final 片段
            # 触发、其后无新语音的那次，推测计算被静音检测窗掩盖；确定性模拟等价处理）。
            # 真实系统中 trigger 与 prefill 并行（D-003），此处顺序执行、不计入 TTFT。
            if self.trigger is not None:
                conf = self.trigger.confidence(accum_text)
                stats["confs"].append(conf)
                if conf >= self.spec_threshold:
                    spec = self._start_speculation()
                    stats["n_spec"] += 1

        # ---- 用户真实说完 ----
        t_user_end = time.perf_counter()
        if spec is not None:
            # 推测存活：回放已就绪 token，再续生成剩余
            stats["survived"] = True
            stats["ready"] = len(spec.tokens)
            remaining = 0 if spec.eos_hit else self.max_spec - len(spec.tokens)
            token_iter = chain(
                iter(spec.tokens),
                self.llm.generate_accumulating(self.acc, max_new_tokens=remaining)
                if remaining > 0 else iter(()),
            )
        else:
            # 无存活推测：现场打开 assistant 并生成
            stats["survived"] = False
            stats["ready"] = 0
            self.llm.open_assistant_role(self.acc)
            token_iter = self.llm.generate_accumulating(self.acc, max_new_tokens=self.max_spec)

        r = self._finish_assistant(token_iter, barge_in_fraction, t_user_end, stats)
        r.user_text = accum_text
        m = r.metrics
        logger.info(f"[spec-turn {self.turn_id} th={self.spec_threshold}] "
                    f"spec={m.n_speculations} inval={m.n_invalidated} wasted={m.spec_wasted_tokens} "
                    f"survived={m.spec_survived} ready={m.ready_tokens_at_user_end} "
                    f"spec_waste={m.spec_waste_rate:.0%} TTFT_eff={m.first_token_ms:.0f}ms")
        return r

    def assert_kv_consistent(self) -> bool:
        a = self.acc
        return a.seq_length == a.attention_mask.shape[1] == a.past_key_values.get_seq_length()
