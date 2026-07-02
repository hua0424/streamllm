# src/dialogue/orchestrator.py
"""
DialogueOrchestrator —— 二期编排闭环 + 实验指标埋点 + 截断模式开关。

闭环（见 docs/paper2_context.md §2.2）：
  用户输入 → LLM 流式生成(推测, cap) → stream2sentence 断句(带 token 区间)
    → TTS 流式合成(时长) → 播放器登记进度 → PlaybackTimeline 建映射
    → [打断: 反查听到位置 → 按 truncation_mode 决定进历史的边界 → crop → 重建 role]
    → 累积下一轮 → 循环

截断模式（experiment_design.md §2 被测条件）：
  - "playback"  (B-ours) : 进历史 = 用户实际听到的（crop 到听到边界）——本文方法
  - "generation"(B-gen)  : 进历史 = LLM 已生成的全部（不 crop）——朴素对照
  - "synthesis" (B-syn)  : 进历史 = TTS 已合成的（本模型里≈全部生成，标注为近似）

指标埋点（experiment_design.md §6）：每轮产出 TurnMetrics（token 计数、推测浪费率、
未听到却进历史的 token 数=E3 的幻觉面、生成墙钟/首 token 时延、mouth-to-ear 建模值）。

确定性编排（P1）：打断以程序化播放比例注入，可复现。软触发用"到点即生成"占位；
real CosyVoice2 通过替换 tts 实现接入（D-010）。
"""

import time
from dataclasses import dataclass, field
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
    n_generated: int            # 本轮生成 token 数
    n_heard: int                # 用户实际听到（播放位置对齐）
    n_in_history: int           # 进入历史/KV 的 token 数（依 truncation_mode）
    n_wasted: int               # 生成但未进历史（被作废）= n_generated - n_in_history
    n_unheard_in_history: int   # 进了历史但用户没听到 = n_in_history - n_heard（E3 幻觉面）
    waste_rate: float           # n_wasted / n_generated
    gen_wall_ms: float          # 生成+断句+TTS 循环墙钟
    first_token_ms: float       # 到首 token 时延（TTFT_text 近似）
    mouth_to_ear_ms: float      # 建模：first_token_ms + TTS 首块延迟
    total_audio_s: float        # 本轮合成音频总时长


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


class DialogueOrchestrator:
    def __init__(self, llm: StreamLLMInference, tts: StreamingTTS, *,
                 system_prompt: str = "You are a helpful assistant. Reply in English.",
                 language: str = "en", tokenizer: str = "nltk",
                 max_speculative_tokens: int = 48,
                 truncation_mode: str = "playback"):
        if truncation_mode not in TRUNCATION_MODES:
            raise ValueError(f"truncation_mode 须为 {TRUNCATION_MODES}")
        self.llm = llm
        self.tts = tts
        self.system_prompt = system_prompt
        self.language = language
        self.tokenizer = tokenizer
        self.max_spec = max_speculative_tokens
        self.truncation_mode = truncation_mode
        self.acc: Optional[StreamLLMInference.AccumKVCache] = None
        self.turn_id = 0
        self._started = False

    def _timed_tokens(self, gen):
        """包裹 token 生成器以记录首 token 时间。"""
        self._t_first = None
        for tok in gen:
            if self._t_first is None:
                self._t_first = time.perf_counter()
            yield tok

    def _assistant_turn(self, barge_in_fraction: Optional[float]) -> TurnResult:
        timeline = PlaybackTimeline(turn_id=self.turn_id)
        player = SimulatedPlayer(timeline)
        fragments: List[SentenceFragment] = []

        t_start = time.perf_counter()
        self._t_first = None
        for frag in chunk_llm_tokens(
            self._timed_tokens(self.llm.generate_accumulating(self.acc, max_new_tokens=self.max_spec)),
            language=self.language, tokenizer=self.tokenizer,
        ):
            fid = timeline.add_fragment(frag.text, frag.token_start, frag.token_end)
            for chunk in self.tts.synthesize(frag.text):
                player.enqueue(fid, chunk)
            fragments.append(frag)
        gen_wall_ms = (time.perf_counter() - t_start) * 1000
        first_token_ms = ((self._t_first - t_start) * 1000) if self._t_first else 0.0

        full_ids = list(self.acc.assistant_token_ids)      # crop 前完整生成
        n_gen = len(full_ids)

        # 反查听到位置（只读，不改状态）；再按 truncation_mode 决定进历史边界
        if barge_in_fraction is not None and player.total_samples > 0:
            player.seek_fraction(barge_in_fraction)
            res = timeline.barge_in_readonly()
            heard_rel = res.crop_token_end
            partial = res.partial
            interrupted = True
        else:
            heard_rel = n_gen
            partial = False
            interrupted = False

        if self.truncation_mode == "playback":
            keep_rel = heard_rel                       # B-ours：只留听到的
        else:
            keep_rel = n_gen                           # B-gen / B-syn：留全部生成

        keep = self.acc.assistant_start + keep_rel
        self.llm.crop_to_token(self.acc, keep)         # keep==seq_length 时等价 no-op

        heard_text = self.llm.tokenizer.decode(full_ids[:heard_rel], skip_special_tokens=True)
        history_text = self.llm.tokenizer.decode(full_ids[:keep_rel], skip_special_tokens=True)
        unheard_in_hist = self.llm.tokenizer.decode(full_ids[heard_rel:keep_rel], skip_special_tokens=True)

        # 关闭 assistant、打开 user（为下一轮）
        self.llm.reopen_user_role(self.acc)

        total_audio_s = player.total_samples / 16000.0
        metrics = TurnMetrics(
            turn_id=self.turn_id, truncation_mode=self.truncation_mode,
            n_generated=n_gen, n_heard=heard_rel, n_in_history=keep_rel,
            n_wasted=n_gen - keep_rel, n_unheard_in_history=keep_rel - heard_rel,
            waste_rate=(n_gen - keep_rel) / n_gen if n_gen else 0.0,
            gen_wall_ms=gen_wall_ms, first_token_ms=first_token_ms,
            mouth_to_ear_ms=first_token_ms + self.tts.first_chunk_latency_ms,
            total_audio_s=total_audio_s,
        )
        return TurnResult(
            turn_id=self.turn_id, user_text="", full_assistant_text=self.llm.tokenizer.decode(full_ids, skip_special_tokens=True),
            history_text=history_text, heard_text=heard_text, unheard_in_history_text=unheard_in_hist,
            interrupted=interrupted, partial=partial, metrics=metrics, fragments=fragments,
        )

    def user_turn(self, user_text: str, barge_in_fraction: Optional[float] = None) -> TurnResult:
        """走一轮对话。barge_in_fraction: None=完整听完；0..1=在该播放比例处打断。"""
        self.turn_id += 1
        if not self._started:
            kv = self.llm.cache_prompt(user_text, is_end=True, system_prompt=self.system_prompt)
            self.acc = self.llm.to_accum_cache(kv)
            self._started = True
        else:
            self.llm.prefill_user_text(self.acc, user_text)
            self.llm.open_assistant_role(self.acc)

        r = self._assistant_turn(barge_in_fraction)
        r.user_text = user_text
        m = r.metrics
        logger.info(f"[turn {self.turn_id}/{self.truncation_mode}] interrupted={r.interrupted} "
                    f"gen={m.n_generated} heard={m.n_heard} hist={m.n_in_history} "
                    f"waste={m.waste_rate:.0%} unheard_in_hist={m.n_unheard_in_history} "
                    f"TTFT~{m.first_token_ms:.0f}ms")
        return r

    def assert_kv_consistent(self) -> bool:
        a = self.acc
        return a.seq_length == a.attention_mask.shape[1] == a.past_key_values.get_seq_length()
