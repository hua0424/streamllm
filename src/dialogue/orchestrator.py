# src/dialogue/orchestrator.py
"""
DialogueOrchestrator —— 二期编排闭环。

把各组件串成"用户感知一致性"的对话循环（见 docs/paper2_context.md §2.2）：

  用户输入 → LLM 流式生成(推测, 可 cap) → stream2sentence 断句(带 token 区间)
    → TTS 流式合成(时长) → 播放器登记进度 → PlaybackTimeline 建映射
    → [打断: 按实际播放位置反查 → crop KV → 重建 role] → 累积下一轮 → 循环

本版为**确定性**编排（experiment_design.md P1）：打断以程序化播放比例注入，可复现。
软触发（TEN, D-003）此处用"到点即生成"占位——真实流式 ASR 增量输入接入后再换。
real CosyVoice2 通过替换 tts 实现接入（D-010）。
"""

from dataclasses import dataclass, field
from typing import List, Optional

from src.dialogue.timeline import PlaybackTimeline
from src.llm.stream_llm_inference import StreamLLMInference
from src.player.player import SimulatedPlayer
from src.tts.sentence_chunker import SentenceFragment, chunk_llm_tokens
from src.tts.streaming_tts import StreamingTTS
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class TurnResult:
    turn_id: int
    user_text: str
    full_assistant_text: str        # 生成的完整（capped）回复
    heard_text: str                 # 用户实际听到 → 进入历史
    discarded_text: str             # 未听到 → 作废
    interrupted: bool
    partial: bool                   # 被打断片段是否半截（→ 需贡献3重写）
    n_generated_tokens: int
    n_heard_tokens: int
    crop_seq_len: int               # crop 后绝对 KV 长度
    fragments: List[SentenceFragment] = field(default_factory=list)


class DialogueOrchestrator:
    def __init__(self, llm: StreamLLMInference, tts: StreamingTTS, *,
                 system_prompt: str = "You are a helpful assistant. Reply in English.",
                 language: str = "en", tokenizer: str = "nltk",
                 max_speculative_tokens: int = 48):
        self.llm = llm
        self.tts = tts
        self.system_prompt = system_prompt
        self.language = language
        self.tokenizer = tokenizer
        self.max_spec = max_speculative_tokens
        self.acc: Optional[StreamLLMInference.AccumKVCache] = None
        self.turn_id = 0
        self._started = False

    def _assistant_turn(self, barge_in_fraction: Optional[float]) -> dict:
        """assistant role 已打开、acc.assistant_start 已设。生成→断句→TTS→timeline→(打断)crop。"""
        timeline = PlaybackTimeline(turn_id=self.turn_id)
        player = SimulatedPlayer(timeline)
        fragments: List[SentenceFragment] = []

        # 生成 + 断句 + TTS + 建 timeline（惰性：stream2sentence 按需拉 token，符合真实时序）
        for frag in chunk_llm_tokens(
            self.llm.generate_accumulating(self.acc, max_new_tokens=self.max_spec),
            language=self.language, tokenizer=self.tokenizer,
        ):
            fid = timeline.add_fragment(frag.text, frag.token_start, frag.token_end)
            for chunk in self.tts.synthesize(frag.text):
                player.enqueue(fid, chunk)
            fragments.append(frag)

        full_ids = list(self.acc.assistant_token_ids)   # crop 前的完整生成
        full_text = self.llm.tokenizer.decode(full_ids, skip_special_tokens=True)

        if barge_in_fraction is not None and player.total_samples > 0:
            player.seek_fraction(barge_in_fraction)
            res = timeline.barge_in()
            crop_rel = res.crop_token_end
            keep = self.acc.assistant_start + crop_rel
            self.llm.crop_to_token(self.acc, keep)
            interrupted, partial = True, res.partial
        else:
            crop_rel = len(full_ids)
            keep = self.acc.seq_length
            interrupted, partial = False, False

        heard_ids = full_ids[:crop_rel]
        heard_text = self.llm.tokenizer.decode(heard_ids, skip_special_tokens=True)
        discarded_text = self.llm.tokenizer.decode(full_ids[crop_rel:], skip_special_tokens=True)

        # 关闭 assistant、打开 user（为下一轮累积做准备）——即使没被打断也这样收尾
        self.llm.reopen_user_role(self.acc)

        return dict(full_text=full_text, heard_text=heard_text, discarded_text=discarded_text,
                    interrupted=interrupted, partial=partial, n_generated=len(full_ids),
                    n_heard=crop_rel, crop_seq_len=keep, fragments=fragments)

    def user_turn(self, user_text: str, barge_in_fraction: Optional[float] = None) -> TurnResult:
        """
        走一轮对话。barge_in_fraction: None=完整听完；0..1=在该播放比例处打断。
        """
        self.turn_id += 1
        if not self._started:
            kv = self.llm.cache_prompt(user_text, is_end=True, system_prompt=self.system_prompt)
            self.acc = self.llm.to_accum_cache(kv)
            self._started = True
        else:
            self.llm.prefill_user_text(self.acc, user_text)
            self.llm.open_assistant_role(self.acc)

        r = self._assistant_turn(barge_in_fraction)
        logger.info(f"[turn {self.turn_id}] user={user_text!r} "
                    f"interrupted={r['interrupted']} heard {r['n_heard']}/{r['n_generated']} tokens")
        return TurnResult(
            turn_id=self.turn_id, user_text=user_text,
            full_assistant_text=r['full_text'], heard_text=r['heard_text'],
            discarded_text=r['discarded_text'], interrupted=r['interrupted'], partial=r['partial'],
            n_generated_tokens=r['n_generated'], n_heard_tokens=r['n_heard'],
            crop_seq_len=r['crop_seq_len'], fragments=r['fragments'],
        )

    def assert_kv_consistent(self) -> bool:
        """seq_length == attention_mask 长度 == DynamicCache 长度。"""
        a = self.acc
        return a.seq_length == a.attention_mask.shape[1] == a.past_key_values.get_seq_length()
