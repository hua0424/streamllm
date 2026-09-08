# src/llm/stream_llm_inference.py
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, DynamicCache
import torch
import time
import traceback
import logging
from typing import Generator, Tuple, Dict, Any, List, Optional, Sequence
from threading import Thread
import queue
from dataclasses import dataclass
from enum import Enum, auto

# 从配置导入
from src.config import LLM_MODEL_NAME, DEVICE, HF_HOME, HF_ENDPOINT, HF_TOKEN
from src.utils.logging_utils import get_logger # 导入 logger

logger = get_logger(__name__)

from dotenv import load_dotenv
load_dotenv()  # 加载 .env 文件中的环境变量

class StreamLLMInference:
    class TimingEventType(Enum):
        """
        时间事件类型枚举类
        """
        START_FUNCTION = auto() # 函数调用开始时间
        END_FUNCTION = auto() # 函数调用结束时间
        START_KV_CACHE = auto() # KV缓存计算起始时间
        END_KV_CACHE = auto() # KV缓存计算结束时间
        START_INFERENCE = auto() # 模型推理开始时间
        RETURN_LOGITS = auto() # 模型推理返回logits时间
        DECODE_TOKEN = auto() # 模型推理decode token时间

    def __init__(
        self,
        model_name=LLM_MODEL_NAME,
        device=DEVICE,
        hf_home=HF_HOME,
        hf_endpoint=HF_ENDPOINT,
        hf_token=HF_TOKEN,
        eval_mode=True
    ):
        """
        初始化流式LLM推理引擎。

        Args:
            model_name (str): LLM模型名称或路径。
            device (str): 推理设备 ("cuda" or "cpu")。
            hf_home (str, optional): Hugging Face缓存目录。
            hf_endpoint (str, optional): Hugging Face 端点。
            hf_token (str, optional): Hugging Face API Token.
        """
        # 设备标准化，支持 cuda:0/cuda:1
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        device = device.lower()
        logger.info(f"Loading LLM model {model_name} on {device}")
        logger.debug(f"HF_HOME: {hf_home}, HF_ENDPOINT: {hf_endpoint}")
        self.device = device
        
        # 优先从本地缓存加载（支持离线环境），如果失败再尝试在线下载
        try:
            # 先尝试本地加载
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                cache_dir=hf_home, 
                token=hf_token,
                trust_remote_code=True,
                local_files_only=True  # 仅从本地加载，不访问网络
            )
            logger.debug("Tokenizer 从本地缓存加载成功")
        except Exception as local_e:
            # 本地加载失败，尝试在线下载
            logger.warning(f"Tokenizer 本地加载失败 ({local_e})，尝试从网络下载...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name, 
                    cache_dir=hf_home, 
                    token=hf_token,
                    trust_remote_code=True,
                    local_files_only=False
                )
                logger.info("Tokenizer 从网络下载成功")
            except Exception as e:
                raise RuntimeError(f"无法加载tokenizer: {e}")
        
        try:
            device_map = "auto" if device == "auto" else {"": device}
            # 先尝试本地加载
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map=device_map,
                cache_dir=hf_home,
                token=hf_token,
                trust_remote_code=True,
                local_files_only=True  # 仅从本地加载，不访问网络
            )
            logger.debug("LLM 模型从本地缓存加载成功")
        except Exception as local_e:
            # 本地加载失败，尝试在线下载
            logger.warning(f"LLM 模型本地加载失败 ({local_e})，尝试从网络下载...")
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype="auto",
                    device_map=device_map,
                    cache_dir=hf_home,
                    token=hf_token,
                    trust_remote_code=True,
                    local_files_only=False
                )
                logger.info("LLM 模型从网络下载成功")
            except Exception as e:
                raise RuntimeError(f"无法加载模型: {e}")
        logger.info("LLM模型加载完成。")       

        self.model.eval() # 模型设置为推理模式

        self.eval_mode = eval_mode

        # 提取生成提示符
        # 为了获取正确的生成提示符，我们使用一个临时的messages
        init_user_text = "提取提示符"
        temp_messages = [
            {"role": "system", "content": "You are a helpful assistant responding in Chinese."},
            {"role": "user", "content": "提取提示符"}  # 临时内容
        ]
        
        # 获取带生成提示符的完整模板
        full_template = self.tokenizer.apply_chat_template(
            temp_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        index = full_template.find(init_user_text)
        self.generation_prompt = full_template[index + len(init_user_text):]
        logger.debug(f"generator_text:{self.generation_prompt}")

        # 二期角色切换只信任 tokenizer 的规范 token 模板。字符串字段仅保留给
        # 旧实验代码做可读日志/兼容，实际 prefill 一律走已验证 token-ID delta。
        self._init_p2_role_transitions(
            system_prompt="You are a helpful assistant responding in Chinese."
        )
        logger.debug(f"role_switch_to_user:{self._role_switch_to_user!r}")

        # 用于记录详细延迟的变量
        self.timing_events:Dict[StreamLLMInference.TimingEventType, float] = {}


    class RolePhase(Enum):
        USER_OPEN = "user_open"
        ASSISTANT_OPEN = "assistant_open"
        ASSISTANT_EOT_PENDING = "assistant_eot_pending"

    class GenerationEndReason(Enum):
        NONE = "none"
        EOS = "eos"
        MAX_TOKENS = "max_tokens"
        CONSUMER_STOP = "consumer_stop"
        CROPPED = "cropped"

    @dataclass(frozen=True)
    class RoleBoundary:
        role_header_start: int
        content_start: int
        content_end: Optional[int] = None
        role_end: Optional[int] = None
        next_user_content_start: Optional[int] = None
        end_reason: Optional["StreamLLMInference.GenerationEndReason"] = None

    class KVCache:
        def __init__(self, past_key_values:torch.Tensor, pre_input_ids:torch.Tensor, pre_attention_mask:torch.Tensor, next_token_logits: torch.Tensor = None, token_ids=None):
            self.past_key_values = past_key_values
            self.pre_input_ids = pre_input_ids
            self.pre_attention_mask = pre_attention_mask
            self.next_token_logits = next_token_logits # 新增：保存最后的logits
            # 二期包装需要完整前缀账本；一期调用方不读取该字段。
            if token_ids is None:
                token_ids = pre_input_ids[0].tolist() if pre_input_ids is not None else []
            self.token_ids = [int(token_id) for token_id in token_ids]

    class AccumKVCache:
        """二期持久 KV：完整 token 账本 + role phase + assistant 内容边界。"""
        def __init__(self, past_key_values, attention_mask, next_token_logits,
                     seq_length: int, assistant_start: Optional[int] = None,
                     assistant_token_ids=None, token_ids=None, role_phase=None,
                     assistant_role_start: Optional[int] = None,
                     assistant_content_start: Optional[int] = None,
                     assistant_content_end: Optional[int] = None,
                     assistant_role_end: Optional[int] = None,
                     generation_end_reason=None, role_boundaries=None):
            self.past_key_values = past_key_values
            self.attention_mask = attention_mask
            self.next_token_logits = next_token_logits
            self.seq_length = int(seq_length)
            self.token_ids = [int(token_id) for token_id in (token_ids or [])]
            content_start = assistant_content_start
            if content_start is None:
                content_start = assistant_start
            self.assistant_role_start = assistant_role_start
            self.assistant_content_start = content_start
            self.assistant_content_end = assistant_content_end
            self.assistant_role_end = assistant_role_end
            self.assistant_token_ids = [
                int(token_id) for token_id in (assistant_token_ids or [])
            ]
            self.role_phase = role_phase or StreamLLMInference.RolePhase.USER_OPEN
            self.generation_end_reason = (
                generation_end_reason or StreamLLMInference.GenerationEndReason.NONE
            )
            self.role_boundaries = list(role_boundaries or [])

        @property
        def assistant_start(self) -> int:
            """旧 timeline/runtime 名称的兼容别名：始终指 assistant 内容起点。"""
            if self.assistant_content_start is None:
                return self.seq_length
            return self.assistant_content_start

        @assistant_start.setter
        def assistant_start(self, value: int) -> None:
            # 旧调用方只能在 user-open 空占位状态写相同的当前位置；其余 fail closed。
            value = int(value)
            if self.role_phase != StreamLLMInference.RolePhase.USER_OPEN or value != self.seq_length:
                raise RuntimeError("assistant_start 由 role API 管理，禁止手工修改")
            self.assistant_content_start = None
            self.assistant_role_start = None
            self.assistant_content_end = None
            self.assistant_role_end = None
            self.assistant_token_ids = []

        @property
        def end_reason(self):
            return self.generation_end_reason

        @property
        def assistant_boundary(self):
            if self.assistant_role_start is None or self.assistant_content_start is None:
                return None
            return StreamLLMInference.RoleBoundary(
                role_header_start=self.assistant_role_start,
                content_start=self.assistant_content_start,
                content_end=self.assistant_content_end,
                role_end=self.assistant_role_end,
                end_reason=self.generation_end_reason,
            )

    @staticmethod
    def _token_list(value) -> List[int]:
        if hasattr(value, "tolist"):
            value = value.tolist()
        if value and isinstance(value[0], list):
            if len(value) != 1:
                raise RuntimeError("二期 KV 状态机仅支持 batch size 1")
            value = value[0]
        return [int(token_id) for token_id in value]

    def _apply_chat_template_ids(self, messages, *, add_generation_prompt: bool) -> List[int]:
        token_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )
        return self._token_list(token_ids)

    @staticmethod
    def _require_prefix(prefix: Sequence[int], full: Sequence[int], label: str) -> List[int]:
        prefix = list(prefix)
        full = list(full)
        if len(full) <= len(prefix) or full[:len(prefix)] != prefix:
            raise RuntimeError(
                f"chat_template 的 {label} 不是可追加 token delta；二期 role API fail closed"
            )
        return full[len(prefix):]

    def _init_p2_role_transitions(self, system_prompt: str) -> None:
        """用 tokenize=True 的规范模板提取并交叉验证 role transition token。"""
        user_text = "P2_USER_SENTINEL_7f4a"
        assistant_text = "P2_ASSISTANT_SENTINEL_9c2e"
        next_user_text = "P2_NEXT_USER_SENTINEL_3d8b"
        user_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        assistant_open = self._apply_chat_template_ids(
            user_messages, add_generation_prompt=True
        )
        user_text_ids = self._token_list(
            self.tokenizer(user_text, add_special_tokens=False).input_ids
        )
        user_content_at = len(assistant_open) - len(user_text_ids)
        while user_content_at >= 0 and (
            assistant_open[user_content_at:user_content_at + len(user_text_ids)]
            != user_text_ids
        ):
            user_content_at -= 1
        if user_content_at < 0:
            raise RuntimeError("chat_template 中无法定位 user 内容 token")
        user_open = assistant_open[:user_content_at + len(user_text_ids)]
        user_to_assistant = self._require_prefix(
            user_open, assistant_open, "user→assistant transition"
        )

        assistant_messages = user_messages + [
            {"role": "assistant", "content": assistant_text}
        ]
        assistant_closed = self._apply_chat_template_ids(
            assistant_messages, add_generation_prompt=False
        )
        assistant_text_ids = self._token_list(
            self.tokenizer(assistant_text, add_special_tokens=False).input_ids
        )
        assistant_content_prefix = list(assistant_open) + assistant_text_ids
        assistant_close = self._require_prefix(
            assistant_content_prefix, assistant_closed, "assistant close transition"
        )
        if not assistant_close:
            raise RuntimeError("chat_template 未提供 assistant EOT token")
        eot_id = self.tokenizer.eos_token_id
        if eot_id is None or int(eot_id) not in assistant_close:
            raise RuntimeError("tokenizer eos_token_id 不在规范 assistant close transition 中")
        if assistant_close.count(int(eot_id)) != 1:
            raise RuntimeError("规范 assistant close transition 必须且只能含一个 EOT")

        assistant_with_user = self._apply_chat_template_ids(
            assistant_messages + [{"role": "user", "content": next_user_text}],
            add_generation_prompt=True,
        )
        next_user_ids = self._token_list(
            self.tokenizer(next_user_text, add_special_tokens=False).input_ids
        )
        next_user_at = len(assistant_with_user) - len(next_user_ids)
        while next_user_at >= len(assistant_content_prefix) and (
            assistant_with_user[next_user_at:next_user_at + len(next_user_ids)]
            != next_user_ids
        ):
            next_user_at -= 1
        if next_user_at < len(assistant_content_prefix):
            raise RuntimeError("chat_template 中无法定位下一 user 内容 token")
        assistant_with_user_open = assistant_with_user[:next_user_at]
        full_transition = self._require_prefix(
            assistant_content_prefix, assistant_with_user_open,
            "assistant→user transition",
        )
        if full_transition[:len(assistant_close)] != assistant_close:
            raise RuntimeError("assistant→user transition 未以规范 EOT 开始")
        assistant_to_user = full_transition[len(assistant_close):]
        if not assistant_to_user:
            raise RuntimeError("chat_template 未提供 user role header")
        if list(assistant_close) + list(assistant_to_user) != full_transition:
            raise RuntimeError("chat_template assistant→user transition 无法分解")

        self._user_to_assistant_ids = user_to_assistant
        self._assistant_close_ids = assistant_close
        self._assistant_to_user_header_ids = assistant_to_user
        self._assistant_to_user_ids = full_transition
        self._assistant_eot_id = int(eot_id)
        self._role_switch_to_user = self.tokenizer.decode(
            full_transition, skip_special_tokens=False
        )

        transition_text = self.tokenizer.decode(
            user_to_assistant, skip_special_tokens=False
        )
        if self._token_list(
            self.tokenizer(transition_text, add_special_tokens=False).input_ids
        ) != user_to_assistant:
            raise RuntimeError("规范 user→assistant transition 无法 token-ID 往返")

    def _assert_accum_consistent(self, cache: "StreamLLMInference.AccumKVCache") -> None:
        kv_len = int(cache.past_key_values.get_seq_length())
        mask_len = int(cache.attention_mask.shape[1])
        if not (len(cache.token_ids) == cache.seq_length == mask_len == kv_len):
            raise AssertionError(
                "AccumKVCache 长度不一致: "
                f"ledger={len(cache.token_ids)}, seq={cache.seq_length}, "
                f"mask={mask_len}, kv={kv_len}"
            )
        phase = cache.role_phase
        if phase == self.RolePhase.USER_OPEN:
            if cache.assistant_token_ids:
                raise AssertionError("USER_OPEN 不得持有当前 assistant 内容账本")
            return
        if cache.assistant_role_start is None or cache.assistant_content_start is None:
            raise AssertionError("assistant phase 缺少 role/content 边界")
        start = cache.assistant_content_start
        end = start + len(cache.assistant_token_ids)
        if cache.token_ids[start:end] != cache.assistant_token_ids:
            raise AssertionError("assistant 内容 span 与 assistant_token_ids 不一致")
        if phase == self.RolePhase.ASSISTANT_OPEN:
            if cache.seq_length != end or cache.assistant_content_end is not None:
                raise AssertionError("ASSISTANT_OPEN 的内容必须位于 ledger 尾部")
        elif phase == self.RolePhase.ASSISTANT_EOT_PENDING:
            if cache.seq_length != end or cache.assistant_content_end != end:
                raise AssertionError("EOT pending 不得把结构 EOT 写入 ledger/KV")
        else:
            raise AssertionError(f"未知 role phase: {phase}")

    def get_last_timings(self):
        return self.timing_events

    def reset_timings(self):
        self.timing_events.clear()
    
    def cache_prompt(self, prompt:str, pre_cache:KVCache | None = None, is_end:bool = False, system_prompt:str = "You are a helpful assistant responding in Chinese.") -> KVCache:
        """
        对prompt计算缓存，返回缓存计算的中间值
        pre_cache传入上一次返回的缓存值，首次可不传
        """
        self.reset_timings()
        self.timing_events[self.TimingEventType.START_FUNCTION] = time.perf_counter()
        if pre_cache is None:
            # 首次计算
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": ""}  # 临时内容
            ]
            full_prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            # 截取prompt部分，便于后续流式添加prompt
            init_prompt_text = full_prompt_text.replace(self.generation_prompt, "") + prompt
            logger.debug(f"init prompt text:{init_prompt_text}")

            if is_end:
                init_prompt_text+=self.generation_prompt    
            result = self._init_kv_cache(init_prompt_text)
        
        else:
            # 非首次，只追加新文本即可
            logger.debug(f"流式添加提示词: {prompt}")
            if is_end:
                prompt += self.generation_prompt
            result = self._add_stream_prompt(pre_cache, prompt)

        self.timing_events[self.TimingEventType.END_FUNCTION] = time.perf_counter()
        return result
    
    def generate(self, pre_cache:KVCache | None, max_new_tokens=50, temperature=0.1, top_p=0.9, repetition_penalty=1.1) -> Generator[str, None, None]:
        self.reset_timings()
        if pre_cache is None:
            raise Exception("未进行kv缓存初始化")
        
        self.timing_events[self.TimingEventType.START_FUNCTION] = time.perf_counter()
        
        # 准备初始状态
        past_key_values = pre_cache.past_key_values
        gen_attention_mask = pre_cache.pre_attention_mask
        # 直接使用预处理阶段留下的 logits，无需再次 forward
        next_token_logits = pre_cache.next_token_logits 

        for i in range(max_new_tokens):
            # 1. 解码当前 Logits (解码上一步的结果)
            self.timing_events[self.TimingEventType.RETURN_LOGITS] = time.perf_counter()
            next_token_id = self._decode_logits(next_token_logits, temperature, top_p, repetition_penalty)
            
            # 检查是否是EOS token
            is_eos = next_token_id.item() == self.tokenizer.eos_token_id
            generated_token_text = self.tokenizer.decode(next_token_id[0], skip_special_tokens=True) 
            self.timing_events[self.TimingEventType.DECODE_TOKEN] = time.perf_counter()
            
            yield generated_token_text

            if self.eval_mode or is_eos:
                break

            # 2. 准备下一步推理的输入
            gen_input_ids = next_token_id
            gen_attention_mask = torch.cat(
                [gen_attention_mask, torch.ones(next_token_id.shape, device=self.device)], 
                dim=-1
            )

            # 3. 执行模型推理 (为下一次循环计算 Logits)
            self.timing_events[self.TimingEventType.START_INFERENCE] = time.perf_counter()
            with torch.no_grad():
                outputs = self.model(
                    input_ids=gen_input_ids,
                    attention_mask=gen_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True
                )
            
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :] # 获取新的 Logits
            self.timing_events[self.TimingEventType.END_FUNCTION] = time.perf_counter()

        return None

    # ===================================================================
    # 二期（bargeincache）新增：assistant-side KV 累积 + 播放感知 crop + role 重建
    # 一期的 generate()/cache_prompt()/once_add_and_generate() 保持不动（一期可复现）
    # ===================================================================

    def _as_dynamic_cache(self, pkv) -> DynamicCache:
        """把 past_key_values 统一为 DynamicCache（legacy tuple 则转换）。见 D-001。"""
        if isinstance(pkv, DynamicCache):
            return pkv
        return DynamicCache.from_legacy_cache(pkv)

    def to_accum_cache(self, pre_cache: "StreamLLMInference.KVCache") -> "StreamLLMInference.AccumKVCache":
        """把一期 KV 包装为二期完整账本，并识别 user-open/assistant-open 状态。"""
        seq_len = int(pre_cache.pre_attention_mask.shape[1])
        token_ids = list(pre_cache.token_ids)
        if len(token_ids) != seq_len:
            raise RuntimeError("一期 KVCache 未携带完整 token ledger")
        if len(token_ids) >= len(self._assistant_to_user_ids) and (
            token_ids[-len(self._assistant_to_user_ids):] == self._assistant_to_user_ids
        ):
            raise RuntimeError("to_accum_cache 不接受已手工拼接 assistant→user transition 的状态")
        assistant_open = bool(self._user_to_assistant_ids) and (
            len(token_ids) >= len(self._user_to_assistant_ids)
            and token_ids[-len(self._user_to_assistant_ids):] == self._user_to_assistant_ids
        )
        content_start = seq_len if assistant_open else None
        role_start = seq_len - len(self._user_to_assistant_ids) if assistant_open else None
        cache = self.AccumKVCache(
            past_key_values=self._as_dynamic_cache(pre_cache.past_key_values),
            attention_mask=pre_cache.pre_attention_mask,
            next_token_logits=pre_cache.next_token_logits,
            seq_length=seq_len,
            token_ids=token_ids,
            role_phase=(self.RolePhase.ASSISTANT_OPEN if assistant_open
                        else self.RolePhase.USER_OPEN),
            assistant_role_start=role_start,
            assistant_content_start=content_start,
            assistant_token_ids=[],
        )
        self._assert_accum_consistent(cache)
        return cache

    def generate_accumulating(self, cache: "StreamLLMInference.AccumKVCache",
                              max_new_tokens=50, temperature=0.1, top_p=0.9,
                              repetition_penalty=1.1, on_token_decoded=None):
        """生成 assistant 内容；EOT 仅标为 pending，不进入内容 ledger 或 KV。"""
        if cache is None or cache.next_token_logits is None:
            raise RuntimeError("AccumKVCache 未初始化或缺少起始 logits")
        if cache.role_phase != self.RolePhase.ASSISTANT_OPEN:
            raise RuntimeError(f"generate 需要 ASSISTANT_OPEN，当前为 {cache.role_phase.value}")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens 不能为负")
        cache.generation_end_reason = self.GenerationEndReason.NONE
        next_token_logits = cache.next_token_logits
        completed = False
        try:
            for _ in range(max_new_tokens):
                next_token_id = self._decode_logits(
                    next_token_logits, temperature, top_p, repetition_penalty
                )
                token_id = int(next_token_id.item())
                rel_idx = len(cache.assistant_token_ids)
                token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
                if on_token_decoded is not None:
                    # 回调表示 token selection/compute readiness，而非内容 yield；
                    # EOT 也要打点，但仍不得进入 assistant 内容流或 KV。
                    on_token_decoded(token_text, rel_idx, token_id)
                if token_id == self._assistant_eot_id:
                    cache.role_phase = self.RolePhase.ASSISTANT_EOT_PENDING
                    cache.assistant_content_end = cache.seq_length
                    cache.generation_end_reason = self.GenerationEndReason.EOS
                    completed = True
                    self._assert_accum_consistent(cache)
                    return
                self._prefill_ids_p2(cache, [token_id])
                cache.assistant_token_ids.append(token_id)
                next_token_logits = cache.next_token_logits
                self._assert_accum_consistent(cache)
                yield token_text, rel_idx

            cache.generation_end_reason = self.GenerationEndReason.MAX_TOKENS
            completed = True
            self._assert_accum_consistent(cache)
        finally:
            if not completed and cache.role_phase == self.RolePhase.ASSISTANT_OPEN:
                cache.generation_end_reason = self.GenerationEndReason.CONSUMER_STOP

    def crop_to_token(self, cache: "StreamLLMInference.AccumKVCache",
                      keep_seq_len: int) -> "StreamLLMInference.AccumKVCache":
        """裁剪 KV，并从完整 token ledger 恢复可解释的 role/end 状态。"""
        self._assert_accum_consistent(cache)
        keep_seq_len = int(keep_seq_len)
        if not (0 <= keep_seq_len <= cache.seq_length):
            raise ValueError(f"keep_seq_len {keep_seq_len} 越界 (0..{cache.seq_length})")
        if keep_seq_len == cache.seq_length:
            # 真正的 no-op：保留 natural EOS 的 pending-close 与 end reason。
            # 播放到回复末端后，reopen_user_role() 仍需按该状态唯一提交 EOT。
            self._assert_accum_consistent(cache)
            return cache

        role_start = cache.assistant_role_start
        content_start = cache.assistant_content_start
        structural_spans = []
        for boundary in cache.role_boundaries:
            if boundary.content_end is not None and boundary.role_end is not None:
                structural_spans.append((boundary.content_end, boundary.role_end))
            if (boundary.role_end is not None
                    and boundary.next_user_content_start is not None):
                structural_spans.append(
                    (boundary.role_end, boundary.next_user_content_start)
                )
        if role_start is not None:
            structural_spans.append((role_start, content_start))
        for span_start, span_end in structural_spans:
            if span_start < keep_seq_len < span_end:
                raise ValueError("crop 不能落在 role 结构 token 中间")

        cache.past_key_values.crop(keep_seq_len)
        cache.attention_mask = cache.attention_mask[:, :keep_seq_len]
        cache.token_ids = cache.token_ids[:keep_seq_len]
        cache.seq_length = keep_seq_len
        cache.next_token_logits = None
        cache.generation_end_reason = self.GenerationEndReason.CROPPED
        cache.role_boundaries = [
            boundary for boundary in cache.role_boundaries
            if boundary.role_end is not None and boundary.role_end <= keep_seq_len
        ]

        if role_start is None or keep_seq_len <= role_start:
            if role_start is not None and keep_seq_len == role_start:
                cache.role_phase = self.RolePhase.USER_OPEN
                cache.assistant_role_start = None
                cache.assistant_content_start = None
                cache.assistant_content_end = None
                cache.assistant_role_end = None
                cache.assistant_token_ids = []
            else:
                cache.role_phase = self.RolePhase.USER_OPEN
                cache.assistant_role_start = None
                cache.assistant_content_start = None
                cache.assistant_content_end = None
                cache.assistant_role_end = None
                cache.assistant_token_ids = []
        else:
            if keep_seq_len < content_start:
                raise AssertionError("不可解释的 assistant header crop")
            kept = keep_seq_len - content_start
            cache.role_phase = self.RolePhase.ASSISTANT_OPEN
            cache.assistant_token_ids = cache.assistant_token_ids[:kept]
            cache.assistant_content_end = None
            cache.assistant_role_end = None
        self._assert_accum_consistent(cache)
        logger.debug(
            f"[p2] crop → seq_length={cache.seq_length}, "
            f"role={cache.role_phase.value}, kept_assistant={len(cache.assistant_token_ids)}"
        )
        return cache

    def _prefill_ids_p2(self, cache: "StreamLLMInference.AccumKVCache",
                        token_ids: Sequence[int]) -> "StreamLLMInference.AccumKVCache":
        """二期唯一 KV 追加核心：直接 prefill token IDs 并维护完整 ledger。"""
        ids_list = [int(token_id) for token_id in token_ids]
        if not ids_list:
            raise ValueError("prefill token IDs 为空")
        ids = torch.tensor([ids_list], dtype=torch.long, device=self.device)
        attn = torch.cat(
            [cache.attention_mask,
             torch.ones(ids.shape, device=self.device, dtype=cache.attention_mask.dtype)],
            dim=-1,
        )
        position_ids = torch.arange(
            cache.seq_length, cache.seq_length + len(ids_list),
            dtype=torch.long, device=self.device,
        ).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(
                input_ids=ids,
                attention_mask=attn,
                position_ids=position_ids,
                past_key_values=cache.past_key_values,
                use_cache=True,
                return_dict=True,
            )
        cache.past_key_values = self._as_dynamic_cache(outputs.past_key_values)
        cache.attention_mask = attn
        cache.token_ids.extend(ids_list)
        cache.seq_length += len(ids_list)
        cache.next_token_logits = outputs.logits[:, -1, :]
        return cache

    def _prefill_text_p2(self, cache: "StreamLLMInference.AccumKVCache",
                         text: str) -> "StreamLLMInference.AccumKVCache":
        if not text:
            raise ValueError("prefill 文本为空")
        token_ids = self._token_list(
            self.tokenizer(text, add_special_tokens=False).input_ids
        )
        return self._prefill_ids_p2(cache, token_ids)

    def reopen_user_role(self, cache: "StreamLLMInference.AccumKVCache") -> "StreamLLMInference.AccumKVCache":
        """唯一提交 pending assistant EOT 的操作，并打开下一 user role。"""
        if cache.role_phase not in (
            self.RolePhase.ASSISTANT_OPEN, self.RolePhase.ASSISTANT_EOT_PENDING
        ):
            raise RuntimeError(f"reopen_user_role 需要 assistant phase，当前为 {cache.role_phase.value}")
        content_end = cache.seq_length
        role_end = content_end + len(self._assistant_close_ids)
        self._prefill_ids_p2(cache, self._assistant_to_user_ids)
        cache.role_boundaries.append(self.RoleBoundary(
            role_header_start=cache.assistant_role_start,
            content_start=cache.assistant_content_start,
            content_end=content_end,
            role_end=role_end,
            next_user_content_start=content_end + len(self._assistant_to_user_ids),
            end_reason=cache.generation_end_reason,
        ))
        cache.assistant_content_end = content_end
        cache.assistant_role_end = role_end
        cache.role_phase = self.RolePhase.USER_OPEN
        cache.assistant_role_start = None
        cache.assistant_content_start = None
        cache.assistant_content_end = None
        cache.assistant_role_end = None
        cache.assistant_token_ids = []
        cache.generation_end_reason = self.GenerationEndReason.NONE
        self._assert_accum_consistent(cache)
        return cache

    def open_assistant_role(self, cache: "StreamLLMInference.AccumKVCache") -> "StreamLLMInference.AccumKVCache":
        if cache.role_phase != self.RolePhase.USER_OPEN:
            raise RuntimeError(f"open_assistant_role 需要 USER_OPEN，当前为 {cache.role_phase.value}")
        role_start = cache.seq_length
        self._prefill_ids_p2(cache, self._user_to_assistant_ids)
        cache.role_phase = self.RolePhase.ASSISTANT_OPEN
        cache.assistant_role_start = role_start
        cache.assistant_content_start = cache.seq_length
        cache.assistant_content_end = None
        cache.assistant_role_end = None
        cache.assistant_token_ids = []
        cache.generation_end_reason = self.GenerationEndReason.NONE
        self._assert_accum_consistent(cache)
        return cache

    def prefill_user_text(self, cache: "StreamLLMInference.AccumKVCache", text: str) -> "StreamLLMInference.AccumKVCache":
        if cache.role_phase != self.RolePhase.USER_OPEN:
            raise RuntimeError(f"prefill_user_text 需要 USER_OPEN，当前为 {cache.role_phase.value}")
        self._prefill_text_p2(cache, text)
        cache.generation_end_reason = self.GenerationEndReason.NONE
        self._assert_accum_consistent(cache)
        return cache

    def prefill_assistant_text(self, cache: "StreamLLMInference.AccumKVCache", text: str) -> "StreamLLMInference.AccumKVCache":
        if cache.role_phase != self.RolePhase.ASSISTANT_OPEN:
            raise RuntimeError(f"prefill_assistant_text 需要 ASSISTANT_OPEN，当前为 {cache.role_phase.value}")
        token_ids = self._token_list(
            self.tokenizer(text, add_special_tokens=False).input_ids
        )
        self._prefill_ids_p2(cache, token_ids)
        cache.assistant_token_ids.extend(token_ids)
        cache.generation_end_reason = self.GenerationEndReason.NONE
        self._assert_accum_consistent(cache)
        return cache

    def _init_kv_cache(self, prompt_text) -> KVCache:
        """
        使用初始化prompt进行KV缓存首次计算
        Returns:
            tuple: (past_key_values, input_ids, attention_mask)
        """
        self.timing_events[self.TimingEventType.START_KV_CACHE] = time.perf_counter()
        model_inputs = self.tokenizer([prompt_text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(
                input_ids=model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                use_cache=True,
                return_dict=True
            )
        
        self.timing_events[self.TimingEventType.END_KV_CACHE] = time.perf_counter()
        return self.KVCache(
            outputs.past_key_values, model_inputs.input_ids,
            model_inputs.attention_mask, outputs.logits[:, -1, :],
            token_ids=model_inputs.input_ids[0].tolist(),
        )

    def _add_stream_prompt(self, pre_cache:KVCache, text_fragments) -> KVCache:
        """
        流式添加提示词，并更新KV缓存。
        """
        # 流式添加提示词，并更新KV缓存。
        # 返回新的past_key_values, current_attention_mask
        # 处理新的文本片段（如果存在）
        if text_fragments == None or len(text_fragments) == 0:
            raise Exception("要添加的文本为空！")
        self.timing_events[self.TimingEventType.START_KV_CACHE] = time.perf_counter()
        new_fragment_inputs = self.tokenizer(text_fragments, return_tensors="pt", add_special_tokens=False).to(self.device)
        new_fragment_ids = new_fragment_inputs.input_ids
        
        if new_fragment_ids.shape[1] == 0: # 如果新片段没有有效token
            raise Exception("新片段没有有效token！")
        logger.debug(f"处理新的文本片段，token数量: {new_fragment_ids.shape[1]}")
        
        # attention_mask需要小心处理，因为它需要覆盖整个序列的长度
        # 包括缓存的部分和新的部分
        attention_mask = torch.cat(
            [pre_cache.pre_attention_mask, torch.ones(new_fragment_ids.shape, device=self.device)], 
            dim=-1
        )

        # 显式计算 position_ids，防止位置编码错乱
        past_length = pre_cache.pre_attention_mask.shape[1]
        current_length = new_fragment_ids.shape[1]
        position_ids = torch.arange(past_length, past_length + current_length, dtype=torch.long, device=self.device)
        position_ids = position_ids.unsqueeze(0) # 增加 batch 维度

        with torch.no_grad():
            outputs = self.model(
                input_ids=new_fragment_ids,
                past_key_values=pre_cache.past_key_values,
                attention_mask=attention_mask, # 传入拼接后的完整 attention mask
                position_ids=position_ids, # 传入 position_ids
                use_cache=True,
                return_dict=True
                )
        self.timing_events[self.TimingEventType.END_KV_CACHE] = time.perf_counter()
        prior_ids = getattr(pre_cache, "token_ids", [])
        full_ids = list(prior_ids) + new_fragment_ids[0].tolist()
        return self.KVCache(
            outputs.past_key_values, new_fragment_ids, attention_mask,
            outputs.logits[:, -1, :], token_ids=full_ids,
        )

    def _decode_logits(self, logits, temperature, top_p, repetition_penalty):
        """
        根据温度、top_p和重复惩罚系数解码logits。
        """
        # 应用温度和top_p采样
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                probs[indices_to_remove] = 0
            next_token_id = torch.multinomial(probs, num_samples=1)
        else: # greedy
            next_token_id = torch.argmax(logits, dim=-1).unsqueeze(-1)

        return next_token_id

    def once_add_and_generate(self, prompt:str, system_prompt:str="You are a helpful assistant responding in Chinese.", max_new_tokens=50, temperature=0.1, top_p=0.9, repetition_penalty=1.1) -> Generator[str, None, None]:
        """
        一次性添加提示词并生成token。
        """
        self.timing_events[self.TimingEventType.START_FUNCTION] = time.perf_counter()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        logger.info(f"prompt_text: {prompt_text}")
        model_inputs = self.tokenizer([prompt_text], return_tensors="pt", padding=True).to(self.device)
        gen_input_ids = model_inputs.input_ids
        gen_attention_mask = model_inputs.attention_mask

        past_key_values = None  
        for i in range(max_new_tokens):
            self.timing_events[self.TimingEventType.START_INFERENCE] = time.perf_counter()
            # 使用模型生成下一个token
            with torch.no_grad():
                outputs = self.model(
                    input_ids=gen_input_ids,
                    attention_mask=gen_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True
                )
            
            past_key_values = outputs.past_key_values

            # 获取logits并生成下一个token
            next_token_logits = outputs.logits[:, -1, :]
            self.timing_events[self.TimingEventType.RETURN_LOGITS] = time.perf_counter()
            next_token_id = self._decode_logits(next_token_logits, temperature, top_p, repetition_penalty)
            # 检查是否是EOS token
            is_eos = next_token_id.item() == self.tokenizer.eos_token_id
            # 解码生成的token
            generated_token_text = self.tokenizer.decode(next_token_id[0], skip_special_tokens=True) 
            self.timing_events[self.TimingEventType.DECODE_TOKEN] = time.perf_counter()
           
            # 更新输入ID序列
            gen_input_ids = next_token_id
            
            # 更新attention_mask
            # attention_mask需要小心处理，因为它需要覆盖整个序列的长度
            # 包括缓存的部分和新的部分
            gen_attention_mask = torch.cat(
                [gen_attention_mask, torch.ones(next_token_id.shape, device=self.device)], 
                dim=-1
            )
            self.timing_events[self.TimingEventType.END_FUNCTION] = time.perf_counter()
            yield generated_token_text

            # 如果生成结束，则返回
            if self.eval_mode or is_eos:
                break

        return None

    def _log_kv_cache_size(self, past_key_values):
        if past_key_values is None:
            logger.debug("KV缓存为空。")
            return
        total_size_bytes = 0
        num_elements = 0
        for layer_past in past_key_values:
            for tensor in layer_past: # key 和 value tensor
                total_size_bytes += tensor.element_size() * tensor.nelement()
                num_elements += tensor.nelement()
        logger.debug(
            f"KV缓存状态: {len(past_key_values)}层, "
            f"总元素数量: {num_elements}, "
            f"预估大小: {total_size_bytes / (1024 * 1024):.2f} MB"
        )
