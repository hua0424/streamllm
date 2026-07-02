# src/llm/stream_llm_inference.py
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, DynamicCache
import torch
import time
import traceback
import logging
from typing import Generator, Tuple, Dict, Any, List
from threading import Thread
import queue
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

        # 二期（bargeincache）：从 generation_prompt 派生"关闭 assistant + 打开 user"的角色切换串。
        # generation_prompt 形如 "<|im_end|>\n<|im_start|>assistant\n"（ChatML）。
        assert self.generation_prompt.rstrip().endswith("assistant"), \
            f"unexpected generation_prompt (need ChatML assistant open): {self.generation_prompt!r}"
        self._role_switch_to_user = self.generation_prompt.replace("assistant\n", "user\n")
        logger.debug(f"role_switch_to_user:{self._role_switch_to_user!r}")

        # 用于记录详细延迟的变量
        self.timing_events:Dict[StreamLLMInference.TimingEventType, float] = {}


    class KVCache:
        def __init__(self, past_key_values:torch.Tensor, pre_input_ids:torch.Tensor, pre_attention_mask:torch.Tensor, next_token_logits: torch.Tensor = None):
            self.past_key_values = past_key_values
            self.pre_input_ids = pre_input_ids
            self.pre_attention_mask = pre_attention_mask
            self.next_token_logits = next_token_logits # 新增：保存最后的logits

    class AccumKVCache:
        """
        二期（bargeincache）：边生成边累积、可被 crop 的 assistant-side KV 容器。

        与一期 KVCache 的区别（见 docs/paper2_context.md Q4/Q5、docs/decisions.md D-001/D-008）：
        - past_key_values 显式为 DynamicCache（支持 crop）
        - 显式维护 seq_length（= attention_mask.shape[1] = past_key_values.get_seq_length()），
          crop 时三者同步，避免靠 shape 间接推断
        - 记录 assistant_start（本轮 assistant 内容起始的 KV 位置）与
          assistant_token_ids（本轮已生成的 assistant token），供反向映射与推测回滚
        """
        def __init__(self, past_key_values, attention_mask, next_token_logits,
                     seq_length: int, assistant_start: int, assistant_token_ids=None):
            self.past_key_values = past_key_values        # DynamicCache
            self.attention_mask = attention_mask          # [1, seq_length]，全 1
            self.next_token_logits = next_token_logits    # 续生成所需的最后 logits（crop 后置 None）
            self.seq_length = seq_length
            self.assistant_start = assistant_start
            self.assistant_token_ids = assistant_token_ids if assistant_token_ids is not None else []

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
        """
        把一期 cache_prompt(is_end=True) 产出的 KVCache 包装为二期 AccumKVCache。
        此时 KV 里已含 system+user+generation_prompt（assistant role 已打开），
        assistant_start = 当前序列长度（后续生成的 token 从这里开始）。
        """
        seq_len = pre_cache.pre_attention_mask.shape[1]
        return self.AccumKVCache(
            past_key_values=self._as_dynamic_cache(pre_cache.past_key_values),
            attention_mask=pre_cache.pre_attention_mask,
            next_token_logits=pre_cache.next_token_logits,
            seq_length=seq_len,
            assistant_start=seq_len,
            assistant_token_ids=[],
        )

    def generate_accumulating(self, cache: "StreamLLMInference.AccumKVCache",
                              max_new_tokens=50, temperature=0.1, top_p=0.9,
                              repetition_penalty=1.1):
        """
        二期版流式生成：边生成边把 assistant token 的 KV 写回 cache（可被 crop）。
        yield (token_text, assistant_relative_idx)。第 i 个 assistant token 占据
        KV 位置 assistant_start + i；生成后 cache.seq_length == assistant_start + 已生成数。
        打断只需消费侧停止迭代；cache 即为可被 crop 的对象。
        """
        if cache is None or cache.next_token_logits is None:
            raise Exception("AccumKVCache 未初始化或缺少起始 logits")
        next_token_logits = cache.next_token_logits
        for _ in range(max_new_tokens):
            next_token_id = self._decode_logits(next_token_logits, temperature, top_p, repetition_penalty)
            is_eos = next_token_id.item() == self.tokenizer.eos_token_id
            token_text = self.tokenizer.decode(next_token_id[0], skip_special_tokens=True)
            rel_idx = len(cache.assistant_token_ids)  # 该 token 的 assistant 相对下标

            # 显式拼 attention_mask 与 position_ids（与一期 _add_stream_prompt 同风格，Q2）
            gen_attention_mask = torch.cat(
                [cache.attention_mask,
                 torch.ones((1, 1), device=self.device, dtype=cache.attention_mask.dtype)],
                dim=-1,
            )
            position_ids = torch.tensor([[cache.seq_length]], dtype=torch.long, device=self.device)
            with torch.no_grad():
                outputs = self.model(
                    input_ids=next_token_id,
                    attention_mask=gen_attention_mask,
                    position_ids=position_ids,
                    past_key_values=cache.past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
            cache.past_key_values = outputs.past_key_values
            cache.attention_mask = gen_attention_mask
            cache.seq_length += 1
            cache.assistant_token_ids.append(int(next_token_id.item()))
            next_token_logits = outputs.logits[:, -1, :]
            cache.next_token_logits = next_token_logits

            yield token_text, rel_idx
            if is_eos:
                break

    def crop_to_token(self, cache: "StreamLLMInference.AccumKVCache",
                      keep_seq_len: int) -> "StreamLLMInference.AccumKVCache":
        """
        播放感知 KV 截断（贡献2核心）：把 KV 裁到绝对长度 keep_seq_len（保留 [0, keep_seq_len)）。
        keep_seq_len 由编排层依据 PlaybackTimeline 反查得到（= assistant_start + 听到的 assistant token 数；
        推测整段作废时 = assistant_start）。crop 后同步截短 attention_mask、更新 seq_length（Q2 陷阱）。
        crop 不重算 logits，next_token_logits 置 None——续轮会经 reopen/prefill 重新产生。
        """
        keep_seq_len = int(keep_seq_len)
        if not (0 <= keep_seq_len <= cache.seq_length):
            raise ValueError(f"keep_seq_len {keep_seq_len} 越界 (0..{cache.seq_length})")
        cache.past_key_values.crop(keep_seq_len)
        cache.attention_mask = cache.attention_mask[:, :keep_seq_len]
        cache.seq_length = keep_seq_len
        kept_assistant = max(0, keep_seq_len - cache.assistant_start)
        cache.assistant_token_ids = cache.assistant_token_ids[:kept_assistant]
        cache.next_token_logits = None
        logger.debug(f"[p2] crop → seq_length={cache.seq_length}, kept_assistant={kept_assistant}")
        return cache

    def _prefill_text_p2(self, cache: "StreamLLMInference.AccumKVCache",
                         text: str) -> "StreamLLMInference.AccumKVCache":
        """二期内部：把裸文本（含特殊 token）prefill 进 KV，显式管理 mask/position_ids。"""
        if not text:
            raise Exception("prefill 文本为空")
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.device)
        ids = inputs.input_ids
        n = ids.shape[1]
        if n == 0:
            raise Exception("prefill 文本无有效 token")
        attn = torch.cat(
            [cache.attention_mask, torch.ones(ids.shape, device=self.device, dtype=cache.attention_mask.dtype)],
            dim=-1,
        )
        position_ids = torch.arange(cache.seq_length, cache.seq_length + n,
                                    dtype=torch.long, device=self.device).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(
                input_ids=ids,
                attention_mask=attn,
                position_ids=position_ids,
                past_key_values=cache.past_key_values,
                use_cache=True,
                return_dict=True,
            )
        cache.past_key_values = outputs.past_key_values
        cache.attention_mask = attn
        cache.seq_length += n
        cache.next_token_logits = outputs.logits[:, -1, :]
        return cache

    def reopen_user_role(self, cache: "StreamLLMInference.AccumKVCache") -> "StreamLLMInference.AccumKVCache":
        """crop 后重建 role 边界：注入 "<|im_end|>\\n<|im_start|>user\\n" 关闭 assistant、打开 user（Q3）。"""
        return self._prefill_text_p2(cache, self._role_switch_to_user)

    def open_assistant_role(self, cache: "StreamLLMInference.AccumKVCache") -> "StreamLLMInference.AccumKVCache":
        """关闭 user、打开新 assistant（注入 generation_prompt）；之后可再 generate_accumulating。"""
        cache = self._prefill_text_p2(cache, self.generation_prompt)
        cache.assistant_start = cache.seq_length   # 新一轮 assistant 内容起点
        cache.assistant_token_ids = []
        return cache

    def prefill_user_text(self, cache: "StreamLLMInference.AccumKVCache", text: str) -> "StreamLLMInference.AccumKVCache":
        """向当前打开的 user role 追加用户新输入文本（裸文本，续累积）。"""
        return self._prefill_text_p2(cache, text)

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
        return self.KVCache(outputs.past_key_values, model_inputs.input_ids, model_inputs.attention_mask, outputs.logits[:, -1, :])

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
        return self.KVCache(outputs.past_key_values, new_fragment_ids, attention_mask, outputs.logits[:, -1, :])

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
