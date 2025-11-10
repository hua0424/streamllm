# src/llm/stream_llm_inference.py
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
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
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading LLM model {model_name} on {device}")
        logger.debug(f"HF_HOME: {hf_home}, HF_ENDPOINT: {hf_endpoint}")
        self.device = device
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                cache_dir=hf_home, 
                token=hf_token,
                trust_remote_code=True, # 对于某些模型如Qwen是必要的
                local_files_only=False  # 先尝试在线加载
            )
        except Exception as e:
            raise RuntimeError(f"无法加载tokenizer: {e}")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map= "auto", #device if device == "auto" else device, # device_map="auto" or device_map=device for single GPU
                cache_dir=hf_home,
                token=hf_token,
                trust_remote_code=True, # 对于某些模型如Qwen是必要的
                local_files_only=False  # 先尝试在线加载
            )
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

        # 用于记录详细延迟的变量
        self.timing_events:Dict[StreamLLMInference.TimingEventType, float] = {}


    class KVCache:
        def __init__(self, past_key_values:torch.Tensor, pre_input_ids:torch.Tensor, pre_attention_mask:torch.Tensor):
            self.past_key_values = past_key_values
            self.pre_input_ids = pre_input_ids
            self.pre_attention_mask = pre_attention_mask

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
        """
        使用预计算缓存生成回复
        @eval 是否评估模式，True=评估模式，只生成首个token
        """
        self.reset_timings()
        if pre_cache is None:
            raise Exception("未进行kv缓存初始化")
        
        self.timing_events[self.TimingEventType.START_FUNCTION] = time.perf_counter()
        # 使用最后一个token作为输入
        gen_input_ids = pre_cache.pre_input_ids[:, -1:]
        gen_attention_mask = pre_cache.pre_attention_mask
        past_key_values = pre_cache.past_key_values

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
            
            # 获取logits并生成下一个token
            next_token_logits = outputs.logits[:, -1, :]
            self.timing_events[self.TimingEventType.RETURN_LOGITS] = time.perf_counter()
            next_token_id = self._decode_logits(next_token_logits, temperature, top_p, repetition_penalty)
            # 检查是否是EOS token
            is_eos = next_token_id.item() == self.tokenizer.eos_token_id
            # 解码生成的token
            generated_token_text = self.tokenizer.decode(next_token_id[0], skip_special_tokens=True) 
            self.timing_events[self.TimingEventType.DECODE_TOKEN] = time.perf_counter()
           
            # 更新KV缓存和输入ID序列
            past_key_values = outputs.past_key_values
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
        return self.KVCache(outputs.past_key_values, model_inputs.input_ids, model_inputs.attention_mask)

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
        with torch.no_grad():
            outputs = self.model(
                input_ids=new_fragment_ids,
                past_key_values=pre_cache.past_key_values,
                attention_mask=attention_mask, # 传入拼接后的完整 attention mask
                use_cache=True,
                return_dict=True
                )
        self.timing_events[self.TimingEventType.END_KV_CACHE] = time.perf_counter()
        return self.KVCache(outputs.past_key_values, new_fragment_ids, attention_mask)

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
