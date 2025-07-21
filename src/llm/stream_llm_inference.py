# src/llm/stream_llm_inference.py
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
import time
import traceback
import logging
from typing import Generator, Tuple
from threading import Thread
import queue

# 从配置导入
from src.config import LLM_MODEL_NAME, DEVICE, HF_HOME, HF_ENDPOINT, HF_TOKEN
from src.utils.logging_utils import get_logger # 导入 logger

logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()  # 加载 .env 文件中的环境变量

class StreamLLMInference:
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
        logger.info(f"正在加载LLM模型 {model_name} 到 {device}...")
        logger.info(f"HF_HOME: {hf_home}, HF_ENDPOINT: {hf_endpoint}")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            cache_dir=hf_home, 
            token=hf_token,
            trust_remote_code=True # 对于某些模型如Qwen是必要的
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map=device if device == "auto" else None, # device_map="auto" or device_map=device for single GPU
            cache_dir=hf_home,
            token=hf_token,
            trust_remote_code=True # 对于某些模型如Qwen是必要的
        )
        if device != "auto": # 如果不是自动分配，则手动移到指定设备
            self.model.to(device)
        
        self.model.eval() # 设置为评估模式
        self.past_key_values = None
        self.current_input_ids = None
        self.current_attention_mask = None
        self.eval_mode = eval_mode
        
        # 用于记录详细延迟的变量
        self.timings = {
            "last_token_gen_time_ms": 0.0,
            "last_kv_update_time_ms": 0.0,
            "last_precompute_time_ms": 0.0,
            "events": [] # (event_name, timestamp, duration_ms)
        }
        logger.info("LLM模型加载完成。")

        # 提取生成提示符
        # 为了获取正确的生成提示符，我们使用一个临时的messages
        temp_messages = [
            {"role": "system", "content": "You are a helpful assistant responding in Chinese."},
            {"role": "user", "content": "temp"}  # 临时内容
        ]
        
        # 获取带生成提示符的完整模板
        full_template = self.tokenizer.apply_chat_template(
            temp_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 获取不带生成提示符的模板
        no_gen_template = self.tokenizer.apply_chat_template(
            temp_messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # 提取生成提示符部分
        generation_prompt = full_template.replace(no_gen_template, "")
        self.generation_prompt = generation_prompt
        logger.debug(f"生成提示符: '{generation_prompt}'")

    def _record_timing(self, event_name, start_time, end_time=None):
        if end_time is None:
            end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000
        self.timings['events'].append((event_name, time.time(), duration_ms))
        if "precompute" in event_name:
            self.timings['last_precompute_time_ms'] = duration_ms
        elif "kv_update" in event_name:
            self.timings['last_kv_update_time_ms'] = duration_ms
        elif "token_gen" in event_name:
            self.timings['last_token_gen_time_ms'] = duration_ms
        logger.debug(f"Timing: {event_name} - {duration_ms:.2f} ms")

    def get_last_timings(self):
        return {
            "last_token_gen_time_ms": self.timings['last_token_gen_time_ms'],
            "last_kv_update_time_ms": self.timings['last_kv_update_time_ms'],
            "last_precompute_time_ms": self.timings['last_precompute_time_ms'],
        }

    def get_all_timing_events(self):
        return self.timings['events']
    
    def stream_add_and_generate(self, prompt:Generator[tuple[str, bool], None, None], system_prompt:str="You are a helpful assistant responding in Chinese.", max_new_tokens=50, temperature=0.1, top_p=0.9, repetition_penalty=1.1) -> Generator[tuple[str, float], None, None]:
        """
        通过文本生成器流式添加提示词并生成下一个token。
        prompt: 文本生成器，每个元素为tuple[str, bool]，其中str为文本片段，bool为是否结束。
        """

        # 初始化messages，因为流式场景主要是对话系统，所以需要初始化一个对话历史
        first_text = next(prompt)
        logger.info(f"first_text: {first_text}")
        init_user_text = first_text[0]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": init_user_text}
        ]

        # 初始化KV缓存
        past_key_values, current_input_ids, current_attention_mask = self._init_kv_cache(messages)

        # 如果生成结束，直接生成token，并返回
        if first_text[1]:
            yield self.generate_next_token(max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty)
            return

        # 如果生成未结束，则流式添加提示词
        for text, is_end in prompt:
            logger.info(f"流式添加提示词: {text}, 是否结束: {is_end}")
            if is_end:            
                text += self.generation_prompt
                logger.info(f"流式添加提示词结束生成提示符: {text}")
            past_key_values, current_input_ids, current_attention_mask = self._add_stream_prompt(past_key_values, current_input_ids, current_attention_mask, text)

        
        # 流式生成token
        logger.info(f"流式生成token")
        logger.info(f"is eval mode: {self.eval_mode}")
        for token, token_time in self._stream_generate_tokens(past_key_values, current_input_ids, current_attention_mask, max_new_tokens, temperature, top_p, repetition_penalty):
            yield token, token_time
            
    def stream_add_and_generate_queue(self, prompt_queue: queue.Queue[Tuple[str, bool]], system_prompt:str="You are a helpful assistant responding in Chinese.", max_new_tokens=50, temperature=0.1, top_p=0.9, repetition_penalty=1.1) -> Generator[tuple[str, float], None, None]:
        """
        通过文本队列流式添加提示词并生成下一个token。
        prompt_queue: 文本队列，每个元素为tuple[str, bool]，其中str为文本片段，bool为是否结束。
        """

        # 初始化messages，因为流式场景主要是对话系统，所以需要初始化一个对话历史
        first_text = prompt_queue.get()
        logger.info(f"first_text: {first_text}")
        init_user_text = first_text[0]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": init_user_text}
        ]

        # 初始化KV缓存
        past_key_values, current_input_ids, current_attention_mask = self._init_kv_cache(messages)

        # 如果生成结束，直接生成token，并返回
        if first_text[1]:
            yield self.generate_next_token(max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty)
            return

        # 如果生成未结束，则流式添加提示词
        while True:
            text, is_end = prompt_queue.get()
            logger.info(f"流式添加提示词: {text}, 是否结束: {is_end}")
            if is_end:            
                text += self.generation_prompt
                logger.info(f"流式添加提示词结束生成提示符: {text}")
                past_key_values, current_input_ids, current_attention_mask = self._add_stream_prompt(past_key_values, current_input_ids, current_attention_mask, text)
                break
            past_key_values, current_input_ids, current_attention_mask = self._add_stream_prompt(past_key_values, current_input_ids, current_attention_mask, text)

        
        # 流式生成token
        logger.info(f"流式生成token")
        logger.info(f"is eval mode: {self.eval_mode}")
        for token, token_time in self._stream_generate_tokens(past_key_values, current_input_ids, current_attention_mask, max_new_tokens, temperature, top_p, repetition_penalty):
            yield token, token_time

    def _init_kv_cache(self, messages) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        使用初始化prompt进行KV缓存首次计算
        Returns:
            tuple: (past_key_values, input_ids, attention_mask)
        """
        prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        model_inputs = self.tokenizer([prompt_text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(
                input_ids=model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                use_cache=True,
                return_dict=True
            )
        return outputs.past_key_values, model_inputs.input_ids, model_inputs.attention_mask
        
    def _add_stream_prompt(self, past_key_values, current_input_ids, current_attention_mask, text_fragments) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        流式添加提示词，并更新KV缓存。
        """
        # 流式添加提示词，并更新KV缓存。
        # 返回新的past_key_values, current_input_ids, current_attention_mask
        # 处理新的文本片段（如果存在）
        if text_fragments:
            new_fragment_inputs = self.tokenizer(text_fragments, return_tensors="pt", add_special_tokens=False).to(self.device)
            new_fragment_ids = new_fragment_inputs.input_ids
            
            if new_fragment_ids.shape[1] > 0: # 如果新片段有有效token
                logger.debug(f"处理新的文本片段，token数量: {new_fragment_ids.shape[1]}")
                
                # 直接处理新片段，不传递position_ids，让模型自动处理
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=new_fragment_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True
                    )
                
                # 更新状态
                new_past_key_values = outputs.past_key_values
                new_current_input_ids = torch.cat([current_input_ids, new_fragment_ids], dim=1)

                # 为新片段创建attention_mask
                new_fragment_attention_mask = torch.ones_like(new_fragment_ids, device=self.device)
                new_current_attention_mask = torch.cat([
                    current_attention_mask,
                    new_fragment_attention_mask
                ], dim=1)

        return new_past_key_values, new_current_input_ids, new_current_attention_mask

    def _stream_generate_tokens(self, past_key_values, current_input_ids, current_attention_mask, max_new_tokens, temperature, top_p, repetition_penalty) -> Generator[tuple[str, float], None, None]:
        """
        流式生成token。
        """

        if self.eval_mode:
            # 使用最后一个token作为输入
            gen_input_ids = current_input_ids[:, -1:]
            
            # 计算正确的position_ids（参考stream_input_cache.py）
            seq_length = current_input_ids.shape[1]
            position_ids = torch.arange(seq_length - 1, seq_length, dtype=torch.long, device=gen_input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(gen_input_ids)
            
            # 处理attention_mask
            gen_attention_mask = torch.cat([
                current_attention_mask, 
                torch.ones((current_attention_mask.shape[0], 1), 
                            dtype=current_attention_mask.dtype, 
                            device=current_attention_mask.device)
            ], dim=-1)
            # 如果是评估模式，只需要第一个token的响应时间即可，不需要将token进行转换
            # 使用模型生成下一个token
            with torch.no_grad():
                outputs = self.model(
                    input_ids=gen_input_ids,
                    attention_mask=gen_attention_mask,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                    use_cache=True,
                    return_dict=True
                )
            
            # 获取logits并生成下一个token
            next_token_logits = outputs.logits[:, -1, :]

            # 这里实际token已经生成，可以作为首个token生成的评估时间点
            yield self._decode_logits(next_token_logits, temperature, top_p, repetition_penalty), time.perf_counter()
            return

        # 非评估模式，需要完整生成token并解码
        for i in range(max_new_tokens):
            # 使用最后一个token作为输入
            gen_input_ids = current_input_ids[:, -1:]
            
            # 计算正确的position_ids（参考stream_input_cache.py）
            seq_length = current_input_ids.shape[1]
            position_ids = torch.arange(seq_length - 1, seq_length, dtype=torch.long, device=gen_input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(gen_input_ids)
            
            # 处理attention_mask
            gen_attention_mask = torch.cat([
                current_attention_mask, 
                torch.ones((current_attention_mask.shape[0], 1), 
                            dtype=current_attention_mask.dtype, 
                            device=current_attention_mask.device)
            ], dim=-1)

            # 使用模型生成下一个token
            with torch.no_grad():
                outputs = self.model(
                    input_ids=gen_input_ids,
                    attention_mask=gen_attention_mask,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                    use_cache=True,
                    return_dict=True
                )
            
            # 获取logits并生成下一个token
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = self._decode_logits(next_token_logits, temperature, top_p, repetition_penalty)
            generated_token_time = time.perf_counter()
            
            # 检查是否是EOS token
            is_eos = next_token_id.item() == self.tokenizer.eos_token_id
            logger.debug(f"生成的token ID: {next_token_id.item()}, EOS token ID: {self.tokenizer.eos_token_id}, 是否EOS: {is_eos}")
            
            # 更新KV缓存和输入ID序列
            past_key_values = outputs.past_key_values
            
            # 确保next_token_id的维度正确
            if next_token_id.dim() == 1:
                next_token_id = next_token_id.unsqueeze(0)
            
            current_input_ids = torch.cat([current_input_ids, next_token_id], dim=1)
            
            # 更新attention_mask
            current_attention_mask = torch.cat([
                current_attention_mask, 
                torch.ones((current_attention_mask.shape[0], 1), 
                            dtype=current_attention_mask.dtype, 
                            device=current_attention_mask.device)
            ], dim=1)

            # 解码生成的token
            generated_token_text = self.tokenizer.decode(next_token_id[0], skip_special_tokens=True)

            yield generated_token_text, generated_token_time

            # 如果生成结束，则返回
            if is_eos:
                break
        
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

    def once_add_and_generate(self, prompt:str, system_prompt:str="You are a helpful assistant responding in Chinese.", max_new_tokens=50, temperature=0.1, top_p=0.9, repetition_penalty=1.1) -> Generator[tuple[str, float], None, None]:
        """
        一次性添加提示词并生成token。
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        logger.info(f"prompt_text: {prompt_text}")
        model_inputs = self.tokenizer([prompt_text], return_tensors="pt", padding=True).to(self.device)

        # 如果是评估模式，只需要第一个token的响应时间即可，不用generate
        if self.eval_mode:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=model_inputs.input_ids,
                    attention_mask=model_inputs.attention_mask,
                    use_cache=True,
                    return_dict=True
                )

            # 获取logits并生成下一个token
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = self._decode_logits(next_token_logits, temperature, top_p, repetition_penalty)
            generated_token_time = time.perf_counter()
            yield self.tokenizer.decode(next_token_id[0], skip_special_tokens=True), generated_token_time
            return
        
        # 非评估模式，需要完整生成token并解码
        streamer = TextIteratorStreamer(self.tokenizer)
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature if temperature > 0 else 0.001, # HF generate 不喜欢 T=0
            "top_p": top_p if temperature > 0 else None, # top_p只在采样时有效
            "repetition_penalty": repetition_penalty,
            "use_cache": True, # 使用缓存,保持一致性
            "pad_token_id": self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "do_sample": True if temperature > 0 else False,
            "streamer": streamer,  # 关键修复：将 streamer 添加到 generation_config 中
        }
        # 移除None值的参数
        generation_config = {k: v for k, v in generation_config.items() if v is not None}

        def generate_with_streamer():
            self.model.generate(
                input_ids=model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                **generation_config  # 展开 generation_config 为关键字参数
            )

        thread = Thread(target=generate_with_streamer)
        thread.start()
        for new_text  in streamer:
            yield new_text, time.perf_counter()
        thread.join()        

    def _prepare_inputs(self, text_fragments, add_generation_prompt=True):
        """准备模型输入，考虑历史记录。"""
        # 这里的messages格式需要根据你的对话管理策略来调整
        # 简单示例：将所有片段连接起来
        # 更复杂的场景可能需要维护一个对话历史列表
        full_text = "".join(text_fragments)
        messages = [
            {"role": "system", "content": "You are a helpful assistant responding in Chinese."},
            {"role": "user", "content": full_text}
        ]
        
        try:
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt
            )
        except Exception as e:
            logger.warning(f"Failed to apply chat template: {e}. Using raw text. This might be incorrect for some models.")
            prompt_text = full_text
            if add_generation_prompt and hasattr(self.tokenizer, 'generation_prompt_template'):
                # 尝试手动添加，但这非常依赖模型
                 logger.warning("Attempting to add generation prompt manually, might not be standard.")

        model_inputs = self.tokenizer([prompt_text], return_tensors="pt", padding=True).to(self.device)
        return model_inputs.input_ids, model_inputs.attention_mask

    def precompute_kv_cache_for_prompt(self, prompt_text_fragments):
        """
        为给定的提示文本片段预计算KV缓存。
        prompt_text_fragments (list of str): 用户输入的文本片段列表。
        """
        logger.info(f"为输入预计算KV缓存: {''.join(prompt_text_fragments)[:100]}...")
        start_time = time.perf_counter()

        input_ids, attention_mask = self._prepare_inputs(prompt_text_fragments, add_generation_prompt=False)
        logger.debug(f"预计算模式 (add_generation_prompt=False): 序列长度={input_ids.shape[1]}")
        logger.debug(f"Token序列: {self.tokenizer.decode(input_ids[0], skip_special_tokens=False)}")
        
        if input_ids.shape[1] == 0:
            logger.warning("输入为空，跳过KV缓存预计算。")
            self._record_timing("precompute_kv_empty_input", start_time)
            return 0.0

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True
            )
        
        self.past_key_values = outputs.past_key_values
        self.current_input_ids = input_ids
        self.current_attention_mask = attention_mask
        
        self._record_timing("precompute_kv_cache", start_time)
        logger.info(f"KV缓存预计算完成，耗时: {self.timings['last_precompute_time_ms']:.2f} ms")
        return self.timings['last_precompute_time_ms'] / 1000 # 返回秒

    def precompute_kv_cache_for_streaming(self, prompt_text_fragments):
        """
        为流式输入场景预计算KV缓存，会添加generation prompt。
        适用于逐步输入完整问题后开始生成的场景。
        """
        logger.info(f"为流式场景预计算KV缓存: {''.join(prompt_text_fragments)[:100]}...")
        start_time = time.perf_counter()

        input_ids, attention_mask = self._prepare_inputs(prompt_text_fragments, add_generation_prompt=True)
        logger.debug(f"流式预计算模式 (add_generation_prompt=True): 序列长度={input_ids.shape[1]}")
        logger.debug(f"Token序列: {self.tokenizer.decode(input_ids[0], skip_special_tokens=False)}")
        
        if input_ids.shape[1] == 0:
            logger.warning("输入为空，跳过KV缓存预计算。")
            self._record_timing("precompute_kv_empty_input", start_time)
            return 0.0

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True
            )
        
        self.past_key_values = outputs.past_key_values
        self.current_input_ids = input_ids
        self.current_attention_mask = attention_mask
        
        self._record_timing("precompute_kv_cache", start_time)
        logger.info(f"流式KV缓存预计算完成，耗时: {self.timings['last_precompute_time_ms']:.2f} ms")
        return self.timings['last_precompute_time_ms'] / 1000 # 返回秒

    def generate_next_token(self, new_text_fragment="", temperature=0.1, top_p=0.9, repetition_penalty=1.1):
        """
        基于预计算的KV缓存和新的文本片段（如果有）生成下一个token。
        如果 new_text_fragment 不为空，会先更新KV缓存。
        
        Returns:
            tuple: (generated_token_text, latency_seconds, is_eos)
        """
        try:
            logger.debug(f"开始生成下一个token... 新文本片段: '{new_text_fragment}'")
            start_time = time.perf_counter()

            if not self.past_key_values or self.current_input_ids is None:
                # 如果没有KV缓存，或者没有当前输入，说明需要先处理完整输入
                logger.warning("警告: 没有预计算的KV缓存或当前输入。请先调用 precompute_kv_cache_for_prompt 或确保输入已处理。")
                # 尝试从 new_text_fragment 构建初始输入
                if not new_text_fragment:
                    raise ValueError("无法生成token，没有KV缓存且没有新的输入文本。")
                self.precompute_kv_cache_for_prompt([new_text_fragment])
                # 预计算后，past_key_values 和 current_input_ids 会被设置
            
            # 处理新的文本片段（如果存在）
            if new_text_fragment:
                # 参考stream_input_cache.py的方式，不添加特殊token
                new_fragment_tokens = self.tokenizer(new_text_fragment, return_tensors="pt", add_special_tokens=False)
                new_fragment_ids = new_fragment_tokens.input_ids.to(self.device)
                
                if new_fragment_ids.shape[1] > 0: # 如果新片段有有效token
                    logger.debug(f"处理新的文本片段，token数量: {new_fragment_ids.shape[1]}")
                    
                    # 直接处理新片段，不传递position_ids，让模型自动处理
                    with torch.no_grad():
                        outputs = self.model(
                            input_ids=new_fragment_ids,
                            past_key_values=self.past_key_values,
                            use_cache=True,
                            return_dict=True
                        )
                    
                    # 更新状态
                    self.past_key_values = outputs.past_key_values
                    self.current_input_ids = torch.cat([self.current_input_ids, new_fragment_ids], dim=1)
                    if self.current_attention_mask is not None:
                        # 为新片段创建attention_mask
                        new_fragment_attention_mask = torch.ones_like(new_fragment_ids, device=self.device)
                        self.current_attention_mask = torch.cat([
                            self.current_attention_mask,
                            new_fragment_attention_mask
                        ], dim=1)
            
            # 参考stream_input_cache.py的方式生成下一个token
            if self.past_key_values is not None:
                # 使用最后一个token作为输入
                gen_input_ids = self.current_input_ids[:, -1:]
                
                # 计算正确的position_ids（参考stream_input_cache.py）
                seq_length = self.current_input_ids.shape[1]
                position_ids = torch.arange(seq_length - 1, seq_length, dtype=torch.long, device=gen_input_ids.device)
                position_ids = position_ids.unsqueeze(0).expand_as(gen_input_ids)
                
                # 处理attention_mask
                if self.current_attention_mask is not None:
                    gen_attention_mask = torch.cat([
                        self.current_attention_mask, 
                        torch.ones((self.current_attention_mask.shape[0], 1), 
                                  dtype=self.current_attention_mask.dtype, 
                                  device=self.current_attention_mask.device)
                    ], dim=-1)
                else:
                    gen_attention_mask = None
                    logger.debug("未使用attention_mask")
            else:
                gen_input_ids = self.current_input_ids
                gen_attention_mask = self.current_attention_mask
                position_ids = None
            
            # 使用模型生成下一个token
            with torch.no_grad():
                outputs = self.model(
                    input_ids=gen_input_ids,
                    attention_mask=gen_attention_mask,
                    past_key_values=self.past_key_values,
                    position_ids=position_ids,
                    use_cache=True,
                    return_dict=True
                )
            
            # 获取logits并生成下一个token
            next_token_logits = outputs.logits[:, -1, :]
            
            # 应用温度和top_p采样
            if temperature > 0:
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
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
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

            first_token_time = time.perf_counter() - start_time
            
            # 检查是否是EOS token
            is_eos = next_token_id.item() == self.tokenizer.eos_token_id
            logger.debug(f"生成的token ID: {next_token_id.item()}, EOS token ID: {self.tokenizer.eos_token_id}, 是否EOS: {is_eos}")
            
            # 更新KV缓存和输入ID序列
            self.past_key_values = outputs.past_key_values
            
            # 确保next_token_id的维度正确
            if next_token_id.dim() == 1:
                next_token_id = next_token_id.unsqueeze(0)
            
            self.current_input_ids = torch.cat([self.current_input_ids, next_token_id], dim=1)
            
            if self.current_attention_mask is not None:
                self.current_attention_mask = torch.cat([
                    self.current_attention_mask, 
                    torch.ones((self.current_attention_mask.shape[0], 1), 
                              dtype=self.current_attention_mask.dtype, 
                              device=self.device)
                ], dim=1)

            # 解码生成的token
            generated_token_text = self.tokenizer.decode(next_token_id[0], skip_special_tokens=True)
            
            # 添加调试信息：显示当前序列末尾和生成的token
            if logger.isEnabledFor(logging.DEBUG):
                last_5_tokens = self.current_input_ids[0, -5:].tolist()
                last_5_text = self.tokenizer.decode(last_5_tokens, skip_special_tokens=False)
                logger.debug(f"序列末尾5个token: {last_5_tokens} -> '{last_5_text}'")
                logger.debug(f"新生成token: {next_token_id.item()} -> '{generated_token_text}' (原始: '{self.tokenizer.decode(next_token_id[0], skip_special_tokens=False)}')")
            
            logger.info(f"生成token: '{generated_token_text}', 耗时: {first_token_time:.4f}秒, 是否EOS: {is_eos}")
            return generated_token_text, first_token_time, is_eos

        except Exception as e:
            logger.error(f"生成过程中出现错误: {str(e)}")
            logger.error("\n详细错误信息:")
            logger.error(traceback.format_exc())
            return None, 0, False

    def generate_full_response_with_cache(self, initial_prompt_fragments, max_new_tokens=50, temperature=0.1, top_p=0.9, repetition_penalty=1.1, stop_sequences=None):
        """
        基于初始提示，使用KV缓存流式生成完整回复。
        """
        logger.info("\n--- 开始使用KV缓存流式生成完整回复 ---")
        self.reset_state() # 清空历史状态
        self.precompute_kv_cache_for_prompt(initial_prompt_fragments)
        
        generated_tokens = []
        total_generation_time = 0
        first_token_latency = -1.0

        if stop_sequences is None:
            stop_sequences = [self.tokenizer.eos_token, "<|endoftext|>", "<|im_end|>"] # 常见的停止符
        if self.tokenizer.eos_token not in stop_sequences:
            stop_sequences.append(self.tokenizer.eos_token)

        for i in range(max_new_tokens):
            token_text, gen_time, is_eos = self.generate_next_token()
            total_generation_time += gen_time
            if first_token_latency < 0 and token_text is not None:
                first_token_latency = gen_time
            
            if token_text is None: # 生成出错
                logger.warning("生成token失败，停止生成")
                break
            
            # 优先检查EOS token
            if is_eos:
                logger.info(f"检测到EOS token，停止生成。")
                break
            
            # 只有非空token才添加到结果中
            if token_text.strip():  # 过滤掉纯空白字符的token
                generated_tokens.append(token_text)
            else:
                logger.debug(f"跳过空token: '{token_text}'")
                # 即使是空token，也要检查是否应该停止
                # 如果连续生成多个空token，可能模型已经"结束"了
                continue
            
            # 检查停止序列 (基于文本内容)
            current_response = "".join(generated_tokens)
            
            # 检查是否包含停止序列
            should_stop = False
            for stop_seq in stop_sequences:
                if stop_seq and stop_seq in current_response:
                    logger.info(f"检测到停止序列: '{stop_seq}'，停止生成。")
                    # 移除停止序列本身
                    if current_response.endswith(stop_seq):
                        current_response = current_response[:-len(stop_seq)]
                        # 重新构建token列表
                        if current_response:
                            # 简单地去掉最后几个字符对应的token（近似处理）
                            while generated_tokens and "".join(generated_tokens).endswith(stop_seq):
                                generated_tokens.pop()
                    should_stop = True
                    break
            
            if should_stop:
                break
        
        full_response = "".join(generated_tokens).strip()
        logger.info(f"完整回复: {full_response}")
        logger.info(f"首个token延迟: {first_token_latency:.4f}秒 (如果成功生成)")
        logger.info(f"总生成时间: {total_generation_time:.4f}秒, Tokens: {len(generated_tokens)}")
        return full_response, first_token_latency, total_generation_time, len(generated_tokens)

    def generate_without_cache_for_comparison(self, prompt_text_fragments, max_new_tokens=50, temperature=0.1, top_p=0.9, repetition_penalty=1.1, stop_sequences=None):
        """
        不使用预计算KV缓存（但模型内部仍会用cache），用于对比。
        """
        logger.info("\n--- 开始无KV缓存生成完整回复 (用于对比) ---")
        start_time = time.perf_counter()

        input_ids, attention_mask = self._prepare_inputs(prompt_text_fragments, add_generation_prompt=True)
        
        if input_ids.shape[1] == 0:
            logger.warning("输入为空，无法生成。")
            return "", 0.0, 0.0, 0

        # 记录从调用 generate 到第一个 token 产生的时间
        first_token_gen_start_time = time.perf_counter()
        
        # 使用模型的 generate 方法
        # 为了获取首个token的延迟，我们不能直接用max_new_tokens，需要一些技巧
        # 或者，我们直接测量整个 generate 调用的时间作为粗略比较
        # 这里我们简化，测量到第一个 token 的时间可能比较困难，除非用自定义生成循环
        # 因此，这里的 first_token_latency 将是整个生成的开始到结束（近似）
        
        # 设置停止序列
        effective_stop_sequences = []
        if stop_sequences:
            effective_stop_sequences.extend(stop_sequences)
        if self.tokenizer.eos_token and self.tokenizer.eos_token not in effective_stop_sequences:
             effective_stop_sequences.append(self.tokenizer.eos_token)
        common_stops = ["<|endoftext|>", "<|im_end|>", "\n\nUSER:", "\n\nASSISTANT:"]
        for cs in common_stops:
            if cs not in effective_stop_sequences:
                effective_stop_sequences.append(cs)
        
        eos_token_id = self.tokenizer.eos_token_id
        # 如果有多个停止序列，需要更复杂的 StoppingCriteria
        # 此处简化为只使用eos_token_id
        # stopping_criteria = transformers.StoppingCriteriaList()
        # stopping_criteria.append(transformers.MaxNewTokensCriteria(max_new_tokens=max_new_tokens))
        
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature if temperature > 0 else 0.001, # HF generate 不喜欢 T=0
            "top_p": top_p if temperature > 0 else None, # top_p只在采样时有效
            "repetition_penalty": repetition_penalty,
            "use_cache": False, # 明确禁用缓存
            "pad_token_id": self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
            "eos_token_id": eos_token_id,
            "do_sample": True if temperature > 0 else False,
        }
        # 移除None值的参数
        generation_config = {k: v for k, v in generation_config.items() if v is not None}


        with torch.no_grad():
            output_sequences = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_config
            )
        
        generation_time_s = time.perf_counter() - start_time # 这是总时间
        # 对于 .generate(), 第一个token的精确延迟不好直接获取
        # 我们将整个生成时间作为 first_token_latency 和 total_generation_time
        
        # 解码输出
        # output_sequences 包含输入的 prompt
        response_ids = output_sequences[0][input_ids.shape[1]:]
        full_response = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        
        # 尝试去除停止符
        cleaned_response = full_response
        for stop_seq in effective_stop_sequences:
            if stop_seq and cleaned_response.endswith(stop_seq):
                cleaned_response = cleaned_response[:-len(stop_seq)].strip()
                break # 只移除第一个匹配的末尾停止符
        full_response = cleaned_response

        num_generated_tokens = len(response_ids)

        logger.info(f"完整回复 (无缓存): {full_response}")
        logger.info(f"总生成时间 (无缓存): {generation_time_s:.4f}秒, Tokens: {num_generated_tokens}")
        
        # 在这种模式下，"first_token_latency" 实际上是总时间，因为我们无法简单分离
        # 如果需要精确的无缓存FTL，需要手动实现生成循环
        return full_response, generation_time_s, generation_time_s, num_generated_tokens

    def reset_state(self):
        """重置KV缓存和当前输入状态。"""
        self.past_key_values = None
        self.current_input_ids = None
        self.current_attention_mask = None
        self.timings = {
            "last_token_gen_time_ms": 0.0,
            "last_kv_update_time_ms": 0.0,
            "last_precompute_time_ms": 0.0,
            "events": []
        }
        logger.debug("LLM状态已重置 (KV缓存清除)。")

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

def test_stream_llm_inference_once_input():
    """
    测试一次输入的推理是否能正确运行
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='LLM非流式推理测试')
    parser.add_argument('--log-level', type=str, default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='设置日志级别')
    parser.add_argument('--prompt', type=str, default='你好，请问你叫什么名字？', help='用户提示词')
    parser.add_argument('--eval', action='store_true', help='是否为评估模式')
    args = parser.parse_args()

    # 根据参数设置日志级别
    log_level = getattr(logging, args.log_level)
    logging.basicConfig(level=log_level)
    logger = logging.getLogger(__name__)

    llm_streamer = StreamLLMInference(device="cuda", eval_mode=args.eval)
    prompt = args.prompt
    def run_single_generation(llm_streamer, prompt):
        """运行单次生成测试并返回统计信息"""
        text = llm_streamer.once_add_and_generate(prompt)
        response = ""
        start_time = time.perf_counter()
        first_token_time = None
        num_tokens = 0
        for new_text, timestamp in text:
            if first_token_time is None:
                first_token_time = timestamp - start_time
            logger.info(f"新文本: {new_text}, 时间戳: {timestamp}")
            response += new_text
            num_tokens += 1
        end_time = time.perf_counter()
        total_time = end_time - start_time
        logger.info(f"完整回复: {response}")
        logger.info(f"首个token延迟: {first_token_time*1000:.2f} ms" if first_token_time else "无首个token")
        logger.info(f"总生成时间: {total_time*1000:.2f} ms, Tokens: {num_tokens}")
        return {
            'response': response,
            'first_token_time': first_token_time,
            'total_time': total_time,
            'num_tokens': num_tokens
        }
    
    # 运行第一次生成
    logger.info("=" * 50)
    logger.info("第一次生成测试")
    logger.info("=" * 50)
    result1 = run_single_generation(llm_streamer, prompt)
    
    # 运行第二次生成，使用不同的prompt
    second_prompt = "你能做什么？"
    logger.info("=" * 50)
    logger.info("第二次生成测试")
    logger.info("=" * 50)
    result2 = run_single_generation(llm_streamer, second_prompt)
    
    # 对比两次结果
    logger.info("=" * 50)
    logger.info("测试结果对比")
    logger.info("=" * 50)
    logger.info(f"第一次: 首token延迟 {result1['first_token_time']*1000:.2f}ms, 总时间 {result1['total_time']*1000:.2f}ms, Tokens: {result1['num_tokens']}")
    logger.info(f"第二次: 首token延迟 {result2['first_token_time']*1000:.2f}ms, 总时间 {result2['total_time']*1000:.2f}ms, Tokens: {result2['num_tokens']}")
    
def test_stream_llm_inference_stream_input():
    """
    测试流式输入的推理是否能正确运行
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='LLM流式推理测试')
    parser.add_argument('--log-level', type=str, default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='设置日志级别')
    parser.add_argument('--eval', action='store_true', help='是否为评估模式')
    args = parser.parse_args()
    
    # 根据参数设置日志级别
    log_level = getattr(logging, args.log_level)

    logging.basicConfig(level=log_level)
    logger = logging.getLogger(__name__)

    llm_streamer = StreamLLMInference(device="cuda", eval_mode=args.eval)

    # 模拟流式文本生成器
    timing_info = {'stream_end_time': None}
    def prompt_generator(prompts: list[str]):
        """模拟流式文本生成器"""
        texts = prompts
        for i, text in enumerate(texts):
            is_end = (i == len(texts) - 1)  # 最后一个文本片段标记为结束
            if is_end:
                timing_info['stream_end_time'] = time.perf_counter()
            yield text, is_end
    
    def run_stream_generation(prompts: list[str]):
        """运行流式生成测试"""
        text_generator = llm_streamer.stream_add_and_generate(prompt_generator(prompts))
        response = ""
        first_token_time = None
        num_tokens = 0
        
        for new_text, timestamp in text_generator:
            if first_token_time is None:
                first_token_time = timestamp - timing_info['stream_end_time']
            logger.info(f"新文本: {new_text}, 时间戳: {timestamp}")
            response += new_text
            num_tokens += 1
        
        end_time = time.perf_counter()
        total_time = end_time - timing_info['stream_end_time']
        logger.info(f"完整回复: {response}")
        logger.info(f"首个token延迟: {first_token_time*1000:.2f} ms" if first_token_time else "无首个token")
        logger.info(f"总生成时间: {total_time*1000:.2f} ms, Tokens: {num_tokens}")
        return response, first_token_time, total_time, num_tokens
    
    # 运行两次测试，取第二次的结果
    logger.info("进行第一次测试运行（预热）...")
    run_stream_generation(["你好，", "请问你叫", "什么名字？"])
    
    logger.info("进行第二次测试运行（正式结果）...")
    response, first_token_time, total_time, num_tokens = run_stream_generation(["你，", "能做", "什么？"])
    logger.info(f"正式首个token延迟: {first_token_time*1000:.2f} ms")
    logger.info(f"正式总生成时间: {total_time*1000:.2f} ms, Tokens: {num_tokens}")



# --- 主测试逻辑 (示例) ---
if __name__ == '__main__':
    test_stream_llm_inference_once_input()
    test_stream_llm_inference_stream_input()

def test_stream_llm_inference_old():
    """
    旧的流式推理测试
    """
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='LLM流式推理测试')
    parser.add_argument('--log-level', type=str, default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='设置日志级别')
    args = parser.parse_args()
    
    # 根据参数设置日志级别
    log_level = getattr(logging, args.log_level)

    logging.basicConfig(level=log_level)
    logger = logging.getLogger(__name__)

    logger.info("开始LLM流式推理测试...")
    
    # 使用一个较小的模型进行测试，例如 'gpt2' 或适合中文的 'Qwen/Qwen-1_8B-Chat'
    # test_model_name = "gpt2" # 如果测试英文
    test_model_name = "Qwen/Qwen1.5-0.5B-Chat" # 确保有权限和资源下载
    # test_model_name = LLM_MODEL_NAME # 使用config中的模型
    
    # 强制使用CPU进行本地测试，除非你有GPU且配置正确
    test_device = "cuda" if torch.cuda.is_available() else "cpu" # "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"测试将使用设备: {test_device}")

    try:
        llm_streamer = StreamLLMInference(model_name=test_model_name, device=test_device)
        initial_prompt = ["你好，请问你叫什么名字？"]

        # --- 测试3: 对比无缓存生成 ---
        logger.info("\n\n--- 测试3: 无缓存生成对比 ---")
        llm_streamer.reset_state()
        # 注意：无缓存测试会重新处理整个prompt，所以通常会慢很多
        response_no_cache, ftl_no_cache, total_time_no_cache, num_tokens_no_cache = \
            llm_streamer.generate_without_cache_for_comparison(
                prompt_text_fragments=initial_prompt, 
                max_new_tokens=30,
                temperature=0.7
            )
        logger.info(f"  完整回复 (无缓存): {response_no_cache}")
        logger.info(f"  总延迟 (无缓存,近似FTL): {ftl_no_cache*1000:.2f} ms") # 这里是总时间
        logger.info(f"  总生成时间 (无缓存): {total_time_no_cache*1000:.2f} ms, Tokens: {num_tokens_no_cache}")

        # --- 测试1: 逐步输入并生成 ---
        logger.info("\n--- 测试1: 逐步输入并生成 ---")
        # 模拟更实际的流式场景：逐步接收ASR输出，累积后触发LLM生成
        prompt_parts = ["你好，", "请问你叫", "什么名字？"]
        llm_streamer.reset_state()
        
        # 模拟逐步接收用户输入（ASR输出）
        accumulated_input = ""
        for i, prompt_part in enumerate(prompt_parts):
            accumulated_input += prompt_part
            logger.info(f"接收输入片段 {i+1}: '{prompt_part}' -> 累积输入: '{accumulated_input}'")
            
            # 在实际应用中，这里可能有触发条件，比如检测到句号、问号等
            # 或者ASR表示用户说话结束
        
        # 当累积到完整问题时，使用流式预计算模式
        logger.info("检测到完整问题，开始LLM处理...")
        llm_streamer.precompute_kv_cache_for_streaming([accumulated_input])
        
        # 开始生成回复
        generated_tokens = []
        for i in range(10):  # 生成几个token作为示例
            token, latency, is_eos = llm_streamer.generate_next_token()
            if token and token.strip():  # 只记录非空token
                generated_tokens.append(token)
                logger.info(f"  生成Token {len(generated_tokens)}: '{token}' (延迟: {latency*1000:.2f} ms)")
            
            if is_eos:
                logger.info("检测到EOS，生成结束")
                break
            elif not token:
                logger.info("生成失败，停止")
                break
        
        full_response = "".join(generated_tokens)
        logger.info(f"测试1完整回复: '{full_response}'")
        logger.info(f"测试1所有计时事件: {llm_streamer.get_all_timing_events()}")


        # --- 测试2: 使用 precompute_kv_cache_for_prompt 和 generate_full_response_with_cache ---
        logger.info("\n\n--- 测试2: 完整流程 (precompute + generate_full_response_with_cache) ---")
        llm_streamer.reset_state()
        logger.info(f"完整提示: {initial_prompt}")
        
        response, ftl, total_time, num_tokens = llm_streamer.generate_full_response_with_cache(
            initial_prompt_fragments=initial_prompt, 
            max_new_tokens=30,
            temperature=0.7
        )
        logger.info(f"  完整回复 (有缓存): {response}")
        logger.info(f"  首Token总延迟 (含预计算): {ftl*1000:.2f} ms")
        logger.info(f"  总生成时间 (含预计算): {total_time*1000:.2f} ms, Tokens: {num_tokens}")
        logger.info(f"测试2所有计时事件: {llm_streamer.get_all_timing_events()}")
        llm_streamer._log_kv_cache_size(llm_streamer.past_key_values)

        # --- 测试4: 再次对比无缓存生成 ---
        logger.info("\n\n--- 测试4: 无缓存生成对比 ---")
        llm_streamer.reset_state()
        # 注意：无缓存测试会重新处理整个prompt，所以通常会慢很多
        response_no_cache, ftl_no_cache, total_time_no_cache, num_tokens_no_cache = \
            llm_streamer.generate_without_cache_for_comparison(
                prompt_text_fragments=initial_prompt, 
                max_new_tokens=30,
                temperature=0.7
            )
        logger.info(f"  完整回复 (无缓存): {response_no_cache}")
        logger.info(f"  总延迟 (无缓存,近似FTL): {ftl_no_cache*1000:.2f} ms") # 这里是总时间
        logger.info(f"  总生成时间 (无缓存): {total_time_no_cache*1000:.2f} ms, Tokens: {num_tokens_no_cache}")


        logger.info("\nLLM流式推理测试结束。")

    except ImportError as e:
        logger.error(f"导入错误，请确保所有依赖已安装: {e}")
    except Exception as e:
        logger.error(f"测试过程中发生未预料的错误: {e}")
        logger.error(traceback.format_exc())
