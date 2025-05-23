# src/llm/stream_llm_inference.py

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import traceback
import logging

# 从配置导入
from src.config import LLM_MODEL_NAME, DEVICE, HF_HOME, HF_ENDPOINT, HF_TOKEN
from src.utils.logging_utils import get_logger # 导入 logger

logger = get_logger(__name__) # 初始化 logger

class StreamLLMInference:
    def __init__(
        self,
        model_name=LLM_MODEL_NAME,
        device=DEVICE,
        hf_home=HF_HOME,
        hf_endpoint=HF_ENDPOINT,
        hf_token=HF_TOKEN
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
        
        # 用于记录详细延迟的变量
        self.timings = {
            "last_token_gen_time_ms": 0.0,
            "last_kv_update_time_ms": 0.0,
            "last_precompute_time_ms": 0.0,
            "events": [] # (event_name, timestamp, duration_ms)
        }
        logger.info("LLM模型加载完成。")

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

    def generate_next_token(self, new_text_fragment="", temperature=0.1, top_p=0.9, repetition_penalty=1.1):
        """
        基于预计算的KV缓存和新的文本片段（如果有）生成下一个token。
        如果 new_text_fragment 不为空，会先更新KV缓存。
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
                new_fragment_ids = self.tokenizer(new_text_fragment, return_tensors="pt").input_ids[:, 1:].to(self.device) # 去掉BOS token
                if new_fragment_ids.shape[1] > 0: # 如果新片段有有效token
                    logger.debug(f"处理新的文本片段，token数量: {new_fragment_ids.shape[1]}")
                    with torch.no_grad():
                        outputs = self.model(
                            input_ids=new_fragment_ids,
                            attention_mask=torch.ones_like(new_fragment_ids, device=self.device), # 新片段的mask全为1
                            past_key_values=self.past_key_values,
                            use_cache=True,
                            return_dict=True
                        )
                    self.past_key_values = outputs.past_key_values
                    self.current_input_ids = torch.cat([self.current_input_ids, new_fragment_ids], dim=1)
                    if self.current_attention_mask is not None:
                        self.current_attention_mask = torch.cat([
                            self.current_attention_mask,
                            torch.ones_like(new_fragment_ids, device=self.device)
                        ], dim=1)
            
            # 使用KV缓存生成下一个token
            # 输入应该是最后一个真实token (来自current_input_ids的最后一个)
            last_token_input_id = self.current_input_ids[:, -1:]
            current_seq_len = self.current_input_ids.shape[1]

            # 注意：这里的 position_ids 需要正确设置
            # 对于使用 past_key_values 的情况，新的 position_id 应该是过去的序列长度
            position_ids = torch.tensor([[current_seq_len -1]], dtype=torch.long, device=self.device)
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=last_token_input_id,
                    attention_mask=self.current_attention_mask, # 完整的attention mask
                    past_key_values=self.past_key_values,
                    position_ids=position_ids, # 关键：指定正确的位置ID
                    use_cache=True,
                    return_dict=True
                )
            
            next_token_logits = outputs.logits[:, -1, :]
            # 温度采样等逻辑可以加在这里
            # next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            
            # 应用温度和top_p (简化版)
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
            
            # 更新KV缓存和输入ID序列以备下次使用
            self.past_key_values = outputs.past_key_values
            self.current_input_ids = torch.cat([self.current_input_ids, next_token_id], dim=1)
            if self.current_attention_mask is not None:
                 self.current_attention_mask = torch.cat([
                    self.current_attention_mask, 
                    torch.ones((self.current_attention_mask.shape[0], 1), dtype=self.current_attention_mask.dtype, device=self.device)
                ], dim=1)

            # 解码生成的token
            generated_token_text = self.tokenizer.decode(next_token_id[0], skip_special_tokens=True)
            
            logger.info(f"生成token: '{generated_token_text}', 耗时: {first_token_time:.4f}秒")
            return generated_token_text, first_token_time

        except Exception as e:
            logger.error(f"生成过程中出现错误: {str(e)}")
            logger.error("\n详细错误信息:")
            logger.error(traceback.format_exc())
            return None, 0 # 或者抛出异常

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
            token_text, gen_time = self.generate_next_token()
            total_generation_time += gen_time
            if first_token_latency < 0 and token_text is not None:
                first_token_latency = gen_time
            
            if token_text is None: # 生成出错
                break
            
            generated_tokens.append(token_text)
            # 检查停止条件
            current_response = "".join(generated_tokens)
            # print(f"已生成 ({i+1}/{max_new_tokens}): {current_response}") # 实时打印
            
            # 更可靠的停止序列检查
            if any(stop_seq in current_response for stop_seq in stop_sequences if stop_seq):
                # 如果检测到停止序列，可能需要移除它本身
                for stop_seq in stop_sequences:
                    if stop_seq and current_response.endswith(stop_seq):
                        current_response = current_response[:-len(stop_seq)]
                        generated_tokens = self.tokenizer.tokenize(current_response) # 重新tokenize以获得干净的列表
                        break
                logger.info(f"检测到停止序列，停止生成。")
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

# --- 主测试逻辑 (示例) ---
if __name__ == '__main__':
    # 配置日志级别为DEBUG，以便看到详细输出
    logging.basicConfig(level=logging.DEBUG)
    main_logger = get_logger(__name__, level="DEBUG") # 确保此处的logger也设置为DEBUG

    main_logger.info("开始LLM流式推理测试...")
    
    # 使用一个较小的模型进行测试，例如 'gpt2' 或适合中文的 'Qwen/Qwen-1_8B-Chat'
    # test_model_name = "gpt2" # 如果测试英文
    test_model_name = "Qwen/Qwen1.5-0.5B-Chat" # 确保有权限和资源下载
    # test_model_name = LLM_MODEL_NAME # 使用config中的模型
    
    # 强制使用CPU进行本地测试，除非你有GPU且配置正确
    test_device = "cpu" # "cuda" if torch.cuda.is_available() else "cpu"
    main_logger.info(f"测试将使用设备: {test_device}")

    try:
        llm_streamer = StreamLLMInference(model_name=test_model_name, device=test_device)

        # --- 测试1: 逐步输入并生成 ---
        main_logger.info("\n--- 测试1: 逐步输入并生成 ---")
        llm_streamer.reset_state()
        
        prompt_part1 = "请介绍一下北京的"
        main_logger.info(f"输入片段1: '{prompt_part1}'")
        # 首次输入，会预计算KV，但不一定立即生成token，取决于外部逻辑
        # 在 StreamLLMInference 的设计中，generate_next_token 会处理
        # 如果是第一次调用，且有 new_text_fragment，它会先precompute
        token1, latency1 = llm_streamer.generate_next_token(new_text_fragment=prompt_part1)
        main_logger.info(f"  LLM Token1: '{token1}' (Latency: {latency1*1000:.2f} ms)")
        llm_streamer._log_kv_cache_size(llm_streamer.past_key_values)

        prompt_part2 = "天气和"
        main_logger.info(f"输入片段2: '{prompt_part2}'")
        token2, latency2 = llm_streamer.generate_next_token(new_text_fragment=prompt_part2)
        main_logger.info(f"  LLM Token2: '{token2}' (Latency: {latency2*1000:.2f} ms)")
        llm_streamer._log_kv_cache_size(llm_streamer.past_key_values)
        
        prompt_part3 = "美食。"
        main_logger.info(f"输入片段3: '{prompt_part3}'")
        token3, latency3 = llm_streamer.generate_next_token(new_text_fragment=prompt_part3)
        main_logger.info(f"  LLM Token3: '{token3}' (Latency: {latency3*1000:.2f} ms)")
        llm_streamer._log_kv_cache_size(llm_streamer.past_key_values)

        # 继续生成几个token
        main_logger.info("继续生成后续几个token...")
        full_sentence_tokens = [token1, token2, token3]
        for i in range(5):
            next_tok, next_lat = llm_streamer.generate_next_token() # 无新输入，纯粹基于现有KV生成
            if next_tok:
                main_logger.info(f"  LLM Next Token {i+1}: '{next_tok}' (Latency: {next_lat*1000:.2f} ms)")
                full_sentence_tokens.append(next_tok)
            else:
                main_logger.info("  LLM未能生成更多token。")
                break
        main_logger.info(f"测试1完整句子: {''.join(filter(None,full_sentence_tokens))}")
        main_logger.info(f"测试1所有计时事件: {llm_streamer.get_all_timing_events()}")


        # --- 测试2: 使用 precompute_kv_cache_for_prompt 和 generate_full_response_with_cache ---
        main_logger.info("\n\n--- 测试2: 完整流程 (precompute + generate_full_response_with_cache) ---")
        initial_prompt = ["你好，请问你叫什么名字？"]
        main_logger.info(f"完整提示: {initial_prompt}")
        
        response, ftl, total_time, num_tokens = llm_streamer.generate_full_response_with_cache(
            initial_prompt_fragments=initial_prompt, 
            max_new_tokens=30,
            temperature=0.7
        )
        main_logger.info(f"  完整回复 (有缓存): {response}")
        main_logger.info(f"  首Token总延迟 (含预计算): {ftl*1000:.2f} ms")
        main_logger.info(f"  总生成时间 (含预计算): {total_time*1000:.2f} ms, Tokens: {num_tokens}")
        main_logger.info(f"测试2所有计时事件: {llm_streamer.get_all_timing_events()}")
        llm_streamer._log_kv_cache_size(llm_streamer.past_key_values)

        # --- 测试3: 对比无缓存生成 ---
        main_logger.info("\n\n--- 测试3: 无缓存生成对比 ---")
        # 注意：无缓存测试会重新处理整个prompt，所以通常会慢很多
        response_no_cache, ftl_no_cache, total_time_no_cache, num_tokens_no_cache = \
            llm_streamer.generate_without_cache_for_comparison(
                prompt_text_fragments=initial_prompt, 
                max_new_tokens=30,
                temperature=0.7
            )
        main_logger.info(f"  完整回复 (无缓存): {response_no_cache}")
        main_logger.info(f"  总延迟 (无缓存,近似FTL): {ftl_no_cache*1000:.2f} ms") # 这里是总时间
        main_logger.info(f"  总生成时间 (无缓存): {total_time_no_cache*1000:.2f} ms, Tokens: {num_tokens_no_cache}")

        main_logger.info("\nLLM流式推理测试结束。")

    except ImportError as e:
        main_logger.error(f"导入错误，请确保所有依赖已安装: {e}")
    except Exception as e:
        main_logger.error(f"测试过程中发生未预料的错误: {e}")
        main_logger.error(traceback.format_exc())
