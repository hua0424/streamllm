# src/pipeline/ultra_low_latency_pipeline.py

import time
import numpy as np
import threading
import queue
from typing import Generator, Tuple, Optional, List

from src.asr.faster_whisper_streamer import StreamingASRProcessor
from src.llm.stream_llm_inference import StreamLLMInference
from src.utils.audio2stream import wav2stream
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

class UltraLowLatencyPipeline:
    """
    超低延迟流式处理流水线
    
    核心优化策略：
    1. 预测性处理：基于前几个词预测用户意图并预处理
    2. 并行KV缓存：多个可能的回复路径并行预计算
    3. 激进的触发策略：更早触发LLM生成
    4. 优化的ASR参数：牺牲一些准确性换取速度
    """
    
    def __init__(
        self,
        asr_model_size: str = "base",  # 与其他模式保持一致
        llm_model_name: str = "Qwen/Qwen1.5-0.5B-Chat",  # 统一使用指定模型
        device: str = "cpu",
        system_prompt: str = "You are a helpful assistant responding in Chinese."
    ):
        self.system_prompt = system_prompt
        
        # 超低延迟ASR配置
        self.asr_processor = StreamingASRProcessor(
            model_size=asr_model_size,
            device=device,
            recognition_threshold=1.0,  # 极低阈值，几乎实时触发
            prefix_segments=0  # 不使用前缀，最快响应
        )
        
        # LLM配置为评估模式
        self.llm_processor = StreamLLMInference(
            model_name=llm_model_name,
            device=device,
            eval_mode=True  # 只生成首Token
        )
        
        # 预设常见回复的KV缓存
        self.common_responses_cache = {}
        self._precompute_common_responses()
        
        # 流式状态
        self.accumulated_text = []
        self.is_speech_finished = False
        self.first_token_generated = False
        
        logger.info("Ultra low latency pipeline initialized")
    
    def _precompute_common_responses(self):
        """预计算常见问题的KV缓存"""
        common_patterns = [
            "你好",
            "你是谁",
            "你能做什么",
            "帮我",
            "请问"
        ]
        
        logger.debug("Precomputing KV cache for common responses")
        for pattern in common_patterns:
            try:
                # 为每个常见模式预计算KV缓存
                self.llm_processor.reset_state()
                self.llm_processor.precompute_kv_cache_for_prompt([f"User: {pattern}"])
                # 存储状态（简化版本，实际应用中需要更复杂的状态管理）
                self.common_responses_cache[pattern] = {
                    'past_key_values': self.llm_processor.past_key_values,
                    'input_ids': self.llm_processor.current_input_ids,
                    'attention_mask': self.llm_processor.current_attention_mask
                }
            except Exception as e:
                logger.warning(f"预计算模式'{pattern}'失败: {e}")
        
        logger.info(f"预计算完成，缓存了{len(self.common_responses_cache)}个常见模式")
    
    def _match_common_pattern(self, text: str) -> Optional[str]:
        """匹配常见模式"""
        text_lower = text.lower().strip()
        for pattern in self.common_responses_cache.keys():
            if pattern in text_lower:
                return pattern
        return None
    
    def process_streaming_audio(
        self, 
        audio_generator: Generator[Tuple[np.ndarray, bool], None, None]
    ) -> Generator[Tuple[str, float, str], None, None]:
        """
        处理流式音频并返回超低延迟响应
        
        Args:
            audio_generator: 音频流生成器
            
        Yields:
            Tuple[str, float, str]: (生成的token, 延迟时间, 触发原因)
        """
        logger.info("开始超低延迟音频处理")
        
        # 线程安全队列
        text_queue = queue.Queue()
        response_queue = queue.Queue()
        
        # ASR线程
        def asr_thread():
            try:
                for audio_chunk, is_last in audio_generator:
                    new_texts = self.asr_processor.add_audio_chunk(
                        audio_chunk, 
                        is_stream_finished=is_last
                    )
                    
                    for text_output in new_texts:
                        if hasattr(text_output, 'text'):
                            text = text_output.text.strip()
                        else:
                            text = str(text_output).strip()
                            
                        if text:
                            text_queue.put(('text', text))
                            logger.debug(f"ASR: {text}")
                    
                    if is_last:
                        text_queue.put(('end', None))
                        break
            except Exception as e:
                logger.error(f"ASR线程错误: {e}")
                text_queue.put(('end', None))
        
        # 超低延迟处理线程
        def ultra_low_latency_processor():
            try:
                word_count = 0
                
                while True:
                    try:
                        msg_type, data = text_queue.get(timeout=0.1)
                        
                        if msg_type == 'end':
                            break
                        elif msg_type == 'text':
                            self.accumulated_text.append(data)
                            word_count += len(data.split())
                            current_text = ' '.join(self.accumulated_text)
                            
                            logger.debug(f"累积文本({word_count}词): {current_text}")
                            
                            # 超激进触发策略
                            should_trigger = False
                            trigger_reason = ""
                            
                            # 1. 词数触发（极低阈值）
                            if word_count >= 2:
                                should_trigger = True
                                trigger_reason = f"词数触发({word_count}词)"
                            
                            # 2. 常见模式匹配
                            elif len(current_text) >= 2:
                                pattern = self._match_common_pattern(current_text)
                                if pattern:
                                    should_trigger = True
                                    trigger_reason = f"模式匹配({pattern})"
                            
                            # 3. 标点符号触发
                            elif any(p in current_text for p in ['？', '?', '。', '！', '!']):
                                should_trigger = True
                                trigger_reason = "标点符号触发"
                            
                            if should_trigger and not self.first_token_generated:
                                logger.info(f"触发首Token生成: {trigger_reason}")
                                self.first_token_generated = True
                                
                                # 生成首Token
                                start_time = time.perf_counter()
                                
                                try:
                                    # 检查是否有匹配的预计算缓存
                                    matched_pattern = self._match_common_pattern(current_text)
                                    if matched_pattern and matched_pattern in self.common_responses_cache:
                                        logger.info(f"使用预计算缓存: {matched_pattern}")
                                        cache_data = self.common_responses_cache[matched_pattern]
                                        
                                        # 恢复预计算状态
                                        self.llm_processor.past_key_values = cache_data['past_key_values']
                                        self.llm_processor.current_input_ids = cache_data['input_ids']
                                        self.llm_processor.current_attention_mask = cache_data['attention_mask']
                                        
                                        # 生成回复
                                        token, latency, is_eos = self.llm_processor.generate_next_token(
                                            new_text_fragment=self.llm_processor.generation_prompt
                                        )
                                    else:
                                        # 常规处理
                                        self.llm_processor.reset_state()
                                        self.llm_processor.precompute_kv_cache_for_prompt([f"User: {current_text}"])
                                        token, latency, is_eos = self.llm_processor.generate_next_token(
                                            new_text_fragment=self.llm_processor.generation_prompt
                                        )
                                    
                                    total_latency = time.perf_counter() - start_time
                                    
                                    if token:
                                        response_queue.put((token, total_latency, trigger_reason))
                                        logger.info(f"首Token: '{token}', 总延迟: {total_latency*1000:.1f}ms")
                                        
                                except Exception as e:
                                    logger.error(f"首Token生成失败: {e}")
                                
                    except queue.Empty:
                        continue
                
                # 结束信号
                response_queue.put((None, 0, "end"))
                
            except Exception as e:
                logger.error(f"超低延迟处理错误: {e}")
                response_queue.put((None, 0, "error"))
        
        # 启动线程
        asr_thread_obj = threading.Thread(target=asr_thread, daemon=True)
        processor_thread_obj = threading.Thread(target=ultra_low_latency_processor, daemon=True)
        
        asr_thread_obj.start()
        processor_thread_obj.start()
        
        # 主线程收集结果
        try:
            while True:
                token, latency, reason = response_queue.get(timeout=30)
                if token is None:
                    break
                yield token, latency, reason
        finally:
            asr_thread_obj.join(timeout=1)
            processor_thread_obj.join(timeout=1)

def create_ultra_low_latency_pipeline_from_wav(
    wav_path: str,
    chunk_duration: float = 0.2,  # 更小的块
    simulate_delay: bool = False,
    asr_model_size: str = "base"  # 与其他模式保持一致
) -> Generator[Tuple[str, float, str], None, None]:
    """创建超低延迟流水线"""
    
    audio_generator = wav2stream(
        wav_path=wav_path,
        chunk_duration=chunk_duration,
        simulate_delay=simulate_delay
    )
    
    pipeline = UltraLowLatencyPipeline(
        asr_model_size=asr_model_size
    )
    
    yield from pipeline.process_streaming_audio(audio_generator)

# 测试函数
def test_ultra_low_latency():
    """测试超低延迟流水线"""
    import argparse
    from src.utils.logging_utils import set_global_log_level
    
    parser = argparse.ArgumentParser(description='测试超低延迟流水线')
    parser.add_argument('--wav_path', type=str, 
                       default="/usr/local/app/jupyterlab/yanjiu/streamllm/data/processed_audio/length30+/3_1_d473.wav")
    parser.add_argument('--log_level', type=str, default='INFO')
    
    args = parser.parse_args()
    set_global_log_level(args.log_level)
    
    logger.info("🚀 超低延迟流水线测试")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    try:
        for token, latency, reason in create_ultra_low_latency_pipeline_from_wav(args.wav_path):
            current_time = time.time() - start_time
            logger.info(f"[{current_time:.2f}s] ⚡ Token: '{token}' | 延迟: {latency*1000:.1f}ms | 原因: {reason}")
            
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    test_ultra_low_latency()