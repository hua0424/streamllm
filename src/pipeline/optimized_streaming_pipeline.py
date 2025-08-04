# src/pipeline/optimized_streaming_pipeline.py

import time
import numpy as np
import threading
import queue
from collections import deque
from typing import Generator, Tuple, Optional, List

from src.asr.faster_whisper_streamer import StreamingASRProcessor
from src.llm.stream_llm_inference import StreamLLMInference
from src.utils.audio2stream import wav2stream
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

class OptimizedStreamingPipeline:
    """
    优化的流式处理流水线
    关键优化：
    1. ASR和LLM并行处理
    2. LLM预处理KV缓存，在语音输入时就开始处理已有文本
    3. 最小化首Token延迟
    """
    
    def __init__(
        self,
        asr_model_size: str = "base",
        llm_model_name: str = "Qwen/Qwen1.5-0.5B-Chat",  # 统一使用指定模型
        device: str = "cpu",
        recognition_threshold: float = 2.0,  # 降低阈值，更快响应
        system_prompt: str = "You are a helpful assistant responding in Chinese."
    ):
        self.system_prompt = system_prompt
        
        # 创建ASR处理器 - 优化参数
        self.asr_processor = StreamingASRProcessor(
            model_size=asr_model_size,
            device=device,
            recognition_threshold=recognition_threshold,
            prefix_segments=1  # 减少前缀段，加快响应
        )
        
        # 创建LLM处理器
        self.llm_processor = StreamLLMInference(
            model_name=llm_model_name,
            device=device,
            eval_mode=True  # 评估模式，只生成首Token
        )
        
        # 流式处理状态
        self.accumulated_text = ""
        self.is_processing = False
        self.is_speech_finished = False
        
        # 线程安全的队列
        self.text_queue = queue.Queue()
        self.response_queue = queue.Queue()
        
        logger.info("优化流式流水线初始化完成")
    
    def process_streaming_audio(
        self, 
        audio_generator: Generator[Tuple[np.ndarray, bool], None, None]
    ) -> Generator[Tuple[str, float], None, None]:
        """
        处理流式音频并返回LLM响应
        
        Args:
            audio_generator: 音频流生成器 (audio_chunk, is_last)
            
        Yields:
            Tuple[str, float]: (生成的token, 时间戳)
        """
        logger.info("开始优化流式音频处理")
        
        # 启动ASR处理线程
        asr_thread = threading.Thread(
            target=self._asr_processing_thread,
            args=(audio_generator,),
            daemon=True
        )
        asr_thread.start()
        
        # 启动LLM预处理线程
        llm_thread = threading.Thread(
            target=self._llm_preprocessing_thread,
            daemon=True
        )
        llm_thread.start()
        
        # 主线程等待最终响应
        try:
            while True:
                try:
                    response_data = self.response_queue.get(timeout=30)  # 30秒超时
                    if response_data is None:  # 结束信号
                        break
                    yield response_data
                except queue.Empty:
                    logger.warning("响应队列超时")
                    break
        finally:
            # 清理线程
            asr_thread.join(timeout=1)
            llm_thread.join(timeout=1)
            logger.info("流式处理完成")
    
    def _asr_processing_thread(self, audio_generator: Generator[Tuple[np.ndarray, bool], None, None]):
        """ASR处理线程"""
        logger.info("ASR处理线程启动")
        
        try:
            for audio_chunk, is_last in audio_generator:
                # 处理音频块
                new_texts = self.asr_processor.add_audio_chunk(
                    audio_chunk, 
                    is_stream_finished=is_last
                )
                
                # 将新文本添加到队列
                for text_output in new_texts:
                    if hasattr(text_output, 'text'):
                        text = text_output.text.strip()
                    else:
                        text = str(text_output).strip()
                        
                    if text:
                        self.text_queue.put(text)
                        logger.debug(f"ASR输出新文本: {text}")
                
                if is_last:
                    self.is_speech_finished = True
                    self.text_queue.put(None)  # 结束信号
                    logger.info("语音输入结束")
                    break
                    
        except Exception as e:
            logger.error(f"ASR处理线程错误: {e}")
            self.text_queue.put(None)
    
    def _llm_preprocessing_thread(self):
        """LLM预处理线程 - 关键优化点"""
        logger.info("LLM预处理线程启动")
        
        try:
            # 初始化对话历史
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": ""}  # 空的用户消息，将逐步填充
            ]
            
            # 预计算系统消息的KV缓存
            system_text = self.llm_processor.tokenizer.apply_chat_template(
                [{"role": "system", "content": self.system_prompt}],
                tokenize=False,
                add_generation_prompt=False
            )
            
            logger.info("预计算系统消息KV缓存")
            self.llm_processor.precompute_kv_cache_for_prompt([system_text])
            
            accumulated_user_text = ""
            
            # 持续处理ASR输出的文本
            while True:
                try:
                    text_fragment = self.text_queue.get(timeout=1)
                    
                    if text_fragment is None:  # 结束信号
                        logger.info("收到ASR结束信号，开始最终生成")
                        break
                    
                    # 累积文本
                    accumulated_user_text += text_fragment + " "
                    logger.debug(f"累积文本: {accumulated_user_text}")
                    
                    # 更新KV缓存（增量处理）
                    user_message_text = f"\\n\\nUSer: {accumulated_user_text.strip()}"
                    
                    # 使用流式添加更新KV缓存
                    try:
                        # 这里我们不立即生成，只是更新缓存
                        # 检查是否需要初始化KV缓存
                        if not hasattr(self.llm_processor, 'past_key_values') or self.llm_processor.past_key_values is None:
                            # 初始化用户消息的KV缓存
                            self.llm_processor.precompute_kv_cache_for_prompt([f"User: {text_fragment}"])
                        else:
                            # 增量更新KV缓存
                            self.llm_processor.generate_next_token(
                                new_text_fragment=text_fragment
                            )
                        logger.debug("KV缓存已更新")
                    except Exception as e:
                        logger.warning(f"KV缓存更新失败: {e}")
                
                except queue.Empty:
                    # 没有新文本，继续等待
                    continue
                except Exception as e:
                    logger.error(f"LLM预处理错误: {e}")
                    break
            
            # 语音结束后，生成最终响应
            if accumulated_user_text.strip():
                logger.info(f"开始生成最终响应，用户输入: {accumulated_user_text.strip()}")
                
                # 添加生成提示符并生成首个token
                generation_prompt = self.llm_processor.generation_prompt
                
                start_time = time.perf_counter()
                first_token, latency, is_eos = self.llm_processor.generate_next_token(
                    new_text_fragment=generation_prompt
                )
                
                if first_token:
                    self.response_queue.put((first_token, start_time))
                    logger.info(f"首Token生成: '{first_token}', 延迟: {latency*1000:.2f}ms")
                else:
                    logger.warning("首Token生成失败")
            
            # 发送结束信号
            self.response_queue.put(None)
            
        except Exception as e:
            logger.error(f"LLM预处理线程错误: {e}")
            self.response_queue.put(None)

def create_optimized_pipeline_from_wav(
    wav_path: str,
    chunk_duration: float = 0.3,  # 更小的块大小，更快响应
    simulate_delay: bool = False,
    asr_model_size: str = "base",
    llm_model_name: str = "Qwen/Qwen1.5-0.5B-Chat"  # 统一使用指定模型
) -> Generator[Tuple[str, float], None, None]:
    """
    从WAV文件创建优化的流式处理流水线
    
    Args:
        wav_path: 音频文件路径
        chunk_duration: 音频块时长
        simulate_delay: 是否模拟实时延迟
        asr_model_size: ASR模型大小
        llm_model_name: LLM模型名称
        
    Yields:
        Tuple[str, float]: (生成的token, 时间戳)
    """
    # 创建音频流生成器
    audio_generator = wav2stream(
        wav_path=wav_path,
        chunk_duration=chunk_duration,
        simulate_delay=simulate_delay
    )
    
    # 创建优化流水线
    pipeline = OptimizedStreamingPipeline(
        asr_model_size=asr_model_size,
        llm_model_name=llm_model_name
    )
    
    # 处理流式音频
    yield from pipeline.process_streaming_audio(audio_generator)

# 测试函数
def test_optimized_pipeline():
    """测试优化流水线"""
    import argparse
    from src.utils.logging_utils import set_global_log_level
    
    parser = argparse.ArgumentParser(description='测试优化流式流水线')
    parser.add_argument('--wav_path', type=str, 
                       default="/usr/local/app/jupyterlab/yanjiu/streamllm/data/processed_audio/length30+/3_1_d473.wav",
                       help='音频文件路径')
    parser.add_argument('--chunk_duration', type=float, default=0.3, help='音频块时长')
    parser.add_argument('--simulate_delay', action='store_true', help='模拟实时延迟')
    parser.add_argument('--log_level', type=str, default='INFO', help='日志级别')
    
    args = parser.parse_args()
    set_global_log_level(args.log_level)
    
    logger.info("=" * 60)
    logger.info("测试优化流式流水线")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    try:
        # 运行优化流水线
        for token, timestamp in create_optimized_pipeline_from_wav(
            wav_path=args.wav_path,
            chunk_duration=args.chunk_duration,
            simulate_delay=args.simulate_delay
        ):
            current_time = time.time() - start_time
            logger.info(f"[{current_time:.2f}s] 生成Token: '{token}'")
            
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    test_optimized_pipeline()