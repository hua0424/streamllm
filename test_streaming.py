import numpy as np
import time
import logging
import argparse
import queue
import threading
from src.llm.stream_llm_inference import StreamLLMInference
from src.asr.faster_whisper_streamer import StreamingASRProcessor
from src.utils.audio2stream import wav2stream
from src.utils.logging_utils import get_logger, set_global_log_level
from src.pipeline.optimized_streaming_pipeline import create_optimized_pipeline_from_wav
from src.pipeline.ultra_low_latency_pipeline import create_ultra_low_latency_pipeline_from_wav
from typing import Generator, Tuple, Any

# 获取当前模块的logger
logger = get_logger(__name__)

def _produce_and_start_thread(generator: Generator[tuple[Any, bool], None, None]) -> tuple[threading.Thread, queue.Queue[Tuple[Any, bool]]]:
    """从原始生成器获取数据块并放入队列"""
    my_queue = queue.Queue[Tuple[any, bool]]()
    
    def producer_task():
        """
        线程任务：从生成器获取所有项目并放入队列，
        """
        for item in generator:
            my_queue.put(item)

    producer_thread = threading.Thread(target=producer_task)
    producer_thread.start()
    return producer_thread, my_queue

def _simulate_streamming_audio_by_wav(wav_path, chunk_duration, simulate_delay) -> tuple[threading.Thread, queue.Queue[Tuple[np.ndarray, bool]]]:
    """
    生成流式音频，并返回音频流生成器
    Args:
        wav_path: 音频文件路径
        chunk_duration: 每个音频块时长(秒)
        simulate_delay: 是否模拟实时延迟
    Returns:
        chunk_generator: 音频流生成器
    """
    
    # 创建音频流生成器
    chunk_generator = wav2stream(
        wav_path=wav_path,
        chunk_duration=chunk_duration,
        sample_rate=16000,
        simulate_delay=simulate_delay
    )

    return _produce_and_start_thread(chunk_generator)

def _asr_processor(asr_processor: StreamingASRProcessor, audio_queue: queue.Queue[Tuple[np.ndarray, bool]]) -> tuple[threading.Thread, queue.Queue[Tuple[str, bool]]]:
    """
    音频分段并ASR
    """
    asr_processor.reset()
    text_generator = asr_processor.add_audio_chunk_queue(audio_queue)

    return _produce_and_start_thread(text_generator)

def _llm_processor(llm_processor: StreamLLMInference, text_queue: queue.Queue[Tuple[str, bool]]) -> tuple[threading.Thread, queue.Queue[Tuple[str, bool]]]:
    """
    流式LLM推理
    """
    llm_generator = llm_processor.stream_add_and_generate_queue(text_queue)
    return _produce_and_start_thread(llm_generator)


def test_ultra_low_latency_pipeline(args):
    """测试超低延迟流水线"""
    logger.info("=" * 60)
    logger.info("测试超低延迟流水线")
    logger.info("=" * 60)
    logger.info(f"音频文件: {args.wav_path}")
    logger.info(f"音频块时长: {args.chunk_duration}s")
    logger.info(f"模拟延迟: {args.simulate_delay}")
    logger.info("-" * 60)
    
    start_time = time.time()
    
    try:
        # 使用超低延迟流水线
        for token, latency, reason in create_ultra_low_latency_pipeline_from_wav(
            wav_path=args.wav_path,
            chunk_duration=args.chunk_duration,
            simulate_delay=args.simulate_delay,
            asr_model_size=args.model_size
        ):
            current_time = time.time() - start_time
            logger.info(f"[{current_time:.2f}s] ⚡ Token: '{token}' | 延迟: {latency*1000:.1f}ms | 触发: {reason}")
            
        total_time = time.time() - start_time
        logger.info(f"超低延迟流水线总耗时: {total_time:.2f}s")
        
    except Exception as e:
        logger.error(f"超低延迟流水线测试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())

def test_optimized_pipeline(args):
    """测试优化的流式流水线"""
    logger.info("=" * 60)
    logger.info("测试优化流式流水线")
    logger.info("=" * 60)
    logger.info(f"音频文件: {args.wav_path}")
    logger.info(f"音频块时长: {args.chunk_duration}s")
    logger.info(f"模拟延迟: {args.simulate_delay}")
    logger.info("-" * 60)
    
    start_time = time.time()
    
    try:
        # 使用优化流水线
        for token, timestamp in create_optimized_pipeline_from_wav(
            wav_path=args.wav_path,
            chunk_duration=args.chunk_duration,
            simulate_delay=args.simulate_delay,
            asr_model_size=args.model_size,
            llm_model_name="Qwen/Qwen1.5-0.5B-Chat"  # 明确指定模型
        ):
            current_time = time.time() - start_time
            logger.info(f"[{current_time:.2f}s] 🤖 LLM回复: '{token}'")
            
        total_time = time.time() - start_time
        logger.info(f"优化流水线总耗时: {total_time:.2f}s")
        
    except Exception as e:
        logger.error(f"优化流水线测试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())

def test_original_pipeline(args):
    """测试原始流水线（对比用）"""
    logger.info("=" * 60)
    logger.info("测试原始流水线")
    logger.info("=" * 60)
    
    try:
        # 1. 生成音频
        logger.info("开始生成音频")
        audio_producer_thread, audio_queue = _simulate_streamming_audio_by_wav(args.wav_path, args.chunk_duration, args.simulate_delay)

        # 2. ASR
        # 创建流式ASR处理器
        logger.info("开始创建ASR处理器")
        asr_processor=StreamingASRProcessor(
            model_size=args.model_size,
            recognition_threshold=args.threshold,
            text_callback=None
        )
        asr_producer_thread, asr_queue = _asr_processor(
            asr_processor,
            audio_queue=audio_queue
        )

        # 3. llm
        # 创建流式llm处理器
        logger.info("开始创建LLM处理器")
        llm_processor=StreamLLMInference(
            model_name="Qwen/Qwen1.5-0.5B-Chat"  # 明确指定模型
        )
        llm_producer_thread, llm_queue = _llm_processor(
            llm_processor,
            asr_queue
        )

        # 4. 定义消费者生成器，打印最终文本
        def consumer_generator():
            """从队列中获取文本并输出"""
            while True:
                text, is_end = llm_queue.get()
                logger.info(f"LLM输出: {text}")     
                if is_end:
                    break

            logger.info("消费者线程已完成。")

        consumer_thread = threading.Thread(target=consumer_generator)
        consumer_thread.start()

        # 5. 等待线程结束
        audio_producer_thread.join()
        asr_producer_thread.join()
        llm_producer_thread.join()
        consumer_thread.join()
        
    except Exception as e:
        logger.error(f"测试失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def main():
    """主测试函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='测试流式语音对话系统')
    parser.add_argument('--wav_path', type=str, 
                       default="/usr/local/app/jupyterlab/yanjiu/streamllm/data/processed_audio/length30+/3_1_d473.wav",
                       help='测试音频文件路径')
    parser.add_argument('--chunk_duration', type=float, default=0.3, help='每个音频块时长(秒)')
    parser.add_argument('--simulate_delay', action='store_true', help='是否模拟实时延迟')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='日志级别')
    parser.add_argument('--model_size', type=str, default="base", help='ASR模型大小')
    parser.add_argument('--threshold', type=float, default=2.0, help='识别阈值(秒)')
    parser.add_argument('--mode', type=str, default='ultra', 
                       choices=['ultra', 'optimized', 'original', 'all'],
                       help='测试模式：超低延迟、优化版、原始版或全部对比')
    
    args = parser.parse_args()
    
    # 设置全局日志级别
    set_global_log_level(args.log_level)
    
    logger.info("🚀 流式语音对话系统测试程序")
    logger.info(f"音频文件: {args.wav_path}")
    logger.info(f"音频块时长: {args.chunk_duration}s")
    logger.info(f"模拟延迟: {args.simulate_delay}")
    logger.info(f"日志级别: {args.log_level}")
    logger.info(f"测试模式: {args.mode}")
    
    if args.mode in ['ultra', 'all']:
        test_ultra_low_latency_pipeline(args)
    
    if args.mode in ['optimized', 'all']:
        test_optimized_pipeline(args)
    
    if args.mode in ['original', 'all']:
        test_original_pipeline(args)

if __name__ == "__main__":
    main()
