#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLM流式推理测试程序
测试 stream_add_and_generate 方法，使用逗号分隔的文本作为生成器输入
"""

import time
from src.llm.stream_llm_inference import StreamLLMInference
from src.utils.logging_utils import get_logger, set_global_log_level

llm_streamer = StreamLLMInference(device="auto", eval_mode=False)
# 测试文本
test_text = '''
安妮，你相信吗？我最近迷上了迷路的孩子，他们简直太令人震撼了，他们最新的回归曲雷震子一直都在我的播放列表里循环。
'''
# 设置日志级别
set_global_log_level('INFO')
logger = get_logger(__name__)

def text_generator(text: str):
    """
    将文本按逗号分隔，生成文本片段生成器
    
    Args:
        text: 输入文本
        
    Yields:
        tuple[str, bool]: (文本片段, 是否结束)
    """
    parts = text.split('，')  # 按中文逗号分隔
    
    for i, part in enumerate(parts):
        is_end = (i == len(parts) - 1)  # 最后一个片段标记为结束
        if not is_end:
            part += '，'  # 除最后一个片段外，保留逗号
        yield part, is_end

    
def warm_up():
    """
    大模型加载预热
    """

    logger.info("开始LLM一次性推理测试...")

    try:
        warm_test = "测试一下"
        logger.info(f"预热文本: {warm_test}")
        logger.info("=" * 60)
        
        # 记录开始时间
        start_time = time.perf_counter()
        
        # 调用 stream_add_and_generate 方法
        response_tokens = []
        first_token_time = None
        
        for token in llm_streamer.once_add_and_generate(warm_test):
            if first_token_time is None:
                first_token_event = llm_streamer.get_last_timings()
                first_token_time = first_token_event[llm_streamer.TimingEventType.RETURN_LOGITS] - start_time
                first_decode_time = first_token_event[llm_streamer.TimingEventType.DECODE_TOKEN] - start_time
             
            logger.info(f"生成Token: '{token}'")
            response_tokens.append(token)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # 输出结果
        full_response = "".join(response_tokens)
        logger.info("=" * 60)
        logger.info(f"完整回复: {full_response}")
        logger.info(f"首Token延迟: {first_token_time*1000:.2f} ms" if first_token_time else "无首Token")
        logger.info(f"总生成时间: {total_time*1000:.2f} ms")
        logger.info(f"生成Token数量: {len(response_tokens)}")
        logger.info(f"========预热结束===========")
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        raise e

def main():
    # 设置日志级别
    set_global_log_level('INFO')
    logger = get_logger(__name__)
    
    logger.info("开始LLM流式推理测试...")
     
    try:
        logger.info(f"测试文本: {test_text}")
        logger.info("=" * 60)
        
        # 创建文本生成器
        prompt_gen = text_generator(test_text)
        
        # 记录开始时间
        start_time = time.perf_counter()
        
        # 调用 stream_add_and_generate 方法
        response_tokens = []
        first_token_time = None

        # 缓存提示文本
        kv_cache  = None
        for text, is_end in prompt_gen:
            kv_cache = llm_streamer.cache_prompt(text, pre_cache = kv_cache, is_end=is_end)

        prompt_end_time = time.perf_counter()   # 缓存结束时间
        # 生成回复
        for token in llm_streamer.generate(pre_cache = kv_cache):
            # 提取首次响应时间
            if not first_token_time:
                first_token_event = llm_streamer.get_last_timings()
                first_token_time = first_token_event[llm_streamer.TimingEventType.RETURN_LOGITS] - prompt_end_time
                first_decode_time = first_token_event[llm_streamer.TimingEventType.DECODE_TOKEN] - prompt_end_time
            
            # 输出文本
            response_tokens.append(token)
            logger.debug(f"输出文本{token}")
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # 输出结果
        full_response = "".join(response_tokens)
        logger.info("=" * 60)
        logger.info(f"完整回复: {full_response}")
        logger.info(f"首Token延迟: {first_token_time*1000:.2f} ms" if first_token_time else "无首Token")
        logger.info(f"总生成时间: {total_time*1000:.2f} ms")
        logger.info(f"生成Token数量: {len(response_tokens)}")
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())

def main_once():
    # 设置日志级别
    set_global_log_level('INFO')
    logger = get_logger(__name__)
    
    logger.info("开始LLM一次性推理测试...")
     
    try:
        logger.info(f"测试文本: {test_text}")
        logger.info("=" * 60)
        
        # 记录开始时间
        start_time = time.perf_counter()
        
        # 调用 stream_add_and_generate 方法
        response_tokens = []
        first_token_time = None
        
        for token in llm_streamer.once_add_and_generate(test_text):
            if first_token_time is None:
                first_token_event = llm_streamer.get_last_timings()
                first_token_time = first_token_event[llm_streamer.TimingEventType.RETURN_LOGITS] - start_time
                first_decode_time = first_token_event[llm_streamer.TimingEventType.DECODE_TOKEN] - start_time
            
            logger.info(f"生成Token: '{token}'")
            response_tokens.append(token)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # 输出结果
        full_response = "".join(response_tokens)
        logger.info("=" * 60)
        logger.info(f"完整回复: {full_response}")
        logger.info(f"首Token延迟: {first_token_time*1000:.2f} ms" if first_token_time else "无首Token")
        logger.info(f"总生成时间: {total_time*1000:.2f} ms")
        logger.info(f"生成Token数量: {len(response_tokens)}")
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    warm_up()
    main_once()    
    main()
