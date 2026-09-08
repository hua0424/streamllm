#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM流式推理测试程序
测试流式KV缓存和一次性推理的功能
"""

import argparse
import time
from src.llm.stream_llm_inference import StreamLLMInference
from src.utils.logging_utils import get_logger, set_global_log_level
from src.config import LLM_MODEL_NAME

logger = get_logger(__name__)

# 测试文本
DEFAULT_TEST_TEXT = '''安妮，你相信吗？我最近迷上了迷路的孩子，他们简直太令人震撼了，他们最新的回归曲雷震子一直都在我的播放列表里循环。'''


def text_generator(text: str):
    """
    将文本按中文逗号分隔，生成文本片段生成器
    
    Args:
        text: 输入文本
        
    Yields:
        tuple[str, bool]: (文本片段, 是否结束)
    """
    parts = text.split('，')
    
    for i, part in enumerate(parts):
        is_end = (i == len(parts) - 1)
        if not is_end:
            part += '，'
        yield part, is_end


def warmup(llm_streamer: StreamLLMInference):
    """模型预热"""
    logger.info("执行模型预热...")
    warmup_text = "测试一下"
    
    start_time = time.perf_counter()
    response = ""
    for token in llm_streamer.once_add_and_generate(warmup_text, max_new_tokens=5):
        response += token
    
    warmup_time = time.perf_counter() - start_time
    logger.info(f"预热完成，耗时: {warmup_time*1000:.2f}ms，响应: '{response}'")


def test_streaming_inference(llm_streamer: StreamLLMInference, text: str):
    """
    测试流式KV缓存推理
    模拟流式文本输入，逐段缓存后生成回复
    """
    logger.info("=" * 60)
    logger.info("测试流式KV缓存推理")
    logger.info("=" * 60)
    logger.info(f"测试文本: {text[:50]}...")
    
    prompt_gen = text_generator(text)
    
    # 缓存阶段
    cache_start = time.perf_counter()
    kv_cache = None
    for prompt_part, is_end in prompt_gen:
        kv_cache = llm_streamer.cache_prompt(prompt_part, pre_cache=kv_cache, is_end=is_end)
    cache_end = time.perf_counter()
    
    # 生成阶段
    response_tokens = []
    first_token_time = None
    
    for token in llm_streamer.generate(pre_cache=kv_cache):
        if first_token_time is None:
            first_token_time = time.perf_counter() - cache_end
        response_tokens.append(token)
    
    total_time = time.perf_counter() - cache_start
    full_response = "".join(response_tokens)
    
    logger.info("-" * 40)
    logger.info(f"完整回复: {full_response}")
    logger.info(f"缓存耗时: {(cache_end - cache_start)*1000:.2f}ms")
    logger.info(f"首Token延迟: {first_token_time*1000:.2f}ms" if first_token_time else "无首Token")
    logger.info(f"总耗时: {total_time*1000:.2f}ms")
    logger.info(f"生成Token数: {len(response_tokens)}")
    
    return {
        "cache_time_ms": (cache_end - cache_start) * 1000,
        "first_token_ms": first_token_time * 1000 if first_token_time else None,
        "total_time_ms": total_time * 1000,
        "token_count": len(response_tokens),
        "response": full_response
    }


def test_once_inference(llm_streamer: StreamLLMInference, text: str):
    """
    测试一次性推理
    完整文本一次性输入后生成回复
    """
    logger.info("=" * 60)
    logger.info("测试一次性推理")
    logger.info("=" * 60)
    logger.info(f"测试文本: {text[:50]}...")
    
    start_time = time.perf_counter()
    response_tokens = []
    first_token_time = None
    
    for token in llm_streamer.once_add_and_generate(text):
        if first_token_time is None:
            first_token_time = time.perf_counter() - start_time
        response_tokens.append(token)
    
    total_time = time.perf_counter() - start_time
    full_response = "".join(response_tokens)
    
    logger.info("-" * 40)
    logger.info(f"完整回复: {full_response}")
    logger.info(f"首Token延迟: {first_token_time*1000:.2f}ms" if first_token_time else "无首Token")
    logger.info(f"总耗时: {total_time*1000:.2f}ms")
    logger.info(f"生成Token数: {len(response_tokens)}")
    
    return {
        "first_token_ms": first_token_time * 1000 if first_token_time else None,
        "total_time_ms": total_time * 1000,
        "token_count": len(response_tokens),
        "response": full_response
    }


def main():
    parser = argparse.ArgumentParser(
        description="LLM流式推理测试程序",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--text", type=str, default=DEFAULT_TEST_TEXT,
                        help="测试文本")
    parser.add_argument("--model", type=str, default=LLM_MODEL_NAME,
                        help="LLM模型名称")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="计算设备")
    parser.add_argument("--mode", type=str, default="both",
                        choices=["streaming", "once", "both"],
                        help="测试模式: streaming(流式), once(一次性), both(两者)")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING"],
                        help="日志级别")
    parser.add_argument("--skip-warmup", action="store_true",
                        help="跳过模型预热")
    
    args = parser.parse_args()
    
    set_global_log_level(args.log_level)
    
    # 初始化LLM
    logger.info(f"加载LLM模型: {args.model} on {args.device}")
    llm_streamer = StreamLLMInference(
        model_name=args.model,
        device=args.device,
        eval_mode=False
    )
    
    # 预热
    if not args.skip_warmup:
        warmup(llm_streamer)
    
    # 运行测试
    results = {}
    
    if args.mode in ["once", "both"]:
        results["once"] = test_once_inference(llm_streamer, args.text)
    
    if args.mode in ["streaming", "both"]:
        results["streaming"] = test_streaming_inference(llm_streamer, args.text)
    
    # 对比结果
    if args.mode == "both" and results.get("once") and results.get("streaming"):
        logger.info("=" * 60)
        logger.info("对比结果")
        logger.info("=" * 60)
        once_first = results["once"]["first_token_ms"] or 0
        stream_first = results["streaming"]["first_token_ms"] or 0
        diff = once_first - stream_first
        logger.info(f"一次性首Token: {once_first:.2f}ms")
        logger.info(f"流式首Token: {stream_first:.2f}ms (缓存后)")
        logger.info(f"流式优势: {diff:.2f}ms")


if __name__ == "__main__":
    main()
