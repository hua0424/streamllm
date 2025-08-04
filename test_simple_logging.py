#!/usr/bin/env python3
"""
简单测试各个模块的日志功能
"""

import argparse
from src.utils.logging_utils import get_logger, set_global_log_level

def test_module_logging():
    """测试各个模块的日志功能"""
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='测试各模块日志功能')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='日志级别')
    
    args = parser.parse_args()
    
    # 设置全局日志级别
    set_global_log_level(args.log_level)
    
    # 获取主logger
    main_logger = get_logger(__name__)
    main_logger.info(f"设置全局日志级别为: {args.log_level}")
    
    # 测试导入各个模块并获取它们的logger
    try:
        main_logger.info("测试ASR模块日志...")
        from src.asr.audio_segmenter import logger as asr_segmenter_logger
        asr_segmenter_logger.debug("ASR音频分段器: DEBUG日志")
        asr_segmenter_logger.info("ASR音频分段器: INFO日志")
        asr_segmenter_logger.warning("ASR音频分段器: WARNING日志")
    except Exception as e:
        main_logger.error(f"ASR模块测试失败: {e}")
    
    try:
        main_logger.info("测试LLM模块日志...")
        from src.llm.stream_llm_inference import logger as llm_logger
        llm_logger.debug("LLM推理模块: DEBUG日志")
        llm_logger.info("LLM推理模块: INFO日志")
        llm_logger.warning("LLM推理模块: WARNING日志")
    except Exception as e:
        main_logger.error(f"LLM模块测试失败: {e}")
    
    try:
        main_logger.info("测试音频工具模块日志...")
        from src.utils.audio2stream import logger as audio_logger
        audio_logger.debug("音频流工具: DEBUG日志")
        audio_logger.info("音频流工具: INFO日志")
        audio_logger.warning("音频流工具: WARNING日志")
    except Exception as e:
        main_logger.error(f"音频工具模块测试失败: {e}")
    
    try:
        main_logger.info("测试流水线模块日志...")
        from src.pipeline.streaming_pipeline import logger as pipeline_logger
        pipeline_logger.debug("流水线模块: DEBUG日志")
        pipeline_logger.info("流水线模块: INFO日志")
        pipeline_logger.warning("流水线模块: WARNING日志")
    except Exception as e:
        main_logger.error(f"流水线模块测试失败: {e}")
    
    main_logger.info("所有模块日志测试完成！")

if __name__ == "__main__":
    test_module_logging()
