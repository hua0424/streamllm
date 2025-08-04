#!/usr/bin/env python3
"""
测试统一日志系统的脚本
"""

import argparse
from src.utils.logging_utils import get_logger, set_global_log_level

def test_logging():
    """测试日志功能"""
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='测试统一日志系统')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='日志级别')
    
    args = parser.parse_args()
    
    # 设置全局日志级别
    set_global_log_level(args.log_level)
    
    # 获取不同模块的logger
    main_logger = get_logger(__name__)
    test_logger1 = get_logger('test_module1')
    test_logger2 = get_logger('test_module2')
    
    # 测试不同级别的日志
    main_logger.debug("这是一条DEBUG级别的日志")
    main_logger.info("这是一条INFO级别的日志")
    main_logger.warning("这是一条WARNING级别的日志")
    main_logger.error("这是一条ERROR级别的日志")
    main_logger.critical("这是一条CRITICAL级别的日志")
    
    print("-" * 60)
    
    # 测试其他模块的logger
    test_logger1.debug("模块1: DEBUG日志")
    test_logger1.info("模块1: INFO日志")
    test_logger1.warning("模块1: WARNING日志")
    
    test_logger2.debug("模块2: DEBUG日志")
    test_logger2.info("模块2: INFO日志")
    test_logger2.error("模块2: ERROR日志")
    
    print("-" * 60)
    print(f"当前日志级别: {args.log_level}")
    print("测试完成！")

if __name__ == "__main__":
    test_logging()
