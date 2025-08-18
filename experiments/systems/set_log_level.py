#!/usr/bin/env python3
"""
日志级别配置工具
用于在运行实验程序时动态设置日志级别
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logging_utils import set_global_log_level, get_logger

def set_log_level_for_experiment(level="DEBUG"):
    """
    为实验设置日志级别
    
    Args:
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # 设置全局日志级别
    set_global_log_level(level)
    
    # 设置环境变量，影响后续启动的程序
    os.environ['LOG_LEVEL'] = level
    
    print(f"✅ 日志级别已设置为: {level}")
    print("这将影响所有StreamLLM组件的日志输出")
    
    # 测试日志输出
    logger = get_logger("test_logger")
    logger.debug("这是DEBUG级别日志")
    logger.info("这是INFO级别日志")
    logger.warning("这是WARNING级别日志")
    logger.error("这是ERROR级别日志")
    
    print(f"\n如果看到上述日志输出，说明日志配置生效")
    print(f"当前级别 {level} 及以上的日志会显示")

def main():
    """主函数"""
    print("StreamLLM 日志级别配置工具")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        level = sys.argv[1].upper()
        if level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            print(f"❌ 无效的日志级别: {level}")
            print("有效的级别: DEBUG, INFO, WARNING, ERROR, CRITICAL")
            sys.exit(1)
    else:
        # 交互式选择
        print("请选择日志级别:")
        print("1. DEBUG   - 显示所有日志（最详细）")
        print("2. INFO    - 显示一般信息及以上")
        print("3. WARNING - 显示警告及以上")
        print("4. ERROR   - 只显示错误")
        print("5. CRITICAL - 只显示严重错误")
        
        choice = input("\n请输入选项 (1-5) [默认: 1]: ").strip()
        
        if choice == "" or choice == "1":
            level = "DEBUG"
        elif choice == "2":
            level = "INFO"
        elif choice == "3":
            level = "WARNING"
        elif choice == "4":
            level = "ERROR"
        elif choice == "5":
            level = "CRITICAL"
        else:
            print("❌ 无效选择，使用默认DEBUG级别")
            level = "DEBUG"
    
    print(f"\n设置日志级别为: {level}")
    set_log_level_for_experiment(level)
    
    print(f"\n💡 使用方法:")
    print(f"   export LOG_LEVEL={level}")
    print(f"   python system_a_baseline.py")
    print(f"\n或者在Python代码中:")
    print(f"   from src.utils.logging_utils import set_global_log_level")
    print(f"   set_global_log_level('{level}')")

if __name__ == "__main__":
    main()