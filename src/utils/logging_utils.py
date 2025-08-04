# src/utils/logging_utils.py

import logging
import sys
import os

from src.config import LOG_LEVEL, LOG_FILE

# 全局日志级别变量，用于运行时动态设置
_global_log_level = None

# 确保日志目录存在
if LOG_FILE:
    log_dir = os.path.dirname(LOG_FILE)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

def set_global_log_level(level):
    """
    设置全局日志级别，影响所有通过get_logger创建的logger

    Args:
        level (str): 日志级别 (e.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
    """
    global _global_log_level
    _global_log_level = level.upper()

    # 更新根logger的级别
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, _global_log_level, logging.INFO))

    # 更新所有已存在的logger的级别
    for name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(name)
        if logger.handlers:  # 只更新已配置的logger
            logger.setLevel(getattr(logging, _global_log_level, logging.INFO))

def get_logger(name, level=None, log_file=None):
    """
    获取一个配置好的logger实例。

    Args:
        name (str): logger的名称，通常是 __name__。
        level (str, optional): 日志级别 (e.g., 'DEBUG', 'INFO')。如果设置了全局级别，则优先使用全局级别。
        log_file (str, optional): 日志文件路径。默认为 config.LOG_FILE。

    Returns:
        logging.Logger: 配置好的logger实例。
    """
    logger = logging.getLogger(name)

    # 优先使用全局日志级别，然后是参数级别，最后是配置文件级别
    if _global_log_level:
        _level = _global_log_level
    elif level:
        _level = level.upper()
    else:
        _level = LOG_LEVEL.upper()

    _log_file = log_file if log_file else LOG_FILE

    # 防止重复添加 handlers
    if not logger.handlers:
        logger.setLevel(getattr(logging, _level, logging.INFO))

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # 控制台输出 handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # 文件输出 handler (如果指定了log_file)
        if _log_file:
            try:
                fh = logging.FileHandler(_log_file, mode='a', encoding='utf-8')
                fh.setFormatter(formatter)
                logger.addHandler(fh)
            except Exception as e:
                logger.error(f"Failed to create file handler for log file {_log_file}: {e}", exc_info=False)
                # 不要因为日志文件创建失败而崩溃，但要打印错误到控制台
                # 此时，logger只有控制台handler
    else:
        # 如果logger已存在但需要更新级别
        logger.setLevel(getattr(logging, _level, logging.INFO))

    return logger

# 示例用法:
if __name__ == '__main__':
    # 获取默认配置的logger
    default_logger = get_logger(__name__)
    default_logger.debug("This is a debug message (default config).")
    default_logger.info("This is an info message (default config).")
    default_logger.warning("This is a warning message (default config).")

    # 获取一个自定义级别的logger，不输出到文件
    custom_logger = get_logger("MyCustomLogger", level="DEBUG", log_file=None)
    custom_logger.debug("This is a debug message from custom_logger.")
    custom_logger.info("This is an info message from custom_logger (should not go to file).")

    # 测试日志文件输出 (如果 LOG_FILE 在 config.py 中被设置)
    if LOG_FILE:
        file_logger = get_logger("FileTestLogger", level="INFO", log_file="logs/test_specific.log")
        file_logger.info(f"This message should go to logs/test_specific.log and console.")
        default_logger.info(f"This message from default_logger should go to {LOG_FILE} and console.")
        print(f"Check logs in '{os.path.dirname(LOG_FILE)}' and specifically 'logs/test_specific.log' if LOG_FILE is set.")
    else:
        print("LOG_FILE not set in config, file logging tests skipped.") 