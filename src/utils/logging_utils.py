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

        # 生产环境优化的格式：更简洁
        formatter = logging.Formatter(
            '%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # 控制台输出 handler - 根据级别调整输出
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        
        # 生产环境：ERROR及以上级别输出到stderr，其他输出到stdout
        if _level in ['ERROR', 'CRITICAL']:
            ch = logging.StreamHandler(sys.stderr)
            ch.setFormatter(formatter)
        
        logger.addHandler(ch)

        # 文件输出 handler (如果指定了log_file)
        if _log_file:
            try:
                # 确保日志目录存在
                log_dir = os.path.dirname(_log_file)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)
                
                fh = logging.FileHandler(_log_file, mode='a', encoding='utf-8')
                fh.setFormatter(formatter)
                logger.addHandler(fh)
            except Exception as e:
                # 降级处理：仅控制台输出
                print(f"WARNING: Failed to create log file {_log_file}: {e}", file=sys.stderr)
    else:
        # 如果logger已存在但需要更新级别
        logger.setLevel(getattr(logging, _level, logging.INFO))

    return logger

# 生产环境工具函数
def silence_external_loggers():
    """静默外部库的冗余日志输出"""
    external_loggers = [
        'transformers',
        'torch',
        'librosa',
        'faster_whisper',
        'httpx',
        'urllib3',
        'huggingface_hub'
    ]
    
    for logger_name in external_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

def configure_production_logging():
    """配置生产环境日志"""
    # 静默外部库日志
    silence_external_loggers()
    
    # 设置根日志级别
    logging.getLogger().setLevel(logging.INFO)
    
    # 禁用不必要的警告
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
    warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")


# 测试用法 (仅在直接运行时执行)
if __name__ == '__main__':
    # 测试基本功能
    logger = get_logger(__name__)
    logger.info("Logger test: INFO level")
    logger.debug("Logger test: DEBUG level")
    logger.warning("Logger test: WARNING level")
    
    # 测试生产环境配置
    configure_production_logging()
    logger.info("Production logging configured") 