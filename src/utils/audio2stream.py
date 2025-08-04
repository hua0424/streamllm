import numpy as np
import librosa
import time
import logging
from typing import Generator, Tuple
from src.utils.logging_utils import get_logger

# 获取当前模块的logger
logger = get_logger(__name__)

def wav2stream(
    wav_path: str,
    chunk_duration: float = 0.5,
    sample_rate: int = 16000,
    simulate_delay: bool = False
) -> Generator[Tuple[np.ndarray, bool], None, None]:
    """
    将WAV文件转换为音频块流的生成器
    
    Args:
        wav_path: WAV文件路径
        chunk_duration: 每个音频块的时长(秒)，默认0.5秒
        sample_rate: 音频采样率，默认16000Hz
        simulate_delay: 是否模拟流式延迟，如果为True会在每个chunk前等待相应时间
        
    Yields:
        Tuple[np.ndarray, bool]: (音频数据, 是否为最后一个chunk)
        
    Raises:
        FileNotFoundError: 如果WAV文件不存在
        ValueError: 如果音频文件无法加载
    """
    try:
        # 加载音频文件
        audio_data, sr = librosa.load(wav_path, sr=sample_rate, mono=True)
        logger.info(f"音频加载成功: {wav_path}, 时长 {len(audio_data)/sr:.2f}s, 采样率 {sr}Hz")
        
    except FileNotFoundError:
        logger.error(f"WAV文件不存在: {wav_path}")
        raise FileNotFoundError(f"WAV文件不存在: {wav_path}")
    except Exception as e:
        logger.error(f"无法加载音频文件 {wav_path}: {str(e)}")
        raise ValueError(f"无法加载音频文件 {wav_path}: {str(e)}")
    
    # 计算每个chunk的样本数
    chunk_size = int(chunk_duration * sample_rate)
    total_duration = len(audio_data) / sample_rate
    
    delay_info = "启用延迟模拟" if simulate_delay else "无延迟"
    logger.info(f"开始生成音频流，每个chunk {chunk_duration}s ({chunk_size}样本), {delay_info}")
    
    # 生成音频块
    for i in range(0, len(audio_data), chunk_size):
        # 模拟延迟
        if simulate_delay:
            logger.debug(f"模拟延迟 {chunk_duration:.2f}s...")
            time.sleep(chunk_duration)
        
        # 获取当前chunk
        chunk = audio_data[i:i + chunk_size]
        
        # 计算时间信息（仅用于日志）
        start_time = i / sample_rate
        end_time = (i + len(chunk)) / sample_rate
        
        # 判断是否为最后一个chunk
        is_last = (i + chunk_size >= len(audio_data))
        
        logger.debug(f"生成音频块: [{start_time:.2f}s-{end_time:.2f}s] "
                    f"样本数{len(chunk)}, 最后块:{is_last}")
        
        # 返回简化的tuple: (音频数据, 是否最后一个chunk)
        yield (chunk, is_last)

def main():
    """测试函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='测试wav2stream函数')
    parser.add_argument('--wav_path', type=str, required=True, help='WAV文件路径')
    parser.add_argument('--chunk_duration', type=float, default=0.5, help='音频块时长(秒)')
    parser.add_argument('--simulate_delay', action='store_true', help='是否模拟流式延迟')
    parser.add_argument('--log_level', type=str, default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='日志级别')
    
    args = parser.parse_args()
    
    # 设置全局日志级别
    from src.utils.logging_utils import set_global_log_level
    set_global_log_level(args.log_level)
    
    logger.info(f"测试wav2stream函数")
    logger.info(f"WAV文件: {args.wav_path}")
    logger.info(f"音频块时长: {args.chunk_duration}s")
    logger.info(f"模拟延迟: {args.simulate_delay}")
    logger.info(f"日志级别: {args.log_level}")
    logger.info("-" * 60)
    
    try:
        # 创建音频流生成器
        chunk_generator = wav2stream(
            args.wav_path, 
            args.chunk_duration,
            simulate_delay=args.simulate_delay
        )
        
        # 遍历所有音频块
        chunk_count = 0
        start_time = time.time()
        
        for chunk_data, is_last in chunk_generator:
            chunk_count += 1
            current_time = time.time() - start_time
            
            logger.info(f"处理第{chunk_count}个音频块 (耗时: {current_time:.2f}s):")
            logger.debug(f"  样本数: {len(chunk_data)}")
            logger.debug(f"  数据类型: {chunk_data.dtype}")
            logger.debug(f"  数据范围: [{chunk_data.min():.4f}, {chunk_data.max():.4f}]")
            logger.debug(f"  是否最后块: {is_last}")
            
            if is_last:
                logger.info("🎉 检测到最后一个音频块，流结束")
                break
        
        total_time = time.time() - start_time
        logger.info(f"总共生成了 {chunk_count} 个音频块，总耗时: {total_time:.2f}s")
        
    except Exception as e:
        logger.error(f"测试失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
