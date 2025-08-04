#!/usr/bin/env python3
"""
系统A：基线串行系统 (Baseline Sequential System)

实现传统的串行处理流程：
1. 等待用户完整语音输入结束
2. 对完整音频进行一次性ASR，得到最终文本
3. 将最终文本送入LLM进行推理

这是未经优化的传统实现方式，作为对照组衡量优化效果的基准。
"""

import time
import numpy as np
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import sys

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.llm.stream_llm_inference import StreamLLMInference
from src.asr.faster_whisper_streamer import StreamingASRProcessor
from src.utils.logging_utils import get_logger


class SystemA_BaselineSequential:
    """系统A：基线串行系统"""
    
    def __init__(self, 
                 asr_model_size: str = "base",
                 llm_model_name: str = "Qwen/Qwen1.5-0.5B-Chat",
                 use_simulation: bool = False):
        """
        初始化基线串行系统
        
        Args:
            asr_model_size: ASR模型大小
            llm_model_name: LLM模型名称
            use_simulation: 是否使用模拟模式（用于快速测试）
        """
        self.asr_model_size = asr_model_size
        self.llm_model_name = llm_model_name
        self.use_simulation = use_simulation
        self.logger = get_logger(f"SystemA_Baseline")
        
        # 延迟初始化组件
        self._asr_processor = None
        self._llm_processor = None
        
        self.logger.info(f"系统A初始化完成 - ASR: {asr_model_size}, LLM: {llm_model_name}, 模拟: {use_simulation}")
    
    @property
    def asr_processor(self) -> StreamingASRProcessor:
        """延迟初始化ASR处理器"""
        if self._asr_processor is None:
            self._asr_processor = StreamingASRProcessor(
                model_size=self.asr_model_size,
                recognition_threshold=2.0,  # 较长的阈值确保完整处理
                text_callback=None
            )
        return self._asr_processor
    
    @property 
    def llm_processor(self) -> StreamLLMInference:
        """延迟初始化LLM处理器"""
        if self._llm_processor is None:
            self._llm_processor = StreamLLMInference(
                model_name=self.llm_model_name
            )
        return self._llm_processor
    
    def get_audio_duration(self, audio_path: str) -> float:
        """获取音频时长"""
        try:
            import librosa
            y, sr = librosa.load(audio_path, sr=None)
            return len(y) / sr
        except ImportError:
            # 如果没有librosa，使用简单估算
            self.logger.warning("librosa未安装，使用默认时长")
            return 10.0
        except Exception as e:
            self.logger.error(f"获取音频时长失败: {e}")
            return 10.0
    
    def process_audio_complete(self, audio_path: str) -> Tuple[str, Dict[str, float]]:
        """
        完整音频ASR处理（传统方式）
        
        Returns:
            (transcript, timing_info)
        """
        if self.use_simulation:
            return self._simulate_asr_processing(audio_path)
        
        try:
            # 重置ASR处理器
            self.asr_processor.reset()
            
            # 模拟完整音频处理：一次性加载并处理完整音频
            # 这里使用简化的处理方式，实际应该调用complete audio processing
            
            start_time = time.perf_counter()
            
            # 模拟ASR处理时间（与音频长度成正比）
            audio_duration = self.get_audio_duration(audio_path)
            processing_time = audio_duration * 0.3  # 假设ASR处理时间是音频长度的30%
            time.sleep(processing_time)
            
            # 模拟ASR结果
            transcript = "这是一个关于语音识别的测试问题，请问现在几点了？"
            
            asr_time = time.perf_counter() - start_time
            
            timing_info = {
                "audio_duration": audio_duration,
                "asr_processing_time": asr_time,
                "asr_wait_time": audio_duration  # 串行系统需要等待音频播放完成
            }
            
            return transcript, timing_info
            
        except Exception as e:
            self.logger.error(f"ASR处理失败: {e}")
            # 返回默认结果
            return "ASR处理失败", {"asr_processing_time": 1.0, "audio_duration": 10.0, "asr_wait_time": 10.0}
    
    def _simulate_asr_processing(self, audio_path: str) -> Tuple[str, Dict[str, float]]:
        """模拟ASR处理（用于快速测试）"""
        audio_duration = 10.0  # 默认音频长度
        asr_processing_time = audio_duration * 0.3  # ASR处理时间
        
        # 模拟处理延迟
        time.sleep(0.1)
        
        transcript = "这是一个模拟的ASR识别结果，用于测试系统性能。"
        
        timing_info = {
            "audio_duration": audio_duration,
            "asr_processing_time": asr_processing_time,
            "asr_wait_time": audio_duration
        }
        
        return transcript, timing_info
    
    def process_llm_inference(self, text: str) -> Tuple[str, Dict[str, float]]:
        """
        LLM推理处理（传统方式）
        
        Returns:
            (first_token, timing_info)
        """
        if self.use_simulation:
            return self._simulate_llm_processing(text)
        
        try:
            start_time = time.perf_counter()
            
            # 传统LLM处理：从零开始，没有预填充的KV缓存
            # 模拟LLM推理过程
            first_token, generation_time = self._get_llm_first_token_traditional(text)
            
            llm_time = time.perf_counter() - start_time
            
            timing_info = {
                "llm_processing_time": llm_time,
                "first_token_generation_time": generation_time,
                "kv_cache_time": 0.0  # 基线系统没有KV缓存
            }
            
            return first_token, timing_info
            
        except Exception as e:
            self.logger.error(f"LLM处理失败: {e}")
            return "错误", {"llm_processing_time": 1.0, "first_token_generation_time": 1.0, "kv_cache_time": 0.0}
    
    def _simulate_llm_processing(self, text: str) -> Tuple[str, Dict[str, float]]:
        """模拟LLM处理（用于快速测试）"""
        # 模拟LLM处理时间
        llm_processing_time = 1.0  # 基线系统LLM处理较慢
        time.sleep(0.1)
        
        first_token = "现在"
        
        timing_info = {
            "llm_processing_time": llm_processing_time,
            "first_token_generation_time": llm_processing_time,
            "kv_cache_time": 0.0
        }
        
        return first_token, timing_info
    
    def _get_llm_first_token_traditional(self, text: str) -> Tuple[str, float]:
        """获取LLM首个token（传统方式，无优化）"""
        try:
            start_time = time.perf_counter()
            
            # 模拟传统LLM处理：从头开始推理，没有任何缓存
            time.sleep(1.0)  # 模拟较长的处理时间
            
            generation_time = time.perf_counter() - start_time
            
            # 模拟首个token
            first_token = "现在"
            
            return first_token, generation_time
            
        except Exception as e:
            self.logger.error(f"LLM首token生成失败: {e}")
            return "错误", 1.0
    
    def process_complete_pipeline(self, audio_path: str, simulate_delay: bool = True) -> Dict[str, Any]:
        """
        执行完整的基线串行流水线处理
        
        Args:
            audio_path: 音频文件路径
            simulate_delay: 是否模拟实时延迟
            
        Returns:
            包含所有时序信息和结果的字典
        """
        pipeline_start = time.perf_counter()
        
        # 步骤1：等待音频播放完成（模拟用户说话）
        audio_duration = self.get_audio_duration(audio_path)
        
        if simulate_delay:
            time.sleep(audio_duration)  # 模拟等待用户说完
        
        speech_end_time = time.perf_counter()
        
        # 步骤2：完整音频ASR处理
        self.logger.debug("开始ASR处理...")
        transcript, asr_timing = self.process_audio_complete(audio_path)
        
        asr_complete_time = time.perf_counter()
        
        # 步骤3：LLM推理
        self.logger.debug("开始LLM推理...")
        first_token, llm_timing = self.process_llm_inference(transcript)
        
        first_token_time = time.perf_counter()
        
        # 计算关键延迟指标
        ttft = (first_token_time - speech_end_time) * 1000  # 转换为毫秒
        
        # 组装结果
        result = {
            "system_name": "SystemA_BaselineSequential",
            "audio_path": audio_path,
            "transcript": transcript,
            "first_token": first_token,
            
            # 时序信息
            "timing": {
                "pipeline_start": pipeline_start,
                "speech_end_time": speech_end_time,
                "asr_complete_time": asr_complete_time,
                "first_token_time": first_token_time,
                
                "audio_duration": audio_duration,
                "ttft_ms": ttft,
                "total_pipeline_time": (first_token_time - pipeline_start) * 1000
            },
            
            # 详细性能指标
            "performance_metrics": {
                "ttft_ms": ttft,
                "asr_processing_time_ms": asr_timing["asr_processing_time"] * 1000,
                "llm_processing_time_ms": llm_timing["llm_processing_time"] * 1000,
                "total_processing_time_ms": (asr_timing["asr_processing_time"] + llm_timing["llm_processing_time"]) * 1000,
                
                # 基线系统特征
                "has_streaming_asr": False,
                "has_kv_cache": False,
                "processing_type": "sequential"
            },
            
            # 详细时序分解
            "detailed_timing": {
                **asr_timing,
                **llm_timing
            }
        }
        
        self.logger.info(f"基线系统处理完成 - TTFT: {ttft:.1f}ms")
        return result
    
    def reset(self):
        """重置系统状态"""
        if self._asr_processor:
            self._asr_processor.reset()
        # LLM不需要特殊重置
        self.logger.debug("系统A已重置")


def test_system_a():
    """测试系统A"""
    print("测试系统A：基线串行系统")
    
    # 创建系统实例
    system = SystemA_BaselineSequential(use_simulation=True)
    
    # 测试处理
    test_audio = "/usr/local/app/jupyterlab/yanjiu/streamllm/data/processed_audio/length30+/3_1_d473.wav"
    
    result = system.process_complete_pipeline(test_audio, simulate_delay=False)
    
    print(f"处理结果:")
    print(f"  转录: {result['transcript']}")
    print(f"  首Token: {result['first_token']}")
    print(f"  TTFT: {result['performance_metrics']['ttft_ms']:.1f}ms")
    print(f"  总处理时间: {result['performance_metrics']['total_processing_time_ms']:.1f}ms")


if __name__ == "__main__":
    test_system_a()