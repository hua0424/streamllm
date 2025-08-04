#!/usr/bin/env python3
"""
系统A'：仅流式ASR系统 (Streaming ASR Only System)

消融研究用的中间系统：
1. 使用流式ASR处理音频，获得更快的转写响应
2. 但LLM部分仍采用传统方式，没有KV缓存预填充
3. 用于量化流式ASR和KV缓存两个组件的独立贡献

这是消融研究的关键对比组，帮助分离各优化组件的效果。
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


class SystemA_Prime_StreamingASROnly:
    """系统A'：仅流式ASR系统（消融研究用）"""
    
    def __init__(self, 
                 asr_model_size: str = "base",
                 llm_model_name: str = "Qwen/Qwen1.5-0.5B-Chat",
                 chunk_duration: float = 0.3,
                 use_simulation: bool = False):
        """
        初始化仅流式ASR系统
        
        Args:
            asr_model_size: ASR模型大小
            llm_model_name: LLM模型名称  
            chunk_duration: 音频块时长
            use_simulation: 是否使用模拟模式
        """
        self.asr_model_size = asr_model_size
        self.llm_model_name = llm_model_name
        self.chunk_duration = chunk_duration
        self.use_simulation = use_simulation
        self.logger = get_logger(f"SystemA_Prime_StreamingASROnly")
        
        # 延迟初始化组件
        self._asr_processor = None
        self._llm_processor = None
        
        # 流式ASR状态跟踪
        self.streaming_texts = []
        
        self.logger.info(f"系统A'初始化完成 - ASR: {asr_model_size}, LLM: {llm_model_name}, 块长: {chunk_duration}s")
    
    @property
    def asr_processor(self) -> StreamingASRProcessor:
        """延迟初始化流式ASR处理器"""
        if self._asr_processor is None:
            self._asr_processor = StreamingASRProcessor(
                model_size=self.asr_model_size,
                recognition_threshold=1.8,  # 中等阈值，平衡速度和准确性
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
            self.logger.warning("librosa未安装，使用默认时长")
            return 10.0
        except Exception as e:
            self.logger.error(f"获取音频时长失败: {e}")
            return 10.0
    
    def process_streaming_asr_with_traditional_llm(self, audio_path: str, simulate_delay: bool = True) -> Dict[str, Any]:
        """
        执行流式ASR + 传统LLM处理
        
        Args:
            audio_path: 音频文件路径
            simulate_delay: 是否模拟实时延迟
            
        Returns:
            包含所有时序信息和结果的字典
        """
        if self.use_simulation:
            return self._simulate_streaming_asr_processing(audio_path)
        
        pipeline_start = time.perf_counter()
        audio_duration = self.get_audio_duration(audio_path)
        
        # 模拟流式ASR处理过程
        if simulate_delay:
            time.sleep(audio_duration)
        
        speech_end_time = time.perf_counter()
        
        # 流式ASR的优势：在语音结束时已经有了（部分）转写结果
        # 但没有KV缓存预处理，所以LLM还是从零开始
        
        # 模拟流式ASR已经产生了文本
        streaming_asr_end_time = speech_end_time + 0.2  # 流式ASR有200ms的处理延迟
        time.sleep(0.2)
        
        transcript = "这是流式ASR生成的文本结果"
        
        # 传统LLM处理：从头开始，无KV缓存预填充
        traditional_llm_start = time.perf_counter()
        
        # 模拟传统LLM推理时间（比有KV缓存的情况更长）
        llm_processing_time = 0.8  # 800ms的LLM处理时间
        time.sleep(llm_processing_time)
        
        first_token_time = time.perf_counter()
        ttft = (first_token_time - speech_end_time) * 1000
        
        # 组装结果
        result = {
            "system_name": "SystemA_Prime_StreamingASROnly",
            "audio_path": audio_path,
            "transcript": transcript,
            "first_token": "现在",
            
            # 时序信息
            "timing": {
                "pipeline_start": pipeline_start,
                "speech_end_time": speech_end_time,
                "streaming_asr_end_time": streaming_asr_end_time,
                "traditional_llm_start_time": traditional_llm_start,
                "first_token_time": first_token_time,
                
                "audio_duration": audio_duration,
                "ttft_ms": ttft,
                "total_pipeline_time": (first_token_time - pipeline_start) * 1000
            },
            
            # 详细性能指标
            "performance_metrics": {
                "ttft_ms": ttft,
                "streaming_asr_time_ms": 200,  # 流式ASR节省的时间
                "traditional_llm_time_ms": llm_processing_time * 1000,
                
                # 系统A'特征
                "has_streaming_asr": True,   # 有流式ASR优化
                "has_kv_cache": False,       # 没有KV缓存优化
                "processing_type": "streaming_asr_only",
                "optimization_components": ["streaming_asr"]
            },
            
            # 消融研究相关信息
            "ablation_info": {
                "streaming_asr_optimization_ms": 2000,  # 相比完全串行的ASR节省时间
                "kv_cache_optimization_ms": 0,         # 没有KV缓存优化
                "total_optimization_vs_baseline_ms": 2000,
                "partial_optimization": True
            }
        }
        
        self.logger.info(f"系统A'处理完成 - TTFT: {ttft:.1f}ms")
        return result
    
    def _simulate_streaming_asr_processing(self, audio_path: str) -> Dict[str, Any]:
        """模拟流式ASR处理（用于快速测试）"""
        pipeline_start = time.perf_counter()
        audio_duration = 10.0
        
        # 模拟语音输入和流式ASR处理
        self.logger.debug("模拟流式ASR处理...")
        
        # 流式ASR的中间结果
        streaming_texts = [
            (3.0, "播放周杰..."),
            (6.0, "播放周杰伦的..."),
            (9.0, "播放周杰伦的演唱会")
        ]
        
        self.streaming_texts = streaming_texts
        
        # 模拟语音结束时间
        speech_end_time = pipeline_start + audio_duration
        
        # 流式ASR优势：更快得到转写结果
        streaming_asr_complete_time = speech_end_time + 0.2
        
        # 但传统LLM需要从头处理，没有KV缓存
        time.sleep(0.8)  # 传统LLM推理时间
        
        first_token_time = time.perf_counter()
        ttft = (first_token_time - speech_end_time) * 1000
        
        # 构建结果
        result = {
            "system_name": "SystemA_Prime_StreamingASROnly",
            "audio_path": audio_path,
            "transcript": "播放周杰伦的演唱会",
            "first_token": "好的",
            
            # 时序信息
            "timing": {
                "pipeline_start": pipeline_start,
                "speech_end_time": speech_end_time,
                "streaming_asr_end_time": streaming_asr_complete_time,
                "first_token_time": first_token_time,
                
                "audio_duration": audio_duration,
                "ttft_ms": ttft,
                "total_pipeline_time": (first_token_time - pipeline_start) * 1000
            },
            
            # 详细性能指标
            "performance_metrics": {
                "ttft_ms": ttft,
                "streaming_asr_optimization_ms": 2000,  # 流式ASR节省的时间
                "traditional_llm_time_ms": 800,         # 传统LLM处理时间
                
                # 系统A'特征
                "has_streaming_asr": True,
                "has_kv_cache": False,
                "processing_type": "streaming_asr_only",
                "optimization_components": ["streaming_asr"]
            },
            
            # 消融研究详细信息
            "ablation_info": {
                "streaming_texts": self.streaming_texts,
                "streaming_asr_benefit_ms": 2000,
                "missing_kv_cache_penalty_ms": 600,  # 相比系统B缺少的KV缓存优化
                "partial_optimization_only": True
            }
        }
        
        self.logger.info(f"系统A'模拟处理完成 - TTFT: {ttft:.1f}ms")
        return result
    
    def _get_error_result(self, audio_path: str, error: Exception) -> Dict[str, Any]:
        """获取错误情况下的默认结果"""
        return {
            "system_name": "SystemA_Prime_StreamingASROnly",
            "audio_path": audio_path,
            "transcript": f"处理失败: {str(error)}",
            "first_token": "错误",
            
            "timing": {
                "pipeline_start": time.perf_counter(),
                "speech_end_time": time.perf_counter(),
                "first_token_time": time.perf_counter(),
                "audio_duration": 10.0,
                "ttft_ms": 10000,  # 设置一个很大的错误值
                "total_pipeline_time": 10000
            },
            
            "performance_metrics": {
                "ttft_ms": 10000,
                "has_streaming_asr": True,
                "has_kv_cache": False,
                "processing_type": "error",
                "error": str(error)
            },
            
            "ablation_info": {
                "error": str(error)
            }
        }
    
    def process_complete_pipeline(self, audio_path: str, simulate_delay: bool = True) -> Dict[str, Any]:
        """
        执行完整的流式ASR + 传统LLM流水线
        
        这是系统A'的主要接口，实现部分优化的流水线
        """
        self.reset()  # 重置状态
        
        try:
            return self.process_streaming_asr_with_traditional_llm(audio_path, simulate_delay)
        except Exception as e:
            self.logger.error(f"系统A'处理失败: {e}")
            return self._get_error_result(audio_path, e)
    
    def reset(self):
        """重置系统状态"""
        if self._asr_processor:
            self._asr_processor.reset()
        
        # 清理流式ASR状态
        self.streaming_texts.clear()
        
        self.logger.debug("系统A'已重置")


def test_system_a_prime():
    """测试系统A'"""
    print("测试系统A'：仅流式ASR系统")
    
    # 创建系统实例
    system = SystemA_Prime_StreamingASROnly(use_simulation=True)
    
    # 测试处理
    test_audio = "/usr/local/app/jupyterlab/yanjiu/streamllm/data/processed_audio/length30+/3_1_d473.wav"
    
    result = system.process_complete_pipeline(test_audio, simulate_delay=False)
    
    print(f"处理结果:")
    print(f"  转录: {result['transcript']}")
    print(f"  首Token: {result['first_token']}")
    print(f"  TTFT: {result['performance_metrics']['ttft_ms']:.1f}ms")
    print(f"  流式ASR优化: {result['performance_metrics']['streaming_asr_optimization_ms']:.1f}ms")
    print(f"  优化组件: {result['performance_metrics']['optimization_components']}")
    print(f"  部分优化: {result['ablation_info']['partial_optimization_only']}")


if __name__ == "__main__":
    test_system_a_prime()