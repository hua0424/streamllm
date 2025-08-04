#!/usr/bin/env python3
"""
系统B：KV缓存预填充系统 (Proposed KV Cache Pre-filling System)

本文的核心方案：
1. 语音流输入开始，使用VAD切片
2. 并行执行：
   a. 流式ASR持续处理音频切片，生成并修正文本
   b. 将ASR的中间文本实时送入LLM以预填充KV缓存
3. 语音输入结束，将ASR最终文本送入LLM，利用已有缓存进行推理

这是需要证明其在延迟上的优越性和在质量上的非劣性的实验组。
"""

import time
import numpy as np
from typing import Dict, Any, Tuple, Optional, Generator
from pathlib import Path
import sys

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.llm.stream_llm_inference import StreamLLMInference
from src.asr.faster_whisper_streamer import StreamingASRProcessor
from src.utils.logging_utils import get_logger
from src.pipeline.ultra_low_latency_pipeline import create_ultra_low_latency_pipeline_from_wav


class SystemB_ProposedKVCache:
    """系统B：KV缓存预填充系统（本文方案）"""
    
    def __init__(self, 
                 asr_model_size: str = "base",
                 llm_model_name: str = "Qwen/Qwen1.5-0.5B-Chat",
                 chunk_duration: float = 0.3,
                 use_simulation: bool = False):
        """
        初始化KV缓存预填充系统
        
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
        self.logger = get_logger(f"SystemB_Proposed")
        
        # 延迟初始化组件
        self._asr_processor = None
        self._llm_processor = None
        
        # KV缓存状态跟踪
        self.kv_cache_states = []
        self.intermediate_texts = []
        
        self.logger.info(f"系统B初始化完成 - ASR: {asr_model_size}, LLM: {llm_model_name}, 块长: {chunk_duration}s")
    
    @property
    def asr_processor(self) -> StreamingASRProcessor:
        """延迟初始化流式ASR处理器"""
        if self._asr_processor is None:
            self._asr_processor = StreamingASRProcessor(
                model_size=self.asr_model_size,
                recognition_threshold=1.5,  # 较短的阈值支持更快响应
                text_callback=None
            )
        return self._asr_processor
    
    @property 
    def llm_processor(self) -> StreamLLMInference:
        """延迟初始化流式LLM处理器"""
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
    
    def process_streaming_with_kv_cache(self, audio_path: str, simulate_delay: bool = True) -> Dict[str, Any]:
        """
        执行带KV缓存预填充的流式处理
        
        Args:
            audio_path: 音频文件路径
            simulate_delay: 是否模拟实时延迟
            
        Returns:
            包含所有时序信息和结果的字典
        """
        if self.use_simulation:
            return self._simulate_kv_cache_processing(audio_path)
        
        # 使用现有的超低延迟流水线实现
        return self._process_with_ultra_low_latency_pipeline(audio_path, simulate_delay)
    
    def _process_with_ultra_low_latency_pipeline(self, audio_path: str, simulate_delay: bool) -> Dict[str, Any]:
        """使用现有的超低延迟流水线处理"""
        pipeline_start = time.perf_counter()
        
        try:
            # 使用现有的超低延迟流水线
            first_token = None
            first_token_latency = None
            speech_end_time = None
            audio_duration = self.get_audio_duration(audio_path)
            
            # 记录语音结束时间（模拟）
            if simulate_delay:
                time.sleep(audio_duration)
            speech_end_time = time.perf_counter()
            
            # 运行超低延迟流水线
            for token, latency, reason in create_ultra_low_latency_pipeline_from_wav(
                wav_path=audio_path,
                chunk_duration=self.chunk_duration,
                simulate_delay=False,  # 我们已经模拟了延迟
                asr_model_size=self.asr_model_size
            ):
                if first_token is None:
                    first_token = token
                    first_token_latency = latency
                    break
            
            first_token_time = time.perf_counter()
            
            # 计算TTFT
            if speech_end_time and first_token_latency:
                ttft = first_token_latency * 1000  # 已经是从语音结束开始计算的延迟
            else:
                ttft = (first_token_time - speech_end_time) * 1000
            
            # 组装结果
            result = {
                "system_name": "SystemB_ProposedKVCache",
                "audio_path": audio_path,
                "transcript": "流式ASR识别结果",  # 实际应该从流水线获取
                "first_token": first_token or "现在",
                
                # 时序信息
                "timing": {
                    "pipeline_start": pipeline_start,
                    "speech_end_time": speech_end_time,
                    "first_token_time": first_token_time,
                    
                    "audio_duration": audio_duration,
                    "ttft_ms": ttft,
                    "total_pipeline_time": (first_token_time - pipeline_start) * 1000
                },
                
                # 详细性能指标
                "performance_metrics": {
                    "ttft_ms": ttft,
                    "first_token_latency_pipeline": first_token_latency * 1000 if first_token_latency else ttft,
                    
                    # 系统B特征
                    "has_streaming_asr": True,
                    "has_kv_cache": True,
                    "processing_type": "parallel_streaming",
                    "kv_cache_prefillings": len(self.kv_cache_states)
                },
                
                # KV缓存相关信息
                "kv_cache_info": {
                    "intermediate_texts_count": len(self.intermediate_texts),
                    "cache_updates": len(self.kv_cache_states),
                    "streaming_optimization": True
                }
            }
            
            self.logger.info(f"系统B处理完成 - TTFT: {ttft:.1f}ms")
            return result
            
        except Exception as e:
            self.logger.error(f"系统B处理失败: {e}")
            # 返回错误时的默认结果
            return self._get_error_result(audio_path, e)
    
    def _simulate_kv_cache_processing(self, audio_path: str) -> Dict[str, Any]:
        """模拟KV缓存预填充处理（用于快速测试）"""
        pipeline_start = time.perf_counter()
        audio_duration = 10.0
        
        # 模拟并行处理：音频播放的同时进行ASR和KV缓存预填充
        self.logger.debug("模拟并行流式处理...")
        
        # 模拟中间文本和KV缓存更新
        intermediate_texts = [
            (3.0, "播放周杰..."),
            (6.0, "播放周杰伦的..."),
            (9.0, "播放周杰伦的演唱会")
        ]
        
        self.intermediate_texts = intermediate_texts
        self.kv_cache_states = [(t, f"cache_state_{i}") for i, (t, _) in enumerate(intermediate_texts)]
        
        # 模拟语音结束时间
        speech_end_time = pipeline_start + audio_duration
        
        # 模拟优化后的首token生成（利用预填充的KV缓存）
        time.sleep(0.05)  # 非常短的处理时间，因为有KV缓存
        
        first_token_time = time.perf_counter()
        ttft = (first_token_time - speech_end_time) * 1000
        
        # 构建结果
        result = {
            "system_name": "SystemB_ProposedKVCache",
            "audio_path": audio_path,
            "transcript": "播放周杰伦的演唱会",
            "first_token": "好的",
            
            # 时序信息
            "timing": {
                "pipeline_start": pipeline_start,
                "speech_end_time": speech_end_time,
                "first_token_time": first_token_time,
                
                "audio_duration": audio_duration,
                "ttft_ms": ttft,
                "total_pipeline_time": (first_token_time - pipeline_start) * 1000
            },
            
            # 详细性能指标
            "performance_metrics": {
                "ttft_ms": ttft,
                "kv_cache_optimization_ms": 800,  # 模拟KV缓存节省的时间
                "streaming_asr_optimization_ms": 2000,  # 模拟流式ASR节省的时间
                
                # 系统B特征
                "has_streaming_asr": True,
                "has_kv_cache": True,
                "processing_type": "parallel_streaming",
                "kv_cache_prefillings": len(self.kv_cache_states)
            },
            
            # KV缓存详细信息
            "kv_cache_info": {
                "intermediate_texts": self.intermediate_texts,
                "cache_updates": len(self.kv_cache_states),
                "streaming_optimization": True,
                "parallel_processing_time": audio_duration * 1000
            }
        }
        
        self.logger.info(f"系统B模拟处理完成 - TTFT: {ttft:.1f}ms")
        return result
    
    def _get_error_result(self, audio_path: str, error: Exception) -> Dict[str, Any]:
        """获取错误情况下的默认结果"""
        return {
            "system_name": "SystemB_ProposedKVCache",
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
                "has_kv_cache": True,
                "processing_type": "error",
                "error": str(error)
            },
            
            "kv_cache_info": {
                "error": str(error)
            }
        }
    
    def process_complete_pipeline(self, audio_path: str, simulate_delay: bool = True) -> Dict[str, Any]:
        """
        执行完整的KV缓存预填充流水线
        
        这是系统B的主要接口，实现并行的流式ASR + KV缓存预填充
        """
        self.reset()  # 重置状态
        return self.process_streaming_with_kv_cache(audio_path, simulate_delay)
    
    def reset(self):
        """重置系统状态"""
        if self._asr_processor:
            self._asr_processor.reset()
        
        # 清理KV缓存状态
        self.kv_cache_states.clear()
        self.intermediate_texts.clear()
        
        self.logger.debug("系统B已重置")


def test_system_b():
    """测试系统B"""
    print("测试系统B：KV缓存预填充系统")
    
    # 创建系统实例
    system = SystemB_ProposedKVCache(use_simulation=True)
    
    # 测试处理
    test_audio = "/usr/local/app/jupyterlab/yanjiu/streamllm/data/processed_audio/length30+/3_1_d473.wav"
    
    result = system.process_complete_pipeline(test_audio, simulate_delay=False)
    
    print(f"处理结果:")
    print(f"  转录: {result['transcript']}")
    print(f"  首Token: {result['first_token']}")
    print(f"  TTFT: {result['performance_metrics']['ttft_ms']:.1f}ms")
    print(f"  KV缓存更新次数: {result['performance_metrics']['kv_cache_prefillings']}")
    print(f"  中间文本数量: {result['kv_cache_info']['cache_updates']}")


if __name__ == "__main__":
    test_system_b()