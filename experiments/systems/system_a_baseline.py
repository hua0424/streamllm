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
import torch

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.llm.stream_llm_inference import StreamLLMInference
from src.asr.faster_whisper_streamer import StreamingASRProcessor
from src.utils.logging_utils import get_logger, set_global_log_level


class SystemA_BaselineSequential:
    """系统A：基线串行系统"""
    
    def __init__(self, 
                 asr_model_size: str = "large-v3",
                 llm_model_name: str = "Qwen/Qwen2-7B-Instruct",
                 llm_device: Optional[str] = None):
        """
        初始化基线串行系统
        
        Args:
            asr_model_size: ASR模型大小
            llm_model_name: LLM模型名称
            llm_device: LLM计算设备（'cuda'、'cpu'或None自动检测）
        """
        self.asr_model_size = asr_model_size
        self.llm_model_name = llm_model_name
        
        # ASR始终使用CPU
        self.asr_device = "cpu"
        
        # LLM使用CUDA（如果可用）
        if llm_device is None:
            self.llm_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.llm_device = llm_device
            
        self.logger = get_logger(f"SystemA_Baseline")
        
        # 延迟初始化组件
        self._asr_processor = None
        self._llm_processor = None
        
        self.logger.info(f"系统A初始化完成 - ASR: {asr_model_size}(CPU), LLM: {llm_model_name}({self.llm_device})")
        if self.llm_device == "cuda":
            self.logger.info(f"LLM使用GPU加速 - GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.logger.info(f"LLM使用CPU运行")
    
    def get_data_path(self, experiment_name: str = "core_comparison", length_group: str = "long", sample_id: str = "sample_001") -> str:
        """
        获取实验数据路径
        
        Args:
            experiment_name: 实验名称 (core_comparison, asr_context, ablation_study, case_analysis)
            length_group: 长度分组 (short: 1-3s, medium: 3-10s, long: 10s+)
            sample_id: 样本ID
            
        Returns:
            音频文件路径
        """
        base_path = Path(__file__).parent.parent.parent / "experiments" / "datasets" / "processed" / "experiments"
        audio_path = base_path / experiment_name / "audio" / length_group / f"{sample_id}.wav"
        return str(audio_path)
    
    def load_sample_metadata(self, experiment_name: str = "core_comparison", length_group: str = "long", sample_id: str = "sample_001") -> Dict:
        """
        加载样本元数据
        
        Returns:
            样本元数据字典
        """
        try:
            base_path = Path(__file__).parent.parent.parent / "experiments" / "datasets" / "processed" / "experiments"
            metadata_path = base_path / experiment_name / "transcripts" / length_group / f"{sample_id}.json"
            
            if metadata_path.exists():
                import json
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                return metadata
            else:
                self.logger.warning(f"元数据文件不存在: {metadata_path}")
                return {
                    "audio_file": f"{sample_id}.wav",
                    "duration": 10.0,
                    "text": "测试用文本",
                    "language": "zh",
                    "ground_truth": "测试用文本"
                }
        except Exception as e:
            self.logger.error(f"加载元数据失败: {e}")
            return {
                "audio_file": f"{sample_id}.wav",
                "duration": 10.0,
                "text": "默认测试文本",
                "language": "zh",
                "ground_truth": "默认测试文本"
            }
    
    @property
    def asr_processor(self) -> StreamingASRProcessor:
        """延迟初始化ASR处理器"""
        if self._asr_processor is None:
            # ASR模型始终使用CPU
            self._asr_processor = StreamingASRProcessor(
                model_size=self.asr_model_size,
                recognition_threshold=2.0,  # 较长的阈值确保完整处理
                text_callback=None,
                device=self.asr_device  # 始终使用CPU
            )
        return self._asr_processor
    
    @property 
    def llm_processor(self) -> StreamLLMInference:
        """延迟初始化LLM处理器"""
        if self._llm_processor is None:
            self._llm_processor = StreamLLMInference(
                model_name=self.llm_model_name,
                device=self.llm_device  # LLM使用CUDA加速
            )
        return self._llm_processor
    
    def get_audio_duration(self, audio_path: str) -> float:
        """获取音频时长"""
        try:
            import librosa
            y, sr = librosa.load(audio_path, sr=None)
            duration = len(y) / sr
            self.logger.debug(f"音频 {audio_path} 时长: {duration:.2f}秒")
            return duration
        except ImportError:
            self.logger.error("librosa未安装，无法获取音频时长")
            raise ImportError("请安装librosa库: uv add librosa")
        except FileNotFoundError:
            self.logger.error(f"音频文件不存在: {audio_path}")
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")
        except Exception as e:
            self.logger.error(f"获取音频时长失败: {e}")
            raise
    
    def process_audio_complete(self, audio_path: str) -> Tuple[str, Dict[str, float]]:
        """
        完整音频ASR处理（传统方式）
        
        Returns:
            (transcript, timing_info)
        """
        try:
            # 使用新增的完整音频转录功能
            asr_result = self.asr_processor.transcribe_complete_audio(audio_path)
            
            # 检查是否有错误
            if 'error' in asr_result:
                self.logger.error(f"ASR转录失败: {asr_result['error']}")
                raise RuntimeError(f"ASR转录失败: {asr_result['error']}")
            
            transcript = asr_result['text']
            timing = asr_result['timing']
            
            # 修复：使用total_time包含模型加载等所有处理时间
            timing_info = {
                "audio_duration": timing['audio_duration'],
                "asr_processing_time": timing['total_time'],  # 使用总时间，包含模型加载
                "asr_transcription_only": timing['transcription_time'],  # 仅转录时间
                "asr_wait_time": timing['audio_duration']  # 串行系统需要等待音频播放完成
            }
            
            self.logger.info(f"ASR转录完成: '{transcript[:50]}...', 总耗时: {timing['total_time']:.3f}s (转录: {timing['transcription_time']:.3f}s)")
            return transcript, timing_info
            
        except Exception as e:
            self.logger.error(f"ASR处理失败: {e}")
            raise
    
    def process_llm_inference(self, text: str) -> Tuple[str, Dict[str, float]]:
        """
        LLM推理处理（传统方式）
        
        Returns:
            (first_token, timing_info)
        """
        try:
            start_time = time.perf_counter()
            
            # 传统LLM处理：从零开始，没有预填充的KV缓存
            first_token, generation_time = self._get_llm_first_token_traditional(text)
            
            llm_time = time.perf_counter() - start_time
            
            timing_info = {
                "llm_processing_time": llm_time,
                "first_token_generation_time": generation_time,
                "kv_cache_time": 0.0  # 基线系统没有KV缓存
            }
            
            self.logger.info(f"LLM推理完成，首token: '{first_token}'，耗时: {llm_time:.3f}s")
            return first_token, timing_info
            
        except Exception as e:
            self.logger.error(f"LLM处理失败: {e}")
            raise
    
    def _get_llm_first_token_traditional(self, text: str) -> Tuple[str, float]:
        """获取LLM首个token（传统方式，无优化）"""
        try:
            # 使用LLM的无缓存生成方式来获取首个token
            start_time = time.perf_counter()
            
            # 使用once_add_and_generate获取完整响应，只取第一个token
            response_generator = self.llm_processor.once_add_and_generate(
                prompt=text,
                max_new_tokens=1,  # 只生成一个token 
                temperature=0.1
            )
            
            # 获取首个token
            first_token = ""
            
            for token_text, _ in response_generator:
                first_token = token_text
                # 只要第一个token
                break
            
            generation_time = time.perf_counter() - start_time
            
            if not first_token:
                raise RuntimeError("LLM未能生成任何token")
            
            self.logger.info(f"LLM首token生成: '{first_token}', 耗时: {generation_time:.3f}s")
            return first_token, generation_time
            
        except Exception as e:
            self.logger.error(f"LLM首token生成失败: {e}")
            raise
    
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
        self.logger.debug(f"⏱️  语音结束时间基准: {speech_end_time:.6f}")
        
        # 步骤2：完整音频ASR处理
        self.logger.debug("开始ASR处理...")
        asr_start_time = time.perf_counter()
        transcript, asr_timing = self.process_audio_complete(audio_path)
        
        asr_complete_time = time.perf_counter()
        actual_asr_duration = (asr_complete_time - asr_start_time) * 1000
        self.logger.debug(f"⏱️  实际ASR耗时: {actual_asr_duration:.1f}ms，报告ASR耗时: {asr_timing['asr_processing_time']*1000:.1f}ms")
        
        # 步骤3：LLM推理
        self.logger.debug("开始LLM推理...")
        llm_start_time = time.perf_counter()
        first_token, llm_timing = self.process_llm_inference(transcript)
        
        first_token_time = time.perf_counter()
        actual_llm_duration = (first_token_time - llm_start_time) * 1000
        self.logger.debug(f"⏱️  实际LLM耗时: {actual_llm_duration:.1f}ms，报告LLM耗时: {llm_timing['llm_processing_time']*1000:.1f}ms")
        
        # 计算关键延迟指标
        ttft = (first_token_time - speech_end_time) * 1000  # 转换为毫秒
        expected_ttft = actual_asr_duration + actual_llm_duration
        self.logger.debug(f"⏱️  计算TTFT: {ttft:.1f}ms，预期TTFT: {expected_ttft:.1f}ms")
        
        # 验证时序一致性
        if abs(ttft - expected_ttft) > 100:  # 允许100ms误差
            self.logger.warning(f"⚠️  TTFT计算异常! TTFT={ttft:.1f}ms, 预期={expected_ttft:.1f}ms, 差异={abs(ttft-expected_ttft):.1f}ms")
        
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
    
    def warmup(self):
        """
        预热系统，确保所有模型都已加载完成
        这个方法应该在正式实验前调用，其耗时不计入TTFT
        """
        self.logger.info("开始系统A预热，加载模型...")
        
        # 预热ASR模型
        warmup_start = time.perf_counter()
        _ = self.asr_processor  # 触发ASR模型加载
        asr_warmup_time = time.perf_counter() - warmup_start
        self.logger.debug(f"ASR模型加载完成，耗时: {asr_warmup_time:.2f}s")
        
        # 预热LLM模型
        llm_warmup_start = time.perf_counter()
        _ = self.llm_processor  # 触发LLM模型加载
        llm_warmup_time = time.perf_counter() - llm_warmup_start
        self.logger.debug(f"LLM模型加载完成，耗时: {llm_warmup_time:.2f}s")
        
        # 进行一次虚拟推理以确保模型完全就绪
        try:
            dummy_generator = self.llm_processor.once_add_and_generate(
                prompt="测试",
                max_new_tokens=1,
                temperature=0.1
            )
            for _ in dummy_generator:
                break  # 只需要第一个token
            self.logger.debug("LLM虚拟推理完成")
        except Exception as e:
            self.logger.warning(f"LLM虚拟推理失败: {e}")
        
        total_warmup_time = time.perf_counter() - warmup_start
        self.logger.info(f"系统A预热完成，总耗时: {total_warmup_time:.2f}s")
    
    def reset(self):
        """重置系统状态（不重新加载模型）"""
        if self._asr_processor:
            self._asr_processor.reset()
        # LLM不需要特殊重置
        self.logger.debug("系统A已重置")
    
    def process_sample(self, audio_path: str, ground_truth: Optional[str] = None, skip_warmup: bool = False) -> Dict[str, Any]:
        """
        处理单个样本的标准接口（供实验调用）
        
        Args:
            audio_path: 音频文件路径
            ground_truth: 真实转录文本（可选，用于质量评估）
            skip_warmup: 是否跳过预热（如果已经预热过则可以跳过）
            
        Returns:
            标准化的处理结果
        """
        try:
            # 确保模型已预热（排除模型加载时间）
            if not skip_warmup:
                self.warmup()
            
            # 执行完整流水线（此时TTFT不包含模型加载时间）
            result = self.process_complete_pipeline(audio_path, simulate_delay=True)
            
            # 添加质量评估信息
            if ground_truth:
                # 这里可以计算WER等质量指标
                result['quality_metrics'] = {
                    'ground_truth': ground_truth,
                    'transcript_match': result['transcript'].strip() == ground_truth.strip()
                }
            
            # 标准化输出格式
            standardized_result = {
                'sample_id': Path(audio_path).stem,
                'system_name': 'SystemA_BaselineSequential',
                'audio_path': audio_path,
                'transcript': result['transcript'],
                'first_token': result['first_token'],
                
                # 核心性能指标
                'ttft_ms': result['performance_metrics']['ttft_ms'],
                'asr_time_ms': result['performance_metrics']['asr_processing_time_ms'],
                'llm_time_ms': result['performance_metrics']['llm_processing_time_ms'],
                'total_time_ms': result['performance_metrics']['total_processing_time_ms'],
                
                # 音频信息
                'audio_duration_s': result['timing']['audio_duration'],
                
                # 系统特征
                'has_streaming_asr': False,
                'has_kv_cache': False,
                'processing_type': 'sequential',
                
                # 完整原始结果
                'raw_result': result
            }
            
            if ground_truth:
                standardized_result['quality_metrics'] = result.get('quality_metrics', {})
            
            return standardized_result
            
        except Exception as e:
            self.logger.error(f"处理样本失败 {audio_path}: {e}")
            return {
                'sample_id': Path(audio_path).stem,
                'system_name': 'SystemA_BaselineSequential',
                'audio_path': audio_path,
                'transcript': 'ERROR',
                'first_token': 'ERROR',
                'ttft_ms': -1,
                'error': str(e)
            }


def test_system_a():
    """测试系统A"""
    print("测试系统A：基线串行系统")
    print("使用真实ASR和LLM处理\n")
    
    # 设置日志级别为DEBUG以显示详细日志
    set_global_log_level("DEBUG")
    print("日志级别已设置为DEBUG\n")
    
    # 检测CUDA可用性
    if torch.cuda.is_available():
        print(f"CUDA可用 - GPU: {torch.cuda.get_device_name(0)}")
        print(f"LLM将使用GPU加速")
        llm_device = "cuda"
    else:
        print("CUDA不可用，使用CPU")
        llm_device = "cpu"
    
    print(f"ASR使用CPU，LLM使用{llm_device}\n")
    
    # 创建系统实例
    system = SystemA_BaselineSequential(
        asr_model_size="base",  # 使用更小模型进行测试，避免CPU上large-v3过慢
        llm_model_name="Qwen/Qwen2-7B-Instruct",
        llm_device=llm_device
    )
    
    # 测试不同长度组的样本
    test_cases = [
        # ("core_comparison", "short", "sample_001"),
        # ("core_comparison", "medium", "sample_001"),
        ("core_comparison", "long", "sample_001")
    ]
    
    for exp_name, length_group, sample_id in test_cases:
        print(f"\n测试 {length_group} 长度组:")
        print("="*50)
        
        # 获取测试音频路径
        test_audio = system.get_data_path(
            experiment_name=exp_name,
            length_group=length_group,
            sample_id=sample_id
        )
        
        # 检查文件是否存在
        from pathlib import Path
        if not Path(test_audio).exists():
            print(f"  ⚠️ 音频文件不存在: {test_audio}")
            print(f"  请先运行 fill_experiments_data.py 准备数据")
            continue
        
        # 加载样本元数据
        metadata = system.load_sample_metadata(
            experiment_name=exp_name,
            length_group=length_group,
            sample_id=sample_id
        )
        
        print(f"  测试音频: {Path(test_audio).name}")
        print(f"  预期文本: {metadata.get('text', '未知')[:50]}...")
        print(f"  语言: {metadata.get('language', '未知')}")
        print(f"  时长: {metadata.get('duration', 0):.1f}秒")
        
        try:
            # 先进行预热（排除模型加载时间）
            print(f"  🔥 预热模型...")
            system.warmup()
            
            # 处理样本（不模拟延迟以加快测试，此时TTFT不包含模型加载时间）
            result = system.process_complete_pipeline(test_audio, simulate_delay=False)
            
            print(f"\n  处理结果:")
            print(f"    转录: {result['transcript'][:50]}...")
            print(f"    首Token: {result['first_token']}")
            print(f"    TTFT: {result['performance_metrics']['ttft_ms']:.1f}ms")
            print(f"    ASR时间: {result['performance_metrics']['asr_processing_time_ms']:.1f}ms")
            print(f"    LLM时间: {result['performance_metrics']['llm_processing_time_ms']:.1f}ms")
            print(f"    总处理时间: {result['performance_metrics']['total_processing_time_ms']:.1f}ms")
            
            # 验证结果结构
            assert 'system_name' in result
            assert 'performance_metrics' in result
            assert 'ttft_ms' in result['performance_metrics']
            print(f"  ✅ {length_group} 测试通过")
            
        except Exception as e:
            print(f"  ❌ 处理失败: {e}")
            print(f"  提示: 请确保ASR和LLM模型已正确安装并可访问")
    
    print(f"\n✅ 系统A测试完成")


if __name__ == "__main__":
    test_system_a()