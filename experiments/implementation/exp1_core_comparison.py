#!/usr/bin/env python3
"""
实验一：核心性能与质量对比
四系统全面对比实验，评估效率和质量指标
"""

import sys
from pathlib import Path
from typing import Dict, List, Any

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from base_experiment import BaseExperiment, ExperimentConfig, SampleResult
from src.utils.logging_utils import get_logger


class CoreComparisonExperiment(BaseExperiment):
    """核心性能与质量对比实验"""
    
    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.logger = get_logger("exp1_core_comparison")
        
    def prepare_test_data(self) -> List[Dict[str, Any]]:
        """准备测试数据"""
        # 这里应该加载实际的音频数据
        # 暂时使用模拟数据
        test_data = []
        
        # 模拟不同长度的音频样本
        sample_lengths = [3, 5, 8, 10, 15]  # 秒
        
        for i, length in enumerate(sample_lengths):
            for j in range(2):  # 每个长度2个样本
                sample_id = f"core_sample_{length}s_{j+1}"
                test_data.append({
                    "sample_id": sample_id,
                    "audio_file": f"/path/to/audio/{sample_id}.wav",
                    "audio_length": length,
                    "expected_transcript": f"这是一个{length}秒长度的测试语音样本",
                    "category": f"length_{length}s"
                })
        
        self.logger.info(f"准备了 {len(test_data)} 个核心对比测试样本")
        return test_data
    
    def run_single_sample(self, sample_data: Dict[str, Any]) -> SampleResult:
        """运行单个样本的四系统对比测试"""
        sample_id = sample_data["sample_id"]
        audio_file = sample_data["audio_file"]
        audio_length = sample_data["audio_length"]
        
        self.logger.debug(f"开始处理样本: {sample_id}")
        
        try:
            # 运行四个系统
            system_results = self.run_four_systems(audio_file)
            
            # 创建样本结果
            result = SampleResult(
                sample_id=sample_id,
                audio_file=audio_file,
                audio_length=audio_length,
                system_a_result=system_results.get('system_a'),
                system_b_result=system_results.get('system_b'),
                system_c_result=system_results.get('system_c'),
                system_a_prime_result=system_results.get('system_a_prime')
            )
            
            # 记录详细信息
            if result.ttft_comparisons:
                ttft_info = ", ".join([f"{k}: {v:.1f}ms" for k, v in result.ttft_comparisons.items()])
                self.logger.debug(f"样本 {sample_id} TTFT结果: {ttft_info}")
            
            if result.optimization_ratios:
                opt_info = ", ".join([f"{k}: {v:.1f}%" for k, v in result.optimization_ratios.items()])
                self.logger.debug(f"样本 {sample_id} 优化比例: {opt_info}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"样本 {sample_id} 处理失败: {e}")
            return SampleResult(
                sample_id=sample_id,
                audio_file=audio_file,
                audio_length=audio_length,
                error_message=str(e)
            )


def create_core_comparison_experiment(use_small_data: bool = True) -> CoreComparisonExperiment:
    """创建核心性能与质量对比实验"""
    
    # 实验配置
    config = ExperimentConfig(
        experiment_name="exp1_core_comparison",
        version="1.0",
        num_runs=10 if not use_small_data else 5,
        asr_model_size="base",
        llm_model_name="Qwen/Qwen1.5-0.5B-Chat",
        chunk_duration=0.3,
        simulate_delay=True,
        output_dir="experiments/results",
        log_level="INFO"
    )
    
    return CoreComparisonExperiment(config)


def test_core_comparison():
    """测试核心对比实验"""
    print("测试实验一：核心性能与质量对比")
    
    # 创建实验
    experiment = create_core_comparison_experiment(use_small_data=True)
    
    # 运行实验
    result = experiment.run_experiment()
    
    # 输出结果摘要
    print(f"\n实验结果摘要:")
    print(f"样本总数: {result.experiment_info['sample_count']}")
    print(f"成功样本: {result.experiment_info['success_count']}")
    print(f"执行时间: {result.execution_time:.2f}秒")
    
    # 输出主要结论
    print(f"\n主要结论:")
    for i, conclusion in enumerate(result.conclusions, 1):
        print(f"  {i}. {conclusion}")
    
    # 输出系统性能对比
    system_performance = result.summary_statistics.get('system_performance', {})
    if system_performance:
        print(f"\n系统性能对比:")
        for system, perf in system_performance.items():
            print(f"  {system}: 平均TTFT {perf['mean_ttft_ms']:.1f}ms "
                  f"(标准差: {perf['std_ttft_ms']:.1f}ms)")
    
    return result


if __name__ == "__main__":
    test_core_comparison()