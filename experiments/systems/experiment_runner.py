#!/usr/bin/env python3
"""
实验运行器 - 演示正确的实验流程：排除模型加载时间

这个类展示了如何正确进行TTFT实验：
1. 预热阶段：加载所有模型，耗时不计入实验结果
2. 正式实验：多次运行，第一次结果丢弃
3. 数据收集：只收集预热后的纯推理延迟
"""

import time
from typing import List, Dict, Any
from pathlib import Path
import sys
import logging

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from experiments.systems.system_a_baseline import SystemA_BaselineSequential
from experiments.systems.system_b_proposed import SystemB_ProposedKVCache
from src.utils.logging_utils import get_logger


class ExperimentRunner:
    """正确的实验运行器，排除模型加载时间"""
    
    def __init__(self):
        self.logger = get_logger("ExperimentRunner")
    
    def run_single_system_experiment(self, system, test_samples: List[str], 
                                   system_name: str, runs_per_sample: int = 3) -> List[Dict[str, Any]]:
        """
        运行单个系统的实验
        
        Args:
            system: 系统实例 (SystemA 或 SystemB)
            test_samples: 测试音频文件路径列表
            system_name: 系统名称
            runs_per_sample: 每个样本运行次数
            
        Returns:
            实验结果列表，不包含模型加载时间
        """
        self.logger.info(f"开始 {system_name} 实验")
        
        # ===== 阶段1：预热阶段（模型加载） =====
        self.logger.info("🔥 阶段1：预热阶段 - 加载模型（耗时不计入实验结果）")
        warmup_start = time.perf_counter()
        
        try:
            system.warmup()
            warmup_time = time.perf_counter() - warmup_start
            self.logger.info(f"✅ 预热完成，模型加载耗时: {warmup_time:.2f}s")
        except Exception as e:
            self.logger.error(f"❌ 预热失败: {e}")
            return []
        
        # ===== 阶段2：首次试运行（结果丢弃） =====
        self.logger.info("🧪 阶段2：首次试运行 - 确保系统稳定（结果丢弃）")
        if test_samples:
            try:
                first_result = system.process_sample(test_samples[0], skip_warmup=True)
                self.logger.debug(f"首次试运行TTFT: {first_result.get('ttft_ms', -1):.1f}ms（此结果将被丢弃）")
            except Exception as e:
                self.logger.warning(f"首次试运行失败: {e}")
        
        # ===== 阶段3：正式实验（收集数据） =====
        self.logger.info(f"📊 阶段3：正式实验 - 收集纯推理延迟数据")
        
        results = []
        for sample_idx, audio_path in enumerate(test_samples):
            self.logger.info(f"处理样本 {sample_idx + 1}/{len(test_samples)}: {Path(audio_path).name}")
            
            sample_results = []
            
            # 每个样本运行多次
            for run_idx in range(runs_per_sample):
                try:
                    # 运行实验（跳过预热，因为已经预热过）
                    result = system.process_sample(audio_path, skip_warmup=True)
                    sample_results.append(result)
                    
                    ttft = result.get('ttft_ms', -1)
                    self.logger.debug(f"  运行 {run_idx + 1}: TTFT = {ttft:.1f}ms")
                    
                    # 重置系统状态（但不重新加载模型）
                    system.reset()
                    
                except Exception as e:
                    self.logger.error(f"  运行 {run_idx + 1} 失败: {e}")
                    continue
            
            # 添加样本结果（包含多次运行）
            if sample_results:
                # 计算统计信息
                ttfts = [r.get('ttft_ms', -1) for r in sample_results if r.get('ttft_ms', -1) > 0]
                
                summary = {
                    'sample_id': Path(audio_path).stem,
                    'sample_path': audio_path,
                    'system_name': system_name,
                    'runs': len(sample_results),
                    'ttft_ms_mean': sum(ttfts) / len(ttfts) if ttfts else -1,
                    'ttft_ms_min': min(ttfts) if ttfts else -1,
                    'ttft_ms_max': max(ttfts) if ttfts else -1,
                    'raw_results': sample_results
                }
                results.append(summary)
                
                self.logger.info(f"  ✅ 完成，平均TTFT: {summary['ttft_ms_mean']:.1f}ms")
        
        self.logger.info(f"🎉 {system_name} 实验完成，共处理 {len(results)} 个样本")
        return results
    
    def compare_systems_experiment(self, test_samples: List[str], runs_per_sample: int = 3) -> Dict[str, Any]:
        """
        对比系统A和系统B的实验
        
        Args:
            test_samples: 测试音频文件路径列表
            runs_per_sample: 每个样本运行次数
            
        Returns:
            对比实验结果
        """
        self.logger.info("🚀 开始系统A vs 系统B 对比实验")
        
        # 系统A实验
        system_a = SystemA_BaselineSequential(
            asr_model_size="base",
            llm_model_name="Qwen/Qwen2-7B-Instruct"
        )
        results_a = self.run_single_system_experiment(
            system=system_a, 
            test_samples=test_samples[:2],  # 限制样本数量避免超时
            system_name="SystemA_Baseline",
            runs_per_sample=runs_per_sample
        )
        
        # 系统B实验
        system_b = SystemB_ProposedKVCache(
            asr_model_size="base",
            llm_model_name="Qwen/Qwen2-7B-Instruct",
        )
        results_b = self.run_single_system_experiment(
            system=system_b,
            test_samples=test_samples[:2],
            system_name="SystemB_KVCache",
            runs_per_sample=runs_per_sample
        )
        
        # 对比分析
        comparison_results = self.analyze_comparison(results_a, results_b)
        
        return {
            'system_a_results': results_a,
            'system_b_results': results_b,
            'comparison_analysis': comparison_results
        }
    
    def analyze_comparison(self, results_a: List[Dict], results_b: List[Dict]) -> Dict[str, Any]:
        """分析系统对比结果"""
        if not results_a or not results_b:
            return {'error': '缺少对比数据'}
        
        # 计算平均TTFT
        ttft_a = [r['ttft_ms_mean'] for r in results_a if r['ttft_ms_mean'] > 0]
        ttft_b = [r['ttft_ms_mean'] for r in results_b if r['ttft_ms_mean'] > 0]
        
        if not ttft_a or not ttft_b:
            return {'error': 'TTFT数据不足'}
        
        avg_ttft_a = sum(ttft_a) / len(ttft_a)
        avg_ttft_b = sum(ttft_b) / len(ttft_b)
        
        improvement = (avg_ttft_a - avg_ttft_b) / avg_ttft_a * 100
        
        analysis = {
            'avg_ttft_system_a_ms': avg_ttft_a,
            'avg_ttft_system_b_ms': avg_ttft_b,
            'ttft_improvement_percent': improvement,
            'samples_compared': min(len(ttft_a), len(ttft_b)),
            'conclusion': f"系统B相比系统A延迟降低 {improvement:.1f}%" if improvement > 0 else f"系统B延迟增加 {abs(improvement):.1f}%"
        }
        
        return analysis


def demo_correct_experiment():
    """演示正确的实验流程"""
    print("🎯 演示正确的TTFT实验流程（排除模型加载时间）")
    print("=" * 60)
    
    # 创建实验运行器
    runner = ExperimentRunner()
    
    # 准备测试样本（使用不存在的路径进行演示）
    test_samples = [
        "/demo/sample1.wav",  # 演示用路径
        "/demo/sample2.wav",
        "/demo/sample3.wav"
    ]
    
    print("📋 实验设置:")
    print(f"  测试样本: {len(test_samples)}个")
    print(f"  每样本运行: 3次")
    print(f"  评估指标: TTFT (Time-to-First-Token)")
    print(f"  关键原则: 模型加载时间不计入TTFT")
    
    print("\n📖 实验步骤说明:")
    print("1. 🔥 预热阶段: 加载所有模型（ASR + LLM）")
    print("2. 🧪 试运行: 首次运行确保系统稳定（结果丢弃）")
    print("3. 📊 正式实验: 多次运行收集纯推理延迟")
    print("4. 📈 统计分析: 计算平均值、标准差等")
    
    print("\n⚠️  重要注意事项:")
    print("- 第一次运行的结果必须丢弃（包含模型加载）")
    print("- 只有预热后的运行才能反映真实推理性能")
    print("- 多次运行可以降低测量误差")
    
    print("\n✅ 这样设计的实验结果才能准确对比系统A和系统B的TTFT性能!")
    
    # 由于是演示，不实际运行避免超时
    print("\n🚫 演示完成（实际运行请使用真实音频文件）")


if __name__ == "__main__":
    demo_correct_experiment()