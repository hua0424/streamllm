#!/usr/bin/env python3
"""
基础实验类 - 为所有具体实验提供通用功能
"""

import time
import json
import logging
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logging_utils import get_logger
from src.llm.stream_llm_inference import StreamLLMInference
from src.asr.faster_whisper_streamer import StreamingASRProcessor
from src.pipeline.optimized_streaming_pipeline import create_optimized_pipeline_from_wav
from src.pipeline.ultra_low_latency_pipeline import create_ultra_low_latency_pipeline_from_wav

# 导入四个对比系统
sys.path.append(str(Path(__file__).parent.parent))
from systems.system_a_baseline import SystemA_BaselineSequential
from systems.system_b_proposed import SystemB_ProposedKVCache
from systems.system_c_endtoend import SystemC_EndToEndOracle
from systems.system_a_prime import SystemA_Prime_StreamingASROnly


@dataclass
class ExperimentConfig:
    """实验配置数据类"""
    experiment_name: str
    version: str = "1.0"
    num_runs: int = 5
    asr_model_size: str = "base"
    llm_model_name: str = "Qwen/Qwen1.5-0.5B-Chat"
    chunk_duration: float = 0.3
    simulate_delay: bool = True
    output_dir: str = "experiments/results"
    log_level: str = "INFO"


@dataclass 
class SampleResult:
    """单个样本的实验结果（支持多系统对比）"""
    sample_id: str
    audio_file: str
    audio_length: float
    
    # 四系统结果
    system_a_result: Optional[Dict[str, Any]] = None  # 基线串行系统
    system_b_result: Optional[Dict[str, Any]] = None  # KV缓存预填充系统
    system_c_result: Optional[Dict[str, Any]] = None  # 端到端系统
    system_a_prime_result: Optional[Dict[str, Any]] = None  # 仅流式ASR系统
    
    # 计算得出的比较指标
    ttft_comparisons: Optional[Dict[str, float]] = None
    optimization_ratios: Optional[Dict[str, float]] = None
    quality_metrics: Optional[Dict[str, float]] = None
    
    error_message: Optional[str] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        
        # 自动计算比较指标
        self._calculate_comparison_metrics()
    
    def _calculate_comparison_metrics(self):
        """计算系统间比较指标"""
        try:
            # 提取TTFT值
            ttft_values = {}
            if self.system_a_result and 'performance_metrics' in self.system_a_result:
                ttft_values['system_a'] = self.system_a_result['performance_metrics'].get('ttft_ms', 0)
            if self.system_b_result and 'performance_metrics' in self.system_b_result:
                ttft_values['system_b'] = self.system_b_result['performance_metrics'].get('ttft_ms', 0)
            if self.system_c_result and 'performance_metrics' in self.system_c_result:
                ttft_values['system_c'] = self.system_c_result['performance_metrics'].get('ttft_ms', 0)
            if self.system_a_prime_result and 'performance_metrics' in self.system_a_prime_result:
                ttft_values['system_a_prime'] = self.system_a_prime_result['performance_metrics'].get('ttft_ms', 0)
            
            self.ttft_comparisons = ttft_values
            
            # 计算相对于基线的优化比例
            if 'system_a' in ttft_values and ttft_values['system_a'] > 0:
                baseline_ttft = ttft_values['system_a']
                optimization_ratios = {}
                
                for system, ttft in ttft_values.items():
                    if system != 'system_a' and ttft > 0:
                        optimization_ratios[f'{system}_vs_baseline'] = ((baseline_ttft - ttft) / baseline_ttft) * 100
                
                self.optimization_ratios = optimization_ratios
        
        except Exception as e:
            # 如果计算失败，设置为空字典
            self.ttft_comparisons = {}
            self.optimization_ratios = {}


@dataclass
class ExperimentResult:
    """完整实验结果"""
    experiment_info: Dict[str, Any]
    sample_results: List[SampleResult]
    summary_statistics: Dict[str, float]
    conclusions: List[str]
    execution_time: float
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class LatencyMeasurer:
    """延迟测量工具"""
    
    def __init__(self):
        self.start_time = None
        self.checkpoints = {}
        
    def start(self):
        """开始计时"""
        self.start_time = time.time()
        return self.start_time
    
    def checkpoint(self, name: str) -> float:
        """设置检查点"""
        if self.start_time is None:
            raise ValueError("请先调用start()开始计时")
        
        current_time = time.time()
        elapsed = current_time - self.start_time
        self.checkpoints[name] = elapsed
        return elapsed
    
    def get_duration(self, start_checkpoint: str = None, end_checkpoint: str = None) -> float:
        """获取两个检查点之间的时长"""
        if start_checkpoint is None:
            start_time = self.start_time
        else:
            start_time = self.start_time + self.checkpoints[start_checkpoint]
            
        if end_checkpoint is None:
            end_time = time.time()
        else:
            end_time = self.start_time + self.checkpoints[end_checkpoint]
            
        return end_time - start_time


class BaseExperiment(ABC):
    """基础实验类"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.logger = get_logger(f"experiment.{config.experiment_name}")
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 实验结果存储
        self.sample_results: List[SampleResult] = []
        self.experiment_start_time = None
        
        # 创建结果子目录
        self.experiment_dir = self.output_dir / config.experiment_name
        self.experiment_dir.mkdir(exist_ok=True)
        
        # 初始化四个对比系统
        self._init_comparison_systems()
        
        self.logger.info(f"初始化实验: {config.experiment_name}")
        self.logger.info(f"配置: {asdict(config)}")
        
    def _init_comparison_systems(self):
        """初始化四个对比系统"""
        try:
            # 系统A：基线串行系统
            self.system_a = SystemA_BaselineSequential(
                asr_model_size=self.config.asr_model_size,
                llm_model_name=self.config.llm_model_name,
                use_simulation=True  # 默认使用模拟模式快速测试
            )
            
            # 系统B：KV缓存预填充系统（本文方案）
            self.system_b = SystemB_ProposedKVCache(
                asr_model_size=self.config.asr_model_size,
                llm_model_name=self.config.llm_model_name,
                chunk_duration=self.config.chunk_duration,
                use_simulation=True
            )
            
            # 系统C：理想化端到端系统
            self.system_c = SystemC_EndToEndOracle(
                model_name="Qwen2-Audio-7B-Instruct",
                use_simulation=True
            )
            
            # 系统A'：仅流式ASR系统（消融研究用）
            self.system_a_prime = SystemA_Prime_StreamingASROnly(
                asr_model_size=self.config.asr_model_size,
                llm_model_name=self.config.llm_model_name,
                chunk_duration=self.config.chunk_duration,
                use_simulation=True
            )
            
            self.logger.info("四个对比系统初始化完成")
            
        except Exception as e:
            self.logger.error(f"系统初始化失败: {e}")
            raise
    
    @abstractmethod
    def prepare_test_data(self) -> List[Dict[str, Any]]:
        """准备测试数据 - 子类必须实现"""
        pass
    
    @abstractmethod 
    def run_single_sample(self, sample_data: Dict[str, Any]) -> SampleResult:
        """运行单个样本测试 - 子类必须实现"""
        pass
    
    def run_four_systems(self, audio_path: str) -> Dict[str, Dict[str, Any]]:
        """运行四个对比系统"""
        results = {}
        
        # 系统A：基线串行系统
        try:
            self.logger.debug(f"运行系统A - 基线串行系统")
            results['system_a'] = self.system_a.process_complete_pipeline(
                audio_path, simulate_delay=self.config.simulate_delay
            )
        except Exception as e:
            self.logger.error(f"系统A运行失败: {e}")
            results['system_a'] = None
        
        # 系统B：KV缓存预填充系统（本文方案）
        try:
            self.logger.debug(f"运行系统B - KV缓存预填充系统")
            results['system_b'] = self.system_b.process_complete_pipeline(
                audio_path, simulate_delay=self.config.simulate_delay
            )
        except Exception as e:
            self.logger.error(f"系统B运行失败: {e}")
            results['system_b'] = None
        
        # 系统C：理想化端到端系统
        try:
            self.logger.debug(f"运行系统C - 理想化端到端系统")
            results['system_c'] = self.system_c.process_complete_pipeline(
                audio_path, simulate_delay=self.config.simulate_delay
            )
        except Exception as e:
            self.logger.error(f"系统C运行失败: {e}")
            results['system_c'] = None
        
        # 系统A'：仅流式ASR系统（消融研究用）
        try:
            self.logger.debug(f"运行系统A' - 仅流式ASR系统")
            results['system_a_prime'] = self.system_a_prime.process_complete_pipeline(
                audio_path, simulate_delay=self.config.simulate_delay
            )
        except Exception as e:
            self.logger.error(f"系统A'运行失败: {e}")
            results['system_a_prime'] = None
        
        return results
    
    def get_audio_duration(self, audio_path: str) -> float:
        """获取音频时长"""
        try:
            import librosa
            y, sr = librosa.load(audio_path, sr=None)
            return len(y) / sr
        except ImportError:
            # 如果没有librosa，使用简单估算
            self.logger.warning("librosa未安装，使用简单估算")
            return 10.0  # 默认10秒
        except Exception as e:
            self.logger.error(f"获取音频时长失败: {e}")
            return 10.0
    
    def calculate_optimization_ratio(self, baseline: float, optimized: float) -> float:
        """计算优化比例"""
        if baseline <= 0:
            return 0.0
        return ((baseline - optimized) / baseline) * 100
    
    def run_experiment(self) -> ExperimentResult:
        """运行完整实验"""
        self.logger.info(f"开始执行实验: {self.config.experiment_name}")
        self.experiment_start_time = time.time()
        
        # 准备测试数据
        test_data = self.prepare_test_data()
        self.logger.info(f"准备了 {len(test_data)} 个测试样本")
        
        # 清空之前的结果
        self.sample_results.clear()
        
        # 执行测试
        for i, sample_data in enumerate(test_data):
            self.logger.info(f"正在处理样本 {i+1}/{len(test_data)}: {sample_data.get('sample_id', 'unknown')}")
            
            try:
                result = self.run_single_sample(sample_data)
                self.sample_results.append(result)
                # 输出主要系统B的优化比例
                if result.optimization_ratios and 'system_b_vs_baseline' in result.optimization_ratios:
                    opt_ratio = result.optimization_ratios['system_b_vs_baseline']
                    self.logger.info(f"样本 {result.sample_id} 完成，系统B优化比例: {opt_ratio:.1f}%")
                else:
                    self.logger.info(f"样本 {result.sample_id} 完成")
            except Exception as e:
                self.logger.error(f"样本 {sample_data.get('sample_id', 'unknown')} 处理失败: {e}")
                # 创建错误结果
                error_result = SampleResult(
                    sample_id=sample_data.get('sample_id', f'sample_{i}'),
                    audio_file=sample_data.get('audio_file', ''),
                    audio_length=sample_data.get('audio_length', 0.0),
                    error_message=str(e)
                )
                self.sample_results.append(error_result)
        
        # 计算总体统计
        execution_time = time.time() - self.experiment_start_time
        summary_stats = self.calculate_summary_statistics()
        conclusions = self.generate_conclusions()
        
        # 创建实验结果
        experiment_result = ExperimentResult(
            experiment_info={
                "name": self.config.experiment_name,
                "version": self.config.version,
                "config": asdict(self.config),
                "sample_count": len(self.sample_results),
                "success_count": len([r for r in self.sample_results if r.error_message is None])
            },
            sample_results=self.sample_results,
            summary_statistics=summary_stats,
            conclusions=conclusions,
            execution_time=execution_time
        )
        
        # 保存结果
        self.save_results(experiment_result)
        
        self.logger.info(f"实验完成，总耗时: {execution_time:.2f}秒")
        
        # 输出主要优化效果
        if summary_stats.get('primary_optimization_mean'):
            self.logger.info(f"主要优化比例: {summary_stats['primary_optimization_mean']:.1f}%")
        
        # 输出系统性能概况
        system_performance = summary_stats.get('system_performance', {})
        if system_performance:
            self.logger.info(f"参与对比的系统数量: {len(system_performance)}")
            for system, perf in system_performance.items():
                self.logger.info(f"  {system}: 平均TTFT {perf['mean_ttft_ms']:.1f}ms")
        
        return experiment_result
    
    def calculate_summary_statistics(self) -> Dict[str, Any]:
        """计算总体统计信息（支持四系统对比）"""
        # 过滤出有效的结果
        valid_results = [r for r in self.sample_results if r.error_message is None]
        
        if not valid_results:
            return {"error": "没有有效结果"}
        
        # 收集各系统的TTFT数据
        system_ttft_data = {
            'system_a': [],
            'system_b': [],
            'system_c': [],
            'system_a_prime': []
        }
        
        # 收集优化比例数据
        optimization_data = {}
        
        for result in valid_results:
            if result.ttft_comparisons:
                for system, ttft in result.ttft_comparisons.items():
                    if system in system_ttft_data and ttft > 0:
                        system_ttft_data[system].append(ttft)
            
            if result.optimization_ratios:
                for comparison, ratio in result.optimization_ratios.items():
                    if comparison not in optimization_data:
                        optimization_data[comparison] = []
                    optimization_data[comparison].append(ratio)
        
        # 计算系统统计信息
        system_stats = {}
        for system, ttft_values in system_ttft_data.items():
            if ttft_values:
                system_stats[system] = {
                    "sample_count": len(ttft_values),
                    "mean_ttft_ms": float(np.mean(ttft_values)),
                    "std_ttft_ms": float(np.std(ttft_values)),
                    "median_ttft_ms": float(np.median(ttft_values)),
                    "min_ttft_ms": float(np.min(ttft_values)),
                    "max_ttft_ms": float(np.max(ttft_values))
                }
        
        # 计算优化比例统计
        optimization_stats = {}
        for comparison, ratios in optimization_data.items():
            if ratios:
                optimization_stats[comparison] = {
                    "mean_optimization": float(np.mean(ratios)),
                    "std_optimization": float(np.std(ratios)),
                    "median_optimization": float(np.median(ratios)),
                    "min_optimization": float(np.min(ratios)),
                    "max_optimization": float(np.max(ratios))
                }
                
                # 95%置信区间
                if len(ratios) > 1:
                    std_error = np.std(ratios) / np.sqrt(len(ratios))
                    margin_error = 1.96 * std_error
                    optimization_stats[comparison]["confidence_interval_lower"] = float(np.mean(ratios) - margin_error)
                    optimization_stats[comparison]["confidence_interval_upper"] = float(np.mean(ratios) + margin_error)
        
        # 综合统计信息
        stats = {
            "total_samples": len(valid_results),
            "system_performance": system_stats,
            "optimization_comparisons": optimization_stats,
            "experiment_success_rate": len(valid_results) / len(self.sample_results) * 100 if self.sample_results else 0
        }
        
        # 如果有系统B的数据，计算主要优化效果
        if 'system_b_vs_baseline' in optimization_stats:
            stats["primary_optimization_mean"] = optimization_stats['system_b_vs_baseline']['mean_optimization']
            stats["primary_optimization_std"] = optimization_stats['system_b_vs_baseline']['std_optimization']
        
        return stats
    
    def generate_conclusions(self) -> List[str]:
        """生成实验结论（支持四系统对比）"""
        conclusions = []
        stats = self.calculate_summary_statistics()
        
        if "error" in stats:
            conclusions.append("实验执行过程中出现错误，无法生成有效结论")
            return conclusions
        
        # 系统性能分析
        system_performance = stats.get("system_performance", {})
        if system_performance:
            # 按平均TTFT排序系统
            sorted_systems = sorted(system_performance.items(), 
                                  key=lambda x: x[1].get("mean_ttft_ms", float('inf')))
            
            if len(sorted_systems) >= 2:
                best_system = sorted_systems[0]
                worst_system = sorted_systems[-1]
                
                conclusions.append(f"性能最优系统：{best_system[0]}，平均TTFT为 {best_system[1]['mean_ttft_ms']:.1f}ms")
                conclusions.append(f"性能最差系统：{worst_system[0]}，平均TTFT为 {worst_system[1]['mean_ttft_ms']:.1f}ms")
        
        # 优化效果分析
        optimization_comparisons = stats.get("optimization_comparisons", {})
        
        # 分析系统B相对于基线的优化效果
        if 'system_b_vs_baseline' in optimization_comparisons:
            b_vs_a = optimization_comparisons['system_b_vs_baseline']
            mean_opt = b_vs_a['mean_optimization']
            
            if mean_opt > 50:
                conclusions.append(f"KV缓存预填充方案（系统B）显著优于基线系统，平均优化幅度达到 {mean_opt:.1f}%")
            elif mean_opt > 20:
                conclusions.append(f"KV缓存预填充方案（系统B）有效优于基线系统，平均优化幅度为 {mean_opt:.1f}%")
            elif mean_opt > 0:
                conclusions.append(f"KV缓存预填充方案（系统B）略优于基线系统，优化幅度为 {mean_opt:.1f}%")
            else:
                conclusions.append("KV缓存预填充方案（系统B）相对基线系统无明显优势")
            
            # 稳定性分析
            if b_vs_a['std_optimization'] < 10:
                conclusions.append("系统B的优化效果稳定，变异性较小")
            else:
                conclusions.append("系统B的优化效果存在一定波动")
        
        # 消融研究分析
        if 'system_a_prime_vs_baseline' in optimization_comparisons:
            a_prime_vs_a = optimization_comparisons['system_a_prime_vs_baseline']
            conclusions.append(f"仅流式ASR优化（系统A'）相对基线的改进为 {a_prime_vs_a['mean_optimization']:.1f}%")
        
        # 理想上限分析
        if 'system_c_vs_baseline' in optimization_comparisons:
            c_vs_a = optimization_comparisons['system_c_vs_baseline']
            conclusions.append(f"理想化端到端系统（系统C）相对基线的改进为 {c_vs_a['mean_optimization']:.1f}%")
        
        # 实验成功率
        success_rate = stats.get("experiment_success_rate", 0)
        conclusions.append(f"实验总体成功率：{success_rate:.1f}%")
        
        return conclusions
    
    def save_results(self, result: ExperimentResult):
        """保存实验结果"""
        # 保存JSON格式的原始结果
        result_file = self.experiment_dir / "experiment_results.json"
        
        # 转换为可序列化的格式
        serializable_result = {
            "experiment_info": result.experiment_info,
            "sample_results": [asdict(r) for r in result.sample_results],
            "summary_statistics": result.summary_statistics,
            "conclusions": result.conclusions,
            "execution_time": result.execution_time,
            "timestamp": result.timestamp
        }
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_result, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"实验结果已保存到: {result_file}")
        
        # 保存简化的摘要
        summary_file = self.experiment_dir / "experiment_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"实验名称: {result.experiment_info['name']}\n")
            f.write(f"执行时间: {result.timestamp}\n")
            f.write(f"样本数量: {result.experiment_info['sample_count']}\n")
            f.write(f"成功样本: {result.experiment_info['success_count']}\n")
            f.write(f"执行耗时: {result.execution_time:.2f}秒\n\n")
            
            f.write("主要统计结果:\n")
            for key, value in result.summary_statistics.items():
                if isinstance(value, float):
                    f.write(f"  {key}: {value:.2f}\n")
                else:
                    f.write(f"  {key}: {value}\n")
            
            f.write("\n实验结论:\n")
            for i, conclusion in enumerate(result.conclusions, 1):
                f.write(f"  {i}. {conclusion}\n")
        
        self.logger.info(f"实验摘要已保存到: {summary_file}")


class ExperimentRunner:
    """实验运行器 - 管理多个实验的执行"""
    
    def __init__(self, output_dir: str = "experiments/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger("experiment_runner")
        
        # 实验注册表
        self.experiments = {}
        
    def register_experiment(self, name: str, experiment_class, config: ExperimentConfig):
        """注册实验"""
        self.experiments[name] = {
            "class": experiment_class,
            "config": config
        }
        self.logger.info(f"注册实验: {name}")
    
    def run_experiment(self, name: str) -> ExperimentResult:
        """运行指定实验"""
        if name not in self.experiments:
            raise ValueError(f"未注册的实验: {name}")
        
        exp_info = self.experiments[name]
        experiment = exp_info["class"](exp_info["config"])
        
        self.logger.info(f"开始运行实验: {name}")
        result = experiment.run_experiment()
        self.logger.info(f"实验 {name} 完成")
        
        return result
    
    def run_all_experiments(self) -> Dict[str, ExperimentResult]:
        """运行所有注册的实验"""
        results = {}
        
        for name in self.experiments:
            try:
                results[name] = self.run_experiment(name)
            except Exception as e:
                self.logger.error(f"实验 {name} 执行失败: {e}")
                results[name] = None
        
        # 生成总体报告
        self.generate_overall_report(results)
        
        return results
    
    def generate_overall_report(self, results: Dict[str, ExperimentResult]):
        """生成总体实验报告"""
        report_file = self.output_dir / "overall_experiment_report.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 级联式语音对话系统延迟优化实验报告\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 实验概述\n\n")
            f.write(f"总实验数量: {len(results)}\n")
            f.write(f"成功实验数量: {len([r for r in results.values() if r is not None])}\n\n")
            
            for name, result in results.items():
                f.write(f"### {name}\n\n")
                if result is None:
                    f.write("❌ 实验执行失败\n\n")
                    continue
                
                f.write("✅ 实验执行成功\n\n")
                f.write(f"- 样本数量: {result.experiment_info['sample_count']}\n")
                f.write(f"- 成功样本: {result.experiment_info['success_count']}\n")
                f.write(f"- 平均优化比例: {result.summary_statistics.get('mean_optimization', 0):.1f}%\n")
                f.write(f"- 执行时间: {result.execution_time:.2f}秒\n\n")
                
                f.write("主要结论:\n")
                for conclusion in result.conclusions:
                    f.write(f"- {conclusion}\n")
                f.write("\n")
        
        self.logger.info(f"总体实验报告已保存到: {report_file}")