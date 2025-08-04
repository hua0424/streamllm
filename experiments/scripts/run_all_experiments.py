#!/usr/bin/env python3
"""
运行所有实验的主脚本
用法: python experiments/scripts/run_all_experiments.py --config experiments/configs/default_config.json
"""

import argparse
import json
import logging
import time
from pathlib import Path
import sys
import traceback

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logging_utils import get_logger, set_global_log_level

class ExperimentRunner:
    def __init__(self, config_path):
        """初始化实验运行器"""
        self.config_path = Path(config_path)
        self.config = self.load_config()
        self.logger = get_logger(__name__)
        self.results = {}
        
        # 设置日志级别
        set_global_log_level(self.config.get("experiment_config", {}).get("log_level", "INFO"))
        
        # 创建结果目录
        self.results_dir = Path("experiments/results")
        self.results_dir.mkdir(exist_ok=True)
        
    def load_config(self):
        """加载实验配置"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Failed to load config from {self.config_path}: {e}")
            sys.exit(1)
    
    def run_experiment_1(self):
        """实验1: 语音长度对延迟优化效果的影响"""
        if not self.config.get("experiment_1_length_impact", {}).get("enabled", False):
            self.logger.info("实验1已禁用，跳过")
            return None
            
        self.logger.info("=" * 60)
        self.logger.info("开始执行实验1: 语音长度对延迟优化效果的影响")
        self.logger.info("=" * 60)
        
        try:
            from .experiment_1_length_impact import LengthImpactExperiment
            
            experiment = LengthImpactExperiment(self.config)
            results = experiment.run()
            
            # 保存结果
            output_dir = self.results_dir / "exp1_length_impact"
            output_dir.mkdir(exist_ok=True)
            
            with open(output_dir / "results.json", 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"实验1完成，结果已保存到 {output_dir}")
            return results
            
        except Exception as e:
            self.logger.error(f"实验1执行失败: {e}")
            self.logger.error(traceback.format_exc())
            return None
    
    def run_experiment_2(self):
        """实验2: 与原生语音模型对比"""
        if not self.config.get("experiment_2_model_comparison", {}).get("enabled", False):
            self.logger.info("实验2已禁用，跳过")
            return None
            
        self.logger.info("=" * 60)
        self.logger.info("开始执行实验2: 与原生语音模型对比")
        self.logger.info("=" * 60)
        
        try:
            from .experiment_2_model_comparison import ModelComparisonExperiment
            
            experiment = ModelComparisonExperiment(self.config)
            results = experiment.run()
            
            # 保存结果
            output_dir = self.results_dir / "exp2_model_comparison"
            output_dir.mkdir(exist_ok=True)
            
            with open(output_dir / "results.json", 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"实验2完成，结果已保存到 {output_dir}")
            return results
            
        except Exception as e:
            self.logger.error(f"实验2执行失败: {e}")
            self.logger.error(traceback.format_exc())
            return None
    
    def run_experiment_3(self):
        """实验3: 消融实验"""
        if not self.config.get("experiment_3_ablation_study", {}).get("enabled", False):
            self.logger.info("实验3已禁用，跳过")
            return None
            
        self.logger.info("=" * 60)
        self.logger.info("开始执行实验3: 消融实验")
        self.logger.info("=" * 60)
        
        try:
            from .experiment_3_ablation_study import AblationExperiment
            
            experiment = AblationExperiment(self.config)
            results = experiment.run()
            
            # 保存结果
            output_dir = self.results_dir / "exp3_ablation_study"
            output_dir.mkdir(exist_ok=True)
            
            with open(output_dir / "results.json", 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"实验3完成，结果已保存到 {output_dir}")
            return results
            
        except Exception as e:
            self.logger.error(f"实验3执行失败: {e}")
            self.logger.error(traceback.format_exc())
            return None
    
    def run_experiment_4(self):
        """实验4: ASR模型规模对系统性能的影响"""
        if not self.config.get("experiment_4_asr_model_scale", {}).get("enabled", False):
            self.logger.info("实验4已禁用，跳过")
            return None
            
        self.logger.info("=" * 60)
        self.logger.info("开始执行实验4: ASR模型规模对系统性能的影响")
        self.logger.info("=" * 60)
        
        try:
            from .experiment_4_asr_model_scale import ASRModelScaleExperiment
            
            experiment = ASRModelScaleExperiment(self.config)
            results = experiment.run()
            
            # 保存结果
            output_dir = self.results_dir / "exp4_asr_model_scale"
            output_dir.mkdir(exist_ok=True)
            
            with open(output_dir / "results.json", 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"实验4完成，结果已保存到 {output_dir}")
            return results
            
        except Exception as e:
            self.logger.error(f"实验4执行失败: {e}")
            self.logger.error(traceback.format_exc())
            return None
    
    def run_experiment_5(self):
        """实验5: 音频质量鲁棒性测试"""
        if not self.config.get("experiment_5_audio_quality", {}).get("enabled", False):
            self.logger.info("实验5已禁用，跳过")
            return None
            
        self.logger.info("=" * 60)
        self.logger.info("开始执行实验5: 音频质量鲁棒性测试")
        self.logger.info("=" * 60)
        
        try:
            from .experiment_5_audio_quality import AudioQualityExperiment
            
            experiment = AudioQualityExperiment(self.config)
            results = experiment.run()
            
            # 保存结果
            output_dir = self.results_dir / "exp5_audio_quality"
            output_dir.mkdir(exist_ok=True)
            
            with open(output_dir / "results.json", 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"实验5完成，结果已保存到 {output_dir}")
            return results
            
        except Exception as e:
            self.logger.error(f"实验5执行失败: {e}")
            self.logger.error(traceback.format_exc())
            return None
    
    def run_experiment_6(self):
        """实验6: 实时性能与准确率权衡分析"""
        if not self.config.get("experiment_6_tradeoff_analysis", {}).get("enabled", False):
            self.logger.info("实验6已禁用，跳过")
            return None
            
        self.logger.info("=" * 60)
        self.logger.info("开始执行实验6: 实时性能与准确率权衡分析")
        self.logger.info("=" * 60)
        
        try:
            from .experiment_6_tradeoff_analysis import TradeoffAnalysisExperiment
            
            experiment = TradeoffAnalysisExperiment(self.config)
            results = experiment.run()
            
            # 保存结果
            output_dir = self.results_dir / "exp6_tradeoff_analysis"
            output_dir.mkdir(exist_ok=True)
            
            with open(output_dir / "results.json", 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"实验6完成，结果已保存到 {output_dir}")
            return results
            
        except Exception as e:
            self.logger.error(f"实验6执行失败: {e}")
            self.logger.error(traceback.format_exc())
            return None
    
    def run_experiment_7(self):
        """实验7: 并发处理能力测试"""
        if not self.config.get("experiment_7_concurrent_performance", {}).get("enabled", False):
            self.logger.info("实验7已禁用，跳过")
            return None
            
        self.logger.info("=" * 60)
        self.logger.info("开始执行实验7: 并发处理能力测试")
        self.logger.info("=" * 60)
        
        try:
            from .experiment_7_concurrent_performance import ConcurrentPerformanceExperiment
            
            experiment = ConcurrentPerformanceExperiment(self.config)
            results = experiment.run()
            
            # 保存结果
            output_dir = self.results_dir / "exp7_concurrent_performance"
            output_dir.mkdir(exist_ok=True)
            
            with open(output_dir / "results.json", 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"实验7完成，结果已保存到 {output_dir}")
            return results
            
        except Exception as e:
            self.logger.error(f"实验7执行失败: {e}")
            self.logger.error(traceback.format_exc())
            return None
    
    def run_all_experiments(self):
        """运行所有实验"""
        start_time = time.time()
        
        self.logger.info("🚀 开始执行完整实验套件")
        self.logger.info(f"配置文件: {self.config_path}")
        self.logger.info(f"结果目录: {self.results_dir}")
        
        # 按顺序执行所有实验
        experiments = [
            ("实验1", self.run_experiment_1),
            ("实验2", self.run_experiment_2), 
            ("实验3", self.run_experiment_3),
            ("实验4", self.run_experiment_4),
            ("实验5", self.run_experiment_5),
            ("实验6", self.run_experiment_6),
            ("实验7", self.run_experiment_7)
        ]
        
        success_count = 0
        for exp_name, exp_func in experiments:
            try:
                result = exp_func()
                if result is not None:
                    self.results[exp_name] = result
                    success_count += 1
                    self.logger.info(f"✅ {exp_name} 执行成功")
                else:
                    self.logger.warning(f"⚠️ {exp_name} 跳过或失败")
            except Exception as e:
                self.logger.error(f"❌ {exp_name} 执行出错: {e}")
        
        # 保存汇总结果
        summary = {
            "execution_time": time.time() - start_time,
            "total_experiments": len(experiments),
            "successful_experiments": success_count,
            "failed_experiments": len(experiments) - success_count,
            "results": self.results,
            "config": self.config,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(self.results_dir / "experiment_summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        self.logger.info("=" * 60)
        self.logger.info("🎉 实验套件执行完成")
        self.logger.info(f"总耗时: {summary['execution_time']:.2f}秒")
        self.logger.info(f"成功实验: {success_count}/{len(experiments)}")
        self.logger.info(f"结果汇总: {self.results_dir}/experiment_summary.json")
        self.logger.info("=" * 60)
        
        return summary

def main():
    parser = argparse.ArgumentParser(description="运行所有语音对话系统延迟优化实验")
    parser.add_argument("--config", type=str, 
                       default="experiments/configs/default_config.json",
                       help="实验配置文件路径")
    parser.add_argument("--experiments", type=str, nargs='+',
                       choices=['1', '2', '3', '4', '5', '6', '7', 'all'],
                       default=['all'],
                       help="要运行的实验编号，默认运行所有实验")
    
    args = parser.parse_args()
    
    try:
        runner = ExperimentRunner(args.config)
        
        if 'all' in args.experiments:
            # 运行所有实验
            summary = runner.run_all_experiments()
        else:
            # 运行指定实验
            experiment_methods = {
                '1': runner.run_experiment_1,
                '2': runner.run_experiment_2,
                '3': runner.run_experiment_3,
                '4': runner.run_experiment_4,
                '5': runner.run_experiment_5,
                '6': runner.run_experiment_6,
                '7': runner.run_experiment_7
            }
            
            for exp_num in args.experiments:
                if exp_num in experiment_methods:
                    experiment_methods[exp_num]()
                    
    except KeyboardInterrupt:
        logging.info("实验被用户中断")
        sys.exit(1)
    except Exception as e:
        logging.error(f"实验执行失败: {e}")
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()