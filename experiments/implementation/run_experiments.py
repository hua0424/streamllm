#!/usr/bin/env python3
"""
实验运行器 - 统一管理和执行所有实验
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logging_utils import get_logger
from .base_experiment import ExperimentRunner, ExperimentConfig
from .length_impact_exp import create_length_impact_experiment
from .ablation_exp import create_ablation_experiment
from .asr_scale_exp import create_asr_scale_experiment
from .audio_quality_exp import create_audio_quality_experiment
from .native_model_comparison_exp import create_native_model_comparison_experiment
from .concurrent_performance_exp import create_concurrent_performance_experiment


class StreamingExperimentRunner:
    """流式语音对话系统实验运行器"""
    
    def __init__(self, output_dir: str = "experiments/results", use_small_data: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_small_data = use_small_data
        self.logger = get_logger("streaming_experiment_runner")
        
        # 实验注册表
        self.available_experiments = {
            "length_impact": {
                "name": "语音长度影响实验",
                "description": "测试不同语音长度对优化效果的影响",
                "create_func": create_length_impact_experiment,
                "priority": 1
            },
            "ablation": {
                "name": "消融实验",
                "description": "量化各优化组件的贡献度",
                "create_func": create_ablation_experiment,
                "priority": 2
            },
            "asr_scale": {
                "name": "ASR模型规模实验",
                "description": "测试不同ASR模型大小对性能的影响",
                "create_func": create_asr_scale_experiment,
                "priority": 3
            },
            "audio_quality": {
                "name": "音频质量鲁棒性实验",
                "description": "测试系统在不同音频质量下的性能稳定性",
                "create_func": create_audio_quality_experiment,
                "priority": 4
            },
            "native_comparison": {
                "name": "原生模型对比实验",
                "description": "对比级联方案与原生语音模型的性能",
                "create_func": create_native_model_comparison_experiment,
                "priority": 5
            },
            "concurrent_performance": {
                "name": "并发性能实验",
                "description": "测试系统在多用户并发场景下的表现",
                "create_func": create_concurrent_performance_experiment,
                "priority": 6
            }
        }
        
        self.logger.info(f"实验运行器初始化完成，共注册 {len(self.available_experiments)} 个实验")
        if use_small_data:
            self.logger.info("使用小数据模式进行快速验证")
    
    def list_experiments(self):
        """列出所有可用实验"""
        print("\n可用实验列表：")
        print("=" * 80)
        
        # 按优先级排序
        sorted_experiments = sorted(self.available_experiments.items(), 
                                  key=lambda x: x[1]["priority"])
        
        for exp_id, exp_info in sorted_experiments:
            print(f"{exp_info['priority']}. {exp_info['name']} ({exp_id})")
            print(f"   描述: {exp_info['description']}")
            print()
    
    def run_single_experiment(self, experiment_id: str) -> bool:
        """运行单个实验"""
        if experiment_id not in self.available_experiments:
            self.logger.error(f"未知实验: {experiment_id}")
            return False
        
        exp_info = self.available_experiments[experiment_id]
        self.logger.info(f"开始运行实验: {exp_info['name']}")
        
        try:
            # 创建实验实例
            experiment = exp_info["create_func"](use_small_data=self.use_small_data)
            
            # 运行实验
            start_time = time.time()
            result = experiment.run_experiment()
            end_time = time.time()
            
            # 输出结果摘要
            self.logger.info(f"实验 {exp_info['name']} 完成")
            self.logger.info(f"执行时间: {end_time - start_time:.2f}秒")
            self.logger.info(f"样本数量: {result.experiment_info['sample_count']}")
            self.logger.info(f"成功样本: {result.experiment_info['success_count']}")
            
            if result.summary_statistics and 'mean_optimization' in result.summary_statistics:
                opt_ratio = result.summary_statistics['mean_optimization']
                self.logger.info(f"平均优化比例: {opt_ratio:.1f}%")
            
            print(f"\n{exp_info['name']} 主要结论:")
            for i, conclusion in enumerate(result.conclusions, 1):
                print(f"  {i}. {conclusion}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"实验 {exp_info['name']} 执行失败: {e}")
            return False
    
    def run_core_experiments(self) -> Dict[str, bool]:
        """运行核心实验（实验1和实验3）"""
        core_experiments = ["length_impact", "ablation"]
        results = {}
        
        print("\n运行核心实验...")
        print("=" * 50)
        
        for exp_id in core_experiments:
            success = self.run_single_experiment(exp_id)
            results[exp_id] = success
            print("-" * 50)
        
        return results
    
    def run_all_experiments(self) -> Dict[str, bool]:
        """运行所有实验"""
        results = {}
        
        print("\n运行所有实验...")
        print("=" * 50)
        
        # 按优先级顺序运行
        sorted_experiments = sorted(self.available_experiments.items(), 
                                  key=lambda x: x[1]["priority"])
        
        for exp_id, exp_info in sorted_experiments:
            success = self.run_single_experiment(exp_id)
            results[exp_id] = success
            print("-" * 50)
        
        return results
    
    def run_selected_experiments(self, experiment_ids: List[str]) -> Dict[str, bool]:
        """运行选定的实验"""
        results = {}
        
        print(f"\n运行选定实验: {', '.join(experiment_ids)}")
        print("=" * 50)
        
        for exp_id in experiment_ids:
            if exp_id in self.available_experiments:
                success = self.run_single_experiment(exp_id)
                results[exp_id] = success
            else:
                self.logger.error(f"未知实验ID: {exp_id}")
                results[exp_id] = False
            print("-" * 50)
        
        return results
    
    def generate_overall_report(self, results: Dict[str, bool]):
        """生成总体实验报告"""
        report_file = self.output_dir / "overall_experiment_report.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 级联式语音对话系统延迟优化实验报告\n\n")
            f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 实验概述
            f.write("## 实验概述\n\n")
            f.write(f"- 总实验数量: {len(results)}\n")
            f.write(f"- 成功实验数量: {sum(results.values())}\n")
            f.write(f"- 失败实验数量: {len(results) - sum(results.values())}\n")
            f.write(f"- 数据模式: {'小数据验证模式' if self.use_small_data else '完整数据模式'}\n\n")
            
            # 实验结果列表
            f.write("## 实验执行结果\n\n")
            
            for exp_id, success in results.items():
                exp_info = self.available_experiments[exp_id]
                status = "✅ 成功" if success else "❌ 失败"
                f.write(f"### {exp_info['name']}\n\n")
                f.write(f"- 状态: {status}\n")
                f.write(f"- 描述: {exp_info['description']}\n")
                
                # 尝试读取实验结果文件
                exp_result_file = self.output_dir / f"{exp_id}_experiment" / "experiment_summary.txt"
                if exp_result_file.exists():
                    f.write(f"- 结果文件: `{exp_result_file}`\n")
                
                f.write("\n")
            
            # 实验文件结构
            f.write("## 生成的文件\n\n")
            f.write("```\n")
            f.write("experiments/results/\n")
            for exp_id in results.keys():
                f.write(f"├── {exp_id}_experiment/\n")
                f.write(f"│   ├── experiment_results.json\n")
                f.write(f"│   ├── experiment_summary.txt\n")
                f.write(f"│   └── [具体分析文件]\n")
            f.write("└── overall_experiment_report.md\n")
            f.write("```\n\n")
            
            # 下一步建议
            f.write("## 下一步建议\n\n")
            if self.use_small_data:
                f.write("1. 当前使用小数据验证模式，建议使用完整数据集重新运行\n")
                f.write("2. 检查失败的实验并解决相关问题\n")
                f.write("3. 基于实验结果进行论文写作\n")
            else:
                f.write("1. 分析实验结果并撰写论文\n")
                f.write("2. 根据结果调整系统优化策略\n")
                f.write("3. 准备实验数据的可视化图表\n")
        
        self.logger.info(f"总体实验报告已保存到: {report_file}")
        return report_file
    
    def validate_experiment_setup(self) -> bool:
        """验证实验环境配置"""
        self.logger.info("验证实验环境配置...")
        
        # 检查必要的目录
        required_dirs = [
            self.output_dir,
            Path("src/asr"),
            Path("src/llm"), 
            Path("src/pipeline"),
            Path("src/utils")
        ]
        
        missing_dirs = []
        for dir_path in required_dirs:
            if not dir_path.exists():
                missing_dirs.append(str(dir_path))
        
        if missing_dirs:
            self.logger.error(f"缺少必要目录: {missing_dirs}")
            return False
        
        # 验证可以创建实验实例
        try:
            test_exp = create_length_impact_experiment(use_small_data=True)
            self.logger.info("实验环境验证成功")
            return True
        except Exception as e:
            self.logger.error(f"实验环境验证失败: {e}")
            return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="级联式语音对话系统延迟优化实验")
    parser.add_argument("--list", action="store_true", help="列出所有可用实验")
    parser.add_argument("--validate", action="store_true", help="验证实验环境")
    parser.add_argument("--run", type=str, help="运行指定实验 (实验ID)")
    parser.add_argument("--run-core", action="store_true", help="运行核心实验")
    parser.add_argument("--run-all", action="store_true", help="运行所有实验")
    parser.add_argument("--run-selected", nargs="+", help="运行选定的实验")
    parser.add_argument("--full-data", action="store_true", help="使用完整数据集（默认使用小数据）")
    parser.add_argument("--output-dir", type=str, default="experiments/results", help="实验结果输出目录")
    
    args = parser.parse_args()
    
    # 创建实验运行器
    use_small_data = not args.full_data
    runner = StreamingExperimentRunner(
        output_dir=args.output_dir,
        use_small_data=use_small_data
    )
    
    if args.list:
        runner.list_experiments()
        return
    
    if args.validate:
        success = runner.validate_experiment_setup()
        if success:
            print("✅ 实验环境验证成功")
        else:
            print("❌ 实验环境验证失败")
        return
    
    # 运行实验
    results = {}
    
    if args.run:
        results = {args.run: runner.run_single_experiment(args.run)}
    elif args.run_core:
        results = runner.run_core_experiments()
    elif args.run_all:
        results = runner.run_all_experiments()
    elif args.run_selected:
        results = runner.run_selected_experiments(args.run_selected)
    else:
        # 默认显示帮助信息
        parser.print_help()
        return
    
    # 生成总体报告
    if results:
        print("\n" + "=" * 60)
        print("实验执行总结:")
        successful = sum(results.values())
        total = len(results)
        print(f"成功: {successful}/{total}")
        
        if successful < total:
            failed_experiments = [exp_id for exp_id, success in results.items() if not success]
            print(f"失败的实验: {', '.join(failed_experiments)}")
        
        # 生成报告
        report_file = runner.generate_overall_report(results)
        print(f"\n📊 总体报告已生成: {report_file}")


if __name__ == "__main__":
    main()