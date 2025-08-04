#!/usr/bin/env python3
"""
分析运行器 - 运行所有实验结果分析
"""

import sys
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from .result_processor import ExperimentResultProcessor


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="实验结果分析工具")
    parser.add_argument("--results-dir", type=str, default="experiments/results", 
                       help="实验结果目录")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="分析输出目录（默认为results-dir/analysis_output）")
    parser.add_argument("--experiment", type=str, default=None,
                       help="只分析指定实验")
    parser.add_argument("--quick-summary", action="store_true",
                       help="只显示快速摘要")
    
    args = parser.parse_args()
    
    # 创建结果处理器
    processor = ExperimentResultProcessor(
        experiment_results_dir=args.results_dir,
        output_dir=args.output_dir
    )
    
    if args.quick_summary:
        # 显示快速摘要
        summary = processor.generate_quick_summary()
        print(summary)
        return
    
    if args.experiment:
        # 处理单个实验
        print(f"分析实验: {args.experiment}")
        result = processor.process_single_experiment(args.experiment)
        
        if result["status"] == "success":
            print(f"✅ 实验 {args.experiment} 分析完成")
            generated_files = result.get("generated_files", {})
            if generated_files:
                print("生成的文件:")
                for file_type, file_info in generated_files.items():
                    if isinstance(file_info, dict):
                        for format_type, file_path in file_info.items():
                            print(f"  - {file_type} ({format_type}): {file_path}")
                    else:
                        print(f"  - {file_type}: {file_info}")
        else:
            print(f"❌ 实验 {args.experiment} 分析失败: {result.get('error', '未知错误')}")
    
    else:
        # 处理所有实验
        print("开始分析所有实验...")
        results = processor.process_all_experiments()
        
        summary = results.get("summary", {})
        print(f"\n分析完成:")
        print(f"  总实验数: {summary.get('total_experiments', 0)}")
        print(f"  成功分析: {summary.get('successful_experiments', 0)}")
        print(f"  失败分析: {summary.get('failed_experiments', 0)}")
        print(f"  成功率: {summary.get('success_rate', 0):.1f}%")
        print(f"  总耗时: {summary.get('total_processing_time', 0):.2f}秒")
        
        print(f"\n📊 分析结果保存在: {processor.output_dir}")
        print(f"📈 图表目录: {processor.figures_dir}")
        print(f"📋 表格目录: {processor.tables_dir}")
        print(f"📄 报告目录: {processor.reports_dir}")


if __name__ == "__main__":
    main()