#!/usr/bin/env python3
"""
论文工具运行器 - 统一管理所有论文写作辅助工具
"""

import sys
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from .report_generator import PaperReportGenerator
from .latex_exporter import LaTeXExporter
from .template_generator import PaperTemplateGenerator


class PaperToolsRunner:
    """论文工具运行器"""
    
    def __init__(self, experiment_results_dir: str, output_dir: str = None):
        self.experiment_results_dir = Path(experiment_results_dir)
        
        if output_dir is None:
            output_dir = self.experiment_results_dir.parent / "paper_materials"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        self.reports_dir = self.output_dir / "reports"
        self.latex_dir = self.output_dir / "latex"
        self.templates_dir = self.output_dir / "templates"
        
        for dir_path in [self.reports_dir, self.latex_dir, self.templates_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # 初始化工具
        self.report_generator = PaperReportGenerator(str(self.reports_dir))
        self.latex_exporter = LaTeXExporter(str(self.latex_dir))
        self.template_generator = PaperTemplateGenerator(str(self.templates_dir))
        
        print(f"📝 论文工具初始化完成")
        print(f"📊 实验结果目录: {self.experiment_results_dir}")
        print(f"📄 论文材料输出目录: {self.output_dir}")
    
    def discover_experiments(self) -> list[str]:
        """发现所有实验"""
        experiments = []
        
        if not self.experiment_results_dir.exists():
            print(f"⚠️  实验结果目录不存在: {self.experiment_results_dir}")
            return experiments
        
        for exp_dir in self.experiment_results_dir.iterdir():
            if exp_dir.is_dir():
                result_file = exp_dir / "experiment_results.json"
                if result_file.exists():
                    experiments.append(exp_dir.name)
        
        print(f"🔍 发现 {len(experiments)} 个实验: {experiments}")
        return experiments
    
    def generate_experiment_reports(self, experiment_names: list[str] = None) -> dict[str, str]:
        """生成实验报告"""
        if experiment_names is None:
            experiment_names = self.discover_experiments()
        
        if not experiment_names:
            print("⚠️  没有发现任何实验")
            return {}
        
        print("📊 生成实验报告...")
        
        generated_reports = {}
        
        # 生成单个实验报告
        for exp_name in experiment_names:
            try:
                report_file = self.report_generator.generate_experiment_report(
                    str(self.experiment_results_dir), exp_name
                )
                generated_reports[exp_name] = report_file
                print(f"✅ {exp_name} 报告已生成: {report_file}")
                
            except Exception as e:
                print(f"❌ {exp_name} 报告生成失败: {e}")
        
        # 生成综合报告
        if len(experiment_names) > 1:
            try:
                comprehensive_report = self.report_generator.generate_comprehensive_report(
                    str(self.experiment_results_dir), experiment_names
                )
                generated_reports["comprehensive"] = comprehensive_report
                print(f"✅ 综合报告已生成: {comprehensive_report}")
                
            except Exception as e:
                print(f"❌ 综合报告生成失败: {e}")
        
        return generated_reports
    
    def export_latex_materials(self, experiment_names: list[str] = None) -> dict[str, str]:
        """导出LaTeX材料"""
        if experiment_names is None:
            experiment_names = self.discover_experiments()
        
        if not experiment_names:
            print("⚠️  没有发现任何实验")
            return {}
        
        print("📝 导出LaTeX材料...")
        
        generated_files = {}
        
        # 导出实验结果汇总表格
        try:
            summary_table = self.latex_exporter.export_experiment_results_table(
                str(self.experiment_results_dir), experiment_names
            )
            generated_files["summary_table"] = summary_table
            print(f"✅ 实验汇总表格已导出: {summary_table}")
            
        except Exception as e:
            print(f"❌ 实验汇总表格导出失败: {e}")
        
        # 导出消融实验表格
        for exp_name in experiment_names:
            if "ablation" in exp_name.lower():
                try:
                    ablation_file = self.experiment_results_dir / exp_name / "ablation_analysis.json"
                    if ablation_file.exists():
                        ablation_table = self.latex_exporter.export_ablation_results_table(
                            str(ablation_file), f"{exp_name}_ablation"
                        )
                        generated_files[f"{exp_name}_ablation"] = ablation_table
                        print(f"✅ {exp_name} 消融表格已导出: {ablation_table}")
                        
                except Exception as e:
                    print(f"❌ {exp_name} 消融表格导出失败: {e}")
        
        # 导出模型对比表格
        for exp_name in experiment_names:
            if "comparison" in exp_name.lower() or "model" in exp_name.lower():
                try:
                    comparison_file = self.experiment_results_dir / exp_name / "native_model_comparison_analysis.json"
                    if comparison_file.exists():
                        comparison_table = self.latex_exporter.export_model_comparison_table(
                            str(comparison_file), f"{exp_name}_comparison"
                        )
                        generated_files[f"{exp_name}_comparison"] = comparison_table
                        print(f"✅ {exp_name} 对比表格已导出: {comparison_table}")
                        
                except Exception as e:
                    print(f"❌ {exp_name} 对比表格导出失败: {e}")
        
        # 生成数学公式
        try:
            formulas = self.latex_exporter.export_mathematical_formulas()
            generated_files["formulas"] = formulas
            print(f"✅ 数学公式已导出: {formulas}")
            
        except Exception as e:
            print(f"❌ 数学公式导出失败: {e}")
        
        # 生成参考文献
        try:
            bibliography = self.latex_exporter.generate_bibliography()
            generated_files["bibliography"] = bibliography
            print(f"✅ 参考文献已生成: {bibliography}")
            
        except Exception as e:
            print(f"❌ 参考文献生成失败: {e}")
        
        # 生成论文模板
        try:
            template = self.latex_exporter.generate_paper_template(
                str(self.experiment_results_dir), experiment_names
            )
            generated_files["paper_template"] = template
            print(f"✅ 论文模板已生成: {template}")
            
        except Exception as e:
            print(f"❌ 论文模板生成失败: {e}")
        
        return generated_files
    
    def generate_writing_templates(self, experiment_names: list[str] = None) -> dict[str, str]:
        """生成写作模板"""
        print("📝 生成写作模板...")
        
        # 生成所有模板
        templates = self.template_generator.generate_all_templates(
            str(self.experiment_results_dir)
        )
        
        for template_type, file_path in templates.items():
            print(f"✅ {template_type} 模板已生成: {file_path}")
        
        # 生成写作指南
        try:
            guide = self.template_generator.generate_writing_guide()
            templates["writing_guide"] = guide
            print(f"✅ 写作指南已生成: {guide}")
            
        except Exception as e:
            print(f"❌ 写作指南生成失败: {e}")
        
        # 生成LaTeX命令
        try:
            commands = self.template_generator.generate_latex_commands()
            templates["latex_commands"] = commands
            print(f"✅ LaTeX命令已生成: {commands}")
            
        except Exception as e:
            print(f"❌ LaTeX命令生成失败: {e}")
        
        return templates
    
    def generate_all_materials(self, experiment_names: list[str] = None) -> dict[str, any]:
        """生成所有论文材料"""
        if experiment_names is None:
            experiment_names = self.discover_experiments()
        
        print("🚀 开始生成所有论文材料...")
        
        all_materials = {
            "reports": {},
            "latex": {},
            "templates": {}
        }
        
        # 生成报告
        try:
            all_materials["reports"] = self.generate_experiment_reports(experiment_names)
        except Exception as e:
            print(f"❌ 报告生成过程失败: {e}")
        
        # 导出LaTeX材料
        try:
            all_materials["latex"] = self.export_latex_materials(experiment_names)
        except Exception as e:
            print(f"❌ LaTeX材料导出失败: {e}")
        
        # 生成模板
        try:
            all_materials["templates"] = self.generate_writing_templates(experiment_names)
        except Exception as e:
            print(f"❌ 模板生成失败: {e}")
        
        # 生成材料清单
        self._generate_materials_index(all_materials)
        
        print("🎉 所有论文材料生成完成！")
        return all_materials
    
    def _generate_materials_index(self, materials: dict[str, any]):
        """生成材料清单"""
        index_file = self.output_dir / "materials_index.md"
        
        with open(index_file, 'w', encoding='utf-8') as f:
            f.write("# 论文材料清单\n\n")
            f.write(f"生成时间: {Path().cwd()}\n\n")
            
            # 报告文件
            f.write("## 📊 实验报告\n\n")
            if materials["reports"]:
                for exp_name, report_file in materials["reports"].items():
                    f.write(f"- **{exp_name}**: `{report_file}`\n")
            else:
                f.write("- 无报告文件\n")
            f.write("\n")
            
            # LaTeX文件
            f.write("## 📝 LaTeX材料\n\n")
            if materials["latex"]:
                for file_type, file_path in materials["latex"].items():
                    f.write(f"- **{file_type}**: `{file_path}`\n")
            else:
                f.write("- 无LaTeX文件\n")
            f.write("\n")
            
            # 模板文件
            f.write("## 📋 写作模板\n\n")
            if materials["templates"]:
                for template_type, template_file in materials["templates"].items():
                    f.write(f"- **{template_type}**: `{template_file}`\n")
            else:
                f.write("- 无模板文件\n")
            f.write("\n")
            
            # 使用说明
            f.write("## 📖 使用说明\n\n")
            f.write("### 报告文件\n")
            f.write("- 实验报告包含详细的结果分析和讨论\n")
            f.write("- 综合报告提供所有实验的总结\n\n")
            
            f.write("### LaTeX材料\n")
            f.write("- 表格文件可直接插入论文\n")
            f.write("- 公式文件包含常用数学表达式\n")
            f.write("- 模板文件提供论文框架\n\n")
            
            f.write("### 写作模板\n")
            f.write("- 各章节模板提供写作指导\n")
            f.write("- 写作指南包含详细建议\n")
            f.write("- LaTeX命令简化排版工作\n\n")
        
        print(f"📋 材料清单已生成: {index_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="论文写作辅助工具")
    parser.add_argument("--results-dir", type=str, default="experiments/results",
                       help="实验结果目录")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="论文材料输出目录")
    parser.add_argument("--experiments", nargs="+", default=None,
                       help="指定实验名称")
    parser.add_argument("--reports-only", action="store_true",
                       help="只生成报告")
    parser.add_argument("--latex-only", action="store_true", 
                       help="只导出LaTeX材料")
    parser.add_argument("--templates-only", action="store_true",
                       help="只生成模板")
    
    args = parser.parse_args()
    
    # 创建工具运行器
    runner = PaperToolsRunner(
        experiment_results_dir=args.results_dir,
        output_dir=args.output_dir
    )
    
    if args.reports_only:
        # 只生成报告
        runner.generate_experiment_reports(args.experiments)
    elif args.latex_only:
        # 只导出LaTeX材料
        runner.export_latex_materials(args.experiments)
    elif args.templates_only:
        # 只生成模板
        runner.generate_writing_templates(args.experiments)
    else:
        # 生成所有材料
        runner.generate_all_materials(args.experiments)
    
    print(f"\n📁 所有文件已保存到: {runner.output_dir}")


if __name__ == "__main__":
    main()