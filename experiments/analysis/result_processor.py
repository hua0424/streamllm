#!/usr/bin/env python3
"""
结果处理器 - 统一处理实验结果分析
调用统计分析器、图表生成器和表格生成器生成完整的分析报告
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

from .statistical_analyzer import ExperimentStatisticalAnalyzer
from .chart_generator import ChartGenerator
from .table_generator import TableGenerator


class ExperimentResultProcessor:
    """实验结果处理器"""
    
    def __init__(self, experiment_results_dir: str, output_dir: str = None):
        self.experiment_results_dir = Path(experiment_results_dir)
        
        if output_dir is None:
            output_dir = self.experiment_results_dir.parent / "analysis_output"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        self.figures_dir = self.output_dir / "figures"
        self.tables_dir = self.output_dir / "tables"
        self.reports_dir = self.output_dir / "reports"
        
        for dir_path in [self.figures_dir, self.tables_dir, self.reports_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # 初始化分析器
        self.statistical_analyzer = ExperimentStatisticalAnalyzer(str(self.experiment_results_dir))
        self.chart_generator = ChartGenerator(str(self.figures_dir))
        self.table_generator = TableGenerator(str(self.tables_dir))
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"结果处理器初始化完成")
        self.logger.info(f"实验结果目录: {self.experiment_results_dir}")
        self.logger.info(f"输出目录: {self.output_dir}")
    
    def discover_experiments(self) -> List[str]:
        """发现所有可用的实验"""
        experiments = []
        
        if not self.experiment_results_dir.exists():
            self.logger.warning(f"实验结果目录不存在: {self.experiment_results_dir}")
            return experiments
        
        for exp_dir in self.experiment_results_dir.iterdir():
            if exp_dir.is_dir():
                result_file = exp_dir / "experiment_results.json"
                if result_file.exists():
                    experiments.append(exp_dir.name)
        
        self.logger.info(f"发现 {len(experiments)} 个实验: {experiments}")
        return experiments
    
    def process_single_experiment(self, experiment_name: str) -> Dict[str, Any]:
        """处理单个实验的结果"""
        self.logger.info(f"开始处理实验: {experiment_name}")
        
        try:
            # 加载实验数据
            experiment_data = self.statistical_analyzer.load_experiment_result(experiment_name)
            
            results = {
                "experiment_name": experiment_name,
                "processing_time": time.time(),
                "status": "success",
                "generated_files": {}
            }
            
            # 1. 统计分析
            try:
                self.logger.info(f"进行统计分析: {experiment_name}")
                statistical_analysis = self.statistical_analyzer.analyze_optimization_effectiveness(experiment_name)
                results["statistical_analysis"] = statistical_analysis
                
                # 生成统计检验表格
                stat_tables = self.table_generator.create_statistical_tests_table(
                    statistical_analysis, f"{experiment_name}_statistical_tests"
                )
                results["generated_files"]["statistical_tables"] = stat_tables
                
            except Exception as e:
                self.logger.error(f"统计分析失败 {experiment_name}: {e}")
                results["statistical_analysis_error"] = str(e)
            
            # 2. 生成基础图表
            try:
                self.logger.info(f"生成基础图表: {experiment_name}")
                latency_chart = self.chart_generator.create_latency_comparison_chart(
                    experiment_data, f"{experiment_name}_latency_comparison"
                )
                results["generated_files"]["latency_chart"] = latency_chart
                
            except Exception as e:
                self.logger.error(f"基础图表生成失败 {experiment_name}: {e}")
                results["chart_error"] = str(e)
            
            # 3. 特定实验的专门分析
            if "ablation" in experiment_name.lower():
                results.update(self._process_ablation_experiment(experiment_name))
            elif "length" in experiment_name.lower():
                results.update(self._process_length_experiment(experiment_name))
            elif "model" in experiment_name.lower() or "comparison" in experiment_name.lower():
                results.update(self._process_model_comparison_experiment(experiment_name))
            elif "concurrent" in experiment_name.lower():
                results.update(self._process_concurrent_experiment(experiment_name))
            elif "quality" in experiment_name.lower():
                results.update(self._process_quality_experiment(experiment_name))
            elif "asr" in experiment_name.lower() and "scale" in experiment_name.lower():
                results.update(self._process_asr_scale_experiment(experiment_name))
            
            self.logger.info(f"实验 {experiment_name} 处理完成")
            return results
            
        except Exception as e:
            self.logger.error(f"处理实验 {experiment_name} 失败: {e}")
            return {
                "experiment_name": experiment_name,
                "processing_time": time.time(),
                "status": "failed",
                "error": str(e)
            }
    
    def _process_ablation_experiment(self, experiment_name: str) -> Dict[str, Any]:
        """处理消融实验"""
        results = {"ablation_analysis": {}}
        
        try:
            # 查找消融分析文件
            ablation_file = self.experiment_results_dir / experiment_name / "ablation_analysis.json"
            
            if ablation_file.exists():
                # 生成消融图表
                ablation_chart = self.chart_generator.create_ablation_chart(
                    str(ablation_file), f"{experiment_name}_ablation"
                )
                results["ablation_analysis"]["chart"] = ablation_chart
                
                # 生成消融表格
                ablation_tables = self.table_generator.create_ablation_results_table(
                    str(ablation_file), f"{experiment_name}_ablation_results"
                )
                results["ablation_analysis"]["tables"] = ablation_tables
                
        except Exception as e:
            self.logger.error(f"消融实验专门分析失败: {e}")
            results["ablation_analysis_error"] = str(e)
        
        return results
    
    def _process_length_experiment(self, experiment_name: str) -> Dict[str, Any]:
        """处理语音长度实验"""
        results = {"length_analysis": {}}
        
        try:
            # 查找长度分析文件
            length_file = self.experiment_results_dir / experiment_name / "length_analysis.json"
            
            if length_file.exists():
                with open(length_file, 'r', encoding='utf-8') as f:
                    length_data = json.load(f)
                
                # 生成长度影响图表
                length_chart = self.chart_generator.create_length_impact_chart(
                    length_data, f"{experiment_name}_length_impact"
                )
                results["length_analysis"]["chart"] = length_chart
                
                # 生成长度影响表格
                length_tables = self.table_generator.create_length_impact_table(
                    str(length_file), f"{experiment_name}_length_impact"
                )
                results["length_analysis"]["tables"] = length_tables
                
        except Exception as e:
            self.logger.error(f"长度实验专门分析失败: {e}")
            results["length_analysis_error"] = str(e)
        
        return results
    
    def _process_model_comparison_experiment(self, experiment_name: str) -> Dict[str, Any]:
        """处理模型对比实验"""
        results = {"model_comparison_analysis": {}}
        
        try:
            # 查找模型对比分析文件
            comparison_file = self.experiment_results_dir / experiment_name / "native_model_comparison_analysis.json"
            
            if comparison_file.exists():
                with open(comparison_file, 'r', encoding='utf-8') as f:
                    comparison_data = json.load(f)
                
                # 生成模型对比图表
                comparison_chart = self.chart_generator.create_model_comparison_chart(
                    comparison_data, f"{experiment_name}_model_comparison"
                )
                results["model_comparison_analysis"]["chart"] = comparison_chart
                
                # 生成模型对比表格
                comparison_tables = self.table_generator.create_model_comparison_table(
                    str(comparison_file), f"{experiment_name}_model_comparison"
                )
                results["model_comparison_analysis"]["tables"] = comparison_tables
                
        except Exception as e:
            self.logger.error(f"模型对比实验专门分析失败: {e}")
            results["model_comparison_analysis_error"] = str(e)
        
        return results
    
    def _process_concurrent_experiment(self, experiment_name: str) -> Dict[str, Any]:
        """处理并发性能实验"""
        results = {"concurrent_analysis": {}}
        
        try:
            # 查找并发分析文件
            concurrent_file = self.experiment_results_dir / experiment_name / "concurrent_performance_analysis.json"
            
            if concurrent_file.exists():
                with open(concurrent_file, 'r', encoding='utf-8') as f:
                    concurrent_data = json.load(f)
                
                # 生成并发性能图表
                concurrent_chart = self.chart_generator.create_concurrent_performance_chart(
                    concurrent_data, f"{experiment_name}_concurrent_performance"
                )
                results["concurrent_analysis"]["chart"] = concurrent_chart
                
        except Exception as e:
            self.logger.error(f"并发实验专门分析失败: {e}")
            results["concurrent_analysis_error"] = str(e)
        
        return results
    
    def _process_quality_experiment(self, experiment_name: str) -> Dict[str, Any]:
        """处理音频质量实验"""
        results = {"quality_analysis": {}}
        
        try:
            # 查找质量分析文件
            quality_file = self.experiment_results_dir / experiment_name / "audio_quality_analysis.json"
            
            if quality_file.exists():
                with open(quality_file, 'r', encoding='utf-8') as f:
                    quality_data = json.load(f)
                
                # 生成质量鲁棒性图表
                quality_chart = self.chart_generator.create_quality_robustness_chart(
                    quality_data, f"{experiment_name}_quality_robustness"
                )
                results["quality_analysis"]["chart"] = quality_chart
                
        except Exception as e:
            self.logger.error(f"质量实验专门分析失败: {e}")
            results["quality_analysis_error"] = str(e)
        
        return results
    
    def _process_asr_scale_experiment(self, experiment_name: str) -> Dict[str, Any]:
        """处理ASR规模实验"""
        results = {"asr_scale_analysis": {}}
        
        try:
            # 查找ASR规模分析文件
            asr_file = self.experiment_results_dir / experiment_name / "asr_model_analysis.json"
            
            if asr_file.exists():
                with open(asr_file, 'r', encoding='utf-8') as f:
                    asr_data = json.load(f)
                
                # 这里可以添加专门的ASR规模图表生成
                # 暂时使用基础的延迟对比图
                
        except Exception as e:
            self.logger.error(f"ASR规模实验专门分析失败: {e}")
            results["asr_scale_analysis_error"] = str(e)
        
        return results
    
    def process_all_experiments(self) -> Dict[str, Any]:
        """处理所有实验"""
        experiments = self.discover_experiments()
        
        if not experiments:
            self.logger.warning("没有发现任何实验")
            return {"error": "没有发现任何实验"}
        
        self.logger.info(f"开始处理 {len(experiments)} 个实验")
        
        processing_results = {
            "processing_start_time": time.time(),
            "total_experiments": len(experiments),
            "experiment_results": {},
            "summary": {}
        }
        
        # 处理各个实验
        successful_experiments = []
        failed_experiments = []
        
        for experiment_name in experiments:
            result = self.process_single_experiment(experiment_name)
            processing_results["experiment_results"][experiment_name] = result
            
            if result["status"] == "success":
                successful_experiments.append(experiment_name)
            else:
                failed_experiments.append(experiment_name)
        
        # 生成综合分析
        if successful_experiments:
            try:
                processing_results.update(self._generate_comprehensive_analysis(successful_experiments))
            except Exception as e:
                self.logger.error(f"综合分析失败: {e}")
                processing_results["comprehensive_analysis_error"] = str(e)
        
        # 生成处理摘要
        processing_results["summary"] = {
            "total_experiments": len(experiments),
            "successful_experiments": len(successful_experiments),
            "failed_experiments": len(failed_experiments),
            "success_rate": len(successful_experiments) / len(experiments) * 100 if experiments else 0,
            "processing_end_time": time.time(),
            "total_processing_time": time.time() - processing_results["processing_start_time"]
        }
        
        # 保存处理结果
        self._save_processing_results(processing_results)
        
        self.logger.info(f"所有实验处理完成，成功: {len(successful_experiments)}, 失败: {len(failed_experiments)}")
        
        return processing_results
    
    def _generate_comprehensive_analysis(self, experiment_names: List[str]) -> Dict[str, Any]:
        """生成综合分析"""
        self.logger.info("生成综合分析")
        
        comprehensive_results = {"comprehensive_analysis": {}}
        
        try:
            # 1. 生成总结仪表板
            dashboard_chart = self.chart_generator.create_summary_dashboard(
                str(self.experiment_results_dir), experiment_names, "experiment_dashboard"
            )
            comprehensive_results["comprehensive_analysis"]["dashboard_chart"] = dashboard_chart
            
            # 2. 生成实验总结表格
            summary_tables = self.table_generator.create_experiment_summary_table(
                str(self.experiment_results_dir), experiment_names, "experiment_summary"
            )
            comprehensive_results["comprehensive_analysis"]["summary_tables"] = summary_tables
            
            # 3. 生成综合结果表格
            comprehensive_tables = self.table_generator.create_comprehensive_results_table(
                str(self.experiment_results_dir), "comprehensive_results"
            )
            comprehensive_results["comprehensive_analysis"]["comprehensive_tables"] = comprehensive_tables
            
            # 4. 实验间对比分析
            if len(experiment_names) > 1:
                comparison_analysis = self.statistical_analyzer.compare_experiments(experiment_names)
                comprehensive_results["comprehensive_analysis"]["inter_experiment_comparison"] = comparison_analysis
                
                # 生成对比统计表格
                comparison_stat_tables = self.table_generator.create_statistical_tests_table(
                    comparison_analysis, "inter_experiment_statistical_tests"
                )
                comprehensive_results["comprehensive_analysis"]["comparison_stat_tables"] = comparison_stat_tables
            
            # 5. 生成统计分析报告
            report_content = self.statistical_analyzer.generate_statistical_report(
                experiment_names, str(self.reports_dir / "statistical_analysis_report.md")
            )
            comprehensive_results["comprehensive_analysis"]["statistical_report"] = str(self.reports_dir / "statistical_analysis_report.md")
            
        except Exception as e:
            self.logger.error(f"综合分析生成失败: {e}")
            comprehensive_results["comprehensive_analysis_error"] = str(e)
        
        return comprehensive_results
    
    def _save_processing_results(self, results: Dict[str, Any]):
        """保存处理结果"""
        # 保存详细的处理结果
        results_file = self.output_dir / "processing_results.json"
        
        # 转换为可序列化的格式
        serializable_results = self._make_serializable(results)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        # 生成处理摘要报告
        self._generate_processing_summary_report(results)
        
        self.logger.info(f"处理结果已保存到: {results_file}")
    
    def _make_serializable(self, obj):
        """使对象可序列化"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            # 处理自定义对象
            return self._make_serializable(obj.__dict__)
        else:
            return obj
    
    def _generate_processing_summary_report(self, results: Dict[str, Any]):
        """生成处理摘要报告"""
        report_file = self.reports_dir / "processing_summary.md"
        
        summary = results.get("summary", {})
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 实验结果处理摘要报告\n\n")
            f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 处理摘要
            f.write("## 处理摘要\n\n")
            f.write(f"- 总实验数量: {summary.get('total_experiments', 0)}\n")
            f.write(f"- 成功处理数量: {summary.get('successful_experiments', 0)}\n")
            f.write(f"- 失败处理数量: {summary.get('failed_experiments', 0)}\n")
            f.write(f"- 成功率: {summary.get('success_rate', 0):.1f}%\n")
            f.write(f"- 总处理时间: {summary.get('total_processing_time', 0):.2f}秒\n\n")
            
            # 各实验处理状态
            f.write("## 实验处理状态\n\n")
            experiment_results = results.get("experiment_results", {})
            
            for exp_name, exp_result in experiment_results.items():
                status = exp_result.get("status", "unknown")
                status_icon = "✅" if status == "success" else "❌"
                
                f.write(f"### {exp_name}\n\n")
                f.write(f"状态: {status_icon} {status}\n")
                
                if status == "success":
                    generated_files = exp_result.get("generated_files", {})
                    if generated_files:
                        f.write("生成的文件:\n")
                        for file_type, file_info in generated_files.items():
                            if isinstance(file_info, dict):
                                for format_type, file_path in file_info.items():
                                    f.write(f"- {file_type} ({format_type}): `{file_path}`\n")
                            else:
                                f.write(f"- {file_type}: `{file_info}`\n")
                else:
                    error = exp_result.get("error", "未知错误")
                    f.write(f"错误: {error}\n")
                
                f.write("\n")
            
            # 生成的文件总览
            f.write("## 生成文件总览\n\n")
            f.write("### 图表文件\n")
            f.write(f"目录: `{self.figures_dir}`\n\n")
            
            f.write("### 表格文件\n")
            f.write(f"目录: `{self.tables_dir}`\n\n")
            
            f.write("### 报告文件\n")
            f.write(f"目录: `{self.reports_dir}`\n\n")
            
            # 使用建议
            f.write("## 使用建议\n\n")
            f.write("1. 查看生成的图表文件，用于论文插图\n")
            f.write("2. 使用LaTeX表格文件插入论文\n")
            f.write("3. 参考统计分析报告进行结果讨论\n")
            f.write("4. 检查处理失败的实验并重新运行\n")
        
        self.logger.info(f"处理摘要报告已保存到: {report_file}")
    
    def generate_quick_summary(self) -> str:
        """生成快速摘要"""
        experiments = self.discover_experiments()
        
        if not experiments:
            return "没有发现任何实验结果"
        
        summary_lines = ["实验结果快速摘要:", "=" * 30]
        
        for exp_name in experiments:
            try:
                data = self.statistical_analyzer.load_experiment_result(exp_name)
                summary_stats = data.get("summary_statistics", {})
                exp_info = data.get("experiment_info", {})
                
                mean_opt = summary_stats.get("mean_optimization", 0)
                sample_count = exp_info.get("sample_count", 0)
                success_count = exp_info.get("success_count", 0)
                
                summary_lines.append(f"{exp_name}:")
                summary_lines.append(f"  样本: {sample_count}, 成功: {success_count}")
                summary_lines.append(f"  平均优化: {mean_opt:.1f}%")
                summary_lines.append("")
                
            except Exception as e:
                summary_lines.append(f"{exp_name}: 读取失败 - {e}")
                summary_lines.append("")
        
        return "\n".join(summary_lines)