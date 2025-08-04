#!/usr/bin/env python3
"""
表格生成器 - 为论文生成各种格式的表格
包括LaTeX、Markdown、CSV等格式
"""

import json
import csv
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
import numpy as np


@dataclass
class TableColumn:
    """表格列定义"""
    name: str
    display_name: str
    format_str: str = "{:.2f}"
    alignment: str = "c"  # l, c, r for left, center, right


class TableGenerator:
    """表格生成器"""
    
    def __init__(self, output_dir: str = "experiments/results/tables"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_experiment_data(self, experiment_results_dir: str, experiment_name: str) -> Dict[str, Any]:
        """加载实验数据"""
        result_file = Path(experiment_results_dir) / experiment_name / "experiment_results.json"
        
        if not result_file.exists():
            raise FileNotFoundError(f"实验结果文件不存在: {result_file}")
        
        with open(result_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def create_experiment_summary_table(self, experiment_results_dir: str, 
                                       experiment_names: List[str],
                                       output_name: str = "experiment_summary") -> Dict[str, str]:
        """创建实验总结表格"""
        # 收集数据
        table_data = []
        
        for exp_name in experiment_names:
            try:
                data = self.load_experiment_data(experiment_results_dir, exp_name)
                summary_stats = data.get("summary_statistics", {})
                exp_info = data.get("experiment_info", {})
                
                row = {
                    "experiment": exp_name.replace("_", " ").title(),
                    "samples": exp_info.get("sample_count", 0),
                    "success": exp_info.get("success_count", 0),
                    "success_rate": (exp_info.get("success_count", 0) / 
                                   max(exp_info.get("sample_count", 1), 1) * 100),
                    "mean_optimization": summary_stats.get("mean_optimization", 0),
                    "std_optimization": summary_stats.get("std_optimization", 0),
                    "mean_baseline": summary_stats.get("mean_baseline_latency", 0),
                    "mean_optimized": summary_stats.get("mean_optimized_latency", 0),
                    "improvement_ms": summary_stats.get("total_improvement_ms", 0)
                }
                table_data.append(row)
                
            except Exception as e:
                print(f"警告：无法加载实验 {exp_name}: {e}")
        
        if not table_data:
            raise ValueError("没有有效的实验数据")
        
        # 定义列
        columns = [
            TableColumn("experiment", "实验名称", "{}", "l"),
            TableColumn("samples", "样本数", "{:.0f}", "c"),
            TableColumn("success_rate", "成功率(%)", "{:.1f}", "c"),
            TableColumn("mean_optimization", "平均优化比例(%)", "{:.1f}", "c"),
            TableColumn("std_optimization", "标准差", "{:.1f}", "c"),
            TableColumn("mean_baseline", "基线延迟(ms)", "{:.0f}", "c"),
            TableColumn("mean_optimized", "优化延迟(ms)", "{:.0f}", "c"),
            TableColumn("improvement_ms", "改进量(ms)", "{:.0f}", "c")
        ]
        
        # 生成各种格式的表格
        output_files = {}
        
        # LaTeX表格
        latex_file = self.output_dir / f"{output_name}.tex"
        latex_content = self._generate_latex_table(table_data, columns, 
                                                  "实验结果总结",
                                                  "tab:experiment_summary")
        with open(latex_file, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        output_files["latex"] = str(latex_file)
        
        # Markdown表格
        markdown_file = self.output_dir / f"{output_name}.md"
        markdown_content = self._generate_markdown_table(table_data, columns, "实验结果总结")
        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        output_files["markdown"] = str(markdown_file)
        
        # CSV表格
        csv_file = self.output_dir / f"{output_name}.csv"
        self._generate_csv_table(table_data, columns, csv_file)
        output_files["csv"] = str(csv_file)
        
        return output_files
    
    def create_ablation_results_table(self, ablation_analysis_file: str,
                                     output_name: str = "ablation_results") -> Dict[str, str]:
        """创建消融实验结果表格"""
        # 加载消融实验数据
        with open(ablation_analysis_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        config_comparison = data.get("config_comparison", {})
        component_contributions = data.get("component_contributions", {})
        
        if not config_comparison:
            raise ValueError("没有有效的消融实验数据")
        
        # 构建表格数据
        table_data = []
        
        # 配置顺序：基线 -> 单独优化 -> 完整优化
        config_order = ["baseline", "asr_only", "llm_only", "full_optimization"]
        
        for config_name in config_order:
            if config_name in config_comparison:
                config_data = config_comparison[config_name]
                config_info = config_data["configuration"]
                
                row = {
                    "configuration": config_info["name"],
                    "streaming_asr": "✓" if config_info["streaming_asr"] else "✗",
                    "kv_cache": "✓" if config_info["kv_cache"] else "✗",
                    "mean_latency": config_data["mean_latency"],
                    "std_latency": config_data["std_latency"],
                    "improvement": config_data.get("improvement_over_baseline", 0),
                    "sample_count": config_data["sample_count"]
                }
                table_data.append(row)
        
        # 定义列
        columns = [
            TableColumn("configuration", "配置", "{}", "l"),
            TableColumn("streaming_asr", "流式ASR", "{}", "c"),
            TableColumn("kv_cache", "KV缓存", "{}", "c"),
            TableColumn("mean_latency", "平均延迟(ms)", "{:.0f}", "c"),
            TableColumn("std_latency", "标准差", "{:.1f}", "c"),
            TableColumn("improvement", "相比基线改进(%)", "{:.1f}", "c"),
            TableColumn("sample_count", "样本数", "{:.0f}", "c")
        ]
        
        # 生成表格
        output_files = {}
        
        # LaTeX表格
        latex_file = self.output_dir / f"{output_name}.tex"
        latex_content = self._generate_latex_table(table_data, columns, 
                                                  "消融实验结果",
                                                  "tab:ablation_results")
        with open(latex_file, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        output_files["latex"] = str(latex_file)
        
        # 添加组件贡献度表格
        if component_contributions:
            contrib_data = [{
                "component": "流式ASR",
                "contribution": component_contributions.get("asr_streaming_contribution", 0),
                "description": "并行音频处理，减少等待时间"
            }, {
                "component": "KV缓存LLM",
                "contribution": component_contributions.get("llm_kv_cache_contribution", 0),
                "description": "缓存键值对，加速生成"
            }, {
                "component": "协同效应",
                "contribution": component_contributions.get("interaction_effect", 0),
                "description": "两种优化的交互作用"
            }]
            
            contrib_columns = [
                TableColumn("component", "优化组件", "{}", "l"),
                TableColumn("contribution", "贡献度(%)", "{:.1f}", "c"),
                TableColumn("description", "作用机制", "{}", "l")
            ]
            
            contrib_latex_file = self.output_dir / f"{output_name}_contributions.tex"
            contrib_latex_content = self._generate_latex_table(contrib_data, contrib_columns,
                                                              "优化组件贡献度分析",
                                                              "tab:component_contributions")
            with open(contrib_latex_file, 'w', encoding='utf-8') as f:
                f.write(contrib_latex_content)
            output_files["contributions_latex"] = str(contrib_latex_file)
        
        # Markdown表格
        markdown_file = self.output_dir / f"{output_name}.md"
        markdown_content = self._generate_markdown_table(table_data, columns, "消融实验结果")
        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        output_files["markdown"] = str(markdown_file)
        
        return output_files
    
    def create_model_comparison_table(self, comparison_analysis_file: str,
                                     output_name: str = "model_comparison") -> Dict[str, str]:
        """创建模型对比表格"""
        # 加载模型对比数据
        with open(comparison_analysis_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        model_comparison = data.get("model_comparison", {})
        
        if not model_comparison:
            raise ValueError("没有有效的模型对比数据")
        
        # 构建表格数据
        table_data = []
        
        # 分类并排序：级联模型在前，原生模型在后
        cascaded_models = []
        native_models = []
        
        for model_name, model_data in model_comparison.items():
            model_config = model_data["model_config"]
            metrics = model_data["performance_metrics"]
            
            row = {
                "model_name": model_config["name"],
                "model_type": "级联式" if model_config["type"] == "cascaded" else "原生语音",
                "model_size": model_config.get("size", "N/A"),
                "optimization": model_config.get("optimization", "N/A"),
                "mean_latency": metrics["mean_latency"],
                "std_latency": metrics["std_latency"],
                "response_quality": metrics["mean_response_quality"],
                "relative_performance": metrics["mean_relative_performance"]
            }
            
            if model_config["type"] == "cascaded":
                cascaded_models.append(row)
            else:
                native_models.append(row)
        
        # 合并数据，级联模型在前
        table_data = cascaded_models + native_models
        
        # 定义列
        columns = [
            TableColumn("model_name", "模型名称", "{}", "l"),
            TableColumn("model_type", "类型", "{}", "c"),
            TableColumn("model_size", "规模", "{}", "c"),
            TableColumn("mean_latency", "平均延迟(ms)", "{:.0f}", "c"),
            TableColumn("std_latency", "标准差", "{:.1f}", "c"),
            TableColumn("response_quality", "响应质量", "{:.1f}", "c"),
            TableColumn("relative_performance", "相对性能(%)", "{:.1f}", "c")
        ]
        
        # 生成表格
        output_files = {}
        
        # LaTeX表格
        latex_file = self.output_dir / f"{output_name}.tex"
        latex_content = self._generate_latex_table(table_data, columns,
                                                  "级联方案与原生模型性能对比",
                                                  "tab:model_comparison")
        with open(latex_file, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        output_files["latex"] = str(latex_file)
        
        # Markdown表格
        markdown_file = self.output_dir / f"{output_name}.md"
        markdown_content = self._generate_markdown_table(table_data, columns, "模型性能对比")
        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        output_files["markdown"] = str(markdown_file)
        
        return output_files
    
    def create_statistical_tests_table(self, statistical_analysis: Dict[str, Any],
                                      output_name: str = "statistical_tests") -> Dict[str, str]:
        """创建统计检验结果表格"""
        # 构建统计检验结果表格
        table_data = []
        
        # 如果有配对t检验结果
        if "paired_t_test" in statistical_analysis:
            t_test = statistical_analysis["paired_t_test"]
            table_data.append({
                "test_name": "配对t检验",
                "statistic": t_test.statistic if hasattr(t_test, 'statistic') else 0,
                "p_value": t_test.p_value if hasattr(t_test, 'p_value') else 1,
                "effect_size": t_test.effect_size if hasattr(t_test, 'effect_size') else 0,
                "significant": "是" if getattr(t_test, 'significant', False) else "否",
                "interpretation": getattr(t_test, 'interpretation', "无")
            })
        
        # 如果有方差分析结果
        if "anova" in statistical_analysis:
            anova = statistical_analysis["anova"]
            table_data.append({
                "test_name": "单因素方差分析",
                "statistic": anova.statistic if hasattr(anova, 'statistic') else 0,
                "p_value": anova.p_value if hasattr(anova, 'p_value') else 1,
                "effect_size": anova.effect_size if hasattr(anova, 'effect_size') else 0,
                "significant": "是" if getattr(anova, 'significant', False) else "否",
                "interpretation": getattr(anova, 'interpretation', "无")
            })
        
        # 如果有相关分析结果
        for corr_type in ["pearson_correlation", "spearman_correlation"]:
            if corr_type in statistical_analysis:
                corr = statistical_analysis[corr_type]
                table_data.append({
                    "test_name": f"{corr_type.replace('_', ' ').title()}",
                    "statistic": corr.statistic if hasattr(corr, 'statistic') else 0,
                    "p_value": corr.p_value if hasattr(corr, 'p_value') else 1,
                    "effect_size": abs(corr.statistic) if hasattr(corr, 'statistic') else 0,
                    "significant": "是" if getattr(corr, 'significant', False) else "否",
                    "interpretation": getattr(corr, 'interpretation', "无")
                })
        
        if not table_data:
            table_data.append({
                "test_name": "无统计检验结果",
                "statistic": 0,
                "p_value": 1,
                "effect_size": 0,
                "significant": "否",
                "interpretation": "没有可用的统计检验数据"
            })
        
        # 定义列
        columns = [
            TableColumn("test_name", "统计检验", "{}", "l"),
            TableColumn("statistic", "检验统计量", "{:.3f}", "c"),
            TableColumn("p_value", "p值", "{:.3f}", "c"),
            TableColumn("effect_size", "效应量", "{:.3f}", "c"),
            TableColumn("significant", "显著性", "{}", "c"),
            TableColumn("interpretation", "解释", "{}", "l")
        ]
        
        # 生成表格
        output_files = {}
        
        # LaTeX表格
        latex_file = self.output_dir / f"{output_name}.tex"
        latex_content = self._generate_latex_table(table_data, columns,
                                                  "统计检验结果",
                                                  "tab:statistical_tests")
        with open(latex_file, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        output_files["latex"] = str(latex_file)
        
        # Markdown表格
        markdown_file = self.output_dir / f"{output_name}.md"
        markdown_content = self._generate_markdown_table(table_data, columns, "统计检验结果")
        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        output_files["markdown"] = str(markdown_file)
        
        return output_files
    
    def create_length_impact_table(self, length_analysis_file: str,
                                  output_name: str = "length_impact") -> Dict[str, str]:
        """创建语音长度影响表格"""
        # 加载长度分析数据
        with open(length_analysis_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        length_statistics = data.get("length_statistics", {})
        
        if not length_statistics:
            raise ValueError("没有有效的长度分析数据")
        
        # 构建表格数据
        table_data = []
        
        # 按长度排序
        sorted_lengths = sorted(length_statistics.items(), 
                               key=lambda x: int(x[0].replace('s', '')))
        
        for length_key, stats in sorted_lengths:
            length = int(length_key.replace('s', ''))
            
            row = {
                "length": length,
                "sample_count": stats["sample_count"],
                "mean_optimization": stats["mean_optimization"],
                "std_optimization": stats["std_optimization"],
                "mean_baseline": stats["mean_baseline_latency"],
                "mean_optimized": stats["mean_optimized_latency"]
            }
            table_data.append(row)
        
        # 定义列
        columns = [
            TableColumn("length", "语音长度(s)", "{:.0f}", "c"),
            TableColumn("sample_count", "样本数", "{:.0f}", "c"),
            TableColumn("mean_optimization", "平均优化比例(%)", "{:.1f}", "c"),
            TableColumn("std_optimization", "标准差", "{:.1f}", "c"),
            TableColumn("mean_baseline", "基线延迟(ms)", "{:.0f}", "c"),
            TableColumn("mean_optimized", "优化延迟(ms)", "{:.0f}", "c")
        ]
        
        # 生成表格
        output_files = {}
        
        # LaTeX表格
        latex_file = self.output_dir / f"{output_name}.tex"
        latex_content = self._generate_latex_table(table_data, columns,
                                                  "语音长度对优化效果的影响",
                                                  "tab:length_impact")
        with open(latex_file, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        output_files["latex"] = str(latex_file)
        
        # 添加相关性分析表格
        correlation_data = data.get("overall_correlation", {})
        if correlation_data and "correlation_coefficient" in correlation_data:
            corr_table_data = [{
                "metric": "相关系数",
                "value": correlation_data["correlation_coefficient"],
                "interpretation": self._interpret_correlation(correlation_data["correlation_coefficient"])
            }, {
                "metric": "回归斜率",
                "value": correlation_data.get("regression_slope", 0),
                "interpretation": "每秒语音长度的优化改进"
            }, {
                "metric": "数据点数",
                "value": correlation_data.get("data_points", 0),
                "interpretation": "参与分析的长度组数"
            }]
            
            corr_columns = [
                TableColumn("metric", "指标", "{}", "l"),
                TableColumn("value", "数值", "{:.3f}", "c"),
                TableColumn("interpretation", "解释", "{}", "l")
            ]
            
            corr_latex_file = self.output_dir / f"{output_name}_correlation.tex"
            corr_latex_content = self._generate_latex_table(corr_table_data, corr_columns,
                                                           "语音长度与优化效果相关性分析",
                                                           "tab:length_correlation")
            with open(corr_latex_file, 'w', encoding='utf-8') as f:
                f.write(corr_latex_content)
            output_files["correlation_latex"] = str(corr_latex_file)
        
        # Markdown表格
        markdown_file = self.output_dir / f"{output_name}.md"
        markdown_content = self._generate_markdown_table(table_data, columns, "语音长度影响分析")
        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        output_files["markdown"] = str(markdown_file)
        
        return output_files
    
    def _generate_latex_table(self, data: List[Dict[str, Any]], 
                             columns: List[TableColumn],
                             caption: str, label: str) -> str:
        """生成LaTeX表格"""
        # 表格头部
        alignment = "".join([col.alignment for col in columns])
        latex_lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            f"\\begin{{tabular}}{{{alignment}}}",
            "\\toprule"
        ]
        
        # 表头
        header = " & ".join([col.display_name for col in columns]) + " \\\\"
        latex_lines.append(header)
        latex_lines.append("\\midrule")
        
        # 数据行
        for row in data:
            formatted_values = []
            for col in columns:
                value = row.get(col.name, "")
                if isinstance(value, (int, float)) and col.format_str != "{}":
                    formatted_value = col.format_str.format(value)
                else:
                    formatted_value = str(value)
                formatted_values.append(formatted_value)
            
            row_line = " & ".join(formatted_values) + " \\\\"
            latex_lines.append(row_line)
        
        # 表格尾部
        latex_lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}"
        ])
        
        return "\n".join(latex_lines)
    
    def _generate_markdown_table(self, data: List[Dict[str, Any]], 
                                columns: List[TableColumn],
                                title: str) -> str:
        """生成Markdown表格"""
        markdown_lines = [f"# {title}\n"]
        
        # 表头
        header = "| " + " | ".join([col.display_name for col in columns]) + " |"
        markdown_lines.append(header)
        
        # 分隔符
        separator_parts = []
        for col in columns:
            if col.alignment == "l":
                separator_parts.append(":---")
            elif col.alignment == "r":
                separator_parts.append("---:")
            else:
                separator_parts.append(":---:")
        
        separator = "| " + " | ".join(separator_parts) + " |"
        markdown_lines.append(separator)
        
        # 数据行
        for row in data:
            formatted_values = []
            for col in columns:
                value = row.get(col.name, "")
                if isinstance(value, (int, float)) and col.format_str != "{}":
                    formatted_value = col.format_str.format(value)
                else:
                    formatted_value = str(value)
                formatted_values.append(formatted_value)
            
            row_line = "| " + " | ".join(formatted_values) + " |"
            markdown_lines.append(row_line)
        
        return "\n".join(markdown_lines)
    
    def _generate_csv_table(self, data: List[Dict[str, Any]], 
                           columns: List[TableColumn],
                           output_file: Path):
        """生成CSV表格"""
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [col.name for col in columns]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # 写入表头（使用显示名称）
            header_row = {col.name: col.display_name for col in columns}
            writer.writerow(header_row)
            
            # 写入数据
            for row in data:
                formatted_row = {}
                for col in columns:
                    value = row.get(col.name, "")
                    if isinstance(value, (int, float)) and col.format_str != "{}":
                        formatted_value = col.format_str.format(value)
                    else:
                        formatted_value = str(value)
                    formatted_row[col.name] = formatted_value
                writer.writerow(formatted_row)
    
    def _interpret_correlation(self, correlation: float) -> str:
        """解释相关系数"""
        abs_corr = abs(correlation)
        if abs_corr >= 0.8:
            strength = "强"
        elif abs_corr >= 0.5:
            strength = "中等"
        elif abs_corr >= 0.3:
            strength = "弱"
        else:
            strength = "很弱"
        
        direction = "正" if correlation > 0 else "负"
        return f"{direction}相关，{strength}相关性"
    
    def create_comprehensive_results_table(self, experiment_results_dir: str,
                                          output_name: str = "comprehensive_results") -> Dict[str, str]:
        """创建综合结果表格，包含所有关键指标"""
        # 扫描所有实验结果
        results_dir = Path(experiment_results_dir)
        experiment_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
        
        table_data = []
        
        for exp_dir in experiment_dirs:
            exp_name = exp_dir.name
            result_file = exp_dir / "experiment_results.json"
            
            if result_file.exists():
                try:
                    with open(result_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    summary_stats = data.get("summary_statistics", {})
                    exp_info = data.get("experiment_info", {})
                    
                    # 计算关键指标
                    row = {
                        "experiment": exp_name.replace("_experiment", "").replace("_", " ").title(),
                        "samples": exp_info.get("sample_count", 0),
                        "success_rate": (exp_info.get("success_count", 0) / 
                                       max(exp_info.get("sample_count", 1), 1) * 100),
                        "mean_optimization": summary_stats.get("mean_optimization", 0),
                        "std_optimization": summary_stats.get("std_optimization", 0),
                        "ci_lower": summary_stats.get("confidence_interval_lower", 0),
                        "ci_upper": summary_stats.get("confidence_interval_upper", 0),
                        "baseline_latency": summary_stats.get("mean_baseline_latency", 0),
                        "optimized_latency": summary_stats.get("mean_optimized_latency", 0),
                        "improvement_ms": summary_stats.get("total_improvement_ms", 0)
                    }
                    table_data.append(row)
                    
                except Exception as e:
                    print(f"警告：无法处理实验 {exp_name}: {e}")
        
        if not table_data:
            raise ValueError("没有找到有效的实验结果")
        
        # 按平均优化比例排序
        table_data.sort(key=lambda x: x["mean_optimization"], reverse=True)
        
        # 定义列
        columns = [
            TableColumn("experiment", "实验类型", "{}", "l"),
            TableColumn("samples", "样本数", "{:.0f}", "c"),
            TableColumn("success_rate", "成功率(%)", "{:.1f}", "c"),
            TableColumn("mean_optimization", "平均优化比例(%)", "{:.1f}", "c"),
            TableColumn("ci_lower", "95%CI下限", "{:.1f}", "c"),
            TableColumn("ci_upper", "95%CI上限", "{:.1f}", "c"),
            TableColumn("baseline_latency", "基线延迟(ms)", "{:.0f}", "c"),
            TableColumn("optimized_latency", "优化延迟(ms)", "{:.0f}", "c"),
            TableColumn("improvement_ms", "绝对改进(ms)", "{:.0f}", "c")
        ]
        
        # 生成表格
        output_files = {}
        
        # LaTeX表格
        latex_file = self.output_dir / f"{output_name}.tex"
        latex_content = self._generate_latex_table(table_data, columns,
                                                  "级联式语音对话系统延迟优化实验综合结果",
                                                  "tab:comprehensive_results")
        with open(latex_file, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        output_files["latex"] = str(latex_file)
        
        # Markdown表格
        markdown_file = self.output_dir / f"{output_name}.md"
        markdown_content = self._generate_markdown_table(table_data, columns, "实验综合结果")
        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        output_files["markdown"] = str(markdown_file)
        
        # CSV表格
        csv_file = self.output_dir / f"{output_name}.csv"
        self._generate_csv_table(table_data, columns, csv_file)
        output_files["csv"] = str(csv_file)
        
        return output_files