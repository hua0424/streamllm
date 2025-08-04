#!/usr/bin/env python3
"""
LaTeX导出器 - 生成论文需要的LaTeX格式内容
包括表格、图片引用、公式等
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class LaTeXFigure:
    """LaTeX图片引用"""
    filename: str
    caption: str
    label: str
    width: str = "0.8\\textwidth"
    placement: str = "htbp"


@dataclass
class LaTeXTable:
    """LaTeX表格"""
    data: List[List[str]]
    headers: List[str]
    caption: str
    label: str
    alignment: str = None
    placement: str = "htbp"


class LaTeXExporter:
    """LaTeX导出器"""
    
    def __init__(self, output_dir: str = "experiments/results/latex"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # LaTeX包和设置
        self.packages = [
            "\\usepackage{booktabs}",
            "\\usepackage{array}",
            "\\usepackage{multirow}",
            "\\usepackage{graphicx}",
            "\\usepackage{float}",
            "\\usepackage{caption}",
            "\\usepackage{subcaption}",
            "\\usepackage{amsmath}",
            "\\usepackage{amsfonts}",
            "\\usepackage{amssymb}",
            "\\usepackage{siunitx}",
            "\\usepackage{xcolor}"
        ]
    
    def export_experiment_results_table(self, experiment_results_dir: str,
                                       experiment_names: List[str],
                                       table_name: str = "experiment_summary") -> str:
        """导出实验结果汇总表格"""
        # 收集数据
        table_data = []
        
        for exp_name in experiment_names:
            try:
                result_file = Path(experiment_results_dir) / exp_name / "experiment_results.json"
                with open(result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                summary_stats = data.get("summary_statistics", {})
                exp_info = data.get("experiment_info", {})
                
                # 格式化实验名称
                display_name = self._format_experiment_name(exp_name)
                
                row = [
                    display_name,
                    str(exp_info.get("sample_count", 0)),
                    f"{exp_info.get('success_count', 0)}/{exp_info.get('sample_count', 1)}",
                    f"{summary_stats.get('mean_optimization', 0):.1f}",
                    f"{summary_stats.get('std_optimization', 0):.1f}",
                    f"{summary_stats.get('mean_baseline_latency', 0):.0f}",
                    f"{summary_stats.get('mean_optimized_latency', 0):.0f}"
                ]
                table_data.append(row)
                
            except Exception as e:
                print(f"警告：无法加载实验 {exp_name}: {e}")
        
        headers = [
            "实验类型",
            "样本数",
            "成功数/总数",
            "优化比例(\\%)",
            "标准差",
            "基线延迟(ms)",
            "优化延迟(ms)"
        ]
        
        alignment = "l" + "c" * (len(headers) - 1)
        
        latex_table = LaTeXTable(
            data=table_data,
            headers=headers,
            caption="级联式语音对话系统延迟优化实验结果汇总",
            label="tab:experiment_summary",
            alignment=alignment
        )
        
        return self._generate_latex_table(latex_table, table_name)
    
    def export_ablation_results_table(self, ablation_analysis_file: str,
                                     table_name: str = "ablation_results") -> str:
        """导出消融实验结果表格"""
        with open(ablation_analysis_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        config_comparison = data.get("config_comparison", {})
        
        if not config_comparison:
            raise ValueError("没有有效的消融实验数据")
        
        # 构建表格数据
        table_data = []
        config_order = ["baseline", "asr_only", "llm_only", "full_optimization"]
        
        for config_name in config_order:
            if config_name in config_comparison:
                config_data = config_comparison[config_name]
                config_info = config_data["configuration"]
                
                row = [
                    config_info["name"],
                    "\\checkmark" if config_info["streaming_asr"] else "\\texttimes",
                    "\\checkmark" if config_info["kv_cache"] else "\\texttimes",
                    f"{config_data['mean_latency']:.0f}",
                    f"{config_data['std_latency']:.1f}",
                    f"{config_data.get('improvement_over_baseline', 0):.1f}"
                ]
                table_data.append(row)
        
        headers = [
            "配置",
            "流式ASR",
            "KV缓存",
            "延迟(ms)",
            "标准差",
            "改进(\\%)"
        ]
        
        alignment = "l" + "c" * (len(headers) - 1)
        
        latex_table = LaTeXTable(
            data=table_data,
            headers=headers,
            caption="消融实验结果：不同优化组件的性能贡献",
            label="tab:ablation_results",
            alignment=alignment
        )
        
        return self._generate_latex_table(latex_table, table_name)
    
    def export_statistical_tests_table(self, statistical_analysis: Dict[str, Any],
                                      table_name: str = "statistical_tests") -> str:
        """导出统计检验结果表格"""
        table_data = []
        
        # 配对t检验
        if "paired_t_test" in statistical_analysis:
            t_test = statistical_analysis["paired_t_test"]
            row = [
                "配对t检验",
                f"{t_test.statistic:.3f}",
                f"{t_test.p_value:.3f}",
                f"{t_test.effect_size:.3f}",
                "是" if t_test.significant else "否"
            ]
            table_data.append(row)
        
        # 方差分析
        if "anova" in statistical_analysis:
            anova = statistical_analysis["anova"]
            row = [
                "单因素ANOVA",
                f"{anova.statistic:.3f}",
                f"{anova.p_value:.3f}",
                f"{anova.effect_size:.3f}",
                "是" if anova.significant else "否"
            ]
            table_data.append(row)
        
        # 相关分析
        for corr_type in ["pearson_correlation", "spearman_correlation"]:
            if corr_type in statistical_analysis:
                corr = statistical_analysis[corr_type]
                test_name = "Pearson相关" if "pearson" in corr_type else "Spearman相关"
                row = [
                    test_name,
                    f"{corr.statistic:.3f}",
                    f"{corr.p_value:.3f}",
                    f"{abs(corr.statistic):.3f}",
                    "是" if corr.significant else "否"
                ]
                table_data.append(row)
        
        if not table_data:
            # 添加空行
            table_data.append(["无统计检验数据", "-", "-", "-", "-"])
        
        headers = [
            "统计检验",
            "检验统计量",
            "$p$值",
            "效应量",
            "显著性"
        ]
        
        alignment = "l" + "c" * (len(headers) - 1)
        
        latex_table = LaTeXTable(
            data=table_data,
            headers=headers,
            caption="统计检验结果",
            label="tab:statistical_tests",
            alignment=alignment
        )
        
        return self._generate_latex_table(latex_table, table_name)
    
    def export_model_comparison_table(self, comparison_analysis_file: str,
                                     table_name: str = "model_comparison") -> str:
        """导出模型对比表格"""
        with open(comparison_analysis_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        model_comparison = data.get("model_comparison", {})
        
        if not model_comparison:
            raise ValueError("没有有效的模型对比数据")
        
        # 构建表格数据
        table_data = []
        
        # 分类并排序
        cascaded_models = []
        native_models = []
        
        for model_name, model_data in model_comparison.items():
            model_config = model_data["model_config"]
            metrics = model_data["performance_metrics"]
            
            row = [
                model_config["name"],
                "级联" if model_config["type"] == "cascaded" else "原生",
                model_config.get("size", "N/A"),
                f"{metrics['mean_latency']:.0f}",
                f"{metrics['std_latency']:.1f}",
                f"{metrics['mean_response_quality']:.1f}",
                f"{metrics['mean_relative_performance']:.1f}"
            ]
            
            if model_config["type"] == "cascaded":
                cascaded_models.append(row)
            else:
                native_models.append(row)
        
        # 合并数据，级联模型在前
        table_data = cascaded_models + native_models
        
        headers = [
            "模型",
            "类型",
            "规模",
            "延迟(ms)",
            "标准差",
            "质量评分",
            "相对性能(\\%)"
        ]
        
        alignment = "l" + "c" * (len(headers) - 1)
        
        latex_table = LaTeXTable(
            data=table_data,
            headers=headers,
            caption="级联方案与原生语音模型性能对比",
            label="tab:model_comparison",
            alignment=alignment
        )
        
        return self._generate_latex_table(latex_table, table_name)
    
    def _generate_latex_table(self, latex_table: LaTeXTable, filename: str) -> str:
        """生成LaTeX表格"""
        if latex_table.alignment is None:
            latex_table.alignment = "c" * len(latex_table.headers)
        
        # 构建LaTeX代码
        latex_lines = [
            f"\\begin{{table}}[{latex_table.placement}]",
            "\\centering",
            f"\\caption{{{latex_table.caption}}}",
            f"\\label{{{latex_table.label}}}",
            f"\\begin{{tabular}}{{{latex_table.alignment}}}",
            "\\toprule"
        ]
        
        # 表头
        header_line = " & ".join(latex_table.headers) + " \\\\"
        latex_lines.append(header_line)
        latex_lines.append("\\midrule")
        
        # 数据行
        for row in latex_table.data:
            row_line = " & ".join(str(cell) for cell in row) + " \\\\"
            latex_lines.append(row_line)
        
        # 表格尾部
        latex_lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}"
        ])
        
        latex_content = "\n".join(latex_lines)
        
        # 保存文件
        output_file = self.output_dir / f"{filename}.tex"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        
        return str(output_file)
    
    def export_figure_references(self, figures_dir: str, 
                                caption_mapping: Dict[str, str] = None) -> str:
        """导出图片引用"""
        figures_path = Path(figures_dir)
        
        if not figures_path.exists():
            raise FileNotFoundError(f"图片目录不存在: {figures_path}")
        
        # 扫描图片文件
        figure_files = list(figures_path.glob("*.png")) + list(figures_path.glob("*.pdf"))
        
        if not figure_files:
            raise ValueError("没有找到图片文件")
        
        latex_figures = []
        
        for fig_file in sorted(figure_files):
            filename = fig_file.stem
            
            # 生成标题
            if caption_mapping and filename in caption_mapping:
                caption = caption_mapping[filename]
            else:
                caption = self._generate_figure_caption(filename)
            
            # 生成标签
            label = f"fig:{filename}"
            
            latex_figure = LaTeXFigure(
                filename=fig_file.name,
                caption=caption,
                label=label
            )
            
            latex_figures.append(latex_figure)
        
        # 生成LaTeX代码
        latex_content = self._generate_figures_latex(latex_figures)
        
        # 保存文件
        output_file = self.output_dir / "figures.tex"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        
        return str(output_file)
    
    def _generate_figures_latex(self, figures: List[LaTeXFigure]) -> str:
        """生成图片LaTeX代码"""
        latex_lines = []
        
        for figure in figures:
            fig_latex = f"""
\\begin{{figure}}[{figure.placement}]
\\centering
\\includegraphics[width={figure.width}]{{figures/{figure.filename}}}
\\caption{{{figure.caption}}}
\\label{{{figure.label}}}
\\end{{figure}}
"""
            latex_lines.append(fig_latex)
        
        return "\n".join(latex_lines)
    
    def _generate_figure_caption(self, filename: str) -> str:
        """生成图片标题"""
        caption_mapping = {
            "latency_comparison": "基线延迟与优化延迟对比",
            "ablation": "消融实验结果分析",
            "length_impact": "语音长度对优化效果的影响",
            "model_comparison": "不同模型性能对比",
            "concurrent_performance": "并发性能测试结果",
            "quality_robustness": "音频质量鲁棒性分析",
            "experiment_dashboard": "实验结果总览仪表板"
        }
        
        # 查找匹配的关键词
        for key, caption in caption_mapping.items():
            if key in filename.lower():
                return caption
        
        # 默认标题
        formatted_name = filename.replace("_", " ").title()
        return f"{formatted_name}实验结果"
    
    def _format_experiment_name(self, exp_name: str) -> str:
        """格式化实验名称"""
        name_mapping = {
            "length_impact": "语音长度影响",
            "ablation": "消融实验",
            "asr_scale": "ASR模型规模",
            "audio_quality": "音频质量鲁棒性",
            "native_comparison": "原生模型对比",
            "concurrent_performance": "并发性能"
        }
        
        # 查找匹配的关键词
        for key, display_name in name_mapping.items():
            if key in exp_name.lower():
                return display_name
        
        # 默认格式化
        return exp_name.replace("_experiment", "").replace("_", " ").title()
    
    def generate_paper_template(self, experiment_results_dir: str,
                               experiment_names: List[str]) -> str:
        """生成论文模板"""
        template_lines = [
            "\\documentclass[conference]{IEEEtran}",
            "",
            "% 导入包",
            *self.packages,
            "",
            "\\begin{document}",
            "",
            "\\title{级联式语音对话系统延迟优化研究}",
            "",
            "\\author{",
            "\\IEEEauthorblockN{作者姓名}",
            "\\IEEEauthorblockA{",
            "学校名称\\\\",
            "学院名称\\\\",
            "邮箱地址}",
            "}",
            "",
            "\\maketitle",
            "",
            "\\begin{abstract}",
            "% 这里插入摘要内容",
            "\\end{abstract}",
            "",
            "\\section{引言}",
            "% 引言内容",
            "",
            "\\section{相关工作}",
            "% 相关工作",
            "",
            "\\section{方法}",
            "% 方法描述",
            "",
            "\\section{实验}",
            "",
            "\\subsection{实验设置}",
            "% 实验设置",
            ""
        ]
        
        # 为每个实验添加小节
        for exp_name in experiment_names:
            exp_title = self._format_experiment_name(exp_name)
            template_lines.extend([
                f"\\subsection{{{exp_title}}}",
                f"% {exp_title}实验内容",
                f"% 参考表格: \\ref{{tab:{exp_name}_results}}",
                f"% 参考图片: \\ref{{fig:{exp_name}_comparison}}",
                ""
            ])
        
        template_lines.extend([
            "\\section{结果与讨论}",
            "% 结果讨论",
            "",
            "\\section{结论}",
            "% 结论",
            "",
            "\\bibliographystyle{IEEEtran}",
            "\\bibliography{references}",
            "",
            "\\end{document}"
        ])
        
        template_content = "\n".join(template_lines)
        
        # 保存模板
        template_file = self.output_dir / "paper_template.tex"
        with open(template_file, 'w', encoding='utf-8') as f:
            f.write(template_content)
        
        return str(template_file)
    
    def export_mathematical_formulas(self) -> str:
        """导出数学公式"""
        formulas = {
            "optimization_ratio": r"""
优化比例的计算公式：
\begin{equation}
\text{Optimization Ratio} = \frac{T_{\text{baseline}} - T_{\text{optimized}}}{T_{\text{baseline}}} \times 100\%
\end{equation}
其中，$T_{\text{baseline}}$为基线方法的延迟，$T_{\text{optimized}}$为优化方法的延迟。
""",
            
            "cohen_d": r"""
Cohen's d效应量的计算公式：
\begin{equation}
d = \frac{\bar{x}_1 - \bar{x}_2}{s_p}
\end{equation}
其中，$s_p = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1+n_2-2}}$为合并标准差。
""",
            
            "confidence_interval": r"""
置信区间的计算公式：
\begin{equation}
CI = \bar{x} \pm t_{\alpha/2,df} \cdot \frac{s}{\sqrt{n}}
\end{equation}
其中，$t_{\alpha/2,df}$为t分布的临界值。
""",
            
            "streaming_latency": r"""
流式处理延迟模型：
\begin{equation}
T_{\text{streaming}} = \max(T_{\text{audio}}, T_{\text{ASR}}) + T_{\text{LLM}}
\end{equation}
传统处理延迟模型：
\begin{equation}
T_{\text{traditional}} = T_{\text{audio}} + T_{\text{ASR}} + T_{\text{LLM}}
\end{equation}
"""
        }
        
        # 合并所有公式
        formula_content = "% 数学公式定义\n\n"
        for name, formula in formulas.items():
            formula_content += f"% {name}\n{formula}\n\n"
        
        # 保存公式文件
        formula_file = self.output_dir / "formulas.tex"
        with open(formula_file, 'w', encoding='utf-8') as f:
            f.write(formula_content)
        
        return str(formula_file)
    
    def generate_bibliography(self) -> str:
        """生成参考文献模板"""
        bibliography = """
@article{streaming_asr_2023,
  title={Streaming Speech Recognition: Recent Advances and Applications},
  author={Author, A. and Author, B.},
  journal={IEEE Transactions on Audio, Speech, and Language Processing},
  year={2023},
  volume={31},
  pages={1234--1245}
}

@inproceedings{kv_cache_2024,
  title={KV-Cache Optimization for Large Language Models},
  author={Author, C. and Author, D.},
  booktitle={Proceedings of AAAI},
  year={2024},
  pages={5678--5689}
}

@article{speech_dialogue_2023,
  title={End-to-End Speech Dialogue Systems: A Survey},
  author={Author, E. and Author, F.},
  journal={ACM Computing Surveys},
  year={2023},
  volume={56},
  number={3},
  pages={1--35}
}

@inproceedings{latency_optimization_2024,
  title={Real-time Latency Optimization in Conversational AI},
  author={Author, G. and Author, H.},
  booktitle={Proceedings of ICASSP},
  year={2024},
  pages={9876--9880}
}
"""
        
        # 保存参考文献
        bib_file = self.output_dir / "references.bib"
        with open(bib_file, 'w', encoding='utf-8') as f:
            f.write(bibliography)
        
        return str(bib_file)