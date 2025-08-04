#!/usr/bin/env python3
"""
实验结果统计分析模块
提供各种统计检验和效应量计算功能
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_rel, ttest_ind, f_oneway, kruskal
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging

class StatisticalAnalyzer:
    def __init__(self, significance_level: float = 0.05):
        """
        初始化统计分析器
        
        Args:
            significance_level: 显著性水平，默认0.05
        """
        self.alpha = significance_level
        self.logger = logging.getLogger(__name__)
    
    def paired_t_test(self, group1: List[float], group2: List[float]) -> Dict[str, Any]:
        """
        配对t检验
        用于比较同一组样本在两种条件下的表现
        
        Args:
            group1: 第一组数据（如基线方法的延迟）
            group2: 第二组数据（如优化方法的延迟）
            
        Returns:
            包含统计结果的字典
        """
        if len(group1) != len(group2):
            raise ValueError("配对t检验要求两组数据长度相同")
        
        # 执行配对t检验
        t_statistic, p_value = ttest_rel(group1, group2)
        
        # 计算效应量 (Cohen's d for paired samples)
        differences = np.array(group1) - np.array(group2)
        effect_size = np.mean(differences) / np.std(differences, ddof=1)
        
        # 计算置信区间
        n = len(group1)
        df = n - 1
        t_critical = stats.t.ppf(1 - self.alpha/2, df)
        se_diff = np.std(differences, ddof=1) / np.sqrt(n)
        ci_lower = np.mean(differences) - t_critical * se_diff
        ci_upper = np.mean(differences) + t_critical * se_diff
        
        return {
            "test_type": "paired_t_test",
            "t_statistic": float(t_statistic),
            "p_value": float(p_value),
            "degrees_of_freedom": df,
            "effect_size_cohen_d": float(effect_size),
            "mean_difference": float(np.mean(differences)),
            "std_difference": float(np.std(differences, ddof=1)),
            "confidence_interval_95": [float(ci_lower), float(ci_upper)],
            "is_significant": p_value < self.alpha,
            "significance_level": self.alpha,
            "interpretation": self._interpret_effect_size(abs(effect_size))
        }
    
    def independent_t_test(self, group1: List[float], group2: List[float]) -> Dict[str, Any]:
        """
        独立样本t检验
        用于比较两个独立组的均值差异
        """
        # 执行独立样本t检验
        t_statistic, p_value = ttest_ind(group1, group2)
        
        # 计算效应量 (Cohen's d for independent samples)
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        n1, n2 = len(group1), len(group2)
        
        # 合并标准差
        pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
        effect_size = (mean1 - mean2) / pooled_std
        
        return {
            "test_type": "independent_t_test",
            "t_statistic": float(t_statistic),
            "p_value": float(p_value),
            "degrees_of_freedom": n1 + n2 - 2,
            "effect_size_cohen_d": float(effect_size),
            "group1_mean": float(mean1),
            "group2_mean": float(mean2),
            "group1_std": float(std1),
            "group2_std": float(std2),
            "mean_difference": float(mean1 - mean2),
            "is_significant": p_value < self.alpha,
            "significance_level": self.alpha,
            "interpretation": self._interpret_effect_size(abs(effect_size))
        }
    
    def one_way_anova(self, *groups: List[float]) -> Dict[str, Any]:
        """
        单因素方差分析
        用于比较三个或更多组的均值差异
        """
        if len(groups) < 3:
            raise ValueError("方差分析至少需要3组数据")
        
        # 执行单因素ANOVA
        f_statistic, p_value = f_oneway(*groups)
        
        # 计算效应量 (eta squared)
        group_means = [np.mean(group) for group in groups]
        grand_mean = np.mean(np.concatenate(groups))
        
        ss_between = sum(len(group) * (np.mean(group) - grand_mean)**2 for group in groups)
        ss_within = sum(sum((x - np.mean(group))**2 for x in group) for group in groups)
        ss_total = ss_between + ss_within
        
        eta_squared = ss_between / ss_total
        
        # 计算组间描述性统计
        group_stats = []
        for i, group in enumerate(groups):
            group_stats.append({
                "group": i + 1,
                "n": len(group),
                "mean": float(np.mean(group)),
                "std": float(np.std(group, ddof=1)),
                "min": float(np.min(group)),
                "max": float(np.max(group))
            })
        
        return {
            "test_type": "one_way_anova",
            "f_statistic": float(f_statistic),
            "p_value": float(p_value),
            "degrees_of_freedom_between": len(groups) - 1,
            "degrees_of_freedom_within": sum(len(group) for group in groups) - len(groups),
            "effect_size_eta_squared": float(eta_squared),
            "group_statistics": group_stats,
            "is_significant": p_value < self.alpha,
            "significance_level": self.alpha,
            "interpretation": self._interpret_eta_squared(eta_squared)
        }
    
    def kruskal_wallis_test(self, *groups: List[float]) -> Dict[str, Any]:
        """
        Kruskal-Wallis检验（非参数版本的ANOVA）
        当数据不满足正态分布假设时使用
        """
        if len(groups) < 3:
            raise ValueError("Kruskal-Wallis检验至少需要3组数据")
        
        # 执行Kruskal-Wallis检验
        h_statistic, p_value = kruskal(*groups)
        
        # 计算描述性统计
        group_stats = []
        for i, group in enumerate(groups):
            group_stats.append({
                "group": i + 1,
                "n": len(group),
                "median": float(np.median(group)),
                "q1": float(np.percentile(group, 25)),
                "q3": float(np.percentile(group, 75)),
                "min": float(np.min(group)),
                "max": float(np.max(group))
            })
        
        return {
            "test_type": "kruskal_wallis",
            "h_statistic": float(h_statistic),
            "p_value": float(p_value),
            "degrees_of_freedom": len(groups) - 1,
            "group_statistics": group_stats,
            "is_significant": p_value < self.alpha,
            "significance_level": self.alpha
        }
    
    def correlation_analysis(self, x: List[float], y: List[float]) -> Dict[str, Any]:
        """
        相关性分析
        计算Pearson和Spearman相关系数
        """
        if len(x) != len(y):
            raise ValueError("相关性分析要求两组数据长度相同")
        
        # Pearson相关系数
        pearson_r, pearson_p = stats.pearsonr(x, y)
        
        # Spearman相关系数
        spearman_r, spearman_p = stats.spearmanr(x, y)
        
        return {
            "test_type": "correlation_analysis",
            "sample_size": len(x),
            "pearson_correlation": {
                "r": float(pearson_r),
                "p_value": float(pearson_p),
                "is_significant": pearson_p < self.alpha,
                "interpretation": self._interpret_correlation(abs(pearson_r))
            },
            "spearman_correlation": {
                "rho": float(spearman_r),
                "p_value": float(spearman_p),
                "is_significant": spearman_p < self.alpha,
                "interpretation": self._interpret_correlation(abs(spearman_r))
            },
            "significance_level": self.alpha
        }
    
    def normality_test(self, data: List[float]) -> Dict[str, Any]:
        """
        正态性检验
        使用Shapiro-Wilk检验
        """
        if len(data) < 3:
            raise ValueError("正态性检验至少需要3个数据点")
        
        # Shapiro-Wilk检验
        w_statistic, p_value = stats.shapiro(data)
        
        return {
            "test_type": "shapiro_wilk_normality",
            "w_statistic": float(w_statistic),
            "p_value": float(p_value),
            "is_normal": p_value > self.alpha,
            "significance_level": self.alpha,
            "sample_size": len(data),
            "recommendation": "使用参数检验" if p_value > self.alpha else "建议使用非参数检验"
        }
    
    def descriptive_statistics(self, data: List[float]) -> Dict[str, Any]:
        """
        描述性统计
        """
        data_array = np.array(data)
        
        return {
            "n": len(data),
            "mean": float(np.mean(data_array)),
            "median": float(np.median(data_array)),
            "std": float(np.std(data_array, ddof=1)),
            "variance": float(np.var(data_array, ddof=1)),
            "min": float(np.min(data_array)),
            "max": float(np.max(data_array)),
            "q1": float(np.percentile(data_array, 25)),
            "q3": float(np.percentile(data_array, 75)),
            "iqr": float(np.percentile(data_array, 75) - np.percentile(data_array, 25)),
            "skewness": float(stats.skew(data_array)),
            "kurtosis": float(stats.kurtosis(data_array)),
            "cv": float(np.std(data_array, ddof=1) / np.mean(data_array)) if np.mean(data_array) != 0 else None
        }
    
    def _interpret_effect_size(self, effect_size: float) -> str:
        """解释Cohen's d效应量"""
        if effect_size < 0.2:
            return "很小效应"
        elif effect_size < 0.5:
            return "小效应"
        elif effect_size < 0.8:
            return "中等效应"
        else:
            return "大效应"
    
    def _interpret_eta_squared(self, eta_squared: float) -> str:
        """解释eta squared效应量"""
        if eta_squared < 0.01:
            return "很小效应"
        elif eta_squared < 0.06:
            return "小效应"
        elif eta_squared < 0.14:
            return "中等效应"
        else:
            return "大效应"
    
    def _interpret_correlation(self, correlation: float) -> str:
        """解释相关系数"""
        if correlation < 0.1:
            return "很弱相关"
        elif correlation < 0.3:
            return "弱相关"
        elif correlation < 0.5:
            return "中等相关"
        elif correlation < 0.7:
            return "强相关"
        else:
            return "很强相关"

class ExperimentAnalyzer:
    """实验结果分析器"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.analyzer = StatisticalAnalyzer()
        self.logger = logging.getLogger(__name__)
    
    def analyze_length_impact_experiment(self, results_file: str) -> Dict[str, Any]:
        """分析语音长度影响实验结果"""
        with open(self.results_dir / results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # 提取数据
        lengths = []
        baseline_latencies = []
        streaming_latencies = []
        optimization_ratios = []
        
        for result in results:
            lengths.append(result['audio_length'])
            baseline_latencies.append(result['baseline_latency'])
            streaming_latencies.append(result['streaming_latency'])
            optimization_ratios.append(result['optimization_ratio'])
        
        analysis = {
            "experiment": "length_impact",
            "sample_size": len(results),
            "descriptive_stats": {
                "baseline_latency": self.analyzer.descriptive_statistics(baseline_latencies),
                "streaming_latency": self.analyzer.descriptive_statistics(streaming_latencies),
                "optimization_ratio": self.analyzer.descriptive_statistics(optimization_ratios)
            },
            "paired_t_test": self.analyzer.paired_t_test(baseline_latencies, streaming_latencies),
            "correlation_analysis": {
                "length_vs_optimization": self.analyzer.correlation_analysis(lengths, optimization_ratios),
                "length_vs_baseline": self.analyzer.correlation_analysis(lengths, baseline_latencies)
            }
        }
        
        return analysis
    
    def analyze_ablation_experiment(self, results_file: str) -> Dict[str, Any]:
        """分析消融实验结果"""
        with open(self.results_dir / results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # 提取各配置的延迟数据
        configurations = list(results.keys())
        latencies = [results[config]['latency_measurements'] for config in configurations]
        
        analysis = {
            "experiment": "ablation_study",
            "configurations": configurations,
            "descriptive_stats": {
                config: self.analyzer.descriptive_statistics(latencies[i])
                for i, config in enumerate(configurations)
            },
            "one_way_anova": self.analyzer.one_way_anova(*latencies),
            "pairwise_comparisons": {}
        }
        
        # 成对比较
        for i in range(len(configurations)):
            for j in range(i+1, len(configurations)):
                comparison_name = f"{configurations[i]}_vs_{configurations[j]}"
                analysis["pairwise_comparisons"][comparison_name] = \
                    self.analyzer.independent_t_test(latencies[i], latencies[j])
        
        return analysis
    
    def generate_statistical_report(self, experiment_results: Dict[str, Any]) -> str:
        """生成统计分析报告"""
        report_lines = []
        
        report_lines.append("# 实验统计分析报告")
        report_lines.append("")
        
        for exp_name, analysis in experiment_results.items():
            report_lines.append(f"## {exp_name}")
            report_lines.append("")
            
            # 描述性统计
            if "descriptive_stats" in analysis:
                report_lines.append("### 描述性统计")
                for metric, stats in analysis["descriptive_stats"].items():
                    report_lines.append(f"**{metric}:**")
                    report_lines.append(f"- 平均值: {stats['mean']:.4f}")
                    report_lines.append(f"- 标准差: {stats['std']:.4f}")
                    report_lines.append(f"- 中位数: {stats['median']:.4f}")
                    report_lines.append(f"- 范围: [{stats['min']:.4f}, {stats['max']:.4f}]")
                    report_lines.append("")
            
            # 假设检验结果
            if "paired_t_test" in analysis:
                test = analysis["paired_t_test"]
                report_lines.append("### 配对t检验结果")
                report_lines.append(f"- t统计量: {test['t_statistic']:.4f}")
                report_lines.append(f"- p值: {test['p_value']:.4f}")
                report_lines.append(f"- 效应量(Cohen's d): {test['effect_size_cohen_d']:.4f} ({test['interpretation']})")
                report_lines.append(f"- 显著性: {'是' if test['is_significant'] else '否'}")
                report_lines.append("")
            
            if "one_way_anova" in analysis:
                anova = analysis["one_way_anova"]
                report_lines.append("### 单因素方差分析结果")
                report_lines.append(f"- F统计量: {anova['f_statistic']:.4f}")
                report_lines.append(f"- p值: {anova['p_value']:.4f}")
                report_lines.append(f"- 效应量(η²): {anova['effect_size_eta_squared']:.4f} ({anova['interpretation']})")
                report_lines.append(f"- 显著性: {'是' if anova['is_significant'] else '否'}")
                report_lines.append("")
            
            # 相关性分析
            if "correlation_analysis" in analysis:
                report_lines.append("### 相关性分析")
                for corr_name, corr_data in analysis["correlation_analysis"].items():
                    report_lines.append(f"**{corr_name}:**")
                    pearson = corr_data["pearson_correlation"]
                    report_lines.append(f"- Pearson r: {pearson['r']:.4f} ({pearson['interpretation']})")
                    report_lines.append(f"- 显著性: {'是' if pearson['is_significant'] else '否'} (p={pearson['p_value']:.4f})")
                    report_lines.append("")
            
            report_lines.append("---")
            report_lines.append("")
        
        return "\n".join(report_lines)

def main():
    """主函数：运行统计分析"""
    analyzer = ExperimentAnalyzer("experiments/results")
    
    # 分析各个实验
    analyses = {}
    
    # 实验1：语音长度影响
    try:
        analyses["length_impact"] = analyzer.analyze_length_impact_experiment(
            "exp1_length_impact/results.json"
        )
    except FileNotFoundError:
        logging.warning("未找到实验1结果文件")
    
    # 实验3：消融实验
    try:
        analyses["ablation_study"] = analyzer.analyze_ablation_experiment(
            "exp3_ablation_study/results.json"
        )
    except FileNotFoundError:
        logging.warning("未找到实验3结果文件")
    
    # 生成统计报告
    if analyses:
        report = analyzer.generate_statistical_report(analyses)
        
        # 保存报告
        output_dir = Path("experiments/results/statistical_analysis")
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / "statistical_report.md", 'w', encoding='utf-8') as f:
            f.write(report)
        
        # 保存详细分析结果
        with open(output_dir / "detailed_analysis.json", 'w', encoding='utf-8') as f:
            json.dump(analyses, f, indent=2, ensure_ascii=False)
        
        print(f"统计分析完成，结果保存到 {output_dir}")
    else:
        print("未找到可分析的实验结果")

if __name__ == "__main__":
    main()