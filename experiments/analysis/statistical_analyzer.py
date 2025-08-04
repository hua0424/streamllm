#!/usr/bin/env python3
"""
统计分析器 - 提供实验结果的统计分析功能
包括描述性统计、假设检验、效应量计算等
"""

import json
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
from pathlib import Path
from dataclasses import dataclass
from scipy import stats
import warnings

# 抑制统计检验的警告
warnings.filterwarnings('ignore', category=UserWarning)


@dataclass
class StatisticalResult:
    """统计分析结果"""
    test_name: str
    statistic: float
    p_value: float
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    interpretation: str = ""
    significant: bool = False


class StatisticalAnalyzer:
    """统计分析器"""
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha  # 显著性水平
        
    def descriptive_statistics(self, data: List[float]) -> Dict[str, float]:
        """计算描述性统计"""
        if not data:
            return {"error": "数据为空"}
        
        data_array = np.array(data)
        
        return {
            "count": len(data),
            "mean": float(np.mean(data_array)),
            "median": float(np.median(data_array)),
            "std": float(np.std(data_array, ddof=1)),
            "variance": float(np.var(data_array, ddof=1)),
            "min": float(np.min(data_array)),
            "max": float(np.max(data_array)),
            "range": float(np.max(data_array) - np.min(data_array)),
            "q25": float(np.percentile(data_array, 25)),
            "q75": float(np.percentile(data_array, 75)),
            "iqr": float(np.percentile(data_array, 75) - np.percentile(data_array, 25)),
            "skewness": float(stats.skew(data_array)),
            "kurtosis": float(stats.kurtosis(data_array))
        }
    
    def confidence_interval(self, data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """计算置信区间"""
        if len(data) < 2:
            return (0.0, 0.0)
        
        data_array = np.array(data)
        mean = np.mean(data_array)
        sem = stats.sem(data_array)  # 标准误
        
        # 使用t分布
        df = len(data) - 1
        t_value = stats.t.ppf((1 + confidence) / 2, df)
        margin_error = t_value * sem
        
        return (float(mean - margin_error), float(mean + margin_error))
    
    def paired_t_test(self, before: List[float], after: List[float]) -> StatisticalResult:
        """配对t检验"""
        if len(before) != len(after) or len(before) < 2:
            return StatisticalResult(
                test_name="paired_t_test",
                statistic=0.0,
                p_value=1.0,
                interpretation="数据不足或长度不匹配"
            )
        
        # 执行配对t检验
        statistic, p_value = stats.ttest_rel(before, after)
        
        # 计算效应量 (Cohen's d for paired samples)
        differences = np.array(after) - np.array(before)
        effect_size = np.mean(differences) / np.std(differences, ddof=1)
        
        # 解释结果
        significant = p_value < self.alpha
        if significant:
            if effect_size > 0:
                interpretation = f"后测显著高于前测 (p={p_value:.3f}, d={effect_size:.3f})"
            else:
                interpretation = f"后测显著低于前测 (p={p_value:.3f}, d={effect_size:.3f})"
        else:
            interpretation = f"差异不显著 (p={p_value:.3f})"
        
        return StatisticalResult(
            test_name="paired_t_test",
            statistic=float(statistic),
            p_value=float(p_value),
            effect_size=float(effect_size),
            interpretation=interpretation,
            significant=significant
        )
    
    def independent_t_test(self, group1: List[float], group2: List[float], 
                          equal_var: bool = True) -> StatisticalResult:
        """独立样本t检验"""
        if len(group1) < 2 or len(group2) < 2:
            return StatisticalResult(
                test_name="independent_t_test",
                statistic=0.0,
                p_value=1.0,
                interpretation="样本量不足"
            )
        
        # 执行独立样本t检验
        statistic, p_value = stats.ttest_ind(group1, group2, equal_var=equal_var)
        
        # 计算效应量 (Cohen's d)
        pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + 
                             (len(group2) - 1) * np.var(group2, ddof=1)) / 
                            (len(group1) + len(group2) - 2))
        effect_size = (np.mean(group2) - np.mean(group1)) / pooled_std
        
        # 解释结果
        significant = p_value < self.alpha
        if significant:
            if effect_size > 0:
                interpretation = f"组2显著高于组1 (p={p_value:.3f}, d={effect_size:.3f})"
            else:
                interpretation = f"组2显著低于组1 (p={p_value:.3f}, d={effect_size:.3f})"
        else:
            interpretation = f"组间差异不显著 (p={p_value:.3f})"
        
        return StatisticalResult(
            test_name="independent_t_test",
            statistic=float(statistic),
            p_value=float(p_value),
            effect_size=float(effect_size),
            interpretation=interpretation,
            significant=significant
        )
    
    def one_way_anova(self, groups: List[List[float]], group_names: List[str] = None) -> StatisticalResult:
        """单因素方差分析"""
        if len(groups) < 2:
            return StatisticalResult(
                test_name="one_way_anova",
                statistic=0.0,
                p_value=1.0,
                interpretation="组数不足"
            )
        
        # 检查每组样本量
        valid_groups = [group for group in groups if len(group) >= 2]
        if len(valid_groups) < 2:
            return StatisticalResult(
                test_name="one_way_anova",
                statistic=0.0,
                p_value=1.0,
                interpretation="有效组数不足"
            )
        
        # 执行单因素ANOVA
        statistic, p_value = stats.f_oneway(*valid_groups)
        
        # 计算效应量 (eta squared)
        # 先计算总平方和和组间平方和
        all_data = np.concatenate(valid_groups)
        grand_mean = np.mean(all_data)
        
        # 组间平方和
        ss_between = sum(len(group) * (np.mean(group) - grand_mean) ** 2 for group in valid_groups)
        
        # 总平方和
        ss_total = sum((x - grand_mean) ** 2 for x in all_data)
        
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        # 解释结果
        significant = p_value < self.alpha
        if significant:
            interpretation = f"组间差异显著 (p={p_value:.3f}, η²={eta_squared:.3f})"
        else:
            interpretation = f"组间差异不显著 (p={p_value:.3f})"
        
        return StatisticalResult(
            test_name="one_way_anova",
            statistic=float(statistic),
            p_value=float(p_value),
            effect_size=float(eta_squared),
            interpretation=interpretation,
            significant=significant
        )
    
    def correlation_analysis(self, x: List[float], y: List[float], 
                           method: str = "pearson") -> StatisticalResult:
        """相关性分析"""
        if len(x) != len(y) or len(x) < 3:
            return StatisticalResult(
                test_name=f"{method}_correlation",
                statistic=0.0,
                p_value=1.0,
                interpretation="数据不足或长度不匹配"
            )
        
        # 选择相关系数方法
        if method == "pearson":
            statistic, p_value = stats.pearsonr(x, y)
        elif method == "spearman":
            statistic, p_value = stats.spearmanr(x, y)
        else:
            raise ValueError(f"不支持的相关分析方法: {method}")
        
        # 解释相关强度
        abs_r = abs(statistic)
        if abs_r >= 0.8:
            strength = "强"
        elif abs_r >= 0.5:
            strength = "中等"
        elif abs_r >= 0.3:
            strength = "弱"
        else:
            strength = "很弱"
        
        direction = "正" if statistic > 0 else "负"
        significant = p_value < self.alpha
        
        if significant:
            interpretation = f"{direction}相关，{strength}相关性 (r={statistic:.3f}, p={p_value:.3f})"
        else:
            interpretation = f"相关性不显著 (r={statistic:.3f}, p={p_value:.3f})"
        
        return StatisticalResult(
            test_name=f"{method}_correlation",
            statistic=float(statistic),
            p_value=float(p_value),
            interpretation=interpretation,
            significant=significant
        )
    
    def effect_size_cohen_d(self, group1: List[float], group2: List[float]) -> float:
        """计算Cohen's d效应量"""
        if len(group1) < 2 or len(group2) < 2:
            return 0.0
        
        # 计算合并标准差
        pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + 
                             (len(group2) - 1) * np.var(group2, ddof=1)) / 
                            (len(group1) + len(group2) - 2))
        
        # 计算Cohen's d
        cohen_d = (np.mean(group2) - np.mean(group1)) / pooled_std
        return float(cohen_d)
    
    def interpret_effect_size(self, effect_size: float, measure: str = "cohen_d") -> str:
        """解释效应量大小"""
        abs_effect = abs(effect_size)
        
        if measure == "cohen_d":
            if abs_effect >= 0.8:
                return "大效应"
            elif abs_effect >= 0.5:
                return "中等效应"
            elif abs_effect >= 0.2:
                return "小效应"
            else:
                return "微小效应"
        
        elif measure == "eta_squared":
            if abs_effect >= 0.14:
                return "大效应"
            elif abs_effect >= 0.06:
                return "中等效应"
            elif abs_effect >= 0.01:
                return "小效应"
            else:
                return "微小效应"
        
        else:
            return "未知效应量类型"
    
    def power_analysis(self, effect_size: float, sample_size: int, alpha: float = None) -> float:
        """功效分析（简化版）"""
        if alpha is None:
            alpha = self.alpha
        
        # 使用简化的功效计算公式
        # 这是一个近似计算，实际应用中建议使用专门的功效分析包
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = effect_size * np.sqrt(sample_size / 2) - z_alpha
        power = stats.norm.cdf(z_beta)
        
        return max(0.0, min(1.0, float(power)))
    
    def normality_test(self, data: List[float]) -> StatisticalResult:
        """正态性检验（Shapiro-Wilk检验）"""
        if len(data) < 3:
            return StatisticalResult(
                test_name="shapiro_wilk_test",
                statistic=0.0,
                p_value=1.0,
                interpretation="样本量不足"
            )
        
        # Shapiro-Wilk检验适用于样本量<5000的情况
        if len(data) > 5000:
            data = np.random.choice(data, 5000, replace=False)
        
        statistic, p_value = stats.shapiro(data)
        
        significant = p_value < self.alpha
        if significant:
            interpretation = f"拒绝正态性假设，数据不符合正态分布 (p={p_value:.3f})"
        else:
            interpretation = f"接受正态性假设，数据符合正态分布 (p={p_value:.3f})"
        
        return StatisticalResult(
            test_name="shapiro_wilk_test",
            statistic=float(statistic),
            p_value=float(p_value),
            interpretation=interpretation,
            significant=significant
        )


class ExperimentStatisticalAnalyzer:
    """实验结果统计分析器"""
    
    def __init__(self, experiment_results_dir: str):
        self.results_dir = Path(experiment_results_dir)
        self.analyzer = StatisticalAnalyzer()
        
    def load_experiment_result(self, experiment_name: str) -> Dict[str, Any]:
        """加载实验结果"""
        result_file = self.results_dir / f"{experiment_name}" / "experiment_results.json"
        
        if not result_file.exists():
            raise FileNotFoundError(f"实验结果文件不存在: {result_file}")
        
        with open(result_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def analyze_optimization_effectiveness(self, experiment_name: str) -> Dict[str, Any]:
        """分析优化效果的统计显著性"""
        result = self.load_experiment_result(experiment_name)
        sample_results = result["sample_results"]
        
        # 提取基线和优化延迟数据
        baseline_latencies = []
        optimized_latencies = []
        optimization_ratios = []
        
        for sample in sample_results:
            if sample.get("error_message") is None:
                baseline_latencies.append(sample["baseline_latency"])
                optimized_latencies.append(sample["optimized_latency"])
                optimization_ratios.append(sample["optimization_ratio"])
        
        if not baseline_latencies:
            return {"error": "没有有效的实验数据"}
        
        analysis = {
            "experiment_name": experiment_name,
            "sample_size": len(baseline_latencies)
        }
        
        # 描述性统计
        analysis["baseline_stats"] = self.analyzer.descriptive_statistics(baseline_latencies)
        analysis["optimized_stats"] = self.analyzer.descriptive_statistics(optimized_latencies)
        analysis["optimization_ratio_stats"] = self.analyzer.descriptive_statistics(optimization_ratios)
        
        # 配对t检验
        paired_t_result = self.analyzer.paired_t_test(baseline_latencies, optimized_latencies)
        analysis["paired_t_test"] = paired_t_result
        
        # 置信区间
        analysis["optimization_ratio_ci"] = self.analyzer.confidence_interval(optimization_ratios)
        
        # 效应量
        cohen_d = self.analyzer.effect_size_cohen_d(baseline_latencies, optimized_latencies)
        analysis["effect_size"] = {
            "cohen_d": cohen_d,
            "interpretation": self.analyzer.interpret_effect_size(cohen_d, "cohen_d")
        }
        
        # 功效分析
        power = self.analyzer.power_analysis(abs(cohen_d), len(baseline_latencies))
        analysis["statistical_power"] = power
        
        return analysis
    
    def compare_experiments(self, experiment_names: List[str], metric: str = "optimization_ratio") -> Dict[str, Any]:
        """比较多个实验的结果"""
        groups = []
        group_names = []
        
        for exp_name in experiment_names:
            try:
                result = self.load_experiment_result(exp_name)
                sample_results = result["sample_results"]
                
                metric_values = []
                for sample in sample_results:
                    if sample.get("error_message") is None and metric in sample:
                        metric_values.append(sample[metric])
                
                if metric_values:
                    groups.append(metric_values)
                    group_names.append(exp_name)
            
            except Exception as e:
                print(f"警告：无法加载实验 {exp_name}: {e}")
        
        if len(groups) < 2:
            return {"error": "有效实验数量不足"}
        
        analysis = {
            "compared_experiments": group_names,
            "metric": metric,
            "group_sizes": [len(group) for group in groups]
        }
        
        # 描述性统计
        for i, (group, name) in enumerate(zip(groups, group_names)):
            analysis[f"group_{i+1}_stats"] = self.analyzer.descriptive_statistics(group)
            analysis[f"group_{i+1}_name"] = name
        
        # 单因素方差分析
        anova_result = self.analyzer.one_way_anova(groups, group_names)
        analysis["anova"] = anova_result
        
        # 两两比较（如果只有2组）
        if len(groups) == 2:
            t_test_result = self.analyzer.independent_t_test(groups[0], groups[1])
            analysis["t_test"] = t_test_result
        
        return analysis
    
    def analyze_correlation(self, experiment_name: str, x_metric: str, y_metric: str) -> Dict[str, Any]:
        """分析两个指标之间的相关性"""
        result = self.load_experiment_result(experiment_name)
        sample_results = result["sample_results"]
        
        x_values = []
        y_values = []
        
        for sample in sample_results:
            if (sample.get("error_message") is None and 
                x_metric in sample and y_metric in sample):
                x_values.append(sample[x_metric])
                y_values.append(sample[y_metric])
        
        if len(x_values) < 3:
            return {"error": "数据点不足"}
        
        analysis = {
            "experiment_name": experiment_name,
            "x_metric": x_metric,
            "y_metric": y_metric,
            "sample_size": len(x_values)
        }
        
        # Pearson相关
        pearson_result = self.analyzer.correlation_analysis(x_values, y_values, "pearson")
        analysis["pearson_correlation"] = pearson_result
        
        # Spearman相关
        spearman_result = self.analyzer.correlation_analysis(x_values, y_values, "spearman")
        analysis["spearman_correlation"] = spearman_result
        
        return analysis
    
    def generate_statistical_report(self, experiment_names: List[str], output_file: str = None) -> str:
        """生成统计分析报告"""
        report_lines = []
        report_lines.append("# 级联式语音对话系统延迟优化实验统计分析报告\n")
        report_lines.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 各实验的优化效果分析
        for exp_name in experiment_names:
            try:
                analysis = self.analyze_optimization_effectiveness(exp_name)
                
                report_lines.append(f"\n## {exp_name} 优化效果分析\n")
                report_lines.append(f"样本量: {analysis['sample_size']}\n")
                
                # 描述性统计
                ratio_stats = analysis["optimization_ratio_stats"]
                report_lines.append(f"平均优化比例: {ratio_stats['mean']:.1f}% ± {ratio_stats['std']:.1f}%\n")
                report_lines.append(f"95%置信区间: [{analysis['optimization_ratio_ci'][0]:.1f}%, {analysis['optimization_ratio_ci'][1]:.1f}%]\n")
                
                # 统计检验结果
                t_test = analysis["paired_t_test"]
                report_lines.append(f"配对t检验: {t_test.interpretation}\n")
                
                # 效应量
                effect = analysis["effect_size"]
                report_lines.append(f"效应量: Cohen's d = {effect['cohen_d']:.3f} ({effect['interpretation']})\n")
                
                # 统计功效
                report_lines.append(f"统计功效: {analysis['statistical_power']:.3f}\n")
                
            except Exception as e:
                report_lines.append(f"\n## {exp_name} 分析失败: {e}\n")
        
        # 实验间比较
        if len(experiment_names) > 1:
            try:
                comparison = self.compare_experiments(experiment_names)
                
                report_lines.append(f"\n## 实验间比较分析\n")
                anova = comparison["anova"]
                report_lines.append(f"单因素方差分析: {anova.interpretation}\n")
                
                if len(experiment_names) == 2:
                    t_test = comparison["t_test"]
                    report_lines.append(f"独立样本t检验: {t_test.interpretation}\n")
                
            except Exception as e:
                report_lines.append(f"\n## 实验间比较失败: {e}\n")
        
        report_content = "\n".join(report_lines)
        
        # 保存报告
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
        
        return report_content