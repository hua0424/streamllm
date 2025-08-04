#!/usr/bin/env python3
"""
图表生成器 - 为实验结果生成可视化图表
包括延迟对比图、消融实验图、相关性图等
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import warnings

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

# 忽略matplotlib警告
warnings.filterwarnings('ignore', category=UserWarning)


class ChartGenerator:
    """图表生成器"""
    
    def __init__(self, output_dir: str = "experiments/results/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 图表样式配置
        self.figure_size = (10, 6)
        self.dpi = 300
        self.color_palette = sns.color_palette("husl", 8)
        
    def load_experiment_data(self, experiment_results_dir: str, experiment_name: str) -> Dict[str, Any]:
        """加载实验数据"""
        result_file = Path(experiment_results_dir) / experiment_name / "experiment_results.json"
        
        if not result_file.exists():
            raise FileNotFoundError(f"实验结果文件不存在: {result_file}")
        
        with open(result_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def create_latency_comparison_chart(self, experiment_data: Dict[str, Any], 
                                       output_name: str = "latency_comparison") -> str:
        """创建延迟对比图"""
        sample_results = experiment_data["sample_results"]
        
        # 提取数据
        baseline_latencies = []
        optimized_latencies = []
        sample_ids = []
        
        for sample in sample_results:
            if sample.get("error_message") is None:
                baseline_latencies.append(sample["baseline_latency"])
                optimized_latencies.append(sample["optimized_latency"])
                sample_ids.append(sample["sample_id"])
        
        if not baseline_latencies:
            raise ValueError("没有有效的延迟数据")
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 子图1：对比柱状图
        x = np.arange(len(sample_ids))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, baseline_latencies, width, 
                        label='基线延迟', color=self.color_palette[0], alpha=0.8)
        bars2 = ax1.bar(x + width/2, optimized_latencies, width,
                        label='优化延迟', color=self.color_palette[1], alpha=0.8)
        
        ax1.set_xlabel('样本')
        ax1.set_ylabel('延迟 (ms)')
        ax1.set_title('基线 vs 优化延迟对比')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'样本{i+1}' for i in range(len(sample_ids))], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}', ha='center', va='bottom', fontsize=8)
        
        # 子图2：优化比例分布
        optimization_ratios = [(b - o) / b * 100 for b, o in zip(baseline_latencies, optimized_latencies)]
        
        ax2.hist(optimization_ratios, bins=10, color=self.color_palette[2], alpha=0.7, edgecolor='black')
        ax2.set_xlabel('优化比例 (%)')
        ax2.set_ylabel('频数')
        ax2.set_title('优化比例分布')
        ax2.axvline(np.mean(optimization_ratios), color='red', linestyle='--', 
                   label=f'平均值: {np.mean(optimization_ratios):.1f}%')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        output_path = self.output_dir / f"{output_name}.png"
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def create_ablation_chart(self, ablation_data: Dict[str, Any], 
                             output_name: str = "ablation_analysis") -> str:
        """创建消融实验图表"""
        if "ablation_analysis.json" in str(ablation_data):
            # 如果传入的是文件路径，加载数据
            with open(ablation_data, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = ablation_data
        
        config_comparison = data.get("config_comparison", {})
        component_contributions = data.get("component_contributions", {})
        
        if not config_comparison:
            raise ValueError("没有有效的消融实验数据")
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 子图1：各配置的延迟对比
        configurations = list(config_comparison.keys())
        mean_latencies = [config_comparison[config]["mean_latency"] for config in configurations]
        config_names = [config_comparison[config]["configuration"]["name"] for config in configurations]
        
        bars = ax1.bar(range(len(configurations)), mean_latencies, 
                      color=self.color_palette[:len(configurations)], alpha=0.8)
        
        ax1.set_xlabel('配置')
        ax1.set_ylabel('平均延迟 (ms)')
        ax1.set_title('不同配置的延迟表现')
        ax1.set_xticks(range(len(configurations)))
        ax1.set_xticklabels(config_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, latency in zip(bars, mean_latencies):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{latency:.0f}ms', ha='center', va='bottom', fontsize=10)
        
        # 子图2：组件贡献度
        if component_contributions:
            contributions = [
                component_contributions.get("asr_streaming_contribution", 0),
                component_contributions.get("llm_kv_cache_contribution", 0)
            ]
            component_labels = ['流式ASR', 'KV缓存LLM']
            
            colors = [self.color_palette[0], self.color_palette[1]]
            wedges, texts, autotexts = ax2.pie(contributions, labels=component_labels, 
                                              autopct='%1.1f%%', colors=colors,
                                              startangle=90, explode=(0.05, 0.05))
            
            ax2.set_title('优化组件贡献度')
            
            # 美化饼图
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(12)
                autotext.set_weight('bold')
        else:
            ax2.text(0.5, 0.5, '没有组件贡献度数据', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=14)
            ax2.set_title('组件贡献度分析')
        
        plt.tight_layout()
        
        # 保存图表
        output_path = self.output_dir / f"{output_name}.png"
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def create_length_impact_chart(self, length_analysis_data: Dict[str, Any],
                                  output_name: str = "length_impact") -> str:
        """创建语音长度影响图表"""
        length_statistics = length_analysis_data.get("length_statistics", {})
        
        if not length_statistics:
            raise ValueError("没有有效的长度分析数据")
        
        # 提取数据
        lengths = []
        optimizations = []
        accuracies = []
        sample_counts = []
        
        for length_key, stats in length_statistics.items():
            length = int(length_key.replace('s', ''))
            lengths.append(length)
            optimizations.append(stats["mean_optimization"])
            sample_counts.append(stats["sample_count"])
            
            # 如果有准确率数据
            if "mean_accuracy" in stats:
                accuracies.append(stats["mean_accuracy"])
        
        # 按长度排序
        sorted_data = sorted(zip(lengths, optimizations, sample_counts))
        lengths, optimizations, sample_counts = zip(*sorted_data)
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 子图1：语音长度 vs 优化比例
        ax1.plot(lengths, optimizations, 'o-', color=self.color_palette[0], 
                linewidth=2, markersize=8, label='优化比例')
        
        # 添加趋势线
        z = np.polyfit(lengths, optimizations, 1)
        p = np.poly1d(z)
        ax1.plot(lengths, p(lengths), "--", color=self.color_palette[1], 
                alpha=0.8, label=f'趋势线 (斜率: {z[0]:.2f})')
        
        ax1.set_xlabel('语音长度 (秒)')
        ax1.set_ylabel('优化比例 (%)')
        ax1.set_title('语音长度对优化效果的影响')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for x, y in zip(lengths, optimizations):
            ax1.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9)
        
        # 子图2：样本数量分布
        bars = ax2.bar(lengths, sample_counts, color=self.color_palette[2], alpha=0.7)
        ax2.set_xlabel('语音长度 (秒)')
        ax2.set_ylabel('样本数量')
        ax2.set_title('各长度组样本分布')
        ax2.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, count in zip(bars, sample_counts):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{count}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        # 保存图表
        output_path = self.output_dir / f"{output_name}.png"
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def create_model_comparison_chart(self, comparison_data: Dict[str, Any],
                                     output_name: str = "model_comparison") -> str:
        """创建模型对比图表"""
        model_comparison = comparison_data.get("model_comparison", {})
        
        if not model_comparison:
            raise ValueError("没有有效的模型对比数据")
        
        # 分类模型
        cascaded_models = {}
        native_models = {}
        
        for model_name, data in model_comparison.items():
            model_config = data["model_config"]
            if model_config["type"] == "cascaded":
                cascaded_models[model_name] = data
            else:
                native_models[model_name] = data
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 子图1：延迟对比
        all_models = list(model_comparison.keys())
        latencies = [model_comparison[model]["performance_metrics"]["mean_latency"] 
                    for model in all_models]
        model_names = [model_comparison[model]["model_config"]["name"] 
                      for model in all_models]
        
        # 设置颜色（级联模型和原生模型使用不同颜色）
        colors = []
        for model in all_models:
            if model_comparison[model]["model_config"]["type"] == "cascaded":
                colors.append(self.color_palette[0])
            else:
                colors.append(self.color_palette[1])
        
        bars = ax1.bar(range(len(all_models)), latencies, color=colors, alpha=0.8)
        
        ax1.set_xlabel('模型')
        ax1.set_ylabel('平均延迟 (ms)')
        ax1.set_title('不同模型延迟对比')
        ax1.set_xticks(range(len(all_models)))
        ax1.set_xticklabels(model_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # 添加图例
        cascaded_patch = plt.Rectangle((0,0),1,1, color=self.color_palette[0], alpha=0.8)
        native_patch = plt.Rectangle((0,0),1,1, color=self.color_palette[1], alpha=0.8)
        ax1.legend([cascaded_patch, native_patch], ['级联式', '原生模型'], loc='upper right')
        
        # 添加数值标签
        for bar, latency in zip(bars, latencies):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{latency:.0f}ms', ha='center', va='bottom', fontsize=9)
        
        # 子图2：响应质量对比
        qualities = [model_comparison[model]["performance_metrics"]["mean_response_quality"] 
                    for model in all_models]
        
        bars2 = ax2.bar(range(len(all_models)), qualities, color=colors, alpha=0.8)
        
        ax2.set_xlabel('模型')
        ax2.set_ylabel('响应质量评分')
        ax2.set_title('不同模型响应质量对比')
        ax2.set_xticks(range(len(all_models)))
        ax2.set_xticklabels(model_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, quality in zip(bars2, qualities):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{quality:.1f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # 保存图表
        output_path = self.output_dir / f"{output_name}.png"
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def create_concurrent_performance_chart(self, concurrent_data: Dict[str, Any],
                                          output_name: str = "concurrent_performance") -> str:
        """创建并发性能图表"""
        concurrency_comparison = concurrent_data.get("concurrency_comparison", {})
        
        if not concurrency_comparison:
            raise ValueError("没有有效的并发性能数据")
        
        # 提取数据并排序
        concurrency_levels = []
        latencies = []
        throughputs = []
        success_rates = []
        
        for level, data in concurrency_comparison.items():
            concurrency_levels.append(level)
            metrics = data["performance_metrics"]
            latencies.append(metrics["latency"])
            throughputs.append(metrics["throughput"])
            success_rates.append(metrics["success_rate"])
        
        # 按并发级别排序
        sorted_data = sorted(zip(concurrency_levels, latencies, throughputs, success_rates))
        concurrency_levels, latencies, throughputs, success_rates = zip(*sorted_data)
        
        # 创建图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 子图1：延迟 vs 并发数
        ax1.plot(concurrency_levels, latencies, 'o-', color=self.color_palette[0], 
                linewidth=2, markersize=8)
        ax1.set_xlabel('并发用户数')
        ax1.set_ylabel('平均延迟 (ms)')
        ax1.set_title('并发数对延迟的影响')
        ax1.grid(True, alpha=0.3)
        
        # 子图2：吞吐量 vs 并发数
        ax2.plot(concurrency_levels, throughputs, 'o-', color=self.color_palette[1], 
                linewidth=2, markersize=8)
        ax2.set_xlabel('并发用户数')
        ax2.set_ylabel('吞吐量 (会话/秒)')
        ax2.set_title('并发数对吞吐量的影响')
        ax2.grid(True, alpha=0.3)
        
        # 子图3：成功率 vs 并发数
        ax3.plot(concurrency_levels, success_rates, 'o-', color=self.color_palette[2], 
                linewidth=2, markersize=8)
        ax3.set_xlabel('并发用户数')
        ax3.set_ylabel('成功率 (%)')
        ax3.set_title('并发数对成功率的影响')
        ax3.set_ylim(0, 105)
        ax3.grid(True, alpha=0.3)
        
        # 子图4：综合性能热力图
        performance_matrix = np.array([
            [(100-l/max(latencies)*100) for l in latencies],  # 延迟性能（越低越好，转换为越高越好）
            [t/max(throughputs)*100 for t in throughputs],    # 吞吐量性能
            success_rates                                      # 成功率
        ])
        
        im = ax4.imshow(performance_matrix, cmap='RdYlGn', aspect='auto')
        ax4.set_xticks(range(len(concurrency_levels)))
        ax4.set_xticklabels(concurrency_levels)
        ax4.set_yticks(range(3))
        ax4.set_yticklabels(['延迟性能', '吞吐量性能', '成功率'])
        ax4.set_xlabel('并发用户数')
        ax4.set_title('综合性能热力图')
        
        # 添加数值标签
        for i in range(3):
            for j in range(len(concurrency_levels)):
                text = ax4.text(j, i, f'{performance_matrix[i, j]:.1f}',
                               ha="center", va="center", color="black", fontsize=9)
        
        # 添加颜色条
        plt.colorbar(im, ax=ax4, shrink=0.8)
        
        plt.tight_layout()
        
        # 保存图表
        output_path = self.output_dir / f"{output_name}.png"
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def create_quality_robustness_chart(self, quality_data: Dict[str, Any],
                                       output_name: str = "quality_robustness") -> str:
        """创建音频质量鲁棒性图表"""
        quality_comparison = quality_data.get("quality_comparison", {})
        
        if not quality_comparison:
            raise ValueError("没有有效的音频质量数据")
        
        # 按质量顺序排序
        quality_order = ["clean", "high_quality", "medium_quality", "low_quality", "poor_quality", "challenging"]
        
        ordered_data = []
        for condition in quality_order:
            if condition in quality_comparison:
                data = quality_comparison[condition]
                ordered_data.append((
                    data["condition_config"]["name"],
                    data["performance_metrics"]["mean_optimization"],
                    data["performance_metrics"]["mean_accuracy"],
                    data["condition_config"]["snr_db"]
                ))
        
        if not ordered_data:
            raise ValueError("没有有序的质量数据")
        
        condition_names, optimizations, accuracies, snr_values = zip(*ordered_data)
        
        # 创建图表
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # 子图1：优化效果 vs 音频质量
        x_pos = range(len(condition_names))
        bars1 = ax1.bar(x_pos, optimizations, color=self.color_palette[0], alpha=0.8)
        
        ax1.set_xlabel('音频质量条件')
        ax1.set_ylabel('优化比例 (%)')
        ax1.set_title('音频质量对优化效果的影响')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([name.replace('音频', '') for name in condition_names], rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, optimization in zip(bars1, optimizations):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{optimization:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # 子图2：准确率 vs 音频质量
        bars2 = ax2.bar(x_pos, accuracies, color=self.color_palette[1], alpha=0.8)
        
        ax2.set_xlabel('音频质量条件')
        ax2.set_ylabel('ASR准确率 (%)')
        ax2.set_title('音频质量对ASR准确率的影响')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([name.replace('音频', '') for name in condition_names], rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, accuracy in zip(bars2, accuracies):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{accuracy:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # 子图3：SNR vs 性能散点图
        # 过滤无限SNR值
        finite_snr_data = [(snr, opt, acc) for snr, opt, acc in zip(snr_values, optimizations, accuracies) 
                          if snr != float('inf')]
        
        if finite_snr_data:
            snr_finite, opt_finite, acc_finite = zip(*finite_snr_data)
            
            scatter = ax3.scatter(snr_finite, opt_finite, c=acc_finite, 
                                s=100, cmap='viridis', alpha=0.7)
            
            ax3.set_xlabel('SNR (dB)')
            ax3.set_ylabel('优化比例 (%)')
            ax3.set_title('SNR vs 优化效果')
            ax3.grid(True, alpha=0.3)
            
            # 添加颜色条
            cbar = plt.colorbar(scatter, ax=ax3)
            cbar.set_label('ASR准确率 (%)')
            
            # 添加趋势线
            if len(snr_finite) > 2:
                z = np.polyfit(snr_finite, opt_finite, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(min(snr_finite), max(snr_finite), 100)
                ax3.plot(x_trend, p(x_trend), "--", color='red', alpha=0.8)
        
        plt.tight_layout()
        
        # 保存图表
        output_path = self.output_dir / f"{output_name}.png"
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def create_summary_dashboard(self, experiment_results_dir: str, 
                                experiment_names: List[str],
                                output_name: str = "experiment_dashboard") -> str:
        """创建实验总结仪表板"""
        # 收集所有实验的关键指标
        experiment_summaries = []
        
        for exp_name in experiment_names:
            try:
                data = self.load_experiment_data(experiment_results_dir, exp_name)
                summary_stats = data.get("summary_statistics", {})
                
                experiment_summaries.append({
                    "name": exp_name,
                    "display_name": exp_name.replace("_", " ").title(),
                    "mean_optimization": summary_stats.get("mean_optimization", 0),
                    "sample_count": summary_stats.get("sample_count", 0),
                    "success_rate": data["experiment_info"]["success_count"] / data["experiment_info"]["sample_count"] * 100
                })
            except Exception as e:
                print(f"警告：无法加载实验 {exp_name}: {e}")
        
        if not experiment_summaries:
            raise ValueError("没有有效的实验数据")
        
        # 创建仪表板
        fig = plt.figure(figsize=(16, 12))
        
        # 网格布局
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. 优化效果总览（左上）
        ax1 = fig.add_subplot(gs[0, :2])
        
        exp_names = [exp["display_name"] for exp in experiment_summaries]
        optimizations = [exp["mean_optimization"] for exp in experiment_summaries]
        
        bars = ax1.barh(exp_names, optimizations, color=self.color_palette[:len(exp_names)], alpha=0.8)
        ax1.set_xlabel('平均优化比例 (%)')
        ax1.set_title('各实验优化效果总览', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, opt in zip(bars, optimizations):
            ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f'{opt:.1f}%', ha='left', va='center', fontsize=10)
        
        # 2. 样本量分布（右上）
        ax2 = fig.add_subplot(gs[0, 2])
        
        sample_counts = [exp["sample_count"] for exp in experiment_summaries]
        
        ax2.pie(sample_counts, labels=exp_names, autopct='%1.0f', 
               colors=self.color_palette[:len(exp_names)], startangle=90)
        ax2.set_title('样本量分布', fontsize=12, fontweight='bold')
        
        # 3. 成功率对比（中左）
        ax3 = fig.add_subplot(gs[1, 0])
        
        success_rates = [exp["success_rate"] for exp in experiment_summaries]
        
        ax3.bar(range(len(exp_names)), success_rates, 
               color=self.color_palette[:len(exp_names)], alpha=0.8)
        ax3.set_ylabel('成功率 (%)')
        ax3.set_title('实验成功率', fontsize=12, fontweight='bold')
        ax3.set_xticks(range(len(exp_names)))
        ax3.set_xticklabels([name[:8] + '...' if len(name) > 8 else name 
                            for name in exp_names], rotation=45)
        ax3.set_ylim(0, 105)
        ax3.grid(True, alpha=0.3)
        
        # 4. 优化效果分布（中中）
        ax4 = fig.add_subplot(gs[1, 1])
        
        ax4.hist(optimizations, bins=5, color=self.color_palette[2], alpha=0.7, edgecolor='black')
        ax4.set_xlabel('优化比例 (%)')
        ax4.set_ylabel('实验数量')
        ax4.set_title('优化效果分布', fontsize=12, fontweight='bold')
        ax4.axvline(np.mean(optimizations), color='red', linestyle='--', 
                   label=f'平均: {np.mean(optimizations):.1f}%')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. 关键统计信息（中右）
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis('off')
        
        # 计算关键统计信息
        total_samples = sum(sample_counts)
        avg_optimization = np.mean(optimizations)
        best_experiment = experiment_summaries[np.argmax(optimizations)]
        avg_success_rate = np.mean(success_rates)
        
        stats_text = f"""
关键统计信息

总实验数: {len(experiment_summaries)}
总样本数: {total_samples}
平均优化: {avg_optimization:.1f}%
最佳实验: {best_experiment['display_name']}
  ({best_experiment['mean_optimization']:.1f}%)
平均成功率: {avg_success_rate:.1f}%
        """.strip()
        
        ax5.text(0.1, 0.9, stats_text, transform=ax5.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                facecolor=self.color_palette[0], alpha=0.2))
        
        # 6. 性能雷达图（下方）
        ax6 = fig.add_subplot(gs[2, :], projection='polar')
        
        # 选择前5个实验进行雷达图展示
        top_experiments = sorted(experiment_summaries, 
                               key=lambda x: x['mean_optimization'], reverse=True)[:5]
        
        angles = np.linspace(0, 2 * np.pi, len(top_experiments), endpoint=False)
        
        # 归一化指标到0-100范围
        metrics = {
            'optimization': [exp['mean_optimization'] for exp in top_experiments],
            'success_rate': [exp['success_rate'] for exp in top_experiments],
            'sample_efficiency': [exp['mean_optimization'] / exp['sample_count'] * 100 
                                for exp in top_experiments]
        }
        
        for i, (metric_name, values) in enumerate(metrics.items()):
            ax6.plot(angles, values, 'o-', linewidth=2, 
                    label=metric_name.replace('_', ' ').title(),
                    color=self.color_palette[i])
            ax6.fill(angles, values, alpha=0.1, color=self.color_palette[i])
        
        ax6.set_xticks(angles)
        ax6.set_xticklabels([exp['display_name'][:10] for exp in top_experiments])
        ax6.set_ylim(0, 100)
        ax6.set_title('前5个实验性能雷达图', fontsize=12, fontweight='bold', pad=20)
        ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax6.grid(True)
        
        # 添加总标题
        fig.suptitle('级联式语音对话系统延迟优化实验总结仪表板', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # 保存图表
        output_path = self.output_dir / f"{output_name}.png"
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return str(output_path)