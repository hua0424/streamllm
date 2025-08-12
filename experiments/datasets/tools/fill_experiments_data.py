#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验数据填充工具 - fill_experiments_data.py

根据实验设计要求，将已筛选的数据集复制到experiments目录，并进行以下处理：
1. CS-Dialogue数据：从filter_dataset目录复制到各实验目录
2. TTS数据：从tts_dataset目录复制，并根据实际音频时长重新分组
3. 生成详细的数据填充报告

符合统一标准：短语音(1-3秒)、中等语音(3-10秒)、长语音(10秒以上)
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any
import argparse
import time
from collections import defaultdict
import librosa
import soundfile as sf
import datetime

class ExperimentDataFiller:
    """实验数据填充器"""
    
    def __init__(self, base_path: str = None):
        """
        初始化数据填充器
        
        Args:
            base_path: 数据集基础路径
        """
        if base_path is None:
            base_path = Path(__file__).parent.parent / "processed"
        
        self.base_path = Path(base_path)
        self.filter_dataset_path = self.base_path / "filter_dataset"
        self.tts_dataset_path = self.base_path / "tts_dataset"
        self.experiments_path = self.base_path / "experiments"
        
        # 统一长度标准
        self.length_standards = {
            'short': (1.0, 3.0),    # 短语音：1-3秒
            'medium': (3.0, 10.0),  # 中等语音：3-10秒
            'long': (10.0, float('inf'))  # 长语音：10秒以上
        }
        
        # ASR期望的采样率
        self.target_sample_rate = 16000
        
        # 实验目录配置
        self.experiment_configs = {
            'core_comparison': {
                'description': '核心性能与质量对比实验',
                'samples_per_group': 100,  # 每组需要的样本数
                'groups': ['short', 'medium', 'long'],
                'datasets': ['cs-dialogue', 'dailydialog', 'lccc']
            },
            'length_analysis': {
                'description': '输入长度对优化效果的影响分析',
                'samples_per_group': 50,
                'groups': ['short', 'medium', 'long'],
                'datasets': ['cs-dialogue', 'dailydialog', 'lccc']
            },
            'asr_context': {
                'description': '前置与后置音频段对ASR准确性的影响',
                'samples_per_group': 30,
                'groups': ['long'],  # 主要使用长语音测试
                'datasets': ['cs-dialogue']
            },
            'ablation_study': {
                'description': '消融研究',
                'samples_per_group': 20,
                'groups': ['medium', 'long'],  # 固定样本，用于对比
                'datasets': ['cs-dialogue', 'dailydialog']
            },
            'case_analysis': {
                'description': '典型案例分析',
                'samples_per_group': 5,
                'groups': ['long'],  # 典型长语音案例
                'datasets': ['cs-dialogue', 'dailydialog', 'lccc']
            }
        }
        
        # 统计信息
        self.stats = {
            'total_processed': 0,
            'regrouped_samples': 0,
            'copied_samples': 0,
            'resampled_audio': 0,  # 新增：记录重采样的音频数量
            'errors': [],
            'experiment_summary': defaultdict(lambda: defaultdict(int))
        }
        
        print(f"初始化实验数据填充器")
        print(f"基础路径: {self.base_path}")
        print(f"实验路径: {self.experiments_path}")
    
    def get_audio_duration(self, audio_path: Path) -> float:
        """
        获取音频文件的实际时长
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            音频时长（秒）
        """
        try:
            y, sr = librosa.load(str(audio_path), sr=None)
            duration = len(y) / sr
            return duration
        except Exception as e:
            print(f"获取音频时长失败 {audio_path}: {e}")
            return 0.0
    
    def determine_length_group(self, duration: float) -> str:
        """
        根据实际时长确定长度分组
        
        Args:
            duration: 音频时长（秒）
            
        Returns:
            长度分组 ('short', 'medium', 'long')
        """
        for group, (min_dur, max_dur) in self.length_standards.items():
            if min_dur <= duration < max_dur:
                return group
        
        # 如果超出范围，默认为long
        if duration >= 10.0:
            return 'long'
        else:
            return 'short'
    
    def create_experiment_directories(self):
        """创建实验目录结构"""
        print("\\n创建实验目录结构...")
        
        for exp_name, config in self.experiment_configs.items():
            exp_path = self.experiments_path / exp_name
            
            # 创建主目录
            exp_path.mkdir(parents=True, exist_ok=True)
            
            # 创建音频和转录目录
            for group in config['groups']:
                (exp_path / "audio" / group).mkdir(parents=True, exist_ok=True)
                (exp_path / "transcripts" / group).mkdir(parents=True, exist_ok=True)
            
            print(f"  创建 {exp_name}: {config['description']}")
    
    def process_cs_dialogue_data(self) -> Dict[str, List]:
        """
        处理CS-Dialogue数据集
        
        Returns:
            按长度分组的样本列表
        """
        print("\\n处理CS-Dialogue数据集...")
        
        cs_dialogue_path = self.filter_dataset_path / "CS-Dialogue"
        grouped_samples = defaultdict(list)
        
        if not cs_dialogue_path.exists():
            print(f"警告：CS-Dialogue路径不存在 {cs_dialogue_path}")
            return grouped_samples
        
        # 处理JSON文件
        json_files = list(cs_dialogue_path.glob("*.json"))
        print(f"找到 {len(json_files)} 个CS-Dialogue JSON文件")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 获取对应的音频文件
                audio_filename = data.get('audio_file', '')
                original_group = data.get('length_group', '').replace('_10plus', '').replace('_3to10s', '').replace('_1to3s', '')
                
                # 查找音频文件
                audio_path = cs_dialogue_path / "audio" / original_group / audio_filename
                
                if not audio_path.exists():
                    print(f"警告：音频文件不存在 {audio_path}")
                    continue
                
                # 获取实际音频时长
                actual_duration = self.get_audio_duration(audio_path)
                if actual_duration <= 0:
                    continue
                
                # 根据实际时长重新分组
                new_group = self.determine_length_group(actual_duration)
                
                # 更新数据
                data['actual_duration'] = actual_duration
                data['original_group'] = original_group
                data['new_group'] = new_group
                
                if new_group != original_group:
                    self.stats['regrouped_samples'] += 1
                    print(f"  重新分组: {json_file.name} {original_group} -> {new_group} ({actual_duration:.1f}s)")
                
                # 添加到相应分组
                grouped_samples[new_group].append({
                    'json_data': data,
                    'json_path': json_file,
                    'audio_path': audio_path
                })
                
                self.stats['total_processed'] += 1
                
            except Exception as e:
                error_msg = f"处理CS-Dialogue文件失败 {json_file}: {e}"
                print(f"错误: {error_msg}")
                self.stats['errors'].append(error_msg)
        
        print(f"CS-Dialogue处理完成: {self.stats['total_processed']} 个样本")
        for group, samples in grouped_samples.items():
            print(f"  {group}: {len(samples)} 个样本")
        
        return grouped_samples
    
    def process_tts_dataset(self, dataset_name: str) -> Dict[str, List]:
        """
        处理TTS数据集（dailydialog/lccc）
        
        Args:
            dataset_name: 数据集名称
            
        Returns:
            按长度分组的样本列表
        """
        print(f"\\n处理TTS数据集: {dataset_name}...")
        
        dataset_path = self.tts_dataset_path / dataset_name
        grouped_samples = defaultdict(list)
        
        if not dataset_path.exists():
            print(f"警告：TTS数据集路径不存在 {dataset_path}")
            return grouped_samples
        
        # 遍历所有长度组目录
        for group_dir in dataset_path.iterdir():
            if not group_dir.is_dir():
                continue
            
            print(f"  处理分组目录: {group_dir.name}")
            
            # 处理该分组的所有JSON文件
            json_files = list(group_dir.glob("*.json"))
            for json_file in json_files:
                try:
                    # 加载JSON数据
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # 查找对应的音频文件
                    audio_filename = data.get('audio_file', json_file.stem + '.wav')
                    audio_path = group_dir / audio_filename
                    
                    if not audio_path.exists():
                        print(f"警告：音频文件不存在 {audio_path}")
                        continue
                    
                    # 获取实际音频时长
                    actual_duration = self.get_audio_duration(audio_path)
                    if actual_duration <= 0:
                        continue
                    
                    # 根据实际时长重新分组
                    original_group = group_dir.name
                    new_group = self.determine_length_group(actual_duration)
                    
                    # 更新数据
                    data['actual_duration'] = actual_duration
                    data['original_group'] = original_group
                    data['new_group'] = new_group
                    
                    if new_group != original_group:
                        self.stats['regrouped_samples'] += 1
                        print(f"    重新分组: {json_file.name} {original_group} -> {new_group} ({actual_duration:.1f}s)")
                    
                    # 添加到相应分组
                    grouped_samples[new_group].append({
                        'json_data': data,
                        'json_path': json_file,
                        'audio_path': audio_path
                    })
                    
                    self.stats['total_processed'] += 1
                    
                except Exception as e:
                    error_msg = f"处理TTS文件失败 {json_file}: {e}"
                    print(f"错误: {error_msg}")
                    self.stats['errors'].append(error_msg)
        
        print(f"{dataset_name}处理完成: {sum(len(samples) for samples in grouped_samples.values())} 个样本")
        for group, samples in grouped_samples.items():
            print(f"  {group}: {len(samples)} 个样本")
        
        return grouped_samples
    
    def copy_and_resample_audio(self, source_path: Path, dest_path: Path) -> bool:
        """
        复制音频文件，如果需要则进行重采样
        
        Args:
            source_path: 源音频文件路径
            dest_path: 目标音频文件路径
            
        Returns:
            是否进行了重采样
        """
        try:
            # 加载音频，保持原始采样率
            y, sr = librosa.load(str(source_path), sr=None)
            
            # 检查是否需要重采样
            if sr != self.target_sample_rate:
                # 需要重采样到16000Hz
                print(f"      重采样: {sr}Hz -> {self.target_sample_rate}Hz")
                y = librosa.resample(y, orig_sr=sr, target_sr=self.target_sample_rate)
                
                # 保存重采样后的音频
                sf.write(str(dest_path), y, self.target_sample_rate)
                self.stats['resampled_audio'] += 1
                return True
            else:
                # 采样率已经正确，直接复制
                shutil.copy2(source_path, dest_path)
                return False
                
        except Exception as e:
            # 如果处理失败，尝试直接复制
            print(f"      警告：音频处理失败，尝试直接复制: {e}")
            shutil.copy2(source_path, dest_path)
            return False
    
    def copy_samples_to_experiment(self, exp_name: str, group: str, samples: List[Dict], max_samples: int):
        """
        复制样本到实验目录
        
        Args:
            exp_name: 实验名称
            group: 长度分组
            samples: 样本列表
            max_samples: 最大样本数
        """
        if not samples:
            return
        
        exp_path = self.experiments_path / exp_name
        audio_dir = exp_path / "audio" / group
        transcripts_dir = exp_path / "transcripts" / group
        
        # 确保目录存在
        audio_dir.mkdir(parents=True, exist_ok=True)
        transcripts_dir.mkdir(parents=True, exist_ok=True)
        
        # 限制样本数量
        selected_samples = samples[:max_samples] if len(samples) > max_samples else samples
        
        for i, sample in enumerate(selected_samples):
            try:
                json_data = sample['json_data']
                audio_path = sample['audio_path']
                
                # 生成新的文件名
                sample_id = f"sample_{i+1:03d}"
                new_audio_name = f"{sample_id}.wav"
                new_json_name = f"{sample_id}.json"
                
                # 复制音频文件（如果需要会进行重采样）
                dest_audio_path = audio_dir / new_audio_name
                
                # 判断是否是TTS生成的音频（来自dailydialog或lccc）
                is_tts_audio = 'dailydialog' in str(audio_path) or 'lccc' in str(audio_path)
                
                if is_tts_audio:
                    # TTS音频需要检查并可能重采样
                    resampled = self.copy_and_resample_audio(audio_path, dest_audio_path)
                    if resampled:
                        # 如果进行了重采样，需要更新音频时长
                        json_data['actual_duration'] = self.get_audio_duration(dest_audio_path)
                        json_data['resampled'] = True
                        json_data['original_sample_rate'] = 22050
                        json_data['target_sample_rate'] = self.target_sample_rate
                else:
                    # CS-Dialogue音频直接复制
                    shutil.copy2(audio_path, dest_audio_path)
                
                # 更新JSON数据并保存
                json_data['audio_file'] = new_audio_name
                json_data['experiment_sample_id'] = sample_id
                json_data['experiment_name'] = exp_name
                json_data['final_group'] = group
                
                dest_json_path = transcripts_dir / new_json_name
                with open(dest_json_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=2)
                
                self.stats['copied_samples'] += 1
                self.stats['experiment_summary'][exp_name][group] += 1
                
            except Exception as e:
                error_msg = f"复制样本失败到 {exp_name}/{group}: {e}"
                print(f"错误: {error_msg}")
                self.stats['errors'].append(error_msg)
    
    def fill_experiment_data(self, experiment_name: str = None):
        """
        填充实验数据
        
        Args:
            experiment_name: 指定实验名称，None表示处理所有实验
        """
        print(f"\\n开始填充实验数据...")
        
        # 处理源数据
        cs_dialogue_data = self.process_cs_dialogue_data()
        dailydialog_data = self.process_tts_dataset('dailydialog')
        lccc_data = self.process_tts_dataset('lccc')
        
        # 合并数据源
        all_data = {
            'cs-dialogue': cs_dialogue_data,
            'dailydialog': dailydialog_data,
            'lccc': lccc_data
        }
        
        # 确定要处理的实验
        if experiment_name:
            experiments_to_process = {experiment_name: self.experiment_configs[experiment_name]}
        else:
            experiments_to_process = self.experiment_configs
        
        # 为每个实验填充数据
        for exp_name, config in experiments_to_process.items():
            print(f"\\n填充实验: {exp_name} - {config['description']}")
            
            for group in config['groups']:
                print(f"  处理分组: {group}")
                
                # 收集该分组的所有可用样本
                available_samples = []
                for dataset_name in config['datasets']:
                    if dataset_name in all_data and group in all_data[dataset_name]:
                        dataset_samples = all_data[dataset_name][group]
                        available_samples.extend(dataset_samples)
                        print(f"    从 {dataset_name} 获取 {len(dataset_samples)} 个样本")
                
                # 复制样本到实验目录
                max_samples = config['samples_per_group']
                if available_samples:
                    print(f"    复制 {min(len(available_samples), max_samples)} 个样本到 {exp_name}/{group}")
                    self.copy_samples_to_experiment(exp_name, group, available_samples, max_samples)
                else:
                    print(f"    警告：{exp_name}/{group} 没有可用样本")
    
    def generate_report(self) -> str:
        """
        生成数据填充报告
        
        Returns:
            报告内容
        """
        report_lines = [
            "=" * 80,
            "StreamLLM 实验数据填充报告",
            "=" * 80,
            f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"基础路径: {self.base_path}",
            "",
            "统计摘要:",
            f"  总处理样本数: {self.stats['total_processed']}",
            f"  重新分组样本数: {self.stats['regrouped_samples']}",
            f"  成功复制样本数: {self.stats['copied_samples']}",
            f"  重采样音频数: {self.stats['resampled_audio']}",
            f"  错误数量: {len(self.stats['errors'])}",
            ""
        ]
        
        # 长度标准说明
        report_lines.extend([
            "长度分组标准:",
            f"  短语音 (short): {self.length_standards['short'][0]}-{self.length_standards['short'][1]}秒",
            f"  中等语音 (medium): {self.length_standards['medium'][0]}-{self.length_standards['medium'][1]}秒",  
            f"  长语音 (long): {self.length_standards['long'][0]}秒以上",
            ""
        ])
        
        # 实验数据统计
        report_lines.append("实验数据统计:")
        for exp_name, groups in self.stats['experiment_summary'].items():
            config = self.experiment_configs.get(exp_name, {})
            report_lines.append(f"  {exp_name} - {config.get('description', '')}:")
            
            total_samples = sum(groups.values())
            report_lines.append(f"    总样本数: {total_samples}")
            
            for group in ['short', 'medium', 'long']:
                if group in groups:
                    count = groups[group]
                    required = config.get('samples_per_group', 0)
                    status = "✅" if count >= required else "⚠️" if count > 0 else "❌"
                    report_lines.append(f"    {group}: {count} 个样本 (需要{required}个) {status}")
            report_lines.append("")
        
        # 错误报告
        if self.stats['errors']:
            report_lines.extend([
                "错误详情:",
                *[f"  - {error}" for error in self.stats['errors'][:10]],  # 只显示前10个错误
                f"  ... 共 {len(self.stats['errors'])} 个错误" if len(self.stats['errors']) > 10 else "",
                ""
            ])
        
        # 目录结构验证
        report_lines.append("目录结构验证:")
        for exp_name in self.experiment_configs.keys():
            exp_path = self.experiments_path / exp_name
            if exp_path.exists():
                report_lines.append(f"  ✅ {exp_name}")
                
                # 检查每个分组
                config = self.experiment_configs[exp_name]
                for group in config['groups']:
                    audio_dir = exp_path / "audio" / group
                    transcripts_dir = exp_path / "transcripts" / group
                    
                    audio_count = len(list(audio_dir.glob("*.wav"))) if audio_dir.exists() else 0
                    json_count = len(list(transcripts_dir.glob("*.json"))) if transcripts_dir.exists() else 0
                    
                    if audio_count == json_count and audio_count > 0:
                        report_lines.append(f"    ✅ {group}: {audio_count} 个配对文件")
                    elif audio_count == json_count:
                        report_lines.append(f"    ⚠️ {group}: 无数据文件")
                    else:
                        report_lines.append(f"    ❌ {group}: 音频({audio_count})和JSON({json_count})文件不匹配")
            else:
                report_lines.append(f"  ❌ {exp_name} (目录不存在)")
        
        report_lines.extend([
            "",
            "=" * 80,
            "数据填充完成",
            "=" * 80
        ])
        
        return "\\n".join(report_lines)
    
    def save_report(self, report_content: str):
        """
        保存报告到文件
        
        Args:
            report_content: 报告内容
        """
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.base_path / f"fill_experiments_data_report_{timestamp}.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"\\n报告已保存到: {report_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='实验数据填充工具')
    parser.add_argument('--experiment', type=str, 
                       choices=['core_comparison', 'length_analysis', 'asr_context', 'ablation_study', 'case_analysis'],
                       help='指定要处理的实验名称（默认处理所有实验）')
    parser.add_argument('--base-path', type=str,
                       default=None,
                       help='数据集基础路径')
    parser.add_argument('--dry-run', action='store_true',
                       help='仅分析数据，不进行实际复制')
    
    args = parser.parse_args()
    
    # 创建数据填充器
    filler = ExperimentDataFiller(base_path=args.base_path)
    
    # 创建实验目录结构
    if not args.dry_run:
        filler.create_experiment_directories()
    
    # 填充实验数据
    if not args.dry_run:
        filler.fill_experiment_data(args.experiment)
    else:
        print("DRY RUN 模式：仅分析数据，不进行复制")
        # 只进行数据分析
        filler.process_cs_dialogue_data()
        filler.process_tts_dataset('dailydialog') 
        filler.process_tts_dataset('lccc')
    
    # 生成并保存报告
    report = filler.generate_report()
    print("\\n" + report)
    
    if not args.dry_run:
        filler.save_report(report)
    
    print(f"\\n✅ 数据填充任务完成")
    print(f"总计处理: {filler.stats['total_processed']} 个样本")
    print(f"重新分组: {filler.stats['regrouped_samples']} 个样本")
    print(f"成功复制: {filler.stats['copied_samples']} 个样本")
    print(f"重采样音频: {filler.stats['resampled_audio']} 个文件")


if __name__ == "__main__":
    main()