#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据过滤处理程序
从原始数据集中筛选并过滤符合要求的样本到filter_dataset目录
根据DATA_PREPARATION_GUIDE.md的要求处理CS-Dialogue、DailyDialog和LCCC数据集
"""

import json
import random
import re
import shutil
from pathlib import Path
from typing import List, Dict, Optional
import argparse
import librosa


class DatasetFilter:
    """数据集过滤器基类"""
    
    def __init__(self, raw_data_path: Path, output_path: Path):
        self.raw_data_path = raw_data_path
        self.output_path = output_path
        
        # 清空输出目录，避免多次运行文件混杂
        if self.output_path.exists():
            shutil.rmtree(self.output_path)
        
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # 时长估算参数（基于文档提供的参考数据）
        self.duration_params = {
            'zh': 0.19,    # 中文：约0.15-0.23秒/字符
            'en': 0.10,    # 英文：约0.07-0.13秒/字符  
            'mix': 0.15    # 混合：约0.10-0.15秒/字符
        }
        
        # 长度分组阈值（基于估算时长，单位：秒）
        self.length_groups = {
            'short': (1, 3),      # 1-3秒
            'medium': (3, 10),    # 3-10秒
            'long': (10, 30)      # 10-30秒（设置上限避免过长）
        }
        
        # 每个分组的目标样本数
        self.samples_per_group = 100
        
    def estimate_duration(self, text: str, language: str = 'zh') -> float:
        """估算文本的音频时长"""
        char_count = len(text.replace(' ', ''))
        duration_per_char = self.duration_params.get(language, 0.15)
        return char_count * duration_per_char
    
    def get_audio_duration(self, audio_path: Path) -> Optional[float]:
        """获取音频文件的实际时长"""
        try:
            if audio_path.exists():
                audio_data, sr = librosa.load(str(audio_path), sr=None)
                duration = len(audio_data) / sr
                return duration
        except Exception as e:
            print(f"  警告：无法读取音频文件 {audio_path}: {e}")
        return None
    
    def get_length_group(self, duration: float) -> Optional[str]:
        """根据时长获取长度分组"""
        for group, (min_dur, max_dur) in self.length_groups.items():
            if min_dur <= duration < max_dur:
                return group
        return None  # 不在任何分组范围内
    
    def detect_language(self, text: str) -> str:
        """检测文本语言"""
        # 简单的语言检测
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        
        if chinese_chars > 0 and english_chars > 0:
            return 'mix'
        elif chinese_chars > english_chars:
            return 'zh'
        else:
            return 'en'
    
    def clean_text_for_tts(self, text: str) -> str:
        """
        清理文本以适配TTS处理
        移除标点符号附近的多余空格，避免TTS错误发音
        
        Args:
            text: 原始文本
            
        Returns:
            清理后的文本
        """
        if not text:
            return text
        
        # 移除标点符号前的空格
        # 处理常见英文标点：? ! . , ; : ' "
        text = re.sub(r'\s+([?!.,;:\'""])', r'\1', text)
        
        # 移除标点符号后的多余空格（保留一个空格）
        text = re.sub(r'([?!.,;:\'""])\s+', r'\1 ', text)
        
        # 处理英文缩写的撇号：如 It ' s -> It's
        text = re.sub(r"(\w)\s+'\s*(\w)", r"\1'\2", text)
        
        # 处理引号内的空格：如 " hello " -> "hello"
        text = re.sub(r'"\s+([^"]*?)\s+"', r'"\1"', text)
        text = re.sub(r"'\s+([^']*?)\s+'", r"'\1'", text)
        
        # 移除多余的连续空格
        text = re.sub(r'\s+', ' ', text)
        
        # 去除首尾空格
        text = text.strip()
        
        return text
    
    def filter_samples_by_duration(self, samples: List[Dict]) -> Dict[str, List[Dict]]:
        """根据时长过滤并分组样本"""
        grouped_samples = {
            'short': [],
            'medium': [],
            'long': []
        }
        
        for sample in samples:
            # 过滤中英混合数据
            if sample['language'] == 'mix':
                continue
                
            # 过滤包含'*'号的样本
            if '*' in sample['text']:
                continue
                
            # 过滤包含<FIL/>标记的样本
            if '<FIL/>' in sample['text']:
                continue
                
            # 优先使用实际音频时长，否则使用估算时长
            duration = sample.get('actual_duration', sample['estimated_duration'])
            group = self.get_length_group(duration)
            
            # 只保留符合分组要求的样本
            if group and group in grouped_samples:
                grouped_samples[group].append(sample)
        
        return grouped_samples
    
    def save_samples(self, samples: List[Dict], dataset_name: str):
        """保存过滤后的样本"""
        # 先根据时长过滤和分组
        grouped_samples = self.filter_samples_by_duration(samples)
        
        # 统计信息
        total_saved = 0
        
        # 为每个分组保存样本
        for group_name in ['short', 'medium', 'long']:
            group_samples = grouped_samples[group_name]
            
            # 对样本进行排序，确保选择最合适的样本
            # 按时长从小到大排序（在该组范围内）
            group_samples.sort(key=lambda x: x.get('actual_duration', x['estimated_duration']))
            
            # 限制每组样本数量
            selected_samples = group_samples[:self.samples_per_group]
            
            # 为该分组创建音频目录（如果需要）
            if dataset_name == 'cs-dialogue' and selected_samples:
                audio_dir = self.output_path / 'audio' / group_name
                audio_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存为JSON文件
            for idx, sample in enumerate(selected_samples, 1):
                filename = f"sample_{group_name}_{idx:03d}.json"
                filepath = self.output_path / filename
                
                # 处理长度分组字符串
                min_dur, max_dur = self.length_groups[group_name]
                if group_name == 'long':
                    length_group_str = f"{group_name}_{min_dur}plus"
                else:
                    length_group_str = f"{group_name}_{min_dur}to{max_dur}s"
                
                # 确定音频文件名
                audio_filename = f"sample_{group_name}_{idx:03d}.wav"
                
                # 处理CS-Dialogue的音频文件
                if dataset_name == 'cs-dialogue' and 'audio_path' in sample:
                    # 复制音频文件
                    src_audio = sample['audio_path']
                    if src_audio.exists():
                        dst_audio = self.output_path / 'audio' / group_name / audio_filename
                        shutil.copy2(src_audio, dst_audio)
                        print(f"    复制音频: {src_audio.name} -> {dst_audio.name}")
                
                # 使用实际时长（如果有）
                duration = sample.get('actual_duration', sample['estimated_duration'])
                
                # 转换为目标格式
                output_data = {
                    "sample_id": f"{dataset_name}_{group_name}_{idx:03d}",
                    "source_dataset": dataset_name,
                    "text": sample['text'],
                    "language": sample['language'],
                    "duration": round(duration, 1),  # 使用实际时长
                    "audio_file": audio_filename,
                    "length_group": length_group_str,
                    "word_count": len(sample['text'])
                }
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)
                
                total_saved += 1
        
        print(f"已保存 {dataset_name} 数据集样本到 {self.output_path}")
        print(f"  总计保存: {total_saved} 个样本")
        for group_name in ['short', 'medium', 'long']:
            count = min(len(grouped_samples[group_name]), self.samples_per_group)
            available = len(grouped_samples[group_name])
            print(f"  - {group_name} ({self.length_groups[group_name][0]}-{self.length_groups[group_name][1]}s): "
                  f"保存 {count} 个，可用 {available} 个")


class CSDialogueFilter(DatasetFilter):
    """CS-Dialogue数据集过滤器"""
    
    def filter(self) -> List[Dict]:
        """过滤CS-Dialogue数据集"""
        samples = []
        
        # 读取短音频文本数据和音频路径映射
        text_file = self.raw_data_path / "data/index/short_wav/train/text"
        wav_scp_file = self.raw_data_path / "data/index/short_wav/train/wav.scp"
        
        # 构建音频路径映射
        audio_paths = {}
        if wav_scp_file.exists():
            with open(wav_scp_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(' ', 1)
                    if len(parts) == 2:
                        utterance_id, relative_path = parts
                        # 构建完整路径 - 注意short_wav目录出现两次
                        audio_path = self.raw_data_path / "data/short_wav" / relative_path
                        audio_paths[utterance_id] = audio_path
        
        if text_file.exists():
            with open(text_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    utterance_id, text = parts
                    # 只选择首轮对话（_2结尾的是第一个说话人的首次发言）
                    if utterance_id.endswith('_2'):
                        # 清理文本
                        text = text.strip()
                        if not text:
                            continue
                        
                        language = self.detect_language(text)
                        
                        # 获取音频文件路径和实际时长
                        audio_path = audio_paths.get(utterance_id)
                        
                        # CS-Dialogue数据集只从有音频文件的样本中抽样
                        if not (audio_path and audio_path.exists()):
                            continue
                        
                        actual_duration = self.get_audio_duration(audio_path)
                        # 如果无法获取实际时长，跳过该样本
                        if actual_duration is None:
                            continue
                        
                        estimated_duration = self.estimate_duration(text, language)
                        
                        # 同时满足实际音频时长和估算文本时长的要求
                        actual_group = self.get_length_group(actual_duration)
                        
                        # 实际音频时长必须在合理范围内
                        if actual_group is None:
                            continue
                        
                        # 文本估算时长只需满足下限要求
                        min_duration = self.length_groups[actual_group][0]
                        if estimated_duration < min_duration:
                            continue
                        
                        # 清理文本以适配TTS处理
                        cleaned_text = self.clean_text_for_tts(text)
                        
                        sample_data = {
                            'text': cleaned_text,
                            'language': language,
                            'estimated_duration': estimated_duration,
                            'actual_duration': actual_duration,
                            'audio_path': audio_path,
                            'utterance_id': utterance_id
                        }
                        
                        samples.append(sample_data)
        
        # 随机打乱
        random.shuffle(samples)
        return samples


class DailyDialogFilter(DatasetFilter):
    """DailyDialog数据集过滤器"""
    
    def filter(self) -> List[Dict]:
        """过滤DailyDialog数据集"""
        samples = []
        
        # 读取训练数据
        train_file = self.raw_data_path / "train/dialogues_train.txt"
        if train_file.exists():
            with open(train_file, 'r', encoding='utf-8') as f:
                dialogues = f.readlines()
            
            for dialogue in dialogues:
                # 分割对话轮次
                utterances = dialogue.strip().split(' __eou__ ')
                if utterances:
                    # 获取第一轮对话的第一个发言
                    first_utterance = utterances[0].strip()
                    if first_utterance and len(first_utterance) > 5:  # 过滤太短的句子
                        language = 'en'  # DailyDialog是英文数据集
                        
                        # 清理文本以适配TTS处理
                        cleaned_text = self.clean_text_for_tts(first_utterance)
                        duration = self.estimate_duration(cleaned_text, language)
                        
                        samples.append({
                            'text': cleaned_text,
                            'language': language,
                            'estimated_duration': duration
                        })
        
        # 随机打乱
        random.shuffle(samples)
        return samples


class LCCCFilter(DatasetFilter):
    """LCCC数据集过滤器"""
    
    def filter(self) -> List[Dict]:
        """过滤LCCC数据集"""
        samples = []
        
        # 读取训练数据（分批读取避免内存溢出）
        train_file = self.raw_data_path / "lccc_base_train.jsonl"
        if train_file.exists():
            with open(train_file, 'r', encoding='utf-8') as f:
                # 读取更多行以获得足够的长样本
                for i, line in enumerate(f):
                    if i >= 50000:  # 增加读取量以获得更多样本
                        break
                    
                    try:
                        dialogue = json.loads(line)
                        if dialogue and len(dialogue) > 0:
                            # 获取第一轮对话
                            first_utterance = dialogue[0]
                            # 处理可能的空格分词
                            text = first_utterance.replace(' ', '')
                            
                            # 过滤太短的文本
                            if len(text) < 2:
                                continue
                            
                            language = 'zh'  # LCCC是中文数据集
                            
                            # 清理文本以适配TTS处理
                            cleaned_text = self.clean_text_for_tts(text)
                            duration = self.estimate_duration(cleaned_text, language)
                            
                            samples.append({
                                'text': cleaned_text,
                                'language': language,
                                'estimated_duration': duration
                            })
                    except:
                        continue
        
        # 随机打乱
        random.shuffle(samples)
        return samples


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='过滤原始数据集生成实验数据')
    parser.add_argument('--dataset', type=str, choices=['cs-dialogue', 'dailydialog', 'lccc', 'all'],
                       default='all', help='要处理的数据集')
    parser.add_argument('--test', action='store_true', help='测试模式，只处理少量数据')
    args = parser.parse_args()
    
    # 设置路径
    base_path = Path("/usr/local/app/jupyterlab/yanjiu/streamllm/experiments/datasets")
    raw_data_path = base_path / "raw_data"
    filter_dataset_path = base_path / "processed/filter_dataset"
    
    # 处理CS-Dialogue数据集
    if args.dataset in ['cs-dialogue', 'all']:
        print("\n处理 CS-Dialogue 数据集...")
        cs_filter = CSDialogueFilter(
            raw_data_path / "CS-Dialogue",
            filter_dataset_path / "CS-Dialogue"
        )
        if args.test:
            cs_filter.samples_per_group = 5  # 测试模式只生成5个样本
        cs_samples = cs_filter.filter()
        print(f"  找到 {len(cs_samples)} 个原始样本")
        # 统计有音频的样本
        audio_samples = [s for s in cs_samples if 'audio_path' in s]
        print(f"  其中有音频文件的样本: {len(audio_samples)} 个")
        cs_filter.save_samples(cs_samples, 'cs-dialogue')
    
    # 处理DailyDialog数据集
    if args.dataset in ['dailydialog', 'all']:
        print("\n处理 DailyDialog 数据集...")
        daily_filter = DailyDialogFilter(
            raw_data_path / "dailydialog",
            filter_dataset_path / "dailydialog"
        )
        if args.test:
            daily_filter.samples_per_group = 5
        daily_samples = daily_filter.filter()
        print(f"  找到 {len(daily_samples)} 个原始样本")
        daily_filter.save_samples(daily_samples, 'dailydialog')
    
    # 处理LCCC数据集
    if args.dataset in ['lccc', 'all']:
        print("\n处理 LCCC 数据集...")
        lccc_filter = LCCCFilter(
            raw_data_path / "lccc",
            filter_dataset_path / "lccc"
        )
        if args.test:
            lccc_filter.samples_per_group = 5
        lccc_samples = lccc_filter.filter()
        print(f"  找到 {len(lccc_samples)} 个原始样本")
        lccc_filter.save_samples(lccc_samples, 'lccc')
    
    print("\n数据过滤完成！")
    
    # 生成统计报告
    print("\n=== 数据统计报告 ===")
    for dataset_name in ['CS-Dialogue', 'dailydialog', 'lccc']:
        dataset_path = filter_dataset_path / dataset_name
        if dataset_path.exists():
            files = list(dataset_path.glob("*.json"))
            if not files:
                continue
                
            print(f"\n{dataset_name}:")
            print(f"  总样本数: {len(files)}")
            
            # 检查音频文件（仅CS-Dialogue）
            if dataset_name == 'CS-Dialogue':
                audio_path = dataset_path / 'audio'
                if audio_path.exists():
                    audio_files = list(audio_path.rglob("*.wav"))
                    print(f"  音频文件数: {len(audio_files)}")
            
            # 统计各组样本数和时长
            for group in ['short', 'medium', 'long']:
                group_files = [f for f in files if f'_{group}_' in f.name]
                if group_files:
                    durations = []
                    for file in group_files[:5]:  # 检查前5个
                        with open(file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            durations.append(data.get('duration', data.get('estimated_duration', 0)))
                    
                    if durations:
                        print(f"  - {group}: {len(group_files)} 个样本, "
                              f"时长范围: {min(durations):.1f}s - {max(durations):.1f}s")


if __name__ == "__main__":
    main()