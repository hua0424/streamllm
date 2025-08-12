#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TTS Filter Dataset Processing Tool
处理filter_dataset目录下的dailydialog和lccc数据集，使用TTS服务生成语音文件，
并将生成的音频文件按实验设计要求放置到tts_dataset目录下。
"""

import json
import shutil
from pathlib import Path
from typing import List, Dict
import argparse
import time
from tts import TTSClient, BatchTTSProcessor

class TTSFilterDatasetProcessor:
    """TTS过滤数据集处理器"""
    
    def __init__(self, base_path: Path, tts_url: str = "http://host.docker.internal:20401", force_regenerate: bool = False):
        """
        初始化TTS数据集处理器
        
        Args:
            base_path: experiments/datasets基础路径
            tts_url: TTS服务URL
            force_regenerate: 是否强制重新生成（删除已有文件）
        """
        self.base_path = Path(base_path)
        self.filter_dataset_path = self.base_path / "processed/filter_dataset"
        self.tts_dataset_path = self.base_path / "processed/tts_dataset"
        self.force_regenerate = force_regenerate
        
        # 创建TTS客户端
        self.tts_client = TTSClient(tts_url)
        
        # 支持的数据集（不包括CS-Dialogue，因为它已有音频文件）
        self.datasets = ['dailydialog', 'lccc']
        
        # 语言到说话人的映射
        self.language_speaker_map = {
            'zh': '晓伊',
            'en': '英文女'
        }
    
    def validate_paths(self) -> bool:
        """验证路径有效性"""
        if not self.filter_dataset_path.exists():
            print(f"错误：filter_dataset路径不存在: {self.filter_dataset_path}")
            return False
        
        for dataset in self.datasets:
            dataset_path = self.filter_dataset_path / dataset
            if not dataset_path.exists():
                print(f"警告：数据集目录不存在: {dataset_path}")
            else:
                json_files = list(dataset_path.glob("*.json"))
                print(f"找到 {dataset} 数据集: {len(json_files)} 个JSON文件")
        
        return True
    
    def test_tts_connection(self) -> bool:
        """测试TTS服务连接"""
        print("测试TTS服务连接...")
        if self.tts_client.test_connection():
            print("✅ TTS服务连接正常")
            return True
        else:
            print("❌ TTS服务连接失败，请检查服务状态")
            return False
    
    def clear_existing_files(self, dataset_name: str):
        """清理已存在的文件"""
        dataset_path = self.tts_dataset_path / dataset_name
        if not dataset_path.exists():
            return
        
        print(f"清理 {dataset_name} 数据集的已有文件...")
        
        files_removed = 0
        for group in ['short', 'medium', 'long']:
            group_path = dataset_path / group
            if group_path.exists():
                # 删除音频文件
                for audio_file in group_path.glob("*.wav"):
                    audio_file.unlink()
                    files_removed += 1
                
                # 删除JSON文件
                for json_file in group_path.glob("*.json"):
                    json_file.unlink()
                    files_removed += 1
        
        print(f"已删除 {files_removed} 个文件")
    
    def load_json_data(self, json_file: Path) -> Dict:
        """加载JSON数据"""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"读取JSON文件失败 {json_file}: {e}")
            return None
    
    def process_dataset(self, dataset_name: str, speed_factor: float = 1.0) -> bool:
        """
        处理单个数据集
        
        Args:
            dataset_name: 数据集名称 (dailydialog/lccc)
            speed_factor: 语速调节系数
        
        Returns:
            处理是否成功
        """
        print(f"\n处理数据集: {dataset_name}")
        
        # 输入和输出路径
        input_path = self.filter_dataset_path / dataset_name
        output_path = self.tts_dataset_path / dataset_name
        
        if not input_path.exists():
            print(f"跳过不存在的数据集: {input_path}")
            return False
        
        # 如果强制重新生成，先清理已有文件
        if self.force_regenerate:
            self.clear_existing_files(dataset_name)
        
        # 创建输出目录结构
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 创建分组目录
        for group in ['short', 'medium', 'long']:
            (output_path / group).mkdir(exist_ok=True)
        
        # 获取所有JSON文件
        json_files = list(input_path.glob("*.json"))
        if not json_files:
            print(f"在 {input_path} 中未找到JSON文件")
            return False
        
        print(f"找到 {len(json_files)} 个样本文件")
        
        success_count = 0
        failed_count = 0
        
        for json_file in sorted(json_files):
            try:
                # 加载JSON数据
                data = self.load_json_data(json_file)
                if not data:
                    failed_count += 1
                    continue
                
                text = data.get('text', '').strip()
                language = data.get('language', 'zh')
                audio_filename = data.get('audio_file', json_file.stem + '.wav')
                length_group = data.get('length_group', '')
                
                if not text:
                    print(f"跳过空文本文件: {json_file.name}")
                    failed_count += 1
                    continue
                
                # 确定分组目录
                group_dir = 'short'  # 默认
                if 'medium' in length_group:
                    group_dir = 'medium'
                elif 'long' in length_group:
                    group_dir = 'long'
                
                # 输出音频路径
                output_audio_path = output_path / group_dir / audio_filename
                
                # 跳过已存在的文件（除非强制重新生成）
                if output_audio_path.exists() and not self.force_regenerate:
                    print(f"跳过已存在文件: {output_audio_path.name}")
                    success_count += 1
                    continue
                
                # 根据语言选择说话人
                spk_id = self.language_speaker_map.get(language, '晓伊')
                
                print(f"处理样本: {json_file.name}")
                print(f"  文本: {text[:50]}{'...' if len(text) > 50 else ''}")
                print(f"  语言: {language} -> 说话人: {spk_id}")
                print(f"  分组: {group_dir}")
                print(f"  输出: {output_audio_path.name}")
                
                # 调用TTS服务生成音频
                result = self.tts_client.synthesize(
                    tts_text=text,
                    spk_id=spk_id,
                    output_path=str(output_audio_path),
                    speed_factor=speed_factor
                )
                
                if result:
                    success_count += 1
                    print(f"  ✅ 生成成功")
                    
                    # 验证音频文件
                    if output_audio_path.exists():
                        file_size = output_audio_path.stat().st_size
                        print(f"  文件大小: {file_size / 1024:.1f} KB")
                    
                    # 复制对应的JSON文件到输出目录
                    output_json_path = output_path / group_dir / json_file.name
                    if not output_json_path.exists():
                        shutil.copy2(json_file, output_json_path)
                        print(f"  复制JSON: {output_json_path.name}")
                else:
                    failed_count += 1
                    print(f"  ❌ 生成失败")
                
                # 添加延迟避免过载
                time.sleep(0.5)
                
            except Exception as e:
                print(f"处理文件 {json_file.name} 时出错: {e}")
                failed_count += 1
        
        print(f"\n{dataset_name} 处理完成:")
        print(f"  成功: {success_count} 个")
        print(f"  失败: {failed_count} 个")
        
        return success_count > 0
    
    def generate_statistics_report(self):
        """生成统计报告"""
        print("\n" + "="*50)
        print("TTS数据集生成统计报告")
        print("="*50)
        
        for dataset_name in self.datasets:
            dataset_path = self.tts_dataset_path / dataset_name
            if not dataset_path.exists():
                continue
            
            print(f"\n{dataset_name.upper()} 数据集:")
            
            total_audio_files = 0
            total_json_files = 0
            
            for group in ['short', 'medium', 'long']:
                group_path = dataset_path / group
                if group_path.exists():
                    audio_files = list(group_path.glob("*.wav"))
                    json_files = list(group_path.glob("*.json"))
                    
                    print(f"  {group}: {len(audio_files)} 音频, {len(json_files)} JSON")
                    total_audio_files += len(audio_files)
                    total_json_files += len(json_files)
                    
                    # 检查文件大小
                    if audio_files:
                        sizes = [f.stat().st_size for f in audio_files[:3]]  # 检查前3个
                        avg_size = sum(sizes) / len(sizes) / 1024
                        print(f"    平均文件大小: {avg_size:.1f} KB")
            
            print(f"  总计: {total_audio_files} 音频, {total_json_files} JSON")
            
            # 检查目录结构完整性
            if total_audio_files != total_json_files:
                print(f"  ⚠️  警告: 音频文件数与JSON文件数不匹配")
    
    def validate_generated_data(self):
        """验证生成的数据"""
        print("\n验证生成的数据...")
        
        issues = []
        
        for dataset_name in self.datasets:
            dataset_path = self.tts_dataset_path / dataset_name
            if not dataset_path.exists():
                issues.append(f"{dataset_name} 目录不存在")
                continue
            
            for group in ['short', 'medium', 'long']:
                group_path = dataset_path / group
                if not group_path.exists():
                    issues.append(f"{dataset_name}/{group} 目录不存在")
                    continue
                
                # 检查音频和JSON文件匹配
                audio_files = {f.stem for f in group_path.glob("*.wav")}
                json_files = {f.stem for f in group_path.glob("*.json")}
                
                missing_audio = json_files - audio_files
                missing_json = audio_files - json_files
                
                if missing_audio:
                    issues.append(f"{dataset_name}/{group} 缺少音频文件: {missing_audio}")
                if missing_json:
                    issues.append(f"{dataset_name}/{group} 缺少JSON文件: {missing_json}")
                
                # 检查音频文件大小
                for audio_file in group_path.glob("*.wav"):
                    if audio_file.stat().st_size < 1000:  # 小于1KB可能有问题
                        issues.append(f"音频文件过小: {audio_file}")
        
        if issues:
            print("⚠️  发现以下问题:")
            for issue in issues[:10]:  # 只显示前10个问题
                print(f"  - {issue}")
            if len(issues) > 10:
                print(f"  ... 还有 {len(issues) - 10} 个问题")
        else:
            print("✅ 数据验证通过，未发现问题")
    
    def cleanup_empty_directories(self):
        """清理空目录"""
        print("\n清理空目录...")
        
        for dataset_name in self.datasets:
            dataset_path = self.tts_dataset_path / dataset_name
            if not dataset_path.exists():
                continue
            
            for group in ['short', 'medium', 'long']:
                group_path = dataset_path / group
                if group_path.exists() and not any(group_path.iterdir()):
                    print(f"删除空目录: {group_path}")
                    group_path.rmdir()
            
            # 如果数据集目录为空，也删除
            if dataset_path.exists() and not any(dataset_path.iterdir()):
                print(f"删除空数据集目录: {dataset_path}")
                dataset_path.rmdir()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='TTS Filter Dataset处理工具')
    parser.add_argument('--dataset', type=str, 
                       choices=['dailydialog', 'lccc', 'all'],
                       default='all', 
                       help='要处理的数据集')
    parser.add_argument('--url', type=str, 
                       default='http://host.docker.internal:20401',
                       help='TTS服务URL')
    parser.add_argument('--speed', type=float, default=1.0,
                       help='语速调节系数 (>1.0加速, <1.0减速)')
    parser.add_argument('--test-connection', action='store_true',
                       help='仅测试TTS连接')
    parser.add_argument('--validate-only', action='store_true',
                       help='仅验证已生成的数据')
    parser.add_argument('--base-path', type=str,
                       default='/usr/local/app/jupyterlab/yanjiu/streamllm/experiments/datasets',
                       help='数据集基础路径')
    parser.add_argument('--force-regenerate', action='store_true',
                       help='强制重新生成所有音频文件（删除已有文件）')
    
    args = parser.parse_args()
    
    # 创建处理器
    processor = TTSFilterDatasetProcessor(
        base_path=args.base_path,
        tts_url=args.url,
        force_regenerate=args.force_regenerate
    )
    
    # 验证路径
    if not processor.validate_paths():
        print("路径验证失败，退出")
        return
    
    # 仅测试连接
    if args.test_connection:
        processor.test_tts_connection()
        return
    
    # 仅验证数据
    if args.validate_only:
        processor.validate_generated_data()
        processor.generate_statistics_report()
        return
    
    # 测试TTS连接
    if not processor.test_tts_connection():
        print("TTS连接测试失败，请检查服务后重试")
        return
    
    print(f"\n开始TTS数据生成，语速系数: {args.speed}")
    if args.force_regenerate:
        print("⚠️  强制重新生成模式：将删除所有已有的音频文件")
    
    # 处理指定数据集
    if args.dataset == 'all':
        datasets_to_process = processor.datasets
    else:
        datasets_to_process = [args.dataset]
    
    success_datasets = []
    
    for dataset in datasets_to_process:
        if processor.process_dataset(dataset, speed_factor=args.speed):
            success_datasets.append(dataset)
    
    # 清理空目录
    processor.cleanup_empty_directories()
    
    # 验证生成的数据
    processor.validate_generated_data()
    
    # 生成统计报告
    processor.generate_statistics_report()
    
    print(f"\n处理完成!")
    print(f"成功处理的数据集: {success_datasets}")
    
    if success_datasets:
        print(f"\n✅ TTS数据已生成到: {processor.tts_dataset_path}")
        print("可以开始使用这些数据进行实验了。")
    else:
        print("\n❌ 未成功处理任何数据集")


if __name__ == "__main__":
    main()