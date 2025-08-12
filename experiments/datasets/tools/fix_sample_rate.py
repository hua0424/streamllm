#!/usr/bin/env python3
"""
修复音频采样率不一致的问题
将所有音频统一转换为16000Hz
"""

import os
import sys
from pathlib import Path
import shutil
import librosa
import soundfile as sf
import json
from typing import Optional

def convert_audio_sample_rate(input_path: Path, output_path: Path, target_sr: int = 16000) -> bool:
    """
    转换音频文件的采样率
    
    Args:
        input_path: 输入音频文件路径
        output_path: 输出音频文件路径
        target_sr: 目标采样率（默认16000Hz）
    
    Returns:
        是否转换成功
    """
    try:
        # 加载音频（保持原始采样率）
        y, sr = librosa.load(str(input_path), sr=None)
        
        # 如果采样率已经正确，直接复制
        if sr == target_sr:
            if input_path != output_path:
                shutil.copy2(input_path, output_path)
            return True
        
        # 重采样到目标采样率
        print(f"  转换采样率: {sr}Hz -> {target_sr}Hz")
        y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        
        # 保存转换后的音频
        sf.write(str(output_path), y_resampled, target_sr)
        return True
        
    except Exception as e:
        print(f"  ❌ 转换失败: {e}")
        return False

def fix_experiment_audio_sample_rates(base_path: str = None):
    """
    修复实验数据中的音频采样率问题
    """
    if base_path is None:
        base_path = Path(__file__).parent.parent / "processed" / "experiments"
    else:
        base_path = Path(base_path)
    
    print("=" * 60)
    print("音频采样率修复工具")
    print("=" * 60)
    print(f"基础路径: {base_path}")
    print(f"目标采样率: 16000Hz")
    print()
    
    # 统计信息
    total_files = 0
    converted_files = 0
    already_correct = 0
    failed_files = 0
    
    # 遍历所有实验目录
    experiments = ["core_comparison", "length_analysis", "asr_context", "ablation_study", "case_analysis"]
    
    for exp_name in experiments:
        exp_path = base_path / exp_name / "audio"
        if not exp_path.exists():
            print(f"⚠️ 实验目录不存在: {exp_path}")
            continue
        
        print(f"\n处理实验: {exp_name}")
        print("-" * 40)
        
        # 遍历所有音频文件
        for audio_file in exp_path.rglob("*.wav"):
            total_files += 1
            
            # 检查当前采样率
            try:
                y, sr = librosa.load(str(audio_file), sr=None)
                
                if sr == 16000:
                    already_correct += 1
                    print(f"  ✅ {audio_file.relative_to(exp_path)}: 已是16000Hz")
                elif sr == 22050:
                    # 需要转换
                    print(f"  🔄 {audio_file.relative_to(exp_path)}: {sr}Hz -> 16000Hz")
                    
                    # 先保存到临时文件
                    temp_file = audio_file.with_suffix('.tmp.wav')
                    if convert_audio_sample_rate(audio_file, temp_file, 16000):
                        # 替换原文件
                        shutil.move(temp_file, audio_file)
                        converted_files += 1
                        print(f"     转换成功")
                    else:
                        failed_files += 1
                        if temp_file.exists():
                            temp_file.unlink()
                else:
                    print(f"  ⚠️ {audio_file.relative_to(exp_path)}: 未知采样率 {sr}Hz")
                    failed_files += 1
                    
            except Exception as e:
                print(f"  ❌ {audio_file.relative_to(exp_path)}: 处理失败 - {e}")
                failed_files += 1
    
    # 打印统计结果
    print("\n" + "=" * 60)
    print("处理完成")
    print("=" * 60)
    print(f"总文件数: {total_files}")
    print(f"已是16000Hz: {already_correct}")
    print(f"成功转换: {converted_files}")
    print(f"处理失败: {failed_files}")
    
    if converted_files > 0:
        print(f"\n✅ 成功将 {converted_files} 个文件转换为16000Hz")
    if failed_files > 0:
        print(f"\n⚠️ 有 {failed_files} 个文件处理失败，请检查")

def fix_tts_source_audio(tts_base_path: str = None):
    """
    修复TTS源目录中的音频采样率
    """
    if tts_base_path is None:
        tts_base_path = Path(__file__).parent.parent / "processed" / "tts_dataset"
    else:
        tts_base_path = Path(tts_base_path)
    
    print("\n" + "=" * 60)
    print("修复TTS源音频采样率")
    print("=" * 60)
    print(f"TTS目录: {tts_base_path}")
    
    total = 0
    converted = 0
    
    for dataset in ["dailydialog", "lccc"]:
        dataset_path = tts_base_path / dataset
        if not dataset_path.exists():
            print(f"⚠️ 数据集目录不存在: {dataset_path}")
            continue
        
        print(f"\n处理数据集: {dataset}")
        
        for audio_file in dataset_path.rglob("*.wav"):
            total += 1
            try:
                y, sr = librosa.load(str(audio_file), sr=None)
                
                if sr == 22050:
                    print(f"  转换: {audio_file.name}")
                    # 转换为16000Hz
                    y_resampled = librosa.resample(y, orig_sr=sr, target_sr=16000)
                    
                    # 保存到临时文件
                    temp_file = audio_file.with_suffix('.tmp.wav')
                    sf.write(str(temp_file), y_resampled, 16000)
                    
                    # 替换原文件
                    shutil.move(temp_file, audio_file)
                    converted += 1
                    
            except Exception as e:
                print(f"  ❌ 处理失败 {audio_file.name}: {e}")
    
    print(f"\n转换完成: {converted}/{total} 个文件")

def verify_sample_rates(base_path: str = None):
    """
    验证所有音频文件的采样率
    """
    if base_path is None:
        base_path = Path(__file__).parent.parent / "processed"
    else:
        base_path = Path(base_path)
    
    print("\n" + "=" * 60)
    print("验证音频采样率")
    print("=" * 60)
    
    # 检查所有相关目录
    dirs_to_check = [
        base_path / "filter_dataset" / "CS-Dialogue",
        base_path / "tts_dataset" / "dailydialog",
        base_path / "tts_dataset" / "lccc",
        base_path / "experiments"
    ]
    
    for dir_path in dirs_to_check:
        if not dir_path.exists():
            continue
        
        print(f"\n{dir_path.relative_to(base_path)}:")
        
        # 统计采样率分布
        sample_rates = {}
        for audio_file in dir_path.rglob("*.wav"):
            try:
                y, sr = librosa.load(str(audio_file), sr=None)
                sample_rates[sr] = sample_rates.get(sr, 0) + 1
            except:
                pass
        
        for sr, count in sorted(sample_rates.items()):
            status = "✅" if sr == 16000 else "⚠️"
            print(f"  {status} {sr}Hz: {count} 个文件")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='修复音频采样率不一致问题')
    parser.add_argument('--verify', action='store_true', help='仅验证采样率，不进行修复')
    parser.add_argument('--fix-tts', action='store_true', help='修复TTS源目录')
    parser.add_argument('--fix-experiments', action='store_true', help='修复实验目录')
    parser.add_argument('--fix-all', action='store_true', help='修复所有目录')
    parser.add_argument('--base-path', help='基础路径')
    
    args = parser.parse_args()
    
    if args.verify:
        verify_sample_rates(args.base_path)
    elif args.fix_tts:
        fix_tts_source_audio(args.base_path)
        verify_sample_rates(args.base_path)
    elif args.fix_experiments:
        fix_experiment_audio_sample_rates(args.base_path)
        verify_sample_rates(args.base_path)
    elif args.fix_all:
        print("修复所有音频文件的采样率...")
        fix_tts_source_audio(args.base_path)
        fix_experiment_audio_sample_rates(args.base_path)
        verify_sample_rates(args.base_path)
    else:
        # 默认只验证
        verify_sample_rates(args.base_path)
        print("\n提示：")
        print("  --verify: 仅验证采样率")
        print("  --fix-tts: 修复TTS源目录")
        print("  --fix-experiments: 修复实验目录")
        print("  --fix-all: 修复所有目录")

if __name__ == "__main__":
    main()