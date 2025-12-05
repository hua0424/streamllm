#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据集处理器

将 MultiWOZ / CrossWOZ 数据集处理成累积对话格式，用于实验。

筛选策略：
1. 计算每个对话的总文本长度（所有轮次累积）
2. 按总文本长度降序排序，取前 N 条对话（默认100条）
3. 对选中的对话进行累积处理，生成 turn1, turn2, turn3...
4. 过滤掉累积文本长度超过上限的样本（中英文分开设置）

累积对话逻辑：
轮次1 (用户)  -> 输出文本: a1
轮次2 (系统)  -> 跳过（只在下一轮用户发言时累积）
轮次3 (用户)  -> 输出文本: a1 + b1 + a2
轮次4 (系统)  -> 跳过
轮次5 (用户)  -> 输出文本: a1 + b1 + a2 + b2 + a3
...
"""

import json
import shutil
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any


def clear_output_dir(directory: Path):
    """如果目录存在则清空，否则创建"""
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)


def calculate_dialog_total_length_crosswoz(messages: List[Dict]) -> int:
    """计算 CrossWOZ 对话的总文本长度"""
    total_length = 0
    for msg in messages:
        text = msg.get('content', '').strip()
        total_length += len(text)
    return total_length


def calculate_dialog_total_length_multiwoz(logs: List[Dict]) -> int:
    """计算 MultiWOZ 对话的总文本长度"""
    total_length = 0
    for turn in logs:
        text = turn.get('text', '').strip()
        total_length += len(text)
    return total_length


def process_crosswoz(
    input_file: str, 
    output_dir: str, 
    top_n_dialogs: int = 100,
    max_samples_per_dialog: Optional[int] = None,
    max_text_length: Optional[int] = None
) -> int:
    """
    处理 CrossWOZ 数据集 (中文)
    
    CrossWOZ 格式: {dialog_id: {messages: [{content: ..., role: usr/sys}, ...]}}
    
    Args:
        input_file: 输入JSON文件路径
        output_dir: 输出目录
        top_n_dialogs: 取文本最长的前N个对话（默认100）
        max_samples_per_dialog: 每个对话最多生成的样本数（None表示不限制）
        max_text_length: 累积文本的最大字符数（超过则跳过，None表示不限制）
    
    Returns:
        生成的任务文件数量
    """
    print(f"\n正在处理 CrossWOZ: {input_file} ...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 第一步：计算每个对话的总文本长度
    dialog_lengths: List[Tuple[str, int, Dict]] = []
    
    for dialog_id, content in data.items():
        messages = content.get('messages', [])
        if not messages:
            continue
        
        total_length = calculate_dialog_total_length_crosswoz(messages)
        dialog_lengths.append((dialog_id, total_length, content))
    
    # 第二步：按总文本长度降序排序，取前 top_n_dialogs 条
    dialog_lengths.sort(key=lambda x: x[1], reverse=True)
    selected_dialogs = dialog_lengths[:top_n_dialogs]
    
    print(f"共 {len(dialog_lengths)} 个对话，选取文本最长的 {len(selected_dialogs)} 个")
    if selected_dialogs:
        print(f"  最长对话文本长度: {selected_dialogs[0][1]} 字符")
        print(f"  最短对话文本长度: {selected_dialogs[-1][1]} 字符")
    
    if max_text_length:
        print(f"  文本长度上限: {max_text_length} 字符")

    # 第三步：对选中的对话进行累积处理
    total_count = 0
    skipped_count = 0
    
    for dialog_id, total_length, content in selected_dialogs:
        messages = content.get('messages', [])
        
        # 累积对话历史
        accumulated_text = ""
        user_turn_index = 0
        sample_count_in_dialog = 0
        
        for idx, msg in enumerate(messages):
            role = msg.get('role', '')
            text = msg.get('content', '').strip()
            
            if not text:
                continue
            
            # 累积所有消息
            if accumulated_text:
                accumulated_text += " " + text
            else:
                accumulated_text = text
            
            # 只在用户轮次生成样本
            if role == 'usr':
                user_turn_index += 1
                
                # 检查样本数限制
                if max_samples_per_dialog and sample_count_in_dialog >= max_samples_per_dialog:
                    continue
                
                # 检查文本长度限制
                if max_text_length and len(accumulated_text) > max_text_length:
                    skipped_count += 1
                    continue
                
                sample_id = f"crosswoz_{dialog_id}_turn{user_turn_index}"
                
                task_data = {
                    "sample_id": sample_id,
                    "dialog_id": str(dialog_id),
                    "turn_index": user_turn_index,
                    "text": accumulated_text,
                    "text_length": len(accumulated_text),
                    "audio_file": f"{sample_id}.wav",
                    "audio_duration": None,  # TTS 后填充
                    "language": "zh",
                    "dataset": "crosswoz"
                }

                output_file = output_path / f"{sample_id}.json"
                with open(output_file, 'w', encoding='utf-8') as tf:
                    json.dump(task_data, tf, ensure_ascii=False, indent=2)
                
                total_count += 1
                sample_count_in_dialog += 1

    print(f"CrossWOZ 处理完成：{len(selected_dialogs)} 个对话，生成 {total_count} 个任务文件")
    if skipped_count > 0:
        print(f"  跳过超长文本: {skipped_count} 个")
    return total_count


def process_multiwoz(
    input_file: str, 
    output_dir: str, 
    top_n_dialogs: int = 100,
    max_samples_per_dialog: Optional[int] = None,
    max_text_length: Optional[int] = None
) -> int:
    """
    处理 MultiWOZ 数据集 (英文)
    
    MultiWOZ 格式: {dialog_id: {log: [{text: ...}, ...]}}
    log 数组中偶数索引(0,2,4...)是用户，奇数索引(1,3,5...)是系统
    
    Args:
        input_file: 输入JSON文件路径
        output_dir: 输出目录
        top_n_dialogs: 取文本最长的前N个对话（默认100）
        max_samples_per_dialog: 每个对话最多生成的样本数（None表示不限制）
        max_text_length: 累积文本的最大字符数（超过则跳过，None表示不限制）
    
    Returns:
        生成的任务文件数量
    """
    print(f"\n正在处理 MultiWOZ: {input_file} ...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 第一步：计算每个对话的总文本长度
    dialog_lengths: List[Tuple[str, int, Dict]] = []
    
    for dialog_id, content in data.items():
        logs = content.get('log', [])
        if not logs:
            continue
        
        total_length = calculate_dialog_total_length_multiwoz(logs)
        dialog_lengths.append((dialog_id, total_length, content))
    
    # 第二步：按总文本长度降序排序，取前 top_n_dialogs 条
    dialog_lengths.sort(key=lambda x: x[1], reverse=True)
    selected_dialogs = dialog_lengths[:top_n_dialogs]
    
    print(f"共 {len(dialog_lengths)} 个对话，选取文本最长的 {len(selected_dialogs)} 个")
    if selected_dialogs:
        print(f"  最长对话文本长度: {selected_dialogs[0][1]} 字符")
        print(f"  最短对话文本长度: {selected_dialogs[-1][1]} 字符")
    
    if max_text_length:
        print(f"  文本长度上限: {max_text_length} 字符")

    # 第三步：对选中的对话进行累积处理
    total_count = 0
    skipped_count = 0
    
    for dialog_id, total_length, content in selected_dialogs:
        logs = content.get('log', [])
        
        # 清理 dialog_id (移除 .json 后缀)
        clean_dialog_id = dialog_id.replace('.json', '')
        
        # 累积对话历史
        accumulated_text = ""
        user_turn_index = 0
        sample_count_in_dialog = 0
        
        for idx, turn in enumerate(logs):
            text = turn.get('text', '').strip()
            
            if not text:
                continue
            
            # 累积所有消息
            if accumulated_text:
                accumulated_text += " " + text
            else:
                accumulated_text = text
            
            # MultiWOZ: 偶数索引是用户，奇数索引是系统
            is_user_turn = (idx % 2 == 0)
            
            # 只在用户轮次生成样本
            if is_user_turn:
                user_turn_index += 1
                
                # 检查样本数限制
                if max_samples_per_dialog and sample_count_in_dialog >= max_samples_per_dialog:
                    continue
                
                # 检查文本长度限制
                if max_text_length and len(accumulated_text) > max_text_length:
                    skipped_count += 1
                    continue
                
                sample_id = f"multiwoz_{clean_dialog_id}_turn{user_turn_index}"
                
                task_data = {
                    "sample_id": sample_id,
                    "dialog_id": clean_dialog_id,
                    "turn_index": user_turn_index,
                    "text": accumulated_text,
                    "text_length": len(accumulated_text),
                    "audio_file": f"{sample_id}.wav",
                    "audio_duration": None,  # TTS 后填充
                    "language": "en",
                    "dataset": "multiwoz"
                }

                output_file = output_path / f"{sample_id}.json"
                with open(output_file, 'w', encoding='utf-8') as tf:
                    json.dump(task_data, tf, ensure_ascii=False, indent=2)
                
                total_count += 1
                sample_count_in_dialog += 1

    print(f"MultiWOZ 处理完成：{len(selected_dialogs)} 个对话，生成 {total_count} 个任务文件")
    if skipped_count > 0:
        print(f"  跳过超长文本: {skipped_count} 个")
    return total_count


def main():
    """命令行入口（用于独立测试）"""
    import argparse
    
    parser = argparse.ArgumentParser(description='数据集预处理工具')
    parser.add_argument('--dataset', choices=['crosswoz', 'multiwoz', 'all'], 
                        default='all', help='要处理的数据集')
    parser.add_argument('--input-file', help='输入文件路径')
    parser.add_argument('--output-dir', required=True, help='输出目录')
    parser.add_argument('--top-n-dialogs', type=int, default=100, 
                        help='取文本最长的前N个对话（默认100）')
    parser.add_argument('--max-samples-per-dialog', type=int, default=None,
                        help='每个对话最多生成的样本数')
    parser.add_argument('--max-text-length', type=int, default=None,
                        help='累积文本的最大字符数（超过则跳过）')
    
    args = parser.parse_args()
    
    if args.dataset in ['crosswoz', 'all'] and args.input_file:
        process_crosswoz(
            args.input_file,
            args.output_dir,
            top_n_dialogs=args.top_n_dialogs,
            max_samples_per_dialog=args.max_samples_per_dialog,
            max_text_length=args.max_text_length
        )
    
    if args.dataset in ['multiwoz', 'all'] and args.input_file:
        process_multiwoz(
            args.input_file,
            args.output_dir,
            top_n_dialogs=args.top_n_dialogs,
            max_samples_per_dialog=args.max_samples_per_dialog,
            max_text_length=args.max_text_length
        )


if __name__ == "__main__":
    main()
