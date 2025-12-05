#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据集处理器

将 MultiWOZ / CrossWOZ 数据集处理成累积对话格式，用于实验。

累积对话逻辑：
轮次1 (用户)  -> 输出文本: a1
轮次2 (系统)  -> 跳过（只在下一轮用户发言时累积）
轮次3 (用户)  -> 输出文本: a1 + b1 + a2
轮次4 (系统)  -> 跳过
轮次5 (用户)  -> 输出文本: a1 + b1 + a2 + b2 + a3
...

这样可以模拟用户在多轮对话中提供越来越长的上下文输入。
"""

import json
import shutil
from pathlib import Path
from typing import Optional


def clear_output_dir(directory: Path):
    """如果目录存在则清空，否则创建"""
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)


def process_crosswoz(
    input_file: str, 
    output_dir: str, 
    max_dialogs: Optional[int] = None,
    max_samples_per_dialog: Optional[int] = None
) -> int:
    """
    处理 CrossWOZ 数据集 (中文)
    
    CrossWOZ 格式: {dialog_id: {messages: [{content: ..., role: usr/sys}, ...]}}
    
    Args:
        input_file: 输入JSON文件路径
        output_dir: 输出目录
        max_dialogs: 最大处理对话数（None表示不限制）
        max_samples_per_dialog: 每个对话最多生成的样本数（None表示不限制）
    
    Returns:
        生成的任务文件数量
    """
    print(f"\n正在处理 CrossWOZ: {input_file} ...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_count = 0
    dialog_count = 0
    
    for dialog_id, content in data.items():
        if max_dialogs and dialog_count >= max_dialogs:
            break
            
        messages = content.get('messages', [])
        if not messages:
            continue
        
        dialog_count += 1
        
        # 累积对话历史
        accumulated_text = ""
        user_turn_index = 0  # 用户轮次计数
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
            
            # 只在用户轮次生成样本（模拟用户说完话后系统响应的场景）
            if role == 'usr':
                user_turn_index += 1
                
                if max_samples_per_dialog and sample_count_in_dialog >= max_samples_per_dialog:
                    continue
                
                sample_id = f"crosswoz_{dialog_id}_turn{user_turn_index}"
                
                task_data = {
                    "sample_id": sample_id,
                    "dialog_id": str(dialog_id),
                    "turn_index": user_turn_index,
                    "text": accumulated_text,
                    "audio_file": f"{sample_id}.wav",
                    "language": "zh",
                    "dataset": "crosswoz"
                }

                output_file = output_path / f"{sample_id}.json"
                with open(output_file, 'w', encoding='utf-8') as tf:
                    json.dump(task_data, tf, ensure_ascii=False, indent=2)
                
                total_count += 1
                sample_count_in_dialog += 1

    print(f"CrossWOZ 处理完成：{dialog_count} 个对话，生成 {total_count} 个任务文件")
    return total_count


def process_multiwoz(
    input_file: str, 
    output_dir: str, 
    max_dialogs: Optional[int] = None,
    max_samples_per_dialog: Optional[int] = None
) -> int:
    """
    处理 MultiWOZ 数据集 (英文)
    
    MultiWOZ 格式: {dialog_id: {log: [{text: ...}, ...]}}
    log 数组中偶数索引(0,2,4...)是用户，奇数索引(1,3,5...)是系统
    
    Args:
        input_file: 输入JSON文件路径
        output_dir: 输出目录
        max_dialogs: 最大处理对话数（None表示不限制）
        max_samples_per_dialog: 每个对话最多生成的样本数（None表示不限制）
    
    Returns:
        生成的任务文件数量
    """
    print(f"\n正在处理 MultiWOZ: {input_file} ...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_count = 0
    dialog_count = 0
    
    for dialog_id, content in data.items():
        if max_dialogs and dialog_count >= max_dialogs:
            break
            
        logs = content.get('log', [])
        if not logs:
            continue
        
        dialog_count += 1
        
        # 清理 dialog_id (移除 .json 后缀)
        clean_dialog_id = dialog_id.replace('.json', '')
        
        # 累积对话历史
        accumulated_text = ""
        user_turn_index = 0  # 用户轮次计数
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
                
                if max_samples_per_dialog and sample_count_in_dialog >= max_samples_per_dialog:
                    continue
                
                sample_id = f"multiwoz_{clean_dialog_id}_turn{user_turn_index}"
                
                task_data = {
                    "sample_id": sample_id,
                    "dialog_id": clean_dialog_id,
                    "turn_index": user_turn_index,
                    "text": accumulated_text,
                    "audio_file": f"{sample_id}.wav",
                    "language": "en",
                    "dataset": "multiwoz"
                }

                output_file = output_path / f"{sample_id}.json"
                with open(output_file, 'w', encoding='utf-8') as tf:
                    json.dump(task_data, tf, ensure_ascii=False, indent=2)
                
                total_count += 1
                sample_count_in_dialog += 1

    print(f"MultiWOZ 处理完成：{dialog_count} 个对话，生成 {total_count} 个任务文件")
    return total_count


def main():
    """命令行入口（用于独立测试）"""
    import argparse
    
    parser = argparse.ArgumentParser(description='数据集预处理工具')
    parser.add_argument('--dataset', choices=['crosswoz', 'multiwoz', 'all'], 
                        default='all', help='要处理的数据集')
    parser.add_argument('--input-file', help='输入文件路径')
    parser.add_argument('--output-dir', required=True, help='输出目录')
    parser.add_argument('--max-dialogs', type=int, default=None, 
                        help='最大处理对话数')
    parser.add_argument('--max-samples-per-dialog', type=int, default=None,
                        help='每个对话最多生成的样本数')
    
    args = parser.parse_args()
    
    if args.dataset in ['crosswoz', 'all'] and args.input_file:
        process_crosswoz(
            args.input_file,
            args.output_dir,
            max_dialogs=args.max_dialogs,
            max_samples_per_dialog=args.max_samples_per_dialog
        )
    
    if args.dataset in ['multiwoz', 'all'] and args.input_file:
        process_multiwoz(
            args.input_file,
            args.output_dir,
            max_dialogs=args.max_dialogs,
            max_samples_per_dialog=args.max_samples_per_dialog
        )


if __name__ == "__main__":
    main()
