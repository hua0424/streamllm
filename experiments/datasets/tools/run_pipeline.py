#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据处理管线

阶段1：数据预处理 - 选取文本最长的对话，转换为累积对话格式的JSON文件
阶段2：TTS生成 - 批量调用TTS服务生成音频文件
阶段3：更新元数据 - 读取音频时长，更新到JSON文件中

使用方式（在项目根目录下运行）:
    uv run python -m experiments.datasets.tools.run_pipeline [参数]
"""

import argparse
import json
import wave
import sys
from pathlib import Path

# 获取项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # tools -> datasets -> experiments -> streamllm
sys.path.insert(0, str(PROJECT_ROOT))

# 导入处理器
from experiments.datasets.tools.data_processor import process_crosswoz, process_multiwoz
from experiments.datasets.tools.tts import TTSClient, BatchTTSProcessor


# 默认路径配置（相对于项目根目录）
DEFAULT_PATHS = {
    "crosswoz_train": "experiments/datasets/raw_data/CrossWOZ/train.json",
    "crosswoz_val": "experiments/datasets/raw_data/CrossWOZ/val.json",
    "crosswoz_test": "experiments/datasets/raw_data/CrossWOZ/test.json",
    "multiwoz_data": "experiments/datasets/raw_data/MultiWOZ/MultiWOZ_2.1/data.json",
    "output_base": "experiments/datasets/processed",
}

# 默认文本长度限制（字符数）
# 中文语速约 4-5 字/秒，英文约 13-14 字符/秒
# 150秒音频对应：中文 ~720字，英文 ~2050字符
DEFAULT_MAX_TEXT_LENGTH_ZH = 720   # 中文：约对应 150 秒音频
DEFAULT_MAX_TEXT_LENGTH_EN = 2050  # 英文：约对应 150 秒音频


def get_audio_duration(audio_path: Path) -> float:
    """
    获取音频文件时长（秒）
    
    Args:
        audio_path: 音频文件路径
    
    Returns:
        音频时长（秒），失败返回 -1
    """
    try:
        with wave.open(str(audio_path), 'rb') as wav_file:
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
            duration = frames / sample_rate
            return round(duration, 3)
    except Exception as e:
        print(f"  警告：无法读取音频时长 {audio_path}: {e}")
        return -1


def run_phase_1_preprocess(args, project_root: Path) -> Path:
    """
    阶段一：数据预处理
    
    选取每个数据集中文本最长的对话，转换为累积对话格式的JSON文件
    
    Returns:
        处理后的JSON文件目录路径
    """
    print("\n" + "=" * 60)
    print("阶段 1：数据预处理 (Data Preprocessing)")
    print("=" * 60)
    print(f"策略：选取每个数据集中文本最长的 {args.top_n_dialogs} 个对话")
    print(f"文本长度限制：")
    print(f"  中文 (CrossWOZ): {args.max_text_length_zh if args.max_text_length_zh else '不限制'} 字符")
    print(f"  英文 (MultiWOZ): {args.max_text_length_en if args.max_text_length_en else '不限制'} 字符")
    
    output_base = project_root / args.output_dir
    processed_json_dir = output_base / "json"
    
    total_samples = 0
    
    # 处理 CrossWOZ (中文)
    if args.dataset in ['crosswoz', 'all']:
        crosswoz_file = project_root / args.crosswoz_path
        if crosswoz_file.exists():
            crosswoz_output = processed_json_dir / "crosswoz"
            count = process_crosswoz(
                str(crosswoz_file),
                str(crosswoz_output),
                top_n_dialogs=args.top_n_dialogs,
                max_samples_per_dialog=args.max_samples_per_dialog,
                max_text_length=args.max_text_length_zh
            )
            total_samples += count
        else:
            print(f"警告：CrossWOZ 文件不存在: {crosswoz_file}")
    
    # 处理 MultiWOZ (英文)
    if args.dataset in ['multiwoz', 'all']:
        multiwoz_file = project_root / args.multiwoz_path
        if multiwoz_file.exists():
            multiwoz_output = processed_json_dir / "multiwoz"
            count = process_multiwoz(
                str(multiwoz_file),
                str(multiwoz_output),
                top_n_dialogs=args.top_n_dialogs,
                max_samples_per_dialog=args.max_samples_per_dialog,
                max_text_length=args.max_text_length_en
            )
            total_samples += count
        else:
            print(f"警告：MultiWOZ 文件不存在: {multiwoz_file}")
    
    print(f"\n阶段 1 完成：共生成 {total_samples} 个任务文件")
    print(f"输出目录: {processed_json_dir}")
    
    return processed_json_dir


def run_phase_2_tts(args, json_input_dir: Path, project_root: Path) -> Path:
    """
    阶段二：TTS 批量生成
    
    读取阶段1生成的JSON文件，调用TTS服务生成音频
    
    Returns:
        音频输出目录路径
    """
    print("\n" + "=" * 60)
    print("阶段 2：TTS 批量生成 (TTS Batch Processing)")
    print("=" * 60)
    
    # 测试 TTS 服务连接
    client = TTSClient(args.tts_url)
    print(f"\n测试 TTS 服务连接: {args.tts_url}")
    if not client.test_connection():
        print("错误：无法连接到 TTS 服务，请确保服务已启动")
        print("提示：可以使用 --skip-tts 参数跳过 TTS 阶段")
        return None
    
    processor = BatchTTSProcessor(client, max_workers=args.tts_workers)
    
    output_base = project_root / args.output_dir
    audio_output_base = output_base / "audio"
    
    # 处理各个数据集
    datasets_to_process = []
    if args.dataset in ['crosswoz', 'all']:
        datasets_to_process.append("crosswoz")
    if args.dataset in ['multiwoz', 'all']:
        datasets_to_process.append("multiwoz")
    
    for dataset in datasets_to_process:
        json_dir = json_input_dir / dataset
        if not json_dir.exists():
            print(f"跳过 {dataset}：JSON目录不存在 ({json_dir})")
            continue
        
        audio_output_dir = audio_output_base / dataset
        
        print(f"\n处理数据集: {dataset}")
        print(f"  输入目录: {json_dir}")
        print(f"  输出目录: {audio_output_dir}")
        
        # 调用批量处理
        processor.process_json_files(
            input_dir=str(json_dir),
            output_dir=str(audio_output_dir),
            speed_factor=args.tts_speed
        )
    
    print(f"\n阶段 2 完成")
    print(f"音频输出目录: {audio_output_base}")
    
    return audio_output_base


def run_phase_3_update_duration(args, json_dir: Path, audio_dir: Path, project_root: Path):
    """
    阶段三：更新音频时长
    
    读取音频文件时长，更新到对应的JSON文件中
    """
    print("\n" + "=" * 60)
    print("阶段 3：更新音频时长 (Update Audio Duration)")
    print("=" * 60)
    
    # 处理各个数据集
    datasets_to_process = []
    if args.dataset in ['crosswoz', 'all']:
        datasets_to_process.append("crosswoz")
    if args.dataset in ['multiwoz', 'all']:
        datasets_to_process.append("multiwoz")
    
    total_updated = 0
    total_missing = 0
    
    for dataset in datasets_to_process:
        json_dataset_dir = json_dir / dataset
        audio_dataset_dir = audio_dir / dataset
        
        if not json_dataset_dir.exists():
            print(f"跳过 {dataset}：JSON目录不存在")
            continue
        
        if not audio_dataset_dir.exists():
            print(f"跳过 {dataset}：音频目录不存在")
            continue
        
        print(f"\n更新数据集: {dataset}")
        
        # 遍历所有 JSON 文件
        json_files = list(json_dataset_dir.glob("*.json"))
        updated = 0
        missing = 0
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                audio_filename = data.get('audio_file', '')
                audio_path = audio_dataset_dir / audio_filename
                
                if audio_path.exists():
                    duration = get_audio_duration(audio_path)
                    data['audio_duration'] = duration
                    
                    with open(json_file, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    
                    updated += 1
                else:
                    missing += 1
                    
            except Exception as e:
                print(f"  错误处理 {json_file}: {e}")
        
        print(f"  更新: {updated} 个文件")
        if missing > 0:
            print(f"  缺少音频: {missing} 个文件")
        
        total_updated += updated
        total_missing += missing
    
    print(f"\n阶段 3 完成：更新 {total_updated} 个文件，缺少音频 {total_missing} 个")


def main():
    parser = argparse.ArgumentParser(
        description='实验数据处理管线',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 仅预处理数据（生成JSON）
  uv run python -m experiments.datasets.tools.run_pipeline --skip-tts
  
  # 完整管线（预处理 + TTS + 更新时长）
  uv run python -m experiments.datasets.tools.run_pipeline --tts-url http://localhost:20401
  
  # 使用8个并发加速TTS生成
  uv run python -m experiments.datasets.tools.run_pipeline --tts-workers 8
  
  # 选取文本最长的50个对话
  uv run python -m experiments.datasets.tools.run_pipeline --top-n-dialogs 50 --skip-tts
  
  # 设置文本长度限制（过滤过长数据）
  uv run python -m experiments.datasets.tools.run_pipeline --max-text-length-zh 300 --max-text-length-en 800 --skip-tts
  
  # 跳过预处理，直接进行TTS
  uv run python -m experiments.datasets.tools.run_pipeline --skip-preprocess
  
  # 仅更新音频时长（已完成TTS）
  uv run python -m experiments.datasets.tools.run_pipeline --skip-preprocess --skip-tts
        """
    )
    
    # 数据集选择
    parser.add_argument('--dataset', choices=['crosswoz', 'multiwoz', 'all'],
                        default='all', help='要处理的数据集 (默认: all)')
    
    # 输入路径
    parser.add_argument('--crosswoz-path', 
                        default=DEFAULT_PATHS["crosswoz_train"],
                        help=f'CrossWOZ 数据文件路径 (默认: {DEFAULT_PATHS["crosswoz_train"]})')
    parser.add_argument('--multiwoz-path',
                        default=DEFAULT_PATHS["multiwoz_data"],
                        help=f'MultiWOZ 数据文件路径 (默认: {DEFAULT_PATHS["multiwoz_data"]})')
    
    # 输出路径
    parser.add_argument('--output-dir',
                        default=DEFAULT_PATHS["output_base"],
                        help=f'输出基础目录 (默认: {DEFAULT_PATHS["output_base"]})')
    
    # 数量控制
    parser.add_argument('--top-n-dialogs', type=int, default=100,
                        help='选取文本最长的前N个对话 (默认: 100)')
    parser.add_argument('--max-samples-per-dialog', type=int, default=None,
                        help='每个对话最多生成的样本数 (默认: 不限制)')
    
    # 文本长度限制（中英文分开设置）
    parser.add_argument('--max-text-length-zh', type=int, default=DEFAULT_MAX_TEXT_LENGTH_ZH,
                        help=f'中文文本最大字符数，超过则跳过 (默认: {DEFAULT_MAX_TEXT_LENGTH_ZH}，约150秒音频)')
    parser.add_argument('--max-text-length-en', type=int, default=DEFAULT_MAX_TEXT_LENGTH_EN,
                        help=f'英文文本最大字符数，超过则跳过 (默认: {DEFAULT_MAX_TEXT_LENGTH_EN}，约150秒音频)')
    
    # TTS 参数
    parser.add_argument('--tts-url', default='http://host.docker.internal:20401',
                        help='TTS 服务 URL (默认: http://host.docker.internal:20401)')
    parser.add_argument('--tts-speed', type=float, default=0.8,
                        help='TTS 语速系数 (默认: 0.8)')
    parser.add_argument('--tts-workers', type=int, default=4,
                        help='TTS 并发数 (默认: 4，可根据GPU显存调整)')
    
    # 阶段控制
    parser.add_argument('--skip-preprocess', action='store_true',
                        help='跳过阶段1（数据预处理），使用已有JSON文件')
    parser.add_argument('--skip-tts', action='store_true',
                        help='跳过阶段2（TTS生成）')
    
    args = parser.parse_args()
    
    # 确定项目根目录
    project_root = PROJECT_ROOT
    print(f"项目根目录: {project_root}")
    
    # 确定目录路径
    output_base = project_root / args.output_dir
    json_dir = output_base / "json"
    audio_dir = output_base / "audio"
    
    # 阶段1：数据预处理
    if not args.skip_preprocess:
        json_dir = run_phase_1_preprocess(args, project_root)
    else:
        print("\n跳过阶段1（数据预处理）")
        if not json_dir.exists():
            print(f"错误：JSON目录不存在: {json_dir}")
            print("请先运行数据预处理，或检查 --output-dir 参数")
            sys.exit(1)
    
    # 阶段2：TTS 生成
    if not args.skip_tts:
        audio_dir = run_phase_2_tts(args, json_dir, project_root)
        if audio_dir is None:
            print("TTS 阶段失败，退出")
            sys.exit(1)
    else:
        print("\n跳过阶段2（TTS生成）")
    
    # 阶段3：更新音频时长
    if audio_dir and audio_dir.exists():
        run_phase_3_update_duration(args, json_dir, audio_dir, project_root)
    else:
        print("\n跳过阶段3（更新音频时长）：音频目录不存在")
    
    print("\n" + "=" * 60)
    print("管线执行完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
