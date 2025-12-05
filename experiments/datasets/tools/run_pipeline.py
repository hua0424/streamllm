#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据处理管线

阶段1：数据预处理 - 将原始数据集转换为累积对话格式的JSON文件
阶段2：TTS生成 - 批量调用TTS服务生成音频文件

使用方式（在项目根目录下运行）:
    uv run python -m experiments.datasets.tools.run_pipeline [参数]
"""

import argparse
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


def run_phase_1_preprocess(args, project_root: Path) -> Path:
    """
    阶段一：数据预处理
    
    将原始数据集转换为累积对话格式的JSON文件
    
    Returns:
        处理后的JSON文件目录路径
    """
    print("\n" + "=" * 60)
    print("阶段 1：数据预处理 (Data Preprocessing)")
    print("=" * 60)
    
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
                max_dialogs=args.max_dialogs,
                max_samples_per_dialog=args.max_samples_per_dialog
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
                max_dialogs=args.max_dialogs,
                max_samples_per_dialog=args.max_samples_per_dialog
            )
            total_samples += count
        else:
            print(f"警告：MultiWOZ 文件不存在: {multiwoz_file}")
    
    print(f"\n阶段 1 完成：共生成 {total_samples} 个任务文件")
    print(f"输出目录: {processed_json_dir}")
    
    return processed_json_dir


def run_phase_2_tts(args, json_input_dir: Path, project_root: Path):
    """
    阶段二：TTS 批量生成
    
    读取阶段1生成的JSON文件，调用TTS服务生成音频
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
        return
    
    processor = BatchTTSProcessor(client)
    
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


def main():
    parser = argparse.ArgumentParser(
        description='实验数据处理管线',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 仅预处理数据（生成JSON）
  uv run python -m experiments.datasets.tools.run_pipeline --skip-tts
  
  # 完整管线（预处理 + TTS）
  uv run python -m experiments.datasets.tools.run_pipeline --tts-url http://localhost:20401
  
  # 仅处理 CrossWOZ 数据集，限制10个对话
  uv run python -m experiments.datasets.tools.run_pipeline --dataset crosswoz --max-dialogs 10
  
  # 跳过预处理，直接进行TTS（使用已有的JSON文件）
  uv run python -m experiments.datasets.tools.run_pipeline --skip-preprocess
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
    parser.add_argument('--max-dialogs', type=int, default=None,
                        help='每个数据集最多处理的对话数 (默认: 不限制)')
    parser.add_argument('--max-samples-per-dialog', type=int, default=None,
                        help='每个对话最多生成的样本数 (默认: 不限制)')
    
    # TTS 参数
    parser.add_argument('--tts-url', default='http://host.docker.internal:20401',
                        help='TTS 服务 URL (默认: http://host.docker.internal:20401)')
    parser.add_argument('--tts-speed', type=float, default=1.0,
                        help='TTS 语速系数 (默认: 1.0)')
    
    # 阶段控制
    parser.add_argument('--skip-preprocess', action='store_true',
                        help='跳过阶段1（数据预处理），直接使用已有JSON文件')
    parser.add_argument('--skip-tts', action='store_true',
                        help='跳过阶段2（TTS生成），仅进行数据预处理')
    
    args = parser.parse_args()
    
    # 确定项目根目录
    project_root = PROJECT_ROOT
    print(f"项目根目录: {project_root}")
    
    # 确定 JSON 输入目录
    output_base = project_root / args.output_dir
    json_dir = output_base / "json"
    
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
        run_phase_2_tts(args, json_dir, project_root)
    else:
        print("\n跳过阶段2（TTS生成）")
    
    print("\n" + "=" * 60)
    print("管线执行完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
