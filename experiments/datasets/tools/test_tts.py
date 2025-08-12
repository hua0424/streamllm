#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TTS工具测试脚本
演示如何使用TTS客户端进行单个和批量转换
"""

import json
from pathlib import Path
from tts import TTSClient, BatchTTSProcessor


def create_test_data():
    """创建测试数据"""
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)
    
    # 创建测试JSON文件
    test_samples = [
        {
            "sample_id": "test_001",
            "text": "Hello, this is a test sentence for TTS.",
            "audio_file": "test_001.wav",
            "language": "en"
        },
        {
            "sample_id": "test_002", 
            "text": "你好，这是一个中文测试句子。",
            "audio_file": "test_002.wav",
            "language": "zh"
        },
        {
            "sample_id": "test_003",
            "text": "This is a longer test sentence to verify the TTS system works correctly with various text lengths.",
            "audio_file": "test_003.wav", 
            "language": "en"
        }
    ]
    
    for sample in test_samples:
        file_path = test_dir / f"{sample['sample_id']}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(sample, f, ensure_ascii=False, indent=2)
    
    print(f"Created {len(test_samples)} test JSON files in {test_dir}")
    return test_dir


def test_single_synthesis():
    """测试单个文本合成"""
    print("\n=== 测试单个文本合成 ===")
    
    # 创建TTS客户端
    client = TTSClient("http://localhost:8000")
    
    # 测试连接
    print("测试连接...")
    if not client.test_connection():
        print("无法连接到TTS服务，请确保服务正在运行")
        return False
    
    # 单个合成示例（需要提供实际的prompt音频文件）
    print("\n要进行单个合成测试，需要:")
    print("1. TTS服务运行在 http://localhost:8000")
    print("2. 提供一个提示音频文件路径")
    print("\n示例命令:")
    print("python tts.py --text 'Hello world' --prompt-wav prompt.wav --output output.wav")
    
    return True


def test_batch_processing():
    """测试批量处理"""
    print("\n=== 测试批量处理 ===")
    
    # 创建测试数据
    test_dir = create_test_data()
    
    # 创建TTS客户端和批量处理器
    client = TTSClient("http://localhost:8000")
    processor = BatchTTSProcessor(client)
    
    print("\n要进行批量处理测试，需要:")
    print("1. TTS服务运行在 http://localhost:8000") 
    print("2. 提供一个提示音频文件路径")
    print(f"3. JSON测试文件已创建在: {test_dir}")
    print("\n示例命令:")
    print(f"python tts.py --batch --input-dir {test_dir} --output-dir output_audio --prompt-wav prompt.wav")
    
    return True


def show_usage_examples():
    """显示使用示例"""
    print("\n=== TTS工具使用示例 ===")
    
    print("\n1. 测试连接:")
    print("python tts.py --url http://your-server:8000 --test")
    
    print("\n2. 单个文本合成:")
    print("python tts.py \\")
    print("  --url http://your-server:8000 \\")
    print("  --text '你好，这是测试文本' \\")
    print("  --prompt-text '你好，我是语音助手' \\") 
    print("  --prompt-wav prompt.wav \\")
    print("  --output test_output.wav")
    
    print("\n3. 批量处理JSON文件:")
    print("python tts.py \\")
    print("  --url http://your-server:8000 \\")
    print("  --batch \\")
    print("  --input-dir ../processed/filter_dataset/dailydialog \\")
    print("  --output-dir ../processed/tts_dataset/dailydialog \\")
    print("  --prompt-text 'Hello, I am a voice assistant.' \\")
    print("  --prompt-wav english_prompt.wav")
    
    print("\n4. 为实验数据生成音频:")
    print("# 英文数据集")
    print("python tts.py --batch \\")
    print("  --input-dir ../processed/filter_dataset/dailydialog \\")
    print("  --output-dir ../processed/tts_dataset/dailydialog \\")
    print("  --prompt-wav english_prompt.wav")
    
    print("\n# 中文数据集") 
    print("python tts.py --batch \\")
    print("  --input-dir ../processed/filter_dataset/lccc \\")
    print("  --output-dir ../processed/tts_dataset/lccc \\")
    print("  --prompt-text '你好，我是语音助手。' \\")
    print("  --prompt-wav chinese_prompt.wav")


def main():
    """主函数"""
    print("TTS客户端工具测试")
    print("=" * 50)
    
    # 显示使用示例
    show_usage_examples()
    
    # 测试功能
    test_single_synthesis()
    test_batch_processing()
    
    print("\n=" * 50)
    print("测试完成！")
    print("\n注意事项:")
    print("1. 确保TTS服务正在运行")
    print("2. 准备合适的提示音频文件")
    print("3. 根据实际服务地址调整URL参数")


if __name__ == "__main__":
    main()