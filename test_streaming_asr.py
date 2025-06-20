#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
流式ASR系统测试脚本
使用WAV音频文件模拟流式输入来验证系统运行正确性
"""

import numpy as np
import time
import librosa
import argparse
import os
from typing import List

# 导入流式ASR处理器
from src.asr.faster_whisper_streamer import StreamingASRProcessor, TranscriptionResult


class StreamingAudioSimulator:
    """模拟流式音频输入"""
    
    def __init__(self, audio_file_path: str, chunk_duration: float = 0.5, sample_rate: int = 16000):
        """
        初始化流式音频模拟器
        
        Args:
            audio_file_path (str): 音频文件路径
            chunk_duration (float): 每个音频块的时长(秒)
            sample_rate (int): 目标采样率
        """
        self.audio_file_path = audio_file_path
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        
        # 加载音频文件
        print(f"正在加载音频文件: {audio_file_path}")
        self.audio_data, self.original_sr = librosa.load(
            audio_file_path, 
            sr=sample_rate, 
            mono=True
        )
        
        self.total_duration = len(self.audio_data) / sample_rate
        self.chunk_size = int(chunk_duration * sample_rate)
        self.current_position = 0
        
        print(f"音频加载完成: 时长 {self.total_duration:.2f}s, 采样率 {sample_rate}Hz")
    
    def get_next_chunk(self) -> tuple[np.ndarray, bool]:
        """
        获取下一个音频块
        
        Returns:
            tuple: (audio_chunk, is_finished)
        """
        if self.current_position >= len(self.audio_data):
            return np.array([], dtype=np.float32), True
        
        end_position = min(self.current_position + self.chunk_size, len(self.audio_data))
        chunk = self.audio_data[self.current_position:end_position]
        
        self.current_position = end_position
        is_finished = (self.current_position >= len(self.audio_data))
        
        return chunk.astype(np.float32), is_finished
    
    def get_current_time(self) -> float:
        """获取当前播放时间"""
        return self.current_position / self.sample_rate


def test_streaming_asr(audio_file_path: str, model_size: str = "base"):
    """测试流式ASR系统"""
    
    if not os.path.exists(audio_file_path):
        print(f"错误: 音频文件不存在: {audio_file_path}")
        return
    
    print("=" * 60)
    print("流式ASR系统测试")
    print("=" * 60)
    
    # 初始化流式ASR处理器
    print("正在初始化流式ASR处理器...")
    asr_processor = StreamingASRProcessor(
        model_size=model_size,
        min_chunk_duration=4.0,  # 最小处理长度3秒
        context_pre_duration=2.0,  # 前置上下文1秒
        context_post_duration=2.0,  # 后置上下文1秒
        vad_aggressiveness=2,  # VAD敏感度
        sample_rate=16000
    )
    
    # 初始化音频模拟器
    print("正在初始化音频模拟器...")
    audio_simulator = StreamingAudioSimulator(
        audio_file_path=audio_file_path,
        chunk_duration=0.5,  # 每0.5秒发送一次音频块
        sample_rate=16000
    )
    
    print("\n开始流式处理...")
    print("-" * 60)
    
    # 模拟流式处理
    chunk_count = 0
    all_new_results: List[TranscriptionResult] = []
    
    start_time = time.time()
    
    while True:
        # 获取下一个音频块
        audio_chunk, is_finished = audio_simulator.get_next_chunk()
        
        if len(audio_chunk) == 0:
            break
        
        chunk_count += 1
        current_time = audio_simulator.get_current_time()
        
        print(f"\n处理音频块 {chunk_count}: {current_time:.2f}s, 长度: {len(audio_chunk)/16000:.2f}s")
        
        # 添加音频块到ASR处理器
        try:
            new_results = asr_processor.add_audio_chunk(audio_chunk)
            
            if new_results:
                print(f"  新转录结果 ({len(new_results)} 条):")
                for result in new_results:
                    print(f"    [{result.start_time:.2f}s -> {result.end_time:.2f}s] {result.text}")
                    all_new_results.extend(new_results)
            else:
                print("  没有新的转录结果")
                
        except Exception as e:
            print(f"  处理音频块时出错: {e}")
            import traceback
            traceback.print_exc()

        if is_finished:
            break
    
    # 处理剩余的音频段
    print("\n音频输入结束，处理剩余音频段...")
    final_results = asr_processor.finish_recording()
    
    if final_results:
        print(f"最终剩余结果 ({len(final_results)} 条):")
        for result in final_results:
            print(f"  [{result.start_time:.2f}s -> {result.end_time:.2f}s] {result.text}")
    
    total_time = time.time() - start_time
    
    # 输出最终结果
    print("\n" + "=" * 60)
    print("流式转录完成")
    print("=" * 60)
    
    final_transcription = asr_processor.get_final_transcription()
    all_results = asr_processor.get_all_results()
    
    print(f"处理时间: {total_time:.2f}s")
    print(f"音频时长: {audio_simulator.total_duration:.2f}s")
    print(f"实时因子: {total_time / audio_simulator.total_duration:.2f}x")
    print(f"总音频块数: {chunk_count}")
    print(f"总转录段数: {len(all_results)}")
    
    print(f"\n完整转录文本:")
    print("-" * 40)
    print(final_transcription)
    print("-" * 40)
    
    print(f"\n详细转录结果:")
    print("-" * 40)
    for i, result in enumerate(all_results, 1):
        print(f"{i:2d}. [{result.start_time:6.2f}s -> {result.end_time:6.2f}s] {result.text}")
    
    # 分析处理统计
    print(f"\n处理统计:")
    print("-" * 40)
    
    # 计算音频段之间的间隔
    if len(all_results) > 1:
        gaps = []
        for i in range(1, len(all_results)):
            gap = all_results[i].start_time - all_results[i-1].end_time
            gaps.append(gap)
        
        if gaps:
            avg_gap = sum(gaps) / len(gaps)
            print(f"平均段间隔: {avg_gap:.2f}s")
            print(f"最大段间隔: {max(gaps):.2f}s")
            print(f"最小段间隔: {min(gaps):.2f}s")
    
    # 计算覆盖率
    total_speech_duration = sum(result.end_time - result.start_time for result in all_results)
    coverage = (total_speech_duration / audio_simulator.total_duration) * 100
    print(f"语音覆盖率: {coverage:.1f}%")
    

def create_test_audio(output_path: str, duration: float = 30.0):
    """创建测试音频文件（合成语音）"""
    from scipy.io.wavfile import write
    
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # 创建多段不同频率的音调，模拟语音段
    audio = np.zeros_like(t)
    
    # 段1: 0-3秒，440Hz（A4）+ 调制
    mask1 = (t >= 0) & (t < 3)
    audio[mask1] = 0.3 * np.sin(2 * np.pi * 440 * t[mask1]) * (1 + 0.1 * np.sin(2 * np.pi * 5 * t[mask1]))
    
    # 静默: 3-4秒
    
    # 段2: 4-8秒，523Hz（C5）+ 调制
    mask2 = (t >= 4) & (t < 8)
    audio[mask2] = 0.3 * np.sin(2 * np.pi * 523 * t[mask2]) * (1 + 0.1 * np.sin(2 * np.pi * 3 * t[mask2]))
    
    # 静默: 8-9秒
    
    # 段3: 9-15秒，659Hz（E5）+ 调制
    mask3 = (t >= 9) & (t < 15)
    audio[mask3] = 0.3 * np.sin(2 * np.pi * 659 * t[mask3]) * (1 + 0.1 * np.sin(2 * np.pi * 4 * t[mask3]))
    
    # 静默: 15-17秒
    
    # 段4: 17-25秒，784Hz（G5）+ 调制
    mask4 = (t >= 17) & (t < 25)
    audio[mask4] = 0.3 * np.sin(2 * np.pi * 784 * t[mask4]) * (1 + 0.1 * np.sin(2 * np.pi * 6 * t[mask4]))
    
    # 段5: 26-30秒，880Hz（A5）+ 调制
    mask5 = (t >= 26) & (t < 30)
    audio[mask5] = 0.3 * np.sin(2 * np.pi * 880 * t[mask5]) * (1 + 0.1 * np.sin(2 * np.pi * 7 * t[mask5]))
    
    # 添加一些噪声使其更真实
    noise = np.random.normal(0, 0.01, len(audio))
    audio += noise
    
    # 归一化并转换为16位整数
    audio = audio / np.max(np.abs(audio)) * 0.8  # 防止削波
    audio_int16 = (audio * 32767).astype(np.int16)
    
    # 保存音频文件
    write(output_path, sample_rate, audio_int16)
    print(f"测试音频文件已创建: {output_path}")
    print(f"音频时长: {duration}s, 包含5个语音段和静默间隔")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="流式ASR系统测试")
    parser.add_argument("--audio", type=str, help="音频文件路径")
    parser.add_argument("--model", type=str, default="base", help="Whisper模型大小")
    parser.add_argument("--create-test", action="store_true", help="创建测试音频文件")
    parser.add_argument("--test-file", type=str, default="test_audio.wav", help="测试音频文件名")
    
    args = parser.parse_args()
    
    try:
        if args.create_test:
            # 创建测试音频文件
            create_test_audio(args.test_file, duration=30.0)
            print(f"\n可以使用以下命令测试:")
            print(f"python test_streaming_asr.py --audio {args.test_file}")
        
        elif args.audio:
            # 测试指定的音频文件
            test_streaming_asr(args.audio, args.model)
        
        else:
            print("请指定音频文件路径或使用 --create-test 创建测试文件")
            print("使用方法:")
            print("  python test_streaming_asr.py --audio your_audio.wav")
            print("  python test_streaming_asr.py --create-test")
            
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc() 