"""
流式音频分段器使用示例
展示如何使用并发API处理多个音频流
"""

from streamaudio_segmenter import StreamAudioSegmenter, StreamState
import numpy as np


def example_single_stream():
    """单个流处理示例"""
    print("=== 单个流处理示例 ===")
    
    # 创建分段器
    segmenter = StreamAudioSegmenter(
        sampling_rate=16000,
        silence_threshold=0.5,
        min_speech_duration_ms=250,
        min_silence_duration_ms=300
    )
    
    # 创建流状态
    state = segmenter.create_state()
    
    # 模拟音频数据块（这里用随机数据演示）
    for i in range(10):
        # 生成100ms的音频块（16000Hz * 0.1s = 1600 samples）
        audio_chunk = np.random.randn(1600).astype(np.float32) * 0.1
        
        # 处理音频块
        segment, state, metadata = segmenter.process_audio(audio_chunk, state)
        
        if segment is not None:
            print(f"  检测到语音段: {len(segment)} samples")
        else:
            print(f"  处理块 {i+1}: 累积 {metadata['accumulated_samples']} samples")
    
    # 清空剩余数据
    remaining, state = segmenter.flush(state)
    if remaining is not None:
        print(f"  清空剩余: {len(remaining)} samples")
    
    print()


def example_concurrent_streams():
    """并发处理多个流示例"""
    print("=== 并发处理多个流示例 ===")
    
    # 创建单个分段器实例
    segmenter = StreamAudioSegmenter(sampling_rate=16000)
    
    # 为每个流创建独立状态
    streams = {
        'stream1': segmenter.create_state(),
        'stream2': segmenter.create_state(),
        'stream3': segmenter.create_state()
    }
    
    # 模拟并发处理
    for iteration in range(5):
        print(f"\n迭代 {iteration+1}:")
        
        for stream_id, state in streams.items():
            # 生成不同的音频数据块
            audio_chunk = np.random.randn(1600).astype(np.float32) * 0.1
            
            # 处理音频块
            segment, new_state, metadata = segmenter.process_audio(audio_chunk, state)
            
            # 更新状态
            streams[stream_id] = new_state
            
            if segment is not None:
                print(f"  [{stream_id}] 检测到语音段: {len(segment)} samples")
            else:
                print(f"  [{stream_id}] 累积: {metadata['accumulated_samples']} samples, "
                      f"说话中: {metadata['is_speaking']}")
    
    # 清空所有流
    print("\n清空所有流:")
    for stream_id, state in streams.items():
        remaining, final_state = segmenter.flush(state)
        streams[stream_id] = final_state
        if remaining is not None:
            print(f"  [{stream_id}] 清空: {len(remaining)} samples")
    
    print()


def example_state_management():
    """状态管理示例"""
    print("=== 状态管理示例 ===")
    
    # 创建分段器
    segmenter = StreamAudioSegmenter()
    
    # 创建状态
    state = segmenter.create_state()
    print(f"初始状态: 累积音频长度={len(state.accumulated_audio)}, 说话中={state.is_speaking}")
    
    # 处理一些数据
    audio_chunk = np.random.randn(1600).astype(np.float32) * 0.1
    _, state, _ = segmenter.process_audio(audio_chunk, state)
    print(f"处理后: 累积音频长度={len(state.accumulated_audio)}")
    
    # 创建状态副本
    state_copy = state.copy()
    print(f"状态副本: 累积音频长度={len(state_copy.accumulated_audio)}")
    
    # 重置状态
    state.reset()
    print(f"重置后: 累积音频长度={len(state.accumulated_audio)}")
    print(f"副本不受影响: 累积音频长度={len(state_copy.accumulated_audio)}")
    
    print()


if __name__ == "__main__":
    # 运行示例
    example_single_stream()
    example_concurrent_streams()
    example_state_management()
    
    print("✅ 所有示例运行完成！")