# src/utils/wav2stream_example.py

"""
使用wav2stream函数进行流式语音识别的示例
"""

import time
from src.utils.audio2stream import wav2stream
from src.asr.faster_whisper_streamer import StreamingASRProcessor

def process_wav_with_streaming_asr(
    wav_path: str,
    chunk_duration: float = 0.5,
    model_size: str = 'base',
    recognition_threshold: float = 3.0
):
    """
    使用wav2stream函数和StreamingASRProcessor进行流式语音识别
    
    Args:
        wav_path: WAV文件路径
        chunk_duration: 音频块时长(秒)
        model_size: ASR模型大小
        recognition_threshold: 识别阈值(秒)
    """
    
    def text_callback(texts, start_time, end_time):
        """文本输出回调函数"""
        if texts:
            for text in texts:
                print(f"[实时输出] [{text.start_time:.2f}s-{text.end_time:.2f}s] {text.text}")
    
    # 创建流式ASR处理器
    print(f"正在初始化ASR处理器...")
    processor = StreamingASRProcessor(
        model_size=model_size,
        recognition_threshold=recognition_threshold,
        text_callback=text_callback
    )
    
    # 创建音频流生成器
    print(f"正在创建音频流生成器...")
    chunk_generator = wav2stream(wav_path, chunk_duration)
    
    print(f"\n开始流式处理...")
    print("=" * 80)
    
    start_time = time.time()
    all_new_texts = []
    
    # 逐个处理音频块
    for chunk_idx, audio_chunk in enumerate(chunk_generator, 1):
        print(f"\n[块 {chunk_idx}] 处理音频块 [{audio_chunk.start_time:.2f}s-{audio_chunk.end_time:.2f}s]")
        
        # 将音频块添加到ASR处理器
        new_texts = processor.add_audio_chunk(
            audio_chunk.data, 
            is_stream_finished=audio_chunk.is_stream_finished
        )
        
        # 收集新输出的文本
        all_new_texts.extend(new_texts)
        
        # 如果是最后一个块，跳出循环
        if audio_chunk.is_stream_finished:
            print(f"[块 {chunk_idx}] 流式处理完成")
            break
    
    # 处理完成
    process_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("流式语音识别完成!")
    print(f"处理时间: {process_time:.2f}s")
    print(f"输出文本段数: {len(all_new_texts)}")
    
    # 获取完整文本
    final_text = processor.get_final_text()
    print(f"\n完整识别结果:")
    print("-" * 80)
    print(final_text)
    
    # 获取所有文本输出详情
    all_outputs = processor.get_all_text_outputs()
    print(f"\n详细文本输出 (共{len(all_outputs)}段):")
    print("-" * 80)
    for i, output in enumerate(all_outputs, 1):
        print(f"{i:2d}. [{output.start_time:.2f}s-{output.end_time:.2f}s] {output.text}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='使用wav2stream进行流式语音识别')
    parser.add_argument('--wav_path', type=str, required=True, help='WAV文件路径')
    parser.add_argument('--chunk_duration', type=float, default=0.5, help='音频块时长(秒)')
    parser.add_argument('--model_size', type=str, default='base', help='ASR模型大小')
    parser.add_argument('--threshold', type=float, default=3.0, help='识别阈值(秒)')
    parser.add_argument('--fast_mode', action='store_true', help='快速模式')
    
    args = parser.parse_args()
    
    # 快速模式设置
    if args.fast_mode:
        args.model_size = 'tiny'
        args.threshold = 2.0
        print("启用快速模式: 使用tiny模型，阈值2.0s")
    
    print("wav2stream + StreamingASR 示例")
    print(f"WAV文件: {args.wav_path}")
    print(f"音频块时长: {args.chunk_duration}s")
    print(f"ASR模型: {args.model_size}")
    print(f"识别阈值: {args.threshold}s")
    
    try:
        process_wav_with_streaming_asr(
            wav_path=args.wav_path,
            chunk_duration=args.chunk_duration,
            model_size=args.model_size,
            recognition_threshold=args.threshold
        )
    except Exception as e:
        print(f"处理失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 