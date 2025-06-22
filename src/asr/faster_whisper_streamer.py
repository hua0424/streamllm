# src/asr/faster_whisper_streamer.py

from faster_whisper import WhisperModel
import numpy as np
import time
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
import os
from dotenv import load_dotenv

# 从配置导入
from src.config import ASR_MODEL_NAME, DEVICE, VAD_PARAMETERS

# 导入音频分段器
from src.asr.audio_segmenter import VADAudioSegmenter

load_dotenv()  # 加载 .env 文件中的环境变量

@dataclass
class AudioSegment:
    """音频段数据结构"""
    id: str
    audio_data: np.ndarray
    start_time: float
    end_time: float
    duration: float

@dataclass
class TextOutput:
    """文本输出结构"""
    segment_id: str
    text: str
    start_time: float
    end_time: float
    timestamp: float  # 输出时间戳

class StreamingASRProcessor:
    """
    流式ASR处理器
    实现语音段队列管理和流式文本输出
    """
    
    def __init__(
        self,
        model_size: str = ASR_MODEL_NAME,
        device: str = DEVICE,
        compute_type: str = 'auto',  # 使用auto自动选择
        recognition_threshold: float = 3.0,  # 识别阈值(秒)
        sample_rate: int = 16000,
        text_callback: Optional[Callable[[str, float, float], None]] = None
    ):
        """
        初始化流式ASR处理器
        
        Args:
            model_size: faster-whisper 模型大小
            device: 推理设备
            compute_type: 计算类型，'auto'为自动选择
            recognition_threshold: 识别阈值，队列总长度达到此值时开始识别(秒)
            sample_rate: 音频采样率
            text_callback: 文本输出回调函数 callback(text, start_time, end_time)
        """
        print(f'正在加载ASR模型: {model_size} 设备: {device}')
        
        # 根据设备自动选择计算类型
        if compute_type == 'auto':
            if device.lower() == 'cpu':
                compute_type = 'int8'  # CPU使用int8
                print(f'CPU设备自动选择计算类型: {compute_type}')
            else:
                compute_type = 'float16'  # GPU使用float16
                print(f'GPU设备自动选择计算类型: {compute_type}')
        
        # 加载Whisper模型，添加错误处理
        try:
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        except ValueError as e:
            if "float16" in str(e):
                print(f"float16不支持，回退到int8...")
                compute_type = 'int8'
                self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
            else:
                raise e
        
        # 配置参数
        self.recognition_threshold = recognition_threshold
        self.sample_rate = sample_rate
        self.text_callback = text_callback
        
        # 初始化音频分段器
        self.audio_segmenter = VADAudioSegmenter(
            sample_rate=sample_rate,
            threshold=0.5,
            min_speech_duration=0.3,
            min_silence_duration=0.2
        )
        
        # 语音段队列
        self.segment_queue: List[AudioSegment] = []
        self.segment_counter = 0
        
        # 文本输出队列
        self.text_outputs: List[TextOutput] = []
        
        # 当前识别结果缓存
        self.current_recognition: Optional[Dict] = None
        self.current_segments_count = 0
        
        print('ASR模型加载完成')

    def _generate_segment_id(self) -> str:
        """生成音频段ID"""
        self.segment_counter += 1
        return f"seg_{self.segment_counter:04d}"

    def _get_queue_duration(self) -> float:
        """获取队列总时长"""
        return sum(seg.duration for seg in self.segment_queue)

    def _combine_segments(self, segments: List[AudioSegment]) -> np.ndarray:
        """合并多个音频段"""
        if not segments:
            return np.array([], dtype=np.float32)
        return np.concatenate([seg.audio_data for seg in segments]).astype(np.float32)

    def _transcribe_segments(self, segments: List[AudioSegment]) -> Dict:
        """转录音频段"""
        if not segments:
            return {'segments': []}
        
        combined_audio = self._combine_segments(segments)
        
        print(f"正在转录 {len(segments)} 个语音段，总时长: {len(combined_audio)/self.sample_rate:.2f}s")
        
        start_time = time.time()
        segments_result, info = self.model.transcribe(
            combined_audio,
            beam_size=1,  # 减少beam size提高速度
            language="zh",  # 直接指定中文，避免语言检测
            word_timestamps=True,  # 启用词级时间戳，用于精确匹配
            vad_filter=False,  # 关闭VAD过滤，因为我们已经用分段器处理过了
            temperature=0.0,  # 使用贪心解码，更快
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6
        )
        
        transcription_time = time.time() - start_time
        
        # 处理结果
        transcription_segments = []
        for segment in segments_result:
            transcription_segments.append({
                'text': segment.text.strip(),
                'start': segment.start,
                'end': segment.end,
                'words': [
                    {
                        'text': word.word,
                        'start': word.start,
                        'end': word.end,
                        'probability': word.probability
                    }
                    for word in (segment.words or [])
                ]
            })
        
        result = {
            'segments': transcription_segments,
            'info': {
                'language': info.language,
                'language_probability': info.language_probability,
                'duration': info.duration
            },
            'transcription_time': transcription_time
        }
        
        print(f"转录完成，耗时: {transcription_time:.2f}s")
        return result

    def _extract_segment_text(self, recognition_result: Dict, segment_index: int, 
                             total_segments: int) -> Optional[str]:
        """从识别结果中提取指定段的文本，使用词级时间戳精确匹配"""
        if not recognition_result or 'segments' not in recognition_result:
            return None
        
        transcription_segments = recognition_result['segments']
        if not transcription_segments:
            return None
        
        # 获取目标音频段的时间范围（在合并音频中的相对时间）
        target_audio_segment = self.segment_queue[segment_index]
        
        # 计算目标段在合并音频中的时间偏移
        segment_start_offset = 0.0
        for i in range(segment_index):
            segment_start_offset += self.segment_queue[i].duration
        
        segment_end_offset = segment_start_offset + target_audio_segment.duration
        
        print(f"    目标段 {segment_index+1}: 合并音频中时间范围 [{segment_start_offset:.2f}s - {segment_end_offset:.2f}s]")
        
        # 收集目标段对应的词
        matched_words = []
        
        for trans_seg in transcription_segments:
            if 'words' not in trans_seg or not trans_seg['words']:
                # 如果没有词级时间戳，使用段级时间戳进行粗略匹配
                if trans_seg['end'] > segment_start_offset and trans_seg['start'] < segment_end_offset:
                    # 计算重叠比例
                    overlap_start = max(trans_seg['start'], segment_start_offset)
                    overlap_end = min(trans_seg['end'], segment_end_offset)
                    overlap_duration = overlap_end - overlap_start
                    trans_duration = trans_seg['end'] - trans_seg['start']
                    
                    if overlap_duration > trans_duration * 0.5:  # 超过50%重叠
                        matched_words.append(trans_seg['text'].strip())
                        print(f"      段级匹配: {trans_seg['text'].strip()}")
                continue
            
            # 使用词级时间戳进行精确匹配
            for word in trans_seg['words']:
                word_start = word['start']
                word_end = word['end']
                
                # 检查词是否在目标段的时间范围内
                # 词的中心点在目标段内，或者词与目标段有显著重叠
                word_center = (word_start + word_end) / 2
                
                # 方法1：词的中心点在目标段内
                if segment_start_offset <= word_center <= segment_end_offset:
                    matched_words.append(word['text'])
                    print(f"      词匹配: {word['text']} [{word_start:.2f}s-{word_end:.2f}s]")
                    continue
                
                # 方法2：词与目标段有显著重叠（超过50%）
                overlap_start = max(word_start, segment_start_offset)
                overlap_end = min(word_end, segment_end_offset)
                if overlap_end > overlap_start:
                    overlap_duration = overlap_end - overlap_start
                    word_duration = word_end - word_start
                    if word_duration > 0 and overlap_duration > word_duration * 0.5:
                        matched_words.append(word['text'])
                        print(f"      词重叠匹配: {word['text']} [{word_start:.2f}s-{word_end:.2f}s] 重叠率{overlap_duration/word_duration:.1%}")
        
        if matched_words:
            result_text = ''.join(matched_words).strip()
            print(f"    段 {segment_index+1} 提取的文本: '{result_text}'")
            return result_text
        else:
            print(f"    段 {segment_index+1} 没有匹配到文本")
            return None

    def _process_queue(self) -> List[str]:
        """处理队列，返回新输出的文本列表"""
        new_texts = []
        
        while self.segment_queue:
            queue_duration = self._get_queue_duration()
            
            # 检查是否达到识别阈值
            if queue_duration < self.recognition_threshold:
                break
            
            # 检查是否需要重新识别
            queue_length = len(self.segment_queue)
            
            if (self.current_recognition is None or 
                self.current_segments_count != queue_length):
                
                print(f"队列长度: {queue_length}, 总时长: {queue_duration:.2f}s - 开始识别")
                
                # 进行新的识别
                self.current_recognition = self._transcribe_segments(self.segment_queue)
                self.current_segments_count = queue_length
            
            # 提取第一个段的文本（即将被删除的段）
            first_segment = self.segment_queue[0]
            print(f"  正在提取第一个段的文本: {first_segment.id} [{first_segment.start_time:.2f}s-{first_segment.end_time:.2f}s]")
            
            segment_text = self._extract_segment_text(
                self.current_recognition, 0, self.current_segments_count
            )
            
            if segment_text:
                # 创建文本输出
                text_output = TextOutput(
                    segment_id=first_segment.id,
                    text=segment_text,
                    start_time=first_segment.start_time,
                    end_time=first_segment.end_time,
                    timestamp=time.time()
                )
                
                self.text_outputs.append(text_output)
                new_texts.append(segment_text)
                
                print(f"  ✓ 输出文本: [{first_segment.start_time:.2f}s-{first_segment.end_time:.2f}s] '{segment_text}'")
                
                # 调用回调函数
                if self.text_callback:
                    self.text_callback(segment_text, first_segment.start_time, first_segment.end_time)
            else:
                print(f"  ✗ 第一个段没有提取到文本")
            
            # 删除第一个段
            self.segment_queue.pop(0)
            print(f"  删除段: {first_segment.id}")
            
            # 由于删除了第一个段，需要清除当前识别缓存，下次重新识别剩余段
            self.current_recognition = None
            self.current_segments_count = 0
        
        return new_texts

    def add_audio_chunk(self, audio_chunk: np.ndarray) -> List[str]:
        """
        添加音频块并返回新的文本输出
        
        Args:
            audio_chunk: 音频数据 (float32, 单声道, 16kHz)
            
        Returns:
            List[str]: 新输出的文本列表
        """
        # 使用分段器处理音频块
        new_segments_data = self.audio_segmenter.process_streaming_audio(audio_chunk)
        
        # 转换为AudioSegment对象并添加到队列
        for seg_data in new_segments_data:
            segment = AudioSegment(
                id=self._generate_segment_id(),
                audio_data=seg_data['audio_data'],
                start_time=seg_data['start_time'],
                end_time=seg_data['end_time'],
                duration=seg_data['duration']
            )
            self.segment_queue.append(segment)
            print(f"新增语音段: {segment.id} [{segment.start_time:.2f}s-{segment.end_time:.2f}s] {segment.duration:.2f}s")
        
        # 处理队列并返回新文本
        return self._process_queue()

    def finish_stream(self) -> List[str]:
        """
        结束流式处理，输出剩余文本
        
        Returns:
            List[str]: 剩余的文本输出
        """
        print("流式处理结束，处理剩余语音段...")
        
        # 获取分段器中剩余的段
        remaining_segments = self.audio_segmenter.finish_streaming()
        
        # 添加剩余段到队列
        for seg_data in remaining_segments:
            segment = AudioSegment(
                id=self._generate_segment_id(),
                audio_data=seg_data['audio_data'],
                start_time=seg_data['start_time'],
                end_time=seg_data['end_time'],
                duration=seg_data['duration']
            )
            self.segment_queue.append(segment)
        
        # 处理剩余的队列
        remaining_texts = []
        
        while self.segment_queue:
            # 对剩余的段进行识别
            if self.current_recognition is None or self.current_segments_count != len(self.segment_queue):
                self.current_recognition = self._transcribe_segments(self.segment_queue)
                self.current_segments_count = len(self.segment_queue)
            
            # 输出第一个段的文本
            first_segment = self.segment_queue[0]
            segment_text = self._extract_segment_text(
                self.current_recognition, 0, self.current_segments_count
            )
            
            if segment_text:
                text_output = TextOutput(
                    segment_id=first_segment.id,
                    text=segment_text,
                    start_time=first_segment.start_time,
                    end_time=first_segment.end_time,
                    timestamp=time.time()
                )
                
                self.text_outputs.append(text_output)
                remaining_texts.append(segment_text)
                
                print(f"最终输出文本: [{first_segment.start_time:.2f}s-{first_segment.end_time:.2f}s] {segment_text}")
                
                # 调用回调函数
                if self.text_callback:
                    self.text_callback(segment_text, first_segment.start_time, first_segment.end_time)
            
            # 删除第一个段
            self.segment_queue.pop(0)
            self.current_segments_count -= 1
        
        # 清理状态
        self.current_recognition = None
        self.current_segments_count = 0
        
        print(f"流式ASR处理完成，总共输出 {len(self.text_outputs)} 个文本段")
        return remaining_texts

    def get_all_text_outputs(self) -> List[TextOutput]:
        """获取所有文本输出"""
        return self.text_outputs.copy()

    def get_final_text(self) -> str:
        """获取完整的文本"""
        return ' '.join([output.text for output in self.text_outputs])

    def reset(self):
        """重置处理器状态"""
        self.segment_queue.clear()
        self.text_outputs.clear()
        self.current_recognition = None
        self.current_segments_count = 0
        self.audio_segmenter._reset_streaming_state()
        print("ASR处理器状态已重置")

# 向后兼容的别名
FasterWhisperStreamer = StreamingASRProcessor

def main():
    """测试函数"""
    import argparse
    import librosa
    
    parser = argparse.ArgumentParser(description='测试流式ASR处理器')
    parser.add_argument('--wav_path', type=str, required=True, help='WAV文件路径')
    parser.add_argument('--model_size', type=str, default='base', help='模型大小 (tiny/base/small/medium/large)')
    parser.add_argument('--chunk_duration', type=float, default=0.5, help='音频块时长(秒)')
    parser.add_argument('--threshold', type=float, default=3.0, help='识别阈值(秒)')
    parser.add_argument('--fast_mode', action='store_true', help='快速模式，使用tiny模型和优化参数')
    
    args = parser.parse_args()
    
    # 快速模式设置
    if args.fast_mode:
        args.model_size = 'tiny'
        args.threshold = 2.0
        print("启用快速模式: 使用tiny模型，阈值2.0s")
    
    print(f"正在测试WAV文件: {args.wav_path}")
    print(f"模型大小: {args.model_size}")
    print(f"音频块时长: {args.chunk_duration}s")
    print(f"识别阈值: {args.threshold}s")
    
    def text_callback(text: str, start_time: float, end_time: float):
        """文本输出回调"""
        print(f"[实时输出] [{start_time:.2f}s-{end_time:.2f}s] {text}")
    
    try:
        # 加载音频文件
        audio_data, sr = librosa.load(args.wav_path, sr=16000, mono=True)
        print(f"音频加载成功: 时长 {len(audio_data)/sr:.2f}s")
        
        # 创建流式ASR处理器
        processor = StreamingASRProcessor(
            model_size=args.model_size,
            recognition_threshold=args.threshold,
            text_callback=text_callback
        )
        
        # 模拟流式输入
        chunk_size = int(args.chunk_duration * 16000)
        print(f"\n开始流式处理 (每{args.chunk_duration}s一个块)...")
        print("-" * 60)
        
        start_time = time.time()
        all_texts = []
        
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            current_time = i / 16000
            
            print(f"\n处理块: {current_time:.2f}s-{(i+len(chunk))/16000:.2f}s")
            
            # 处理音频块
            new_texts = processor.add_audio_chunk(chunk)
            all_texts.extend(new_texts)
        
        # 完成处理
        print(f"\n完成流式输入，处理剩余段...")
        final_texts = processor.finish_stream()
        all_texts.extend(final_texts)
        
        process_time = time.time() - start_time
        
        print(f"\n" + "="*60)
        print(f"流式ASR测试完成")
        print(f"处理时间: {process_time:.2f}s")
        print(f"音频时长: {len(audio_data)/16000:.2f}s")
        print(f"实时因子: {(len(audio_data)/16000)/process_time:.2f}x")
        print(f"输出文本段数: {len(all_texts)}")
        print(f"\n完整文本:")
        print("-" * 60)
        print(processor.get_final_text())
        
    except Exception as e:
        print(f"测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
