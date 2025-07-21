# src/asr/faster_whisper_streamer.py

from faster_whisper import WhisperModel
import numpy as np
import time
import logging
from typing import List, Dict, Optional, Callable, Generator, Tuple
from dataclasses import dataclass
import os
from dotenv import load_dotenv
from datetime import datetime
from queue import Queue

# 从配置导入
from src.config import ASR_MODEL_NAME, DEVICE, VAD_PARAMETERS

# 导入音频分段器
from src.asr.audio_segmenter import VADAudioSegmenter

load_dotenv()  # 加载 .env 文件中的环境变量

# 获取当前模块的logger
logger = logging.getLogger(__name__)

@dataclass
class TextOutput:
    """文本输出结构"""
    segment_id: str
    text: str
    start_time: float
    end_time: float
    timestamp: float  # 输出时间戳

@dataclass
class AudioSegment:
    """音频段数据结构"""
    id: str
    audio_data: np.ndarray
    start_time: float
    end_time: float
    duration: float
    text: Optional[TextOutput] = None

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
        sample_rate: int = 16000, # 音频采样率
        prefix_segments: int = 1, # 前缀段数
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
        logger.info(f'正在加载ASR模型: {model_size} 设备: {device}')
        
        # 根据设备自动选择计算类型
        if compute_type == 'auto':
            if device.lower() == 'cpu':
                compute_type = 'int8'  # CPU使用int8
                logger.info(f'CPU设备自动选择计算类型: {compute_type}')
            else:
                compute_type = 'float16'  # GPU使用float16
                logger.info(f'GPU设备自动选择计算类型: {compute_type}')
        
        # 加载Whisper模型，添加错误处理
        try:
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        except ValueError as e:
            if "float16" in str(e):
                logger.warning(f"float16不支持，回退到int8...")
                compute_type = 'int8'
                self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
            else:
                raise e
        
        # 配置参数
        self.recognition_threshold = recognition_threshold
        self.sample_rate = sample_rate
        self.prefix_segments = prefix_segments
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
        
        # 流式输入状态标志
        self.is_stream_started = True
        self.is_stream_finished = False
        
        logger.info('ASR模型加载完成')

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
        """将队列中的音频段合并后转录音频段"""
        if not segments:
            return {'segments': []}
        
        combined_audio = self._combine_segments(segments)
        
        logger.debug(f"{datetime.now().strftime('%H:%M:%S.%f')[:-3]}正在转录 {len(segments)} 个语音段，总时长: {len(combined_audio)/self.sample_rate:.2f}s")
        
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
        for i, segment in enumerate(segments_result):
            logger.debug(f"  段 {i+1}: [{segment.start:.2f}s-{segment.end:.2f}s] {segment.text}")
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
        
        logger.debug(f"{datetime.now().strftime('%H:%M:%S.%f')[:-3]}转录完成，耗时: {time.time() - start_time:.2f}s")
        return result

    def _extract_segment_text(self, recognition_result: Dict, segment_index: int, 
                             total_segments: int) -> Optional[str]:
        """从识别结果中提取指定段的文本，使用词级时间戳简单匹配"""
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
        
        logger.debug(f"{datetime.now().strftime('%H:%M:%S.%f')[:-3]}    目标段 {segment_index+1}: 合并音频中时间范围 [{segment_start_offset:.2f}s - {segment_end_offset:.2f}s]")
        
        # 收集目标段对应的词
        matched_words = []
        
        for trans_seg in transcription_segments:
            if 'words' not in trans_seg or not trans_seg['words']:
                # 如果没有词级时间戳，使用段级时间戳进行匹配
                seg_center = (trans_seg['start'] + trans_seg['end']) / 2
                if segment_start_offset <= seg_center <= segment_end_offset:
                    matched_words.append(trans_seg['text'].strip())
                    logger.debug(f"      段级匹配: {trans_seg['text'].strip()}")
                continue
            
            # 使用词级时间戳进行匹配
            for word in trans_seg['words']:
                word_start = word['start']
                word_end = word['end']
                word_center = (word_start + word_end) / 2
                
                # 简单判断：
                # 方案1：如果词的中心点在目标段内，就归属于该段
                # if segment_start_offset <= word_center <= segment_end_offset:
                #     matched_words.append(word['text'])
                #     print(f"      词匹配: {word['text']} [{word_start:.2f}s-{word_end:.2f}s]")
                # 方案2：如果词的开始时间在目标段内，就归属于该段
                # if segment_start_offset <= word_start <= segment_end_offset:
                #     matched_words.append(word['text'])
                #     print(f"      词匹配: {word['text']} [{word_start:.2f}s-{word_end:.2f}s]")
                # 方案3：如果词的结束时间在目标段内，就归属于该段
                if segment_start_offset <= word_end <= segment_end_offset:
                    matched_words.append(word['text'])
        
        if matched_words:
            result_text = ''.join(matched_words).strip()
            logger.debug(f"{datetime.now().strftime('%H:%M:%S.%f')[:-3]}    段 {segment_index+1} 提取的文本: '{result_text}'")
            return result_text
        else:
            logger.debug(f"    段 {segment_index+1} 没有匹配到文本")
            return None

    def _process_queue(self) -> List[str]:
        """处理队列，返回新输出的文本列表"""
        new_texts = []
        
        if self.segment_queue:
            queue_duration = self._get_queue_duration()
            queue_length = len(self.segment_queue)
            
            # 检查是否达到识别阈值
            if queue_duration < self.recognition_threshold and not self.is_stream_finished:
                return new_texts
            
            # 检查语音段数是否满足最低要求（前缀段数 + 本身 + 后续至少1段）
            if queue_length < self.prefix_segments + 2 and not self.is_stream_finished:
                return new_texts
            
            logger.debug(f"{datetime.now().strftime('%H:%M:%S.%f')[:-3]}队列长度: {queue_length}, 总时长: {queue_duration:.2f}s, 流式结束: {self.is_stream_finished} - 开始识别")
            
            # 进行新的识别
            current_recognition = self._transcribe_segments(self.segment_queue)
            current_segments_count = queue_length

            # 记录每个段的识别结果
            for i, segment in enumerate(self.segment_queue):
                segment.text = self._extract_segment_text(
                    current_recognition, i, current_segments_count
                )
                if segment.text:
                    self.segment_queue[i].text = TextOutput(
                        segment_id=self.segment_queue[i].id,
                        text=segment.text,
                        start_time=self.segment_queue[i].start_time,
                        end_time=self.segment_queue[i].end_time,
                        timestamp=time.time()
                    )
                else:
                    logger.debug(f" 第 {i} 段没有提取到文本")
            
            # 决定哪个段的文本已经确定了需要输出
            # 一般情况是输出第前缀段数+1个段，这个是要识别的段，如果是流式开头，那么应该输出前缀段数+1个段，因为这些段的前缀不可能再多了
            if self.is_stream_finished:
                output_segment_index = list(range(self.prefix_segments, queue_length - self.prefix_segments + 1))
                logger.debug(f"流式结束，输出所有段:{output_segment_index}")

            elif self.is_stream_started:
                output_segment_index = list(range(0, self.prefix_segments + 1))
                self.is_stream_started = False
            else:
                output_segment_index = [self.prefix_segments]
            
            # 输出前缀段数+1个段
            for i in output_segment_index:
                self.text_outputs.append(self.segment_queue[i].text)
                new_texts.append(self.segment_queue[i].text)
                logger.info(f"{datetime.now().strftime('%H:%M:%S.%f')[:-3]}  ✓ 输出文本: index[{i}] [{self.segment_queue[i].start_time:.2f}s-{self.segment_queue[i].end_time:.2f}s] '{self.segment_queue[i].text}'")

            if self.text_callback:
                self.text_callback(new_texts, self.segment_queue[output_segment_index[0]].start_time, self.segment_queue[output_segment_index[-1]].end_time)
            
            # 删除队列首段
            self.segment_queue.pop(0)
        
        return new_texts
    
    def add_audio_chunk(self, audio_chunk: np.ndarray, is_stream_finished: bool = False) -> List[str]:
        """
        添加音频块并返回新的文本输出
        """
        logger.info(f"添加音频块: {is_stream_finished}")
        if is_stream_finished:
            logger.info("🎉 检测到最后一个音频块，流结束")
            self.is_stream_finished = True

        start_time = time.time()
        new_segments_data = self.audio_segmenter.process_streaming_audio(audio_chunk)

        for seg_data in new_segments_data:
            segment = AudioSegment(
                id=self._generate_segment_id(),
                audio_data=seg_data['audio_data'],
                start_time=seg_data['start_time'],
                end_time=seg_data['end_time'],
                duration=seg_data['duration']
            )
            self.segment_queue.append(segment)
            logger.debug(f"{datetime.now().strftime('%H:%M:%S.%f')[:-3]}新增语音段: {segment.id} [{segment.start_time:.2f}s-{segment.end_time:.2f}s] {segment.duration:.2f}s")
            
        logger.debug(f"{datetime.now().strftime('%H:%M:%S.%f')[:-3]}音频分段处理时间: {time.time() - start_time:.2f}s")

        start_time = time.time()
        new_texts = self._process_queue()
        logger.info(f"ASR队列处理时间: {time.time() - start_time:.2f}s, 生成文本: {new_texts}")
        logger.debug(f"{datetime.now().strftime('%H:%M:%S.%f')[:-3]}ASR队列处理时间: {time.time() - start_time:.2f}s")
        return new_texts
    
    def add_audio_chunk_queue(self, chunk_queue: Queue[Tuple[np.ndarray, bool]]) -> Generator[str, None, None]:
        """
        添加音频块生成器并返回新的文本输出
        
        Args:
            audio_chunk_generator: 音频数据生成器
        """
        logger.info("开始添加音频块生成器")
        while True:
            chunk_data, is_last = chunk_queue.get()
            new_texts = self.add_audio_chunk(chunk_data, is_stream_finished=is_last)
            yield ' '.join([text.text for text in new_texts])
            if is_last:
                break

    def add_audio_chunk_generator(self, chunk_generator: Generator[Tuple[np.ndarray, bool], None, None]) -> Generator[str, None, None]:
        """
        添加音频块生成器并返回新的文本输出
        
        Args:
            audio_chunk_generator: 音频数据生成器
        """
        chunk_count = 0
        start_time = time.time()
        
        for chunk_data, is_last in chunk_generator:
            chunk_count += 1
            current_time = time.time() - start_time
            
            logger.debug(f"处理第{chunk_count}个音频块 (耗时: {current_time:.2f}s):")
            logger.debug(f"  样本数: {len(chunk_data)}")
            logger.debug(f"  数据类型: {chunk_data.dtype}")
            logger.debug(f"  数据范围: [{chunk_data.min():.4f}, {chunk_data.max():.4f}]")
            logger.debug(f"  是否最后块: {is_last}")

            new_texts = self.add_audio_chunk(chunk_data, is_stream_finished=is_last)

            if new_texts:
                yield ' '.join([text.text for text in new_texts])

    # def add_audio_chunk(self, audio_chunk: np.ndarray, is_stream_finished: bool = False) -> List[str]:
    #     """
    #     添加音频块并返回新的文本输出
        
    #     Args:
    #         audio_chunk: 音频数据 (float32, 单声道, 16kHz)
            
    #     Returns:
    #         List[str]: 新输出的文本列表
    #     """

    #     if self.is_stream_finished:
    #         raise ValueError("流式处理已结束，无法添加音频块")
        
    #     # 使用分段器处理音频块
    #     if is_stream_finished:
    #         self.is_stream_finished = True
        
    #     start_time = time.time()
    #     new_segments_data = self.audio_segmenter.process_streaming_audio(audio_chunk)
    #     logger.debug(f"{datetime.now().strftime('%H:%M:%S.%f')[:-3]}音频分段处理时间: {time.time() - start_time:.2f}s")
        
    #     # 转换为AudioSegment对象并添加到队列
    #     for seg_data in new_segments_data:
    #         segment = AudioSegment(
    #             id=self._generate_segment_id(),
    #             audio_data=seg_data['audio_data'],
    #             start_time=seg_data['start_time'],
    #             end_time=seg_data['end_time'],
    #             duration=seg_data['duration']
    #         )
    #         self.segment_queue.append(segment)
    #         logger.debug(f"{datetime.now().strftime('%H:%M:%S.%f')[:-3]}新增语音段: {segment.id} [{segment.start_time:.2f}s-{segment.end_time:.2f}s] {segment.duration:.2f}s")
        
    #     # 处理队列并返回新文本
    #     return self._process_queue()

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
        self.is_stream_finished = False
        self.audio_segmenter._reset_streaming_state()
        logger.info("ASR处理器状态已重置")

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
    parser.add_argument('--no_segmentation', action='store_true', help='不分段模式，直接使用faster_whisper转录整个音频')
    
    args = parser.parse_args()
    
    # 快速模式设置
    if args.fast_mode:
        args.model_size = 'tiny'
        args.threshold = 2.0
        logger.info("启用快速模式: 使用tiny模型，阈值2.0s")
    
    print(f"正在测试WAV文件: {args.wav_path}")
    print(f"模型大小: {args.model_size}")
    
    if args.no_segmentation:
        print("模式: 不分段直接转录")
    else:
        print(f"音频块时长: {args.chunk_duration}s")
        print(f"识别阈值: {args.threshold}s")
        print("模式: 流式分段转录")
    
    def text_callback(text: str, start_time: float, end_time: float):
        """文本输出回调"""
        logger.info(f"[实时输出] [{start_time:.2f}s-{end_time:.2f}s] {text}")
    
    try:
        # 加载音频文件
        audio_data, sr = librosa.load(args.wav_path, sr=16000, mono=True)
        logger.info(f"音频加载成功: 时长 {len(audio_data)/sr:.2f}s")
        
        start_time = time.time()
        
        if args.no_segmentation:
            # 不分段模式：直接使用faster_whisper转录整个音频
            print(f"\n开始不分段转录...")
            print("-" * 60)
            
            # 根据设备自动选择计算类型
            compute_type = 'auto'
            if compute_type == 'auto':
                if DEVICE.lower() == 'cpu':
                    compute_type = 'int8'
                    logger.info(f'CPU设备自动选择计算类型: {compute_type}')
                else:
                    compute_type = 'float16'
                    logger.info(f'GPU设备自动选择计算类型: {compute_type}')
            
            # 加载Whisper模型
            try:
                model = WhisperModel(args.model_size, device=DEVICE, compute_type=compute_type)
            except ValueError as e:
                if "float16" in str(e):
                    logger.warning(f"float16不支持，回退到int8...")
                    compute_type = 'int8'
                    model = WhisperModel(args.model_size, device=DEVICE, compute_type=compute_type)
                else:
                    raise e
            
            # 转录整个音频
            segments_result, info = model.transcribe(
                audio_data,
                beam_size=1,
                language="zh",
                word_timestamps=True,
                vad_filter=False,
                temperature=0.0,
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6
            )
            
            # 收集转录结果
            all_text = []
            for segment in segments_result:
                text = segment.text.strip()
                if text:
                    all_text.append(text)
                    print(f"[{segment.start:.2f}s-{segment.end:.2f}s] {text}")
            
            final_text = ' '.join(all_text)
            process_time = time.time() - start_time
            
            print(f"\n" + "="*60)
            print(f"不分段转录完成")
            print(f"处理时间: {process_time:.2f}s")
            print(f"音频时长: {len(audio_data)/sr:.2f}s")
            print(f"实时因子: {(len(audio_data)/sr)/process_time:.2f}x")
            print(f"\n完整文本:")
            print("-" * 60)
            print(final_text)
            
            return
        
        # 流式分段模式
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
        
        all_texts = []
        
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            current_time = i / 16000
            
            print(f"\n处理块: {current_time:.2f}s-{(i+len(chunk))/16000:.2f}s")
            
            # 处理音频块
            is_stream_finished = (i >= len(audio_data) - chunk_size)
            logger.debug(f"流式结束:{is_stream_finished} i = {i}, chunk_size = {chunk_size}, len(audio_data) = {len(audio_data)}")
            new_texts = processor.add_audio_chunk(chunk, is_stream_finished=is_stream_finished)
            all_texts.extend(new_texts)
        
        # 完成处理        
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
        logger.error(f"测试失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())



if __name__ == "__main__":
    main()
