# src/asr/faster_whisper_streamer.py

from enum import Enum, auto
from faster_whisper import WhisperModel
import numpy as np
import time
import logging
from typing import List, Dict, Optional, Callable, Generator, Tuple
from dataclasses import dataclass, field
import os
from dotenv import load_dotenv
from datetime import datetime
from queue import Queue

# 从配置导入
from src.config import ASR_MODEL_NAME, DEVICE, VAD_PARAMETERS

# 导入音频分段器
from src.asr.audio_segmenter import VADAudioSegmenter

# 导入日志工具
from src.utils.logging_utils import get_logger

load_dotenv()  # 加载 .env 文件中的环境变量

# 获取当前模块的logger
logger = get_logger(__name__)

@dataclass
class ASRAudioSegment:
    """音频段数据结构"""
    id: str
    audio_data: np.ndarray
    start_time: float
    end_time: float
    duration: float
    text: str | None = None  # 转录文本
    is_final: bool = False  # 是否为流式最后音频段
    is_start: bool = False  # 是否为流式第一个音频段

class TimingEventType(Enum):
    """
    时间事件类型枚举类
    """
    START_FUNCTION = auto() # 函数调用开始时间
    END_FUNCTION = auto() # 函数调用结束时间
    START_ASR = auto() # ASR处理开始时间
    END_ASR = auto() # ASR处理结束时间
    START_TRANSCRIBE = auto() # 模型转录开始时间
    END_TRANSCRIBE = auto() # 模型转录结束时间
    DECODE_TOKEN = auto() # 模型推理decode token时间

@dataclass
class ASRCache:
    """
    内部缓存类，封装原来属于 StreamingASRProcessor 的状态属性
    """
    # 音频段
    segment_queue: List["ASRAudioSegment"] = field(default_factory=list)
    segment_counter: int = 0
    current_recognition: Optional[Dict] = None
    current_segments_count: int = 0
        
class StreamingASRProcessor:
    """
    流式ASR处理器
    实现语音段队列管理和流式文本输出
    """

    def __init__(
        self,
        model_size: str = ASR_MODEL_NAME or "base",
        device: str = DEVICE,
        compute_type: str = 'auto',  # 使用auto自动选择
        recognition_threshold: float = 3.0,  # 识别阈值(秒)
        sample_rate: int = 16000, # 音频采样率
        prefix_segments: int = 1, # 前缀段数
    ):
        """
        初始化流式ASR处理器
        
        Args:
            model_size: faster-whisper 模型大小
            device: 推理设备
            compute_type: 计算类型，'auto'为自动选择
            recognition_threshold: 识别阈值，队列总长度达到此值时开始识别(秒)
            sample_rate: 音频采样率
        """
        logger.info(f'Loading ASR model: {model_size} on {device}')
        
        # 根据设备自动选择计算类型
        if compute_type == 'auto':
            if device.lower() == 'cpu':
                compute_type = 'int8'  # CPU使用int8
                logger.debug(f'Auto-selected compute type for CPU: {compute_type}')
            else:
                compute_type = 'float16'  # GPU使用float16
                logger.debug(f'Auto-selected compute type for GPU: {compute_type}')
        
        # 加载Whisper模型
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        
        # 配置参数
        self.recognition_threshold = recognition_threshold
        self.sample_rate = sample_rate
        self.prefix_segments = prefix_segments
        
        # # 语音段队列
        # self.segment_queue: List[AudioSegment] = []
        # self.segment_counter = 0
        
        # # 文本输出队列
        # self.text_outputs: List[TextOutput] = []
        
        # # 当前识别结果缓存
        # self.current_recognition: Optional[Dict] = None
        # self.current_segments_count = 0
        
        # # 流式输入状态标志
        # self.is_stream_started = True
        # self.is_stream_finished = False

        # 用于记录详细延迟的变量
        self.timing_events:Dict[TimingEventType, float] = {}

        logger.info('ASR model loaded successfully')

    def get_last_timings(self):
        return self.timing_events

    def reset_timings(self):
        return self.timing_events.clear    
    
    def transcribe_complete_audio(self, audio_path: str) -> Dict:
        """
        完整音频文件转录（用于基线系统）
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            转录结果字典，包含文本和时序信息
        """
        try:
            import librosa
            
            # 加载音频文件
            start_time = time.time()
            audio_data, sr = librosa.load(audio_path, sr=self.sample_rate)
            audio_duration = len(audio_data) / sr
            load_time = time.time() - start_time
            
            logger.info(f"加载音频文件: {audio_path}, 时长: {audio_duration:.2f}s")
            
            # 使用Whisper进行完整转录
            transcription_start = time.time()
            segments_result, info = self.model.transcribe(
                audio_data,
                beam_size=5,  # 基线系统使用较高beam size以获得更好质量
                language="zh",
                word_timestamps=True,
                vad_filter=True,
                temperature=0.0,
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6
            )
            
            transcription_time = time.time() - transcription_start
            
            # 整理转录结果
            full_text = ""
            segments = []
            
            for segment in segments_result:
                segment_text = segment.text.strip()
                full_text += segment_text
                segments.append({
                    'text': segment_text,
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
                'text': full_text.strip(),
                'segments': segments,
                'timing': {
                    'audio_duration': audio_duration,
                    'load_time': load_time,
                    'transcription_time': transcription_time,
                    'total_time': time.time() - start_time
                },
                'info': {
                    'language': info.language,
                    'language_probability': info.language_probability,
                    'duration': info.duration
                }
            }
            
            logger.info(f"完整转录完成: '{full_text}', 耗时: {transcription_time:.2f}s")
            return result
            
        except ImportError as e:
            logger.error("librosa未安装，无法加载音频文件")
            raise e
        except Exception as e:
            logger.error(f"音频转录失败: {e}")
            raise e
    
    def transcribe_audio_segment(self, cache: ASRCache, audio_segment: ASRAudioSegment) -> Tuple[ASRCache, str | None]:
        """
        添加音频块并返回新的转录文本，输出文本可能为空
        @param cache ASR缓存对象
        @param audio_segment 音频信号段AudioSegment对象
        @return cache ASR缓存对象, str 新输出的文本，可能为空
        备注：音频块必须是 float32, 单声道, 16kHz
        例如： np.ndarray = np.random.randn(16000).astype(np.float32)
        """

        self.reset_timings()
        self.timing_events[TimingEventType.START_FUNCTION] = time.perf_counter()       

        if cache is None:
            cache = ASRCache()

        if audio_segment is None:
            return cache, None
        
        cache.segment_queue.append(audio_segment)

        # 开始识别音频队列里累计的音频段
        queue_duration = sum(seg.duration for seg in cache.segment_queue)
        queue_length = len(cache.segment_queue)

        # 检查是否达到识别阈值
        if queue_duration < self.recognition_threshold and not cache.is_stream_finished:
            return cache, None

        # 检查语音段数是否满足最低要求（前缀段数 + 本身 + 后续至少1段）
        if queue_length < self.prefix_segments + 2 and not cache.is_stream_finished:
            return cache, None

        # 开始进行语音识别
        self.timing_events[TimingEventType.START_ASR] = time.perf_counter()

        current_recognition = self._transcribe_segments(cache.segment_queue)

        if current_recognition["segments"] is None or len(current_recognition["segments"]) == 0:
            logger.warning("当前累积音频段的识别结果为空")
            return cache, None

        # 记录每个段的识别结果
        segment_start_offset = 0.0
        for i, segment in enumerate(cache.segment_queue):
            segment_start_offset += segment.duration
            segment.text = self._extract_segment_text(current_recognition, i, segment, segment_start_offset)

            if not segment.text:
                logger.warning(f" 第 {i} 段没有提取到文本")

        # 决定哪个段的文本已经确定了需要输出
        # 一般情况是输出第前缀段数+1个段，这个是要识别的段。但如果是流式开头，那么应该输出前缀段数+1个段，因为这些段不可能再有前缀了
        # 如果是流式结尾，那么输出所有剩余段
        if audio_segment.is_final:
            output_segment_index = list(range(self.prefix_segments, queue_length - self.prefix_segments + 1))

        # 如果是流式开始
        elif cache.segment_queue[0].is_start:
            output_segment_index = list(range(0, self.prefix_segments + 1))
        else:
            output_segment_index = [self.prefix_segments]

        logger.debug(f"本次输出的段为:{[cache.segment_queue[i].id for i in output_segment_index]}")
        
        # 输出队列中选定的段，确保所有元素都是字符串（将 None 转为空串）
        output_text = ''.join([cache.segment_queue[i].text or '' for i in output_segment_index])
        logger.debug(f"{datetime.now().strftime('%H:%M:%S.%f')[:-3]}  ✓ 输出文本: index{output_segment_index} [{cache.segment_queue[output_segment_index[0]].start_time:.2f}s-{cache.segment_queue[output_segment_index[-1]].end_time:.2f}s] '{output_text}'")

        # 删除队列首段
        cache.segment_queue.pop(0)

        self.timing_events[TimingEventType.END_ASR] = time.perf_counter()

        self.timing_events[TimingEventType.END_FUNCTION] = time.perf_counter()

        return cache, output_text

    def _transcribe_segments(self, segments: List[ASRAudioSegment]) -> Dict:
        """将队列中的音频段合并后转录音频段"""
        
        # 合并音频段，扩展上下文增加识别准确率
        combined_audio = np.concatenate([seg.audio_data for seg in segments]).astype(np.float32)
        
        logger.debug(f"{datetime.now().strftime('%H:%M:%S.%f')[:-3]}正在转录 {len(segments)} 个语音段，总时长: {len(combined_audio)/self.sample_rate:.2f}s")

        
        self.timing_events[TimingEventType.START_TRANSCRIBE] = time.perf_counter()
        segments_result, info = self.model.transcribe(
            combined_audio,
            beam_size=1,  # 减少beam size提高速度
#            language="zh",  # 直接指定中文，避免语言检测
            word_timestamps=True,  # 启用词级时间戳，用于精确匹配
            vad_filter=False,  # 关闭VAD过滤，因为我们已经用分段器处理过了
            temperature=0.0,  # 使用贪心解码，更快
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6
        )
        
        # 在迭代的时候才触发真正的转录
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

        self.timing_events[TimingEventType.END_TRANSCRIBE] = time.perf_counter()
        
        result = {
            'segments': transcription_segments,
            'info': {
                'language': info.language,
                'language_probability': info.language_probability,
                'duration': info.duration
            },
        }
        
        logger.debug(f"{datetime.now().strftime('%H:%M:%S.%f')[:-3]}转录完成，耗时: {time.perf_counter() - self.timing_events[TimingEventType.START_TRANSCRIBE]:.2f}s")
        return result

    def _extract_segment_text(self, recognition_result: Dict, segment_index: int, segment: ASRAudioSegment, segment_start_offset: float) -> Optional[str]:
        """从识别结果中提取指定音频段的文本，使用词级时间戳简单匹配"""
        
        transcription_segments = recognition_result['segments']

        # 计算目标段在合并音频中的时间偏移        
        segment_end_offset = segment_start_offset + segment.duration
        logger.debug(f"{datetime.now().strftime('%H:%M:%S.%f')[:-3]}    目标段 {segment_index+1}: 合并音频中时间范围 [{segment_start_offset:.2f}s - {segment_end_offset:.2f}s]")
        
        # 收集目标段对应的词
        matched_words = []
        
        for trans_seg in transcription_segments:
            if 'words' not in trans_seg or not trans_seg['words']:
                logger.error("转录结果中缺少词级时间戳，无法进行精确匹配")
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
                
                # word_center = (word_start + word_end) / 2
                
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


