# src/asr/faster_whisper_streamer.py

from enum import Enum, auto
from faster_whisper import WhisperModel
import numpy as np
import time
import logging
from typing import List, Dict, Optional, Callable, Generator, Tuple, Union
from dataclasses import dataclass, field
import threading
from dotenv import load_dotenv
from datetime import datetime

# 从配置导入
from src.config import ASR_MODEL_NAME, DEVICE, VAD_PARAMETERS

# 导入日志工具
from src.utils.logging_utils import get_logger

load_dotenv()  # 加载 .env 文件中的环境变量

# 获取当前模块的logger
logger = get_logger(__name__)

# 常量定义
DEFAULT_AUDIO_FORMAT = "float32"
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_BEAM_SIZE = 5
DEFAULT_TEMPERATURE = 0.0
DEFAULT_COMPRESSION_RATIO_THRESHOLD = 2.4
DEFAULT_LOG_PROB_THRESHOLD = -1.0
DEFAULT_NO_SPEECH_THRESHOLD = 0.6

# 性能优化常量
MAX_AUDIO_BUFFER_SIZE = 30  # 最大音频缓冲区大小（秒）
MIN_SEGMENT_DURATION = 0.5  # 最小段长度（秒）
TIMESTAMP_FORMAT = "%H:%M:%S.%f"  # 时间戳格式
TIMESTAMP_PRECISION = 3  # 时间戳精度（毫秒）

def format_timestamp(precision: int = TIMESTAMP_PRECISION) -> str:
    """格式化当前时间戳为字符串"""
    return datetime.now().strftime(TIMESTAMP_FORMAT)[:-precision]

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
    内部缓存类，封装流式ASR处理的状态属性
    
    Attributes:
        segment_queue: 音频段队列，存储待处理的音频段
        segment_counter: 段计数器，用于生成唯一ID
        current_recognition: 当前识别结果缓存
        current_segments_count: 当前处理的段数
        total_duration: 队列中音频的总时长（秒）
        transcription_in_progress: 转录是否正在进行中
    """
    def __init__(self):
        self._lock = threading.Lock()
        self.segment_queue: List["ASRAudioSegment"] = []
        self.waiting_segment_queue: List["ASRAudioSegment"] = []
        self.segment_counter: int = 0
        self.current_recognition: Optional[Dict] = None
        self.current_segments_count: int = 0
        self.total_duration: float = 0.0  # 缓存总时长，避免重复计算
        self.transcription_in_progress: bool = False  # 转录是否正在进行中
    
    def add_segment(self, segment: "ASRAudioSegment") -> None:
        """添加音频段到队列并更新总时长"""
        with self._lock:
            self.waiting_segment_queue.append(segment)
    
    def add_to_asr_segments(self) -> None:
        """添加音频段到队列并更新总时长"""
        with self._lock:
            self.segment_queue.extend(self.waiting_segment_queue)
            self.total_duration += sum(segment.duration for segment in self.waiting_segment_queue)  
            self.segment_counter += len(self.waiting_segment_queue)
            self.waiting_segment_queue.clear()
        
    def remove_first_segment(self) -> Optional["ASRAudioSegment"]:
        """移除队列首段并更新总时长"""
        if not self.segment_queue:
            return None
        segment = self.segment_queue.pop(0)
        return segment
    
    def set_processing(self) -> bool:
        """设置转录处理状态，返回是否成功"""
        with self._lock:
            if not self.transcription_in_progress:
                self.transcription_in_progress = True
                return True
            return False

    def set_processed(self) -> None:
        """清除转录处理状态"""
        self.transcription_in_progress = False

    def is_processing(self) -> bool:
        """判断是否正在处理"""
        return self.transcription_in_progress

    def should_process(self, recognition_threshold: float, prefix_segments: int, suffix_segments_atleast: int, is_final: bool) -> bool:
        """判断是否应该处理当前队列中的音频段"""
        
        queue_length = len(self.segment_queue)

        if is_final:
            return True
        
        # 检查是否达到识别阈值
        if self.total_duration < recognition_threshold:
            return False
            
        # 检查语音段数是否满足最低要求（前缀段数 + 本身 + 后续最少段数）
        if queue_length < prefix_segments + suffix_segments_atleast + 1:
            return False
            
        return True
        
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
        suffix_segments_atleast: int = 1, # 最少后缀段数
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
        
        # 存储设备类型
        self._device = device.lower()
        
        # 根据设备自动选择计算类型
        if compute_type == 'auto':
            if self._device == 'cpu':
                compute_type = 'int8'  # CPU使用int8
                logger.debug(f'Auto-selected compute type for CPU: {compute_type}')
            else:
                compute_type = 'float16'  # GPU使用float16
                logger.debug(f'Auto-selected compute type for GPU: {compute_type}')
        
        # 加载Whisper模型
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        
        # 存储模型大小以备后用
        self._model_size = model_size
        
        # 配置参数
        self.recognition_threshold = recognition_threshold
        self.sample_rate = sample_rate
        self.prefix_segments = prefix_segments
        self.suffix_segments_atleast = suffix_segments_atleast
        
        # 用于记录详细延迟的变量
        self.timing_events: Dict[TimingEventType, float] = {}
        
        # 性能优化：预分配音频缓冲区
        self._audio_buffer = np.array([], dtype=np.float32)
        self._max_buffer_size = MAX_AUDIO_BUFFER_SIZE * self.sample_rate

        logger.info('ASR model loaded successfully')

    def get_last_timings(self) -> Dict[TimingEventType, float]:
        """获取最后一次处理的时间事件记录"""
        return self.timing_events.copy()  # 返回副本以避免外部修改
        
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        获取性能指标
        
        Returns:
            Dict[str, float]: 包含各种性能指标的字典
        """
        if not self.timing_events:
            return {}
            
        metrics = {}
        
        # 计算各阶段耗时
        if TimingEventType.START_FUNCTION in self.timing_events and TimingEventType.END_FUNCTION in self.timing_events:
            metrics['total_processing_time'] = (
                self.timing_events[TimingEventType.END_FUNCTION] -
                self.timing_events[TimingEventType.START_FUNCTION]
            )
            
        if TimingEventType.START_ASR in self.timing_events and TimingEventType.END_ASR in self.timing_events:
            metrics['asr_processing_time'] = (
                self.timing_events[TimingEventType.END_ASR] -
                self.timing_events[TimingEventType.START_ASR]
            )
            
        if TimingEventType.START_TRANSCRIBE in self.timing_events and TimingEventType.END_TRANSCRIBE in self.timing_events:
            metrics['transcription_time'] = (
                self.timing_events[TimingEventType.END_TRANSCRIBE] -
                self.timing_events[TimingEventType.START_TRANSCRIBE]
            )
            
        # 计算延迟指标
        if 'total_processing_time' in metrics and 'transcription_time' in metrics:
            metrics['overhead_time'] = metrics['total_processing_time'] - metrics['transcription_time']
            
        return metrics
        
    def log_performance_metrics(self, segment_count: int, audio_duration: float) -> None:
        """
        记录性能指标到日志
        
        Args:
            segment_count: 处理的段数
            audio_duration: 音频总时长（秒）
        """
        metrics = self.get_performance_metrics()
        
        if not metrics:
            return
            
        logger.info(f"{format_timestamp()} 性能指标 - 段数: {segment_count}, "
                   f"音频时长: {audio_duration:.2f}s")
        
        if 'transcription_time' in metrics:
            transcription_time = metrics['transcription_time']
            real_time_factor = transcription_time / audio_duration if audio_duration > 0 else 0
            logger.info(f"  转录耗时: {transcription_time:.3f}s, 实时率: {real_time_factor:.2f}")
            
        if 'total_processing_time' in metrics:
            logger.info(f"  总处理时间: {metrics['total_processing_time']:.3f}s")
            
        if 'overhead_time' in metrics:
            logger.info(f"  系统开销: {metrics['overhead_time']:.3f}s")

    def generate_performance_report(self, audio_duration: float) -> Dict[str, Union[str, float, int]]:
        """
        生成详细的性能报告
        
        Args:
            audio_duration: 音频总时长（秒）
            
        Returns:
            Dict[str, Union[str, float, int]]: 包含详细性能指标的字典
        """
        metrics = self.get_performance_metrics()
        
        report = {
            'model_info': {
                'model_size': getattr(self.model, 'model_size', 'unknown'),
                'device': getattr(self.model, 'device', 'unknown'),
                'compute_type': getattr(self.model, 'compute_type', 'unknown')
            },
            'processing_metrics': metrics,
            'audio_info': {
                'duration_seconds': audio_duration,
                'sample_rate': self.sample_rate
            },
            'configuration': {
                'recognition_threshold': self.recognition_threshold,
                'prefix_segments': self.prefix_segments
            }
        }
        
        # 计算额外指标
        if 'transcription_time' in metrics and audio_duration > 0:
            report['performance_metrics']['real_time_factor'] = (
                metrics['transcription_time'] / audio_duration
            )
            report['performance_metrics']['processing_speed'] = (
                audio_duration / metrics['transcription_time'] if metrics['transcription_time'] > 0 else 0
            )
            
        return report

    def reset_timings(self) -> None:
        """重置时间事件记录"""
        self.timing_events.clear()
    
    def transcribe_complete_audio(self, audio_path: str, audio_data: Optional[np.ndarray] = None, sample_rate: Optional[int] = None) -> Dict:
        """
        完整音频文件转录（用于基线系统）
        
        Args:
            audio_path: 音频文件路径
            audio_data: 可选的已加载音频数据，如果提供则不会重新加载
            sample_rate: 可选的音频采样率，与audio_data一起提供
            
        Returns:
            转录结果字典，包含文本和时序信息
        """
        try:
            import librosa
            
            # 记录开始时间
            start_time = time.time()
            
            # 如果提供了音频数据，直接使用；否则加载音频文件
            if audio_data is not None:
                if sample_rate is None:
                    raise ValueError("如果提供了audio_data，则必须提供sample_rate")
                
                # 确保音频数据是float32格式
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32)
                
                # 如果是立体声，转换为单声道
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.mean(axis=1)
                
                audio_duration = len(audio_data) / sample_rate
                load_time = 0  # 因为没有重新加载，所以加载时间为0
                
                logger.debug(f"使用提供的音频数据: shape={audio_data.shape}, sample_rate={sample_rate}, duration={audio_duration:.2f}s")
            else:
                # 加载音频文件
                audio_data, sr = librosa.load(audio_path, sr=self.sample_rate)
                sample_rate = int(sr)  # 确保类型为int
                audio_duration = len(audio_data) / sample_rate
                load_time = time.time() - start_time
                
                logger.info(f"加载音频文件: {audio_path}, 时长: {audio_duration:.2f}s")
            
            # 使用Whisper进行完整转录
            transcription_start = time.time()
            segments_result, info = self.model.transcribe(
                audio_data,
                beam_size=DEFAULT_BEAM_SIZE,  # 基线系统使用较高beam size以获得更好质量
                # language="zh",
                word_timestamps=True,
                vad_filter=False,
                temperature=DEFAULT_TEMPERATURE,
                compression_ratio_threshold=DEFAULT_COMPRESSION_RATIO_THRESHOLD,
                log_prob_threshold=DEFAULT_LOG_PROB_THRESHOLD,
                no_speech_threshold=DEFAULT_NO_SPEECH_THRESHOLD
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
                    'total_time': time.time() - start_time,
                    'real_time_factor': transcription_time / audio_duration if audio_duration > 0 else 0
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
    
    def transcribe_audio_segment(self, cache: ASRCache) -> Tuple[ASRCache, Optional[str], bool]:
        """
        添加音频块并返回新的转录文本，输出文本可能为空
        
        Args:
            cache: ASR缓存对象
            audio_segment: 音频信号段ASRAudioSegment对象
            
        Returns:
            Tuple[ASRCache, Optional[str]]: 更新后的缓存对象和新输出的文本,是否最后一个音频段
            
        Notes:
            音频块必须是 float32, 单声道, 16kHz
            例如： np.ndarray = np.random.randn(16000).astype(np.float32)
        """
        # 性能监控：记录函数开始时间
        self.reset_timings()
        self.timing_events[TimingEventType.START_FUNCTION] = time.perf_counter()

        # 设置处理中，避免多线程竞争
        if not cache.set_processing():
            return cache, None, False

        cache.add_to_asr_segments()  # 将等待中的音频段添加到ASR段队列
        audio_segment = cache.segment_queue[-1]  # 获取最新添加的音频段
         
        # 使用ASRCache的新方法添加段
        logger.debug(f"最新音频段 {audio_segment.id} 到队列，当前队列长度: {len(cache.segment_queue)}")

       # 检查是否应该处理当前队列
        if not cache.should_process(self.recognition_threshold, self.prefix_segments, self.suffix_segments_atleast, audio_segment.is_final):
            cache.set_processed()
            return cache, None, False
   
        # 开始进行语音识别

        try:
            self.timing_events[TimingEventType.START_ASR] = time.perf_counter()
            logger.debug(f"开始处理音频队列，包含 {len(cache.segment_queue)} 个段，总时长: {cache.total_duration:.2f}s")

            current_recognition = self._transcribe_segments(cache.segment_queue)
            
            if current_recognition["segments"] is None or len(current_recognition["segments"]) == 0:
                logger.warning("当前累积音频段的识别结果为空")
                return cache, None, False

            # 记录每个段的识别结果
            segment_start_offset = 0.0
            for i, segment in enumerate[ASRAudioSegment](cache.segment_queue):
                segment.text = self._extract_segment_text(current_recognition, i, segment, segment_start_offset)
                segment_start_offset += segment.duration

                if not segment.text:
                    logger.debug(f"段 {i} ({segment.id}) 没有提取到文本")

            # 确定要输出的段索引
            output_segment_indices = self._determine_output_segments(cache, audio_segment)
            logger.debug(f"本次输出的段为: {[cache.segment_queue[i].id for i in output_segment_indices]}")
            
            # 提取输出文本
            output_text = self._extract_output_text(cache, output_segment_indices)
            
            # 记录性能指标
            processing_time = time.perf_counter() - self.timing_events[TimingEventType.START_ASR]
            logger.info(f"{format_timestamp()} ✓ 处理完成: 输出文本长度 {len(output_text)} 字符，耗时 {processing_time:.3f}s")

            # 删除已处理的段
            if output_segment_indices:
                # 确定在已输出的段中要保留的段序号
                keep_segment: ASRAudioSegment = cache.segment_queue[output_segment_indices[-1] - self.prefix_segments + 1]
                # 删除保留段数以前的所有段
                for _ in range(len(cache.segment_queue)):  
                    if keep_segment.id == cache.segment_queue[0].id:
                        break
                    cache.remove_first_segment()

            # 记录结束时间
            self.timing_events[TimingEventType.END_ASR] = time.perf_counter()
            self.timing_events[TimingEventType.END_FUNCTION] = time.perf_counter()
            
            # 记录详细性能指标
            self.log_performance_metrics(len(cache.segment_queue), cache.total_duration)

            return cache, output_text, audio_segment.is_final
        finally:
            cache.set_processed()
        
    def _determine_output_segments(self, cache: ASRCache, current_segment: ASRAudioSegment) -> List[int]:
        """
        确定应该输出的段索引
        
        Args:
            cache: ASR缓存对象
            current_segment: 当前处理的音频段
            
        Returns:
            List[int]: 应该输出的段索引列表
        """
        queue_length = len(cache.segment_queue)
        
        # 如果是流式结尾，输出所有剩余段
        if current_segment.is_final:
            return list(range(self.prefix_segments, queue_length))

        # 如果是流式开始，输出保留段数之前的所有段
        if cache.segment_queue[0].is_start:
            logger.debug(f"queue_length: {queue_length}, suffix_segments_atleast: {self.suffix_segments_atleast},selecting {list(range(0, queue_length - self.suffix_segments_atleast))}")
            return list(range(0, queue_length - self.suffix_segments_atleast))
        
        # 一般情况：输出中间段，不要输出前缀和后缀段
        return list(range(self.prefix_segments, queue_length - self.suffix_segments_atleast))
        
    def _extract_output_text(self, cache: ASRCache, output_indices: List[int]) -> str:
        """
        从缓存中提取指定索引的文本并合并
        
        Args:
            cache: ASR缓存对象
            output_indices: 要输出的段索引列表
            
        Returns:
            str: 合并后的文本
        """
        if not output_indices:
            return ""
            
        # 输出队列中选定的段，确保所有元素都是字符串（将 None 转为空串）
        output_text = ''.join([cache.segment_queue[i].text or '' for i in output_indices])
        
        # 记录详细信息
        if output_text:
            first_seg = cache.segment_queue[output_indices[0]]
            last_seg = cache.segment_queue[output_indices[-1]]
            logger.debug(f"{format_timestamp()} ✓ 输出文本: 索引{output_indices} "
                        f"[{first_seg.start_time:.2f}s-{last_seg.end_time:.2f}s] '{output_text}'")
        
        return output_text

    def _transcribe_segments(self, segments: List[ASRAudioSegment]) -> Dict:
        """
        将队列中的音频段合并后转录音频段
        
        Args:
            segments: 要转录的音频段列表
            
        Returns:
            Dict: 转录结果，包含段信息和词级时间戳
        """
        # 性能优化：使用预分配缓冲区
        total_samples = sum(len(seg.audio_data) for seg in segments)
        combined_audio = np.empty(total_samples, dtype=DEFAULT_AUDIO_FORMAT)
        
        # 高效合并音频段
        offset = 0
        for seg in segments:
            seg_samples = len(seg.audio_data)
            combined_audio[offset:offset+seg_samples] = seg.audio_data
            offset += seg_samples
        
        total_duration = len(combined_audio) / self.sample_rate
        logger.debug(f"{format_timestamp()} 正在转录 {len(segments)} 个语音段，总时长: {total_duration:.2f}s")

        # 记录转录开始时间
        self.timing_events[TimingEventType.START_TRANSCRIBE] = time.perf_counter()
        
        
        segments_result, info = self.model.transcribe(
            combined_audio,
            beam_size=DEFAULT_BEAM_SIZE,  # 使用常量
            # language="zh",  # 明确指定中文，避免语言检测开销
            word_timestamps=True,  # 启用词级时间戳，用于精确匹配
            vad_filter=False,  # 关闭VAD过滤，因为我们已经用分段器处理过了
            temperature=DEFAULT_TEMPERATURE,  # 使用常量
            compression_ratio_threshold=DEFAULT_COMPRESSION_RATIO_THRESHOLD,
            log_prob_threshold=DEFAULT_LOG_PROB_THRESHOLD,
            no_speech_threshold=DEFAULT_NO_SPEECH_THRESHOLD
        )
            
        # 处理转录结果
        transcription_segments = []
        for i, segment in enumerate(segments_result):
            segment_text = segment.text.strip()
            logger.debug(f"  段 {i+1}: [{segment.start:.2f}s-{segment.end:.2f}s] {segment_text}")
            
            # 性能优化：只在有词级时间戳时才处理
            words = []
            if segment.words:
                words = [
                    {
                        'text': word.word,
                        'start': word.start,
                        'end': word.end,
                        'probability': word.probability
                    }
                    for word in segment.words
                ]
            
            transcription_segments.append({
                'text': segment_text,
                'start': segment.start,
                'end': segment.end,
                'words': words
            })

        # 记录转录结束时间
        self.timing_events[TimingEventType.END_TRANSCRIBE] = time.perf_counter()
        
        # 计算转录耗时
        transcription_time = (self.timing_events[TimingEventType.END_TRANSCRIBE] -
                            self.timing_events[TimingEventType.START_TRANSCRIBE])
        
        result = {
            'segments': transcription_segments,
            'info': {
                'language': info.language,
                'language_probability': info.language_probability,
                'duration': info.duration
            },
            'timing': {
                'transcription_time': transcription_time,
                'real_time_factor': transcription_time / total_duration if total_duration > 0 else 0
            }
        }
        
        logger.debug(f"{format_timestamp()} 转录完成，耗时: {transcription_time:.2f}s，实时率: {result['timing']['real_time_factor']:.2f}")
        return result

    def _extract_segment_text(self, recognition_result: Dict, segment_index: int,
                             segment: ASRAudioSegment, segment_start_offset: float) -> Optional[str]:
        """
        从识别结果中提取指定音频段的文本，使用优化的词级时间戳匹配算法
        
        Args:
            recognition_result: 转录结果字典
            segment_index: 段索引
            segment: 音频段对象
            segment_start_offset: 段在合并音频中的开始时间偏移
            
        Returns:
            Optional[str]: 提取的文本，如果没有匹配到则返回None
        """
        transcription_segments = recognition_result['segments']
        
        # 计算目标段在合并音频中的时间范围
        segment_end_offset = segment_start_offset + segment.duration
        logger.debug(f"{format_timestamp()}    目标段 {segment_index+1} ({segment.id}): "
                    f"合并音频中时间范围 [{segment_start_offset:.2f}s - {segment_end_offset:.2f}s]")
        
        # 收集目标段对应的词
        matched_words = []
        
        for trans_seg in transcription_segments:
            # 检查是否有词级时间戳
            if 'words' not in trans_seg or not trans_seg['words']:
                logger.warning("转录结果中缺少词级时间戳，使用段级匹配")
                # 如果没有词级时间戳，使用段级时间戳进行匹配
                seg_center = (trans_seg['start'] + trans_seg['end']) / 2
                if segment_start_offset <= seg_center <= segment_end_offset:
                    matched_words.append(trans_seg['text'].strip())
                    logger.debug(f"      段级匹配: {trans_seg['text'].strip()}")
                continue
            
            # 使用优化的词级时间戳匹配算法
            for word in trans_seg['words']:
                word_start = word['start']
                word_end = word['end']
                
                # 优化匹配策略：使用词的重叠度进行匹配
                # 计算词与目标段的重叠度
                overlap_start = max(word_start, segment_start_offset)
                overlap_end = min(word_end, segment_end_offset)
                overlap_duration = max(0, overlap_end - overlap_start)
                word_duration = word_end - word_start

                if word_end >= segment_start_offset and word_end <= segment_end_offset:
                    matched_words.append(word['text'])
                    logger.debug(f"      词匹配: {word['text']} [{word_start:.2f}s-{word_end:.2f}s] ")
                        
                # # 如果重叠度超过词的50%，则归属于该段
                # if word_duration > 0 and (overlap_duration / word_duration) >= 0.5:
                #     matched_words.append(word['text'])
                #     logger.debug(f"      词匹配: {word['text']} [{word_start:.2f}s-{word_end:.2f}s] "
                #                f"重叠度: {overlap_duration/word_duration:.2f}")
        
        if matched_words:
            result_text = ''.join(matched_words).strip()
            logger.debug(f"{format_timestamp()}    段 {segment_index+1} ({segment.id}) 提取的文本: '{result_text}'")
            return result_text
        else:
            logger.debug(f"    段 {segment_index+1} ({segment.id}) 没有匹配到文本")
            return None


