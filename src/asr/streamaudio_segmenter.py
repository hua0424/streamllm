"""
流式音频分段器
使用 Silero VAD 进行语音活动检测，实现流式音频的实时分段
支持并发处理多个流式音频
"""

import torch
import numpy as np
from typing import Optional, Tuple
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict
import time

from src.utils.logging_utils import get_logger
logger = get_logger(__name__)

class TimingEventType(Enum):
    """时间事件类型枚举类，用于性能分析"""
    START_FUNCTION = auto()  # 函数调用开始时间
    END_FUNCTION = auto()    # 函数调用结束时间

@dataclass
class AudioSegment:
    """
    音频段结构体，包含音频数据和相关的元数据
    """
    audio: np.ndarray  # 音频数据
    segment_id: int  # 段唯一标识符
    is_speaking: bool  # 是否正在说话
    accumulated_samples: int  # 累积的采样点数
    accumulated_duration_s: float  # 累积的时长（秒）
    segment_detected: bool  # 是否检测到音频段
    abs_start_time: float  # 绝对开始时间
    abs_end_time: float  # 绝对结束时间
    segment_duration_s: float  # 段持续时间
    buffer_cleared: bool = False  # 缓冲区是否被清理
    contains_speech: bool = False  # 段内是否含语音（VAD 闭段恒 True；flush 段按残余实际判定）
    
    def is_empty(self) -> bool:
        """检查音频段是否为空"""
        return len(self.audio) == 0

@dataclass
class StreamState:
    """
    流式音频处理的状态对象
    每个音频流对应一个独立的状态对象
    """
    accumulated_audio: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    current_speech_start: float = 0.0  # 当前语音段在整个音频流的开始时间（秒）
    silence_counter: int = 0
    is_speaking: bool = False
    segment_counter: int = 0  # 用于生成唯一的segment_id
    
    def copy(self) -> 'StreamState':
        """创建状态的深拷贝（已弃用，仅用于特殊情况）"""
        return StreamState(
            accumulated_audio=self.accumulated_audio.copy(),
            current_speech_start=self.current_speech_start,
            silence_counter=self.silence_counter,
            is_speaking=self.is_speaking,
            segment_counter=self.segment_counter
        )
    
    def reset(self):
        """重置状态"""
        self.accumulated_audio = np.array([], dtype=np.float32)
        self.current_speech_start = 0.0
        self.silence_counter = 0
        self.is_speaking = False
        # 注意：不重置segment_counter，保持递增
    
    def get_next_segment_id(self) -> int:
        """获取下一个唯一的segment_id"""
        self.segment_counter += 1
        return self.segment_counter


class StreamAudioSegmenter:
    """
    流式音频分段器
    支持并发处理多个音频流
    """
    
    def __init__(
        self,
        sampling_rate: int = 16000,
        silence_threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 300,
        window_size_ms: int = 64,
        silero_model=None,
        silero_utils=None
    ):
        """
        初始化流式音频分段器

        参数:
            sampling_rate: 采样率，默认16000Hz
            silence_threshold: 静音阈值，0-1之间，默认0.5
            min_speech_duration_ms: 最小语音段时长（毫秒），默认250
            min_silence_duration_ms: 最小静音时长（毫秒），默认300
            window_size_ms: VAD窗口大小（毫秒），默认64ms
            silero_model: 可选，注入已加载并固定 revision/hash 的 Silero VAD 模型
                （W1 统一 TTFA：与 PSE 使用同一 artifact，避免二次浮动 hub 加载）
            silero_utils: 可选，与 silero_model 配套的 utils 元组（第一项须为
                get_speech_timestamps）；注入时不再访问 torch.hub
        """
        # Silero VAD 仅支持 8000 或 16000 (或 16000 的倍数) 的采样率
        if sampling_rate not in (8000, 16000):
            raise ValueError(
                f"Silero VAD 仅支持 8k/16k 采样率，收到 {sampling_rate}，"
                "请在输入前将音频重采样到 16k 或 8k。"
            )
        self.sampling_rate = sampling_rate
        self.silence_threshold = silence_threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.window_size_ms = window_size_ms
        self.silero_injected = silero_model is not None

        # 计算采样点数
        self.min_speech_samples = int(sampling_rate * min_speech_duration_ms / 1000)
        self.min_silence_samples = int(sampling_rate * min_silence_duration_ms / 1000)
        self.window_size_samples = int(sampling_rate * window_size_ms / 1000)

        if silero_model is not None:
            if silero_utils is None:
                raise ValueError("注入 silero_model 时必须同时提供 silero_utils")
            model, utils = silero_model, silero_utils
            logger.debug("Silero VAD 使用外部注入的固定模型（未访问 torch.hub）")
        else:
            # 加载Silero VAD模型（优先从本地缓存加载，避免网络访问）
            # torch.hub.load 会自动使用 ~/.cache/torch/hub/ 中的缓存，无需特殊处理
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,  # 优先使用本地缓存
                onnx=False,
                trust_repo='check',  # 如果已缓存则跳过检查
                verbose=False  # 减少输出
            )
            logger.debug("Silero VAD 模型加载成功（从本地缓存或网络）")

        self.model = model
        # 获取VAD检测函数
        (self.get_speech_timestamps, _, _, _, _) = utils

        self.timing_events:Dict[TimingEventType, float] = {}
        
        logger.info(f"StreamAudioSegmenter initialized with sampling_rate={sampling_rate}, "
                   f"silence_threshold={silence_threshold}")
    
    def get_last_timings(self):
        return self.timing_events

    def reset_timings(self):
        return self.timing_events.clear()
    
    def create_state(self) -> StreamState:
        """
        创建一个新的流状态对象
        
        返回:
            新的StreamState对象
        """
        return StreamState()
    
    def process_audio(
        self,
        audio_chunk: np.ndarray,
        state: Optional[StreamState] = None
    ) -> Tuple[Optional[AudioSegment], StreamState]:
        """
        处理音频块，返回完成的语音段和更新后的状态
        
        参数:
            audio_chunk: 新的音频数据块，numpy数组，dtype为float32，范围[-1, 1]
            state: 当前流的状态对象，如果为None则创建新状态
        
        返回:
            (audio_segment, updated_state)
            - audio_segment: 完成的语音段对象，如果没有则为None
            - updated_state: 更新后的状态对象
        """
        # 记录开始时间（用于性能分析）
        self.reset_timings()
        self.timing_events[TimingEventType.START_FUNCTION] = time.perf_counter()

        # 如果没有提供状态，创建新状态
        if state is None:
            state = self.create_state()
        
        # 确保音频数据格式正确并累积
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)
        state.accumulated_audio = np.concatenate([state.accumulated_audio, audio_chunk])
        
        # 如果累积数据不足一个窗口，直接返回
        if len(state.accumulated_audio) < self.window_size_samples:
            return None, state
        
        # 转换为torch张量
        audio_tensor = torch.from_numpy(state.accumulated_audio)
        
        # 获取语音时间戳
        speech_timestamps = self.get_speech_timestamps(
            audio_tensor, 
            self.model,
            sampling_rate=self.sampling_rate,
            threshold=self.silence_threshold,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms,
            return_seconds=False
        )
        
        # 初始化变量
        completed_segment = None
        segment_detected = False
        abs_start_time = 0.0
        abs_end_time = 0.0
        segment_duration_s = 0.0
        buffer_cleared = False
        
        if speech_timestamps:
            # 有语音活动，检查是否有完整的语音段（语音结束后有足够的静音）
            current_position = len(state.accumulated_audio)
            
            for i, timestamp in enumerate(speech_timestamps):
                segment_end = timestamp['end']
                
                # 检查段后是否有足够的静音
                silence_after = current_position - segment_end
                is_last_segment = i == len(speech_timestamps) - 1
                
                # 如果是最后一段且有足够静音，或者不是最后一段且与下一段间有足够静音
                if (is_last_segment and silence_after >= self.min_silence_samples) or \
                   (not is_last_segment and speech_timestamps[i + 1]['start'] - segment_end >= self.min_silence_samples):
                    # 找到完整段，提取并返回
                    completed_segment = state.accumulated_audio[:segment_end].copy()
                    state.accumulated_audio = state.accumulated_audio[segment_end:]
                    segment_detected = True
                    segment_duration_s = len(completed_segment) / self.sampling_rate
                    
                    # 设置时间信息
                    abs_start_time = state.current_speech_start
                    abs_end_time = state.current_speech_start + segment_duration_s
                    state.current_speech_start = abs_end_time
                    
                    break

            # 更新语音状态。
            # 注意（R3-P0-1）：闭段已消费 segment_end 之前的音频，is_speaking 必须按
            # "残余缓冲是否仍含语音"重算，不能无条件置 True——否则闭段发生在最后一次
            # process_audio 调用时，纯静音的 flush 残余会带上残留的 True。
            if segment_detected:
                state.is_speaking = any(ts['end'] > segment_end for ts in speech_timestamps)
            else:
                state.is_speaking = True
            state.silence_counter = 0
        else:
            # 没有语音活动
            state.is_speaking = False
        
        # 如果检测到完整段，创建AudioSegment对象
        if segment_detected and completed_segment is not None:
            segment_id = state.get_next_segment_id()
            accumulated_samples = len(state.accumulated_audio)
            accumulated_duration_s = len(state.accumulated_audio) / self.sampling_rate
            
            audio_segment = AudioSegment(
                audio=completed_segment,
                segment_id=segment_id,
                is_speaking=state.is_speaking,
                accumulated_samples=accumulated_samples,
                accumulated_duration_s=accumulated_duration_s,
                segment_detected=segment_detected,
                abs_start_time=abs_start_time,
                abs_end_time=abs_end_time,
                segment_duration_s=segment_duration_s,
                buffer_cleared=buffer_cleared,
                contains_speech=True  # VAD 闭段以语音时间戳结尾，构造上必含语音
            )

            # 记录结束时间并返回
            self.timing_events[TimingEventType.END_FUNCTION] = time.perf_counter()
            return audio_segment, state
        
        # 记录结束时间并返回None
        self.timing_events[TimingEventType.END_FUNCTION] = time.perf_counter()
        return None, state
    
    def flush(self, state: StreamState) -> Tuple[Optional[AudioSegment], StreamState]:
        """
        强制输出剩余的音频数据
        
        参数:
            state: 当前流的状态对象
        
        返回:
            (audio_segment, updated_state)
            - audio_segment: 剩余的音频段对象，如果没有则返回None
            - updated_state: 更新后的状态对象
        """
        if len(state.accumulated_audio) > 0:
            # 获取剩余音频数据
            remaining = state.accumulated_audio.copy()
            remaining_duration = len(remaining) / self.sampling_rate
            
            # 创建AudioSegment对象（R3-P0-1：contains_speech 按残余实际判定，
            # 修复后 state.is_speaking 准确反映残余是否含语音）
            audio_segment = AudioSegment(
                audio=remaining,
                segment_id=state.get_next_segment_id(),
                is_speaking=state.is_speaking,
                accumulated_samples=len(remaining),
                accumulated_duration_s=remaining_duration,
                segment_detected=True,
                abs_start_time=state.current_speech_start,
                abs_end_time=state.current_speech_start + remaining_duration,
                segment_duration_s=remaining_duration,
                buffer_cleared=False,
                contains_speech=state.is_speaking
            )
            
            # 重置状态并返回
            state.reset()
            return audio_segment, state
        
        return None, state