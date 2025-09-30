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

logger = logging.getLogger(__name__)


@dataclass
class StreamState:
    """
    流式音频处理的状态对象
    每个音频流对应一个独立的状态对象
    """
    accumulated_audio: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    current_speech_start: Optional[int] = None
    silence_counter: int = 0
    is_speaking: bool = False
    
    def copy(self) -> 'StreamState':
        """创建状态的深拷贝"""
        return StreamState(
            accumulated_audio=self.accumulated_audio.copy(),
            current_speech_start=self.current_speech_start,
            silence_counter=self.silence_counter,
            is_speaking=self.is_speaking
        )
    
    def reset(self):
        """重置状态"""
        self.accumulated_audio = np.array([], dtype=np.float32)
        self.current_speech_start = None
        self.silence_counter = 0
        self.is_speaking = False


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
        window_size_ms: int = 64
    ):
        """
        初始化流式音频分段器
        
        参数:
            sampling_rate: 采样率，默认16000Hz
            silence_threshold: 静音阈值，0-1之间，默认0.5
            min_speech_duration_ms: 最小语音段时长（毫秒），默认250ms
            min_silence_duration_ms: 最小静音时长（毫秒），默认300ms
            window_size_ms: VAD窗口大小（毫秒），默认64ms
        """
        self.sampling_rate = sampling_rate
        self.silence_threshold = silence_threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.window_size_ms = window_size_ms
        
        # 计算采样点数
        self.min_speech_samples = int(sampling_rate * min_speech_duration_ms / 1000)
        self.min_silence_samples = int(sampling_rate * min_silence_duration_ms / 1000)
        self.window_size_samples = int(sampling_rate * window_size_ms / 1000)
        
        # 加载Silero VAD模型
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )
        
        self.model = model
        # 获取VAD检测函数
        (self.get_speech_timestamps, _, _, _, _) = utils
        
        logger.info(f"StreamAudioSegmenter initialized with sampling_rate={sampling_rate}, "
                   f"silence_threshold={silence_threshold}")
    
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
    ) -> Tuple[Optional[np.ndarray], StreamState, dict]:
        """
        处理音频块，返回完成的语音段和更新后的状态
        
        参数:
            audio_chunk: 新的音频数据块，numpy数组，dtype为float32，范围[-1, 1]
            state: 当前流的状态对象，如果为None则创建新状态
        
        返回:
            (completed_segment, updated_state, metadata)
            - completed_segment: 完成的语音段，如果没有则为None
            - updated_state: 更新后的状态对象
            - metadata: 元数据字典，包含分段信息
        """
        # 如果没有提供状态，创建新状态
        if state is None:
            state = self.create_state()
        else:
            # 创建状态副本，避免修改原始状态
            state = state.copy()
        
        # 确保音频数据格式正确
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)
        
        # 累积音频数据
        state.accumulated_audio = np.concatenate([state.accumulated_audio, audio_chunk])
        
        # 准备返回的元数据
        metadata = {
            'is_speaking': state.is_speaking,
            'accumulated_samples': len(state.accumulated_audio),
            'accumulated_duration_s': len(state.accumulated_audio) / self.sampling_rate
        }
        
        # 如果累积数据不足一个窗口，直接返回
        if len(state.accumulated_audio) < self.window_size_samples:
            return None, state, metadata
        
        # 转换为torch张量进行VAD检测
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
        
        # 处理检测结果
        completed_segment = None
        
        if speech_timestamps:
            # 有语音活动
            current_position = len(state.accumulated_audio)
            
            # 检查是否有完整的语音段（语音结束后有足够的静音）
            if len(speech_timestamps) > 0:
                for i, timestamp in enumerate(speech_timestamps):
                    # 检查当前段后是否有足够的静音
                    segment_end = timestamp['end']
                    
                    # 如果是最后一个语音段
                    if i == len(speech_timestamps) - 1:
                        silence_after = current_position - segment_end
                        if silence_after >= self.min_silence_samples:
                            # 找到完整段，提取并返回
                            completed_segment = state.accumulated_audio[:segment_end].copy()
                            state.accumulated_audio = state.accumulated_audio[segment_end:]
                            metadata['segment_detected'] = True
                            metadata['segment_duration_s'] = len(completed_segment) / self.sampling_rate
                            break
                    else:
                        # 检查与下一段之间的静音
                        next_start = speech_timestamps[i + 1]['start']
                        if next_start - segment_end >= self.min_silence_samples:
                            # 找到完整段
                            completed_segment = state.accumulated_audio[:segment_end].copy()
                            state.accumulated_audio = state.accumulated_audio[segment_end:]
                            metadata['segment_detected'] = True
                            metadata['segment_duration_s'] = len(completed_segment) / self.sampling_rate
                            break
            
            state.is_speaking = True
            state.silence_counter = 0
        else:
            # 没有语音活动
            state.is_speaking = False
            
            # 如果累积了太多静音数据，清理缓冲区
            if len(state.accumulated_audio) > self.sampling_rate * 5:  # 5秒静音
                state.accumulated_audio = np.array([], dtype=np.float32)
                metadata['buffer_cleared'] = True
        
        metadata['is_speaking'] = state.is_speaking
        
        return completed_segment, state, metadata
    
    def flush(self, state: StreamState) -> Tuple[Optional[np.ndarray], StreamState]:
        """
        强制输出剩余的音频数据
        
        参数:
            state: 当前流的状态对象
        
        返回:
            (remaining_audio, updated_state)
            - remaining_audio: 剩余的音频数据，如果没有则返回None
            - updated_state: 更新后的状态对象
        """
        # 创建状态副本
        state = state.copy()
        
        if len(state.accumulated_audio) > 0:
            remaining = state.accumulated_audio.copy()
            state.reset()
            return remaining, state
        return None, state