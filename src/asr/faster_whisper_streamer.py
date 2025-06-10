# src/asr/faster_whisper_streamer.py

from faster_whisper import WhisperModel
import numpy as np
import time
import hashlib
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import librosa
from scipy import signal

# 从配置导入
from src.config import ASR_MODEL_NAME, DEVICE, VAD_PARAMETERS

@dataclass
class AudioSegment:
    """音频段数据结构"""
    id: str
    audio_data: np.ndarray
    start_time: float
    end_time: float
    duration: float
    is_processed: bool = False

@dataclass
class TranscriptionResult:
    """转录结果数据结构"""
    segment_id: str
    text: str
    start_time: float
    end_time: float
    words: List[Dict] = None

class StreamingASRProcessor:
    def __init__(
        self,
        model_size=ASR_MODEL_NAME,
        device=DEVICE,
        compute_type='float16',
        min_chunk_duration=3.0,  # 最小处理长度(秒)
        context_pre_duration=1.0,  # 前置上下文时长(秒)
        context_post_duration=1.0,  # 后置上下文时长(秒)
        vad_aggressiveness=3,  # VAD敏感度 0-3
        sample_rate=16000
    ):
        """
        初始化流式ASR处理器
        
        Args:
            model_size (str): faster-whisper 模型大小
            device (str): 推理设备
            compute_type (str): 计算类型
            min_chunk_duration (float): 开始ASR识别的最小音频长度(秒)
            context_pre_duration (float): 前置上下文时长(秒)
            context_post_duration (float): 后置上下文时长(秒)
            vad_aggressiveness (int): VAD敏感度 0-3
            sample_rate (int): 音频采样率
        """
        print(f'正在加载ASR模型: {model_size} 设备: {device} 计算类型: {compute_type}')
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        
        # 配置参数
        self.min_chunk_duration = min_chunk_duration
        self.context_pre_duration = context_pre_duration
        self.context_post_duration = context_post_duration
        self.sample_rate = sample_rate
        
        # VAD设置
        self.vad_aggressiveness = vad_aggressiveness
        self.frame_duration_ms = 30  # VAD帧长度(ms)
        self.frame_size = int(sample_rate * self.frame_duration_ms / 1000)
        
        # 基于能量的VAD参数
        self.energy_threshold = 0.01 * (vad_aggressiveness + 1)  # 动态调整阈值
        self.min_speech_frames = 3  # 最少连续语音帧数
        
        # 音频段管理
        self.audio_segments: List[AudioSegment] = []
        self.segment_counter = 0
        self.current_audio_time = 0.0
        
        # 结果管理
        self.final_results: List[TranscriptionResult] = []
        self.result_cache: Dict[str, Dict] = {}  # 缓存ASR结果
        
        # 状态标记
        self.is_recording_ended = False
        
        print('ASR模型加载完成')

    def _generate_segment_id(self) -> str:
        """生成音频段ID"""
        self.segment_counter += 1
        return f"seg_{self.segment_counter}"

    def _create_cache_key(self, segment_ids: List[str]) -> str:
        """为音频段组合创建缓存键"""
        combined_ids = "_".join(sorted(segment_ids))
        return hashlib.md5(combined_ids.encode()).hexdigest()

    def _detect_speech_segments(self, audio_data: np.ndarray, start_time: float) -> List[AudioSegment]:
        """使用基于能量的VAD检测语音段"""
        segments = []
        
        # 确保音频是float32格式
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # 预处理：高通滤波去除低频噪声
        b, a = signal.butter(5, 300 / (self.sample_rate / 2), 'high')
        audio_filtered = signal.filtfilt(b, a, audio_data)
        
        # 按帧计算能量
        frame_energies = []
        frame_positions = []
        
        for i in range(0, len(audio_filtered) - self.frame_size + 1, self.frame_size // 2):  # 50%重叠
            frame = audio_filtered[i:i + self.frame_size]
            # 计算RMS能量
            energy = np.sqrt(np.mean(frame ** 2))
            frame_energies.append(energy)
            frame_positions.append(i)
        
        if not frame_energies:
            return segments
        
        # 自适应阈值：使用能量分布的统计特性
        energy_array = np.array(frame_energies)
        energy_mean = np.mean(energy_array)
        energy_std = np.std(energy_array)
        
        # 动态阈值计算
        base_threshold = energy_mean + energy_std * 0.5
        adaptive_threshold = max(base_threshold, self.energy_threshold)
        
        # VAD决策：标记语音帧
        speech_frames = []
        for i, energy in enumerate(frame_energies):
            is_speech = energy > adaptive_threshold
            if is_speech:
                speech_frames.append((frame_positions[i], frame_positions[i] + self.frame_size))
        
        if not speech_frames:
            return segments
        
        # 平滑处理：去除孤立的帧
        smoothed_frames = []
        for i, (start, end) in enumerate(speech_frames):
            # 检查前后帧
            context_count = 0
            for j in range(max(0, i-2), min(len(speech_frames), i+3)):
                if j != i:
                    context_count += 1
            
            # 如果有足够的上下文，保留这个帧
            if context_count >= 1 or len(speech_frames) < 3:
                smoothed_frames.append((start, end))
        
        if not smoothed_frames:
            return segments
        
        # 合并连续的语音帧
        merged_segments = []
        current_start = smoothed_frames[0][0]
        current_end = smoothed_frames[0][1]
        
        for frame_start, frame_end in smoothed_frames[1:]:
            # 如果间隔小于200ms，合并
            gap = frame_start - current_end
            if gap <= 0.2 * self.sample_rate:  
                current_end = frame_end
            else:
                merged_segments.append((current_start, current_end))
                current_start = frame_start
                current_end = frame_end
        
        merged_segments.append((current_start, current_end))
        
        # 创建AudioSegment对象
        for seg_start, seg_end in merged_segments:
            # 扩展边界，包含更多上下文
            extended_start = max(0, seg_start - int(0.1 * self.sample_rate))  # 前100ms
            extended_end = min(len(audio_data), seg_end + int(0.1 * self.sample_rate))  # 后100ms
            
            seg_audio = audio_data[extended_start:extended_end].astype(np.float32)
            seg_duration = len(seg_audio) / self.sample_rate
            
            # 过滤太短的段
            if seg_duration >= 0.5:  # 最小0.5秒
                segment = AudioSegment(
                    id=self._generate_segment_id(),
                    audio_data=seg_audio,
                    start_time=start_time + extended_start / self.sample_rate,
                    end_time=start_time + extended_end / self.sample_rate,
                    duration=seg_duration
                )
                segments.append(segment)
        
        return segments

    def add_audio_chunk(self, audio_chunk: np.ndarray) -> List[TranscriptionResult]:
        """
        添加音频块并返回新的转录结果
        
        Args:
            audio_chunk (np.ndarray): 音频数据 (float32, 单声道, 16kHz)
            
        Returns:
            List[TranscriptionResult]: 新的转录结果列表
        """
        # 检测语音段
        new_segments = self._detect_speech_segments(audio_chunk, self.current_audio_time)
        self.audio_segments.extend(new_segments)
        
        # 更新时间
        self.current_audio_time += len(audio_chunk) / self.sample_rate
        
        # 处理可以识别的音频段
        new_results = self._process_ready_segments()
        
        return new_results

    def _calculate_total_duration(self, segments: List[AudioSegment]) -> float:
        """计算音频段总时长"""
        return sum(seg.duration for seg in segments)

    def _get_context_segments(self, target_idx: int) -> Tuple[List[AudioSegment], AudioSegment, List[AudioSegment]]:
        """获取目标段的前置、目标、后置段"""
        target_segment = self.audio_segments[target_idx]
        
        # 前置段
        pre_segments = []
        pre_duration = 0.0
        for i in range(target_idx - 1, -1, -1):
            if pre_duration >= self.context_pre_duration:
                break
            pre_segments.insert(0, self.audio_segments[i])
            pre_duration += self.audio_segments[i].duration
        
        # 后置段
        post_segments = []
        post_duration = 0.0
        for i in range(target_idx + 1, len(self.audio_segments)):
            if post_duration >= self.context_post_duration:
                break
            post_segments.append(self.audio_segments[i])
            post_duration += self.audio_segments[i].duration
        
        return pre_segments, target_segment, post_segments

    def _can_process_segment(self, target_idx: int) -> bool:
        """检查是否可以处理指定的音频段"""
        if target_idx >= len(self.audio_segments):
            return False
        
        if self.audio_segments[target_idx].is_processed:
            return False
        
        # 计算到目标段为止的总时长
        total_duration = sum(seg.duration for seg in self.audio_segments[:target_idx + 1])
        if total_duration < self.min_chunk_duration:
            return False
        
        # 检查是否有足够的后置上下文（除非录音已结束）
        pre_segments, target_segment, post_segments = self._get_context_segments(target_idx)
        post_duration = sum(seg.duration for seg in post_segments)
        
        # 如果录音未结束且后置上下文不足，不处理
        if not self.is_recording_ended and post_duration < self.context_post_duration:
            return False
        
        return True

    def _combine_audio_segments(self, segments: List[AudioSegment]) -> np.ndarray:
        """合并多个音频段"""
        if not segments:
            return np.array([], dtype=np.float32)
        
        combined = np.concatenate([seg.audio_data for seg in segments])
        return combined.astype(np.float32)

    def _transcribe_with_cache(self, segments: List[AudioSegment], target_segment: AudioSegment) -> Dict:
        """使用缓存进行转录"""
        segment_ids = [seg.id for seg in segments]
        cache_key = self._create_cache_key(segment_ids)
        
        # 检查缓存
        if cache_key in self.result_cache:
            print(f"使用缓存结果 for segments: {segment_ids}")
            return self.result_cache[cache_key]
        
        # 执行转录
        combined_audio = self._combine_audio_segments(segments)
        
        print(f"正在转录音频段: {segment_ids}, 总时长: {len(combined_audio)/self.sample_rate:.2f}s")
        
        start_time = time.time()
        segments_result, info = self.model.transcribe(
            combined_audio,
            beam_size=5,
            language=None,  # 自动检测
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=VAD_PARAMETERS
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
        
        # 缓存结果
        self.result_cache[cache_key] = result
        
        print(f"转录完成，耗时: {transcription_time:.2f}s, 语言: {info.language} (概率: {info.language_probability:.2f})")
        
        return result

    def _extract_target_result(self, transcription_result: Dict, target_segment: AudioSegment, 
                              all_segments: List[AudioSegment]) -> Optional[TranscriptionResult]:
        """从转录结果中提取目标段的结果"""
        
        # 计算目标段在合并音频中的时间偏移
        pre_duration = 0.0
        target_idx_in_combined = -1
        
        for i, seg in enumerate(all_segments):
            if seg.id == target_segment.id:
                target_idx_in_combined = i
                break
            pre_duration += seg.duration
        
        if target_idx_in_combined == -1:
            return None
        
        # 在转录结果中找到对应的段
        target_start_time = pre_duration
        target_end_time = pre_duration + target_segment.duration
        
        matched_texts = []
        matched_words = []
        
        for seg_result in transcription_result['segments']:
            seg_start = seg_result['start']
            seg_end = seg_result['end']
            
            # 检查是否与目标时间范围重叠
            if seg_end > target_start_time and seg_start < target_end_time:
                # 计算重叠比例
                overlap_start = max(seg_start, target_start_time)
                overlap_end = min(seg_end, target_end_time)
                overlap_duration = overlap_end - overlap_start
                seg_duration = seg_end - seg_start
                
                if overlap_duration > seg_duration * 0.5:  # 超过50%重叠
                    matched_texts.append(seg_result['text'])
                    if seg_result.get('words'):
                        for word in seg_result['words']:
                            if word['end'] > target_start_time and word['start'] < target_end_time:
                                # 调整词级时间戳到绝对时间
                                adjusted_word = word.copy()
                                adjusted_word['start'] = target_segment.start_time + (word['start'] - target_start_time)
                                adjusted_word['end'] = target_segment.start_time + (word['end'] - target_start_time)
                                matched_words.append(adjusted_word)
        
        if matched_texts:
            combined_text = ' '.join(matched_texts).strip()
            return TranscriptionResult(
                segment_id=target_segment.id,
                text=combined_text,
                start_time=target_segment.start_time,
                end_time=target_segment.end_time,
                words=matched_words
            )
        
        return None

    def _process_ready_segments(self) -> List[TranscriptionResult]:
        """处理准备好的音频段"""
        new_results = []
        
        # 从前往后检查可以处理的段
        for i in range(len(self.audio_segments)):
            if not self._can_process_segment(i):
                continue
            
            target_segment = self.audio_segments[i]
            if target_segment.is_processed:
                continue
            
            # 获取上下文段
            pre_segments, target_segment, post_segments = self._get_context_segments(i)
            all_segments = pre_segments + [target_segment] + post_segments
            
            # 执行转录
            transcription_result = self._transcribe_with_cache(all_segments, target_segment)
            
            # 提取目标段结果
            target_result = self._extract_target_result(transcription_result, target_segment, all_segments)
            
            if target_result:
                new_results.append(target_result)
                self.final_results.append(target_result)
                print(f"[{target_result.start_time:.2f}s -> {target_result.end_time:.2f}s] {target_result.text}")
            
            # 标记为已处理
            target_segment.is_processed = True
        
        # 清理已处理的前置段
        self._cleanup_processed_segments()
        
        return new_results

    def _cleanup_processed_segments(self):
        """清理已处理的音频段"""
        # 保留最近的一些段作为后续的前置上下文
        keep_duration = self.context_pre_duration * 2  # 保留2倍前置上下文时长
        
        # 从后往前计算需要保留的段
        keep_segments = []
        total_duration = 0.0
        
        for i in range(len(self.audio_segments) - 1, -1, -1):
            segment = self.audio_segments[i]
            keep_segments.insert(0, segment)
            total_duration += segment.duration
            
            if total_duration >= keep_duration:
                break
        
        # 更新音频段列表
        removed_count = len(self.audio_segments) - len(keep_segments)
        if removed_count > 0:
            self.audio_segments = keep_segments
            print(f"清理了 {removed_count} 个已处理的音频段")

    def finish_recording(self) -> List[TranscriptionResult]:
        """
        结束录音，处理剩余的音频段
        
        Returns:
            List[TranscriptionResult]: 最终剩余的转录结果
        """
        self.is_recording_ended = True
        print("录音结束，处理剩余音频段...")
        
        # 处理所有剩余的段
        final_results = self._process_ready_segments()
        
        print(f"流式转录完成，总共输出 {len(self.final_results)} 个结果")
        return final_results

    def get_final_transcription(self) -> str:
        """获取完整的转录文本"""
        return ' '.join([result.text for result in self.final_results])

    def get_all_results(self) -> List[TranscriptionResult]:
        """获取所有转录结果"""
        return self.final_results.copy()

# 向后兼容的别名
FasterWhisperStreamer = StreamingASRProcessor