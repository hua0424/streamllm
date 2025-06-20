# src/asr/faster_whisper_streamer.py

from faster_whisper import WhisperModel
import numpy as np
import time
import hashlib
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import librosa
from scipy import signal
import os
from dotenv import load_dotenv

# 从配置导入
from src.config import ASR_MODEL_NAME, DEVICE, VAD_PARAMETERS

load_dotenv() # 加载 .env 文件中的环境变量

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
        # Hugging Face 配置 (如果需要从特定端点或使用token)
        HF_ENDPOINT = os.getenv("HF_ENDPOINT")
        HF_TOKEN = os.getenv("HF_TOKEN")
        HF_HOME = os.getenv("HF_HOME") # 默认缓存路径
        print(f'HF_ENDPOINT: {HF_ENDPOINT}')
        print(f'HF_TOKEN: {HF_TOKEN}')
        print(f'HF_HOME: {HF_HOME}')

        self.model = WhisperModel(model_size, device=device)
        
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
        
        print(f"VAD处理音频块: 时长 {len(audio_data)/self.sample_rate:.2f}s, RMS: {np.sqrt(np.mean(audio_data**2)):.4f}")
        
        # 简化的VAD：直接基于能量，不使用高通滤波避免过度过滤
        # 按帧计算能量
        frame_energies = []
        frame_positions = []
        
        for i in range(0, len(audio_data) - self.frame_size + 1, self.frame_size // 2):  # 50%重叠
            frame = audio_data[i:i + self.frame_size]
            # 计算RMS能量
            energy = np.sqrt(np.mean(frame ** 2))
            frame_energies.append(energy)
            frame_positions.append(i)
        
        if not frame_energies:
            print("  没有足够的帧进行分析")
            return segments
        
        # 自适应阈值：使用能量分布的统计特性
        energy_array = np.array(frame_energies)
        energy_mean = np.mean(energy_array)
        energy_std = np.std(energy_array)
        energy_max = np.max(energy_array)
        
        print(f"  能量统计: 均值={energy_mean:.4f}, 标准差={energy_std:.4f}, 最大值={energy_max:.4f}")
        
        # 更灵敏的动态阈值计算
        if energy_max > 0.001:  # 如果有明显的信号
            # 使用更低的阈值
            base_threshold = energy_mean + energy_std * 0.2  # 降低系数从0.5到0.2
            adaptive_threshold = max(base_threshold, energy_max * 0.1)  # 使用最大值的10%作为最低阈值
        else:
            # 对于很小的信号，使用固定的很低阈值
            adaptive_threshold = 0.0001
        
        print(f"  使用阈值: {adaptive_threshold:.4f}")
        
        # VAD决策：标记语音帧
        speech_frames = []
        speech_frame_count = 0
        for i, energy in enumerate(frame_energies):
            is_speech = energy > adaptive_threshold
            if is_speech:
                speech_frames.append((frame_positions[i], frame_positions[i] + self.frame_size))
                speech_frame_count += 1
        
        print(f"  检测到 {speech_frame_count}/{len(frame_energies)} 个语音帧")
        
        if not speech_frames:
            print("  没有检测到语音帧")
            return segments
        
        # 简化平滑处理：保留所有检测到的帧
        smoothed_frames = speech_frames  # 简化：不进行平滑处理
        
        # 合并连续的语音帧
        merged_segments = []
        current_start = smoothed_frames[0][0]
        current_end = smoothed_frames[0][1]
        
        for frame_start, frame_end in smoothed_frames[1:]:
            # 如果间隔小于500ms，合并
            gap = frame_start - current_end
            if gap <= 0.5 * self.sample_rate:  # 增加合并间隔
                current_end = frame_end
            else:
                merged_segments.append((current_start, current_end))
                current_start = frame_start
                current_end = frame_end
        
        merged_segments.append((current_start, current_end))
        
        print(f"  合并后得到 {len(merged_segments)} 个语音段")
        
        # 创建AudioSegment对象
        for seg_start, seg_end in merged_segments:
            # 扩展边界，包含更多上下文
            extended_start = max(0, seg_start - int(0.2 * self.sample_rate))  # 前200ms
            extended_end = min(len(audio_data), seg_end + int(0.2 * self.sample_rate))  # 后200ms
            
            seg_audio = audio_data[extended_start:extended_end].astype(np.float32)
            seg_duration = len(seg_audio) / self.sample_rate
            
            print(f"    语音段: 开始={extended_start/self.sample_rate:.2f}s, 结束={extended_end/self.sample_rate:.2f}s, 时长={seg_duration:.2f}s")
            
            # 降低最小时长要求
            if seg_duration >= 0.3:  # 最小0.3秒
                segment = AudioSegment(
                    id=self._generate_segment_id(),
                    audio_data=seg_audio,
                    start_time=start_time + extended_start / self.sample_rate,
                    end_time=start_time + extended_end / self.sample_rate,
                    duration=seg_duration
                )
                segments.append(segment)
                print(f"      -> 创建音频段 {segment.id}")
        
        print(f"  最终返回 {len(segments)} 个音频段")
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
        
        print(f"添加了 {len(new_segments)} 个新音频段，总段数: {len(self.audio_segments)}")
        
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
            print(f"      索引超出范围: {target_idx} >= {len(self.audio_segments)}")
            return False
        
        if self.audio_segments[target_idx].is_processed:
            print(f"      段已处理")
            return False
        
        # 计算到目标段为止的总时长
        total_duration = sum(seg.duration for seg in self.audio_segments[:target_idx + 1])
        print(f"      累计时长: {total_duration:.2f}s (需要: {self.min_chunk_duration:.2f}s)")
        if total_duration < self.min_chunk_duration:
            print(f"      累计时长不足")
            return False
        
        # 检查是否有足够的后置上下文（除非录音已结束）
        pre_segments, target_segment, post_segments = self._get_context_segments(target_idx)
        post_duration = sum(seg.duration for seg in post_segments)
        
        print(f"      后置上下文: {post_duration:.2f}s (需要: {self.context_post_duration:.2f}s), 录音结束: {self.is_recording_ended}")
        
        # 如果录音未结束且后置上下文不足，不处理
        if not self.is_recording_ended and post_duration < self.context_post_duration:
            print(f"      后置上下文不足且录音未结束")
            return False
        
        print(f"      可以处理")
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
        
        if not self.audio_segments:
            return new_results
        
        print(f"检查 {len(self.audio_segments)} 个音频段是否可以处理...")
        
        # 从前往后检查可以处理的段
        for i in range(len(self.audio_segments)):
            segment = self.audio_segments[i]
            print(f"  段 {i+1}/{len(self.audio_segments)}: {segment.id} ({segment.duration:.2f}s, 已处理: {segment.is_processed})")
            
            if not self._can_process_segment(i):
                print(f"    -> 不能处理")
                continue
            
            target_segment = self.audio_segments[i]
            if target_segment.is_processed:
                print(f"    -> 已经处理过")
                continue
            
            print(f"    -> 可以处理，开始转录...")
            
            # 获取上下文段
            pre_segments, target_segment, post_segments = self._get_context_segments(i)
            all_segments = pre_segments + [target_segment] + post_segments
            
            print(f"      上下文: {len(pre_segments)} 前置 + 1 目标 + {len(post_segments)} 后置 = {len(all_segments)} 段")
            
            # 执行转录
            transcription_result = self._transcribe_with_cache(all_segments, target_segment)
            
            # 提取目标段结果
            target_result = self._extract_target_result(transcription_result, target_segment, all_segments)
            
            if target_result:
                new_results.append(target_result)
                self.final_results.append(target_result)
                print(f"      -> 转录成功: [{target_result.start_time:.2f}s -> {target_result.end_time:.2f}s] {target_result.text}")
            else:
                print(f"      -> 转录结果为空")
            
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

def main():
    """测试函数，通过指定wav文件路径测试whisper是否能正常工作"""
    import argparse
    import librosa
    
    parser = argparse.ArgumentParser(description='测试Whisper ASR')
    parser.add_argument('--wav_path', type=str, required=True, help='WAV文件路径')
    parser.add_argument('--model_size', type=str, default='base', help='模型大小')
    parser.add_argument('--chunk_size', type=int, default=16000, help='音频块大小(样本数)')
    
    args = parser.parse_args()
    
    print(f"正在测试WAV文件: {args.wav_path}")
    
    try:
        # 加载音频文件
        audio_data, sr = librosa.load(args.wav_path, sr=16000, mono=True)
        print(f"音频加载成功: 时长 {len(audio_data)/sr:.2f}s, 采样率 {sr}Hz")
        
        # 创建Whisper模型实例进行测试
        from faster_whisper import WhisperModel
        
        print(f"正在加载Whisper模型: {args.model_size}")
        HF_ENDPOINT = os.getenv("HF_ENDPOINT")
        HF_TOKEN = os.getenv("HF_TOKEN")
        HF_HOME = os.getenv("HF_HOME") # 默认缓存路径
        print(f'HF_ENDPOINT: {HF_ENDPOINT}')
        print(f'HF_TOKEN: {HF_TOKEN}')
        print(f'HF_HOME: {HF_HOME}')
        model = WhisperModel(args.model_size, device="cpu")
        
        # 直接转录整个音频文件
        print("开始转录...")
        segments, info = model.transcribe(audio_data, language="zh", beam_size=1)
        
        print(f"检测到的语言: {info.language} (概率: {info.language_probability:.2f})")
        print("转录结果:")
        print("-" * 50)
        
        full_text = ""
        import datetime
        start_time = time.time()
        start_datetime = datetime.datetime.fromtimestamp(start_time)
        print(f"开始转录时间: {start_datetime.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
        for segment in segments:
            print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-6]}[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
            full_text += segment.text
        
        print(f"转录结束时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
        print("-" * 50)
        print(f"完整转录文本: {full_text}")
        
    except Exception as e:
        print(f"测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
