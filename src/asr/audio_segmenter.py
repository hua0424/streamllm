# src/asr/audio_segmenter.py

import numpy as np
import librosa
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

# 从配置导入
from src.config import VAD_PARAMETERS, STREAMING_ASR_CHUNK_SECONDS, STREAMING_ASR_OVERLAP_SECONDS

class AudioSegmenter:
    def __init__(
        self,
        sample_rate=16000,
        min_silence_len_ms=None, # 从 VAD_PARAMETERS 获取
        silence_thresh_db=None, # VAD_PARAMETERS 中没有直接对应，需要设定或使用pydub默认
        chunk_duration_s=STREAMING_ASR_CHUNK_SECONDS,
        overlap_duration_s=STREAMING_ASR_OVERLAP_SECONDS
    ):
        """
        初始化 AudioSegmenter。

        Args:
            sample_rate (int): 音频采样率。
            min_silence_len_ms (int, optional): 用于VAD的最小静音长度（毫秒）。
                                              默认为 config.VAD_PARAMETERS['min_silence_duration_ms']。
            silence_thresh_db (int, optional): VAD的静音阈值 (dBFS)。需要经验设定，pydub默认为-16dBFS。
            chunk_duration_s (float): 固定分块策略中每个块的持续时间（秒）。
            overlap_duration_s (float): 固定分块策略中块之间的重叠持续时间（秒）。
        """
        self.sample_rate = sample_rate
        self.min_silence_len_ms = min_silence_len_ms if min_silence_len_ms is not None \
                                  else VAD_PARAMETERS.get('min_silence_duration_ms', 1000)
        # pydub 的 silence_thresh 是负值，数值越小（越接近负无穷）越灵敏
        self.silence_thresh_db = silence_thresh_db if silence_thresh_db is not None else -40 # -40dBFS 是一个经验值
        self.chunk_duration_s = chunk_duration_s
        self.overlap_duration_s = overlap_duration_s

    def segment_by_silence(self, audio_np, audio_absolute_start_time=0):
        """
        使用 VAD (基于pydub) 将音频分割成基于静默的段落。

        Args:
            audio_np (np.ndarray): 单声道 NumPy 音频数组 (float32, 范围 [-1, 1])。
            audio_absolute_start_time (float): 当前音频块在整个流中的绝对开始时间（秒）。

        Returns:
            list: 包含 (start_time_abs, end_time_abs, segment_np) 元组的列表。
                  时间戳是相对于原始完整音频流的绝对时间。
        """
        if audio_np.dtype != np.float32:
            audio_np = audio_np.astype(np.float32)
        if np.max(np.abs(audio_np)) > 1.0:
            audio_np = audio_np / np.max(np.abs(audio_np)) # 归一化到 [-1, 1]

        # 将 NumPy 数组转换为 pydub AudioSegment
        # pydub 需要 int16，所以需要转换范围并更改类型
        audio_int16 = (audio_np * 32767).astype(np.int16)
        audio_segment = AudioSegment(audio_int16.tobytes(), frame_rate=self.sample_rate, sample_width=2, channels=1)

        non_silent_ranges = detect_nonsilent(
            audio_segment,
            min_silence_len=self.min_silence_len_ms,
            silence_thresh=self.silence_thresh_db,
            seek_step=1 # ms
        )

        speech_segments = []
        for start_ms, end_ms in non_silent_ranges:
            start_sample = int(start_ms * self.sample_rate / 1000)
            end_sample = int(end_ms * self.sample_rate / 1000)
            segment_np = audio_np[start_sample:end_sample]

            abs_start_s = audio_absolute_start_time + (start_ms / 1000.0)
            abs_end_s = audio_absolute_start_time + (end_ms / 1000.0)

            speech_segments.append({
                'start_abs': abs_start_s,
                'end_abs': abs_end_s,
                'audio_np': segment_np
            })
        return speech_segments

    def segment_by_fixed_chunks(self, audio_np, audio_absolute_start_time=0):
        """
        将音频分割成固定长度的重叠块。

        Args:
            audio_np (np.ndarray): 单声道 NumPy 音频数组。
            audio_absolute_start_time (float): 当前音频块在整个流中的绝对开始时间（秒）。

        Yields:
            dict: 包含 {'start_abs': float, 'end_abs': float, 'audio_np': np.ndarray} 的字典。
                  时间戳是相对于原始完整音频流的绝对时间。
        """
        total_samples = len(audio_np)
        chunk_samples = int(self.chunk_duration_s * self.sample_rate)
        overlap_samples = int(self.overlap_duration_s * self.sample_rate)
        step_samples = chunk_samples - overlap_samples

        if chunk_samples <= 0:
            # 如果块太小或采样率为0，则返回整个音频
            if total_samples > 0:
                yield {
                    'start_abs': audio_absolute_start_time,
                    'end_abs': audio_absolute_start_time + total_samples / self.sample_rate,
                    'audio_np': audio_np
                }
            return

        current_pos_samples = 0
        while current_pos_samples < total_samples:
            start_sample_in_chunk = current_pos_samples
            end_sample_in_chunk = current_pos_samples + chunk_samples
            
            # 确保不会超出音频末尾
            actual_end_sample_in_chunk = min(end_sample_in_chunk, total_samples)
            chunk_audio_np = audio_np[start_sample_in_chunk:actual_end_sample_in_chunk]

            if len(chunk_audio_np) == 0:
                break # 防止无限循环（如果step_samples为0且total_samples也为0）
            
            abs_start_s = audio_absolute_start_time + (start_sample_in_chunk / self.sample_rate)
            abs_end_s = audio_absolute_start_time + (actual_end_sample_in_chunk / self.sample_rate)

            yield {
                'start_abs': abs_start_s,
                'end_abs': abs_end_s,
                'audio_np': chunk_audio_np
            }

            if actual_end_sample_in_chunk == total_samples:
                break # 已到达音频末尾
            
            current_pos_samples += step_samples
            if step_samples <= 0 and chunk_samples > 0 : # 防止因overlap过大或chunk_duration_s为0导致无限循环
                current_pos_samples = end_sample_in_chunk # 强制前进，避免卡住
                if current_pos_samples >= total_samples: # 如果强制前进后已经到末尾
                    break

# 示例用法 (用于测试):
if __name__ == '__main__':
    try:
        sample_rate = 16000
        # 创建一个包含语音和静音的虚拟音频
        duration_speech1 = 2 #s
        duration_silence1 = 1.5 #s
        duration_speech2 = 3 #s
        duration_silence2 = 0.5 #s
        duration_speech3 = 1 #s

        speech1 = np.sin(2 * np.pi * 300 * np.linspace(0, duration_speech1, int(duration_speech1 * sample_rate), endpoint=False)) * 0.5
        silence1 = np.zeros(int(duration_silence1 * sample_rate))
        speech2 = np.sin(2 * np.pi * 440 * np.linspace(0, duration_speech2, int(duration_speech2 * sample_rate), endpoint=False)) * 0.6
        silence2 = np.zeros(int(duration_silence2 * sample_rate))
        speech3 = np.sin(2 * np.pi * 500 * np.linspace(0, duration_speech3, int(duration_speech3 * sample_rate), endpoint=False)) * 0.4
        
        test_audio_np = np.concatenate([speech1, silence1, speech2, silence2, speech3]).astype(np.float32)
        total_duration = len(test_audio_np) / sample_rate
        print(f"Test audio created: duration={total_duration:.2f}s")

        segmenter_vad = AudioSegmenter(sample_rate=sample_rate, min_silence_len_ms=500, silence_thresh_db=-45)
        print("\n--- Segmenting by silence (VAD) ---")
        vad_segments = segmenter_vad.segment_by_silence(test_audio_np, audio_absolute_start_time=10.0) # 假设这段音频在整个流的10s处开始
        for i, seg_info in enumerate(vad_segments):
            print(f"  VAD Segment {i+1}: start_abs={seg_info['start_abs']:.2f}s, end_abs={seg_info['end_abs']:.2f}s, duration={seg_info['end_abs'] - seg_info['start_abs']:.2f}s, data_len={len(seg_info['audio_np'])}")

        segmenter_fixed = AudioSegmenter(
            sample_rate=sample_rate, 
            chunk_duration_s=2.0, 
            overlap_duration_s=0.5
        )
        print("\n--- Segmenting by fixed chunks ---")
        # 模拟一个更大的音频流，然后取其中一部分进行分块
        long_audio_prefix = np.zeros(int(5 * sample_rate)) # 假设前面有5s的音频
        current_block_audio_np = test_audio_np # 当前处理的音频块
        current_block_start_time_abs = 5.0 # 这个块在整个流中的开始时间

        for i, chunk_info in enumerate(segmenter_fixed.segment_by_fixed_chunks(current_block_audio_np, audio_absolute_start_time=current_block_start_time_abs)):
            chunk_duration = chunk_info['end_abs'] - chunk_info['start_abs']
            print(f"  Fixed Chunk {i+1}: start_abs={chunk_info['start_abs']:.2f}s, end_abs={chunk_info['end_abs']:.2f}s, duration={chunk_duration:.2f}s, data_len={len(chunk_info['audio_np'])}")

        print("\n--- Segmenting short audio by fixed chunks (edge case) ---")
        short_audio_np = speech1[:int(0.7*sample_rate)] # 0.7秒音频
        for i, chunk_info in enumerate(segmenter_fixed.segment_by_fixed_chunks(short_audio_np, audio_absolute_start_time=0.0)):
            chunk_duration = chunk_info['end_abs'] - chunk_info['start_abs']
            print(f"  Short Audio Chunk {i+1}: start_abs={chunk_info['start_abs']:.2f}s, end_abs={chunk_info['end_abs']:.2f}s, duration={chunk_duration:.2f}s, data_len={len(chunk_info['audio_np'])}")


    except Exception as e:
        import traceback
        print(f"An error occurred during the example: {e}")
        traceback.print_exc() 