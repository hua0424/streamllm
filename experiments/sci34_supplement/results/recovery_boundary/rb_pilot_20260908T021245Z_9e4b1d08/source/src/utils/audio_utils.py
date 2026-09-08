# src/utils/audio_utils.py

import numpy as np
import librosa
import soundfile as sf # 用于读写音频文件，比librosa更通用
from pydub import AudioSegment
import io

class AudioUtils:
    @staticmethod
    def load_audio(file_path, sample_rate=16000, mono=True, dtype=np.float32):
        """
        加载音频文件。

        Args:
            file_path (str): 音频文件路径。
            sample_rate (int, optional): 目标采样率。默认为 16000。
            mono (bool, optional): 是否转换为单声道。默认为 True。
            dtype (np.dtype, optional): 返回的 NumPy 数组的数据类型。默认为 np.float32。

        Returns:
            np.ndarray: 音频数据。
            int: 音频的实际采样率 (转换后的)。
        """
        try:
            # 使用 soundfile 加载，它能处理更多格式，并且可以指定输出类型
            audio_data, original_sr = sf.read(file_path, dtype=dtype)
            
            # 转换为单声道 (如果需要且音频不是单声道)
            if mono and audio_data.ndim > 1:
                if audio_data.shape[1] == 1: #已经是 (N,1) 的单声道了
                    audio_data = audio_data.flatten()
                else: # 取平均值转为单声道
                    audio_data = np.mean(audio_data, axis=1)
            
            # 重采样 (如果需要)
            if original_sr != sample_rate:
                audio_data = librosa.resample(audio_data, orig_sr=original_sr, target_sr=sample_rate)
            
            return audio_data, sample_rate
        except Exception as e:
            print(f"Error loading audio file {file_path}: {e}")
            # 尝试使用 librosa 作为备选
            try:
                audio_data, original_sr = librosa.load(file_path, sr=sample_rate, mono=mono)
                return audio_data.astype(dtype), sample_rate # librosa 默认 float32
            except Exception as e2:
                print(f"Fallback librosa loading also failed for {file_path}: {e2}")
                return np.array([], dtype=dtype), 0

    @staticmethod
    def save_audio(file_path, audio_data, sample_rate=16000):
        """
        保存音频数据到文件。

        Args:
            file_path (str): 保存的音频文件路径。
            audio_data (np.ndarray): 要保存的音频数据。
            sample_rate (int, optional): 音频采样率。默认为 16000。
        """
        try:
            sf.write(file_path, audio_data, sample_rate)
            print(f"Audio saved to {file_path}")
        except Exception as e:
            print(f"Error saving audio file {file_path}: {e}")

    @staticmethod
    def convert_to_wav_16khz_mono_float32(audio_data_bytes, input_format='wav'):
        """
        将音频字节数据转换为16kHz单声道float32的NumPy数组。
        对于 faster-whisper 尤其有用。
        Args:
            audio_data_bytes (bytes): 音频文件的字节内容。
            input_format (str): 输入音频的格式 (例如 'wav', 'mp3')。
        Returns:
            np.ndarray: 16kHz, 单声道, float32 的音频数据，或者None如果转换失败。
        """
        try:
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_data_bytes), format=input_format)
            audio_segment = audio_segment.set_frame_rate(16000)
            audio_segment = audio_segment.set_channels(1)
            
            # 转换为 float32 NumPy 数组，范围 [-1.0, 1.0]
            # pydub 的 samples 是 int16 数组
            samples_int16 = np.array(audio_segment.get_array_of_samples())
            samples_float32 = samples_int16.astype(np.float32) / 32768.0 # 标准化到 [-1, 1]
            return samples_float32
        except Exception as e:
            print(f"Error converting audio bytes to 16kHz mono float32: {e}")
            return None

    @staticmethod
    def audio_to_float32(audio_np):
        """确保音频是 float32 格式。如果已经是，则不转换。"""
        if audio_np.dtype == np.float32:
            return audio_np
        elif audio_np.dtype == np.int16:
            return audio_np.astype(np.float32) / 32767.0
        elif audio_np.dtype == np.uint8:
            return (audio_np.astype(np.float32) - 128.0) / 128.0
        # 可以根据需要添加更多类型的转换
        else:
            print(f"Warning: Unsupported audio dtype {audio_np.dtype} for direct float32 conversion. Attempting astype.")
            try:
                # 尝试一个通用的转换，可能不准确
                return audio_np.astype(np.float32) 
            except Exception as e:
                print(f"Could not convert audio of dtype {audio_np.dtype} to float32: {e}")
                return None # 或者抛出错误

# 临时的BeispielAudioRecorder，与pipeline中的定义保持一致，但应放在这里
class BeispielAudioRecorder:
    def __init__(self, buffer_duration_seconds=10, sample_rate=16000):
        self.buffer_duration_samples = int(buffer_duration_seconds * sample_rate)
        self.sample_rate = sample_rate
        self.audio_buffer = np.array([], dtype=np.float32)
        self.current_absolute_time = 0.0 # 记录当前缓冲区的绝对开始时间
        print(f"BeispielAudioRecorder initialized with buffer for {buffer_duration_seconds}s.")

    def add_frames(self, frames_np, frame_start_time_abs=None):
        """
        添加音频帧到缓冲区。
        Args:
            frames_np (np.ndarray): 新的音频帧。
            frame_start_time_abs (float, optional): 新帧在整个流中的绝对开始时间。
                                                 如果为None，则假定是连续的。
        """
        if not isinstance(frames_np, np.ndarray) or frames_np.dtype != np.float32:
            frames_np = AudioUtils.audio_to_float32(frames_np)
            if frames_np is None:
                print("Error: Could not convert input frames to float32.")
                return

        if not self.audio_buffer.any(): # 如果缓冲区为空
            self.current_absolute_time = frame_start_time_abs if frame_start_time_abs is not None else 0.0
        
        self.audio_buffer = np.concatenate((self.audio_buffer, frames_np))
        
        # 保持缓冲区大小
        if len(self.audio_buffer) > self.buffer_duration_samples:
            samples_to_cut = len(self.audio_buffer) - self.buffer_duration_samples
            self.audio_buffer = self.audio_buffer[samples_to_cut:]
            self.current_absolute_time += samples_to_cut / self.sample_rate
    
    def get_buffered_data_with_time(self):
        """返回当前缓冲区中的所有数据及其绝对开始时间。"""
        return self.audio_buffer.copy(), self.current_absolute_time
    
    def get_buffer_data(self): # 保持与pipeline中临时版本兼容
        return self.audio_buffer.copy()

    def clear_buffer(self):
        self.audio_buffer = np.array([], dtype=np.float32)
        self.current_absolute_time = 0.0

# 示例用法
if __name__ == '__main__':
    # 创建一个虚拟的WAV文件字节流 (静音)
    samplerate = 44100
    duration = 1
    frequency = 440
    # t = np.linspace(0., duration, int(samplerate * duration), endpoint=False)
    # audio = np.sin(2. * np.pi * frequency * t) * 0.5
    # audio_int16 = (audio * 32767).astype(np.int16)
    
    # # 使用 soundfile 创建内存中的 WAV
    # virtual_wav_file = io.BytesIO()
    # sf.write(virtual_wav_file, audio_int16, samplerate, format='WAV', subtype='PCM_16')
    # wav_bytes = virtual_wav_file.getvalue()

    # print("--- Testing audio conversion ---")
    # converted_audio = AudioUtils.convert_to_wav_16khz_mono_float32(wav_bytes)
    # if converted_audio is not None:
    #     print(f"Converted audio shape: {converted_audio.shape}, dtype: {converted_audio.dtype}, sample rate should be 16000")
    #     # 可以尝试保存或播放来验证
    #     # AudioUtils.save_audio("test_converted.wav", converted_audio, 16000)
    # else:
    #     print("Audio conversion failed.")

    print("\n--- Testing BeispielAudioRecorder ---")
    recorder = BeispielAudioRecorder(buffer_duration_seconds=2, sample_rate=16000)
    sr = 16000
    chunk1 = np.random.rand(int(1 * sr)).astype(np.float32) * 0.1
    chunk2 = np.random.rand(int(1.5 * sr)).astype(np.float32) * 0.1
    chunk3 = np.random.rand(int(0.8 * sr)).astype(np.float32) * 0.1

    recorder.add_frames(chunk1, frame_start_time_abs=0.0)
    buffered, start_time = recorder.get_buffered_data_with_time()
    print(f"After chunk1: buffer_len={len(buffered)/sr:.2f}s, start_abs_time={start_time:.2f}s")
    assert abs(len(buffered)/sr - 1.0) < 0.01
    assert abs(start_time - 0.0) < 0.01

    recorder.add_frames(chunk2, frame_start_time_abs=1.0) # chunk2 在 1.0s 开始
    buffered, start_time = recorder.get_buffered_data_with_time()
    print(f"After chunk2: buffer_len={len(buffered)/sr:.2f}s, start_abs_time={start_time:.2f}s") # 总共2.5s，缓冲区2s，所以应该从0.5s开始
    assert abs(len(buffered)/sr - 2.0) < 0.01 # 缓冲区已满
    # 0s (chunk1) + 1.5s (chunk2) = 2.5s total. Buffer is 2s. Should cut 0.5s from start.
    # So, buffer should contain last 0.5s of chunk1 and all of chunk2.
    # Start time should be 0.5s.
    assert abs(start_time - 0.5) < 0.01 

    recorder.add_frames(chunk3, frame_start_time_abs=2.5) # chunk3 在 2.5s 开始
    buffered, start_time = recorder.get_buffered_data_with_time()
    print(f"After chunk3: buffer_len={len(buffered)/sr:.2f}s, start_abs_time={start_time:.2f}s") # 缓冲区2s
    # 之前是 0.5s-2.5s 的音频。新块是 2.5s-3.3s。 总共 0.5s - 3.3s (长度2.8s)。
    # 缓冲区保留最后2s，即 1.3s - 3.3s。
    assert abs(len(buffered)/sr - 2.0) < 0.01
    assert abs(start_time - (0.5 + 1.5 + 0.8 - 2.0)) < 0.01 # (1.3s)

    print("BeispielAudioRecorder tests seem OK.")

    # 注意：直接运行此文件可能因为缺少真实的音频文件而部分功能无法完整测试
    # 需要在您的项目中放置测试音频，或修改路径
    # dummy_audio_data, sr = AudioUtils.load_audio("non_existent_file.wav")
    # print(f"Loaded dummy audio: {len(dummy_audio_data)} samples at {sr} Hz") 