# src/asr/faster_whisper_streamer.py

from faster_whisper import WhisperModel
import numpy as np
import time

# 从配置导入
from src.config import ASR_MODEL_NAME, DEVICE, VAD_PARAMETERS

class FasterWhisperStreamer:
    def __init__(
        self,
        model_size=ASR_MODEL_NAME,
        device=DEVICE,
        compute_type='float16', # or int8_float16, int8
        vad_filter=True,
        vad_parameters=None
    ):
        '''
        初始化 FasterWhisperStreamer.

        Args:
            model_size (str): faster-whisper 模型大小 (e.g., "tiny", "base", "small", "medium", "large-v2", "large-v3").
            device (str): 推理设备 ("cuda" or "cpu").
            compute_type (str): 计算类型 (e.g., "float16", "int8_float16").
            vad_filter (bool): 是否启用 VAD 语音活动检测滤波器。
            vad_parameters (dict, optional): VAD 参数。默认为 config.py 中的 VAD_PARAMETERS。
        '''
        print(f'Loading ASR model: {model_size} on {device} with {compute_type}')
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.vad_filter = vad_filter
        self.vad_parameters = vad_parameters if vad_parameters is not None else VAD_PARAMETERS
        print('ASR model loaded.')

    def transcribe_audio_chunk(self, audio_chunk_np, language=None, word_timestamps=True, initial_prompt=None):
        '''
        转录单个音频块。

        Args:
            audio_chunk_np (np.ndarray): 浮点型的 NumPy 数组，表示音频数据 (应为单声道，16kHz)。
            language (str, optional): 音频语言代码 (e.g., "en", "zh")。如果为None，则自动检测。
            word_timestamps (bool): 是否返回单词级时间戳。
            initial_prompt (str, optional): 可选的初始提示，用于指导解码。

        Returns:
            tuple: (segments, info) - faster-whisper 的转录结果。
                   segments 是一个生成器，包含每个语音段的 Segment 对象。
                   info 包含检测到的语言等信息。
        '''
        if not isinstance(audio_chunk_np, np.ndarray):
            raise ValueError('audio_chunk_np must be a NumPy array')
        if audio_chunk_np.ndim > 1 and audio_chunk_np.shape[0] > 1 and audio_chunk_np.shape[1] > 1:
             # 假设如果是多通道，选择第一个通道
            print(f'Warning: Multi-channel audio detected, shape: {audio_chunk_np.shape}. Using the first channel.')
            audio_chunk_np = audio_chunk_np[:, 0]
        elif audio_chunk_np.ndim == 2 and audio_chunk_np.shape[0] == 1: # e.g. shape (1, N)
            audio_chunk_np = audio_chunk_np.flatten()

        # faster-whisper期望float32
        if audio_chunk_np.dtype != np.float32:
            audio_chunk_np = audio_chunk_np.astype(np.float32)

        start_time = time.time()
        segments, info = self.model.transcribe(
            audio_chunk_np,
            beam_size=5, # 可以作为参数调整
            language=language,
            word_timestamps=word_timestamps,
            vad_filter=self.vad_filter,
            vad_parameters=self.vad_parameters if self.vad_filter else None,
            initial_prompt=initial_prompt
        )
        transcription_time = time.time() - start_time
        # print(f'Transcription time for chunk: {transcription_time:.4f}s')
        # print(f'Detected language: {info.language} with probability {info.language_probability:.2f}')
        return segments, info, transcription_time

    def process_segment(self, segment, audio_start_time_abs=0):
        '''处理单个转录段，添加绝对时间戳。'''
        processed_words = []
        if segment.words:
            for word in segment.words:
                processed_words.append({
                    'text': word.word,
                    'start': audio_start_time_abs + word.start,
                    'end': audio_start_time_abs + word.end,
                    'probability': word.probability
                })
        return {
            'text': segment.text,
            'start_abs': audio_start_time_abs + segment.start,
            'end_abs': audio_start_time_abs + segment.end,
            'words_abs': processed_words
        }

# 示例用法 (用于测试):
if __name__ == '__main__':
    # 这部分需要一个真实的音频文件或模拟的numpy数组
    # 假设我们有一个10秒的16kHz单声道音频文件 "test_audio.wav"
    # 你需要安装pydub或librosa来加载音频: pip install pydub librosa
    try:
        import librosa
        print('Attempting to load test audio with librosa...')
        # 创建一个虚拟的10秒音频信号
        sample_rate = 16000
        duration = 10  # 秒
        frequency = 440 # Hz (A4 note)
        # t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        # audio_data_np = 0.5 * np.sin(2 * np.pi * frequency * t)
        # audio_data_np = audio_data_np.astype(np.float32)

        # 或者从文件加载 (确保你有一个 test_audio.wav 或者替换为你的文件)
        # audio_data_np, sample_rate = librosa.load(\'path_to_your_audio.wav\', sr=16000, mono=True)

        # 由于我们不能在此处实际加载文件，我们将创建一个静音的numpy数组作为占位符
        print('Creating a dummy silent audio array for testing as file loading is not possible here.')
        audio_data_np = np.zeros(sample_rate * duration, dtype=np.float32)

        print(f'Dummy audio data shape: {audio_data_np.shape}, dtype: {audio_data_np.dtype}')

        streamer = FasterWhisperStreamer()

        # 模拟流式输入，处理整个音频
        print('\n--- Transcribing entire dummy audio ---')
        segments, info, trans_time = streamer.transcribe_audio_chunk(audio_data_np, language='en')
        print(f'Transcription completed in {trans_time:.4f}s')
        print(f'Detected language: {info.language} (prob: {info.language_probability:.2f}) for {info.duration:.2f}s audio')

        full_text = []
        for segment in segments:
            processed_segment = streamer.process_segment(segment)
            print(f'[%.2fs -> %.2fs] {segment.text}')
            if segment.words:
                for word in segment.words:
                    print(f'  [%.2fs -> %.2fs] {word.word} (p=%.2f)' % (word.start, word.end, word.probability))
            full_text.append(segment.text)
        print('Full transcription:', ' '.join(full_text)) # segments 是生成器，需要迭代

        # 模拟分块处理 (例如，每2秒一块，重叠0.5秒)
        # 这部分逻辑会更复杂，需要 audio_segmenter.py 的辅助
        # 这里仅简单演示调用
        print('\n--- Simulating chunk-wise transcription (conceptual) ---')
        chunk_duration_samples = 2 * sample_rate
        chunk_1 = audio_data_np[:chunk_duration_samples]

        print(f'Transcribing first chunk (0.00s to 2.00s) of dummy audio...')
        segments_chunk1, info_chunk1, time_c1 = streamer.transcribe_audio_chunk(chunk_1, language='en')
        print(f'Chunk 1 (duration {info_chunk1.duration:.2f}s) transcription: {time_c1:.2f}s')
        for segment in segments_chunk1:
            processed_segment = streamer.process_segment(segment, audio_start_time_abs=0)
            print(f'  Chunk1 Seg: [%.2fs -> %.2fs] {processed_segment["text"]}')
            # print(f'    Words: {processed_segment["words_abs"]}')

    except ImportError as e:
        print(f'Error: librosa (or its dependency) is not installed. Please install it to run the example: pip install librosa. Details: {e}')
    except Exception as e:
        import traceback
        print(f'An error occurred during the example: {e}')
        traceback.print_exc() 