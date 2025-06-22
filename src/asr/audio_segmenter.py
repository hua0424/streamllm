# src/asr/audio_segmenter.py

import numpy as np
import torch
import librosa
import argparse
import os
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import logging
import soundfile as sf
from pathlib import Path

# 尝试导入 silero-vad
try:
    from silero_vad import load_silero_vad, get_speech_timestamps, VADIterator
    SILERO_VAD_AVAILABLE = True
except ImportError:
    SILERO_VAD_AVAILABLE = False
    
from src.config import VAD_PARAMETERS

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VADAudioSegmenter:
    """
    基于 Silero VAD 的音频分段器
    支持批量和流式两种模式，确保分割后的音频段能完整重构原始文件
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        min_speech_duration: float = 0.25,
        min_silence_duration: float = 0.2,
        speech_pad_ms: int = 30,
        buffer_size_seconds: float = 10.0
    ):
        """
        初始化VAD音频分段器
        
        Args:
            sample_rate: 音频采样率，必须是8000或16000
            threshold: VAD检测阈值，0-1之间
            min_speech_duration: 最小语音段长度（秒）
            min_silence_duration: 最小静音段长度（秒）
            speech_pad_ms: 语音段前后填充时间（毫秒）
            buffer_size_seconds: 流式模式下的缓冲区大小（秒）
        """
        if sample_rate not in [8000, 16000]:
            raise ValueError("采样率必须是8000或16000Hz")
            
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_speech_duration = min_speech_duration
        self.min_silence_duration = min_silence_duration
        self.speech_pad_ms = speech_pad_ms
        self.buffer_size_seconds = buffer_size_seconds
        
        # 加载VAD模型
        self._load_vad_model()
        
        # 流式处理状态
        self._reset_streaming_state()
    
    def _load_vad_model(self):
        """加载Silero VAD模型"""
        try:
            if SILERO_VAD_AVAILABLE:
                self.vad_model = load_silero_vad(onnx=False)
                logger.info("已加载 Silero VAD 模型 (pip版本)")
            else:
                # 使用torch.hub方式
                torch.set_num_threads(1)
                self.vad_model, self.vad_utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False,
                    onnx=False
                )
                logger.info("已加载 Silero VAD 模型 (torch.hub版本)")
        except Exception as e:
            logger.error(f"无法加载VAD模型: {e}")
            raise RuntimeError(f"VAD模型加载失败: {e}")
    
    def _reset_streaming_state(self):
        """重置流式处理状态"""
        self.audio_buffer = np.array([], dtype=np.float32)
        self.total_processed_samples = 0
        self.completed_segments = []
        self.current_speech_start = None
        
        # 创建VAD迭代器（流式模式）
        if SILERO_VAD_AVAILABLE:
            self.vad_iterator = VADIterator(
                model=self.vad_model,
                sampling_rate=self.sample_rate,
                threshold=self.threshold,
                min_silence_duration_ms=int(self.min_silence_duration * 1000),
                speech_pad_ms=self.speech_pad_ms
            )
        else:
            # torch.hub版本的VADIterator
            _, _, _, VADIterator_class, _ = self.vad_utils
            self.vad_iterator = VADIterator_class(
                model=self.vad_model,
                sampling_rate=self.sample_rate,
                threshold=self.threshold,
                min_silence_duration_ms=int(self.min_silence_duration * 1000),
                speech_pad_ms=self.speech_pad_ms
            )
    
    def segment_audio_file(self, audio_file_path: str, output_dir: str = None) -> List[Dict]:
        """
        对音频文件进行批量VAD分段
        
        Args:
            audio_file_path: 输入音频文件路径
            output_dir: 输出目录，如果为None则不保存文件
            
        Returns:
            分段信息列表，每个元素包含：
            {
                'segment_id': int,
                'start_time': float,
                'end_time': float,  
                'duration': float,
                'audio_data': np.ndarray,
                'file_path': str (如果保存了文件)
            }
        """
        logger.info(f"开始处理音频文件: {audio_file_path}")
        
        # 加载音频文件
        try:
            audio_data, original_sr = librosa.load(
                audio_file_path, 
                sr=self.sample_rate, 
                mono=True
            )
            logger.info(f"音频加载成功: 时长{len(audio_data)/self.sample_rate:.2f}秒, 采样率{self.sample_rate}Hz")
        except Exception as e:
            logger.error(f"无法加载音频文件: {e}")
            raise
        
        # 执行VAD分段
        segments = self._segment_audio_batch(audio_data)
        
        # 保存分段音频（如果指定了输出目录）
        if output_dir:
            segments = self._save_segments(segments, audio_file_path, output_dir)
        
        logger.info(f"分段完成: 共{len(segments)}个语音段")
        return segments
    
    def _segment_audio_batch(self, audio_data: np.ndarray) -> List[Dict]:
        """批量模式的VAD分段"""
        # 确保音频数据格式正确
        audio_data = self._normalize_audio(audio_data)
        
        try:
            if SILERO_VAD_AVAILABLE:
                # 使用pip版本的get_speech_timestamps
                speech_timestamps = get_speech_timestamps(
                    audio_data,
                    self.vad_model,
                    sampling_rate=self.sample_rate,
                    threshold=self.threshold,
                    min_speech_duration_ms=int(self.min_speech_duration * 1000),
                    min_silence_duration_ms=int(self.min_silence_duration * 1000),
                    window_size_samples=512 if self.sample_rate == 16000 else 256,
                    speech_pad_ms=self.speech_pad_ms,
                    return_seconds=True
                )
            else:
                # 使用torch.hub版本
                get_speech_timestamps_func = self.vad_utils[0]
                speech_timestamps = get_speech_timestamps_func(
                    audio_data,
                    self.vad_model,
                    sampling_rate=self.sample_rate,
                    threshold=self.threshold,
                    min_speech_duration_ms=int(self.min_speech_duration * 1000),
                    min_silence_duration_ms=int(self.min_silence_duration * 1000),
                    window_size_samples=512 if self.sample_rate == 16000 else 256,
                    speech_pad_ms=self.speech_pad_ms,
                    return_seconds=True
                )
        except Exception as e:
            logger.error(f"VAD分段失败: {e}")
            return []
        
        # 转换为统一格式
        segments = []
        for i, timestamp in enumerate(speech_timestamps):
            start_time = timestamp['start']
            end_time = timestamp['end']
            
            # 提取音频数据
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)
            
            # 确保索引有效
            start_sample = max(0, min(start_sample, len(audio_data)))
            end_sample = max(start_sample, min(end_sample, len(audio_data)))
            
            segment_audio = audio_data[start_sample:end_sample].copy()
            
            segments.append({
                'segment_id': i + 1,
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time,
                'audio_data': segment_audio,
                'file_path': None
            })
        
        return segments
    
    def process_streaming_audio(self, audio_chunk: np.ndarray) -> List[Dict]:
        """
        处理流式音频块
        
        Args:
            audio_chunk: 音频数据块
            
        Returns:
            新检测到的完整语音段列表
        """
        # 归一化音频数据
        audio_chunk = self._normalize_audio(audio_chunk)
        
        # 添加到缓冲区
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])
        chunk_start_sample = self.total_processed_samples
        self.total_processed_samples += len(audio_chunk)
        
        # 使用VAD迭代器处理
        new_segments = []
        
        # 处理音频块
        window_size = 512 if self.sample_rate == 16000 else 256
        
        for i in range(0, len(audio_chunk), window_size):
            window_end = min(i + window_size, len(audio_chunk))
            window = audio_chunk[i:window_end]
            
            if len(window) < window_size:
                break
                
            try:
                # 使用VAD迭代器
                speech_dict = self.vad_iterator(window, return_seconds=True)
                
                if speech_dict:
                    if 'start' in speech_dict:
                        # 语音开始
                        abs_start_time = (chunk_start_sample + i) / self.sample_rate
                        self.current_speech_start = abs_start_time
                        logger.debug(f"语音开始: {abs_start_time:.3f}s")
                    
                    if 'end' in speech_dict:
                        # 语音结束
                        if self.current_speech_start is not None:
                            abs_end_time = (chunk_start_sample + i + len(window)) / self.sample_rate
                            segment = self._create_segment_from_buffer(
                                self.current_speech_start, 
                                abs_end_time
                            )
                            if segment:
                                new_segments.append(segment)
                            self.current_speech_start = None
                            logger.debug(f"语音结束: {abs_end_time:.3f}s")
                            
            except Exception as e:
                logger.warning(f"VAD处理错误: {e}")
                continue
        
        # 维护缓冲区大小
        self._maintain_buffer()
        
        return new_segments
    
    def finish_streaming(self) -> List[Dict]:
        """
        完成流式处理，返回剩余的语音段
        
        Returns:
            剩余的语音段列表
        """
        remaining_segments = []
        
        # 如果还有未完成的语音段，强制结束
        if self.current_speech_start is not None:
            current_time = self.total_processed_samples / self.sample_rate
            segment = self._create_segment_from_buffer(self.current_speech_start, current_time)
            if segment:
                remaining_segments.append(segment)
        
        # 重置状态
        self._reset_streaming_state()
        
        return remaining_segments
    
    def _create_segment_from_buffer(self, start_time: float, end_time: float) -> Optional[Dict]:
        """从缓冲区创建音频段"""
        duration = end_time - start_time
        
        # 检查最小长度
        if duration < self.min_speech_duration:
            logger.debug(f"语音段太短 ({duration:.3f}s)，跳过")
            return None
        
        # 计算缓冲区的时间范围
        buffer_duration = len(self.audio_buffer) / self.sample_rate
        buffer_start_time = (self.total_processed_samples / self.sample_rate) - buffer_duration
        
        # 计算在缓冲区中的位置
        rel_start_time = start_time - buffer_start_time
        rel_end_time = end_time - buffer_start_time
        
        start_sample = int(rel_start_time * self.sample_rate)
        end_sample = int(rel_end_time * self.sample_rate)
        
        # 确保索引有效
        start_sample = max(0, min(start_sample, len(self.audio_buffer)))
        end_sample = max(start_sample, min(end_sample, len(self.audio_buffer)))
        
        if end_sample <= start_sample:
            logger.warning(f"无法从缓冲区提取音频段 [{start_time:.3f}s -> {end_time:.3f}s]")
            return None
        
        audio_data = self.audio_buffer[start_sample:end_sample].copy()
        
        segment = {
            'segment_id': len(self.completed_segments) + 1,
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'audio_data': audio_data,
            'file_path': None
        }
        
        self.completed_segments.append(segment)
        return segment
    
    def _maintain_buffer(self):
        """维护缓冲区大小"""
        max_buffer_samples = int(self.buffer_size_seconds * self.sample_rate)
        
        if len(self.audio_buffer) > max_buffer_samples:
            # 保留最近的音频数据
            self.audio_buffer = self.audio_buffer[-max_buffer_samples:]
    
    def _normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """归一化音频数据"""
        if not isinstance(audio_data, np.ndarray):
            audio_data = np.array(audio_data, dtype=np.float32)
        
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # 确保音频范围在[-1, 1]
        max_val = np.max(np.abs(audio_data))
        if max_val > 1.0:
            audio_data = audio_data / max_val
            
        return audio_data
    
    def _save_segments(self, segments: List[Dict], original_file_path: str, output_dir: str) -> List[Dict]:
        """保存音频段到文件"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 生成输出文件名前缀
        original_name = Path(original_file_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"保存音频段到: {output_path}")
        
        for segment in segments:
            # 生成文件名
            filename = f"{original_name}_seg{segment['segment_id']:03d}_{segment['start_time']:.2f}s-{segment['end_time']:.2f}s.wav"
            file_path = output_path / filename
            
            try:
                # 保存音频文件
                sf.write(str(file_path), segment['audio_data'], self.sample_rate)
                segment['file_path'] = str(file_path)
                logger.debug(f"已保存: {filename}")
            except Exception as e:
                logger.error(f"保存失败 {filename}: {e}")
        
        return segments
    
    def verify_segments_completeness(self, segments: List[Dict], original_duration: float, tolerance: float = 0.1) -> Dict:
        """
        验证分段的完整性
        
        Args:
            segments: 音频段列表
            original_duration: 原始音频时长
            tolerance: 容忍的时间差（秒）
            
        Returns:
            验证结果字典
        """
        if not segments:
            return {
                'is_complete': False,
                'coverage_ratio': 0.0,
                'total_segment_duration': 0.0,
                'gaps': [],
                'overlaps': []
            }
        
        # 按开始时间排序
        sorted_segments = sorted(segments, key=lambda x: x['start_time'])
        
        # 计算总时长
        total_segment_duration = sum(seg['duration'] for seg in segments)
        coverage_ratio = total_segment_duration / original_duration
        
        # 检查间隙和重叠
        gaps = []
        overlaps = []
        
        for i in range(len(sorted_segments) - 1):
            current_end = sorted_segments[i]['end_time']
            next_start = sorted_segments[i + 1]['start_time']
            
            if next_start > current_end + tolerance:
                # 有间隙
                gaps.append({
                    'start': current_end,
                    'end': next_start,
                    'duration': next_start - current_end
                })
            elif next_start < current_end - tolerance:
                # 有重叠
                overlaps.append({
                    'start': next_start,
                    'end': current_end,
                    'duration': current_end - next_start
                })
        
        is_complete = (
            coverage_ratio >= 0.8 and  # 至少80%覆盖率
            len(gaps) == 0 and  # 无间隙
            abs(total_segment_duration - original_duration) <= tolerance  # 总时长差异在容忍范围内
        )
        
        return {
            'is_complete': is_complete,
            'coverage_ratio': coverage_ratio,
            'total_segment_duration': total_segment_duration,
            'gaps': gaps,
            'overlaps': overlaps
        }


class StreamingAudioSimulator:
    """流式音频模拟器，用于测试流式分段功能"""
    
    def __init__(self, audio_file_path: str, chunk_duration: float = 0.5, sample_rate: int = 16000):
        self.audio_file_path = audio_file_path
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        
        # 加载音频
        self.audio_data, _ = librosa.load(audio_file_path, sr=sample_rate, mono=True)
        self.total_duration = len(self.audio_data) / sample_rate
        self.chunk_size = int(chunk_duration * sample_rate)
        self.current_position = 0
        
        logger.info(f"音频模拟器初始化: {self.total_duration:.2f}秒, 块大小{chunk_duration}秒")
    
    def get_next_chunk(self) -> Tuple[np.ndarray, bool]:
        """获取下一个音频块"""
        if self.current_position >= len(self.audio_data):
            return np.array([], dtype=np.float32), True
        
        end_pos = min(self.current_position + self.chunk_size, len(self.audio_data))
        chunk = self.audio_data[self.current_position:end_pos]
        
        self.current_position = end_pos
        is_finished = (self.current_position >= len(self.audio_data))
        
        return chunk.astype(np.float32), is_finished
    
    def get_current_time(self) -> float:
        """获取当前时间"""
        return self.current_position / self.sample_rate


def test_batch_segmentation(audio_file: str, output_dir: str):
    """测试批量分段功能"""
    print("=" * 60)
    print("批量分段测试")
    print("=" * 60)
    
    if not os.path.exists(audio_file):
        print(f"错误: 音频文件不存在: {audio_file}")
        return
    
    try:
        # 创建分段器
        segmenter = VADAudioSegmenter(
            sample_rate=16000,
            threshold=0.5,
            min_speech_duration=0.25,
            min_silence_duration=0.2
        )
        
        # 执行分段
        start_time = time.time()
        segments = segmenter.segment_audio_file(audio_file, output_dir)
        process_time = time.time() - start_time
        
        # 输出结果
        print(f"分段完成:")
        print(f"处理时间: {process_time:.3f}秒")
        print(f"检测到 {len(segments)} 个语音段")
        print("-" * 40)
        
        total_duration = 0
        for segment in segments:
            total_duration += segment['duration']
            print(f"段{segment['segment_id']:2d}: [{segment['start_time']:6.2f}s -> {segment['end_time']:6.2f}s] "
                  f"时长: {segment['duration']:5.2f}s")
        
        print(f"-" * 40)
        print(f"总语音时长: {total_duration:.2f}秒")
        
        # 验证完整性
        audio_data, _ = librosa.load(audio_file, sr=16000, mono=True)
        original_duration = len(audio_data) / 16000
        verification = segmenter.verify_segments_completeness(segments, original_duration)
        
        print(f"\n完整性验证:")
        print(f"覆盖率: {verification['coverage_ratio']:.1%}")
        print(f"是否完整: {'是' if verification['is_complete'] else '否'}")
        if verification['gaps']:
            print(f"间隙数量: {len(verification['gaps'])}")
        if verification['overlaps']:
            print(f"重叠数量: {len(verification['overlaps'])}")
        
        return segments
        
    except Exception as e:
        print(f"批量分段测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_streaming_segmentation(audio_file: str, output_dir: str, chunk_duration: float = 0.5):
    """测试流式分段功能"""
    print("=" * 60)
    print("流式分段测试")
    print("=" * 60)
    
    if not os.path.exists(audio_file):
        print(f"错误: 音频文件不存在: {audio_file}")
        return
    
    try:
        # 创建流式音频模拟器
        simulator = StreamingAudioSimulator(audio_file, chunk_duration, 16000)
        
        # 创建分段器
        segmenter = VADAudioSegmenter(
            sample_rate=16000,
            threshold=0.5,
            min_speech_duration=0.25,
            min_silence_duration=0.2
        )
        
        print(f"开始流式处理 (每{chunk_duration}秒一个块)...")
        print("-" * 40)
        
        all_segments = []
        chunk_count = 0
        start_time = time.time()
        
        while True:
            chunk, is_finished = simulator.get_next_chunk()
            
            if len(chunk) == 0:
                break
            
            chunk_count += 1
            current_time = simulator.get_current_time()
            
            print(f"处理块 {chunk_count:2d}: {current_time-chunk_duration:.2f}s -> {current_time:.2f}s")
            
            # 处理音频块
            new_segments = segmenter.process_streaming_audio(chunk)
            
            if new_segments:
                print(f"  ✓ 新增 {len(new_segments)} 个语音段")
                for seg in new_segments:
                    print(f"    段{seg['segment_id']}: [{seg['start_time']:6.2f}s -> {seg['end_time']:6.2f}s] "
                          f"时长: {seg['duration']:5.2f}s")
                all_segments.extend(new_segments)
            else:
                print("  - 无新语音段")
            
            if is_finished:
                break
        
        # 完成处理
        print("\n完成流式处理...")
        final_segments = segmenter.finish_streaming()
        if final_segments:
            print(f"最终段数: {len(final_segments)}")
            all_segments.extend(final_segments)
        
        process_time = time.time() - start_time
        
        # 保存所有段
        if all_segments and output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stream_output_dir = os.path.join(output_dir, f"streaming_{timestamp}")
            all_segments = segmenter._save_segments(all_segments, audio_file, stream_output_dir)
        
        # 输出结果
        print(f"\n流式分段完成:")
        print(f"处理时间: {process_time:.3f}秒")
        print(f"总块数: {chunk_count}")
        print(f"总段数: {len(all_segments)}")
        
        total_duration = sum(seg['duration'] for seg in all_segments)
        print(f"总语音时长: {total_duration:.2f}秒")
        
        # 验证完整性
        verification = segmenter.verify_segments_completeness(all_segments, simulator.total_duration)
        print(f"\n完整性验证:")
        print(f"覆盖率: {verification['coverage_ratio']:.1%}")
        print(f"是否完整: {'是' if verification['is_complete'] else '否'}")
        
        return all_segments
        
    except Exception as e:
        print(f"流式分段测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_test_audio(output_path: str, duration: float = 10.0, sample_rate: int = 16000):
    """创建测试音频文件"""
    print(f"创建测试音频: {output_path}")
    
    # 创建包含语音和静音的测试音频
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    
    # 语音段: 440Hz正弦波
    speech1 = 0.3 * np.sin(2 * np.pi * 440 * t[:int(2 * sample_rate)])  # 0-2秒
    silence1 = np.zeros(int(1 * sample_rate))  # 2-3秒静音
    speech2 = 0.3 * np.sin(2 * np.pi * 523 * t[:int(3 * sample_rate)])  # 3-6秒  
    silence2 = np.zeros(int(1 * sample_rate))  # 6-7秒静音
    speech3 = 0.3 * np.sin(2 * np.pi * 659 * t[:int(3 * sample_rate)])  # 7-10秒
    
    test_audio = np.concatenate([speech1, silence1, speech2, silence2, speech3]).astype(np.float32)
    
    # 保存测试音频
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, test_audio, sample_rate)
    
    print(f"测试音频已保存: {output_path}")
    print(f"包含语音段: [0-2s], [3-6s], [7-10s]")
    
    return output_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VAD音频分段器测试程序")
    parser.add_argument("--mode", choices=["batch", "streaming", "both", "test"], 
                        default="test", help="测试模式")
    parser.add_argument("--audio", type=str, help="音频文件路径")
    parser.add_argument("--output-dir", type=str, 
                        default="/home/project/streamllm/results/wav_segments",
                        help="输出目录")
    parser.add_argument("--chunk-duration", type=float, default=0.5,
                        help="流式模式块大小（秒）")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="VAD阈值")
    parser.add_argument("--min-speech", type=float, default=2,
                        help="最小语音段长度（秒）")
    parser.add_argument("--min-silence", type=float, default=0.2,
                        help="最小静音段长度（秒）")
    
    args = parser.parse_args()
    
    print("VAD 音频分段器测试程序")
    print(f"输出目录: {args.output_dir}")
    print(f"VAD阈值: {args.threshold}")
    print(f"最小语音段: {args.min_speech}秒")
    print(f"最小静音段: {args.min_silence}秒")
    print()
    
    try:
        if args.mode == "test":
            # 创建并测试合成音频
            test_audio_path = "/tmp/test_audio.wav"
            create_test_audio(test_audio_path, duration=10.0)
            
            print("=" * 60)
            print("使用合成音频进行测试")
            print("=" * 60)
            
            # 批量测试
            batch_segments = test_batch_segmentation(test_audio_path, args.output_dir)
            
            print("\n")
            
            # 流式测试
            streaming_segments = test_streaming_segmentation(test_audio_path, args.output_dir, args.chunk_duration)
            
            # 清理测试文件
            if os.path.exists(test_audio_path):
                os.remove(test_audio_path)
                print(f"\n已清理测试文件: {test_audio_path}")
            
        elif args.mode == "batch":
            if not args.audio:
                print("错误: batch模式需要指定--audio参数")
                exit(1)
            test_batch_segmentation(args.audio, args.output_dir)
            
        elif args.mode == "streaming":
            if not args.audio:
                print("错误: streaming模式需要指定--audio参数")
                exit(1)
            test_streaming_segmentation(args.audio, args.output_dir, args.chunk_duration)
            
        elif args.mode == "both":
            if not args.audio:
                print("错误: both模式需要指定--audio参数")
                exit(1)
            print("批量模式测试:")
            test_batch_segmentation(args.audio, args.output_dir)
            print("\n" + "="*60 + "\n")
            print("流式模式测试:")
            test_streaming_segmentation(args.audio, args.output_dir, args.chunk_duration)
        
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc() 