#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Text-to-Speech (TTS) Client Tool
HTTP API client for remote TTS service (CosyVoice)

支持单个文本合成和多并发批量处理
"""

import requests
import json
import wave
from pathlib import Path
from typing import Optional, List, Tuple
import argparse
import time
import struct
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import threading


@dataclass
class TTSTask:
    """TTS 任务数据结构"""
    json_file: Path
    output_path: Path
    text: str
    spk_id: str
    speed_factor: float
    language: str


class TTSClient:
    """TTS Service Client"""
    
    def __init__(self, base_url: str = "http://host.docker.internal:20401"):
        """
        Initialize TTS client
        
        Args:
            base_url: Base URL of TTS service
        """
        self.base_url = base_url.rstrip('/')
        self.endpoint = f"{self.base_url}/inference_sft"
        
    def synthesize(
        self, 
        tts_text: str, 
        spk_id: str = "晓伊",
        output_path: str = None,
        speed_factor: float = 0.8,
        timeout: int = 120,
        verbose: bool = True
    ) -> Optional[bytes]:
        """
        Call TTS service to generate speech
        
        Args:
            tts_text: Text to convert to speech
            spk_id: Speaker ID (e.g., "晓伊", "云皓")
            output_path: Output audio file path, if not provided, won't save file
            speed_factor: Speed adjustment factor (>1.0 for faster, <1.0 for slower)
            timeout: Request timeout in seconds
            verbose: Whether to print progress messages
            
        Returns:
            Generated audio data (bytes), None if failed
        """
        try:
            # Prepare request data
            data = {
                'tts_text': tts_text,
                'spk_id': spk_id,
                'stream': True,
                'speed': speed_factor
            }
            
            if verbose:
                print(f"Synthesizing speech...")
                print(f"  Text: {tts_text[:50]}{'...' if len(tts_text) > 50 else ''}")
                print(f"  Speaker ID: {spk_id}")
            
            # Send POST request
            response = requests.post(
                self.endpoint,
                data=data,
                timeout=timeout,
                stream=True
            )
            
            if response.status_code == 200:
                # Read audio stream data
                audio_data = b''
                for chunk in response.iter_content(chunk_size=16000):
                    if chunk:
                        audio_data += chunk
                
                # Save file if output path provided
                if output_path:
                    output_path1 = Path(output_path)
                    output_path1.parent.mkdir(parents=True, exist_ok=True)
                    
                    # 将原始PCM数据转换为WAV格式
                    wav_data = self._convert_pcm_to_wav(audio_data)
                    
                    with open(output_path, 'wb') as f:
                        f.write(wav_data)
                    
                    if verbose:
                        print(f"  Audio saved to: {output_path}")
                        
                        # Validate generated audio file
                        if self._validate_audio(output_path1):
                            print(f"  Audio file validation passed")
                        else:
                            print(f"  Warning: Audio file may be corrupted")
                
                return audio_data
            
            else:
                if verbose:
                    print(f"TTS request failed: {response.status_code}")
                    try:
                        error_msg = response.json()
                        print(f"  Error message: {error_msg}")
                    except:
                        print(f"  Error message: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            if verbose:
                print(f"Request timeout (>{timeout}s)")
            return None
        except requests.exceptions.ConnectionError:
            if verbose:
                print(f"Connection failed, please check if TTS service is running: {self.base_url}")
            return None
        except Exception as e:
            if verbose:
                print(f"TTS synthesis failed: {str(e)}")
            return None
    
    def _convert_pcm_to_wav(self, pcm_data: bytes, sample_rate: int = 22050, channels: int = 1, bit_depth: int = 16) -> bytes:
        """
        Convert raw PCM data to WAV format
        
        Args:
            pcm_data: Raw PCM audio data
            sample_rate: Sample rate in Hz (default: 22050, CosyVoice output)
            channels: Number of channels (default: 1 for mono)
            bit_depth: Bits per sample (default: 16)
            
        Returns:
            WAV format audio data
        """
        # Calculate parameters
        byte_depth = bit_depth // 8
        block_align = channels * byte_depth
        byte_rate = sample_rate * block_align
        data_size = len(pcm_data)
        file_size = 36 + data_size
        
        # Create WAV header
        wav_header = struct.pack('<4sI4s4sIHHIIHH4sI',
            b'RIFF',           # ChunkID
            file_size,         # ChunkSize
            b'WAVE',           # Format
            b'fmt ',           # Subchunk1ID
            16,                # Subchunk1Size (PCM)
            1,                 # AudioFormat (PCM = 1)
            channels,          # NumChannels
            sample_rate,       # SampleRate
            byte_rate,         # ByteRate
            block_align,       # BlockAlign
            bit_depth,         # BitsPerSample
            b'data',           # Subchunk2ID
            data_size          # Subchunk2Size
        )
        
        return wav_header + pcm_data
    
    def _validate_audio(self, audio_path: Path) -> bool:
        """Validate audio file"""
        try:
            with wave.open(str(audio_path), 'rb') as wav_file:
                frames = wav_file.getnframes()
                sample_rate = wav_file.getframerate()
                duration = frames / sample_rate
                
                print(f"    Sample rate: {sample_rate}Hz, Duration: {duration:.2f}s")
                return frames > 0 and sample_rate > 0
        except Exception as e:
            print(f"    Audio validation error: {e}")
            return False
    
    def test_connection(self) -> bool:
        """Test connection to TTS service"""
        try:
            # Try to access root path or health check endpoint
            response = requests.get(f"{self.base_url}/", timeout=5)
            if response.status_code in [200, 404]:  # 404 also counts as connection success
                print(f"TTS service connection OK: {self.base_url}")
                return True
            else:
                print(f"TTS service response abnormal: {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print(f"Cannot connect to TTS service: {self.base_url}")
            return False
        except Exception as e:
            print(f"Connection test failed: {str(e)}")
            return False


class BatchTTSProcessor:
    """
    批量 TTS 处理器
    
    支持多并发处理，提高 GPU 利用率
    """
    
    def __init__(self, tts_client: TTSClient, max_workers: int = 4):
        """
        初始化批量处理器
        
        Args:
            tts_client: TTS 客户端实例
            max_workers: 最大并发数（默认4，可根据GPU显存调整）
        """
        self.tts_client = tts_client
        self.max_workers = max_workers
        
        # 统计信息（线程安全）
        self._lock = threading.Lock()
        self._success_count = 0
        self._failed_count = 0
        self._skipped_count = 0
    
    def _process_single_task(self, task: TTSTask) -> Tuple[bool, str]:
        """
        处理单个 TTS 任务（在工作线程中执行）
        
        Args:
            task: TTS 任务
            
        Returns:
            (是否成功, 消息)
        """
        try:
            result = self.tts_client.synthesize(
                tts_text=task.text,
                spk_id=task.spk_id,
                output_path=str(task.output_path),
                speed_factor=task.speed_factor,
                verbose=False  # 并发时关闭详细输出
            )
            
            if result:
                return True, f"✓ {task.json_file.name}"
            else:
                return False, f"✗ {task.json_file.name}: TTS failed"
                
        except Exception as e:
            return False, f"✗ {task.json_file.name}: {str(e)}"
    
    def process_json_files(
        self, 
        input_dir: str, 
        output_dir: str,
        speed_factor: float = 0.8,
        pattern: str = "*.json",
        max_workers: int = None
    ):
        """
        批量处理 JSON 文件生成语音（多并发）
        
        Args:
            input_dir: 输入目录（包含 JSON 文件）
            output_dir: 输出目录（保存音频文件）
            speed_factor: 语速系数
            pattern: JSON 文件匹配模式
            max_workers: 并发数（None 则使用初始化时的值）
        """
        input_dir_path = Path(input_dir)
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        
        workers = max_workers or self.max_workers
        
        # 收集所有任务
        json_files = list(input_dir_path.rglob(pattern))
        print(f"Found {len(json_files)} JSON files")
        print(f"Concurrent workers: {workers}")
        
        # 准备任务列表
        tasks: List[TTSTask] = []
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                text = data.get('text', '')
                language = data.get('language', 'zh')
                
                if not text:
                    print(f"Skipping empty text: {json_file.name}")
                    continue
                
                # 设置说话人
                spk_id = "晓伊"
                
                # 构建输出路径
                audio_filename = data.get('audio_file', json_file.stem + '.wav')
                relative_path = json_file.relative_to(input_dir_path)
                output_subdir = output_dir_path / relative_path.parent
                output_path = output_subdir / audio_filename
                
                # 跳过已存在的文件
                if output_path.exists():
                    self._skipped_count += 1
                    continue
                
                # 确保输出目录存在
                output_subdir.mkdir(parents=True, exist_ok=True)
                
                tasks.append(TTSTask(
                    json_file=json_file,
                    output_path=output_path,
                    text=text,
                    spk_id=spk_id,
                    speed_factor=speed_factor,
                    language=language
                ))
                
            except Exception as e:
                print(f"Error reading {json_file}: {e}")
                self._failed_count += 1
        
        if self._skipped_count > 0:
            print(f"Skipped {self._skipped_count} existing files")
        
        if not tasks:
            print("No tasks to process")
            return
        
        print(f"Processing {len(tasks)} tasks...")
        print("-" * 50)
        
        # 重置计数器
        self._success_count = 0
        self._failed_count = 0
        
        start_time = time.time()
        
        # 使用线程池并发处理
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # 提交所有任务
            future_to_task = {
                executor.submit(self._process_single_task, task): task 
                for task in tasks
            }
            
            # 处理完成的任务
            completed = 0
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                completed += 1
                
                try:
                    success, message = future.result()
                    
                    with self._lock:
                        if success:
                            self._success_count += 1
                        else:
                            self._failed_count += 1
                    
                    # 打印进度
                    progress = completed / len(tasks) * 100
                    print(f"[{completed}/{len(tasks)} {progress:.1f}%] {message}")
                    
                except Exception as e:
                    with self._lock:
                        self._failed_count += 1
                    print(f"[{completed}/{len(tasks)}] ✗ {task.json_file.name}: {e}")
        
        elapsed = time.time() - start_time
        
        print("-" * 50)
        print(f"Batch processing completed in {elapsed:.1f}s")
        print(f"  Success: {self._success_count}")
        print(f"  Failed: {self._failed_count}")
        print(f"  Skipped: {self._skipped_count}")
        if self._success_count > 0:
            print(f"  Average: {elapsed / self._success_count:.2f}s per file")


def main():
    """Command line main function"""
    parser = argparse.ArgumentParser(description='TTS Client Tool')
    parser.add_argument('--url', default='http://host.docker.internal:20401', 
                       help='TTS service URL')
    parser.add_argument('--test', action='store_true',
                       help='Test connection')
    
    # Single file synthesis
    parser.add_argument('--text', help='Text to synthesize')
    parser.add_argument('--spk-id', default='晓伊',
                       help='Speaker ID (e.g., 晓伊, 云皓)')
    parser.add_argument('--speed', type=float, default=0.8,
                       help='Speed adjustment factor (>1.0 for faster, <1.0 for slower)')
    parser.add_argument('--output', default='ttsresult/result.wav', 
                       help='Output audio file path')
    
    # Batch processing
    parser.add_argument('--batch', action='store_true',
                       help='Batch processing mode')
    parser.add_argument('--input-dir', 
                       help='Input directory (containing JSON files)')
    parser.add_argument('--output-dir',
                       help='Output directory (saving audio files)')
    parser.add_argument('--pattern', default='*.json',
                       help='JSON file match pattern')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of concurrent workers (default: 4)')
    
    args = parser.parse_args()
    
    # Create TTS client
    client = TTSClient(args.url)
    
    # Test connection
    if args.test:
        client.test_connection()
        return
    
    # Batch processing mode
    if args.batch:
        if not args.input_dir or not args.output_dir:
            print("Error: Batch mode requires --input-dir and --output-dir")
            return
        
        processor = BatchTTSProcessor(client, max_workers=args.workers)
        processor.process_json_files(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            speed_factor=args.speed,
            pattern=args.pattern
        )
        return
    
    # Single file synthesis mode
    if args.text and args.output:
        result = client.synthesize(
            tts_text=args.text,
            spk_id=args.spk_id,
            output_path=args.output,
            speed_factor=args.speed
        )
        
        if result:
            print("Speech synthesis completed successfully")
        else:
            print("Speech synthesis failed")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
