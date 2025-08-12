#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Text-to-Speech (TTS) Client Tool
HTTP API client for remote TTS service
"""

import requests
import json
import wave
from pathlib import Path
from typing import Optional
import argparse
import time
import struct

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
        output_path: str = "/usr/local/app/jupyterlab/yanjiu/streamllm/experiments/datasets/tools/ttsresult",
        speed_factor: float = 1.0,
        timeout: int = 60
    ) -> Optional[bytes]:
        """
        Call TTS service to generate speech
        
        Args:
            tts_text: Text to convert to speech
            spk_id: Speaker ID (e.g., "晓伊", "云皓")
            output_path: Output audio file path, if not provided, won't save file
            speed_factor: Speed adjustment factor (>1.0 for faster, <1.0 for slower) - handled by backend
            timeout: Request timeout in seconds
            
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
                    
                    print(f"  Audio saved to: {output_path}")
                    
                    # Validate generated audio file
                    if self._validate_audio(output_path1):
                        print(f"  Audio file validation passed")
                    else:
                        print(f"  Warning: Audio file may be corrupted")
                
                return audio_data
            
            else:
                print(f"TTS request failed: {response.status_code}")
                try:
                    error_msg = response.json()
                    print(f"  Error message: {error_msg}")
                except:
                    print(f"  Error message: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            print(f"Request timeout (>{timeout}s)")
            return None
        except requests.exceptions.ConnectionError:
            print(f"Connection failed, please check if TTS service is running: {self.base_url}")
            return None
        except Exception as e:
            print(f"TTS synthesis failed: {str(e)}")
            return None
    
    def _convert_pcm_to_wav(self, pcm_data: bytes, sample_rate: int = 22050, channels: int = 1, bit_depth: int = 16) -> bytes:
        """
        Convert raw PCM data to WAV format
        
        Args:
            pcm_data: Raw PCM audio data
            sample_rate: Sample rate in Hz (default: 22050)
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
    """Batch TTS Processor"""
    
    def __init__(self, tts_client: TTSClient):
        self.tts_client = tts_client
    
    def process_json_files(
        self, 
        input_dir: str, 
        output_dir: str,
        speed_factor: float = 1.0,
        pattern: str = "*.json"
    ):
        """
        Batch process JSON files to generate speech for each text
        
        Args:
            input_dir: Input directory containing JSON files
            output_dir: Output directory for audio files
            speed_factor: Speed adjustment factor (>1.0 for faster, <1.0 for slower) - handled by backend
            pattern: JSON file match pattern
        """
        input_dir_path = Path(input_dir)
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        
        # Find all JSON files
        json_files = list(input_dir_path.rglob(pattern))
        print(f"Found {len(json_files)} JSON files")
        
        success_count = 0
        failed_count = 0
        
        for json_file in json_files:
            try:
                # Read JSON data
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                text = data.get('text', '')
                language = data.get('language', 'zh')  # 默认中文
                if not text:
                    print(f"Skipping empty text file: {json_file}")
                    continue
                
                # 根据语言自动设置spk_id
                spk_id = "晓伊"
                
                # Build output file path
                audio_filename = data.get('audio_file', json_file.stem + '.wav')
                
                # Maintain directory structure
                relative_path = json_file.relative_to(input_dir_path)
                output_subdir = output_dir_path / relative_path.parent
                output_subdir.mkdir(parents=True, exist_ok=True)
                
                output_path = output_subdir / audio_filename
                
                # Skip existing files
                if output_path.exists():
                    print(f"Skipping existing file: {output_path}")
                    continue
                
                # Generate speech
                print(f"\nProcessing: {json_file.name} (Language: {language}, Speaker: {spk_id}, Speed: {speed_factor}x)")
                result = self.tts_client.synthesize(
                    tts_text=text,
                    spk_id=spk_id,
                    output_path=str(output_path),
                    speed_factor=speed_factor
                )
                
                if result:
                    success_count += 1
                else:
                    failed_count += 1
                
                # Add small delay to avoid overloading server
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Failed to process file {json_file}: {str(e)}")
                failed_count += 1
        
        print(f"\nBatch processing completed:")
        print(f"  Success: {success_count}")
        print(f"  Failed: {failed_count}")


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
    parser.add_argument('--speed', type=float, default=1.0,
                       help='Speed adjustment factor (>1.0 for faster, <1.0 for slower) - handled by backend')
    parser.add_argument('--output', default='/usr/local/app/jupyterlab/yanjiu/streamllm/experiments/datasets/tools/ttsresult/result.wav', help='Output audio file path')
    
    # Batch processing
    parser.add_argument('--batch', action='store_true',
                       help='Batch processing mode')
    parser.add_argument('--input-dir', 
                       help='Input directory (containing JSON files)')
    parser.add_argument('--output-dir',
                       help='Output directory (saving audio files)')
    parser.add_argument('--pattern', default='*.json',
                       help='JSON file match pattern')
    
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
        
        processor = BatchTTSProcessor(client)
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