# src/pipeline/streaming_pipeline.py

import time
import numpy as np
from collections import deque

from src.asr.audio_segmenter import AudioSegmenter
from src.asr.faster_whisper_streamer import FasterWhisperStreamer
from src.llm.stream_llm_inference import StreamLLMInference
# from src.tts.tts_engine import TTSEngine # 未来引入
from src.config import (
    STREAMING_ASR_CHUNK_SECONDS,
    STREAMING_ASR_OVERLAP_SECONDS,
    VAD_PARAMETERS,
    ASR_MODEL_NAME, # 确保从config导入
    LLM_MODEL_NAME, # 确保从config导入
    DEVICE
)
# from src.utils.audio_utils import BeispielAudioRecorder # 暂时注释掉，因为文件还未创建
from src.utils.audio_utils import BeispielAudioRecorder # 从 utils 导入
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

class StreamingPipeline:
    def __init__(
        self,
        asr_model_size=None, 
        llm_model_name=None, 
        device=None, 
        silence_based_segmentation=True, 
        asr_buffer_size_seconds=15, 
        asr_context_window_chunks=3 
    ):
        logger.info("Initializing streaming pipeline...")
        self.device = device if device else DEVICE
        _asr_model_size = asr_model_size if asr_model_size else ASR_MODEL_NAME
        _llm_model_name = llm_model_name if llm_model_name else LLM_MODEL_NAME

        self.asr_streamer = FasterWhisperStreamer(
            model_size=_asr_model_size,
            device=self.device,
            vad_parameters=VAD_PARAMETERS 
        )
        self.llm_streamer = StreamLLMInference(
            model_name=_llm_model_name, 
            device=self.device
        )

        self.silence_based_segmentation = silence_based_segmentation
        self.audio_segmenter = AudioSegmenter(
            min_silence_len_ms=VAD_PARAMETERS.get('min_silence_duration_ms', 1000),
            chunk_duration_s=STREAMING_ASR_CHUNK_SECONDS, 
            overlap_duration_s=STREAMING_ASR_OVERLAP_SECONDS
        )
        
        self.audio_buffer = BeispielAudioRecorder(buffer_duration_seconds=asr_buffer_size_seconds, sample_rate=16000)
        self.transcribed_text_parts = deque() 
        self.current_full_transcript = ""
        self.asr_chunk_history = deque(maxlen=asr_context_window_chunks) 
        self.llm_input_buffer = [] 
        self.last_asr_processed_time_abs = 0.0 
        logger.info("Streaming pipeline initialized.")

    def _process_audio_chunk_for_asr(self, audio_chunk_np, chunk_start_time_abs):
        logger.debug(f"Processing audio chunk for ASR: start_abs={chunk_start_time_abs:.2f}s, duration={len(audio_chunk_np)/16000.0:.2f}s")
        context_prompt = self.current_full_transcript 

        segments, info, trans_time = self.asr_streamer.transcribe_audio_chunk(
            audio_chunk_np,
            language='zh', 
            word_timestamps=True,
            initial_prompt=context_prompt if context_prompt else None
        )
        logger.info(f"ASR for chunk completed in {trans_time:.3f}s. Detected lang: {info.language}")

        chunk_transcript_parts = []
        segment_list = list(segments)

        if not segment_list:
            logger.debug("ASR produced no segments for this chunk.")
            return "" 

        for segment_obj in segment_list:
            processed_segment = self.asr_streamer.process_segment(segment_obj, audio_start_time_abs=chunk_start_time_abs)
            chunk_transcript_parts.append(processed_segment)
            logger.debug(f"  ASR Segment: [{processed_segment['start_abs']:.2f}s -> {processed_segment['end_abs']:.2f}s] {processed_segment['text']}")

        temp_full_text_words = []
        for old_part_info in list(self.transcribed_text_parts):
            if old_part_info['end_abs'] <= chunk_start_time_abs or old_part_info['start_abs'] >= (chunk_start_time_abs + len(audio_chunk_np)/16000.0):
                if old_part_info.get('words_abs'):
                    temp_full_text_words.extend(old_part_info['words_abs'])
        
        for new_part in chunk_transcript_parts:
            if new_part.get('words_abs'):
                temp_full_text_words.extend(new_part['words_abs'])
        
        temp_full_text_words.sort(key=lambda w: w['start'])
        
        final_words = []
        if temp_full_text_words:
            final_words.append(temp_full_text_words[0])
            for i in range(1, len(temp_full_text_words)):
                prev_word = final_words[-1]
                curr_word = temp_full_text_words[i]
                if not (prev_word['text'] == curr_word['text'] and abs(prev_word['start'] - curr_word['start']) < 0.1):
                    final_words.append(curr_word)
        
        self.current_full_transcript = " ".join([word['text'].strip() for word in final_words]).strip()
        newly_transcribed_text = self.current_full_transcript 

        self.transcribed_text_parts.append({
            'start_abs': chunk_start_time_abs,
            'end_abs': chunk_start_time_abs + len(audio_chunk_np)/16000.0,
            'text': " ".join([p['text'] for p in chunk_transcript_parts]),
            'words_abs': [word for p in chunk_transcript_parts for word in p.get('words_abs', [])]
        })

        logger.info(f"Updated full transcript: '{self.current_full_transcript}'")
        return newly_transcribed_text 

    def _process_text_for_llm(self, text_fragment, is_final_asr=False):
        if not text_fragment.strip():
            logger.debug("LLM received empty text fragment, skipping.")
            return None, 0.0

        logger.info(f"Processing text for LLM: '{text_fragment}', is_final_asr: {is_final_asr}")
        self.llm_input_buffer.append(text_fragment)
        llm_response_token = None
        first_token_latency = 0.0

        # Corrected list of punctuation marks
        punctuation_marks = ['。', '？', '！', '.', '?', '!']
        trigger_generation = is_final_asr or any(p in text_fragment for p in punctuation_marks)

        if trigger_generation or not self.llm_streamer.past_key_values:
            full_prompt_for_llm = "".join(self.llm_input_buffer)
            if not self.llm_streamer.past_key_values:
                logger.info(f"LLM: No KV cache. Precomputing for prompt: '{full_prompt_for_llm}'")
                self.llm_streamer.precompute_kv_cache_for_prompt([full_prompt_for_llm])
            else:
                logger.info(f"LLM: Triggering generation. KV will be updated by generate_next_token for: '{full_prompt_for_llm}'")
                pass

            # 如果触发了生成，则生成一个token
            if trigger_generation:
                fragment_for_generate = text_fragment if not self.llm_streamer.past_key_values else ""
                llm_response_token, first_token_latency, is_eos = self.llm_streamer.generate_next_token(new_text_fragment=fragment_for_generate)
                logger.info(f"LLM generated token: '{llm_response_token}', latency: {first_token_latency:.3f}s, is_eos: {is_eos}")
                self.llm_input_buffer = [] # 清空缓冲区
                return llm_response_token, first_token_latency
            else:
                # 只是更新KV缓存，不生成token
                _ , _, _ = self.llm_streamer.generate_next_token(new_text_fragment=text_fragment) # Call to update KV
                logger.info(f"LLM: Updated KV cache with text fragment: '{text_fragment}'")
                return None, 0.0
        else:
            logger.info(f"LLM: Non-triggering input. Updating KV with fragment: '{text_fragment}'")
            _ , _, _ = self.llm_streamer.generate_next_token(new_text_fragment=text_fragment) # Call to update KV
            logger.debug("LLM KV cache updated with new fragment.")
            
        return llm_response_token, first_token_latency

    def process_audio_stream(self, audio_frames_producer):
        logger.info("Starting to process audio stream...")
        
        def simulate_audio_producer():
            total_sim_duration = 20 
            sample_rate = 16000
            block_duration = 0.5 
            current_time_abs = 0.0
            for _ in range(int(total_sim_duration / block_duration)):
                sim_chunk = np.random.uniform(low=-0.2, high=0.2, size=int(sample_rate * block_duration)).astype(np.float32)
                if 2 < current_time_abs < 5 or 8 < current_time_abs < 12 or 15 < current_time_abs < 18:
                    freq = np.random.choice([200, 300, 400])
                    t = np.linspace(0, block_duration, int(sample_rate*block_duration), endpoint=False)
                    speech_signal = 0.5 * np.sin(2 * np.pi * freq * t)
                    sim_chunk = speech_signal.astype(np.float32)
                else:
                    sim_chunk *= 0.1 
                
                yield sim_chunk, current_time_abs
                time.sleep(block_duration) 
                current_time_abs += block_duration
            yield None, current_time_abs 

        if audio_frames_producer is None:
            audio_frames_producer = simulate_audio_producer()

        accumulated_audio_for_segmentation = []
        acc_audio_start_time_abs = -1.0
        min_segmentation_duration_samples = int(0.5 * 16000) 

        for audio_input_block, block_start_time_abs in audio_frames_producer:
            if audio_input_block is None: 
                logger.info("Audio stream ended. Processing any remaining audio.")
                if accumulated_audio_for_segmentation:
                    final_asr_text = self._process_audio_chunk_for_asr(np.concatenate(accumulated_audio_for_segmentation), acc_audio_start_time_abs)
                    if final_asr_text:
                        self._process_text_for_llm(final_asr_text, is_final_asr=True)
                break 

            if acc_audio_start_time_abs < 0: 
                acc_audio_start_time_abs = block_start_time_abs
            accumulated_audio_for_segmentation.append(audio_input_block)
            
            current_accumulated_samples = sum(len(s) for s in accumulated_audio_for_segmentation)

            if current_accumulated_samples < min_segmentation_duration_samples and not self.silence_based_segmentation:
                if current_accumulated_samples < STREAMING_ASR_CHUNK_SECONDS * 16000:
                    continue
            
            combined_audio_np = np.concatenate(accumulated_audio_for_segmentation)

            if self.silence_based_segmentation:
                speech_segments_info = self.audio_segmenter.segment_by_silence(combined_audio_np, acc_audio_start_time_abs)
                processed_upto_sample_in_combined = 0
                for seg_info in speech_segments_info:
                    asr_input_audio = seg_info['audio_np']
                    asr_input_start_abs = seg_info['start_abs']
                    
                    if asr_input_start_abs < self.last_asr_processed_time_abs:
                        continue

                    new_text = self._process_audio_chunk_for_asr(asr_input_audio, asr_input_start_abs)
                    if new_text:
                        self._process_text_for_llm(new_text, is_final_asr=False) 
                    
                    self.last_asr_processed_time_abs = seg_info['end_abs']
                    processed_upto_sample_in_combined = max(processed_upto_sample_in_combined, 
                                                              int((seg_info['end_abs'] - acc_audio_start_time_abs) * 16000))
                
                if processed_upto_sample_in_combined > 0 and processed_upto_sample_in_combined < len(combined_audio_np):
                    accumulated_audio_for_segmentation = [combined_audio_np[processed_upto_sample_in_combined:]]
                    acc_audio_start_time_abs += processed_upto_sample_in_combined / 16000.0
                elif processed_upto_sample_in_combined >= len(combined_audio_np):
                    accumulated_audio_for_segmentation = []
                    acc_audio_start_time_abs = -1.0

            else:
                for chunk_info in self.audio_segmenter.segment_by_fixed_chunks(combined_audio_np, acc_audio_start_time_abs):
                    asr_input_audio = chunk_info['audio_np']
                    asr_input_start_abs = chunk_info['start_abs']
                    asr_input_end_abs = chunk_info['end_abs']

                    chunk_center_time = (asr_input_start_abs + asr_input_end_abs) / 2
                    if chunk_center_time < self.last_asr_processed_time_abs and \
                       self.last_asr_processed_time_abs - asr_input_start_abs > STREAMING_ASR_OVERLAP_SECONDS * 0.75: 
                        logger.debug(f"Skipping largely overlapping fixed chunk: start_abs={asr_input_start_abs:.2f}")
                        continue
                    
                    new_text = self._process_audio_chunk_for_asr(asr_input_audio, asr_input_start_abs)
                    if new_text:
                        self._process_text_for_llm(new_text, is_final_asr=False)
                    
                    self.last_asr_processed_time_abs = max(self.last_asr_processed_time_abs, asr_input_end_abs)

                overlap_samples = int(self.audio_segmenter.overlap_duration_s * 16000)
                if len(combined_audio_np) > overlap_samples:
                    retained_audio = combined_audio_np[-overlap_samples:]
                    retained_start_time = acc_audio_start_time_abs + (len(combined_audio_np) - overlap_samples) / 16000.0
                else:
                    retained_audio = combined_audio_np
                    retained_start_time = acc_audio_start_time_abs
                
                accumulated_audio_for_segmentation = [retained_audio] if len(retained_audio) > 0 else []
                acc_audio_start_time_abs = retained_start_time if len(retained_audio) > 0 else -1.0

        logger.info("Audio stream processing finished.")

if __name__ == '__main__':
    from src.utils.logging_utils import set_global_log_level
    set_global_log_level('INFO')
    pipeline = StreamingPipeline(silence_based_segmentation=False, asr_buffer_size_seconds=5)
    pipeline.process_audio_stream(audio_frames_producer=None) 

    print("\n--- Final ASR Transcript ---")
    print(pipeline.current_full_transcript)

    if pipeline.llm_streamer.past_key_values:
        print("\n--- LLM Final State --- (example of generating more if needed)")
        pipeline.llm_streamer._log_kv_cache_size(pipeline.llm_streamer.past_key_values)
    else:
        print("\n--- LLM Final State ---: No LLM generation was triggered or completed successfully.")

    print("\nPipeline demonstration finished.") 