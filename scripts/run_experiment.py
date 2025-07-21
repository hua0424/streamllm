import os
import json
import time
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import glob

from src.pipeline.streaming_pipeline import StreamingPipeline
from src.utils.audio_utils import AudioUtils
from src.utils.logging_utils import get_logger
from src.config import (
    ASR_MODEL_NAME, LLM_MODEL_NAME, DEVICE, 
    PROCESSED_AUDIO_DIR, TRANSCRIPTS_DIR, EXPERIMENT_RESULTS_DIR,
    STREAMING_ASR_CHUNK_SECONDS, STREAMING_ASR_OVERLAP_SECONDS,
    VAD_SILENCE_MS, MIN_SPEECH_DURATION_S
)

logger = get_logger(__name__)

def run_single_experiment(audio_file_path, reference_transcript_path, config_override=None):
    logger.info(f"Running experiment for audio: {audio_file_path}")
    if reference_transcript_path and os.path.exists(reference_transcript_path):
        with open(reference_transcript_path, 'r', encoding='utf-8') as f:
            reference_transcript = f.read().strip()
    else:
        reference_transcript = ""
        logger.warning(f"Reference transcript not found or not provided for {audio_file_path}")

    results = {
        "audio_file": os.path.basename(audio_file_path),
        "reference_transcript": reference_transcript,
        "generated_transcript_asr_only": "",
        "generated_transcript_llm_input": "",
        "llm_first_token_response": "",
        "llm_full_response": "",
        "timestamps": [], # (event_type, absolute_time_ms, relative_time_ms_since_start, data)
        "latencies": {
            "asr_total_processing_time_s": 0.0,
            "llm_first_token_latency_s": -1.0, # Time from end of final ASR to first LLM token
            "llm_time_to_first_response_s": -1.0, # Time from start of audio to first LLM token
            "total_pipeline_time_s": 0.0,
            "asr_segment_latencies": [], # [(segment_duration_s, processing_time_s)]
            "llm_kv_precompute_latencies_s": [],
            "llm_token_generation_latencies_s": []
        },
        "config": config_override or {},
        "error": None
    }
    pipeline_start_time = time.time()
    current_abs_time_offset = 0 # Simulates the absolute time for audio chunks

    try:
        # 覆盖默认配置（如果提供）
        effective_asr_model = config_override.get("asr_model_size", ASR_MODEL_NAME) if config_override else ASR_MODEL_NAME
        effective_llm_model = config_override.get("llm_model_name", LLM_MODEL_NAME) if config_override else LLM_MODEL_NAME
        effective_device = config_override.get("device", DEVICE) if config_override else DEVICE
        silence_segmentation = config_override.get("silence_based_segmentation", True) if config_override else True
        asr_buffer_s = config_override.get("asr_buffer_size_seconds", 15) if config_override else 15
        asr_context_chunks = config_override.get("asr_context_window_chunks", 3) if config_override else 3

        pipeline = StreamingPipeline(
            asr_model_size=effective_asr_model,
            llm_model_name=effective_llm_model,
            device=effective_device,
            silence_based_segmentation=silence_segmentation,
            asr_buffer_size_seconds=asr_buffer_s,
            asr_context_window_chunks=asr_context_chunks
        )

        results["config"] = {
            "asr_model_size": effective_asr_model,
            "llm_model_name": effective_llm_model,
            "device": effective_device,
            "silence_based_segmentation": silence_segmentation,
            "asr_buffer_size_seconds": asr_buffer_s,
            "asr_context_window_chunks": asr_context_chunks,
            "vad_min_silence_ms": pipeline.audio_segmenter.min_silence_len_ms,
            "asr_chunk_s": pipeline.audio_segmenter.chunk_duration_s,
            "asr_overlap_s": pipeline.audio_segmenter.overlap_duration_s
        }

        audio_data_np, sr = AudioUtils.load_audio(audio_file_path, sample_rate=16000)
        if sr != 16000:
            raise ValueError(f"Audio sample rate is {sr}, but 16000 is required.")
        results["audio_duration_s"] = len(audio_data_np) / sr
        results["timestamps"].append(("pipeline_start", time.time() * 1000, 0, None))

        # 模拟音频流输入
        # 我们将整个音频文件一次性加载，然后切片模拟流式输入
        # 这有助于控制实验的可复现性，而不是依赖实时麦克风输入
        # chunk_duration_ms = 500 # 模拟每次送入0.5秒的音频
        # chunk_samples = int(sr * (chunk_duration_ms / 1000.0))
        # 实际的pipeline.process_audio_stream处理的是更小的音频块，然后内部做VAD或固定分块
        # 所以这里的producer应该模拟pipeline的外部调用者，它可能以任意大小的块提供音频

        simulated_block_duration_s = 0.5 # 模拟外部调用方每0.5s提供一次数据
        simulated_block_samples = int(sr * simulated_block_duration_s)

        def audio_frames_producer(full_audio_data, block_samples, total_duration_s):
            nonlocal current_abs_time_offset
            idx = 0
            start_event_time = time.time()
            while idx < len(full_audio_data):
                chunk_processing_start_time = time.time()
                end_idx = min(idx + block_samples, len(full_audio_data))
                audio_chunk = full_audio_data[idx:end_idx]
                
                # 模拟真实时间流逝，但我们希望实验尽可能快
                # time.sleep(len(audio_chunk) / sr) # 真实时间流逝
                # 为了实验，我们不sleep，而是记录绝对时间戳
                
                chunk_abs_start_time = current_abs_time_offset
                results["timestamps"].append((
                    "audio_chunk_sent_to_pipeline", 
                    time.time() * 1000, 
                    (time.time() - pipeline_start_time) * 1000,
                    {"chunk_abs_start_s": chunk_abs_start_time, "chunk_duration_s": len(audio_chunk)/sr}
                ))
                
                yield audio_chunk, chunk_abs_start_time
                
                current_abs_time_offset += len(audio_chunk) / sr
                idx += block_samples
                
                # 模拟外部块之间的延迟，对于文件处理，可以为0
                # time.sleep(0.01) # 可选的小延迟
            
            # 发送结束信号
            results["timestamps"].append((
                "audio_stream_end_sent", 
                time.time() * 1000, 
                (time.time() - pipeline_start_time) * 1000,
                {"final_abs_time_s": current_abs_time_offset}
            ))
            yield None, current_abs_time_offset
            logger.info("Audio frames producer finished.")

        # ---- Hook into pipeline internals for more detailed logging ----
        original_asr_process_chunk = pipeline._process_audio_chunk_for_asr
        original_llm_process_text = pipeline._process_text_for_llm
        
        asr_processing_times = []
        llm_kv_precompute_times = []
        llm_token_gen_times = []

        def hooked_asr_process_chunk(audio_chunk_np, chunk_start_time_abs):
            asr_event_start_time = time.time()
            newly_transcribed_text = original_asr_process_chunk(audio_chunk_np, chunk_start_time_abs)
            asr_event_end_time = time.time()
            duration_s = asr_event_end_time - asr_event_start_time
            asr_processing_times.append(duration_s)
            results["latencies"]["asr_segment_latencies"].append((len(audio_chunk_np)/sr, duration_s))
            results["timestamps"].append((
                "asr_chunk_processed", 
                asr_event_end_time * 1000, 
                (asr_event_end_time - pipeline_start_time) * 1000,
                {
                    "chunk_abs_start_s": chunk_start_time_abs, 
                    "chunk_duration_s": len(audio_chunk_np)/sr, 
                    "processing_time_s": duration_s,
                    "transcribed_text": newly_transcribed_text
                }
            ))
            return newly_transcribed_text

        def hooked_llm_process_text(text_fragment, is_final_asr=False):
            llm_event_start_time = time.time()
            llm_response_token, first_token_latency_s = original_llm_process_text(text_fragment, is_final_asr)
            llm_event_end_time = time.time()
            
            # 从LLM模块获取详细计时
            llm_timings = pipeline.llm_streamer.get_last_timings()
            if llm_timings["last_precompute_time_ms"] > 0:
                llm_kv_precompute_times.append(llm_timings["last_precompute_time_ms"] / 1000.0)
            if llm_timings["last_token_gen_time_ms"] > 0 and first_token_latency_s > 0: #确保是生成token的计时
                llm_token_gen_times.append(llm_timings["last_token_gen_time_ms"] / 1000.0)

            if llm_response_token and results["latencies"]["llm_first_token_latency_s"] < 0:
                results["latencies"]["llm_first_token_latency_s"] = first_token_latency_s
                results["latencies"]["llm_time_to_first_response_s"] = llm_event_end_time - pipeline_start_time
                results["llm_first_token_response"] = llm_response_token
                results["timestamps"].append((
                    "llm_first_token_generated", 
                    llm_event_end_time * 1000, 
                    (llm_event_end_time - pipeline_start_time) * 1000,
                    {
                        "token": llm_response_token, 
                        "latency_from_asr_end_s": first_token_latency_s,
                        "latency_from_audio_start_s": results["latencies"]["llm_time_to_first_response_s"],
                        "input_text_fragment": text_fragment
                    }
                ))
            
            results["timestamps"].append((
                "llm_text_processed", 
                llm_event_end_time * 1000, 
                (llm_event_end_time - pipeline_start_time) * 1000,
                {
                    "input_text_fragment": text_fragment,
                    "is_final_asr": is_final_asr,
                    "generated_token_this_step": llm_response_token,
                    "llm_reported_ftl_s": first_token_latency_s,
                    "kv_precompute_ms": llm_timings["last_precompute_time_ms"],
                    "token_gen_ms": llm_timings["last_token_gen_time_ms"]
                }
            ))
            return llm_response_token, first_token_latency_s

        pipeline._process_audio_chunk_for_asr = hooked_asr_process_chunk
        pipeline._process_text_for_llm = hooked_llm_process_text
        # ---- End Hooks ----

        producer = audio_frames_producer(audio_data_np, simulated_block_samples, results["audio_duration_s"])
        pipeline.process_audio_stream(audio_frames_producer=producer)
        
        results["generated_transcript_asr_only"] = pipeline.current_full_transcript
        results["generated_transcript_llm_input"] = "".join(pipeline.llm_input_buffer)
        
        # 如果想获取完整的LLM回复（如果适用）
        # 这个pipeline设计主要是流式的，可能没有一个明确的"完整LLM回复"
        # 除非我们在最后用累积的KV缓存生成一次
        if pipeline.llm_streamer.past_key_values and not results["llm_first_token_response"]:
            logger.info("No first token generated during stream, attempting to generate full response from final KV cache.")
            # 确保generate_full_response_with_cache能用当前的KV状态
            # 需要修改 StreamLLMInference 使其能从当前状态继续生成，或提供这样一个接口
            # 暂时假设，如果 first_token 未生成，则可能没有 LLM 输出
            pass 
        elif results["llm_first_token_response"]:
            # 尝试从第一个token开始继续生成一个短回复作为示例
            # 注意: 这会改变LLM的状态，如果后面有其他评估就不合适了
            # 这里只是为了获取一个示例输出
            temp_full_response = [results["llm_first_token_response"]]
            for _ in range(10): # Gen a few more tokens
                next_tok, _, is_eos = pipeline.llm_streamer.generate_next_token()
                if next_tok and next_tok not in [pipeline.llm_streamer.tokenizer.eos_token, "<|im_end|>"] and not is_eos:
                    temp_full_response.append(next_tok)
                else:
                    break
            results["llm_full_response"] = "".join(temp_full_response)

        results["latencies"]["asr_total_processing_time_s"] = sum(asr_processing_times)
        if llm_kv_precompute_times:
            results["latencies"]["llm_kv_precompute_latencies_s"] = llm_kv_precompute_times
        if llm_token_gen_times:
            results["latencies"]["llm_token_generation_latencies_s"] = llm_token_gen_times

    except Exception as e:
        logger.error(f"Error during experiment for {audio_file_path}: {e}", exc_info=True)
        results["error"] = str(e)
    finally:
        pipeline_end_time = time.time()
        results["latencies"]["total_pipeline_time_s"] = pipeline_end_time - pipeline_start_time
        results["timestamps"].append(("pipeline_end", pipeline_end_time * 1000, (pipeline_end_time - pipeline_start_time) * 1000, None))
        logger.info(f"Experiment for {audio_file_path} finished. Total time: {results['latencies']['total_pipeline_time_s']:.3f}s")
        
        # 还原，以防pipeline对象被重用（虽然在此脚本中不会）
        if 'pipeline' in locals() and hasattr(pipeline, '_process_audio_chunk_for_asr_original'): # Check if original_asr_process_chunk was defined
            pipeline._process_audio_chunk_for_asr = original_asr_process_chunk
            pipeline._process_text_for_llm = original_llm_process_text

    return results

def run_batch_experiments(data_dir, output_dir, num_files=None, config_override_file=None):
    logger.info(f"Starting batch experiments. Data dir: {data_dir}, Output dir: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    config_overrides = []
    if config_override_file and os.path.exists(config_override_file):
        with open(config_override_file, 'r', encoding='utf-8') as f:
            loaded_configs = json.load(f)
            if isinstance(loaded_configs, list): # Expect a list of configs
                config_overrides = loaded_configs
            else: # If it's a single dict, treat as one config
                config_overrides = [loaded_configs]
        logger.info(f"Loaded {len(config_overrides)} config overrides from {config_override_file}")
    
    if not config_overrides: # 如果没有覆盖文件或文件为空，则使用默认配置运行一次
        config_overrides.append({"name": "default_config"}) # Ensure 'name' key for consistency

    audio_files_pattern = os.path.join(data_dir, PROCESSED_AUDIO_DIR, "*.wav")
    text_files_dir = TRANSCRIPTS_DIR 

    audio_files = sorted(glob.glob(audio_files_pattern)) # Sort for consistent order
    if num_files is not None and num_files > 0:
        audio_files = audio_files[:num_files]
    
    logger.info(f"Found {len(audio_files)} audio files to process.")
    if not audio_files:
        logger.warning("No audio files found. Exiting.")
        return

    all_experiment_results = []

    for i, config_item in enumerate(config_overrides):
        current_config = config_item.copy() # Work with a copy
        config_name = current_config.pop("name", f"config_{i}")
        logger.info(f"Running experiments with configuration: {config_name}")
        config_output_dir = os.path.join(output_dir, config_name)
        os.makedirs(config_output_dir, exist_ok=True)
        
        experiment_suite_results = []
        for audio_file in tqdm(audio_files, desc=f"Experiments for {config_name}"):
            base_name = os.path.splitext(os.path.basename(audio_file))[0]
            transcript_file = os.path.join(text_files_dir, base_name + ".txt")
            
            single_run_results = run_single_experiment(audio_file, transcript_file, config_override=current_config)
            single_run_results["config_name"] = config_name 
            experiment_suite_results.append(single_run_results)
            
            output_file_path = os.path.join(config_output_dir, f"{base_name}_results.json")
            with open(output_file_path, 'w', encoding='utf-8') as f_out:
                json.dump(single_run_results, f_out, ensure_ascii=False, indent=4, cls=NpEncoder)
        
        all_experiment_results.extend(experiment_suite_results)
        logger.info(f"Finished all experiments for configuration: {config_name}")

    summary_file_path = os.path.join(output_dir, "all_experiments_summary.json")
    with open(summary_file_path, 'w', encoding='utf-8') as f_summary:
        json.dump(all_experiment_results, f_summary, ensure_ascii=False, indent=4, cls=NpEncoder)
    logger.info(f"All batch experiments finished. Summary saved to {summary_file_path}")

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run streaming ASR-LLM experiments.")
    parser.add_argument("--data_dir", type=str, default="data", 
                        help="Root directory containing processed_audio and transcripts.")
    parser.add_argument("--output_dir", type=str, default=os.path.join("results", "experiment_outputs"), 
                        help="Directory to save experiment results.")
    parser.add_argument("--num_files", type=int, default=None, 
                        help="Number of audio files to process (for quick testing).")
    parser.add_argument("--config_file", type=str, default=None, 
                        help="Path to a JSON file containing a list of configurations to run.")
    parser.add_argument("--single_audio_file", type=str, default=None, help="Path to a single audio file to test.")
    parser.add_argument("--reference_transcript", type=str, default=None, help="Path to the reference transcript for the single audio file.")

    args = parser.parse_args()

    # Ensure EXPERIMENT_RESULTS_DIR and output_dir exist
    for dir_path in [EXPERIMENT_RESULTS_DIR, args.output_dir]:
        if not os.path.isabs(dir_path): # If relative, ensure base 'results' exists if it's the parent
             # Check if the parent of the output_dir is 'results' when output_dir is relative
            if os.path.dirname(dir_path) == "results" and not os.path.exists("results"):
                 os.makedirs("results", exist_ok=True)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)


    if args.single_audio_file:
        if not os.path.exists(args.single_audio_file):
            logger.error(f"Single audio file not found: {args.single_audio_file}")
            exit(1)
        
        config_override_single = None
        if args.config_file and os.path.exists(args.config_file):
            with open(args.config_file, 'r', encoding='utf-8') as f:
                all_configs = json.load(f)
                if all_configs :
                    if isinstance(all_configs, list) and all_configs:
                        config_override_single = all_configs[0].copy() # Use a copy of the first config
                        config_override_single.pop("name", None) # Remove name if present, not needed for single run
                        logger.info(f"Using first configuration from {args.config_file} for single run.")
                    elif isinstance(all_configs, dict): # If it's a single config dict
                        config_override_single = all_configs.copy()
                        config_override_single.pop("name", None)
                        logger.info(f"Using the configuration from {args.config_file} for single run.")


        single_results = run_single_experiment(args.single_audio_file, args.reference_transcript, config_override_single)
        results_file_name = f"{os.path.splitext(os.path.basename(args.single_audio_file))[0]}_results.json"
        final_results_path = os.path.join(args.output_dir, results_file_name)
        
        with open(final_results_path, 'w', encoding='utf-8') as f:
            json.dump(single_results, f, ensure_ascii=False, indent=4, cls=NpEncoder)
        logger.info(f"Single experiment finished. Results saved to {final_results_path}")
    else:
        run_batch_experiments(args.data_dir, args.output_dir, args.num_files, args.config_file)

    logger.info("Experiment run script finished.")
 