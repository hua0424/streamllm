# scripts/evaluate_results.py

import os
import json
import argparse
import pandas as pd
import numpy as np
import jiwer
import glob
from collections import defaultdict

from src.utils.logging_utils import get_logger
from src.config import EXPERIMENT_RESULTS_DIR # Optional: use for default input

logger = get_logger(__name__)

# Helper function to calculate WER, handling empty strings
def calculate_wer(reference, hypothesis):
    if not reference and not hypothesis:
        return 0.0
    if not reference:
        return 1.0 # All hypothesized words are errors if reference is empty
    if not hypothesis:
        return 1.0 # All reference words are missed if hypothesis is empty
    try:
        return jiwer.wer(reference, hypothesis)
    except Exception as e:
        logger.error(f"Error calculating WER for ref='{reference}', hyp='{hypothesis}': {e}")
        return 1.0 # Return worst case if error

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def analyze_experiment_results(results_file_path):
    """Analyzes a single experiment result JSON file."""
    try:
        with open(results_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load or parse results file {results_file_path}: {e}")
        return None

    if data.get("error"):
        logger.warning(f"Experiment for {data.get('audio_file', 'Unknown_file')} had an error: {data['error']}")
        # Still, we might have partial data to analyze, or we can mark it as failed
        # For now, we'll try to extract what we can

    metrics = {
        "audio_file": data.get("audio_file", os.path.basename(results_file_path)),
        "config_name": data.get("config_name", "default"),
        "audio_duration_s": data.get("audio_duration_s", 0),
        "wer_asr_only": None,
        "wer_llm_input": None, # WER of the text that was fed to LLM vs reference
        "ftl_asr_end_to_llm_token_s": data.get("latencies", {}).get("llm_first_token_latency_s", -1),
        "ftl_audio_start_to_llm_token_s": data.get("latencies", {}).get("llm_time_to_first_response_s", -1),
        "total_pipeline_time_s": data.get("latencies", {}).get("total_pipeline_time_s", 0),
        "num_asr_segments": len(data.get("latencies", {}).get("asr_segment_latencies", [])),
        "avg_asr_segment_processing_time_s": np.mean([s[1] for s in data.get("latencies", {}).get("asr_segment_latencies", [])]) if data.get("latencies", {}).get("asr_segment_latencies") else 0,
        "avg_llm_kv_precompute_s": np.mean(data.get("latencies", {}).get("llm_kv_precompute_latencies_s", [])) if data.get("latencies", {}).get("llm_kv_precompute_latencies_s") else 0,
        "avg_llm_token_generation_s": np.mean(data.get("latencies", {}).get("llm_token_generation_latencies_s", [])) if data.get("latencies", {}).get("llm_token_generation_latencies_s") else 0,
        "llm_first_token_response": data.get("llm_first_token_response", ""),
        "llm_full_response_preview": data.get("llm_full_response", "")[:100], # Preview of full response
        "reference_transcript_length": len(data.get("reference_transcript", "").split()),
        "asr_transcript_length": len(data.get("generated_transcript_asr_only", "").split()),
        "error_in_run": data.get("error")
    }

    ref_transcript = data.get("reference_transcript", "")
    asr_transcript = data.get("generated_transcript_asr_only", "")
    llm_input_transcript = data.get("generated_transcript_llm_input", "")

    if ref_transcript and asr_transcript:
        metrics["wer_asr_only"] = calculate_wer(ref_transcript, asr_transcript)
    
    if ref_transcript and llm_input_transcript: # Compare what LLM received vs original reference
        metrics["wer_llm_input"] = calculate_wer(ref_transcript, llm_input_transcript)
    elif not llm_input_transcript and asr_transcript: # If LLM input is empty, use ASR output for this metric
         metrics["wer_llm_input"] = metrics["wer_asr_only"] 

    return metrics

def process_results_directory(results_dir, output_csv_path=None, output_summary_path=None):
    logger.info(f"Processing experiment results from directory: {results_dir}")
    json_files = glob.glob(os.path.join(results_dir, "**", "*_results.json"), recursive=True)
    
    if not json_files:
        logger.warning(f"No JSON result files found in {results_dir} or its subdirectories.")
        return

    all_metrics = []
    for file_path in json_files:
        logger.debug(f"Analyzing file: {file_path}")
        analysis = analyze_experiment_results(file_path)
        if analysis:
            all_metrics.append(analysis)
    
    if not all_metrics:
        logger.warning("No valid metrics could be extracted from the result files.")
        return

    df_metrics = pd.DataFrame(all_metrics)
    
    if output_csv_path:
        try:
            df_metrics.to_csv(output_csv_path, index=False, encoding='utf-8-sig') # utf-8-sig for Excel compatibility
            logger.info(f"Detailed metrics saved to CSV: {output_csv_path}")
        except Exception as e:
            logger.error(f"Failed to save detailed metrics CSV to {output_csv_path}: {e}")

    # Generate summary statistics
    summary_stats = {
        "total_files_processed": len(df_metrics),
        "files_with_errors": int(df_metrics["error_in_run"].notna().sum()),
        "overall_avg_wer_asr_only": df_metrics["wer_asr_only"].mean(),
        "overall_avg_wer_llm_input": df_metrics["wer_llm_input"].mean(),
        "overall_avg_ftl_audio_start_to_llm_token_s": df_metrics[df_metrics["ftl_audio_start_to_llm_token_s"] >= 0]["ftl_audio_start_to_llm_token_s"].mean(),
        "overall_avg_total_pipeline_time_s": df_metrics["total_pipeline_time_s"].mean(),
        "avg_audio_duration_s": df_metrics["audio_duration_s"].mean()
    }
    
    # Per-config summary (if 'config_name' column exists)
    if "config_name" in df_metrics.columns:
        summary_stats["per_config_summary"] = df_metrics.groupby("config_name").agg(
            num_files=("audio_file", "count"),
            avg_wer_asr_only=("wer_asr_only", "mean"),
            avg_wer_llm_input=("wer_llm_input", "mean"),
            avg_ftl_s=("ftl_audio_start_to_llm_token_s", lambda x: x[x>=0].mean()), # Only avg valid FTLs
            avg_pipeline_time_s=("total_pipeline_time_s", "mean"),
            files_with_errors=("error_in_run", lambda x: x.notna().sum())
        ).reset_index().to_dict(orient='records')

    logger.info("Summary Statistics:")
    logger.info(json.dumps(summary_stats, indent=4, cls=NpEncoder))

    if output_summary_path:
        try:
            with open(output_summary_path, 'w', encoding='utf-8') as f_summary:
                json.dump(summary_stats, f_summary, ensure_ascii=False, indent=4, cls=NpEncoder)
            logger.info(f"Summary statistics saved to JSON: {output_summary_path}")
        except Exception as e:
            logger.error(f"Failed to save summary JSON to {output_summary_path}: {e}")
    
    return df_metrics, summary_stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate experiment results.")
    parser.add_argument("--results_dir", type=str, 
                        default=os.path.join("results", "experiment_outputs"), 
                        help="Directory containing the experiment JSON result files (can have subfolders per config). Defaults to results/experiment_outputs.")
    parser.add_argument("--output_csv", type=str, 
                        default=os.path.join("results", "evaluation", "detailed_metrics.csv"),
                        help="Path to save the detailed metrics as a CSV file. Defaults to results/evaluation/detailed_metrics.csv")
    parser.add_argument("--output_summary_json", type=str, 
                        default=os.path.join("results", "evaluation", "summary_statistics.json"),
                        help="Path to save the summary statistics as a JSON file. Defaults to results/evaluation/summary_statistics.json")

    args = parser.parse_args()

    # Create output directories if they don't exist
    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    if args.output_summary_json:
        os.makedirs(os.path.dirname(args.output_summary_json), exist_ok=True)

    if not os.path.isdir(args.results_dir):
        logger.error(f"Results directory not found: {args.results_dir}")
        exit(1)

    process_results_directory(args.results_dir, args.output_csv, args.output_summary_json)

    logger.info("Evaluation script finished.") 