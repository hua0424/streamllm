# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands
- **Run ASR tests**: `python test_streaming.py`
- **Run LLM inference tests**: `python test_streaming.py --test-llm`
- **Run pipeline tests**: `python test_streaming.py --test-pipeline`

## Environment Setup
- This project uses a conda-managed uv environment
- make enviroment the same as user terminal: `source /root/.bashrc`
- Activate the environment before running programs: `conda activate uv`
- Use uv commands to run programs, e.g., `uv run python -m test_streaming`
- **Package Installation**: Use `uv add` to install packages in this project

## File Path Guidelines for Claude Code
- **Working Directory**: `/usr/local/app/jupyterlab/yanjiu/streamllm`
- **Always use absolute paths** when editing files to avoid "file not found" errors
- Key files with absolute paths:
  - Main test: `/usr/local/app/jupyterlab/yanjiu/streamllm/test_streaming.py`
  - Config: `/usr/local/app/jupyterlab/yanjiu/streamllm/.env`
  - ASR module: `/usr/local/app/jupyterlab/yanjiu/streamllm/src/asr/faster_whisper_streamer.py`
  - LLM module: `/usr/local/app/jupyterlab/yanjiu/streamllm/src/llm/stream_llm_inference.py`

## Architecture Overview
- **ASR Module**: Uses `faster-whisper` for streaming ASR with dynamic segmentation (`src/asr/faster_whisper_streamer.py`).
- **LLM Module**: Implements KV caching for low-latency streaming responses (`src/llm/stream_llm_inference.py`).
- **Pipeline**: Orchestrates ASR -> LLM flow (`src/pipeline/streaming_pipeline.py`).
- **Utils**: Includes audio processing (`src/utils/audio2stream.py`) and logging (`src/utils/logging_utils.py`).