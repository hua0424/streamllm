# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research codebase for low-latency cascaded speech dialogue (ASR → LLM → TTS). The thesis claim being validated: in a streaming pipeline, ASR text fragments are fed into the LLM's KV cache *as they are recognized*, so the LLM has pre-filled most of the prompt by the time the user stops speaking — making **Time-To-First-Token (TTFT)** roughly constant w.r.t. utterance length, instead of growing linearly as in the non-streaming baseline. Code, experiments, and the thesis (`paper/`) all serve this one claim.

## Commands

Environment is `uv` (not pip/conda). Always run modules with `uv run python -m ...` from the repo root — code uses absolute `from src.config import ...` imports, so `python path/to/file.py` breaks.

```bash
uv venv --python 3.10 && uv sync          # setup
uv run python -m src.run_test_simple --mode both --audio path/to.wav   # e2e latency A/B
uv run python -m src.asr.run_stream_asr_test     # ASR-only smoke
uv run python -m src.llm.run_llm_test            # LLM-only smoke
```

Experiments (each supports checkpointing + resume; outputs land in `experiments/results/expN_*/`):
```bash
uv run python -m experiments.scripts.run_exp_latency    # exp1: TTFT vs utterance length
uv run python -m experiments.scripts.run_exp_ablation   # exp2: ablation (streaming-ASR vs +KV-prefill)
uv run python -m experiments.scripts.run_exp_quality    # exp3: output quality
```

No pytest suite — validate by running the harnesses above and reading the printed latency numbers / saved JSON+CSV. The `*.sh` wrappers in `src/` and `experiments/scripts/` are thin Bash wrappers around these module invocations.

## Architecture

Two engines, glued by the test harnesses. There is no always-on server — everything is run as a benchmark.

**`src/llm/stream_llm_inference.py` — `StreamLLMInference`** is the core of the thesis. Manual KV-cache management, *not* `model.generate()`:
- `cache_prompt(prompt, pre_cache, is_end)` — incremental prefill. First call (`pre_cache=None`) builds the cache from the chat template up to (but excluding) the generation prompt; later calls run a forward pass on *only the new tokens* with the prior `past_key_values`, manually concatenating `attention_mask` and computing explicit `position_ids` (line ~290 — getting these wrong corrupts positional encoding silently). `is_end=True` appends the stored `generation_prompt` to trigger the assistant turn.
- `KVCache` (inner class) carries `past_key_values`, ids, mask, **and `next_token_logits`** — the prefill already produced the first token's logits, so `generate()` decodes token 0 with zero extra forward passes. That's where the TTFT win comes from.
- `generate()` honors `self.eval_mode`: when `True` it yields exactly ONE token then breaks (we only measure first-token latency, not full decode). Set `eval_mode=False` for real multi-token output.
- `once_add_and_generate()` is the non-streaming baseline (full prompt in one shot) for A/B comparison.

**`src/asr/faster_whisper_streamer.py` — `StreamingASRProcessor` + `ASRCache`**: uses `openai-whisper` (PyTorch) despite the filename, plus Silero VAD for segmentation. Loads offline-first from local cache. Sliding window keeps `prefix_segments` as recognition context; emits only "confirmed" text fragments to forward into the LLM cache.

Both engines try `local_files_only=True` first, then fall back to network download — designed to run on air-gapped GPU boxes.

**Timing** is captured via `StreamLLMInference.TimingEventType` enum events written into `self.timing_events` (a dict keyed by event type → `time.perf_counter()`). Read them with `get_last_timings()` after a call; `reset_timings()` is called at the start of each public method. This is the measurement substrate the whole project rests on — don't reorder or drop these markers when editing the inference loop.

## Config

All config flows through `src/config.py`, which reads `.env` via `python-dotenv`. Never hard-code model names, devices, or paths in modules — add an env var with a default in `config.py` instead (and document it in `.env.example` if introducing one). Key knobs: `DEVICE` (auto/cuda/cpu), `LLM_MODEL_NAME` (default `Qwen/Qwen2.5-0.5B-Instruct`), `ASR_MODEL_NAME` (whisper size, default `tiny`), `HF_HOME`, `VAD_*`, `ASR_CHUNK_SECONDS`.

## Conventions

- Logging via `src/utils/logging_utils.get_logger(__name__)`, not `print()`, in core code.
- Code comments and docstrings are largely Chinese; commit messages too (short imperative: 修复/增加/调整/完成/更新). Match the surrounding language when editing a file.
- If you change experimental methodology, update `experiments/EXPERIMENT_DESIGN.md`. The thesis drafts in `paper/` cite specific result files — don't silently regenerate result CSVs without noting it.
