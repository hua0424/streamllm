#!/usr/bin/env bash
set -euo pipefail

: "${DIALOGUES:?Set DIALOGUES to the formal p2_turns.json path}"
: "${MAIN_MODEL:?Set MAIN_MODEL to the local Qwen2-7B-Instruct path}"
: "${JUDGE_MODEL:?Set JUDGE_MODEL to the local Mistral judge path}"
: "${OUT_ROOT:?Set OUT_ROOT to the supplementary result root}"
: "${CAMPAIGN:?Set CAMPAIGN to a unique fixed run id}"

export HF_TOKEN=
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

SEED="${SEED:-20260831}"
A1_REPEATS="${A1_REPEATS:-20}"
A1_WARMUP="${A1_WARMUP:-3}"
DEVICE="${DEVICE:-cuda:0}"

uv run python -m experiments.sci34_supplement.smoke

CUDA_VISIBLE_DEVICES=0 uv run python -m experiments.sci34_supplement.e3_fixed_trajectory \
  --dialogues "$DIALOGUES" \
  --run-id "${CAMPAIGN}_e3" \
  --results-root "$OUT_ROOT/e3" \
  --runtime transformers \
  --model "$MAIN_MODEL" \
  --device "$DEVICE" \
  --formal \
  --limit 100 \
  --max-first-tokens 40 \
  --max-probe-tokens 40 \
  --seed "$SEED" \
  --resume

CUDA_VISIBLE_DEVICES=1 uv run python -m experiments.sci34_supplement.e3_judge \
  --e3-run-dir "$OUT_ROOT/e3/${CAMPAIGN}_e3" \
  --run-id "${CAMPAIGN}_judge" \
  --results-root "$OUT_ROOT/judge" \
  --judge-model "$JUDGE_MODEL" \
  --device cuda:0 \
  --resume

uv run python -m experiments.sci34_supplement.analyze_e3 \
  --e3-run-dir "$OUT_ROOT/e3/${CAMPAIGN}_e3" \
  --judge-records "$OUT_ROOT/judge/${CAMPAIGN}_judge/judge_records.jsonl" \
  --bootstrap-repeats 10000 \
  --seed "$SEED"

CUDA_VISIBLE_DEVICES=0 uv run python -m experiments.sci34_supplement.a1_joint_latency \
  --run-id "${CAMPAIGN}_a1" \
  --results-root "$OUT_ROOT/a1" \
  --runtime transformers \
  --model "$MAIN_MODEL" \
  --device "$DEVICE" \
  --lengths 256 512 1024 2048 4096 8192 \
  --crop-tokens 32 \
  --warmup "$A1_WARMUP" \
  --repeats "$A1_REPEATS" \
  --resume

uv run python -m experiments.sci34_supplement.analyze_latency \
  --run-dir "$OUT_ROOT/a1/${CAMPAIGN}_a1" \
  --kind a1

CUDA_VISIBLE_DEVICES=0 uv run python -m experiments.sci34_supplement.async_bargein \
  --run-id "${CAMPAIGN}_async" \
  --results-root "$OUT_ROOT/async_bargein" \
  --runtime transformers \
  --model "$MAIN_MODEL" \
  --device "$DEVICE" \
  --lengths 512 2048 8192 \
  --fractions 0.25 0.5 0.75 \
  --repeats 20 \
  --sample-rate 24000 \
  --duration-s 0.8 \
  --fragments 4 \
  --block-ms 20 \
  --time-scale 1 \
  --resume

uv run python -m experiments.sci34_supplement.analyze_latency \
  --run-dir "$OUT_ROOT/async_bargein/${CAMPAIGN}_async" \
  --kind async

printf 'SCI supplement completed: %s\n' "$CAMPAIGN"
