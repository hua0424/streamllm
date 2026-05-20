# Repository Guidelines

## Project Structure & Module Organization
- `src/`: core implementation
  - `src/asr/`: streaming ASR (Whisper/Faster-Whisper + VAD)
  - `src/llm/`: streaming LLM inference + KV cache
  - `src/utils/`: audio + logging utilities
  - `src/run_test_simple.py`: end-to-end latency comparison harness
- `experiments/`: experiment design, scripts, datasets, and saved outputs (`experiments/results/`)
- `notebooks/`: exploratory analysis and prototyping
- `paper/`: thesis drafts and references (Markdown)

## Build, Test, and Development Commands
Preferred environment manager is `uv` (see `pyproject.toml` / `uv.lock`):
```bash
uv venv --python 3.10
uv sync
```

Run the end-to-end benchmark (recommended cross-platform entrypoint):
```bash
uv run python -m src.run_test_simple --mode both --audio path/to.wav
```

Module-level smoke runs:
```bash
uv run python -m src.asr.run_stream_asr_test
uv run python -m src.llm.run_llm_test
```

## Coding Style & Naming Conventions
- Python 3.10+, 4-space indentation; follow PEP 8.
- Naming: `snake_case` (modules/functions), `PascalCase` (classes), `UPPER_SNAKE_CASE` (constants).
- Keep configuration in `src/config.py` and `.env`; avoid hard-coded paths/devices in modules.
- Prefer structured logs via `src/utils/logging_utils.py` over `print()` in core code.

## Testing Guidelines
- This repo is script-driven (no `pytest` suite). Validate changes by running `src/run_test_simple.py` and checking printed latency metrics.
- Save new experiment artifacts under `experiments/results/` (e.g., `*.json`) and keep generation scripts in `experiments/scripts/`.
- If you change experimental methodology, update `experiments/EXPERIMENT_DESIGN.md`.

## Commit & Pull Request Guidelines
- Commit history favors short, imperative summaries (often Chinese) like “修复…/增加…/调整…/完成…/更新…”. Keep one logical change per commit.
- PRs should include: purpose, exact command(s) to reproduce, and before/after latency numbers when relevant. Attach logs or a small result JSON.
- Do not commit secrets. Use `.env` locally; add an `.env.example` when introducing new variables.
