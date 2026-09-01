# GPU 主机执行说明

以下命令从项目根目录执行。示例假设仓库位于 `/dataA/streamllm`，模型已在本地缓存或本地目录中。正式实验禁止临时下载模型。

## 0. 首次签出与固定版本

```bash
set -euo pipefail
export REPO=/dataA/streamllm
cd "$REPO"

git fetch origin
git checkout paper2
git pull --ff-only
# 正式跑数前记录并固定本次补实验 commit：
git rev-parse HEAD
git status --short --branch
# 此处必须无修改；结果目录已忽略，不会污染后续步骤。
```

正式 run 默认要求干净提交；runner 会在加载模型前检查并拒绝 dirty tree。prepared-state P1 v2 的新 run、日志与打包件默认忽略，因此断点续传不会被自身结果误判为 dirty；验收后若需把新结果纳入仓库，必须由设计方显式 `git add -f`，GPU 正式运行阶段不要自行入库。`--allow-dirty` 仅用于诊断，不得用于投稿正式结果。

## 1. 环境变量

```bash
export HF_TOKEN=
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=20260831
export HF_HOME=/workspace/hfhome                 # 按实验机实际路径修改
export P2_LLM_MODEL_NAME=/dataA/models/Qwen2-7B-Instruct
export LD_LIBRARY_PATH="$REPO/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"

export DIALOGUES="$REPO/experiments/datasets/processed/p2_turns.json"
export MAIN_MODEL=/dataA/models/Qwen2-7B-Instruct
export JUDGE_MODEL=/dataA/models/Mistral-7B-Instruct-v0.3
export OUT_ROOT="$REPO/experiments/sci34_supplement/results"
export CAMPAIGN=sci34_$(git rev-parse --short HEAD)_20260831
```

如果模型使用 Hugging Face 名称，必须确认 `local_files_only=True` 能成功加载后再开始正式 run。

## 2. 同步环境与硬件检查

```bash
uv sync
uv run python - <<'PY'
import nltk, torch, transformers
print('torch', torch.__version__)
print('transformers', transformers.__version__)
print('cuda', torch.cuda.is_available(), torch.version.cuda)
for resource in ('tokenizers/punkt', 'tokenizers/punkt_tab'):
    try:
        print('nltk', resource, nltk.data.find(resource))
    except LookupError as exc:
        raise SystemExit(f'Missing offline NLTK resource {resource}; install it before the formal run') from exc
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
nvidia-smi
sha256sum "$DIALOGUES" pyproject.toml uv.lock
```

验收：两张 RTX 3090 可见；主卡空闲；`p2_turns.json` 存在。

## 3. 无模型 smoke

```bash
uv run python -m py_compile experiments/sci34_supplement/*.py
uv run python -m experiments.sci34_supplement.smoke
uv run python -m src.dialogue.run_timeline_test
```

三项都通过后再加载 7B。

## 4. E3 三条数据 integration

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m experiments.sci34_supplement.e3_fixed_trajectory \
  --dialogues "$DIALOGUES" \
  --run-id "${CAMPAIGN}_e3_smoke3" \
  --results-root "$OUT_ROOT/e3" \
  --runtime transformers \
  --model "$MAIN_MODEL" \
  --device cuda:0 \
  --formal \
  --limit 3 \
  --max-first-tokens 40 \
  --max-probe-tokens 40 \
  --seed 20260831
```

若这里只是调试尚未提交的临时代码，可显式加 `--allow-dirty`；该结果不得作为投稿正式数据。

检查：

```bash
wc -l "$OUT_ROOT/e3/${CAMPAIGN}_e3_smoke3/trajectories.jsonl"
wc -l "$OUT_ROOT/e3/${CAMPAIGN}_e3_smoke3/records.jsonl"
# 预期 3 trajectories / 24 records
```

## 5. E3 正式 100 条

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m experiments.sci34_supplement.e3_fixed_trajectory \
  --dialogues "$DIALOGUES" \
  --run-id "${CAMPAIGN}_e3" \
  --results-root "$OUT_ROOT/e3" \
  --runtime transformers \
  --model "$MAIN_MODEL" \
  --device cuda:0 \
  --formal \
  --limit 100 \
  --max-first-tokens 40 \
  --max-probe-tokens 40 \
  --seed 20260831
```

若中断，用**完全相同参数**加 `--resume`。参数、数据哈希或模型路径变化会拒绝恢复。

完成验收：

```bash
wc -l "$OUT_ROOT/e3/${CAMPAIGN}_e3/trajectories.jsonl"  # 100
wc -l "$OUT_ROOT/e3/${CAMPAIGN}_e3/records.jsonl"       # 800
```

## 6. E3 Mistral 裁判

主模型退出后释放显存，再运行：

```bash
CUDA_VISIBLE_DEVICES=1 uv run python -m experiments.sci34_supplement.e3_judge \
  --e3-run-dir "$OUT_ROOT/e3/${CAMPAIGN}_e3" \
  --run-id "${CAMPAIGN}_judge" \
  --results-root "$OUT_ROOT/judge" \
  --judge-model "$JUDGE_MODEL" \
  --device cuda:0
```

中断后加 `--resume`。完成后检查 `parse_failures` 应为 0；若非 0，保留原始输出并先修复解析，不得把失败默认记为否。

## 7. E3 分析

```bash
uv run python -m experiments.sci34_supplement.analyze_e3 \
  --e3-run-dir "$OUT_ROOT/e3/${CAMPAIGN}_e3" \
  --judge-records "$OUT_ROOT/judge/${CAMPAIGN}_judge/judge_records.jsonl" \
  --bootstrap-repeats 10000 \
  --seed 20260831
```

正式汇总：`$OUT_ROOT/e3/${CAMPAIGN}_e3/summary.json`。

## 8. A1 预跑与正式运行

预跑 20 repeats：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m experiments.sci34_supplement.a1_joint_latency \
  --run-id "${CAMPAIGN}_a1_pilot" \
  --results-root "$OUT_ROOT/a1" \
  --runtime transformers \
  --model "$MAIN_MODEL" \
  --device cuda:0 \
  --lengths 256 512 1024 2048 4096 8192 \
  --crop-tokens 32 \
  --warmup 3 \
  --repeats 20
```

如果时间允许，论文正式数字推荐 50 repeats：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m experiments.sci34_supplement.a1_joint_latency \
  --run-id "${CAMPAIGN}_a1" \
  --results-root "$OUT_ROOT/a1" \
  --runtime transformers \
  --model "$MAIN_MODEL" \
  --device cuda:0 \
  --lengths 256 512 1024 2048 4096 8192 \
  --crop-tokens 32 \
  --warmup 5 \
  --repeats 50

uv run python -m experiments.sci34_supplement.analyze_latency \
  --run-dir "$OUT_ROOT/a1/${CAMPAIGN}_a1" \
  --kind a1
```

## 9. Headless 异步播放 P1 prepared-state 定向重跑

旧 `${CAMPAIGN}_async` 的准备态异步工作污染结果只保留作审计，不修改、不追加。本次只重跑 P1，详细协议、环境快照与专用打包命令见 [P1_PREPARED_RERUN.md](P1_PREPARED_RERUN.md)。该实验不需要声卡，主模型只用于真实 KV crop/role 软件路径。

```bash
export P1_RUN_ID="${CAMPAIGN}_async_prepared_v2"
CUDA_VISIBLE_DEVICES=0 uv run python -m experiments.sci34_supplement.async_bargein \
  --run-id "$P1_RUN_ID" \
  --results-root "$OUT_ROOT/async_bargein" \
  --runtime transformers \
  --model "$MAIN_MODEL" \
  --device cuda:0 \
  --lengths 512 2048 8192 \
  --fractions 0.25 0.5 0.75 \
  --warmups 3 \
  --repeats 20 \
  --sample-rate 24000 \
  --duration-s 0.8 \
  --fragments 6 \
  --block-ms 20 \
  --time-scale 1

uv run python -m experiments.sci34_supplement.analyze_latency \
  --run-dir "$OUT_ROOT/async_bargein/$P1_RUN_ID" \
  --kind async
```

协议在每次播放前执行 `ensure_full()` 后立即同步 GPU，记录 `setup_ms`，并对每个 `(length, fraction)` 单元先做 3 次不落盘 warmup。断点恢复用完全相同命令加 `--resume`；只对尚缺正式记录的单元重新 warmup。验收：180 条正式 records；warmup 不在 `records.jsonl`；0.25/0.75 全为 `partial=true`，0.5 全为 `partial=false`；所有记录 `protocol=async_prepared_v2` 且 `prepared_state_synchronized=true`。正式结果必须称“headless wall-clock-paced software playback microbenchmark”。

## 10. 一键运行

确认前述变量后：

```bash
bash experiments/sci34_supplement/run_all_gpu.sh
# 当前 P1 定向重跑：
P1_ONLY=1 bash experiments/sci34_supplement/run_all_gpu.sh
```

脚本保留完整补实验编排能力，但 P1 默认使用新 prepared-state v2 协议和独立 `${CAMPAIGN}_async_prepared_v2` 目录。当前批准的是 P1-only 定向重跑，优先直接执行第 9 节或设置 `P1_ONLY=1` 运行脚本。若要 A1 50 repeats，设置：

```bash
export A1_REPEATS=50
export A1_WARMUP=5
bash experiments/sci34_supplement/run_all_gpu.sh
```

## 11. 打包和回传

本次为 **P1-only 定向重跑**。不得重新打包 E3/judge/A1，也不得把旧 `${CAMPAIGN}_async` 混入新包。按 [P1_PREPARED_RERUN.md](P1_PREPARED_RERUN.md) 仅打包：

- `async_bargein/${P1_RUN_ID}/`（新 run 的 manifest、raw records、summary、analysis）；
- `run_logs/${P1_RUN_ID}.log`；
- `run_logs/${P1_RUN_ID}_snapshots/`（CPU/RAM/kernel/driver/GPU 精确快照）。

回传 tarball 和 sha256，不要只复制 summary。

## 12. 失败处理

- **OOM**：先结束其他 GPU 进程；不要改正式上下文长度。若仍 OOM，记录失败，另起新 run ID 调整 dtype/模型设置，不能 resume 到旧目录。
- **模型本地缺失**：停止，不在正式 run 中临时联网下载。
- **config hash mismatch**：使用新 run ID，禁止强行追加。
- **judge parse failure**：保留 raw output，修复 prompt/解析后使用新 judge run ID。
- **GPU 干扰**：A1 前后保存 `nvidia-smi`；若中途出现其他进程，整次 A1 使用新 run ID 重跑。
