# P1 prepared-state 定向重跑

本轮只重跑 S-P1 的 headless 软件控制路径；不使用声卡、不运行真实 TTS/ASR、不做完整音频闭环。旧 `${CAMPAIGN}_async` 保持只读，新 run-id 必须以 `async_prepared_v2` 结尾。

## 1. 固定环境与精确快照

从项目根目录执行，模型必须已在本地且 git tree 必须干净：

```bash
set -euo pipefail
export REPO=/dataA/streamllm
export OUT_ROOT="$REPO/experiments/sci34_supplement/results"
export MAIN_MODEL=/dataA/models/Qwen2-7B-Instruct
export CAMPAIGN=sci34_$(git rev-parse --short HEAD)_$(date +%Y%m%d)
export P1_RUN_ID="${CAMPAIGN}_async_prepared_v2"
export LOG_DIR="$OUT_ROOT/run_logs"
export SNAP_DIR="$LOG_DIR/${P1_RUN_ID}_snapshots"
mkdir -p "$LOG_DIR" "$SNAP_DIR"

export HF_TOKEN=
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=20260831

git rev-parse HEAD > "$SNAP_DIR/git_commit.txt"
git status --short --branch > "$SNAP_DIR/git_status.txt"
test -z "$(git status --porcelain)"

# Exact CPU model, sockets/cores/threads, flags and NUMA topology.
lscpu --all --extended > "$SNAP_DIR/lscpu_extended.txt"
lscpu > "$SNAP_DIR/lscpu.txt"
printf 'model_name=' > "$SNAP_DIR/cpu_model_exact.txt"
awk -F: '/^model name/{gsub(/^[ \t]+/, "", $2); print $2; exit}' /proc/cpuinfo >> "$SNAP_DIR/cpu_model_exact.txt"

# Exact installed/available RAM and DIMM inventory when permitted.
free -b > "$SNAP_DIR/memory_free_bytes.txt"
grep -E '^(MemTotal|MemFree|MemAvailable|Buffers|Cached|SwapTotal|SwapFree):' /proc/meminfo > "$SNAP_DIR/meminfo_exact.txt"
(dmidecode --type memory || true) > "$SNAP_DIR/dmidecode_memory.txt" 2>&1

# Kernel/OS snapshot.
uname -a > "$SNAP_DIR/uname.txt"
cat /proc/version > "$SNAP_DIR/proc_version.txt"
(cat /etc/os-release || true) > "$SNAP_DIR/os_release.txt"

# Exact NVIDIA driver/GPU identity, firmware-visible PCI IDs, clocks, memory and processes.
nvidia-smi > "$SNAP_DIR/nvidia_smi_before.txt"
nvidia-smi -q > "$SNAP_DIR/nvidia_smi_query_before.txt"
nvidia-smi --query-gpu=timestamp,index,uuid,pci.bus_id,name,serial,vbios_version,driver_version,pstate,temperature.gpu,power.draw,power.limit,clocks.current.graphics,clocks.current.sm,clocks.current.memory,memory.total,memory.used,memory.free,compute_mode --format=csv,noheader,nounits > "$SNAP_DIR/gpu_exact_before.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits > "$SNAP_DIR/gpu_processes_before.csv" || true
lspci -nn | grep -iE 'nvidia|vga|3d controller' > "$SNAP_DIR/gpu_pci_ids.txt" || true

uv sync
uv run python - <<'PY' > "$SNAP_DIR/python_runtime.txt"
import platform, sentencepiece, torch, transformers
print("python", platform.python_version())
print("sentencepiece", sentencepiece.__version__)
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("cuda_runtime", torch.version.cuda)
print("cudnn", torch.backends.cudnn.version())
for index in range(torch.cuda.device_count()):
    properties = torch.cuda.get_device_properties(index)
    print(index, properties.name, properties.total_memory, properties.major, properties.minor)
PY
sha256sum pyproject.toml uv.lock > "$SNAP_DIR/dependencies.sha256"
```

验收：`sentencepiece 0.2.2`；目标 3090 空闲，无未知 compute process。正式 run 不允许下载模型。

## 2. smoke 与正式重跑

协议要点：每个 trial 在播放器启动前完成 `ensure_full()` 并立即 GPU synchronize；`setup_ms` 单独记录且不计入 stop 路径。stop 后同步单独记录为 `post_stop_sync_ms`，从而验证准备工作没有泄漏到控制路径。每个 `(length,fraction)` 先跑 3 次完整 warmup，warmup 不写入正式 records。

```bash
uv run python -m py_compile experiments/sci34_supplement/*.py
uv run python -m experiments.sci34_supplement.smoke

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
  --time-scale 1 \
  2>&1 | tee "$LOG_DIR/${P1_RUN_ID}.log"

uv run python -m experiments.sci34_supplement.analyze_latency \
  --run-dir "$OUT_ROOT/async_bargein/$P1_RUN_ID" \
  --kind async \
  2>&1 | tee -a "$LOG_DIR/${P1_RUN_ID}.log"
```

若中断，用完全相同参数加 `--resume` 并 `tee -a`。Resume 只 warm 有缺失正式 repeat 的 cell；完整 cell 不 warm、不重跑。

## 3. 验收和运行后快照

```bash
P1_DIR="$OUT_ROOT/async_bargein/$P1_RUN_ID"
test "$(wc -l < "$P1_DIR/records.jsonl")" -eq 180
uv run python - "$P1_DIR/records.jsonl" <<'PY'
import json, sys
from collections import Counter
rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
assert len(rows) == 180
assert all(row["protocol"] == "async_prepared_v2" for row in rows)
assert all(row["prepared_state"] == "full_kv_synchronized_before_playback" for row in rows)
assert all(row["prepared_state_synchronized"] for row in rows)
assert all(row["trial_kind"] == "formal" and row["repeat"] >= 0 for row in rows)
assert all(row["setup_ms"] >= 0 and row["post_stop_sync_ms"] >= 0 for row in rows)
assert all(row["stop_to_sync_done_ms"] >= row["stop_ack_ms"] for row in rows)
assert all(row["played_at_request"] == row["target_samples"] for row in rows)
assert all(row["played_at_ack"] == row["target_samples"] for row in rows)
assert all(row["leaked_samples"] == 0 for row in rows)
expected = {0.25: True, 0.5: False, 0.75: True}
assert all(row["partial"] is expected[row["fraction"]] for row in rows)
counts = Counter((row["context_length_target"], row["fraction"]) for row in rows)
assert set(counts.values()) == {20} and len(counts) == 9
print("P1 prepared-state validation PASS", dict(sorted(counts.items())))
PY

nvidia-smi > "$SNAP_DIR/nvidia_smi_after.txt"
nvidia-smi -q > "$SNAP_DIR/nvidia_smi_query_after.txt"
nvidia-smi --query-gpu=timestamp,index,uuid,pci.bus_id,name,serial,vbios_version,driver_version,pstate,temperature.gpu,power.draw,power.limit,clocks.current.graphics,clocks.current.sm,clocks.current.memory,memory.total,memory.used,memory.free,compute_mode --format=csv,noheader,nounits > "$SNAP_DIR/gpu_exact_after.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits > "$SNAP_DIR/gpu_processes_after.csv" || true
sha256sum "$P1_DIR"/* "$LOG_DIR/${P1_RUN_ID}.log" "$SNAP_DIR"/* > "$SNAP_DIR/artifacts.sha256"
```

## 4. P1-only 打包

只包含新 P1 run、其日志和快照；不得包含旧 async run、E3、judge、A1 或其他历史结果。

```bash
cd "$OUT_ROOT"
tar -czf "${P1_RUN_ID}.tar.gz" \
  "async_bargein/$P1_RUN_ID" \
  "run_logs/${P1_RUN_ID}.log" \
  "run_logs/${P1_RUN_ID}_snapshots"
sha256sum "${P1_RUN_ID}.tar.gz" > "${P1_RUN_ID}.tar.gz.sha256"
tar -tzf "${P1_RUN_ID}.tar.gz"
```

回传 `${P1_RUN_ID}.tar.gz` 与 `.sha256`。结果口径仅为 headless wall-clock-paced software playback；不能外推为真实声卡停播、声学 user-heard latency 或完整生产 barge-in latency。
