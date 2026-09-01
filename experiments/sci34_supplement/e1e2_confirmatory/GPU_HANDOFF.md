# E1/E2 确认性 campaign GPU 交接

> 面向 GPU 操作者。本文只使用当前已实现 CLI，所有命令均以本地 `--help` 输出为准。正式结果通过 analysis、validation 与 acceptance 前，不修改论文稿或旧结果。

## 0. 证据边界与计时口径

本 campaign 使用本地 MultiWOZ 2.1 派生的同步预切分文本段，不是实际音频；不运行真实 ASR、在线 TTS、播放器或声卡。

必须区分三个事件：

1. `last_segment_arrival`：harness 收到最后一个受控文本段的墙钟时刻；
2. `first_token_ready`：首个最终可用 token 实际准备完成的墙钟时刻；
3. `endpoint_accept`：同步 oracle 决定接受候选的时刻。

**实际受控墙钟主指标**是：

```text
arrival_to_first_token_ready_ns = first_token_ready_ns - last_segment_arrival_ns
```

`TTFT_eff` 不是上述实际墙钟延迟。它把“候选已准备好，oracle 在接受时允许立即交付”记为 0，因此是**候选准备完成后、同步 oracle 接受策略下的时延的乐观下界（推测收益的上界）**。`endpoint_accept` 也不是最后一段到达瞬间；不得把 `endpoint_accept_ns` 改称或解释成 `last_segment_arrival_ns`。

当前 raw records 已正式保存 `last_segment_arrival_ns`、`first_token_ready_ns`、`arrival_to_first_token_ready_ns`、`endpoint_accept_ns` 和 `oracle_preaccept_processing_ns`。Runner 与 validator 会核对 `arrival_to_first_token_ready_ns = first_token_ready_ns - last_segment_arrival_ns`，analyzer 将其作为主指标；`TTFT_eff` 仅作为同步 oracle 时延的乐观下界（推测收益的上界）单列。

浪费率正式主定义固定为：

```text
sum(wasted_tokens) / sum(wasted_tokens + final_tokens)
```

`speculative_tokens` 只作诊断分母，不是正式 pooled waste 主定义。

主模型固定 Qwen2-7B-Instruct、greedy、batch size 1、`max_new_tokens=32`、`spec_chunk=12`；`0.92` 是旧探索结果预冻结的 confirmatory candidate。正式条件为 System A、八个阈值和 never-speculate，共 10 个条件。

## 1. 最终 CLI 与帮助核对

从仓库根目录执行。以下六个模块提供 argparse 帮助；`smoke` 没有参数解析器，直接运行测试，因此不要把 `--help` 传给它。

```bash
set -euo pipefail

for MODULE in \
  holdout_builder \
  trigger_cache \
  campaign \
  run_session \
  analyze \
  validate
do
  uv run python -m "experiments.sci34_supplement.e1e2_confirmatory.${MODULE}" --help
done

uv run python -m experiments.sci34_supplement.e1e2_confirmatory.smoke
```

当前实际入口只有：

- `holdout_builder`
- `trigger_cache`
- `campaign`
- `run_session`
- `analyze`
- `validate`
- `smoke`

命令必须逐项复制本文已核对的模块和参数，不附加旧草案中的额外选项。

`campaign` 生成 formal session 强制使用的不可变 `campaign_manifest.json`。五个 formal session 必须传同一个 `--campaign-manifest`；runner、records 与 validator 会交叉核对 manifest SHA-256、campaign identity、输入、TEN cache、主模型、runtime、device、dtype 和 attention backend。

## 2. 固定 clean commit 与旧结果 hash guard

```bash
export REPO=/dataA/streamllm
cd "$REPO"

git fetch origin
git checkout paper2
git pull --ff-only
export CODE_COMMIT="$(git rev-parse HEAD)"
test -z "$(git status --porcelain)" || { git status --short; exit 1; }

mkdir -p /tmp/e1e2_confirmatory_guard
sha256sum \
  experiments/results/exp1_latency.json \
  experiments/results/exp2_tradeoff.json \
  experiments/results/paper2_reanalysis.json \
  > /tmp/e1e2_confirmatory_guard/legacy_before.sha256
sha256sum pyproject.toml uv.lock \
  > /tmp/e1e2_confirmatory_guard/env_lock.sha256
```

`--allow-dirty` 仅用于开发诊断，产生的结果不得进入 formal campaign。

## 3. 冻结离线环境与本地资产

```bash
export HF_TOKEN=
unset HUGGING_FACE_HUB_TOKEN 2>/dev/null || true
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=20260901
export HF_HOME=/workspace/hfhome
export LD_LIBRARY_PATH="$REPO/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"

uv sync --frozen
sha256sum -c /tmp/e1e2_confirmatory_guard/env_lock.sha256

export MAIN_MODEL=/dataA/models/Qwen2-7B-Instruct
export TEN_MODEL=/dataA/models/TEN_Turn_Detection
export MULTIWOZ_RAW=/dataA/datasets/MultiWOZ_2.1/data.json
export OLD_E1=experiments/results/exp1_latency.json
export OLD_E2=experiments/results/exp2_tradeoff.json
export ACCEPTED_E3_MANIFEST=experiments/sci34_supplement/results/e3/sci34_f11ccba_20260901_e3/manifest.json
export OUT_ROOT=experiments/sci34_supplement/results/e1e2_confirmatory

for P in \
  "$MAIN_MODEL" \
  "$TEN_MODEL" \
  "$MULTIWOZ_RAW" \
  "$OLD_E1" \
  "$OLD_E2" \
  "$ACCEPTED_E3_MANIFEST"
do
  test -e "$P" || { printf '缺少本地资产：%s\n' "$P" >&2; exit 1; }
done
```

固定 E3 排除源必须是上述 accepted run 的真实 manifest，不得替换成目录占位符。

记录环境，不打印 token 值：

```bash
uv run python - <<'PY'
import os
from pathlib import Path
import torch, transformers
assert os.environ.get("HF_TOKEN", "") == ""
assert os.environ.get("HF_HUB_OFFLINE") == "1"
assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
for key in ("MAIN_MODEL", "TEN_MODEL", "MULTIWOZ_RAW", "ACCEPTED_E3_MANIFEST"):
    path = Path(os.environ[key]).resolve()
    assert path.exists(), (key, path)
    print(key, path)
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("cuda_available", torch.cuda.is_available())
print("gpu_count", torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_name(index))
PY
```

缺少本地模型、数据、accepted E3 manifest、CUDA 或离线加载能力时立即停止；不得联网补资产。

## 4. 建立 campaign 目录与 smoke

```bash
export CAMPAIGN="e1e2c_$(git rev-parse --short HEAD)_$(date -u +%Y%m%dT%H%M%SZ)"
export CAMPAIGN_DIR="$OUT_ROOT/$CAMPAIGN"
mkdir -p \
  "$CAMPAIGN_DIR/inputs" \
  "$CAMPAIGN_DIR/trigger_cache" \
  "$CAMPAIGN_DIR/run_logs" \
  "$CAMPAIGN_DIR/snapshots/before" \
  "$CAMPAIGN_DIR/snapshots/after"

{
  uv run python -m py_compile experiments/sci34_supplement/*.py experiments/sci34_supplement/e1e2_confirmatory/*.py
  uv run python -m experiments.sci34_supplement.smoke
  uv run python -m src.dialogue.run_timeline_test
  uv run python -m experiments.sci34_supplement.e1e2_confirmatory.smoke
  git diff --check
} 2>&1 | tee "$CAMPAIGN_DIR/run_logs/smoke.log"
```

全部 PASS 才继续。

## 5. Before snapshot

```bash
SNAP_BEFORE="$CAMPAIGN_DIR/snapshots/before"
git rev-parse HEAD > "$SNAP_BEFORE/git_commit.txt"
git status --short --branch > "$SNAP_BEFORE/git_status.txt"
uname -a > "$SNAP_BEFORE/uname.txt"
(lscpu || true) > "$SNAP_BEFORE/lscpu.txt"
(free -h || true) > "$SNAP_BEFORE/memory.txt"
(nvidia-smi || true) > "$SNAP_BEFORE/nvidia_smi.txt"
(nvidia-smi --query-gpu=index,uuid,name,pci.bus_id,driver_version,memory.total,memory.free,temperature.gpu,power.draw --format=csv,noheader || true) \
  > "$SNAP_BEFORE/gpu_fields.csv"
(nvidia-smi pmon -c 1 || true) > "$SNAP_BEFORE/gpu_processes.txt"
(uv pip freeze || true) > "$SNAP_BEFORE/uv_freeze.txt"
sha256sum pyproject.toml uv.lock "$MAIN_MODEL/config.json" "$TEN_MODEL/config.json" \
  > "$SNAP_BEFORE/config_hashes.sha256"
```

## 6. 构建并冻结 100 条 disjoint holdout

`holdout_builder` 的 formal 模式是默认值；不要传不存在的 `--formal`。显式传三次 `--exclude`，避免依赖隐式默认。

```bash
export HOLDOUT_SEED=20260901

uv run python -m experiments.sci34_supplement.e1e2_confirmatory.holdout_builder \
  --input "$MULTIWOZ_RAW" \
  --output "$CAMPAIGN_DIR/inputs/holdout.json" \
  --provenance "$CAMPAIGN_DIR/inputs/holdout.provenance.json" \
  --exclude "$OLD_E1" \
  --exclude "$OLD_E2" \
  --exclude "$ACCEPTED_E3_MANIFEST" \
  --count 100 \
  --seed "$HOLDOUT_SEED" \
  2>&1 | tee "$CAMPAIGN_DIR/run_logs/holdout.log"

sha256sum \
  "$MULTIWOZ_RAW" \
  "$OLD_E1" \
  "$OLD_E2" \
  "$ACCEPTED_E3_MANIFEST" \
  "$CAMPAIGN_DIR/inputs/holdout.json" \
  "$CAMPAIGN_DIR/inputs/holdout.provenance.json"
```

`holdout_builder` 自身校验数量、fixture、ID 唯一、至少两个非空 segment、无损拼接及排除交集。holdout 的校验由 builder 完成；campaign validator 只在 session grid 形成后运行。

## 7. 构建一次 TEN cache

`trigger_cache` 输出单个 JSON 文件，不是目录 manifest + JSONL。

```bash
CUDA_VISIBLE_DEVICES=1 uv run python -m experiments.sci34_supplement.e1e2_confirmatory.trigger_cache \
  --input "$CAMPAIGN_DIR/inputs/holdout.json" \
  --output "$CAMPAIGN_DIR/trigger_cache/trigger_cache.json" \
  --model "$TEN_MODEL" \
  --device cuda:0 \
  2>&1 | tee "$CAMPAIGN_DIR/run_logs/trigger_cache.log"

sha256sum "$CAMPAIGN_DIR/trigger_cache/trigger_cache.json"
```

cache 保存每个累积 prefix 的未舍入 confidence、文本 hash、template、类别 token/聚合规则、TEN identity、输入 hash 和 cache identity。五个 formal session 必须使用同一个文件。

cache 的输入 hash 与 entry identity 会在 `ReplayTrigger` 加载时校验。

## 8. 生成并冻结 formal campaign manifest

必须在 TEN cache 后、pilot/formal session 前生成。`campaign` 默认 formal，会检查 clean tree、严格离线、100 条输入、TEN cache strong identity 和主模型 strong identity；输出已存在时拒绝覆盖。

```bash
uv run python -m experiments.sci34_supplement.e1e2_confirmatory.campaign \
  --campaign-id "$CAMPAIGN" \
  --input "$CAMPAIGN_DIR/inputs/holdout.json" \
  --trigger-cache "$CAMPAIGN_DIR/trigger_cache/trigger_cache.json" \
  --main-model "$MAIN_MODEL" \
  --device cuda:0 \
  --output "$CAMPAIGN_DIR/campaign_manifest.json" \
  2>&1 | tee "$CAMPAIGN_DIR/run_logs/campaign_manifest.log"

sha256sum "$CAMPAIGN_DIR/campaign_manifest.json"
```

该文件冻结 protocol、input、TEN cache、TEN identity、主模型 strong identity、runtime/device 和 campaign identity。后续五个 formal session 必须逐字节复用此文件。

## 9. Pilot：代码实际支持的非 formal 方式

非 formal pilot 使用独立 campaign/session ID，省略 `--formal`，通过 `--runtime transformers` 走真实模型路径，并用代码实际支持的 `--limit 3` 选前三条。`run_session` 明确禁止 non-formal session 传 campaign manifest，因此 pilot 不传 formal manifest；它与 `$CAMPAIGN_DIR` 的正式网格隔离。

```bash
export PILOT_CAMPAIGN="${CAMPAIGN}-pilot"

CUDA_VISIBLE_DEVICES=0 uv run python -m experiments.sci34_supplement.e1e2_confirmatory.run_session \
  --campaign-id "$PILOT_CAMPAIGN" \
  --session-id pilot01 \
  --session-index 0 \
  --input "$CAMPAIGN_DIR/inputs/holdout.json" \
  --trigger-cache "$CAMPAIGN_DIR/trigger_cache/trigger_cache.json" \
  --results-root "$OUT_ROOT" \
  --runtime transformers \
  --model "$MAIN_MODEL" \
  --device cuda:0 \
  --seed 20260901 \
  --order-seed 20260901 \
  --max-new-tokens 32 \
  --spec-chunk 12 \
  --warmup-repeats 3 \
  --limit 3 \
  2>&1 | tee "$CAMPAIGN_DIR/run_logs/pilot01.log"
```

Pilot 输出位于 `$OUT_ROOT/$PILOT_CAMPAIGN/`，天然不在 `$CAMPAIGN_DIR` 的 formal validation/analysis 范围内。不得根据 pilot 改阈值、解码、prompt、模型或统计定义。

## 10. 五个独立 formal 进程

session index 固定为 `0..4`。每次 shell 调用启动一个新 Python 进程，上一进程完全退出后再启动下一进程。

```bash
SESSION_IDS=(s01 s02 s03 s04 s05)

for INDEX in 0 1 2 3 4; do
  SID="${SESSION_IDS[$INDEX]}"
  test -z "$(git status --porcelain)" || { git status --short; exit 1; }
  (nvidia-smi pmon -c 1 || true) > "$CAMPAIGN_DIR/snapshots/${SID}_gpu_processes_before.txt"

  CUDA_VISIBLE_DEVICES=0 uv run python -m experiments.sci34_supplement.e1e2_confirmatory.run_session \
    --campaign-id "$CAMPAIGN" \
    --session-id "$SID" \
    --session-index "$INDEX" \
    --input "$CAMPAIGN_DIR/inputs/holdout.json" \
    --trigger-cache "$CAMPAIGN_DIR/trigger_cache/trigger_cache.json" \
    --campaign-manifest "$CAMPAIGN_DIR/campaign_manifest.json" \
    --results-root "$OUT_ROOT" \
    --runtime transformers \
    --model "$MAIN_MODEL" \
    --device cuda:0 \
    --seed 20260901 \
    --order-seed 20260901 \
    --max-new-tokens 32 \
    --spec-chunk 12 \
    --warmup-repeats 3 \
    --formal \
    2>&1 | tee "$CAMPAIGN_DIR/run_logs/${SID}.log"

  (nvidia-smi pmon -c 1 || true) > "$CAMPAIGN_DIR/snapshots/${SID}_gpu_processes_after.txt"
done
```

每 session 预期 `100 × 10 = 1000` records，总计 5000。阈值网格、greedy、batch size 和 system prompt 来自冻结 `ProtocolConfig`，没有对应的 CLI 参数。

中断规则：

- `--resume` 只允许同一 Python 进程、同一 `process_start_id`；进程重启后的 resume 会被拒绝；
- OOM、进程退出或 GPU 干扰后，以新 session ID 从头重跑，失败目录保留审计；
- 不拼接不同进程的墙钟记录；
- duplicate key、截断 JSONL 或 identity/hash mismatch 立即停止。

## 11. 严格 validation 与 immutable analysis

先 validation，再 analysis。两个命令都默认 formal；`--out` 是实际输出参数名。

```bash
uv run python -m experiments.sci34_supplement.e1e2_confirmatory.validate \
  --campaign-dir "$CAMPAIGN_DIR" \
  --out "$CAMPAIGN_DIR/validation.json" \
  --expected-sessions 5 \
  --expected-dialogues 100 \
  2>&1 | tee "$CAMPAIGN_DIR/run_logs/validation.log"

uv run python -m experiments.sci34_supplement.e1e2_confirmatory.analyze \
  --campaign-dir "$CAMPAIGN_DIR" \
  --out "$CAMPAIGN_DIR/analysis_v1.json" \
  --bootstrap-repeats 10000 \
  --bootstrap-seed 20260901 \
  --expected-sessions 5 \
  --expected-dialogues 100 \
  2>&1 | tee "$CAMPAIGN_DIR/run_logs/analysis_v1.log"
```

输出文件已存在时两个命令都会拒绝覆盖；口径变化必须使用 versioned 新路径。

必须核对：

- 5 个独立 process identity；
- 五个 session manifest 与全部 records 共享同一个非空 campaign manifest SHA-256；
- 5000 records，十条件网格完整且顺序平衡；
- 所有 session 共享 input/cache/model/protocol identity；
- E1 直接复用 E2 的 B@0.92 raw records；
- `TTFT_eff=0` 仅表示接受时有存活 ready candidate；
- 正式 pooled waste 为 `sum(wasted) / sum(wasted + final)`；
- bootstrap 先 session、再 session 内 dialogue，并共同保留全部条件；
- 无 outlier trimming 或未声明排除；
- `TTFT_eff` 没有被写成实际 `last_segment_arrival→first_token_ready` 墙钟主指标。

复制模板并填写：

```bash
cp experiments/sci34_supplement/e1e2_confirmatory/ACCEPTANCE_TEMPLATE.md \
  "$CAMPAIGN_DIR/ACCEPTANCE.md"
```

验收必须从 raw records 直接复算 `arrival_to_first_token_ready_ns = first_token_ready_ns - last_segment_arrival_ns`，并确认 analyzer 以该字段作为 C-E1/C-E2 主比较；不得用 `endpoint_accept` 或 `TTFT_eff` 替代。

## 12. After snapshot、旧结果保护与 checksums

```bash
SNAP_AFTER="$CAMPAIGN_DIR/snapshots/after"
git rev-parse HEAD > "$SNAP_AFTER/git_commit.txt"
git status --short --branch > "$SNAP_AFTER/git_status.txt"
uname -a > "$SNAP_AFTER/uname.txt"
(lscpu || true) > "$SNAP_AFTER/lscpu.txt"
(free -h || true) > "$SNAP_AFTER/memory.txt"
(nvidia-smi || true) > "$SNAP_AFTER/nvidia_smi.txt"
(nvidia-smi --query-gpu=index,uuid,name,pci.bus_id,driver_version,memory.total,memory.free,temperature.gpu,power.draw --format=csv,noheader || true) \
  > "$SNAP_AFTER/gpu_fields.csv"
(nvidia-smi pmon -c 1 || true) > "$SNAP_AFTER/gpu_processes.txt"

sha256sum \
  experiments/results/exp1_latency.json \
  experiments/results/exp2_tradeoff.json \
  experiments/results/paper2_reanalysis.json \
  > /tmp/e1e2_confirmatory_guard/legacy_after.sha256

diff -u \
  /tmp/e1e2_confirmatory_guard/legacy_before.sha256 \
  /tmp/e1e2_confirmatory_guard/legacy_after.sha256

test "$(git rev-parse HEAD)" = "$CODE_COMMIT"

(
  cd "$CAMPAIGN_DIR"
  find . -type f \
    ! -name 'checksums.sha256' \
    ! -name '*.tar.gz' \
    ! -name '*.tar.gz.sha256' \
    -print0 | sort -z | xargs -0 sha256sum > checksums.sha256
  sha256sum -c checksums.sha256
)

export TARBALL="$OUT_ROOT/${CAMPAIGN}.tar.gz"
tar -C "$OUT_ROOT" -czf "$TARBALL" "$CAMPAIGN"
sha256sum "$TARBALL" | tee "${TARBALL}.sha256"
```

## 13. 回传与红线

至少回传：campaign/code identity、`campaign_manifest.json` 路径/内容 hash/SHA-256、环境与离线证明、模型/TEN/MultiWOZ/holdout/cache hash、accepted E3 manifest hash、pilot 处置、五个 session process identity 与共同 campaign manifest SHA、5000-record grid、主 `arrival_to_first_token_ready_ns` 指标、`TTFT_eff` 时延的乐观下界（推测收益的上界）、wasted/(wasted+final)、analysis/validation/checksum/tarball hash、旧结果前后 hash、异常审计和限定性结论。

红线：

1. 不覆盖旧结果或 `analysis_v1.json`；
2. 不修改论文稿、chapter、abstract、thesis draft 或 IEEE；
3. 不用 pilot、失败 session 或部分 records 更新论文；
4. 不在新 holdout 上重选阈值；
5. 不把 replay TEN 称为在线零成本；
6. 不把 `endpoint_accept` 称为最后一段到达；
7. 不把 `TTFT_eff` 称为实际受控墙钟主指标；
8. 不把受控文本段称为真实音频或生产端到端；
9. 不把浪费率主分母改为 `speculative_tokens`；
10. 未经明确授权不 commit 或 push。
