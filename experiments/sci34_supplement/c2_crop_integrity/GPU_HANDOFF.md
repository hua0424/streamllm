# C2 v3 crop-integrity GPU 执行交接

> **实验机唯一操作入口。** v3 是独立 addendum，不覆盖或修改 C2 v1/v2。正式固定 24 cases、27 次 crop、1 session、无统计重复、无数值容差。任一失败保留目录并停止 acceptance/seal。

## 0. Exact clean commit、旧结果 guard 与协议预检

```bash
set -euo pipefail
export REPO=/dataA/streamllm
cd "$REPO"

git fetch origin
git checkout paper2
git pull --ff-only
export CODE_COMMIT="$(git rev-parse HEAD)"
test -z "$(git status --porcelain)" || { git status --short; exit 1; }

export OUT_ROOT=experiments/sci34_supplement/results/c2_crop_integrity
export GUARD=/tmp/c2_crop_integrity_guard
mkdir -p "$GUARD"

# 所有既有结果（含 C2 v1/v2、pilot 与 E3 rescue）全部纳入只读 guard。
git ls-files experiments/results experiments/sci34_supplement/results \
  | sort > "$GUARD/legacy_paths.txt"
while IFS= read -r path; do sha256sum "$path"; done < "$GUARD/legacy_paths.txt" \
  > "$GUARD/legacy_before.sha256"
sha256sum pyproject.toml uv.lock > "$GUARD/env_before.sha256"

uv run python - <<'PY'
from pathlib import Path
from experiments.sci34_supplement.common import sha256_file
from experiments.sci34_supplement.c2_crop_integrity.protocol import (
    EXPECTED_CASES_SHA256, FORMAL_CASE_COUNT, FORMAL_CROP_EVENT_COUNT,
    PRIOR_V2_RUN_ID, PROTOCOL_VERSION,
)
local = Path('experiments/sci34_supplement/c2_crop_integrity/cases.json')
source = Path('experiments/sci34_supplement/c2_equivalence/cases.json')
assert PROTOCOL_VERSION == 3
assert FORMAL_CASE_COUNT == 24 and FORMAL_CROP_EVENT_COUNT == 27
assert PRIOR_V2_RUN_ID == 'c2eq_5c56b014_20260903T040829Z'
assert sha256_file(local) == sha256_file(source) == EXPECTED_CASES_SHA256
print('C2 v3 preflight PASS')
PY
```

v2 run 只作为固定 provenance；v3 runtime 不读取其工件、不重跑 termination probe。不得删除或覆盖任何 v1/v2 rejected 记录。

## 1. 严格离线环境与本地模型

```bash
export HF_TOKEN=
unset HUGGING_FACE_HUB_TOKEN 2>/dev/null || true
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=20260903
export HF_HOME=/root/autodl-tmp/hfhome
export LD_LIBRARY_PATH="$REPO/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
export MAIN_MODEL=/root/autodl-tmp/dataA/models/Qwen2-7B-Instruct

test -d "$MAIN_MODEL" || { printf '缺少本地模型：%s\n' "$MAIN_MODEL" >&2; exit 1; }
uv sync --frozen
sha256sum -c "$GUARD/env_before.sha256"
```

Formal 期间不得联网、不得使用 HF repo ID、不得修改环境锁文件。

## 2. CLI、smoke 与模型预检

```bash
for MODULE in campaign run validate analyze seal; do
  uv run python -m "experiments.sci34_supplement.c2_crop_integrity.${MODULE}" --help
done
uv run python -m py_compile experiments/sci34_supplement/c2_crop_integrity/*.py
uv run python -m experiments.sci34_supplement.c2_crop_integrity.smoke
uv run python -m src.llm.run_kvcrop_test
uv run python -m src.dialogue.run_timeline_test
uv run python -m src.dialogue.run_speculative_test
uv run python -m experiments.sci34_supplement.smoke
git diff --check

uv run python - <<'PY'
import hashlib, os
from pathlib import Path
from transformers import AutoTokenizer
model = Path(os.environ['MAIN_MODEL']).resolve()
tok = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=True)
eot = tok.convert_tokens_to_ids('<|im_end|>')
assert tok.eos_token_id == eot
assert isinstance(tok.chat_template, str) and tok.chat_template
print('model', model)
print('tokenizer', type(tok).__name__)
print('eos/eot/pad', tok.eos_token_id, eot, tok.pad_token_id)
print('chat_template_sha256', hashlib.sha256(tok.chat_template.encode()).hexdigest())
PY
```

V3 smoke 必须报告 `status=PASS`、`protocol_version=3`、24 cases、27 crop events，且 `wrong_keep/layer_hash/duplicate_eot_ledger/missing_event` 全检出。

## 3. 独立 pilot（覆盖全部短上下文场景与第二次 crop）

Pilot 使用独立 run ID/目录，仅作兼容性预检；不进入 formal。跑前 8 cases，覆盖全部 8 scenario，并包含 `c2_08` 的第二次 crop。

```bash
export PILOT_ID="c2crop_pilot_${CODE_COMMIT:0:8}_$(date -u +%Y%m%dT%H%M%SZ)"
export PILOT_DIR="/tmp/$PILOT_ID"

CUDA_VISIBLE_DEVICES=0 uv run python -m experiments.sci34_supplement.c2_crop_integrity.campaign \
  --run-id "$PILOT_ID" --output-dir "$PILOT_DIR" --runtime transformers \
  --model "$MAIN_MODEL" --device cuda:0 --seed 20260903 --non-formal
CUDA_VISIBLE_DEVICES=0 uv run python -m experiments.sci34_supplement.c2_crop_integrity.run \
  --campaign-dir "$PILOT_DIR" --runtime transformers --model "$MAIN_MODEL" \
  --device cuda:0 --seed 20260903 --limit 8
uv run python -m experiments.sci34_supplement.c2_crop_integrity.validate \
  --campaign-dir "$PILOT_DIR" --non-formal --expected-cases 8
```

必须确认 8 records / 9 crop events 全通过，并人工检查：assistant fixture 每 token 一次 `_prefill_ids_p2`、production crop 与 pre-prefix/oracle K/V hash exact、crop 后 mask/token/seq/KV exact、recovery 每步 K/V/logits/state exact、wrong-length negative control detected。失败则保留 pilot，formal 不启动。

## 4. Formal before snapshot 与不可变 manifest

```bash
export RUN_ID="c2crop_${CODE_COMMIT:0:8}_$(date -u +%Y%m%dT%H%M%SZ)"
export RUN_DIR="$OUT_ROOT/$RUN_ID"
mkdir -p "$OUT_ROOT" "$GUARD/snapshot_before"

git rev-parse HEAD > "$GUARD/snapshot_before/git_commit.txt"
git status --short --branch > "$GUARD/snapshot_before/git_status.txt"
uname -a > "$GUARD/snapshot_before/uname.txt"
(lscpu || true) > "$GUARD/snapshot_before/lscpu.txt"
(free -h || true) > "$GUARD/snapshot_before/memory.txt"
(nvidia-smi || true) > "$GUARD/snapshot_before/nvidia_smi.txt"
(nvidia-smi pmon -c 1 || true) > "$GUARD/snapshot_before/gpu_processes.txt"
(uv pip freeze || true) > "$GUARD/snapshot_before/uv_freeze.txt"

test -z "$(git status --porcelain)" || { git status --short; exit 1; }
CUDA_VISIBLE_DEVICES=0 uv run python -m experiments.sci34_supplement.c2_crop_integrity.campaign \
  --run-id "$RUN_ID" --output-dir "$RUN_DIR" --runtime transformers \
  --model "$MAIN_MODEL" --device cuda:0 --seed 20260903
mkdir -p "$RUN_DIR/snapshots/before" "$RUN_DIR/snapshots/after"
cp -a "$GUARD/snapshot_before/." "$RUN_DIR/snapshots/before/"
sha256sum "$RUN_DIR/campaign_manifest.json" "$RUN_DIR/cases.json"
```

Manifest 冻结代码、24-case byte-copy、每 case token plan、模型/模板/dtype/backend、negative control 与 v2 provenance。Campaign 创建后代码/模型/cases 任一变化均拒绝运行。

## 5. 单 session formal 与 case 原子 resume

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m experiments.sci34_supplement.c2_crop_integrity.run \
  --campaign-dir "$RUN_DIR" --runtime transformers --model "$MAIN_MODEL" \
  --device cuda:0 --seed 20260903 \
  2>&1 | tee "$RUN_DIR/logs/formal.log"
```

中断后保持目录、commit、模型与 cases 不变，加 `--resume` 并 `tee -a` 同一日志。不得删除 failed record/attempt。正式必须恰好 24 records / 27 crop events；不做统计重复。

## 6. After snapshot 与旧结果 guard

```bash
git rev-parse HEAD > "$RUN_DIR/snapshots/after/git_commit.txt"
git status --short --branch > "$RUN_DIR/snapshots/after/git_status.txt"
uname -a > "$RUN_DIR/snapshots/after/uname.txt"
(lscpu || true) > "$RUN_DIR/snapshots/after/lscpu.txt"
(free -h || true) > "$RUN_DIR/snapshots/after/memory.txt"
(nvidia-smi || true) > "$RUN_DIR/snapshots/after/nvidia_smi.txt"
(nvidia-smi pmon -c 1 || true) > "$RUN_DIR/snapshots/after/gpu_processes.txt"

while IFS= read -r path; do sha256sum "$path"; done < "$GUARD/legacy_paths.txt" \
  > "$GUARD/legacy_after.sha256"
diff -u "$GUARD/legacy_before.sha256" "$GUARD/legacy_after.sha256"
test "$(git rev-parse HEAD)" = "$CODE_COMMIT"
test -z "$(git status --porcelain)" || { git status --short; exit 1; }
cp "$GUARD/legacy_before.sha256" "$RUN_DIR/snapshots/before/legacy.sha256"
cp "$GUARD/legacy_after.sha256" "$RUN_DIR/snapshots/after/legacy.sha256"
```

## 7. Validate → analyze → acceptance → seal → tar

```bash
uv run python -m experiments.sci34_supplement.c2_crop_integrity.validate \
  --campaign-dir "$RUN_DIR" --out "$RUN_DIR/validation.json" \
  2>&1 | tee "$RUN_DIR/logs/validation.log"
uv run python -m experiments.sci34_supplement.c2_crop_integrity.analyze \
  --campaign-dir "$RUN_DIR" --out "$RUN_DIR/analysis_v1.json" \
  2>&1 | tee "$RUN_DIR/logs/analysis_v1.log"
cp experiments/sci34_supplement/c2_crop_integrity/ACCEPTANCE_TEMPLATE.md \
  "$RUN_DIR/ACCEPTANCE.md"
```

独立复核并填写 acceptance。只有 validation `ok=true`、24/27 全 exact、negative control detected、无旧结果变化时，才加入独立一行：

```text
Status: accepted
```

随后：

```bash
uv run python -m experiments.sci34_supplement.c2_crop_integrity.seal \
  --campaign-dir "$RUN_DIR" --create
uv run python -m experiments.sci34_supplement.c2_crop_integrity.seal \
  --campaign-dir "$RUN_DIR" --verify

export TARBALL="$OUT_ROOT/${RUN_ID}.tar.gz"
test ! -e "$TARBALL" && test ! -e "${TARBALL}.sha256" || {
  printf '回传包已存在，拒绝覆盖：%s\n' "$TARBALL" >&2; exit 1;
}
tar -C "$OUT_ROOT" -czf "$TARBALL" "$RUN_ID"
sha256sum "$TARBALL" | tee "${TARBALL}.sha256"
```

Seal 会重跑 validator/analyzer、核对 stored verdict、要求非空 logs 和 before/after snapshot，并拒绝覆盖。

## 8. 回传、允许主张与红线

回传：完整 `$RUN_DIR`、tarball、tarball SHA-256、exact commit、GPU/driver 信息和执行说明。不能只回 summary/analysis。

若全部通过，只允许主张：

> 在冻结 Qwen2-7B snapshot、BF16/SDPA/backend 和 24-case/27-event v3 addendum 下，生产 `crop_to_token` 保留的 K/V 前缀与 crop 前前缀及独立切片 oracle 逐张量 bitwise exact；以相同 token-ID chunk 恢复后的 K/V、logits、mask、账本和 role/end 状态也 bitwise/exact 一致。

禁止主张 clean re-prefill 数值等价、跨模型/backend 普适性、真实 ASR/TTS/声卡正确性、时延/质量提升或生产端到端正确性。v2 仍保持 rejected；不得把 v3 结果改写为“v2 通过”。

红线：

1. 不覆盖/修改 C2 v1/v2、E1/E2/E3/A1/P1、E3 rescue 或论文正文。
2. 不改变 24 cases、27 events、tokenwise fixture、exact-only gates 或 negative control。
3. 不删除失败记录或现场改代码/协议配合结果。
4. 不用 fake/0.5B/pilot 代替 7B formal。
5. 未经明确授权不 commit/push（正式结果验收后按既有 `git add -f` 入库惯例除外）。
