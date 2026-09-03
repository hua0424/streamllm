# C2 equivalence GPU 执行交接

> 实验机唯一操作入口。以下命令按当前已实现 CLI 写成。正式证据固定 24 cases、1 session、无统计重复。任何失败保留目录并停止 acceptance/seal。

## 0. Exact clean commit、目录与旧结果 guard

```bash
set -euo pipefail
export REPO=/dataA/streamllm
cd "$REPO"

git fetch origin
git checkout paper2
git pull --ff-only
export CODE_COMMIT="$(git rev-parse HEAD)"
test -z "$(git status --porcelain)" || { git status --short; exit 1; }

export OUT_ROOT=experiments/sci34_supplement/results/c2_equivalence
export GUARD=/tmp/c2_equivalence_guard
mkdir -p "$GUARD"

# 旧结果统一内容 guard；不依赖手工维护文件列表。
git ls-files experiments/results experiments/sci34_supplement/results \
  | grep -v '^experiments/sci34_supplement/results/c2_equivalence/' \
  | sort > "$GUARD/legacy_paths.txt"
while IFS= read -r path; do sha256sum "$path"; done < "$GUARD/legacy_paths.txt" \
  > "$GUARD/legacy_before.sha256"
sha256sum pyproject.toml uv.lock > "$GUARD/env_before.sha256"
```

正式前必须先形成包含 C2 与配套 core 修复的 exact clean commit。本交接不授权 commit/push。

## 1. 严格离线与显式本地模型

```bash
export HF_TOKEN=
unset HUGGING_FACE_HUB_TOKEN 2>/dev/null || true
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=20260902
export HF_HOME=/workspace/hfhome
export LD_LIBRARY_PATH="$REPO/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"

export MAIN_MODEL=/root/autodl-tmp/dataA/models/Qwen2-7B-Instruct
test -d "$MAIN_MODEL" || { printf '缺少本地模型：%s\n' "$MAIN_MODEL" >&2; exit 1; }

uv sync --frozen
sha256sum -c "$GUARD/env_before.sha256"
```

不得在 formal 期间联网下载或以 HF repo ID 替代显式本地目录。

## 2. CLI help 与本机 smoke

当前带 argparse 的真实入口只有 `campaign/run/validate/analyze/seal`；`smoke` 直接执行。

```bash
for MODULE in campaign run validate analyze seal; do
  uv run python -m "experiments.sci34_supplement.c2_equivalence.${MODULE}" --help
done

uv run python -m py_compile experiments/sci34_supplement/c2_equivalence/*.py
uv run python -m experiments.sci34_supplement.c2_equivalence.smoke
uv run python -m src.llm.run_kvcrop_test
uv run python -m src.dialogue.run_timeline_test
uv run python -m src.dialogue.run_speculative_test
uv run python -m experiments.sci34_supplement.smoke
git diff --check
```

`c2_equivalence.smoke` 输出必须明确 `models_loaded=false`、`network_used=false`、24 cases PASS。任一测试失败停止。

## 3. 模型/tokenizer 预检

```bash
uv run python - <<'PY'
import hashlib, os
from pathlib import Path
from transformers import AutoTokenizer
model = Path(os.environ["MAIN_MODEL"]).resolve()
assert model.is_dir()
tok = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=True)
eot = tok.convert_tokens_to_ids("<|im_end|>")
assert isinstance(eot, int) and eot >= 0
assert tok.eos_token_id == eot, (tok.eos_token_id, eot)
assert isinstance(tok.chat_template, str) and tok.chat_template
print("model", model)
print("tokenizer", type(tok).__name__)
print("eos", tok.eos_token_id, "eot", eot, "pad", tok.pad_token_id)
print("chat_template_sha256", hashlib.sha256(tok.chat_template.encode()).hexdigest())
PY
```

## 4. E3 exact `p2_turns.json` 抢救（与 C2 分开保存）

Accepted E3 manifest 记录的输入路径是：

```text
/root/autodl-tmp/dataA/streamllm/experiments/datasets/processed/p2_turns.json
```

记录的 SHA-256 是：

```text
a2116b83b38509e45d641ade17ae1791282729836bdca251aa8aba8aa9248a0c
```

在原实验机执行；不存在时只记录 missing，不重建后冒充 exact 文件。

```bash
export E3_RESCUE=/tmp/e3_exact_rescue
mkdir -p "$E3_RESCUE"
export E3_TURNS=/root/autodl-tmp/dataA/streamllm/experiments/datasets/processed/p2_turns.json
export E3_MANIFEST=experiments/sci34_supplement/results/e3/sci34_f11ccba_20260901_e3/manifest.json

sha256sum "$E3_MANIFEST" > "$E3_RESCUE/e3_manifest.sha256"
if test -f "$E3_TURNS"; then
  sha256sum "$E3_TURNS" | tee "$E3_RESCUE/p2_turns.sha256"
  grep -q '^a2116b83b38509e45d641ade17ae1791282729836bdca251aa8aba8aa9248a0c ' \
    "$E3_RESCUE/p2_turns.sha256"
  cp -a "$E3_TURNS" "$E3_RESCUE/p2_turns.json"
else
  printf 'MISSING exact path: %s\n' "$E3_TURNS" > "$E3_RESCUE/p2_turns.MISSING.txt"
fi

# 抢救 raw MultiWOZ、builder/provenance；路径以实验机实际情况为准，全部 hash。
find /root/autodl-tmp/dataA /dataA -type f \
  \( -path '*/MultiWOZ_2.1/data.json' -o -name 'prepare_multiwoz_data.py' \
     -o -iname '*p2_turns*provenance*.json' \) -print 2>/dev/null \
  | sort > "$E3_RESCUE/provenance_paths.txt"
while IFS= read -r path; do sha256sum "$path"; done < "$E3_RESCUE/provenance_paths.txt" \
  > "$E3_RESCUE/provenance.sha256"

# 保存 E3 manifest 中的模型 snapshot 路径/identity；若目录仍在则重新 strong-hash。
cp "$E3_MANIFEST" "$E3_RESCUE/e3_manifest.json"
uv run python - <<'PY' > "$E3_RESCUE/e3_model_identity.json"
import json, os
from pathlib import Path
from experiments.sci34_supplement.e1e2_confirmatory.strong_identity import strong_model_identity
manifest=json.load(open(os.environ["E3_MANIFEST"], encoding="utf-8"))
path=Path(manifest["config"]["model"])
print(json.dumps({
  "manifest_model": manifest["config"].get("model_identity"),
  "current_strong_identity": strong_model_identity(path) if path.is_dir() else None,
  "status": "rehashed" if path.is_dir() else "snapshot path missing",
}, indent=2))
PY
```

把 `$E3_RESCUE` 与 C2 tarball 一并回传，但不要把它混入 C2 correctness records。

## 5. 独立 integration pilot

Pilot 使用不同 run ID/目录，`campaign --non-formal`，实际 Transformers 7B，仅跑前三个 cases。它不能进入 formal validation/analysis，也不能据此改变 frozen protocol。

```bash
export PILOT_ID="c2pilot_${CODE_COMMIT:0:8}_$(date -u +%Y%m%dT%H%M%SZ)"
export PILOT_DIR="$OUT_ROOT/$PILOT_ID"
mkdir -p "$OUT_ROOT"

CUDA_VISIBLE_DEVICES=0 uv run python -m experiments.sci34_supplement.c2_equivalence.campaign \
  --run-id "$PILOT_ID" \
  --output-dir "$PILOT_DIR" \
  --runtime transformers \
  --model "$MAIN_MODEL" \
  --device cuda:0 \
  --seed 20260902 \
  --non-formal

CUDA_VISIBLE_DEVICES=0 uv run python -m experiments.sci34_supplement.c2_equivalence.run \
  --campaign-dir "$PILOT_DIR" \
  --runtime transformers \
  --model "$MAIN_MODEL" \
  --device cuda:0 \
  --seed 20260902 \
  --limit 3

uv run python -m experiments.sci34_supplement.c2_equivalence.validate \
  --campaign-dir "$PILOT_DIR" \
  --non-formal
```

人工检查 pilot 的 termination probe、token/state/logit/continuation 字段与失败 sidecar 机制。Pilot 只做兼容性/成本预检，**不授予 formal termination 资格**；formal 的每个 case 会重新运行并硬校验自身 probe。若 pilot 已暴露 natural EOS 无法在冻结 128 上限内命中等问题，可提前停止以节省 GPU，但即使 pilot 通过也不能跳过 formal probe。

## 6. 创建 formal 目录与 before snapshot

```bash
export RUN_ID="c2eq_${CODE_COMMIT:0:8}_$(date -u +%Y%m%dT%H%M%SZ)"
export RUN_DIR="$OUT_ROOT/$RUN_ID"

# campaign 命令负责原子创建目录；先记录外部 before snapshot，创建后再复制进去。
mkdir -p "$GUARD/snapshot_before"
git rev-parse HEAD > "$GUARD/snapshot_before/git_commit.txt"
git status --short --branch > "$GUARD/snapshot_before/git_status.txt"
uname -a > "$GUARD/snapshot_before/uname.txt"
(lscpu || true) > "$GUARD/snapshot_before/lscpu.txt"
(free -h || true) > "$GUARD/snapshot_before/memory.txt"
(nvidia-smi || true) > "$GUARD/snapshot_before/nvidia_smi.txt"
(nvidia-smi pmon -c 1 || true) > "$GUARD/snapshot_before/gpu_processes.txt"
(uv pip freeze || true) > "$GUARD/snapshot_before/uv_freeze.txt"
```

## 7. 不可变 formal manifest

```bash
test -z "$(git status --porcelain)" || { git status --short; exit 1; }

CUDA_VISIBLE_DEVICES=0 uv run python -m experiments.sci34_supplement.c2_equivalence.campaign \
  --run-id "$RUN_ID" \
  --output-dir "$RUN_DIR" \
  --runtime transformers \
  --model "$MAIN_MODEL" \
  --device cuda:0 \
  --seed 20260902

mkdir -p "$RUN_DIR/snapshots/before" "$RUN_DIR/snapshots/after"
cp -a "$GUARD/snapshot_before/." "$RUN_DIR/snapshots/before/"
sha256sum "$RUN_DIR/campaign_manifest.json" "$RUN_DIR/cases.json"
```

正式 `campaign` 默认 formal：拒绝 dirty tree、fake runtime、非离线环境和任何非 24-case 网格；模型必须与 D-017 已接受的 Qwen2-7B-Instruct 逐文件内容身份一致且实际以 BF16 加载。Manifest 冻结核心 `src`、campaign/shared code、cases、模型、tokenizer/chat template/special token、dtype/backend、环境和 lock hash；code identity 明确包含 StreamLLM、orchestrator、timeline、supplement common 与 strong-identity 实现，并保存 repo-relative path/hash/size。

## 8. 单 session formal run 与 case 原子 resume

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m experiments.sci34_supplement.c2_equivalence.run \
  --campaign-dir "$RUN_DIR" \
  --runtime transformers \
  --model "$MAIN_MODEL" \
  --device cuda:0 \
  --seed 20260902 \
  2>&1 | tee "$RUN_DIR/logs/formal.log"
```

进程中断后，保持同一目录、模型、代码与 cases 不变，执行：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m experiments.sci34_supplement.c2_equivalence.run \
  --campaign-dir "$RUN_DIR" \
  --runtime transformers \
  --model "$MAIN_MODEL" \
  --device cuda:0 \
  --seed 20260902 \
  --resume \
  2>&1 | tee -a "$RUN_DIR/logs/formal.log"
```

Resume 只跳过 `records.jsonl` 中已完整原子落盘的 case。`attempts.jsonl` 保存每次 attempt/process identity。Identity 变化、重复 case、截断 JSONL、模型/code/cases mismatch 均 fail closed。

Runner 对每个 formal case 先做独立 termination probe，再做共享 retained IDs 的 teacher-force equivalence comparison。`natural_eos` 必须真实 greedy 在 128 内 EOS；`max_tokens` 必须真实 greedy 在预算 2 内无 EOS并报告 `MAX_TOKENS`；`eos_at_cap` 是明确 controlled 的 cap=4 token-selection fixture，但内容仍走 production KV append，最终 EOT 必须走 `generate_accumulating` EOS 分支。另对 `crop_pending_eot`/`reply_tail_noop` 在 teacher-force 内容后强制一次 EOT decode 并调用真实 `generate_accumulating(max_new_tokens=1)`：前者截断 crop 必须清除 pending，后者 current-seq no-op 必须保留 pending，再 reopen。任一 probe 或 equivalence correctness failure 都会保存 raw（logit 失败另存 `failures/*.npz`）后非零退出。不要删除失败目录，不要继续 acceptance/seal。

## 9. After snapshot 与旧结果 guard

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

## 10. 强制顺序：validate → analyze → acceptance → seal → tar

### 10.1 Validate

```bash
uv run python -m experiments.sci34_supplement.c2_equivalence.validate \
  --campaign-dir "$RUN_DIR" \
  --out "$RUN_DIR/validation.json" \
  2>&1 | tee "$RUN_DIR/logs/validation.log"
```

必须 `ok=true`、`acceptance_eligible=true`，并且 `termination_probes.required == observed == qualified == 24`。Formal validator 要求 `summary.json` 存在，并从 records 独立核对 case/checkpoint/failed/process/identity/probe 计数与 verdict；同时从 raw token IDs、hash、cap/EOS step、end reason、role phase、EOT ledger/KV 和 scenario execution 标志独立判定资格，不信任 stored pass。任一失败保留目录并停止。

### 10.2 Analyze

```bash
uv run python -m experiments.sci34_supplement.c2_equivalence.analyze \
  --campaign-dir "$RUN_DIR" \
  --out "$RUN_DIR/analysis_v1.json" \
  2>&1 | tee "$RUN_DIR/logs/analysis_v1.log"
```

Analysis 只做描述性汇总，除 context/scenario/termination/checkpoint 外，单独汇总 termination probe 的资格数、observed end reason、mode、cap、内容 token 数、EOS step、EOT 入 KV/ledger 计数；无 bootstrap。Raw checkpoint 的 `continuation_source` 必须全部为 `actual_crop_cache`，且 p0 empty-assistant boundary=1、speculation full invalidation boundary=0。

### 10.3 人工 acceptance

```bash
cp experiments/sci34_supplement/c2_equivalence/ACCEPTANCE_TEMPLATE.md \
  "$RUN_DIR/ACCEPTANCE.md"
```

从 raw records 独立复算并填写。全部通过后，把状态精确写为一行：

```text
Status: accepted
```

未通过不能伪写 accepted。

### 10.4 Seal

```bash
uv run python -m experiments.sci34_supplement.c2_equivalence.seal \
  --campaign-dir "$RUN_DIR" --create
uv run python -m experiments.sci34_supplement.c2_equivalence.seal \
  --campaign-dir "$RUN_DIR" --verify
```

`seal --create` 会先要求 campaign/cases/records/summary/attempts/progress/logs/snapshots/failures/validation/analysis/acceptance 全套工件存在且关键目录非空，再直接 formal 重跑 validator/analyzer，并比较 stored validation/analysis 的核心 verdict/provenance；三文件伪造不能封存。`checksums.sha256` 使用相对路径字典序，拒绝覆盖；seal 不包含自身和 tarball。

### 10.5 Tar

```bash
export TARBALL="$OUT_ROOT/${RUN_ID}.tar.gz"
test ! -e "$TARBALL" && test ! -e "${TARBALL}.sha256" || { printf '回传包已存在，拒绝覆盖：%s
' "$TARBALL" >&2; exit 1; }
tar -C "$OUT_ROOT" -czf "$TARBALL" "$RUN_ID"
sha256sum "$TARBALL" | tee "${TARBALL}.sha256"
```

## 11. 回传与红线

回传：完整 `$RUN_DIR`、tarball、tarball hash、`$E3_RESCUE`。不能只回传 summary/analysis。

红线：

1. 不覆盖旧结果、validation、analysis、seal 或 tarball。
2. 不删除 failed records/attempts/NPZ。
3. 不修改 24 cases、32 continuation、top-k 或 BF16 `0.1/0.01` 阈值。
4. 不用 pilot/fake smoke 更新论文。
5. 不把单 snapshot 正确性推广到其他模型/backend/dtype。
6. 不修改 `src`、其他文档或论文以“配合结果”。
7. 未经明确授权不 commit/push。
