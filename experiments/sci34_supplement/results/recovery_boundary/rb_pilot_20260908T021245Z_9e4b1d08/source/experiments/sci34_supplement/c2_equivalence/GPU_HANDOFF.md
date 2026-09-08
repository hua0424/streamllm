# C2 equivalence GPU 执行交接（协议 v2）

> 实验机唯一操作入口。以下命令按当前已实现 CLI 写成。正式证据固定 24 cases、1 session、无统计重复。任何失败保留目录并停止 acceptance/seal。
>
> **v2 背景（D-019）**：v1 run `c2eq_563dd22a_20260903T013547Z` 判定 rejected 并永久归档——token/state 层 100% 等价，但绝对 BF16 logit 阈（0.1/0.01）与 32-token greedy exact 对任何正确实现不可达成（增量 append vs 整段 prefill 的核归约差异），且 4/10 natural_eos greedy 在 128 cap 内 run-on。v2 引入噪声对照臂 + 相对门槛 + margin 感知 + natural_eos cap 256/重资格化，常数全部先验冻结。**本轮所有 v1 目录（`c2eq_563dd22a_*`、`c2pilot_563dd22a_*`、`e3_exact_rescue/`）只读，不得改动。**

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

# 协议版本预检：必须是 v2
uv run python - <<'PY'
from experiments.sci34_supplement.c2_equivalence.protocol import PROTOCOL_VERSION
assert PROTOCOL_VERSION == 2, PROTOCOL_VERSION
print("C2 protocol version:", PROTOCOL_VERSION)
PY

export OUT_ROOT=experiments/sci34_supplement/results/c2_equivalence
export GUARD=/tmp/c2_equivalence_guard_v2
mkdir -p "$GUARD"

# 旧结果统一内容 guard（含 v1 C2 归档 run 与 E3 抢救件，全部只读）。
git ls-files experiments/results experiments/sci34_supplement/results \
  | sort > "$GUARD/legacy_paths.txt"
while IFS= read -r path; do sha256sum "$path"; done < "$GUARD/legacy_paths.txt" \
  > "$GUARD/legacy_before.sha256"
sha256sum pyproject.toml uv.lock > "$GUARD/env_before.sha256"
```

正式前必须先在本地形成包含协议 v2 的 exact clean commit 并推送（设计侧提供 commit；本交接不授权实验机 commit/push）。

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

`c2_equivalence.smoke` 输出必须明确 `protocol_version=2`、`models_loaded=false`、`network_used=false`、24 cases PASS、`checkpoint_sidecars=45`。任一测试失败停止。

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

## 4. E3 exact `p2_turns.json` 抢救（v1 轮已完成，本轮默认跳过）

Accepted E3 输入已在 v1 轮抢救并入库 `results/e3_exact_rescue/`（SHA `a2116b83…` 核验一致）。仅当该目录缺失时按 v1 步骤重做；本轮 guard 已把它纳入只读保护。

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

人工检查 pilot 的 termination probe（注意 natural_eos cap 现为 256，未命中者 `requalified=true` 且记录自洽）、noise_control/logit_gates/margin 字段与 `checkpoints/*.npz` sidecar 机制（v2 起 45 个 checkpoint 全量保存）。Pilot 只做兼容性/成本预检，**不授予 formal termination 资格**；formal 的每个 case 会重新运行并硬校验自身 probe。参考量级：v2 每 case 约多一次对照臂增量 prefill，v1 formal 全程约 5 分钟，v2 预计 10 分钟内；tarball 约 60–100MB（45×3 个 FP32 logits 数组）。

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

Runner 对每个 formal case 先做独立 termination probe，再做共享 retained IDs 的 teacher-force equivalence comparison，并为每个 checkpoint 增算 v2 噪声对照臂（canonical 序列按结构 seam 分块增量 prefill，结尾单 token refresh）。`natural_eos` 真实 greedy 在 cap 256 内命中 EOS 记 genuine，未命中则重资格化为 max_tokens 语义（须自洽；campaign 级要求 ≥5/10 genuine）；`max_tokens` 必须真实 greedy 在预算 2 内无 EOS 并报告 `MAX_TOKENS`；`eos_at_cap` 是明确 controlled 的 cap=4 token-selection fixture，但内容仍走 production KV append，最终 EOT 必须走 `generate_accumulating` EOS 分支。另对 `crop_pending_eot`/`reply_tail_noop` 在 teacher-force 内容后强制一次 EOT decode 并调用真实 `generate_accumulating(max_new_tokens=1)`：前者截断 crop 必须清除 pending，后者 current-seq no-op 必须保留 pending，再 reopen。等价门槛为 v2 相对门槛（2× 控制臂噪声 + 绝对安全上限 + 近并列 margin 规则，常数见 EXPERIMENT_PLAN §0）。任一 probe 或 correctness failure 都会保存 raw 与 `checkpoints/*.npz` 后非零退出。不要删除失败目录，不要继续 acceptance/seal。

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

必须 `ok=true`、`acceptance_eligible=true`，`termination_probes.required == observed == qualified == 24`，且 `termination_probes.natural_eos.genuine >= 5`。Formal validator 要求 `summary.json` 存在并含 `protocol_version=2` 与 natural_eos_gate，从 records 独立核对 case/checkpoint/failed/process/identity/probe 计数与 verdict；**从 `checkpoints/*.npz` 三数组独立重算**全部 logit/control 统计、top-1/top-5/margin 与 v2 相对门槛，同时从 raw token IDs、hash、cap/EOS step、end reason、role phase、EOT ledger/KV 和 scenario execution 标志独立判定资格，不信任 stored pass。任一失败保留目录并停止。

### 10.2 Analyze

```bash
uv run python -m experiments.sci34_supplement.c2_equivalence.analyze \
  --campaign-dir "$RUN_DIR" \
  --out "$RUN_DIR/analysis_v1.json" \
  2>&1 | tee "$RUN_DIR/logs/analysis_v1.log"
```

Analysis 只做描述性汇总，除 context/scenario/termination/checkpoint 外，单独汇总 termination probe 的资格数（含 genuine/requalified）、observed end reason、mode、cap、内容 token 数、EOS step、EOT 入 KV/ledger 计数，以及 v2 `noise_control`（控制臂噪声量级与 path/control 比值）；无 bootstrap。Raw checkpoint 的 `continuation_source` 必须全部为 `actual_crop_cache`，且 p0 empty-assistant boundary=1、speculation full invalidation boundary=0。

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

`seal --create` 会先要求 campaign/cases/records/summary/attempts/progress/logs/snapshots/checkpoints/validation/analysis/acceptance 全套工件存在且关键目录非空，再直接 formal 重跑 validator/analyzer，并比较 stored validation/analysis 的核心 verdict/provenance；三文件伪造不能封存。`checksums.sha256` 使用相对路径字典序，拒绝覆盖；seal 不包含自身和 tarball。

### 10.5 Tar

```bash
export TARBALL="$OUT_ROOT/${RUN_ID}.tar.gz"
test ! -e "$TARBALL" && test ! -e "${TARBALL}.sha256" || { printf '回传包已存在，拒绝覆盖：%s
' "$TARBALL" >&2; exit 1; }
tar -C "$OUT_ROOT" -czf "$TARBALL" "$RUN_ID"
sha256sum "$TARBALL" | tee "${TARBALL}.sha256"
```

## 11. 回传与红线

回传：完整 `$RUN_DIR`（含 `checkpoints/` 全量 sidecar）、tarball、tarball hash。不能只回传 summary/analysis。

红线：

1. 不覆盖旧结果、validation、analysis、seal 或 tarball；v1 归档 run 与 `e3_exact_rescue/` 只读。
2. 不删除 failed records/attempts/NPZ。
3. 不修改 24 cases、32 continuation、top-k，或 v2 冻结常数（2.0× 比率、0.05/0.01 噪声下限、2.0/0.5 安全上限、0.125 margin 下限、256 cap、≥5 genuine）。绝对不得回退到 v1 的 0.1/0.01 绝对阈。
4. 不用 pilot/fake smoke 更新论文。
5. 不把单 snapshot 正确性推广到其他模型/backend/dtype。
6. 不修改 `src`、其他文档或论文以“配合结果”。
7. 未经明确授权不 commit/push（结果回传按既有惯例 `git add -f` 入库除外）。
