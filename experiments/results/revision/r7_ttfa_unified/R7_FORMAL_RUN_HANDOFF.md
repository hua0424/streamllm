# R7 正式实验交接文档（Gate 版 r3，2026-08-22）

- **本版针对 Gate 复核的核心修正**：正式 code_commit 唯一化、clean 记录与材料归档分离、fatal smoke 显式绑定 platform conditions、完整 manifest 覆盖全部放行材料；manifest hash 统一明示为 **LF-normalized 内容 SHA-256**（不是 Git blob hash）。
- **执行权限**：G1–G8 采集、§2b fatal smoke、§2c GPU self-test 仍为书面放行前允许的 Gate；仅 §2 r7_main 与其后的 §3 control 需书面放行。
- **代码基线**：GPU 主机先 pull 本文档所在提交，随后选定唯一 `code_commit` 并在该提交 clean checkout；后续 Gate 运行不得混用其他 commit。
- self-test 期望：90 PASS / 0 FAIL

## 0. GPU 主机待执行清单（放行前 Gate，**r3 原子材料包流程，按序执行**）

> **关键规则**：先在唯一 code_commit 的 clean checkout 完成全部 Gate；运行会产生未跟踪材料文件，
> 这是预期的 artifact，不可再把 `git status --porcelain` 当作 Gate 后期 clean 条件。
> clean 证明（G1）必须在任何 Gate artifact 生成**之前**保存；Gate 完成后将材料提交为单独
> `result_artifact_commit`，提交后再次记录 porcelain 为空。

| 步骤 | 产物（随放行申请提交） |
|---|---|
| §1 G1/G2 | `env/gate/gate_clean_git.txt` | **Gate 前**唯一 code_commit 的 HEAD + 真正空 porcelain（材料生成前） |
| §1 G7 | `env/gate/tts_provenance/` | CosyVoice commit+本地 diff、镜像 digest、模型与 `spk2info.pt` hash、启动配置 |
| §1 G8 | `env/platform_conditions.txt` | 驱动/CUDA/fallback 登记/双 3090/独占声明/nvidia-smi |
| §2b | `fatal_smoke/checkpoint_r7_smoke_fatal.jsonl` + `fatal_smoke/RUNINFO_r7_smoke_fatal.md`/`fatal_smoke/QA_r7_smoke_fatal.md`/**`fatal_smoke/r7_smoke_fatal_run.log`** | 非末位 fatal→cancelled 证据（独立子目录） |
| §2c | `env/gate/gate_selftest_gpu.log` + `.md` | GPU clean 树 90 PASS + exit code；md 附命令/HEAD/环境/输出 sha256 |
| Gate 后 | `env/gate/gate_artifact_commit.txt` | Gate 材料提交后的 artifact commit + porcelain 空证明 |

全部齐备 → 按 §0b 打包 Gate manifest → 开发侧核验 → 提交审查方最终放行复核 → 书面放行后执行 §2、§3。

### 0b. 最小放行材料包（与审查 `review-reply-final-gate-20260822.md` §5 命名逐项对应）

| # | 材料 | 生成 |
|---|---|---|
| 1 | `env/gate/gate_clean_git.txt` | `{ echo "HEAD=$(git rev-parse HEAD)"; echo "--- porcelain ---"; git status --porcelain; echo "---(空=clean)---"; } > …/env/gate/gate_clean_git.txt` |
| 2 | `env/gate/gate_selftest_gpu.log` + `.md` | §2c 命令输出（90 PASS + exit code；.md 附命令/HEAD/环境/输出 sha256） |
| 3 | `fatal_smoke/checkpoint_r7_smoke_fatal.jsonl` + 同目录 `RUNINFO_r7_smoke_fatal.md`/`QA_r7_smoke_fatal.md`/run log | §2b 小 smoke 产物（独立子目录，见 §0c） |
| 4 | `env/platform_conditions.txt` | §1 G8 采集（CPU/OS/kernel/线程、双 3090/驱动/CUDA、Triton fallback 状态、ASR/LLM/TTS 显存分配、独占声明、nvidia-smi 原始快照、TTS 与 ASR 共用 cuda:0 说明） |
| 5 | `env/gate/tts_provenance/` | §1 G7：CosyVoice commit + 本地 diff、image ID/registry digest、模型 snapshot/hash、`spk2info.pt` hash、启动命令/挂载/环境变量非敏感摘要、依赖快照 |
| 6 | `env/gate/tts_probe_new.json` | 新一轮探活（header/payload 允许策略；speaker 映射注记已在 probe 输出） |
| 7 | `env/gate/GATE_MANIFEST.md` | 见下方命令 |

Gate manifest 生成（**全部 hash 按 `dos2unix` 等价的 LF-normalized 内容 SHA-256，非 Git blob hash/工作树原始字节 hash**；覆盖全部放行依据）：

```bash
cd /dataA/streamllm/experiments/results/revision/r7_ttfa_unified
mkdir -p env/gate
lfsha() { printf '%s  %s\n' "$(sed 's/\r$//' "$1" | sha256sum | awk '{print $1}')" "$1"; }
{
  echo "# R7 放行前 Gate manifest";
  echo "hash_scheme=LF-normalized-content-sha256";
  echo "code_commit=$(cat env/gate/gate_clean_git.txt | sed -n 's/^HEAD=//p')";
  echo "generated=$(date -Is)";
  lfsha env/gate/gate_clean_git.txt;
  lfsha env/gate/gate_selftest_gpu.log;
  lfsha env/gate/gate_selftest_gpu.md;
  lfsha fatal_smoke/checkpoint_r7_smoke_fatal.jsonl;
  lfsha fatal_smoke/RUNINFO_r7_smoke_fatal.md;
  lfsha fatal_smoke/QA_r7_smoke_fatal.md;
  lfsha fatal_smoke/r7_smoke_fatal_run.log;
  lfsha env/platform_conditions.txt;
  lfsha env/gate/tts_probe_new.json;
  for f in env/gate/tts_provenance/*; do lfsha "$f"; done;
} > env/gate/GATE_MANIFEST.md
git add -A && git commit -m "R7放行前Gate材料包" && git push
{ echo "artifact_commit=$(git rev-parse HEAD)"; echo "--- porcelain ---"; git status --porcelain; } \
  > env/gate/gate_artifact_commit.txt
git add env/gate/gate_artifact_commit.txt && git commit -m "登记R7 Gate材料artifact commit" && git push
```

> 注意：先 `mkdir -p fatal_smoke` 再执行 §2b 命令并用 `tee fatal_smoke/r7_smoke_fatal_run.log`
> 保存完整控制台日志；run log 属 Gate 材料，不能缺失。


### 0c. 目录安排说明（消除与"一目录一 checkpoint"守卫的冲突，方案 A）

守卫语义**不变**（同目录存在其他 run 的 checkpoint 即拒——审查已确认的 fail-closed 行为）。
改为**每个 run 独立子目录**：§2b → `fatal_smoke/`，§2 → `r7_main/`，§3 → `tts_control/`。
既有产物（r7_smoke 三件套、env/、selftest_archive/）留在主目录不动；G1/G2/G7/G8/2c
已采集产物无需重跑，仅 §2b 待按新目录重跑并重新生成 GATE_MANIFEST。

## 1. 启动前 Gate 清单（逐项执行并留存证据）

```bash
cd /dataA/streamllm && git pull
# G1 clean 工作树（dirty 不得开跑；若有本地改动须留 patch 并获批准）
git status --porcelain | tee /tmp/g1_status.txt      # 期望：空
git rev-parse HEAD | tee /tmp/g2_code_commit.txt     # G2 批准的 code_commit（回传待批）
# G7 TTS 服务端 provenance
docker inspect --format='{{.Image}}' $(docker ps -q --filter ancestor=cosyvoice:v2.0) \
  | tee /tmp/g7_image_digest.txt                     # 镜像 digest
docker images --digests | grep cosyvoice | tee -a /tmp/g7_image_digest.txt
sha256sum <CosyVoice模型目录>/….pt <TTS 服务 spk2info.pt 路径> | tee /tmp/g7_model_hashes.txt
# CosyVoice 服务代码 commit + 本地修改 diff（server.py/requirements 有本地修改）
cd <CosyVoice 仓库目录> && git rev-parse HEAD && git diff > /tmp/g7_server_dirty.patch && cd /dataA/streamllm
# G8 平台条件记录（--platform-conditions-file 将绑定其 hash）
{ nvidia-smi; nvcc --version 2>/dev/null || cat /usr/local/cuda/version.txt 2>/dev/null;
  grep -c "falling back to a slower" <(tail -100 r7_ttfa_unified/r7_smoke_run.log) || true;
  echo "triton_fallback=observed_in_smoke_log(已登记为第二平台固定条件)";
  echo "GPU 独占声明: 实验期间无其他 GPU 作业"; nvidia-smi --query-gpu=memory.used,memory.total --format=csv; } \
  > experiments/results/revision/r7_ttfa_unified/env/platform_conditions.txt
```

## 2. 正式主实验 + 重复子集（新 checkpoint，绝不从 smoke 续跑；**须书面放行后执行**。§2b/§2c 编号在后，但属放行前 Gate，先于本节完成）

> **版本操作（书面放行后）**：正式 run 在 **origin/main 的 HEAD**（当前 `b5355ee` 及其后的
> 放行提交）上执行，`git pull --ff-only` 后工作树须 clean。脚本 `run_ttfa_unified.py`、
> `src/`、`sample-list` 在 `b8893d6` 与 origin/main 间**逐字节一致**，故 RUNINFO 记录的
> `code_commit` 仍为 `b8893d6`（正式代码基线不变）；但放行绑定的 `platform_conditions.txt`
> （hash `a4c40057…`）只在 Gate 材料包 `a1fbb82` 及之后存在，`b8893d6` 中是旧版 `6b0a2fcd…`，
> 因此**必须在 origin/main 上运行**，`--platform-conditions-file` 才指向放行版。

```bash
cd /dataA/streamllm && git pull --ff-only origin main && git status --porcelain   # 期望：空
uv run python -m experiments.scripts.run_ttfa_unified \
    --sample-list experiments/results/revision/r1_stats/repeat_subset_ids.json \
    --json-dir experiments/datasets/processed/json \
    --audio-dir experiments/datasets/processed/audio \
    --datasets crosswoz multiwoz \
    --asr-model turbo --asr-device cuda:0 \
    --llm-model Qwen/Qwen2-7B-Instruct --llm-device cuda:1 \
    --tts-url http://127.0.0.1:20401 --tts-spk 晓伊 --tts-speed 0.8 \
    --silero-dir ~/.cache/torch/hub/snakers4_silero-vad_master \
    --platform-conditions-file experiments/results/revision/r7_ttfa_unified/env/platform_conditions.txt \
    --output-dir experiments/results/revision/r7_ttfa_unified/r7_main \
    --run-id r7_main
```

- **140 任务**（50×2 主实验 = 100；10 子集 ×2 模式 ×2 补轮 = 40；合计 140），预计 4–6 小时；
  脚本启动时自动做**新一轮探活**并把 probe 绑定进 checkpoint（G5）；Silero artifact hash 自动
  核验并断言 PSE/分段器一致（G6）；
  - 任务数由 `build_schedule` 实际产出：`repeat 0`=50 样本×2 模式=100，`repeat 1/2`=10 子集
    ×2 模式×2 轮=40。QA 用 `tasks` 动态计算预期，不硬编码 120/140；RUNINFO 会打印实际
    `任务数`，以脚本输出为准。
- RUNINFO 自动记录：speaker 映射注记、platform_conditions_sha256、code_commit（`b8893d6`）、
  config/schedule/git/env hash（G3/G4/G10）；
- 产物 push 后形成 result_artifact_commit，与 code_commit 一并回传（区分三者）；
- fatal/thread_leak/pair_timeout/schema 错误 → 立即停止，保留终态记录，反馈现场。

### 2b. 非末位 fatal 小 smoke（Gate 第 10 项：cancelled 运行级证据；**在 §2 正式 run 之前执行**；独立 run，不入正式结果）

```bash
mkdir -p experiments/results/revision/r7_ttfa_unified/fatal_smoke
uv run python -m experiments.scripts.run_ttfa_unified \
    --sample-list experiments/results/revision/r1_stats/repeat_subset_ids.json \
    --json-dir experiments/datasets/processed/json \
    --audio-dir experiments/datasets/processed/audio \
    --datasets crosswoz multiwoz \
    --asr-model turbo --asr-device cuda:0 \
    --llm-model Qwen/Qwen2-7B-Instruct --llm-device cuda:1 \
    --tts-url http://127.0.0.1:20401 --tts-spk 晓伊 --tts-speed 0.8 \
    --silero-dir ~/.cache/torch/hub/snakers4_silero-vad_master \
    --smoke 3 --inject-fault asr_error --inject-fault-index 1 \
    --output-dir experiments/results/revision/r7_ttfa_unified/fatal_smoke \
    --run-id r7_smoke_fatal \
    2>&1 | tee experiments/results/revision/r7_ttfa_unified/fatal_smoke/r7_smoke_fatal_run.log
```

验收：任务 0 success；任务 1 error（含 `fault_injection`）且 fatal=True；任务 2–5 全部
`terminal_state=cancelled`、`error=cancelled_after_fatal`；QA 记录数=6。该 run 与 r7_main
完全独立，不参与任何论文数据。

### 2c. self-test 归档（Gate 第 11 项；**在 §2 正式 run 之前执行**）

开发侧已归档本机 90 PASS（`selftest_archive/selftest_20260822.md/.log`，含命令/退出码/
HEAD/输出 hash）。GPU 主机请在 clean 树上复跑并另存：

```bash
uv run python -m experiments.scripts.run_ttfa_unified --self-test \
    > experiments/results/revision/r7_ttfa_unified/selftest_archive/selftest_gpu_$(date +%Y%m%d).log 2>&1
echo "exit=$?" >> experiments/results/revision/r7_ttfa_unified/selftest_archive/selftest_gpu_$(date +%Y%m%d).log
```

## 3. 匹配文本 TTS 控制（主实验完成后；`--tts-control-only` 已实现并自测）

```bash
uv run python -m experiments.scripts.run_ttfa_unified \
    --tts-control-only \
    --control-from experiments/results/revision/r7_ttfa_unified/r7_main/checkpoint_r7_main.jsonl \
    --sample-list experiments/results/revision/r1_stats/repeat_subset_ids.json \
    --platform-conditions-file experiments/results/revision/r7_ttfa_unified/env/platform_conditions.txt \
    --tts-url http://127.0.0.1:20401 --tts-spk 晓伊 --tts-speed 0.8 \
    --output-dir experiments/results/revision/r7_ttfa_unified/tts_control \
    --run-id r7_tts_control
```

- 自动分层选 10 条成功配对（zh/en 各 5），每样本 3 次 TTS 调用（B 首句重测 / A 回复首句 /
  A 全文）+ 固定校准句中英各一 = 32 调用；产物 `tts_control_r7_tts_control.csv` + RUNINFO；
- 不加载 ASR/LLM/Silero；成功配对不足 10 或 zh/en 不足 5 → fail-closed。

## 4. 正式验收清单（在原八项上按 Gate 加三项）

1. 原 r4 版八项（QA 0 / 140 全终态 / 子集恰三轮 / validate 全过 / TTS 无 error / RUNINFO 齐全…）；
2. **G1-G8 证据齐备**（clean 树、code_commit、新 checkpoint、探活绑定、Silero hash、
   TTS 服务 provenance、平台条件文件 hash 入 binding）；
3. **commit 三元组回传**：code_commit（RUNINFO 内）/ result_artifact_commit（push 后）/
   verification_commit（本机核验后）；
4. 晓伊→内置中文女映射、CUDA/Triton fallback、max_tokens=128 等限定已由 RUNINFO/注记固化
   （论文边界照 review §6 执行，不在 GPU 侧处理）。

## 5. 产物与反馈

`r7_ttfa_unified/`（`r7_main/` 子目录：checkpoint/RUNINFO/QA/summary/CV；
`tts_control/` 子目录：控制三件套；主目录：run.log、tts_probe.json、
env/ 含 platform_conditions.txt 与 G1-G8 证据）→ push →
本机结果级核验（区分三元 commit）→ 装配新 Table VIII → W8 阶段 2 → 审查复核 → 论文放行。
