# R7 正式实验交接文档（Gate 版 r2，2026-08-22）

- **执行权限划分（消除流程循环，对应审查 2026-08-22 终裁）**：
  - **放行前允许执行的 Gate（无需书面放行，GPU 主机即可执行）**：
    §1 的 G1–G8 采集（clean 树/provenance/TTS 服务端/平台条件）＋ §2b 非末位 fatal 小 smoke
    ＋ §2c GPU clean 树 self-test 归档。产物 push 后提交**最终放行复核**；
  - **需审查方书面放行后才可执行**：仅 §2 正式 run（r7_main，120 任务）与
    §3 匹配文本控制（依赖 r7_main 产物）。两者在放行前不得启动。
- 代码基线：`git pull` 至本次 push（含 `--tts-control-only`/`--inject-fault-index`/
  speaker 与平台条件绑定）或更新
- 脚本 self-test 期望值：**90 PASS / 0 FAIL**

## 0. GPU 主机待执行清单（放行前 Gate，按序）

| 步骤 | 产物（随放行申请提交） |
|---|---|
| §1 G1 | `git status --porcelain` 空输出记录（clean 树） |
| §1 G2 | 拟用于正式 run 的 code_commit（pull 后 HEAD） |
| §1 G7 | CosyVoice commit+本地 diff、镜像 digest、模型与 `spk2info.pt` hash、启动配置 |
| §1 G8 | `env/platform_conditions.txt`（驱动/CUDA/fallback 登记/双 3090/独占声明/nvidia-smi） |
| §2b | `checkpoint_r7_smoke_fatal.jsonl` + `QA_r7_smoke_fatal.md`（非末位 fatal→cancelled 证据） |
| §2c | `selftest_archive/selftest_gpu_YYYYMMDD.log`（GPU clean 树 90 PASS + exit code） |

全部齐备 → 开发侧核验 → 提交审查方最终放行复核 → 书面放行后执行 §2、§3。

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

```bash
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
    --output-dir experiments/results/revision/r7_ttfa_unified \
    --run-id r7_main
```

- 120 任务（50×2 + 10 子集 ×2 模式 ×补 2 轮），预计 3–5 小时；脚本启动时自动做**新一轮探活**
  并把 probe 绑定进 checkpoint（G5）；Silero artifact hash 自动核验并断言 PSE/分段器一致（G6）；
- RUNINFO 自动记录：speaker 映射注记、platform_conditions_sha256、code_commit、
  config/schedule/git/env hash（G3/G4/G10）；
- 产物 push 后形成 result_artifact_commit，与 code_commit 一并回传（区分三者）；
- fatal/thread_leak/pair_timeout/schema 错误 → 立即停止，保留终态记录，反馈现场。

### 2b. 非末位 fatal 小 smoke（Gate 第 10 项：cancelled 运行级证据；**在 §2 正式 run 之前执行**；独立 run，不入正式结果）

```bash
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
    --output-dir experiments/results/revision/r7_ttfa_unified \
    --run-id r7_smoke_fatal
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
    --control-from experiments/results/revision/r7_ttfa_unified/checkpoint_r7_main.jsonl \
    --sample-list experiments/results/revision/r1_stats/repeat_subset_ids.json \
    --platform-conditions-file experiments/results/revision/r7_ttfa_unified/env/platform_conditions.txt \
    --tts-url http://127.0.0.1:20401 --tts-spk 晓伊 --tts-speed 0.8 \
    --output-dir experiments/results/revision/r7_ttfa_unified \
    --run-id r7_tts_control
```

- 自动分层选 10 条成功配对（zh/en 各 5），每样本 3 次 TTS 调用（B 首句重测 / A 回复首句 /
  A 全文）+ 固定校准句中英各一 = 32 调用；产物 `tts_control_r7_tts_control.csv` + RUNINFO；
- 不加载 ASR/LLM/Silero；成功配对不足 10 或 zh/en 不足 5 → fail-closed。

## 4. 正式验收清单（在原八项上按 Gate 加三项）

1. 原 r4 版八项（QA 0 / 120 全终态 / 子集恰三轮 / validate 全过 / TTS 无 error / RUNINFO 齐全…）；
2. **G1-G8 证据齐备**（clean 树、code_commit、新 checkpoint、探活绑定、Silero hash、
   TTS 服务 provenance、平台条件文件 hash 入 binding）；
3. **commit 三元组回传**：code_commit（RUNINFO 内）/ result_artifact_commit（push 后）/
   verification_commit（本机核验后）；
4. 晓伊→内置中文女映射、CUDA/Triton fallback、max_tokens=128 等限定已由 RUNINFO/注记固化
   （论文边界照 review §6 执行，不在 GPU 侧处理）。

## 5. 产物与反馈

`r7_ttfa_unified/`（checkpoint_r7_main、RUNINFO/QA/summary/CV、tts_control 三件套、
run.log、tts_probe.json、env/ 含 platform_conditions.txt 与 G1-G8 证据）→ push →
本机结果级核验（区分三元 commit）→ 装配新 Table VIII → W8 阶段 2 → 审查复核 → 论文放行。
