# GPU 交接：R 同策略恢复 + B 软件边界（2026-09-06）

## 0. 状态及只读范围

实现位于 `experiments/sci34_supplement/recovery_boundary/`。权威中文/SC 稿未修改；旧 E1/E2/E3/A1/P1、C2 v1/v2/v3 只读。

**交付时为本地未提交代码，未 commit/push。用户需明确授权并完成审阅/commit/push 后 GPU 才能 pull。不能把当前 HEAD 当作包含本实现。** 原有 docs/decisions.md、docs/paper2_context.md 及 review/tougao 未提交改动须由作者处理；不能替作者丢弃、stash 或混入提交。formal gate 会拒绝 dirty tree。

GPU 主机不可访问，未核验当前安装/缓存/空闲显存。历史正式配置证据（普通 1-based 行号）：
`experiments/sci34_supplement/results/c2_crop_integrity/c2crop_82103004_20260903T080512Z/campaign_manifest.json`
- L231：`/root/autodl-tmp/dataA/models/Qwen2-7B-Instruct`
- L316：`torch.bfloat16`
- L322：`sdpa`
- L36–39：artifact 指纹 `fae2ece10b76512237cf28957f98e7b0d2c609455a173031e3bd16b3dff7c5ab` / qwen2 / Qwen2ForCausalLM。

不是 Qwen2.5-7B。运行栈冻结 torch 2.8.0+cu128 / transformers 4.57.1，版本不符停止，不自动升级或降级。runner 对所有稳定模型文件内容校验，不接受名字相同但文件不同的目录。不要在模型目录添加下载日志以免改变该文件集合身份。

## 1. 拉取与环境（Linux Bash，源码根目录）

以下路径为历史默认，可按实际 checkout 位置设置 REPO；不是已验证远程路径。全部使用 uv；不会自动下载模型。

```bash
set -euo pipefail
export REPO=/root/autodl-tmp/dataA/streamllm
cd "$REPO"
test -z "$(git status --porcelain --untracked-files=all)"
git -c core.autocrlf=false pull --ff-only origin paper2
test -z "$(git status --porcelain --untracked-files=all)"
export SOURCE_COMMIT=$(git rev-parse HEAD)
# 必须核对作者交付的完整 commit，且该 commit 包含 recovery_boundary 模块。
git log -1 --format='%H %s'
test -f experiments/sci34_supplement/recovery_boundary/campaign.py
export MODEL=/root/autodl-tmp/dataA/models/Qwen2-7B-Instruct
export OUTPUT=/root/autodl-tmp/recovery_boundary_runs
export CUDA_VISIBLE_DEVICES=0
export HF_TOKEN=
unset HUGGING_FACE_HUB_TOKEN
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export TOKENIZERS_PARALLELISM=false
# 不要 env/printenv，不读取或发送 .env，不输出凭据。
nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
df -h "$REPO" /root/autodl-tmp
uv --version
# 离线、锁定同步；缺包就停，先报告，不默默换版本/下载模型。
uv sync --frozen --offline
uv run --no-sync python -c 'import torch,transformers; print("torch",torch.__version__,"transformers",transformers.__version__,"cuda",torch.version.cuda,"available",torch.cuda.is_available()); assert torch.cuda.is_available(); assert torch.cuda.device_count()==1; print("GPU",torch.cuda.get_device_name(0),"free/total",torch.cuda.mem_get_info(0)); assert torch.cuda.is_bf16_supported()'
```

2×3090 24GB 默认仅暴露一张空闲卡，按 session 顺序运行。7B BF16 权重约 14–16GB，另需 KV/activations/框架；实际峰值以 pilot 为准，不保证 24GB 一定容纳所有现场环境。无需 NVLink、tensor parallel 或两卡分片。另一张卡不启动额外研究模型。尽量无其他 GPU 工作负载；OOM/模型身份错误立即停止，保留目录，不量化/换模型/缩 grid。

## 2. 本地软件检查及工程 pilot

```bash
uv run --no-sync python -m experiments.sci34_supplement.recovery_boundary.smoke
uv run --no-sync python -m src.dialogue.run_timeline_test
# chunker 依赖的 nltk 资源必须已缓存；不自动下载，缺失则记录并停止检查。
P2_LLM_MODEL_NAME="$MODEL" DEVICE=cuda:0 uv run --no-sync python -m src.tts.run_chunker_test
uv run --no-sync python -m experiments.sci34_supplement.c2_equivalence.smoke
uv run --no-sync python -m experiments.sci34_supplement.recovery_boundary.campaign run \
  --pilot --model "$MODEL" --output-root "$OUTPUT"
```

pilot 只有 3 cases（512 普通、2048 两次打断、8192 普通），每 case 1 对预热+1 对计时；1 个进程、8 arm-event records。run 会打印 `RUN_DIR=...` 与 `ARCHIVE=...`，成功自动 validate/analyze/seal/verify/tar。保存全部 pilot 工件但**不作为正式支持证据**。

读取 `session_0/complete.json` elapsed_s、records 的 peak_allocated/reserved、日志错误；不要根据差值正负决定运行规模。Formal 固定 4 sessions×9 cases×(1 warmup+4 measured pairs)，320 records。pilot 所测不同 case 的成本不可简单当作整个 grid 精确预测；工程上可按正式各长度/事件工作量粗估并留加载/哈希余量，不能保证完成小时数。作者尚无明确 GPU 小时上限；若预计无法接受，停在这里反馈，不擅自改 preset。无自动无限重试或第二轮找正面结果。

## 3. 正式 all-in-one（pilot 通过且资源/预算可接受后）

```bash
set -euo pipefail
cd "$REPO"
test -z "$(git status --porcelain --untracked-files=all)"
test "$(git rev-parse HEAD)" = "$SOURCE_COMMIT"
uv run --no-sync python -m experiments.sci34_supplement.recovery_boundary.campaign run \
  --source-commit "$SOURCE_COMMIT" --model "$MODEL" --output-root "$OUTPUT"
```

该命令：强身份检查→冻结 manifest/source→B 重放→4 个独立模型子进程（单卡顺序）→完整网格/账本/有限性验证→配对 session 分析→seal/verify→tar.gz+SHA256。每个 run UUID 新目录，不覆盖、不 resume。first_deliverable 为排除 CPU 审计暂停后的两个同步区间之和；包含 generator 的首 token KV commit，不能当作不间断声学 E2E。

若失败：停止，不删除/重跑覆盖；发送 RUN_DIR、FAILED 类型、对应 session log（先人工检查无凭据）、完整失败目录。可在目录之外手工创建 failure tar：

```bash
# RUN_DIR 填失败实际路径；目录不再写入后才归档。
export RUN_DIR=/root/autodl-tmp/recovery_boundary_runs/rb_formal_REPLACE
FAIL_TAR="${RUN_DIR}.failed.tar.gz"
test ! -e "$FAIL_TAR"
tar -czf "$FAIL_TAR" -C "$(dirname "$RUN_DIR")" "$(basename "$RUN_DIR")"
sha256sum "$FAIL_TAR" > "${FAIL_TAR}.sha256"
```

没有真实 GPU 运行的结果不得写 accepted；有限数值差不要求 BF16 cross-topology bitwise，不能仿 C2 v1/v2 添加经验容差。

## 4. 成功复核与回传

```bash
# 使用 run 命令打印的真实目录，禁止猜测。
export RUN_DIR=/root/autodl-tmp/recovery_boundary_runs/rb_formal_REPLACE
uv run --no-sync python -m experiments.sci34_supplement.recovery_boundary.campaign validate --run-dir "$RUN_DIR"
uv run --no-sync python -m experiments.sci34_supplement.recovery_boundary.campaign verify --run-dir "$RUN_DIR"
# seal 已由 run 完成；不要再次 seal，不往 sealed dir tee 日志。
cd "$(dirname "$RUN_DIR")"
sha256sum -c "$(basename "$RUN_DIR").tar.gz.sha256"
```

从设计机取回 run 打印的 tar.gz 和 .sha256（示例占位 host，不假装能访问）：

```bash
scp GPU_HOST:/root/autodl-tmp/recovery_boundary_runs/rb_formal_ACTUAL.tar.gz ./
scp GPU_HOST:/root/autodl-tmp/recovery_boundary_runs/rb_formal_ACTUAL.tar.gz.sha256 ./
```

同时回传 pilot tar/hash、准确 SOURCE_COMMIT、启动命令、资源检查和任何失败目录。tar 内含 raw rows、NPY logits、manifest、source snapshot、session logs/environment、B reference、legacy before/after、validation、analysis、seal。成功 seal 只是工件完整性/软件门通过，成本结论需设计侧读实际分布，不能保证优势。

**CRLF 警告**：不要让 Windows git checkout 转写工件；使用原 tarball 或 `git -c core.autocrlf=false clone`。校验工具复读既有 B 工件，若普通 Windows checkout 发生 CRLF，字节身份会不同；在 LF checkout/原归档环境验证，不把转换误报解释成实验缺陷。

## 5. 本机实际测试

实际 PASS：R/B smoke（随机 tiny CPU Qwen，不载入 pretrained 模型，含实际 execute 两臂、crop/role/consumer commit、exclusive write/seal tamper）、独立 B 100 trajectories/800 rows/1118 cursors/27 C2 closures、`src.dialogue.run_timeline_test`、`src.tts.run_chunker_test`、`experiments.sci34_supplement.c2_equivalence.smoke`、compileall、CLI help、错误 source commit 的 formal 拒绝门、git diff --check。

未运行：7B pilot/formal、真实 GPU 数值/显存/时长、完整 7B session→validate→analyze→seal 集成。远程硬件、缓存、驱动未核验。生产 src 无修改，run_kvcrop_test/run_speculative_test 的真实模型/GPU 回归不伪称通过。工程 pilot 是正式前必需验证，不得跳过。
