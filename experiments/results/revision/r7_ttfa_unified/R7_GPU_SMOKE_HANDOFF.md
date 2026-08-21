# R7 统一 TTFA：GPU 主机冒烟阶段交接文档（W1 Gate 1 后）

- 日期：2026-08-21
- 对应审查：`experiments/review/20260821-PRE-PAPER-AUDIT/review-implementation-r2-20260821.md`
  （**已放行**：TTS 独立探活 + 3 条分层 GPU 冒烟；**未放行**：正式 50×A/B）
- 代码基线：`git pull` 至 `1d81cf1` 或更新
- 脚本：`experiments/scripts/run_ttfa_unified.py`（本机 self-test 69 PASS / 0 FAIL）

## 0. 前置

```bash
cd /dataA/streamllm   # GPU 主机仓库根
git pull
uv sync
# 启动 TTS 服务（与 E6 相同：CosyVoice 流式 PCM，127.0.0.1:20401，spk 晓伊，speed 0.8）
# 并把 TTS 服务的 commit/模型 revision/启动参数记入 env 记录（见 §5）
```

## 1. 确定固定 Silero 来源（P0-4 要求，缺一不可启动）

优先用**本地缓存仓库目录**（离线安全）：

```bash
ls ~/.cache/torch/hub/ | grep silero          # 通常是 snakers4_silero-vad_master
cd ~/.cache/torch/hub/snakers4_silero-vad_master && git rev-parse HEAD && git status --porcelain | head -3
```

记下 commit（如 `3245b1a`）。后续命令统一用：

```
--silero-dir ~/.cache/torch/hub/snakers4_silero-vad_master
```

脚本会自动：哈希该仓库内的模型 artifact（.jit）、记录 repo commit/dirty、并把它注入正式
`StreamAudioSegmenter`（与 PSE 同一实例，RUNINFO 有断言）。
若仓库内找不到 >100KB 的模型 artifact，改用 `--silero-ref <commit>`（需网络）并反馈。

## 2. 任务 1：TTS 独立探活（已放行）

```bash
uv run python -m experiments.scripts.run_ttfa_unified \
    --tts-probe --tts-url http://127.0.0.1:20401 --tts-spk 晓伊 --tts-speed 0.8 \
    --output-dir experiments/results/revision/r7_ttfa_unified
```

- 期望：`"ok": true`、`payload_class: "pcm"`；产物 `r7_ttfa_unified/tts_probe.json` 随冒烟一并提交；
- **探活失败不得临时放宽允许策略**（allow_content_type/encoding 已由探活固定，正式请求逐项比对）。

## 3. 任务 2：self-test（期望 69 PASS / 0 FAIL）

```bash
uv run python -m experiments.scripts.run_ttfa_unified --self-test
```

## 4. 任务 3：3 条分层 GPU 冒烟（成功路径 + 故障注入）

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
    --smoke 3 --inject-fault asr_error \
    --output-dir experiments/results/revision/r7_ttfa_unified \
    --run-id r7_smoke
```

- `--smoke 3` 自动分层选样（2 zh + 1 en），共 6 个任务（3 样本 × A/B）；脚本内置校验：
  精确命中 3 样本、双语覆盖、任务数=6，不满足直接退出；
- `--inject-fault asr_error` 仅在冒烟模式合法：对**最后一个任务**注入 ASR 异常，验证
  error 落盘与 fail-stop（后续若有剩余任务会补 cancelled；本例注入任务即最后一条）；
- 预计时长：模型加载 3–5 分钟 + 6 任务（每任务含实时回放+生成+TTS 全程读毕，约 1–4 分钟）
  ≈ 20–40 分钟；checkpoint 支持断点（同 run_id 重跑会跳过已终态任务）。

### 冒烟验收清单（审查 §3.3/§5，逐项确认后随产物反馈）

1. `QA_r7_smoke.md`：QA 问题数 = 0；记录数 = 6；
2. 成功 ≥1 条且 A/B 两模式都有成功记录（正常应为 5 成功 + 1 error）；
3. 注入任务终态 = error 且 error 含 `fault_injection`；
4. **无** `final_drain_empty` / `thread_leak` / `pair_timeout` / `validate:` 类错误
   （任一出现即停止，不进入正式实验，反馈现场）；
5. `RUNINFO_r7_smoke.md`：`segmenter_meta.segmenter_silero_injected=true` 且
   `silero_meta.artifact_sha256` 与其一致（PSE 与分段器同一 artifact）；
6. 成功记录的 `tts` 字段无 error（header/payload 策略与探活一致）；
7. 成功记录 TTFA_playable 非负、量级合理（B 首句就绪即 TTS，应在数秒级；A 等全文，
   视回复长度 15–40s 级）；
8. 慢流证据：若出现任何 TTS 错误，保留原始记录（不许删改）；如见 thread_leak 立即停止。

## 5. W2 环境记录（顺带收集，供正式实验 RUNINFO 引用）

```bash
mkdir -p experiments/results/revision/r7_ttfa_unified/env
{lscpu; echo ---; uname -a; echo ---; free -h; echo ---; nvidia-smi; \
 cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null;} \
 > experiments/results/revision/r7_ttfa_unified/env/cpu_gpu.txt
uv pip freeze > experiments/results/revision/r7_ttfa_unified/env/pip_freeze.txt
# TTS 服务 commit/模型 revision/启动命令 → env/tts_service.txt（无法取到的写 unknown）
```

## 6. 提交与反馈

- 产物目录：`experiments/results/revision/r7_ttfa_unified/`
  （checkpoint_r7_smoke.jsonl、tts_probe.json、RUNINFO/QA/summary、env/）；
- 提交 push 后在本机登记；**冒烟结果需再过一次结果级复核，通过后才放行正式
  50×2 + 子集补轮 + 匹配文本控制**（handoff 将另行给出）；
- 禁止事项：跳过冒烟直接跑正式；修改任何允许策略/时间戳字段；删除非成功记录。

## 7. 已知边界（审查 §3.1 登记）

- TTS total deadline 在"headers 已到、body 长时间不发"场景最终仍由动态收紧的 read
  timeout / pair deadline 兜底（内部 total deadline 只在 chunk 间检查）——冒烟如遇此
  场景，按 §4.8 保存证据并反馈，不得静默重试。
