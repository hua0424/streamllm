# SCI 3–4 区补实验实施计划

## 一、范围与取舍

本轮只实施以下三类实验，全部放入新目录 `experiments/sci34_supplement/`：

1. **固定生成轨迹 E3（P0，必须）**：消除 playback/generation 两条件首轮回复不同的混杂。
2. **规范化 A1 联合计时（P0，必须）**：保存原始重复，直接测量联合 `crop + role recovery`，报告 median/IQR。
3. **无人工评测的 P1 简化异步播放微基准**：使用 headless、wall-clock paced sample player；测停播确认、反查、KV crop 和角色恢复联合路径，不建设完整生产式音频闭环。

明确不做：

- 人工盲法双标；
- A2 重跑；A2 后续从主贡献中删弱或移入局限/附录；
- E1 全量重跑；
- E2 全九点多 seed；
- 完整 ASR→并发 LLM→可取消 CosyVoice→真实声卡的生产式闭环；
- B-syn、E4、中文/CrossWOZ、多主模型复验。

新实验统一采用固定 seed 或 greedy、运行 manifest、配置哈希和独立结果目录。现有未提交的论文修订和原始 GPU JSON 不覆盖、不回退。

## 二、目录与文件设计

新建：

```text
experiments/sci34_supplement/
├── __init__.py
├── README.md                    # 目录说明、快速命令、实验边界
├── EXPERIMENT_PLAN.md           # RQ、假设、变量、样本量、统计和允许主张
├── GPU_RUNBOOK.md               # GPU 主机从签出到打包结果的逐步操作
├── CLAIMS_MATRIX.md             # 做/不做哪些实验分别允许论文声称什么
├── common.py                    # 原子写盘、SHA-256、稳定 seed、manifest、分位数
├── model_runtime.py             # 正式 greedy chat runtime + 无模型 fake runtime
├── e3_fixed_trajectory.py       # 固定轨迹 E3 主实验，支持断点续传
├── e3_judge.py                  # 复用现有 Mistral Judge，保留 manifest/来源哈希
├── analyze_e3.py                # 配对统计、dialogue bootstrap、Wilson CI
├── a1_joint_latency.py          # joint crop+role 原始计时和 re-prefill 对照
├── paced_player.py              # 无声卡 wall-clock sample consumer
├── async_bargein.py             # P1 停播→反查→crop→role 联合微基准
├── analyze_latency.py           # A1/P1 median、IQR、P90/P95、原始值检查
├── smoke.py                     # 纯 Python、无模型、无网络 smoke
├── run_all_gpu.sh               # GPU 正式运行编排，不含人工步骤
├── fixtures/
│   └── mini_dialogues.json      # 仅 smoke；正式模式禁止 fixture 回退
└── results/
    └── .gitkeep
```

结果按 run ID 隔离：

```text
results/<run_id>/
├── manifest.json
├── records.jsonl
├── summary.json
├── progress.json
└── run.log
```

`manifest.json` 至少记录：git commit/dirty、CLI、模型名与 revision、输入 SHA-256、样本 ID、seed/解码参数、Python/PyTorch/Transformers/CUDA、GPU、`uv.lock` hash、TTS profile、配置哈希和运行时间。恢复运行时若配置哈希不一致立即拒绝，避免再次混入 fixture 或旧 schema。

## 三、固定轨迹 E3

### 设计

每条 MultiWOZ 对话只生成一次首轮 assistant 轨迹：

- 完整 generated token IDs；
- 每 token 解码文本；
- TTS 文本片段及 token span；
- Mock TTS 的 fragment/chunk/sample span；
- 完整 assistant 文本；
- trajectory hash。

随后从同一轨迹离线派生四个注入位置：25%、50%、75%、clean boundary；每个位置派生：

- `playback`：历史只保留命中片段末端前缀；
- `generation`：历史保留完整首轮轨迹；
- fragment-level unheard；
- character-proportional, whitespace-snapped proxy tail。

probe 生成使用新建标准 chat history：

```text
system → user turn 1 → fixed assistant history → probe 1 → reply 1 → probe 2
```

正式运行使用 greedy 解码，避免 seed 噪声。E3 的语义实验不依赖同一 KV 对象分叉：固定 token 轨迹保证处理前回复一致；各历史重新 prefill 后生成 probe。KV crop 的状态与性能由现有机制测试和新 A1 验证。因此不深改 `_finish_assistant()`，也不要求复制 `DynamicCache`。

### 计算去重

- generation 历史在四个 fraction 下相同，probe 链只生成一次；
- playback 若多个 fraction 命中同一 fragment endpoint，probe 链只生成一次；
- 规则检测、timeline 反查、proxy tail、统计全在 CPU 离线完成。

正式规模仍为 100 对话、4 位置、2 条件，但预计主 LLM decode 从旧脚本约 2400 次降至约 900 次。

### 输出与统计

保持兼容字段：`unheard_text`、`strict_unheard_text`（文档说明它实际是 proxy）、`probe_replies`、`referenced_unheard*`，并新增：

- `trajectory_id`；
- `assistant_token_ids` 或其单独轨迹表引用；
- `history_key`；
- `shared_trajectory=true`；
- decode config 与 seed。

统计：

- B-gen 失败率和 Wilson 95% CI；
- exact McNemar；
- 以 dialogue 为单元的 10,000 次 cluster bootstrap；
- 分位置结果；
- 不再带“独立生成混杂”限制，改为固定轨迹的配对解释；
- B-ours fragment-level zero 仍标为构造性结果。

### 断点与验收

- 正式运行必须显式传 `--dialogues`；缺失时直接退出，绝不自动用 fixture；
- resume key 含 trajectory hash、history key、probe index 和配置哈希；
- 断言同一 dialogue 的所有条件共享 trajectory ID；
- 断言 playback history token IDs 是完整轨迹前缀；
- 断言 generation 四位置历史一致；
- 正式数据中禁止任何 `id` 以 `fx` 开头。

## 四、规范化 A1

### 实现

基于现有 `StreamLLMInference`、`crop_to_token()`、`reopen_user_role()` 和 CUDA 同步计时，默认：

- 六个上下文长度：256、512、1024、2048、4096、8192（实际长度随模板略偏移并落盘）；
- warmup 3 次；
- 每点正式重复 20 次；
- 可通过 `--repeats 50` 做更稳健版本。

每次保存四组原始数组：

1. `crop_only_ms`；
2. `role_recovery_only_ms`；
3. **同一个计时区间内执行的 `crop_role_joint_ms`**；
4. `reprefill_ms`。

每组计算 median、Q1、Q3、IQR、P90、min、max。主加速比定义为：

```text
median(reprefill raw) / median(joint crop+role raw)
```

不再使用 `median(crop)+median(role)` 代替联合路径中位数。每个长度完成后原子保存，支持中断恢复。

### 验收

- raw 数组长度等于 repeats；
- joint 计时包含一次 crop 和一次 role recovery，前后 CUDA synchronize；
- re-prefill 的输入 token 数与报告的 keep length 一致；
- 报告 IQR，不做无意义的显著性检验；
- 论文仅称“模型侧联合恢复微基准”，不称完整 barge-in latency。

## 五、P1 简化异步播放微基准

### 选择的实现

采用 **headless wall-clock paced sample player**，不依赖声卡、PortAudio 或 sounddevice，保证 GPU 服务器也能运行：

- 预先将 fragment 的 sample spans 注册到 `PlaybackTimeline`；
- 后台线程以 `time.perf_counter()` 和 sample rate 推进 `played_samples`；
- 到目标播放比例时发出 stop request；
- player 立即停止推进、返回 stop acknowledgment 和最终 sample；
- 主线程随后执行 timeline lookup、实际 GPU crop 和 role recovery。

可读取：

- 默认 `TimingProfile`/现有 `cosyvoice_profile.json` 生成的 sample durations；
- 可选外部 audio manifest（真实 CosyVoice 缓存片段的 sample counts）。

音频波形内容不参与测量；本实验名为“异步墙钟播放路径微基准”，不宣称真实声卡、硬件 buffer 或在线 TTS 取消。

### 设计

默认：

- 上下文长度：512、2048、8192；
- 注入位置：25%、50%、75%；
- 每组合 20 次；
- 共 180 个事件；
- smoke 模式使用短时长和 fake KV callbacks，数秒完成；
- GPU 模式使用真实 7B `crop_to_token`/`reopen_user_role`，状态准备不计入被测区间。

逐事件保存：

- target samples；
- stop request、stop ack、lookup done、crop done、role done 时间戳；
- stop ack latency；
- stop 后泄漏 samples/ms；
- lookup latency；
- crop-only latency；
- role recovery latency；
- stop→crop done；
- stop→role done；
- 实际 crop token end；
- 上下文长度、fraction、repeat。

### 允许与禁止主张

允许：

- 在 headless wall-clock paced playback 下，停播控制、反查和 GPU 状态修正的联合延迟分布；
- crop 与 role recovery 在异步控制路径中的模型侧开销；
- 播放停止后 sample counter 的泄漏量。

禁止：

- 真实声卡最终用户听觉延迟；
- 在线 CosyVoice 推理取消；
- ASR/LLM/TTS 真并发竞争；
- 生产级完整 barge-in latency。

## 六、本机验证策略（不下载大模型）

实施后在本机执行：

```bash
uv run python -m py_compile experiments/sci34_supplement/*.py
uv run python -m experiments.sci34_supplement.smoke
uv run python -m src.dialogue.run_timeline_test
```

`smoke.py` 使用 fake tokenizer/runtime/KV callbacks，验证：

- 固定轨迹只生成一次；
- 条件共享 trajectory ID；
- history 去重和断点恢复；
- fixture 禁止进入 formal 模式；
- manifest/config hash；
- A1 raw 统计聚合；
- paced player 的 stop/ack/泄漏 samples；
- 结果 schema 与原子写盘。

不会加载 Hugging Face 模型，不访问网络，不下载权重。若本机已有小模型，可由 runbook 给出可选 0.5B integration 命令，但不作为验收前提。

## 七、GPU 主机 runbook

`GPU_RUNBOOK.md` 将写成可复制执行的步骤：

1. 签出指定 commit/branch，确认工作树；
2. `uv sync`，检查 CUDA、两张 3090 和模型路径；
3. 设置 `.env`/HF_HOME，禁止临时下载；
4. 运行无模型 smoke；
5. 校验正式 MultiWOZ 文件、样本数和 SHA-256；
6. 先用 3 条正式数据做 E3 integration；
7. 跑 100 条固定轨迹 E3；
8. 运行 Mistral judge；
9. 运行 E3 分析并检查验收断言；
10. 跑 A1 六长度×20 repeats；
11. 跑异步墙钟播放 3 长度×3 位置×20 repeats；
12. 运行延迟分析；
13. 将 run ID、manifest、records、summary、日志和校验和打包；
14. 失败恢复、OOM 降级和只重跑缺失单元的命令。

`run_all_gpu.sh` 只编排上述无人工步骤，并要求显式提供：

- 正式 dialogues 路径；
- Qwen2-7B-Instruct 路径；
- Mistral judge 路径；
- 输出根目录；
- GPU/device 配置。

## 八、完成标准

- 所有新代码和文档只新增在 `experiments/sci34_supplement/`，不修改或覆盖旧实验结果；
- 无模型 smoke 和 `py_compile` 通过；
- E3 formal 模式不会使用 fixture，具备配置哈希断点保护；
- A1 保存联合 raw timing 和 median/IQR；
- P1 在无声卡 GPU 服务器可执行并明确证据边界；
- GPU runbook 从干净签出可逐步执行；
- 人工评测步骤完全不出现；
- 不提交、不推送，保留当前工作区已有修改。