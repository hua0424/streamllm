# R7 冒烟阻塞：PSE 预扫描 Silero 调用签名错配（请开发侧修复）

> **读者**：开发/审查侧。GPU 主机已按 `R7_GPU_SMOKE_HANDOFF_R2.md` 执行，
> 探活（任务 1）与 self-test（任务 2）均通过，任务 3 冒烟在 **PSE 预扫描阶段 fail-closed 退出**。
> **状态**：待开发侧修复后重跑。**未改任何代码、未绕过 fail-closed、未进入任何任务执行。**
> **关联**：`experiments/results/revision/r7_ttfa_unified/R7_GPU_SMOKE_HANDOFF_R2.md`、
> `handoff/reply-R7_TTS_CLIENT_CONTRACT_BUG.md`。

---

## 1. 现象（GPU 主机实测，任务 3 冒烟 exit 0 但被 fail-closed 拦停）

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

run.log 末尾：

```
PSE 失败 crosswoz_7701_turn3: pse_single_algorithm_failure（停止）
```

进程 exit 0（fail-closed 走了 `SystemExit` 前无显式非零），但**没有生成
checkpoint_r7_smoke.jsonl / RUNINFO / QA / summary**，一个任务都没跑。

## 2. 根因（已定位，非数据问题）

`run_ttfa_unified.py:210-214` 的 `silero_pse_sample()` 调用 Silero 的
`get_speech_timestamps` 时**漏传了 `model` 位置参数**：

```python
ts = get_speech_timestamps(torch.from_numpy(wave), sampling_rate=ANALYSIS_SR,
                           **SILERO_PARAMS)
```

但本机缓存仓库（`~/.cache/torch/hub/snakers4_silero-vad_master`）的签名是：

```python
def get_speech_timestamps(audio: torch.Tensor,
                          model,          # ← 必填位置参数
                          threshold=0.5, sampling_rate=16000, ...)
```

（`src/silero_vad/utils_vad.py:212`）。`model` 无默认值，漏传即抛
`TypeError: get_speech_timestamps() missing 1 required positional argument: 'model'`。

该异常被 `analyze_pse()`（`:238-241`）吞进 `silero_error`，`s` 记为 `None`；
而能量法 `e = 488920`（非 None）。按协议 §3.1.7「单算法失败 fail-closed」，
`e is not None and s is None` 命中 `:247-250`，返回 `pse_single_algorithm_failure`。

**已用现场音频实测闭环**（`crosswoz_7701_turn3.wav`，30.56s zh）：

```python
# 复现（脚本现行调用，漏 model）：
analyze_pse(wav, get_speech_timestamps)  # -> error=pse_single_algorithm_failure,
                                         #    silero_error="... missing ... 'model'"
# 修复验证（补 model 位置参数）：
get_speech_timestamps(torch.from_numpy(wave), silero_model,
                      sampling_rate=16000, **SILERO_PARAMS)
# -> n_segments=14, last_end=488920（与能量法一致）
```

能量法 e=488920 = 全文件末尾（30.56s×16000），与 Silero 补 model 后的
last_end=488920 一致，说明**音频本身正常、有语音**，纯属调用签名错配。

## 3. 为什么 self-test 75 项没拦住

self-test 的 Silero 是注入的假函数（`:1611-1612`）：

```python
def fake_silero(val):
    return lambda wave, sampling_rate=16000, **kw: [{"end": val}] if val else []
```

这个假实现**不要求 `model` 参数**，所以 `analyze_pse` 的漏参路径在自测里是通的；
真实仓库的 `get_speech_timestamps` 才需要 `model`。契约假 TTS 服务已按真实 FastAPI 对齐，
但 Silero 假实现未按真实签名对齐——这正是本次缺口。

对比：`src/asr/streamaudio_segmenter.py:209-217` 的分段器调用是**正确**的
（`self.get_speech_timestamps(audio_tensor, self.model, sampling_rate=..., ...)`），
模型注入分段器后的 `process_audio` 路径没问题；出问题的只有 `silero_pse_sample()` 这一处。

## 4. 修复建议（供开发侧参考，勿由 GPU 主机代改）

最小改动：`silero_pse_sample` 增加 `model` 参数并透传。

```python
def silero_pse_sample(wave, model, get_speech_timestamps) -> int | None:
    ts = get_speech_timestamps(torch.from_numpy(wave), model,
                               sampling_rate=ANALYSIS_SR, **SILERO_PARAMS)
```

调用点 `analyze_pse()`（`:238`）与签名（`:220`）同步加 `model` 参数；`main()` 里
`analyze_pse(s["audio_path"], silero_model, get_speech_timestamps)`（`:2456`）一并传参。

同时建议（回归防护）：
1. self-test 的 `fake_silero` 改为**签名严格**（要求 `model` 位置参数），或新增一条
   「真实签名调用不漏 model」的断言，避免同类错配再次漏过；
2. 可选：给 `silero_pse_sample`/`analyze_pse` 增加「传入 model 为空/未传」的显式拒止，
   而不是静默落进 fail-closed 再靠 run.log 反推。

## 5. 现场与基线

- 仓库：`/dataA/streamllm`，HEAD = `ba0ec8edc32085e604fe3aab31e66e71ec8224f5`。
- `run_ttfa_unified.py` sha256 = `052fd6ecc653a0dc8cdcd5b410336fc52dadcd4df9a0a6c3b73ffb31f7cd7a9c`。
- **任务 1（探活）已通过**：`ok: true`、`payload_class: "pcm"`、status 200、`policy_note` 存在；
  `tts_probe.json` 已落盘。
- **任务 2（self-test）已通过**：**75 PASS / 0 FAIL**（exit 0）。
- 任务 3 冒烟：未产生 checkpoint/RUNINFO/QA/summary（PSE 预扫描阶段 fail-closed）。
- 冒烟命中的样本与分层选样校验**未被执行到**（PSE 预扫描在建 schedule 之后、加载模型之前）。
- TTS 服务：CosyVoice v2.0，commit `8555549e`（`server.py`/`requirements.txt` 有本地修改），
  `127.0.0.1:20401` 正常。
- W2 环境记录已采集：`env/cpu_gpu.txt`、`env/pip_freeze.txt`（项目 `.venv` 3.10.18，
  torch 2.5.1+cu121 / faster-whisper 1.2.0 / librosa 0.11.0）。`scaling_governor`
  文件不存在（此主机无该 sysfs 节点），属预期。

## 6. 修复完成后

开发侧修正并推送后，GPU 主机重跑顺序不变：任务 1 探活（期望 ok:true/pcm）→
任务 2 self-test（期望 75 PASS）→ 任务 3 三层冒烟（八项验收，见 R2 handoff §2）。
