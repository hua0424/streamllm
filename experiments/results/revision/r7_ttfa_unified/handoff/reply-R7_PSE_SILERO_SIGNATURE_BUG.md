# 开发侧回复：PSE Silero 调用签名错配（对应 handoff/R7_PSE_SILERO_SIGNATURE_BUG_HANDOFF.md）

- 日期：2026-08-21
- 结论：现场诊断**完全属实，已修复并推送**。GPU 主机请按 `R7_GPU_SMOKE_HANDOFF_R3.md`
  重跑（探活已过可跳过；self-test 期望值更新为 **76 PASS**）。

## 1. 核实与修复

现场根因定位正确：`silero_pse_sample()` 漏传真实签名（`utils_vad.py`）的必填位置参数
`model`，异常被 `analyze_pse` 吞进 `silero_error` → 单算法失败 → fail-closed 拦停。
对照分段器 `streamaudio_segmenter.py` 的正确调用（`get_speech_timestamps(audio, model, …)`）
确认属实。已修复：

1. `silero_pse_sample(wave, model, get_speech_timestamps)`：model 位置参数透传；
2. `analyze_pse(wav_path, model, get_speech_timestamps)`：提供了函数但 model 为 None →
   **显式拒止** `pse_missing_model`（采纳现场建议 2，不再静默落进单算法失败靠日志反推）；
3. `main()` PSE 预扫描传 `silero_model`；本地集成测试同步修复（此前其能量法兜底
   **掩盖了该契约错误**——现已加防护：`silero_error` 含 "missing" 或缺 model 时直接失败，
   不允许能量法兜底掩盖契约类错误）。

## 2. 自测缺口补齐（采纳现场建议 1）

- self-test 的 `fake_silero` 改为**签名严格**：`_call(audio, model, sampling_rate=…)`——
  model 为必填位置参数，调用方漏传立即 TypeError 断言失败（本类错配不再可能漏过自测）；
- 新增用例：「PSE 缺 model 显式拒止」（`pse_missing_model`）；
- self-test **76 PASS / 0 FAIL**；本机真实组件集成检查复跑 ALL PASS（真实 Silero 路径
  本次实际跑通，噪声输入下无语音属预期、能量法注入仅限该场景）。

## 3. 现场产物已接收核验

- `tts_probe.json`：ok=true / pcm / policy_note 在——**任务 1 通过，无需重跑**；
- 任务 2 已过（75/75），修复后请复跑一次（期望 76）；
- `env/cpu_gpu.txt`、`env/pip_freeze.txt` 已收妥（scaling_governor 缺失属预期，已知）；
- TTS 服务 commit `8555549e`（含本地修改）已登记，正式 RUNINFO 将引用。

## 4. 后续

按 `R7_GPU_SMOKE_HANDOFF_R3.md`：任务 2 复跑（期望 76 PASS）→ 任务 3 三层冒烟
（命令不变）→ 八项验收 → 产物 push。冒烟通过后本机做结果级核验，再申请放行正式实验。
