# R7 放行前 Gate 阻塞：§2b fatal 小 smoke 与「checkpoint 不得复用目录」守卫冲突（请开发侧定夺）

> **读者**：开发/审查侧。GPU 主机已按 `R7_FORMAL_RUN_HANDOFF.md` 执行放行前 Gate。
> G1/G2/G7/G8/新探活/§2c 均已采集完成；**§2b 非末位 fatal 小 smoke 被脚本 fail-closed 拦停**。
> **状态**：待开发侧定夺（改 handoff 输出目录 或 改脚本守卫语义）。**未绕过守卫、未改代码。**
> **关联**：`R7_FORMAL_RUN_HANDOFF.md` §0b/§1/§2b、
> `experiments/review/20260821-PRE-PAPER-AUDIT/review-reply-final-gate-20260822.md` §3.4/§5.3。

---

## 1. 现象（GPU 主机实测，§2b 冒烟，exit 0 但 fail-closed）

`--run-id r7_smoke_fatal` 命令在「PSE 预扫描 + 模型加载」完成后、写入 checkpoint 前被拦停：

```
目录中存在其他 run 的 checkpoint: experiments/results/revision/r7_ttfa_unified/checkpoint_r7_smoke.jsonl（不得复用目录，停止）
```

`checkpoint_r7_smoke_fatal.jsonl` **未生成**，无 RUNINFO/QA。

## 2. 根因（handoff 输出目录与脚本守卫直接冲突，非环境问题）

`R7_FORMAL_RUN_HANDOFF.md` §2b 把三个 run 都指向**同一个输出目录**
`experiments/results/revision/r7_ttfa_unified`：

- §2 `r7_main`（`--run-id r7_main`）
- §2b `r7_smoke_fatal`（`--run-id r7_smoke_fatal`）
- §3 `r7_tts_control`（`--run-id r7_tts_control`）

但脚本 `Checkpoint.__init__`（`run_ttfa_unified.py:1196-1199`）有硬守卫：

```python
# 同目录其他 run 的 checkpoint 混入检查
for other in sorted(path.parent.glob("checkpoint_*.jsonl")):
    if other != path:
        raise SystemExit(f"目录中存在其他 run 的 checkpoint: {other}（不得复用目录，停止）")
```

即**同一目录只能存在一个 `checkpoint_*.jsonl`**。§2b 跑的时候，目录里已有上次 happy-path 冒烟的
`checkpoint_r7_smoke.jsonl`（已提交、属于正式产物），于是任何新 run 都会被拒。

`--no-resume` 不能解决：它只在 `ck_path.exists()` 时 unlink **本 run 自己的** checkpoint
（`:2763-2766`），不清理「其他 run」的 checkpoint；且清掉 `r7_smoke` 会破坏已提交的冒烟产物，不可行。

该守卫源自审查要求「checkpoint 损坏/截断/hash 不匹配必须退出且不得复用目录」
（`review-implementation-v3.1-20260821.md:145`）——守卫本身是对的，但 handoff §2/§2b/§3 的
「同目录多 run」安排与它矛盾。**这意味着 §2 r7_main 与 §3 r7_tts_control 将来也会撞同一个守卫**
（三者同 output-dir），不是只有 §2b 受影响。

## 3. 已完成的放行前 Gate 产物（未受此阻塞影响）

| 步骤 | 状态 | 产物 |
|---|---|---|
| G1/G2 | ✅ | `env/gate/gate_clean_git.txt`（HEAD=2e54ac2、porcelain 空） |
| G7 | ✅ | `env/gate/tts_provenance/`：`TTS_PROVENANCE.md` + `server_dirty.patch`(163行) + `tts_pip_freeze.txt` |
| G8 | ✅ | `env/platform_conditions.txt`（双 3090/驱动 550.127.05/CUDA 12.4/Triton fallback×4/显存分配/nvidia-smi 快照） |
| 新探活 | ✅ | `env/gate/tts_probe_new.json`（ok=true/pcm） |
| §2c | ✅ | `selftest_archive/selftest_gpu_20260822.log` + `.md`（**90 PASS / 0 FAIL，exit 0**） |
| §2b | ❌ | 被守卫拦停（见上） |

关键结论已核验并写入 G7：**`spk2info.pt` 中 `晓伊` embedding 与 `中文女` 完全相等（max abs diff=0.0，shape (1,192)）**，即「晓伊→内置中文女」的模型级证据；`spk2info.pt` sha256 `1095c8d8…`。

## 4. 建议（供开发侧定夺，勿由 GPU 主机代改）

二选一（倾向方案 A，改动最小、语义清晰）：

- **方案 A（改 handoff 输出目录）**：§2b 用独立子目录，如
  `--output-dir experiments/results/revision/r7_ttfa_unified/fatal_smoke`（或 `…/r7_smoke_fatal/`）。
  §2 r7_main 与 §3 r7_tts_control 若继续共用主目录，同样需评估：要么各自独立子目录，要么
  脚本守卫放宽为「同目录允许多个不同 run_id 的 checkpoint，但禁止同 run_id 混入」。
  材料包 §5.3 仍引用 `checkpoint_r7_smoke_fatal.jsonl` 路径，需同步更新。
- **方案 B（改脚本守卫语义）**：把「不得复用目录」从「目录级一 checkpoint」放宽为
  「同 run_id 的 checkpoint 互斥」——即允许 `checkpoint_r7_smoke.jsonl` 与
  `checkpoint_r7_smoke_fatal.jsonl` 共存（run_id 不同、binding 不同），仅拒绝
  同 run_id 重复/混入。需确认这不违背审查「不得复用目录」的原始意图，并补自测用例。

无论哪种方案，请同步更新 §0b manifest 命令里的文件路径，使 `sha256sum` 列表与实际产物一致。

## 5. 现场与基线

- 仓库：`/dataA/streamllm`，HEAD = `2e54ac297d51076f10c2445c9485b107544cf16f`。
- `run_ttfa_unified.py` sha256 = `2c1c1ac605280f0207cca464118a9642656026c2bd306cefed230e00cd6a26ce`。
- 自测：**90 PASS / 0 FAIL**（clean 树，exit 0，已归档）。
- TTS 服务：CosyVoice v2.0，commit `8555549e`（dirty，server.py/requirements），image digest
  `sha256:1c0974a4…`；`/inference_sft` 裸 PCM 无 Content-Type；探活 ok=true/pcm（300/300 稳定，前一轮已验）。
- Silero：artifact sha256 `e1122837…d3720`（`repo_commit=None` 属预期）。
- 未改代码、未绕过守卫、未提交任何改动；§2b 未产生 checkpoint/RUNINFO/QA。

## 6. 定夺后

开发侧修正 handoff（或脚本）并推送后，GPU 主机重跑 §2b 小 smoke（期望：task0 success /
task1 error 含 `fault_injection` 且 fatal / task2–5 cancelled_after_fatal / QA 记录数 6），
再按 §0b 打包 Gate manifest 提交最终放行复核。
