# 开发侧回复：§2b 与 checkpoint 目录守卫冲突的定夺（对应 R7_FORMAL_GATE_CHECKPOINT_DIR_CONFLICT_HANDOFF.md）

- 日期：2026-08-22
- **定夺：方案 A**（与现场倾向一致），且不止 §2b——**三个 run 全部改独立子目录**。
- 代码守卫**零改动**（"一目录一 checkpoint"是审查已确认通过的 fail-closed 行为，
  放宽为方案 B 需重新过审且无必要）。

## 1. 定夺理由

现场根因分析正确：handoff 把 §2/§2b/§3 三个 run 指向同一目录，与守卫冲突；
且正如现场指出，**不只 §2b——r7_main 与 r7_tts_control 未来也会撞同一守卫**
（主目录已有已提交的 checkpoint_r7_smoke.jsonl）。因此一次改到位：

| run | 新输出目录 |
|---|---|
| §2b `r7_smoke_fatal` | `r7_ttfa_unified/fatal_smoke/` |
| §2 `r7_main` | `r7_ttfa_unified/r7_main/` |
| §3 `r7_tts_control` | `r7_ttfa_unified/tts_control/`（`--control-from` 同步指向 `r7_main/` 子目录） |

既有产物（r7_smoke 三件套、env/、selftest_archive/）留在主目录不动；
**G1/G2/G7/G8/新探活/§2c 已采集产物全部有效，无需重跑**。

## 2. handoff 已更新（本次 push）

- §0c 目录安排说明（守卫语义不变 + 三子目录映射）；
- §0b 材料包第 3 项路径改 `fatal_smoke/checkpoint_r7_smoke_fatal.jsonl`；
- §0b manifest 命令 sha256sum 列表同步（`fatal_smoke/…`）；
- §2/§2b/§3 命令 output-dir 与 §5 产物说明同步。

## 3. GPU 主机后续（仅两步）

1. **重跑 §2b**（新目录 `fatal_smoke/`，命令见 handoff §2b）：验收 = task0 success /
   task1 error 含 `fault_injection` 且 fatal=True / task2–5 全部
   `cancelled_after_fatal` / QA 记录数 6；另确认 cancelled 记录无 ASR/LLM/TTS 事件；
2. **重新生成 GATE_MANIFEST**（§0b 命令，路径已更新）→ commit + push。

材料包齐备后本机核验，提交最终书面放行复核。

## 4. 现场成果确认

- G7 的关键发现收妥并登记：`spk2info.pt` 中"晓伊"与"中文女"embedding **完全相等**
  （max abs diff=0.0，shape (1,192)）——speaker 映射注记自此有了模型级证据
  （`1095c8d8…` hash 已绑定）；
- GPU clean 树 90 PASS、platform_conditions、新探活（300/300 稳定）均已入库核验通过。
