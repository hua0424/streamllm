# RUNINFO — r6_ttfa 目录（E5 端点 / E6 TTS 首包 / TTFA 解码至首句补测）

> 本文件登记 `r6_ttfa/` 下各次测量的执行信息。E5/E6 的 RUNINFO 见各自产物同名 `.runinfo.md`；
> 本节为 2026-08-21 TTFA 预算缺项补测（`T_decode_to_first_sentence`）的执行登记与跑后 QA。
> 执行依据：`../r3_baseline_la/handoff/E6_TTFA_DECODE_MEASURE_RUN_HANDOFF.md`（评审 r3 放行）。

## 2026-08-21 TTFA 解码至首句补测（R6 补测，意见2 / 计划 §7.3）

- **任务**：补测 TTFA 预算表缺项 `T_decode_to_first_sentence`（首 token → 首个句末标点 token 的 LLM 解码延迟）。
- **代码**：git commit `dd2e6e0`（≥ 评审要求的 `1efc809`，含 r2 P1 修复）；`--self-test` 26 PASS / 0 FAIL，exit 0。
- **冒烟**（步骤 1）：`--max-samples 3`，3/3 完成、error 0、exit 0；`fragment_count`/`fragments_sha256`/`prompt_tokens`
  非空；`decode_to_first_sentence_smoke.runinfo.md` 字段齐全（重放模式 `fragment_replay`）；
  逐条值 98.0/262.3/49.2ms（正常量级）。四项冒烟验收全过。
- **正式命令**（步骤 2）：
  `uv run python -m experiments.scripts.measure_decode_to_first_sentence --llm-device cuda:1 --output experiments/results/revision/r6_ttfa/decode_to_first_sentence.csv`
  （`2>&1 | tee .../decode_to_first_sentence_run.log`）
- **起止**：2026-08-21 10:58:56 → 11:04:07（约 5.2 分钟，远短于预估 0.5 GPU 小时）。
- **输入**：`r4_commit/exp1_results_20260820_171522.json`（E4 复跑产物），
  sha256 `32dc6f5b6889bf9984623653f6a5b2aa497a06d15ff6b3c14511c22b2f8d7277`（与 `decode_to_first_sentence.runinfo.md` 记录一致）。
- **一句话总结**：50/50 完成、error 0；`decode_to_first_sentence` **mean 389.0ms / p50 92.0ms / p90 967.1ms**
  （en mean 635.7ms，zh mean 142.3ms）；`sentence_end_found` **50/50（100%）**；解码速率 mean 26.0 tok/s。

### 跑后结果级 QA（评审 r3 放行条件，七项逐条结论）

| # | 检查项 | 结论 |
|---|---|---|
| 1 | 完成度 | **PASS**：50/50，CSV `error` 列全空，exit 0 |
| 2 | RUNINFO | **PASS**：`decode_to_first_sentence.runinfo.md` 字段齐全；输入文件 sha256 与 E4 复跑结果 `exp1_results_20260820_171522.json` 实测一致 |
| 3 | 样本完整 | **PASS**：50 个唯一 sample_id，与 E4 streaming 清单逐一对齐（排序比对相等） |
| 4 | 重放口径 | **PASS**：全部行 `fragment_count ≥ 1`、`fragments_sha256` 非空、`prompt_tokens > 0` |
| 5 | 计时关系 | **PASS**：全部行 `decode_to_first_sentence_ms ≤ decode_total_ms`、`n_tokens > 0`；`sentence_end_found=1` 行首句索引均在 `[0, n_tokens)` 内 |
| 6 | 句末覆盖 | **PASS**：`sentence_end_found` 50/50 = 100%（≥90%） |
| 7 | 量级 | **PASS**：mean 389.0ms 落在 0.3–1.5s 预期带（max 1769.8ms，无 >5s 异常） |

**总判定：七项全过，数字可入表。** 无停止规则触发项。

### 产物清单（本目录）

- 正式：`decode_to_first_sentence.csv`（50 行）、`.summary.txt`、`.runinfo.md`、`.checkpoint.jsonl`、`decode_to_first_sentence_run.log`
- 冒烟：`decode_to_first_sentence_smoke.csv`、`.summary.txt`、`.runinfo.md`、`.checkpoint.jsonl`

### 移交说明（给本机侧）

- 本测分项用于装配 `ttfa_budget.csv`（Table VIII）；装配与 changelog 登记由本机侧完成。
  **装配口径（2026-08-21 审查 P0 裁决=方案2，以此为准，替换此处此前的 E4-TTFT 链条公式）**：
  `TTFA = E5.endpoint_detection_wait(53ms) + E5.post_final_enqueue(1012.5ms，final 段入队→首 token) + 本测 decode_to_first_sentence(389ms) + E6.TTFC(zh 13.99s / en 11.86s)`——
  2s 追加静音的装置等待从 B 行对称剔除（A 行 ttft 从 audio_end 起算本就不含）；
  装配结果 B ALL 14.38s / A ALL 22.67s。
- CSV 中 `first_token_latency_ms` 为重放口径参考值，**不是**预算表分项。
- 中英文差异明显（en 636ms vs zh 142ms）：英文首句普遍更长（token 数多），装配分语种行时可直接引用 summary 分组数字。
- 全部数字绑定本 GPU 主机平台口径（裁决 D），不与他机混排。
