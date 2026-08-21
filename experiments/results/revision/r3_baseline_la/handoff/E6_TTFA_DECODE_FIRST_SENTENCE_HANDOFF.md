# TTFA 预算缺项补测交接文档（T_decode_to_first_sentence，R6 / 意见2）

> **读者**：代码审查人员；GPU 主机实验执行者。
> **状态**：补测脚本已完成并提交，本机自检通过；待审查确认后在 GPU 主机执行（约 0.5 GPU 小时）。
> **关联**：`experiments/CISR_REVISION_PLAN.md` §7.3（TTFA 预算表定义）、
> `experiments/results/revision/PAPER_HANDOFF.md` §一 E5/E6、§三 待决清单。

---

## 1. 前因后果（为什么需要这次补测）

1. **审稿意见 2** 要求给出端到端"语音结束 → 首个可听音频帧"（TTFA）的分解预算。
2. 修订计划 §7.3 定义预算表四个组成项：
   `TTFA = T_endpoint（端点等待）+ TTFT（首 token）+ T_decode_to_first_sentence（首 token → 首个句末标点 token）+ T_TTS_first_chunk（TTS 首包）`。
   其中 TTS 按句触发合成，首句文本齐了之后才能请求 TTS，因此第三项是预算表的必要环节。
3. E1–E6 已全部跑完并通过验收（2026-08-19 ~ 08-21，见 `PAPER_HANDOFF.md`）：
   T_endpoint（E5：mean 53ms）、TTFT（E1/E4）、T_TTS_first_chunk（E6：zh 13.99s / en 11.86s）均已在手。
4. **缺项发现**（2026-08-21 本机离线核对产物时发现）：E4 复跑（`r4_commit/`，50 样本，
   max_tokens=128，full_response 落盘）的结果 JSON 只有 `first_token_time` 等聚合时刻，
   **没有逐 token 时刻**；`StreamLLMInference.generate()` 的 `timing_events` 每轮覆写、
   不写逐 token 日志，run.log 亦无 token 级记录。因此
   **T_decode_to_first_sentence 无法从任何既有产物离线恢复**。
5. 结论：需要一次小规模补测。这是本次唯一需要 GPU 主机执行的补充实验；
   其余论文缺口（离线 WER/CER、分词接缝、语义一致性、表格装配）均为本机离线分析，不涉及主机。

## 2. 为什么独立 LLM 测量在方法学上成立（请审查重点确认）

- 待测量是**纯 LLM 解码段时延**。生产管线中，首 token 之后 ASR 已收尾，解码在 GPU 独占状态下
  逐 token 进行；该时段与 ASR/VAD 队列无耦合。因此"独立喂同一提示文本、测量同一解码段"与
  管线内测量的是同一个物理量。
- 输入提示取 **E4 落盘的 50 条 streaming 模式 `transcribed_text`**——即 LLM 当时真实收到的用户输入，
  不是参考文本、不是重新识别的文本。
- 预填/解码走生产同款路径：`StreamLLMInference.cache_prompt(text, is_end=True)` + `generate()`，
  同模型（Qwen/Qwen2-7B-Instruct）同设备（默认 cuda:1）同 `max_tokens=128`。
  一次性预填与生产的增量预填在提示内容上等价（chat template 相同）；差异只影响预填段
  （TTFT 已有 E4/E5 实测值，不取本脚本的预填计时），**不影响解码段测量**。
- 逐 token 时刻在 `generate()` 每次 yield 处记录（yield 紧接该 token 解码完成，见
  `src/llm/stream_llm_inference.py:218`）。
- 句末判定规则：首个 `。！？!?`；英文 `.` 仅在非数字夹击中计数（豁免 `3.5` 类小数）。
  无句末标点的回复回退为整段解码时间并置 `sentence_end_found=0`（汇总时可见）。
- 平台口径：本测量与 E1–E6 同一 GPU 主机，绝对值同样绑定该平台（裁决 D），不与他机混排。

## 3. 补测脚本

`experiments/scripts/measure_decode_to_first_sentence.py`

- 输入：E4 结果 JSON（默认 `r4_commit/exp1_results_*.json` 最新一个，取 streaming 模式、
  error 空、transcribed_text 非空的样本；E4 为 50 条）。
- 输出：`decode_to_first_sentence.csv`（逐样本：n_tokens / ttft_ms（独立预填口径，仅参考）/
  first_sentence_token_idx / decode_to_first_sentence_ms / decode_total_ms / tokens_per_s /
  sentence_end_found / first_sentence_text）+ 同名 `.summary.txt`（overall 与分语言
  mean/std/p50/p90）。
- 预热 3 轮（与生产一致）；`--repeat` 可重复测量（默认 1）；`--max-samples` 供冒烟。
- **本机自检**：`--self-test` 不加载模型，用脚本化 token 流验证句末检测（中文/英文/小数豁免/
  无句末回退）、首句 token 定位、计时关系、CSV 与汇总链路——2026-08-21 本机 12/12 断言通过。

## 4. GPU 主机执行

```bash
git pull   # 需包含本脚本（提交见 git log）
uv run python -m experiments.scripts.measure_decode_to_first_sentence --self-test   # 期望全 PASS
# 冒烟（3 条，确认模型加载与产出格式）
uv run python -m experiments.scripts.measure_decode_to_first_sentence \
  --llm-device cuda:1 --max-samples 3 \
  --output experiments/results/revision/r6_ttfa/decode_to_first_sentence_smoke.csv
# 正式（50 条，约 0.5 GPU 小时以内）
uv run python -m experiments.scripts.measure_decode_to_first_sentence \
  --llm-device cuda:1 \
  --output experiments/results/revision/r6_ttfa/decode_to_first_sentence.csv
```

## 5. 验收标准

1. 50/50 样本完成、无异常（脚本遇 generate 空产出会抛错，应为 0）；
2. `sentence_end_found` 比例合理（预期 ≥90%；无句末样本回退整段解码，属正常口径）；
3. summary 中 `decode_to_first_sentence` mean 量级预期 0.3–1.5s（首句约 10–30 token × 3090 解码速率）；
   若显著超出（如 >5s）应停止并保留现场上报，不得强行入表；
4. CSV + summary 落盘，RUNINFO 记录命令、commit、耗时。

## 6. 产出如何进论文（Table VIII 装配，本机离线完成）

`TTFA = E5.endpoint_detection_wait(53ms) + E4.TTFT(streaming mean) + 本测 decode_to_first_sentence(mean) + E6.TTFC(zh 13.99s / en 11.86s)`，
逐项 mean±std 装配为 `r6_ttfa/ttfa_budget.csv`（Table VIII），全部数字绑定第二平台口径；
E5 的 `final_enqueue_wait≈2s` 为测量装置属性（等追加静音推完），不进预算表，正文需说明。
