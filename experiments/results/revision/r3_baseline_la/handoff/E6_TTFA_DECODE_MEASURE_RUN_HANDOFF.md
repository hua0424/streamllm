# TTFA 解码至首句补测执行交接（GPU 主机）—— 评审已放行

> **读者**：GPU 主机实验执行者。
> **任务**：补测 TTFA 预算表缺项 `T_decode_to_first_sentence`（首 token → 首个句末标点 token 的 LLM 解码延迟）。
> **状态**：三轮评审完毕，r3 **放行**（`experiments/review/20260821-TTFA/`：r1 提 P0/P1 → 已修正；r2 提 1 项 P1 → 已修复；r3 通过）。
> **预计耗时**：约 0.5 GPU 小时（50 条提示，仅 LLM，只用 cuda:1；不涉及 ASR，cuda:0 不受影响）。
> **设计/方法学论证**：同目录 `E6_TTFA_DECODE_FIRST_SENTENCE_HANDOFF.md`（先读 §1 前因后果）。

---

## 1. 这次测什么、为什么（一句话版）

E4 复跑没记录逐 token 时刻，TTFA 预算表（计划 §7.3）缺"首 token → 首句 token"分项且无法离线恢复；
本补测用 E4 落盘的 50 条 `committed_fragments` 逐片段重放生产增量预填序列，再逐 token 计时解码，
取首个句末标点 token 的时间差。结果进论文 Table VIII（`ttfa_budget.csv`，本机侧装配）。

## 2. 执行步骤（顺序不可跳）

### 步骤 0：代码就位 + 自检

```bash
cd <repo> && git pull
git log --oneline -1   # 应为 424e1f9 或更新（必须 ≥ 1efc809，含评审 r2 修复）
uv run python -m experiments.scripts.measure_decode_to_first_sentence --self-test
# 期望：26 PASS / 0 FAIL，exit 0
```

### 步骤 1：冒烟（3 条，约 5 分钟）

```bash
uv run python -m experiments.scripts.measure_decode_to_first_sentence \
  --llm-device cuda:1 --max-samples 3 \
  --output experiments/results/revision/r6_ttfa/decode_to_first_sentence_smoke.csv
```

**冒烟验收**（全过才继续）：
- 3/3 完成、exit 0、CSV `error` 列全空；
- CSV 含 `fragment_count`/`fragments_sha256`/`prompt_tokens` 列且非空（确认走的是 fragment replay）；
- `decode_to_first_sentence_smoke.runinfo.md` 生成且字段齐全（命令、commit、输入文件 sha256、
  样本清单哈希、生成参数、重放模式 `fragment_replay`）；
- 每条日志打印的 `decode_to_first_sentence` 为正的毫秒值（量级几十到几百 ms 属正常）。

### 步骤 2：正式 50 条（约 0.5 GPU 小时）

```bash
uv run python -m experiments.scripts.measure_decode_to_first_sentence \
  --llm-device cuda:1 \
  --output experiments/results/revision/r6_ttfa/decode_to_first_sentence.csv \
  2>&1 | tee experiments/results/revision/r6_ttfa/decode_to_first_sentence_run.log
```

说明：
- 输入默认取 `r4_commit/exp1_results_*.json` 最新一个（E4 复跑产物）；脚本强制校验
  恰好 50 条 streaming 样本、ID 唯一、E4 config（Qwen2-7B / max_tokens=128 / cuda:1），
  **校验失败会直接退出——遇到退出不要改参数凑合，停止并上报**；
- 逐样本容错：单条失败写 CSV `error` 列并继续，结束时有失败 exit 1；
  `.checkpoint.jsonl` 逐行追加防中断丢失（仅防丢、不续跑，重跑即全新）；
- 正式模式不要加 `--max-samples` / `--repeat`（默认 1）。

## 3. 跑后结果级 QA（评审 r3 放行条件，逐条核对）

| # | 检查项 | 通过标准 |
|---|---|---|
| 1 | 完成度 | 50/50，CSV `error` 列全空，exit 0 |
| 2 | RUNINFO | `decode_to_first_sentence.runinfo.md` 存在且字段齐全；其中输入文件 sha256 对应的确是 E4 复跑结果 |
| 3 | 样本完整 | 50 个唯一 sample_id，与 E4 清单一致 |
| 4 | 重放口径 | `fragment_count ≥ 1`、`fragments_sha256` 非空、`prompt_tokens > 0` |
| 5 | 计时关系 | 每行 `decode_to_first_sentence_ms ≤ decode_total_ms`、`n_tokens > 0`；`sentence_end_found=1` 的行首句索引在 `[0, n_tokens)` 内 |
| 6 | 句末覆盖 | `sentence_end_found` 比例 ≥90% 预期（无句末样本回退整段解码为正常口径） |
| 7 | 量级 | summary 的 mean 预期 0.3–1.5s；若 >5s 停止保留现场上报，不得入表 |

**停止规则**：任一项不过 → 停止、保留现场（CSV/log/RUNINFO 原样保留）、上报本机侧；
QA 全过前任何数字不得写入论文。

## 4. 完成后登记与回传

1. 结果目录应含：`decode_to_first_sentence.csv`、`.summary.txt`、`.runinfo.md`、
   `.checkpoint.jsonl`、`decode_to_first_sentence_run.log`（+ 冒烟三件套）；
2. 把上表 7 项 QA 结论写进 `experiments/results/revision/r6_ttfa/RUNINFO.md`（或在已有 RUNINFO 追加一节），
   连同一句话总结（mean/p50/p90、sentence_end_found 比例）通知本机侧；
3. 本机侧收到后：装配 `r6_ttfa/ttfa_budget.csv`（Table VIII：
   E5 端点 53ms + E4 TTFT + 本测解码段 + E6 TTFC zh 13.99s/en 11.86s），登记 changelog。

## 5. 注意

- 本任务只用 cuda:1 上的 Qwen2-7B，可与任何 cuda:0 侧工作并行；
- CSV 中 `first_token_latency_ms` 是重放口径参考值，**不是**预算表的 TTFT（TTFT 用 E4/E5 实测值）；
- E5 的 `final_enqueue_wait≈2s` 是测量装置属性，不进预算表。
