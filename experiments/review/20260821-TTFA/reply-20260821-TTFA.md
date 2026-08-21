# TTFA 补测审查回复（2026-08-21，对应 review-20260821-TTFA.md）

> 审查报告：`experiments/review/20260821-TTFA/review-20260821-TTFA.md`
> 回复范围：P0 一项、P1 三项、P2 一项、文档建议。**全部采纳，无一驳回**；无需需求方决策的事项。
> 修改载体：`experiments/scripts/measure_decode_to_first_sentence.py`（修订版）、
> `experiments/results/revision/r3_baseline_la/handoff/E6_TTFA_DECODE_FIRST_SENTENCE_HANDOFF.md`（§2/§3/§5 改写）。

## 总体回应

审查的 P0 成立。我们对照生产代码核实：`run_exp_latency.py:731-732` 逐片段
`text_queue.put((output_text, False))`、`:734` 收尾 `("", True)`、`:762` 增量
`cache_prompt(text, pre_cache=kv_cache, is_end=is_end)`，而 `:801`
`result.transcribed_text = " ".join(transcribed_text)` 确实是插空格的重构。
此前 handoff 中"transcribed_text 即 LLM 当时真实收到的用户输入""一次性预填与增量预填等价"
两处表述不成立，已按建议改为 committed_fragments 重放口径，并同步补齐全部审计增强。

## 逐项回复

### P0（输入不是管线实际收到的提示文本）—— 采纳，已修正为 fragment replay

`run_one()` 改为接收 `fragments` 并按生产调用序列重放：

1. `kv = None` 起，对每个 `committed_fragments` 片段依次 `cache_prompt(frag, pre_cache=kv, is_end=False)`；
2. 最后 `cache_prompt("", pre_cache=kv, is_end=True)`（与 `run_exp_latency.py:734` 收尾一致），再 `generate()`；
3. 审计字段：`fragment_count`、`fragments_sha256`、`prompt_tokens`（生成前 `kv.pre_input_ids.shape[-1]`）入 CSV；
4. 样本无 `committed_fragments` → `load_e4_samples()` 直接 `SystemExit`，不静默退回 `transcribed_text`。

self-test 新增"重放调用序列"断言：假 LLM 记录 `cache_prompt` 调用，验证
`[("你好",False),("，这是",False),("测试。",False),("",True)]` 的确切顺序与 is_end 标记。

### P1-1（无 RUNINFO）—— 采纳，已实现

`write_runinfo()` 生成 `decode_to_first_sentence.runinfo.md`，含：完整命令行、git commit
（`git rev-parse HEAD`）、起止时间与耗时、输入文件路径 + sha256、E4 config 摘录、
样本数 + sample_ids sha256、重放模式（`fragment_replay`）、模型/设备/max_tokens/warmup/repeat、
generate 采样参数（temperature 0.1 / top_p 0.9 / repetition_penalty 1.1，生产默认值）、
计时口径说明、结果行数与 error 数。验收标准第 2 条已改为"无 RUNINFO 不通过验收"。

### P1-2（输入校验缺失）—— 采纳，已实现运行时强校验

`load_e4_samples()` 现在强制校验，任一不满足即 `SystemExit`：

- E4 config：`llm_model`（默认 Qwen/Qwen2-7B-Instruct）、`max_tokens`（与本次 --max-tokens 一致，默认 128）、
  `llm_device`（与本次 --llm-device 一致，默认 cuda:1）；
- sample_id 唯一性（重复即退出并列出重复项）；
- 正式模式恰好 `--expected-samples`（默认 50）条 streaming 样本，少了拒绝；
  `--max-samples` 仅作冒烟截断豁免；
- 输入文件取 glob 最新一个，其路径与 sha256 记入 meta 与 RUNINFO（文件选错可被审计发现）。

### P1-3（失败处理不可审计）—— 采纳，已实现

- 正式循环逐样本 try/except：异常写入该行的 `error` 列（含 sample_id/pass_idx）并继续；
- 每个样本完成即追加 `.checkpoint.jsonl`（进程中断不丢已完成测量）；
- `write_outputs()` 汇总只统计 `error` 为空的行，summary 头部带 `rows/ok/error` 计数；
- 存在失败样本时进程 exit 1；验收标准第 1 条改为"error 列全空"。

### P2（句末检测与计时）—— 确认无阻塞，QA 关系已补入验收

感谢确认。计时关系 QA（`decode_to_first_sentence_ms <= decode_total_ms`、`n_tokens > 0`、
`sentence_end_found` 与首句索引一致性）已写入 handoff §5 验收标准第 5 条；
生成参数已入 RUNINFO（见 P1-1）。

### 文档修改建议 —— 采纳，已改写

handoff §2 已删除/改写下列表述："transcribed_text 即 LLM 当时真实收到的用户输入"、
"一次性预填与增量预填等价"、"同一个物理量"的旧论证；改为明确：本补测重放 E4 保存的
committed fragment 序列，使用同一 `cache_prompt` 增量调用与同一最终生成路径；
测量区间为首个 `generate()` yield 到首句标点 token yield，预填时间不并入本分项
（CSV 的 `first_token_latency_ms` 标注为重放口径参考值，不进预算表）。§3/§5 同步更新。

## 验证证据

```text
uv run python -m experiments.scripts.measure_decode_to_first_sentence --self-test
23/23 断言 PASS（修订前 12 项 + 新增：重放调用序列/fragment_count/prompt_tokens/空片段拒绝/
正常加载/meta 哈希/冒烟截断/缺 fragments 退出/数量不符退出/配置不符退出/重复 ID 退出/
error 行不进汇总/RUNINFO 关键字段）

uv run python -m py_compile experiments/scripts/measure_decode_to_first_sentence.py
通过
```

## 放行条件对照

| 审查放行条件 | 状态 |
|---|---|
| 1. committed_fragments 重放或降级口径 | ✅ 已按 fragment replay 实现 |
| 2. 强制校验 50 条/唯一 ID/E4 配置/输入文件 | ✅ 运行时 SystemExit 级校验 |
| 3. RUNINFO（commit/输入哈希/参数/计时/重放模式） | ✅ `write_runinfo()` |
| 4. 逐样本异常/失败清单 | ✅ error 列 + checkpoint JSONL + exit 1 |
| 5. 重新运行 self-test | ✅ 23/23 |
| 6. 先 3 条冒烟再正式 50 条 | ✅ handoff §4 命令已含冒烟步骤（--max-samples 3） |
| 7. 正式结果 QA 后才装配 ttfa_budget.csv | ✅ handoff §5 验收标准（含 RUNINFO 存在性与计时关系检查） |

请审查人员复核修订版脚本与 handoff；放行后 GPU 主机按 handoff §4 执行（self-test → 3 条冒烟 → 正式 50 条）。
