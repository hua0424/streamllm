# TTFA 补测审查回复 r2（2026-08-21，对应 review-20260821-TTFA-r2.md）

> 审查报告：`experiments/review/20260821-TTFA/review-20260821-TTFA-r2.md`
> 回复范围：唯一 P1 阻塞项（异常行身份字段覆盖）+ 4 条非阻塞注意事项。**阻塞项采纳并修复**。

## P1（异常行丢失 sample_id/language/pass_idx）—— 确认成立，已修复

核实：`CSV_FIELDS` 含 `sample_id`/`language`/`pass_idx`，旧异常分支
`rec = {k: "" for k in CSV_FIELDS}` 把身份字段造成空串，`row = {...identity..., **rec}`
被 `**rec` 覆盖——审查属实，且与上轮回复中"含 sample_id/pass_idx"的承诺不符。

修复（`measure_decode_to_first_sentence.py`）：

1. 抽取 `measure_sample(llm, sample, pass_idx, max_tokens)` 承载逐样本测量与异常处理，
   异常分支构造 `rec` 时**排除身份字段**
   （`{k: "" for k in CSV_FIELDS if k not in _IDENTITY_FIELDS}`），身份只在外层写入一次；
2. 异常行补充 `fragments_sha256`（与正常行一致的片段审计口径）；
3. self-test 新增两条真实异常行断言（假 LLM `cache_prompt` 抛错）：
   - `measure_sample` 返回的异常行保留 `sample_id='crosswoz_bad1'`、`language='zh'`、`pass_idx=2`，
     `error='boom'`、`fragment_count=2`、`fragments_sha256` 完整；
   - 该异常行经 `write_outputs()` 写 CSV 后身份字段仍在（`CSV 异常行身份保留`），
     且不进入汇总（`n=2, error=1`）。

验证：

```text
uv run python -m experiments.scripts.measure_decode_to_first_sentence --self-test
26/26 PASS（含 2 条新增异常行身份断言）

uv run python -m py_compile experiments/scripts/measure_decode_to_first_sentence.py
通过
```

## 非阻塞注意事项回应

1. **glob 取最新文件**：同意。正式执行后 RUNINFO 记录实际输入文件路径 + SHA-256，
   结果审计时据此确认确为 E4 复跑产物（handoff §5 第 2 条已要求 RUNINFO 存在且字段齐全）。
2. **checkpoint 启动清空**：同意其定位——仅作中断防丢，不作续跑机制；正式补测均为全新执行。
3. **`--repeat > 1` 行数超 50**：同意。正式执行使用默认 `--repeat 1`；若重复测量，
   验收按 `sample_id × pass_idx` 二元组检查完整性（handoff §5 第 1 条口径）。
4. **self-test 不替代 GPU 冒烟**：同意。执行顺序保持 handoff §4：self-test → 3 条冒烟 → 正式 50 条，
   正式结果按 §5 七项 QA（50/50、error 全空、唯一 ID、RUNINFO、fragment replay 口径、
   计时关系）独立检查后才装配 `ttfa_budget.csv`。

## 复审决定对照

r2 的唯一阻塞项已修复并通过 self-test（26/26）与 py_compile。请审查人员复核；
如无新问题，GPU 主机按 handoff §4 顺序执行。
