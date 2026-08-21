# TTFA 补测复审报告 r3（2026-08-21，放行）

审查对象：

- `experiments/review/20260821-TTFA/reply-20260821-TTFA-r2.md`
- 修复 commit `1efc809`（`experiments/scripts/measure_decode_to_first_sentence.py`）

## 阻塞项修复核验

r2 唯一阻塞项（异常行身份字段被 `**rec` 覆盖）已正确修复：

1. 逐样本测量与异常处理抽取为 `measure_sample()`，异常分支构造 `rec` 时排除
   `_IDENTITY_FIELDS = ("sample_id", "language", "pass_idx")`，身份只在外层写入一次，
   `**rec` 不再可能覆盖身份字段；
2. 异常行补充 `fragments_sha256`，与正常行的片段审计口径一致；
3. 主循环改用 `measure_sample()`，正常/异常路径统一，日志引用同步改为 `row`。

本机独立验证（非转引开发数据）：

```text
uv run python -m experiments.scripts.measure_decode_to_first_sentence --self-test
26 PASS / 0 FAIL，exit 0（含"异常行身份保留""CSV 异常行身份保留"两条新增断言；
其中函数级断言验证 sample_id='crosswoz_bad1'、language='zh'、pass_idx=2、
error='boom'、fragment_count=2、fragments_sha256 完整；CSV 级断言验证
round-trip 后身份字段仍在且该行不进汇总）

uv run python -m py_compile experiments/scripts/measure_decode_to_first_sentence.py
通过

git diff --check 1efc809^..1efc809
通过
```

说明：一次中间运行的控制台输出曾显示缺"英文句末"一行，重定向到文件后完整复核为
26 PASS / 0 FAIL，属显示层问题，非脚本缺陷。

## 非阻塞事项确认

开发对 r2 四条非阻塞注意事项的回应（glob 审计依赖 RUNINFO、checkpoint 仅防丢不续跑、
`--repeat>1` 按 sample_id × pass_idx 验收、self-test 不替代 GPU 冒烟）均与 r2 建议一致，
无异议。

## 复审决定

**通过。批准 GPU 主机按 handoff §4 顺序执行：**

1. `git pull` 至 `1efc809` 或更新；
2. `--self-test`（期望 26/26）；
3. 3 条冒烟（`--max-samples 3`，输出 `decode_to_first_sentence_smoke.csv`）：确认模型加载、
   fragment replay 路径、CSV/RUNINFO 字段完整；
4. 正式 50 条（默认 `--repeat 1`，输出 `decode_to_first_sentence.csv`）。

正式结果仍需独立结果级 QA（handoff §5 七项：50/50、error 全空、唯一 ID、RUNINFO 齐全、
fragment_replay 口径、计时关系 `decode_to_first_sentence_ms ≤ decode_total_ms` 等、
`sentence_end_found` 比例与 mean 量级合理）通过后，方可装配 `r6_ttfa/ttfa_budget.csv`
进论文 Table VIII。QA 未过前，任何数字不得写入论文。
