# TTFA 补测复审报告 r2（2026-08-21）

审查对象：

- `experiments/review/20260821-TTFA/reply-20260821-TTFA.md`
- 修订版 `experiments/scripts/measure_decode_to_first_sentence.py`
- 修订版 `experiments/results/revision/r3_baseline_la/handoff/E6_TTFA_DECODE_FIRST_SENTENCE_HANDOFF.md`

## 已验证内容

本机执行：

```text
uv run python -m experiments.scripts.measure_decode_to_first_sentence --self-test
结果：全部 23 项 PASS

uv run python -m py_compile experiments/scripts/measure_decode_to_first_sentence.py
结果：通过

git diff --check 706c07a..HEAD
结果：通过
```

## P1 阻塞问题：异常行会丢失 sample_id、language 和 pass_idx

位置：`experiments/scripts/measure_decode_to_first_sentence.py:456-459`。

当前异常处理逻辑是：

```python
rec = {k: "" for k in CSV_FIELDS}
rec.update({"fragment_count": len(s["fragments"]), "error": str(e)})
row = {"sample_id": s["sample_id"], "language": s["language"],
       "pass_idx": pass_idx, **rec}
```

`CSV_FIELDS` 本身包含 `sample_id`、`language`、`pass_idx`。因此 `**rec` 会覆盖前面刚写入的真实身份字段，异常行最终会变成：

```text
sample_id="", language="", pass_idx=""
```

这与回复中承诺的“异常写入该行的 `error` 列（含 sample_id/pass_idx）”不一致，也使 checkpoint JSONL 无法按样本定位失败。正式运行即使最终 exit 1，失败清单仍不完整。

### 修改建议

构造异常记录时只填充测量字段，不要把身份字段放进 `rec`，例如：

```python
rec = {k: "" for k in CSV_FIELDS
       if k not in {"sample_id", "language", "pass_idx"}}
rec.update({
    "fragment_count": len(s["fragments"]),
    "fragments_sha256": _sha256_text(json.dumps(s["fragments"], ensure_ascii=False)),
    "error": str(e),
})
row = {
    "sample_id": s["sample_id"],
    "language": s["language"],
    "pass_idx": pass_idx,
    **rec,
}
```

同时在 self-test 中增加一个真实异常行断言，确认 CSV 和 checkpoint 中仍保留具体的 `sample_id` 与 `pass_idx`，而不是只测试“error 行不进汇总”。修复后应重新运行 self-test 和 py_compile。

## 已通过的上一轮阻塞项

以下内容已按上一轮意见正确修正：

1. **P0 fragment replay**：`run_one()` 从 `kv=None` 开始，逐个调用 `cache_prompt(fragment, is_end=False)`，最后调用 `cache_prompt("", is_end=True)`，随后 `generate()`。这与生产 `run_exp_latency.py` 的调用序列一致；不再使用带空格重构的 `transcribed_text`。
2. **空片段收尾语义**：`StreamLLMInference.cache_prompt()` 在已有 KV 状态且 `is_end=True` 时会追加 generation prompt，因此 `cache_prompt("", is_end=True)` 不会触发空文本异常。
3. **输入校验**：正式模式默认要求 50 条，检查 sample_id 唯一性，并校验 E4 的模型、设备和 max_tokens；冒烟通过 `--max-samples` 明确截断。
4. **审计产物**：已加入输入文件 SHA-256、sample ID 清单哈希、片段哈希、prompt token 数、RUNINFO 和 checkpoint JSONL。
5. **汇总过滤**：`write_outputs()` 只将 `error` 为空的行纳入统计，并在 summary 中报告 rows/ok/error。
6. **计时与句末逻辑**：首 token 到首句标点 token 的 yield 时间差、无句末回退、数字夹击小数豁免均实现清楚。

## 非阻塞注意事项

- `load_e4_samples()` 使用 glob 排序后取最后一个文件。当前实际目录只有预期 E4 结果时没有问题；正式执行后 RUNINFO 会记录实际文件和 SHA-256，仍应在结果审计时确认路径确实是 E4 复跑产物。
- checkpoint JSONL 在程序启动时会清空旧文件；这适合 `--no-resume` 式全新补测，但不要把旧 checkpoint 当作可恢复续跑机制。
- `--repeat > 1` 时输出行数会超过 50，正式验收应按 `sample_id × pass_idx` 检查完整性，而不能只看总行数。
- 当前 self-test 的“23/23”是脚本内部检查项；仍不能替代 GPU 冒烟和正式结果级 QA。

## 复审决定

**当前版本仍暂缓 GPU 正式 50 条执行，唯一阻塞项是异常行身份字段覆盖。**

修复该处并新增异常行身份保留的回归断言后，重新通过 self-test 和 py_compile，即可批准执行 handoff §4 的顺序：self-test → 3 条 GPU 冒烟 → 正式 50 条。正式结果仍需独立检查 50/50、error=0、唯一 ID、RUNINFO、fragment replay 口径和计时关系后，才可装配 `ttfa_budget.csv`。
