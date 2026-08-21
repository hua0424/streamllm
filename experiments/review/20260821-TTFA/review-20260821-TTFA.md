# TTFA 解码至首句补测审查报告（2026-08-21）

审查对象：

- `experiments/scripts/measure_decode_to_first_sentence.py`（commit `706c07a`）
- `experiments/results/revision/r3_baseline_la/handoff/E6_TTFA_DECODE_FIRST_SENTENCE_HANDOFF.md`
- 对照：E4 产物 `experiments/results/revision/r4_commit/exp1_results_20260820_171522.json`、`run_exp_latency.py`、`src/llm/stream_llm_inference.py`、`CISR_REVISION_PLAN.md` §7.3。

已执行的本机验证：

```text
uv run python -m experiments.scripts.measure_decode_to_first_sentence --self-test
11 项 PASS，脚本输出为 4/4 组通过

git diff 706c07a^..HEAD --check
通过
```

> 脚本自检只覆盖检测、计时和文件链路，不能证明正式测量使用了与 E4 管线相同的提示 token 序列。

## 结论

**暂不批准按当前脚本直接在 GPU 主机执行正式 50 条测量。** 句末检测和首 token 到句末 token 的计时实现本身基本正确，但脚本实际使用的 `transcribed_text` 不是 E4 运行时传给 LLM 的原始增量输入，导致 handoff 中“即 LLM 当时真实收到的用户输入”和“与生产路径等价”的论证不成立。修正输入重放口径并补齐结果审计后，可以执行 E0 和正式补测。

## P0：当前输入不是管线实际收到的提示文本

### 现象

脚本 `load_e4_prompts()`（约第 138–157 行）读取 E4 结果中的 `transcribed_text`。但生产流式路径在 `run_exp_latency.py` 中是：

```python
text_queue.put((output_text, False))
...
cache_prompt(text, pre_cache=kv_cache, is_end=is_end)
```

也就是说，LLM 接收的是 ASR 每次提交的原始 `output_text` 片段，并通过多次 `cache_prompt()` 增量追加；只有最终空片段用 `is_end=True` 加生成提示。

E4 落盘时却把这些片段重新拼接为：

```python
result.transcribed_text = " ".join(transcribed_text)
```

该字段因此人为插入了空格，而且丢失了片段边界。E4 同时保存了 `committed_fragments`，其中才保留了实际提交片段。对于中文、英文标点附着、数字和 BPE 子词边界，这个空格插入和边界丢失都可能改变 tokenizer 的 token 序列；而生产路径还使用了多次增量 `_add_stream_prompt()`，不能仅凭字符串语义相同就认为 KV 状态和生成条件相同。

### 影响

当前脚本测量的是“对重构后的带空格完整文本重新预填，再生成”的解码延迟，不是 E4 管线实际 prompt 状态下的解码延迟。独立测量仍可作为一个近似的 LLM 解码基准，但不能按 handoff 当前文字直接装配为生产管线的 `T_decode_to_first_sentence`，否则会过度声称方法学等价。

### 修改建议（阻塞）

优先改为使用 E4 的 `committed_fragments` 重放生产调用序列：

1. 对每个样本从 `pre_cache=None` 开始；
2. 对每个非空 `committed_fragments` 依次调用 `cache_prompt(fragment, pre_cache=kv, is_end=False)`；
3. 最后调用 `cache_prompt("", pre_cache=kv, is_end=True)`，再调用 `generate()`；
4. 记录 `fragment_count`、输入来源字段和最终 prompt token 长度（至少记录可审计的片段哈希或 JSONL 输入快照）；
5. 若某样本没有 `committed_fragments`，必须报错退出，不能静默退回 `transcribed_text`。

如果开发方坚持使用 `transcribed_text`，则必须把方法名称和 handoff 改为“完整转录文本独立 LLM 解码基准”，不得声称是 E4 真实 prompt 的等价重放，也不能直接作为严格的生产 `T_decode_to_first_sentence` 预算项。

## P1：没有生成 RUNINFO，交付要求未实现

handoff §5 第 4 条要求 RUNINFO 记录命令、commit、耗时；但当前脚本只写 CSV 和 `.summary.txt`（`write_outputs()` 约第 112–135 行），没有写 RUNINFO，也没有记录：

- E4 输入文件的实际路径和内容哈希；
- 输入样本数、唯一 sample_id 数量和样本清单哈希；
- 模型名、设备、`max_tokens`、temperature、top-p、repetition penalty、warmup 轮数；
- 当前 git commit；
- 开始/结束时间和总耗时；
- 输入重放模式（修正后应为 fragment replay）。

建议正式输出同时写 `decode_to_first_sentence.runinfo.md` 或 JSON RUNINFO。GPU 命令完成后没有 RUNINFO 时，不应通过正式结果验收。

## P1：输入样本选择没有强制 50 条、唯一性和配置校验

`load_e4_prompts()` 找到最新 glob 文件后，只过滤 `mode=streaming`、无 error 和非空文本，不检查：

- 是否正好 50 条；
- `sample_id` 是否唯一；
- E4 的 `max_tokens` 是否为 128；
- E4 模型名是否为 `Qwen/Qwen2-7B-Instruct`；
- E4 的 LLM 设备和本次设备是否符合锁定方案；
- 取到的文件是否确实是预期 E4 复跑，而不是目录中的另一份 JSON。

当前实际文件验证得到 50 条 streaming 样本、50 个唯一 ID，配置为 Qwen/Qwen2-7B-Instruct、cuda:1、max_tokens=128，因此现有现场没有暴露数据错误；但脚本应把这些条件变成运行时检查，避免 GPU 侧 glob 选择错误文件后仍产出“完整”结果。`--max-samples` 仅允许冒烟模式，正式模式应拒绝少于 50 条的输入。

## P1：失败处理不能形成可审计的 50/50 结果

正式循环（约第 258–266 行）中，`run_one()` 任何异常都会直接中止整个进程；CSV 只有在所有样本完成后才写出。因此中途 GPU/模型异常时既没有逐样本错误记录，也没有可恢复的部分结果。handoff 的“50/50 完成、无异常”可以通过人工判断，但不利于审计和重跑。

建议至少：

- 捕获每个样本异常并写入 `error`、`sample_id`、`pass_idx`；
- 正式验收要求 `error` 为空且样本 ID 集合完整；
- 使用临时文件或逐行 JSONL/checkpoint，避免进程中止丢失已完成测量；
- 失败样本不得被汇总函数当作有效延迟样本。

该项可在 P0 修正后一起补，不建议用当前脚本直接长跑。

## P2：句末检测和计时实现的审查结论

以下部分未发现当前阻塞错误：

- `detect_first_sentence_end()` 对中文句末和英文 `. ` 的基本判断正确；数字夹击的小数（如 `3.5`）会被豁免；
- `run_one()` 在 `generate()` 每次 yield 后记录时刻，首句标点所在 yield 的时间减首 token yield 的时间，符合“首 token → 首句 token”的定义；
- 没有句末时回退到整段解码时间并设置 `sentence_end_found=0`，口径明确；
- `eval_mode=False`、`max_new_tokens=128`、默认 temperature/top-p/repetition penalty 与生产 `generate()` 调用一致；
- 本机 self-test 覆盖了中文、英文、小数、无标点、首句定位、CSV 和 summary 链路。

但正式结果仍应记录生成参数，并在结果级 QA 中检查 `decode_to_first_sentence_ms <= decode_total_ms`、`n_tokens > 0`、`sentence_end_found` 与首句索引的一致性。

## 文档修改建议

`E6_TTFA_DECODE_FIRST_SENTENCE_HANDOFF.md` §2 的下列表述需要在代码修正前删除或改写：

- “E4 落盘的 50 条 streaming 模式 `transcribed_text`——即 LLM 当时真实收到的用户输入”；
- “一次性预填与生产的增量预填在提示内容上等价”；
- “与管线内测量的是同一个物理量”。

修正为 fragment replay 后，可以保留等价性论证，但应明确：本补测重放的是 E4 保存的 committed fragment 序列，使用同一 `cache_prompt` 增量调用和同一最终生成路径；测量时间从首个 `generate()` yield 到首句标点 token yield，不把独立预填时间并入该分项。

## 放行条件

1. 使用 `committed_fragments` 进行生产调用序列重放，或明确降级为独立基准并修改论文装配口径；
2. 强制校验 50 条、唯一 ID、E4 配置和输入文件；
3. 生成 RUNINFO，记录 commit、输入哈希、参数、计时和重放模式；
4. 增加逐样本异常/恢复或至少完整失败清单；
5. 重新运行 self-test；
6. 先执行 3 条冒烟，确认 fragment replay、生成收尾和输出字段，再执行正式 50 条；
7. 正式结果通过 50/50、无 error、输入 ID 完整、参数一致和计时关系 QA 后，才可装配 `ttfa_budget.csv`。

**最终决定：当前版本不放行正式 GPU 50 条补测；P0 修正并完成上述审计增强后放行。**
