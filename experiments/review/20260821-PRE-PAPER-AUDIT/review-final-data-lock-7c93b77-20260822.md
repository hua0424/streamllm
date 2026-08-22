# 最终数据锁定复核（7c93b77，2026-08-22）

- **对比基线**：`6069868`
- **审查提交**：`7c93b77255857d51ee4d9230dd3e66ff3bc61e8e`
- **规格来源**：`review-reply-paper-data-readiness-f54f2bd-20260822.md`
- **最终裁决**：**通过。`7c93b77` 可以作为论文修订的最终数据锁定基线，正式开始修改 `main.tex`。未发现仍需补实验、重跑或修改统计数字的问题。**

## Standards

### 已通过

- Table VIII 生成脚本已移至 `experiments/scripts/assemble_table_viii.py`，符合 `AGENTS.md` 关于生成脚本位置的规定；CSV/Markdown 结果仍保存在 `experiments/results/`。
- 装配方法学已同步至 `experiments/EXPERIMENT_DESIGN.md`，包含 repeat0、ddof=1、分位数插值、单位与精度、received-only-QA、旧估计项排除和四项 QA。
- 当前提交未引入密钥、机器专属设备配置或实验依赖变更。
- 当前工作树 clean，`git diff --check` 通过；装配脚本在当前环境可直接运行，生成后工作树仍无差异。

### 非阻塞规范残留

`reply-review-paper-data-readiness-f54f2bd-20260822.md` 是整改前的历史回复函，仍保留旧生成命令：

```text
uv run python experiments/results/revision/r7_ttfa_unified/table_viii/assemble_table_viii.py
```

正确命令现为：

```text
uv run python experiments/scripts/assemble_table_viii.py
```

该旧命令只存在于历史回复函，不是当前锁定数据或论文写作入口，因此不阻塞数据锁定；但若继续把该回复函当作复核操作手册，应做一次文档更正。

提交 `7c93b77` 同时包含标签修正、脚本迁移、W7 措辞、状态清理和锁定登记，范围略宽于“一提交一逻辑变化”的理想约定。不过这些修改共同服务于同一数据锁定闭环，不影响审查结论。

## Spec

### Table VIII 边界修正通过

第二分项现在准确声明为：

```text
t_feed_to_close_wait
= pipeline_input_close_ns - feed_end_ns
= 喂入结束 → 管线输入关闭
```

独立复算仍为：

```text
feed_end → input_close          133.014 ms
feed_end → explicit_flush_start 132.682 ms
explicit_flush_start → close      0.332 ms
explicit flush 本身               0.210 ms
flush_done → close                0.122 ms
```

装配稿、CSV、装配脚本、`PAPER_HANDOFF.md`、`PAPER_WRITING_REFERENCE.md` 和 `EXPERIMENT_DESIGN.md` 均使用了准确边界，并明确禁止把 133ms 解释为 flush 计算开销。

### 数字和 QA 不变

本机重新执行：

```text
uv run python experiments/scripts/assemble_table_viii.py
```

结果：

- QA-1：100 条六分项闭合最大残差 0；
- QA-2：received→playable mean 0.1ms、max 0.2ms；
- QA-3：与运行侧 summary 48 行全部一致；
- QA-4：checkpoint LF SHA-256 前缀 `4edcd6ec28189d00` 一致；
- 生成后的 CSV/MD 与 Git 锁定版本无差异。

Table VIII 锁定数字保持：

| 系统 | mean TTFA | P50 TTFA |
|---|---:|---:|
| System B | 5481.9ms | 3113.7ms |
| System A | 22425.7ms | 22269.9ms |

论文可选择：

- mean 口径：System B 降低 75.6%；或
- P50 口径：System B 降低 86.0%。

两者不能在同一结论中混成一个中心趋势口径。

### 其余整改通过

- W7 对爆音来源的无证据归因已经删除；当前仅记录听到爆音/削波、来源未区分、仍正常可懂。
- `PAPER_WRITING_REFERENCE.md` §十已改为 P0 整改闭环记录。
- 装配层方法学已完整同步。
- `REVISION_CHANGELOG.md` 已登记数据冻结和书面变更规则。
- R7 仍是 Table VIII 唯一数据源；旧 R6 跨运行装配全部作废。
- `r7_tts_control` 仍按流程偏差豁免结果使用，7076ms 不作为 Table VIII 行项，引用时必须保留披露。

## 非阻塞文档残留

以下内容不影响数据有效性，也不阻塞 `main.tex` 修改，但建议在论文写作初期顺手清理：

1. `PAPER_HANDOFF.md` §三仍写“仅剩装配”，但装配已经完成；其六项“待决”也已在 `PAPER_WRITING_REFERENCE.md` 判定完毕。
2. `PAPER_WRITING_REFERENCE.md` 末尾仍写“剩余：宣布最终数据锁定”，在本复核通过后应视为历史状态。
3. `PAPER_WRITING_REFERENCE.md` §九仍提到旧跨运行 TTFA“方案 (a) 定稿”；应明确这是历史 P0 裁决，已被 R7 统一实测替代，避免写稿人误读。
4. 历史回复函仍有旧脚本路径及旧 W7 归因。这些不属于权威写作数据源，但可在文档维护时修正。

上述均属于状态或路径文字，不涉及样本、结果、统计、过滤或指标定义，因此无需推迟论文修订，也无需修改锁定基线中的实验产物。

## 数据锁定规则

自 `7c93b77` 起：

- `r7_main/`、`tts_control/`、R1–R5 原始结果和统计 CSV 冻结；
- Table III–VIII、Fig.6 和正文数字只引用锁定写作总册及结果文件；
- 新增数字、样本过滤、统计方法、指标定义或实验条件变化必须书面变更；
- 不改变数据边界和数值的拼写、路径、状态及标签说明可登记后直接修正；
- 不得恢复旧 TTFA 14.79s/22.67s、CV<5%、AISHELL CER 6.72% 或“统计不可区分”等撤销口径。

## 最终结论

**未发现剩余的数据问题。`7c93b77` 正式确认为最终数据锁定基线，可以全面进入 `main.tex` 修改与最终审稿回复撰写。**

后续文档清理可与论文修改并行进行，不构成前置条件；若论文编写过程中发现需要新增数字或改变统计口径，则必须暂停对应论断并走书面变更，而不能直接改动冻结结果。