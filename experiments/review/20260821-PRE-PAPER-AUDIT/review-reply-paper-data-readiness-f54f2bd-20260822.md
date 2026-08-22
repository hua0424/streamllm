# 数据就绪整改回复函复审（f54f2bd…6069868，2026-08-22）

- **审查对象**：`reply-review-paper-data-readiness-f54f2bd-20260822.md` 及提交区间 `f54f2bd…6069868`。
- **规格来源**：`review-paper-data-readiness-f54f2bd-20260822.md`、`CISR_REVISION_PLAN.md`。
- **结论**：**实验数据已经齐全，可以开始论文正文修改；但 `6069868` 暂不能声明为最终数据锁定基线。Table VIII 有一处分项边界误标，须先修正标签和关联状态文字。该问题不改变任何数值，不需要重跑实验。**

## 1. 已确认关闭的就绪度问题

### 1.1 W7 人工抽检已完成

`MANUAL_SPOT_CHECK_FORM.md` 已记录 5 条样本，覆盖中英文、Long/Very Long/Extra Long；试听者、日期、可懂度、截断、错序、爆音/削波、异常静音、音量、拼接缝和结论字段均已填写，5/5 判定通过。样本 ID、时长和分组与构建 manifest 一致。

该材料可以作为 **manual spot check** 使用，不能称为 human evaluation。备注区为空和表头“初始值”措辞不影响数据有效性，均为非阻塞项。

### 1.2 历史口径已经有效隔离

以下主要冲突已处理：

- CV 改为 ddof=1：B 5.19%/4.05%/10.73%/18.96%，A 5.23%/4.65%/9.92%/14.01%；
- AISHELL sanity CER 的 6.72% 标为旧口径，终版使用 10.73% 并保留数字写法失配说明；
- 旧 TTFA 装配 B 14.79s/A 22.67s 明确作废；
- Table VIII 数据源改为 R7 统一时间轴实测；
- `CISR_REVISION_PLAN.md` 的 R6→R7 映射已更新；
- `EXPERIMENT_DESIGN.md` §5.3 已历史化。

因此旧数据不会再合法地进入新 Table VIII。

### 1.3 Table VIII 的统计数字与 QA 正确

独立复核确认：

- repeat0，n=50/模式，zh/en 各 25；
- 主指标为 `first_playable_pcm_ns - physical_speech_end_ns`；
- mean/std(ddof=1)/P50/P90/P95 数值正确；
- CSV 48/48 行与 checkpoint/运行侧 summary 一致；
- 100 条记录的首尾闭合残差为 0；
- received→playable 差值、checkpoint hash、84/16 截断计数均正确；
- 旧 R6 估计项未混入；
- tts_control 7076ms 未作为 Table VIII 行项，偏差豁免披露仍在。

所以总 TTFA 数字可继续使用：

- B ALL mean 5481.9ms，P50 3113.7ms；
- A ALL mean 22425.7ms，P50 22269.9ms；
- mean 降幅 75.6%，P50 降幅 86.0%，两种表述二选一。

## 2. 唯一必须先修正的问题：Table VIII 分项误标

装配脚本实际计算：

```text
pipeline_input_close_ns - feed_end_ns
```

但表格将其写成：

```text
t_flush_to_close（flush→管线输入关闭）
```

两者不是同一个时间区间。对 streaming repeat0 的独立复算为：

```text
feed_end → pipeline_input_close:       mean 133.014ms
explicit_flush_start → input_close:    mean   0.332ms
feed_end → explicit_flush_start:       mean 132.682ms
explicit flush 本身:                    mean   0.210ms
flush_done → input_close:              mean   0.122ms
```

因此表中 133.0ms **不能解释为 flush 执行耗时或 flush→关闭延迟**。它是“喂入结束→管线输入关闭”的完整等待，其中绝大部分发生在 `feed_end` 到 `explicit_flush_start` 之间。

这不会改变 TTFA 总量或闭合，因为 R7 六段链条本来就是：

```text
physical_speech_end
→ feed_end
→ pipeline_input_close
→ first_model_token
→ text_ready
→ tts_request_start
→ first_playable_pcm
```

### 最小修正要求

无需重算、无需 GPU、无需改 checkpoint。只需：

1. 将 Table VIII 展示标签改为：

   ```text
   t_feed_to_close_wait（喂入结束→管线输入关闭）
   ```

   或使用同义且边界准确的名称；
2. 在装配脚本中将输出层的 `t_flush_to_close` 映射为上述准确论文标签，并注明源 summary 字段名是历史命名；
3. `TABLE_VIII_ASSEMBLED.md`、`table_viii_r7.csv`、`PAPER_HANDOFF.md`、`PAPER_WRITING_REFERENCE.md`、`EXPERIMENT_DESIGN.md` 中若出现“flush→关闭=133ms”均同步修正；
4. 不得在论文中把 133ms 归因为 flush 计算开销；
5. 若论文确实要报告真正的 flush→关闭耗时，应另列约 0.332ms，并相应增加 `feed_end→flush_start` 分项；否则保持六段闭合链、只改标签是最简且推荐的处理。

修正后重新运行装配 QA，数字应完全不变。

## 3. 可忽略或非阻塞事项

### 3.1 W7 备注与表头

不要求补齐备注区，也不要求修改“初始值”表头才能开始论文。但 `MANUAL_SPOT_CHECK.md` 将一条爆音直接归因为“LibriSpeech 源音频属性”，现有记录没有时间位置或源文件与拼接文件对照证据。建议将其软化为：

> 该样本听到爆音/削波，但仍正常可懂；其来源未进一步区分，不影响延迟测量，作为拼接真人朗读语音的质量边界披露。

这比无证据断言“源音频属性”更稳妥。无需重新试听。

### 3.2 总册的历史状态文字

`PAPER_WRITING_REFERENCE.md` 后段仍同时出现“P0 整改中”“装配为唯一剩余动作”“试听待人完成”，随后又写 W7/W8 已完成。数据不受影响，但写作者容易误读。建议把 §十标题改为“P0 整改闭环记录”，并把上述旧状态明确标成历史划线或删除。

这属于文档去歧义，非实验阻塞；可与分项标签修正同一提交完成。

### 3.3 装配脚本目录

`assemble_table_viii.py` 放在 `experiments/results/` 下，不符合 `AGENTS.md`“生成脚本放 `experiments/scripts/`”的约定。推荐移动到 `experiments/scripts/assemble_table_viii.py`，结果 CSV/MD 继续留在 results。

该问题不影响数据或论文结论。如果此时移动会扰动锁定路径，可保留现状并登记为历史分析脚本；不应因此阻塞论文。

### 3.4 方法学同步细节

`EXPERIMENT_DESIGN.md` 已登记 R7 测量方法，但尚未完整列出装配层的 repeat0、ddof=1、线性分位数、1 位小数和 received-only-QA 决策。建议补入 §6.6。该项有助于复现，但不影响现有数字。

## 4. 是否具备开始修改论文的条件

### 可以立即开始

- Abstract、Introduction 和指标定义的结构性修改；
- Table III–VII；
- Fig.6；
- 真实语音、LA、append-only/漂移、tokenizer seam、语义评估、limitations；
- 审稿回复信除 Table VIII 分项名称外的主体内容。

### Table VIII 的限制

- 总量表数字可以使用；
- 组件分解表必须先把 133ms 的标签改准确，之后才可粘入 `main.tex`；
- 在修正完成前，不能把 `6069868` 宣布为最终数据锁定提交。

## 5. 数据锁定与改动控制裁定

### 当前裁定

- `6069868`：**候选锁定基线，暂不最终锁定**；
- 原因仅为 Table VIII 分项边界误标，不是数据错误；
- 不要求补实验、不要求重跑 R7、不要求重跑 TTS control。

### 修正后的锁定规则

在一个后续提交完成以下三项后，可直接锁定，无需再次进行全量实验审查：

1. 133ms 分项改名为 feed_end→input_close 的准确边界；
2. 装配 QA 复跑仍为 4/4，全部数字不变；
3. 清理总册中仍处于“实时状态”语气的旧 P0/W7/W8 文本。

锁定后：

- `r7_main/`、`tts_control/`、R1–R5 原始结果与统计 CSV 全部冻结；
- `main.tex` 只能引用锁定文档和表；
- 排版、措辞和引用位置可正常修改；
- 任何新增数字、指标定义变化、样本过滤变化或统计口径变化必须走书面变更；
- 纯拼写/标签修复若不改变边界和数值，可记录后直接更正。

## 6. 最终结论

**已经具备开始修改论文的条件，且没有需要补跑的实验。**

但是，“数据全部准备完成”应分成两层理解：

- **实验数据层：完成并通过；**
- **最终写作锁定层：还差一处 Table VIII 分项标签修正。**

W7 的空备注、表头措辞、脚本目录、旧状态文字等均可按用户要求视为非阻塞；其中建议至少顺手软化未经证明的“源音频属性”归因。完成 133ms 标签修正后，即可把后续修正提交设为最终锁定基线并全面修改 `main.tex`。