# r7_main / r7_tts_control 流程越界裁量复核（2026-08-22）

- **审查对象**：`experiments/review/20260821-PRE-PAPER-AUDIT/results-qa-r7-main-20260822.md`
- **审查事项**：r7_main 完成后，在审查方尚未确认结果级 QA、且未取得单独书面放行前执行了真实 `r7_tts_control`。
- **裁量结论：认定为非实质性流程偏差；本次不要求因该偏差重跑 32 条控制数据，但不得将其表述为事前已获授权。允许在论文中有条件采信，须完成偏差登记和证据链保留。**

## 1. 事实认定

流程要求是：

```text
Gate 放行 → r7_main → 结果级 QA 通过 → 单独书面放行 → r7_tts_control
```

实际顺序是：

```text
Gate 放行 → r7_main → r7_tts_control → 本机结果级 QA 核验
```

因此，开发侧对“r7_tts_control 需单独放行”的理解确实造成了一处流程越界。handoff §3 当时漏写“须单独书面放行”是流程文件缺陷，但不能改变既有放行函已经明确限定的授权边界。

该行为应记录为：

> `tts_control` 在完成 r7_main 后、完成审查方结果级 QA 和单独书面放行前执行；属于未经授权的提前执行，不构成事后追认，也不应在记录中写成“按已获批准流程执行”。

## 2. 为什么不要求重跑

本次判断不是因为流程越界可以忽略，而是因为现有证据表明该越界没有造成数据或条件污染：

1. **r7_main 结果级 QA 已通过**：47/47 项通过，140/140 记录成功，`validate_record` 全量重放无违规，A/B 配对、重复子集、TTFA 非负、hash 绑定和分层覆盖均通过。
2. **控制数据自身完整**：32/32 success、error=0，`control_from_sha256` 与主 checkpoint 精确匹配，控制文本哈希全部可回溯到主实验同一样本的首句/全文。
3. **控制调用范围与预定设计一致**：10 个匹配样本，中文/英文各 5；每个样本 B 首句、A 首句、A 全文各 1 次，加中英校准句 2 次，共 32 次。
4. **数值可复算**：`tts_request_start → first_pcm` 均值 7076 ms 与 RUNINFO 一致，请求时间链单调，绑定了 `c9437c3`、platform conditions hash 和 control-from hash。
5. **代码变更不影响测量逻辑**：r7_main/control 使用的实验代码与 Gate baseline `b8893d6` 在 `run_ttfa_unified.py`、`src/` 和 sample list 上无差异；后续 mkdir 修复只影响控制模式输出目录创建，不改变 TTS 请求或计时逻辑。
6. **无“先看到结果再挑样本”的证据**：控制样本按脚本固定规则从 repeat-0 成功配对中选择，中文/英文各 5；选择哈希和主 checkpoint 已绑定。

在这种情况下，重跑 32 次不会修复已发生的授权时序问题，只会生成另一批受网络、服务负载和设备状态影响的控制测量。若重跑数据与现有数据不同，也不能证明现有数据错误。因而没有足够的科学或审计理由把现有干净数据判为无效。

## 3. 对“采信”与“重跑”的裁决

### 3.1 本次裁决

**采信现有 32 条控制数据，但以“流程偏差豁免后的正式结果”身份采信，不以“事前已获单独放行”身份采信。**

这是一项审查裁量/偏差豁免，不是对越界行为的事后追认。其含义是：

- 认可数据的科学有效性和可审计性；
- 不认可当时的执行权限解释；
- 将越界作为方法与流程偏差留档；
- 不要求仅为修复权限顺序而重复测量。

### 3.2 采信的边界

现有控制数据可以用于：

- 论文 TTFA 组成或 TTS 控制实验的正式统计表；
- `r7_main` 同样本、同文本、同平台的匹配文本 TTS 对照；
- 审稿回复中关于“控制请求/全文长度或首句文本差异”的证据。

但论文和回复信必须避免以下说法：

- “控制实验已在 r7_main QA 通过并获单独放行后执行”；
- “全部实验严格按预注册/预授权顺序执行”；
- 用本次审查裁量掩盖 handoff §3 漏标或执行顺序错误。

建议在内部结果说明或补充材料中加入一句：

> `r7_tts_control` was launched after completion of `r7_main` but before the separately required written authorization and reviewer QA sign-off. The run was retained under an explicit procedural-deviation waiver because post-run audit found exact checkpoint/text/hash binding, 32/32 successful calls, and no code or platform divergence affecting measurement validity; this waiver is not retroactive authorization of the original execution.

若主论文篇幅不适合写入，可至少在审稿回复、实验归档和修订 changelog 中保留该披露。

## 4. 必须补做的治理动作

虽然不要求重跑控制数据，但正式采信前仍应完成以下记录：

1. 将本裁量报告与 `results-qa-r7-main-20260822.md`、control RUNINFO、checkpoint、CSV 一起归档；
2. 在 `REVISION_CHANGELOG.md` 新增一条：
   - 越界事实；
   - handoff 漏标责任；
   - 数据级 QA 通过；
   - 审查裁量为非实质性偏差豁免；
   - 不要求重跑；
   - 不构成事后追认；
3. 把 handoff §3 的“须另行书面放行”修正保留在正式版本中；
4. 将 `r7_tts_control` 的运行提交、control-from hash、platform hash、文本哈希和 32 条 CSV 固定为不可变归档；
5. 论文数据表和 `PAPER_WRITING_REFERENCE.md` 只能引用本次已审查通过的 control 结果，不能引用旧 TTFA 表或未 QA 的估计值；
6. 今后任何后置实验必须取得独立书面放行，不能依据“前一阶段已经完成”自行推断授权延伸。

## 5. 关于是否应在单独放行后再重跑

若需求方把“程序合规性”设为绝对硬门槛，而非允许审查裁量的治理要求，则可以选择重跑；但那属于**更严格的合规补救**，不是本次证据显示出的科学必要性。基于当前材料，我不建议把重跑设为强制条件，理由是：

- 重跑无法改变历史上的流程事实；
- 现有数据已与主 checkpoint、文本、平台和代码严格绑定；
- 结果级 QA 已证明不存在数据污染、样本选择漂移或代码分叉；
- 仅为重新获得一个形式上的事前授权而重跑，会增加随机服务延迟噪声，却不增加有效科学信息。

## 6. 最终裁决

**`r7_main`：通过。**

**`r7_tts_control`：数据级 QA 通过；流程上存在未经单独书面放行的提前执行。经本审查裁量，认定为非实质性流程偏差，允许保留并采信，不要求重跑；但必须按偏差豁免披露，且不得声称事前已获授权。**

当前可以进入结果汇总和论文修订前的最终数据锁定流程，但在正式修改 `main.tex` 之前，仍需完成：

- 本裁量及偏差登记；
- control 结果纳入写作参考文档；
- `EXPERIMENT_DESIGN.md`、`PAPER_HANDOFF.md`、`REVISION_CHANGELOG.md` 的口径同步；
- 确认旧 TTFA 结果不会与新的 r7_main/control 数据混用。
