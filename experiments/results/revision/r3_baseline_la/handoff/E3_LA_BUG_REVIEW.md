# E3-LA 结果无效问题审查交接文档（DEV-3 下标错帧 bug）

> **读者**：代码审查人员。
> **目的**：请审查本文对 `src/asr/local_agreement_streamer.py`（DEV-3，LocalAgreement-2 实现）的 bug 定位与修复建议，
> 确认后由开发方修复并授权重跑 E3-LA。
> **状态**：E3-LA 已于 2026-08-20 16:16 跑完（498/498，error 0），但结果因本 bug 无效，
> 结果文件已标记 `*.INVALID_dev3_frame_bug` 保留在 `r3_baseline_la/` 下备查。
> 相关记录：`../../REVISION_CHANGELOG.md`（2026-08-20 条目）、`../../PAPER_IMPACT_NOTES.md`（影响项 4）。

---

## 1. 问题现象（E3-LA QA 发现）

| 指标 | 实测 | 预期/参照 |
|---|---|---|
| LA WER mean | **0.545**（zh 0.579 / en 0.476） | 同引擎 System B 量级应 ≪ 此值 |
| LA/System B 转写长度比 | mean 0.52，median 0.47，**79% 样本 < 0.7** | ≈1.0 |
| LA TTFT mean | 2500.4ms（long/very_long/extra_long = 1773/2331/2934ms） | 与 System B 1573.9ms 同机可比，但因丢文本，对比无意义 |
| 分歧事件 | mean 6.0 次/样本，max 19 | 预期偶发 |

典型样本（crosswoz_10296_turn2，26.7s，WER=0.88）：
- LA 输出（24 字符）：`'你好,我想找个高档型的酒店住 支持明镜与点点栏目'`
- System B 输出（118 字符）：`'你好,我想找个高档型的酒店住宿,我希望酒店的最低价格是500到600元。有什么合适的吗 请推荐您去…'`
- 参考文本要点：`…酒店住宿,我希望酒店的最低价格是500到600元。有什么合适的吗…`

LA 输出的正确开头之后直接丢失中段 `'我希望酒店的最低价格是500到'`，尾部为噪声段幻听残留。

## 2. 根因定位（逐轮重放实证）

对该样本用相同分段器逐段重放 LA（trace 如下，关键三行）：

```
[段3] buffer=10.07s -> 提交 '你好,我想找个高档型的酒店住宿' (12 词), n_committed=12
      → _trim_buffer() 裁剪至最后提交词 end−0.1s，buffer 剩 5.8s
[段4] 对裁剪后 buffer 重解码 -> 假设 H2(20 词)，H2[0]≠H1[0]，agreed=0 < committed=12（分歧 WARNING），无提交
[段5] LCP(H2,H3) 重新延伸 -> 提交 H3[12:20]='600元有什么合适的吗?'，n_committed 12→20
```

**机制**：`_trim_buffer()` 把音频缓冲裁到"最后提交词 end − 0.1s"之后，下一轮 `model.transcribe(buffer)`
产出的假设 H2 是**尾段帧**的词序列——H2 的词 0 对应未提交区域的第一个词。
但 `self.n_committed=12` 仍是**裁剪前全序列帧** H1 的下标。第 5 段提交时执行
`agreed[self.n_committed:]`（即 H3[12:]），把 H3 前 12 个词（=`'我希望酒店的最低价格是500到'`，
**尚未提交过的内容**）当作"已提交"跳过 → 中段文本静默丢失。
每次裁剪后错帧都会再丢一段，累积造成 79% 样本大比例丢文本；`flush()` 的
`prev_words[self.n_committed:]` 在同一错误下标上截取，导致尾部丢失或把幻听残留当作尾部提交。

代码位置（`src/asr/local_agreement_streamer.py`）：
- `feed_segment()`：`agreed` 计算（L117-121）、`agreed[self.n_committed:]` 提交（L141-145）、`self.prev_words = cur_words`（L156）
- `_trim_buffer()`（L192-206）：裁剪 buffer 并前移 `prev_words` 时间戳，**但未重置 `prev_words`/`n_committed` 的序列帧**——前移时间戳只解决了时间轴，没解决"新假设只覆盖尾部、词数与旧假设不同"的计数帧问题。
- `flush()`（L160-168）：`prev_words[self.n_committed:]` 继承同一错帧。

与 ufal/whisper_streaming 原版对照：LA-2 在缓冲裁剪后丢弃/重建假设簿记（committed 文本独立保存、
假设比较只在同帧内进行），本实现漏掉了这一步。

## 3. 为什么判定结果无效

- 提交文本大面积中段丢失 → `transcribed_text`、WER/CER 全部失真（偏高）；
- TTFT 定义虽仍成立（first_token − audio_end），但 LA 队列工作量分布被错误提交/分歧重解码改变，
  且"LLM 收到的文本"与真实策略行为不符 → TTFT 数字不可作为策略对比证据；
- System A/B 本机重跑（`system_ab_rerun/`，996 条 error 0）**不受影响、仍然有效**——它走的是
  run_exp_latency 的 System B 路径，与 LA 组件零耦合；E1/E2/E4/E5 同理不受影响。

## 4. 修复建议（供审查与开发方参考，未实施）

原则：裁剪后让"已提交"的判定不依赖跨帧下标。

1. `_trim_buffer()` 裁剪后：`self.prev_words = []`、`self.n_committed = 0`（帧重置），
   并保留 `self.last_committed_end` 的新帧值（现有代码已前移，保留即可）；
2. 提交条件由"下标 `agreed[n_committed:]`"改为"时间下限"：只提交满足
   `word.end > last_committed_end + ε`（新帧）且 `word.end ≤ buffer_duration − trailing_margin` 的词；
   LCP 仍用于防回滚（只提交与上轮假设一致的稳定前缀），但提交起点用时间戳而非下标；
3. `flush()` 同理改为按时间下限截取 `prev_words` 中 `end > last_committed_end + ε` 的词；
4. 回归验证建议：
   a. 用本样本能复现修复前后差异（修复前丢中段，修复后完整）；
   b. 追加单元测试：模拟"提交后裁剪→新假设前缀不同→再提交"序列，断言提交文本拼接后与最终假设一致、无跳段；
   c. 修复后 E0 冒烟（`--max-samples 2`）确认 WER 回到 System B 同量级，再重跑 E3-LA 全量（约 6h）。

## 5. 重跑前置条件

- [ ] 审查确认本定位与修复方向；
- [ ] 开发方修复 DEV-3 并 push（本机 `git pull` 获取）；
- [ ] 修复后回归套件 `test_revision_regressions` 10/10 + E0 冒烟通过；
- [ ] 重跑 E3-LA：`uv run python -m experiments.scripts.run_exp_baseline_la --dataset all --sample-list experiments/results/revision/r3_baseline_la/exp2_ablation_sample_list.json --asr-device cuda:0 --llm-device cuda:1 --output-dir experiments/results/revision/r3_baseline_la --no-resume`（需删除/移走现有 checkpoint.json 与 INVALID 结果，避免续传混淆）。
