# SCI 3–4 区补实验计划

## 1. 研究目标

### S-E3：固定轨迹一致性

**RQ：** 在同一首轮 assistant token 轨迹、同一片段时间轴和同一注入位置下，保留完整生成历史相对片段级 playback 历史，是否提高后续回复对差异文本的复现率？

- 独立变量：历史条件 `playback` / `generation`；播放位置 25%、50%、75%、clean boundary。
- 配对单位：`(dialogue_id, fraction)`。
- 聚类单位：dialogue。
- 主目标：`delta_text = assistant_ids[heard_token_end:]`，两条件使用完全相同的 target。
- 次目标：字符比例—空白边界 proxy tail + delta。
- 分析 eligibility 按目标独立定义：片段指标仅纳入 `unheard_text.strip()` 非空的 `(dialogue_id, fraction)` pair；proxy 指标仅纳入 `strict_unheard_text.strip()` 非空的 pair。规则与 judge 对同一 target 共用该 target 的 eligibility，但片段与 proxy 的分母可不同。
- 分析器必须逐 pair 验证两条件的 `unheard_text` 完全一致，且 `strict_unheard_text` 也完全一致；只验证布尔 eligibility 相同不足以保证配对 target 正确。
- 历史输出中的 `design.eligible_pairs` / `design.empty_target_pairs` 仅保留为片段 eligibility 的兼容别名；正式分析同时报告片段与 proxy 的 overall/by-fraction eligibility。
- 机制检查：playback 的 condition-local 完整未听片段恒为空；不得将此构造性零解释为经验效果。
- 样本：100 条纯 MultiWOZ 对话，每条至少三个 user turns，共 800 records。
- 解码：首轮和 probes 均 greedy；同一 retained history 只生成一次 probe chain。
- 统计：事件数、Wilson CI、exact McNemar、dialogue-cluster bootstrap 95% CI；分位置为次分析。离线分析产物记录 analyzer/records/judge/manifest 哈希、源 run ID、裁判 prompt/model identity hash、bootstrap 次数与 seed，不记录凭据。

### S-A1：联合 KV 恢复微基准

**RQ：** 联合执行 KV crop 与角色恢复的墙钟中位延迟相对同长度重新 prefill 的比值是多少？

- 上下文目标长度：256、512、1024、2048、4096、8192。
- crop：尾部 32 token。
- warmup：3（正式推荐可改为 5）。
- repeats：20（若时间允许建议 50）。
- 原始路径：crop-only、role-only、joint crop+role、re-prefill。
- 主加速比：`median(re-prefill raw) / median(joint raw)`。
- 报告：median、Q1、Q3、IQR、P90、P95、min、max。
- 不做 outlier trimming，不用 `median(crop)+median(role)` 替代 joint median。

### S-P1：headless 异步控制路径（prepared-state v2）

**RQ：** 无声卡、真实墙钟节拍的播放器收到 stop 后，软件停播确认、时间轴反查和 GPU 状态修正的延迟分布如何？

旧 v1 在 `ensure_full()` 后没有于播放器启动前同步，使尚未完成的准备态恢复在 stop 后首次同步时被计入控制路径；其联合 stop→crop/role 数值仅作协议审计，不进入论文。v2 采用以下锁定协议：

- 上下文：512、2048、8192。
- 位置：25%、50%、75%；6 个等长片段下，25%/75% 位于片段内，50% 位于片段边界。
- warmup：每个存在未完成正式记录的 `(length, fraction)` 单元 3 次，不写入 `records.jsonl`。
- repeats：20，共 180 个正式 events。
- 主播放条件：24 kHz、20 ms block、0.8 s synthetic duration、6 fragments。
- 每个 trial 在 `player.start()` 前完成 `ensure_full()` 与设备同步；准备耗时、stop 后同步耗时分别记录。
- 指标：setup、stop ack、post-stop sync、leaked samples、lookup、crop-only、role-only、joint crop/role、stop→crop、stop→role、wakeup error。
- player 使用绝对 deadline、进度条件通知和可中断 `Event.wait()`，TTS chunk 与播放 block 分离。
- 本实验不使用声卡，不测 DAC/hardware buffer，不测在线 CosyVoice cancellation。

## 2. 数据完整性

正式 E3 必须显式传入 `p2_turns.json`，并满足：

- 恰好或至少 100 条正式对话；
- ID 唯一；
- 每条至少 3 个非空 user turns；
- 无任何 ID 以 `fx` 开头；
- manifest 记录文件 SHA-256 和样本 ID。

缺失正式数据时 runner 直接失败，不自动回退 fixture。

## 3. 断点续传

- `manifest.json` 首次创建后不可变。
- Resume 时比较 config hash 和输入 SHA-256；不同则拒绝。
- E3 trajectories 与 records 使用 JSONL 逐条 fsync；已完成 trajectory 不重新生成。
- E3 probe cache 可从已有 records 重建；相同 history 不重复解码。
- A1/P1 使用实验单元 key 跳过已完成组合。

## 4. 预注册式验收

### S-E3

- 每条 dialogue 只有一个 trajectory ID。
- generation 四位置只有一个 history key。
- 同一个 pair 的两条件共享 trajectory ID，并逐字符串验证片段 target (`unheard_text`) 与 proxy target (`strict_unheard_text`) 各自完全一致。
- 片段/proxy eligibility 分别由对应 target 的非空白性决定并独立报告 overall/by-fraction；规则与 judge 的 target-specific `n` 一致。
- playback condition-local 完整未听文本为空。
- 正式 records=800，无 fixture。

### S-A1

- 每点 raw 数组长度等于 repeats。
- joint 路径在同一同步计时区间内执行 crop 与 role recovery。
- 所有时间有限且非负。
- `q1 <= median <= q3`，IQR 可由 raw 精确复算。
- 8k re-prefill median 大于 joint median 是报告性期望；若不成立，照实报告并检查 GPU 干扰。

### S-P1

- 新 run ID 以 `async_prepared_v2` 结尾，不覆盖或续写 v1。
- 180 条正式事件完整落盘，每个单元 20 条；warmup 不落盘。
- 所有记录均在播放器启动前完成准备态同步，并单独保存 setup/post-stop sync。
- stop request/ack 均精确落在目标 sample，leaked samples 为 0，stop 后 sample counter 不再推进。
- 25%/75% 的 `partial=true`，50% 的 `partial=false`；timeline 返回合法 token endpoint。
- 不对真实声卡或声学停止做任何主张。

## 5. 论文更新规则

只有 GPU 正式 run、judge 和分析全部完成后才替换第六章数字。旧实验与新实验不得合并为同一总体。新 E3 应说明解码从旧实验的温度采样改为 greedy，并把“独立生成混杂”限制替换为固定轨迹的配对设计说明。
