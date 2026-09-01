# P1 v2 验收与论文最终补充计划

## 总体判断

按 GPU 实验机报告，P1 prepared-state v2 已满足预注册式验收：9 个单元、180 条正式记录、120 条片段内与 60 条边界事件、180/180 精确采样目标、零泄漏、准备态同步完成，旧 v1 的异步准备态污染消失。若拉取 `ee1dcc7` 后对 raw records、analysis、manifest、日志和快照的复算一致，**本轮必要 GPU 实验即全部结束，不再补跑 E1/E2/E3/A1/A2，也不再新增 P1 重复**。

`git add -f` 入库不是问题：`.gitignore` 只控制未跟踪文件的默认发现，已经强制加入并提交的结果可以正常被 Git 追踪。保留 tarball/sha256 在实验机作为外部回传校验，同时在仓库保存原始 records、analysis、manifest、日志和快照，审计性更强。论文会说明正式结果经强制纳入版本库，不把这一操作视为协议偏离。

## 1. 拉取并独立验收 `ee1dcc7`

1. fast-forward 本地 `paper2` 到 `origin/paper2`，确认远端提交历史线性且 `ee1dcc7` 只增加 P1 v2 结果、日志、快照等工件，不修改已执行的协议代码或旧结果。
2. 核对 run identity：
   - run ID `sci34_dc52978_20260901_async_prepared_v2`；
   - 执行代码 commit `dc52978`；
   - 结果入库 commit `ee1dcc7`；
   - manifest 为 clean tree、transformers runtime、正式模型和锁定配置。
3. 从 180 条 `records.jsonl` 独立复算：
   - 9 个 `(length, fraction)` 单元，每单元 20 条，key 唯一；
   - warmup 不在正式 records；
   - protocol/prepared-state/trial-kind 均正确；
   - request 与 ack 均等于 target sample；
   - leaked samples 全为 0，停止后游标稳定；
   - 0.25/0.75 全为 `partial=true`（120 条），0.5 全为 `partial=false`（60 条）；
   - 所有 timing 有限、非负。
4. 复核分段计时恒等关系与嵌套关系：stop→sync、lookup、joint crop、joint role 与 stop→crop/role 的逐条差值仅允许序列化舍入误差；不得把累计区间与组件中位数相加。
5. 独立重算九个单元的 median/IQR/P95/min/max，核对提交的 `analysis.json` 与报告区间：
   - stop ack median 0.055–0.062 ms，最大 cell P95 ≤0.077 ms；
   - post-stop sync median 0.167–0.176 ms，最大 cell P95 ≤0.35 ms；
   - lookup median 0.47–0.50 ms，最大 cell P95 ≤0.94 ms；
   - stop→crop median 2.44–2.53 ms，最大 cell P95 ≤3.49 ms；
   - stop→role median 78.6–80.8 ms，最大 cell P95 ≤86.1 ms；
   - setup 41–1717 ms，全部发生在 playback 前并排除于 stop 路径。
6. 核对 CPU/RAM/kernel/NVIDIA driver/两张 RTX 3090、GPU 空闲快照、依赖哈希和运行日志；记录精确 P1 主机 CPU 信息，但不补造旧主机 CPU。
7. 检查 tarball SHA-256 `4c6188249f1226e5692a85468cf1e9c3b05e648494a5ce9a6e5a475b264c0bc8` 是否在日志/说明中有可追溯记录；tarball 本身无需入 Git。

若以上全部通过，形成正式结论：P1 v2 接受；v1 保留为协议失败审计；不需要更多 GPU 实验。若只发现论文/分析汇总舍入差异，离线修正即可；只有 raw records 不完整、计时恒等式失败、GPU 干扰或 manifest 配置不符才考虑重跑。

## 2. 论文中将 P1 作为 RQ4 的第二部分，不新增 RQ

保持五个 RQ 编号不变：

- 6.5 改为“RQ4：KV 状态复用与 prepared-state 软件控制路径”；
- 6.5.1 保留联合 A1 模型侧微基准；
- 6.5.2 新增 P1 v2 headless 软件播放控制路径；
- 6.6 RQ5 和 6.7 总结编号不变。

新增表 6-5，汇总九个 cell 的 median 范围和最大 cell P95：

| 区间 | 九个 cell 中位数范围 | 最大 cell P95 | 解释 |
|---|---:|---:|---|
| 软件停播确认 | 0.055–0.062 ms | ≤0.077 ms | 请求至播放器线程确认 |
| stop 后设备同步 | 0.167–0.176 ms | ≤0.35 ms | 单独计量，不与 setup 混合 |
| 时间轴反查 | 0.47–0.50 ms | ≤0.94 ms | 采样位置至片段/token 端点 |
| stop→crop 完成 | 2.44–2.53 ms | ≤3.49 ms | 累计至同步 KV crop 完成 |
| stop→角色恢复完成 | 78.6–80.8 ms | ≤86.1 ms | 累计至角色边界恢复完成 |

表下注明：n=180；120 个 mid-fragment、60 个 boundary；180/180 精确目标、零软件采样泄漏；setup 41–1717 ms 在播放前完成并排除；累计区间相互嵌套，不能相加；结果是按 cell 汇总，不是跨 campaign 池化分布。

不新增主图：P1 九个 cell 的中位数范围很窄，表格比主图更清晰；图 6-1 至 6-4 编号保持稳定。详细九单元分布由入库 `analysis.json` 和原始 records 提供。

## 3. 更新权威论文源

### 摘要

中英文摘要在 A1 后增加一句紧凑的 P1 结果：180 次事件、精确目标、零泄漏，stop ack / stop→crop / stop→role 的中位数范围和最大 cell P95；立即限定为 headless 软件播放控制路径，不是声学停播或生产端到端 barge-in。摘要不写 41–1717 ms setup 细节，只说明播放前 setup 单独计量并排除。

### 第一至第五章

- 第一章 C2 增加从模型侧 A1 到软件控制路径 P1 的证据闭环，不把 P1 称为真实用户听觉验证。
- 第三章定义 stop acknowledgment、post-stop sync、timeline lookup、stop→crop、stop→role、setup 排除、精确目标与泄漏检查；RQ4 映射到 A1+P1。
- 第四章说明 prepared-state barrier 和有序计时边界；Mock TTS 仍不代表真实音频，但 P1 已覆盖墙钟软件播放器调度。
- 第五章记录 P1 的 180 条 raw records、analysis、manifest、日志和环境快照；补实验目录由“协议待运行”改为“已执行并归档”。

### 第六至八章

- 第六章新增 §6.5.2 和表 6-5，RQ4 总结同时区分 A1 模型侧计算与 P1 软件控制路径。
- 第七章删除当前叙述中的“P1 v2 pending”，保留 v1 失败原因作为审计历史；讨论 CPU/OS/thread scheduler 敏感性，不声称上下文长度无影响或硬件不变性。
- 第八章把 P1 写入限定性结论；未来工作从“先重跑 P1”改为“真实 ASR/TTS/队列/声卡/声学停止和生产并发闭环”。固定轨迹 A2、人工双标、词/token 物理对齐、跨模型/语言和 E4 仍保留。
- 更新 `outline.md` 后通过生成脚本重建 `thesis_draft.md`，不直接手改合并稿。

## 4. 三个 campaign 与 CPU 差异写法

正文从“两轮 campaign”改为三个独立工件组：

1. 旧 E1/E2/A2；
2. 固定轨迹 E3 + 联合 A1；
3. prepared-state P1 v2。

三者均使用双 RTX 3090，但 CPU 主机不同。只在各自协议内部报告绝对墙钟分布，不池化、不直接用 A1 的 31–48 ms 与 P1 的 79–81 ms 相减或解释为固定“系统开销”。P1 中软件停播、timeline、Python 线程和部分 launch 路径对 CPU/OS 调度敏感；setup/role 又包含 GPU 工作，因此也不称为纯 CPU 基准。

## 5. 决策、上下文和实验文档

1. 新增 D-015，而不改写 D-014 的历史：记录 P1 v2 接受、run/commit/hash、180 条验收、五项延迟、setup 排除、第三 campaign 和主张边界；D-015 仅替代 D-014 的 P1 pending 状态，不改 E3/A1 决策。
2. `paper2_context.md` 追加新里程碑，旧 D-014 行保留为当时历史；下一里程碑改为生产音频闭环、固定 A2 和人评。
3. 新增 P1 v2 验收补充报告，保留原报告中 v1 失败与当时 pending 状态，不静默覆盖审计历史。
4. 更新 `CLAIMS_MATRIX.md`、`EXPERIMENT_PLAN.md`、`README.md`：把 P1 v2 从条件式“完成后”改为已接受证据，保留禁止声学/生产外推的红线。
5. `P1_PREPARED_RERUN.md`、`GPU_RUNBOOK.md`、`HANDOFF_FOR_GPU.md` 增加“已完成/仅供复现”标记，防止下一位 agent 再次重跑；命令和红线仍保留。
6. `GPU_RUN_NOTES.md` 在 v1 诊断之后追加独立 v2 结果与验收，不删除 v1。
7. 记录结果由 `git add -f` 入库的原因；保持 `.gitignore` 规则不变即可，已追踪文件不会因 ignore 丢失。

## 6. 验证与交付

- 用原始 records 独立复算并与 `analysis.json` 一致性比较。
- 运行 Python 编译、SCI smoke、timeline smoke、P1 数据审计脚本。
- 重建并 `--check` 完整合并稿。
- 扫除当前态中的 `P1 v2 pending`、`先重跑 P1`、“协议有效异步路径尚未测量”等陈旧表述，同时保留历史日志中的 dated pending/v1 failure。
- 检查摘要/表 6-5/讨论/结论数字一致；所有图片链接、引用、Whisper [17]、陈旧 E3/A1 数字和 `git diff --check`。
- 不修改或运行 IEEE 衍生稿；其同步继续等权威 Markdown 最终稳定后整体进行。

完成后，若 P1 审计通过，将明确报告：**SCI Q3/Q4 目标下本轮必要补实验全部完成，无强制新增 GPU 实验；论文可进入导师审阅与格式/引用定稿阶段。**