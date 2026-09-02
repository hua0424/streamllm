# E1/E2 确认性 campaign 设计侧验收（2026-09-02）

> 设计侧对 GPU 端回传的 campaign `e1e2c_b8c758b_20260901T173306Z`（ACCEPTANCE.md 状态 `accepted`，GPU 机侧硬性检查全过）进行独立终审。本文件不修改 campaign 目录内任何被 `checksums.sha256` 覆盖的文件。

## 1. 身份

- Run：`e1e2c_b8c758b_20260901T173306Z`；pilot 独立目录 `…-pilot`（30 条，非 formal，无 campaign manifest）。
- 实验代码 commit：`b8c758bd8e97e519f041ac047d4f6c5f85697bc7`；结果入库 commit：`62508dc79a8843e5dbe58677750f2c22010a1e44`（仅新增 campaign/pilot/GPU_RUN_NOTES 追记，未触碰代码、论文与旧结果）。
- config hash `940ff45e…5703`；campaign identity `897b24fb…75a1`；manifest content hash `b307e054…f146`；manifest SHA-256 `2f4bd76e…f4ed8`（五 session 与 5000 条 records 一致引用）。
- holdout SHA-256 `e86c0ccb…4161`（git blob 口径）；trigger cache SHA-256 `64008c92…35da`，222 entries，input SHA 匹配。

## 2. 独立复算结论（与 analysis_v1.json 及 GPU 汇报一致）

原始 5000 条 records 逐条复算：网格 5×100×10 完整；session_index={0..4}；五进程唯一；单一 campaign manifest/identity/trigger/input 哈希；时间恒等式（`oracle_preaccept_processing = endpoint−arrival`、`arrival_to_ready = ready−arrival`、`waste_denominator = wasted+final`、survived⇒`TTFT_eff=0`）零错误；`excluded_records=0`。

配对 n=500（单位 ms）：

| 比较 | 实际墙钟 arrival→first-token-ready | oracle TTFT_eff（时延乐观下界/推测收益上界） |
|---|---|---|
| C-E1 A−B@0.92 | A 27.70 / B 62.38；差 **−34.69**（CI [−35.30, −34.11]，B 更慢） | **+17.44**（CI [16.12, 18.75]） |
| C-E2 never−B@0.92 | 62.35 / 62.38；差 −0.03（CI [−0.55, +0.51]，含 0） | **+20.80**（CI [19.50, 22.10]） |

B@0.92：pooled waste（wasted/(wasted+final)）2.85%（CI [0.020, 0.037]）；survival 67.0%（CI [0.628, 0.712]）；ready tokens 中位 12；候选首 token 领先端点中位 291 ms；未存活 on-demand 均值 31.09 ms（与 never 的 oracle 均值 31.06 一致）。九点 waste/survival 单调；arrival→ready 全条件平坦于约 62 ms。

## 3. 完整性与保护审计

- 旧 `exp1_latency.json`/`exp2_tradeoff.json`/`paper2_reanalysis.json` 的 git blob 与 GPU 端跑前/跑后 SHA-256 逐字节一致。
- `checksums.sha256` 72 文件对 git blob 全部验证通过。
- holdout 100 条与旧 E1/E2 record ID 及固定轨迹 E3 sample ID 的交集为 0，对话级交集亦为 0；排除源 schema/非空校验通过（old_e1=100、old_e2=106、E3=100）。
- 本机 Windows `core.autocrlf=true` 导致工作树字节哈希与 GPU 端 LF 原文不同；所有哈希核验均改用 git blob 口径后全绿，属检出转换假象，非数据问题。
- session manifests/records 记录真实 `resolved_dtype=torch.bfloat16`、`attention_backend=sdpa` 且五 session 一致；campaign manifest 的 runtime_metadata 为 null 系其采集阶段（模型加载前）如实记录，validator 已通过，接受该说明。

## 4. 科学判读

1. **实际墙钟口径**：同步 harness 中 B@0.92 的最后段到达→首 token 就绪为 62.38 ms，慢于一次性 prefill 的 27.70 ms（结构性，非噪声）：A 的关键路径是单次批量 prefill＋首 token 采样；B 在最后段到达后需依次完成该段增量 prefill、assistant role 边界注入、首 token 采样（≈两次串行前向），短文本下单次前向固定开销主导。
2. **oracle 口径**：67% 场景接受前已有就绪候选（中位领先 291 ms），TTFT_eff=0；均值 10.26 ms。该口径量化"用户在触发后继续说话、真值端点晚于同步端点"时的可立即交付收益，是推测收益的上界。
3. **C-E2**：推测不改变到达→就绪关键路径（差 −0.03 ms 含 0）；2.85% waste 购买的是 oracle 侧就绪概率与 20.80 ms 上界收益。
4. **旧 E1/E2 定性**：旧 0.581 ms/12.1 ms 属 oracle 口径但被误述为墙钟（user_end 记录在同步推测完成后）。确认性 campaign 修正该口径混淆；旧结果保留为探索性 campaign 审计。

## 5. 允许与禁止的主张

允许：上述双口径数值、CI、机制解释（限于本 harness/模型/硬件）、九点权衡、"旧口径 artifact 已修正"。

禁止：把 62.38→10.26 或 +17.44/+20.80 说成实际墙钟改善；把 arrival→ready 平坦说成推测无用（oracle 侧收益另有条件成立）；声称真实 ASR、在线 TEN runtime、在线 TTS、播放器/声卡、声学端点、mouth-to-ear、生产端到端或 0.92 为部署最优；池化跨 campaign 绝对时间。

## 6. 决定

接受该 campaign 为 E1/E2 的确认性正式证据（第四个独立 campaign：旧 E1/E2/A2、固定轨迹 E3+联合 A1、prepared-state P1 v2、本次 C-E1/C-E2）。授权按上述口径替换论文权威 Markdown 的 E1/E2 数字并重画图 6-2/6-3；旧 E1/E2 降级为探索性 campaign 引用。IEEE 衍生稿继续等待权威 Markdown 稳定后整体同步。
