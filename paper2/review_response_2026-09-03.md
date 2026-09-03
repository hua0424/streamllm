# 第二次审稿意见修订响应（2026-09-03）

> 对应审稿文件：`paper2/review/sci_q3_q4_full_review_2026-09-02.md`。本记录用于内部修订追踪；目标期刊选定后可改写为正式 response letter。行号以本次重建后的 `paper2/thesis_draft.md` 为准。

## 总体响应

我们接受 Major Revision 的总体判断，并将稿件从宽泛的“播放感知低延迟系统”收窄为**软件播放游标驱动的 TTS 片段—token—KV/role 状态修正**。C2 是唯一核心机制贡献；C1 降为受控同步 harness 中 candidate-selection/oracle/waste 的支持性刻画；C3 为受混杂探索性负扩展。GPU 补强已经结束，无需新增实验。C2 v3 direct crop-integrity evidence 已正式接受；C2 v1/v2 继续保持 rejected，不以 v3 追溯改判。

## Consolidated Major Findings

| Finding | 响应状态 | 修订与证据 |
|---|---|---|
| CF-01 readiness ≠ deliverability | **Resolved** | 全文把 `first_token_ready` 改称 first-candidate-token selection / internal compute-readiness，明确其位于 cache-update forward 和 generator yield 前。`first_deliverable`、consumer marker 与 291 ms 只作同步程序诊断；E1/E2 v2 另报告 B@0.92 的 257.58/265.57 ms diagnostics，不作为生产 headline。 |
| CF-02 crossed design | **Resolved** | 新增 `analyze_v2.py` 和 `analysis_v2.json`，按 5 session 与全局 100 utterance 独立重采样后取笛卡尔积；10,000 次 product bootstrap。正文统一写 100 unique utterances × 5 technical sessions，而非 500 个独立内容样本。 |
| CF-03 C-E1 非 token-equivalent | **Resolved** | 报告 A/B@0.92 full tokens 280/500、first token 465/500、44/100 utterances 分岔；C-E1 改为 implementation-path comparison，差异可能混合 tokenization、forward topology/shape、role boundary、kernel 和 Python scheduling，不再归因于纯 incremental-prefill effect。B@0.92/never 500/500 exact，C-E2 保留为 token-consistent B-path comparison。 |
| CF-04 clean re-prefill equivalence | **Partially accepted; claim narrowed** | v1/v2 clean-reprefill 协议按冻结门槛 rejected；v2 control 与 production forward topology 不匹配，不能识别 crop effect。我们不再声称 clean-reprefill numerical/logit/continuation equivalence。C2 v3 改为 direct crop-integrity addendum：24/27、28 层 K/V、60 recovery steps 全 bitwise/exact。该替代证据充分支持收窄后的 direct-crop/matched-recovery 主张。若未来恢复 clean-reprefill 主张，仍需新可识别实验。 |
| CF-05 EOS/EOT 重复边界 | **Resolved** | 新增完整 token ledger、`RolePhase`、`GenerationEndReason` 与 `ASSISTANT_EOT_PENDING`；预测 EOT 不进入 assistant content/KV/timeline，`reopen_user_role()` 唯一提交 close；full invalidation、p=0 和新 user 内容清除陈旧 `CROPPED` 均有回归与 C2 v3 evidence。 |
| CF-06 E3 非 HCI 证据 | **Resolved** | 新增 `analyze_e3_v2.py` 与 weighting/dedup v2。E3 统一称 fixed-detector-conditioned information-reproduction rate；主表使用 label-weighted point + 同 estimand dialogue-cluster CI，另报告 dialogue-weighted 和 unique-boundary sensitivity；明确 CI 不含 detector/prompt/model/human error，不作 superiority/equivalence/noninferiority/harm/HCI 结论。 |
| CF-07 software/device/acoustic 区分 | **Resolved** | 标题、摘要、形式化、方法、结果和讨论统一区分 software-consumed-sample cursor、device-presented samples、acoustically heard content；本文只观测第一层，保留边界为 TTS-fragment-level software boundary。 |
| CF-08 novelty 检索闭环 | **Partial** | 新增 `docs/novelty_search_2026-09-03.md`：日期、查询族、来源、纳排规则、snowballing 起点和最近邻矩阵。明确 OpenAI/Azure/LiveKit 高层语义和 KV crop primitive 均为 prior art，只保留跨层联合实现的 scoped non-identification。Google/ACL/ACM/部分 arXiv 本轮访问受限，未记零结果；提交前仍应在可访问环境保存检索导出、结果数和逐条排除日志。 |
| CF-09 贡献成熟度不对称 | **Resolved** | 全文统一 C2 核心、C1 支持、C3 探索；RQ5 改为描述当前分数/耗时，不再问“是否改善”。 |
| CF-10 artifact/journal closure | **Partial** | 新增 `REPRODUCIBILITY.md` campaign matrix、E3 exact-rescue README、analysis-only commands 和 `paper2/declarations.md`。两项 analysis v2 已生成并 hash-pinned，随本批提交固化。尚需作者/机构提供 public URL/DOI/release、LICENSE/权利人、派生数据再分发许可、最终 declarations，并在目标期刊确定后生成压缩独立稿。 |

## Minor Findings

| Finding | 状态 | 修订 |
|---|---|---|
| MF-01 阈值计数 | **Resolved** | 统一为“八个数值阈值 + never，共九个工作点”。 |
| MF-02 A1 固定顺序/裁剪量 | **Resolved** | 披露固定 operation order 与固定 32-token suffix，限制外推。 |
| MF-03 P1 P95 | **Resolved** | 明确 n=20/cell，P95 仅为 empirical/descriptive order statistic，不是 SLO。 |
| MF-04 E3 重复边界 | **Resolved** | fragment 297 labels→169 unique groups，proxy 380→379；完整报告去重敏感性。 |
| MF-05 timeline 顺序合同 | **Resolved** | 实现与论文均明确 token/sample 连续、chunk 唯一归属、fragment lifecycle、游标单调与乱序 fail-closed。 |
| MF-06 字段命名强于证据 | **Resolved** | `heard_text/n_heard/strict_unheard` 明确为兼容别名，只代表 software fragment/proxy，不是声学真值。 |
| MF-07 references/declarations | **Partial** | 引文 1–21 无缺失/孤儿；修正 stream2sentence 版本、Mistral-v0.3 model card、Transformers v4.57.1 source、LiveKit 来源。最终期刊格式、2026 preprint 状态和作者 declarations 仍待投稿前确认。 |

## 修订后核心结果

- E1 readiness A−B@0.92：−34.69 ms，crossed 95% CI [−35.44, −33.95]。
- E2 readiness never−B@0.92：−0.03 ms [−0.64, 0.61]；oracle lower-bound：+20.80 ms [17.85, 23.65]；waste 2.85% [1.12%,4.73%]，survival 67% [58%,76%]。
- E3 label-weighted effects：fragment rule −3.37 pp [−10.49,3.40]，fragment judge −2.02 pp [−10.70,6.13]，proxy rule −1.58 pp [−6.08,2.67]，proxy judge −2.63 pp [−8.57,2.90]；fragment judge unique-group effect 0.00 pp [−7.98,7.47]。
- C2 v3：24/24 cases、27/27 crop events、28 层 K/V、60/60 recovery steps、27/27 wrong-length negative controls，全 exact。

## 剩余投稿前事项

这些事项不需要新 GPU 实验，但必须由作者/机构或目标期刊决定：

1. public/anonymous artifact URL、immutable release/tag、DOI；
2. LICENSE、权利人、third-party notices、E3 派生数据再分发许可；
3. ethics/exemption、participants/consent、funding、COI；
4. 作者名单、顺序、通讯作者、CRediT、accountability 与 AI/tool disclosure；
5. 在可访问环境补全检索导出/筛选日志并复核 2026 preprints；
6. 选择目标期刊后统一引用格式，并从当前权威 Markdown 生成独立压缩投稿稿。
