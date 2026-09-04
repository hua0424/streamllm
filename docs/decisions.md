# 二期工程决策日志

每次做技术决策时按时间倒序追加一条。每条包含：**日期 / 决策 / 背景 / 理由 / 影响 / 状态**。
状态：`proposed` / `accepted` / `superseded`（被后续决策替代时填写替代条目的日期与编号）。

---

## D-025（2026-09-04）落实内部初稿审阅并以 C2→E3→C1→C3 重组论文

**决策**：接受 `paper2/review/paper2_internal_draft_review_2026-09-04.md` 的定位与报告意见，在不新增或重跑实验、不改 accepted/rejected verdict 的前提下，修改权威摘要与第一至八章。采用标题“级联式语音对话打断中的上下文状态修正：从软件播放游标和 TTS 片段到 KV 与角色恢复”；将研究问题与第六章顺序调整为 C2 核心→E3 downstream 支持→C-E2/C-E1 supporting characterization→A2 exploratory description。

**口径收束**：C2 抽象为 external-progress-conditioned joint prefix-state repair 的四层合同，并只主张 direct crop integrity 与同一 accepted run 内的 within-run matched-arm recovery exactness；v1/v2 保持 rejected。E1/E2 改称同步分段文本 harness 中的 pre-oracle-acceptance candidate generation、endpoint candidate availability 和 pooled discarded-token ratio；C-E1 保持非 token-equivalent implementation-path comparison。E3 将“semantic group”更正为 target-specific exact-key grouping/deduplication，补齐 label-weighted estimand、dialogue/exact-key sensitivity 与 rule/judge 操作定义；A2 不再称负结果。

**篇幅与工件**：第五、七、八章下沉重复的 run/hash/CRLF/项目管理信息，保留 `REPRODUCIBILITY.md` 稳定入口和紧凑环境表；英文摘要压至 296 词。`thesis_draft.md` 由 10 个权威源确定性重建并通过 `--check`，参考文献 [1]–[21] 均被正文引用。当前结论边界不需要新 GPU、声学或 HCI 实验。

**状态**：accepted（论文内部审阅修订完成；作者/机构/目标期刊声明仍按既有边界待确认）

---

## D-024（2026-09-03）完成二审统一修订并冻结两项离线 analysis v2

**决策**：在 D-023 解除 GPU 阻塞后，不再新增或重跑实验；从 accepted raw 工件生成两个不覆盖历史分析的 versioned 离线结果，并以 `docs/paper2_revision_evidence_2026-09-03.md` 作为全文唯一数值/主张合同，统一修改权威摘要、第一至八章、参考文献、实验图和合并稿。

**E1/E2 crossed reanalysis**：新增 `e1e2_confirmatory/analyze_v2.py` 与 `analysis_v2.json`（SHA-256 `9bce6db5…4456`）。设计明确为 100 unique utterances × 5 process sessions 的 crossed panel；10,000 次 product bootstrap 独立重采样全局 session 与 dialogue 后取笛卡尔积。正式差值/95% CI：C-E1 candidate readiness A−B@0.92 = −34.6877 ms [−35.4421, −33.9535]；C-E2 never−B = −0.03349 ms [−0.63861, 0.61494]；oracle 下界分别 +17.4367 ms [14.4079, 20.3234] 与 +20.8037 ms [17.8492, 23.6450]；B@0.92 pooled waste 2.8527% [1.1239%,4.7345%]，survival 67% [58%,76%]。A/B 完整 token 280/500、首 token 465/500、44/100 unique utterances 分岔，故 C-E1 只作 implementation-path comparison；B@0.92/never 500/500 exact。

**E3 weighting/dedup reanalysis**：新增 `analyze_e3_v2.py` 与 `analysis_weighting_dedup_v2.json`（SHA-256 `5776db23…0366`）。主表采用 label-weighted 点估计及匹配的 dialogue-cluster bootstrap：fragment rule −3.37 pp [−10.49,3.40]、fragment judge −2.02 pp [−10.70,6.13]、proxy rule −1.58 pp [−6.08,2.67]、proxy judge −2.63 pp [−8.57,2.90]。同时报告 dialogue-weighted 与 target-specific exact semantic-boundary sensitivity：fragment 297 labels/96 dialogues→169 groups（judge 0.00 pp [−7.98,7.47]），proxy 380/100→379 groups。全部结果仅为 fixed-detector-conditioned dialogue-sampling uncertainty，不支持优效、等效、非劣、伤害或无效应主张。

**全文修订**：C2 成为唯一核心贡献，C1 降为 candidate-selection/oracle/waste 支持性刻画，C3 为受混杂探索性负扩展；统一 software-consumed cursor、TTS-fragment retention、device-presented/acoustically-heard 分层；`first_token_ready` 改称内部 candidate selection/compute-readiness，first-deliverable/consumer 仅作同步 harness diagnostic；C2 v1/v2 保持 rejected，v3 仅主张 direct crop integrity/matched recovery exactness；RQ1–RQ5 在第一、六、七、八章一一对齐。图 6-1～6-3 改读两份 analysis v2 并重画中英文版本，视觉验收通过；`thesis_draft.md` 由 10 个权威源确定性重建。

**novelty/artifact**：新增 dated targeted public-source scan，承认 OpenAI/Azure/LiveKit 高层 prior art 与 KV crop primitive prior art；因 Google/ACL/ACM/arXiv 部分渠道访问受限，只作 scoped non-identification，不称 systematic/exhaustive/global first。新增 `REPRODUCIBILITY.md`、E3 exact-rescue README 与 declarations draft。公开 URL/DOI、LICENSE/权利人、派生数据再分发、伦理/consent、funding、COI、作者/CRediT 与 AI disclosure 必须由作者/机构确认，不由仓库推断。

**状态**：accepted（技术与论文实证修订完成；待作者补投稿元数据、选择目标期刊格式并提交本批 commit）

---

## D-023（2026-09-03）接受并封存 C2 v3 crop-integrity 正式证据

**决策**：正式接受 run `c2crop_82103004_20260903T080512Z`（code commit `82103004637dce8f98688f4a685d33ebee363a3b`、结果 commit `7d50624`、manifest SHA-256 `d8c3db4d…d4bc2`）作为 D-018 EOS/EOT/role 修复后的 crop-integrity 正确性证据。24/24 ordered cases、27/27 ordered crop events 全部 exact，validation `ok=true/acceptance_eligible=true/errors=[]`，analysis accepted、ACCEPTANCE 含独立 `Status: accepted`，30-file seal 验证通过（seal SHA-256 `e0997d41…f4a9`）。C2 v1/v2 verdict 不变，均保留 rejected 描述性证据。

**设计侧独立复核**：从 raw `records.jsonl` 与 Git blob 重算 24/27 网格、record content hash、308 次第一轮 tokenwise production append、3 份第二轮逐-token ledger、27 个 wrong-length negative control、3 个 no-op crop、60 个 recovery steps及全部 28 层 K/V manifest/aggregate hash。所有 event 的 pre-crop retained prefix、production post-crop 与独立 clone-oracle 三方逐层 shape/dtype/device/hash exact；所有层 `shape[-2]==keep`；crop 后 mask/token/seq/KV exact；recovery 每步 production/direct-oracle 的 K/V/logits/mask/ledger/retained prefix 与 operation-derived role/end/content state exact；unique EOT 与 final canonical ledger exact。381 个既有结果文件 before/after guard 完全一致。模型/环境身份为 accepted Qwen2-7B artifact `fae2ece1…`、Qwen2ForCausalLM/Qwen2、BF16、SDPA、严格离线、clean commit；v2 只作 provenance 且无 runtime dependency。

**封存复核注意**：Windows checkout 会因 `core.autocrlf` 把结果文本转为 CRLF，直接对工作树跑 byte-level cases/seal 校验会误报。Git blob 30/30 与 seal 匹配；在 `core.autocrlf=false` 的 LF 保留临时 clone 中，formal validator 复跑得到 `ok=true/errors=0/cases=24/events=27/all_exact=true`，seal verify 得 `files=30/ok=true/seal_sha256=e0997d41…f4a9`。后续复核必须使用 LF 保留 checkout、Git blob 或原 tarball，不得把 CRLF 工作树误报解释为工件损坏。

**允许主张**：仅限冻结 Qwen2-7B snapshot、BF16/SDPA/Transformers backend 与 24-case/27-event v3 网格：production `crop_to_token` 保留的 K/V 前缀与 crop 前前缀及独立切片 oracle 逐张量 bitwise exact；以相同 token-ID chunk 恢复后的 K/V、logits、mask、token ledger 与 role/end 状态 exact 一致。

**禁止主张**：不得声称 clean re-prefill 数值等价、v2 通过、跨模型/dtype/backend/硬件普适、真实 ASR/TTS/声卡/用户实际听到边界、生产端到端正确性、时延或质量提升。论文必须透明说明 v1/v2 rejected 的数值对照及 v3 为直接 crop-integrity addendum，不把 v3 改写为 clean-reprefill equivalence。

**影响**：C2 正确性 GPU 阻塞解除，无需再跑 C2 或重跑 E1/E2/E3/A1/P1。下一阶段进入二审统一论文修订：插入 v3 限定证据，同时完成 E1/E2 crossed analysis、E3 weighting/unique-boundary sensitivity、事件命名、C-E1 implementation-path 限定、贡献层级和 artifact/literature 完善。

**状态**：accepted（正式证据已验收封存；GPU 补实验阶段结束）

---

## D-022（2026-09-03）新 user 内容推进时清除陈旧 CROPPED end reason

**问题与证据**：C2 v3 8-case 7B pilot `c2crop_pilot_b2c6f22b_20260903T064135Z`（结果 commit `91ea218`）按规程在 formal 前停止。7/8 cases、c2_08 second crop 和全部 K/V/logit/mask/token exact 门通过；唯一失败为 `speculation_full_invalidation`：crop 到 assistant role 起点后 production 正确进入 `USER_OPEN + CROPPED`，但紧接着 `prefill_user_text()` 成功追加新用户内容后仍残留 `CROPPED`，直到 `open_assistant_role()` 才清为 `NONE`。这是 8 个状态字段中的唯一差异，K/V、logits、mask、ledger 仍 bitwise exact。

**决策**：`GenerationEndReason` 是当前生成/截断阶段状态，不是永久事件日志。`CROPPED` 在 crop 后、任何新内容推进前必须可见；一旦 `prefill_user_text()` 成功追加新 user 内容，当前状态已进入新的用户输入阶段，旧 crop 终因必须重置为 `NONE`。因此修生产侧而非放宽 v3 oracle：`prefill_user_text()` 在成功 `_prefill_text_p2()` 后设置 `generation_end_reason=NONE`，与 `reopen_user_role()`、`open_assistant_role()`、`prefill_assistant_text()` 的推进语义一致；orchestrator 在追加 user segment 后新增 fail-closed 断言。

**测试与影响**：先在真实 `run_kvcrop_test` seam 加 `invalidation crop → prefill_user_text` 断言，修复前稳定复现 GPU 同一失败，修复后 PASS；`run_speculative_test`、timeline、v3 fake full workflow 全绿。0.5B CUDA 直接复跑 c2_06：crop 后 `USER_OPEN+CROPPED` 保留，prefill_user_text 后 `USER_OPEN+NONE`，open_assistant 后 `ASSISTANT_OPEN+NONE`，两步 K/V/logits/mask/token 与 oracle 均 bitwise exact。既有 orchestrator/E1E2 在生成结束时已把 end reason 快照到独立 candidate，A1/P1 recovery 原本经 reopen 清零；该修复不改变 token、KV、logits、计时窗口或既有结果，无需重跑 E1/E2/E3/A1/P1。v3 protocol/cases/exact gates 不变，不 bump 版本；修复 commit 后 GPU 从 §3 pilot 重跑。

**状态**：accepted（生产状态修复完成；Qwen2-7B v3 formal evidence pending）

---

## D-021（2026-09-03）保持 C2 v2 rejected；新增 exact-only v3 crop-integrity addendum

**v2 审计结论**：Qwen2-7B run `c2eq_5c56b014_20260903T040829Z`（code `5c56b01`、结果 `8d9b863`）按冻结协议正确判定 **rejected**：24/24 termination probe、45/45 token/state/账本/EOT/scenario、43/45 top-1（两次均为近并列）、45/45 continuation margin 规则及绝对安全上限全过，但单控制 2× 相对门槛仅 42/45。不得把 2.0 事后放宽到 2.7，也不得把 v2 改判通过。

设计侧对 24 records 与 45×3 个 FP32 logits 数组独立复算，并做两项分离审查：v2 control 以 canonical **语义边界**分块并强制最后 token 单独 forward，而 production path 的初始 512/2048/8192 context 是一次 forward、assistant/role/user 各按真实 API chunk 追加，45 个 checkpoint 也未走 control 所声称镜像的单-token refresh；两者拓扑不匹配。三项失败亦无 crop 特异信号：c2_06 的 max_abs 由 canonical rank 146,048、概率约 7.7×10⁻¹² 的尾部 token 单点决定，且 path→canonical 总变差 0.0078 反而小于 control 的 0.0365；c2_10/c2_21 的 mean_abs 主要是 softmax 不变的全词表常数偏移，中心化后 path/control 比仅 1.24/1.04。故三项失败只说明“超过该预注册但不可识别的单控制门槛”，既不证明 crop bug，也不支持 clean-prefill 数值等价。`src/` 不改。

**决策**：在不改 C2 v1/v2 协议与既有结果的前提下，新增 `experiments/sci34_supplement/c2_crop_integrity/` 作为 protocol v3 exact-only addendum。v3 精确复制 v2 24-case JSON（SHA-256 `acda9afb…0696`），覆盖每 case 首次 crop 与 3 个指定 case 的第二次 crop，共 27 events；不重跑 v2 termination probe。正式 Qwen2-7B/BF16 路径以 `generate_accumulating` + 受控非 EOT token selection 逐 token生成冻结 assistant fixture，硬验每 token 一次 `_prefill_ids_p2` forward。每次 crop 前对全层 retained K/V 前缀生成 shape/dtype/device/SHA-256 manifest 并独立 clone oracle；production arm 唯一调用 `crop_to_token`，要求 crop 前缀、production crop 后、oracle 三者逐张量 `torch.equal`/hash exact。之后 production role API 与 oracle 以相同 token-ID chunk、position/mask 直接 forward，逐步要求 K/V、logits、mask、token ledger、retained-prefix hash 与 role/end/content 状态全部 bitwise/exact。

**防伪与独立性**：manifest 冻结每 case token plan；validator 从 case、逐 fragment token partition、role/content boundary 和 second-crop fraction 独立推导 keep，不信任 stored keep；逐层验证 K/V `shape[-2] == keep`，独立重算 JSON/token/layer aggregate hashes、24/27 网格、第一/第二 assistant 的逐-token event ledger、unique EOT 与 operation-derived role state。wrong-length disposable negative control 与 smoke 的 wrong-keep/layer-hash/duplicate-EOT/missing-event 篡改必须被检出。v3 无经验容差、无 clean re-prefill logit gate；v2 clean-prefill 数据仅保留为 rejected 描述性证据。另修复 v2 summary 的 `runner_qualified` 误与 case `passed` 联合计数问题，v2 raw/verdict 不改。

**本地验证**：compile 与 fake full workflow PASS（24 cases/27 events，四类 tamper + missing-snapshot seal 拒绝）；0.5B CUDA 真模型代表 pilot 覆盖 c2_01 与 c2_08，共 3 个真实 crop（含 next-turn/second crop），所有 pre-prefix=production-post=oracle、post-crop mask/token/length、matched recovery K/V/logits/state 均 bitwise/exact。

**证据边界**：若 7B formal 通过，只能主张冻结 snapshot/dtype/backend 下的 crop/truncation integrity 与 matched recovery determinism；不得主张 clean re-prefill 数值等价、跨模型普适、在线 ASR/TTS/player 或生产端到端正确性。

**状态**：accepted（v2 rejected 归档；v3 代码与协议已实现；Qwen2-7B formal v3 evidence pending）

---

## D-020（2026-09-03）修复 v2 探针分支缺陷（pilot 暴露），协议内容不变

**决策**：`TransformersBackend._termination_probe` 的 termination 检查结构中，`eos_at_cap` 之后的 `else:` 误捕 `natural_eos`，对 genuine EOS case 强加 max-token 断言。7B pilot `c2pilot_a501df43_20260903T033106Z`（结果 commit `899462c`）在 c2_01 上确定性暴露：`genuine_eos=true`、`eos_step=21≤256`、`ASSISTANT_EOT_PENDING`，却记 3 条 max-token 错误并被 runner fail-closed 拦截，formal 按规程未启动。修复为 `elif case.termination == "max_tokens":`。**协议 v2 全部冻结内容不变**（cases、24-case 网格、相对门槛、对照臂、margin 规则、cap 256、≥5/10 genuine 均不动）；这是 v2 实现的分支缺陷，不是协议变更，不构成"事后改协议"。

**覆盖补强**：FakeBackend 自行合成探针字典、此前 0.5B dry-run 恰走 requalified 路径，故两处本地验证都未覆盖 genuine 分支。`smoke.py` 新增 stub 化 `TransformersBackend._termination_probe` 四分支路由单元回归（genuine/requalified natural、max_tokens、受控 eos_at_cap），并用独立 validator 逐条交叉校验 stub 探针输出。

**影响**：仅 `runtime.py` 一行分支条件 + `smoke.py` 回归；`src/` 零改动；v1 归档与 v2 pilot（`899462c`）工件只读。pilot 同时证实噪声对照臂在 7B 上工作正常（c2_01：对照噪声 max_abs 0.3125 vs path 0.289，相对限 0.625，门槛全过），与本地 0.5B dry-run 的 path/control≈1.08 相互印证。修复 commit 推送后实验机从 §5 pilot 重跑。

**状态**：accepted（修复完成；v2 Qwen2-7B formal evidence 仍 pending）

---

## D-019（2026-09-03）判定 C2 v1 formal run rejected；以噪声对照臂 + 相对门槛发布协议 v2

**决策**：Qwen2-7B formal run `c2eq_563dd22a_20260903T013547Z`（code commit `563dd22`、结果 commit `1a47ac1`）判定 **rejected 并永久归档**；不修改其任何容差或工件。`src/` 无需改动。C2 协议升级为 v2 并冻结，以新 run-id 定向重跑（唯一 GPU 任务，入口 `c2_equivalence/GPU_HANDOFF.md`）。

**v1 结果审计（设计侧独立复核 raw records/NPZ）**：
1. **通过层 100%**：canonical↔crop-path token IDs 45/45 exact、KV/mask/seq/ledger、assistant 内容账本、unique EOT、role phase、scenario execution 24/24、top-5 overlap≥4/5（min 4，mean 4.87）。
2. **失败层均非实现缺陷**：
   - 45/45 checkpoint BF16→FP32 logit diff 超绝对阈（max_abs 0.156–0.969 > 0.1；mean_abs 0.020–0.156 > 0.01）。机理：同形状重复计算差精确为 0，而"增量 append vs 整段 prefill"的 kernel 归约顺序不同产生固有差异；设计侧本地 0.5B/RTX 3060 纯 transformers 分块 append 对照（零 crop 代码）复现同量级（max_abs 0.20–0.33），与 7B/RTX 3090 观测（0.16–0.97）同阶，证明环境无关、机制固有。mean_abs 并不随上下文长度单调增长（512/2048/8192 均值 0.056/0.063/0.043）。
   - next-token top-1 运行时口径 43/45（GPU 执行报告的"45/45 exact"系误统计，validator 严格重算口径 41/45）；全部翻转均为 top-5 集合内前两名互换，canonical top1–top2 margin ≤ 0.125（|logit|∈[16,32) 的 BF16 ulp）或精确 0 并列。
   - 32-token continuation 30/45 exact，15 处发散全部始于 margin 受限的平缓/退化 greedy 区（含重复 token 环）。
   - 4/10 `natural_eos` probe greedy 在 128-token cap 内 run-on：cap×snapshot 确定性组合，同机重跑不可解，与等价性无关。

**协议 v2（常数先验冻结，禁止看到 formal 结果后调整）**：
1. **噪声对照臂**：每个 checkpoint 将同一 canonical 序列按结构 seam 分块增量 re-prefill（镜像路径 append 结构、同样单 token refresh 收尾，不含 crop/恢复代码），对照臂 diff 即环境固有增量噪声 `control`。
2. **相对门槛**：path max_abs ≤ 2.0×max(control max_abs, 0.05)、mean_abs ≤ 2.0×max(control mean_abs, 0.01)；绝对安全上限 max_abs ≤ 2.0、mean_abs ≤ 0.5。本地 0.5B dry-run 实测 path/control 比值最差 1.08。
3. **top-1/continuation margin 规则**：top-1 须 exact 或 canonical margin ≤ min(max(2.0×max(control max_abs,0.05),0.125),0.5)；32-token continuation 须 exact 或首个发散步 margin 在同限内，逐步记录 top1/top2/margin，exact 率降为描述性统计。
4. **natural_eos**：cap 256；未命中者确定性重资格化为 max_tokens 语义并完成全部等价比较；campaign 级要求 ≥5/10 genuine EOS。
5. **工件**：每 checkpoint 恒存 `checkpoints/*.npz`（path/canonical/control 三 FP32 数组），validator 从 sidecar 独立重算全部统计与门槛。24-case 矩阵、8 scenario、3 termination、token/state/账本/EOT/role 门槛、模型身份（D-017 artifact）、BF16、单 session、fail-closed/seal 规则全部不变。

**影响**：仅改 `experiments/sci34_supplement/c2_equivalence/`（protocol/runtime/run/validate/analyze/seal/campaign/smoke 及其文档）；`src/`、cases.json、v1 工件、旧 campaign、论文正文均不动。v1 run 及 pilot、`e3_exact_rescue/` 全部只读。本地验证：fake smoke PASS（24 cases/45 sidecars/全 tamper 与重资格化测试）、0.5B CUDA 真模型 dry-run 全绿。论文正确性主张仍待 v2 7B formal 证据；GPU 侧若 v2 仍失败，须回设计侧再决策而非现场放宽。

**状态**：accepted（v1 rejected 归档；协议 v2 冻结；v2 Qwen2-7B formal evidence pending）

---

## D-018（2026-09-02）显式化 EOT/role 状态并冻结 C2 语义等价验收协议

**决策**：修复二期持久化 KV 的 EOS/EOT 状态语义，并新增独立 `experiments/sci34_supplement/c2_equivalence/` 正确性 campaign。Qwen 的结构性 `<|im_end|>` 不再写入 `assistant_token_ids`、TTS fragment timeline 或 assistant 内容 KV；`generate_accumulating()` 预测 EOT 时进入 `ASSISTANT_EOT_PENDING` 并记录显式 `GenerationEndReason.EOS`，由 `reopen_user_role()` 唯一提交一次 assistant close 并打开下一 user role。max-token、consumer-stop、crop 分别记录显式 end reason，不再通过生成 token 数或账本末 token 推断。

**状态合同**：`AccumKVCache` 新增完整 `token_ids` ledger、`RolePhase`、assistant role/content 边界及 end reason；所有追加统一经 token-ID prefill 核心，role transition 从 `apply_chat_template(tokenize=True)` 推导并验证。强制 `len(token_ids) == seq_length == attention_mask length == DynamicCache length`，assistant 内容 span 与 `assistant_token_ids` 完全一致。`prefill_user_text`、`prefill_assistant_text`、`open_assistant_role`、`reopen_user_role` 均校验 role 前置条件；crop 同步恢复 role/end 状态，并拒绝落在结构 token 中间。timeline 同步增加 token span、chunk 唯一/顺序、sample 连续与播放游标单调合同。

**冻结验证协议**：正式 C2 campaign 固定 Qwen2-7B-Instruct、BF16、Transformers、本地模型、严格离线、24 个确定性 case、1 个逻辑 session、无统计重复。覆盖 512/2048/8192 token，p=0、片段边界、中段吸附、reply-tail、pending EOT、推测全作废、下一轮及第二次 crop；natural EOS、EOS-at-cap、max-token 均由 formal case 自身的 termination probe 硬验证。每个 case 比较实际 crop/recovery 与相同 retained token IDs 的 canonical clean re-prefill，冻结门槛为 token/状态/唯一 EOT/top-1/32-token continuation 全部 exact、top-5 overlap≥4/5、BF16 logits 转 FP32 后 max abs≤0.1 且 mean abs≤0.01。任一失败保留 raw/attempt/sidecar 并判定未通过，禁止删除 case 或事后放宽同一协议。

**影响**：`src/llm/stream_llm_inference.py`、编排器和未来 E1/E2 runtime 改用显式状态；一期 `generate()` 保持不动。既有 C-E1/E2、固定轨迹 E3、联合 A1 与 P1 v2 不重跑、不覆盖；本轮不是时延实验。GPU formal 验收前不修改论文正文、图表或旧结果；执行唯一入口为 `experiments/sci34_supplement/c2_equivalence/GPU_HANDOFF.md`，回传后另追加接受/拒绝决策。

**状态**：accepted（代码与协议冻结；Qwen2-7B formal evidence pending）

---

## D-017（2026-09-02）接受确认性 E1/E2 campaign 并以双口径重写 E1/E2 结论

**决策**：接受 run `e1e2c_b8c758b_20260901T173306Z`（代码 commit `b8c758b`、结果 commit `62508dc`、manifest `2f4bd76e…f4ed8`）为 E1/E2 的确认性正式证据（第四个独立 campaign）。设计侧对 5000 条 raw records 独立复算与 analysis_v1.json 全部一致；checksums 72 文件对 git blob 全绿；旧三个结果文件 blob 逐字节不变；holdout 与旧 E1/E2/E3 的 ID 与对话级交集为 0。

**结果与口径**：
1. 实际墙钟主指标（last_segment_arrival→first_token_ready，配对 n=500）：C-E1 中 System A 27.70 ms vs B@0.92 62.38 ms，配对 A−B −34.69（95% CI [−35.30, −34.11]），B 更慢；C-E2 中 B@0.92 与 never 无显著差异（−0.03，CI [−0.55, +0.51]），九条件 arrival→ready 平坦于约 62 ms。
2. oracle TTFT_eff（时延乐观下界/推测收益上界）：C-E1 A−B +17.44（CI [16.12, 18.75]）；C-E2 never−B +20.80（CI [19.50, 22.10]）。B@0.92 pooled waste（wasted/(wasted+final)）2.85%，survival 67.0%，ready 中位 12，候选领先中位 291 ms，未存活 on-demand 31.09 ms≈never oracle 31.06 ms。
3. 机制：同步 harness 中 A 关键路径＝单次批量 prefill＋首 token；B＝最后段增量 prefill＋assistant role 注入＋首 token（≈两次串行前向），短文本下单次前向固定开销主导，故 B 到达→就绪更慢。oracle 口径量化"端点晚于触发"时的可立即交付收益。
4. 旧 E1/E2 的 0.581/12.1 ms 属 oracle 口径误作墙钟（user_end 记录在同步推测完成后）；确认性 campaign 显式修正，旧结果降级为探索性 campaign 审计。

**影响**：摘要、第一/三/五/六/七/八章按双口径重写 E1/E2 表述；图 6-2/6-3 由新 analysis 重画；旧 E1/E2 数字不再作为 headline。禁止把 oracle 收益说成墙钟改善，禁止新增真实 ASR/TTS/声学/生产端到端主张。IEEE 衍生稿待权威 Markdown 稳定后整体同步。

**状态**：accepted

---

## D-016（2026-09-01）冻结 E1/E2 确认性受控文本段 campaign

**决策**：旧 E1/E2 保持只读，不直接重跑旧脚本；新增独立 `experiments/sci34_supplement/e1e2_confirmatory/` campaign，在新的未见 holdout 上确认受控模型侧 E1/E2。正式结果验收前不修改论文数字、权威分章、摘要、`thesis_draft` 或 IEEE 衍生稿。

**预冻结协议与实现对齐**：
1. **阈值与解码**：`0.92` 来源于旧探索 campaign，在新 holdout 结果可见前冻结为唯一 confirmatory candidate；C-E1 使用 System A vs B@0.92，C-E2 使用 B@0.92 vs `never_speculate` 并报告全部离散点。主模型固定 Qwen2-7B-Instruct、greedy、`max_new_tokens=32`、`spec_chunk=12`、batch size 1。
2. **新 holdout**：从本地 MultiWOZ 2.1 确定性派生 100 条话语，显式排除旧 E1/E2 与 accepted 固定轨迹 E3。E3 排除源固定为真实 manifest：`experiments/sci34_supplement/results/e3/sci34_f11ccba_20260901_e3/manifest.json`。formal 默认拒绝 fixture、重复/空 segment、缺文件和联网回退。
3. **TEN cache**：TEN 对每个累积 segment prefix 真实前向一次，输出单个只读 JSON cache，保存未舍入 confidence、文本/template/token/model/input 身份与 hash；五个 formal session replay 同一 cache。TEN runtime 不进入指标窗口，故不作在线 trigger 零开销主张。
4. **独立重复**：运行 5 个独立 Python 进程，`session-index=0..4`，每个重新加载模型并完成 100 条×10 条件；每条 warmup path 默认重复 3 次且不进入 formal records，条件顺序按 session/dialogue 平衡。进程重启后不得拼接同一 session；当前 `--resume` 只允许相同 `process_start_id`。
5. **时间语义修正**：严格区分 `last_segment_arrival`、`first_token_ready` 与同步 oracle 的 `endpoint_accept`。实际受控墙钟主指标是 `first_token_ready - last_segment_arrival`；`endpoint_accept` 不是最后一段到达瞬间。`TTFT_eff` 仅是候选已准备后同步 oracle 接受的时延的乐观下界（推测收益的上界）。Raw records 已保存 `last_segment_arrival_ns`、`first_token_ready_ns`、`arrival_to_first_token_ready_ns`、`endpoint_accept_ns` 和 `oracle_preaccept_processing_ns`；validator 复算恒等式，analyzer 以 arrival-to-ready 为主指标。
6. **浪费率与分析**：正式 pooled waste 固定为 `sum(wasted_tokens) / sum(wasted_tokens + final_tokens)`；`speculative_tokens` 只作诊断。C-E1/C-E2 使用配对记录和 session→dialogue 两层 bootstrap；主分析不删异常值，versioned analysis 不覆盖。
7. **实际 CLI 与 manifest**：已实现并核对 `holdout_builder`、`trigger_cache`、`campaign`、`run_session`、`analyze`、`validate`、`smoke`。`campaign` 在 TEN cache 后生成不可变 formal manifest，冻结 input/cache/TEN/main-model/protocol identity；formal `run_session` 强制传同一个 `--campaign-manifest`。Pilot 使用独立 non-formal session、`--limit 3` 且不传 formal manifest。

**证据边界**：输入是受控预切分文本段，**不是实际音频**。允许主张限于指定模型/硬件下、可审计的最后段到达至首 token 准备墙钟行为、同步 oracle 的 `TTFT_eff` 时延的乐观下界（推测收益的上界）与推测浪费工作点；不得声称真实流式 ASR、实际 endpoint detector、在线 TTS、播放器/声卡、声学停播、真实 mouth-to-ear、生产端到端 barge-in 或 `0.92` 为部署最优阈值。

**影响**：campaign CLI、不可变 manifest、raw 主墙钟字段与文档已对齐，代码已实现，GPU formal 数据待运行。GPU 数据、campaign manifest、analysis、validation 与 acceptance 全部通过后，才由论文主代理决定是否更新权威 Markdown；旧 JSON 永不覆盖，论文稿与旧结果本次不改。

**状态**：accepted

---

## D-015（2026-09-01）接受 prepared-state P1 v2 并区分第三实验 campaign

**决策**：接受 run `sci34_dc52978_20260901_async_prepared_v2` 为 headless 墙钟软件播放控制路径的正式证据；P1 v1 继续仅作失败协议审计。D-015 只取代 D-014 中“P1 v2 待运行”的状态，不改变 D-014 已接受的固定轨迹 E3 与联合 A1 结论。

**证据与口径**：
1. **身份与归档**：实验代码 commit `dc529788e86ecd3e2e4203ba16b1076d6b231ec1`，结果入库 commit `ee1dcc7`，manifest `config_hash=93b7837acdc708ffde48448fc7cb0549475cbf064539d53a5327cda05031e005`，clean tree、Transformers runtime、Qwen2-7B-Instruct 模型指纹已记录。结果文件 SHA-256：records `2dc68896dc52ce2c777b1a6375f1a5c3090f9baffd8f07a6ac1ed0f1769a3b67`，analysis `b9705d58f36909604e3e0df94d2190b3a5050c6a62d35fee1c29987fff4db20a`。回传 tarball SHA-256 为 `4c6188249f1226e5692a85468cf1e9c3b05e648494a5ce9a6e5a475b264c0bc8`。
2. **完整性**：512/2048/8192 token × 0.25/0.50/0.75 × 20 repeats，共 9 个单元、180 条唯一正式记录；3 次单元预热均不落盘。120 条为片段内打断，60 条为片段边界；180/180 request 与 ack 精确命中目标软件采样，零采样泄漏，prepared-state 与 partial 几何检查全部通过。
3. **延迟**：九个单元的中位数范围为：软件停播确认 0.055–0.062 ms、stop 后设备同步 0.167–0.176 ms、时间轴反查 0.47–0.50 ms、stop→crop 完成 2.44–2.53 ms、stop→角色恢复完成 78.6–80.8 ms；各指标最大单元 P95 分别为 0.076842、0.351591、0.939422、3.491824 和 86.084611 ms。准备态 setup 的单元中位数为 41.208–1717.110 ms，在播放前完成并从所有 stop 路径区间中排除。累计区间相互嵌套，禁止与组件中位数相加。
4. **硬件与 campaign**：P1 v2 主机为双路 Intel Xeon Gold 6330（112 逻辑 CPU）、约 756 GiB 内存、Ubuntu 22.04.5、NVIDIA driver 580.105.08、双 RTX 3090；运行前后无其他 GPU 计算进程。它作为第三个独立 campaign 报告，不与旧 E1/E2/A2 或固定轨迹 E3/联合 A1 的绝对墙钟时间池化，也不通过相减解释“系统开销”。
5. **主张边界**：P1 v2 只支持 headless 软件播放器、时间轴查询与模型状态修正路径的协议内分布。它不测声卡/扬声器停止、用户声学上实际听到的最后采样、在线 TTS 取消、真实 ASR/LLM/TTS/播放器并发或生产端到端 barge-in。九个单元范围较窄只作本 campaign 的观察，不证明上下文无关或硬件不变性。

**影响**：第六章在 RQ4 下新增 P1 v2 子节与表 6-5；摘要、讨论和结论加入限定性结果。P1 从当前待办移除，后续工作保留生产音频闭环、固定轨迹 A2、人类双标、细粒度物理对齐和跨模型/语种复验。

**状态**：accepted

---

## D-014（2026-09-01）固定轨迹 E3 与联合 A1 的正文证据升级

**决策**：将 SCI3/4 补实验中的固定轨迹 E3 作为 RQ1 的主要受控结果，以新联合 A1 替换正文旧 A1 数值；headless P1 v1 因联合计时协议把未完成的异步准备工作计入 stop 路径而排除，待按 prepared-state v2 协议完成定向重跑后再决定是否纳入正文。

**关键结果与口径**：
1. **固定轨迹 E3 为主证据**：100 条纯 MultiWOZ 对话生成 400 个配对场景、800 条条件记录；playback/generation 共享同一被打断 assistant 轨迹、片段时间轴和注入位置，首轮及 probe 均 greedy、最多 40 token。片段目标 n=297：规则 199/297（67.0%）vs 189/297（63.6%），McNemar p=.164，generation−playback dialogue-cluster 95% CI [−9.5, 2.8] pp；裁判 127/297（42.8%）vs 121/297（40.7%），p=.512，CI [−8.9, 6.1] pp。
2. **proxy 资格修正**：字符比例—空白边界代理按自身非空目标确定资格，n=380；规则 286/380（75.3%）vs 280/380（73.7%），p=.405，CI [−5.75, 2.50] pp；裁判 167/380（43.9%）vs 157/380（41.3%），p=.229，CI [−8.25, 2.67] pp。四个点估计均小、方向与预设假设相反且不显著，不作优效、等效、非劣或伤害主张。
3. **构造检查独立报告**：playback 条件的局部完整未播放文本在 400/400 场景中为空，是机制/指标共同定义的构造检查，不与 n=297 或 n=380 的语义效果估计合并。0.5 与 clean boundary 在片段目标层面重复；cluster bootstrap 是主要不确定性结果，McNemar 仅作描述性补充。
4. **裁判与人评边界**：固定轨迹 E3 没有随机盲法的人类双标注。Mistral 裁判使用 `specific-reference-v3` 单一提示词；v3 在 v2 格式解析失败后增加首行 YES/NO 约束与一次有界重试，正式运行无解析失败且未触发重试。裁判仅为模型代理。
5. **联合 A1 替换旧结果**：上下文 256/512/1024/2048/4096/8192 token，warmup=5、repeats=50，以设备前后同步包围同一 joint crop+role 区间。联合中位数/IQR/重新预填充相对加速比分别为：31.616/2.356/2.254×，31.852/2.162/4.124×，31.054/3.099/7.707×，31.519/1.197/15.020×，36.903/0.635/25.453×，48.315/0.928/40.620×。该结果是模型侧同步微基准，不是完整 barge-in。
6. **campaign 与 P1**：旧 E1/E2/A2 与新 E3/A1 运行在不同 CPU 主机、但均为同型号双 RTX 3090；不虚构 CPU 名称，不池化跨 campaign 绝对时间。P1 v1 在 `ensure_full()` 后未于播放器启动前同步，stop 后的首次同步把仍在执行的 KV 恢复错误计入 stop→crop/role；该污染跨多次重复持续，并非一次性冷启动。P1 v2 待运行，摘要和正文不得出现其占位数字。

**影响**：权威分章 Markdown、摘要、大纲与论文上下文改用上述口径；固定轨迹 E3 从后续工作移除。后续保留固定轨迹 A2、协议有效的生产式异步链路/真实音频闭环和独立人工双标。IEEE 衍生稿本次不修改。

**状态**：accepted

---

## D-013（2026-08-31）论文统稿的数据完整性审计与结论边界修正

**决策**：在不重跑 GPU 实验、不覆盖原始结果 JSON 的前提下，对二期 E1/E2/E3/A1/A2 做离线完整性审计；将清洗与统计复算独立保存为 `experiments/results/paper2_reanalysis.json`，并以分章 Markdown 作为唯一正文源重新统稿。

**关键结论与修正**：
1. **fixture 污染隔离**：E3 原 103 个 id 实为 100 条 MultiWOZ + 3 条 `fx_*` 开发样例，E2 原文件另含 12 条 `fx*` 记录。正文正式结果排除 fixture：E3 每条件 n=400，B-gen loose 规则/裁判为 50.3%/2.3%；E2 每阈值 n=100，不推测点 TTFT 为 48.3 ms。原始 GPU JSON 保留不动，便于追溯。
2. **统计设计对齐**：E3 由独立样本 Fisher 改为同 `(id, fraction)` 配对的 exact McNemar，并以 dialogue 为重采样单元给出 10,000 次 cluster bootstrap 置信区间。
3. **结论强度收缩**：loose=0 继续明确为构造性保证；strict 改称采样比例近似口径；取消“检测器上界/裁判下界”“无代价”“连续单调前沿”“完整 barge-in 亚毫秒”等超证据表述。
4. **A1 口径分离**：0.308–0.339 ms 仅为 `DynamicCache.crop` 孤立微基准；39.7× 的分母是 crop+role recovery（8k 为 46.88 ms），不再把二者混用。
5. **A2 降格**：三策略分别重新采样首轮与下一轮回复，仅 33/100 的三策略 `heard_text` 完全相同，不能隔离策略因果效应；正文保留为受混杂的探索性负结果，若需正式比较须在实验机固定同一生成轨迹后重跑。
6. **形式化修正**：原始进度不再跨量纲直接比较；以片段级保留边界统一到 token 域。按代码真实语义，片段内尾部评估定义为“播放比例切文本字符 + 向前吸附空白边界”的代理，而非 token 域线性插值；KV 裁剪和角色恢复拆成两个状态阶段，assistant token 账本保持本轮相对长度。
7. **版本源层级**：`abstract.md + chapter1..8 + references.md` 为权威源；`thesis_draft.md` 和中英文 IEEE 稿均为衍生产物。当前先更新学位论文 Markdown 与实验图，IEEE 稿后续从新源同步。

**影响**：论文核心贡献仍成立，但适用范围被限定为受控文本段/Mock TTS 实例化与孤立 KV 微基准；真实异步音频闭环、完整停播延迟、固定轨迹 A2 和独立盲法人评列为后续补强实验。

**状态**：accepted

---

## D-012（2026-07-02）实验前代码审查结论与修复

**决策**：实验开跑前对 `7facaba...HEAD`（二期全部实现+实验代码）做两轴审查（Standards/Spec），确认 3 个 BUG 并全部修复；§6 埋点缺口补齐；配置集中化整改。

**关键修复**：
1. **E3 指标框架修正（review BUG1，最重要）**：选 A 语义下 playback 的"未听引用率=0"是**构造性保证而非实验发现**——论文必须如此表述，实验量化的是 B-gen 失败率。新增 **strict 严格 ground-truth 列**（P1 语义：被打断片段内未播尾部按播放采样比例切分、计入 unheard 检测）——playback 的 strict>0 量化**片段级截断粒度的量化误差**（D-008 选 A/§八取舍的代价），成为 E3 的诚实补充结果。
2. **E1 公平性（BUG2）**：System A 改用与 B 相同的 system prompt（原用默认中文 prompt 导致生成不可比）；mouth-to-ear 建模改为 `first_fragment_ms + TTS首块延迟`（原用首 token 时刻，忽略断句攒首片段的时间）；"B 的 prefill 与说话重叠"是一期机制、属被测系统本身，注释澄清非偏置。
3. **chunker 越界（BUG3）**：纯空白句直接跳过（原兜底推进会偷下一片段首 token 使 crop 点偏移）；token_end 钳制到实际生成数（原可越界致 crop_to_token 崩溃）。
4. **§6 埋点补齐**：8 个时间戳落盘（timestamps dict，模拟量标注）、`ttft_text_ms`（§3 定义可测）、KV 复用计数器/复用率（rewrite<1）、反向映射 timeline_records 落盘、E3 增加 boundary 边界对照注入（P2）。
5. **配置集中化**：新增 `P2_LLM/TRIGGER/REWRITER_MODEL_NAME`、`P2_DEVICE`（src/config.py，.env 可覆盖）——实验机换 7B 只需 .env 或 `--model`；采样率经 `StreamingTTS.sample_rate` 取；`_check` 收拢至 `src/utils/check_utils.py`。
6. **B-syn 措辞修正**：Mock 同步合成下与 generation 等价，仅异步 real TTS 可区分，文档不再称"已验证"。

**已知未修（记录为接受的债务）**：`spec_stats` 字典应为 dataclass；offline-first 加载块三处重复；`_timed_tokens` 的 `self._t_first` 侧信道；E1/A1/A2 无断点续传（微基准快速可重跑）；fixture 规模小（真实数据在实验机）。

**状态**：accepted

---

## D-011（2026-07-02）TEN 规格修正 + 软触发开发替身策略

**决策**：
- **规格修正**：TEN Turn Detection（TEN-framework/TEN_Turn_Detection）实测为 **7.6B 参数 / BF16 ~15GB**（HF API 确认），**不是 D-003 记录的"Qwen 0.5B 微调"**——当时调研信息有误。
- **验证机（16G）装不下 TEN** → 定义统一 `SoftTrigger` 接口（文本→turn 完成度连续置信度），本机用 **prompted Qwen2.5-0.5B 作开发替身**（取 YES/NO 首 token logits softmax 为置信度；D-003 讨论时的备选方案），实验机同一接口加载 TEN 7B（取 finished/unfinished/wait 类别词概率）。
- **实验机分卡布局仍成立**：卡 1 = TEN(15GB) + CosyVoice2(~3GB) + Qwen3-0.6B(~1.5GB) ≈ 19.5GB < 24GB，比 D-002 预估紧但可行。
- 两阈值机制（§3.5）不变；软触发不是论文贡献，不做选型消融（D-003 原则不变）。

**影响**：`src/dialogue/trigger.py` 按接口+双实现设计；E2/A3 在本机用替身出 harness 验证，实验机换 TEN 出正式数值；论文 §实现 需注明软触发模型规格。

**状态**：accepted

---

## D-010（2026-07-02）TTS 策略：Mock-first（时长 profile 驱动）+ real CosyVoice2 仅在实验机

**决策**：
- **验证机（本机 Blackwell）永不装 real CosyVoice2**——它硬 pin torch==2.3.1+cu121，与 sm_120 根本不兼容（正是 D-009 升级掉的版本）。
- 编排闭环用 **Mock TTS**：由**真实测得的时长 profile**（每字符≈多少采样、首块延迟≈45ms）驱动，与真机**时序等价**。定义 `StreamingTTS` 接口，Mock 与 CosyVoice2 都是其实现（swap-in）。
- **real CosyVoice2 只在实验机（3090 Ampere，官方 pin 可装）跑**，且仅用于：① E1 的 mouth-to-ear 最可信数字；② 定性 demo。
- 工作流：本机把**全部实验代码**用 Mock 跑通验证 → 上实验机直接换 real CosyVoice2 实现跑出最终结果。

**背景**：CosyVoice2 官方 requirements pin torch==2.3.1/transformers==4.51.3/cu121，与本机 torch 2.8+cu128 冲突。经分析（见下）其对实验的贡献可归约为"时长 + 延迟 profile"。

**理由（CosyVoice2 在实验中的真实角色）**：
- 全部实验指标都是**时序或文本**类，无一需要真听音频；P1 已定确定性程序注入，不需实时交互播放。
- CosyVoice2 对实验只贡献两样：**片段音频时长**（驱动模拟播放时钟，可用一次性测得的 profile 参数化）+ **首块延迟**（mouth-to-ear，最好真机 live 测）。
- assistant 文本每次动态生成且依赖打断，无法预烤成固定音频集；故 CosyVoice2 是"一次性 characterize + 实验机 live 少量指标"，而非"预处理后丢弃"，也非"每个实验都 live"。

**影响**：
- 新增 `src/tts/streaming_tts.py`（接口 + TimingProfile + MockStreamingTTS）、`src/player/`（SimulatedPlayer）、`src/dialogue/orchestrator.py`（编排闭环）
- TimingProfile 初值为占位（英文 ~1000 samples/char、首块 45ms），**上实验机后用真实 benchmark 替换**
- CosyVoice2 实现类留待实验机；接口先定死

**状态**：accepted

---

## D-009（2026-07-01）torch 升级到 cu128 以支持 Blackwell（5070 Ti）

**决策**：`pyproject.toml` 的 PyTorch 栈从 cu121 升到 **cu128 / torch 2.8.0 trio**：
- index：`https://download.pytorch.org/whl/cu121` → `.../cu128`
- `torch==2.8.0`、`torchvision==0.23.0`、`torchaudio==2.8.0`（cu128，cp310 均已确认可得）
- **移除** 5 个显式 `nvidia-*-cu12==12.1.*` / `cudnn==9.1.0.*` pin，由 torch cu128 wheel 传递依赖自动拉取 12.8.x / cudnn 9.10

**背景**：验证机 5070 Ti 是 Blackwell sm_120，旧 torch(cu121，≤sm_90) 不认这块卡（此即 handoff 所称"venv 损坏"真因）。

**理由**：cu128 支持 sm_120，且**向下兼容 Ampere sm_86（3090 实验机）**——同一份 pyproject/lock 两台机器通用，无需分叉。选 2.8.0 而非更新的 2.9/2.10/2.11：成熟稳定、Blackwell 支持完善。

**验证**（本机 5070 Ti，2026-07-01）：
- `torch 2.8.0+cu128`，`cuda available: True`，`capability (12,0)`，2048² matmul 真跑在 GPU 上 ✓
- 一期栈回归：transformers 4.57.1 / whisper / ctranslate2 4.6.0 / faster_whisper / silero_vad 全部 import 正常；`DynamicCache.crop` 存在 ✓
- `run_timeline_test` 仍 ALL PASS ✓

**影响**：
- `pyproject.toml` + `uv.lock` 已改（未提交，待用户决定 commit 时机）
- 本机 GPU 解锁：可跑 0.5B 全链路验证（含 CosyVoice2/Whisper GPU 路径）
- `run_test_simple.sh` 的 `LD_LIBRARY_PATH=.../nvidia/cudnn/lib` 仍有效（cudnn 9.10 仍装在该路径）

**回退**：`git checkout pyproject.toml uv.lock && uv sync` 即回到 cu121。

**状态**：accepted

---

## D-008（2026-05-21）反向映射表数据结构设计（PlaybackTimeline）

**决策**：
- 反向映射表实现为 **`PlaybackTimeline`**，落 `src/dialogue/timeline.py`。主干 = 按生成顺序排列的 **`FragmentRecord` 列表**（片段是截断单位，故以片段为主轴）。
- `FragmentRecord` 字段：`fragment_id / text / token_start,token_end / chunk_ids / sample_start,sample_end / status`（status ∈ SPECULATIVE/SYNTHESIZING/ENQUEUED/PLAYING/PLAYED/DISCARDED）。
- 反向查询：`playback_ms → samples → 二分查找命中片段 → token 边界`（sample_start 单调，O(log n)）。
- 并发：**一把锁罩整个 timeline**（操作极小，对话速率下竞争可忽略，不过早拆锁）；`played_samples` 游标原子 int 单独走。
- "已合成未播放"处理：打断时游标之后的 SYNTHESIZING/ENQUEUED 片段标 DISCARDED、token 被 crop。
- **mid-fragment 截断语义（选 A）**：打断落在片段中间时，该片段算"已听到"，截断到其 `token_end`（物理仍为片段边界）；若该片段被部分播放（partial）则置 rewrite 标记，供贡献3重写。

**背景**：handoff 方向1 的核心数据结构，KV 截断/推测浪费率/播放感知截断都依赖它。CPU + 0.5B 可验证，不需 GPU。

**理由**：片段主轴与"截断单位=片段"一致；单锁避免过早优化；选 A 与核心原则"历史=用户听到内容"及 P2"mid-fragment 触发重写"自洽。

**影响**：
- `src/dialogue/timeline.py` 实现 + `run_timeline_test.py` smoke（纯 Python，本机 CPU 可跑）
- 打断链路 `on_barge_in(playback_samples)` 返回 crop_token_end / discarded_ids / partial 标记，供 KV crop 与重写触发
- 对应 `experiment_design.md` §6 反向映射表落盘埋点

**环境备注**：本机 5070 Ti(sm_120) 当前 torch(cu121,≤sm_90) 不兼容，GPU 暂不可用（这是 handoff 所称"venv 损坏"的真因）。策略：核心 KV 逻辑先 CPU+0.5B 验证；需 CosyVoice2/全链路时再升 torch→cu128（兼容 3090）。

**状态**：accepted

---

## D-007（2026-05-21）实验设计四项基础决策（/experiment-agent plan 模式）

**决策**：
- **P1 打断产生**：确定性程序注入——"用户听到的"=注入时刻前已播放音频，ground truth 确定、可复现、无需真人/伦理审查
- **P2 打断时机**：混合——固定播放比例 25%/50%/75%（含 mid-fragment，触发重写）+ 片段边界对照（干净截断）
- **P3 一致性指标**：客观"未听到内容引用率"为主 + LLM-judge 连贯性为辅 + 人工小样本验证（Cohen's κ）
- **P4 语种/数据**：英文为主（MultiWOZ 派生 + 自构造英文打断集，对齐 HumDial-FDBench）；中文 CrossWOZ 为可砍扩展

**背景**：进入实验设计阶段，先定这四项决定后续所有实验能否测、代码要埋哪些点。完整设计见 `paper2/experiment_design.md`。

**理由**：均服务于"工程/系统贡献 + 一个月 deadline + 可复现"三重约束。程序注入避免真人成本；混合时机同时覆盖贡献2/3；客观主指标最扎实；英文为主对齐 benchmark 且省工。

**影响**：
- 确立被测系统条件 A / B-ours / B-gen / B-syn / B-noKV / B-naive|mark|rewrite
- 产出 instrumentation 埋点清单（`experiment_design.md` §6）作为 `src/dialogue/` 编码验收标准
- E4（buffer 精确映射对比）确认为锦上添花可砍（呼应 D-006）

**状态**：accepted

---

## D-006（2026-05-21）核心创新点重新定位（据 novelty 核查结论）

**决策**：贡献 2 的创新点从"提出'对话历史=用户实际听到的内容'原则"**降级/重新定位**为——

> **"首个开源、可复现的级联式播放感知上下文一致性管理实现 + 具体 KV 机制（`DynamicCache.crop` + `pre_attention_mask`/`position_ids` 同步重算 + ChatML role 边界重建）+ 可量化对比"**

三条硬约束（写作时必须遵守）：
1. **不把"历史=用户听到的内容"当作本论文 insight 来 headline**——Azure Voice Live 官方文档几乎逐字写过。它作为**组织性原则**可用，但必须**引用** OpenAI Realtime / Azure Voice Live / LiveKit 为 prior art。
2. **intro 显式引用上述商用系统先发制人**，堵审稿人。
3. **不得靠"商用系统做得粗/框架只做检测"立论**——这两条已被对抗核查 0-3 驳回（它们确实做了 played-vs-heard 历史管理）。合法差异只有：开源 vs 闭源、显式 KV crop vs 删 transcript、级联 vs 端到端、测量 buffer vs 假设实时速度。

**背景**：deep-research 核查（Task wi2gfobgx）判 (C) 部分重叠。概念被商用系统 pre-empt，但无学术/开源级联先例。完整报告 `docs/research_novelty_check.md`。

**理由**：
- 与 D-005 工程/系统框架完全一致——工程贡献不要求概念首创，要求开源+可复现+系统评测，正是商用闭源系统留下的空间
- 对硕士毕业论文门槛绰绰有余；对标邻居（RelayS2S/LTS-VoiceAgent/FireRedChat）都是 arXiv/workshop 级 preprint
- **代码工作量不变**——重新定位只改 framing 与引用，不改要实现的东西

**影响**：
- `paper2_context.md` §2.1 framing 已加 prior-art 护栏（本决策同批改）
- intro/related work 必须新增一段：商用系统现状 + 本工作与之的精确差异（用 `research_novelty_check.md` §三差异表）
- 论文可投层次：Interspeech / ICASSP / ASRU / SLT 系统方向（若投稿）

**关联决策**：**"最强 novelty 杠杆"实验**（量化 buffer 精确映射 vs 实时速度假设的 context 正确性差异）**列为"锦上添花"，不进主线**——一个月 deadline 下先保主 pipeline 完整；主体跑通且时间有余再做。需额外构造"合成速度≠实时"场景（TTS 快于播放/buffer 堆积），估 +3~5 天。

**状态**：accepted

---

## D-005（2026-05-21）论文定位：以工程/系统贡献为主骨架

**决策**：二期论文**主框架 = 系统贡献**——"在开源级联栈上实现播放感知的打断-上下文一致性管理"；"按用户实际播放位置截断 KV"作为该系统的**技术创新点/亮点**保留，但**不把论文成败押在它'全球首次'上**。贡献层级：
- **贡献 2（主，系统+技术亮点）**：播放感知 KV 缓存管理，必做
- **贡献 1（辅助）**：软触发推测生成，必做
- **贡献 3（扩展，可砍缓冲垫）**：对话历史自然化重写——时间不够时退化为"论文讨论 + 小规模验证"，甚至只保留零成本的标记法

**不采用"收窄到某个更细 novel 子点"路线**。

**背景**：论文目标 = **硕士毕业论文**，**预期一个月内完成编写**。当时 novelty 对抗核查（deep-research，Task wi2gfobgx）尚在跑，但本决策对报告结论 A/B/C 三种输出都鲁棒，故先定。

**理由**：
- 硕士学位论文评价尺度 = 工作量 + 系统完整性 + 实验充分性 + **一定的**创新性，不是顶会 novelty 门槛。~2000 行完整 pipeline + 系统实验本身即合格主体，与一期（流式架构 + KV prefill）同一评价逻辑
- "收窄"需要精密隔离实验证明某细点首创，更耗时且风险高（点被占则无退路），一个月预算承受不起
- 工程框架对 deep-research 结论鲁棒：判 A 则放大亮点，判 B/C 则作安全港，**不必等报告即可定**
- 工程框架允许干净砍范围（贡献 3 作缓冲），适配紧张时间线

**影响**：
- **实验目标简化**：从"证明全球首创"变为"在本系统上关键指标可测量改善 + 消融证明各组件有用"。核心对比实验（按播放位置截断 vs 按生成/合成位置截断，对多轮连贯性的影响）在自有系统内部自洽完成，不依赖外部 novelty
- 论文大纲与实验清单待 deep-research 报告回来后据此调整
- deadline 风险高，需以工程框架主动控范围

**状态**：accepted

---

## D-004（2026-05-21）重写模型选型

**决策**：使用 **Qwen3-0.6B** 作为对话历史自然化重写模型。

**背景**：贡献 3 的"重写法"分支在截断位置语义不完整时启用，并行运行隐藏延迟。输入 ~50 token，输出 ~60 token。

**理由**：
- Qwen3 系列 2025 年发布，比 Qwen2.5 更新，中文质量好
- 0.6B 规模在 3090 上推理 200-300ms，并行隐藏在用户说话期内
- 与主 LLM 同家族（Qwen 系），但**实例独立部署**，符合多服务工程现实
- 不在论文贡献范围内，不做模型选型消融

**影响**：
- `src/dialogue/rewriter.py` 加载 Qwen3-0.6B-Instruct 实例
- 实验机分卡布局：与软触发 + CosyVoice 共驻卡 1

**状态**：accepted

---

## D-003（2026-05-21）软触发模型选型

**决策**：使用 **TEN Turn Detection**（基于 Qwen 0.5B 微调的文本侧端点检测器，Apache 2.0）作为软触发主选；**不做候选模型消融**，软触发不是论文贡献。

**两阈值机制**：模型输出连续置信度，配两个阈值
- **推测阈值**（激进）：超过即触发主 LLM decode 进入"推测生成"
- **提交阈值**（保守）：超过才允许 TTS 开始播放给用户

> **实现注记（2026-07-29）**：提交阈值在本工作的确定性模拟 harness 中**未启用**——`orchestrator.py:speculative_turn` 仅用单一推测阈值（`spec_threshold`）启动推测，推测的提交（采用）由 ASR 段流终止的真值端点触发（P1 确定性模拟），无需第二阈值门控播放。此为 harness 简化，论文稿（abstract/C1/总结）已据此对齐为"推测阈值"表述；提交阈值作为真实部署的门控设计保留于此。

调整两阈值得到"推测浪费率 vs TTFT"trade-off 曲线（论文核心图之一，paper2_context.md §五）。

**背景**：候选过 Smart-Turn v2（音频侧，~20ms）、TEN（文本侧，50-100ms）、Phoenix-VAD（权重发布不确定）、Qwen prompted（最灵活但慢）。

**理由**：
- 文本侧检测的推理时间**与 KV prefill 并行**，挂在 prefill 的延迟阴影里，**实际零额外成本** —— 这是关键架构观察
- 文本侧错误更易复查与调优（端点判断错时可以打印当前累积文本看原因）
- 中英文双优，Apache 2.0，知名度足，论文里讲故事无争议
- 软触发不是论文贡献，**不需要做模型选型消融实验**

**影响**：
- `src/dialogue/trigger.py` 加载 TEN Turn Detection 实例（卡 1）
- 软触发输入是 ASR final 片段累积文本，触发判断与 LLM `_add_stream_prompt` 并行
- 论文中作为辅助模块描述，**不展开多模型对比**

**状态**：accepted

---

## D-002（2026-05-21）硬件配置、分支、主 LLM 规模策略、模型独立部署

**决策**：
1. **二期工作分支**：`bargeincache`（已切，不污染一期 main）
2. **验证机**：5070 Ti 16GB，主 LLM 用 0.5B 跑通 pipeline
3. **实验机**：3090 24GB × 2 = 48GB，主 LLM 用 7B，与一期实验对齐
4. **三个 LLM 实例完全独立部署**（主 LLM / 软触发 / 重写），不复用权重，模拟真实多服务工程

**3090×2 部署粗算（7B fp16）**：
- 卡 0：主 LLM(~14GB) + 长 KV(2-4GB) + Whisper-small(~1GB) ≈ 17-19GB
- 卡 1：CosyVoice2-0.5B(~2-3GB) + 软触发(~1-2GB) + 重写(~1-2GB) ≈ 5-7GB

**理由**：
- 与一期实验对齐，便于直接对比一期/二期数据
- 多服务独立部署反映工程真实，论文工程价值更可信
- 验证机用 0.5B 跑通，等架构 OK 再上实验机跑 7B，节省迭代时间

**影响**：
- `src/config.py` 二期需要支持**按模块**指定 device（主 LLM、ASR、TTS、trigger、rewriter 各自一项），一期目前只分了 asr_device/llm_device 两路
- 实验脚本要支持单卡（验证）/双卡（实验）两种 device map

**状态**：accepted

---

## D-001（2026-05-21）transformers KV cache 的对象类型与改造路径

**决策**：二期 KV 截断走 `DynamicCache.crop()` 路线。一期 `StreamLLMInference.KVCache` 中的 `past_key_values` 字段保持现状（"transformers 返回什么就用什么"），但二期新增的 KV 操作模块**显式断言**它是 `DynamicCache` 实例；若 transformers 实际返回 legacy tuple，则一开始就 `DynamicCache.from_legacy_cache()` 转换。

**背景**：一期 `src/llm/stream_llm_inference.py` 把 `past_key_values` 当作不透明对象在 `_init_kv_cache` / `_add_stream_prompt` / `generate` 之间传递，从未调用 cache 方法 — 无法从代码静态判断它到底是 DynamicCache 还是 legacy tuple。

**理由**：现代 transformers（4.36+）对 Qwen2.5 默认就返回 `DynamicCache`，`crop()` 自 4.39 起稳定。显式断言/转换让 KV 操作有一个稳定的契约面，二期不再被 transformers 内部默认行为牵着走。

**影响**：
- 二期新增模块（KV 截断、role 重建）依赖 `DynamicCache` API（`crop`、`__len__`、`key_cache` / `value_cache` 访问、`update`）
- 一期的 `KVCache` 数据类需要在二期版本里多带一个字段：**当前 cache 长度**（即 `past_key_values.get_seq_length()`），避免靠 `pre_attention_mask.shape[1]` 间接推断
- 风险：若实际运行的 transformers 版本不返回 DynamicCache，需在加载阶段统一转换

**状态**：accepted

---

## D-000（模板示例）

**决策**：[一句话决定了什么]
**背景**：[当时面临的问题 / 约束]
**理由**：[为什么这么选 — 与备选方案的对比]
**影响**：[改动哪些文件、引入哪些依赖、有哪些后续工作]
**状态**：proposed / accepted / superseded by D-xxx

---

> 这条 D-000 是模板，提交真实决策时删除或保留为占位。
