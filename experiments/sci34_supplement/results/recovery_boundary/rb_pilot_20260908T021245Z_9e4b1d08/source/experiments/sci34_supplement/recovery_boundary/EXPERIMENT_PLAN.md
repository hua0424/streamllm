# R/B 有限补证协议 v1（2026-09-06）

用户批准 SC 聚焦方案点 1/2 后的实验实现，不授权论文改稿或发布。本目录独立于既有 accepted/rejected campaign；禁止覆盖或重跑旧实验。

## 冻结设计

唯一模型为历史正式 artifact Qwen2-7B-Instruct（不是 Qwen2.5），文件集合指纹 `fae2ece10b76512237cf28957f98e7b0d2c609455a173031e3bd16b3dff7c5ab`；Qwen2ForCausalLM、torch.bfloat16、SDPA。历史依据：`../results/c2_crop_integrity/c2crop_82103004_20260903T080512Z/campaign_manifest.json`，模型路径第 231 行、dtype 第 316 行、backend 第 322 行。历史 torch 2.8.0+cu128 / transformers 4.57.1；当前远程安装尚未核验，runner 保存实际版本。

R 使用现有 C2 的 9 个非笛卡尔 fixture：01/02/03/04/05/06/10/16/18。普通边界覆盖 512/2048/8192；短上下文覆盖零播放、片段边界、中部向上保留、reply-tail/pending EOT、推测全作废；16 覆盖第二次打断。EOT 由受控 selector 走真实 production generator，明确不是 natural-EOS 实测。旧 case termination 标签不作为本 campaign 的自然终止主张。

正式：4 个独立子进程 session，每个重新加载单卡模型；每 case 1 对预热（排除）+4 对正式。共 144 case-pairs、160 event-pairs、320 arm-event records。顺序由 `(session+case_index+repeat)%2` 精确平衡。seed=20260906；greedy、top_p=1、repetition_penalty=1、最多 8 个下一轮 selection。固定短 suffix（原 case.next_user 或明确默认句），第二次 suffix 固定。cases、token plan、模板 delta、完整 source snapshot 与 hash 在计时前入 manifest。

Pilot：01 个 session，case 02/16/18，各 1 对预热+1 对计时（8 arm-event records）。仅工程估时、显存/打点检查，无正式推断。完整 formal grid 已在 pilot 前冻结；pilot 不用于选择正面 case/样本数。预算不满足则停止，不能自动缩模型、量化、扩样或更改协议。

## R 公平对照与端点

借用 A1 的 token-ID full-prefill 思路，但修复其仅丢弃输出的基准用途：本次 rebuild 必须构造可继续生成的真实 AccumKVCache。复用 C2 的精确初始上下文、逐 token production fixture append、独立模板 parts。每 arm 从独立相同历史重建 pre-interruption state（排除），不共享可变 KV，不把前一 arm 的结果作为另一 arm 输入。

- crop：真实 `crop_to_token` + 必要 `reopen_user_role`；推测作废退回原 USER_OPEN，不额外 close assistant。
- full rebuild：直接将相同 retained token IDs + 同一合法 close/user-open delta 一次性 model forward，保留输出 K/V/logits/mask，构造 USER_OPEN 状态；不调用 crop 模拟重建。
- 两臂随后相同 `prefill_user_text` + `open_assistant_role`，再真实 greedy `generate_accumulating`。
- 连续打断：第 1 次生成诊断完成后，在计时外将两臂统一到冻结第二 assistant fixture，保留第一轮修复后的真实历史，然后分别测第 2 次；避免 greedy 分岔导致后续条件失配。只覆盖两次有限打断，不外推长对话。

不新增 stable-prefix serving baseline：当前 A1 无可直接作为独立 serving 策略的执行臂；本对照范围明确为完整重建，不声称胜过所有 prefix caching。

所有区间以 perf_counter_ns 墙钟、CUDA 前后同步测量：
1. recovery_ns：已同步 pre-state 后→USER_OPEN 修复完成（含 post-sync）。
2. suffix_ready_ns：USER_OPEN→相同 suffix/header 完成同步；ready_total 为 1+2。
3. decode_first_ns：状态/FP32 logits 审计完成并同步后单独启动→第一次 `next(generator)` 返回并同步。包括首 token 的真实 KV commit、断言与 yield；selection callback 单列，不替代 consumer 端点。
4. first_deliverable_ns = ready_total + decode_first，仅为排除审计暂停的**分段同步区间之和**，不是未插桩连续墙钟，更不是语音 E2E。该和不包含模型加载、初始历史、fixture construction、plan、state/logit CPU 快照、后续最多 7 token；不包含 ASR/TTS/player。组件为逐 trial 差值，禁止相加各组件中位数。

首 selection 为 EOT 时 delivered=false、首交付延迟 null；不伪造零延迟，不丢掉该记录，不改 prompt 找正面结果。next-token logits 永存 FP32 NPY，有限性为门；数值差、top1、短 continuation 同一性只作描述，不设事后容差、bitwise cross-topology 或等价性判定。

## 精确门与统计

预期 token IDs 由冻结模板 delta/fixture partition/suffix 推导；校验完整 token/mask/seq/KV 长度、role/content span、EOT、end reason、相同初始与下一轮 ledger。validator 从 manifest 独立推 keep，不信任 stored keep。原始行逐条 fsync；丢失/重复/非有限/身份漂移 fail closed。

按固定 case/event 报告 rebuild-minus-crop 配对均值及每 session 配对均值；10,000 次 session-cluster percentile bootstrap，固定 seed。case 是预设 fixture，不当作抽样人群；repeats、同一 session、第二事件均非独立样本。4 session CI 粗糙，只描述该固定网格下进程变异。任一 pair 无 deliverable 时该 cell 不生成 complete-case 延迟优势估计，报告缺失/交付计数。无结果正负验收门，不按显著性追加实验。

## B 独立软件闭环

使用既有 E3 100 条 generated-text trajectories、800 条存储 records；独立以 sample_end 有序数组/bisect 构造参考，重放 production timeline。每条检查 0、start、start+1、中部、end、总长及超播；1118 个去重游标。检查连续 token/sample spans、token count、stored heard/history boundary 和 partial。C2 24 fixtures/27 events 独立重算两类零保留、second crop、closure token ledger；该闭环检查是既有软件记录审计，不是新模型证据。

真实 TTS 对齐、声卡 loopback、人类词级/实际听觉标注缺失，明确列出；不把模拟采样或 token 尾部计数称为声学误差。B 加现有 timeline/chunker smoke；不下载 NLP 资源。

## 运行与封存

正式要求 exact full commit + 全工作树 clean（含 untracked）；运行目录必须在源码树外。唯一时间戳+UUID，exclusive writes，无 resume，不覆盖。每 session 单独 log、环境/模型身份、process UUID。保存全依赖 source snapshot、历史工件前后 hash guard。失败保留 FAILED/原始行/log；不自动重试。OOM 停止，无量化/换模型回退。

成功流程 run→validate→analyze→seal→verify→独占创建 tar.gz + SHA256。seal 后不写日志入目录、不 tee；验证可只输出控制台。CRLF 可破坏字节 hash，回传原 tar 或 LF checkout。结果接收后另立接受/拒绝决策，本次不更新论文数字。
