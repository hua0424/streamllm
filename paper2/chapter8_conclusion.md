# 第八章 总结与展望

## 8.1 全文总结

本文研究级联式流式语音对话系统在打断后的上下文状态管理，并将证据范围限定为 software-cursor-conditioned、TTS-fragment-level runtime/prototype。OpenAI、Azure 和 LiveKit 等系统已公开 playback-conditioned history truncation 的高层实践，KV crop 与 prefix reuse 也不是本文原创；本文的工作重点是把 software-consumed-sample cursor、TTS 文本片段、assistant token span、KV cache、attention mask、token ledger、position、role 和 EOT 状态连接为可审计合同。

贡献层级如下。**C2 是核心贡献**：实现 software cursor→fragment→token→KV crop→role/EOT recovery，并以 C2 v3 direct crop-integrity addendum 检验状态完整性。**C1 是支持性贡献**：对 pre-end-of-turn candidate-response generation with invalidation 的 candidate-selection/compute-readiness、同步 oracle acceptance 与 wasted-token 工作点进行受控刻画，不把其解释为 production deliverability 改善。**C3 是探索性扩展**：实现朴素、标记和重写三种历史处理方式；当前 A2 只提供受混杂描述性结果。

C2 v3 正式 run `c2crop_82103004_20260903T080512Z` 覆盖 24/24 cases、27/27 crop events、3 个 no-op、60 个 recovery steps、308 个逐 token production append 和 27/27 wrong-length negative control。在冻结 Qwen2-7B snapshot、BF16/SDPA/Transformers backend 下，28 层 K/V 的 crop 前 retained prefix、production post-crop 与独立 slicing oracle 逐张量 bitwise exact；匹配 token-ID chunks 恢复后的 K/V、logits、attention mask、token ledger 与 role/end/content state 亦 exact。该结论只涉及 direct crop integrity 与 matched-recovery determinism，不涉及 clean re-prefill numerical equivalence、32-token continuation、跨模型/后端/硬件或在线音频系统正确性。v1/v2 clean-reprefill 协议均保持 rejected；v3 不改变其 verdict。

固定轨迹 E3 的 label-weighted 主表包含片段目标 297 个标签/96 条对话，以及代理目标 380 个标签/100 条对话。generation−playback 的四项效应与 95% dialogue-cluster CI 分别为：片段规则 −3.37 pp [−10.49, 3.40]、片段自动裁判 −2.02 pp [−10.70, 6.13]、代理规则 −1.58 pp [−6.08, 2.67]、代理自动裁判 −2.63 pp [−8.57, 2.90]。精确去重后，片段目标为 169 个语义组、代理目标为 379 个语义组；片段自动裁判的 unique-group effect 为 0.00 pp [−7.98, 7.47]。这些结果仅是 fixed-detector-conditioned information-reproduction rate；区间不含检测器、提示词、模型变化或人类感知误差，不能确定方向性优势或差异不存在。规则与裁判的一致数只描述 automated-proxy agreement。

确认性 E1/E2 采用 100 条唯一话语×5 个独立进程 session 的交叉设计；每条件 500 个观测，正式区间由 crossed/product bootstrap 得到。C-E2 中 `never − B@0.92` 的 candidate-readiness 差值为 −0.03 ms [−0.64, 0.61]，oracle TTFT_eff 下界差值为 +20.80 ms [17.85, 23.65]；B@0.92 pooled waste 为 2.85% [1.12%, 4.73%]，survival 为 67% [58%, 76%]。同步 harness 中 B@0.92 的 first-deliverable 和 consumer marker 均值为 257.58 与 265.57 ms，只是程序执行顺序诊断，不代表生产可交付性。

C-E1 是 implementation-path comparison：System A 与 B@0.92 的 candidate-readiness 均值为 27.70 与 62.38 ms，A−B 为 −34.69 ms [−35.44, −33.95]；oracle 下界 A−B 为 +17.44 ms [14.41, 20.32]。两路径完整输出 token 仅 280/500 相同，首 token 为 465/500 相同，长度/EOS/max-token 状态为 495/500 相同，44/100 条唯一话语出现完整输出不一致；B@0.92 与 B-never 则为 500/500 相同。因此差值混合 tokenization、forward topology/shape、role boundary、kernel 和 Python scheduling，不能归因于单一额外 forward 或视为纯 incremental-prefill effect。

联合 A1 在固定 operation order、固定移除 32-token suffix、5 次预热和每点 50 次重复下，256–8192 token 的 joint crop+role 中位数为 31.054–48.315 ms，IQR 为 0.635–3.099 ms，重新预填充/联合路径中位数比为 2.254–40.620。P1 v2 的 9 个单元各含 20 次记录，software stop→crop 与 stop→role 的单元中位数分别为 2.44–2.53 和 78.6–80.8 ms；P95 仅为 empirical/descriptive order statistic。两组结果都不代表 device/acoustic stop、用户实际接收边界或生产端到端 barge-in。

RQ5 的描述性结果为：朴素、重写和标记实现的连贯性均值分别为 3.76、3.62 和 3.29；重写均值耗时 639 ms。由于三条件历史和下一轮生成不一致，这些数值不支持策略因果比较，也不证明重写延迟已被真实用户发言完全隐藏。

综上，本文最稳健的结论是：在冻结模型和后端下，software cursor 驱动的片段级保留可以被落实为具有 bitwise crop integrity 和 matched-recovery exactness 的 KV/role 状态操作；支持性实验进一步给出了固定协议的模型侧成本、headless 软件控制路径时延及候选生成的浪费—oracle-readiness 工作点。本文未测 device-presented samples、acoustically heard content、真实异步音频闭环或 HCI 效果，因而不对生产时延、声学边界或交互自然度作超出证据的结论。

## 8.2 可选后续工作

现有 C2 v3、E1/E2 crossed analysis、E3 weighting/dedup、A1 和 P1 证据已经满足本文收窄后的结论边界，**无需新增 GPU 工作作为当前提交阻塞**。若资源与目标期刊定位允许，可进一步开展以下研究。

1. **真实异步音频与设备/声学边界。** 接入在线 ASR、TTS、bounded audio queue、设备时钟或 loopback 波形，统一记录 stop request、device stop、acoustic stop、timeline query、KV crop 和 role recovery。
2. **固定轨迹 A2。** 缓存同一 assistant token 流、断句和打断点，并固定下一轮解码或成对随机种子，以形成可识别的策略比较。
3. **人工与 HCI 评测。** 使用盲法双标或直接用户实验测量特定信息复现、自然度、信任和主观交互质量，并报告标注一致性与不确定性。
4. **边界粒度与系统迁移。** 比较片段、词、音素和 token 级对齐，并在不同语言、主模型、TTS、chat template、dtype、attention backend 和推理引擎上重新验证状态合同。
5. **时延分布扩展。** 对 A1 随机化 operation order、覆盖多种 crop length；增加 P1 重复并在真实并发条件下估计稳定的高分位数。

## 8.3 工件可用性与投稿声明

实验工件、accepted/rejected campaign 状态、run ID、代码与结果 commit、hash、复算入口及主张边界的权威索引见仓库根目录 `REPRODUCIBILITY.md`。accepted E3 processed input 保存在 `experiments/sci34_supplement/results/e3_exact_rescue/p2_turns.json`；模型权重与原始 MultiWOZ 数据不在仓库中再分发，第三方资产受各自许可与访问条款约束。C2 v3、固定轨迹 E3、联合 A1、P1 v2 和确认性 E1/E2 工件均已在当前研究仓库中保存并以 run/hash 关联；C2 v1/v2 保持 rejected 状态。公共不可变 release/DOI 仍需作者在投稿前提供。

投稿声明草稿见 `paper2/declarations.md`，在以下事项由作者确认前不得改写为已完成、`none` 或 `not applicable`：公开数据与代码 URL/DOI、immutable release/tag 与访问日期；E3 派生数据再分发许可；仓库权利人和 LICENSE；第三方 notices；伦理审查或豁免、参与者与 consent；funding；每位作者的 competing interests；作者名单、顺序、CRediT 角色与 accountability；生成式 AI/自动化工具披露。以上均保留 **AUTHOR CONFIRM** 边界，不从 Git 元数据、实验内容或本稿推断。

## 8.4 结语

本项目一期内部实现关注用户话轮期间的增量预填充，本文则把研究重点推进到系统播报被打断后的显式状态修正。结果表明，可复查的 software cursor—fragment—token—KV/role 合同能够在受控环境中实现并验证；其向真实设备、声学听觉和用户体验的推广仍需对应层级的直接证据。
