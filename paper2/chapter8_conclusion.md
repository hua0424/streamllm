# 第八章 总结与展望

## 8.1 主要结论

本文研究级联式语音对话系统在打断后的上下文状态修正。高层 playback-conditioned history truncation 与 KV crop primitive 均有既有先例；本文不以这些单项的新颖性立论，而提出并实现 **external-progress-conditioned joint prefix-state repair contract**：software-consumed-sample cursor 经 TTS fragment 映射为合法 assistant commit boundary，KV、attention mask、token ledger、position、role 与 EOT 在该边界上作为联合状态同步转换。合同由边界解析、联合状态、不变式保持转换和可证伪验证四层组成。

**C2 是唯一核心贡献。** 在冻结 Qwen2-7B-Instruct snapshot、BF16/SDPA、Transformers backend 和 24-case/27-event 网格内，production crop 后的 28 层 K/V 与同一 pre-crop snapshot 的保留前缀及逐层 slicing oracle bitwise exact；27/27 wrong-length negative controls 被检出。同一 accepted run 内，两条匹配臂从精确匹配的保留状态出发，接收相同 token-ID chunks 与操作序列后，其 K/V、logits、mask、token ledger 和 role/end/content state 在 60 个恢复步骤中逐步 exact。该结果只支持 direct crop integrity 与 within-run matched-arm recovery exactness。v1/v2 clean-reprefill 协议仍按冻结门槛 rejected，v3 不改变其 verdict，也不建立 clean-reprefill、continuation 或跨环境等价性。

A1 与 P1 给出该机制的成本边界。固定 32-token suffix 的联合 crop+role 微基准在 256–8192 token 上为 31.054–48.315 ms，重新预填充与联合路径的中位数比值为 2.254–40.620。prepared-state P1 的 stop→crop 和 stop→role 单元中位数分别为 2.44–2.53 ms 与 78.6–80.8 ms。前者是固定 GPU 微基准，后者是 headless 软件路径；两者均不是声卡、声学停止或生产端到端打断时延。

固定轨迹 E3 是 C2 的 downstream 支持性证据。label-weighted generation-minus-playback 点估计在 fragment/rule、fragment/judge、proxy/rule 和 proxy/judge 四个并列操作化中分别为 −3.37、−2.02、−1.58 和 −2.63 个百分点，对话聚类 95% CI 均跨零。dialogue-weighted 与 target-specific exact-key 去重敏感性分析同样未确定方向。结果仅适用于冻结自动检测器和目标构造，不支持 superiority、equivalence、noninferiority、harm、absence of effect、人类语义或 HCI 推断。

**C1 是支持性刻画。** 在 token-consistent C-E2 中，B@0.92 相对 never 的 candidate selection/readiness 差为 −0.03 ms（crossed 95% CI [−0.64, 0.61]），同步 oracle `TTFT_eff` 乐观下界差为 +20.80 ms（[17.85, 23.65]）。B@0.92 的接受时候选可用率为 335/500（67%），pooled discarded-token ratio 为 2.85%。这些量描述 pre-oracle-acceptance candidate generation，不证明真实 end-of-speech 前就绪或 production deliverability。C-E1 的 27.70 与 62.38 ms 是两条非 token-equivalent implementation paths 的 candidate-readiness 均值；由于输出、tokenization 与 forward topology 不完全一致，该差异不能归因于纯 incremental-prefill effect。

**C3 是探索性实现。** A2 报告朴素、重写和标记路径的描述性评分及重写耗时，但三条件使用不同生成轨迹，策略效应不可识别。因此，A2 不构成负结果、零效应或因果比较。

综上，本文建立的是受测软件 runtime 层的状态合同，而非真实听觉边界或完整系统效益证明。software-consumed cursor 不等于 device-presented samples 或 acoustically heard content；当前证据也不覆盖真实异步 ASR/TTS/播放器闭环、生产时延或用户体验。

## 8.2 后续工作

后续研究可从四个层次扩展本文证据。第一，接入在线 ASR、异步 TTS、bounded audio queue、设备时钟或 loopback 波形，统一测量 software、device 与 acoustic stop。第二，在固定 assistant token 轨迹、断句和打断点下重做 A2，并固定或成对控制下一轮解码，以形成可识别的策略比较。第三，采用盲法双标或直接用户研究评估特定信息复现、自然度、信任与交互质量。第四，在不同语言、模型、TTS、chat template、dtype、attention backend 与推理引擎上重新验证状态合同，并为 A1 随机化操作顺序、覆盖更多 crop length。

## 8.3 工件与声明入口

正式 campaign 状态、run identity、分析文件、复算命令与主张边界由仓库根目录 `REPRODUCIBILITY.md` 统一索引；E3 processed input 的可复算入口亦列于其中。作者、机构、许可、伦理、基金、利益冲突、CRediT、公开 artifact URL/DOI 与 AI 使用声明保留在 `paper2/declarations.md`，待目标期刊和责任主体确认后按期刊格式完成。

## 8.4 结语

本文将级联语音对话打断后的历史修正，从文本层原则推进为外部进度条件下的联合前缀状态转换。受控结果表明，合同中由 v3 覆盖的直接裁剪与匹配恢复性质通过 exact gates；其向设备呈现、声学接收、跨运行环境和人类交互效果的推广，仍需对应层级的直接证据。
