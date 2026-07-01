# 第二章 相关工作

> 初稿（2026-05-21）。写作策略遵循 `docs/decisions.md` D-006：先诚实陈述商用系统已实现的 prior art，再指出学术界与开源级联实现的空白，最后以差异对比表明确本文定位。**本章不主张"对话历史=用户实际听到内容"这一原则为本文首创。**
> 引用编号为本章局部临时编号，统稿时并入全文参考文献；带 † 的 arXiv 预印本编号需在投稿/送审前复核（见 `docs/research_novelty_check.md` §七）。

---

近年来，随着大语言模型（LLM）赋予语音对话系统强大的语义理解与生成能力，级联式架构（流式 ASR → LLM → 流式 TTS）在工业界与学术界仍是主流方案。围绕"低延迟"与"自然打断（barge-in）"这两个核心诉求，相关研究可归纳为三条脉络：商用与工程系统中的打断-上下文管理实践、学术界的流式语音对话与打断处理方法、以及 LLM 推理层的 KV 缓存操作技术。本章依次梳理这三条脉络，并在此基础上界定本文工作的确切位置。

## 2.1 商用与工程系统中的打断-上下文管理

在语音对话系统被用户打断时，一个根本性的问题是：**已经生成、但用户并未完整听到的助手回复，应当以何种形式进入对话历史？** 若系统将"自己生成的全部内容"当作已说出的话记入上下文，则在后续轮次中 LLM 会误以为它表达了实际上并未传达给用户的信息，从而破坏多轮对话的一致性。

值得强调的是，"对话历史应当等于用户实际听到的内容"这一原则**并非本文首次提出，已在多个商用系统中得到实现**。OpenAI 的 Realtime API 提供 `conversation.item.truncate` 事件，客户端在检测到打断时传入 `audio_end_ms`（即音频实际播放到的位置），服务端据此删除未播放的音频及其对应的文本转写，其文档明确指出此举是"为确保上下文中不包含用户尚未听到的文本"[1]。微软 Azure 的 Voice Live 服务提供 `auto_truncate` 能力，在用户于播放期间开始说话时截断上一轮回复，并将会话上下文更新为已播放的部分；其官方文档几乎逐字表述了与上述相同的原则——"会话上下文应当被更新为反映用户实际听到的内容，否则 LLM 会假设它说了从未真正传达给用户的话"[2]。在开源工程框架层面，LiveKit Agents 在打断发生时仅将实际播放的转写文本提交进对话历史，并以 `interrupted=True` 标记该条被打断的消息，其文档说明转写会"被截断以匹配实际说出的输出"[3]。

因此，从"概念"层面看，播放感知的上下文截断已是成熟工程实践。然而，这些系统存在若干共同局限，恰为学术研究留下空间：其一，它们均为**闭源黑盒**（Realtime API、Voice Live）或仅在**框架转发层**做截断（LiveKit），未公开可复现的、面向 LLM 推理内部状态的实现；其二，截断的粒度停留在**受管理的转写文本层面**（删除 transcript 条目），而非对 LLM 的 KV 缓存做显式操作；其三，对"用户听到位置"的确定依赖**近似估计**——Azure 文档承认其"假设回复以实时速度播放"，OpenAI 则依赖客户端上报 `audio_end_ms`。需要审慎指出的是，这些差异是**开源与闭源、显式 KV 操作与转写删除、测量与估计**之别，而非"是否实现了该功能"之别：上述系统确实完成了播放感知的历史管理，本文不以"商用系统未做此事"作为立论依据。

## 2.2 流式语音对话与打断处理的学术研究

学术界围绕级联式与端到端两类架构，从不同角度逼近低延迟自然交互，但**均未涉及基于实际播放位置的 KV 缓存截断**。

在级联式流式延迟优化方向，LTS-VoiceAgent[4]† 提出 Listen-Think-Speak 框架，通过动态语义触发（Dynamic Semantic Trigger）与增量推理降低"边想边说"的延迟，是与本文最接近的级联流水线工作；然而其全文未涉及打断、播放位置、KV 截断或对话角色边界重建，其"Pause-and-Repair"评测针对的是用户输入端的自我修正，而非 TTS 播放被打断的场景。RelayS2S[5]† 采用双路架构，由快速的全双工语音到语音模型推测性地起草回复前缀并立即送入 TTS 以降低音频起始延迟，同时由慢速级联路径续写完整回复，并以一个轻量学习型验证器决定何时提交推测前缀；其"推测换延迟"的框架与本文的推测生成调度部分有重叠，但其目标是延迟起始而非打断作废，不处理 KV 缓存管理。此外，Amazon 的 Predictive ASR 等工作利用部分语音预测完整句子以提前触发 LLM 调用，同样聚焦触发时机而非上下文截断。

在端到端全双工方向，Moshi[6] 将用户与模型语音建模为并行的多流结构，取消显式的说话人轮次，使打断与重叠在架构层面自然涌现。由于其生成与真实时间帧同步、且不存在独立的 TTS 阶段，"生成即播放"隐式成立，因而**从架构上回避了本文所针对的"生成进度、合成进度、播放进度三者不一致"问题**——这从反面印证了级联式场景具有端到端模型所不具备的独特挑战。

在打断检测方向，FireRedChat[7] 通过流式个性化 VAD 抑制误触发的打断，对主说话人语音段打时间戳并在确认打断后立即暂停 TTS 播放；但其贡献停留在**检测与时序控制层**，并未处理已说出一半的助手回复如何进入历史，也不涉及 LLM 的 KV 状态。

## 2.3 LLM 推理与 KV 缓存操作

KV 缓存通过缓存历史 token 的键值状态、避免自回归生成中的重复计算，是 LLM 低延迟推理的基础技术。HuggingFace transformers 库提供 `DynamicCache` 等缓存抽象，其中 `crop` 方法支持将缓存截断到指定长度[8]；vLLM、SGLang 等推理框架则围绕前缀缓存（prefix caching）与缓存复用做了系统级优化。这些机制主要面向上下文窗口溢出、滑动窗口限制与吞吐优化，**并未面向"由外部信号（如音频播放进度）驱动的动态截断与对话角色边界重建"这一场景**。

与本文相关但正交的是若干文本 Agent 场景下的 KV 操作工作：IntentKV[9]† 面向文本 LLM Agent 做跨轮意图感知的 KV 剪枝，Speculative Interaction Agents[10]† 在工具调用被判定失效时丢弃推测性的工具观测结果。两者的截断均由文本/工具逻辑触发，与"用户在音频播放期间听到了多少"无关，也不涉及语音级联流水线。综上，在 transformers/vLLM 生态中，将 KV 缓存按播放进度显式 crop 并重建 ChatML 角色边界，作为一项系统/工程贡献，尚无已公开记录的开源先例。

## 2.4 差异对比与本文定位

综合上述三条脉络，本文与最接近工作的差异如表 2-1 所示。

**表 2-1 本文与相关工作在打断-上下文管理上的差异对比**

| 系统/工作 | 截断依据 | 是否播放感知 | 上下文处理粒度 | 是否级联 | 是否开源 |
|---|---|---|---|---|---|
| OpenAI Realtime API[1] | 客户端上报 `audio_end_ms` | 是 | 删除受管理的转写文本 | 闭源黑盒 | 否 |
| Azure Voice Live[2] | 假设实时播放速度估算 | 是（估计） | 更新会话上下文文本 | 闭源 | 否 |
| LiveKit Agents[3] | 实际转发的转写 | 是 | 词/句级转写，标记 interrupted | 框架级联 | 框架层开源（非 KV） |
| RelayS2S[5]† | 不做打断截断 | 否 | — | 否（双路） | 是 |
| LTS-VoiceAgent[4]† | 不涉及 | 否 | — | 是 | — |
| FireRedChat[7] | 仅检测层暂停 | 否 | 不管理 LLM 上下文 | 是 | 部分 |
| Moshi[6] | 架构隐式 | 隐式 | 无独立 TTS 阶段 | 否（端到端） | 是 |
| **本文** | **实际播放位置→反向映射→token** | **是** | **显式 KV crop + ChatML 角色重建，片段级** | **是** | **是** |

由表可见，播放感知的上下文截断在商用系统中已有实现，但均为闭源且停留在转写文本层面；而学术界与开源级联实现中，尚无工作在 LLM 推理内部以显式的 KV 缓存操作实现这一机制。**本文的定位由此明确**：面向开源、可复现的级联式 ASR→LLM→TTS 流水线，提供播放感知上下文一致性管理的完整实现，其技术核心是基于实际播放位置反向映射的显式 KV 缓存截断（`DynamicCache.crop` 配合 attention mask 与 position id 的同步重算）与 ChatML 角色边界重建。这一定位填补了表 2-1 中"开源、显式 KV 操作、级联"这一空白格，与商用系统形成"可复现的学术参考实现"与"闭源产品"的互补关系。

## 本章小结

本章从商用工程实践、学术研究与 LLM 推理技术三个层面梳理了相关工作。核心结论是：播放感知的打断-上下文管理作为一种理念已被 OpenAI、微软、LiveKit 等系统实现，本文不主张其原创性；但在开源、可复现、面向 LLM KV 缓存内部状态的显式实现层面存在明确空白，这正是本文工作的立足点。下一章将对级联流水线中"生成—合成—播放"三进度不一致问题及相关评测指标进行形式化定义。

---

## 参考文献（本章局部，统稿时并入全文）

> † 标记的 arXiv 预印本编号为核查期（2026 年中）检索所得，正式送审前需复核编号与发表状态。

[1] OpenAI. Realtime API — Conversations. https://developers.openai.com/api/docs/guides/realtime-conversations
[2] Microsoft Azure. Voice Live auto-truncation. https://learn.microsoft.com/en-us/azure/ai-services/speech-service/how-to-voice-live-auto-truncation
[3] LiveKit. Agents — Events & interruption handling. https://docs.livekit.io/agents/build/events/ ; https://github.com/livekit/agents/issues/5038
[4]† LTS-VoiceAgent: A Listen-Think-Speak Framework for Efficient Streaming Voice Interaction. arXiv:2601.19952
[5]† RelayS2S: Dual-Path Speculative Generation for Real-Time Dialogue. arXiv:2603.23346
[6] Défossez et al. Moshi: a speech-text foundation model for real-time dialogue. arXiv:2410.00037
[7] FireRedChat: streaming personalized VAD for barge-in. arXiv:2509.06502
[8] HuggingFace. Transformers — KV cache strategies (DynamicCache). https://huggingface.co/docs/transformers/en/kv_cache
[9]† IntentKV: intent-aware KV pruning for LLM agents. arXiv:2606.09916
[10]† Speculative Interaction Agents. arXiv:2605.13360
