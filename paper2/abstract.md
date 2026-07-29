# 摘要

> 初稿（2026-07-21）。数字与第六章正式数值对齐；主张表述遵守 D-006 prior-art 护栏。

## 中文摘要

级联式架构（流式语音识别 → 大语言模型 → 流式语音合成）是当前语音对话系统的主流方案，但在用户打断（barge-in）场景下存在一个结构性的一致性缺陷：大语言模型的生成进度、语音合成进度与音频实际播放进度三者不同步，若系统将"已生成的全部内容"记入对话历史，模型便会在后续轮次引用用户从未听到的信息，造成对话错乱。本文实验表明，在朴素的生成位置截断策略下，此类未听内容引用在表面重叠口径下可达被打断轮次的 51%，在最保守的"特定信息引用"裁判口径下仍有 2.7%。"对话历史应等于用户实际听到的内容"这一原则已见于闭源商用系统，但学术界与开源社区尚无在大语言模型推理内部以显式 KV 缓存操作实现播放感知上下文管理的工作。

针对上述问题，本文在形式化"生成—合成—播放"三进度指针与听到边界语义的基础上，设计并实现了三项机制：（1）**推测生成调度**：以现成话轮检测模型输出的连续置信度配推测阈值，驱动可作废、可回滚的推测生成过程，以可控的冗余计算换取响应速度（本工作的确定性模拟评测中，推测的提交由用户说完的真值端点触发；真实部署中可增设更高的提交阈值以门控 TTS 播放）；（2）**播放感知的 KV 缓存管理**：建立"播放采样—音频块—文本片段—token 区间"的反向映射时间轴，打断时经显式 KV 截断（注意力掩码与位置编码同步修正）与对话模板角色边界重建，使对话历史与用户所听内容严格一致；（3）**被打断历史的处理策略**：朴素截断、打断标记与轻量模型重写三种策略的实现与消融。上述机制整合为首个开源、可复现的级联式参考实现。

在 7B 主模型与 MultiWOZ 派生数据上的实验表明：播放感知截断使未听内容引用率在片段口径下由机制构造保证为零，且片段级截断粒度的量化误差在人工判定下未产生任何可感知的特定引用（0/24）；打断响应关键路径（反查+截断）恒为亚毫秒级且与上下文长度无关，KV 复用相对重新预填充在 8k 上下文处加速 39.7 倍；推测调度给出浪费率—延迟的连续可调权衡前沿，拐点处以约 4.5% 的 token 浪费将说完后首 token 延迟从 48.5 ms 降至 12.1 ms。本文同时提出"上界检测器—下界裁判—人工仲裁"三层一致性评测协议（人机一致性 κ=0.649），对同类语音对话系统评测具有独立参考价值。

**关键词**：语音对话系统；打断；KV 缓存；推测生成；上下文一致性；低延迟

## Abstract

Cascaded pipelines (streaming ASR → LLM → streaming TTS) remain the mainstream architecture for spoken dialogue systems, yet they suffer from a structural consistency flaw under user barge-in: the LLM's generation progress, the TTS synthesis progress, and the actual audio playback progress advance at different paces. If the system commits everything it has generated to the dialogue history, the model will later refer to content the user never heard, derailing the conversation. Our experiments show that under naive generation-point truncation, such unheard-content references occur in up to 51% of interrupted turns by a surface-overlap criterion, and still 2.7% under the most conservative LLM-judge criterion of specific-information reference. The principle that "dialogue history should equal what the user actually heard" has been practiced in closed-source commercial systems, but no academic or open-source work has realized playback-aware context management inside LLM inference via explicit KV-cache operations.

Building on a formalization of the generation–synthesis–playback progress pointers and the heard-boundary semantics, this thesis designs and implements three mechanisms: (1) **speculative generation scheduling**, which softens endpoint detection into an abortable, roll-backable speculative process driven by the continuous confidence of an off-the-shelf turn-detection model with a speculation threshold, trading controlled redundant computation for responsiveness (in our deterministic-simulation evaluation, speculation is committed by the ground-truth end-of-utterance; a higher commit threshold may gate TTS playback in real deployment); (2) **playback-aware KV-cache management**, which maintains a reverse-mapping timeline across playback samples, audio chunks, text segments, and token spans, and upon barge-in performs explicit KV truncation (with synchronized attention-mask and position-encoding correction) and chat-template role-boundary rebuilding, so that the dialogue history strictly matches what the user heard; (3) **strategies for interrupted history**, implementing and ablating naive truncation, interruption marking, and lightweight-model rewriting. These mechanisms are integrated into the first open-source, reproducible cascaded reference implementation.

Experiments with a 7B backbone on MultiWOZ-derived data show that playback-aware truncation reduces the unheard-reference rate to zero by construction at segment granularity, and the quantization error of segment-level truncation causes no human-perceivable specific reference (0/24); the barge-in critical path (reverse lookup + truncation) is constantly sub-millisecond and independent of context length, while KV reuse achieves a 39.7× speedup over re-prefilling at 8k context; speculative scheduling yields a continuously tunable waste–latency frontier, cutting post-utterance first-token latency from 48.5 ms to 12.1 ms at the knee point for about 4.5% token waste. The thesis further contributes a three-layer consistency evaluation protocol — upper-bound rule detector, lower-bound LLM judge, and human arbitration (human–judge agreement κ=0.649) — of independent reference value for evaluating similar systems.

**Keywords**: spoken dialogue system; barge-in; KV cache; speculative generation; context consistency; low latency
