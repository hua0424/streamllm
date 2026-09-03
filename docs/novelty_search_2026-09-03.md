# Paper 2 targeted/scoping novelty search（2026-09-03）

## 1. 目的与范围

本检索用于回应二审 CF-08，不构成系统综述或专利检索。研究问题是：公开资料中是否已有系统同时公开以下机制与可复算证据：

1. 级联 ASR→LLM→TTS；
2. software-consumed-sample cursor；
3. cursor→TTS fragment→assistant token span 映射；
4. interruption 后 in-place KV crop；
5. attention mask、token ledger 与 role/EOT state 同步恢复；
6. crop-integrity 与恢复路径的可复算验证。

检索截止日期：**2026-09-03**。纳入公开学术记录、正式预印本、第一方产品/框架文档和第一方开源仓库。商业系统只按公开接口语义比较，不推断闭源内部架构。

## 2. 信息源

- 学术发现：Google Scholar 或等价跨出版商索引；
- 预印本：arXiv；
- speech/dialogue：ACL Anthology、ISCA Archive/Interspeech；
- systems：ACM Digital Library、IEEE Xplore、NeurIPS Proceedings；
- 元数据：DOI/Crossref/出版商落地页；
- 第一方实现语义：OpenAI、Microsoft Azure、LiveKit、Hugging Face；
- 第一方开源：GitHub 项目与 issue。

### 本次可复查性状态

- 仓库保留了 2026-05-21 的 22-source novelty 对抗核查，见 `docs/research_novelty_check.md`；该核查有来源与逐项结论，但未保留原始查询日志、每库结果数、去重和逐条排除记录。
- 2026-09-03 补查重建并冻结了下面的查询族。Google 搜索触发 reCAPTCHA，ACL Anthology 页面出现超时/控制失效，必应中文索引将 `barge-in` 错配为航运词义，ACM 页面返回 403，部分 arXiv/官方页面被工具提供方拒绝。因此这些渠道被标记为**访问受限**，不得据此写“检索零结果”。
- Azure 官方页成功核验：*Handle voice interruptions in chat history*，`ms.date=2026-04-28`，明确 `auto_truncate` 在播放中检测到用户说话时只保留打断前播放的响应部分，并明确其估算假设为实时播放速度。
- DOI `10.21437/Interspeech.2023-211` 成功解析到 ISCA Archive 的 Personalized Predictive ASR 页面；DOI `10.1145/3600006.3613165` 成功解析到 ACM PagedAttention 页面，但出版商正文访问受限。
- 其他来源的身份由 DOI、arXiv identifier 或第一方 URL 及仓库既有核查支撑；提交前应再次打开官方记录，核对完整作者、版本和是否已有正式发表版本。

因此，本报告支持的是**公开来源范围内的限定性非识别结论**，不是穷尽性“全球首次”。

## 3. 查询式

数据库不支持完整 Boolean 时，将 OR 子句拆成独立查询并保留相同概念边界。

### A. 级联流式语音对话与延迟

1. `"cascaded spoken dialogue" streaming latency ASR LLM TTS`
2. `("voice agent" OR "spoken dialogue system") AND (streaming OR incremental) AND (latency OR "time to first token")`
3. `("ASR LLM TTS" OR "ASR→LLM→TTS") AND streaming AND latency`
4. `("incremental dialogue processing" OR "incremental spoken dialogue") AND response generation`

### B. turn-taking、interruption 与 barge-in

5. `("barge-in" OR interruption) AND ("spoken dialogue system" OR "voice assistant")`
6. `("barge-in" OR interruption) AND (history OR context OR memory) AND dialogue`
7. `("turn-taking" OR "end-of-turn" OR endpointing) AND streaming AND (LLM OR dialogue)`
8. `("semantic VAD" OR "semantic endpoint detection") AND "voice agent"`
9. `site:aclanthology.org ("barge-in" OR interruption) ("dialogue state" OR history)`
10. `site:ieeexplore.ieee.org ("barge-in" OR "turn-taking") "spoken dialogue"`

### C. pre-end-of-turn candidate response

11. `("speculative response generation" OR "predictive response generation") AND ("voice assistant" OR dialogue)`
12. `("predictive ASR" OR "partial utterance prediction") AND (prefetch OR cache OR response)`
13. `("before end-of-turn" OR "before utterance completion") AND response AND dialogue`
14. `("speculative generation" AND "real-time dialogue") NOT "speculative decoding"`
15. `("early response" OR "anticipatory response") AND partial ASR`

### D. KV state、crop 与 rollback

16. `("KV cache" OR "key-value cache") AND (crop OR truncate OR rollback) AND dialogue`
17. `("KV cache" OR "key-value cache") AND interruption AND conversation`
18. `"DynamicCache.crop" AND (dialogue OR assistant OR conversation)`
19. `("chunked prefill" OR "incremental prefill") AND streaming LLM`
20. `("prefix cache" OR "prefix caching") AND multi-turn dialogue`
21. `("KV cache pruning" OR "KV cache eviction") AND cross-turn agent`

### E. playback/listening-aware history

22. `("playback-aware" OR "playback conditioned") AND (history OR context OR memory) AND dialogue`
23. `("listening-aware" OR "heard by the user") AND conversational history`
24. `("actually heard" OR "spoken output") AND (truncate OR truncation) AND context`
25. `"audio_end_ms" AND truncate`
26. `"conversation.item.truncate" AND transcript`
27. `"auto_truncate" AND "session context"`
28. `("playback cursor" OR "audio cursor") AND (LLM OR dialogue) AND history`
29. `site:github.com ("playback" OR "audio_end_ms") ("KV cache" OR "past_key_values")`

## 4. 纳排规则

### 纳入

至少满足一项：流式/增量语音对话及其延迟；从不完整 user 输入提前计算响应；语音系统中的打断控制；依据播放/转发进度修改 assistant 消息或上下文；KV crop/rollback/prefix reuse/cross-turn pruning；第一方文档明确了截断语义。来源必须具有 DOI、官方出版页、官方预印本或第一方 URL。

### 排除

纯 speculative decoding（固定 prompt 下 draft-target token 验证）、无对话/跨轮/rollback 关系的通用 KV 压缩、只做回声消除而不改历史的 barge-in、只停止音频而未说明历史处理的播放 API、可被第一方来源替代的二手博客，以及仅凭营销材料推断闭源内部实现的内容。

### snowballing 起点

Personalized Predictive ASR、LTS-VoiceAgent、RelayS2S、FireRedChat、Moshi、OpenAI/Azure/LiveKit truncation 文档、PagedAttention、SGLang 与 Transformers DynamicCache。由于本轮索引访问受限，未声称完成 forward-citation 穷尽检索。

## 5. 最近邻矩阵与证据边界

| 来源 | 已建立的公开事实 | 与本文的边界 |
|---|---|---|
| OpenAI Realtime API[1] | 客户端按 `audio_end_ms` 截断未播放音频和对应 transcript | 最接近高层语义先例；公开资料不暴露是否使用级联栈、token span 或 KV crop |
| Azure Voice Live[2] | `auto_truncate` 在播放期打断后更新 session context；采用实时播放速度假设 | 高层“history 反映已播放内容”是 prior art；本文不得声称该原则原创，也不得声称物理精度优于 Azure |
| LiveKit Agents[3] | 截断 interrupted transcript/history，使其匹配 spoken output | 最接近开源框架级先例；不公开 transformer KV/role-state 修复 |
| LTS-VoiceAgent[4] | semantic triggering 与 incremental reasoning | 最近的级联学术邻居；关注提前触发，不公开 playback-driven KV crop |
| RelayS2S[5] | response-level candidate prefix 与验证/续写 | 与 compute-before-commit 重叠；不是 draft-target speculative decoding，也不做 interruption-history repair |
| Predictive ASR[18] | 从 partial ASR 预测完整输入并预取下游响应 | 输入侧提前计算 prior art；不处理被打断 assistant 的已交付边界 |
| FireRedChat[7] | streaming/personalized VAD 与 interruption control | 主要回答何时停；本文回答外部打断到达后哪些模型状态继续保留 |
| Moshi[6] | 端到端 speech/text stream 与重叠交互 | 架构对比；不能由此推断设备/声学交付与模型生成恒等 |
| Transformers DynamicCache[8] | cache abstraction 与 crop primitive | crop 原语不是创新；本文对象是外部 playback state 到 role-safe state correction 的联结与 exact 验证 |
| PagedAttention[19] / SGLang[20] | KV 内存管理、共享、prefix reuse | 通用 serving prior art，不以软件播放游标选择对话提交边界 |
| IntentKV[9] | text-agent 跨轮 intent-aware KV pruning | 最近的跨轮 KV 邻居；信号来自文本意图而非 speech delivery |

## 6. 可使用与禁止的 novelty 表述

### 可使用

> 在截至 2026-09-03 的目标性公开来源检索中，我们识别到按播放进度截断 transcript/session history 的商用与开源先例，以及分别处理提前响应计算、打断检测和 KV 操作的研究。我们未在报告的公开来源范围内识别到这样一个可检视的级联实现：它把 software-consumed-sample cursor 映射至 TTS fragment 和 assistant token span，再执行 in-place KV crop、显式 role/EOT recovery，并提供可复算的 state-integrity 与延迟工件。该结论是范围受限的非识别结果，不排除未发表、闭源或本次索引访问受限而未收录的系统。

### 禁止

- “首次提出只保留用户听到的内容”；
- “现有系统忽略被打断历史”；
- “LiveKit 只做检测”；
- “商业系统没有 KV crop”；
- “KV crop 本身是创新”；
- “端到端模型消除了播放差异”；
- “系统测得用户实际听到的内容”；
- “穷尽检索证明不存在其他方法”。

## 7. 提交前复核清单

- 使用可访问的 Scholar/ACL/IEEE/ACM/Crossref 环境重跑上述查询，保留导出文件、结果数、去重与排除理由；
- 重新核验 2026 年预印本的版本、完整作者和正式发表状态；
- 拆分 LiveKit 文档与 GitHub issue 的引用功能；
- 对每个 DOI 检查出版商元数据；
- 给在线资料补统一访问日期；
- 若无法完成索引重跑，论文应明确称“targeted public-source scan”，不得称“systematic review”。
