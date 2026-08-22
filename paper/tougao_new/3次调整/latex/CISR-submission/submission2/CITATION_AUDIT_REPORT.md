# Citation Audit Report

- **目标目录**：`paper/tougao_new/3次调整/latex/CISR-submission/submission2`
- **审计对象**：`main.tex`、`refs.bib`、`main.bbl`
- **审计日期**：2026-08-22
- **模式**：`academic-paper / citation-check` 等效人工执行（当前会话技能未注册，直接调用返回 `Skill not found`）
- **操作边界**：只读核查；未修改 `main.tex`、`refs.bib` 或其他论文文件

## 1. 总结结论

**引用机械完整性通过，但论文目前仍有需要修改的引用质量问题，不建议在修复高优先级项之前直接投稿。**

### 已通过

- `refs.bib` 共 **25 条文献，25 条全部在正文引用**；无孤立条目。
- 正文共有 **26 个 `\cite{}` 命令、27 次引用使用、25 个唯一 key**。
- 无缺失 citation key、无未定义引用、无重复 BibTeX key、无重复 DOI、无重复规范化标题。
- 参考文献编号连续为 **[1]–[25]**，首次引用顺序与最终编号一致。
- 隔离目录完整执行 `pdflatex → bibtex → pdflatex ×2` 成功；重建的 `main.bbl` 与现有版本字节一致。
- 未发现伪造 DOI；现有 DOI 均可由 Crossref、DataCite、IEEE、ACM、ACL Anthology、PMLR、arXiv 或正式出版页面核验。
- LocalAgreement 的两条引用使用正确：Liu et al. 是 partial-hypothesis/LocalAgreement 策略来源，Macháček et al. 是 Whisper-Streaming 适配来源。
- LibriSpeech、AISHELL-1、MUSAN、MultiWOZ、CrossWOZ、BGE-M3 的数据集/模型文献与正文对象匹配。

### 需要处理

- **9 项高优先级论断—引用问题**：主要集中在 Introduction 和 Related Work，包括生产系统普遍采用级联架构、rVAD/Silero 混用、Conformer、KV cache/vLLM/TGI、FlashAttention、StreamingLLM、Mini-Omni/Moshi/LLaMA-Omni、Whisper 鲁棒性以及 CosyVoice “service” 表述。
- **5 项确定的 BibTeX 元数据错误/缺漏**：`ref5`、`ref7`、`ref9`、`ref14`、`ref20`。
- **3 项专名大小写保护问题**：Whisper、Mandarin、M3-Embedding 在 `IEEEtran.bst` 生成后被错误小写化。
- **DOI 显示风险**：`refs.bib` 中 21 条具有合法 DOI，但仓库的 `IEEEtran.bst` 1.12 忽略 `doi` 字段，最终 `main.bbl` 和 PDF 中 **不显示任何 DOI**。
- Silero VAD 和 `deepseek/deepseek-v4-flash` 属动态软件/服务引用，缺少不可变版本标识，长期复现性有限。

## 2. 高优先级：正文论断与引用不匹配

### H1. “most production systems” 是无来源的行业普遍性断言

- **位置**：`main.tex:48`
- **当前论断**：多数生产系统仍采用 ASR–LLM–TTS 级联架构，并通常在端点后才启动下游处理。
- **问题**：这是行业覆盖率/普遍性断言；当前 GPT-4o 或 Gemini 语境不能证明“多数生产系统”。
- **建议**：
  - 最稳妥：改为本文可控制的范围，例如 “many deployable voice systems use a cascaded architecture” 或 “the conventional serial cascade considered here…”。
  - 若保留“most”，必须补充明确统计生产系统架构采用情况的调查或产业报告。
  - 若继续点名 GPT-4o/Gemini，应分别引用一手系统卡/技术报告。

### H2. rVAD 文献错误地附着到本文 Silero VAD / 级联架构叙述

- **位置**：`main.tex:52`，当前 `\cite{ref2}`。
- **问题**：`ref2` 是 Tan et al. 的 **rVAD**，本文实际实现使用 **Silero VAD**。该引用既不支持 Silero，也不支持部署成本、隐私治理或模块替换等级联架构主张。
- **建议**：删除该处 `ref2`，在第一次出现 Silero 实现处使用 `ref13`。如果需要 VAD 的通用背景，另加 VAD 综述，并明确它与 Silero 的区别。

### H3. Conformer 应引用原始论文，且“mainstream in industry”证据不足

- **位置**：`main.tex:79`，当前 `\cite{ref5}`。
- **问题**：`ref5` 是 ASR 综述，不是 Conformer 原始论文；不能理想支撑“Google proposed”这一归属，更不能充分支撑“迅速成为工业主流”。
- **建议主文献**：Gulati et al., “Conformer: Convolution-augmented Transformer for Speech Recognition,” Interspeech 2020, DOI `10.21437/Interspeech.2020-3015`。
- **建议措辞**：删除或软化 “quickly became mainstream in industry”。

### H4. KV cache、vLLM 和 TGI 被引用到不匹配的模型压缩综述

- **位置**：`main.tex:83`，当前 `\cite{ref8}`。
- **问题**：`ref8` 是 LLM 模型压缩综述，不是 KV-cache 原理、vLLM 或 TGI 的直接来源。
- **建议**：
  - vLLM/PagedAttention：引用 Kwon et al., “Efficient Memory Management for Large Language Model Serving with PagedAttention,” SOSP 2023。
  - TGI：引用 Hugging Face 的稳定版本文档/仓库，并锁定版本或 commit。
  - KV-cache 原理/复杂度：补充权威 Transformer inference/KV-cache 技术来源；不要把模型压缩综述作为唯一依据。

### H5. FlashAttention 使用了非主文献

- **位置**：`main.tex:83`，当前 `\cite{ref9}`。
- **问题**：`ref9` 是 Efficient Transformers 综述，不是 FlashAttention 的直接来源。
- **建议主文献**：Dao et al., “FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness,” NeurIPS 2022 / arXiv:2205.14135。
- **建议措辞**：使用 “improves throughput in reported settings”，避免无数字支撑的 “substantially”。

### H6. StreamingLLM/attention sink 没有直接引用

- **位置**：`main.tex:83`。
- **问题**：正文直接介绍 StreamingLLM 和 attention sink，但相邻的 `ref9` 实际属于前一句 FlashAttention，不能覆盖 StreamingLLM。
- **建议主文献**：Xiao et al., “Efficient Streaming Language Models with Attention Sinks,” ICLR 2024 / arXiv:2309.17453。
- **边界提醒**：StreamingLLM 通过保留初始 sink tokens 与近期窗口维持有界缓存，并不保存任意久远内容。

### H7. Mini-Omni、Moshi、LLaMA-Omni 缺少各自一手引用

- **位置**：`main.tex:87`。
- **问题**：`ref11` 是宽泛的多模态 LLM 综述，不能替代三个具体系统的主文献；“Open-source systems” 对三者授权状态也可能过宽。
- **建议主文献**：
  - Mini-Omni：arXiv:2408.16725；
  - Moshi：arXiv:2410.00037；
  - LLaMA-Omni：arXiv:2409.06666。
- **建议措辞**：在未逐项核实代码/权重许可前，将 “Open-source systems” 改为 “recent systems”。

### H8. Whisper 鲁棒性引用错配

- **位置**：`main.tex:116`，当前 `\cite{ref12}`。
- **问题**：`ref12` 是 2022 年自监督语音表征综述，早于 Whisper 正式论文，不支持 Whisper 的架构、弱监督预训练或多语言/噪声鲁棒性。
- **建议**：改用本文已有的 Whisper 主文献 `ref6`。措辞宜限定为 “strong zero-shot robustness across multiple languages and reported distribution-shift/noise benchmarks”，避免泛化为所有语言/口音/噪声条件。

### H9. CosyVoice 论文不能证明具体的 “Alibaba service”

- **位置**：`main.tex:346`，当前 `\cite{ref17}`。
- **问题**：`ref17` 支持 CosyVoice 模型，但不记录本文使用的具体 hosted API/service、endpoint、版本或部署参数。
- **建议**：
  - 若实际为本地部署：将 “Alibaba's CosyVoice service” 改为 “a CosyVoice implementation/model”，并记录 checkpoint/revision。
  - 若实际为云服务：补充正式 API 文档、服务型号、访问日期和版本。

## 3. 中优先级：建议补强或软化的引用

| ID | 位置 | 问题 | 建议 |
|---|---|---|---|
| M1 | `main.tex:50` | `ref1` 能支持 GPT-4o 的 232/320 ms，但长句末引用范围会让人误以为同时支持“接近人类反应速度”和市场竞争结论 | 将 `\cite{ref1}` 紧跟在 320 ms 后；人类比较另引 turn-taking 文献或明确为 OpenAI 的比较 |
| M2 | `main.tex:77` | “early streaming ASR architectures were dominated by RNN-T” 过强 | 改为 “prominently included RNN-T”，或补 RNN-T 主文献/采用率证据 |
| M3 | `main.tex:77` | Vaswani et al. 支持 Transformer/global attention，但不直接支持 ASR 精度提升；“cannot operate streaming”过于绝对 | 改为“不修改/限制 attention/context 时无法实现有界延迟流式处理”，并补 ASR Transformer 文献 |
| M4 | `main.tex:79` | overlapped stitching、causal conversion、sequence-level regularization 等流式适配方法无直接引用 | 为实际指向的方法补主文献，或改写为不对应特定论文的高层分类 |
| M5 | `main.tex:83` | “second major bottleneck” 是系统依赖的排序断言 | 改为 “can become another major bottleneck” |
| M6 | `main.tex:87` | 多模态 LLM 综述对“speech-native streaming dialogue”支持过宽 | 保留综述作为总背景，同时补具体系统主文献 |
| M7 | `main.tex:98` | “Existing work either…or…” 是遗漏大量管线调度/级联增量工作的穷尽式二分 | 改为 “Much prior work focuses on…” 并补 pipeline/incremental cascade 工作 |
| M8 | `main.tex:117` | Whisper-Turbo 的 128-bin log-Mel 是 checkpoint 特定事实，原始 Whisper 论文主要描述 80 bins | 补官方 `whisper-large-v3-turbo` model card/config 引用 |
| M9 | `main.tex:121–124` | Whisper chunk boundary omission/repetition/rewrite 属外部可泛化主张 | 在此处补 Macháček/Liu 引用，而非只在 baseline 小节引用 |
| M10 | `main.tex:140–157` | KV-cache 数学与历史背景缺少直接技术来源 | 加权威 KV-cache/inference 来源；保留当前“naive model”限制说明 |
| M11 | `main.tex:217–221` | 500 ms/2 s/300 ms 的效用被写得像已实证，但无参数消融 | 改为 “chosen/intended to…”，或补 VAD 参数消融/端点文献 |
| M12 | `main.tex:367` | “five standard groups” 实际是本文自定义分组 | 改为 “five predefined/analysis groups”，无需引用 |
| M13 | `main.tex:379` | Qwen2 技术报告支持模型家族，但未锁定 `Qwen2-7B-Instruct` 具体 checkpoint | 在方法/可复现材料记录 repo ID、revision、Transformers 版本、dtype |
| M14 | `main.tex:518` | BGE-M3 论文支持模型，但具体 CLS pooling/L2 cosine 实现需要 checkpoint/代码信息 | 记录模型 revision、软件版本、pooling 代码；核实是否使用模型推荐 dense output |
| M15 | `main.tex:518` | Command Code “latest” 是可变 alias，非不可变模型版本 | 保留调用日期、endpoint、prompt hash、参数和响应 manifest；若服务返回 build ID 则记录 |
| M16 | `main.tex:568` | `ref22` 支持 half-duplex/turn-taking，但未必独立支持所有 paralinguistic-loss 细节 | 将两项拆开；后者补语音/副语言文献或表述为架构直接观察 |

## 4. 确定的 BibTeX 元数据错误

### B1. `ref5` 缺页码

- **位置**：`refs.bib:40–48`
- **当前问题**：缺少正式页码。
- **修正**：增加 `pages = {1--64}`。
- **证据**：DOI `10.1561/116.00000050` / Crossref。

### B2. `ref7` 作者不全，ACM article number 被误写为页码

- **位置**：`refs.bib:60–69`
- **当前问题**：
  - 仅列 3 位作者后写 `and others`，实际共 6 位作者；
  - `pages = {209}` 错误，209 是 article number；正式页码为 1–35。
- **修正建议**：

```bibtex
author    = {Zheng, Yue and Chen, Yuhao and Qian, Bin and Shi, Xiufang and Shu, Yuanchao and Chen, Jiming},
articleno = {209},
pages     = {1--35},
```

### B3. `ref9` ACM article number 被误写为页码

- **位置**：`refs.bib:81–90`
- **当前问题**：`pages = {109}` 中 109 是 article number；正式页码为 1–28。
- **修正建议**：

```bibtex
articleno = {109},
pages     = {1--28},
```

### B4. `ref14` 把出版社所在地写成会议地点

- **位置**：`refs.bib:131–140`
- **当前问题**：EMNLP 2018 会议地点是 Brussels, Belgium；`Stroudsburg` 是 ACL 出版组织所在地。
- **修正**：`address = {Brussels, Belgium}`。
- **可选**：正式标题写作 `{MultiWOZ} - A Large-Scale ...`。

### B5. `ref20` Qwen2 作者表严重截断

- **位置**：`refs.bib:171–179`
- **当前问题**：`and others` 在 IEEE 输出为 “et al.”，语法合法，但不是完整元数据。
- **修正**：如果投稿要求完整作者表，替换为 arXiv 正式作者列表；若考虑双栏篇幅，可保留 `et al.`，但应视为有意缩写而不是完整记录。
- **同时建议**：按预印本改为 `@misc` + `eprint` + `archivePrefix`。

## 5. 专名大小写保护

`IEEEtran.bst` 会将标题改为 sentence case，下列专名需要额外花括号：

| 位置 | 当前渲染 | 建议 BibTeX 标题 |
|---|---|---|
| `refs.bib:208` | “Turning whisper…” | `Turning {Whisper} into Real-Time Transcription System` |
| `refs.bib:190` | “open-source mandarin…” | `... Open-Source {Mandarin} Speech Corpus ...` |
| `refs.bib:227` | `M3-embedding` | `{{M}3-{Embedding}}: ...` 或保护完整 `M3-Embedding` |

## 6. DOI 显示问题

### 事实

- `refs.bib` 中 21 条记录含合法 DOI。
- 当前仓库的 `IEEEtran.bst` 1.12 不声明/输出 `doi` 字段。
- 因而 `main.bbl` 和最终 PDF 中 DOI 数量为 **0**。
- 这不是 BibTeX 语法错误，而是 bibliography style 的输出行为。

### 如何处理

1. 先确认 CISR 是否要求最终参考文献显示 DOI。
2. 如果必须继续使用会议提供的 `IEEEtran.bst`，可以将 DOI 以该样式实际会输出的字段保存，例如：
   - `url = {https://doi.org/...}`；或
   - 经会议允许的 `note = {doi: ...}`。
3. 不要仅依赖当前 `doi = {...}` 字段，因为它们虽保留在数据库中，却不会出现在 PDF。
4. 不建议未经会议模板许可自行换 `.bst`；应优先核对投稿规范。

## 7. 软件、模型和动态服务的可复现性

### Silero VAD (`ref13`)

- 官方 GitHub 来源有效，但不是同行评审论文，也没有 DOI。
- 当前引用没有 release/tag/commit；官方引用说明建议锁定 commit。
- 建议在不虚构的前提下，从实际实验记录补充所用 commit 或 artifact hash，并增加访问日期。

### CosyVoice (`ref17`)

- arXiv/DOI 真实，属于预印本。
- 支持 CosyVoice 模型，不足以单独支持具体 hosted service/API。
- 建议补实际 checkpoint/service 文档和版本。

### BGE-M3

- ACL 2024 学术引用正确。
- 精确复现仍需模型 revision、tokenizer revision、pooling/normalization 实现；当前实验元数据已承认 HF revision unknown，此限制应保留。

### Command Code judge

- `https://commandcode.ai/models/deepseek-v4-flash` 在 2026-08-22 可访问，能证明 service identifier。
- 这是 Command Code 服务目录，不是 DeepSeek 官方技术报告，也不是同行评审来源。
- `latest` 是动态 alias，不能保证未来指向相同部署。
- 论文当前称其为 Command Code service model 是正确边界；不应改写成官方 DeepSeek-V4 学术模型。

## 8. 可接受的小问题

- `main.bbl` 中 rVAD 条目引发一个 `Underfull \hbox (badness 1642)`；没有 reference-related overfull box。IEEE 双栏参考文献中通常可接受。
- `\begin{thebibliography}{10}` 是两位数编号宽度提示，不代表只有 10 条文献；实际编号 [1]–[25] 正确。
- `ref4`（NeurIPS）和 `ref6`（PMLR）官方元数据没有 DOI；Silero 与 Command Code 是软件/网页来源。未发现可确定但漏填的 DOI。
- `and others` 在 `ref7`/`ref20` 会合法输出 “et al.”，但 `ref7` 仅 6 位作者，不建议截断；`ref20` 是否展开可视版面与会议规范决定。

## 9. 建议修复顺序

### 投稿前必须优先处理

1. 用 `ref6` 替换 Whisper 鲁棒性处的错误 `ref12`。
2. 删除/替换 rVAD 对 Silero/级联架构的误导性引用。
3. 补 Conformer、vLLM/KV cache、FlashAttention、StreamingLLM 的主文献。
4. 为 Mini-Omni、Moshi、LLaMA-Omni 分别加一手文献，或删除具体系统名。
5. 软化或证明 “most production systems” 等行业普遍性断言。
6. 修正 `ref5`、`ref7`、`ref9`、`ref14` 的确定元数据错误。
7. 处理 Whisper/Mandarin/M3-Embedding 的大小写保护。

### 投稿规范确认后处理

8. 确认 CISR 是否要求 DOI 出现在 PDF；若要求，调整 DOI 的输出方式。
9. 决定是否展开 Qwen2 完整作者表、是否将 arXiv 条目统一为 `@misc`。
10. 为 Silero、Qwen2 checkpoint、Whisper-Turbo、CosyVoice、BGE-M3 和 judge 服务补不可变 revision/commit 或明确 unknown 边界。

## 10. 最终判定

| 维度 | 结果 |
|---|---|
| 引用 key 闭合 | PASS |
| BibTeX 可解析与编译 | PASS |
| 重复 key/DOI/标题 | PASS |
| DOI 真实性 | PASS |
| 所有条目均被正文引用 | PASS |
| 文献元数据准确性 | NEEDS REVISION（5 个确定问题） |
| 论断—来源匹配 | NEEDS MAJOR REVISION（9 个高优先级问题） |
| DOI 最终显示 | NEEDS VENUE CHECK（当前 PDF 为 0） |
| 软件/服务长期复现性 | PARTIAL |
| 当前引用层面投稿就绪 | **NOT YET** |

本报告只记录问题与建议，没有对论文源文件做任何修改。
