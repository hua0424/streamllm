# 引用文献复核报告（Citation Recheck）

- **目标稿件**：`main.tex`
- **参考文献库**：`refs.bib`
- **生成日期**：2026-08-22
- **检查模式**：`academic-paper / citation-check`（Fidelity，Low oversight）
- **引用样式**：CISR 模板所用 `IEEEtran.bst`
- **处理边界**：本报告只记录问题与建议修订，不自动改写 `main.tex` 或 `refs.bib`。

## 1. 结论

**当前引用系统在机械层面闭合，但尚不宜将“引用内容匹配”判为完全通过。** 31 个唯一正文引文键与 31 条 BibTeX 记录一一对应，重编译后无未定义引用；全部来源均能通过 DOI、正式论文页面、arXiv、OpenReview、模型仓库或项目页面验证，未发现虚构、正式撤稿或 Expression of Concern。

投稿前建议至少处理以下两项：

1. **SERIOUS：`main.tex:99` 的 `ref11` 与“speech-native dialogue systems”主张不匹配。** `ref11` 是一般多模态大模型综述，不直接覆盖实时语音原生对话系统；应删除该引文、换用专门综述，或收窄表述。
2. **MINOR：`refs.bib:79` 的 `MIT` 未保护大小写。** `IEEEtran.bst` 已在 `main.bbl:156` 将其显示为 `mIT-licensed software`；建议写成 `{{MIT}-licensed software; ...}`。

另有 7 处轻微论断边界漂移、2 处动态来源/完整性警示，以及 6 组建议补引的一般性方法陈述。它们不构成文献虚构或重大歪曲，但会影响“每个外部论断均由最直接来源支撑”的严格合规结论。

> **对既有报告的更正**：`CITATION_AUDIT_REPORT.md:19` 的 `Primary-source attribution | PASS` 已被本次逐条外部核验推翻，应至少改为 `NEEDS REVISION`，直至 `ref11` 问题处理完毕。

## 2. Summary

| Metric | Result |
|---|---:|
| `\cite{...}` 命令数 | 38 |
| Citation key 出现总次数 | 42 |
| 唯一正文引文键 | 31 |
| `refs.bib` 条目 | 31 |
| `main.bbl` 条目 | 31 |
| 正文孤儿引文 | 0 |
| 未被正文引用的 BibTeX 条目 | 0 |
| 重复 key / DOI / URL / 规范化标题 | 0 / 0 / 0 / 0 |
| 已核验 DOI | 23 |
| 已核验 URL 来源 | 6 |
| 无 DOI/URL、由正式会议页面核验 | 2（`ref4`, `ref6`） |
| 经核验应有但缺失的 DOI | 0 |
| Wrong-source attribution | 1 |
| Minor paraphrase drift | 7 个引用键 |
| 动态来源或来源完整性警示 | 2 个引用键 |
| 正式撤稿 / Expression of Concern | 0 / 0 |
| 自引 | 0/31（0%） |
| 近 5 年来源（2022–2026） | 22/31（71.0%） |
| 2023–2026 来源 | 21/31（67.7%） |
| 超过 10 年来源 | 2/31（6.5%；均为 2015 年基础数据集论文） |
| 单句超过 5 条引用 | 0 |
| 可见参考文献格式错误 | 1（`MIT` → `mIT`） |
| 最终 undefined citations / references | 0 / 0 |

## 3. 必须处理的引用错配

### SERIOUS-1：`ref11` 不支持 speech-native dialogue 主张

- **正文位置**：`main.tex:99`
- **参考文献位置**：`refs.bib:62-70`
- **当前文字**：

  > Recent speech-native dialogue systems instead model or generate speech within an integrated architecture~`\cite{ref11}`.

- **来源实际范围**：Yin et al., *A survey on multimodal large language models*，讨论一般多模态 LLM、模态编码器和多模态输入/输出；未直接建立 speech-native、实时 spoken dialogue 或 speech-to-speech dialogue 系统谱系。
- **判定**：`wrong-source attribution`，但不是虚构文献，也不是 `major misrepresentation`。
- **证据**：
  - <https://doi.org/10.1093/nsr/nwae403>
  - <https://arxiv.org/abs/2306.13549>
  - <https://ar5iv.labs.arxiv.org/html/2306.13549>
  - <https://api.openalex.org/works/https://doi.org/10.1093/nsr/nwae403>
- **建议修订（优先）**：删除该句中的 `\cite{ref11}`，让紧随其后的 Mini-Omni、Moshi 与 LLaMA-Omni 原始论文直接支撑系统实例；如需要综述性来源，则另引专门的 speech LLM / spoken dialogue foundation model 综述。
- **可选收窄表述**：若必须保留 `ref11`，将主张改为“multimodal LLMs increasingly support audio input and output”，不要把该来源描述为 speech-native dialogue 综述。

## 4. 建议修订的论断—来源边界

| ID | 位置 | 引文键 | 判定 | 问题与建议 |
|---|---|---|---|---|
| MEDIUM-1 | `main.tex:89` | `ref3` | Minor drift | “RNN-T 的递归结构”过度概括；现代 transducer 不必然使用递归编码器。建议改成：RNN-T 在有限前瞻编码器下支持从左到右流式解码，而递归实现的时序依赖会限制时间维并行。 |
| MEDIUM-2 | `main.tex:91` | `ref6` | Minor drift | Whisper 原论文描述固定窗口推理，但未明示“primarily offline”。建议联合引用 `ref6,machacek2023turning`，或改写为 published inference procedure uses fixed windows and does not specify bounded-latency streaming commitment。`main.tex:126` 对模型与训练信息的引用准确。 |
| MEDIUM-3 | `main.tex:95` | `ref7` | Minor drift | Edge-LLM 综述支持二次注意力、内存带宽与 KV-cache 管理，但不是“扩展上下文及缓存访问是主要延迟瓶颈”的最直接证据。建议增加或改用 `kwon2023pagedattention`。 |
| MEDIUM-4 | `main.tex:99` | `ref10` | Minor drift | 来源直接支持 error propagation、prosody 等非词汇线索丢失和较高 inference latency；“intermediate-representation loss / accumulated inter-module latency”范围更宽。建议按来源原意收窄。 |
| MEDIUM-5 | `main.tex:132` | `liu2020partialhypothesis` | Minor drift | Liu et al. 支持一般 partial-hypothesis selection，不直接证明 Whisper 的 chunk-boundary omission/repetition/revision。该处建议只留 `machacek2023turning` 或增加 Simul-Whisper；Liu 继续用于 LocalAgreement 起源与 `main.tex:506`。 |
| MEDIUM-6 | `main.tex:389` | `ref20` | Minor drift | Qwen2 技术报告支持模型家族，但精确仓库标识 `Qwen/Qwen2-7B-Instruct` 最好另引固定 revision 模型卡：<https://huggingface.co/Qwen/Qwen2-7B-Instruct>。 |
| MEDIUM-7 | `main.tex:532` | `chen2024m3embedding` | Minor drift | ACL 论文支持 M3-Embedding 方法，但精确仓库别名 `BAAI/bge-m3` 最好另引固定 revision 模型卡：<https://huggingface.co/BAAI/bge-m3>。 |
| MEDIUM-8 | `main.tex:532` | `commandcode2026deepseekv4flash` | Reproducibility boundary | 动态页面能验证服务名与发布日期，但 `latest` 没有不可变 build ID。保留当前限定，同时归档网页快照、model-list/API 响应和请求 ID（如有）。 |
| MEDIUM-9 | `main.tex:355` | `ref17` | Integrity notice | CosyVoice 来源与模型家族主张匹配，但 arXiv 页面提示与 `arXiv:2407.04051` 有 substantial text overlap。它不是撤稿或正式 EoC；如有正式版本应优先引用，否则可联合引用 FunAudioLLM。 |

### 建议用于 MEDIUM-1 至 MEDIUM-5 的证据

- RNN-T survey：<https://doi.org/10.1109/TASLP.2023.3328283>；<https://ar5iv.labs.arxiv.org/html/2303.03329>
- Whisper：<https://proceedings.mlr.press/v202/radford23a.html>；<https://arxiv.org/abs/2212.04356>
- Edge LLM review：<https://doi.org/10.1145/3719664>；<https://arxiv.org/abs/2410.11845>
- Speech-to-text translation survey：<https://doi.org/10.1016/j.csl.2024.101751>；<https://arxiv.org/abs/2312.01053>
- Partial hypothesis selection：<https://doi.org/10.21437/Interspeech.2020-2897>

## 5. 建议补引但当前未引的一般性主张

这些段落不是论文自身实验事实，而是可由外部文献验证的一般理论、指标或方法说明。它们不会造成 citation-key closure 失败，但严格审稿时可能被问到依据。

| 位置 | 当前主张 | 建议 |
|---|---|---|
| `main.tex:146` | 缩放因子抑制点积方差、softmax 饱和，单层可聚合长程信息 | 直接补 `\cite{ref4}`。 |
| `main.tex:167` | KV cache 将逐步解码从每步 `O(t^2)` 降为 `O(t)`、累计 `O(N^2)` | 引用直接讨论 causal decoding/KV-cache 复杂度的来源；`kwon2023pagedattention` 可部分支撑。 |
| `main.tex:191-197` | WER/Levenshtein 定义、cluster bootstrap、Wilcoxon 与 Holm correction | 增加标准 WER 来源及统计方法原始/权威来源，尤其 Holm correction。 |
| `main.tex:321` | last-step logits 用于首 token 是 standard causal-LM prompt processing | 引用实际推理框架或 Qwen2/Hugging Face generation/cache 文档。 |
| `main.tex:347` | conventional datasets 缺少 long-speech samples | 给出数据长度分布或长语音 benchmark 文献；否则改成仅描述本研究构造目的。 |
| `main.tex:585` | 文本级模块会丢弃非文本声学线索 | 此处可再次引用 `ref10`，其正文直接讨论 prosody 等非词汇线索。 |

## 6. 确定性格式修正建议

### `MIT` 大小写保护

- **源位置**：`refs.bib:79`
- **当前**：

```bibtex
note = {MIT-licensed software; accessed August 22, 2026}
```

- **IEEEtran 输出**：`main.bbl:156` 显示 `mIT-licensed software`
- **建议**：

```bibtex
note = {{MIT}-licensed software; accessed August 22, 2026}
```

这是本次唯一确认的可见参考文献格式错误，不影响 key 解析或编译。

## 7. 来源存在性与主张匹配核验台账

下表覆盖全部 31 个唯一引用键。`PASS` 表示相邻主要主张由来源直接或合理支持；`CONDITIONAL` 表示来源真实，但存在第 4 节所述边界；`REVISE` 表示应修正引用或表述。

| Key | 正文位置 | 状态 | 核验入口 |
|---|---:|---|---|
| `ref1` | 57 | PASS | <https://arxiv.org/abs/2410.21276> |
| `ref3` | 89 | CONDITIONAL | <https://doi.org/10.1109/TASLP.2023.3328283> |
| `ref4` | 89, 150 | PASS | <https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html> |
| `ref6` | 91, 126 | CONDITIONAL | <https://proceedings.mlr.press/v202/radford23a.html> |
| `ref7` | 95 | CONDITIONAL | <https://doi.org/10.1145/3719664> |
| `ref10` | 99 | CONDITIONAL | <https://doi.org/10.1016/j.csl.2024.101751> |
| `ref11` | 99 | **REVISE** | <https://doi.org/10.1093/nsr/nwae403> |
| `ref13` | 205, 232 | PASS | <https://github.com/snakers4/silero-vad> |
| `ref14` | 327 | PASS | <https://aclanthology.org/D18-1547/> |
| `ref15` | 327 | PASS | <https://doi.org/10.1162/tacl_a_00314> |
| `ref17` | 355 | CONDITIONAL | <https://arxiv.org/abs/2407.05407> |
| `ref20` | 389 | CONDITIONAL | <https://arxiv.org/abs/2407.10671> |
| `ref22` | 585 | PASS | <https://doi.org/10.1016/j.csl.2020.101178> |
| `gulati2020conformer` | 91 | PASS | <https://www.isca-archive.org/interspeech_2020/gulati20_interspeech.html> |
| `kwon2023pagedattention` | 95, 157 | PASS | <https://doi.org/10.1145/3600006.3613165> |
| `dao2022flashattention` | 95 | PASS | <https://arxiv.org/abs/2205.14135> |
| `xiao2024streamingllm` | 95 | PASS | <https://arxiv.org/abs/2309.17453> |
| `wang2024simulwhisper` | 91 | PASS | <https://www.isca-archive.org/interspeech_2024/wang24ea_interspeech.html> |
| `gim2024promptcache` | 95 | PASS | <https://proceedings.mlsys.org/paper_files/paper/2024/hash/a66caa1703fe34705a4368c3014c1966-Abstract-Conference.html> |
| `zheng2024sglang` | 95 | PASS | <https://arxiv.org/abs/2312.07104> |
| `xie2024miniomni` | 99 | PASS | <https://arxiv.org/abs/2408.16725> |
| `defossez2024moshi` | 99 | PASS | <https://arxiv.org/abs/2410.00037> |
| `fang2025llamaomni` | 99 | PASS | <https://arxiv.org/abs/2409.06666> |
| `openai2024whisperlargev3turbo` | 128 | PASS | <https://huggingface.co/openai/whisper-large-v3-turbo/blob/41f01f3fe87f28c78e2fbf8b568835947dd65ed9/config.json> |
| `panayotov2015librispeech` | 383 | PASS | <https://doi.org/10.1109/ICASSP.2015.7178964> |
| `bu2017aishell1` | 383 | PASS | <https://doi.org/10.1109/ICSDA.2017.8384449> |
| `snyder2015musan` | 383 | PASS | <https://arxiv.org/abs/1510.08484> |
| `machacek2023turning` | 91, 115, 132, 506 | PASS | <https://aclanthology.org/2023.ijcnlp-demo.3/> |
| `liu2020partialhypothesis` | 91, 115, 132, 506 | CONDITIONAL | <https://www.isca-archive.org/interspeech_2020/liu20s_interspeech.html> |
| `chen2024m3embedding` | 532 | CONDITIONAL | <https://aclanthology.org/2024.findings-acl.137/> |
| `commandcode2026deepseekv4flash` | 532 | CONDITIONAL | <https://commandcode.ai/models/deepseek-v4-flash> |

## 8. 撤稿、完整性、自引与时效性

- **正式撤稿**：未发现。
- **Expression of Concern**：未发现。
- **Crossref correction/update relation**：所查 DOI 未显示撤稿或更正关系。
- **OpenAlex**：可查询条目均为 `is_retracted:false`。
- **arXiv withdrawal**：未发现被标记为 withdrawn 的引用条目。
- **其他完整性提示**：`ref17` 的 arXiv 页面存在与 `arXiv:2407.04051` 的文本重叠提示；不是撤稿或正式 EoC。
- **自引**：未发现作者 Haihua Mo、Zhengyou Liang 或倒排姓名，比例 0%。
- **时效性**：22/31（71.0%）来源发表于 2022–2026。两篇 2015 年来源分别是 LibriSpeech 与 MUSAN 数据集原始论文，属于合理的基础来源，不建议仅因年份替换。

## 9. 编译与引用格式验证

在临时目录复制输入后执行：

```text
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex
```

结果：

- BibTeX：成功，0 warnings；
- 最终 undefined citations：0；
- 最终 undefined references：0；
- multiply-defined labels：0；
- 重建 `main.bbl` 与仓库中 `main.bbl` 逐字节一致；
- overfull box：0；
- underfull `\hbox`：22；underfull `\vbox`：4；
- 另有 1 条 Fig.4 PDF page-group inclusion warning。

Underfull box 与 PDF page-group 提示不属于引用错误，也不阻断编译。目标目录当前没有保留 `main.log`、`main.blg`、`main.aux`；因此本报告记录的是本次独立临时重建结果，而不是对历史日志的复用。

## 10. Corrected Draft / Corrected Reference List 状态

本轮为检查报告模式，没有静默修改学术论断或增删参考文献：

- **Corrected Draft**：未写回；第 3–5 节给出逐项修订位置与文本方向。
- **Corrected Reference List**：未写回；唯一确定性格式修正见第 6 节。
- **建议重新验收条件**：处理 `ref11`、保护 `{MIT}` 大小写，并对第 4–5 节项目作取舍后，再运行一次 `pdflatex → bibtex → pdflatex → pdflatex` 与双向 key 检查。

## 11. 最终门禁状态

| Gate | Status |
|---|---|
| Citation-key closure | PASS |
| Bibliographic existence | PASS |
| DOI/URL integrity | PASS |
| IEEEtran compilation | PASS |
| Retraction/EoC screen | PASS（带 1 项非正式 arXiv 重叠提示） |
| Claim-to-source alignment | **NEEDS REVISION**（1 项 wrong-source attribution） |
| Reference-list visible formatting | **NEEDS MINOR FIX**（`MIT` → `mIT`） |
| 投稿前引用终检 | **CONDITIONAL PASS** |

**总判定：有条件通过。** 机械引用与文献真实性没有问题；完成 `ref11` 的归因修正和 `MIT` 大小写修复后，可将最关键的引用门禁提升为通过。其余 MEDIUM 项建议在最终投稿前尽量处理，以避免审稿人质疑来源直接性与实验工件可复现性。
