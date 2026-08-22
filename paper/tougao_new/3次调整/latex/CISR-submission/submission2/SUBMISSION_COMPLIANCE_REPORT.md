# CISR 2026 LaTeX 投稿格式与引用合规检查

- **检查日期**：2026-08-22
- **投稿要求网页**：https://www.icisr.com/submission/
- **本地要求文件**：`templates/投稿须知 IEEE.docx`
- **官方 LaTeX 模板**：`templates/ieee-template-latex.zip`
- **目标稿件**：`submission2/main.tex`、`submission2/main.pdf`、`submission2/refs.bib`
- **检查边界**：只读检查；未修改论文源文件或 PDF

## 一、总体结论

**当前稿件主体版式与 CISR/IEEE LaTeX 模板高度一致，但还不能判定为“格式完全就绪”。**

- 页面、双栏、模板类、字体嵌入、正文长度、英文、图像分辨率、可编辑表格和公式均符合或基本符合。
- 参考文献数量、近三年文献和国际来源多样性符合要求；网页没有要求 DOI 必须显示，因此当前 IEEEtran 样式不输出 DOI 不构成违规。
- 投稿前有 **3 项必须优先修复**：
  1. Fig.6 使用 Type 3 字体，存在 IEEE PDF eXpress 拒绝风险；
  2. Table IV 和 Table VII 未在正文中显式交叉引用，不符合“所有表格必须在正文引用”；
  3. 引用语义审计仍有多项来源错配/缺少主文献，影响投稿要求中的“adequate, proper and scholarly citations”。
- 另有公式引用语法、BibTeX 元数据和 PDF metadata 等次要问题，建议一并清理。

## 二、CISR 对 LaTeX 稿件的要求

### 2.1 投稿网页明示要求

网页 `https://www.icisr.com/submission/` 当前明示：

1. 稿件必须使用英文；
2. 正文至少 4 个完整页面，参考文献页不计；
3. 必须具有创新性和科学价值；
4. 必须遵循会议模板；
5. 主题必须属于 CISR 会议范围；
6. 禁止抄袭和重复投稿；
7. 至少 8 条参考文献，来源应多样、权威，并包含国际来源；
8. 图片必须高分辨率、图中文字必须为英文；
9. 公式和表格必须为可编辑格式，不能作为图片；
10. 涉及敏感图片时须有合法授权。

### 2.2 本地“投稿须知 IEEE.docx”的更严格要求

下载目录中的投稿须知进一步规定：

1. 全文双栏，正文不少于 4 页，不含参考文献；
2. 所有作者需按顺序标注序号；
3. 全英文，需完成语言校对和润色；
4. 图片在 100% 阅读模式下文字/数据清晰，且不能含中文；
5. 图片与标题必须在同一页；所有图片须在正文引用并按阿拉伯数字排序；
6. 表格与标题必须在同一页；所有表格须为可编辑格式，并在正文引用、按阿拉伯数字排序；
7. 公式必须可编辑、居中，编号连续且右对齐；
8. 所有文献须在正文正确引用；
9. 参考文献不少于 10 条，必须包含近三年文献，作者来源至少覆盖 3 个国家；
10. 全文查重率不高于 24%，单项不高于 3%；
11. 最终稿必须通过 IEEE PDF eXpress；
12. 录用后须完成版权转让。

网页要求“至少 8 条”，本地投稿须知要求“至少 10 条”。本检查采用更严格的 10 条标准。

## 三、官方模板比对

### 3.1 IEEEtran 类文件

官方 ZIP 内：

- `IEEEtran.cls`：288,304 bytes，SHA-256 `c972aca108fda004...e003f55`
- `IEEEtran.bst`：61,632 bytes，SHA-256 `d83aa3c9b47fc120...5ded24`

`submission2` 中对应文件与官方 ZIP **逐字节完全相同**：

- `submission2/IEEEtran.cls` = 官方 `IEEEtran.cls`
- `submission2/IEEEtran.bst` = 官方 `IEEEtran.bst`

模板版本为：

```text
IEEEtran.cls 2015/08/26 V1.8b
```

### 3.2 文档类与页面

当前稿件：

```latex
\documentclass[conference]{IEEEtran}
```

PDF 属性：

- 13 页；
- Letter：612 × 792 pt；
- IEEE conference 双栏；
- 页面内容边界约为 x=49–563 pt，左右对称；
- 无空白页；
- 无可见内容裁切。

官方模板 PDF 同样为 Letter 612 × 792 pt。因此页面尺寸和版式符合模板。

### 3.3 模板结构

当前稿件包含：

- 英文标题；
- 有序作者编号；
- 作者单位和邮箱；
- 通讯作者脚注；
- Abstract；
- IEEEkeywords；
- 正文章节结构；
- Acknowledgment；
- IEEEtran 参考文献样式。

这些结构符合网页、本地投稿须知和 LaTeX 模板要求。

## 四、逐项格式合规结果

| 要求 | 当前状态 | 判定 | 说明 |
|---|---|---|---|
| 全英文 | 正文、图表和参考文献均为英文 | PASS | PDF 文本层未检测到中文字符；图中视觉检查亦未发现中文 |
| 双栏 | `IEEEtran[conference]` | PASS | 与模板一致 |
| 正文不少于 4 页 | 参考文献从第 13 页开始，正文约 12 页 | PASS | 远高于最低要求；网页未给出最大页数 |
| 使用官方模板 | `.cls/.bst` 与 ZIP 字节相同 | PASS | 无模板版本偏差 |
| 标题/作者/单位/摘要/关键词/章节/致谢/参考文献 | 均存在 | PASS | 结构完整 |
| 作者按序编号 | 1st、2nd | PASS | 符合本地须知 |
| 图片高清 | Fig.1–5 内部位图约 384 ppi，最终排版等效多为 572–802 dpi；Fig.6 为矢量图 | PASS | 分辨率高于常见 300 dpi 标准 |
| 图中文字英文 | 视觉检查无中文 | PASS | Fig.6 清晰、全英文 |
| 图和图题同页 | 均位于同一 figure/figure* float | PASS | LaTeX 保证同一浮动体 |
| 所有图片在正文引用 | 6/6 均被 `\ref{fig:...}` 引用 | PASS | Fig.1–6 全部被引用 |
| 图按顺序编号 | Fig.1–6 连续 | PASS | 无跳号 |
| 表格可编辑 | 8 张表均为 LaTeX tabular | PASS | 不是图片 |
| 表和表题同页 | 均位于 table* float | PASS | 无拆分 |
| 表按顺序编号 | Table I–VIII 连续 | PASS | 无跳号 |
| 所有表格正文引用 | Table IV、Table VII 未使用 `\ref{tab:...}` | **FAIL** | 必须在相关分析段补 “Table~\ref{tab:ablation}” 和 “Table~\ref{tab:la}” |
| 公式可编辑 | LaTeX equation/align | PASS | 不属于图片公式 |
| 公式编号连续且右对齐 | LaTeX 自动编号 | PASS | 使用 10 个 equation + 2 个 align |
| 公式正文交叉引用 | 多个带 label 公式未被正文引用 | WARN | 投稿须知未绝对要求每个公式都被引用，但建议引用或去掉无用 label/编号 |
| 公式引用语法 | 使用 `Eq.~(\ref{...})` 和 `(\ref{...})` | WARN | 官方模板明确建议使用 `\eqref{...}`；应统一 |
| 无 overfull/citation/reference 错误 | 最后完整编译通过 | PASS | 未发现 undefined citation/reference 或 overfull box |
| PDF 字体嵌入 | 所有字体均 embedded/subset | PASS | 无缺失字体 |
| PDF 仅使用合规矢量字体 | Fig.6 引入 Type 3 DejaVuSans | **FAIL / PDF eXpress 风险** | 官方 IEEE HOWTO 要求避免 bitmapped/Type 3 字体 |
| PDF metadata | 缺 Title 和 Author metadata | WARN | 网页未明示，但 PDF QA 建议补齐 |
| 最终 PDF eXpress | 尚未执行 | PENDING | 投稿前必须完成 |
| 查重率 | 未检查 | PENDING | 需使用 iThenticate/CrossCheck 等检查 |
| 版权转让 | 尚未进入流程 | PENDING | 录用后完成 |

## 五、Type 3 字体问题

### 5.1 检测结果

`pdffonts main.pdf` 检测到：

```text
BMQQDV+DejaVuSans  Type 3  embedded=yes  subset=yes
```

该字体来自 `Fig6.pdf`。其余稿件字体均为嵌入和子集化的 Type 1 字体。

官方 IEEEtran HOWTO 明确说明：

> Authors should check their system ... to ensure that only vector (Type 1) fonts are being used and that all fonts are embedded and subsetted. A document that uses bitmapped fonts ... may be rejected by the IEEE.

官方模板示例 PDF 也全部使用 Type 1 字体，没有 Type 3。

### 5.2 建议修复

重新生成 Fig.6 时应禁止 Matplotlib 默认 Type 3 字体。例如可在绘图脚本中设置：

```python
import matplotlib as mpl
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
```

更保守的 IEEE 方案是使用 LaTeX/PGF 或可嵌入 Type 1 的字体生成图中文字。重新生成后执行：

```bash
pdffonts Fig6.pdf
pdffonts main.pdf
```

目标是输出中不再出现 `Type 3`。之后再提交 IEEE PDF eXpress。

## 六、图像和图表检查

### 6.1 分辨率

- Fig.1–5 是 PDF 容器中的高分辨率位图/透明遮罩组合，源图约 384 ppi；在论文中缩小后，最终等效分辨率大多超过 500 dpi。
- 没有检测到最终稿中低于 300 dpi 的常规栅格图像。
- Fig.6 的曲线和坐标轴是矢量内容，100% 阅读清晰；唯一问题是 Type 3 字体。

### 6.2 可读性

视觉检查结果：

- Fig.1–6 均与图题在同一页；
- Fig.6 图例、坐标轴和阴影带清晰；
- Fig.1–5 内容较密，但在论文当前缩放下基本可辨；投稿前仍建议按本地须知用 100% 缩放逐图人工复核一次；
- 未发现中文字符或在线图库水印。

## 七、表格和公式检查

### 7.1 表格

- 8 张表均为 LaTeX 源码，不是截图；
- 表题位于表格上方，符合 IEEE 模板；
- 均采用 `table*` 跨双栏，未越界或裁切；
- 编号连续 Table I–VIII。

**必须修改**：

- Table IV `\label{tab:ablation}` 未在正文通过 `\ref` 引用；
- Table VII `\label{tab:la}` 未在正文通过 `\ref` 引用。

建议在对应结果段开头增加：

```latex
Table~\ref{tab:ablation} shows that ...
```

```latex
As shown in Table~\ref{tab:la}, ...
```

### 7.2 公式

公式均为 LaTeX 可编辑格式、居中、右侧自动编号。编号由 LaTeX 管理，不存在跳号风险。

但官方模板在 LaTeX-specific advice 中明确建议：

```latex
\eqref{eq:total}
```

而不是：

```latex
Eq.~(\ref{eq:total})
```

当前稿件至少有以下 4 处旧写法：

- `main.tex:59`
- `main.tex:130`
- `main.tex:172`
- `main.tex:235`

建议统一改为 `\eqref{...}` 或模板规定的句首 `Equation~\eqref{...}`。

## 八、参考文献是否符合 CISR 要求

### 8.1 数量

- 网站最低要求：8 条；
- 本地投稿须知最低要求：10 条；
- 当前：**25 条**。

判定：**PASS**。

### 8.2 近三年文献

当前年份分布：

| 年份 | 条数 |
|---|---:|
| 2015 | 2 |
| 2017 | 2 |
| 2018 | 1 |
| 2020 | 3 |
| 2021 | 1 |
| 2022 | 3 |
| 2023 | 2 |
| 2024 | 8 |
| 2025 | 2 |
| 2026 | 1 |

- 2023–2026：13/25；
- 2024–2026：11/25。

判定：**PASS**，近三年文献比例充分。

### 8.3 国家和来源多样性

参考文献作者/机构来源明显覆盖至少：

- 美国；
- 中国；
- 捷克；
- 荷兰；
- 德国；
- 日本；
- 法国等。

来源包括 IEEE、ACM、ACL、PMLR、Oxford Academic、Elsevier、TACL、ISCA、arXiv/DataCite 和官方软件/服务文档。

判定：**PASS**，超过“至少三个国家”要求。

### 8.4 权威性与多样性

多数来源为同行评审期刊/会议或正式数据集论文。Silero VAD 和 Command Code judge 是必要的软件/服务文档，不属于同行评审来源，但没有主导文献列表，且在正文中用于标识实际软件/服务，使用场景合理。

判定：数量/多样性层面 **PASS**。

### 8.5 DOI 是否必须显示

CISR 网页和本地投稿须知均未要求参考文献必须在 PDF 中显示 DOI。当前官方 `IEEEtran.bst` 与下载 ZIP 逐字节相同，并会忽略 `doi` 字段。

因此：

- `refs.bib` 中保存 DOI 但 PDF 不显示，不构成 CISR 格式违规；
- 不应为了显示 DOI 擅自更换非官方 `.bst`；
- 如 IEEE PDF eXpress 或编辑部后续提出 DOI 显示要求，再按其指示处理。

### 8.6 引用内容质量仍需修复

尽管数量、时效性和多样性符合，先前引用审计发现的论断—来源错配仍会影响投稿网页中的：

> adequate, proper and scholarly citations to the work of others

投稿前应优先修复：

1. Whisper 鲁棒性错误引用 `ref12`，应换为 Whisper 主文献 `ref6`；
2. rVAD 引用与本文 Silero 实现混淆；
3. Conformer 应补原始论文；
4. vLLM/KV cache/TGI 当前引用不直接；
5. FlashAttention 应补主文献；
6. StreamingLLM 缺主文献；
7. Mini-Omni、Moshi、LLaMA-Omni 缺各自主文献；
8. CosyVoice 模型论文不能单独证明具体 hosted service；
9. “most production systems” 等广泛行业断言应软化或补证据。

详细条目见 `CITATION_AUDIT_REPORT.md`。

### 8.7 BibTeX 元数据问题

投稿前建议修正以下确定错误：

- `ref5` 补页码 1–64；
- `ref7`：209 是 article number，应使用页码 1–35，并补完整作者；
- `ref9`：109 是 article number，应使用页码 1–28；
- `ref14`：EMNLP 2018 地址应为 Brussels, Belgium，而不是 Stroudsburg；
- 保护标题中的 `Whisper`、`Mandarin`、`M3-Embedding` 大小写。

## 九、源文件包完整性

`submission2` 当前包含：

- `main.tex`
- `refs.bib`
- `main.bbl`
- `IEEEtran.cls`
- `IEEEtran.bst`
- `Fig1.pdf`–`Fig6.pdf`
- `main.pdf`

主文件引用的 6 张图均存在。模板说明要求使用 BibTeX 时提交 `.bib` 文件；当前已包含。

正式打包时建议只包含：

```text
main.tex
refs.bib
main.bbl
IEEEtran.cls
IEEEtran.bst
Fig1.pdf ... Fig6.pdf
main.pdf（若系统要求同时上传 PDF）
```

不要将以下内部文件放入正式源文件 ZIP：

```text
CITATION_AUDIT_REPORT.md
SUBMISSION_COMPLIANCE_REPORT.md
REVISION_TRACKING.md
response_to_reviewers.*（除非投稿系统有单独回复信上传入口）
*.aux *.log *.blg *.out
```

## 十、投稿前修复优先级

### P0：必须先处理

1. 重新生成 Fig.6，消除 Type 3 字体；
2. 在正文显式引用 Table IV 和 Table VII；
3. 修复引用审计中的主文献错配，至少处理 Whisper、Silero/rVAD、Conformer、KV-cache/vLLM、FlashAttention、StreamingLLM 和三个具名语音系统；
4. 修正确定的 BibTeX 元数据错误。

### P1：建议处理

5. 将公式引用统一为 `\eqref{...}`；
6. 检查是否需要为每个编号公式增加正文引用，或去除不需要编号/label 的公式；
7. 补 PDF Title/Author metadata；
8. 逐张图以 100% 缩放人工确认最小文字清晰；
9. 做最后一轮英语润色。

### P2：投稿流程

10. 完成相似度检测：全文 ≤24%，单项 ≤3%；
11. 用修复后的 PDF 运行 IEEE PDF eXpress；
12. 按投稿系统要求分别上传 PDF、LaTeX 源包和回复信；
13. 录用后完成版权转让和终校确认。

## 十一、最终判定

| 检查维度 | 判定 |
|---|---|
| CISR 主题相关性 | PASS |
| 英文与论文结构 | PASS |
| 官方 LaTeX 模板 | PASS |
| Letter/双栏/页边距 | PASS |
| 正文最低页数 | PASS |
| 图片分辨率与英文 | PASS |
| 图引用与编号 | PASS |
| 表格可编辑与编号 | PASS |
| 所有表格正文引用 | **FAIL（Table IV、VII）** |
| 公式可编辑与编号 | PASS |
| 公式引用样式 | WARN |
| 参考文献数量 | PASS（25 ≥ 10） |
| 近三年文献 | PASS（13 条为 2023–2026） |
| 至少三个国家来源 | PASS |
| 引用 key 与编号闭合 | PASS |
| 论断—来源匹配 | **NEEDS REVISION** |
| PDF 字体嵌入 | PASS |
| 无 Type 3/位图字体 | **FAIL（Fig.6）** |
| PDF eXpress | PENDING |
| 查重要求 | PENDING |
| 当前投稿格式就绪度 | **NOT YET READY** |

**综合判断**：论文已经正确使用官方 CISR/IEEE LaTeX 模板，主体格式总体合格；完成 P0 四项修复并通过 IEEE PDF eXpress 后，才能视为投稿格式就绪。
