# 投稿自检清单（对照 templates/投稿须知 IEEE.docx，2026-07-21）

| 要求 | 状态 |
|---|---|
| 全英文稿件 | ✅ 正文/图/表全英文 |
| 双栏正文 ≥4 页（不含参考文献） | ✅ 约 5.3 页（总 6 页，文献占末页约半页） |
| 必备要素：标题/作者/通讯作者/单位/摘要/关键词/章节/致谢/参考文献 | ⚠️ 结构齐全；**作者块为占位符，投稿前必须替换真实作者并标注通讯作者**（main.tex 有 TODO 注释） |
| 图片高清、无中文字符 | ✅ 5 图全部矢量 PDF（矢量无分辨率问题），全英文标签 |
| 图与图题同页、按 Figure 1..5 顺序、正文均引用 | ✅（Fig.1~5 均 \ref 引用） |
| 表格可编辑格式、与表题同页、正文引用 | ✅ Table I 为 LaTeX 原生表 |
| 公式可编辑、居中、编号右对齐 (1)(2)(3) | ✅ LaTeX 原生公式（MathType 要求仅适用 Word 投稿路线） |
| 参考文献 ≥10 条 | ✅ 16 条（bib 中另有 whisper 备用未引用） |
| 有近三年文献 | ✅ 2024×3、2025×4、2026×4 |
| 文献作者来源 ≥3 国 | ✅ 美/中/法/爱尔兰等 |
| 查重 ≤24%（单项 ≤3%） | ⏳ 作者自行用 CrossCheck/iThenticate 查（英文稿全新撰写，与中文学位论文不同语言，预期低） |
| IEEE PDF eXpress 格式验证 | ⏳ 投稿前作者在 PDF eXpress 生成终稿 |
| AI 使用说明 | ✅ Acknowledgment 中已含（按会议要求可调整措辞/位置） |

## 编译方式

```powershell
cd paper2\ieee_paper
pdflatex main.tex ; bibtex main ; pdflatex main.tex ; pdflatex main.tex
```

## 与中文学位论文的关系

- 内容为 `paper2/chapter1..8` 的压缩英译（约 5.3 页版），数字全部与 `experiments/results/*.json` 一致
- D-006 护栏保留：商用系统 prior art 明示（Sec.I/II/VI）、"first open-source"限定、B-ours 构造性零、A2 honest null、m2e 建模值声明
- 英文图源：matplotlib `plot_figures.py --en`；drawio `figures/src/fig3_1_en.drawio`、`fig4_1_en.drawio`
