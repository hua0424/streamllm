# 致写稿人：论文数据就绪度整改完成报告，请求复核（2026-08-22）

- **回应对象**：`experiments/review/20260821-PRE-PAPER-AUDIT/review-paper-data-readiness-f54f2bd-20260822.md`
- **整改区间**：`f54f2bd…6069868`（两个提交：`aca4d92` 整改+装配，`6069868` W7 关闭；本报告随第三个提交入库）
- **请求事项**：复核 §2.1–2.6 六项整改与 §3 状态表，确认是否可宣布**最终数据锁定**并开始 main.tex 正式修改。
- **不变边界**：无新增 GPU 实验、无重跑 r7_main / r7_tts_control；所有数字只做装配与文档同步，未做任何缩放或事后修正。

> **更正注（2026-08-22，最终锁定复核 review-final-data-lock-7c93b77 §非阻塞残留 后追加；
> 原文未改，仅登记更正）**：
> ① §三复核命令中的脚本路径 `experiments/results/revision/r7_ttfa_unified/table_viii/assemble_table_viii.py`
> 已迁移，正确命令为 `uv run python experiments/scripts/assemble_table_viii.py`；
> ② §2.1 中"爆音→LibriSpeech 源音频属性"的归因表述已按复审软化（来源未区分、不作归因断言），
> 现行表述见 `r2_real_speech/MANUAL_SPOT_CHECK.md` 状态注；
> ③ 数据锁定已由 `7c93b77` 复核确认，本函"请求事项"已闭环。

---

## 一、逐项整改对照（就绪度审查 §2.1–2.6）

### 2.1 W7 人工 spot check → 已完成

- **记录**：`results/revision/r2_real_speech/MANUAL_SPOT_CHECK_FORM.md`（试听者：华血；日期 2026-08-22；
  5/5 通过）。七类字段逐条填写（可懂度/截断/错序/爆音削波/异常静音/音量/拼接缝/结论）。
- **形式核验（本机，写入 MANUAL_SPOT_CHECK.md 状态注）**：5 个 sample_id 与
  `librispeech_build_manifest.json` / `aishell1_build_manifest.json` 全部命中；时长逐条吻合
  （15.126→15.1、60.841→60.8、15.0、30.179→30.2、60.113→60.1）；duration_group 一致。
- **三项非默认记录解读**（已在状态注）：① 1 条爆音/削波=有 → LibriSpeech 源音频属性，
  该条可懂度"正常"、该样本已在 WER 2.98% 统计内，不影响延迟测量；② 拼接缝可感知
  （1 轻微/1 明显）→ 与拼接构造一致，证实"拼接的真人朗读语音"披露必要，证据已补进
  写作参考 §八 limitations 第 5 条；③ 1 条"可懂但瑕疵"仍通过。
- **口径**：全程维持 manual spot check（人工抽检），不称 human evaluation。
- **留给写稿人裁量的两点（未阻塞）**：记录表备注区为空（爆音"有"可补一句位置/现象，
  非字段清单要求）；表头一句"试听记录初始值"的说明可改定稿语。试听人原始记录本机未改动。

### 2.2 PAPER_HANDOFF.md 旧口径 → 已清理

| 位置 | 处理 |
|---|---|
| E1 CV 4.2%/3.3%/"CV<5%" | 标注 **ddof=0 历史作废、禁止引用**；换 ddof=1 终版（B 5.19%/4.05%/P90 10.73%/max 18.96%；A 5.23%/4.65%/9.92%/14.01%，指向 `r1_stats/repeat_cv_summary.csv`） |
| E2 aishell1 sanity CER 6.72% | 标注旧口径作废；换修正口径 **10.73%** + 中文数字/阿拉伯数字失配脚注指引 |
| TTFA 补测段 B 14.79s / A 22.67s | 整段改为 **"历史作废、不得引用"**，仅留审计追溯（原两轮 P0 裁决口径与 ttfa_budget.csv 路径）；R7 装配稿指针已加入 R7 小节 |

### 2.3 PAPER_WRITING_REFERENCE.md 冲突 → 已消除

- 铁律 1：Table VIII 来源由 E5/E6/补测改为 **R7 统一实测**；
- 铁律 7：重写为 R7 优先（主指标/范围/统计量/装配稿路径/控制脚注），旧装配口径标注历史作废；
- §七证据表意见 3 行：CV 换 **ddof=1 完整分布口径**（总册内不再有 4.2% 与 5.19% 并存）；
- §二：R7 主数字在前、历史装配稿整体标"作废仅供对照"；§十：W7/W8 状态更新为完成，剩余=数据锁定+main.tex。

### 2.4 Table VIII 装配 → 已完成（W8 阶段 2）

产物：`results/revision/r7_ttfa_unified/table_viii/`（`assemble_table_viii.py` 可复现 +
`table_viii_r7.csv` + `TABLE_VIII_ASSEMBLED.md` 论文表）。审查 §2.4 七项决策全部固定：

1. 主指标 `first_playable_pcm`；2. repeat0、n=50/模式（zh/en 各 25，补轮仅 CV）；
3. mean/std(ddof=1)/P50/P90/P95（np.percentile 线性插值）；4. 单位 ms、1 位小数；
5. `ttfa_received` 仅 QA 补充；6. tts_control 7076ms 只作归因/回信证据、必带偏差豁免脚注；
7. 旧 `ttfa_budget.csv` 估计项完全排除。

**装配 QA 四项全过**（结果在装配稿末尾，可复跑脚本复现）：
- QA-1 六分项逐记录闭合：100 条最大残差 **0.00e+00 ms**（首尾相接恒等式）；
- QA-2 received→playable 缓冲差：mean 0.1 / max 0.2 ms（确认 received 不进主表）；
- QA-3 与运行侧 `ttfa_summary_r7_main.csv` **双入口对拍 48 行一致**；
- QA-4 输入 checkpoint sha256(LF)=`4edcd6ec28189d00…`（不可变归档、control_from 同源）。

**最终数字（论文表 (a)/(b)）**：B streaming ALL mean **5481.9** / P50 **3113.7** ms
（zh 3303.3/2603.0；en 7660.5/7577.0）；A ALL mean **22425.7** / P50 **22269.9** ms。
B vs A：ALL mean −75.6%（4.09×）/ P50 −86.0%（7.15×）；zh −85.4%/−88.3%；en −65.5%/−66.2%
（mean 与 P50 两种表述**二选一勿混用**）。组件表六分项闭合校验与表 (a) 完全一致。

### 2.5 CISR_REVISION_PLAN.md → 已同步

§七加"R6 装配作废、R7 替代（E5/E6 单项仅背景）"状态注；§8.1 Table VIII 映射 R6→**R7 统一 TTFA**；
§8.3 两处 R6 引用改 R7（Abstract 口径句、§V 小节数据源）。

### 2.6 EXPERIMENT_DESIGN.md §5.3 → 已历史化

标题改"执行与分析（历史章节，已由归档结果覆盖）"，三项勾选完成并注明归档位置，
消除"基础实验未完成"误读；§6.6 为 R7 方法学唯一权威定义（前次已入）。

---

## 二、数据锁定判定请求

就绪度审查 §3 状态表逐行更新：

| 层级 | 原状态 | 现状态 |
|---|---|---|
| R1–R5 核心数值 | 通过 | 通过（未动） |
| R2 人工 spot check | 未完成 | **完成（5/5 通过 + manifest 核验）** |
| R6 单项背景测量 | 通过（不装配） | 通过（不装配，状态注已入路线图） |
| R7 正式 TTFA | 通过 | 通过（未动） |
| r7_tts_control | 通过（偏差豁免） | 通过（偏差豁免，引用必带披露） |
| 最终 Table VIII | 未装配 | **已装配（QA 4/4，七项决策固定）** |
| 文档口径一致性 | 未完全通过 | **已清理（§2.2/2.3/2.5/2.6 全关）** |
| 论文改稿可开始 | 条件通过 | **请复核确认：最终数据锁定** |

若复核通过，建议的锁定声明基线：**锁定提交 = 6069868（W7 关闭）**，此后 main.tex
修改只引用本报告与写作参考所列数字，任何新数据需求须另行书面提出。

## 三、给写稿人的复核操作清单（可逐条执行验证）

```bash
git log --oneline f54f2bd..HEAD          # 应见 aca4d92 / 6069868（及本报告提交）
# 1) 装配可复现（QA 四项应全过）
uv run python experiments/results/revision/r7_ttfa_unified/table_viii/assemble_table_viii.py
# 2) 旧口径只剩"作废标注"语境
grep -n "CV<5%\|6.72%\|14.79s\|22.67s" experiments/results/revision/PAPER_HANDOFF.md   # 6 处命中均应带 ~~划线~~/作废标注
grep -n "4\.2%" experiments/results/revision/PAPER_WRITING_REFERENCE.md
#   预期仅 2 处：L133 CV 4.2%（划线作废语境）；L176 为 "max 14.2%" 子串误命中（R4 外部一致性数字，非 CV）
# 3) W7 记录与核验注
cat experiments/results/revision/r2_real_speech/MANUAL_SPOT_CHECK_FORM.md
grep -n "W7 试听已完成" experiments/results/revision/r2_real_speech/MANUAL_SPOT_CHECK.md
```

## 四、写作期须随稿携带的约束（已在总册，此处汇总提醒）

1. Table VIII 只用 `table_viii/TABLE_VIII_ASSEMBLED.md`；mean/P50 表述二选一；received 不进主表；
2. tts_control 7076ms 引用必带偏差豁免脚注（原文在装配稿 §(d) 与 deviation-waiver §3）；
   不得写"已获单独放行后执行"或"全部实验严格按预授权顺序执行"；
3. CV 只用 ddof=1 全分布口径；aishell1 CER 用修正口径 + 数字写法失配脚注；
4. 平台绑定与禁缩放红线不变；R7 绝对值与原平台旧表不得混排同栏；
5. 拼接语音称"拼接的真人朗读语音"；babble 边界与 limitations 措辞照 §八；
6. LA 方法写修复后语义；"统计不可区分"等已撤销表述不得复现（总册 §十 P0 清单）。

## 五、未决事项（请写稿人裁定）

1. W7 记录表两处可选增强（§2.1 末）是否要求补齐，或接受现状；
2. 最终数据锁定的声明基线（建议 6069868）与锁定后的改动控制流程（建议：锁定后
   `results/revision/` 数字文件冻结，main.tex 修改若需新口径一律走书面变更）。
