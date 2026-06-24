# 图2 绘图说明（优化定稿版）

**对应论文图题**：图2 基于 KV Cache 的推理机制示意 / Fig. 2 Schematic of the KV Cache-based inference mechanism
**对应正文**：thesis.md §2.2.2（KV Cache 机制的数学原理与复杂度分析）

---

## 〇、与原始描述的最大修正（务必先读）

原始绘图描述把图2画成了「System A 端点后一次性 full prefill 时间轴 vs System B 发声期间增量 prefill」的**流式增量预填充时间轴图**。经与论文核对，这与 §2.2.2 的图2**内容错位**：

- **§2.2.2 是理论/背景节**，正文（line 121）明确说图2画的是：
  - 左侧 = **"重复前向"做法**：生成新 token 时对长度为 N 的序列重算 self-attention，单步开销 ≈ **O(N²)**；
  - 右侧 = **使用 KV Cache**：历史 K,V 缓存于显存并复用，当前步只算新 token 的 K,V 并与缓存交互，单步开销降为 **O(N)**。
- **流式增量预填充时间轴**（prefill 前移到 End-of-Speech 之前）是 **§3.3 / 图5 / 系统贡献**的内容，不应放在背景节的图2。
- 论文 §2.2.2 表格原文把两侧标为 "System A: 原生推理(No Cache)" 与 "System B: 增量推理(KV Cache)"，这与第3章**架构级** System A/B（非流式基线 / 流式）**同名不同义**。本图为消歧，**左右标签改用 `Without KV Cache` / `With KV Cache`**，不再使用 System A/B 字样。

**故本定稿：图2 = 解码单步「无缓存重算 vs 有缓存复用」的机制 + 复杂度对比图。**

---

## 一、设计目标

绘制一张简洁、科学严谨、适合顶刊论文（IEEE / Nature / ACM 方法示意图）风格的 draw.io 矢量对比图，主题为
**"基于 KV Cache 的推理（解码）机制示意"**。

- **左右两栏对比**结构，左右等大、布局对称；白色背景；低饱和学术配色；线条清晰、留白充足。
- 左：`Without KV Cache — Repeated Forward`（浅红主色）。
- 右：`With KV Cache — State Reuse`（浅蓝/浅绿主色，灰色表缓存）。
- 所有文字使用英文，公式只保留单步复杂度 Big-O 角标，不堆叠公式。

---

## 二、相对原始描述的优化点（已纳入本定稿）

1. **纠正图意**：从"流式 prefill 时间轴"改回"解码单步无缓存 vs 有缓存的机制 + 复杂度对比"，与 §2.2.2 caption、公式 (3)(4)(5) 及 line121 完全一致。
2. **去掉图内大标题**：论文已有正式图题，图内不再嵌大标题（与图1保持一致的 IEEE 惯例），仅保留两栏小标题。
3. **左右标签去歧义**：用 `Without / With KV Cache` 取代 `System A / System B`，避免与第3章架构级 System A/B 撞名。
4. **复杂度口径与正文一致**：强调 KV Cache 把**解码单步**复杂度由 O(N²) 降为 O(N)；底部注明**不消除** prompt 预填充的 O(M²) 开销（line 121 原话）。
5. **不画流式时间轴/VAD/ASR/TTS**：这些属于第3章，本图只聚焦 LLM 解码侧的 KV Cache 机制。

---

## 三、模块与数据流（左右对比）

### 左栏：Without KV Cache — Repeated Forward（浅红）
- 小标题：`Without KV Cache — Repeated Forward`
- 步进说明：`Generation step t : recompute all tokens`
- token 序列：`x₁ x₂ x₃ x₄ x₅`（全部浅红，表示每步都重算）。
- **因果 self-attention 矩阵**（下三角全部填色）：5×5 网格，下三角填浅红表"本步全部重算"，上三角空白表因果掩码。
- 注释：`All cells recomputed every step: self-attention over all N tokens`
- 底部小框：`No reuse: K,V for x₁…xₜ are recomputed every step`
- Big-O 角标（醒目）：`Per-step cost ≈ O(N²)`

### 右栏：With KV Cache — State Reuse（浅蓝/浅绿 + 灰）
- 小标题：`With KV Cache — State Reuse`
- 步进说明：`Generation step t : reuse cache, add one token`
- token 序列：`x₁…x₄` 灰（已缓存）、`xₜ` 绿（新增）。
- **因果 self-attention 矩阵**：5×5 网格，前 4 行灰色（历史，复用缓存）、仅最后一行 qₜ 绿色（本步新算），直观表示"只算新行、历史复用"。
- KV Cache 模块（浅蓝）：`Past Key/Value Cache (VRAM)` / `K₁,V₁ … Kₜ₋₁,Vₜ₋₁ (cached)`
- 新增计算块（绿）：`Compute Qₜ,Kₜ,Vₜ (new token only)`，箭头指向缓存：`append Kₜ,Vₜ`
- 注释：`Only the new row (qₜ) is computed; previous K,V reused from cache`
- Big-O 角标：`Per-step cost ≈ O(N)`

### 底部说明条（一句，忠于 line 121）
`KV Cache reuses cached Key/Value states to cut the per-step decoding cost from O(N²) to O(N); it does not remove the O(M²) cost of the initial prompt prefill.`

---

## 四、布局与样式要求

- 单栏宽即可（约 900×600，≈3:2），左右两个等大圆角矩形面板。
- 配色：浅红 `#f8cecc / #b85450`；浅蓝 `#dae8fc / #6c8ebf`；浅绿 `#d5e8d4 / #82b366`；缓存灰 `#eeeeee / #999999`；底部说明浅黄 `#fff2cc / #d6b656`。
- 矩阵、缓存块、token 序列对齐整齐，箭头清晰，留白充足。
- 风格接近 IEEE / ACM / Nature 方法机制图。

---

## 四点五、相对草稿图（2image.png）的评估与取舍

**草稿图准确、值得借鉴的部分：**
- 左 `O(N²) Re-computing History` vs 右 `O(N) Reuse Cached States`，与 line121 一致。
- 右栏"上方多行 Cached + 底部 New Compute"的结构正确。
- **坐标轴标注严谨**（纵轴 `Query step (generated tokens)`、横轴 `Key/Value step (context history)`、标题 `Attention mask`）——**已吸收进本图**，替换原先仅 `qᵢ×kⱼ` 的标注。

**草稿图的错误（本图不沿用）：**
- 用了 `Memory Bank`（违反术语规范）→ 本图用 `Past Key/Value Cache (VRAM)`。
- 用了 `System A / System B`（与第3章架构级撞名）→ 本图用 `Without / With KV Cache`。
- 草稿在每个格子里塞 `K₁ Q₁ V₁` 向量，语义混乱 → 本图格子以颜色编码为主（红=重算/灰=缓存复用/绿=新算）；仅在**右栏对角线**格标注该列对应的 `Kᵢ,Vᵢ`（K 上 V 下两行），末行末列为新 token 的 `Kₜ,Vₜ`（绿），既"实"又不混淆分数矩阵与 K/V。
- 缺 `O(M²) prefill 不被消除` 注脚 → 本图底部已补。

---

## 五、避免的错误（务必遵守）

- 不画流式 prefill 时间轴、prefill 前移、End-of-Speech 等（属 §3.3 图5/系统贡献，不是本图）。
- 不画 VAD、ASR 滑动窗口、TTS（不属于本图）。
- 不声称 KV Cache 把完整 prompt prefill 的总复杂度从 O(N²) 降到 O(N)——本图只讲**解码单步** O(N²)→O(N)，且底部注明**不消除** prefill 的 O(M²)。
- 右栏不要画"完整历史矩阵重算"——历史行必须是灰色复用、仅新行绿色新算。
- 不使用 "Memory Bank" 等不标准术语，统一用 `KV Cache` / `Past Key/Value Cache`。
- 左右标签不要用 System A / System B（与第3章撞名），用 `Without / With KV Cache`。
