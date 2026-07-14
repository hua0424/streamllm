# 图5 绘图说明（定稿版）

**对应论文图题**：图5 LLM KV Cache 增量更新机制示意图 / Fig. 5 Schematic of the LLM KV Cache incremental update mechanism
**对应正文**：thesis.md §3.3（line 191 策略、line 195 图说、line 205 复杂度）

---

## 〇、与正文的核对结论（先读）

§3.3 / line 195 明确图5 = **显存内 KV Cache 的状态变化**：左侧蓝色=历史上下文缓存的 K,V（past_key_values），右侧绿色=新增片段产生的 K,V；增量预填充**复用历史、仅对新增 token 前向并更新缓存**。本图与图2 区分清楚：
- **图2（§2.2.2 背景）**=解码单步"无缓存重算 O(N²) vs 有缓存复用 O(N)"+ 注意力矩阵。
- **图5（§3.3 贡献）**=流式**增量预填充**：KV Cache 随 ASR 文本片段陆续到达而增长，绝大部分 prefill 前移到发声期间，End-of-Speech 后只剩末段 → 复用末步 logits 立即出首 token（line 191/205）。

故图5 不画注意力矩阵、不画复杂度角标，聚焦"缓存随片段增量增长 + 仅新算绿块"。

---

## 〇点五、布局取舍（按草图 5image.png 改为 concat 矩阵视图）

第一版用"一排随时间增长的色块快照（t₁→t₃→EoS）"，抽象、要看图例才懂、且 900px 过宽过扁。草图 5image.png 把图5 画成**矩阵拼接**：大蓝方阵 `K_prev,V_prev (Cached, N×d)` ⊕ 细绿条 `New Projection (k,v, M×d)` → 顶部括号 `Cache_new (Updated)`——**直接画出真实操作、一眼懂、近正方紧凑、且贴合 §3.3.3 复杂度记号 N/M**。故采纳草图布局。草图唯一缺"流式/EoS→首 token"贡献，用一条底注 + 一只"下一片段"回环虚线箭头补上，不加框、不变宽。

## 一、设计目标（concat 矩阵视图，少字）

绘制一张简洁、科学严谨、IEEE / ACM / Nature 方法机制图风格的 draw.io 矢量图，主题
**"LLM KV Cache 增量更新机制"**。白底、低饱和、黑白可读。约 560×320（近正方，不再过宽）。

**核心：一次增量更新的显存矩阵拼接。**
- 顶部括号 `Cache_new (updated)` 罩住蓝绿两块。
- **蓝方阵**（N×d，带网格纹）：`K_prev, V_prev (cached · reused)`，下标 `N × d — reused, not recomputed`——历史复用、不重算。
- `+ concat`。
- **细绿条**（M×d，带网格纹）：`new k,v`，下标 `M × d`——本次新增片段、仅对其投影/前向。
- 右侧 `new ASR text chunk Δ` 框 → 绿条，箭头 `project new tokens only`。
- 底部浅紫虚线回环箭头 `next chunk: appended → cached history`——示意增量在发声期间反复发生（绿条下一轮并入蓝方阵）。
- 底部一句注承载系统贡献（prefill 前移发声期 + EoS 后复用末步 logits 出首 token）。

### 少字原则（回应"文字不用太多"）
- 矩阵格子**不写字**，仅蓝/绿 + 网格纹表意；底层字段（past_key_values/attention_mask/position_ids）正文已详述，图中不堆。
- N、d、M 沿用 §3.3.3 复杂度记号，使图文呼应。

---

## 二、模块与数据流

- 蓝方阵 `K_prev, V_prev (cached, N×d)`：历史上下文已缓存的键值对，复用、不重算。
- 绿条 `new k,v (M×d)`：新增 ASR 文本片段 Δ 经投影得到的键值对，仅对其前向计算。
- `Cache_new = concat(蓝, 绿)`：拼接后即更新缓存。
- 回环：下一片段到达时，本轮绿条并入蓝方阵成为历史，重复增量。
- 底部说明条（忠于 line 191/195/205）：
  `Each arriving chunk computes only its new (k,v) (M×d) and concatenates it to the cached history (N×d); most prefill thus completes during speech, and after End-of-Speech the first token is decoded from the cached last-step logits.`

---

## 三、版面与样式

- 单栏窄幅，约 560×320（近正方，扁平度大幅下降）。
- 配色：蓝（缓存复用）`#dae8fc/#6c8ebf`，网格纹 `#aebfd6`；绿（新算）`#d5e8d4/#82b366`，网格纹 `#9ccb9c`；新片段 `#eaf3e6/#82b366`；回环紫 `#9673a6`；括号/底注灰 `#888888/#666666`。
- 矩阵格纹细、对齐整齐、留白充足、箭头清晰。

---

## 四、避免的错误（务必遵守）

- 不重复图2——不画注意力矩阵、不画 O(N²)/O(N) 角标。
- 不把"历史"画成每步重算——历史块必须蓝色复用，仅当步绿块新算。
- 不堆底层字段文字（past_key_values/attention_mask/position_ids 等正文已详述）；块内不写字。
- 不画 ASR/VAD/TTS——本图只聚焦 LLM 侧 KV Cache 增量更新。
- 体现"prefill 前移到发声期间、EoS 后只剩末段 + 复用末步 logits 出首 token"这一系统贡献。
