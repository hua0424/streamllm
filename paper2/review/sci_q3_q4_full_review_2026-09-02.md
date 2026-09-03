# SCI Q3–Q4 Full Review Report

**评审日期**：2026-09-02  
**评审对象**：`paper2/thesis_draft.md`  
**评审模式**：`academic-paper-reviewer` full review  
**目标标准**：通用 SCI Q3–Q4 系统、应用型语音技术、智能交互与 AI systems 论文标准  
**编辑决定**：**Major Revision**  
**Critical / fatal finding**：**无**  
**稿件修改状态**：本轮评审只读，未修改论文、代码或实验工件

---

## 1. 评审范围与执行说明

本轮采用五个角色分离的审稿席位：

1. **Journal-Fit Reviewer**：论文定位、贡献层级、读者价值和期刊稿成熟度；
2. **Methodology Reviewer**：estimand、统计结构、GPU 计时、baseline 公平性和可复现性；
3. **Domain Reviewer**：流式语音对话、KV cache、增量预填充、播放边界和角色恢复；
4. **Perspective Reviewer**：真实部署、HCI、用户实际听到、异步链路和 artifact 可用性；
5. **Devil’s Advocate**：最强反论证、替代解释、逻辑跳跃和贡献坍缩风险。

五席先在论文内容不可见的状态下冻结审查标准，随后分别阅读同一主稿并提交互不可见的正式报告，最后由 Editorial Synthesizer 进行汇总。角色分离和互盲可以降低同席意见互相影响，但五席使用同一模型家族，因此**不能称为独立误差过程或独立审稿人样本**。

用户没有指定具体目标期刊，也没有提供期刊 criteria binding，因此本轮标记为：

```text
criteria_binding_unavailable
calibration_status=NOT_CALIBRATED
```

本报告不声称稿件符合某一本具体期刊，也不预测录用概率。

---

# 2. Editorial Decision Letter

## 2.1 决定

**Major Revision，修订后重新评审。**

五个席位一致认为稿件存在可辨识的系统研究贡献，没有发现需要否定整个研究的不可修复问题。当前阻碍投稿的主要因素不是负结果，而是：

- 核心 C2 机制尚未完成模型语义正确性闭环；
- E1/E2 的事件含义和交叉重复统计仍需修正；
- C-E1 的两条路径不是 token-equivalent，不能按单因素机制比较解释；
- EOS/EOT 的多轮角色恢复存在具体实现风险；
- 稿件的“播放感知/用户实际听到”叙事仍比软件游标和受控实验的证据范围略宽；
- 当前仍是硕士论文式长稿，而不是完成压缩和投稿声明的 SCI 期刊稿。

这些问题均可以通过定向代码修复、小规模正确性实验、现有数据重分析和论文重构解决，因此不建议 Reject。

## 2.2 稿件的真实优势

### A. Novelty 边界诚实

稿件明确承认“对话历史应反映用户所听内容”已有 OpenAI、Azure 和 LiveKit 等工程先例，没有把高层原则包装为原创。贡献被限定为公开级联栈中的播放位置关联、显式 KV 状态修正、角色恢复和可检视评测，见：

- `paper2/thesis_draft.md:42-63`
- `paper2/thesis_draft.md:77-96`

### B. C2 形成了清晰的系统状态对象

论文没有停留在文本截断层面，而是统一管理：

- 软件播放采样位置；
- TTS 文本片段；
- assistant token span；
- KV cache；
- attention mask；
- assistant token ledger；
- position IDs；
- chat role boundary。

对应形式化和实现见：

- `paper2/thesis_draft.md:151-249`
- `paper2/thesis_draft.md:384-430`
- `src/llm/stream_llm_inference.py:301-414`

五席一致认为，这是当前最具发表价值的核心贡献。

### C. 负结果和旧口径错误披露充分

稿件如实报告：

- B@0.92 在同步 compute-readiness 口径下慢于 System A；
- oracle 指标只是条件性乐观下界/收益上界；
- 旧 E1/E2 的 48.3→12.1 ms 是时间原点错误造成的口径 artifact；
- E3 四项点估计与预设方向相反且不显著；
- A2 不支持策略因果比较；
- A1/P1 不是声卡、声学停播或生产端到端测量。

主要位置：

- `paper2/thesis_draft.md:570-630`
- `paper2/thesis_draft.md:674-739`

没有发现为了保留正结果而删除负结果或隐藏失败协议的证据。

### D. 正式 campaign 的可审计性较强

确认性 E1/E2 保存了 5000 条 raw records、五个进程身份、输入/TEN/模型内容哈希、条件顺序、三类时间戳、manifest、validation、analysis 和 checksums。固定轨迹 E3、联合 A1 和 prepared-state P1 也分别保留了正式工件。稿件没有跨 campaign 池化绝对时间。

### E. 无 fatal trigger

未发现以下情形：

- 全部主要计时无法恢复；
- 主要数据无法确定或无法审计；
- 无法恢复的伪重复；
- 所有 baseline 均完全不可识别；
- 系统性删除失败记录；
- 核心贡献被已有工作完全预示且无剩余增量；
- 核心实现已被证明必然无效。

---

# 3. 可发表的最稳妥定位

现有证据最稳妥支持的定位是：

> **一个以软件播放游标和 TTS 文本片段为边界、显式维护 KV/cache-mask/position/token-ledger/role 状态的级联式推理 runtime/prototype；其证据包括组件状态检查、模型侧恢复微基准、prepared-state 软件控制路径，以及受控文本条件下的限定性代理实验。**

在该定位中：

- **C2 是唯一核心机制贡献**；
- **C1 是 candidate compute-readiness、oracle policy 上界和计算浪费的受控 characterization**，不是生产低延迟收益；
- **C3 是探索性扩展和负结果**，不是已验证的策略改进；
- “playback-aware”表示软件播放游标驱动的片段级状态操作，不等于用户声学上真正听到的物理真值；
- E1/E2、E3、A1 和 P1 分别回答不同层级的问题，不能组合成真实异步音频闭环结果。

若采用这一窄定位，真实声卡闭环和人工 HCI 评测不是无条件投稿门槛；若保留更强的实际语音交互、用户体验或生产端到端主张，则必须增加对应实验。

---

# 4. Consolidated Major Findings

以下编号用于去重和后续追踪，**不表示作者执行优先级**。

## CF-01：`arrival→first_token_ready` 是内部 compute readiness，不是当前 harness 的可交付 TTFT

**来源**：Methodology、Domain、Devil’s Advocate  
**严重性**：Major

### 证据

论文把 `first_token_ready` 描述为首 token 完成计算并可被下游消费：

- `paper2/thesis_draft.md:255-279`

但代码中，`on_token_decoded` 在 token 被选中后立即触发，随后还要执行一次模型 forward 更新 KV，最后才 `yield`：

- `src/llm/stream_llm_inference.py:301-346`

对于 survived speculation：

- `first_token_ready_ns = candidate_first_ns`；
- `first_deliverable_token_ns = endpoint_ns`；
- endpoint 又在同步候选 chunk 生成完毕后才记录。

见：

- `experiments/sci34_supplement/e1e2_confirmatory/runtime.py:319-395`

### 影响

当前 27.70/62.38 ms 比较可以描述内部首 token 选择或 candidate compute readiness，但不能同时称为：

- gate-authorized token latency；
- consumer-observed TTFT；
- 首个真正可播出 token 的实际响应延迟。

对现有 raw records 的只读复算约为：

| 路径 | arrival→candidate-ready | arrival→first-deliverable |
|---|---:|---:|
| System A | 27.70 ms | 27.70 ms |
| B@0.92，全体 | 62.38 ms | 约 257.6 ms |
| B@0.92，survived | 约 62.43 ms | 约 353.8 ms |
| B@0.92，non-survived | 约 62.28 ms | 约 62.28 ms |

B 的 deliverable 数值也不能解释为生产延迟，因为它受到同步 harness “先生成完整候选，再接受”的程序顺序影响。

### Minimum remedy

1. 将 27.70/62.38 ms 统一命名为：
   - 首 token 内部选择/候选计算就绪延迟；
   - candidate-token compute-readiness latency。
2. 删除“实际交付”“已可被下游消费”等无证据等同。
3. 使用现有 raw records 离线补报：
   - arrival→candidate-ready；
   - arrival→endpoint-accept；
   - arrival→first-deliverable；
   - endpoint→first-deliverable；
   - arrival→consumer-observed。
4. 将“候选领先端点中位 291 ms”改称：
   - 同步协议中 candidate-first-token 到 post-candidate oracle acceptance 的内部程序间隔；
   - 不得解释为用户继续发言 291 ms 或自然端点提前量。
5. 摘要、形式化、图 6-2/6-3、讨论和结论采用同一事件词汇。

**是否需要 GPU**：否，可由现有 raw records 完成。  
**Stronger option**：运行真正异步的 endpoint/gate/consumer harness。

---

## CF-02：5×100 是 session×dialogue 交叉重复设计，现有 nested bootstrap 不匹配

**来源**：Methodology、Devil’s Advocate  
**严重性**：Major

### 证据

五个独立进程重复运行同一批 100 条话语：

- `paper2/thesis_draft.md:506`
- `paper2/thesis_draft.md:537`
- `paper2/thesis_draft.md:593-595`

但分析器先抽 session，再在每个抽中 session 内独立抽 dialogue：

- `experiments/sci34_supplement/e1e2_confirmatory/analyze.py:208-263`

这相当于把 dialogue 嵌套在 session 内，而真实设计是共同的 dialogue 跨五个 session 重复。

### 正确的分析单位

- 内容采样单位：100 个唯一 dialogue/utterance；
- 技术重复单位：5 个独立 process sessions；
- 观测单元：500 个 session×dialogue cells；
- 条件比较：每个 cell 内配对。

### 交叉敏感性复算

| Estimand | 当前 CI | crossed audit CI（约） |
|---|---:|---:|
| C-E1 compute-ready A−B | [−35.30, −34.11] ms | [−35.43, −33.94] ms |
| C-E2 compute-ready never−B | [−0.55, 0.51] ms | [−0.63, 0.61] ms |
| C-E1 oracle A−B | [16.12, 18.75] ms | [14.39, 20.30] ms |
| C-E2 oracle never−B | [19.50, 22.10] ms | [17.84, 23.62] ms |
| B@0.92 pooled waste | [0.020, 0.037] | [0.011, 0.047] |
| B@0.92 survival | [0.628, 0.712] | [0.58, 0.76] |

点估计和方向性结论均不变，但 oracle、waste 和 survival 的区间明显变宽。

### Minimum remedy

1. 新建 `analysis_v2.json`，不得覆盖 `analysis_v1.json`；
2. 采用全局 session 与全局 dialogue 的 crossed/product bootstrap；
3. 或先对每个 dialogue 汇总五个技术重复，再做 dialogue-cluster bootstrap，并单列 session variation；
4. 全文把 `n=500` 改成：
   - 100 个唯一话语；
   - 5 个进程重复；
   - 每条件 500 个运行观测；
5. survival/waste 的 CI 由 dialogue 层决定，不能把确定性重复当成 500 个独立 Bernoulli 样本；
6. 报告五个 session 的逐 session 时延效应。

**是否需要 GPU**：否。  
**是否改变结论**：不改变方向性结论，但必须更新正式 CI 和有效样本量解释。

---

## CF-03：System A 与增量路径不是 token-equivalent，C-E1 不能解释为纯 prefill 机制效应

**来源**：Methodology、Domain、Devil’s Advocate  
**严重性**：Major

### 证据

System A 对完整字符串一次性 tokenization；System B 对 ASR segments 分别 tokenization 后追加：

- `experiments/sci34_supplement/e1e2_confirmatory/runtime.py:211-266`
- `experiments/sci34_supplement/e1e2_confirmatory/runtime.py:287-381`
- `src/llm/stream_llm_inference.py:435-475`

对 BPE tokenizer，通常不保证：

\[
T(u_1)+T(u_2)=T(u_1u_2).
\]

正式 raw records 显示：

- System A vs B@0.92：
  - 完整输出 token 序列相同 280/500；
  - 首 token 相同 465/500；
  - 44/100 个唯一话语出现确定性输出分岔。
- B@0.92 vs B-never：
  - 完整输出 token 序列 500/500 相同。

### 影响

C-E1 当前估计的是：

> 一次性字符串 tokenization/full-prefill 实现路径与逐段 tokenization/incremental-path 实现路径的整体差异。

它不能严格估计：

> 完全相同 tokenized context 下，仅改变 prefill 调度的纯效应。

因此 −34.69 ms 不能排他性归因于额外 assistant-role forward；它可能同时包含 tokenization、分块、kernel 形态、Python 调度和前向次数的组合差异。

### Minimum remedy：不重跑路线

1. 披露 280/500 完整输出一致率和 465/500 首 token 一致率；
2. 将 C-E1 明确改称 implementation-path comparison；
3. 将“机制是额外两次串行前向”改成：
   - “结果与额外角色前向、分块和 kernel 形态的组合效应一致”；
   - “本实验未隔离各组成因素”。
4. 保留 B@0.92 vs B-never 500/500 输出一致性，支持 C-E2 的 B 内部比较。

**是否需要 GPU**：否。

### Stronger option：保留纯机制主张

若要继续声称 C-E1 隔离了 incremental prefill 的纯效应，必须：

1. 生成唯一规范 prompt token IDs；
2. 将相同 token IDs 分别走 one-shot 和 incremental forward；
3. 验证 prompt IDs、KV length、首 token logits 和 greedy output；
4. 再进行 token-equivalent C-E1 定向 GPU 重跑。

该情况下需要定向 GPU 重跑，但不必无条件重跑完整九点 C-E2。

---

## CF-04：C2 缺少 crop+role 与 clean re-prefill 的模型语义等价验证

**来源**：Domain、Devil’s Advocate  
**严重性**：Major；窄定位稿件也必须解决

### 当前证据能够证明什么

现有测试证明：

- DynamicCache、attention mask 和 `seq_length` 一致；
- assistant ledger 会同步裁剪；
- crop 后可以继续生成；
- A1 中 crop+role recovery 比从零重新 prefill 快；
- P1 中软件 stop、lookup、crop 和 role recovery 可以执行。

### 当前证据不能证明什么

尚未验证：

\[
\operatorname{crop}(KV)+\operatorname{role\ recovery}
\]

与

\[
\operatorname{clean\ re\text{-}prefill}(retained\ history)
\]

是否产生等价的 next-token distribution 和后续 continuation。

长度一致只是结构合法性的必要条件，不是模型语义正确性的充分条件。

### Minimum remedy：定向正确性实验

在 Qwen2-7B 上建立两条路径：

- 路径 A：从已生成 KV crop，再恢复角色；
- 路径 B：从相同保留 token 和规范 chat serialization 干净 re-prefill。

比较：

1. 规范 token sequence；
2. next-token top-1；
3. top-k 集合；
4. max/mean logit difference；
5. 预先规定数值容差；
6. 16–32 token greedy continuation；
7. 加入下一 user 轮后的 logits 和 continuation。

覆盖：

- `p=0` 全回滚；
- clean fragment boundary；
- mid-fragment 向片段末端吸附；
- 多种 context length；
- 多轮 crop；
- EOS/EOT 已生成和未生成；
- max-token truncation。

**是否需要 GPU**：是，定向小规模 Qwen2-7B 正确性验证。  
**是否需要重跑完整 campaign**：否。

---

## CF-05：EOS/EOT 已进入 KV 后可能被角色恢复重复注入

**来源**：Domain  
**严重性**：Major；核心实现正确性问题

### 证据

`generate_accumulating()` 检测到 EOS 后仍将该 token：

- forward 进 KV；
- 写入 `assistant_token_ids`；
- 然后 break。

见：

- `src/llm/stream_llm_inference.py:315-348`

随后 `reopen_user_role()` 又无条件注入包含 assistant-close 的角色切换串：

- `src/llm/stream_llm_inference.py:401-410`

在 Qwen ChatML 中，EOS/EOT 通常是 `<|im_end|>`，因此可能形成：

```text
assistant content
<|im_end|>
<|im_end|>
<|im_start|>user
```

正式 B-never 记录中有 35/500 条以 EOS 结束，而 C-E1/E2 是单轮实验，没有覆盖 EOS 后 reopen 的下一轮路径。

### Minimum remedy

1. 明确 assistant content token 与 role terminator token 的状态语义；
2. 若 EOT 已进入 KV，reopen 只注入 user-open；
3. 若 EOT 未进入 KV，再注入 EOT + user-open；
4. 或统一规定生成器不把 EOT 写入 content ledger/KV，由 role closer 只写一次；
5. 使用规范 `apply_chat_template(..., tokenize=True)` 对照状态机 token IDs；
6. 增加：
   - EOS 正常完成；
   - max-token 截断；
   - mid-fragment barge-in；
   - 全回滚；
   - 下一 user 轮和下一 assistant 轮；
7. 明确 `assistant_token_ids` 是否包含 EOT，并同步 timeline 与 token waste 定义。

**是否需要 GPU**：需要代码修复；建议在目标 Qwen2-7B 上做多轮定向验收。  
**是否需要重跑旧正式 campaign**：通常不需要，但必须有新的正确性工件和结果。

---

## CF-06：E3 是 fixed-detector-conditioned proxy，不是人类语义或 HCI 效果

**来源**：Methodology、Perspective、Devil’s Advocate  
**严重性**：Major

### 当前限制

E3 的 CI 只包含 dialogue sampling uncertainty，不包含：

- 词面规则误差；
- LLM judge 假阳性/假阴性；
- prompt-version uncertainty；
- 人类感知误差。

同时存在：

1. pair-weighted 点估计与 dialogue-weighted bootstrap point 略有不同；
2. 0.5 和 clean boundary 在部分轨迹上对应相同实际历史/目标；
3. playback 条件的较高阳性率允许“任务域自然重合或模型再生成”解释；
4. 规则与 LLM judge 不是两个人类真值来源。

### Minimum remedy

1. 将 RQ1 改为：
   - “在固定自动规则和固定 LLM judge 下，positive rate 有何差异？”
2. 所有 CI 标明为：
   - fixed-detector-conditioned sampling uncertainty；
3. 统一点估计与 bootstrap weighting，或分别定义两个 estimand；
4. 增加按唯一实际边界去重的 sensitivity analysis；
5. 报告规则与 judge 的 pair-level agreement/confusion；
6. 归档 E3 精确 processed input，或提供可生成同一 SHA-256 的确定性 builder；
7. 保留“不支持优效、等效、非劣、伤害或人类感知结论”。

**是否需要 GPU**：否，可基于现有 records 处理。

### Stronger option

若保留“改善语义一致性”的因果主张，则至少需要：

- 随机 canary/不会由 user prompt 推出的差异事实；
- 预声明保留/删除设计；
- 盲法人工标注或经过校准的检测器。

若声称自然度、信任或用户体验，则需要直接 HCI/人工评测。

---

## CF-07：软件游标、设备播放和用户实际听到仍需严格区分

**来源**：Journal-Fit、Perspective、Devil’s Advocate  
**严重性**：Major

### 问题

当前 `p` 最多表示软件播放器报告的已消费样本。真实部署还存在：

- 应用音频队列；
- audio API 缓冲；
- OS mixer；
- 内核/驱动缓冲；
- 声卡 DMA；
- 蓝牙或网络音频缓冲；
- 扬声器和声学传播；
- stop 后已经提交但尚未发声的采样。

Mock TTS 只构造时长，不产生可测波形。P1 是 prepared-state、无争用的软件控制路径。因此 400/400 构造检查或 180/180 零软件采样泄漏不能解释为真实声学边界正确率。

### Minimum remedy

全稿固定三个层级：

1. `software-consumed samples`；
2. `device-presented samples`；
3. `acoustically heard content`。

并完成：

- 当前保证统一称为“软件播放游标驱动的 TTS-fragment-level retention”；
- `heard_text`、`strict_unheard` 等作为兼容字段时标注 operational alias；
- 400/400 明确是 implementation invariant check；
- 标题、摘要和贡献不得暗示已经测到用户听觉真值。

**是否需要 GPU**：否。

### Stronger option

若保留“用户实际听到”或生产 playback-aware 主张，则需要：

- 在线 TTS；
- bounded audio queue；
- device clock 或 loopback 波形；
- stop request、device stop、acoustic stop、timeline query、crop 和 role recovery 的统一时间轴；
- 不同 buffer、拥塞和并发条件下的边界误差与 p50/p95/p99。

---

## CF-08：组合式 novelty 尚缺可复查检索闭环

**来源**：Journal-Fit、Domain、Devil’s Advocate  
**严重性**：Major

### 问题

论文的主要原创性属于合取式判断：

> 开源级联语音栈 + 播放位置驱动 + 显式 KV 裁剪 + 角色恢复 + 可复算评测。

当前相关工作分类合理，但文献数量较少，混合了产品文档、GitHub、预印本和正式论文；尚未报告检索数据库、查询词、纳入/排除标准、citation snowballing 和最近邻排除理由。

主要位置：

- `paper2/thesis_draft.md:70-119`
- `paper2/thesis_draft.md:92-96`

### Minimum remedy

1. 报告检索截止日期；
2. 给出数据库、代码平台和产品文档渠道；
3. 给出关键词与同义词族；
4. 说明纳入/排除规则；
5. 扩展传统 barge-in、turn-taking、incremental dialogue processing、incremental NLG/TTS、cache rollback、chunked prefill 和 prefix-cache 文献；
6. 对最接近的 3–5 项工作逐项比较；
7. 扩展表 2-1，使每个 novelty 子成分都有来源依据；
8. 将“尚未发现”限定为“在所报告的公开检索范围内尚未发现”。

**是否需要 GPU**：否。

---

## CF-09：三项贡献成熟度不对称，应以 C2 为唯一核心

**来源**：Journal-Fit，其他席位的证据边界支持  
**严重性**：Major

### 当前成熟度

- **C2**：有形式化、状态不变式、代码、A1 和 P1，是最成熟的贡献；
- **C1**：同步 compute-readiness 不优于基线，正向结果仅为 oracle policy 上界；
- **C3**：A2 有独立生成混杂，且没有观察到重写优于朴素策略。

### Minimum remedy

- C2：唯一核心机制贡献；
- C1：次要的 candidate-readiness/waste characterization；
- C3：exploratory extension/negative result；
- 摘要、引言贡献、RQ、结果、讨论和结论统一采用上述层级；
- RQ5 不再问“是否改善”，改为描述当前探索性运行的表现。

**是否需要 GPU**：否。

若坚持三项等强贡献，则需分别补真实异步 C1 和固定轨迹 A2，成本显著提高。

---

## CF-10：artifact 和期刊投稿形态尚未闭环

**来源**：Journal-Fit、Methodology、Perspective  
**严重性**：Major

### 发现

- 顶层 README 仍写确认性 formal campaign 待执行，与正文已完成状态冲突；
- 缺稳定的公开或匿名 artifact URL；
- 仓库缺正式 `LICENSE` 文件，README 文字声明不能替代许可证；
- E3 exact processed input 尚未随当前工件闭环；
- 缺统一的 campaign/result/commit/entrypoint matrix；
- 当前参考文献明确仍是工作格式；
- 缺 ethics、COI、funding、author contribution、data availability、code availability 等投稿声明；
- 当前 830 行合并稿重复较多，仍是硕士论文式结构。

### Minimum remedy

1. 保留当前 thesis authoritative Markdown；
2. 另建独立压缩 SCI 稿；
3. 提供公开或匿名 artifact 入口；
4. 加入正式 LICENSE 和第三方许可说明；
5. 提供 accepted run、代码 commit、结果 commit、输入 hash、模型身份、入口命令和分析文件映射；
6. 发布 E3 exact input 或确定性 builder；
7. 提供 CPU-only smoke 与 analysis-only 复算命令；
8. 更新 README 的 campaign 状态；
9. 统一参考文献格式和访问日期；
10. 增加适用的 submission declarations。

**是否需要 GPU**：否。

---

# 5. Minor / Reporting Findings

## MF-01：阈值数量表述错误

当前应写为：

> 八个数值阈值加一个 never-speculate 对照，共九个工作点。

不应写“九个离散阈值”。

## MF-02：A1 固定执行顺序和固定裁剪量需要披露

A1 每轮按固定顺序执行不同操作，且只裁掉固定 32 token。它可以支持受测协议内的模型侧比较，但不代表真实打断位置和裁剪量分布。

最低修改：

- 披露固定顺序和固定裁剪量；
- 不把 2.254–40.620× 外推为典型用户打断收益。

## MF-03：P1 的 P95 只能是经验描述

P1 每单元 n=20，经验 P95 主要由一至两条最大记录决定。应称 descriptive empirical P95，不解释为稳定生产 SLO 或 p99 行为。

## MF-04：E3 重复实际边界的 weighting

0.5 与 clean boundary 在部分轨迹中对应相同实际历史和目标。按 dialogue 聚类可处理独立性，但主 estimand 仍对重复标签赋权。应报告重复数量并提供去重 sensitivity analysis。

## MF-05：timeline API 的顺序不变量未由接口强制

`add_fragment()` 和 `attach_chunk()` 当前依赖调用方保证 token span、fragment 和 chunk 的顺序。并发部署声明下，应增加：

- token span 单调连续断言；
- fragment 关闭状态；
- chunk 归属/顺序验证；
- 乱序失败测试。

## MF-06：代码字段命名强于证据层级

`heard_text`、`n_heard`、`strict_unheard`、`ground-truth` 容易被解释为声学真值。建议新增兼容别名：

- `fragment_retained_text`；
- `fragment_included_ids`；
- `character_ratio_whitespace_proxy_tail`；
- `measurement_level=software_fragment_proxy`。

## MF-07：参考文献和声明

参考文献需按最终投稿格式统一作者、题名、年份、卷期页码、DOI、URL 和访问日期；检查 2026 年预印本是否已有正式版本。声明中如无人体参与者、资助或利益冲突，也应明确写 `not applicable` 或 `none`，不能留空。

---

# 6. Devil’s Advocate Findings Adjudication

DA 没有提出 CRITICAL finding。其七项 Major challenge 的编辑裁决如下。

| DA finding | 裁决 | 理由 |
|---|---|---|
| DA-M1：组合创新可能只是公共 API 拼装 | **Partly validated** | 组合式 novelty 的检索和最近邻比较不足；但 sample→fragment→token 反查和 KV/mask/position/ledger/role 状态编排仍构成非零系统增量。 |
| DA-M2：缺 crop-vs-reprefill 语义等价 | **Validated** | A1 只比较耗时，长度不变式不证明 logits/续写等价。C2 为核心后，此验证无条件必要。 |
| DA-M3：E1 是非等价实现路径 | **Validated** | A/B 完整输出相同仅 280/500；可以保留为 implementation-path comparison，不能解释为 pure prefill effect。 |
| DA-M4：oracle 291 ms 由同步 acceptance 定义构造 | **Validated as scope limitation** | 数值是真实程序间隔，但不是自然用户端点提前量或可交付延迟。 |
| DA-M5：5×100 应按 crossed design 分析 | **Validated** | 同一 100 条内容跨五个技术 session 重复，现有 nested bootstrap 不匹配。 |
| DA-M6：E3 高背景阳性、构念敏感度不足 | **Partly validated** | 不能支持人类语义效果，但不推翻稿件已经限定的 detector-conditioned null result。 |
| DA-M7：无真实音频时只是软件游标感知 | **Validated** | 阻断 broad system claim，但不阻断明确收窄的 fragment-level runtime/prototype。 |

**DA fatal trigger F1–F7：均未触发。**

---

# 7. 实验需求分层裁决

本节的三层表示“不同主张需要什么证据”，不是作者执行顺序。

## 7.1 无条件必要

即使采用最窄的 runtime/prototype 定位，也需要完成：

| 项目 | 是否需要 GPU | 是否需要完整 campaign 重跑 |
|---|---:|---:|
| crossed session×dialogue `analysis_v2` | 否 | 否 |
| compute-ready / endpoint / deliverable / consumer 指标重命名与离线重分析 | 否 | 否 |
| 291 ms oracle lead 解释修正 | 否 | 否 |
| C-E1 非等价率披露并降为 implementation-path comparison | 否 | 否 |
| E3 estimand、weighting、重复边界 sensitivity 修正 | 否 | 否 |
| crop+role vs clean re-prefill 语义等价验证 | **是，定向小实验** | 否 |
| EOS/EOT 角色恢复修复和多轮回归 | **建议目标 7B GPU 验收** | 否 |
| novelty 检索和 nearest-neighbor matrix | 否 | 否 |
| artifact、README、LICENSE、processed input、submission 声明 | 否 | 否 |

## 7.2 只有保留较强主张时才必要

| 希望保留的主张 | 必须增加的证据 |
|---|---|
| C-E1 隔离了 incremental prefill 的纯效应 | token-equivalent C-E1 定向 GPU 重跑 |
| 系统依据用户真正“听到”的内容裁剪 | device clock、loopback 或声学播放边界实验 |
| 改善生产 barge-in latency | 在线异步 ASR/TTS/队列/声卡/并发实验 |
| 推测改善可交付或用户感知延迟 | 独立 endpoint gate 和 consumer 时间戳 |
| playback 策略改善语义一致性 | 随机 canary 或其他可识别语义实验 |
| 改善自然度、信任或用户体验 | 直接人工/HCI 评测 |
| 重写/标记策略具有因果优势 | 固定轨迹 A2 |

## 7.3 增强 Q3 竞争力但不是窄主张门槛

- 真实异步音频闭环；
- 声卡/loopback 的 acoustic stop；
- 双盲人工标注和 canary calibration；
- 中文或其他非空白分词语言；
- 不同主模型、TTS、chat template 和推理引擎；
- A1 多种裁剪长度和随机执行顺序；
- P1 增加重复并报告 p99；
- 固定轨迹 A2；
- 带 DOI 的 artifact 和容器化环境。

如果资源只允许增加一个较大的系统实验，优先选择：

> **真实异步音频闭环与声学 stop 边界实验**，而不是继续增加同步 GPU 微基准。

---

# 8. Immutable Non-Ranking Revision Roadmap Core

以下条目按评审 finding 的来源顺序记录。ID 仅用于后续追踪，**不表示优先级或执行顺序**。`author_triage` 留待后续组织修订时填写。

| ID | Severity | Revision item | Author triage | 验收标准 |
|---|---:|---|---|---|
| RMAP-EIC-01 | Major | 将真实异步音频缺口与稿件范围一致化 | `not_provided` | 标题、摘要、贡献、方法、结果和结论统一限定为 fragment-level software-cursor runtime/prototype；如保留真实用户听觉或生产闭环主张，则附对应真实异步实验。 |
| RMAP-EIC-02 | Major | 将 C2 明确为核心贡献，重标 C1/C3 证据成熟度 | `not_provided` | C2 为核心机制；C1 只承担 compute-readiness/oracle characterization；C3 为探索性负结果。 |
| RMAP-EIC-03 | Major | 补充组合式 novelty 检索与最近邻比较 | `not_provided` | 报告检索日期、数据库/来源、查询式、筛选规则，并逐项比较各 novelty 子成分。 |
| RMAP-EIC-04 | Major | 将硕士论文长稿压缩为独立期刊稿 | `not_provided` | 形成去除章节性重复和过程审计复述的独立稿，核心方法、结果、限制和工件信息仍可自足理解。 |
| RMAP-EIC-05 | Major | 完成公开 artifact 闭环 | `not_provided` | 提供稳定入口、正式 LICENSE、run/commit mapping、E3 exact input 或 builder、统一复现命令，并修正 README 状态。 |
| RMAP-EIC-06 | Minor | 收窄题名和关键词 | `not_provided` | 题名反映 fragment-level/software-cursor 或 prototype 范围，不暗示生产端到端或用户听觉真值。 |
| RMAP-EIC-07 | Minor | 修正阈值/工作点计数 | `not_provided` | 全稿统一写“八个阈值加 never，共九个工作点”。 |
| RMAP-EIC-08 | Minor | 正式化参考文献 | `not_provided` | 全部条目按目标格式完成并删除工作稿提示。 |
| RMAP-R1-01 | Major | 区分 candidate compute readiness 与 deliverability | `not_provided` | 删除“ready 即可被下游消费”的无证据等同；291 ms 改为内部程序间隔；如 raw 可得，补 first-deliverable/consumer。 |
| RMAP-R1-02 | Major | 按 crossed design 重算 C-E1/C-E2 不确定性 | `not_provided` | 正式分析明确 100 unique dialogues × 5 process sessions，并采用 crossed analysis_v2；更新 CI 和 n 表述。 |
| RMAP-R1-03 | Major | 处理 A/B 非 token-equivalent 比较 | `not_provided` | 披露 280/500、465/500 等价率并降为 implementation-path comparison；如保留 pure effect，则提供定向重跑。 |
| RMAP-R1-04 | Major | 限定 E3 CI 的误差来源 | `not_provided` | 所有 E3 CI 明确是 fixed-detector-conditioned sampling uncertainty，不包含 judge error 或人类感知误差。 |
| RMAP-R1-05 | Minor | 对齐 E3 点估计与 bootstrap estimand | `not_provided` | 点估计和区间采用同一 weighting；或分别定义并解释。 |
| RMAP-R1-06 | Minor | 披露 A1 固定顺序和 P1 P95 边界 | `not_provided` | P1 n=20/cell 的 P95 只称经验描述；A1 固定顺序写入限制。 |
| RMAP-R1-07 | Minor | 修正 RQ5 因果措辞 | `not_provided` | 改为描述当前探索性运行表现，不再询问“是否改善”。 |
| RMAP-R2-01 | Major | 验证 crop+role 与 clean re-prefill 语义等价 | `not_provided` | 在预声明 crop 条件下比较 token serialization、logits、top-k 和 greedy continuation，并报告容差和差异案例。 |
| RMAP-R2-02 | Major | 修复 EOS/EOT 后角色恢复重复边界 | `not_provided` | EOS、max-token、mid-fragment、full rollback 和多轮路径均证明每轮只含一个合法 assistant close。 |
| RMAP-R2-03 | Minor | 在 timeline API 强制顺序不变式 | `not_provided` | 对 token span、fragment、chunk 和 sample range 增加 assertion/validation 与乱序失败测试。 |
| RMAP-R2-04 | Minor | 收窄 heard/strict/ground-truth 术语 | `not_provided` | 术语表区分 physical truth、software cursor、fragment retained boundary、in-fragment proxy 和 oracle endpoint。 |
| RMAP-R3-01 | Major | 处理自然度、信任和用户感知证据缺口 | `not_provided` | 无 HCI 数据时不作体验改善主张；保留时须报告直接人工/HCI 证据。 |
| RMAP-R3-02 | Major | 限定外部有效性 | `not_provided` | 结论明确限于英语任务型对话、单一模型、特定模板/引擎、RTX 3090 和受控 harness。 |
| RMAP-R3-03 | Minor | 补 submission declarations | `not_provided` | 包含适用的 ethics、consent、COI、funding、author contribution、data/code availability 声明。 |
| RMAP-DA-01 | Minor | 说明 A1 固定裁剪 32 token 的代表性 | `not_provided` | 明确固定裁剪量不能代表其他打断位置；扩展实验时覆盖多个裁剪长度。 |
| RMAP-DA-02 | Minor | 处理 E3 同一实际边界重复加权 | `not_provided` | 报告重复边界数量，提供去重或重新加权 sensitivity analysis，并说明结论是否变化。 |

---

# 9. 建议的依赖式修订组织

本节只是为了后续组织工作时减少返工，不改变上一节 non-ranking roadmap 的性质。

## 工作包 A：现有数据的统计和事件重分析

- 新建 crossed `analysis_v2`；
- 重算正式 CI；
- 拆分 compute-ready、endpoint、deliverable、consumer；
- 修正 291 ms 的含义；
- 修正 E3 estimand、weighting 和重复边界 sensitivity。

## 工作包 B：核心 C2 正确性闭环

- 修复 EOS/EOT 状态；
- 建立规范 chat token serialization；
- 实现 crop vs clean re-prefill semantic-equivalence test；
- 覆盖多轮和全部边界条件；
- 在 Qwen2-7B 上执行定向验证并归档结果。

## 工作包 C：论文主线收敛

- C2 作为唯一核心贡献；
- C1 改为受控 compute-readiness/waste characterization；
- C3 降为探索性扩展；
- C-E1 降为 implementation-path comparison，除非完成 token-equivalent rerun；
- “用户实际听到”统一改为设计目标、软件游标代理或条件性表述。

## 工作包 D：期刊稿和 artifact

- 保留当前硕士论文稿；
- 新建精简期刊稿；
- 完成 novelty 检索；
- 完成 artifact、LICENSE、README 和 exact input；
- 统一参考文献；
- 增加 submission declarations。

---

# 10. 投稿准备度判断

## 当前状态

**Not submission-ready。**

主要原因：

- C2 尚缺 semantic-equivalence 和 EOS/EOT 多轮正确性证据；
- C-E1/C-E2 的 metric label、crossed estimand 和 implementation-path 含义尚未完成正式修正；
- 题名和主线仍可能超过软件游标 prototype 的实证范围；
- novelty、artifact、期刊压缩稿和投稿声明尚未闭环。

## 达到窄定位复审状态的条件

1. 完成 crossed `analysis_v2`；
2. 修正 compute-ready/deliverability 口径；
3. 将 C-E1 降为实现路径比较，或提供 token-equivalent 定向重跑；
4. 完成 crop semantic-equivalence 与 EOS/EOT 多轮验证；
5. 修正 E3 的 detector-conditioned estimand 和 weighting；
6. 以 C2 为核心重构全文贡献层级；
7. 完成 novelty 检索、artifact、README、LICENSE、processed input、参考文献和声明；
8. 如果不做真实音频或人工/HCI 数据，删除对应的广义主张。

## 分层判断

- **硕士论文**：主体完整，实验审计和负结果披露成熟；但核心实现仍需修复 EOS/EOT 并补语义等价验证。
- **SCI Q4 系统/应用型路线**：完成本报告的无条件修订后，具备合理投稿基础；真实音频闭环不是窄定位硬门槛。
- **SCI Q3 语音交互路线**：建议增加真实异步音频闭环；若声称用户一致性、自然度或信任改善，还需可识别语义实验或人工评测。

---

# 11. 最终结论

本稿最值得保留的研究贡献不是“已经证明播放感知显著改善真实对话”，而是：

> **公开、可审计的软件播放游标—TTS 片段—token—KV 跨层状态契约，以及该契约的实现、状态不变式、失败边界和受控性能测量。**

当前不建议再次无条件重跑原有 5000 条 C-E1/C-E2 campaign。建议：

- 使用现有 raw records 完成交叉统计和事件口径修正；
- 必须补做小规模 Qwen2-7B 的 crop/re-prefill 语义等价和 EOS/EOT 多轮正确性验证；
- 若不做真实音频闭环，则将论文稳定收窄为 fragment-level software-cursor runtime/prototype；
- 若希望争取更强的 Q3 语音交互定位，再增加真实异步音频和可识别语义/HCI 证据。

**最终编辑决定：Major Revision，修订后重新评审。**
