# 实验完成交接文档（EXP_HANDOFF）—— 供论文写作会话接手

> 面向接手**论文编写**的下一个会话。二期全部正式实验已完成，本文档给出：实验设置、
> 全部结果数字、每个数字进论文哪一章哪张图、以及**写作时必须遵守的红线**。
> 设计/决策背景按需查：`paper2/outline.md`（八章大纲+写作顺序）、`paper2/experiment_design.md`
>（指标定义 §3 / 实验规格 §5 / §9' 状态表）、`docs/decisions.md`（D-001~D-012）、
> `docs/paper2_context.md`（§九 完整时间线）、`docs/research_novelty_check.md`（prior-art）。
> 已有论文产物：`paper2/outline.md`（定稿大纲）+ `paper2/chapter2_related_work.md`（第二章初稿）。

**生成**：2026-07-17。**分支** `paper2`。**结果文件**全部在 `experiments/results/`（已入库）。

---

## 一、实验设置（第六章"实验设置"节的事实来源）

- **硬件**：实验机 RTX 3090 24GB ×2（卡0=主 LLM；卡1=TEN/重写/裁判）；验证机 5070 Ti 16GB（0.5B 开发验证，不出论文数字）
- **模型**：主 LLM **Qwen2-7B-Instruct**（与一期实验对齐）；软触发 **TEN Turn Detection 7B**（实测标定 AUC=1.00，阈值扫描点由 `calibrate_trigger.py` 产出，见 `trigger_calibration.json`）；重写 **Qwen3-0.6B**；裁判 **Mistral-7B-Instruct-v0.3**（非 Qwen 家族）；TTS **CosyVoice2-0.5B**（真机 profile：3175 samples/char@24kHz、首块 2433.6ms@3090、RTF 0.513——论文注明官方 A100+TRT 可至 45ms，引官方数字作对照）
- **数据**：MultiWOZ 2.1 派生（`prepare_multiwoz_data.py`，seed=42）：103 条 turns（≥3 user 轮）+ 100 条 segments（子句切分、严格无损）；英文（D-007 P4）
- **打断注入**（P1/P2）：确定性程序注入，播放比例 {25%,50%,75%} + boundary（片段边界对照）
- **被测条件**：A（非流式基线）/ B-ours（playback 截断）/ B-gen（generation 截断）；历史策略 naive/mark/rewrite。**B-syn 未实现区分（Mock 同步合成下与 B-gen 等价），论文不得称"已验证 B-syn"**

## 二、全部结果数字（→ 论文映射）

### E3 多轮一致性（核心实验 → 第六章主结果表 + 论文核心论点）
文件 `exp3_consistency.json`（含 summary）/ `exp3_consistency_judged.json`。n=412/条件。

| 指标 | B-ours(playback) | B-gen(generation) |
|---|---|---|
| 未听引用率 loose（规则版，片段级） | **0.0%**（构造性保证） | **51.0%** |
| 未听引用率 strict（规则版，含片段尾部量化误差） | 51.0% | 73.3% |
| 未听引用率 loose（**Mistral judge**） | **0.0%** | **2.7%** |
| 未听引用率 strict（judge） | 2.4% | 2.9% |
| 平均未听 token 数 | 0.0 | 10.7 |

边界语义自检：playback×boundary 的 strict_chars **103/103 全为 0**（截断语义在真实数据上精确成立，可写进方法验证）。

### E2 推测浪费率-TTFT 曲线（论文核心图）
文件 `exp2_tradeoff.json`（curve 九点；records=A3 数据）。TEN 阈值扫描：

| 阈值 | 0.0052 | 0.198 | 0.391 | 0.583 | 0.776 | 0.85 | 0.92 | 0.9688 | 1.1(不推测) |
|---|---|---|---|---|---|---|---|---|---|
| 浪费率 | 29.2% | 16.2% | 14.4% | 13.6% | 11.5% | 10.7% | 4.5% | 0.8% | 0% |
| TTFT_eff(ms) | 0.5 | 0.1 | 0.6 | 1.1 | 3.9 | 5.8 | 12.1 | 29.3 | 48.5 |
| 推测存活率 | 100% | 100% | 99% | 98% | 92% | 88% | 75% | 39% | 0% |

### E1 端到端延迟
文件 `exp1_latency.json`。n=100，spec_threshold=0.3906：A TTFT **27.4ms** vs B TTFT_eff **0.6ms**（改善 97.9%）；建模 mouth-to-ear **9080 vs 2482ms**（B 被 3090 的 TTS 首块 2434ms 主导——写作时必须说明是建模值+3090 实测 profile，并讨论 A100+TRT 下 B 可至 ~100ms 而 A 随回复长度线性增长）。

### A1 KV 复用 vs 重新 prefill（7B）
文件 `exp_a1_kvreuse.json`。**barge-in 响应关键路径（反查+crop）恒 0.31-0.34ms、与上下文长度无关**；role 重建 25-47ms（非关键路径，可延迟到下轮输入前）；re-prefill：275→72ms / 1k→235 / 2k→459 / 4k→891 / 8k→**1863ms**；加速比 **2.8x→39.7x**。→ 第六章折线图（对数轴）。

### A2 历史处理三策略
文件 `exp_a2_history(_judged).json`。n=100/策略。judge 连贯性：**naive 3.76 / rewrite 3.62 / mark 3.29**；重写延迟 mean 639ms / P50 679 / **P90 937** / max 1165ms（"P90<1s 可隐藏于用户说话期"成立，max 如实报告）。
**诚实结果**：重写未优于朴素截断（0.6B 重写 + 7B 主模型下）——论文如实报告并讨论（可能原因：7B 对半句历史已鲁棒；0.6B 重写质量有限；Mistral 单点评分敏感度）。贡献3 按 D-005 本就是"可砍缓冲"，据此在论文中收缩为"探索性结果"。

## 三、写作红线（违反会被审稿人/答辩抓）

1. **E3 的 loose 0% 是构造性保证，不是实验发现**（D-012）：表述为"机制保证历史不含未听片段；实验量化的是 B-gen 的失败率与严格 GT 下的片段粒度量化误差"
2. **规则检测器=表面重叠上界，judge=特定引用下界，真值被夹在中间**：两者 κ≈0.05（fixture 上 0.71）——原因是 MultiWOZ 领域常用词使规则 cue 过触发。这是**方法学发现**，第六章单独一段 + 第七章 threats；**两把尺子下 B-ours 均不劣、loose 双双为 0** 是安全结论
3. **「对话历史=用户听到的内容」不是本文首创**（D-006 护栏）：intro/related work 必须引用 OpenAI Realtime / Azure Voice Live / LiveKit 为 prior art；本文定位=首个开源可复现级联实现+显式 KV 机制+可量化对比（第二章初稿已按此写好）
4. mouth-to-ear 是**建模值**（profile 为 3090 实测）；TTFT_eff/TTFT_text/barge-in 响应为真实测量——图表标注区分
5. B-syn 不得称已验证（见 §一）
6. TEN 实为 **7.6B**（D-011，非早期记录的 0.5B）——第五章部署表写对

## 四、待办与其影响

| 待办 | 谁 | 影响 |
|---|---|---|
| **人工校验 37 条**（`experiments/results/e3_human_validation_sample.md`，loose 13+strict 24，填"人判"Y/N） | 用户（~半小时） | **不阻塞写作**：只影响第六章 E3 小节 2-3 句 + 一张 κ 小表（κ_人-规则 / κ_人-judge，填完后写小脚本计算即可）+ 第七章一句 |
| 实验机可能有未推提交（E2 加密数据；push 需交互凭据） | 用户在实验机 `git push` | 若本机 `exp2_tradeoff.json` 的 curve 已是九点则已推齐（写作前 `git pull` 核对） |
| 可选：E1 补保守阈值(0.92)第二工作点；真实音频 ASR 链路 | 实验机，非必需 | 不影响主线 |

## 五、画图（本机普通程序即可）

matplotlib/seaborn 已在依赖里。建议图表：① E2 九点 trade-off 曲线（浪费率 vs TTFT_eff，标注拐点 0.85-0.97，论文核心图）② A1 双线图（crop 常数 vs re-prefill 线性，log-y）③ E3 主结果表（上方表格）+ 分打断比例分解（records 里有 fraction 字段）④ A2 策略对比条形图。绘图脚本建议放 `experiments/scripts/plot_*.py`，图入 `paper2/figures/`。

## 六、写作路线（大纲已定稿于 `paper2/outline.md`，附篇幅与顺序建议）

已完成：第二章初稿（related work，含差异表）。建议顺序：**第三章问题形式化**（指标定义与 `experiment_design.md` §3 咬合，全部有实测语义）→ **第六章**（数据全在本文档+JSON，趁热）→ 第四/五章（方法/实现：素材=已验证代码，关键文件 `src/dialogue/timeline.py`、`src/llm/stream_llm_inference.py` 的 AccumKVCache/crop_to_token/role重建、`src/dialogue/orchestrator.py` 推测状态机）→ 第一/七/八章统稿。语言/格式参照一期 `paper/` 与 `paper2/outline.md` 附注。

## 七、建议 skills

- **`code-review`**：写绘图脚本/κ 计算脚本后顺手审
- **`experiment-agent`（validate 模式）**：写第六章前对结果 JSON 做统计解读与 11 类统计谬误扫描（显著性检验：E3 loose 0/412 vs 210/412 建议报 χ² 或 Fisher + 效应量；配对样本注意同对话跨条件配对）
- 绘图与统计均为本机普通程序，无需实验机
