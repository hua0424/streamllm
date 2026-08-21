# 论文数据就绪度审查回复（2026-08-21，对应 review-20260821-PAPER-DATA.md）

> 审查报告：`experiments/review/20260821-PAPER-DATA/review-20260821-PAPER-DATA.md`
> 回复范围：P0×1、P1×3、P2×8、INFO 落实清单。**审查结论全部属实，无驳回项**。

## 总体回应

我们对审查的每个关键数字做了独立重算，全部吻合：

- P0：E5 streaming 行重算 endpoint_wait=53.1ms、final_enqueue−speech_end=2052.6ms、
  first_token−speech_end=3065.1ms、first_token−final_enqueue=**1012.5ms**，A 行 ttft=3928.9ms——
  "B 行含 ~2.0s 装置等待、A 行不含"的不对称属实；PAPER_HANDOFF"不进预算表"与旧装配结果的矛盾属实。
- P1-2：Table IV 重算值与 `table4_ablation_percentiles.csv` 逐格一致（含 Extra Long KV 增益 40.82ms）。
- P1-3：LA 2115.0 vs B 1573.9 → LA 比 B 高 34.4%、B 比 LA 低 25.6%，口径歧义属实。

## P0（Table VIII 装置等待不对称）—— 需求方已裁决：方案2（对称剔除 2s），已执行

- `assemble_ttfa_budget.py` 修正：B 行 `t_post_endpoint` 由 `first_token − speech_end − endpoint_wait`
  改为 **`first_token − final_is_final_segment_enqueue_time`**（1012.5ms），
  剔除实时喂追加静音的 ~2.0s 装置等待；A 行不变（其 ttft 从 audio_end 起算本就不含）。
  self-test 5/5 通过（含新口径断言）。
- 重装配结果：**System B ALL 14.38s（zh 15.21 / en 13.54）；System A ALL 22.67s**（差距 8.3s）。
  产物 `r6_ttfa/ttfa_budget.csv` 已更新。
- 三处文档已对齐到同一口径：
  1. `PAPER_HANDOFF.md` §TTFA 公式行改写为方案2 公式，并注明剔除规则（"2s 不进预算表"表述自此成立）；
  2. `r6_ttfa/RUNINFO.md` 移交说明的旧 E4-TTFT 链条公式已替换并标注"以此为准"；
  3. `REVISION_CHANGELOG.md` 已登记裁决与重装配数字。
- 论文写作要求（落笔时执行）：正文须写明"B 行 post 分项为 final 段入队→首 token 的真实处理，
  2s 静音窗为测量装置属性已对称剔除"。

## P1 处置

- **P1-1（E4 漂移表述过轻）**：已改。PAPER_HANDOFF E4 条目更新为实测分布口径
  （224 次 / 49~50 样本、涉及段 52.7%、编辑距离 mean 2.3 / p90 6 / max 16、归一化 max 47.1%、
  含实词级漂移、下游不可见），删除"同音字/标点级"。论文 §IV 新小节将按此写。
- **P1-2（Table IV 498 重算连锁）**：接受，列入改稿清单：Table IV 全表替换为 498 口径
  （108/150/240），§IV-B 的 KV 增益叙述重写（Extra Long 占比 9.3%→**3.3%**、Very Long +16.96→**+2.73**
  几乎归零、"收益愈加显著"递进叙述删除）、"1.1s 左右"改为"约 1.1–1.2s"或分组值、样本量表述改为
  "498 条成对干净子集（排除规则：运行错误 + 流式 TTFT>10s 挂起）"，并全文检索旧数字残留（含摘要/结论）。
- **P1-3（LA 优势口径）**：已统一为"**LA-2 基线比 System B 慢约 34%**"（等价："B 比 LA 低约 26%"），
  PAPER_HANDOFF 已改；论文只用其中一种表述，不混用。

## P2 确认（写作时照单执行）

1. "最长分组平均减少"改为 **5.66 秒**（6745.57−1087.70）——确认；
2. zh CER 口径脚注——已补入 PAPER_HANDOFF E2 条目（49/75 数字写法失配，受影响 0.1476/未受影响 0.0313）；
3. Table VI 逐条件行用 `overall` 行或注明变速后重判分组（speed11 出现 medium 子组）——接受；
4. R5 以轨道 A+B2 为主证据、B1 如实归因、披露 judge 型号与随机化——接受（changelog 已先行登记同口径建议）；
5. decode 分语种差异按"首句 token 数、速率相同"归因——接受；
6. E1 CV 只报 mean/median（4.2%/3.3%）——接受（p90 分位约定差异不引用）；
7. "near-zero error"限定合成集，真实上界 librispeech 2.97% / aishell1 10.8%（含数字写法因素）——接受；
8. babble limitations 按完整归因链写（VAD 过度触发→段积压+空输出；长尾 median 2020 vs mean 3425）——接受。

## 澄清说明

1. **A 行 decode/TTFC 为估计项**并非数据缺失的折中，而是有意口径：decode 项与输入提示内容无关
  （同模型同速率，B 实测代理）、TTFC 项用 E6 实测的 TTFC-长度关系（~0.09s/字符）× A 回复均长；
  CSV `source` 列已逐行标注"全实测/估计"，论文表格将保留该标注。
2. **B2 独立盲评是 2026-08-21 增补轨**，用于分离 B1 成对等价中的采样发散；不改变计划既定的
  双轨结构，B1 仍按计划口径报告。
3. P0 选择方案2 而非方案3（E4 链条）的理由：方案2 两系统取自同一次 E5 运行、同为装置外口径，
  不跨运行混合；E4 的 audio_end 另含数据集自带尾隙。GPU RUNINFO 的旧公式为装配前的移交草案，
  已按裁决更新。

整改均已完成并入库（组装脚本修正 + 预算表重装配 + 三处文档对齐 + changelog 登记）。
请审查人员复核；无异议后进入论文修改阶段（R7 清单执行）。
