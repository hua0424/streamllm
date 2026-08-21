# 开发侧回复函 r2（对应 review-reply-dev-assessment-and-plan-20260821-r2.md）

- 日期：2026-08-21
- 方案版本：`dev-assessment-and-plan-20260821.md` v3（v3 修订节已落实 r2 §3–§4 全部细则）

## 一、核实结论：r2 对现有代码的断言全部属实，另有一项重要新发现

| r2 断言 | 核实结果 |
|---|---|
| 旧脚本先 put 整块 500ms chunk 再 sleep（§3.2） | 属实：`run_exp_latency.py:638-642`，`queue.put()` 在前、`time.sleep()` 在后，消费者在 chunk 末样本物理到达前即可见整块 |
| 旧 System A 直接处理内存音频、无实时回放（§3.3） | 属实：非流式路径以 `audio_load_time` 起算，无逐 chunk 回放 |
| 旧 TTS 客户端 `iter_content(chunk_size=16000)`≈363ms（§3.7） | 属实：`measure_tts_first_chunk.py:65,96`；1324-byte  playable 阈值计算复核无误 |
| 生成器只 yield 文本、EOS/截断不可区分（§3.8） | 属实：`stream_llm_inference.py:215-219`，EOS 时 yield 的是 `skip_special_tokens=True` 解码的空串 |
| **repetition_penalty 是否真正生效**（§3.8 要求核实） | **不生效**：`_decode_logits()`（`stream_llm_inference.py:308`）接收该参数但函数体内从未使用，仅有 temperature + top_p + multinomial |

**repetition_penalty 发现的影响与处置**：E1–E6/R2–R6 全部历史生成文本的实际采样配置为 temperature=0.1、top_p=0.9、**无重复惩罚**。处置（v3 修订节第 0 条）：W1 保持现有采样行为不变（与历史数据同引擎同行为、A/B 同码自洽），RUNINFO 记录请求参数与实际生效参数；论文方法部分的采样参数表述更正纳入 W8 残留搜索清单；修复死参数不在本轮范围。

## 二、对 r2 各项要求的接纳确认

1. **§3.1–3.10 全部接纳**，已逐条落实为 v3 修订节第 1 条的实现规范（physical speech end 算法全参数、因果回放、A 同时间轴、事件三枚举拆分、组件闭合公式、句末检测复用已审查实现、PCM 512-byte 粒度与 1324-byte 阈值、生成元信息接口、轮次键定义、fail-closed 工程）。其中两个自选参数按 r2 授权范围内确定为：
   - energy 与 Silero 差 ≤200ms 时取 **energy** 值，>200ms 取 Silero 并标记 `pse_method`；
   - 子集三轮定义采用 **repeat_idx=0 主实验计入三轮、子集仅补 1/2 轮**（节省一轮 GPU），交替顺序 AB/BA/AB 或 BA/AB/BA，schedule 预生成存 hash。
2. **§4.1**：W3 验收锚点（B 5.19/4.05/18.96、A 5.23/4.65/14.01）与本机两轮独立重算一致，按"偏差先查因不覆盖"执行。
3. **§4.2**：W4 覆盖清单按五类来源建立（R2 干净/增强、Table VII、R4 外部一致性、QA 与总册残留），逐数字标注口径。
4. **§4.3**：W5 最小比较族按四组冻结（R2 十二条件 Holm 族、Table VII B-vs-LA 主比较、Table III 分组配对差、R5 仅报 CI），SciPy 版本/Wilcoxon correction/rank-biserial 符号方向/全零差处理在脚本常量中固定。
5. **§5 Gate 1 / §6 结果级 QA**：接受两级门禁。下一步直接产出 W1 代码 + schema + self-test 供脚本级审查，不再扩写总体方案；Gate 1 通过后出 GPU handoff（冒烟 3 条 → 正式）。

## 三、流程确认

r2 的分阶段裁决（放行实现、暂缓正式 GPU 与论文放行）即为当前执行基线。W3/W4/W5 本机脚本与 W1 实现并行开展；W6–W9 文档项在结果级 QA 前完成。
