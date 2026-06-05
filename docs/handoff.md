# 二期工作交接文档（HANDOFF）

> 面向接手的下一个 agent。本文档不重复 `paper2_context.md` / `decisions.md` 已有内容，只给定位、当前断点、下一步方向。先读那两份文档再读本文。

**生成时间**：2026-05-21
**分支**：`bargeincache`（二期专用，与一期 `main` 隔离）
**当前阶段**：纯设计/规划，**本机不运行代码**（venv 已损坏，Python 路径失效；真要跑实验在 5070 Ti 验证机 / 3090×2 实验机上）

---

## 一、必读材料（按顺序）

| 文件 | 作用 |
|---|---|
| `docs/paper2_context.md` | **主交接文档**。二期方向、pipeline、技术选型、Q1-Q8 一期审查结论、§九里程碑日志 |
| `docs/decisions.md` | 技术决策日志 D-001~D-004。每次新决策**倒序追加一条** |
| `paper/thesis_outline.md` | 一期论文大纲（二期暂未单独建 outline） |
| `src/llm/stream_llm_inference.py` | 一期 KV prefill 引擎，二期改造主战场 |
| `src/asr/faster_whisper_streamer.py` | 一期流式 ASR，final 片段粒度衔接 LLM |
| `src/run_test_simple.py` | 一期四线程流水线编排，二期 orchestrator 的参照 |

---

## 二、上一轮（2026-05-21）完成了什么

1. 基于一期源码逐条回答了 `paper2_context.md` §六 **Q1-Q8**，结论内联写回该文档每个 Q 的"**答**"块。最关键三条：
   - **Q3**：一期 chat template 是**手工字符串拼接** `generation_prompt`，二期 role 重建可沿用同一模式（构造 `<|im_end|>\n<|im_start|>user\n` 走 `_add_stream_prompt` 注入）
   - **Q4/Q5**：一期 `generate()` **没把 assistant token 的 KV 写回 caller**，也没做多轮累积 —— 二期必须新建"边生成边累积可被 crop 的 assistant-side KVCache"
   - **Q2**：一期 prefill 已显式管理 `attention_mask` + `position_ids`，crop 后只要同步截 `pre_attention_mask`、用新 past_length 重算 position_ids 即可复用
2. 敲定技术选型（D-002~D-004）：软触发 = **TEN Turn Detection**（文本侧，与 KV prefill 并行零额外耗时）、重写 = **Qwen3-0.6B**、KV = **DynamicCache.crop**、硬件分卡布局（§3.6）
3. 用户确认从 **方向 1（反向映射表数据结构设计）** 开始下一轮讨论 —— **这是本次交接的断点，尚未开始**

---

## 三、下一步方向（接手后从这里继续）

### 主任务：反向映射表（Reverse Mapping Table）数据结构设计

这是论文工程贡献的**核心依赖项**，后续 KV 截断、推测浪费率统计、播放感知截断全部建立在它之上。目标是设计清楚四向映射：

```
LLM token_idx  ↔  text fragment_id  ↔  audio chunk_id  ↔  playback_ms
（generate产出）  （stream2sentence）   （CosyVoice2产出）  （播放器回报）
```

讨论需要覆盖（建议作为 checklist 推进，**纯设计推演，不写实现代码**）：

1. **每一层的产出时机与标识**
   - LLM token：`generate()` 循环里每 yield 一个 token 拿到全局递增 idx
   - fragment：stream2sentence 吐出一个 chunk 时，需记录它**起止覆盖的 token_idx 区间** `[start, end)` —— stream2sentence 本身不给 token 索引，要在喂给它的 generator 里自己计数对齐（**关键工程难点，要想清楚怎么不丢字符地对齐**）
   - audio chunk：CosyVoice2 流式输出每个 chunk，记录其源 fragment_id
   - playback_ms：播放器按已播放采样数 / 采样率回报

2. **反向查询语义**：给定"当前实际播放到 T 毫秒" → 落在哪个 audio chunk → 哪个 fragment → 该 fragment 末尾对应的 token_idx N → `DynamicCache.crop(N)`

3. **截断单位 = fragment 边界**（已在 §八否决"精确到单 token"）。物理 crop 到 fragment 末 token

4. **"已合成未播放" buffer 的处理**：CosyVoice2 可能已合成但播放器还没播的部分，被打断时要丢弃 —— 映射表要能区分"已播放 fragment"与"已合成未播放 fragment"

5. **并发边界**：generate 线程写 token↔fragment，TTS 线程写 fragment↔chunk，播放线程写 chunk↔playback，打断时主线程读全表做 crop。锁粒度怎么设计，避免打断响应被锁阻塞

6. **数据结构落点**：建议落在 `src/dialogue/timeline.py`（见 Q8 工作量表），但**本轮只产出设计，不写代码**

### 推进方式建议

- 用户偏好：**先讨论清楚设计再动手**，每个技术决策记入 `docs/decisions.md`（倒序追加 D-005...）
- 里程碑结束后**同步更新 `paper2_context.md` §九时间线**
- 讨论可以用文字 + ASCII 时序图，不需要跑代码（本机无运行环境）

### 方向 1 之后的排队议题（用户已认可的顺序 1→2→3）

2. **role 边界 KV 重建的精确字符串构造**：从 Qwen2.5/Qwen3 chat_template 反推 assistant→user 切换字符串，验证 position_ids 接续正确性（纯文本推演可确认）
3. **推测生成长度上限策略**：限前 N token / 第一句 / soft+hard cap，决定论文核心 trade-off 曲线横轴范围

---

## 四、注意事项 / 坑

- **不要在本机跑任何 Python/实验**：venv 损坏（uv-managed Python 路径失效），且本机定位为写作机。需要运行时明确告诉用户去验证机
- **不要走 §八已否决的方向**：SNN、端点检测模型创新、打断类型分类、完整长回复推测、TADA 作主 TTS 等
- **软触发/重写不是论文贡献**：选定模型即可，不做选型消融，别在这上面花实验精力
- **保持论文 framing 锚点**：「对话历史 = 用户实际听到的内容」是所有 KV 去留判断的根原则
- 一期 ASR final 片段间隔 ~1s（`recognition_threshold=1.0`），可能影响软触发响应度，方向 1 讨论时留意这个粒度约束

---

## 五、建议调用的 skills

- **`superpowers:brainstorming`** —— 方向 1 是开放式设计探讨（数据结构 + 并发模型），动手定方案前应先用它厘清需求与取舍。**这是接手后第一个该调用的 skill**
- **`superpowers:writing-plans`** —— 当反向映射表设计收敛、准备转入 `src/dialogue/timeline.py` 实现时（届时需在验证机环境），用它把设计写成可执行 plan
- **CodeGraph（`codegraph_*`）** —— 需要再查一期符号关系（如 `cache_prompt` / `_add_stream_prompt` / `generate` 的调用链与影响面）时优先用，比 grep 快且准
- **`update-config`** —— 若需修复 venv / 配置运行环境时（仅验证机场景）

---

## 六、关键路径速查

- 二期主文档：`docs/paper2_context.md`
- 决策日志：`docs/decisions.md`
- KV 引擎（改造核心）：`src/llm/stream_llm_inference.py:195-306`（generate + _add_stream_prompt）
- chat template 处理：`src/llm/stream_llm_inference.py:124-193`
- 流水线编排参照：`src/run_test_simple.py:288-620`
