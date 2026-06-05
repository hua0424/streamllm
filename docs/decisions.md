# 二期工程决策日志

每次做技术决策时按时间倒序追加一条。每条包含：**日期 / 决策 / 背景 / 理由 / 影响 / 状态**。
状态：`proposed` / `accepted` / `superseded`（被后续决策替代时填写替代条目的日期与编号）。

---

## D-004（2026-05-21）重写模型选型

**决策**：使用 **Qwen3-0.6B** 作为对话历史自然化重写模型。

**背景**：贡献 3 的"重写法"分支在截断位置语义不完整时启用，并行运行隐藏延迟。输入 ~50 token，输出 ~60 token。

**理由**：
- Qwen3 系列 2025 年发布，比 Qwen2.5 更新，中文质量好
- 0.6B 规模在 3090 上推理 200-300ms，并行隐藏在用户说话期内
- 与主 LLM 同家族（Qwen 系），但**实例独立部署**，符合多服务工程现实
- 不在论文贡献范围内，不做模型选型消融

**影响**：
- `src/dialogue/rewriter.py` 加载 Qwen3-0.6B-Instruct 实例
- 实验机分卡布局：与软触发 + CosyVoice 共驻卡 1

**状态**：accepted

---

## D-003（2026-05-21）软触发模型选型

**决策**：使用 **TEN Turn Detection**（基于 Qwen 0.5B 微调的文本侧端点检测器，Apache 2.0）作为软触发主选；**不做候选模型消融**，软触发不是论文贡献。

**两阈值机制**：模型输出连续置信度，配两个阈值
- **推测阈值**（激进）：超过即触发主 LLM decode 进入"推测生成"
- **提交阈值**（保守）：超过才允许 TTS 开始播放给用户

调整两阈值得到"推测浪费率 vs TTFT"trade-off 曲线（论文核心图之一，paper2_context.md §五）。

**背景**：候选过 Smart-Turn v2（音频侧，~20ms）、TEN（文本侧，50-100ms）、Phoenix-VAD（权重发布不确定）、Qwen prompted（最灵活但慢）。

**理由**：
- 文本侧检测的推理时间**与 KV prefill 并行**，挂在 prefill 的延迟阴影里，**实际零额外成本** —— 这是关键架构观察
- 文本侧错误更易复查与调优（端点判断错时可以打印当前累积文本看原因）
- 中英文双优，Apache 2.0，知名度足，论文里讲故事无争议
- 软触发不是论文贡献，**不需要做模型选型消融实验**

**影响**：
- `src/dialogue/trigger.py` 加载 TEN Turn Detection 实例（卡 1）
- 软触发输入是 ASR final 片段累积文本，触发判断与 LLM `_add_stream_prompt` 并行
- 论文中作为辅助模块描述，**不展开多模型对比**

**状态**：accepted

---

## D-002（2026-05-21）硬件配置、分支、主 LLM 规模策略、模型独立部署

**决策**：
1. **二期工作分支**：`bargeincache`（已切，不污染一期 main）
2. **验证机**：5070 Ti 16GB，主 LLM 用 0.5B 跑通 pipeline
3. **实验机**：3090 24GB × 2 = 48GB，主 LLM 用 7B，与一期实验对齐
4. **三个 LLM 实例完全独立部署**（主 LLM / 软触发 / 重写），不复用权重，模拟真实多服务工程

**3090×2 部署粗算（7B fp16）**：
- 卡 0：主 LLM(~14GB) + 长 KV(2-4GB) + Whisper-small(~1GB) ≈ 17-19GB
- 卡 1：CosyVoice2-0.5B(~2-3GB) + 软触发(~1-2GB) + 重写(~1-2GB) ≈ 5-7GB

**理由**：
- 与一期实验对齐，便于直接对比一期/二期数据
- 多服务独立部署反映工程真实，论文工程价值更可信
- 验证机用 0.5B 跑通，等架构 OK 再上实验机跑 7B，节省迭代时间

**影响**：
- `src/config.py` 二期需要支持**按模块**指定 device（主 LLM、ASR、TTS、trigger、rewriter 各自一项），一期目前只分了 asr_device/llm_device 两路
- 实验脚本要支持单卡（验证）/双卡（实验）两种 device map

**状态**：accepted

---

## D-001（2026-05-21）transformers KV cache 的对象类型与改造路径

**决策**：二期 KV 截断走 `DynamicCache.crop()` 路线。一期 `StreamLLMInference.KVCache` 中的 `past_key_values` 字段保持现状（"transformers 返回什么就用什么"），但二期新增的 KV 操作模块**显式断言**它是 `DynamicCache` 实例；若 transformers 实际返回 legacy tuple，则一开始就 `DynamicCache.from_legacy_cache()` 转换。

**背景**：一期 `src/llm/stream_llm_inference.py` 把 `past_key_values` 当作不透明对象在 `_init_kv_cache` / `_add_stream_prompt` / `generate` 之间传递，从未调用 cache 方法 — 无法从代码静态判断它到底是 DynamicCache 还是 legacy tuple。

**理由**：现代 transformers（4.36+）对 Qwen2.5 默认就返回 `DynamicCache`，`crop()` 自 4.39 起稳定。显式断言/转换让 KV 操作有一个稳定的契约面，二期不再被 transformers 内部默认行为牵着走。

**影响**：
- 二期新增模块（KV 截断、role 重建）依赖 `DynamicCache` API（`crop`、`__len__`、`key_cache` / `value_cache` 访问、`update`）
- 一期的 `KVCache` 数据类需要在二期版本里多带一个字段：**当前 cache 长度**（即 `past_key_values.get_seq_length()`），避免靠 `pre_attention_mask.shape[1]` 间接推断
- 风险：若实际运行的 transformers 版本不返回 DynamicCache，需在加载阶段统一转换

**状态**：accepted

---

## D-000（模板示例）

**决策**：[一句话决定了什么]
**背景**：[当时面临的问题 / 约束]
**理由**：[为什么这么选 — 与备选方案的对比]
**影响**：[改动哪些文件、引入哪些依赖、有哪些后续工作]
**状态**：proposed / accepted / superseded by D-xxx

---

> 这条 D-000 是模板，提交真实决策时删除或保留为占位。
