# 二期工程决策日志

每次做技术决策时按时间倒序追加一条。每条包含：**日期 / 决策 / 背景 / 理由 / 影响 / 状态**。
状态：`proposed` / `accepted` / `superseded`（被后续决策替代时填写替代条目的日期与编号）。

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
