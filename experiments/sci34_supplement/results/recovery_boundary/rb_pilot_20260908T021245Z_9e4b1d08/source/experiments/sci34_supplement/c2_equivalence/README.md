# C2 crop / clean-prefill 等价性 campaign（协议 v2）

独立、确定性的 Qwen2-7B-Instruct BF16 正确性验收。它执行：

- 独立 termination probe：每个 case 真实调用 `generate_accumulating`；natural EOS/max-token 走真实 greedy（natural cap 256，未命中者确定性重资格化，campaign 级要求 ≥5/10 genuine），EOS-at-cap 明确使用受控 token-selection fixture 并走 production EOS 状态逻辑；
- 路径 A：为等价比较 teacher-force 冻结 assistant IDs，按规定片段边界 crop，再通过角色 API 恢复；
- 路径 B：把同一批 retained assistant token IDs 原样嵌入规范 chat token 序列，从空 cache clean re-prefill；
- v2 噪声对照臂：同一 canonical 序列按结构 seam 分块增量 prefill（不含 crop 代码），测得本环境固有 BF16 增量噪声；crop 路径偏差以 2× 相对门槛验收，top-1 翻转/continuation 发散仅在近并列 margin 内允许（D-019；v1 绝对阈已被证明对任何正确实现不可达成，v1 run `c2eq_563dd22a_*` 归档保留）。

正式设计固定为 24 cases、1 session、无统计重复。任一 case 或 checkpoint 失败都保留工件并使 validation/analysis/acceptance 失败。

## 文件导航

- `EXPERIMENT_PLAN.md`：研究问题、case 覆盖、状态语义、硬门槛与主张边界。
- `GPU_HANDOFF.md`：实验机唯一操作入口；命令对应当前真实 CLI。
- `ACCEPTANCE_TEMPLATE.md`：GPU 返回后的人工设计验收模板。
- `protocol.py` / `cases.json`：冻结协议与 24-case 规格。
- `canonical_chat.py`：`apply_chat_template(tokenize=True)` 结构抽取与 token-ID canonical builder。
- `runtime.py`：Transformers 两路径执行及纯 CPU fake backend。
- `campaign.py` / `run.py`：不可变 manifest、单 session、case 原子 resume。
- `validate.py` / `analyze.py` / `seal.py`：独立验证、描述性汇总、SHA-256 封存。
- `smoke.py`：不加载模型、不联网的完整 fake workflow。

本机验证入口：

```bash
uv run python -m experiments.sci34_supplement.c2_equivalence.smoke
```

正式命令只从 `GPU_HANDOFF.md` 复制。Pilot 和 formal 必须使用不同 run ID 与目录。
