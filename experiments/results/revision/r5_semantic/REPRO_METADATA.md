# R5 语义评测复现元数据（W6，2026-08-21）

对应 PRE-PAPER-AUDIT P1-4 / v3.1 方案。结论定位：**50 条 Very Long 合成样本上的探索性证据**，不建立等价性。

## 样本与输入

- 样本范围：E4 同 50 条 Very Long 合成样本（25 crosswoz zh + 25 multiwoz en），非真人语音、非噪声条件；
- 被评回复来源：`r4_commit/exp1_results_20260820_171522.json` 的 `full_response`（A=non-streaming / B=streaming）；
- 输入与回复逐样本哈希：见 `judge/`、`judge_solo/` 逐样本 JSON（每样本一份，含原始响应）。

## 两组参数（分开记录）

| 角色 | 参数 |
|---|---|
| 被评 A/B 回复（E4 复跑生成） | temperature=0.1、top_p=0.9、max_tokens=128（capped）、`requested_repetition_penalty=1.1` / **`effective_repetition_penalty=not_applied`**（`_decode_logits` 死参数，2026-08-21 核实） |
| judge | temperature=0.0；max_tokens=2048（初轮 1024 有 3 条 reasoning 烧预算失败，提额重试后成功）；请求经兼容 OpenAI 的 chat/completions 接口 |

## 轨道 A：BGE-M3 嵌入

- 模型：BAAI/bge-m3，本地目录 `C:\Users\hua\.cache\models\bge-m3`（本机直接下载，非 HF git snapshot，**snapshot commit 不可回溯 → unknown**）；
- 权重完整性（本机文件 SHA-256）：
  - `model.safetensors`（本机由 `pytorch_model.bin` 转换，torch 2.5.1 CVE-2025-32434 拒载 .bin 所致）: `993b2248881724788dcab8c644a91dfd63584b6e5604ff2037cb5541e1e38e7e`；
  - `config.json`: `26159e7ad0650734…`、`tokenizer_config.json`: `a62b2b6784f99025…`、`special_tokens_map.json`: `8c785abebea9ae32…`、`sentencepiece.bpe.model`（5,069,051 bytes）、`tokenizer.json`（17,098,108 bytes）（后两项大文件未哈希）；
- 打分方式：CLS 向量 L2 归一化余弦（`semantic_consistency.py:track_a`）；
- tokenizer revision：随上述文件固定，HF revision 号 **unknown**。

## 轨道 B / B2：LLM judge

- judge 模型 ID：`deepseek/deepseek-v4-flash`；供应商/base URL：`https://api.commandcode.ai/provider/v1`（OpenAI 兼容网关）；
- 服务版本/部署 revision：**unknown**（网关不暴露，不事后推断）；
- prompt 哈希：
  - 成对等价 `JUDGE_PROMPT` sha256: `fd21011473b1507dd09f93fa9cf982671757a2669dd330156f3c7e0d8a4c0c17`；
  - 独立盲评 `JUDGE_PROMPT_SOLO` sha256: `0f6318b75f9995cb4debe1482de8b289033f315eddaf455e27f520c81a4d5961`；
- 成对顺序随机化：`sum(ord(c)) % 2` 按 sample_id 确定性决定甲乙顺序；
- 调用时间：2026-08-21（本机执行）；调用顺序：track B 后 track B2，按样本顺序；
- 失败/重试：3 条因 reasoning 预算耗尽（finish_reason=length）失败，max_tokens 1024→2048 后重试成功；错误记录不缓存、可重试；
- API key 经环境变量传入，未落盘。

## 结果（引用口径）

- 轨道 A cosine：mean 0.8832（n=50）；
- 轨道 B 成对 judge：mean 2.96/5，≥4 占 40.0%（受两臂独立采样 + 128 token 截断影响，如实归因）；
- 轨道 B2 独立盲评：A 3.10 vs B 3.04，B−A 配对均值差 −0.06，bootstrap 95% CI [−0.34, 0.22]（`stats_inference/paired_inference.csv`）；
- **不得写"统计不可区分/等价/无损"**；单裁判、无重复评分、无多裁判一致性。
