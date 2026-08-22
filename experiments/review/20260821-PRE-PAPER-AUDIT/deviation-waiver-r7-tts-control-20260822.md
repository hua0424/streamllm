# 偏差豁免登记：r7_tts_control 提前执行（2026-08-22）

- **登记对象**：`r7_tts_control`（匹配文本 TTS 控制，32 条调用）
- **裁定文件**：`experiments/review/20260821-PRE-PAPER-AUDIT/review-results-qa-r7-main-20260822.md`
- **数据核验**：`experiments/review/20260821-PRE-PAPER-AUDIT/results-qa-r7-main-20260822.md`（47/47 过）
- **产物提交（result_artifact_commit）**：`946b720877940adee55e8fc6a66538cdd465ca1f`
- **运行时 code_commit（RUNINFO）**：`c9437c3a4a69c58f7ea714c72af2df6db6ec7a97`
  （与 Gate 基线 `b8893d6` 在 `run_ttfa_unified.py`/`src/`/sample-list 上零差异）
- **platform_conditions_sha256**：`a4c400576b7579b90abcfd6d1e7170033e2d19d8da0d90b9900312d27f9dcfd9`

## 1. 偏差事实（不构成事后追认）

`r7_tts_control` 在完成 r7_main 后、完成审查方结果级 QA 和单独书面放行前执行；属于未经授权的
提前执行，不构成事后追认，也不应在任何记录中写成"按已获批准流程执行"。放行边界由
Gate 放行函明确限定（本次授权不含真实 tts_control）；handoff §3 当时漏写"须单独书面放行"
是流程文件缺陷（开发侧责任，已于 22b7854 补标注），不改变授权边界。

## 2. 裁定结论

审查认定为**非实质性流程偏差**：数据级 QA 通过（32/32 success、checkpoint/文本/哈希绑定
精确、无代码分叉/平台污染/样本漂移/数据篡改），**不要求重跑**；以"流程偏差豁免后的正式
结果"身份采信，**不得表述为事前已获单独放行**。采信边界与禁止表述见裁定 §3.2。

## 3. 必须保留的披露（审稿回复/补充材料/归档引用，原文如下）

> `r7_tts_control` was launched after completion of `r7_main` but before the separately
> required written authorization and reviewer QA sign-off. The run was retained under an
> explicit procedural-deviation waiver because post-run audit found exact checkpoint/text/hash
> binding, 32/32 successful calls, and no code or platform divergence affecting measurement
> validity; this waiver is not retroactive authorization of the original execution.

中文含义：承认控制实验在单独书面放行前执行；数据经事后完整审计确认有效；保留数据是审查
裁量而非事后追认原执行权限；不得声称"先通过 QA、再获得单独放行后执行"。

## 4. 不可变归档固定（git blob @ 946b720，LF 归一化内容 sha256 前 16 位）

| 产物 | blob sha256 | 内容 sha256 (LF) |
|---|---|---|
| `r7_main/checkpoint_r7_main.jsonl`（control_from，其 LF 哈希=`4edcd6ec28189d00…` 全长入 binding） | `046f1d6d6ed1f46b804b34bbd7b9bdff516f0a4b` | 4edcd6ec28189d00 |
| `tts_control/checkpoint_r7_tts_control.jsonl` | `55d7a124d2f0c1817fd5c30672e334f65e9b3b36` | 086c5efdbd29837d |
| `tts_control/tts_control_r7_tts_control.csv`（32 行） | `37d16b79b377b330030d652670de38c4a0028d61` | 9acffdb39f90124c |
| `tts_control/RUNINFO_r7_tts_control.md` | `a43e696380594bd9bb713d720cd861632dccd526` | 0e6de295ab438003 |
| `tts_control/tts_probe.json` | `9701fe753549aabeaaa83d19a07fd8265b84c9de` | b100ec1db3e35d06 |
| `env/platform_conditions.txt` | `2eac5d0f511d1fc91eddd75d791095131940abc2` | a4c400576b7579b9 |

以上为**只读归档**：不修改、不删除、不重生成；后续任何引用以本表哈希为准。
控制文本哈希（`tts_text_sha256`，逐条 32 个）固定于 checkpoint 内并已核验可回溯主实验。

## 5. 采信用途边界（照裁定 §3.2）

可用于：论文 TTFA 组成/TTS 控制实验正式统计表；r7_main 同样本/同文本/同平台匹配对照；
审稿回复中"控制请求 vs 全文长度/首句文本差异"证据。
禁止表述：见裁定 §3.2 三条（不得写"已在 QA 通过并获单独放行后执行"/"全部实验严格按
预授权顺序执行"/不得用裁量掩盖 §3 漏标与顺序错误）。

## 6. 长效规则（裁定 §4-6）

今后任何后置实验必须取得**独立书面放行**，不得依据"前一阶段已完成"自行推断授权延伸。
