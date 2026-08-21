# 开发侧回复：Gate 材料复核意见整改（对应 review-gate-material-verification-20260822.md）

- 日期：2026-08-22
- 总体：**意见全部采纳。** 非末位 fatal 机制、TTS 内容、self-test 内容均通过；正式放行
  仍被 provenance/clean/manifest 冲突阻塞。已将 Gate 流程升级为 r3 原子材料包流程，
  需 GPU 主机在**唯一 clean code_commit**上重新生成放行前材料。

## 一、为什么必须重跑 Gate（采纳 §3.1–§3.5）

审查指出的版本/clean 冲突属实，不能用已有混合材料修补放行：

- clean/self-test 在 `2e54ac2`；fatal smoke 在 `34ea12e`；当前 handoff 修订在更晚提交；
- `gate_clean_git.txt` 的 porcelain 实际列出了未跟踪 Gate 文件，不能称"空=clean"；
- fatal smoke binding 的 `platform_conditions_sha256=null`，不能证明它与 platform 文件同环境；
- manifest 当时覆盖不全，且其实际 hash 语义为 LF-normalized 内容 SHA-256，不应称 Git blob/
  原始文件 hash；
- Gate 材料产生后未先完成 artifact commit，工作树材料继续漂移。

因此不尝试局部替换/沿用旧材料；采用单一 code_commit + clean checkout + 原子材料包重建。

## 二、已整改的 handoff（Gate 版 r3）

1. **唯一基线**：Gate 前在拟正式 code_commit clean checkout 先写 `gate_clean_git.txt`，
   Gate 全部运行均用该 HEAD；
2. **clean 语义澄清**：clean 证明必须在材料生成前取得；材料生成后的未跟踪/修改是预期 artifact，
   不再误把它作为不 clean；材料提交后另写 `gate_artifact_commit.txt`（artifact commit + porcelain）；
3. **fatal 平台绑定**：§2b 命令在同一 `env/platform_conditions.txt` 下运行，脚本将其 hash 写入
   fatal checkpoint/RUNINFO；
4. **完整 manifest**：明确 `hash_scheme=LF-normalized-content-sha256`，覆盖 clean、GPU selftest
   log+md、fatal checkpoint+RUNINFO+QA+run.log、platform、probe、**全部** TTS provenance 文件；
5. **fatal run log**：先 `mkdir -p fatal_smoke` 再 `tee fatal_smoke/r7_smoke_fatal_run.log`，
   消除上一轮日志缺失；
6. **目录隔离不变**：fatal_smoke / r7_main / tts_control 各自子目录，保持"一目录一 checkpoint"
   fail-closed 守卫语义；
7. `tts-control-only` 继续保持放行后执行（审查 §4 时序接受）。

## 三、GPU 主机下一步（放行前允许执行）

请按 `R7_FORMAL_RUN_HANDOFF.md` Gate r3：

1. 选择拟正式唯一 `code_commit` 并 clean checkout；
2. 运行 §1 G1–G8；
3. 运行 §2c GPU clean self-test；
4. 运行 §2b 非末位 fatal smoke（新目录、platform hash、完整 run log）；
5. 运行 §0b 完整 manifest + artifact commit；
6. push 整个材料包，提交最终书面放行复核。

正式 r7_main 与真实 tts-control 仍不得启动。
