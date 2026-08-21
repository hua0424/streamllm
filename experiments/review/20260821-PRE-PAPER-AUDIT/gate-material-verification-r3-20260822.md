# R7 放行前 Gate 材料包 r3 本机核验清单（对应 review-gate-material-verification-20260822.md）

- 核验日期：2026-08-22
- GPU 材料包提交：`a1fbb82`（result_artifact_commit）+ `6aaf356`（登记提交）
- 拟批准 code_commit：`b8893d6`
- 核验方式：GPU 推送后，本机以仓库 Git blob 内容 + LF-normalized 内容 SHA-256 逐项重算比对
- 结论：**上一轮 5 项阻塞（§3.1–§3.5）全部关闭；建议审查方出具正式书面放行。**

## 1. 上一轮阻塞项 → 本轮证据逐项对照

| 审查 § | 阻塞点 | 本轮材料 | 核验结果 |
|---|---|---|---|
| 3.1 | code commit 不唯一、clean 记录自相矛盾 | `gate_clean_git.txt` + 全材料统一 HEAD | **已关闭**，见 §2 |
| 3.2 | platform conditions 未绑定 fatal smoke | checkpoint `tts_config.platform_conditions_sha256` | **已关闭**，见 §3 |
| 3.3 | manifest 不完整、hash 语义不明 | 12 项 manifest + 显式 `hash_scheme` | **已关闭**，见 §4 |
| 3.4 | 工作树 Gate 材料 modified | 材料全部落入 `a1fbb82`，提交后 porcelain 空 | **已关闭**，见 §5 |
| 3.5 | self-test 版本绑定未统一 | `gate_selftest_gpu.md/.log` HEAD=`b8893d6` | **已关闭**，见 §6 |

## 2. 唯一 code commit 与 clean 记录（§3.1）

- `env/gate/gate_clean_git.txt`：`HEAD=b8893d63782b32a36eeb08584720993247fe0312`，porcelain **真正为空**（在材料生成前采集，非事后补写"空=clean"文字）；
- fatal smoke checkpoint header `git_commit = b8893d6…`；`gate_selftest_gpu.md` HEAD=`b8893d6…`；manifest `code_commit=b8893d6…`——四处引用**同一** commit，不再出现 `2e54ac2`/`34ea12e`/`81f548f` 混用；
- `b8893d6` 是 `a1fbb82`/`6aaf356` 的线性祖先（`git merge-base --is-ancestor` 通过），Gate 材料包在唯一 clean 基线上产生。

**关于 checkpoint `git_dirty=true` 的说明（预期，非矛盾）：** fatal smoke 运行时，Gate 材料（platform_conditions.txt、selftest log、probe 等）已作为预期 artifact 落盘，`_git_info()` 的 porcelain 因此非空，故 checkpoint 记录 `git_dirty=true`。这与 handoff r3 §0 的规则一致——clean 证明只在材料生成**前**的 G1 有效，运行期 dirty 是预期 artifact，不是"在脏树上跑实验"。真正的 clean 证据是 G1 文件。

## 3. platform conditions 绑定（§3.2）

- fatal smoke checkpoint header 的 `tts_config.platform_conditions_sha256 = a4c400576b7579b90abcfd6d1e7170033e2d19d8da0d90b9900312d27f9dcfd9`；
- RUNINFO 的 config 块与 checkpoint `tts_config` 两处均携带该 hash；
- 与 `env/platform_conditions.txt` 逐字节一致（`sha256_file` 同源），**不再是 `null`**；
- 该 hash 同时计入 `config_hash=336b04df…`（`platform_conditions_sha256` 已进入 config 字典），保证平台文件与 run binding 冻结。

## 4. Gate manifest 完整性与 hash 语义（§3.3）

- manifest 首部显式声明 `hash_scheme=LF-normalized-content-sha256`（不再是"Git blob hash/原始文件 hash"）；
- 覆盖 **12 项**（上轮漏列的 selftest md、fatal RUNINFO/QA/run.log 已补入）：

```text
env/gate/gate_clean_git.txt
env/gate/gate_selftest_gpu.log
env/gate/gate_selftest_gpu.md
fatal_smoke/checkpoint_r7_smoke_fatal.jsonl
fatal_smoke/RUNINFO_r7_smoke_fatal.md
fatal_smoke/QA_r7_smoke_fatal.md
fatal_smoke/r7_smoke_fatal_run.log
env/platform_conditions.txt
env/gate/tts_probe_new.json
env/gate/tts_provenance/server_dirty.patch
env/gate/tts_provenance/tts_pip_freeze.txt
env/gate/tts_provenance/TTS_PROVENANCE.md
```

- 本机按 LF-normalized 内容 SHA-256 重算，**12/12 全部一致**（run.log 含进度条 `\r` 字节，无 CRLF，raw 与 LF-normalized 同值）。

## 5. 工作树与 artifact commit（§3.4）

- 材料全部落入 `a1fbb82`，登记提交 `6aaf356` 记录 `artifact_commit=a1fbb82…` + 提交时 porcelain（仅剩待登记的 `gate_artifact_commit.txt`，随后一并提交）；
- 本机 `git pull --ff-only` 后工作树干净（仅剩审查方输入文档未跟踪），HEAD==origin/main==`6aaf356`。

## 6. GPU clean-tree self-test（§3.5）

- `gate_selftest_gpu.log`：`[PASS]` 计数 90、`[FAIL]` 计数 0，末尾 `self-test 90 PASS / 0 FAIL` + `exit=0`；
- `gate_selftest_gpu.md` 记录命令、期望/结果、`git HEAD: b8893d6…`、环境（python 3.10.18 / torch 2.5.1+cu121）、输出 sha256。

## 7. 未在 manifest 内但同包提交的次要产物（说明）

`fatal_smoke/tts_probe.json` 与 `fatal_smoke/ttfa_summary_r7_smoke_fatal.csv`、`ttfa_subset_cv_r7_smoke_fatal.csv` 为脚本运行副产物，非放行 Gate 依据（放行依据的 probe 为 `env/gate/tts_probe_new.json`），已随 `a1fbb82` 提交但不强制入 manifest。

## 8. 放行申请边界（不变）

审查方现在复核并书面放行：

- 正式主实验 `r7_main`（50×A/B + 10 条子集补轮）；
- `r7_main` 结果级 QA 通过后，执行真实 `r7_tts_control`（32 次匹配文本 TTS 控制）。

正式 run 必须使用新子目录 `r7_main/`、新 checkpoint、启动时重新探活；不得从任何 smoke checkpoint 续跑。
