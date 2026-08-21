# Gate材料验证复核报告 r3（2026-08-22）

- **审查对象**：`experiments/review/20260821-PRE-PAPER-AUDIT/gate-material-verification-r3-20260822.md`
- **复核范围**：r3 核验清单、Git 提交链、Gate manifest、clean 记录、GPU self-test、非末位 fatal smoke、platform conditions、TTS provenance，以及当前仓库状态。
- **复核结论**：r3 已实质关闭上一轮提出的版本统一、platform hash、manifest 覆盖和材料落盘问题；但 `gate_artifact_commit.txt` 仍保留“自身尚未提交”的 porcelain 记录，与 r3 对“提交后 porcelain 空”的叙述不一致。因此建议判定为 **条件通过，暂缓正式书面放行**，先修正这一项自洽性问题，再进行一次短复核。

## 1. 已独立确认通过的内容

### 1.1 唯一代码基线

以下材料均引用同一完整提交：`b8893d63782b32a36eeb08584720993247fe0312`：

- `env/gate/gate_clean_git.txt`；
- `env/gate/gate_selftest_gpu.md`；
- fatal smoke checkpoint / RUNINFO；
- `env/gate/GATE_MANIFEST.md`。

Git 祖先关系也成立：

```text
b8893d6 → a1fbb82 → 6aaf356 → e424eed
```

当前 checkout 为 `e424eed`，工作树 `git status --porcelain` 为空。因而上一轮的多版本混用问题已关闭。

### 1.2 Gate 前 clean 证据

`gate_clean_git.txt` 的实际内容为：

```text
HEAD=b8893d63782b32a36eeb08584720993247fe0312
--- porcelain ---
---(空=clean)---
```

该文件是在材料生成前采集，且其内容本身没有列出未跟踪 Gate 目录；结合 r3 流程说明，足以证明 Gate 运行的起点是 clean checkout。fatal smoke 运行期间的 `git_dirty=true` 可以由运行时生成 Gate artifact 解释，不应再被误读为 Gate 起点脏树。

### 1.3 Platform conditions 绑定

fatal smoke checkpoint 和 RUNINFO 均包含：

```text
platform_conditions_sha256=a4c400576b7579b90abcfd6d1e7170033e2d19d8da0d90b9900312d27f9dcfd9
```

该值与 `env/platform_conditions.txt` 逐字节 SHA-256 一致，并进入 `config_hash=336b04df...`。因此上一轮 `null` 绑定问题已关闭。

平台材料内容足够识别第二平台条件：双 RTX 3090、驱动 550.127.05、CUDA 12.4、Xeon Gold 6133、Triton fallback、ASR/LLM/TTS 设备分配，以及实验期间无其他 GPU 作业声明。正式论文仍须披露该平台和 TTS/ASR 共用 `cuda:0` 的限制。

### 1.4 Manifest 覆盖和 hash 语义

manifest 首部明确声明：

```text
hash_scheme=LF-normalized-content-sha256
```

列出的 12 项材料覆盖：

- clean 记录；
- GPU self-test log/md；
- fatal checkpoint、RUNINFO、QA、完整 run.log；
- platform conditions；
- TTS probe；
- 三项 TTS provenance 文件。

本机按 `CRLF → LF` 后重算，12/12 一致。run.log 的 raw hash 与 LF-normalized hash 相同；provenance 中存在 CRLF 的文件按 manifest 明示的 LF-normalized 规则一致。该 hash 方案现在表述准确，不应再称为 Git blob hash。

### 1.5 非末位 fatal smoke

checkpoint 6 条记录的顺序为：

```text
success
error（fatal=true，fault_injection:asr_error）
4 × cancelled_after_fatal
```

四条 cancelled 记录的 `events`、`chunk_log`、`tts` 均为空；QA 记录数、终态计数和 checkpoint 一致。这项确实证明了非末位 fatal 后的 run-level fail-stop 行为。

### 1.6 GPU self-test 和 TTS provenance

- GPU self-test：90 PASS / 0 FAIL，exit 0，HEAD 与唯一 code commit 一致；
- TTS provenance 含 CosyVoice commit、dirty patch、镜像 digest、模型权重和 `spk2info.pt` hash、启动方式及依赖快照；
- probe 返回 HTTP 200、裸 PCM，且与正式请求契约一致；
- “晓伊”与内置“中文女” embedding 相同的限制已有模型级证据。

这些内容可以作为正式实验的前置证据，但不能替代正式 `r7_main` 或正式 TTS control 结果。

## 2. 仍需修正的唯一阻塞项

### 2.1 `gate_artifact_commit.txt` 的 porcelain 记录不代表提交后 clean

当前文件内容为：

```text
artifact_commit=a1fbb827145eed8a010748e46d7ade6d0e70284d
--- porcelain ---
?? experiments/results/revision/r7_ttfa_unified/env/gate/gate_artifact_commit.txt
```

该内容也已经被提交到 `6aaf356`，并在当前 checkout 中原样保留。因此它记录的是：**在把该文件加入 Git 之前，该文件自身是未跟踪的**；它不是“artifact commit 提交后 porcelain 为空”的证明。

这与 r3 清单 §5 的表述：

> 材料提交后再写 artifact commit+porcelain；提交时 porcelain 空

以及：

> 本机 `git pull --ff-only` 后工作树干净

存在证据语义不一致。当前工作树确实是干净的，但不能把 `gate_artifact_commit.txt` 内的两行当作提交后 clean 记录。

这不是实验数据错误，也不是代码基线错误，而是 Gate 证据文件的自引用/时序问题。由于本轮审查目标是判断能否出具正式放行，仍应修正后再签字，避免日后审计时被指出为“clean 证明记录了自身未提交”。

### 2.2 建议的最小修正

任选一种，但必须在文档中明确时序：

**方案 A（推荐）：**将 `gate_artifact_commit.txt` 改为准确的历史记录，例如：

```text
artifact_commit=a1fbb827145eed8a010748e46d7ade6d0e70284d
porcelain_at_capture=?? experiments/results/revision/r7_ttfa_unified/env/gate/gate_artifact_commit.txt
note=the record was captured before this file was added; post-commit cleanliness was verified separately at e424eed
```

同时保留当前仓库 `git status --porcelain` 为空作为 reviewer 的独立 post-commit verification。

**方案 B：**从记录中删除 porcelain 段，只保留 artifact commit，并新增独立的 `gate_post_artifact_verification.txt`，明确该文件由审查方在拉取后的 checkout 上运行 `git status --porcelain` 生成；不要声称该文件自身证明了提交后的空状态。

不建议通过再次生成一个自包含“空 porcelain”文件来声称它证明自身提交后的 clean，因为该文件在生成时仍然会使工作树变化。

## 3. 其他非阻塞观察

1. `GATE_MANIFEST.md` 不包含 `gate_artifact_commit.txt`。这可以接受，因为 manifest 是材料包在 artifact commit 形成前生成的；但应在 manifest 或 r3 清单中明确其生成时点，避免读者误以为 manifest 覆盖了最终登记文件。
2. 当前 HEAD `e424eed` 是登记和审查提交，不应被写成 formal code commit。正式 run 的 `RUNINFO` 应继续记录 `b8893d6` 为 code commit，并另行记录 result artifact commit 和 verification commit。
3. `REVISION_CHANGELOG.md` 中早期“论文数据全部定稿”属于历史阶段性记录；由于 r7_main 尚未执行，不能将其解释为包含 TTFA 正式结果已经最终定稿。后续应在正式结果 QA 后追加明确的最终状态条目。
4. TTS provenance 中的 CosyVoice 服务端 dirty patch 是已记录、可追溯的部署条件；正式论文不能把该服务称为未经修改的官方默认服务。

## 4. 放行边界

在修正 `gate_artifact_commit.txt` 的时序语义并完成一次短复核后，可以书面放行且仅放行：

- 正式 `r7_main`：50 条 A/B 主实验及 10 条子集补轮；
- `r7_main` 结果级 QA 通过后，再执行真实 `r7_tts_control`。

仍然禁止：

- 从 smoke checkpoint 续跑正式实验；
- 在 `r7_main` 前执行真实 TTS control；
- 将 fatal smoke 或 self-test 作为论文正式样本；
- 将 TTS probe 或 fake control 记录替代正式 control 结果；
- 在论文修改前把旧 TTFA 表或旧 `PAPER_WRITING_REFERENCE` 数字重新标为 final。

## 5. 最终结论

**r3 的主要整改是可信的，上一轮五项实质阻塞已关闭；但 artifact commit 登记文件中的自引用 porcelain 记录仍不准确。当前建议为“条件通过、修正一处后最终复核”，而不是立即无条件放行。**

修正该文件的证据时序后，审查方可以快速复核：唯一 code commit、Gate 前空 porcelain、platform hash、12/12 manifest、90/0 self-test、fatal→cancelled 序列和当前 checkout clean。若均保持一致，即可按限定边界签发 r7_main 书面放行。
