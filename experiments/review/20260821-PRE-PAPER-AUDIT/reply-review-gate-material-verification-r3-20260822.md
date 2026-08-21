# 开发侧回复函：Gate r3 复核意见整改（对应 review-gate-material-verification-r3-20260822.md）

- 日期：2026-08-22
- 对应审查：`experiments/review/20260821-PRE-PAPER-AUDIT/review-gate-material-verification-r3-20260822.md`
- 整改提交：`a070fd6`
- 总体：**意见采纳。** 审查指出的唯一阻塞项——`gate_artifact_commit.txt` 证据时序/自引用问题——已修正；其余 4 项非阻塞观察一并说明。本轮仍只申请书面放行 `r7_main`，不申请任何论文数据锁定。

## 一、唯一阻塞项：`gate_artifact_commit.txt` 的时序/自引用语义（审查 §2.1）

审查意见完全成立：该文件原内容里的 porcelain 记录的是**该文件自身加入 Git 之前**的状态
（`?? …/gate_artifact_commit.txt`），因此它只能证明「采集时除本登记文件外无其它漂移」，
不能声称「artifact commit 提交后 porcelain 为空」。这是 Gate 证据的自引用/时序问题，非数据或代码基线错误。

已按审查推荐的**方案 A**修正。文件现为准确的历史记录：

```text
artifact_commit=a1fbb827145eed8a010748e46d7ade6d0e70284d
porcelain_at_capture=?? experiments/results/revision/r7_ttfa_unified/env/gate/gate_artifact_commit.txt
note=recorded before this file itself was added to git; it proves the artifact commit was the only remaining change at capture time, NOT the cleanliness after this file was committed. Post-commit cleanliness is verified independently by running `git status --porcelain` on the current checkout (empty).
```

关键语义三点，已写入文件本身：

1. `artifact_commit` 明确为 `a1fbb82`（result artifact commit）；
2. `porcelain_at_capture` 如实保留「采集时仅本登记文件未跟踪」，并注明这是采集时点状态；
3. `note=` 明确「提交后 clean 由当前 checkout 独立验证」，不再让该文件声称自证含自身之后的 clean。

## 二、提交后 clean 的独立证据（审查 §2.2 的 post-artifact verification）

按审查要求，登记文件不再承担「提交后 clean」的证明职责。该证明改由**独立、可复跑**的方式给出：

- 本机 `git pull --ff-only` 至 `a070fd6`（已包含 `a1fbb82` 与登记提交 `6aaf356`）后，
  `git status --porcelain` 输出为空；
- 审查方可在其 checkout 上自行执行 `git status --porcelain` 复验，与该登记文件的自引用记录完全分离；
- r3 核验清单 `gate-material-verification-r3-20260822.md` §5 已同步改写，不再沿用
  「提交后 porcelain 空」的旧表述，改述为「提交后 clean 由当前 checkout 独立验证」。

## 三、其余 4 项非阻塞观察的处置说明

| 审查观察 | 处置 |
|---|---|
| 3.1 manifest 不含 `gate_artifact_commit.txt` | 确认。manifest 在 artifact commit 形成前生成，登记文件在随后的 `6aaf356` 才提交，故不在其中；核验清单 §4 已补「manifest 生成时点」说明，避免误读。 |
| 3.2 `e424eed` 等不应写为 formal code commit | 确认。正式 `r7_main` 的 RUNINFO 仍记录 `b8893d6` 为 code commit，`a1fbb82`/`6aaf356`/`e424eed` 分别记 result artifact / 登记 / 核验提交，三元组区分不变。 |
| 3.3 CHANGELOG「论文数据全部定稿」系历史阶段性记录 | 确认。该句属 R2 口径的历史记录，不涵盖尚未执行的 `r7_main`；正式结果级 QA 通过后会追加明确的最终状态条目，不会把旧 TTFA 表/旧 `PAPER_WRITING_REFERENCE` 数字重新标 final。 |
| 3.4 TTS 服务端 dirty patch 是已记录的部署条件 | 确认。CosyVoice 的本地修改（`server_dirty.patch` 163 行、镜像 digest、`spk2info.pt` hash）已入 G7 provenance 并被冻结；论文将如实披露该服务为本地配置版本，不写成未经修改的官方默认服务。 |

## 四、本轮申请边界（与审查 §4 一致）

仅申请书面放行：

1. 正式 `r7_main`：50 条 A/B 主实验 + 10 条子集补轮；
2. `r7_main` 结果级 QA 通过后，执行真实 `r7_tts_control`（32 次匹配文本 TTS 控制）。

继续遵守并执行下列禁令，不越界：

- 不从 smoke checkpoint 续跑正式实验；
- 不在 `r7_main` 前执行真实 TTS control；
- 不将 fatal smoke 或 self-test 作为论文正式样本；
- 不以 TTS probe 或 fake control 记录替代正式 control 结果；
- 不在论文修改前把旧 TTFA 表或旧 `PAPER_WRITING_REFERENCE` 数字重新标 final。

## 五、可供快速复核的事实清单

- 唯一 code commit：`b8893d63782b32a36eeb08584720993247fe0312`（Gate 起点 clean，G1 porcelain 真正为空）；
- platform hash 绑定：`a4c400576b7579b90abcfd6d1e7170033e2d19d8da0d90b9900312d27f9dcfd9`，并进入 `config_hash=336b04df…`；
- Gate manifest：`hash_scheme=LF-normalized-content-sha256`，12/12 本机重算一致；
- GPU self-test：`90 PASS / 0 FAIL`，exit 0，HEAD=`b8893d6`；
- 非末位 fatal smoke：`success → error(fault_injection:asr_error, fatal=true) → 4×cancelled_after_fatal`，QA 0，cancelled 无事件污染；
- 当前 checkout：`a070fd6`，`git status --porcelain` 为空（含登记文件已提交后的独立 post-artifact clean 证据）。

以上整改与事实若复核一致，请按审查 §4 的限定边界签发 `r7_main` 书面放行。
