# Gate材料验证复核报告 r3 回复函复审（2026-08-22）

- **审查对象**：`experiments/review/20260821-PRE-PAPER-AUDIT/reply-review-gate-material-verification-r3-20260822.md`
- **复核范围**：回复函所述整改、登记文件时序、post-artifact clean、Git 提交链、manifest hash、Gate 证据完整性及放行边界。
- **最终结论：通过，可出具限定范围的书面放行。**

## 1. 本轮整改已核实

### 1.1 登记文件自引用问题已正确修正

`env/gate/gate_artifact_commit.txt` 现明确区分：

- `artifact_commit=a1fbb82` 是 Gate 材料提交；
- `porcelain_at_capture` 记录的是登记文件自身加入 Git 前的状态；
- 该记录不声称证明自身提交后的 clean；
- post-commit clean 由 checkout 上独立执行的 `git status --porcelain` 验证。

这正是上一轮审查要求的时序修正，证据语义已经自洽。该文件不需要纳入先前生成的 Gate manifest，因为它属于后续提交簿记，不是 manifest 生成时点的 12 项 Gate 运行依据；当前 manifest 对此已有明确说明。

### 1.2 独立 post-artifact clean 已核实

当前仓库状态：

```text
HEAD=b5355ee8111a97dfc77ae5e8d2d948c54e35406e
origin/main=b5355ee8111a97dfc77ae5e8d2d948c54e35406e
git status --porcelain 为空
```

提交链保持线性且角色可区分：

```text
b8893d6  code baseline
  → a1fbb82  result artifact commit
  → 6aaf356  artifact-register commit
  → a070fd6  registration semantic fix
  → b5355ee  reply/review commit
```

因此，`b8893d6` 仍是 Gate 和未来正式 run 的 code commit；后续提交没有被错误地当作 formal code baseline。

### 1.3 原 Gate 关键材料没有漂移

本机重新核验结果：

- manifest 12/12 LF-normalized SHA-256 全部一致；
- `platform_conditions_sha256` 仍与 fatal smoke、RUNINFO、config 绑定一致；
- GPU self-test 仍为 `90 PASS / 0 FAIL`；
- fatal smoke 序列仍为 `success → fatal error → 4 × cancelled_after_fatal`；
- TTS probe 和 provenance 文件仍在 manifest 覆盖范围内；
- 当前 checkout clean，且 `HEAD == origin/main`。

回复函中“r3 关键事实清单”与实际仓库状态一致。唯一需要注意的是，较早的核验清单文字仍提到 `e424eed` 作为当时的 post-artifact checkout；这是历史核验时点记录，不是当前 code commit，也不影响 Gate 证据链。若后续整理文档，可将其标注为“当时核验 checkout”，避免读者把它误认为当前 HEAD。

## 2. 回复函中其他说明的判断

以下说明均可接受：

1. manifest 不含 `gate_artifact_commit.txt`：因为 manifest 在 `a1fbb82` 形成前生成，登记文件在后续提交，时序合理；
2. `a1fbb82`、`6aaf356`、`e424eed`、`a070fd6`、`b5355ee` 不属于 formal code commit，未来 RUNINFO 应继续记录 `b8893d6`；
3. CHANGELOG 中“论文数据全部定稿”是历史阶段记录，不可替代 r7_main 后的最终状态，但当前回复函已经明确不会据此提前锁定 TTFA 论文数字；
4. CosyVoice dirty patch、镜像、模型和 speaker mapping 已记录，后续论文披露边界明确；
5. fake control/self-test/probe 仅是 Gate 或代码契约证据，不能替代正式 `r7_tts_control`。

## 3. 放行范围

现在可以签发、且仅签发以下书面授权：

### 允许执行

- 正式 `r7_main`：50 条样本的 A/B 主实验，加 10 条子集补轮；
- 使用新的 `r7_main/` 子目录、新 checkpoint、新 run ID；
- 启动时重新探活，并继续绑定固定 Silero artifact、TTS probe、platform conditions 和 code commit `b8893d6`。

### 仍须后置

- 真实 `r7_tts_control` 只能在 `r7_main` 完成且结果级 QA 通过后执行；
- 论文正式修改、Table VIII 更新和 `PAPER_WRITING_REFERENCE.md` 重新标记 final 只能在 r7_main 与 control 结果 QA 完成后进行。

### 明确禁止

- 从任意 smoke checkpoint 续跑；
- 在 r7_main 前运行真实 TTS control；
- 将 smoke、self-test、probe 或 fake control 当作论文正式样本；
- 跨平台拼接绝对延迟；
- 因平台差异对正式结果做缩放或事后修正。

## 4. 最终裁决

**Gate r3 整改通过，正式 `r7_main` 可以书面放行；真实 `r7_tts_control` 不在本次授权内，必须等待 r7_main 结果级 QA 通过。**

本轮不再需要阻塞性 Gate 材料整改。后续审查重点从“是否允许启动”转为：

1. r7_main 是否严格使用 `b8893d6` 和新 checkpoint；
2. 120 条任务是否全部有终态且无 fatal/thread leak/pair timeout/schema 违规；
3. TTFA 组件闭合、A/B 配对、分层覆盖和平台绑定是否通过结果级 QA；
4. 只有上述 QA 通过后，才复核并授权真实 TTS control。
