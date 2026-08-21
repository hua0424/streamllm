# 开发侧回复函（对应 review-reply-dev-smoke-verification-20260822.md）

- 日期：2026-08-22
- 总体：**意见全部采纳，无不必要项，无需与需求方确认的争议项**。逐条处置如下；
  self-test 保持 **90 PASS / 0 FAIL**。

## 一、逐项处置

| 审查项 | 处置 |
|---|---|
| §3.1 clean/provenance 未生成 | 采纳。这些是**正式 run 启动时点**的现场动作（clean 树检查、code_commit、新 checkpoint、启动探活），已固化在 Gate 版 handoff §1/§2；审查要求的顺序为"先完成 Gate 再书面放行"，handoff 已按此排序：G1-G8 采集 → 2b fatal 小 smoke → 2c self-test 归档 → 提请书面放行 → §2 r7_main。旧 smoke 产物保持原样不回写 |
| §3.2 TTS 服务端 provenance 未绑定 | 采纳。G7 命令已在 handoff（commit+本地 diff/镜像 digest/模型与 spk2info.pt hash/启动配置），现场执行后产物随放行申请一并提交——同意"G7 是待执行模板而非证据"的定性，补执行即可 |
| §3.3 平台条件文件未生成 | 采纳。同上属现场动作；另发现并修复一处**真实代码缺口**：`run_tts_control` 原先不读 `--platform-conditions-file`（control binding 只经早前补丁顺带带上字段但 cfg 未实际计算 hash）——现已显式计算并写入 control 的 config hash 与 binding（审查 §4 指出"control 分支读取路径需确认"正是此处） |
| §3.4 非末位 fatal 缺运行级证据 | 采纳。新增 `--inject-fault-index`（默认 -1=末位；非末位可指定任务下标）；handoff 新增 **2b 独立小 smoke**（run_id=r7_smoke_fatal，`--smoke 3 --inject-fault-index 1`）：验收=任务 0 success、任务 1 fault error+fatal、任务 2-5 全部 cancelled_after_fatal、QA 记录数 6；独立 run 不入正式结果 |
| §3.5 speaker 映射正式产物披露 | 采纳。SPEAKER_MAPPING_NOTE 已入正式与 control 两处 RUNINFO/binding；正式产物生成时自动携带（旧 smoke 产物为旧代码生成，维持原样不回写） |
| §4 self-test 归档 | 采纳。已生成本机不可变归档 `selftest_archive/selftest_20260822.md/.log`（命令、exit 0、执行时 HEAD、环境版本、90 项完整输出、输出 sha256；归档内注明当时工作树含待提交改动）；handoff 2c 增加 GPU clean 树复跑归档命令 |

## 二、Gate 12 项现状对照

1-3（clean/code_commit/新 checkpoint）→ handoff §1/§2 启动时点动作，待现场执行；
4（启动探活）→ 脚本启动时自动执行并绑定 run（代码已具备，冒烟已证）；
5-8（TTS provenance/平台文件/fallback 登记/GPU 独占）→ handoff §1 G7/G8 采集；
9（speaker 映射）→ 代码已固化，正式产物自动携带；
**10（非末位 fatal 证据）→ 本次新增 `--inject-fault-index` + handoff 2b，待现场执行**；
**11（self-test 归档）→ 本机归档已生成；GPU clean 树归档待现场**；
12（书面放行）→ 待审查方在前 11 项证据齐备后出具。

## 三、代码变更摘要（本轮）

- `run_tts_control`：显式计算并绑定 `platform_conditions_sha256`（含 control cfg 与 binding）；
- 新增 CLI `--inject-fault-index`（默认 -1）；fault 任务选择由"固定末位"改为按指定下标；
- handoff（Gate 版）新增 2b/2c 两节并明确执行顺序（fatal 小 smoke 与归档先于 r7_main）；
- self-test 90 PASS / 0 FAIL（本机复跑 + 归档内同结果）；py_compile 通过。

**申请**：GPU 主机按 Gate 版 handoff 执行 §1（G1-G8 采集）→ 2b → 2c 后，将证据与
放行申请一并提交审查方；审查方复核 12 项齐备后出具书面放行，随后执行 §2 正式 run。
