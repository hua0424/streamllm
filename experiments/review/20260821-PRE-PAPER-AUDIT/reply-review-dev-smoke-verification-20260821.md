# 开发侧回复函：冒烟复核意见（provenance Gate）（对应 review-dev-smoke-verification-20260821.md）

- 日期：2026-08-22
- 总体：**意见全部采纳，无驳回项**。五项阻塞均为成立的事实（dirty 树、commit 混淆、TTS 服务
  provenance 缺口、`--tts-control-only` 未实现、平台条件未固化），已按 §5 Gate 逐项落实；
  算法未改、冒烟无需重做（除正式 run 启动前的新一轮探活——本就是 Gate 第 5 项）。

## 一、逐项处置

| 审查项 | 处置 |
|---|---|
| §4.1 dirty 工作树 | 采纳。正式 handoff（Gate 版）把 `git status --porcelain` 为空列为启动前置 G1；正式 run 一律新 run_id=r7_main + 新 checkpoint，绝不从 smoke 续跑（本就如此设计，现已明文） |
| §4.2 commit 混淆 | 采纳。核验报告已补勘误 addendum，拆分 code_commit=`1a0ddc8`（运行代码，dirty=true 如实在案）/ result_artifact_commit=`b1e1206` / verification_commit=`cdeb927`；正式流程要求回传三元组 |
| §4.3 TTS 服务端 provenance | 采纳。Gate 版 handoff G7 给出采集命令：CosyVoice commit + 本地修改 diff（server.py/requirements）、docker 镜像 digest、模型与 `spk2info.pt` hash、启动配置；**晓伊→内置中文女映射**已固化为脚本常量 `SPEAKER_MAPPING_NOTE`，自动写入每次 RUNINFO 与 checkpoint binding（G10），论文限制表述照 §6 |
| §4.4 `--tts-control-only` 未实现 | 采纳，选方案 1：**已实现并自测**。`run_tts_control()`：自动分层选 10 条成功配对（zh/en 各 5，不足即 fail-closed），每样本 3 调用（B 首句重测/A 回复首句离线派生/A 全文）+ 中英校准句各一=32 调用；独立 checkpoint/binding（control-from 主 checkpoint 的 SHA-256 入 binding）；新增 4 项 self-test（32 记录全成功、A 首句派生、校准句、配对不足拒绝）——self-test **90 PASS / 0 FAIL** |
| §4.5 CUDA/Triton fallback 与平台条件 | 采纳。新增 `--platform-conditions-file`：宿主机把驱动/CUDA/Triton fallback 登记/GPU 独占声明/nvidia-smi 状态写入文件，其 SHA-256 进 config hash 与 checkpoint binding；fallback 定性为**第二平台固定运行条件**，绝对毫秒只绑该平台、不与原平台混排（§6 边界照执行） |

## 二、Gate 11 项对照（review §5）

1. clean 树 / 2. 批准 code_commit / 3. 新 run_id+checkpoint / 4. 三元 commit 区分 /
5. 新一轮探活绑定正式 run（脚本启动时自动执行并写入 checkpoint）/
6. Silero artifact SHA-256 + PSE/分段器一致断言（脚本内置，冒烟已证）/
7. TTS 服务 commit/镜像/模型/spk2info hash（handoff G7 命令）/
8. fallback、双 3090、ASR/TTS 共存、无其他作业（platform_conditions.txt + binding）/
9. `--tts-control-only` 已实现（见上）/
10. speaker 映射入 RUNINFO 与论文限制（常量自动写入）/
11. 审查方书面放行记录 —— **待审查方对本回复复核后出具**。

## 三、测试与代码状态

- `run_ttfa_unified --self-test`：**90 PASS / 0 FAIL**（新增控制模式 4 项）；
  py_compile 通过；既有 86 项无回归；
- 正式 handoff 已重写为 Gate 版：`r7_ttfa_unified/R7_FORMAL_RUN_HANDOFF.md`
  （启动前 G1-G8 采集命令 + 正式命令 + 控制模式命令 + 加严验收清单）；
- 冒烟数据不重跑、不入论文（§6 边界照单接受；max_tokens=128 仅绑定 R7 等已登记）。

**申请**：复核本回复与 Gate 版 handoff，出具正式实验的书面放行记录（§5 第 11 项）。
