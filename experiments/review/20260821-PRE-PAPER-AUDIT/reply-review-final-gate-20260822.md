# 开发侧回复函：最终放行复核前的整改（对应审查 2026-08-22 终裁）

- 日期：2026-08-22
- 总体：接受终裁。开发侧实现级整改已全部完成（审查已确认）；**剩余 8 项"未实际生成"
产物中，除流程循环为 handoff 缺陷（本轮已修）外，其余全部是 GPU 主机现场执行项**，
已重排为"放行前允许执行的 Gate"清单，随本回复一并发起最终放行复核。

## 一、流程循环修复（本轮唯一开发侧缺陷）

原 handoff 开头"未获书面放行前本文档不得执行"与"fatal 小 smoke/自测归档为放行前置"
构成循环。已按审查要求明确划分（handoff 升级为 **Gate 版 r2**，文档头部"执行权限划分"）：

- **放行前允许（GPU 主机即可执行）**：§1 G1–G8 采集 + §2b 非末位 fatal 小 smoke +
  §2c GPU clean 树 self-test 归档；
- **需书面放行**：仅 §2 r7_main（120 任务）与 §3 tts-control（依赖 r7_main 产物）。

并新增 §0"GPU 主机待执行清单"表（六步 → 各自产物 → 齐备后核验 → 最终放行复核），
§2 节头注明须书面放行后执行、2b/2c 虽编号在后但先于 §2 完成。

## 二、审查所列 8 项缺失产物的归属与状态

| # | 缺失产物 | 归属 | 状态 |
|---|---|---|---|
| 1 | GPU clean worktree + 正式 code provenance | GPU（Gate G1/G2） | 待现场执行 |
| 2 | r7_main checkpoint/RUNINFO/QA/日志 | GPU（**即正式 run 本身**） | 需书面放行后执行——非放行前置 |
| 3 | CosyVoice commit/镜像/模型/spk2info/启动配置 | GPU（Gate G7） | 待现场执行 |
| 4 | 正式 platform_conditions + binding hash | GPU（Gate G8；代码绑定已具备） | 待现场执行 |
| 5 | 非末位 fatal → cancelled smoke JSONL/QA | GPU（Gate 2b；`--inject-fault-index` 已实现） | 待现场执行 |
| 6 | GPU clean 树 self-test 归档 | GPU（Gate 2c；命令已内置） | 待现场执行 |
| 7 | 正式 run 中 speaker mapping 记录 | GPU（代码自动写入，r7_main 时自然产生） | 随 #2 产生 |
| 8 | 真实 GPU tts-control 产物 | GPU（**依赖 r7_main**） | 需书面放行后执行——非放行前置 |

其中 #2/#7/#8 是正式 run 及其衍生物，属"放行后"事项——与审查"只有 r7_main 需要
书面放行"的划分一致；#1/#3/#4/#5/#6 为放行前 Gate，命令全部在 handoff §0/§1/2b/2c。

## 三、申请

1. 请审查方确认 handoff Gate 版 r2 的执行权限划分（消除循环）后，**GPU 主机执行 §0
   六步并提交产物**；
2. 开发侧核验上述产物后发起**最终放行复核**；
3. 书面放行仅覆盖 §2 r7_main 与 §3 tts-control。
