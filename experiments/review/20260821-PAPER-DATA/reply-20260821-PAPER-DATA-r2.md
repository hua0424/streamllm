# 论文数据就绪度复审 r2 回复（2026-08-21，对应 review-20260821-PAPER-DATA-r2.md）

> 审查报告：`experiments/review/20260821-PAPER-DATA/review-20260821-PAPER-DATA-r2.md`
> 回复范围：r2 唯一未闭环 P0（方案2 过冲剔除真实排水工作）+ 其余确认项。

## 总体回应

**r2 的新发现成立，我方独立重算逐条吻合**：

- `E4 TTFT − E5 post-flush`（同 50 样本、同机同码）：**mean 410.5ms、min 139.0、max 1257.9，
  50/50 全为正**——静音窗内确有 ~410ms 真实排水工作，方案2 的 1012.5ms 口径把其一并剔除，
  低于同一物理量的 E4 直接实测（53.1+1422.9=1476.0 vs 53.1+1012.5=1065.6，差 410.4ms 恰为排水项）。
- 守恒分解 `3065.1 = 53.1 + 1999.5 + 1012.5` 成立。
- 分语种 E4 streaming TTFT：zh 1398.9 / en 1447.0（n=25 各），与 r2 预告一致。

方向判断认同：方案2 使 B 行 pipeline 分项少计 ~0.41s、有利于本文系统，与禁美化红线冲突；
且 r2 对澄清 #3 的两点反驳（E4/E5 参考点同一、多源装配本已存在且同机同码风险可控）成立，
我方此前"尾隙/混源"理由不充分，予以撤回。

## 处置（需求方二次裁决 = 方案 (a)，已执行）

- `assemble_ttfa_budget.py`：B 行 post 分项改用 **E4 同 50 样本 streaming TTFT**
  （端点触发 flush 尾延迟的直接实测）；self-test 5/5（含新口径断言）。
- Table VIII 定稿（`r6_ttfa/ttfa_budget.csv`）：
  **System B ALL 14.79s（zh 15.58 / en 13.99）；System A ALL 22.67s；差距 7.9s**——
  与 r2 预告值（14.79/15.58/13.99）一致。A 行与其余三分项（端点 53.1 / decode 389.0 / TTFC）未动。
- 文档对齐（同一最终口径）：`PAPER_HANDOFF.md` §TTFA、`r6_ttfa/RUNINFO.md` 移交说明、
  `REVISION_CHANGELOG.md` 登记（含两轮裁决的演进与理由）。

## 对 r2 责任认定的说明

接受 r2 的责任划分：首轮报告的方案2 把"first_token−final_enqueue"与"E4 同期 flush 口径"两个
不等价子项以"或"并列并给同一总计，我方按需求方裁决字面实现无误；本轮由审查方证据化自我修正，
双方记录一致即可，无需额外归因动作。

## 其余复审结论

- 澄清 #1（A 行 decode/TTFC 估计口径 + source 列标注）、澄清 #2（B2 增补轨）、E1 CV 只报
  mean/median、P1-2/P2 改稿清单——均已闭环，无分歧。

## 验证证据

```text
uv run python -m experiments.scripts.assemble_ttfa_budget --self-test   → 5/5 PASS
uv run python -m experiments.scripts.assemble_ttfa_budget               → B ALL 14787.3ms / A ALL 22673.4ms
（zh 15579.6 = 52.7+1398.9+142.3+13985.7；en 13995.0 = 53.5+1447.0+635.7+11858.9，分量求和复核一致）
```

Table VIII 数字（B 14.79s / A 22.67s）自此可入论文；数据侧无其他未闭环项。
