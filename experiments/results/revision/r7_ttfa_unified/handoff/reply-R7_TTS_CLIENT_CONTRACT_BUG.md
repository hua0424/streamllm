# 开发侧回复：TTS 客户端契约错配（对应 handoff/R7_TTS_CLIENT_CONTRACT_BUG_HANDOFF.md）

- 日期：2026-08-21
- 结论：现场诊断**完全属实，两处契约错配均已修复并推送**；次要两事已定夺（见 §2）。
  GPU 主机请按 `R7_GPU_SMOKE_HANDOFF_R2.md` 重跑（顺序不变，self-test 期望值更新为 75）。

## 1. 核实与修复

对照 E6 `measure_tts_first_chunk.py:55-61` 确认：真实契约是
`POST {base}/inference_sft` + `data=` form 编码（服务端 `Form()` 参数）。
`run_ttfa_unified.py` 的 `tts_probe`/`tts_measure` 在这两点上相对 E6 回归属实
（根路径 + `json=` → 404）。已修复：

1. `_tts_endpoint()`：幂等拼接 `/inference_sft`（裸基址/尾斜杠/已带后缀均不重复拼）；
2. `_tts_form_body()`：全部字段显式字符串（`stream="True"`），`data=` form 编码；
   探活与正式请求共用同一构造；
3. **回归防护**（现场指出"69 项未覆盖真实契约"的缺口）：self-test 假 TTS 服务改为
   契约严格路由——非法路径 404 JSON、非 form body 422 JSON（与真实 FastAPI 行为一致）；
   新增探活契约测试（ok=true 走 form+正确端点）与两条负向（根路径+JSON 得 404、
   对路径+JSON 得 422）、端点归一化 ×3；
4. self-test **75 PASS / 0 FAIL**；本机真实组件集成检查（3060）复跑 ALL PASS。

## 2. 次要两事的定夺

1. **Content-Type: None**：按设计接受——探活把缺头按 `None` 原样固定为允许值
   （probe 输出新增 `policy_note`），正式请求逐项严格相等比对，任何偏离记 error、不放宽。
   旧审查要求"缺 Content-Type 不得视为可接受"针对旧实现"缺头静默放行"；现实现是
   "探活固定 + 正式严格比对"，满足其意图，已在 handoff r2 §1.1 说明；
2. **Silero 目录非 git checkout**：可接受——锁定依据转为 artifact sha256（现场实测
   `e1122837…d3720`）；`_silero_artifact_meta` 对该场景补 `repo_commit_note` 注记，
   `repo_commit=None` 属预期输出。继续用 `--silero-dir`，无需 `--silero-ref`。

## 3. 后续

GPU 主机按 `R7_GPU_SMOKE_HANDOFF_R2.md` 执行：探活（期望 ok:true / payload_class:pcm）
→ self-test（期望 75 PASS）→ 3 条分层冒烟（含 asr_error 注入）→ 八项验收 → 产物 push。
探活若仍失败，不改代码不放宽策略，按 handoff r2 §3 反馈现场信息。
