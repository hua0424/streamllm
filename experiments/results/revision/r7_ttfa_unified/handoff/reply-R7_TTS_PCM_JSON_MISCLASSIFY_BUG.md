# 开发侧回复：TTS 裸 PCM 被 classify_payload 误判 JSON/HTML（对应 handoff/R7_TTS_PCM_JSON_MISCLASSIFY_BUG_HANDOFF.md）

- 日期：2026-08-21
- 结论：现场诊断**完全属实，已修复并推送**。GPU 主机请按 `R7_GPU_SMOKE_HANDOFF_R4.md`
  重跑：任务 2 self-test（期望 **86 PASS**）→ 任务 3 冒烟（命令不变）。

## 1. 核实与修复

现场 200 次实测分布（198 pcm / 1 json / 1 html，首字节均匀）证明这是**概率性误判**：
裸 PCM（int16 字节流）首字节可取任意值，旧 `classify_payload` 用单字节判
`{`/`[`/`<`，撞上即误判——且探活与正式请求共用该分类器，不修则正式 50×2 每条 TTS
请求约 1.6% 概率假失败（现场影响面分析正确）。已按建议 2+3 组合修复：

1. **JSON 严格解析**：lstrip 后以 `{`/`[` 开头**且整段前缀能通过 `json.loads`（UTF-8
   strict）**才判 json；解析失败（音频字节）→ pcm。真实 FastAPI 错误体完整短小且必带
   `application/json` 头，能被正确捕获；
2. **HTML 特征匹配**：`<` 后须匹配 `<!doctype`/`<html`（大小写不敏感），否则 pcm；
3. **响应头辅助**：Content-Type 显式声明 json/html/xml 时以头为准（FastAPI
   HTTPException 必带，覆盖"无头但长 JSON 被前缀截断"的残余场景）；
4. WAV 判定保持 4 字节 "RIFF"（PCM 撞上概率 1/2^32）。

探活与正式请求的三个 classify 调用点均已传入响应 Content-Type。

## 2. 自测补齐（按现场要求）

新增 10 项用例：裸 PCM 首字节 `0x7b`/`0x5b`/`0x3c` 均判 pcm（含现场实测首样本
`(-133,-68,-108,-119)` 还原）、真 JSON 体/数组 JSON 判 json、**截断 JSON 无头判 pcm
（残余风险显式登记）**、响应头声明 json 以头为准、HTML 特征 ×2 判 html、`<` 后无特征
判 pcm、以及 measure 级"首块以 0x7b 开头的正常 PCM 流不被误判且 playable 达标"。

self-test 76 → **86 PASS / 0 FAIL**；本机真实组件集成检查复跑 ALL PASS。

## 3. 残余风险登记（如实披露）

无 Content-Type 头且 JSON 错误体**长于已累积前缀**（>64B）时，截断解析失败会判 pcm——
真实服务（CosyVoice StreamingResponse 无头 + FastAPI 错误带 json 头）不构成该场景；
若未来服务变更，探活的 `payload_class`/`magic_hex` 字段可复核。

## 4. 后续

按 `R7_GPU_SMOKE_HANDOFF_R4.md`：任务 2（期望 86）→ 任务 3 三层冒烟（命令不变）→
八项验收 → 产物 push。冒烟通过后本机结果级核验，再申请放行正式实验。
