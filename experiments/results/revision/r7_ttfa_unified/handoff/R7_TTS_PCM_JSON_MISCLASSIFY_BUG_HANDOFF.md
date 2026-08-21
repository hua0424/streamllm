# R7 冒烟阻塞：TTS 探活 classify_payload 对裸 PCM 的 JSON 误判（请开发侧修复）

> **读者**：开发/审查侧。GPU 主机已按 `R7_GPU_SMOKE_HANDOFF_R3.md` 执行：
> 任务 2 self-test 复跑 **76 PASS / 0 FAIL**；任务 3 冒烟在**正式路径的 TTS 探活**阶段再次 fail-closed。
> **状态**：待开发侧修复后重跑。**未改代码、未绕过 fail-closed、未执行任何任务。**
> **关联**：`R7_GPU_SMOKE_HANDOFF_R3.md`、`handoff/reply-R7_PSE_SILERO_SIGNATURE_BUG.md`。

---

## 1. 现象（GPU 主机实测，任务 3 冒烟，exit 0 但 fail-closed）

r7_smoke_run.log 末尾：

```
TTS 探活失败: {'url': 'http://127.0.0.1:20401', 'endpoint': 'http://127.0.0.1:20401/inference_sft',
 'spk_id': '晓伊', 'speed': 0.8, 'status': 200, 'content_type': None, 'content_encoding': None,
 'allow_content_type': None, 'allow_content_encoding': None,
 'policy_note': '缺失头按 None 原样固定为允许值；正式请求逐项严格相等比对，任何偏离（含头新增/取值变化）记 error，不放宽',
 'payload_class': 'json', 'magic_hex': '7bffbcff94ff89ff', 'ok': False}（停止）
```

关键点：**HTTP status 200**（服务正常返回音频），但 `payload_class: "json"` → `ok: false` → 停止。

## 2. 根因（已定位，非服务异常，是分类器误判 + 概率性）

`classify_payload()`（`run_ttfa_unified.py:91-102`）用**单个字节**判 JSON：

```python
if p[:1] in (b"{", b"["):
    return "json"
```

裸 PCM 是 16-bit 小端 int16 字节流，**首字节可以是任意 0x00–0xff**。本次失败的
`magic_hex = 7bffbcff94ff89ff` 解成 int16 是 `(-133, -68, -108, -119)`——是正常音频，
只是**首字节恰好是 `0x7b`（`{`）**，被误判为 JSON 错误体。

我用真实服务连打 200 次探活，统计如下：

```
payload_class: {'pcm': 198, 'json': 1, 'html': 1}
first-byte top: 62/eb/fb/db/1d/cc/...  （均匀散布）
```

- 200 次里 **1 次 json、1 次 html**（`0x3c` = `<`）误判，其余 198 次 pcm；
- 首字节分布均匀，证明这是**概率性误判**，不是服务偶发返回 JSON。

所以：**同一个 TTS 服务、同一命令**，探活有时 ok 有时 fail，取决于响应音频首字节是否
撞上 `{`(0x7b)/`[`(0x5b)/`<`(0x3c)。任务 1 首次探活通过（首字节非这些字节），
任务 3 正式路径重跑探活撞上 `0x7b` 即 fail-closed。

### 为什么自测没拦住

自测里「探活走 form+/inference_sft 契约」「契约防护·根路径+JSON 得 404」「对路径+JSON 得 422」
用的是**假服务返回的真 JSON 字节**（`{...`），所以能正确分类 json；但没有覆盖
「裸 PCM 首字节恰为 `{`/`[`/`<`」这种**真音频假阳性**场景。契约假服务对齐了路由与 body，
却没有对齐「返回体是裸 PCM、首字节随机」这一真实属性。

## 3. 影响面

`classify_payload` 被**探活（`:344`）与正式 TTS 请求（`:413`、`:427`）共用**。因此该误判
不仅让探活随机失败，正式 50×2 的每条 TTS 请求也有 ~2/256×2（`{`/`[`）≈1.6% 概率被
`tts_format_not_pcm` 整行 error——即使不改，正式实验也会出现**随机 TTS 假失败**，必须修。

## 4. 修复建议（供开发侧参考，勿由 GPU 主机代改）

裸 PCM 与 JSON/HTML/WAV 的判别应从「单字节前缀」改为**更可靠的结构校验**，例如：

1. **先看 Content-Type / 长度**：真实 JSON 错误体有 `application/json` 头（本次探活
   允许策略已固定 `content_type=None`，但这是"允许值"，判别格式时应另看正文）；
2. **JSON 需通过严格解析**：仅当 `lstrip` 后以 `{`/`[` 开头**且能 `json.loads` 成功**
   才判 json；HTML 需 `{`/`<` 后能匹配 `<!doctype`/`<html` 等特征；否则按 PCM 处理。
   单字节 0x7b 不应单独成立。
3. **或用响应头判别**：CosyVoice `StreamingResponse` 默认无 Content-Type（本次实测 None），
   而 FastAPI 错误响应（HTTPException）带 `application/json`。可把「HTTP 200 + 无 JSON 头」
   作为 PCM 强证据，仅对非 200 或明确 JSON 头才走 JSON 校验。

请同时补一条自测：**裸 PCM 首字节为 `0x7b`/`0x5b`/`0x3c` 时 `classify_payload` 仍判 pcm**，
以及「真 JSON 错误体（可 `json.loads`）判 json」，防止下次回归。

## 5. 现场与基线

- 仓库：`/dataA/streamllm`，HEAD = `4a6e418c701fbdbd2730afde19b9a0afdea28338`。
- `run_ttfa_unified.py` sha256 = `06854199170e30b77cb7fc10277fc947434c8883a7046f3968674b0cea5a17e1`。
- 任务 1（探活，r2 版）已通过并收妥（`tts_probe.json` ok=true/pcm）。
- 任务 2 self-test 复跑：**76 PASS / 0 FAIL**（exit 0）。
- 任务 3 冒烟：正式路径 TTS 探活 fail-closed，未产生 checkpoint/RUNINFO/QA/summary，
  未执行任何任务、未加载模型。
- TTS 服务：CosyVoice v2.0，commit `8555549e`（`server.py` 本地修改），`127.0.0.1:20401` 正常；
  `/inference_sft` 返回裸 PCM、无 Content-Type 头（`StreamingResponse` 默认），FastAPI 错误才带 JSON。
- 复现：连续多次调 `tts_probe('http://127.0.0.1:20401','晓伊',0.8)` 可见 pcm/json/html 交替；
  单次失败可稳定复现于「响应音频首字节为 0x7b」时。

## 6. 修复完成后

开发侧修正并推送后，GPU 主机重跑：任务 2 self-test（期望值按修复后新值）→ 任务 3
三层冒烟（命令不变，八项验收）。探活重跑若仍非 ok:true，按 handoff 规则反馈现场、不改码不放宽。
