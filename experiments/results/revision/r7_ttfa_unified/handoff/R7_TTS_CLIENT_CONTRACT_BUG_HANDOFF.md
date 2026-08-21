# R7 统一 TTFA 冒烟阻塞：TTS 客户端契约错配（请开发侧修复）

> **读者**：开发/审查侧。GPU 主机执行者已按 `R7_GPU_SMOKE_HANDOFF.md` 执行到任务 1 即被硬门禁阻断。
> **状态**：待开发侧修复后重跑。**请勿让 GPU 主机临时放宽允许策略或绕过探活**（handoff §2 明令禁止）。
> **关联**：`experiments/results/revision/r7_ttfa_unified/R7_GPU_SMOKE_HANDOFF.md`、
> `experiments/review/20260821-PRE-PAPER-AUDIT/reply-review-implementation-r2-20260821.md`（已放行 TTS 探活 + 3 条冒烟）。

---

## 1. 现象（GPU 主机实测，探活失败）

```bash
uv run python -m experiments.scripts.run_ttfa_unified \
    --tts-probe --tts-url http://127.0.0.1:20401 --tts-spk 晓伊 --tts-speed 0.8 \
    --output-dir experiments/results/revision/r7_ttfa_unified
```

输出（exit 1）：

```json
{
  "url": "http://127.0.0.1:20401",
  "status": 404,
  "content_type": "application/json",
  "payload_class": "json",
  "magic_hex": "7b2264657461696c",   // = {"detail （FastAPI 404 "Not Found"）
  "ok": false
}
```

`magic_hex = 7b2264657461696c` 即 `{"detail` 的 UTF-8 十六进制，说明服务返回的是 FastAPI 404 JSON，不是 PCM。

## 2. 根因（两处契约错配，均在 `run_ttfa_unified.py`）

GPU 主机对真实 TTS 服务（CosyVoice v2.0，commit `8555549e`）做了四种请求组合实测：

| 请求 | 结果 |
|---|---|
| `POST /` + form | 404 JSON |
| `POST /` + JSON（**脚本当前做法**） | 404 JSON ← 探活失败 |
| `POST /inference_sft` + form | **200，PCM**（magic `69009100…`）✓ |
| `POST /inference_sft` + JSON | 422（服务端 `Form()` 不接受 JSON body）|

**错配 1：URL 少拼 `/inference_sft`。**

- `tts_probe()`（`run_ttfa_unified.py:303`）与 `tts_measure()`（`:351`）均直接
  `requests.post(url, ...)`，`url` 就是 `--tts-url http://127.0.0.1:20401`（根路径）。
- 但 CosyVoice FastAPI 服务（`runtime/python/fastapi/server.py`）只有
  `/inference_sft` / `/inference_zero_shot` / … 等路由，根路径无路由 → 404。

**错配 2：body 用 `json=` 而非 `data=`。**

- 服务端 `inference_sft` 是 `Form()` 参数，只吃 `application/x-www-form-urlencoded`。
- 即便 URL 改成 `/inference_sft`，`json={...}` 仍会 422。

**对比（正确契约已在 E6 脚本存在）**：`experiments/scripts/measure_tts_first_chunk.py:55-61`
用的是 `f"{cfg['url'].rstrip('/')}/inference_sft"` + `data={"tts_text":..., "spk_id":...,
"stream": True, "speed":...}`（form 编码）。`run_ttfa_unified.py` 在这两点上相对 E6 回归了。

## 3. 修复建议（供开发侧参考，勿由 GPU 主机代改）

在 `run_ttfa_unified.py` 中把 `tts_probe()` 与 `tts_measure()` 两处的请求改为与 E6 一致：

1. URL 拼后缀：`url = f"{base_url.rstrip('/')}/inference_sft"`（或等价处理，需覆盖
   `--tts-url` 已带 `/inference_sft` 后缀与不带后缀两种情况，避免重复拼接）。
2. body 改 `data={...}`（form 编码），`stream` 字段用 `"True"` 字符串（与 E6 一致，
   服务端 `Form` 反序列化后为 `"true"`/`"True"` 字符串均可，但布尔 `True` 经 JSON 是 422 的诱因之一）。
3. 若担心 `requests` 对 `data` 里非字符串值的编码，请显式全传字符串。

请同步自查：`tts_measure()` 里对 `resp.headers.get("Content-Type")` 与
`probe["allow_content_type"]` 的比对逻辑在 form 编码下是否仍成立（本次真实服务 form 请求
返回的 `Content-Type` 为 `None`/无该头，见下节，需确认「允许策略固定」在缺头场景仍按设计工作）。

## 4. 顺带请开发侧确认的两件次要事项

1. **真实服务 form 请求返回 `Content-Type: None`**：`POST /inference_sft` + form 实测
   响应头无 `Content-Type`（`Content-Encoding` 亦无）。探活把缺失值也作为「固定允许策略」
   记录下来（脚本设计如此）。请确认正式请求的逐项比对在 `allow_content_type=None` 时的
   语义正确（不应导致正式请求误判），并与 review「缺 Content-Type 不得视为可接受」的
   要求核对（该要求针对的是旧实现；现实现已固定策略，语义是否满足请定夺）。
2. **Silero 缓存目录非 git checkout**：`~/.cache/torch/hub/snakers4_silero-vad_master`
   没有 `.git`，`git rev-parse` 失败，`_silero_artifact_meta()` 里 `repo_commit` 会记 `None`。
   脚本仍能哈希 `.jit`（sha256 `e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720`，
   2.27MB）并注入分段器，不阻塞。但 handoff §1 要求「记下 commit（如 `3245b1a`）」，目前
   只能落到 artifact sha256、拿不到 repo commit。请确认是否可接受，或另给 `--silero-ref`。

## 5. 现场与基线

- 仓库：`/dataA/streamllm`，HEAD = `219b8909785d78d0d31cdd5495b2675ea61d7165`。
- `run_ttfa_unified.py` sha256 = `0a1ac1a709c043ebadce974fa21b2604e7b690918babfc66d68a36708cc452f0`。
- self-test 已跑通 **69 PASS / 0 FAIL**（但 69 项未覆盖真实 TTS 服务契约，故探活错配未暴露）。
- TTS 服务已在线：CosyVoice v2.0 容器（image `cosyvoice:v2.0`），commit `8555549e`，
  `127.0.0.1:20401` 监听正常，`/inference_sft` 路由可达。
- 未改任何代码、未跑冒烟、未提交；结果目录除 handoff 文档外无本机新产物。

## 6. 修复完成后

请开发侧修正并推送后通知 GPU 主机重跑（顺序不变）：

1. 任务 1 TTS 探活 → 期望 `"ok": true`、`payload_class: "pcm"`；
2. 任务 2 self-test → 期望 69 PASS / 0 FAIL；
3. 任务 3 三层冒烟 → 按 `R7_GPU_SMOKE_HANDOFF.md` §4 八项验收。
