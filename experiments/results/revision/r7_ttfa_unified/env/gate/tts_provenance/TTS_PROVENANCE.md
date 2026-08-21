# G7 — TTS 服务端 provenance（CosyVoice，127.0.0.1:20401）

采集时间：2026-08-22（GPU 主机）

## 镜像与容器
- 镜像：`cosyvoice:v2.0`
- 镜像 digest（images --digests）：`sha256:1c0974a47ece8e3f97c12d682e6d72e4c3669fd8aa2e6225107d70056d2ecc8b`
- 容器 ID：`8123c3629e3a`（name: cosyvoice）
- entrypoint：`/opt/nvidia/nvidia_entrypoint.sh`
- 启动命令：`cd /workspace/CosyVoice/runtime/python/fastapi && uvicorn server:app --host 0.0.0.0 --port 20401 --workers 1 --loop asyncio --http auto --timeout-keep-alive 60`
- 挂载：
  - `/dataA/app/cosyvoice/models` → `/pretrained_models`
  - `/dataA/app/cosyvoice/fastapi` → `/workspace/CosyVoice/runtime/python/fastapi`

## CosyVoice 服务代码
- 仓库 commit：`8555549e882236e6541748b1042d95693caa82ba`
- 本地修改（dirty）：`requirements.txt`（删 tensorrt-cu12* 三行）、
  `runtime/python/fastapi/server.py`（加 torch 导入 + 全局加载模型 + speed 参数 +
  async→sync 端点）。完整 diff 见 `server_dirty.patch`（163 行）。
- 其余未跟踪文件（非敏感，仅记录）：`requirements.txt.bak`、
  `runtime/python/fastapi/server - singlework.py`、`server_bak20251205.py`

## 模型与 speaker 表（/pretrained_models/CosyVoice-300M-SFT）
- `spk2info.pt` sha256：`1095c8d809d800584307b004dcd04853d0a77339694456da5925d14098fa3474`
  - 内含 speaker：中文女/中文男/日语男/粤语女/英文女/英文男/韩语女/晓伊/云皓
  - **晓伊 embedding 与 中文女 完全相等（max abs diff = 0.0，shape (1,192) float32）**
    —— 即「晓伊 → 内置中文女」映射的模型级证据。
- 模型权重 hash：
  - `llm.pt`：`d198ce56636e1eb1c9d0cb0d6e3529de8fdfd3fd45075c346296b0d6dcfc54ea`
  - `flow.pt`：`21eae78c105b5e1c6c337b04f667843377651b4bcfb2d43247ed3ad7fd0a3470`
  - `hift.pt`：`91e679b6ca1eff71187ffb4f3ab0444935594cdcc20a9bd12afad111ef8d6012`
  - `campplus.onnx`：`a6ac6a63997761ae2997373e2ee1c47040854b4b759ea41ec48e4e42df0f4d73`
  - `speech_tokenizer_v1.onnx`：`23b5a723ed9143aebfd9ffda14ac4c21231f31c35ef837b6a13bb9e5488abb1e`
  - `flow.decoder.estimator.fp32.onnx`：`f2b71b58497f56a5b5e8f2cacc8c2c5088b0fb0e8f9547e1a39269f0a98d0c92`

## 服务端环境（非敏感摘要）
- Python：3.10.20；关键依赖：torch 2.3.1+cu121 / torchaudio 2.3.1+cu121 /
  onnx 1.16.0 / onnxruntime-gpu 1.18.0 / fastapi 0.115.6 / uvicorn 0.30.0 /
  numpy 1.26.4 / pydantic 2.7.0。完整快照见 `tts_pip_freeze.txt`。
- 响应契约：`/inference_sft`（GET/POST，Form 参数 tts_text/spk_id/speed），
  `StreamingResponse` 返回裸 PCM（无 Content-Type 头，16-bit mono 22050Hz）。
