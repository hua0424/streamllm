# TTS 服务部署 handoff（E6 前置）

> 2026-08-20 部署于 GPU 实验机（Ubuntu 22.04，2×RTX 3090）。
> 用途：为实验 E6（TTS 首包延迟测量）提供 CosyVoice `/inference_sft` 流式 TTS 服务。
> 详细部署记录见 `/dataA/app/cosyvoice/DEPLOY_NOTES.md`（宿主机，不在 git 仓库）。

## 一、服务状态与启停

服务当前**已启动并验证可用**（容器 `cosyvoice` 运行中，模型已加载，占 GPU 显存约 3.2GB）。

```bash
# 查看状态 / 日志
docker ps --filter name=cosyvoice
docker logs cosyvoice --tail 50        # 正常应有 "CosyVoice model loaded successfully" + "Uvicorn running on http://0.0.0.0:20401"

# 启动 / 停止（在部署目录下）
cd /dataA/app/cosyvoice
docker compose up -d                    # 启动
docker compose down                     # 停止
docker compose restart                  # 重启（改模型/配置后）

# 容器 restart: no（与原 WSL 配置一致），机器重启后需手动 docker compose up -d
```

## 二、E6 探活与运行（⚠️ 关键：URL 用 127.0.0.1）

**Linux 宿主机上 `host.docker.internal` 不可解析**（它只在容器内有定义，extra_hosts 提供）。
E6 脚本 `measure_tts_first_chunk.py` 默认 `TTS_URL=http://host.docker.internal:20401`，
**在宿主机跑必须显式传 `--url http://127.0.0.1:20401`**，否则探活报"服务不可达"。

```bash
# 1) 探活（可选，脚本正式运行会自动先探活）
uv run python experiments/datasets/tools/tts.py --url http://127.0.0.1:20401 --test

# 2) 单条合成测试（可选）
uv run python experiments/datasets/tools/tts.py --url http://127.0.0.1:20401 \
  --text "你好，测试一下" --spk-id "晓伊" --output /tmp/tts_test.wav

# 3) E6 正式测量（前置：E4 已产出 full_response）
uv run python -m experiments.scripts.measure_tts_first_chunk \
    --from-e4 experiments/results/revision/r4_commit --n-zh 25 --n-en 25 \
    --url http://127.0.0.1:20401 \
    --output experiments/results/revision/r6_ttfa/tts_first_chunk.csv
```

## 三、说话人适配说明（⚠️ 需回告需求方，已做本地别名补丁）

- 官方模型 `FunAudioLLM/CosyVoice-300M-SFT` 内置 7 个说话人：中文女/中文男/英文女/英文男/日语男/粤语女/韩语女，**没有"晓伊/云皓"**。
- 论文数据生成口径（tts.py、measure_tts_first_chunk.py、TTS_USAGE.md）全部用 `spk_id="晓伊"`，直接调用会 `KeyError: '晓伊'`。
- 已做本地补丁：`/dataA/app/cosyvoice/models/CosyVoice-300M-SFT/spk2info.pt` 添加别名
  `晓伊→中文女`、`云皓→中文男`（原文件备份为 `spk2info.pt.orig`）。
  **服务端接受"晓伊/云皓" ID，音色为内置中文女/男。**
- **影响评估**：E6 测的是首包延迟（TTFC/total/rtf），**音色不影响延迟指标**，测量结果有效。
  若需求方要求音色与论文完全一致，需提供原 WSL 的 spk2info.pt 或"晓伊"embedding。

## 四、已生成样例音频（验证用，可听）

- `/dataA/app/cosyvoice/sample_audio/sample_zh.wav`（中文，12.98s）
- `/dataA/app/cosyvoice/sample_audio/sample_en.wav`（英文，12.42s）

## 五、服务依赖与注意事项

1. **模型**：`CosyVoice-300M-SFT`（2.5GB）位于 `/dataA/app/cosyvoice/models/CosyVoice-300M-SFT`，以 volume 挂载进容器 `/pretrained_models`。**不要删除**；改模型后需 `docker compose restart`。
2. **镜像**：`cosyvoice:v2.0` 为本机重建（12.6GB），Dockerfile 与构建上下文在 `/dataA/app/cosyvoice/`。原 WSL 镜像已删无法恢复。
3. **端口**：20401（compose 映射 20400-20499 全段，与原配置一致）。
4. **GPU**：模型加载在 cuda:0（E1-E5 实验也用 cuda:0/cuda:1，TTS 推理只占用 3GB 显存，与实验可并行；E6 本身是 CPU/网络测量，穿插进行无冲突）。
5. 服务不可达时的排查顺序：`docker ps` 看容器在不在 → `docker logs cosyvoice` 看模型加载日志 → 确认 20401 端口监听 → 确认模型目录存在。
