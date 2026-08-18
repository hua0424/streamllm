# GPU 实验机全新安装指引（从裸机到可执行实验）

> **读者**：在全新 GPU 实验机上做初始化的人员（或照此执行的需求方本人）。
> **机器假设**：全新安装的 Ubuntu 22.04，2×RTX 3090，盘内无任何历史数据。
> **与任务书的关系**：本文是**操作手册**，负责"从零装到能跑"；装完后的实验执行以
> `experiments/GPU_EXPERIMENT_HANDOFF.md`（任务书）为准——从其 §1.2 环境核验继续。
> **原则**：每一步都有"✅ 检查点"，全部通过再进入下一步；任何一步失败，保留报错截图/文本
> 回给需求方，不要自行换方案（尤其是驱动版本、Python 版本、依赖来源）。

## 总览

| 步骤 | 内容 | 大约耗时 |
|---|---|---|
| A | 系统包 + NVIDIA 驱动 | 20–40 min（含重启） |
| B | 安装 uv | 2 min |
| C | 克隆仓库 + uv sync 依赖 | 15–40 min（下载约 8 GB） |
| D | `.env` 本机适配（两个必改项） | 5 min |
| E | 模型预下载（约 18 GB） | 30–90 min（视带宽） |
| F | 合成数据集上传（约 3.4 GB，唯一需要传输的数据） | 视传输渠道 |
| G | 安装自检（回归 10/10 + 冒烟 16/16 + 两次试跑） | 30 min |

**磁盘预算（共 ≥150 GB 建议余量）**：venv 约 8 GB（torch cu121）；HF 模型缓存约 18 GB
（turbo 1.6 + Qwen2-7B 15 + 小模型 1.5）；合成数据 3.4 GB×2（tar 包+解压）；R2 语料约 55 GB
（下载包 26 GB + 解压 29 GB）；实验结果若干 GB。建议仓库与 HF 缓存放同一块大盘。

---

## Step A：系统包与驱动

```bash
# A1 系统包（ffmpeg 是 whisper 解码音频的硬依赖，libsndfile1 是 soundfile 的依赖）
sudo apt update && sudo apt install -y git curl ffmpeg libsndfile1 build-essential tmux

# A2 NVIDIA 驱动（torch cu121 轮子要求驱动 ≥ 530；CUDA 运行库由 pip 包自带，
#    无需安装 CUDA Toolkit）
sudo ubuntu-drivers install        # 或: sudo apt install -y nvidia-driver-550
sudo reboot
```

重启后：

```bash
# A3 确认双卡，并选好大磁盘
nvidia-smi                         # 应看到 2×RTX 3090，驱动版本 ≥ 530
df -h                              # 选一块剩余 ≥150 GB 的盘，后文记为 /data（按实际替换）
```

**✅ 检查点 A**：`nvidia-smi` 显示双 3090；`ffmpeg -version` 有输出；已确定大盘路径。

---

## Step B：安装 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env        # 或按安装输出提示操作
uv --version
```

网络受限备选（仅直连失败时用，用完可 unset）：

```bash
export UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"   # pypi 镜像加速 uv sync
```

**✅ 检查点 B**：`uv --version` 正常输出。

---

## Step C：克隆仓库与依赖

```bash
cd /data                           # 你的大盘
git clone https://github.com/hua0424/streamllm.git
cd streamllm
git log -1 --oneline               # 记下 commit hash，回信时报告（实验可追溯）

uv sync                            # 自动按 .python-version(3.10.18) 建 .venv，下载约 8 GB
```

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.device_count())"
```

**✅ 检查点 C**：输出 `2.5.1+cu121 2`（版本号必须一致；设备数=2）。若设备数为 0 回 Step A 查驱动。

---

## Step D：`.env` 本机适配（两个必改项，新机必踩的坑）

`.env` 随仓库带入且被 git 跟踪，其中有两处旧机器遗留，**必须本机修改但不要提交**：

```bash
# D1 修改 .env：
#   HF_HOME="/mhh/model/hfhome"  →  改为 Step A3 选定的大盘路径，如 /data/hfhome（不存在的
#                                  旧主机路径，不改会导致模型下载到错误位置或失败）
#   HF_TOKEN="hf_...."           →  改为 HF_TOKEN=""（该 token 已失效，实测 whoami 401；
#                                  所有模型均为公开，空 token 即可下载）
nano .env
mkdir -p /data/hfhome              # 与你改的 HF_HOME 一致

# D2 防止误提交本地改动（可选但建议；本任务全程不需要 commit/push）
git update-index --skip-worktree .env
```

**✅ 检查点 D**：`grep -E "HF_HOME|HF_TOKEN" .env` 显示新路径与空 token；目录已创建。

---

## Step E：模型预下载（约 18 GB，建议在 tmux 里跑）

```bash
tmux new -s models

# E1 Whisper turbo（约 1.6 GB）
HF_TOKEN= uv run python -c "import whisper; whisper.load_model('turbo')"

# E2 Qwen2-7B（约 15 GB）
HF_TOKEN= uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2-7B-Instruct')"

# E3 Silero VAD 预缓存（torch.hub 首次下载会交互询问信任仓库，非交互 shell 直接 EOF 报错，
#    必须用 echo y 管道预缓存）
echo y | uv run python -c "from src.asr.streamaudio_segmenter import StreamAudioSegmenter; StreamAudioSegmenter()"

# E4 冒烟用小模型（tiny + Qwen2.5-0.5B，约 1.5 GB；Step G 自检要用）
HF_TOKEN= uv run python -c "import whisper; whisper.load_model('tiny')"
HF_TOKEN= uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B-Instruct')"

du -sh /data/hfhome                # 期望 ≈ 18 GB
```

说明：命令前缀 `HF_TOKEN=` 是临时置空环境里的失效 token（若 Step D 已把 .env 里的 token 清空，
此前缀可省；保留无害）。HuggingFace 直连慢/超时的备选：`export HF_ENDPOINT=https://hf-mirror.com`
后重试（也建议写进 .env 持久化）。

**✅ 检查点 E**：四个下载命令全部无 401/EOF 报错；HF 缓存目录 ≈18 GB。

---

## Step F：合成数据集上传（唯一需要从本机传的数据）

**先明确哪些东西不用传**（都已在 git 里，clone 即有）：

- `experiments/results/revision/r1_stats/repeat_subset_ids.json`（E1/E4/E5 用的 50 样本清单）✅ 已随仓库
- `experiments/results/revision/r3_baseline_la/exp2_ablation_sample_list.json`（E3 用的 498 条清单）✅ 已随仓库
- R2 真实语料（LibriSpeech/AISHELL-1/MUSAN）❌ 不要传——按任务书 §4-E2-0 在主机直接下载构建

需要传的只有一份：**合成实验数据集 `experiments/datasets/processed/` 整目录（约 3.4 GB，
1,133 条 JSON+WAV：multiwoz 630 + crosswoz 503）**，该目录被 .gitignore 排除、不在 git 中。

```bash
# F1 本机打包（Windows 本机 git-bash，在仓库根目录）
cd /d/project/my/research/streamllm
tar -czf processed.tar.gz -C experiments/datasets processed

# F2 传到主机（本机执行；user/host/端口按实际替换；也可走网盘/U盘）
scp processed.tar.gz user@<gpu-host>:/data/

# F3 主机解压到仓库（在 /data/streamllm 下）
tar -xzf /data/processed.tar.gz -C experiments/datasets/
ls experiments/datasets/processed/json   # 应看到 crosswoz multiwoz
```

**F4 验收（数量 + 抽验，与任务书 §1.1 相同）**：

```bash
ls experiments/datasets/processed/json/multiwoz  | wc -l   # 630
ls experiments/datasets/processed/audio/multiwoz | wc -l   # 630
ls experiments/datasets/processed/json/crosswoz  | wc -l   # 503
ls experiments/datasets/processed/audio/crosswoz | wc -l   # 503
```

再跑任务书 §1.1 中的"随机抽 15 个样本校验时长/文本"脚本（此处不重复，验收标准一致）。

**✅ 检查点 F**：四个计数为 630/630/503/503，抽验脚本 `problems 0`。
（crosswoz 比论文计数 502 多 1 条为历史运行失败样本，属正常，不要删除。）

---

## Step G：安装自检（全部通过才算装好）

```bash
# G1 修订回归套件（依赖 Step E3 的 silero 缓存；纯 CPU，约 1 分钟）
uv run python -m experiments.scripts.test_revision_regressions
# 期望最后一行: 10/10 passed

# G2 R2 构建链路冒烟（纯 CPU 离线，秒级）
uv run python -m experiments.scripts.test_r2_build_smoke
# 期望最后一行: 冒烟结果: 16/16 通过

# G3 小模型全链路试跑（tiny + Qwen2.5-0.5B，1 样本 × 3 模式，验证线程/队列/落盘正常）
uv run python -m experiments.scripts.run_exp_latency \
  --dataset crosswoz --max-samples 1 \
  --asr-model-size tiny --llm-model-name Qwen/Qwen2.5-0.5B-Instruct \
  --asr-device cuda:0 --llm-device cuda:1 \
  --output-dir /tmp/install_smoke
# 期望: 3 个模式均有结果、无 error 字段，打印延迟指标

# G4 正式配置试跑（turbo + Qwen2-7B，验证双卡分工 cuda:0/cuda:1 与锁定配置可加载；
#     注意 --suffix-segments 0 与设备参数都必须显式传——脚本默认 auto 会把两个模型都放到 cuda:0）
uv run python -m experiments.scripts.run_exp_latency \
  --dataset crosswoz --max-samples 1 \
  --asr-device cuda:0 --llm-device cuda:1 --suffix-segments 0 \
  --output-dir /tmp/install_smoke_full
# 期望: 3 个模式均有结果、无 error；运行时另开终端 nvidia-smi 可见两张卡分别有负载

# G5 环境存档（任务书 §1.2 要求，回传时要附）
mkdir -p experiments/results/revision
uv run pip list > experiments/results/revision/env_versions.txt
nvidia-smi >> experiments/results/revision/env_versions.txt
```

**✅ 检查点 G 汇总**：

| # | 检查项 | 通过标准 |
|---|---|---|
| G1 | 修订回归套件 | `10/10 passed` |
| G2 | R2 构建冒烟 | `16/16 通过` |
| G3 | tiny 全链路 | 1 样本 3 模式无 error |
| G4 | 正式配置试跑 | 1 样本 3 模式无 error，双卡有负载 |
| G5 | 环境存档 | `env_versions.txt` 已生成 |

---

## 完成 → 移交任务书

以上 A–G 全部通过后：

1. 勾完任务书 `GPU_EXPERIMENT_HANDOFF.md` §1.2 环境核验清单、对照 §1.3 配置锁定表；
2. **建议先启动 E2-0**（R2 语料下载约 26 GB + 构建，CPU 为主、可与其它事并行）；
3. 其余按任务书 §四 顺序执行（E1 → E3 → E4/E5，E6 需先部署 CosyVoice TTS，见任务书 §1.0 第 8 条）；
4. 全程不需要 commit/push；结果按任务书 §5.4 打包回传。

## 常见问题（FAQ）

| 现象 | 原因与处理 |
|---|---|
| 模型下载 401 Unauthorized | `.env` 中 HF_TOKEN 已失效：Step D 置空，或命令前缀 `HF_TOKEN=` |
| silero VAD 报 EOF / trust 提示卡死 | torch.hub 交互信任提示：用 Step E3 的 `echo y \|` 预缓存命令 |
| `torch.cuda.device_count()` 为 0 | 驱动未装好或未重启：回 Step A2；`dmesg \| grep -i nvidia` 查原因 |
| whisper 报 FileNotFoundError: ffmpeg | 系统缺 ffmpeg：`sudo apt install -y ffmpeg` |
| HF 下载极慢/超时 | `export HF_ENDPOINT=https://hf-mirror.com`；`uv sync` 慢用 UV_DEFAULT_INDEX 镜像 |
| git clone GitHub 失败 | 网络受限：配置代理后重试；不要改用第三方 fork 源 |
| 长任务怕断线 | 所有下载/实验放 `tmux` 会话里执行（`tmux new -s work`） |
| `git status` 显示 .env 被修改 | 正常（Step D 本机适配），已 skip-worktree 则不显示；**无论如何不要提交** |
