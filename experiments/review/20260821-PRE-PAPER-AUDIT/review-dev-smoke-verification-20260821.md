# GPU 冒烟结果复核报告（2026-08-21）

- **审查对象**：`experiments/review/20260821-PRE-PAPER-AUDIT/dev-smoke-verification-20260821.md`
- **审查范围**：GPU冒烟原始产物、RUNINFO、checkpoint、TTS probe、运行日志、handoff及正式runner配置。
- **目的**：判断冒烟是否通过 Gate 1，以及是否可以放行正式50条 A/B实验。

## 1. 总体裁决

本次 GPU 冒烟可以确认**功能路径已通过**，但当前证据还不足以直接宣布“正式实验已经获得可追溯放行”。

| 阶段 | 裁决 |
|---|---|
| GPU冒烟功能路径 | **通过** |
| 独立TTS探活 | **通过** |
| 冒烟结果内部一致性 | **基本通过** |
| 正式50条 A/B实验 | **暂缓，完成第5节正式运行前Gate后放行** |
| 论文数据锁定/论文修改 | **不放行** |

冒烟已验证核心配置、A/B路径、固定Silero注入、TTS请求和成功/故障处理，但存在以下正式运行前必须解决的 provenance/交付问题：

1. 冒烟实际运行在 `git_dirty=true` 的工作树上，未保存dirty patch或文件清单；
2. 冒烟运行代码实际绑定为 `1a0ddc8`，而验证报告将结果产物提交 `b1e1206`写成核验对象，代码commit与产物commit混淆；
3. TTS服务端代码、镜像、模型和本地音色补丁未完整绑定；
4. handoff要求的 `--tts-control-only` 当前runner不存在；
5. CUDA/Triton fallback、TTS与ASR共用 `cuda:0`等运行条件需要作为正式平台条件固定并记录。

这些问题不要求重新设计TTFA算法，也不要求重新做冒烟功能测试；主要是正式运行前的版本、环境、服务和交付范围收口。

---

## 2. 冒烟配置核对

从RUNINFO、checkpoint和运行日志核对，冒烟配置与R7 TTFA协议一致：

| 项目 | 冒烟值 | 判断 |
|---|---|---|
| ASR | Whisper `turbo` | 一致 |
| ASR设备 | `cuda:0` | 一致 |
| LLM | `Qwen/Qwen2-7B-Instruct` | 一致 |
| LLM设备 | `cuda:1` | 一致 |
| chunk | 500 ms | 一致 |
| prefix/suffix | 1 / 0 | 一致 |
| recognition threshold | 2.0 s | 一致 |
| max_tokens | 128 | 与R7 TTFA协议一致 |
| temperature/top_p | 0.1 / 0.9 | 一致 |
| repetition penalty | requested 1.1；effective not applied | 已披露，论文不得写成实际生效 |

证据：

- `experiments/results/revision/r7_ttfa_unified/RUNINFO_r7_smoke.md`
- `experiments/results/revision/r7_ttfa_unified/r7_smoke_run.log`
- `experiments/results/revision/r7_ttfa_unified/checkpoint_r7_smoke.jsonl`

注意：`max_tokens=128`只应绑定本轮R7统一TTFA实验，不应回写成E1/E2/E3等历史一般实验的统一配置。

---

## 3. 已确认通过的冒烟证据

### 3.1 固定Silero注入

RUNINFO和checkpoint记录：

- PSE与正式segmenter使用同一artifact；
- `segmenter_silero_injected=true`；
- artifact SHA-256：

```text
e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720
```

这足以证明本次冒烟的PSE和正式流式segmenter没有各自加载不同的Silero权重。

但该Silero目录不是Git checkout，`repo_commit=null`；因此正式材料应称“由artifact SHA-256冻结运行权重”，不要声称已经具备Silero源码commit级可复现性。

### 3.2 TTS探活及正式请求

探活和冒烟正式请求的客户端策略一致：

- endpoint：`http://127.0.0.1:20401/inference_sft`；
- speaker请求值：`晓伊`；
- speed：0.8；
- HTTP status：200；
- Content-Type/Encoding：均为None；
- payload分类：PCM；
- 成功样本均收到有效PCM。

探活产物：

- `experiments/results/revision/r7_ttfa_unified/tts_probe.json`

探活通过只能证明协议和payload基本可用，不代表正式TTS首包方差或端到端TTFA已经通过。

### 3.3 冒烟路径

冒烟结果已验证：

- A/B两模式执行；
- 中英文覆盖；
- 成功和故障路径；
- fatal后的cancelled补写；
- 固定Silero注入；
- TTFA/schema/事件和checkpoint相关QA。

因此可以关闭“smoke零样本假阳性”和“只加载模型不执行A/B”的前序风险。

---

## 4. 正式放行前的阻塞项

### 4.1 冒烟运行在dirty工作树

冒烟RUNINFO记录：

```text
git_commit = 1a0ddc83d3082ddedc443695d0be0da58669705c
git_dirty = true
```

仅保存`dirty=true`不能证明实际运行代码。必须补充以下之一：

- GPU主机当时的`git diff`/patch文件；或
- dirty文件清单及每个文件hash，并证明只有允许的本地配置变化；或
- 在clean、明确批准的commit上重新启动正式run。

正式实验不得从dirty smoke checkpoint续跑。必须新建：

```text
run_id = r7_main
```

并使用新checkpoint。

### 4.2 运行commit和产物commit混淆

冒烟实际运行代码绑定为：

```text
1a0ddc8
```

而：

- `b1e1206`是结果产物提交；
- `cdeb927`是开发侧结果核验提交；
- 当前审查副本HEAD另有其值。

`dev-smoke-verification-20260821.md`不应将`b1e1206`描述为实际运行代码commit。正式记录必须拆分：

```text
code_commit
result_artifact_commit
verification_commit
```

### 4.3 TTS服务端 provenance不完整

当前仅能确认客户端请求和响应策略，尚未绑定：

- CosyVoice服务代码commit；
- Docker镜像digest；
- 模型snapshot/revision/hash；
- 修改后的`spk2info.pt` hash；
- 服务启动命令、配置和依赖版本。

此外，TTS handoff说明“晓伊”实际是本地`spk2info.pt`中指向内置“中文女”的别名，不等同于原始论文音色embedding。正式实验可以继续使用该配置，但论文/结果必须明确：

> The requested speaker ID was mapped by the local service configuration to the built-in Chinese female speaker embedding.

不能无条件声称与原论文音色完全相同。

### 4.4 handoff要求的TTS控制参数尚未实现

`R7_FORMAL_RUN_HANDOFF.md`要求主实验后执行：

```text
--tts-control-only
```

但当前`run_ttfa_unified.py`没有该CLI参数。因而正式交付范围和代码能力不一致：

- 主A/B路径存在；
- 匹配文本TTS控制路径尚未实现。

正式实验前二选一：

1. 实现并审查`--tts-control-only`；或
2. 从handoff和正式验收清单中删除该项，并在论文中将TTS策略控制降级为未完成/限制。

不能让GPU操作者自行临时改造。

### 4.5 CUDA/Triton fallback及资源条件

运行日志出现CUDA/Triton相关fallback：

```text
Failed to launch Triton kernels...
falling back to a slower median/DTW implementation
```

这不阻塞功能冒烟，但会影响绝对时延条件。正式实验必须：

- 将该fallback作为第二平台固定运行条件记录；
- 不与原平台绝对毫秒数混排；
- 记录GPU驱动、CUDA、CPU/OS和相关运行状态；
- 实验期间确保没有其他GPU任务；
- 记录ASR和TTS共用`cuda:0`的显存/进程状态。

---

## 5. 正式运行前Gate

在书面放行正式50条实验前，必须完成并保存：

1. GPU主机clean工作树，或完整dirty patch及允许项说明；
2. 明确批准的`code_commit`；
3. 新`run_id=r7_main`和新checkpoint；
4. 正确区分code/artifact/verification commit；
5. 新一轮TTS探活，生成正式run绑定的probe文件；
6. 固定Silero artifact SHA-256并核对PSE/segmenter一致；
7. TTS服务commit/镜像/模型/`spk2info.pt`配置hash；
8. CUDA/Triton fallback、双3090、ASR/TTS GPU共存和无其他作业记录；
9. 实现`--tts-control-only`，或从handoff中明确删除该交付项；
10. 将“晓伊→中文女”的speaker映射写入RUNINFO和正式论文限制说明；
11. 审查方完成一次正式书面放行记录。

### 可接受的正式run命令边界

正式run必须：

- 使用新checkpoint，不从smoke续跑；
- 显式提供固定Silero来源；
- 显式传入样本清单和配置；
- 启动前完成TTS探活；
- 保存环境、配置、hash和资源状态；
- 任何fatal、thread leak、pair timeout、schema错误都立即停止并保留终态记录。

---

## 6. 论文放行边界

即使正式run通过，也只能在后续结果级QA通过后修改论文。当前冒烟不能提供论文TTFA数字，也不能替代50条正式样本。

正式论文仍需保留以下限定：

- R7使用`max_tokens=128`，与历史E1/E2/E3的50不同；
- 第二平台存在CUDA/Triton fallback，绝对毫秒值只绑定该平台；
- TTS speaker请求名“晓伊”是服务内置“中文女”别名映射；
- TTFA是first-playable-PCM，不是声卡真实first-audible；
- A/B包含各自不同的TTS启动策略；
- 不把R7绝对值与旧平台绝对值混排；
- repetition penalty请求值1.1但实际未应用。

---

## 7. 最终裁决

**冒烟功能通过，但正式实验暂缓。**

允许开发侧继续完成正式运行前的 provenance Gate；完成后可申请书面放行正式50条A/B、10条重复子集和匹配文本控制。当前不需要重新设计实验，也不需要重复做本次冒烟功能测试，除非正式clean基线或TTS服务配置发生变化。
