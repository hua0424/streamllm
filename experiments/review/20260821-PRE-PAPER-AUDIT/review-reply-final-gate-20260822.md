# 最终 Gate 回复函复核报告（2026-08-22）

- **审查对象**：`experiments/review/20260821-PRE-PAPER-AUDIT/reply-review-final-gate-20260822.md`
- **前序报告**：`review-reply-dev-smoke-verification-20260822.md`
- **目的**：判断能否出具正式50条A/B实验的书面放行。

## 1. 最终裁决

**当前仍不出具正式50条A/B实验的书面放行。**

最新回复函已经修复handoff中的流程循环，并完成多项实现级工作；但是书面放行前要求的GPU现场Gate产物仍未实际生成。现有材料主要是“将要执行的命令和模板”，不能替代运行证据。

当前状态：

| 项目 | 裁决 |
|---|---|
| 旧GPU冒烟功能路径 | **通过** |
| 本机self-test | **90 PASS / 0 FAIL，通过** |
| 非末位fatal回填代码 | **实现级通过** |
| speaker/platform binding代码 | **实现级通过** |
| `tts-control-only` | **实现级通过** |
| GPU clean-tree放行前Gate | **未完成** |
| 正式50条A/B | **暂不放行** |
| 论文数据锁定 | **不放行** |

## 2. 已确认完成

### 2.1 Handoff流程循环已修复

当前handoff已明确：

1. G1–G8环境/provenance采集；
2. 非末位fatal smoke；
3. GPU clean-tree self-test；
4. 提交以上证据进行最终复核；
5. 获得书面放行后才执行`r7_main`；
6. 主实验完成后执行TTS control。

这一顺序正确。特别说明：**`r7_main`产物不属于书面放行前必须存在的证据**，否则会再次形成循环。书面放行前需要的是clean binding、服务和平台证据、独立fatal smoke及GPU self-test；`r7_main`只能在放行后生成。

### 2.2 本机self-test真实通过

归档日志可独立复算为：

```text
90 PASS / 0 FAIL
```

但该归档执行于dirty开发树，HEAD为旧提交，因此只能证明开发侧实现通过，不能替代GPU clean-tree self-test。

### 2.3 实现级整改

代码已具备：

- `--inject-fault-index`；
- fatal后补写`cancelled_after_fatal`；
- speaker mapping注记写入binding/RUNINFO；
- platform conditions hash绑定；
- `--tts-control-only`；
- control结果绑定主实验checkpoint hash。

这些实现可以进入GPU现场Gate验证。

## 3. 书面放行前仍缺少的实际证据

### 3.1 GPU主机clean/provenance

旧smoke仍记录：

```text
git_commit = 1a0ddc8...
git_dirty = true
```

最新回复函没有提供新的GPU clean状态归档。必须实际保存：

- `git rev-parse HEAD`；
- `git status --porcelain`为空；
- 批准的`code_commit`；
- 如存在本地配置，保存非敏感配置摘要/hash，并证明无tracked代码修改；
- 不从旧`r7_smoke` checkpoint续跑。

### 3.2 TTS服务端provenance

当前仍只有旧客户端`tts_probe.json`，未发现以下实际产物：

- CosyVoice服务代码commit；
- 服务端本地diff；
- Docker image ID及可取得的registry digest；
- 模型snapshot/revision/hash；
- 修改后的`spk2info.pt` hash；
- 启动命令、容器挂载、环境变量的非敏感摘要；
- 服务依赖快照。

handoff中的G7是命令模板，不是完成证据。

### 3.3 Platform conditions

尚未发现正式：

```text
env/platform_conditions.txt
```

及其SHA-256。该文件至少应包含：

- CPU、OS、kernel和线程配置；
- 双RTX 3090、驱动和CUDA；
- Triton/DTW fallback状态；
- ASR、LLM、TTS的GPU分配和显存占用；
- 正式实验期间其他GPU进程；
- `nvidia-smi`原始快照；
- TTS与ASR共用`cuda:0`的说明。

### 3.4 非末位fatal/cancelled运行证据

旧smoke的fatal位于最后一条，原始结果为：

- success 5；
- error/fatal 1；
- cancelled 0。

代码具备回填机制，但仍需单独小smoke生成真实记录，例如：

- task 0：success；
- task 1：故障注入，`error + fatal=true`；
- 后续任务：`cancelled + error=cancelled_after_fatal`；
- JSONL、RUNINFO、QA和日志数量一致；
- cancelled任务没有ASR/LLM/TTS执行事件。

该run必须独立于正式结果。

### 3.5 GPU clean-tree self-test归档

需在上述批准候选commit的GPU clean树运行并保存：

- 完整命令；
- 完整90项输出；
- exit code；
- HEAD和clean状态；
- Python/依赖环境；
- 日志SHA-256。

现有dirty开发机归档不能替代此项。

### 3.6 Speaker mapping现场验证

代码已经支持注记，但旧smoke产物没有该字段。放行前应在GPU Gate/探活或配置归档中确认：

> 请求speaker ID“晓伊”由本地`spk2info.pt`映射到内置中文女embedding，并非原论文音色。

同时绑定修改后`spk2info.pt`的hash。

## 4. 不属于书面放行前阻塞的项目

以下产物按正确流程只能在正式放行后产生，当前不要求提前存在：

- `checkpoint_r7_main.jsonl`；
- `RUNINFO_r7_main.md`；
- `QA_r7_main.md`；
- `ttfa_summary_r7_main.csv`；
- 正式result artifact commit和verification commit；
- 真实`tts-control-only`结果。

但放行后必须：

1. 使用新`run_id=r7_main`和新checkpoint；
2. 绑定放行前生成的clean/provenance、TTS服务和platform conditions；
3. 正式启动时重新探活；
4. 主实验成功后，使用同一platform conditions运行TTS control；
5. 再生成正式code/artifact/verification三元组并做结果级QA。

## 5. 最小放行材料包

开发侧下一次只需提交以下放行前材料，无需再修改宏观方案：

1. `gate_clean_git.txt`：GPU HEAD和空的`git status --porcelain`；
2. `gate_selftest_gpu.log/.md`：GPU clean-tree 90/0归档；
3. `checkpoint_r7_smoke_fatal.jsonl`、对应RUNINFO/QA/log；
4. `env/platform_conditions.txt`及SHA-256；
5. TTS服务provenance目录：commit/diff/image/model/spk2info/startup/dependencies；
6. 新TTS probe及服务header/payload策略；
7. 一份Gate manifest，将上述文件hash与拟批准`code_commit`绑定。

提交后可进行最后一次书面放行复核。若全部一致，可直接放行`r7_main`，无需再次重复旧happy-path smoke。

## 6. 本轮结论

**流程文本和实现已基本准备完成，但放行前GPU现场证据尚未实际提交，因此正式50条A/B继续暂缓。**

下一步应执行handoff中明确允许的放行前Gate，而不是继续扩写回复方案。完成最小放行材料包后，再申请最终书面放行。
