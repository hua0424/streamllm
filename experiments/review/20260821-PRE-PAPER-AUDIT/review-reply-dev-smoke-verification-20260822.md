# GPU冒烟整改回复复核报告（2026-08-22）

- **审查对象**：`experiments/review/20260821-PRE-PAPER-AUDIT/reply-review-dev-smoke-verification-20260821.md`
- **对应冒烟产物**：`experiments/results/revision/r7_ttfa_unified/`
- **前序报告**：`review-dev-smoke-verification-20260821.md`

## 1. 总体裁决

最新回复函和代码已落实大部分上一轮整改，且本机复跑 `run_ttfa_unified --self-test` 得到 **90 PASS / 0 FAIL**。但是，正式运行所需的 provenance、平台条件和服务端绑定证据仍未实际生成，非末位 fatal 后的 cancelled 也没有在GPU冒烟产物中验证。

因此：

| 阶段 | 裁决 |
|---|---|
| 冒烟功能路径 | **通过** |
| 本机self-test | **通过** |
| `tts-control-only`实现 | **通过实现级检查** |
| 正式50条 A/B | **暂不放行** |
| 论文数据锁定 | **不放行** |

这不是要求重新设计实验，而是要求完成正式运行前的可追溯性 Gate。

## 2. 已确认通过的整改

### 2.1 非末位 fatal 处理的代码机制

代码已具备 `fatal_seen`、恢复后停止未完成任务、补写 `cancelled_after_fatal` 的逻辑。相关位置：

- `experiments/scripts/run_ttfa_unified.py:1228-1230`
- `experiments/scripts/run_ttfa_unified.py:2567-2577`
- `experiments/scripts/run_ttfa_unified.py:2762-2774`
- `experiments/scripts/run_ttfa_unified.py:2814-2817`

但本次GPU smoke的fatal仍是最后一个任务，因此原始JSONL没有cancelled记录。代码机制可接受，运行级证据仍不完整。

### 2.2 self-test

本机执行结果为：

```text
90 PASS / 0 FAIL
```

新增覆盖包括fatal恢复、smoke样本校验、`tts-control-only`和负向路径。该结果可作为代码复核证据，但仓库尚未保存带命令、commit、环境和退出码的独立self-test归档。

### 2.3 `tts-control-only`

CLI和`run_tts_control()`已实现，本机self-test通过，control binding包含主实验结果hash。该实现级问题已关闭。

不过，真实GPU主实验尚未产生control checkpoint、RUNINFO和CSV，因此正式交付仍需现场验收。

## 3. 正式运行前仍未关闭的阻塞项

### 3.1 正式clean/provenance尚未生成

旧smoke产物仍记录：

```text
git_commit = 1a0ddc8...
git_dirty = true
```

证据：

- `r7_ttfa_unified/checkpoint_r7_smoke.jsonl:0`
- `r7_ttfa_unified/RUNINFO_r7_smoke.md:8`

开发报告已澄清：

- `1a0ddc8`：实际运行代码；
- `b1e1206`：结果产物提交；
- `cdeb927`：验证提交。

但正式运行仍需要：

- GPU主机clean工作树，或完整dirty patch及文件hash；
- 明确的正式`code_commit`；
- 新的`run_id=r7_main`和checkpoint；
- 不得从dirty smoke checkpoint续跑。

### 3.2 TTS服务端provenance未绑定

当前只有客户端TTS探活信息，没有实际保存：

- CosyVoice服务代码commit；
- 本地diff；
- Docker镜像digest；
- 模型snapshot/revision/hash；
- 修改后的`spk2info.pt` hash；
- 启动命令、配置和依赖快照。

handoff中的G7仍是待现场执行的模板命令，不是已经生成的证据。正式运行前必须实际采集并纳入run binding。

### 3.3 平台条件文件尚未生成

代码已经支持`--platform-conditions-file`并将hash写入binding，但仓库中还没有正式的platform conditions产物。需要现场采集并绑定：

- GPU、驱动、CUDA和Triton状态；
- `-lcuda`/Triton fallback是否启用；
- 双RTX 3090状态；
- ASR/TTS的GPU分配和显存；
- 实验期间是否有其他GPU进程；
- CPU、OS、线程和资源配置。

旧smoke日志出现Triton/DTW fallback，旧资源快照还显示GPU 0有CosyVoice进程。因此正式绝对时延必须绑定同样的平台条件，不能与原平台毫秒数混排。

### 3.4 非末位fatal后的cancelled缺少运行级证据

当前smoke的fatal在最后一个任务，实际补写0条cancelled。这不能证明后续任务会正确写入：

```text
terminal_state=cancelled
error=cancelled_after_fatal
```

建议在正式50条前补一次独立小smoke：将fatal放在非末位，至少保留一个成功任务和一个后续cancelled任务；该测试不应混入正式结果。

### 3.5 speaker映射需要正式产物披露

代码已支持写入speaker mapping note，但旧smoke产物是旧代码生成的，尚未包含该注记。TTS服务实际将请求的“晓伊”映射为内置“中文女”embedding。

正式RUNINFO、checkpoint和论文限制中必须明确该映射，不能声称实际音色与原论文embedding完全一致。

## 4. 其他交付一致性问题

正式handoff要求主实验后执行：

```text
--tts-control-only
```

该参数现在已经实现，问题已不再是“缺少CLI”。但仍需在GPU主机生成真实control结果，并确认control binding也绑定platform conditions；当前control分支对`--platform-conditions-file`的读取路径需要现场确认，不能只依赖主实验配置推断。

另外，self-test虽然已通过，但建议保存以下不可变归档：

- 执行commit；
- 完整90项输出；
- 命令行和退出码；
- Python/依赖环境；
- 输出文件hash。

## 5. 正式50条实验放行条件

正式运行前必须完成：

1. clean工作树或完整dirty patch绑定；
2. 明确code/artifact/verification commit三元组；
3. 新`r7_main` checkpoint和RUNINFO；
4. 正式启动时重新TTS探活；
5. TTS服务commit、镜像、模型和`spk2info.pt` hash；
6. platform conditions文件及hash；
7. 明确并记录Triton/CUDA fallback；
8. 确认无其他GPU作业，记录ASR/TTS共用`cuda:0`资源状态；
9. speaker映射写入正式产物；
10. 非末位fatal小smoke或等价运行级cancelled证据；
11. self-test归档；
12. 审查方出具正式书面放行记录。

完成上述条件后，R7主A/B管线可以申请正式50条、10条重复子集和匹配文本控制实验；无需新增数据集或重新设计核心实验。

## 6. 最终结论

**冒烟功能和代码整改基本通过，但正式运行前Gate尚未完成，因此当前不放行正式50条A/B。**

当前最重要的不是再修改实验逻辑，而是生成真实的`r7_main` clean/provenance、TTS服务绑定、platform conditions、非末位fatal/cancelled证据和self-test归档。完成后再进行一次正式放行复核。
