# Gate材料验证复核报告（2026-08-22）

- **审查对象**：`experiments/review/20260821-PRE-PAPER-AUDIT/gate-material-verification-20260822.md`
- **审查目的**：判断放行前Gate材料是否齐备，能否出具正式`r7_main`书面放行。

## 1. 最终裁决

**当前不能出具无条件书面放行，也不能启动正式`r7_main`。**

本轮有一项重要进展：非末位fatal smoke的运行级证据已经真实通过。但Gate材料仍存在代码版本、clean状态、平台hash绑定和manifest完整性冲突。

| Gate | 结论 |
|---|---|
| 非末位fatal→cancelled | **通过** |
| TTS服务内容和新probe | **内容通过** |
| GPU self-test 90/0 | **测试通过，版本绑定未通过** |
| platform conditions内容 | **内容通过，run binding未通过** |
| clean/provenance | **不通过** |
| Gate manifest | **不通过** |
| 正式`r7_main` | **暂不放行** |
| 真实`r7_tts_control` | **暂不放行，继续保持后置** |

## 2. 已确认通过的Gate材料

### 2.1 非末位fatal smoke

`fatal_smoke/checkpoint_r7_smoke_fatal.jsonl`确实包含：

```text
success
error (fatal=true, fault_injection:asr_error)
4 × cancelled_after_fatal
```

四条cancelled记录的`events`、`chunk_log`和`tts`均为空，没有后续GPU执行污染。对应QA和RUNINFO的任务数、终态数与JSONL一致。这一项已关闭。

### 2.2 TTS服务provenance内容

`env/gate/tts_provenance/`已提供CosyVoice commit、dirty patch、镜像/模型信息、`spk2info.pt`和启动/依赖材料；新TTS probe也显示HTTP 200、PCM payload和严格策略匹配。

正式论文仍需披露“晓伊”映射到内置中文女embedding的限制。

### 2.3 GPU self-test内容

self-test归档可独立复算：

```text
90 PASS / 0 FAIL
```

但其代码版本绑定仍存在冲突，见下节。

## 3. 当前仍阻塞正式放行的问题

### 3.1 code commit不唯一且clean记录自相矛盾

当前材料至少出现以下版本：

- 当前checkout：`81f548f`；
- Gate manifest声明：`34ea12e`；
- `gate_clean_git.txt`和GPU self-test记录：`2e54ac2`；
- fatal smoke RUNINFO/JSONL：`34ea12e`；
- handoff修订：`128ff2e`。

此外，`gate_clean_git.txt`记录的porcelain中实际出现：

```text
?? env/gate/
?? env/platform_conditions.txt
```

随后“空=clean”的文字说明不能替代真实空输出。因此当前不能证明Gate材料是在一个统一、clean、可复现的代码基线上产生的。

**必须修复：**选定唯一正式`code_commit`，在该commit上完成clean checkout；重新生成/绑定self-test、fatal smoke、platform和manifest。不要继续沿用多个分叉版本混合放行。

### 3.2 platform conditions尚未绑定到fatal smoke

`env/platform_conditions.txt`内容本身包含双RTX 3090、驱动、CUDA、Triton fallback、显存和进程状态，内容可接受；但fatal smoke RUNINFO中的：

```text
platform_conditions_sha256 = null
```

所以不能证明fatal smoke和平台条件属于同一冻结环境。

**必须修复：**用同一platform conditions文件重新生成fatal smoke binding，或者明确该文件是在同一运行前采集并将其hash写入RUNINFO/checkpoint。

### 3.3 Gate manifest不完整且hash语义表述不准确

manifest列出的8项文件在将CRLF规范化为LF后hash一致，但工作树直接hash和`git hash-object`均不一致。因此manifest实际采用的是：

> LF-normalized文件内容SHA-256

不能称为Git blob hash或原始文件hash，必须在manifest中明确。

更重要的是，manifest没有覆盖全部作为Gate依据的材料，至少漏列：

- `gate_selftest_gpu.md`；
- fatal RUNINFO；
- fatal QA；
- fatal run log；
- 其他关键platform/provenance文件。

**必须修复：**重新生成完整manifest，覆盖实际放行依据的每个文件，统一hash规范，并在提交后再次核验。

### 3.4 当前checkout仍有Gate材料modified

当前工作树存在多个Gate/fatal材料modified。即便只是换行符或重新生成，也必须先决定最终归档内容，再提交到明确的artifact commit，保证最终审查副本和GPU产物不会继续漂移。

### 3.5 self-test版本绑定未统一

self-test 90/0本身可信，但记录绑定到`2e54ac2`，manifest使用`34ea12e`。在没有统一正式code commit前，不能把self-test作为正式放行材料。

## 4. `tts-control-only`的放行时序

当前只证明control代码及self-test可用。真实`r7_tts_control`依赖正式`r7_main`，因此应继续遵守：

1. 获得书面放行；
2. 执行`r7_main`；
3. `r7_main`结果级QA通过；
4. 再执行真实`r7_tts_control`。

不能用self-test的32条fake control记录替代正式control结果，也不能在`r7_main`之前执行真实control。

## 5. 修复后的最小放行材料包

开发侧不需要再改实验方案，只需：

1. 选定唯一clean正式`code_commit`；
2. 在GPU主机保存真正为空的`git status --porcelain`；
3. 在同一代码基线上重新执行GPU self-test，保存90/0、exit code、环境和hash；
4. 重新执行/绑定非末位fatal smoke，写入platform conditions hash；
5. 重新生成完整Gate manifest，覆盖：
   - clean记录；
   - self-test log/md；
   - fatal JSONL/RUNINFO/QA/log；
   - platform conditions；
   - TTS provenance；
   - TTS probe；
6. 明确manifest使用LF-normalized内容hash；
7. 提交最终artifact commit并再次确认工作树状态；
8. 将speaker mapping note和`spk2info.pt` hash纳入正式binding。

## 6. 最终结论

**非末位fatal smoke和TTS服务材料内容已通过，但正式Gate仍因版本不统一、clean记录错误、platform hash未绑定和manifest不完整而未关闭。**

完成上述最小材料包后，可以重新进行最后一次书面复核。若统一代码基线、所有hash和绑定均一致，将只放行：

- 正式`r7_main`：50条A/B主实验及10条子集补轮；
- `r7_main`结果级QA通过后，再执行真实`r7_tts_control`。

当前不得启动`r7_main`，也不得启动真实TTS control。
