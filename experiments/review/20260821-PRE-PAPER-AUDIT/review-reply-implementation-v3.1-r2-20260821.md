# Gate 1 整改复核报告 r2（2026-08-21）

- **固定点**：`f7e219c`
- **被审提交**：`7323359`
- **回复函**：`reply-review-implementation-v3.1-20260821.md`
- **前序审查**：`review-implementation-v3.1-20260821.md`
- **审查范围**：W1核心实现、本地集成测试、W3/W4/W5整改及申请放行的“TTS探活 + 3条GPU冒烟”。

---

## 1. 最终裁决

本轮整改显著改善了代码，前序审查中的多数问题已经关闭：final-drain框架、整文件原子checkpoint、固定Silero启动门禁、`generate_with_meta()`、请求级Generator、分层schedule、TTS跨read格式识别以及W3/W4/W5均取得实质进展。

但独立复核仍发现数个会影响GPU冒烟有效性的高严重性问题。因此：

| 阶段 | 裁决 |
|---|---|
| W3 CV | **放行** |
| W4 WER/CER | **放行** |
| W5配对统计 | **放行** |
| 本机self-test/集成测试 | **可作为路径证据，但不能替代生产冒烟** |
| **独立TTS探活** | **放行** |
| **3条GPU冒烟** | **暂不放行** |
| 正式50×A/B | **不放行** |

TTS探活可以执行，因为它是独立、短时的协议检查；但其通过只证明服务状态码、header和payload基本格式可用，不证明统一runner的线程、final-drain、取消和正式Silero分段路径正确。

---

## 2. 已确认关闭的前序问题

### 2.1 checkpoint原子绑定

`run_ttfa_unified.py:1080-1156`已改为整文件tmp+fsync+replace快照，并对截断、重复主键、binding不匹配和旧run混入执行fail-closed。配置、schedule、git/env/model/Silero/TTS/sample/subset/audio-map已经进入binding。该项通过。

### 2.2 固定Silero启动参数门禁

正式/冒烟模式缺少`--silero-ref`或`--silero-dir`时会拒绝启动，不再默认使用`master`。该“启动门禁”本身通过；但固定模型尚未注入正式`StreamAudioSegmenter`，见P0-4。

### 2.3 `generate_with_meta()`与请求级Generator

`src/llm/stream_llm_inference.py:194-240,356-378`已支持可选`torch.Generator`并传给`torch.multinomial()`；EOS、正常结束和max_tokens路径已有方法级测试。runner也已在EOS判断之后记录first model token。该项通过。

### 2.4 分层schedule

主实验已按`(language, duration_group)`尽可能平衡AB/BA，stratum内差值不超过1，全局25/25；重复子集三轮顺序也已冻结。该项通过。

### 2.5 TTS跨read与对齐

跨read RIFF、前导空白JSON、HTML、奇数总字节、大read、低于playable阈值和空body均有测试，整体整改有效。仍有deadline取消问题，见P1-3。

### 2.6 W3/W4/W5

- W3已检查逐文件主键唯一、三轮键集一致及关键config一致；结果与锚点一致。
- W4已实现共同sample-ID交集、paired manifest、`reference_full`强制和S/D/I/N；现有集合无排除，旧汇总数字不变。
- W5已补重复键、空配对、LA mode和R5 ID检查；21项结果不变。

三项均可放行。

---

## 3. 阻塞GPU冒烟的问题

## P0-1：final-drain仍存在collector/transcriber竞态

位置：

- `experiments/scripts/run_ttfa_unified.py:688-699`
- `run_ttfa_unified.py:714-718`
- `run_ttfa_unified.py:725-738`

collector先设置`pipeline_input_close_ns`，之后才检查/修改`waiting_segment_queue`、`segment_queue`并设置drain标志；transcriber同时无锁读取关闭状态、队列和标志。

可能发生：

1. collector设置closed；
2. transcriber看到closed且当前waiting为空、drain标志尚未设置，直接退出；
3. collector随后才把segment queue尾段标为final并设置drain。

这样真实尾段可能未执行final ASR。现有self-test没有强制该线程交错。

### 必须修复

- 用同一lock/condition保护“close发布、队列final化、drain状态”和transcriber退出判定；或
- collector必须先完成final化和drain状态设置，再原子发布pipeline closed；
- transcriber退出必须依赖明确的`drain_complete`，而不是仅依赖`closed && queue empty`；
- 新增确定性交错测试：在旧closed写入点暂停collector，确认transcriber不能提前退出；
- 使用真实ASRCache协议断言尾文本进入最终prompt。

---

## P0-2：System A主线程ASR/LLM异常没有fail-stop

位置：

- `run_ttfa_unified.py:977-986`
- 主循环停止判断：`run_ttfa_unified.py:2233-2235`

System A full ASR、`cache_prompt()`和`generate_with_meta()`在主线程执行。异常进入通用`except`后，仅当`exc_q`非空时才设置`fatal=True`；主线程异常不会进入worker exception queue，因此可得到：

```text
terminal_state=error
fatal=False
```

随后主循环继续运行后续GPU任务，违反模型/GPU异常后的fail-stop协议。

### 必须修复

- 明确错误分类；ASR、LLM、CUDA、模型状态异常无条件`fatal=True`；
- 仅预先定义的输入/TTS可恢复错误允许`fatal=False`；
- 增加System A ASR错误、LLM cache错误和generation错误故障注入测试；
- 测试必须断言后续schedule任务被写为cancelled，而不是继续执行。

---

## P0-3：恢复checkpoint时不会恢复fatal-stop状态

位置：

- `run_ttfa_unified.py:1118-1129`
- `run_ttfa_unified.py:2187-2199`
- `run_ttfa_unified.py:2232-2235`

checkpoint加载只恢复主键对应的terminal state，不恢复历史记录中的`fatal=True`。新进程把`fatal_stop=False`重新初始化；如果进程在写入fatal记录后、补写后续cancelled记录前崩溃，恢复后会跳过fatal记录并继续执行剩余GPU任务。

### 必须修复

- checkpoint加载后扫描所有终态记录；任何`fatal=True`均恢复为run级fatal-stop；
- 如果fatal后仍有未完成schedule键，只能补写cancelled，不得执行；
- binding/header可增加`run_fatal`状态并原子持久化；
- 增加“写入fatal后模拟进程崩溃，再恢复”的负向测试。

---

## P0-4：固定Silero只用于PSE，没有注入正式流式分段器

位置：

- 固定Silero加载：`run_ttfa_unified.py:2130-2143`
- 创建正式segmenter：`run_ttfa_unified.py:2163-2166`
- `src/asr/streamaudio_segmenter.py:122-136`

主程序按固定ref/dir加载的Silero实例用于PSE预扫描；随后正式System B使用`StreamAudioSegmenter()`，该类在构造函数中再次从默认`snakers4/silero-vad`加载模型，没有使用前面固定的model/utils。

因此checkpoint记录的Silero meta可能不是实际流式分段所用模型，直接影响VAD触发、flush和TTFA。

### 必须修复

- `StreamAudioSegmenter`支持注入已加载的固定model/utils，或支持固定repo/ref/dir参数；
- W1创建segmenter时必须传入与PSE一致并已hash的Silero artifact；
- RUNINFO分别记录PSE和stream segmenter模型hash，并断言二者一致；
- 集成测试应验证不会触发第二次浮动hub加载。

---

## P0-5：`--smoke`允许零样本/少样本仍成功退出

位置：

- `run_ttfa_unified.py:2084-2108`
- `run_ttfa_unified.py:2240-2259`
- `run_ttfa_unified.py:2274-2276`

smoke模式绕过样本清单完整性检查。如果smoke ID全部不存在，程序仍可完成模型加载和TTS探活，随后写空summary，QA返回0问题，最终exit 0。这会形成GPU冒烟假阳性。

### 必须修复

- smoke必须精确命中预期样本数；零样本或少于要求立即非零退出；
- smoke选择必须验证中英文覆盖及A/B任务数量；
- 故障注入任务必须实际生成预期error终态；
- QA需要断言成功路径和故障路径均执行，而非只看`n_err`。

---

## 4. 冒烟前强烈建议修复的问题

### P1-1：schema未验证TTS文本派生字段一致性

位置：`run_ttfa_unified.py:563-568`

当前只检查字段存在和大于零，没有验证：

```text
tts_n_chars == len(tts_text)
tts_n_bytes_utf8 == len(tts_text.encode('utf-8'))
tts_text_sha256 == sha256(tts_text)
tts_text_source与mode/fallback一致
```

构造错误字节数和bogus hash仍可通过`validate_record()`。应在冒烟前补齐并增加负向self-test。

### P1-2：TTS total deadline不能主动打断阻塞中的HTTP read

位置：

- `run_ttfa_unified.py:348-410`
- `run_ttfa_unified.py:837-876`

`total_deadline_ns`只在`iter_content()`返回chunk后检查。服务器返回headers后不发body时，线程仍会阻塞到requests read timeout。cancel event不能从外部关闭局部response。

建议：

- read timeout动态限制为pair剩余时间和配置read timeout的较小值；
- response句柄放入可由外层取消逻辑访问的holder，取消时主动`close()`；
- TTS worker必须在进入下一task前确认退出；
- 增加headers后停发body的慢流测试。

该问题不阻塞独立探活，但会影响故障冒烟隔离，建议在冒烟前完成。

### P1-3：本地集成测试证据应降级表述

`ttfa_local_integration.py`确实运行了真实tiny Whisper、真实Qwen2-0.5B和真实`generate_with_meta()`，有路径价值；但：

- 输入是宽带噪声+静音，Whisper转写为空；
- PSE使用energy fallback，而非真实Silero终点；
- 真实final-drain本次未触发；
- TTS是fake HTTP PCM；
- Qwen路径没有做真实AB/BA同seed重复性比较；
- 脚本硬编码Windows Qwen路径和CUDA，不能直接在通用Linux GPU主机复现。

因此回复函应称“真实CUDA组件加载与A/B路径集成检查”，不能称完整生产协议验证。

### P1-4：self-test数量表述不一致

实际输出可统计为55个PASS行，而回复函写56项。虽不影响测试成功，但应修正文档计数或让脚本显式输出测试总数，避免审计时产生证据不一致。

---

## 5. TTS探活放行范围

允许在GPU主机执行**独立TTS探活**，前提是：

1. 不启动50条或3条A/B任务；
2. 保存status、Content-Type、Content-Encoding、payload classifier结果、PCM服务配置和探活时间；
3. 探活失败不得临时放宽正式允许策略；
4. 探活结果只用于确定正式允许header/payload策略；
5. 不把探活通过解释为TTS慢流、取消、首包或TTFA链路通过。

探活产物提交后可与上述代码修复一起复核。

---

## 6. W3/W4/W5最终状态

| 工作项 | 结论 | 备注 |
|---|---|---|
| W3 | **放行** | 数字与锚点一致，逐文件唯一、键集和config检查已补 |
| W4 | **放行** | paired manifest、共同ID过滤及`reference_full` fail-closed已补，旧数字不变 |
| W5 | **放行** | 重复键、空配对、LA mode和R5唯一性已补，21项结果不变 |

生成CSV存在CRLF被`git diff --check`报告为trailing whitespace的问题，这不影响数值，但建议统一CSV换行或在仓库规则中明确，不作为本轮冒烟阻塞项。

---

## 7. 重新申请3条冒烟的最低条件

必须先完成：

1. final-drain竞态原子化；
2. System A ASR/LLM异常无条件fatal；
3. checkpoint恢复fatal-stop；
4. 固定Silero注入正式`StreamAudioSegmenter`；
5. smoke精确样本/语言/任务覆盖校验；
6. TTS文本派生字段一致性schema；
7. 建议同步完成HTTP阻塞read可取消。

并提交以下证据：

- 更新后的self-test，包含上述负向用例；
- 真实ASRCache final-drain竞态测试；
- fatal checkpoint恢复测试；
- segmenter固定Silero hash一致性测试；
- 零命中/少命中smoke拒绝测试；
- System A ASR/LLM故障注入测试；
- py_compile和diff check。

完成后可重新评估3条GPU冒烟。冒烟通过结果级QA后，才放行正式50×A/B。

---

## 8. 本轮结论

**整改有效但尚未完全关闭Gate 1。**

本轮仅放行独立TTS探活，不放行3条GPU冒烟。最关键的剩余问题是：final-drain线程竞态、System A模型异常未fail-stop、fatal checkpoint恢复缺失，以及固定Silero未用于正式流式分段器。这些问题均可能产生看似成功但不可采信的冒烟记录，必须先修复。
