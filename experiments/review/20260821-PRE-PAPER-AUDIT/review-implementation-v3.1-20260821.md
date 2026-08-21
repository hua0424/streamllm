# v3.1 方案与首批实现代码审查报告（2026-08-21）

- **固定比较点**：`b60d907`
- **被审提交**：`f7e219c`
- **差异范围**：`git diff b60d907...HEAD`
- **规格来源**：
  - `dev-assessment-and-plan-20260821.md` v3.1
  - `reply-review-dev-assessment-and-plan-v3-20260821.md`
  - `review-dev-assessment-and-plan-v3-20260821.md`
- **审查范围**：W1统一TTFA、LLM元信息接口、W3 CV、W4 WER/CER、W5配对统计，以及W6/W7/W9材料。

---

## 1. 总体裁决

**v3.1方案本身可以接受，但首批代码不通过 Gate 1。当前不放行GPU探活、3条GPU冒烟或正式50×A/B实验。**

已经确认：

- `run_ttfa_unified --self-test`：33项PASS；
- W3/W4/W5各自self-test：PASS；
- Python编译检查：PASS；
- diff whitespace检查：PASS；
- W3当前结果数字正确；
- W4当前macro/corpus及S/D/I/N数字基本正确；
- W5当前21项统计结果可复算。

但这些测试不足以证明生产路径可用。W1 self-test的A/B成功路径使用fake Segmenter、fake ASR和fake LLM，没有覆盖真实ASRCache最终提交语义、真实`generate_with_meta()`、CUDA随机数隔离和真实TTS分片行为。代码中还有4个P0阻塞问题和多项高严重性缺口。

阶段裁决：

| 阶段 | 结论 |
|---|---|
| v3.1方案 | **通过** |
| 本机fake orchestration self-test | **通过，但证据范围有限** |
| W3结果 | **有条件通过** |
| W4结果 | **不通过冻结协议** |
| W5结果 | **有条件通过** |
| GPU/TTS探活 | **暂不放行** |
| 3条GPU冒烟 | **不放行** |
| 正式50×A/B | **禁止启动** |

---

## 2. Standards 轴

### 硬性问题

1. `experiments/scripts/run_ttfa_unified.py:1489-1490,1520-1524`仍硬编码模型和`cuda:0/cuda:1`。虽然这是复现实验锁定配置，但与仓库“配置放入`src/config.py`/`.env`、避免模块硬编码设备”的要求不一致。建议CLI显式默认值或从配置读取，同时把正式运行值写入RUNINFO。
2. 本次改变实验方法后尚未更新`experiments/EXPERIMENT_DESIGN.md`。v3.1已把它列入W8，因此可作为结果QA前必须完成项，不必阻塞当前代码修复。

### 判断项

- 实验脚本中的`print()`可以接受，但正式长时运行最好接入结构化日志。
- `experiments/review/`是用户明确指定的审查目录，虽然一般实验结果应放`experiments/results/`，本轮不将其视为阻塞违规。

---

## 3. Spec 轴：W1 Gate 1阻塞发现

## P0-1：`flush() -> None`仍可能静默丢失尾部ASR内容

位置：

- `experiments/scripts/run_ttfa_unified.py:547-552`
- `experiments/scripts/run_ttfa_unified.py:579-593`
- `src/asr/faster_whisper_streamer.py:531-540,616`

当前实现无条件发送`InputClosed`，解决了旧死锁的一部分，但没有保证真实`ASRCache`的未提交prefix/尾段被final路径消费。

当`flush()`返回None时：

1. 不再产生`is_final=True`音频段；
2. collector收到`InputClosed`；
3. transcriber在waiting queue为空后结束；
4. 真实ASRCache的`segment_queue`可能仍保留未提交内容；
5. 代码仍可能继续LLM/TTS并记录success。

这把“挂起”变成了“静默截断”，不符合“正常完成或显式error”。现有fake ASR没有模拟ASRCache的prefix和final提交语义，因此self-test未发现。

### 修复要求

- close控制事件与final ASR提交语义分离；
- `InputClosed`到达后，必须显式触发真实ASR cache的final drain；
- 如果无法构造可靠final drain，则该record必须error，不得success；
- 增加使用真实`ASRCache`/`StreamingASRProcessor`协议的轻量集成测试，构造缓存中有未提交prefix且`flush=None`的用例，断言尾文本不丢失。

---

## P0-2：没有真正的pair总deadline，线程可能遗留并污染后续样本

位置：

- `run_ttfa_unified.py:599-621`
- `run_ttfa_unified.py:652-695`
- `run_ttfa_unified.py:721-790`
- `run_ttfa_unified.py:1537-1598`

`PAIR_DEADLINE_S`实际主要用于生成结束后等待TTS，并非从pair开始覆盖：

- playout；
- segmentation；
- collector；
- ASR；
- LLM cache/generation；
- System A `sink_q.get()`；
- full ASR。

`join(timeout=5)`后未检查`thread.is_alive()`，worker默认非daemon；超时或异常后主循环仍可开始下一task，共享模型可能被遗留线程继续访问。

### 修复要求

- pair开始即计算唯一绝对deadline；所有等待使用剩余时间；
- queue get、join、LLM、ASR、TTS均受该deadline约束；
- join后检查所有线程是否退出；任何遗留线程均将pair标timeout并fail-stop当前run；
- GPU/模型worker异常后不得继续下一个正式样本；
- TTS worker保存句柄，取消时关闭response并确认退出；
- self-test增加永久阻塞segmenter/ASR、playout不发sentinel、TTS慢流等路径。

---

## P0-3：checkpoint记录追加不是原子的，恢复绑定信息不足

位置：

- `run_ttfa_unified.py:854-904`
- `run_ttfa_unified.py:1488-1517`
- `run_ttfa_unified.py:1532-1535`

header采用临时文件替换，但record仍是普通append。崩溃可产生截断JSON行；下一次虽然退出，但没有达到v3.1要求的记录级原子写。

config hash也未完整绑定：

- git commit/dirty patch；
-软件环境；
- ASR/LLM/Silero模型revision/hash；
- TTS服务revision/config；
- 每个音频hash及subset-list hash；
- downmix/resampler版本。

### 修复要求

- 采用原子snapshot checkpoint，或JSONL加独立原子index/offset并验证record hash；
- config/schedule hash使用canonical serialization；
- checkpoint损坏、截断或hash不匹配必须退出且不得复用目录；
- error/cancelled/timeout均是终态，不静默重跑；
- 配置变化、attempt重跑使用新run_id；
- self-test覆盖record截断、重复主键、pair一侧失败和目录中旧run混入。

---

## P0-4：Silero正式默认仍是浮动`master`

位置：

- `run_ttfa_unified.py:1433-1434`
- `run_ttfa_unified.py:1506-1515`

正式模式允许不传`--silero-ref/--silero-dir`，此时使用`master`。这直接违反v3.1“固定commit、禁止浮动远端”的要求。传本地目录也只记录路径，没有验证实际commit和模型hash。

### 修复要求

- 正式模式必须要求固定commit或经验证的本地snapshot；
- 禁止默认`master`；
- 保存实际仓库commit、dirty状态和模型artifact hash；
- checkpoint config hash包含这些值；
- self-test可用fake，但正式/冒烟模式缺固定revision时必须立即拒绝启动。

---

## 4. W1高严重性问题

### P1-1：self-test未执行生产`generate_with_meta()`，first model token定义也有误

位置：

- `src/llm/stream_llm_inference.py:194-236`
- `run_ttfa_unified.py:632-639,748-755`
- fake：`run_ttfa_unified.py:1066-1082`

调用方在检查`is_eos`前设置`first_model_token_ns`，因此EOS-only响应仍会获得first-model-token时间戳。冻结规范要求首token对应首个非EOS模型token。

同时，self-test全部走`_FakeLLM`，没有调用真实新增接口；`generation_stop_reason`主要由调用方推导，没有验证真实接口的终止状态。

### 修复要求

- EOS判断在first-model-token记录之前；
- 明确first model token与first content token；
- 新增不加载大模型的真实类方法级测试，例如构造对象/替换logits生成路径，实际调用`StreamLLMInference.generate_with_meta()`；
- 测试EOS-only、空decoded special token、max_tokens和正常EOS。

### P1-2：RNG仍使用全局重置，不是请求级隔离

位置：

- `run_ttfa_unified.py:624-627,739-742`
- `stream_llm_inference.py:211-213,351-370`

当前调用`torch.manual_seed()`和`torch.cuda.manual_seed_all()`，而`torch.multinomial()`不接收独立generator。遗留线程或其他GPU随机操作会消费全局随机流，AB/BA顺序仍可能改变回复。

### 修复要求

- 最好给`generate_with_meta()`和`_decode_logits()`增加可选`torch.Generator`并传给`torch.multinomial()`；
- generator绑定目标LLM设备；
- A/B同配对键使用相同基础seed；
- 测试同seed重复生成token序列一致，并验证不同任务顺序不改变该序列。

### P1-3：schedule只有全局25/25，没有语言×时长分层平衡

位置：

- `run_ttfa_unified.py:797-825`
- self-test：`run_ttfa_unified.py:1356-1371`

当前排序后只做全局25/25。独立复现可出现某个完整stratum 13/0、另一个0/13。self-test也只检查全局计数。

### 修复要求

- 在每个`(language, duration_group)`stratum内交替或尽可能平衡AB/BA；
- 奇数stratum的额外一个方向在全局和相邻stratum中平衡；
- self-test逐stratum断言差值≤1；
- 重复子集的两种三轮序列也按语言×时长平衡。

### P1-4：TTS magic可被跨read分片绕过

位置：

- `run_ttfa_unified.py:281-285,322-349`

若读取依次返回`b"R"`、`b"IFF..."`，当前代码会把WAV body当作裸PCM并得到playable success。JSON也可用前导空白或分片绕过。

### 修复要求

- 累积至少足以识别格式的前缀后再判定；
- 处理前导空白后的JSON/XML/HTML错误响应；
- 探活和正式请求使用同一校验器；
- self-test覆盖跨read `RIFF`、跨read JSON、HTML和空body。

### P1-5：512-byte应用粒度未被强制，奇数字节被截掉后仍可能成功

位置：`run_ttfa_unified.py:323-350`

`iter_content(512)`通常但不保证严格返回≤512 bytes；代码没有对返回块二次切分。奇数字节尾部还会被截掉，而不是按冻结协议报格式/对齐错误。

### 修复要求

- 在应用层把任意read重新切成固定≤512-byte granule；
- 按字节流累计，不丢弃奇数字节；只在完整sample边界推进playable计数；
- 结束时若总PCM字节数非sample-width倍数，整行error；
- self-test覆盖大read和奇数分片。

### P1-6：schema与QA校验不足

位置：

- `run_ttfa_unified.py:59-88`
- `run_ttfa_unified.py:418-494`
- `run_ttfa_unified.py:974-999`

缺少或未严格验证：

- clock/endpoint mode；
- last input sample；
- TTS文本、UTF-8长度；
- TTS不可控随机性标志；
- 每chunk scheduler error≥0；
- PCM总字节≥1324；
- RMS/peak有限；
- first-content-token偏序；
- config/schedule hash一致；
- A/B seed和分析波形hash一致；
- pair两侧明确终态；
- error行必填诊断字段。

应增加真正的JSON Schema或等价的完整字段验证，并让QA覆盖所有冻结约束。

### P1-7：CLI没有可控故障注入冒烟

位置：`run_ttfa_unified.py:1422-1440,1484-1486`

`--smoke 3`只是取前三个样本，不保证中英文覆盖，也没有故障注入。不能满足Gate要求的“中英文成功路径+A/B+至少一个可控故障”。

### 修复要求

- smoke样本显式分层选择；
- 增加仅限smoke/test的故障注入选项；
- 生产正式模式禁止故障注入；
- 冒烟报告明确列出2条成功样本和1条故障场景，或等价覆盖组合。

---

## 5. W1中严重性问题

1. PSE元数据仍缺soundfile/librosa版本、实际Silero hash/revision、两法原始sample和详细失败原因。
2. TTS probe把缺Content-Type视为可接受，且没有真正固定允许策略；Content-Encoding也未比较。
3. 如果没有success行，`summary[0]`或`cv_rows[0]`可能抛`IndexError`，导致错误checkpoint已写但RUNINFO/QA未生成。
4. 正式结果目前没有完整记录TTS文本本体和UTF-8字节数，与v3.1协议不完全一致。

---

## 6. Self-test证据范围

开发回复中“成功路径覆盖生产A/B”的表述不准确。当前self-test主要验证fake组件下的orchestration。

### 已覆盖

- energy PSE部分；
-句末检测器；
- fake HTTP下的TTS函数；
- playout worker；
- fake A/B编排；
- schedule/checkpoint部分行为。

### 未覆盖

- 真实`generate_with_meta()`；
- 真实tokenizer累计token decode；
- 真实ASRCache状态机；
- 真实StreamingASRProcessor final drain；
- 真实Segmenter `flush=None`与ASR尾提交组合；
- CUDA RNG隔离；
- 生产TTS分片和慢流；
- worker永久阻塞与线程污染；
- 正式smoke故障注入。

因此33项PASS只能证明脚本结构和模拟路径，不能作为Gate 1通过证据。

---

## 7. W3/W4/W5审查

### W3：有条件通过

当前数字独立复算正确：

| 模式 | mean CV | median CV | P90 CV | max CV | >5% |
|---|---:|---:|---:|---:|---:|
| B | 5.1935% | 4.0492% | 10.7303% | 18.9577% | 19/50 |
| A | 5.2317% | 4.6530% | 9.9213% | 14.0090% | 23/50 |

但脚本只检查合计有3条观测，没有强制三个输入文件各恰一条，也不检查顶层config一致。

修复要求：

- 每个输入文件内主键唯一；
- 三个文件的键集合完全一致；
- 每个键每文件恰一条；
- 对关键config字段做一致性hash/比较；
- self-test覆盖一轮重复、一轮缺失和配置不一致。

### W4：当前不通过

S/D/I/N、macro和corpus算法当前数字正确，但没有执行v3.1要求的A/B共同sample-ID过滤，也没有生成paired filter manifest。未来一侧缺样本时会基于不同集合比较。

另有：

- `reference_full`缺失时会静默回退到已知截断的`reference`；
- self-test没有覆盖完整参考优先、缺失拒绝和模式集合不一致。

修复要求：

- 对每个数据集/条件构造模式交集；
- 生成paired filter manifest，记录候选、保留、排除和原因；
- A/B/LA汇总基于相同ID集合；
- R2指定`--ref-csv`时强制每行非空`reference_full`，禁止回退；
- self-test覆盖不配对、error、重复ID和reference_full缺失。

当前交付数据碰巧集合一致，所以已有数字未受影响，但脚本未达到冻结协议。

### W5：有条件通过

当前21项结果、bootstrap、Wilcoxon、Holm和效应量方向可复算，关键锚点正确。需要补：

- 重复`(sample_id, mode)`必须fail；
- 空配对必须fail；
- LA输入必须限定`mode=la_streaming`；
- R5 sample ID唯一及过滤清单；
- self-test覆盖以上异常。

---

## 8. 文档与其他交付

1. `MANUAL_SPOT_CHECK.md`仍是待填写模板，因此W7尚未完成；这不阻塞W1代码修复，但阻塞论文数据放行。
2. `EXPERIMENT_DESIGN.md`尚未同步；按W8应在结果QA后完成。
3. `run_ttfa_unified.py`中的设备/模型建议改为CLI或配置来源，不要作为模块内部硬编码。
4. 本轮没有发现不必要的范围扩张。

---

## 9. 下一轮最低送审条件

在再次申请GPU探活/冒烟放行前，至少完成：

### 必须修复

1. 真实ASR final drain，保证`flush=None`不丢尾文本；
2. 从pair开始覆盖全部阶段的绝对deadline；
3. 所有线程、HTTP响应和共享模型状态可确认终止，异常后fail-stop run；
4. checkpoint记录级原子性与完整恢复绑定；
5. 正式模式强制固定Silero commit/model hash；
6. EOS/first token语义和真实`generate_with_meta()`测试；
7. 请求级独立RNG generator；
8. 语言×时长分层schedule；
9. TTS跨read magic、大read和奇数字节处理；
10. 完整schema/QA；
11. production smoke故障注入；
12. W4配对过滤和manifest。

### 测试证据

- 原33项self-test更新并通过；
- 增加真实ASRCache/processor协议的轻量集成测试；
- 增加真实`StreamLLMInference.generate_with_meta()`方法级测试；
- 增加永久阻塞/线程遗留测试；
- 增加TTS跨read格式测试；
- W3/W4/W5新增负向self-test；
- py_compile和diff check通过。

完成后可先评估是否放行TTS探活及3条GPU冒烟；正式50×2仍须在冒烟结果QA后单独放行。

---

## 10. 最终结论

**方案v3.1通过，代码提交`f7e219c`不通过Gate 1。**

最严重的问题是：生产`flush=None`路径可能静默丢失尾部ASR文本，pair总deadline和线程fail-stop没有实现，checkpoint与Silero固定不满足协议，而self-test又没有覆盖真实生产ASR/LLM路径。因此当前不得启动GPU探活或冒烟。

W3和W5当前结果可以保留为待补强的有效结果；W4当前数值可核验，但脚本和产物在补齐配对过滤manifest前不能最终放行。
