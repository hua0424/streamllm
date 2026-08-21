# 开发侧回复函：实现审查整改（对应 review-implementation-v3.1-20260821.md）

- 日期：2026-08-21
- 被审提交：`f7e219c`；整改后提交见本次 push（message 含"Gate1 整改"）
- 总体：4 项 P0、7 项 P1、4 项中危、W3/W4/W5 强化要求**全部接纳并完成**，无争议项。评审对代码的核查（flush=None 静默截断风险、pair deadline 未覆盖全程、checkpoint 非原子、Silero 浮动 master 等）经逐行核实全部属实。

## 一、P0 整改（与评审逐条对应）

### P0-1 flush=None 静默截断 → 显式 final drain

- 评审对真实语义的判断属实：`transcribe_audio_segment` 在未达阈值且非 final 时早退
  （`faster_whisper_streamer.py:538-540`），段留存未提交；flush=None 时旧实现会把挂起变成静默截断。
- 修复（`run_ttfa_unified.py` collector/transcriber）：`InputClosed` 到达且未见 is_final 段时，
  显式将尾部段标记 `is_final=True` 触发 final drain（waiting 队列空但 segment_queue 有未提交
  内容时同样处理）；drain 触发后仍无 final 输出 → 该行记 `asr_final_drain_no_output` error，
  **不得 success**；close 控制事件与 final 提交语义分离。
- 测试：fake 路径 drain 用例（尾文本断言进入 LLM 预填）+ drain 无输出 error 用例 +
  **真实 `ASRCache` 协议集成用例**（真实 cache 状态机 + 实现真实调用协议的桩 processor，
  覆盖"未达阈值留存 → close → drain → 尾文本输出"全链）。

### P0-2 pair 总 deadline + 线程终止确认 + fail-stop

- pair 开始即计算唯一绝对 deadline（`--pair-deadline-s`，默认 900s）；queue get/join/LLM/ASR/TTS
  等待全部用剩余时间约束；超时 → `PairTimeout` → 终态 `timeout` + `fatal=True`；
- `_join_all()`：join 后检查 `is_alive()`，任何遗留线程 → timeout + fatal；worker 线程全部
  daemon；TTS worker 保存句柄、取消时经 cancel_event 关闭 response 并确认退出；
- fatal 传播：worker 异常/线程遗留/pair 超时 → 主循环停止，后续任务全部补写 `cancelled`
  终态记录（pair 另一侧必有明确终态）；
- 测试：永久阻塞 segmenter 用例（2.5s deadline → timeout+fatal+thread_leak 记录）、ASR 异常
  fatal 用例。

### P0-3 checkpoint 记录级原子 + 完整恢复绑定

- 改为**整文件原子快照**：每次 append 后 tmp+fsync+replace 重写（记录数小，开销可忽略）；
- 加载校验：header 损坏 / 记录截断 / 重复主键 / 非法终态 / **binding 任一项不匹配** /
  **同目录混入其他 run 的 checkpoint** → SystemExit；
- binding 含：config/schedule hash、git commit+dirty、环境版本（python/torch/numpy/soundfile/
  librosa/requests/scipy）、ASR/LLM 模型、Silero meta（ref/commit/artifact hash）、TTS 配置+
  探活策略、样本清单/子集/音频映射 SHA-256；
- 测试：截断记录、重复主键、binding 不匹配、目录混入旧 run 四个负向用例。

### P0-4 正式模式强制固定 Silero

- 正式/冒烟模式缺 `--silero-ref`/`--silero-dir` 立即拒绝（不再有 master 默认）；
- `_silero_artifact_meta()`：本地目录记录 git commit/dirty；hub ref 记录缓存仓库目录名；
  模型 artifact（.jit/.pt/.onnx）SHA-256 找不到即拒启动；meta 进 config hash 与 checkpoint binding。

## 二、P1 整改

1. **P1-1 EOS/first token**：两个 runner 均先判 is_eos 再记 `first_model_token_ns`
   （首个非 EOS 模型 token）；新增**真实 `StreamLLMInference.generate_with_meta()` 方法级测试**
   （`__new__` 构造 + stub tokenizer/model，正常 EOS / EOS-only / max_tokens 三路径 +
   stop_reason 断言）；self-test 另断言 EOS-only 响应不记 first_model_token；
2. **P1-2 请求级 RNG**：`generate_with_meta()`/`_decode_logits()` 新增可选
   `torch.Generator` 参数并传入 `torch.multinomial`（旧 `generate()` 不动）；runner 用
   绑定设备的独立 Generator（配对键派生 seed）；测试：同 seed 同序列且不受全局 RNG 影响；
3. **P1-3 分层 schedule**：逐 (language, duration_group) stratum 内交替、|AB−BA|≤1，
   stratum 多数方向与全局累计失衡相反 → 全局恰 25/25；子集两种三轮序列按语言分层 5/5；
   self-test 逐 stratum 断言；
4. **P1-4 TTS 跨 read 格式**：先累积 ≥16 字节前缀再判定；前导空白剥离后识别
   RIFF/JSON/HTML/XML；探活与正式共用同一 `classify_payload`；测试含跨 read RIFF、
   前导空白 JSON、HTML；
5. **P1-5 读取粒度/对齐**：任意 read 应用层重切 ≤512B granule；字节连续累积不丢奇数；
   只在完整 sample 边界推进 playable；**改为读到自然结束**（顺带覆盖全程对齐校验）——
   结束总字节非 2 的倍数 → `tts_misaligned_bytes` 整行 error；测试含大 read、奇数分片；
6. **P1-6 schema/QA**：validate_record 扩至 clock_type/endpoint_mode/last_input_sample/
   TTS 文本+UTF-8 字节数/tts_seeded=False 标志/逐 chunk sched_err≥0/tts_total_bytes≥1324/
   RMS·peak 有限/first_content 偏序/config·schedule hash 比对/error 行必填诊断；QA 增加
   A/B 同 seed、同分析波形 hash、配对键两侧终态；
7. **P1-7 冒烟故障注入**：`--smoke` 分层选取（中英交替保证覆盖）；`--inject-fault asr_error`
   仅限 smoke（正式模式拒绝），对最后一条冒烟任务注入 ASR 异常验证 error 落盘。

## 三、中危项

1. PSE 元数据补 soundfile/librosa 版本（loader meta + binding env_versions）、双法原始 sample
   与失败原因字段保留；
2. 探活固定允许策略：`allow_content_type/allow_content_encoding`（含 None 取值），正式请求
   逐项比对，不一致即 error，不临时放宽；
3. 无成功行时 summary/CV 只写表头（消除 IndexError），QA/RUNINFO 照常生成；
4. TTS 文本本体 + UTF-8 字节数落盘（`tts_text`/`tts_n_bytes_utf8`）。

另按 Standards 轴：模型/设备改 CLI 参数（默认值取自 `src/config`），正式值写入 RUNINFO；
`EXPERIMENT_DESIGN.md` 同步按 W8 排在结果 QA 后（阶段 2）。

## 四、W3/W4/W5 强化与结果

- **W3**：文件内主键唯一、三文件键集合一致、关键 config 字段跨文件一致（10 字段比对），
  负向用例×3；重跑数字与评审锚点逐位一致（B 5.1935/4.0492/10.7303/18.9577%、19/50）；
- **W4**：scope 内全模式 sample_id 交集过滤 + `paired_filter_manifest_{tag}.json`
  （候选/保留/排除及原因）；`--ref-csv` 强制非空 reference_full、禁止回退截断列；
  重复 (scope,mode,sample_id) 退出；**重生成 wer_real.csv / wer_la_vs_b.csv 与整改前
  逐列 0 差异**（集合本就一致，评审预判正确），manifest 排除 0 条；
- **W5**：重复记录/空配对退出、LA 限定 la_streaming、R5 ID 唯一；重跑 21 比较数字不变。

## 五、测试证据汇总

| 项 | 结果 |
|---|---|
| `run_ttfa_unified --self-test` | **56 项全 PASS**（新增真实 ASRCache 协议、真实 generate_with_meta 方法级、请求级 RNG、跨 read/大 read/奇数 TTS、阻塞 worker 线程遗留、checkpoint 四负向、stratum 分层等） |
| W3/W4/W5 self-test | 全 PASS（含新增负向用例） |
| py_compile 全部改动文件 | PASS |
| **本机真实组件集成测试**（新增 `ttfa_local_integration.py`，RTX 3060） | **ALL PASS**：真实 Silero/whisper-tiny/ASRCache/Qwen2-0.5B（本地目录）+ 真实 tokenizer 重解码 + 请求级 Generator，A/B 双路径 success、validate 全过、A 未提前启动 ASR。仅验证路径可用性，不产出论文数字 |
| W3/W4/W5 数据重跑 | 数字与整改前一致（0 差异） |

## 六、遗留（不阻塞本次送审）

- W7 人工试听：候选清单与模板已备（`r2_real_speech/MANUAL_SPOT_CHECK.md`），待需求方本人完成；
- `EXPERIMENT_DESIGN.md` 与总册阶段 2 同步：排在正式结果 QA 后；
- 真实 TTS 分片/慢流的生产行为、正式 smoke 故障注入：待 GPU 冒烟阶段验证。

**申请**：按评审 §9 复核整改，若通过请放行 GPU 侧 TTS 探活 + 3 条冒烟（handoff 随即提供）。
