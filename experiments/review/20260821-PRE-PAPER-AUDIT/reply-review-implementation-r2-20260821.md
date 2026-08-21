# 开发侧回复函：Gate 1 r2 复核整改（对应 review-reply-implementation-v3.1-r2-20260821.md）

- 日期：2026-08-21
- 被审提交：`7323359`；整改后提交见本次 push
- 总体：本轮 5 项 P0、4 项 P1 **全部接纳并完成**，无争议项。评审对竞态窗口、System A
  主线程异常路径、fatal 恢复缺口、Silero 双重加载与 smoke 假阳性的判断经逐项核实全部属实。

## 一、P0 整改

### P0-1 final-drain 竞态原子化

- `collector` 在 `state_lock` 内**先完成队列 final 化与 drain 标志设置，再发布
  `pipeline_input_close_ns`**；`transcriber` 的退出判定（closed && !must_call）在同一把锁内
  读取——结构上消除"看到 closed 但 drain 标志未设"的窗口；
- 新增**确定性交错回归测试 ×3**（真实 `ASRCache` + 首次转写被放慢 0.4s，强制 close 发布
  与 final 化在转写调用中交错）：断言尾段（"段2"）经真实 cache 状态机进入 LLM 预填，
  transcriber 未提前退出；
- 原真实 ASRCache 协议尾文本测试保留。

### P0-2 System A 主线程异常无条件 fatal

- 两个 runner 的通用 `except` 改为**无条件 `fatal=True`**（ASR/LLM/模型状态异常均
  fail-stop）；可恢复错误（TTS/输入类）在到达该分支前已提前 return；
- 新增测试：System A ASR 异常 → fatal、System A LLM cache_prompt 异常 → fatal；
- fatal 后续任务补写提取为可测函数 `_backfill_cancelled()`（新增单测：2 条剩余任务全部
  补 cancelled），主循环 fatal（本次或恢复）后统一调用。

### P0-3 checkpoint 恢复 fatal-stop

- `Checkpoint` 加载后扫描全部终态记录，`fatal_seen` 暴露 run 级状态；main 以
  `fatal_stop = ck.fatal_seen` 初始化——恢复后**只补写剩余任务 cancelled，不执行任何
  GPU 任务**；
- 新增测试：写入 fatal 记录后模拟崩溃 → 重开 checkpoint 恢复 `fatal_seen=True` →
  `_backfill_cancelled` 补齐。

### P0-4 固定 Silero 注入正式分段器

- `StreamAudioSegmenter` 新增 `silero_model`/`silero_utils` 注入参数（注入时**不访问
  torch.hub**；缺 utils 拒绝构造）；旧签名/行为不变；
- W1 main：segmenter 用与 PSE **同一**已 hash 的 Silero 实例构造，并断言
  `silero_injected`；RUNINFO 分别记录 PSE/segmenter 侧 meta 并断言同一 artifact；
- 新增测试：monkeypatch `torch.hub.load` 为抛错后注入构造仍成功（证明无二次浮动加载）、
  缺 utils 拒绝。

### P0-5 smoke 精确命中校验

- 选取逻辑提取为 `_select_smoke()`：零命中/少命中 → SystemExit；smoke≥2 必须覆盖中英
  两语种；任务数必须恰为 N×2；
- QA 断言成功路径与故障路径均实际执行：普通冒烟无 success 记录 → QA 问题；故障冒烟
  必须出现 `fault_injection` error 终态**且**存在成功路径记录；exit code 由 QA 问题决定；
- 新增测试：单语种 smoke≥2 拒绝、双语种命中、零命中拒绝。

## 二、P1 整改

1. **P1-1 TTS 派生字段一致性**：validate_record 重算并比对 `tts_n_chars`、
   `tts_n_bytes_utf8`、`tts_text_sha256`，且 `tts_text_source` 必须与
   mode×(sentence_end_found, sentence_fallback) 一致；负向测试×3（篡改字节数/哈希/来源
   均被拦截）；
2. **P1-2 阻塞 read 可取消**：`tts_measure` 支持 `resp_holder`（response 句柄外置）；
   read timeout 动态收紧为 min(配置值, pair 剩余时间)；join 超时时外层主动
   `_close_resp()` 打断阻塞 read；新增 headers 后停发 body 的慢流测试（0.5s read
   timeout 内被打断，全程 <8s）；
3. **P1-3 表述降级**：集成测试 docstring 与本函均改称"**真实 CUDA 组件加载与 A/B 路径
   集成检查**"，明示局限（噪声输入/PSE 能量注入/fake TTS/未做 AB 同 seed 比较）；
   脚本增加 `--qwen-dir/--asr-model/--device` CLI 参数（不再硬编码 Windows 路径）；
4. **P1-4 计数显式**：self-test 末行显式输出 `N PASS / M FAIL`（当前 **69 PASS / 0 FAIL**）。

另：按 §6 建议新增 `.gitattributes` 规则 `experiments/results/**/*.csv -whitespace`
（CSV CRLF 不再被 diff --check 报 trailing whitespace，仓库规则显式化）。

## 三、测试证据汇总

| 项 | 结果 |
|---|---|
| `run_ttfa_unified --self-test` | **69 PASS / 0 FAIL**（新增：竞态回归×3、fatal 恢复+回填、A 侧 ASR/LLM 故障、headers-only 慢流、TTS 派生字段负向×3、smoke 三负向、segmenter 注入不触 hub） |
| W3/W4/W5 self-test | 全 PASS（未改动，数字不变） |
| py_compile（run_ttfa_unified / ttfa_local_integration / streamaudio_segmenter / stream_llm_inference） | PASS |
| `ttfa_local_integration`（CLI 化后复跑） | ALL PASS（B success TTFA≈2.45s / A success TTFA≈1.51s / validate 全过 / A 未提前 ASR） |
| git diff --check | 干净（CSV whitespace 已由 .gitattributes 显式规则覆盖） |

## 四、探活执行说明

按 §5 放行范围，GPU 主机可先执行**独立 TTS 探活**
（`uv run python -m experiments.scripts.run_ttfa_unified --tts-probe --tts-url ...`），
产物 `tts_probe.json`（status/Content-Type/Content-Encoding/payload 分类/允许策略）将随
冒烟 handoff 一并提交复核；探活通过不解释为慢流/取消/TTFA 链路通过。

**申请**：复核本整改 + 探活产物，通过后放行 3 条 GPU 冒烟（handoff 随即提供：
探活 → self-test → 3 条分层冒烟[含故障注入] → QA）。
