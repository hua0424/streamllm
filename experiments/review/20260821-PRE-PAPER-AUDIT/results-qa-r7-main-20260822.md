# r7_main / r7_tts_control 结果级 QA 核验记录（20260822）

核验人：开发侧（本机）。输入：GPU 主机产物提交 **946b720**（`R7正式主实验+匹配文本TTS控制产物`）。

## 0. 结论摘要

| 项 | 结论 |
|---|---|
| r7_main 结果级 QA | **通过**（本机复算 47/47 项，含脚本自带 `validate_record` 140 条全量重放） |
| tts_control 数据级 QA | 通过（32/32 success、control_from hash 精确匹配、文本哈希对应、均值复算一致） |
| 流程合规 | **一处越界须审查裁量**：tts_control 在未获单独书面放行的情况下提前执行（§3） |
| 代码基线 | `b8893d6`→`c9437c3` 实验代码零差异（`git diff` 为空）；RUNINFO `git_commit=c9437c3…` 语义已更正入 handoff |

commit 三元组：
- **code_commit**（RUNINFO 内，运行时 HEAD）：`c9437c3a4a69c58f7ea714c72af2df6db6ec7a97`
- **result_artifact_commit**：`946b720877940adee55e8fc6a66538cdd465ca1f`
- **verification_commit**：本记录所在提交（含核验脚本 `handoff/_qa_r7_local_20260822.py` 与脚本 mkdir 修复）

## 1. r7_main 核验明细（47 项全过，要点）

复算环境：本机 Windows，`uv run`，QA 脚本 `r7_ttfa_unified/handoff/_qa_r7_local_20260822.py`（随本提交归档，可复现）。

1. **规模与终态**：记录 140（header+140 行）；`terminal_state=success` 140/140；
   `error` 全空、`fatal` 全 False；(sample, mode, repeat_idx) 全唯一。
2. **schema+因果闭合重放**：调用脚本自带 `validate_record(rec, config_hash, schedule_hash)`
   逐条重放，140/140 零违规（含 perf_counter_ns 时钟、偏序链
   playput_start≤pse≤last_input≤feed_end≤input_close≤first_token≤generation_end、
   TTS 链 req≤headers≤first_pcm≤first_playable、streaming 侧 flush 链）。
3. **计划复算**：以 `load_samples` 复现运行时加载序（crosswoz→multiwoz）+ 推导子集
   （load 序 zh 前 5+en 前 5，与实际 3 轮样本集合一致），`build_schedule` 复算
   140 任务，**schedule_hash 复算值与 RUNINFO 一致**（`fd989786…`）；
   **checkpoint 行序与 schedule 全序一致**（执行无跳步/无乱序）。
4. **A/B 配对**：70 对双模式齐全；全局 AB/BA=35/35；repeat0=25/25；补轮=10/10；
   每对执行方向与 schedule `order` 字段一致。
5. **子集三轮**：恰 10 样本（5zh+5en）各 3 轮×2 模式；其余 40 样本仅 repeat 0。
6. **TTFA**：`first_playable_pcm − physical_speech_end` 全 70+70 条**非负**；
   streaming p50≈2957ms / max 14188ms（70 条含补轮）；summary（50 条 repeat0）：
   zh p50 2603ms、en p50 7577ms、ALL p50 3113.7ms；non-streaming ALL p50 22269.9ms。
7. **重复性**：子集 CV（ddof=1，n_valid=3 全部成立）20 条均值 7.73%，最大 20.70%
   （仅 1 条触 20%，无 >20% 项）。
8. **hash 绑定全部复算一致**：`platform_conditions_sha256=a4c400576b…`（放行版）、
   Silero artifact `e1122837…`（PSE/分段器同源断言在 RUNINFO）、
   `sample_list_sha256`（LF 归一化）、`subset_sha256`（canonical_json(sorted)）、
   `audio_map_sha256`（由 140 条记录 wav_sha256 重建）、`config_hash`（由 RUNINFO
   config JSON 重放 canonical_json 复算）。
9. **分层覆盖**：zh 25/en 25、duration_group=very_long×50（与样本清单一致）；
   endpoint_mode 与 mode 一一对应（streaming=explicit_flush / A=full_input 各 70）；
   `response_token_count≤128` 全体（stop: max_tokens 116 / eos 24）。

## 2. RUNINFO git 字段语义（两处与 handoff 原文不符，已更正 handoff）

1. **`git_commit=c9437c3…` 而非 b8893d6**：`_git_info()` 记录运行时
   `git rev-parse HEAD`，放行后 HEAD 即 `c9437c3`。handoff 原文"RUNINFO 记录的
   code_commit 仍为 b8893d6"是**措辞错误**（开发侧责任），运行时记录 c9437c3 才是
   正确行为。代码基线不变性改由独立证据支撑：`b8893d6`→`c9437c3` 对
   `run_ttfa_unified.py`/`src/`/sample-list 的 diff 为空（本机核验）。
2. **`git_dirty=True`**：成因为**运行自身日志文件**——`| tee r7_main_run.log` 在
   shell 管道建立瞬间创建未跟踪文件（日志首行 05:13:01.471），早于脚本内
   `_git_info()`（05:13:02 后）采集；探针/tts_probe.json 写盘在采集之后，不参与。
   先例：fatal_smoke RUNINFO 同为 `git_dirty: True`（其 tee 日志同样先于采集）。
   启动前工作树 clean 由 GPU 侧 `git pull --ff-only` 后 porcelain 为空确认。
   **限制如实声明**：脚本只记布尔值不记 porcelain 明细，故"采集时 porcelain 恰只含
   该日志文件"是从时序推断而非产物直接证明；属记录粒度缺陷，建议后续版本记录
   porcelain 内容或排除运行自身输出路径。

## 3. 流程披露：tts_control 提前执行（越界）

放行函（review-reply-gate-material-verification-r3-20260822.md）明确：**本次授权不含
真实 r7_tts_control**，须待 r7_main 完成并通过结果级 QA 后**单独复核并另行放行**。
GPU 主机在 r7_main 后直接执行了 handoff §3。成因之一是 handoff §3 节头当时未标注
"须单独书面放行"（§2 有标注、§3 遗漏——开发侧责任，已补标注）。

- **数据级 QA 本身通过**：32/32 success、error 0；`control_from_sha256=4edcd6ec…`
  与主 checkpoint 哈希精确匹配；10 样本按脚本设计规则选样（repeat-0 完整配对中
  sorted zh 前 5+en 前 5，与主实验子集不同属设计而非异常）；每样本 3 调用
  （B 首句/A 首句/A 全文）+校准 2；非校准条目 `tts_text_sha256` 全部能在主实验
  同样本全文/首句哈希中找到；tts req→first_pcm 均值复算 7076ms 与 RUNINFO 一致；
  请求链单调；binding 记 git_commit=c9437c3、platform=a4c40057。
- **该 32 条是否采信为论文正式控制数据，或要求单独放行后重跑，由审查裁量**；
  本记录仅证明数据自身干净，不构成对越界的事后追认。

## 4. 脚本缺陷与修复（不影响既有结果）

- **缺陷**：`--tts-control-only` 分支在写 `tts_probe.json` 前不创建输出目录，
  目录不存在时首跑 `FileNotFoundError`（GPU 侧实测；mkdir 后重跑成功，产物干净）。
- **修复**：分支入口先 `Path(args.output_dir).mkdir(parents=True, exist_ok=True)`
  （约 line 2631），纯目录创建，不触碰任何测量逻辑；修复后本机 `--self-test`
  90 PASS / 0 FAIL。r7_main/tts_control 结果均在修复前的 c9437c3 上产生，
  RUNINFO 记录不受影响。
- 附带说明：`tts_control_run.log` 仅 1 行摘要（控制模式正常输出即只有末行汇总；
  首次失败 traceback 未保留，成功重跑的 tee 覆盖之），控制结论以
  checkpoint/CSV/RUNINFO 为准。

## 5. 产物哈希清单（LF 归一化 sha256 前 16 位）

```
4edcd6ec28189d00  r7_main/checkpoint_r7_main.jsonl      086c5efdbd29837d  tts_control/checkpoint_r7_tts_control.jsonl
3089f11e89013341  r7_main/RUNINFO_r7_main.md            0e6de295ab438003  tts_control/RUNINFO_r7_tts_control.md
579d280eddb794bc  r7_main/QA_r7_main.md                 9acffdb39f90124c  tts_control/tts_control_r7_tts_control.csv
1dca0a8e0326a360  r7_main/ttfa_summary_r7_main.csv      b100ec1db3e35d06  tts_control/tts_probe.json
660d12c42b9abdc5  r7_main/ttfa_subset_cv_r7_main.csv    9c97c94df196b55f  tts_control_run.log
941fa0e4ff8d8048  r7_main/tts_probe.json
11b35818c76143ab  r7_main_run.log
```

## 6. 边界（不变）

smoke/self-test/fatal_smoke 不作论文样本；绝对延迟不跨平台拼接、不缩放、
不事后修正；本记录不预支 tts_control 放行权；论文侧边界（映射注记、
CUDA/Triton fallback、max_tokens=128 等）照 review §6 在论文中声明。
