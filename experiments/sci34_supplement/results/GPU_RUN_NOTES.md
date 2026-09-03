# GPU 主机运行记录（sci34 补实验，2026-09-01）

## Commit 三元组与 campaign

- **campaign**：`sci34_f11ccba_20260901`
- **code_commit（E3 主实验）**：`f11ccba`（origin/paper2，工作树 clean）
- **judge 修复 commit（裁判及之后所有 run）**：`ca627c7`（本地提交，仅改 `experiments/sci34_supplement/e3_judge.py`；
  `f11ccba..ca627c7` 对 E3/A1/P1 代码零改动，`git diff --stat` 可核验）
- 数据：`p2_turns.json` SHA-256 `a2116b83b38509e45d641ade17ae1791282729836bdca251aa8aba8aa9248a0c`
  （MultiWOZ 2.1 + `prepare_multiwoz_data --seed 42 --max-dialogues 100` 本机派生）

## 环境侧变更（不触及 git 跟踪文件）

1. `uv pip install sentencepiece==0.2.2`（venv 内，未改 pyproject/uv.lock）——
   必要原因：Mistral-7B-Instruct-v0.3 的 tokenizer_config 含 `add_prefix_space=true`，
   transformers 4.57 对 Llama 系 tokenizer 强制 from_slow 转换路径，缺 sentencepiece 直接报错。
2. NLTK `punkt`/`punkt_tab` 离线装入 `/root/nltk_data`（GitHub raw 下载解压）。
3. 模型经 ModelScope 下载至本机目录：
   `Qwen2-7B-Instruct`（Qwen/Qwen2-7B-Instruct）、`Mistral-7B-Instruct-v0.3`（LLM-Research/Mistral-7B-Instruct-v0.3）。

## 裁判格式失败与修复（步骤 6）

- 首次 judge run `sci34_f11ccba_20260901_judge`（prompt v2）在第 920/1600 条遇 Mistral
  输出解释文本（`'Directly refers to specific information'`）而非 YES/NO 开头，脚本按设计
  fail-closed 中止。**该目录（919 条记录 + manifest）原样保留**在 `judge/` 下备查。
- 修复（commit `ca627c7`，prompt 升 v3）：判定标准与语义不变；用户提示尾部追加
  「首行必须是 YES 或 NO」格式指令；新增一次有界重试（重试提示带格式提醒）；仍失败维持
  RuntimeError fail-closed。新增 `retried` 字段落盘。
- 重跑 `sci34_f11ccba_20260901_judge_v2`：1600 条全部完成，`parse_failures=0`，`retried=0`
  （v3 提示词下格式失败完全消失，重试机制未触发）。

## P1 异常观察与设计方复核（步骤 9）

GPU 运行方最初按“无 per-length warmup 的 CUDA 首次分配/kernel 编译”上报异常：2048/f0.25
的 stop→crop/role 中位数约 108 ms，8192/f0.25 约 1388 ms、f0.5 约 657 ms；同长度后续
单元格回落到 stop→crop 约 1–2.5 ms、stop→role 约 77–81 ms。原始数据未做任何剔除或重跑。

设计方随后结合 180 条 raw records 与代码顺序复核，否定了“一次性冷启动”解释：受影响单元的
额外等待跨 20 次重复持续，并随待恢复 KV 后缀长度与播放等待时间变化。根因是每次
`fixture.ensure_full()` 后未在 `player.start()` 前同步；stop 后首次设备同步把尚未完成的准备态
恢复错误计入 stop→crop/role。因此旧 run 的联合路径数值被判为协议无效，不能通过删除首个样本
或挑选所谓稳态单元修复。旧 run 继续作为审计记录保留；后续只能以新 run-id 按 prepared-state v2
协议定向重跑。

## P1 prepared-state v2 定向重跑（2026-09-01 晚，commit `dc52978`）

按 `P1_PREPARED_RERUN.md` 完整执行，run-id `sci34_dc52978_20260901_async_prepared_v2`：

- 协议：每次 trial 先 `ensure_full()` + GPU 同步再启动播放器（`setup_ms` 单独记录、
  不计入 stop 路径），stop 后同步单独记为 `post_stop_sync_ms`；每 `(length,fraction)`
  单元先 3 次 warmup（不落盘），再 20 次正式 repeat。
- 验收全过：180 条 formal records（9 单元 × 20），`protocol=async_prepared_v2`、
  `prepared_state_synchronized=true`、`leaked_samples=0`、request/ack 精确命中目标采样点；
  partial 几何 0.25/0.75→true（120 条 mid_fragment）、0.5→false（60 条 fragment_boundary）。
- 关键数值（median / 最大单元 P95，ms；全 9 单元）：stop ack 0.055–0.062 / 约 0.077；
  post-stop sync 0.167–0.176 / 约 0.352；timeline lookup 0.47–0.50 / 约 0.94；
  stop→crop 2.44–2.53 / 约 3.492；stop→role 78.6–80.8 / 约 86.1。精确最大单元 P95
  分别为 0.076842、0.351591、0.939422、3.491824 和 86.084611 ms。
  旧 run 的跨单元冷启动异常消失：stop→crop/role 在 512/2048/8192 与三个位置间均匀，
  `setup_ms`（41–1717 ms，随长度增长）被正确隔离在 stop 路径之外。
- 环境：`sentencepiece==0.2.2` 已并入 pyproject（`dc52978`），venv 内旧装版本一致；
  快照（lscpu/meminfo/uname/nvidia-smi 前后/依赖哈希）见
  `run_logs/sci34_dc52978_20260901_async_prepared_v2_snapshots/`；两张 3090 运行前后均无其他进程。
- 打包：`results/sci34_dc52978_20260901_async_prepared_v2.tar.gz`（仅新 P1 run + 日志 + 快照，
  30 项），SHA-256 `4c6188249f1226e5692a85468cf1e9c3b05e648494a5ce9a6e5a475b264c0bc8`。
  旧 `sci34_f11ccba_20260901_async` 目录路径/时间戳/内容未动。

## GPU 干扰记录

A1 正式 run 前后 `nvidia-smi` 快照存于 `a1/nvidia_smi_before_formal.txt`、
`a1/nvidia_smi_after_formal.txt`；期间无其他 GPU 进程。

## E1/E2 确认性 campaign（2026-09-01/02，commit `b8c758b`）

按 `e1e2_confirmatory/GPU_HANDOFF.md` 完整执行，campaign `e1e2c_b8c758b_20260901T173306Z`：

- 环境准备：TEN_Turn_Detection 本机原缺失，进入冻结流程前一次性经 ModelScope 下载
  （15GB/17 文件，`/root/autodl-tmp/dataA/models/TEN_Turn_Detection`）；此后全程
  `HF_HUB_OFFLINE=1` 离线，正式 run 零联网。`uv sync --frozen`，pyproject/uv.lock 前后 hash 不变。
- 流程：六模块 CLI 核对 → 四项 smoke 全 PASS → before snapshot → holdout 100 条
  （与旧 E1/E2/E3 交集 0，sha `e86c0ccb…`）→ TEN cache 222 entries（卡1）→ 冻结
  campaign manifest（sha `2f4bd76e…`）→ pilot 30 records（独立非 formal campaign）→
  五个 formal session 顺序各一进程，5 × 1000 = 5000 records，validation `ok=true`
  （无 duplicate/truncation，五 process identity 唯一，条件顺序平衡）→
  `analysis_v1.json`（bootstrap 10000×，seed 20260901）→ ACCEPTANCE.md 填写 →
  after snapshot + 72 文件 checksums + tarball（sha 见 `.tar.gz.sha256`，
  `88de19dd10e344fc70a6a075b481fcba30e34ab6907c119792b5fd7253b31eaf`）。
- 旧结果保护：`exp1_latency.json`/`exp2_tradeoff.json`/`paper2_reanalysis.json` 跑前跑后
  SHA-256 逐字节一致（guard diff 通过）；论文稿与旧实验结果零改动。
- 关键数值（单位 ms，配对 n=500）：C-E1 实际墙钟主指标（last_segment_arrival→
  first_token_ready）A 27.70 vs B@0.92 62.38，配对 A−B −34.69（95% CI [−35.30, −34.11]）；
  TTFT_eff 乐观下界口径 A−B +17.44（CI [16.12, 18.75]）。C-E2 B@0.92 vs never：主墙钟
  −0.03（CI [−0.55, +0.51] 含 0），oracle 下界 +20.80（CI [19.50, 22.10]）；
  B@0.92 pooled waste 2.85%、survival 67.0%。九阈值 waste/survival 单调。
- 全部口径与红线遵守：不把 TTFT_eff 称实际墙钟、不把 endpoint_accept 称最后段到达、
  不把受控文本段称真实音频；结果按 .gitignore 约定验收后 `git add -f` 入库。

## C2 crop/clean-prefill 等价性 campaign（2026-09-03，commit `563dd22`）

按 `c2_equivalence/GPU_HANDOFF.md` 完整执行，run `c2eq_563dd22a_20260903T013547Z`
（campaign identity `1f07a2e9…`，manifest sha `f4960a20…`）。**结果：rejected（硬门槛未过，工件全留，未 seal）**。

- 环境准备：本机原无 `Qwen/Qwen2.5-0.5B-Instruct` HF 缓存（§2 的 kvcrop/speculative
  smoke 需要），formal 前一次性经 HF 下载（`HF_HUB_DISABLE_XET=1`）入
  `/root/autodl-tmp/hfhome`，并以符号链接对齐 `cache_dir=$HF_HOME` 布局；此后全程
  `HF_HUB_OFFLINE=1` 离线。`uv sync --frozen`，pyproject/uv.lock 前后 hash 不变。
- 流程：五模块 CLI 核对 → py_compile + 五项 smoke 全 PASS（c2 fake smoke 24 cases
  `models_loaded=false`）→ 7B/tokenizer 预检（eos==eot==151645，template sha
  `79320228…`）→ E3 exact `p2_turns.json` 抢救（sha 与 manifest 一致 `a2116b83…`）→
  pilot（`c2pilot_563dd22a_20260903T013205Z`，3 cases，probe 全合格但 logit 门槛超阈，
  已预警）→ 冻结 formal manifest → 单进程 formal 24/24 cases（45 checkpoints，
  24 attempts 无 resume）→ validator `ok=false`（probes 24/24/20，227 errors）→
  analyzer fail-closed 拒绝生成 → ACCEPTANCE.md 如实 `rejected` → seal 拒绝封存
  （未写任何状态）→ 回传 tarball（sha
  `c24f28fcc51ae27c77f9290a797e1ce86ab9a309a95c121e83b60cd71cbeb874`，23MB，目录内
  74 文件有 `checksums_return.sha256` 清单）。
- 旧结果保护：186 个 git 跟踪旧结果文件（experiments/results + sci34_supplement/results
  全量，排除 c2_equivalence）跑前跑后 SHA-256 逐字节一致；论文稿零改动。
- 关键结果（RTX 3090 / torch 2.8.0+cu128 / sdpa / BF16）：
  - **通过层 100%**：canonical↔path token IDs exact（45/45）、KV/mask/seq/ledger 状态、
    assistant 内容账本、unique EOT、role phase、scenario execution 24/24、
    next-token top-1（45/45）、top-5 overlap（min 4，门槛 ≥4）。
  - **失败层**：BF16→FP32 logit diff 全部 45/45 checkpoint 超冻结阈（max_abs
    0.15625–0.96875 > 0.1；mean_abs 0.0202–0.1565 > 0.01；RMS 最高 0.1643）；
    32-token continuation 30/45 exact（15 个发散，首个发散位 0–25）；
    termination probe 20/24（4 个 `natural_eos`（c2_07/13/16/19）greedy 在冻结
    128-token cap 内未 EOS；max_tokens 8/8、eos_at_cap 6/6 全合格）。
  - 失败定性与建议已写入 ACCEPTANCE §8：(a) probe 未命中属 cap×snapshot 行为，
    同机重跑不可解；(b) logit 超阈呈系统性（增量 append vs 整段 prefill 的 BF16
    数值核差异叠加上下文长度放大），是否放宽阈/换环境复核由设计侧决策，本 run 容差未动。
- 附带归档：E3 抢救件入库 `results/e3_exact_rescue/`（p2_turns.json、E3 manifest、
  模型身份重哈希 `209f3a9c…`、raw MultiWOZ/builder hash），与 C2 records 分离。
- 红线遵守：未删任何失败工件/NPZ；未改 24 cases/32 continuation/top-k/BF16 阈值；
  未改 src/论文“配合结果”；结果（含 pilot 与失败工件）按用户留存规则 `git add -f`
  入库，tarball 与 E1/E2 轮惯例一致不入 git。

## C2 v2 pilot 阻塞（2026-09-03，commit `a501df4`，**formal 未启动**）

按 v2 `GPU_HANDOFF.md` 执行至 §5 pilot 后提前停止：pilot 暴露 v2 探针检查的确定性
实现缺陷，formal 必然批量失败，按 pilot 预检定位与“失败停止”规则不进入 formal。

- 已完成：§0 协议预检（PROTOCOL_VERSION==2）+ guard 279 文件（含 v1 归档 93 文件
  只读）→ §1 冻结离线（pyproject/uv.lock hash 一致）→ §2 五模块 CLI + 五项 smoke
  全 PASS（c2 smoke `protocol_version=2`、`checkpoint_sidecars=45`）→ §3 模型预检
  （eos==eot==151645，template `79320228…`）→ §4 E3 抢救确认已归档跳过。
- pilot：`c2pilot_a501df43_20260903T033106Z`（identity `5dcb0773…`）。3 cases 中
  c2_02/c2_03 通过；**c2_01（natural_eos，genuine）失败**。
- **缺陷定位**：`runtime.py` `_termination_probe` 末尾的检查分支
  （`if case.termination == "eos_at_cap": … else: <max-token 检查>`）——该 `else`
  同时捕获 `natural_eos`。genuine natural_eos（observed=EOS、role_phase=
  ASSISTANT_EOT_PENDING、content<cap）与 max-token 断言（MAX_TOKENS/ASSISTANT_OPEN/
  content==cap）必然互斥，于是每个 genuine case 都被记 3 条
  “small-budget/MAX_TOKENS/ASSISTANT_OPEN”错误并 `passed=false`；runner
  `_assert_probe_qualified`（run.py:128 `not probe.errors`）随即抛出
  “common invariants failed”。而 run.py:137-144 与 validate.py:106-109 的语义
  正确（genuine → 期望 EOS+EOT_PENDING），即 runtime 与 runner/validator/spec 三方
  中仅 runtime 一处分支写反。fake smoke 不产生 genuine EOS，故 24-case smoke 无法
  捕获。
- pilot 证据（c2_01 probe 摘录）：`{declared: natural_eos, mode: real_greedy,
  cap: 256, observed_end_reason: EOS, eos_step: 21, genuine_eos: true,
  requalified: false, role_phase: ASSISTANT_EOT_PENDING, content_token_count: 20,
  errors: [3 条 max-token 错误], passed: false}`。
- **formal 影响面（确定性）**：v1 同模型同 seed 下 10 个 natural_eos case 中 6 个
  在 ≤128 即 EOS（21/8/20/111/98/80，均 ≤256 → v2 全为 genuine）；另 4 个 128 内
  run-on 在 cap 256 下可能转 genuine。故 formal 至少 6/24、至多 10/24 case 必然
  probe 失败；且 campaign 级 `natural_eos_min_genuine≥5` 与“每个 genuine case 都
  失败”不可同时满足——协议在当前代码下不可通过。
- **建议修复方向（未实施，代码由设计侧提供）**：runtime.py 该处 `else:` 应改为
  仅对 `max_tokens` 声明执行（如 `elif case.termination == "max_tokens":`）；
  natural_eos 的 genuine/requalified 两条路径已分别在 665-674 与 675-676 自洽校验。
  修复后 genuine 探针 0 错误、requalified 保持 675-676 一致性检查。
- v2 等价门槛本身在 pilot 已验证工作正常：3 cases 全部 checkpoint
  `logit_gates.all_ok=true`（相对噪声限 + 绝对安全上限 + margin + backstop +
  continuation 全过；例：c2_01 post_recovery path max_abs 0.289 vs 对照噪声 0.3125，
  限 2×0.3125=0.625），noise_control 臂与 checkpoints sidecar 落盘正常。
- 工件：pilot 目录全留（含 checkpoints/ sidecar 与失败 record）；工作树零改动
  （未修任何代码）；本轮不产生 formal/tarball/ACCEPTANCE。

## C2 v2 formal（2026-09-03，commit `5c56b01`，修复后正式轮）

设计侧修复探针分支（`else:`→`elif case.termination == "max_tokens":`，另补 smoke
genuine 覆盖）后按 v2 `GPU_HANDOFF.md` 重跑全程，run `c2eq_5c56b014_20260903T040829Z`
（identity `165c91f9…`，manifest sha `7a636016…`，content hash `b7d8a474…`）。
**结果：rejected——21/24 cases 通过，3 个 checkpoint 边际超过 v2 噪声相对门槛**。

- 流程：§0–§4 复跑全绿（protocol v2 预检、guard 289 文件含 v1 归档只读、冻结离线、
  五项 smoke、模型预检、E3 抢救确认已归档跳过）→ 新 pilot
  `c2pilot_5c56b014_20260903T040650Z` 3/3 通过（genuine c2_01 零错误，修复生效）→
  冻结 formal manifest → 单进程 formal 24/24 落盘（约 6 分钟，无中断/resume）→
  validator `ok=false`（10 errors，从 checkpoints/*.npz 三数组独立重算一致）→
  analyzer fail-closed 拒绝 → ACCEPTANCE 如实 `rejected`（全部计数独立重算填写，
  遵守模板填写纪律）→ seal 拒绝（未写状态）→ 回传 tarball
  （sha `b70ee32347d3b0064384d1191e3d7f5dd90f4fd2da2a39d4fa6115e266420593`，34MB，
  74 文件 `checksums_return.sha256`）。
- 旧结果保护：289 文件跑前跑后 SHA-256 一致（v1 归档 run 与 e3_exact_rescue 零改动）。
- 通过层（独立重算）：termination probe 24/24（natural_eos 6 genuine/4 requalified，
  ≥5 门槛达成；eos_at_cap 6/6 @step4；max_tokens 8/8 @budget2）；token IDs 45/45 exact；
  KV/mask/seq/ledger、assistant 账本、unique EOT 45/45；scenario 24/24；
  top-1 运行时口径 43/45（2 个 near-tie 翻转均在 margin 限内）；top-5 min 4/5；
  continuation 规则 45/45（30 完全 exact，15 个发散点 margin 0–0.25 ≤ 各自限 0.125–0.5）；
  绝对安全上限（max≤2.0/mean≤0.5）45/45。
- **失败项（3/45 checkpoint，均为 2.0× 噪声相对倍数边际超阈）**：
  `c2_06_invalidate_short_max/next_assistant`（path max_abs 0.96875 vs 限 0.8125，
  比值 2.38）；`c2_10_clean_medium_eos/post_recovery`（mean_abs 0.09014 vs 限 0.07626，
  比值 2.36）；`c2_21_pending_long_max/post_recovery`（mean_abs 0.08432 vs 限 0.06407，
  比值 2.63）。失败 sidecar 与 45 个 checkpoint NPZ（path/canonical/control 三数组）
  全量保留。
- 定性建议（ACCEPTANCE §8）：其余结构层与绝对上限全过，失败仅集中在 2.0× 相对倍数；
  0.5B dry-run 参考比值 1.08 而本机 7B 长上下文更高。是否调整倍数须新协议版本冻结，
  本 run 常数未动。

## C2 v3 crop-integrity pilot 阻塞（2026-09-03，commit `b2c6f22`，**formal 未启动**）

按 v3 `c2_crop_integrity/GPU_HANDOFF.md` 执行至 §3 pilot 后提前停止：pilot 8 cases /
9 crop events 中 7 cases 全部 exact 通过（含 c2_08 第二次 crop），仅
`c2_06_invalidate_short_max`（speculation_full_invalidation）失败，formal 按规程不启动。

- 已完成：§0 预检 PASS（PROTOCOL_VERSION=3、24/27、PRIOR_V2_RUN_ID 与 cases
  byte-copy sha 校验全过；guard 375 文件含 C2 v1/v2 全工件只读）→ §1 冻结离线 →
  §2 五 CLI + v3 smoke（`protocol_version=3`、27 events、wrong_keep/layer_hash/
  duplicate_eot_ledger/missing_event 四类 tamper 全检出）+ 四核心 smoke + 模型预检。
- **失败定位（唯一差异字段）**：c2_06 的 recovery_check[0]（`prefill_user_text`，
  invalidation crop 后首个恢复操作）中 production `generation_end_reason='CROPPED'`
  vs oracle 期望 `'NONE'`；其余全部字段（role_phase=USER_OPEN、seq/mask/KV/ledger=514、
  assistant 计数）及该 event 的 K/V/logits/mask bitwise 比较、crop 保留前缀三方
  （pre-crop prefix / 独立切片 oracle / production）exact、negative control 检出
  **全部通过**。recovery_check[1]（`open_assistant_role`）state_exact=True。
- **根因**：生产语义 `crop_to_token` 置 `CROPPED`（stream_llm_inference.py:619）；
  `reopen_user_role`/`open_assistant_role`/`prefill_assistant_text` 均重置 NONE，
  唯 `prefill_user_text`（:742）不重置。`speculation_full_invalidation` 是唯一不经
  reopen、直接 `prefill_user_text` 的恢复路径，故 `CROPPED` 残留到
  `open_assistant_role` 才清除；v3 oracle `_expected_state`（runtime.py:633）期望
  任何恢复操作后即 NONE。其余 7 个 pilot case 恢复均先经 reopen（重置 NONE）故通过；
  fake smoke 的 prefill/open 实现不走生产 role/end 状态机，无法捕获。
- **formal 影响面（确定性）**：24 cases 中 3 个 invalidation case（c2_06/c2_14/c2_22，
  三个 context 档各一）将同样失败；其余 21 cases 预计通过。
- **决策归属设计侧（两种修法，均非实验机可现场实施）**：
  (a) 生产侧：`prefill_user_text` 追加 `generation_end_reason=NONE` 重置
  （与其他恢复 API 对齐，属 src 语义变更，需设计侧评审对 E2/E3 等既有结论无影响）；
  (b) 协议侧：v3 oracle 对 invalidation 路径的 `prefill_user_text` 一步放行残留
  `CROPPED`（期望态放宽，属协议修订需 bump 版本）。红线 3 禁止现场改任一侧配合结果。
- 工件：pilot 目录全留（`results/c2_crop_integrity/c2crop_pilot_b2c6f22b_20260903T064135Z/`，
  自 /tmp 移入归档）；工作树零改动；本轮不产生 formal/tarball/ACCEPTANCE。

## C2 v3 crop-integrity formal（2026-09-03，commit `8210300`，**accepted + sealed**）

设计侧采用生产侧修法（`prefill_user_text` 重置 `generation_end_reason=NONE`，含
orchestrator 断言与 kvcrop 新语义检查）后按 v3 `GPU_HANDOFF.md` 重跑，run
`c2crop_82103004_20260903T080512Z`（identity `fa6f956d…`，manifest sha
`d8c3db4d…`）。**结果：全部通过——24/24 cases、27/27 crop events 全 exact，
validation `ok=true` 零错误，ACCEPTANCE `Status: accepted`，seal 已创建并验证。**

- 流程：§0–§2 重建全绿（新 commit 预检 PASS、guard 381 文件含 v1/v2 归档与
  上轮失败 pilot 只读、五项 smoke 含新语义 kvcrop、模型预检）→ 新 pilot
  `c2crop_pilot_82103004_20260903T080321Z` 8 records/9 events 全过
  （`validate --non-formal --expected-cases 8` ok=true；c2_06 修复确认
  end_reason NONE/NONE）→ 冻结 formal manifest → 单进程 formal 约 3 分钟一次
  通过（pid-100045-…，无中断）→ guard 干净 → validator `{"errors": [], "ok": true}`
  → analysis_v1（accepted、descriptive-only、保留 v2 rejected 为描述性证据）→
  ACCEPTANCE 填写 `Status: accepted` → seal `--create`/`--verify` 通过
  （30 文件，seal sha `e0997d41793f510fc1120a7c3f08c420097813cc627f08d47716e76b4489f4a9`）→
  tarball（sha `54cbe2edf961e4536add813c477f3fe3c9808256febc979eab6ef046285304a9`）。
- 独立重算确认：24 records/27 events 全 `passed=true`；全部 exact 旗标
  （keep_length/retained_prefix_hash/pre_prefix=post=oracle 三方/post_crop 长度·mask·
  token/logits/mask/shapes/dtypes/devices）True；27/27 negative control 检出；
  3 个 no-op crop 同获全套 exact 证明；全部 recovery_check 的
  kv/logits/masks/production_state exact；production API 为真实
  `StreamLLMInference.reopen_user_role/prefill_user_text/open_assistant_role`，
  oracle 为 direct forward、token chunk 逐块一致。
- 执行插曲（无实质影响）：seal 后我给 seal 日志加 tee 导致 `logs/seal.log`
  偏离封存空文件哈希、`--verify` 一度失配；按封存期望恢复空文件后原样重跑
  `--verify` 通过（handoff §7 本不给 seal 加 tee，系操作偏差已纠正）。
- 旧结果保护：381 文件跑前跑后 SHA-256 一致；v1/v2 rejected 归档与
  `e3_exact_rescue/` 零改动；论文零改动。v2 保持 rejected，v3 结果未改写 v2 结论
  （claim boundary 已在 ACCEPTANCE 限定：仅证 crop/truncation 完整性与
  matched recovery 确定性，不证 clean re-prefill 数值等价）。
