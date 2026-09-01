# 二期工程决策日志

每次做技术决策时按时间倒序追加一条。每条包含：**日期 / 决策 / 背景 / 理由 / 影响 / 状态**。
状态：`proposed` / `accepted` / `superseded`（被后续决策替代时填写替代条目的日期与编号）。

---

## D-015（2026-09-01）接受 prepared-state P1 v2 并区分第三实验 campaign

**决策**：接受 run `sci34_dc52978_20260901_async_prepared_v2` 为 headless 墙钟软件播放控制路径的正式证据；P1 v1 继续仅作失败协议审计。D-015 只取代 D-014 中“P1 v2 待运行”的状态，不改变 D-014 已接受的固定轨迹 E3 与联合 A1 结论。

**证据与口径**：
1. **身份与归档**：实验代码 commit `dc529788e86ecd3e2e4203ba16b1076d6b231ec1`，结果入库 commit `ee1dcc7`，manifest `config_hash=93b7837acdc708ffde48448fc7cb0549475cbf064539d53a5327cda05031e005`，clean tree、Transformers runtime、Qwen2-7B-Instruct 模型指纹已记录。结果文件 SHA-256：records `2dc68896dc52ce2c777b1a6375f1a5c3090f9baffd8f07a6ac1ed0f1769a3b67`，analysis `b9705d58f36909604e3e0df94d2190b3a5050c6a62d35fee1c29987fff4db20a`。回传 tarball SHA-256 为 `4c6188249f1226e5692a85468cf1e9c3b05e648494a5ce9a6e5a475b264c0bc8`。
2. **完整性**：512/2048/8192 token × 0.25/0.50/0.75 × 20 repeats，共 9 个单元、180 条唯一正式记录；3 次单元预热均不落盘。120 条为片段内打断，60 条为片段边界；180/180 request 与 ack 精确命中目标软件采样，零采样泄漏，prepared-state 与 partial 几何检查全部通过。
3. **延迟**：九个单元的中位数范围为：软件停播确认 0.055–0.062 ms、stop 后设备同步 0.167–0.176 ms、时间轴反查 0.47–0.50 ms、stop→crop 完成 2.44–2.53 ms、stop→角色恢复完成 78.6–80.8 ms；各指标最大单元 P95 分别为 0.076842、0.351591、0.939422、3.491824 和 86.084611 ms。准备态 setup 的单元中位数为 41.208–1717.110 ms，在播放前完成并从所有 stop 路径区间中排除。累计区间相互嵌套，禁止与组件中位数相加。
4. **硬件与 campaign**：P1 v2 主机为双路 Intel Xeon Gold 6330（112 逻辑 CPU）、约 756 GiB 内存、Ubuntu 22.04.5、NVIDIA driver 580.105.08、双 RTX 3090；运行前后无其他 GPU 计算进程。它作为第三个独立 campaign 报告，不与旧 E1/E2/A2 或固定轨迹 E3/联合 A1 的绝对墙钟时间池化，也不通过相减解释“系统开销”。
5. **主张边界**：P1 v2 只支持 headless 软件播放器、时间轴查询与模型状态修正路径的协议内分布。它不测声卡/扬声器停止、用户声学上实际听到的最后采样、在线 TTS 取消、真实 ASR/LLM/TTS/播放器并发或生产端到端 barge-in。九个单元范围较窄只作本 campaign 的观察，不证明上下文无关或硬件不变性。

**影响**：第六章在 RQ4 下新增 P1 v2 子节与表 6-5；摘要、讨论和结论加入限定性结果。P1 从当前待办移除，后续工作保留生产音频闭环、固定轨迹 A2、人类双标、细粒度物理对齐和跨模型/语种复验。

**状态**：accepted

---

## D-014（2026-09-01）固定轨迹 E3 与联合 A1 的正文证据升级

**决策**：将 SCI3/4 补实验中的固定轨迹 E3 作为 RQ1 的主要受控结果，以新联合 A1 替换正文旧 A1 数值；headless P1 v1 因联合计时协议把未完成的异步准备工作计入 stop 路径而排除，待按 prepared-state v2 协议完成定向重跑后再决定是否纳入正文。

**关键结果与口径**：
1. **固定轨迹 E3 为主证据**：100 条纯 MultiWOZ 对话生成 400 个配对场景、800 条条件记录；playback/generation 共享同一被打断 assistant 轨迹、片段时间轴和注入位置，首轮及 probe 均 greedy、最多 40 token。片段目标 n=297：规则 199/297（67.0%）vs 189/297（63.6%），McNemar p=.164，generation−playback dialogue-cluster 95% CI [−9.5, 2.8] pp；裁判 127/297（42.8%）vs 121/297（40.7%），p=.512，CI [−8.9, 6.1] pp。
2. **proxy 资格修正**：字符比例—空白边界代理按自身非空目标确定资格，n=380；规则 286/380（75.3%）vs 280/380（73.7%），p=.405，CI [−5.75, 2.50] pp；裁判 167/380（43.9%）vs 157/380（41.3%），p=.229，CI [−8.25, 2.67] pp。四个点估计均小、方向与预设假设相反且不显著，不作优效、等效、非劣或伤害主张。
3. **构造检查独立报告**：playback 条件的局部完整未播放文本在 400/400 场景中为空，是机制/指标共同定义的构造检查，不与 n=297 或 n=380 的语义效果估计合并。0.5 与 clean boundary 在片段目标层面重复；cluster bootstrap 是主要不确定性结果，McNemar 仅作描述性补充。
4. **裁判与人评边界**：固定轨迹 E3 没有随机盲法的人类双标注。Mistral 裁判使用 `specific-reference-v3` 单一提示词；v3 在 v2 格式解析失败后增加首行 YES/NO 约束与一次有界重试，正式运行无解析失败且未触发重试。裁判仅为模型代理。
5. **联合 A1 替换旧结果**：上下文 256/512/1024/2048/4096/8192 token，warmup=5、repeats=50，以设备前后同步包围同一 joint crop+role 区间。联合中位数/IQR/重新预填充相对加速比分别为：31.616/2.356/2.254×，31.852/2.162/4.124×，31.054/3.099/7.707×，31.519/1.197/15.020×，36.903/0.635/25.453×，48.315/0.928/40.620×。该结果是模型侧同步微基准，不是完整 barge-in。
6. **campaign 与 P1**：旧 E1/E2/A2 与新 E3/A1 运行在不同 CPU 主机、但均为同型号双 RTX 3090；不虚构 CPU 名称，不池化跨 campaign 绝对时间。P1 v1 在 `ensure_full()` 后未于播放器启动前同步，stop 后的首次同步把仍在执行的 KV 恢复错误计入 stop→crop/role；该污染跨多次重复持续，并非一次性冷启动。P1 v2 待运行，摘要和正文不得出现其占位数字。

**影响**：权威分章 Markdown、摘要、大纲与论文上下文改用上述口径；固定轨迹 E3 从后续工作移除。后续保留固定轨迹 A2、协议有效的生产式异步链路/真实音频闭环和独立人工双标。IEEE 衍生稿本次不修改。

**状态**：accepted

---

## D-013（2026-08-31）论文统稿的数据完整性审计与结论边界修正

**决策**：在不重跑 GPU 实验、不覆盖原始结果 JSON 的前提下，对二期 E1/E2/E3/A1/A2 做离线完整性审计；将清洗与统计复算独立保存为 `experiments/results/paper2_reanalysis.json`，并以分章 Markdown 作为唯一正文源重新统稿。

**关键结论与修正**：
1. **fixture 污染隔离**：E3 原 103 个 id 实为 100 条 MultiWOZ + 3 条 `fx_*` 开发样例，E2 原文件另含 12 条 `fx*` 记录。正文正式结果排除 fixture：E3 每条件 n=400，B-gen loose 规则/裁判为 50.3%/2.3%；E2 每阈值 n=100，不推测点 TTFT 为 48.3 ms。原始 GPU JSON 保留不动，便于追溯。
2. **统计设计对齐**：E3 由独立样本 Fisher 改为同 `(id, fraction)` 配对的 exact McNemar，并以 dialogue 为重采样单元给出 10,000 次 cluster bootstrap 置信区间。
3. **结论强度收缩**：loose=0 继续明确为构造性保证；strict 改称采样比例近似口径；取消“检测器上界/裁判下界”“无代价”“连续单调前沿”“完整 barge-in 亚毫秒”等超证据表述。
4. **A1 口径分离**：0.308–0.339 ms 仅为 `DynamicCache.crop` 孤立微基准；39.7× 的分母是 crop+role recovery（8k 为 46.88 ms），不再把二者混用。
5. **A2 降格**：三策略分别重新采样首轮与下一轮回复，仅 33/100 的三策略 `heard_text` 完全相同，不能隔离策略因果效应；正文保留为受混杂的探索性负结果，若需正式比较须在实验机固定同一生成轨迹后重跑。
6. **形式化修正**：原始进度不再跨量纲直接比较；以片段级保留边界统一到 token 域。按代码真实语义，片段内尾部评估定义为“播放比例切文本字符 + 向前吸附空白边界”的代理，而非 token 域线性插值；KV 裁剪和角色恢复拆成两个状态阶段，assistant token 账本保持本轮相对长度。
7. **版本源层级**：`abstract.md + chapter1..8 + references.md` 为权威源；`thesis_draft.md` 和中英文 IEEE 稿均为衍生产物。当前先更新学位论文 Markdown 与实验图，IEEE 稿后续从新源同步。

**影响**：论文核心贡献仍成立，但适用范围被限定为受控文本段/Mock TTS 实例化与孤立 KV 微基准；真实异步音频闭环、完整停播延迟、固定轨迹 A2 和独立盲法人评列为后续补强实验。

**状态**：accepted

---

## D-012（2026-07-02）实验前代码审查结论与修复

**决策**：实验开跑前对 `7facaba...HEAD`（二期全部实现+实验代码）做两轴审查（Standards/Spec），确认 3 个 BUG 并全部修复；§6 埋点缺口补齐；配置集中化整改。

**关键修复**：
1. **E3 指标框架修正（review BUG1，最重要）**：选 A 语义下 playback 的"未听引用率=0"是**构造性保证而非实验发现**——论文必须如此表述，实验量化的是 B-gen 失败率。新增 **strict 严格 ground-truth 列**（P1 语义：被打断片段内未播尾部按播放采样比例切分、计入 unheard 检测）——playback 的 strict>0 量化**片段级截断粒度的量化误差**（D-008 选 A/§八取舍的代价），成为 E3 的诚实补充结果。
2. **E1 公平性（BUG2）**：System A 改用与 B 相同的 system prompt（原用默认中文 prompt 导致生成不可比）；mouth-to-ear 建模改为 `first_fragment_ms + TTS首块延迟`（原用首 token 时刻，忽略断句攒首片段的时间）；"B 的 prefill 与说话重叠"是一期机制、属被测系统本身，注释澄清非偏置。
3. **chunker 越界（BUG3）**：纯空白句直接跳过（原兜底推进会偷下一片段首 token 使 crop 点偏移）；token_end 钳制到实际生成数（原可越界致 crop_to_token 崩溃）。
4. **§6 埋点补齐**：8 个时间戳落盘（timestamps dict，模拟量标注）、`ttft_text_ms`（§3 定义可测）、KV 复用计数器/复用率（rewrite<1）、反向映射 timeline_records 落盘、E3 增加 boundary 边界对照注入（P2）。
5. **配置集中化**：新增 `P2_LLM/TRIGGER/REWRITER_MODEL_NAME`、`P2_DEVICE`（src/config.py，.env 可覆盖）——实验机换 7B 只需 .env 或 `--model`；采样率经 `StreamingTTS.sample_rate` 取；`_check` 收拢至 `src/utils/check_utils.py`。
6. **B-syn 措辞修正**：Mock 同步合成下与 generation 等价，仅异步 real TTS 可区分，文档不再称"已验证"。

**已知未修（记录为接受的债务）**：`spec_stats` 字典应为 dataclass；offline-first 加载块三处重复；`_timed_tokens` 的 `self._t_first` 侧信道；E1/A1/A2 无断点续传（微基准快速可重跑）；fixture 规模小（真实数据在实验机）。

**状态**：accepted

---

## D-011（2026-07-02）TEN 规格修正 + 软触发开发替身策略

**决策**：
- **规格修正**：TEN Turn Detection（TEN-framework/TEN_Turn_Detection）实测为 **7.6B 参数 / BF16 ~15GB**（HF API 确认），**不是 D-003 记录的"Qwen 0.5B 微调"**——当时调研信息有误。
- **验证机（16G）装不下 TEN** → 定义统一 `SoftTrigger` 接口（文本→turn 完成度连续置信度），本机用 **prompted Qwen2.5-0.5B 作开发替身**（取 YES/NO 首 token logits softmax 为置信度；D-003 讨论时的备选方案），实验机同一接口加载 TEN 7B（取 finished/unfinished/wait 类别词概率）。
- **实验机分卡布局仍成立**：卡 1 = TEN(15GB) + CosyVoice2(~3GB) + Qwen3-0.6B(~1.5GB) ≈ 19.5GB < 24GB，比 D-002 预估紧但可行。
- 两阈值机制（§3.5）不变；软触发不是论文贡献，不做选型消融（D-003 原则不变）。

**影响**：`src/dialogue/trigger.py` 按接口+双实现设计；E2/A3 在本机用替身出 harness 验证，实验机换 TEN 出正式数值；论文 §实现 需注明软触发模型规格。

**状态**：accepted

---

## D-010（2026-07-02）TTS 策略：Mock-first（时长 profile 驱动）+ real CosyVoice2 仅在实验机

**决策**：
- **验证机（本机 Blackwell）永不装 real CosyVoice2**——它硬 pin torch==2.3.1+cu121，与 sm_120 根本不兼容（正是 D-009 升级掉的版本）。
- 编排闭环用 **Mock TTS**：由**真实测得的时长 profile**（每字符≈多少采样、首块延迟≈45ms）驱动，与真机**时序等价**。定义 `StreamingTTS` 接口，Mock 与 CosyVoice2 都是其实现（swap-in）。
- **real CosyVoice2 只在实验机（3090 Ampere，官方 pin 可装）跑**，且仅用于：① E1 的 mouth-to-ear 最可信数字；② 定性 demo。
- 工作流：本机把**全部实验代码**用 Mock 跑通验证 → 上实验机直接换 real CosyVoice2 实现跑出最终结果。

**背景**：CosyVoice2 官方 requirements pin torch==2.3.1/transformers==4.51.3/cu121，与本机 torch 2.8+cu128 冲突。经分析（见下）其对实验的贡献可归约为"时长 + 延迟 profile"。

**理由（CosyVoice2 在实验中的真实角色）**：
- 全部实验指标都是**时序或文本**类，无一需要真听音频；P1 已定确定性程序注入，不需实时交互播放。
- CosyVoice2 对实验只贡献两样：**片段音频时长**（驱动模拟播放时钟，可用一次性测得的 profile 参数化）+ **首块延迟**（mouth-to-ear，最好真机 live 测）。
- assistant 文本每次动态生成且依赖打断，无法预烤成固定音频集；故 CosyVoice2 是"一次性 characterize + 实验机 live 少量指标"，而非"预处理后丢弃"，也非"每个实验都 live"。

**影响**：
- 新增 `src/tts/streaming_tts.py`（接口 + TimingProfile + MockStreamingTTS）、`src/player/`（SimulatedPlayer）、`src/dialogue/orchestrator.py`（编排闭环）
- TimingProfile 初值为占位（英文 ~1000 samples/char、首块 45ms），**上实验机后用真实 benchmark 替换**
- CosyVoice2 实现类留待实验机；接口先定死

**状态**：accepted

---

## D-009（2026-07-01）torch 升级到 cu128 以支持 Blackwell（5070 Ti）

**决策**：`pyproject.toml` 的 PyTorch 栈从 cu121 升到 **cu128 / torch 2.8.0 trio**：
- index：`https://download.pytorch.org/whl/cu121` → `.../cu128`
- `torch==2.8.0`、`torchvision==0.23.0`、`torchaudio==2.8.0`（cu128，cp310 均已确认可得）
- **移除** 5 个显式 `nvidia-*-cu12==12.1.*` / `cudnn==9.1.0.*` pin，由 torch cu128 wheel 传递依赖自动拉取 12.8.x / cudnn 9.10

**背景**：验证机 5070 Ti 是 Blackwell sm_120，旧 torch(cu121，≤sm_90) 不认这块卡（此即 handoff 所称"venv 损坏"真因）。

**理由**：cu128 支持 sm_120，且**向下兼容 Ampere sm_86（3090 实验机）**——同一份 pyproject/lock 两台机器通用，无需分叉。选 2.8.0 而非更新的 2.9/2.10/2.11：成熟稳定、Blackwell 支持完善。

**验证**（本机 5070 Ti，2026-07-01）：
- `torch 2.8.0+cu128`，`cuda available: True`，`capability (12,0)`，2048² matmul 真跑在 GPU 上 ✓
- 一期栈回归：transformers 4.57.1 / whisper / ctranslate2 4.6.0 / faster_whisper / silero_vad 全部 import 正常；`DynamicCache.crop` 存在 ✓
- `run_timeline_test` 仍 ALL PASS ✓

**影响**：
- `pyproject.toml` + `uv.lock` 已改（未提交，待用户决定 commit 时机）
- 本机 GPU 解锁：可跑 0.5B 全链路验证（含 CosyVoice2/Whisper GPU 路径）
- `run_test_simple.sh` 的 `LD_LIBRARY_PATH=.../nvidia/cudnn/lib` 仍有效（cudnn 9.10 仍装在该路径）

**回退**：`git checkout pyproject.toml uv.lock && uv sync` 即回到 cu121。

**状态**：accepted

---

## D-008（2026-05-21）反向映射表数据结构设计（PlaybackTimeline）

**决策**：
- 反向映射表实现为 **`PlaybackTimeline`**，落 `src/dialogue/timeline.py`。主干 = 按生成顺序排列的 **`FragmentRecord` 列表**（片段是截断单位，故以片段为主轴）。
- `FragmentRecord` 字段：`fragment_id / text / token_start,token_end / chunk_ids / sample_start,sample_end / status`（status ∈ SPECULATIVE/SYNTHESIZING/ENQUEUED/PLAYING/PLAYED/DISCARDED）。
- 反向查询：`playback_ms → samples → 二分查找命中片段 → token 边界`（sample_start 单调，O(log n)）。
- 并发：**一把锁罩整个 timeline**（操作极小，对话速率下竞争可忽略，不过早拆锁）；`played_samples` 游标原子 int 单独走。
- "已合成未播放"处理：打断时游标之后的 SYNTHESIZING/ENQUEUED 片段标 DISCARDED、token 被 crop。
- **mid-fragment 截断语义（选 A）**：打断落在片段中间时，该片段算"已听到"，截断到其 `token_end`（物理仍为片段边界）；若该片段被部分播放（partial）则置 rewrite 标记，供贡献3重写。

**背景**：handoff 方向1 的核心数据结构，KV 截断/推测浪费率/播放感知截断都依赖它。CPU + 0.5B 可验证，不需 GPU。

**理由**：片段主轴与"截断单位=片段"一致；单锁避免过早优化；选 A 与核心原则"历史=用户听到内容"及 P2"mid-fragment 触发重写"自洽。

**影响**：
- `src/dialogue/timeline.py` 实现 + `run_timeline_test.py` smoke（纯 Python，本机 CPU 可跑）
- 打断链路 `on_barge_in(playback_samples)` 返回 crop_token_end / discarded_ids / partial 标记，供 KV crop 与重写触发
- 对应 `experiment_design.md` §6 反向映射表落盘埋点

**环境备注**：本机 5070 Ti(sm_120) 当前 torch(cu121,≤sm_90) 不兼容，GPU 暂不可用（这是 handoff 所称"venv 损坏"的真因）。策略：核心 KV 逻辑先 CPU+0.5B 验证；需 CosyVoice2/全链路时再升 torch→cu128（兼容 3090）。

**状态**：accepted

---

## D-007（2026-05-21）实验设计四项基础决策（/experiment-agent plan 模式）

**决策**：
- **P1 打断产生**：确定性程序注入——"用户听到的"=注入时刻前已播放音频，ground truth 确定、可复现、无需真人/伦理审查
- **P2 打断时机**：混合——固定播放比例 25%/50%/75%（含 mid-fragment，触发重写）+ 片段边界对照（干净截断）
- **P3 一致性指标**：客观"未听到内容引用率"为主 + LLM-judge 连贯性为辅 + 人工小样本验证（Cohen's κ）
- **P4 语种/数据**：英文为主（MultiWOZ 派生 + 自构造英文打断集，对齐 HumDial-FDBench）；中文 CrossWOZ 为可砍扩展

**背景**：进入实验设计阶段，先定这四项决定后续所有实验能否测、代码要埋哪些点。完整设计见 `paper2/experiment_design.md`。

**理由**：均服务于"工程/系统贡献 + 一个月 deadline + 可复现"三重约束。程序注入避免真人成本；混合时机同时覆盖贡献2/3；客观主指标最扎实；英文为主对齐 benchmark 且省工。

**影响**：
- 确立被测系统条件 A / B-ours / B-gen / B-syn / B-noKV / B-naive|mark|rewrite
- 产出 instrumentation 埋点清单（`experiment_design.md` §6）作为 `src/dialogue/` 编码验收标准
- E4（buffer 精确映射对比）确认为锦上添花可砍（呼应 D-006）

**状态**：accepted

---

## D-006（2026-05-21）核心创新点重新定位（据 novelty 核查结论）

**决策**：贡献 2 的创新点从"提出'对话历史=用户实际听到的内容'原则"**降级/重新定位**为——

> **"首个开源、可复现的级联式播放感知上下文一致性管理实现 + 具体 KV 机制（`DynamicCache.crop` + `pre_attention_mask`/`position_ids` 同步重算 + ChatML role 边界重建）+ 可量化对比"**

三条硬约束（写作时必须遵守）：
1. **不把"历史=用户听到的内容"当作本论文 insight 来 headline**——Azure Voice Live 官方文档几乎逐字写过。它作为**组织性原则**可用，但必须**引用** OpenAI Realtime / Azure Voice Live / LiveKit 为 prior art。
2. **intro 显式引用上述商用系统先发制人**，堵审稿人。
3. **不得靠"商用系统做得粗/框架只做检测"立论**——这两条已被对抗核查 0-3 驳回（它们确实做了 played-vs-heard 历史管理）。合法差异只有：开源 vs 闭源、显式 KV crop vs 删 transcript、级联 vs 端到端、测量 buffer vs 假设实时速度。

**背景**：deep-research 核查（Task wi2gfobgx）判 (C) 部分重叠。概念被商用系统 pre-empt，但无学术/开源级联先例。完整报告 `docs/research_novelty_check.md`。

**理由**：
- 与 D-005 工程/系统框架完全一致——工程贡献不要求概念首创，要求开源+可复现+系统评测，正是商用闭源系统留下的空间
- 对硕士毕业论文门槛绰绰有余；对标邻居（RelayS2S/LTS-VoiceAgent/FireRedChat）都是 arXiv/workshop 级 preprint
- **代码工作量不变**——重新定位只改 framing 与引用，不改要实现的东西

**影响**：
- `paper2_context.md` §2.1 framing 已加 prior-art 护栏（本决策同批改）
- intro/related work 必须新增一段：商用系统现状 + 本工作与之的精确差异（用 `research_novelty_check.md` §三差异表）
- 论文可投层次：Interspeech / ICASSP / ASRU / SLT 系统方向（若投稿）

**关联决策**：**"最强 novelty 杠杆"实验**（量化 buffer 精确映射 vs 实时速度假设的 context 正确性差异）**列为"锦上添花"，不进主线**——一个月 deadline 下先保主 pipeline 完整；主体跑通且时间有余再做。需额外构造"合成速度≠实时"场景（TTS 快于播放/buffer 堆积），估 +3~5 天。

**状态**：accepted

---

## D-005（2026-05-21）论文定位：以工程/系统贡献为主骨架

**决策**：二期论文**主框架 = 系统贡献**——"在开源级联栈上实现播放感知的打断-上下文一致性管理"；"按用户实际播放位置截断 KV"作为该系统的**技术创新点/亮点**保留，但**不把论文成败押在它'全球首次'上**。贡献层级：
- **贡献 2（主，系统+技术亮点）**：播放感知 KV 缓存管理，必做
- **贡献 1（辅助）**：软触发推测生成，必做
- **贡献 3（扩展，可砍缓冲垫）**：对话历史自然化重写——时间不够时退化为"论文讨论 + 小规模验证"，甚至只保留零成本的标记法

**不采用"收窄到某个更细 novel 子点"路线**。

**背景**：论文目标 = **硕士毕业论文**，**预期一个月内完成编写**。当时 novelty 对抗核查（deep-research，Task wi2gfobgx）尚在跑，但本决策对报告结论 A/B/C 三种输出都鲁棒，故先定。

**理由**：
- 硕士学位论文评价尺度 = 工作量 + 系统完整性 + 实验充分性 + **一定的**创新性，不是顶会 novelty 门槛。~2000 行完整 pipeline + 系统实验本身即合格主体，与一期（流式架构 + KV prefill）同一评价逻辑
- "收窄"需要精密隔离实验证明某细点首创，更耗时且风险高（点被占则无退路），一个月预算承受不起
- 工程框架对 deep-research 结论鲁棒：判 A 则放大亮点，判 B/C 则作安全港，**不必等报告即可定**
- 工程框架允许干净砍范围（贡献 3 作缓冲），适配紧张时间线

**影响**：
- **实验目标简化**：从"证明全球首创"变为"在本系统上关键指标可测量改善 + 消融证明各组件有用"。核心对比实验（按播放位置截断 vs 按生成/合成位置截断，对多轮连贯性的影响）在自有系统内部自洽完成，不依赖外部 novelty
- 论文大纲与实验清单待 deep-research 报告回来后据此调整
- deadline 风险高，需以工程框架主动控范围

**状态**：accepted

---

## D-004（2026-05-21）重写模型选型

**决策**：使用 **Qwen3-0.6B** 作为对话历史自然化重写模型。

**背景**：贡献 3 的"重写法"分支在截断位置语义不完整时启用，并行运行隐藏延迟。输入 ~50 token，输出 ~60 token。

**理由**：
- Qwen3 系列 2025 年发布，比 Qwen2.5 更新，中文质量好
- 0.6B 规模在 3090 上推理 200-300ms，并行隐藏在用户说话期内
- 与主 LLM 同家族（Qwen 系），但**实例独立部署**，符合多服务工程现实
- 不在论文贡献范围内，不做模型选型消融

**影响**：
- `src/dialogue/rewriter.py` 加载 Qwen3-0.6B-Instruct 实例
- 实验机分卡布局：与软触发 + CosyVoice 共驻卡 1

**状态**：accepted

---

## D-003（2026-05-21）软触发模型选型

**决策**：使用 **TEN Turn Detection**（基于 Qwen 0.5B 微调的文本侧端点检测器，Apache 2.0）作为软触发主选；**不做候选模型消融**，软触发不是论文贡献。

**两阈值机制**：模型输出连续置信度，配两个阈值
- **推测阈值**（激进）：超过即触发主 LLM decode 进入"推测生成"
- **提交阈值**（保守）：超过才允许 TTS 开始播放给用户

> **实现注记（2026-07-29）**：提交阈值在本工作的确定性模拟 harness 中**未启用**——`orchestrator.py:speculative_turn` 仅用单一推测阈值（`spec_threshold`）启动推测，推测的提交（采用）由 ASR 段流终止的真值端点触发（P1 确定性模拟），无需第二阈值门控播放。此为 harness 简化，论文稿（abstract/C1/总结）已据此对齐为"推测阈值"表述；提交阈值作为真实部署的门控设计保留于此。

调整两阈值得到"推测浪费率 vs TTFT"trade-off 曲线（论文核心图之一，paper2_context.md §五）。

**背景**：候选过 Smart-Turn v2（音频侧，~20ms）、TEN（文本侧，50-100ms）、Phoenix-VAD（权重发布不确定）、Qwen prompted（最灵活但慢）。

**理由**：
- 文本侧检测的推理时间**与 KV prefill 并行**，挂在 prefill 的延迟阴影里，**实际零额外成本** —— 这是关键架构观察
- 文本侧错误更易复查与调优（端点判断错时可以打印当前累积文本看原因）
- 中英文双优，Apache 2.0，知名度足，论文里讲故事无争议
- 软触发不是论文贡献，**不需要做模型选型消融实验**

**影响**：
- `src/dialogue/trigger.py` 加载 TEN Turn Detection 实例（卡 1）
- 软触发输入是 ASR final 片段累积文本，触发判断与 LLM `_add_stream_prompt` 并行
- 论文中作为辅助模块描述，**不展开多模型对比**

**状态**：accepted

---

## D-002（2026-05-21）硬件配置、分支、主 LLM 规模策略、模型独立部署

**决策**：
1. **二期工作分支**：`bargeincache`（已切，不污染一期 main）
2. **验证机**：5070 Ti 16GB，主 LLM 用 0.5B 跑通 pipeline
3. **实验机**：3090 24GB × 2 = 48GB，主 LLM 用 7B，与一期实验对齐
4. **三个 LLM 实例完全独立部署**（主 LLM / 软触发 / 重写），不复用权重，模拟真实多服务工程

**3090×2 部署粗算（7B fp16）**：
- 卡 0：主 LLM(~14GB) + 长 KV(2-4GB) + Whisper-small(~1GB) ≈ 17-19GB
- 卡 1：CosyVoice2-0.5B(~2-3GB) + 软触发(~1-2GB) + 重写(~1-2GB) ≈ 5-7GB

**理由**：
- 与一期实验对齐，便于直接对比一期/二期数据
- 多服务独立部署反映工程真实，论文工程价值更可信
- 验证机用 0.5B 跑通，等架构 OK 再上实验机跑 7B，节省迭代时间

**影响**：
- `src/config.py` 二期需要支持**按模块**指定 device（主 LLM、ASR、TTS、trigger、rewriter 各自一项），一期目前只分了 asr_device/llm_device 两路
- 实验脚本要支持单卡（验证）/双卡（实验）两种 device map

**状态**：accepted

---

## D-001（2026-05-21）transformers KV cache 的对象类型与改造路径

**决策**：二期 KV 截断走 `DynamicCache.crop()` 路线。一期 `StreamLLMInference.KVCache` 中的 `past_key_values` 字段保持现状（"transformers 返回什么就用什么"），但二期新增的 KV 操作模块**显式断言**它是 `DynamicCache` 实例；若 transformers 实际返回 legacy tuple，则一开始就 `DynamicCache.from_legacy_cache()` 转换。

**背景**：一期 `src/llm/stream_llm_inference.py` 把 `past_key_values` 当作不透明对象在 `_init_kv_cache` / `_add_stream_prompt` / `generate` 之间传递，从未调用 cache 方法 — 无法从代码静态判断它到底是 DynamicCache 还是 legacy tuple。

**理由**：现代 transformers（4.36+）对 Qwen2.5 默认就返回 `DynamicCache`，`crop()` 自 4.39 起稳定。显式断言/转换让 KV 操作有一个稳定的契约面，二期不再被 transformers 内部默认行为牵着走。

**影响**：
- 二期新增模块（KV 截断、role 重建）依赖 `DynamicCache` API（`crop`、`__len__`、`key_cache` / `value_cache` 访问、`update`）
- 一期的 `KVCache` 数据类需要在二期版本里多带一个字段：**当前 cache 长度**（即 `past_key_values.get_seq_length()`），避免靠 `pre_attention_mask.shape[1]` 间接推断
- 风险：若实际运行的 transformers 版本不返回 DynamicCache，需在加载阶段统一转换

**状态**：accepted

---

## D-000（模板示例）

**决策**：[一句话决定了什么]
**背景**：[当时面临的问题 / 约束]
**理由**：[为什么这么选 — 与备选方案的对比]
**影响**：[改动哪些文件、引入哪些依赖、有哪些后续工作]
**状态**：proposed / accepted / superseded by D-xxx

---

> 这条 D-000 是模板，提交真实决策时删除或保留为占位。
