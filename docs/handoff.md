# 二期实验执行交接文档（HANDOFF）

> 面向接手的下一个 agent / 实验机会话。目标：**在实验机（3090 24G ×2）上跑出论文的正式实验数值**。
> 本文档只讲"怎么跑、跑什么、要什么结果"；设计与指标定义见 `paper2/experiment_design.md`（含 §9' harness 状态表），决策史见 `docs/decisions.md`（D-001~D-012），项目全景见 `docs/paper2_context.md`。
> （2026-05-21 的旧版 handoff 是设计期交接，已被本版取代，内容存于 git 历史。）

**生成时间**：2026-07-02
**分支**：`paper2`（HEAD `9567785`，已推送）
**代码状态**：全部 6 个实验 harness 在验证机（5070 Ti，0.5B）跑通；经**两轮代码审查**（发现 3 BUG + 5 minor，全部修复，见 D-012），回归全绿。**已知零未修 BUG**。

---

## 一、实验机环境准备（一次性）

```bash
git clone <repo> && cd streamllm && git checkout paper2   # 或已有仓库 git pull
uv venv --python 3.10 && uv sync        # torch 2.8.0+cu128（兼容 3090 sm_86，D-009）
cp .env.example .env                     # 然后按下表改
export LD_LIBRARY_PATH=".venv/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
```

`.env` 实验机配置（P2_* 变量是二期模型开关，改这里即可全局换模型，**不用改代码**）：

```bash
HF_HOME="<实验机模型缓存目录>"
HF_TOKEN=""                                    # 留空匿名下载（历史 token 已失效，勿用）
P2_LLM_MODEL_NAME="Qwen/Qwen2-7B-Instruct"     # 主 LLM 换 7B（与一期实验对齐）
P2_TRIGGER_MODEL_NAME="TEN-framework/TEN_Turn_Detection"   # 软触发换 TEN 7B（见 §四.2）
P2_REWRITER_MODEL_NAME="Qwen/Qwen3-0.6B"
P2_DEVICE="cuda:1"                             # trigger/rewriter 放卡1；主 LLM 由 --asr/--llm 逻辑走卡0
```

分卡布局（D-002/D-011）：卡0 = 主 LLM 7B(~14G)+KV；卡1 = TEN 7B(~15G)+CosyVoice2(~3G)+Qwen3-0.6B(~1.5G) ≈ 19.5G<24G。

**环境验证**（跑通即环境 OK，纯逻辑不费时）：
```bash
uv run python -m src.dialogue.run_timeline_test      # 纯 Python，秒级
HF_TOKEN= uv run python -m src.llm.run_kvcrop_test   # 首跑会下载 P2_LLM 模型
HF_TOKEN= uv run python -m src.dialogue.run_speculative_test
```
三个都 `ALL PASS ✓` 才继续。

---

## 二、六个实验：命令、产出、预期结果

全部从**项目根目录**跑；结果 JSON 入 `experiments/results/`；E2/E3 支持断点续传（中断重跑同命令即续）。每个脚本都有 `--model`（默认取 P2_LLM_MODEL_NAME）。

| # | 命令 | 产出 | 预期结果形状（0.5B fixture 参考 → 7B 应更显著） |
|---|---|---|---|
| **E3**（核心） | `uv run python -m experiments.scripts.run_exp3_consistency --dialogues <MultiWOZ派生.json>` | `exp3_consistency.json`：summary 含 **loose/strict 双列**未听引用率 + 逐条 records（含 timeline 映射落盘） | loose：B-ours **恒 0%**（构造性保证，论文如此表述）vs B-gen 显著>0（fixture 67-75%）；strict：双列都>0，B-ours 的 strict 值=**片段粒度量化误差**（诚实补充列，见 §五.1 注意事项） |
| **E2**（核心图） | `uv run python -m experiments.scripts.run_exp2_tradeoff --dialogues <segments格式.json>` | `exp2_tradeoff.json`：curve 数组（threshold→waste_rate/ttft_eff/存活率） | 单调 trade-off 曲线 + 明显拐点（fixture：th0.02→30%浪费/0.5ms ↔ th高→0%/43-75ms；**7B 的 TTFT 全额会到数百 ms，曲线张力更大**）。⚠️ 换 TEN 后阈值扫描区间必须重标（§四.2） |
| **E1** | `uv run python -m experiments.scripts.run_exp1_latency --dialogues <segments格式.json>` | `exp1_latency.json`：A/B 的 TTFT + 建模 mouth-to-ear | B TTFT_eff ≈0 vs A 全额；m2e 建模值 B<<A。**real CosyVoice2 实测 m2e 见 §四.3** |
| **A1** | `uv run python -m experiments.scripts.run_exp_a1_kvreuse --lengths 256 512 1024 2048 4096 8192` | `exp_a1_kvreuse.json`：crop/role/re-prefill 三条延迟 vs 上下文长度 | crop 亚 ms 近常数（barge-in 响应延迟卖点）；re-prefill 线性涨（**7B 下更陡，加速比更大**）。计时已用 median 抗噪；趋势异常只 WARN，见输出提示空载重跑 |
| **A2** | `uv run python -m experiments.scripts.run_exp_a2_history --dialogues <turns格式.json>` | `exp_a2_history.json`：三策略历史样例 + rewrite_ms + **judge_coherence 字段（null，待 §四.4 填）** | 三策略跑通；rewrite mean 数百 ms（"可隐藏"论点）；连贯性结论靠 LLM-judge |
| **A3** | 无独立脚本 | 复用 `exp2_tradeoff.json` 的 records | 逐阈值分解报告（与 E2 同数据） |

**建议执行顺序**：A1（最快，验证 7B 环境）→ E3 → E2（最耗时，扫阈值×全数据）→ E1 → A2。

---

## 三、实验数据准备（跑正式实验的前置）

本机验证全用内置 fixture；正式数值需要 **MultiWOZ 派生数据**（D-007 P4：英文为主）。三种输入格式（都是 JSON 列表）：

1. **E3/A2 用 turns 格式**：`[{"id": "...", "turns": ["user轮1(被打断轮)", "probe轮2", "probe轮3", ...]}]`——turn1 要能引出多部分回答，probe 轮诱导复述（≥3 轮，§4）
2. **E2/E1 用 segments 格式**：`[{"id": "...", "segments": ["片段1", " 片段2", ...]}]`——段边界=停顿点，部分首段句法上近似完整（制造假停顿）；段自带前导空格
3. 规模（experiment_design.md §4）：每条件 50-100 段对话；从 MultiWOZ 的 user 轮抽取+切分（切分脚本**尚未写**，是实验机上第一件开发工作；一期 `experiments/datasets/tools/` 有 MultiWOZ 处理工具可参考）

---

## 四、实验机准备步骤（脚本已全部备好并本机验证，2026-07-02——**照单执行即可**）

1. **派生 MultiWOZ 数据**（脚本已验证，兼容 2.0/2.1 与 2.2 格式，切分严格无损）：
   ```bash
   uv run python -m experiments.scripts.prepare_multiwoz_data \
       --input experiments/datasets/raw_data/MultiWOZ/data.json \
       --out-turns experiments/datasets/processed/p2_turns.json \
       --out-segments experiments/datasets/processed/p2_segments.json --max-dialogues 100
   ```
2. **TEN 7B 接入 + 阈值重标**（标定脚本已验证，替身 AUC 0.84）：.env 设 `P2_TRIGGER_MODEL_NAME=TEN-framework/TEN_Turn_Detection` 后：
   ```bash
   HF_TOKEN= uv run python -m experiments.scripts.calibrate_trigger --config ten
   # 产出 suggested_thresholds → 传给 run_exp2_tradeoff 的 --thresholds
   # 脚本自带 AUC>=0.65 可分性体检，过不了会拒绝放行 E2
   ```
3. **real CosyVoice2**（⚠️ 唯一未真机验证的部分——适配器与 benchmark 已写好、编译通过）：
   按官方 requirements 在**独立环境**装（pin torch 2.3.1+cu121，勿污染主 uv 环境）；
   适配器 `src/tts/cosyvoice_tts.py`（StreamingTTS 接口，守卫式导入）；然后：
   ```bash
   uv run python -m experiments.scripts.benchmark_cosyvoice --ref-audio ref.wav
   # 产出 cosyvoice_profile.json → 三个实测值回填 TimingProfile 与 SYNTH_RTF，重跑 E1/E2/E3
   # E1 实测 mouth-to-ear：编排层 tts=CosyVoiceStreamingTTS(...)
   ```
4. **LLM-judge**（脚本已验证：替身裁判下 loose κ=0.71 / strict κ=0.23，印证规则 strict 是噪声上界、必须 judge 交叉）：
   ```bash
   HF_TOKEN= uv run python -m experiments.scripts.run_llm_judge e3 \
       --results experiments/results/exp3_consistency.json --judge-model <与主LLM不同家族的强模型>
   HF_TOKEN= uv run python -m experiments.scripts.run_llm_judge a2 \
       --results experiments/results/exp_a2_history.json --judge-model <同上>
   # 产出 *_judged.json（judge率 + Cohen κ / judge_coherence）；另抽 ~50 条人工验证裁判可靠性（P3）
   ```

---

## 五、写论文的人需要知道的三个结果解读要点

1. **E3 的 loose 0% 是构造性保证，不是实验发现**（D-012 BUG1 修正）——论文表述："B-ours 由机制保证历史不含未听片段（loose=0），实验量化的是 B-gen 的失败率与 B-ours 在严格 ground-truth 下的片段粒度量化误差（strict 列）"。strict 列在规则检测器下是**上界**（诱导复述型 probe 易误报），正式解读以 LLM-judge 交叉验证为准
2. **barge-in 响应延迟 = 反查+crop（亚 ms、与上下文无关）**；role 重建（~十几 ms）不在关键路径（可延迟到下轮输入前）——A1 数据支撑"打断即停"卖点
3. **B-syn（合成位置截断）在 Mock 同步合成下与 generation 等价**，只有接入异步 real CosyVoice2 后才可区分——论文不得称"已验证 B-syn"；若时间不够可砍（可砍项清单见 experiment_design.md §5 E4 与 D-005 贡献分层）

---

## 六、坑与提醒

- **HF_TOKEN 必须为空/有效**：历史 token 已失效会让公开模型也 401（跑命令前缀 `HF_TOKEN=` 最稳）
- 模型加载 offline-first：首跑联网下载后即可离线
- `.env` 已 gitignore、每机自维护；勿提交真实 token
- A1 若趋势 WARN：GPU 有其它负载，空载重跑（数据已落盘，不会丢）
- E2/E3 断点续传按 (id, threshold/fraction, condition) 去重——**换数据集/模型后务必删旧结果 JSON 再跑**，否则旧记录混入聚合
- 提交约定：短祈使句中文；决策倒序追加 `docs/decisions.md`；里程碑同步 `paper2_context.md` §九；改方法学同步 `paper2/experiment_design.md`

---

## 七、建议调用的 skills

- **`code-review`（/code-review）**：写完 MultiWOZ 数据脚本 / CosyVoice2 接入类 / LLM-judge 后，跑一轮再进正式实验（本项目两轮审查抓出 3 个会污染实验数字的 BUG，值得保持）
- **`verify`**：CosyVoice2 接入后验证真实音频链路行为
- **`experiment-agent`（/experiment-agent）**：`validate` 模式可用于正式结果 JSON 的统计解读与 11 类统计谬误扫描（论文第六章写作前过一遍）；`run` 模式可托管长实验的监控
- **`loop`（/loop）**：E2 全量扫描耗时较长，可配合后台监控

---

## 八、关键路径速查

- 实验设计+harness 状态表：`paper2/experiment_design.md`（§9' 是总控表）
- 编排器（所有实验的核心）：`src/dialogue/orchestrator.py`
- 软触发（TEN 接入点）：`src/dialogue/trigger.py`（`TEN_CONFIG` 现成）
- TTS 接口（CosyVoice2 实现点）：`src/tts/streaming_tts.py`（`StreamingTTS` ABC + `TimingProfile` 占位值）
- 检测器（LLM-judge 交叉验证对象）：`src/dialogue/unheard_detector.py`
- 论文正文：`paper2/`（outline.md + chapter2 初稿；第三章待写，指标定义与实验设计已咬合）
