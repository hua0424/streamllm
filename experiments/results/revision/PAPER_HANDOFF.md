# 修订实验数据交接文档（论文编写用）

> **读者**：论文编写人。
> **内容**：CISR 修订补充实验（E1–E6 + TTFA 解码补测）全部完成并通过验收（2026-08-19 ~ 2026-08-21，GPU 实验机）。
> 本文给出每个实验的产物位置、论文可用数字、引用口径与注意事项。
> **配套**：`PAPER_IMPACT_NOTES.md`（影响与待决清单，7 项）、`REVISION_CHANGELOG.md`（执行流水）、
> `r1_stats/attribution/README.md`（平台差异归因证据链）。各结果目录内均有 `RUNINFO.md`（命令/时间/样本数/error 数）。
> **验收状态**：E1–E6 总验收 33/33 项通过（产物完整性、格式、config 锁定、error 率 0/2316、预期值区间、异常值扫描）；
> TTFA 解码补测（2026-08-21）跑后 QA 7/7 通过（见 `r6_ttfa/RUNINFO.md`）。

---

## 〇、引用口径（三条铁律，违反会被审稿人抓）

1. **平台绑定**：本次全部修订实验在第二平台完成（KVM 虚拟机，Xeon Gold 6133 @2.5GHz + 2×RTX 3090）。
   绝对延迟比原平台系统性偏高（System A +18%、System B +36~46%，归因证据见 attribution/README.md）。
   论文中引用本次实验的绝对毫秒数时**必须注明平台**；与原论文 Table III/IV 的旧绝对值**不得混排同栏**。
2. **禁缩放红线**（需求方裁决）：不对本机数字做任何缩放去对齐旧机绝对值。所有数字如实呈现。
3. **同机对比**：每个新结论只使用同机内产生的对比数字（本文所列对比均已满足）。

## 一、逐实验产物与可用数字

### E1 TTFT 稳定性（意见3）— `r1_stats/repeat_r{1,2,3}/`

50 个 very_long 样本 × 2 模式 × 3 轮，error 0。

| 轮次 | streaming TTFT mean | non-streaming mean |
|---|---|---|
| r1 / r2 / r3 | 1434.0 / 1482.1 / 1454.9 ms | 4063.3 / 4244.2 / 4118.5 ms |

- **论文可用**：跨轮 mean 极差 3.3%；逐样本 CV mean 4.2% / median 3.3% / p90 8.8%。结论："TTFT 测量可复现（3 轮 CV<5%）"。

### E2 真实语音 A/B（意见1）— `r2_real_speech/`

- 数据集自建（E2-0）：librispeech/aishell1 各 75 条（long 30/very_long 30/extra_long 15，seed=42 确定性可重建），
  12 个增强变体各 30 条；构建/增强 manifest 与 QA 报告（`qa_static.csv`、`qa_transcribe.csv`）齐全；
  转写 sanity：librispeech WER 2.98% / aishell1 CER 6.72%（验收线 ≤10%）。
- 干净集（各 75×2，error 0）：

  | 数据集 | System B extra_long | System A extra_long | 改善率 |
  |---|---|---|---|
  | librispeech | 1559ms | 4805ms | **67.5%** |
  | aishell1 | 1763ms | 5140ms | **65.7%** |

- 增强集（各 30×2，error 全 0）：snr20/15/10、speed09/11 共 10 组改善率 **+22.8%~+31.6%**；
  **babble 例外**：librispeech_babble −47.3%（流式反而更慢）、aishell1_babble +6.5%——VAD 对 babble 过度触发
  造成段积压 + whisper 噪声段空输出（已归因，非缺陷）。
- **论文可用**：真实语音上流式优势成立（与原论文合成集结论同量级）；babble 为诚实披露的边界条件，
  建议进 limitations。⚠️ 分析注意：空输出样本的 `asr_time`/`llm_prefill_time` 是哨兵值（`last_text_time=0`），
  统计这两个字段时剔除零提交样本；`ttft` 不受影响。
- ⚠️ **zh CER 口径脚注（引用 aishell1 CER 必加）**：参考文本中文数字 vs Whisper 阿拉伯数字写法失配
  影响 49/75 样本（受影响 mean CER 0.1476，未受影响仅 0.0313）；不注明会被质疑与 librispeech
  WER 的反差（changelog 2026-08-21 已登记）。
- ⚠️ speed 变体时长重判分组出现 medium 子组（librispeech 5 条、aishell1 3 条）；Table VI 逐条件行
  建议用 `overall` 行，或注明分组按变速后时长重判。

### E3 LocalAgreement-2 基线（意见4）— `r3_baseline_la/`

498 消融清单样本，三方同机（本机重跑 System A/B 消除跨机污染，需求方裁决项）：

| 系统 | mean TTFT | long | very_long | extra_long |
|---|---|---|---|---|
| System A（非流式） | 5310.8ms | 1958 | 3906 | 7698 |
| **System B（本文）** | **1573.9ms** | 1464 | 1551 | 1638 |
| LA-2 基线 | 2115.0ms | 1741 | 2200 | 2230 |

- LA 质量：WER mean 0.130 / CER 0.118（与 System B 同引擎同量级）；divergence mean 1.0。
- **论文可用**（口径修正 2026-08-21）："**LA-2 基线比 System B 慢约 34%**（LA 2115.0 vs B 1573.9ms；
  等价表述：B 比 LA 低约 26%——两种表述不得混用），同等转写质量下成立（LA 需全缓冲重解码），
  且 LA 在长音频上退化更明显（very_long 以上 LA 2230ms vs B 1551–1638ms）"。
- ⚠️ **方法描述必须写修复后语义**（绝对时间轴提交 + 句界裁剪 + la_max_buffer_s=15.0），不得只写
  "LocalAgreement-2"（评审 R2 保留项）；LA 实现经历一次 bug 修复重跑，过程文档在
  `r3_baseline_la/handoff/` 与 `experiments/review/20260820-E3LA/`（如被追问可出示）。
- 产物：`la_results/la_summary/la_statistics` 三件套、`system_ab_rerun/`（A/B 本机数字）、
  `e0_smoke/`（修复后冒烟）、`invalid_dev3_frame_bug/`（旧无效现场，勿入表）。

### E4 插桩复跑：提交分歧日志 + 完整回复（意见5）— `r4_commit/`

50 样本 × 2 模式（max_tokens=128），error 0。

- `commit_log.jsonl`：599 行 = 375 commit + **224 correction**（已提交段在后续重识别中文本漂移，下游不可见）；
- `full_response` 50/50 完整（mean 208 字符）供分词接缝与语义一致性分析；`committed_fragments` 50/50。
- **论文可用**（2026-08-21 按完整统计修正，不得再用"同音字/标点级"表述）：
  "append-only 对下游输出成立（回滚下发 = 0，构造保证）；内部重识别漂移实测存在
  （224 次 / 49~50 样本，涉及段 52.7%），编辑距离 mean 2.3 字符、p90 6、max 16
  （归一化比率 mean 5.6%、max 47.1%，含实词级漂移，见 commit_divergence.json top 示例）；
  下游不可见"——若原文有"内部从不变化"类强声明需按此软化。

### E5 端点等待（意见2）— `r6_ttfa/endpoint/`

50 样本（尾拼 2s 静音），error 0；QA 四项全过（间隔 2.002s、无 asr_no_speech、时序无违例）。

| 指标 | mean | 说明 |
|---|---|---|
| endpoint_detection_wait | **53ms**（median 109ms，p90 208ms） | 语音结束→最后语音段入队 |
| post_endpoint_ttft | 3012ms | 末段 ASR+LLM（含本机平台偏移） |
| total（speech_end→首 token） | 3065ms | 管线侧尾延迟 |

- **论文可用**：本管线（VAD 闭段驱动 + suffix=0）端点检测等待 ~0.05–0.11s，不是瓶颈；
  尾延迟主体在末段 ASR+LLM。`final_enqueue_wait`≈2s 是测量装置属性（等追加静音推完），勿当系统等待。

### E6 TTS 首包（意见2）— `r6_ttfa/tts_first_chunk.csv`

50 条 E4 真实回复（中英各 25），error 0（服务：CosyVoice-300M-SFT，spk 晓伊→中文女 别名，speed 0.8）。

- TTFC mean：**zh 13.99s / en 11.86s**（回复均长 ~200 字符）；RTF ~0.71；total ~33–35s。
- 冷热启动对照已排除预热因素（14.84s vs 12.92s）；**TTFC 与文本长度近似线性**（~0.09s/字符：
  3 字符→1.4s，45→7.6s，200→17.9s），该部署为句段级流式（句内不流式）。
- **论文可用**：TTFA 预算中 TTS 首包是最大项（≫ 管线侧 3.1s ≫ 端点 0.05s）。若原文暗示"TTS 流式
  首包很快"需修正；建议给 TTFC-长度关系或限定短回复场景。注意本机 CPU 放大 TTFC（平台披露覆盖）。

### TTFA 解码至首句补测（意见2，预算表第三项）— `r6_ttfa/decode_to_first_sentence.csv`

**为什么有这次补测**：E4 复跑未记录逐 token 时刻，TTFA 预算表缺"首 token → 首个句末标点 token"分项且
无法离线恢复；2026-08-21 用 E4 落盘的 `committed_fragments` 逐片段重放生产增量预填序列后逐 token 计时
补测（方法与三轮评审记录见 `r3_baseline_la/handoff/E6_TTFA_DECODE_FIRST_SENTENCE_HANDOFF.md`）。
50 样本（与 E4 同 50 条），error 0，QA 7/7 全过。

| 分项 | mean | p50 | p90 | 分语种 |
|---|---|---|---|---|
| decode_to_first_sentence | **389.0ms** | 92.0ms | 967.1ms | en 635.7ms / zh 142.3ms |

- `sentence_end_found` 50/50 = 100%（全部回复在 max_tokens=128 内出现句末标点）；解码速率 mean 26.0 tok/s。
- 中英文差异来自首句长度（英文首句 token 数更多），非速度差异（分语种 tok/s 相同）。
- **论文可用**（2026-08-21 审查 P0 裁决=方案2，对称剔除 2s 装置等待；以此为准）：
  `TTFA = T_endpoint(E5: 53ms) + T_post_final_enqueue(E5: 1012.5ms) + T_decode_to_first_sentence(本测: 389ms) + T_TTS_first_chunk(E6: zh 13.99s / en 11.86s)`；
  Table VIII 装配稿 `r6_ttfa/ttfa_budget.csv` 已按此口径生成：**B ALL 14.38s / A ALL 22.67s**。
  逐项 mean±std 由 `r6_ttfa/decode_to_first_sentence.summary.txt` 与各实验 RUNINFO 取数。
- ⚠️ 口径要点（审查 P0）：E5 的 `final_enqueue_wait≈2s` 是实时喂追加静音的**测量装置等待**，
  已从 B 行 post 分项剔除（A 行 ttft 从 audio_end 起算本就不含）——两系统口径对称；
  B 行 post 为 final 段入队→首 token 的真实处理（末段 ASR + LLM 预填解码，1012.5ms）。
  论文正文须写明该剔除规则。
- ⚠️ CSV 中 `first_token_latency_ms` 是重放口径参考值，**不是** TTFT。

## 二、环境存档与可追溯

- `env_versions.txt`（项目 venv 真实版本 + nvidia-smi）；repo 状态见 git log（结果已随 main 提交）。
- 配置锁定（全部运行一致）：turbo / Qwen2-7B-Instruct / cuda:0+cuda:1 / chunk 500ms / prefix 1 /
  **suffix 0** / recognition_threshold 2.0s / max_tokens 50（E4 为 128）/ 预热 3 轮。
- 每次运行均有 RUNINFO.md + run.log；changelog 逐条登记。

## 三、待论文侧决策清单（汇总自 PAPER_IMPACT_NOTES.md）

> 注：TTFA 预算表的数据缺口（`T_decode_to_first_sentence`）已于 2026-08-21 补测关闭，四组成项数字齐备，
> 仅剩装配（机械工作，本机侧进行），**不属于**待决项。以下为仍需论文侧拍板的 6 项：

1. LA 对比表是否引用 exp2 中间消融臂 `streaming_asr_only`（旧机数字 1171.0ms；引用需脚注或补跑 +4h）；
2. babble 结果进正文的口径（10 变体 vs 12 变体）与 limitations 措辞；
3. 摘要/结论中改善率表述：建议"70%–74%（两平台复现）"或绑定平台（原平台 74.3%/65.6%，本平台 70.4%/64.7%）；
4. E6 是否需要在原机/高主频 CPU 机器补测对照；
5. "append-only"表述精确化（见 E4 条目）；
6. TTS 首包结论的长度限定（见 E6 条目）。
