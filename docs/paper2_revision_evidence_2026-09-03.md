# Paper 2 二审统一修订证据合同（2026-09-03）

> 本文件是本轮论文改写的唯一数值与主张合同。若与旧正文、旧 analysis_v1 或 reviewer 近似复算冲突，以正式 raw 派生的 versioned analysis_v2、D-023 和本文件为准。不得据此修改 accepted raw/manifest/analysis_v1/seal。

## 1. 贡献层级

1. **核心贡献 C2**：software-consumed-sample cursor → TTS text fragment → assistant token span → KV crop → mask/token ledger/position/role/EOT state recovery；正式正确性证据为 C2 v3 direct crop-integrity addendum。
2. **支持性 C1**：pre-end-of-turn candidate-response generation 的 candidate-selection/compute-readiness、oracle acceptance 和 wasted-token 工作点刻画；不是 speculative decoding，不是 production deliverability 改善证明。
3. **探索性 C3**：naive/mark/rewrite 历史自然化实现与受混杂负结果；不得作策略因果比较。
4. 高层“history 应反映 delivered/spoken output”是 OpenAI/Azure/LiveKit prior art，不是本文原创原则；KV crop primitive 也不是创新。

## 2. 术语合同

- `first_token_ready`：**首候选 token 选择/内部计算就绪事件**（first-candidate-token selection / candidate compute-readiness）。回调发生于 token selection 后、cache-update forward 与 generator yield 前；不是可交付 token、consumer observation、TTS admission 或 acoustic output。
- `endpoint_accept`：同步 harness 中的 **post-candidate oracle acceptance**；不是自然端点检测输出，也不是最后文本段到达瞬间。
- `first_deliverable_token` / `consumer_delivery`：仅称 **同步 harness marker/diagnostic**，不称 production deliverability。
- `p`：**software-consumed-sample cursor**；不等于 device-presented samples 或 acoustically heard content。
- `H_hat(p)`：**TTS-fragment-level software retention boundary**；不是用户实际听到的逐词/token 真值。
- legacy `heard_text/n_heard/strict_unheard`：artifact compatibility alias；语义仅为 fragment retention 或 character-proportional whitespace-snapped proxy。
- E3：**fixed-detector-conditioned information-reproduction rate**；不是人类语义真值/HCI 效果。
- 本文 speculation：**pre-end-of-turn candidate-response generation with invalidation**；不是 draft-target speculative decoding。

## 3. E1/E2 crossed analysis_v2

Artifact：`experiments/sci34_supplement/results/e1e2_confirmatory/e1e2c_b8c758b_20260901T173306Z/analysis_v2.json`  
SHA-256：`9bce6db5d93c1faccb4069b295df32ce5ee0778899b31ac6be17526bfb644456`

设计：100 unique utterances × 5 independently initialized process sessions × 10 conditions；每条件 500 observations，但内容采样单位是 100 utterances、session 是技术重复。Crossed/product bootstrap 独立抽样全局 session 与 dialogue，再取笛卡尔积；10,000 repeats，seed 20260901，percentile 95%。

### 3.1 正式点估计与 crossed CI

| Estimand | Point | 95% CI |
|---|---:|---:|
| C-E1 candidate readiness，System A − B@0.92 | −34.6877 ms | [−35.4421, −33.9535] |
| C-E2 candidate readiness，never − B@0.92 | −0.03349 ms | [−0.63861, 0.61494] |
| C-E1 oracle TTFT_eff lower bound，A − B@0.92 | +17.4367 ms | [14.4079, 20.3234] |
| C-E2 oracle TTFT_eff lower bound，never − B@0.92 | +20.8037 ms | [17.8492, 23.6450] |
| B@0.92 pooled waste | 2.8527% | [1.1239%, 4.7345%] |
| B@0.92 survival | 67.0% | [58.0%, 76.0%] |

### 3.2 条件事件均值（仅受控同步 harness）

| Condition | arrival→candidate selection | arrival→endpoint accept | arrival→first-deliverable marker | arrival→consumer marker | oracle TTFT_eff |
|---|---:|---:|---:|---:|---:|
| System A | 27.6959 ms | 0.0 | 27.6959 ms | 51.8937 ms | 27.6959 ms |
| B@0.92 | 62.3837 ms | 247.3239 ms | 257.5832 ms | 265.5727 ms | 10.2592 ms |
| B-never | 62.3502 ms | 31.2873 ms | 62.3502 ms | 86.6751 ms | 31.0629 ms |

257.58/265.57 ms 受同步执行顺序支配，只作 diagnostic，不作为系统 headline。291 ms 只能称 candidate-first-selection 到 post-candidate oracle acceptance 的内部间隔中位数，不是自然端点 lead 或用户继续说话时长。

### 3.3 C-E1 output identity 与因果边界

- A vs B@0.92 full `output_token_ids` exact：280/500（56%）。
- first token exact：465/500（93%）。
- length/EOS/max-token agreement：495/500（99%）。
- 44/100 unique utterances 至少一次 full-output mismatch；五个 session 中每个均为 44/100，状态跨 session 稳定。
- B@0.92 vs B-never：full tokens/first token/length/EOS/max-token/text 全 500/500。

因此 C-E1 是 **implementation-path comparison**，差异混合 full-string vs segment-wise tokenization、forward topology/shape、role boundary、kernel 与 Python scheduling；不得归因于“纯 incremental-prefill effect”或单一额外 forward。不得按 280 matched outputs 过滤主延迟（post-treatment selection）。C-E2 是 token-consistent B-path comparison。

## 4. E3 weighting/dedup analysis_v2

Artifact：`experiments/sci34_supplement/results/e3/sci34_f11ccba_20260901_e3/analysis_weighting_dedup_v2.json`  
SHA-256：`5776db23e534767c6ca266872967e228962eef59924cec3a1e3dd5cdbcd30366`

设计：100 dialogues、400 `(dialogue,injection_label)` pairs、800 condition records、1600 judge records；bootstrap 以 dialogue 为 cluster，10,000 repeats，seed 20260831。主表采用 label-weighted estimand，并用同一 label-weighted cluster bootstrap，点估计与 CI 同 estimand。另报告 dialogue-weighted 与 target-specific unique-semantic-boundary sensitivity。

### 4.1 主表：label-weighted（generation − playback）

| Target/detector | Playback | Generation | Difference | 95% dialogue-cluster CI |
|---|---:|---:|---:|---:|
| Fragment/rule | 67.00% (199/297) | 63.64% (189/297) | −3.37 pp | [−10.49, 3.40] pp |
| Fragment/judge | 42.76% (127/297) | 40.74% (121/297) | −2.02 pp | [−10.70, 6.13] pp |
| Proxy/rule | 75.26% (286/380) | 73.68% (280/380) | −1.58 pp | [−6.08, 2.67] pp |
| Proxy/judge | 43.95% (167/380) | 41.32% (157/380) | −2.63 pp | [−8.57, 2.90] pp |

### 4.2 Dialogue-weighted effect/CI（作为 estimand 对照）

- Fragment/rule：−3.21 pp [−9.55, 2.78]
- Fragment/judge：−1.30 pp [−8.94, 6.08]
- Proxy/rule：−1.50 pp [−5.75, 2.50]
- Proxy/judge：−2.58 pp [−8.25, 2.67]

### 4.3 Unique semantic boundary sensitivity

- Fragment：297 eligible labels、96 dialogues → 169 unique semantic groups，移除 128 重复 label weight。
  - Rule group-weighted：71.60% vs 68.64%，−2.96 pp [−9.04, 2.63]
  - Judge group-weighted：43.20% vs 43.20%，**0.00 pp** [−7.98, 7.47]
- Proxy：380 labels、100 dialogues → 379 groups，仅移除 1 个重复。
  - Rule group-weighted：75.20% vs 73.61%，−1.58 pp [−6.10, 2.69]
  - Judge group-weighted：43.80% vs 41.16%，−2.64 pp [−8.57, 2.90]

所有 weighting/dedup 结果均不支持 superiority、equivalence、noninferiority、harm 或 absence-of-effect。CI 仅表示 dialogue-sampling uncertainty conditional on frozen lexical rule、specific-reference-v3 Mistral judge、targets、trajectory、prompt 与 40-token cap；不包含 detector error、prompt/model variation 或 human perception error。

Automated proxy agreement（不得称人类验证）：label fragment 370/594 agreement；label proxy 442/760；unique-group fragment 207/338；unique-group proxy 440/758。

## 5. C2 v3 accepted exact-only 证据

Run：`c2crop_82103004_20260903T080512Z`；code `8210300`；result `7d50624`；manifest `d8c3db4d…`；seal `e0997d41…`。

- 24/24 cases、27/27 crop events、3 no-op、60 recovery steps；28 层 K/V；308 fixture tokens 逐 token production append；27/27 wrong-length negative control。
- 每 event：pre-crop retained prefix = production post-crop = independent slicing oracle，逐层 shape/dtype/device/hash 与 runtime `torch.equal` exact；keep/mask/token/seq/KV exact。
- Matched recovery：相同 token-ID chunks 后 K/V、logits、mask、token ledger、retained prefix 与 role/end/content state exact。
- 允许主张：tested snapshot/backend 下 direct crop integrity 与 matched-recovery determinism。
- 禁止：clean-reprefill numerical equivalence、v2 passed、32-token continuation equivalence、跨模型/backend/硬件、online audio/production correctness。

### v1/v2 透明度

v1/v2 clean-reprefill 协议均按冻结门槛 rejected。v2 24/24 probe 与 45/45 token/state/EOT/scenario 通过，但单控制 2× numerical gate 42/45；control 与 production forward topology 不匹配，因此三项失败既不能识别 crop bug，也不能建立 clean-reprefill equivalence。v3 是 direct crop-integrity addendum，不改变 v1/v2 verdict。

## 6. A1/P1 与 A2 报告边界

- A1：固定 operation order，固定移除 32-token suffix；5 warmups、50 repeats；256–8192 context，joint crop+role median 31.054–48.315 ms，IQR 0.635–3.099 ms，re-prefill/joint median ratio 2.254–40.620。只限该固定协议，不代表自然打断位置/其他 crop length。
- P1：9 cells×20；P95 仅称 empirical/descriptive order statistic（每 cell 20 个值，主要由 1–2 个上尾观测决定），不称 production SLO。software cursor/headless only。
- A2/RQ5：改为描述性问题：“当前探索性运行中三种实现的连贯性分数与重写耗时如何？”不得问/答“是否改善”。

## 7. 播放与 novelty 边界

- OpenAI、Azure、LiveKit 已建立 playback-conditioned transcript/session-history truncation prior art；本文不主张高层原则原创。
- crop/prefix reuse 也是 prior art。
- 可辩护 novelty：在报告的公开来源范围内未识别同时公开 software cursor→fragment→token→KV crop→role recovery 和可复算 exact/latency evidence 的 cascaded implementation；这是 scoped non-identification，不是 global first。
- 检索方法与限制：`docs/novelty_search_2026-09-03.md`。

## 8. Artifact/declaration 状态

- Artifact matrix：`REPRODUCIBILITY.md`。
- E3 exact input rescue：`experiments/sci34_supplement/results/e3_exact_rescue/README.md`。
- Declarations draft：`paper2/declarations.md`。
- 不得编造作者、基金、COI、伦理、consent、license、公开 URL/DOI。正文 submission declaration 可链接 draft 并保留明确 AUTHOR CONFIRM，占位未解决前不得写“none/not applicable”。
