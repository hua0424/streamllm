# E1/E2 确认性重跑实施计划

## 总体判断

方案 2 更有利于保留 E1/E2 定量结果，但不能直接重跑旧脚本。旧 runner 存在 TTFT 埋点、阈值事后选择、无独立重复、正式输入未归档和 resume 不安全等审稿风险。新实现将建立独立的 **confirmatory controlled text-segment campaign**：用于确认同步文本段、真值端点接受协议下的定量结果，不声称真实 ASR、在线 TTS、声卡或生产端到端时延。

新结果可能不同于旧的 48.3→12.1 ms。论文后续必须采用新结果，不能为了保留旧数字而调整协议或筛选数据。

## 1. 独立 campaign 与不可变旧工件

在 `experiments/sci34_supplement/e1e2_confirmatory/` 新建代码和文档，结果写到：

```text
experiments/sci34_supplement/results/e1e2_confirmatory/<campaign_id>/
```

永久不修改：

- `experiments/results/exp1_latency.json`
- `experiments/results/exp2_tradeoff.json`
- `experiments/results/paper2_reanalysis.json`

结果目录包含 campaign/input/trigger/session manifests、append-only raw JSONL、aggregate analysis、环境快照、日志、checksums 和验收文件。新增专用 ignore 规则，验收后再显式 `git add -f` 入库。

## 2. 新的未见 holdout

复用 `prepare_multiwoz_data.py` 的确定性切分逻辑，在 GPU 主机从本地 MultiWOZ 2.1 派生 100 条新话语：

- 排除旧 E1/E2 的 100 个 ID；
- 排除固定轨迹 E3 的样本 ID；
- 固定 seed、筛选与切分规则；
- 保存完整 `id/full_text/segments`、原始数据哈希、排除列表和输入 SHA-256；
- formal 模式禁止 fixture、重复 ID、空 segment、缺文件和联网下载。

## 3. 冻结条件与参数

每个 session 对同一 100 条输入执行十个条件：

- System A：一次性完整 prefill，无推测；
- B：0.0052、0.1979、0.3906、0.5833、0.776、0.85、0.92、0.9688；
- B：显式 `never_speculate`。

统一：

- Qwen2-7B-Instruct；
- greedy 主分析；
- `max_new_tokens=32`；
- `spec_chunk=12`；
- batch size 1；
- 相同 system prompt、dtype 和 attention backend。

0.92 明确标记为由旧探索 campaign 产生、在本次新 holdout 上预先冻结的 confirmatory candidate。新结果不能再次用于选择另一个“最佳阈值”。E1 的 B 条件统一采用 0.92，并直接复用 E2 的相同 records。

## 4. TEN 置信度缓存

增加正式 trigger-cache 阶段：TEN 在第二张 GPU 上对每条话语的累积 segment 前缀计算一次真实置信度，并保存：

- 未舍入置信度；
- 累积文本哈希；
- trigger template、正负类别 token；
- TEN 模型指纹和环境；
- cache SHA-256。

正式 E1/E2 session 使用只读 ReplayTrigger，避免九阈值×五进程重复确定性 TEN 前向。论文明确 trigger runtime 不在 E1/E2 的 TTFT 窗口内。

## 5. 新 runner 与五个独立进程

新 runner 复用 `sci34_supplement/common.py` 的 strict-offline、clean-tree、manifest、hash、原子写盘和安全 resume：

- 5 个独立 Python 进程 session；
- 每 session 重新加载模型；
- 每 session 100 条话语×10 条件；
- full-prefill、存活推测、作废 crop、never-speculate、断句器路径分别预热 3–5 次，warmup 不落盘；
- 条件顺序按 session/dialogue 循环平衡，避免低阈值和 System A 固定占据冷机位置；
- 保存 session/process/restart identity；发生进程重启的墙钟 session 默认以新 session ID 重跑，不拼接。

## 6. 修正 TTFT 与原始记录

不再使用旧 `first_token_ms` 作为唯一指标。每条 B record 保存：

- 真值端点接受时间；
- 推测是否存活；
- 接受时 ready token 数；
- `TTFT_eff`：候选已存活时严格为 0，否则为接受后现场首 token 可交付时间；
- consumer delivery latency，单列为实现诊断；
- candidate lead time；
- 无存活时 on-demand TTFT；
- speculation/invalidations/wasted/final tokens 和 EOS 状态；
- 原始 `perf_counter_ns`，只在分析阶段舍入。

System A 保存完整 prefill 至首 token、完整生成、token 数和输出文本。TTS 画像只形成次要的 modeled start-of-playback 情景估计，不称 mouth-to-ear 实测。

## 7. 主分析

E1 主比较：同 session、同 dialogue 的 System A vs B@0.92。

E2 主比较：B@0.92 vs `never_speculate`，并报告全部九个离散工作点。

指标包括：

- mean/median/IQR/P95 TTFT；
- 配对绝对差与相对差；
- pooled token waste ratio 与 utterance-level waste；
- survival、ready tokens、invalidations；
- candidate lead time 和未存活 on-demand TTFT。

采用两层 bootstrap：先重采样独立 session，再在 session 内重采样 dialogue，并保留该 dialogue 的全部条件。分析文件保存 bootstrap seed/repeats、estimand、所有 source SHA、config/model/TEN identity 和 excluded-record 审计。

`analysis_v1.json` 不覆盖；后续口径修订新增 versioned 文件并在验收文档中标记 superseded 关系。

## 8. 本地无模型验证

新增 fake backend 和 smoke，覆盖：

- holdout 排除旧 ID、formal 拒绝 fixture；
- config/input/trigger/model mismatch 拒绝 resume；
- duplicate/truncated JSONL；
- 十条件平衡顺序；
- warmup 不落盘；
- 完整网格与断点补跑；
- survived 时 `TTFT_eff=0`；
- pooled waste、配对 E1 差值和两层 bootstrap；
- analysis provenance；
- 旧结果文件哈希保持不变。

运行 Python 编译、现有 SCI smoke、timeline smoke、新 campaign smoke 和 `git diff --check`。本机不加载或下载模型。

## 9. GPU 单一交接入口

创建：

`experiments/sci34_supplement/e1e2_confirmatory/GPU_HANDOFF.md`

提供可复制命令，按顺序执行：

1. 签出指定 clean commit；
2. `uv sync` 和锁文件核验；
3. 设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1` 并清空 `HF_TOKEN`；
4. 验证 Qwen2-7B、TEN 和 MultiWOZ 已在本地，缺失即停止；
5. 运行无模型 smoke；
6. 构建并核验 disjoint holdout；
7. 采集 before 环境快照；
8. 运行 TEN trigger-cache；
9. 跑一个不进入正式汇总的 pilot；
10. 跑 5 个独立 formal sessions；
11. 运行 aggregate analysis 和严格验收；
12. 采集 after 快照与 GPU process；
13. 生成 artifact SHA-256、tarball 和 tarball SHA；
14. force-add 仅接受的新结果，不修改旧 JSON；
15. 回传 run ID、code/result commit、配置/模型/输入 hashes、记录数、主要区间和包哈希。

同时创建 `EXPERIMENT_PLAN.md`、`CLAIMS_MATRIX.md` 和 `ACCEPTANCE_TEMPLATE.md`，明确允许“受控同步文本段、真值端点接受下的确认性 E1/E2”，禁止“真实 ASR、在线 TTS、声学停播、生产端到端或部署最优阈值”。

## 10. 决策与论文状态

新增 D-016 并同步 `docs/paper2_context.md`：记录 0.92 的预冻结来源、新 holdout、greedy、五个 session、层级 bootstrap、修正 TTFT 和证据边界。

在 GPU 数据回传前：

- 不把新数字写入摘要或第六章；
- 不删除旧数字，只标记新 campaign 代码待执行；
- 不修改 IEEE 衍生稿。

GPU 结果验收后再统一替换权威 Markdown、重画 E1/E2 图和重建 `thesis_draft.md`。

## 11. 交付边界

本轮完成代码、配置、分析器、smoke、实验文档和 GPU handoff；不会下载模型或数据、运行正式 GPU、修改旧结果、执行人工双标、构建真实音频闭环、提交 `.zcode/`，也不会自动 push。