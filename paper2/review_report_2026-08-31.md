# 第二篇论文审查与修订报告（2026-08-31）

## 1. 审查范围

本次审查覆盖：

- `paper2/abstract.md` 与八章分章 Markdown；
- `paper2/outline.md`、`paper2/thesis_draft.md`、`paper2/references.md`；
- 二期 E1、E2、E3、A1、A2、A3 的实验脚本与结果 JSON；
- 图 3-1、图 4-4 和图 6-1 至图 6-4；
- `docs/decisions.md`、`docs/paper2_context.md` 等方法与交接材料。

本机未重跑正式 GPU 实验。所有新增统计均为对已有 JSON 的离线清洗和复算，原始 GPU 结果未被覆盖。

## 2. 大纲结论

八章结构总体合理，适合工程/系统型硕士论文：绪论 → 相关工作 → 问题形式化 → 方法 → 实现 → 实验 → 讨论 → 总结。主要调整如下：

1. 第四章取消不规范的“4.0”，改为 4.1–4.5。
2. 第五章按模块架构、时间轴、KV 操作、编排记录、部署与复现拆分，避免把工程贡献压缩成代码清单。
3. 第六章改为研究问题驱动，而非 E3→E2→E1 的编号倒序：RQ1 一致性、RQ2 推测权衡、RQ3 文本驱动响应、RQ4 KV 微基准、RQ5 历史策略。
4. A3 明确与 E2 共用阈值扫描数据；未实施的 E4 转入后续工作。
5. 讨论章按构念、内部、外部和结论效度组织。

## 3. 已修复的关键问题

### 3.1 数据完整性

- E3 原结果含 3 条内置 fixture，正文“103 条 MultiWOZ”不成立。正式分析改为 100 条 MultiWOZ、每条件 400 个场景。
- E2 原结果含 12 条 fixture；不推测点原聚合 n=106。正式分析改为九点各 n=100，不推测点 TTFT 为 48.3 ms。
- 原始 GPU JSON 保留不动；`experiments/scripts/reanalyze_paper2_results.py` 生成独立 `paper2_reanalysis.json`。

### 3.2 统计与结论

- E3 从独立样本 Fisher 改为按 `(id, fraction)` 匹配的 exact McNemar 描述性比较，并增加 dialogue-cluster bootstrap 置信区间。
- 因 playback/generation 未共享同一生成轨迹，正文明确统计结果不能隔离边界策略因果效应。
- B-ours 片段口径的零继续表述为构造性保证，而非实验发现。
- 规则检测器和 LLM 裁判不再称为上界和下界。
- “未检出差异”不再解释为等价、无代价或非劣。

### 3.3 形式化与实现对齐

- 删除跨量纲的 $p(t)\le s(t)\le g(t)$，先将播放位置解析到片段级 token 端点后再比较。
- 取消没有实现依据的 token 域线性播放边界；按代码真实语义定义“播放比例切文本字符并向前吸附到空白边界”的代理。
- 反向查询改为 `playback samples → hit fragment → token span`；当前实现不按采样位置反查具体音频块。
- KV 和掩码使用绝对端点 $N$；assistant token 账本使用本轮相对长度 $N-a_0$。角色串不写入 assistant 账本。

### 3.4 实验口径

- A1 的 0.308–0.339 ms 只称 `DynamicCache.crop` 孤立微基准，不再称完整打断响应。
- 46.88 ms 明确为 crop 中位数与角色恢复中位数之和，不是联合路径中位数；39.7 为重新预填充中位数与该组件和的比值。
- E1 改称文本段驱动响应实验；mouth-to-ear 明确为 TTS 画像建模。
- E2 改称九个离散工作点；最低阈值点被次低阈值点支配，不再称严格单调连续前沿。
- A2 三策略只在 33/100 对话中共享相同 `heard_text`，降格为受混杂的描述性观察。

### 3.5 引用与图表

- 修正 OpenAI、Azure、LiveKit、FireRedChat、TEN 和 Hugging Face 文献条目。
- 新增 Predictive ASR、vLLM、SGLang 与 stream2sentence 引用。
- 删除无法核实的“CosyVoice2 A100+TensorRT 45 ms”外部数字。
- 更新图 3-1、图 4-4、图 6-1 和图 6-4 的语义；图 6-1 只表达片段口径，图 6-4 的 39.7 标注落在组件中位数之和曲线上。

## 4. 当前可保留的核心结论

1. 在本文片段口径下，playback 策略按构造不把完整未播放片段写入历史。
2. 在同步全生成、40-token 上限的纯 MultiWOZ 模拟中，B-gen 的片段口径词面复现率为 50.3%，单一异构 LLM 裁判率为 2.3%。
3. 九个阈值点显示总体的推测浪费—有效 TTFT 权衡；$\theta=0.92$ 对应约 4.5% 浪费和 12.1 ms。
4. 在受测模型和硬件上，KV crop 与角色恢复组件耗时低于重新预填充；8k 处相应比值为 39.7。
5. A2 没有提供重写策略的正向或因果证据，诚实保留为受混杂的描述性结果。

## 5. 仍需 GPU 实验机补强的项目

### 必要程度最高

1. **固定生成轨迹重跑 E3。** 同一 assistant token 流、断句和时间轴分别应用 playback/generation 边界，隔离策略效应。
2. **修复 A2 并重跑。** 同一 `heard_text` 派生 naive/mark/rewrite；固定下一轮解码或使用成对随机种子。
3. **真实异步音频闭环。** 联合测量打断检测、播放器停播、时间轴查询、KV crop 和角色恢复；让 ASR 段与推测解码真实竞争。

### 强烈建议

4. A1 保存每次重复的联合 `crop + role recovery` 总时延，报告联合路径 median/IQR，而不是组件中位数之和。
5. E1/E2 保存输入文本、分段和时长，补输入长度分析与每点置信区间。
6. 主模型生成固定 seed 或使用多 seed 重复；结果 JSON 归档 commit、模型 revision、环境版本、数据 hash 和完整配置。
7. 重新进行随机、盲法、至少双标注员的人工评测。

## 6. 本机已完成与未完成验证

- 离线重分析脚本：通过，输出可确定性复算。
- 绘图脚本：通过，中英文图 6-1 至图 6-4 均已重新生成。
- 图表视觉验收：图 6-1 至图 6-4 全部通过。
- `PlaybackTimeline` smoke：通过。
- `sentence_chunker` smoke：未完成。脚本会加载 Qwen2.5-0.5B；本机缓存缺少模型，约 1 GB 权重下载中断。该失败是环境/下载问题，不是断句断言失败，本次未继续重试。

## 7. 版本说明

- 权威正文源：`abstract.md`、`chapter1_introduction.md` 至 `chapter8_conclusion.md`、`references.md`。
- `thesis_draft.md`：由分章源自动合并。
- `ieee_paper/main.tex` 和 `ieee_paper_cn/main.tex`：仍是旧衍生版本，尚未同步本次数据和论证修正，不应作为当前定稿依据。
