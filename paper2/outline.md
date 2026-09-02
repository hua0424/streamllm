# 第二篇论文大纲

**建议题目**：播放感知的级联式流式语音对话上下文管理

**English title**: Playback-Aware Context Management for Cascaded Streaming Spoken Dialogue

**论文定位**：工程与系统型硕士论文。高层原则“对话历史应反映用户实际听到的内容”已有商用服务和开源框架实践；本文的贡献边界是公开级联栈中的播放位置关联、显式 KV 状态修正、角色恢复和受控量化评测。

**权威正文源**：`abstract.md`、`chapter1_introduction.md` 至 `chapter8_conclusion.md`、`references.md`。`thesis_draft.md` 由 `uv run python -m experiments.scripts.build_thesis_draft` 自动合并；中英文 IEEE 稿为后续衍生版本，不应反向覆盖分章源稿。

---

## 第一章 绪论

### 1.1 研究背景与动机
- 级联语音对话的模块化优势与低延迟需求
- 用户打断后的生成、合成、播放状态错位
- 承认 OpenAI、Azure、LiveKit 的 prior art
- 将研究范围收窄到公开级联实现中的显式 KV 状态管理

### 1.2 研究问题与挑战
- 核心问题一：播放采样、音频块、TTS 文本片段和 token 区间的关联与反查
- 核心问题二：KV、掩码、token 账本、位置编码和角色边界的合法恢复
- 扩展问题一：单一推测阈值下的计算浪费—有效 TTFT 权衡
- 扩展问题二：被打断历史的标记与自然化

### 1.3 本文工作与贡献
- C1：可作废的推测生成调度；oracle 接受语义下的首 token 提前就绪为收益上界，墙钟收益取决于真实端点是否晚于触发；同步 harness 中到达→就绪不优于一次性 prefill；在线门控未实证
- C2：片段关联时间轴、KV 裁剪和角色恢复；核心贡献
- C3：三种历史策略；报告受混杂的探索性负结果
- 可检视研究工件；避免无条件“首个”“完全可复现”主张

### 1.4 论文组织结构

---

## 第二章 相关工作

### 2.1 商用与工程系统中的打断—上下文管理
- OpenAI Realtime API
- Azure Voice Live
- LiveKit Agents
- 比较公开接口语义、实现可见性和研究工件，不猜测闭源内部架构

### 2.2 流式语音对话与打断处理
- 提前触发与增量推理：LTS-VoiceAgent、RelayS2S、Predictive ASR
- 端到端全双工：Moshi；说明帧同步减弱但不自动消除播放缓冲差异
- 打断检测：FireRedChat；与打断后的历史修正互补

### 2.3 LLM 推理与 KV 缓存操作
- `DynamicCache.crop`、前缀缓存、vLLM/SGLang
- IntentKV、Speculative Interaction Agents
- 播放信号驱动的显式 KV 修正与角色恢复

### 2.4 差异表与本文定位
- 列拆为截断依据、播放感知、上下文层次、公开架构和实现可见性
- novelty 陈述限定检索范围和时间

---

## 第三章 问题形式化

### 3.1 系统模型与异构进度
- ASR 稳定文本段、LLM token、TTS 文本片段、音频块和播放采样
- 原始进度不直接比较；映射到 token 域后描述顺序约束

### 3.2 播放边界与片段级历史一致性
- 片段级保留边界 $\widehat H(p)$
- 片段操作语义下的历史对齐
- 字符比例—空白边界代理；明确不是物理 token 真值

### 3.3 反向查询与 KV 合法性
- 片段记录关联与反向索引 $\Phi$
- 裁剪阶段：KV、掩码、账本端点一致
- 恢复阶段：角色串长度 $q$，下一轮 user 从 $N+q$ 开始

### 3.4 评测指标
- 三个事件：last_segment_arrival、first_token_ready、endpoint_accept（接受不在到达瞬间）
- 主指标 $L_{\mathrm{arr}\to\mathrm{rdy}}=t_{\mathrm{rdy}}-t_{\mathrm{arr}}$（实际墙钟）；$\mathrm{TTFT}_{\mathrm{eff}}$ 为 oracle 接受下界（存活且已就绪为 0，否则端点后按需生成首 token 的时延）
- TTFT 触发口径与 mouth-to-ear 画像建模
- A1 crop-only/联合恢复与 P1 prepared-state 软件控制路径的区分
- stop 确认、stop 后同步、反查、stop→crop、stop→role 的嵌套区间定义；不相加
- 片段口径与字符比例—空白边界近似口径
- 词面检测、LLM 裁判和人工样本均为代理，不称上下界
- 推测浪费率 wasted/(wasted+final)（与确认性 campaign 的正式 estimand 相同）和离散工作点

---

## 第四章 方法设计

### 4.1 总体设计
- 输入增量流水、输出断句/TTS/播放和打断状态修正

### 4.2 可作废的推测生成调度
- 单一推测阈值与候选预算
- 真值终点接受；在线 TTS 门控为未验证扩展
- 推测作废状态机

### 4.3 播放感知的 KV 缓存管理
- 片段关联时间轴和断句—token 对齐
- `DynamicCache.crop`、掩码和 token 账本同步裁剪
- 角色边界恢复和 assistant 侧 KV 累积
- P1 prepared-state 屏障：播放前 `ensure_full()` + 设备同步，准备耗时排除
- B-gen、B-noKV、B-syn 的语义边界

### 4.4 被打断历史处理策略
- 朴素、标记、重写
- 重写“不新增信息”是提示约束，不是形式保证
- 延迟可重叠是潜力，不声称完全隐藏

---

## 第五章 系统实现

### 5.1 模块架构
- 模块、接口和验证对象表

### 5.2 片段时间轴与断句对齐
- 边界计数语义
- 非空白字符前缀和
- 部署并发目标与实验模拟的区别

### 5.3 KV 状态操作
- assistant 侧 KV 累积
- KV/掩码/账本统一不变式
- chat template 角色串推导与适配范围

### 5.4 编排与实验记录
- 两条用户输入路径、输出闭环和条件开关
- Mock TTS 是真机画像参数化近似
- 原始结果与离线重分析分离
- 确认性 E1/E2 campaign 工件：run `e1e2c_b8c758b_20260901T173306Z`、代码 `b8c758b`/结果 `62508dc`、5 独立进程×100 条零交集 holdout×10 条件（5000 条 records）、TEN replay cache 222 条目且不进计时窗口、greedy/bfloat16/sdpa、manifest/checksums（72 文件）/环境快照

### 5.5 部署、验证与可复现性
- 双 3090 模型布局
- 四个独立 campaign：旧 E1/E2/A2（探索性旧 campaign）、固定轨迹 E3/联合 A1、prepared-state P1 v2、确认性 E1/E2；绝对时间不池化，A1/P1 不相减
- P1 主机：双路 Xeon Gold 6330、112 逻辑 CPU、约 756 GiB、Ubuntu 22.04.5、driver 580.105.08、双 RTX 3090
- 组件检查与数据级检查的证据边界
- 固定轨迹 E3/联合 A1/P1 v2/确认性 E1/E2 已归档 manifest、运行身份、checksums 与原始记录；旧 E1/E2 口径 artifact 已修正并降级为审计引用，旧 campaign 仍缺部分元数据

---

## 第六章 实验与结果分析

### 6.1 研究问题与实验设置
- RQ1 一致性；RQ2 推测浪费—双口径延迟权衡；RQ3 组合系统双口径响应延迟；RQ4 KV 复用与 prepared-state 软件打断控制路径；RQ5 历史策略
- 旧 E2 清除 fixture；固定轨迹 E3 独立使用 100 条纯 MultiWOZ，不与旧 E3 合并
- 确认性 E1/E2（C-E1/C-E2）为第四个独立 campaign：5 独立进程×100 条新 holdout（与旧 E1/E2/E3 零交集）、greedy、TEN 置信度离线回放；旧 E1/E2 明确标为探索性旧 campaign
- 墙钟实测、oracle 下界、微基准、画像建模和构造性结果分类
- 明确同步推测、40-token E3 上限和无真实 ASR 闭环

### 6.2 RQ1：固定轨迹历史一致性（E3）
- 100 条对话、400 配对场景、800 条条件记录；共享首轮轨迹/时间轴/注入位置，greedy probes，40-token 上限
- dialogue-cluster bootstrap 为主要不确定性结果；exact McNemar 仅作描述性补充
- 片段目标 n=297：规则 67.0% vs 63.6%，裁判 42.8% vs 40.7%；CI 均跨零
- 修正代理资格 n=380：规则 75.3% vs 73.7%，裁判 43.9% vs 41.3%；CI 均跨零
- 四个小点估计均与预设方向相反且不显著；不作优效/等效/非劣/伤害主张
- playback 局部完整未播放文本 400/400 为空仅是构造检查；0.5/boundary 片段目标重复
- 无人类双标；LLM judge v3 为单模型单提示词代理

### 6.3 RQ2：推测浪费—双口径延迟权衡（C-E2/E2/A3）
- 确认性 campaign 九条件（八阈值＋不推测），每点配对 n=500；表 6-3 报告九点
- 实际墙钟 arrival→ready：B@0.92 与不推测差 −0.03 ms（CI [−0.55, +0.51]），九条件均平坦于约 62 ms
- B@0.92 pooled waste 2.85%（CI [0.020, 0.037]）、survival 67.0%（CI [0.628, 0.712]）、就绪 token 中位 12、候选首 token 领先端点中位 291 ms
- oracle 上界：never−B +20.80 ms（CI [19.50, 22.10]）；未存活 on-demand 31.09 ms ≈ never oracle 31.06 ms
- 旧 E2 九点为探索性旧 campaign 审计引用（0.92 预冻结来源）；不称连续 Pareto 前沿

### 6.4 RQ3：组合系统双口径响应延迟（C-E1/E1）
- 确认性配对 n=500；表 6-4 报告 C-E1 配对结果
- 实际墙钟 arrival→ready：System A 27.70 ms vs B@0.92 62.38 ms；A−B −34.69 ms（CI [−35.30, −34.11]），B 更慢
- 机制：A＝单次批量 prefill＋首 token；B＝最后段增量 prefill＋assistant role 注入＋首 token（≈两次串行前向），短文本下单次前向固定开销主导
- oracle 口径：A−B +17.44 ms（CI [16.12, 18.75]），为推测收益上界，适用条件是真实端点晚于触发
- 旧 E1 的 0.581/27.407 ms 与 mouth-to-ear 画像建模降级为探索性旧 campaign 审计引用

### 6.5 RQ4：KV 状态复用与软件打断控制路径

#### 6.5.1 A1：KV 复用联合微基准
- 表 6-5：256–8192 token；warmup=5，repeats=50；设备同步的同一区间联合计时
- 联合中位数 31.054–48.315 ms，IQR 0.635–3.099 ms
- 重新预填充中位数 / 联合路径中位数 = 2.254–40.620
- 不是播放器链路

#### 6.5.2 P1：Prepared-state 软件打断控制路径
- run `sci34_dc52978_20260901_async_prepared_v2`；代码 `dc52978`；结果 `ee1dcc7`
- 3 长度 × 3 位置 × 20 = 180；120 片段内、60 边界；180/180 精确目标、零软件采样泄漏
- 播放前 setup 原始 40.499–1722.228 ms、单元中位数 41.208–1717.110 ms，明确排除
- 紧凑表 6-6：stop ack 0.055–0.062 / 最大单元 P95≈0.077；post-stop sync 0.167–0.176 / ≈0.352；lookup 0.47–0.50 / ≈0.94；stop→crop 2.44–2.53 / ≈3.492；stop→role 78.6–80.8 / ≈86.1 ms
- 两个 stop 累计区间嵌套；组件与累计端点不相加；不与 A1 池化或相减；不声称上下文无关
- 仅 headless 墙钟节拍软件播放器/模型状态；不代表声卡、声学/用户所听、在线 TTS、真实并发或生产端到端

### 6.6 RQ5：历史策略（A2）
- 描述性评分与重写耗时
- 仅 33/100 三策略 heard_text 相同，明确独立生成混杂
- 作为探索性负结果，不作因果比较

### 6.7 按 RQ 汇总结论与适用边界

---

## 第七章 讨论

### 7.1 与已有系统的关系
- 公开研究实现与既有工程实践互补
- 不猜测商业系统内部实现

### 7.2 效度威胁
- 构念效度：边界代理、词面规则、单一 judge v3、无人类双标
- 内部效度：旧 E1/E2 口径 artifact（user_end 记录在同步推测完成后，oracle 误作墙钟）已由确认性 campaign 修正；两串行前向机制解释九点平坦的 arrival→ready；同步时序、40-token 上限、0.5/boundary 重复、A2 条件轨迹不一致
- 外部效度：单模型、双 3090、英文任务型对话、四个 campaign 不池化
- 结论效度：cluster bootstrap 主、McNemar 描述；不显著不等于等效/非劣，负点估计不等于伤害；oracle 上界收益不得表述为墙钟改善；P1 v1 协议失败只作审计，v2 当前有效但限于软件控制路径

### 7.3 可推广性与适用条件
- TTS 需提供片段—音频归属
- 推理引擎需支持缓存裁剪和角色恢复
- 阈值须按新领域和在线时序重新标定

---

## 第八章 总结与展望

### 8.1 全文总结
- 按 C1/C2/C3 分别总结“证据与边界”
- C2 的状态机制、A1 模型侧计算结果和 P1 prepared-state 软件控制路径结果成立；固定轨迹 E3 未检出预设方向语义收益；C1 为双口径结论（oracle 上界收益＋同步 harness 中到达→就绪不优于一次性 prefill）；C3 为受混杂的探索性负结果

### 8.2 后续工作
- P1 v2 已完成；后续接入真实声卡/声学停播、在线异步 TTS、真实 ASR/LLM/TTS/播放器并发与生产控制闭环
- 固定轨迹 A2 因果对照（固定 E3 已完成，不再列未来工作）
- 词级/token 级播放边界
- 随机、盲法、双标注员的人评
- 跨语种、跨模型、跨引擎复验和完整运行 manifest
- 完成实测播放位置与实时速度假设的 E4

---

## 定稿前引用与工件待办

1. 为 Predictive ASR、vLLM、SGLang 和 stream2sentence 补原始文献或官方技术文档。
2. 为“一期工作已验证的结果”提供正式论文/章节引用，而不是只引用实验设计文件。
3. 不使用未核实的 CosyVoice2 45 ms 外部数字；如讨论官方 150 ms 宣称或 L20/TensorRT-LLM 约 190–220 ms 基准，须完整注明来源和条件，且不直接代入本文系统对比。
4. 按学校要求把统一文献表转换为 GB/T 7714 等正式格式。
5. 用 `uv run python -m experiments.scripts.build_thesis_draft` 从分章 Markdown 重新生成 `thesis_draft.md`，再同步中文和英文 IEEE 衍生稿。
