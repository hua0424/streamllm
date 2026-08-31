# 第三章 问题形式化

## 3.1 系统模型与符号

### 3.1.1 级联流水线

考虑级联式流式语音对话系统

$$
\mathcal{S}=\langle \mathrm{ASR},\mathrm{LLM},\mathrm{CHK},\mathrm{TTS},\mathrm{PLY}\rangle,
$$

其中各模块按流水线衔接。

- **流式 ASR** 将用户语音转写为稳定文本段序列 $U=\langle u_1,u_2,\ldots\rangle$。本文下游仅接收不再修正的 final segment，而不直接消费可撤销的 partial transcript。
- **LLM** 维护对话上下文的 KV 缓存 $\mathcal{K}$。用户文本段到达后增量预填充，assistant 回复以零起始的 token 序列 $Y=\langle y_0,\ldots,y_{G-1}\rangle$ 逐步生成。
- **断句器 CHK** 将 assistant token 流切分为 TTS 文本片段 $F=\langle f_1,\ldots,f_m\rangle$。每个片段关联一个左闭右开的 token 区间：

$$
f_j\mapsto[\operatorname{ts}(f_j),\operatorname{te}(f_j)),\qquad
\operatorname{ts}(f_1)=0,\quad
\operatorname{ts}(f_{j+1})=\operatorname{te}(f_j).
$$

  本文以 TTS 文本片段作为历史裁剪的原子单位。
- **流式 TTS** 接收完整文本片段并输出一个或多个音频块。片段 $f_j$ 在累计音频轴上占据采样区间 $[\operatorname{ss}(f_j),\operatorname{se}(f_j))$，采样率记为 $r$。
- **播放器 PLY** 顺序消费音频块并维护已播放采样计数 $p\in\mathbb{N}$。$p$ 采用计数语义，即采样区间 $[0,p)$ 已被播放。

本文固定使用“ASR 稳定文本段”表示 $u_i$，使用“TTS 文本片段”表示 $f_j$，以避免两类片段混淆。

### 3.1.2 异构进度及其关联

在时刻 $t$，系统同时维护三种原始进度：生成 token 端点 $G(t)$、已登记到 TTS/播放时间轴的最大文本片段 token 端点 $S(t)$，以及播放采样计数 $p(t)$。三者分别处于 token 域和采样域，不能直接写成 $p(t)\leq S(t)\leq G(t)$。

系统只把播放位置映射到**命中的 TTS 文本片段**，并以该片段的 token 末端作为保留边界 $\widehat H(p)$。在“播放器只消费已登记片段、片段按生成顺序入队”的假设下，比较对象均转换为 token 端点后有

$$
\widehat H(p(t))\leq S(t)\leq G(t). \tag{3-1}
$$

式（3-1）描述的是本文系统中的顺序约束，而不是对所有 TTS 实现和流控状态无条件成立的物理定律。只要打断时 $\widehat H(p)<G$，按生成边界保留历史就可能纳入尚未播放的内容。

端到端帧同步模型可以减少独立 TTS 引入的错位，但网络和播放缓冲仍可能使模型产出与实际听到位置不同；因此本文不将“端到端”简单等同于三种进度完全一致。

![图 3-1](figures/fig3_1.png)

**图 3-1　三种异构进度与片段级保留边界。** 播放采样位置 $p$ 落在片段 $f_3$ 内，片段级保留边界吸附到 $\widehat H(p)=\operatorname{te}(f_3)$。片段中尚未播放的文本尾部由定义 3.3 的字符比例—空白边界代理估计。

## 3.2 播放位置与片段级历史对齐

**定义 3.1（片段级保留边界）**　若 $p$ 落在片段 $f_k$ 的采样区间内，即

$$
\operatorname{ss}(f_k)<p\leq\operatorname{se}(f_k),
$$

则定义

$$
\widehat H(p)=\operatorname{te}(f_k). \tag{3-2}
$$

若 $p=\operatorname{se}(f_k)$，打断发生在干净片段边界；若 $p<\operatorname{se}(f_k)$，则当前片段只播放了一部分。式（3-2）选择把命中片段整体保留在历史中，从而避免在缺少片段内文本—音频对齐时裁剪到任意 token。

当 $p=0$ 时，表示推测内容尚未播出，定义 $\widehat H(0)=0$。若播放位置越过全部已登记音频，则边界钳制到最后一个具有音频记录的片段末端；仅当本轮全部 assistant 内容均已登记时，该端点才等于本轮 assistant 结束位置。

**定义 3.2（片段级历史对齐）**　设本轮 assistant 内容相对起点的保留范围为 $[0,\widehat H(p))$。若打断后对话历史及其 KV 表示只保留该范围，则称其满足片段级历史对齐：

$$
\mathcal{H}_{\mathrm{frag}}=
\langle y_0,\ldots,y_{\widehat H(p)-1}\rangle. \tag{3-3}
$$

作为对照，按生成位置保留的历史为 $\mathcal{H}_{\mathrm{gen}}=\langle y_0,\ldots,y_{G-1}\rangle$。当 $G>\widehat H(p)$ 时，区间 $[\widehat H(p),G)$ 对应完整未播放片段或其后续内容。

需要强调，式（3-3）保证的是**片段操作语义下的对齐**，而非与连续物理播放内容逐 token 严格相等。本文实现没有 TTS 词级 duration 或强制对齐，因而不能从播放采样数得到真实 token 播放位置。

**定义 3.3（字符比例—空白边界代理）**　当 $p$ 命中片段 $f_k$ 且只播放了该片段的一部分时，定义片段内播放比例

$$
\alpha(p)=\frac{p-\operatorname{ss}(f_k)}{
\operatorname{se}(f_k)-\operatorname{ss}(f_k)}.
$$

设片段文本长度为 $L_k$ 个字符，先计算原始字符切点

$$
c_{\mathrm{raw}}=\operatorname{round}\bigl(\alpha(p)L_k\bigr),
$$

再将其向前移动到最近的空白边界，得到 $c_{\mathrm{ws}}$。文本后缀

$$
W_{\mathrm{tail}}(p)=f_k[c_{\mathrm{ws}}:L_k] \tag{3-4}
$$

作为命中片段尚未播放部分的代理。该口径是**字符比例—空白边界近似**，既不是音素/词级对齐真值，也不是 token 域线性插值。它只用于第六章分析片段级向上吸附可能带来的文本尾部风险。

本文包含两类状态修正事件：播放期用户打断按 $\widehat H(p)$ 保留历史；未播出的推测结果作废时，缓存回滚到本轮 assistant 起点，对应 $p=0$ 和 $\widehat H=0$。

## 3.3 反向查询与 KV 状态合法性

**定义 3.4（播放位置反向查询）**　给定播放采样数 $p$，关联时间轴执行

$$
\Phi:p\longrightarrow
f_k\ \text{s.t.}\ \operatorname{ss}(f_k)<p\leq\operatorname{se}(f_k)
\longrightarrow[\operatorname{ts}(f_k),\operatorname{te}(f_k))
\longrightarrow\widehat H(p). \tag{3-5}
$$

时间轴记录片段关联的 `chunk_ids`，但当前反查按片段聚合采样区间定位，并不从采样位置解析到某个具体音频块。$\Phi$ 也不是四层之间的可逆双射，而是由片段记录维护的关联与反向索引。

设当前 assistant 内容在整段 KV 序列中的绝对起点为 $a_0$，裁剪位置为

$$
N=a_0+\widehat H(p).
$$

状态恢复分为两个阶段。

**定义 3.5（裁剪阶段合法性）**　裁剪至 $N$ 后，应满足：

1. KV 序列长度与注意力掩码长度均为绝对端点 $N$；本轮 assistant token 账本长度为 $N-a_0=\widehat H(p)$；
2. 被移除的 assistant token 不再出现在 KV、掩码所覆盖的序列和 assistant token 账本中；
3. 下一次预填充使用裁剪后的 past length 构造连续位置编码。

**定义 3.6（角色恢复阶段合法性）**　设 assistant 关闭和下一轮 user 开启所需的模板串包含 $q$ 个 token。该串预填充后，KV 与注意力掩码的绝对端点均为 $N+q$；角色串不计入本轮 assistant token 账本，账本仍保存 $\widehat H(p)$ 个 assistant token。下一轮 user 文本从位置 $N+q$ 开始。

## 3.4 评测指标

### 3.4.1 延迟指标

**推测触发到首 token 延迟 $\mathrm{TTFT}_{\mathrm{spec}}$**：推测阈值被触发到推测生成产生首 token 的墙钟时间。该指标描述触发—生成链路，不表示内容已经获准播出。

**有效首 token 延迟 $\mathrm{TTFT}_{\mathrm{eff}}$**：用户话轮真值终点到首个可继续使用的 assistant token 的时间。当提前生成结果在真值终点被接受时，已有 token 可使该值接近零。本文实验中的接受由真值终点触发，因而该指标属于受控模拟口径。

**mouth-to-ear 延迟**：用户话轮结束到首块音频可播放的时间。第六章将 LLM 计算墙钟时间与 TTS 真机画像组合建模；该数值不是实际音频闭环的端到端实测。

**KV 裁剪操作延迟 $L_{\mathrm{crop}}$**：孤立执行缓存裁剪的墙钟时间。A1 另将 crop 中位数与角色恢复中位数相加，作为两个组件耗时之和与重新预填充比较；该和不是联合路径逐次计时所得的中位数。实际打断响应还应包含播放器停播、队列清理、$\Phi$ 查询、服务通信和线程调度，本文没有在真实异步播放链路上联合测得这一总延迟。

### 3.4.2 一致性指标

设按某种边界进入历史但未向用户完整播放的文本为 $W$，其后两轮回复集合为 $R$。本文记录后续回复是否复现 $W$ 中的信息，并采用两种边界口径。

- **片段口径（loose）**：$W_{\mathrm{frag}}$ 只包含进入历史的完整未播放片段。按定义 3.2，本文方法下 $W_{\mathrm{frag}}=\varnothing$，因此相应引用率为构造性零；实验主要估计按生成位置保留历史的对照失败率。
- **字符比例—空白边界近似口径（proxy）**：$W_{\mathrm{proxy}}$ 还包含式（3-4）的命中片段文本尾部。该口径比片段口径纳入更多候选文本，但不是真实播放边界。

引用判定使用两种代理：词面检测器用于高敏感度筛查，异构 LLM 裁判用于较保守的特定信息判断。二者均可能产生误报或漏报，不能预设为数学上的上界和下界；人工分层样本仅用于描述其差异，不能无条件外推到总体。

同时，本文区分“未播放内容进入历史”这一结构合规问题和“后续回复复现特定信息”这一语义后果。前者可由缓存边界和 $W$ 的长度直接检查，后者只能由规则、模型或人工代理估计。

### 3.4.3 效率指标

推测浪费率定义为

$$
\rho=\frac{\sum\text{作废的推测 token 数}}
{\sum\text{作废的推测 token 数}+\sum\text{最终生成 token 数}}. \tag{3-6}
$$

不同推测阈值 $\theta$ 对应离散工作点

$$
\bigl(\rho(\theta),\mathrm{TTFT}_{\mathrm{eff}}(\theta)\bigr).
$$

阈值降低通常提高提前生成覆盖率，也可能增加作废计算。有限个测试点只能支持总体权衡趋势，不自动构成连续或严格单调的 Pareto 前沿。

KV 复用收益通过“重新预填充耗时与 crop、角色恢复两个组件中位数之和的比值”描述。为避免口径混淆，本文分别报告 crop-only、角色恢复、二者中位数之和及重新预填充耗时。

### 3.4.4 指标与实验对应关系

| 研究问题 | 主要指标 | 实验 |
|---|---|---|
| 播放感知历史是否减少未播放信息复现 | 片段口径、字符比例—空白边界近似口径 | E3 |
| 推测阈值如何影响计算与响应 | $\rho$、$\mathrm{TTFT}_{\mathrm{eff}}$ | E2（同时作为 A3） |
| 组合系统在受控文本输入下的响应差异 | TTFT、建模 mouth-to-ear | E1 |
| KV 状态复用是否降低恢复计算 | crop、角色恢复组件耗时及重新预填充耗时 | A1 |
| 三种历史处理策略的描述性表现 | 连贯性评分、重写耗时 | A2 |

## 3.5 本章小结

本章首先避免在不同量纲之间直接比较进度，使用片段级保留边界 $\widehat H(p)$ 将播放采样位置解析为 token 端点；随后将本文保证限定为片段操作语义下的历史对齐，并按代码真实实现定义字符比例—空白边界代理。KV 修正被拆为裁剪和角色恢复两个阶段，assistant token 账本保持本轮相对长度语义。最后，本章区分墙钟实测、孤立微基准、组件中位数之和、TTS 画像建模和构造性结果，为第六章限制结论强度提供统一口径。
