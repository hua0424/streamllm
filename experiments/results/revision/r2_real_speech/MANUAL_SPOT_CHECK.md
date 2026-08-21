# R2 真人语音 QA：人工抽检记录与补充说明（W7）

对应 PRE-PAPER-AUDIT P1-2 / 复审 §5.2。**结论标为 manual spot check（人工抽检），不称 human evaluation。**

## 一、抽检任务（需需求方本人完成）

候选 5 条（clean 集，中英覆盖，含 extra_long；音频在 GPU 主机 `experiments/datasets/processed/audio/` 对应数据集目录下）：

| # | sample_id | 语种 | 分组 | 时长(s) |
|---|---|---|---|---|
| 1 | librispeech_121-121726 | en | long | 15.1 |
| 2 | librispeech_6829-68771 | en | extra_long | 60.8 |
| 3 | aishell1_S0005_01 | zh | long | 15.0 |
| 4 | aishell1_S0041_01 | zh | very_long | 30.2 |
| 5 | aishell1_S0064_01 | zh | extra_long | 60.1 |

### 记录模板（逐条填写）

```text
sample_id:
试听者:                日期:
可懂度（正常/可懂但瑕疵/不可懂）:
截断（无/开头/结尾）:          错序（无/有，说明）:
爆音/削波（无/有）:            异常静音段（无/有，大致位置）:
音量异常（无/有）:             拼接缝可感知（无/轻微/明显）:
结论（通过/不通过+原因）:
```

## 二、数据构造说明（论文方法节用）

- 样本为**同章节/同说话人朗读句拼接**的长语音：LibriSpeech（en）与 AISHELL-1（zh）各自按
  说话人/章节内连续 utterance 顺序拼接， utterance 间插入 U(0.2, 1.0)s 人工静音
  （见 `librispeech_build_manifest.json` / `aishell1_build_manifest.json` 的 gap_policy、seed=42）；
- 非自然对话；论文统一称"拼接的真人朗读语音"；
- 增强条件（snr10/15/20、babble、speed09/11）各自由 clean 75 条中独立抽 30 条
  （条件间样本仅重叠 14–18 条），跨条件比较不作严格因果排序。

## 三、参考文本恢复与校验（`qa_transcribe.corrected.csv`）

- `reference_full` 为完整拼接参考（各 utterance 参考依次连接），是评分用参考；
- `reference` 列为 QA 展示用截断版（90/150 行更短）——2026-08-21 本机核实：误用 reference
  会把 aishell1_clean 非流式 CER 从 0.1077 抬高到 0.2009；`score_wer_offline.py --ref-csv`
  固定使用 `reference_full`；
- 校验过程：`qa_real_speech.py` 对 clean 音频做免模型重转写比对，`reference_full` 与
  构建清单的 source_utterances 逐句对齐核验；英文 WER 口径重算与原值一致（交叉验证）。

## 四、失败边界与统计口径

- **babble 空输出率单列**：LibriSpeech babble 12/30、AISHELL-1 babble 5/30 条流式零提交
  （VAD 对多人 babble 过度触发 → 段积压/空输出；median 2020ms vs mean 3425ms 长尾）；
- `error=0` 只表示程序未崩溃，**不代表有效转写率 100%**；空输出样本记 WER/CER=1.0 并计 n_empty；
- 空输出样本的 asr_time/llm_prefill_time 为哨兵值，不纳入延迟统计。
