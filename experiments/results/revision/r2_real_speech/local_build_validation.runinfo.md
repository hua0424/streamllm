# R2 真实语音构建链路 —— 本机验证记录

**日期**：2026-08-18
**机器**：本机 Windows + RTX 3060（开发机；正式 75 条构建按 handoff §4-E2-0 由 GPU 主机执行）
**验证范围**：构建/增强/QA 三脚本在伪造迷你语料与真实 LibriSpeech test-clean 上的正确性；
AISHELL-1 解析器经伪造语料（内层 tar.gz 与解压目录两种官方布局）覆盖，真实 15 GB 语料的
终验在主机 E2-0 第 4 步 QA 中完成。

## 一、验证结果汇总

| 验证项 | 结果 |
|---|---|
| 伪造迷你语料全链路冒烟 `test_r2_build_smoke.py` | **16/16 通过**（构建/schema/字节级确定性/SNR 独立复算/变速时长比/QA 正负例） |
| 真实 test-clean 小配额构建（long:2/very_long:2/extra_long:1） | 5 条成功，时长 21.27/19.90/39.26/31.51/61.56 s |
| 真实数据确定性 | 同种子二次构建，json+wav 逐文件 md5 完全一致 |
| 满配额构建（30/30/15，共 75 条，44.6 min 音频） | 成功，配额足额（test-clean 87 章节） |
| 静态 QA（75 条） | 通过（16k 单声道 / 时长 ±50ms / 文本非空 / RMS） |
| turbo 转写 sanity（5 条，cuda:0，System A 同参 beam=5/temp=0） | **mean WER = 2.75%**（max 5.56%），≤10% 验收线通过 |
| 旧回归套件 `test_revision_regressions.py` | 10/10 不受影响 |

## 二、验证中发现并修复的问题

1. AISHELL 内层 tar 说话人的流式读取闭包晚绑定（所有单元引用循环末尾的句表）→
   改为参数早绑定（`make_tar_stream(..., bound_items=...)`）；伪造语料双布局冒烟捕获。
2. `select_subset` 未做 long/very_long 各半上限（40 条池会取 20/10 而非 15/15）→
   补 `half = ceil(subset/2)` 上限；离线自检捕获。
3. QA 英文 WER 未折叠大小写：LibriSpeech 参考全大写、Whisper 输出混合大小写，
   归一化后逐词失配（首跑 WER≈1.0，但 hypothesis 与 reference 内容逐词一致）→
   QA 内补 `.lower()`（LibriSpeech 官方评估惯例），**未改动** `run_exp_quality.normalize_text`
   （exp3 中文口径不受大小写影响，保持原样）。

## 三、本机运行的完整命令（供复现）

```bash
# 语料（本机仅下载 test-clean 346MB，AISHELL/MUSAN 留给 GPU 主机）
curl -L -o experiments/datasets/raw_data/librispeech/test-clean.tar.gz \
  https://www.openslr.org/resources/12/test-clean.tar.gz

# 1) 冒烟（16/16）
uv run python -m experiments.scripts.test_r2_build_smoke

# 2) 真实小配额构建 + 静态 QA + turbo 转写 QA（输出在临时目录，不入库）
uv run python -m experiments.scripts.build_real_speech_set --source librispeech \
  --quota long:2,very_long:2,extra_long:1 \
  --json-dir <tmp>/json --audio-dir <tmp>/audio --manifest-dir <tmp>/manifest
uv run python -m experiments.scripts.qa_real_speech --datasets librispeech \
  --json-dir <tmp>/json --audio-dir <tmp>/audio --report-dir <tmp>/qa \
  --expected-quota long=2,very_long=2,extra_long=1 \
  --transcribe --asr-model-size turbo --device cuda:0

# 3) 满配额可行性验证（30/30/15）
uv run python -m experiments.scripts.build_real_speech_set --source librispeech \
  --json-dir <tmp>/full/json --audio-dir <tmp>/full/audio --manifest-dir <tmp>/full/manifest
uv run python -m experiments.scripts.qa_real_speech --datasets librispeech \
  --json-dir <tmp>/full/json --audio-dir <tmp>/full/audio --report-dir <tmp>/full/qa
```

## 四、环境备注

- `.env` 的 `HF_HOME` 指向 GPU 主机路径，本机转写 QA 需覆盖：
  `HF_HOME=/c/Users/hua/.cache/hf_r2qa`（turbo 已缓存于此，约 1.5 GB）；
  HF_TOKEN 失效需置空：`HF_TOKEN= uv run ...`。
- 本机构建/转写产物均在临时目录，未写入 `experiments/datasets/processed/`；
  正式 75 条×2 语料 + 增强变体由 GPU 主机构建（handoff §4-E2-0）。
