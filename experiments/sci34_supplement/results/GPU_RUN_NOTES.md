# GPU 主机运行记录（sci34 补实验，2026-09-01）

## Commit 三元组与 campaign

- **campaign**：`sci34_f11ccba_20260901`
- **code_commit（E3 主实验）**：`f11ccba`（origin/paper2，工作树 clean）
- **judge 修复 commit（裁判及之后所有 run）**：`ca627c7`（本地提交，仅改 `experiments/sci34_supplement/e3_judge.py`；
  `f11ccba..ca627c7` 对 E3/A1/P1 代码零改动，`git diff --stat` 可核验）
- 数据：`p2_turns.json` SHA-256 `a2116b83b38509e45d641ade17ae1791282729836bdca251aa8aba8aa9248a0c`
  （MultiWOZ 2.1 + `prepare_multiwoz_data --seed 42 --max-dialogues 100` 本机派生）

## 环境侧变更（不触及 git 跟踪文件）

1. `uv pip install sentencepiece==0.2.2`（venv 内，未改 pyproject/uv.lock）——
   必要原因：Mistral-7B-Instruct-v0.3 的 tokenizer_config 含 `add_prefix_space=true`，
   transformers 4.57 对 Llama 系 tokenizer 强制 from_slow 转换路径，缺 sentencepiece 直接报错。
2. NLTK `punkt`/`punkt_tab` 离线装入 `/root/nltk_data`（GitHub raw 下载解压）。
3. 模型经 ModelScope 下载至本机目录：
   `Qwen2-7B-Instruct`（Qwen/Qwen2-7B-Instruct）、`Mistral-7B-Instruct-v0.3`（LLM-Research/Mistral-7B-Instruct-v0.3）。

## 裁判格式失败与修复（步骤 6）

- 首次 judge run `sci34_f11ccba_20260901_judge`（prompt v2）在第 920/1600 条遇 Mistral
  输出解释文本（`'Directly refers to specific information'`）而非 YES/NO 开头，脚本按设计
  fail-closed 中止。**该目录（919 条记录 + manifest）原样保留**在 `judge/` 下备查。
- 修复（commit `ca627c7`，prompt 升 v3）：判定标准与语义不变；用户提示尾部追加
  「首行必须是 YES 或 NO」格式指令；新增一次有界重试（重试提示带格式提醒）；仍失败维持
  RuntimeError fail-closed。新增 `retried` 字段落盘。
- 重跑 `sci34_f11ccba_20260901_judge_v2`：1600 条全部完成，`parse_failures=0`，`retried=0`
  （v3 提示词下格式失败完全消失，重试机制未触发）。

## P1 异常观察与设计方复核（步骤 9）

GPU 运行方最初按“无 per-length warmup 的 CUDA 首次分配/kernel 编译”上报异常：2048/f0.25
的 stop→crop/role 中位数约 108 ms，8192/f0.25 约 1388 ms、f0.5 约 657 ms；同长度后续
单元格回落到 stop→crop 约 1–2.5 ms、stop→role 约 77–81 ms。原始数据未做任何剔除或重跑。

设计方随后结合 180 条 raw records 与代码顺序复核，否定了“一次性冷启动”解释：受影响单元的
额外等待跨 20 次重复持续，并随待恢复 KV 后缀长度与播放等待时间变化。根因是每次
`fixture.ensure_full()` 后未在 `player.start()` 前同步；stop 后首次设备同步把尚未完成的准备态
恢复错误计入 stop→crop/role。因此旧 run 的联合路径数值被判为协议无效，不能通过删除首个样本
或挑选所谓稳态单元修复。旧 run 继续作为审计记录保留；后续只能以新 run-id 按 prepared-state v2
协议定向重跑。

## GPU 干扰记录

A1 正式 run 前后 `nvidia-smi` 快照存于 `a1/nvidia_smi_before_formal.txt`、
`a1/nvidia_smi_after_formal.txt`；期间无其他 GPU 进程。
