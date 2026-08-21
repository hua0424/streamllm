# RUNINFO — decode_to_first_sentence 补测

- 命令: `/dataA/streamllm/experiments/scripts/measure_decode_to_first_sentence.py --llm-device cuda:1 --output experiments/results/revision/r6_ttfa/decode_to_first_sentence.csv`
- git commit: dd2e6e03c4613e835ef632956b76a5b8f97d6523
- 起止: 2026-08-21T10:59:34 → 2026-08-21T11:04:07（耗时 273s）
- 输入文件: /dataA/streamllm/experiments/results/revision/r4_commit/exp1_results_20260820_171522.json
  - sha256: 32dc6f5b6889bf9984623653f6a5b2aa497a06d15ff6b3c14511c22b2f8d7277
  - E4 config: llm_model=Qwen/Qwen2-7B-Instruct, max_tokens=128, llm_device=cuda:1
- 样本: 50 条（streaming 模式），sample_ids sha256: b73ed71a9565f5af73f8ff8cc2589c1db60ba38bb0644af87076d7c49ead4c79
- 重放模式: fragment_replay（committed_fragments 增量 cache_prompt + 空片段 is_end=True 收尾）
- 参数: llm_model=(config 默认), llm_device=cuda:1,
  max_tokens=128, warmup=3, repeat=1,
  generate 采样参数 {'temperature': 0.1, 'top_p': 0.9, 'repetition_penalty': 1.1}
- 结果: rows=50, error=0；CSV=decode_to_first_sentence.csv；summary=decode_to_first_sentence.summary.txt
- 计时口径: decode_to_first_sentence_ms = 首个句末标点 token yield 时刻 − 首个 token yield 时刻（不含预填）
