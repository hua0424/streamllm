# TTFA 统一实测 RUNINFO

- 命令参数: ['/dataA/streamllm/experiments/scripts/run_ttfa_unified.py', '--sample-list', 'experiments/results/revision/r1_stats/repeat_subset_ids.json', '--json-dir', 'experiments/datasets/processed/json', '--audio-dir', 'experiments/datasets/processed/audio', '--datasets', 'crosswoz', 'multiwoz', '--asr-model', 'turbo', '--asr-device', 'cuda:0', '--llm-model', 'Qwen/Qwen2-7B-Instruct', '--llm-device', 'cuda:1', '--tts-url', 'http://127.0.0.1:20401', '--tts-spk', '晓伊', '--tts-speed', '0.8', '--silero-dir', '/root/.cache/torch/hub/snakers4_silero-vad_master', '--smoke', '3', '--inject-fault', 'asr_error', '--output-dir', 'experiments/results/revision/r7_ttfa_unified', '--run-id', 'r7_smoke']
- schema_version: ttfa_unified/1
- run_id: r7_smoke
- config: {"asr_model": "turbo", "asr_device": "cuda:0", "llm_model": "Qwen/Qwen2-7B-Instruct", "llm_device": "cuda:1", "chunk_ms": 500, "prefix_segments": 1, "suffix_segments": 0, "recognition_threshold": 2.0, "max_tokens": 128, "pair_deadline_s": 900.0, "tts_url": "http://127.0.0.1:20401", "tts_spk": "晓伊", "tts_speed": 0.8, "sample_list_sha256": "e5a30a3507a742ae691de7a53b1ea77400986b5368c7ab564af16e99a9f38175", "requested_repetition_penalty": 1.1, "effective_repetition_penalty": "not_applied", "temperature": 0.1, "top_p": 0.9, "silero_meta": {"ref": null, "dir": "/root/.cache/torch/hub/snakers4_silero-vad_master", "repo_commit": null, "repo_dirty": null, "artifact_path": "/root/.cache/torch/hub/snakers4_silero-vad_master/src/silero_vad/data/silero_vad.jit", "artifact_sha256": "e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720", "repo_commit_note": "目录非 git checkout，锁定依据为 artifact_sha256"}}
- config_hash: d60136b3c2bd335481aceebb9a7e32b6eb1693cbe77fc9af9934825170b12ea2
- schedule_hash: 9a5888a1dd5446642a1d72a548f8a7fe10663a6f0fb224b1399bd7e6835a8ae0
- git: {'git_commit': '1a0ddc83d3082ddedc443695d0be0da58669705c', 'git_dirty': True}
- env_versions: {"python": "3.10.18", "torch": "2.5.1+cu121", "numpy": "1.26.4", "soundfile": "0.13.1", "librosa": "0.11.0", "requests": "2.32.5", "scipy": "1.15.3"}
- silero_meta: {"ref": null, "dir": "/root/.cache/torch/hub/snakers4_silero-vad_master", "repo_commit": null, "repo_dirty": null, "artifact_path": "/root/.cache/torch/hub/snakers4_silero-vad_master/src/silero_vad/data/silero_vad.jit", "artifact_sha256": "e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720", "repo_commit_note": "目录非 git checkout，锁定依据为 artifact_sha256"}
- segmenter_meta: {"pse_and_segmenter_same_artifact": true, "artifact_sha256": "e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720", "segmenter_silero_injected": true}（PSE 与流式分段器同一固定 artifact，已断言一致）
- subset_sha256: 99b68bcf5924d5d0c63b0f1de11fecdec72d7d2476aa9b2d8e6b247fc5a80f9d
- audio_map_sha256: f8889486d6fac3fee9f89c5183f57d9272e3d38450fdaa20d7b97ed267e585c4
- playable 阈值: 1324 bytes（22050Hz×16bit×30ms）
- 采样实际生效参数: temperature=0.1, top_p=0.9, repetition_penalty=not_applied
- 故障注入: asr_error（仅冒烟）
