# TTFA 统一实测 RUNINFO

- 命令参数: ['/dataA/streamllm/experiments/scripts/run_ttfa_unified.py', '--sample-list', 'experiments/results/revision/r1_stats/repeat_subset_ids.json', '--json-dir', 'experiments/datasets/processed/json', '--audio-dir', 'experiments/datasets/processed/audio', '--datasets', 'crosswoz', 'multiwoz', '--asr-model', 'turbo', '--asr-device', 'cuda:0', '--llm-model', 'Qwen/Qwen2-7B-Instruct', '--llm-device', 'cuda:1', '--tts-url', 'http://127.0.0.1:20401', '--tts-spk', '晓伊', '--tts-speed', '0.8', '--silero-dir', '/root/.cache/torch/hub/snakers4_silero-vad_master', '--smoke', '3', '--inject-fault', 'asr_error', '--inject-fault-index', '1', '--output-dir', 'experiments/results/revision/r7_ttfa_unified/fatal_smoke', '--run-id', 'r7_smoke_fatal']
- schema_version: ttfa_unified/1
- run_id: r7_smoke_fatal
- config: {"asr_model": "turbo", "asr_device": "cuda:0", "llm_model": "Qwen/Qwen2-7B-Instruct", "llm_device": "cuda:1", "chunk_ms": 500, "prefix_segments": 1, "suffix_segments": 0, "recognition_threshold": 2.0, "max_tokens": 128, "pair_deadline_s": 900.0, "tts_url": "http://127.0.0.1:20401", "tts_spk": "晓伊", "tts_speed": 0.8, "sample_list_sha256": "e5a30a3507a742ae691de7a53b1ea77400986b5368c7ab564af16e99a9f38175", "requested_repetition_penalty": 1.1, "effective_repetition_penalty": "not_applied", "temperature": 0.1, "top_p": 0.9, "tts_speaker_mapping_note": "requested speaker id '晓伊' was mapped by the local service configuration (spk2info.pt) to the built-in Chinese female speaker embedding; not identical to the original paper voice", "platform_conditions_sha256": null, "silero_meta": {"ref": null, "dir": "/root/.cache/torch/hub/snakers4_silero-vad_master", "repo_commit": null, "repo_dirty": null, "artifact_path": "/root/.cache/torch/hub/snakers4_silero-vad_master/src/silero_vad/data/silero_vad.jit", "artifact_sha256": "e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720", "repo_commit_note": "目录非 git checkout，锁定依据为 artifact_sha256"}}
- config_hash: 466f62dccea046964436b6a0917b0ab7ff6f39cd2a09e26d772954f5e8afcc6d
- schedule_hash: 9a5888a1dd5446642a1d72a548f8a7fe10663a6f0fb224b1399bd7e6835a8ae0
- git: {'git_commit': '34ea12e93a18dcd9e3ed66fdfebfd41edc043ca1', 'git_dirty': False}
- env_versions: {"python": "3.10.18", "torch": "2.5.1+cu121", "numpy": "1.26.4", "soundfile": "0.13.1", "librosa": "0.11.0", "requests": "2.32.5", "scipy": "1.15.3"}
- silero_meta: {"ref": null, "dir": "/root/.cache/torch/hub/snakers4_silero-vad_master", "repo_commit": null, "repo_dirty": null, "artifact_path": "/root/.cache/torch/hub/snakers4_silero-vad_master/src/silero_vad/data/silero_vad.jit", "artifact_sha256": "e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720", "repo_commit_note": "目录非 git checkout，锁定依据为 artifact_sha256"}
- segmenter_meta: {"pse_and_segmenter_same_artifact": true, "artifact_sha256": "e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720", "segmenter_silero_injected": true}（PSE 与流式分段器同一固定 artifact，已断言一致）
- subset_sha256: 99b68bcf5924d5d0c63b0f1de11fecdec72d7d2476aa9b2d8e6b247fc5a80f9d
- audio_map_sha256: f8889486d6fac3fee9f89c5183f57d9272e3d38450fdaa20d7b97ed267e585c4
- playable 阈值: 1324 bytes（22050Hz×16bit×30ms）
- 采样实际生效参数: temperature=0.1, top_p=0.9, repetition_penalty=not_applied
- speaker 映射注记: requested speaker id '晓伊' was mapped by the local service configuration (spk2info.pt) to the built-in Chinese female speaker embedding; not identical to the original paper voice
- platform_conditions_sha256: None
- 故障注入: asr_error（仅冒烟）
