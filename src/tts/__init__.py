# src/tts/
# 二期新增：LLM token 流 → 句子片段（stream2sentence）→ （后续）CosyVoice2 流式合成
# - sentence_chunker.py : 断句并把每个片段映射回 assistant token 区间（喂 PlaybackTimeline）
