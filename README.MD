
# 项目说明
本项目主要研究大模型流式输入的优化问题，即文本的输入是流式的，通过优化，使流式文本输入完成后模型首个字符的推理延迟缩短。可以应用在流式语音输入场景以减少语音响应的延迟。

# 研究资料
大模型使用QWEN-2 7B，主要是因为该模型有一个语音模态的版本，便于后续比较。另外模型大小适合在8GB以下显存的配置中进行，方便研究。
具体要用到的模型从huggingface获取，具体为：
- 纯文本模型：Qwen/Qwen2-7B-Instruct，地址：https://huggingface.co/Qwen/Qwen2-7B-Instruct
- 语音模型：Qwen/Qwen2-Audio-7B-Instruct，地址：https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct
