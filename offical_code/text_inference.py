from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
from dotenv import load_dotenv
import os
from transformers import TextIteratorStreamer
from threading import Thread

is_loaded = load_dotenv()  # 默认加载项目根目录的 .env 文件

if not is_loaded:
    print("Warning: .env file not loaded successfully")

# 2. 打印环境变量
hf_home = os.getenv("HF_HOME")
if hf_home is None:
    print("HF_HOME environment variable is not set")
else:
    print(f"HF_HOME: {hf_home}")

# 退出程序
device = "cuda" # the device to load the model onto

model_name = "Qwen/Qwen2-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

# Convert messages to model input format
input_text = tokenizer.apply_chat_template(messages, tokenize=False)
model_inputs = tokenizer([input_text], return_tensors="pt").to(device)

# 创建 streamer
streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)

# Record start time
start_time = time.time()
first_token_time = None

# 在新线程中运行生成
generation_kwargs = dict(
    **model_inputs,
    streamer=streamer,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.95,
)
thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

# 在主线程中打印生成的文本
generated_text = ""
for new_text in streamer:
    if first_token_time is None:
        first_token_time = time.time() - start_time
    print(new_text, end="", flush=True)
    generated_text += new_text

# 等待生成完成
thread.join()

end_time = time.time()
total_time = end_time - start_time

# 计算生成的token数量
total_tokens = len(tokenizer.encode(generated_text))

print("\n\nGeneration Statistics:")
print(f"First token latency: {first_token_time:.2f} seconds")
print(f"Total generation time: {total_time:.2f} seconds")
print(f"Total tokens generated: {total_tokens}")
print(f"Tokens per second: {total_tokens/total_time:.2f}")

