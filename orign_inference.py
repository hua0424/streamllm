import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 配置参数
MODEL_NAME = "Qwen/Qwen2.5-7B"  # 请确认实际模型名称/路径
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=DTYPE,
    device_map=DEVICE,
    trust_remote_code=True
).eval()

# 推理函数（带KV缓存）
def generate_with_kv_cache(
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_k: int = 50
):
    # 编码输入文本
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids.to(DEVICE)
    attention_mask = inputs.attention_mask.to(DEVICE)
    
    # 初始化生成参数
    generated_ids = input_ids.clone()
    past_key_values = None
    stopping_criteria = False
    
    # 逐步生成
    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True  # 启用KV缓存
            )
        
        # 更新KV缓存
        next_token_logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values
        
        # 采样下一个token
        probs = torch.softmax(next_token_logits / temperature, dim=-1)
        top_probs, top_indices = torch.topk(probs, top_k)
        next_token = top_indices[0, torch.multinomial(top_probs, 1)]
        
        # 检查终止条件
        if next_token == tokenizer.eos_token_id:
            stopping_criteria = True
            break
        
        # 更新生成序列
        generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=-1)
        input_ids = next_token.unsqueeze(0)
        # 更新attention_mask，为新token添加mask值1
        attention_mask = torch.ones((1, 1), dtype=torch.long, device=DEVICE)
        
    # 解码结果
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# 执行推理
prompt = "中国的首都是"
result = generate_with_kv_cache(prompt)
print(f"Input: {prompt}\nOutput: {result}")