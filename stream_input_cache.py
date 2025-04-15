from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
from dotenv import load_dotenv
import os
import traceback
is_loaded = load_dotenv()  # 默认加载项目根目录的 .env 文件

if not is_loaded:
    print("Warning: .env file not loaded successfully")

# 2. 打印环境变量
hf_home = os.getenv("HF_HOME")
if hf_home is None:
    print("HF_HOME environment variable is not set")
else:
    print(f"HF_HOME: {hf_home}")

hf_endpoint = os.getenv("HF_ENDPOINT")
if hf_endpoint is None:
    print("HF_ENDPOINT environment variable is not set")
else:
    print(f"HF_ENDPOINT: {hf_endpoint}")

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model_and_tokenizer(model_name):
    """加载模型和分词器"""
    print(f"正在加载模型 {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("模型加载完成")
    return model, tokenizer

def precompute_kv_cache(model, input_ids, attention_mask=None):
    """预计算输入序列的KV缓存"""
    print("预计算输入序列的KV缓存...")
    start_time = time.time()
    
    # 使用模型前向传播但不生成输出，只获取KV缓存
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True
        )
    
    # 获取KV缓存
    past_key_values = outputs.past_key_values
    
    precompute_time = time.time() - start_time
    print(f"KV缓存预计算完成，耗时: {precompute_time:.4f}秒")
    # 计算并打印KV缓存占用的存储空间
    total_size_bytes = 0
    for layer_cache in past_key_values:
        for tensor in layer_cache:
            total_size_bytes += tensor.nelement() * tensor.element_size()
    
    # 转换为更友好的单位
    total_size_mb = total_size_bytes / (1024 * 1024)
    print(f"KV缓存占用存储空间: {total_size_mb:.2f} MB")
    
    return past_key_values, precompute_time

def generate1token_with_precomputed_cache(
    model, 
    tokenizer,
    input_ids, 
    attention_mask=None, 
    past_key_values=None,
    temperature=0.7,
    top_p=0.9
):
    """使用预计算的KV缓存生成文本"""
    try:
        print("开始使用预计算缓存生成文本...")
        
        # 准备用于生成的输入
        if past_key_values is not None:
            # 使用最后一个token作为输入
            gen_input_ids = input_ids[:, -1:]
            
            # 计算正确的position_ids
            seq_length = input_ids.shape[1]
            position_ids = torch.arange(seq_length - 1, seq_length, dtype=torch.long, device=gen_input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(gen_input_ids)
            
            if attention_mask is not None:
                gen_attention_mask = torch.cat([
                    attention_mask, 
                    torch.ones((attention_mask.shape[0], 1), 
                              dtype=attention_mask.dtype, 
                              device=attention_mask.device)
                ], dim=-1)
            else:
                gen_attention_mask = None
                print("未使用attention_mask")
        else:
            gen_input_ids = input_ids
            gen_attention_mask = attention_mask
            position_ids = None
        
        # 开始计时
        start_time = time.time()
        
        # 使用预计算的KV缓存生成文本
        with torch.no_grad():
            # 尝试直接使用forward方法而不是generate
            outputs = model(
                input_ids=gen_input_ids,
                attention_mask=gen_attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                use_cache=True,
                return_dict=True
            )
            
            # 获取logits并生成下一个token
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            # 打印首个token生成时间
            first_token_time = time.time() - start_time
            
            # 将生成的token添加到输入中，确保维度匹配
            next_token = next_token.view(1, 1)  # 将形状从 [1] 改为 [1, 1]
            generated_ids = torch.cat([gen_input_ids, next_token], dim=1)
            
       
        # 提取生成的部分
        if past_key_values is not None:
            generated_part = generated_ids[0, 1:]  # 排除输入的最后一个token
        else:
            generated_part = generated_ids[0, input_ids.shape[1]:]
        
        # 解码结果
        response = tokenizer.decode(generated_part, skip_special_tokens=True)
        
        return response, first_token_time, len(generated_part)
        
    except Exception as e:
        import traceback
        print(f"生成过程中出现错误: {str(e)}")
        print("\n详细错误信息:")
        print(traceback.format_exc())
        raise

def generate1token_with_orign(model, tokenizer, input_ids, attention_mask=None, temperature=0.7, top_p=0.9):
    """使用模型的generate方法生成回复
    
    Args:
        model: 语言模型
        tokenizer: 分词器
        input_ids: 输入序列的token ids
        attention_mask: 注意力掩码
        max_new_tokens: 最大生成token数
        temperature: 采样温度
        top_p: top-p采样参数
        
    Returns:
        tuple: (生成的回复文本, 首个token生成时间, 生成的token数量)
    """
    try:
        start_time = time.time()
        
        # 生成第一个token
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1,  # 只生成一个token
            use_cache=True
        )
        
        # 计算总生成时间
        first_token_time = time.time() - start_time
        
        # 只解码新生成的token
        generated_ids = outputs[0, input_ids.shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return response, first_token_time, len(generated_ids)
        
    except Exception as e:
        print(f"生成过程中出现错误: {str(e)}")
        print("\n详细错误信息:")
        print(traceback.format_exc())
        raise


def run_comparison_test(prompt, model_name):
    """运行对比测试，比较使用预计算缓存和不使用预计算缓存的性能差异"""
    model, tokenizer = load_model_and_tokenizer(model_name)
    
    # 准备输入
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    
    print("\n" + "="*50)
    print(f"输入提示词: '{len(prompt)}'")
    print("="*50)
    
    # 1. 使用预计算缓存的生成
    print("\n方法1: 使用预计算缓存")
    # 预计算KV缓存
    past_key_values, precompute_time = precompute_kv_cache(
        model,
        model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask
    )
    
    with_cache_response, with_cache_first_token_time, with_cache_tokens = generate1token_with_precomputed_cache(
        model,
        tokenizer,
        model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        past_key_values=past_key_values
    )

    # 2. 不使用预计算缓存的生成
    print("\n方法2: 不使用预计算缓存")
    no_cache_response, no_cache_first_token_time, no_cache_tokens = generate1token_with_orign(
        model,
        tokenizer,
        model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask
    )
    # # 1. 不使用预计算缓存的生成
    # print("\n方法1: 不使用预计算缓存")
    # no_cache_response, no_cache_first_token_time, no_cache_tokens = generate1token_with_orign(
    #     model,
    #     tokenizer,
    #     model_inputs.input_ids,
    #     attention_mask=model_inputs.attention_mask
    # )
    
    # # 2. 使用预计算缓存的生成
    # print("\n方法2: 使用预计算缓存")
    # # 预计算KV缓存
    # past_key_values, precompute_time = precompute_kv_cache(
    #     model,
    #     model_inputs.input_ids,
    #     attention_mask=model_inputs.attention_mask
    # )
    
    with_cache_response, with_cache_first_token_time, with_cache_tokens = generate1token_with_precomputed_cache(
        model,
        tokenizer,
        model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        past_key_values=past_key_values
    )
    
    # 输出比较结果
    print("\n" + "="*50)
    print("\n输出结果比较:")
    print(f"不使用缓存的输出: {no_cache_response}")
    print(f"使用缓存的输出: {with_cache_response}")
    print("\n" + "="*50)
    print("性能比较:")
    print("="*50)
    print(f"不使用预计算缓存:")
    print(f"  - 首个token生成时间: {no_cache_first_token_time:.4f}秒")
    
    print(f"\n使用预计算缓存:")
    print(f"  - 缓存预计算时间: {precompute_time:.4f}秒")
    print(f"  - 首个token生成时间: {with_cache_first_token_time:.4f}秒")
    print(f"  - 总生成时间(含预计算): {precompute_time + with_cache_first_token_time:.4f}秒")
    
    print(f"\n首个token性能提升: {(no_cache_first_token_time - with_cache_first_token_time) / no_cache_first_token_time * 100:.2f}%")
    



def simulate_stream_input(prompt, model_name, char_delay=0.2):
    """模拟流式输入场景，字符逐个输入，并预计算KV缓存"""
    model, tokenizer = load_model_and_tokenizer(model_name)
    
    print(f"\n开始模拟流式输入，每个字符延迟{char_delay}秒...")
    
    # 模拟流式输入
    current_input = ""
    for i, char in enumerate(prompt):
        current_input += char
        print(f"流式输入: {current_input}")
        
        # 准备当前输入
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": current_input}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        
        # 预计算KV缓存
        past_key_values, precompute_time = precompute_kv_cache(
            model,
            model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask
        )
        
        # 保存最新的KV缓存
        latest_past_key_values = past_key_values
        latest_input_ids = model_inputs.input_ids
        latest_attention_mask = model_inputs.attention_mask
        
        # 模拟字符输入延迟
        if i < len(prompt) - 1:  # 不在最后一个字符后延迟
            time.sleep(char_delay)
    
    print("\n流式输入完成，开始生成回答...")
    
    # 使用最终的KV缓存生成回答
    response, first_token_time, total_time, tokens_count = generate_with_precomputed_cache(
        model,
        tokenizer,
        latest_input_ids,
        attention_mask=latest_attention_mask,
        past_key_values=latest_past_key_values
    )
    
    print("\n" + "="*50)
    print("流式输入生成结果:")
    print("="*50)
    print(f"输入: {prompt}")
    print(f"回答: {response}")
    print(f"首个token生成时间: {first_token_time:.4f}秒")
    print(f"总生成时间: {total_time:.4f}秒")

def test_basic_inference(prompt, model_name):
    """测试模型基本推理功能
    
    Args:
        prompt (str): 输入提示词
        model_name (str): 模型名称
        
    Returns:
        tuple: (是否成功, 响应文本)
    """
    try:
        print("\n=== 测试基本推理功能 ===")
        
        # 加载模型和tokenizer
        model, tokenizer = load_model_and_tokenizer(model_name)
        
        # 准备输入
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        
        # 执行推理
        response, first_token_time, total_time, tokens_count = generate_with_precomputed_cache(
            model,
            tokenizer,
            model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            past_key_values=None  # 不使用预计算缓存
        )
        
        # 打印结果
        print(f"输入: {prompt}")
        print(f"输出: {response}")
        print(f"生成token数: {tokens_count}")
        print(f"总耗时: {total_time:.4f}秒")
        
        return True, response
        
    except Exception as e:
        print(f"推理测试失败: {str(e)}")
        return False, str(e)

def test_inference_with_cache(prompt, model_name):
    """使用预计算的KV缓存进行推理测试
    
    Args:
        prompt (str): 原始输入提示词
        model_name (str): 模型名称（当model和tokenizer未提供时使用）
        
    Returns:
        tuple: (是否成功, 响应文本, 性能指标)
    """
    try:
        print("\n=== 测试预计算缓存推理 ===")
        model, tokenizer = load_model_and_tokenizer(model_name)

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        input_ids = model_inputs.input_ids
        attention_mask=model_inputs.attention_mask

        # 预计算KV缓存
        past_key_values, precompute_time = precompute_kv_cache(
            model,
            input_ids,
            attention_mask=attention_mask
        )

        print(f"KV缓存预计算耗时: {precompute_time:.4f}秒")
        
        # 使用预计算缓存进行推理
        response, first_token_time, total_time, tokens_count = generate_with_precomputed_cache(
            model,
            tokenizer,
            input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values
        )
        
        # 打印结果
        print(f"输入: {prompt}")
        print(f"输出: {response}")
        print(f"生成token数: {tokens_count}")
        print(f"首个token生成时间: {first_token_time:.4f}秒")
        print(f"总耗时: {total_time:.4f}秒")
        
        return True, response, {
            "first_token_time": first_token_time,
            "total_time": total_time,
            "tokens_count": tokens_count
        }
        
    except Exception as e:

        print(f"预计算缓存推理测试失败: {str(e)}")
        print("\n详细错误信息:")
        print(traceback.format_exc())
        return False, str(e), None

def test_basic_inference(prompt, model_name):
    """测试基本推理功能，不使用预计算缓存"""
    try:
        print("\n开始基本推理测试...")
        model, tokenizer = load_model_and_tokenizer(model_name)
        
        # 准备输入
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        input_ids = model_inputs.input_ids
        attention_mask = model_inputs.attention_mask

        # 使用原始generate方法进行推理
        response, first_token_time, total_time, tokens_count = generate_with_orign(
            model,
            tokenizer,
            input_ids,
            attention_mask=attention_mask
        )
        
        # 打印结果
        print(f"输入: {prompt}")
        print(f"输出: {response}")
        print(f"生成token数: {tokens_count}")
        print(f"首个token生成时间: {first_token_time:.4f}秒")
        print(f"总耗时: {total_time:.4f}秒")
        
        return True, response, {
            "first_token_time": first_token_time,
            "total_time": total_time,
            "tokens_count": tokens_count
        }
        
    except Exception as e:
        print(f"基本推理测试失败: {str(e)}")
        print("\n详细错误信息:")
        print(traceback.format_exc())
        return False, str(e), None


if __name__ == "__main__":
    # test_prompt = "请简要介绍一下大语言模型的工作原理。"
    model_name = "Qwen/Qwen2-7B-Instruct"

    # 原生推理测试
    # success, response, metrics = test_basic_inference(test_prompt, model_name)


    # model_name = "/usr/local/app/jupyterlab/model/download/DeepSeek-R1-Distill-Llama-8B"
    # 运行对比测试
    test_prompt = "一、VLLM和Ollama是什么？基础知识解析\n在深入探讨之前，我们先来了解一下这两个框架的核心功能。\n\n什么是VLLM？\nVLLM（超大型语言模型）是SKYPILOT开发的推理优化框架，主要用于提升大语言模型在GPU上的运行效率。它的优势体现在以下几个方面：\n\n快速令牌生成：采用连续批处理技术，让令牌生成速度大幅提升。\n高效内存利用：借助PagedAttention技术，在处理大上下文窗口时，能有效控制GPU内存消耗。\n无缝集成：与PyTorch、TensorFlow等主流深度学习平台兼容，可轻松融入AI工作流程。\n\n\nVLLM深受AI研究人员和需要大规模高性能推理的企业青睐。\n\n什么是奥拉玛（Ollama）？\nOllama是一个本地大语言模型运行时环境，能简化开源AI模型的部署和使用流程。它具备以下特点：\n\n预打包模型丰富：内置了LLaMA、Mistral、Falcon等多种模型。\n硬件适配性强：针对日常使用的硬件进行了CPU和GPU推理优化，无论是MacBook、PC还是边缘设备，都能流畅运行AI模型。\n操作便捷：提供简洁的API和命令行界面（CLI），开发人员只需简单配置，就能快速启动大语言模型。\n对于想在个人电脑上尝试AI模型的开发人员和AI爱好者来说，Ollama是个不错的选择。\n\n二、性能大比拼：速度、内存与可扩展性\n性能是衡量推理框架优劣的关键指标，下面我们从速度、内存效率和可扩展性三个方面，对VLLM和Ollama进行对比。\n\n\n\n关键性能指标分析\nVLLM借助PagedAttention技术，在推理速度上优势明显，处理大上下文窗口时也能游刃有余。这让它成为聊天机器人、搜索引擎、AI写作辅助工具等高性能AI应用的首选。\n\nOllama的速度也还不错，但受限于本地硬件配置。在MacBook、PC和边缘设备上运行小型模型时表现良好，不过遇到超大模型就有些力不从心了。\n\n结论：Ollama更适合初学者，而需要深度定制的开发人员则可以选择VLLM。\n\n三、应用场景：VLLM和Ollama分别适用于哪些场景？\nVLLM的最佳应用场景\n企业AI应用：如客户服务聊天机器人、AI驱动的搜索引擎等。\n云端高端GPU部署：适用于A100、H100、RTX 4090等高端GPU的云端大语言模型部署。\n模型微调与定制：方便进行模型微调和运行自定义模型。\n大上下文窗口需求：适用于对上下文窗口要求较高的应用。\n不太适用的场景：个人笔记本电脑、日常AI实验。\n\nOllama的最佳应用场景\n本地设备运行：无需借助云资源，就能在Mac、Windows或Linux系统的设备上运行大语言模型。\n本地模型试验：不需要复杂的设置，就能在本地轻松试验各种模型。\n简易API集成：开发人员可以通过简单的API将AI功能集成到应用程序中。\n边缘计算应用：在边缘计算场景中表现出色。\n不太适用的场景：大规模AI部署、高强度GPU计算任务。\n\n总结：VLLM更适合AI工程师，而Ollama则是开发人员和AI爱好者的好帮手。\n\n四、如何上手使用？（分步指南）\nVLLM入门教程\n安装依赖项：在命令行中输入pip install vllm，按提示完成安装。\n在LLaMA模型上运行推理：在Python环境中，输入以下代码：\n\nfrom vllm import LLM\nllm = LLM(model=\"meta-llama/Llama-2-7b\")\noutput = llm.generate(\"What is VLLM?\")\n上述代码中，首先从vllm库中导入LLM类，然后创建LLM对象，并指定使用meta-llama/Llama-2-7b模型。最后，使用generate方法输入问题\"What is VLLM?\"，就能得到模型的输出结果。\n\n上面是vllm和ollama的介绍，我现在要开发mcp工作流，打算使用n8n或者dify作为工作流引擎，你建议使用vllm还是ollama来部署大模型？"
    run_comparison_test(test_prompt, model_name)
    
    # 模拟流式输入
    # simulate_stream_input(test_prompt, model_name=model_name, char_delay=0.5)
    
    # 测试基本推理功能
    # 1. 首先测试基本推理功能
    # print("\n=== 第一步：测试基本推理 ===")
    # success, response = test_basic_inference(test_prompt, model_name)
    
    # KV缓存推理测试
    # cache_success, cache_response, metrics = test_inference_with_cache(
    #         test_prompt,
    #         model_name=model_name
    #     )
        
    # if cache_success:
    #     print("\n预计算缓存推理测试成功！")
    #     print("\n=== 性能对比 ===")

    #     print(f"使用缓存后首个token生成时间: {metrics['first_token_time']:.4f}秒")
    #     print(f"使用缓存后总生成时间: {metrics['total_time']:.4f}秒")
    # else:
    #     print("\n预计算缓存推理测试失败，请检查错误信息。")

