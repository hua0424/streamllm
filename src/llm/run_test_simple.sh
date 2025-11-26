#!/bin/bash

# 简单的流式LLM测试脚本

echo "===== 流式LLM测试（直接运行版） ====="

# 切换到脚本所在目录
cd "$(dirname "$0")"

# 创建结果目录
mkdir -p ../../experiments/results
export LD_LIBRARY_PATH="/usr/local/app/jupyterlab/yanjiu/streamllm/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib/:$LD_LIBRARY_PATH"

# 运行测试
echo "运行流式LLM测试..."
uv run python -m src.llm.run_llm_test

echo "测试完成！"