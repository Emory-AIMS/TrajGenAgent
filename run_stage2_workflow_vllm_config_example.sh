#!/bin/bash
#SBATCH --job-name=    
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=
#SBATCH --gres=gpu:1           
#SBATCH --mem=
#SBATCH --time=
#SBATCH --output=output/

cd /your_path
source /your_path
conda activate your_env

export HF_HOME="/your_path"

echo "=== [1/3] Starting vLLM API Server in background ==="

python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-32B-Instruct \
    --served-model-name Qwen/Qwen2.5-32B-Instruct \
    --gpu-memory-utilization 0.95 \
    --max-model-len 8192 \
    --port 8000 \
    --disable-log-requests \
    --disable-log-stats &   


VLLM_PID=$!

echo "=== [2/3] Waiting for vLLM Server to be ready ==="

elapsed=0
while ! curl -s http://localhost:8000/v1/models > /dev/null; do
    mins=$((elapsed / 60))
    secs=$((elapsed % 60))
    echo "⏳ Waiting for vLLM to load model... [Elapsed time: ${mins}m ${secs}s]"
    sleep 10
    elapsed=$((elapsed + 10))
    
    if [ $elapsed -ge 1800 ]; then
        echo "❌ ERROR: vLLM server failed to start within 30 minutes. Exiting."
        kill $VLLM_PID
        exit 1
    fi
done
echo "vLLM Server is successfully up and running!"

echo "=== [3/3] Starting LangGraph Hybrid Workflow (Client) ==="

python -u location_time_tools_llm_numosim.py

echo "=== Job Finished. Cleaning up vLLM Server ==="

kill $VLLM_PID
echo "Cleanup done. Exiting."