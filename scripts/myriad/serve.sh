#!/bin/bash -l
# Myriad SGE jobscript: serve one model behind vLLM's OpenAI-compatible server.
#
# The resource directives below are defaults for the bf16 workhorse. Override per tier on the qsub
# line (resource flags take precedence over the directives), and pass model/quant settings via -v:
#
#   8B / 14B (bf16):  qsub scripts/myriad/serve.sh
#   32B (bf16):       qsub -l gpu=1 -v MODEL=Qwen/Qwen3-32B-Instruct,GPU_MEM_UTIL=0.95 scripts/myriad/serve.sh
#   70B-AWQ (TP=2):   qsub -l gpu=2 -v MODEL=meta-llama/Llama-3.3-70B-Instruct-AWQ-INT4,QUANT=awq,TP=2 scripts/myriad/serve.sh
#
# Greedy decoding is enforced client-side (LLMClient temperature=0); the server only pins the seed
# and the model revision. See docs/serving.md for the full tier/GPU table and queue notes.

#$ -l gpu=1
#$ -l h_rt=8:00:00
#$ -pe smp 8
#$ -l mem=32G
#$ -N vllm-serve
#$ -cwd
#$ -j y
# Set your Free or priority allocation:
# -P <project>

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-14B-Instruct}"
REVISION="${REVISION:-main}"
PORT="${PORT:-8000}"
DTYPE="${DTYPE:-bfloat16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
SEED="${SEED:-0}"
TP="${TP:-1}"                       # tensor-parallel size; 2 for 70B-AWQ on 2x A100-40GB
QUANT="${QUANT:-}"                  # e.g. 'awq' for the 70B-AWQ tier; empty for bf16
GUIDED_BACKEND="${GUIDED_BACKEND:-xgrammar}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.4}"
VENV="${VENV:-$HOME/venvs/precept-research}"

module load "$CUDA_MODULE"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

args=(
  serve "$MODEL"
  --revision "$REVISION"
  --port "$PORT"
  --dtype "$DTYPE"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --seed "$SEED"
  --tensor-parallel-size "$TP"
  --guided-decoding-backend "$GUIDED_BACKEND"
)
if [[ -n "$QUANT" ]]; then
  args+=(--quantization "$QUANT")
fi

echo "[serve] $(date -u +%FT%TZ) host=$(hostname) launching: vllm ${args[*]}"
exec vllm "${args[@]}"
