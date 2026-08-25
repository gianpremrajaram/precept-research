#!/bin/bash -l
# Myriad SGE jobscript: serve one model behind vLLM's OpenAI-compatible server.
#
# The resource directives below are defaults for the bf16 workhorse. Override per tier on the qsub
# line (resource flags take precedence over the directives) with -v TIER=<group>; -P takes your
# project code.
#
# The served name and revision come from configs/model/<TIER>.yaml, which is the same file the run
# manifest records them from. They are deliberately NOT arguments: passed separately they can
# disagree with what the manifest claims, and nothing downstream can detect it - the health check
# compares the served model id, but /v1/models carries no revision, so a wrong one is invisible.
#
#   14B (bf16, default):
#     qsub -P <project> scripts/myriad/serve.sh
#   8B (bf16, V100 class):
#     qsub -P <project> -ac allow=EF -v TIER=qwen8b scripts/myriad/serve.sh
#   32B (bf16, 80 GB A100):
#     qsub -P <project> -ac allow=U -v TIER=qwen32b,GPU_MEM_UTIL=0.95 scripts/myriad/serve.sh
#   70B-AWQ (TP=2), the one tier with no config file yet:
#     qsub -P <project> -l gpu=2 -v MODEL=<70B-AWQ-repo-id>,REVISION=<sha>,QUANT=awq,TP=2 \
#          scripts/myriad/serve.sh
#     (the 70B repo id is a PLACEHOLDER until DSE-005 verifies and pins it; MODEL/REVISION remain
#      overridable for exactly this case and log a warning when they diverge from the config)
#
# Dense Qwen3 ids carry no -Instruct suffix (P0-3).
# Qwen3's hybrid thinking is disabled per-request by the client (chat_template_kwargs), not here.
# Greedy decoding is enforced client-side (LLMClient temperature=0); the server only pins the seed
# and the model revision. See docs/serving.md for the full tier/GPU table and queue notes.

#$ -l gpu=1
#$ -l h_rt=8:00:00
#$ -pe smp 8
# `mem` is PER SLOT, not per job: this is 8 x 4G = 32G total. The earlier `mem=32G` asked for 256G
# against an L node's 160G usable, which is not a rejection - it is a job that queues forever.
#$ -l mem=4G
#$ -N vllm-serve
#$ -cwd
#$ -j y
# Node class: L = 40 GB A100, required by the bf16 14B workhorse (~28-30 GB in use). The 8B tier
# also fits the V100 class (edit to `-ac allow=EF`); 32B bf16 needs an 80 GB A100 (`allow=U`/`V`).
#$ -ac allow=L
# Project allocation: pass it on the qsub line (`qsub -P <project> ...`). No usable default
# exists, and an SGE directive cannot read the environment, so it is deliberately not set here.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$HERE/../.." && pwd)}"

TIER="${TIER:-qwen14b}"             # Hydra model group; configs/model/<TIER>.yaml
TIER_FILE="$REPO_ROOT/configs/model/$TIER.yaml"
PORT="${PORT:-8000}"
DTYPE="${DTYPE:-bfloat16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
SEED="${SEED:-0}"
TP="${TP:-1}"                       # tensor-parallel size; 2 for 70B-AWQ on 2x A100-40GB
QUANT="${QUANT:-}"                  # e.g. 'awq' for the 70B-AWQ tier; empty for bf16
GUIDED_BACKEND="${GUIDED_BACKEND:-xgrammar}"
# `none` by default since DSE-051: the job runs containerised and there is no module system inside
# the image. Nothing is lost - torch's bundled cu12 libraries are the CUDA userspace and
# `apptainer --nv` injects the host driver. Kept overridable for a bare run on a future non-RHEL7
# node; see the load below.
CUDA_MODULE="${CUDA_MODULE:-none}"
# uv's own default location, so `uv sync` populates exactly what these jobs activate - no
# UV_PROJECT_ENVIRONMENT, no second path to keep in step. Override with -v VENV=<path>.
VENV="${VENV:-$REPO_ROOT/.venv}"

# shellcheck source=scripts/myriad/_common.sh
source "$HERE/_common.sh"

# Myriad is glibc 2.17 and the lock is manylinux_2_28 throughout, so vLLM can only run inside the
# container (DSE-051). Asserted, never pulled: an image pull here would spend A100 time on network
# I/O. A no-op when pilot.sh has already entered the container - which is what keeps serve and
# drive in one process tree, so pilot.sh's trap on $! still kills the right process.
require_image
enter_container "$HERE/serve.sh" "$@"

# Caches onto Scratch before anything can populate them under $HOME. Override HF_HOME to relocate;
# prefetch.sh resolves the same paths, so a login-node pre-pull lands where this job reads.
cache_to_scratch

# vLLM's wheels bundle their own CUDA runtime through torch, so the module is a convenience, not a
# requirement - and inside the container it is unreachable, which is why CUDA_MODULE now defaults to
# `none`. The locked stack is cu12 throughout (nvidia-cublas-cu12 12.8.4.1) and the L-node driver is
# 550.127.05 / CUDA 12.4, so CUDA minor-version compatibility covers it. If it is ever loaded for a
# bare run the name must be one Myriad actually has (`module avail cuda`, e.g.
# cuda/12.2.2/gnu-10.2.0).
if [[ "$CUDA_MODULE" != "none" ]]; then
  module load "$CUDA_MODULE" || {
    echo "[serve] module load '$CUDA_MODULE' failed." >&2
    echo "[serve] Run 'module avail cuda' on a login node and pass the exact name:" >&2
    echo "[serve]   qsub -v CUDA_MODULE=<name> scripts/myriad/serve.sh" >&2
    echo "[serve] or CUDA_MODULE=none to skip it (torch ships its own CUDA runtime)." >&2
    exit 1
  }
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

resolve_tier "$TIER_FILE" serve

# Serving-environment capture (P2-10). Written to a sidecar, not just echoed: manifest.serve_env()
# reads it via PRECEPTX_SERVE_ENV so the server-side stack lands in the run manifest instead of
# depending on someone copying it out of the job log. Echoed too, so the log still stands alone.
SERVE_ENV_OUT="$(serve_env_path)"
write_serve_env "$SERVE_ENV_OUT"
echo "[serve-env] wrote $SERVE_ENV_OUT"
sed 's/^/[serve-env] /' "$SERVE_ENV_OUT"

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
