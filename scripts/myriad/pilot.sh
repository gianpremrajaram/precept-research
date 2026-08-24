#!/bin/bash -l
# Myriad SGE jobscript: serve a model AND drive the E3 pilot against it, in one job (DSE-050).
#
# Why one job rather than two. The vLLM endpoint listens on a compute node; a login node running
# `preceptx-pilot --base-url localhost:8000` would resolve localhost to the login node and find
# nothing. Splitting serve and drive across two jobs means discovering the server's hostname,
# holding a port open between nodes, and coordinating two queue waits. Co-locating them makes the
# endpoint a loopback address again, which is what the client already expects.
#
# The served name and revision come from configs/model/<TIER>.yaml - the same file the manifest
# records them from - so a run cannot serve one checkpoint while recording another. -P takes your
# project code and is the only thing you must supply:
#
#   qsub -P <project> scripts/myriad/pilot.sh
#
#   # the 8B tier on a V100 node instead:
#   qsub -P <project> -ac allow=EF -v TIER=qwen8b scripts/myriad/pilot.sh
#
#   # the one permitted retune (PREREGISTRATION §6):
#   qsub -P <project> -v ATTEMPT=2 scripts/myriad/pilot.sh
#
# Cost the sweep first, on the login node, where it issues no model calls and needs no GPU:
#   uv run preceptx-pilot --dry-run --model qwen14b
#
# See docs/myriad.md for the first-session runbook and docs/serving.md for the tier/GPU table.

#$ -l gpu=1
#$ -l h_rt=6:00:00
#$ -pe smp 8
# `mem` is PER SLOT: 8 x 4G = 32G total. Requesting mem=32G here would ask for 256G against an L
# node's 160G usable and queue forever rather than fail.
#$ -l mem=4G
#$ -N precept-pilot
#$ -cwd
#$ -j y
# L = 40 GB A100, what the bf16 14B workhorse (~28-30 GB) needs. EF = V100 for the 8B tier;
# U/V = 80 GB A100 for 32B bf16.
#$ -ac allow=L
# Project allocation: pass it on the qsub line (`qsub -P <project> ...`). An SGE directive cannot
# read the environment and there is no usable default, so it is deliberately not set here.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$HERE/../.." && pwd)}"

TIER="${TIER:-qwen14b}"                 # Hydra model group (configs/model/<tier>.yaml)
PORT="${PORT:-8000}"
ATTEMPT="${ATTEMPT:-1}"                 # 1 = first pass, 2 = the one permitted retune
RUNS_ROOT="${RUNS_ROOT:-runs}"
SERVE_TIMEOUT="${SERVE_TIMEOUT:-1800}"  # generous: a cold HF cache downloads ~28 GB before serving
# uv's own default location, so `uv sync` populates exactly what these jobs activate - no
# UV_PROJECT_ENVIRONMENT, no second path to keep in step. Override with -v VENV=<path>.
VENV="${VENV:-$REPO_ROOT/.venv}"
# TIER is exported so serve.sh serves the tier this pilot drives. Left unexported they are set
# independently, and `-v TIER=qwen8b` alone would drive the 8B pilot against a 14B server.
export PORT VENV TIER REPO_ROOT

# Checked here rather than in serve.sh so a typo'd tier fails now, not after the model has loaded.
if [[ ! -f "$REPO_ROOT/configs/model/$TIER.yaml" ]]; then
  echo "[pilot] no configs/model/$TIER.yaml - available tiers:" >&2
  ls "$REPO_ROOT/configs/model/" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

# Label the substrate from the GPU actually allocated, not from the node class requested: the
# manifest should record where the episodes were really served (the CLI refuses to run unlabelled).
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 |
  tr '[:upper:] ' '[:lower:]-' | tr -cd 'a-z0-9-')"
export PRECEPTX_SERVING_SUBSTRATE="${PRECEPTX_SERVING_SUBSTRATE:-myriad-${gpu_name:-unknown}}"

# The embedding encoder downloads on first use, which is at ANALYSIS time - after every episode has
# run. If this node has no outbound network that failure would land at the end of a full GPU hour
# with the dataset already paid for. Pull it now, when failing costs seconds.
echo "[pilot] warming the embedding encoder before any GPU time is spent"
python -c 'from preceptx.measure.featuriser import EncoderConfig, Featuriser
Featuriser(EncoderConfig()).embed_texts(["warm"])'

echo "[pilot] $(date -u +%FT%TZ) host=$(hostname) substrate=$PRECEPTX_SERVING_SUBSTRATE"

# One launch path: serve.sh owns the vLLM command line, and it `exec`s vllm, so $! is the server
# itself and the trap kills the right process on every exit path - success, failure, or the
# scheduler's SIGTERM at wallclock.
bash "$HERE/serve.sh" &
server_pid=$!
trap 'echo "[pilot] stopping server (pid $server_pid)"; kill "$server_pid" 2>/dev/null || true' EXIT

echo "[pilot] waiting up to ${SERVE_TIMEOUT}s for the endpoint on :$PORT"
deadline=$((SECONDS + SERVE_TIMEOUT))
until curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; do
  # A crashed server (bad revision, OOM, missing CUDA module) must fail now rather than burn the
  # whole timeout waiting for a process that is already gone.
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "[pilot] server exited before becoming ready - see the job log above" >&2
    exit 1
  fi
  if ((SECONDS >= deadline)); then
    echo "[pilot] endpoint not ready after ${SERVE_TIMEOUT}s" >&2
    exit 1
  fi
  sleep 5
done
echo "[pilot] endpoint live after ${SECONDS}s"

# The pilot's own health check verifies the endpoint serves the model that was configured, so a
# leftover job on this port serving another tier fails here rather than being recorded as this one.
# Conditions/difficulties/seeds are left to the CLI defaults, which are the pre-registered E3 cell
# (PREREGISTRATION §6): naming them here would let the two drift apart silently.
preceptx-pilot \
  --model "$TIER" \
  --base-url "http://localhost:${PORT}/v1" \
  --root "$RUNS_ROOT" \
  --attempt "$ATTEMPT"

echo "[pilot] $(date -u +%FT%TZ) complete"
