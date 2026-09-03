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
# records them from - so a run cannot serve one checkpoint while recording another. No -P project
# code: the Free allocation is the default (see the directive note below).
#
#   qsub scripts/myriad/pilot.sh
#
#   # the 8B tier on a V100 node instead:
#   qsub -ac allow=EF -v TIER=qwen8b scripts/myriad/pilot.sh
#
#   # the one permitted retune (PREREGISTRATION §6):
#   qsub -v ATTEMPT=2 scripts/myriad/pilot.sh
#
#   # a characterisation grid, which must NOT emit a gate verdict (see the DRIVER note below).
#   # --no-analysis releases the GPU at the last episode; job 227886 held an A100 for 2h37m
#   # running statsmodels, and the analysis is also the only part that can fail AFTER the
#   # episodes are paid for. Analyse afterwards on a login node, with no GPU and no time limit:
#   #   preceptx-analyse --dataset-hash <the hash the driver printed>
#   qsub -N precept-rq1 -l h_rt=6:00:00 -v DRIVER=preceptx-rq1 scripts/myriad/pilot.sh \
#     --conditions C0,C1,C2,C3,C4 --difficulties easy,medium,hard --seeds "$(seq -s, 0 31)" \
#     --no-analysis
#
#   # the RQ3a judge replication (DSE-065). Its --root is the CORPUS root, not the runs root, and
#   # the corpus must already be on disk - prefetch.sh pulls it. Cost it first on a login node:
#   #   preceptx-rq3a --root ~/Scratch/rq3a --judge --dry-run
#   qsub -N precept-rq3a -v DRIVER=preceptx-rq3a,RQ3A_ROOT=$HOME/Scratch/rq3a scripts/myriad/pilot.sh
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
# No -P directive: the Free allocation is Myriad's default for UCL internal users, verified live
# 25-26 Aug 2026 - jobs 212241 and 212796 were both accepted without one (docs/myriad.md section 10).
# A paid/priority allocation, if one ever exists, goes on the qsub line; an SGE directive cannot
# read the environment, so it could not live here anyway.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# SGE runs a SPOOLED COPY of this script (/var/opt/sge/<node>/job_scripts/<jobid>), so BASH_SOURCE
# names the spool directory, not the checkout, and $HERE/_common.sh does not exist there - job
# 212796 died on that source line in under a second. `#$ -cwd` puts us in the submit directory,
# which is the repo root (RUNS_ROOT is already relative to it), so recover the checkout from there.
# HERE also feeds enter_container, which would otherwise re-exec a spool path the container has no
# bind mount for. Running the file in place (`bash scripts/myriad/pilot.sh`) keeps the first branch.
[[ -f "$HERE/_common.sh" ]] || HERE="$PWD/scripts/myriad"
[[ -f "$HERE/_common.sh" ]] || {
  echo "[pilot] cannot find scripts/myriad/_common.sh from the spool directory or \$PWD ($PWD)" >&2
  echo "[pilot] submit from the repo root: cd ~/Scratch/precept-research && qsub scripts/myriad/pilot.sh" >&2
  exit 1
}
REPO_ROOT="${REPO_ROOT:-$(cd "$HERE/../.." && pwd)}"

TIER="${TIER:-qwen14b}"                 # Hydra model group (configs/model/<tier>.yaml)
# Offset by the job id rather than fixed at 8000. Myriad nodes are shared, and a fixed port is a
# port someone else may already hold: jobs 244519/244520/244523 all landed on node-l00a-003 beside
# another tenant's vLLM serving Devstral-Small-2-24B, and each spent its queue wait to reach the
# readiness loop below in one second and then fail the driver's model check. The pre-flight before
# the launch covers the rest; this only keeps our own concurrent jobs off each other by
# construction. Override with -v PORT=<n>.
PORT="${PORT:-$((8000 + ${JOB_ID:-0} % 1000))}"
ATTEMPT="${ATTEMPT:-1}"                 # 1 = first pass, 2 = the one permitted retune
RUNS_ROOT="${RUNS_ROOT:-runs}"
SERVE_TIMEOUT="${SERVE_TIMEOUT:-1800}"  # generous: a cold HF cache downloads ~28 GB before serving
# uv's own default location, so `uv sync` populates exactly what these jobs activate - no
# UV_PROJECT_ENVIRONMENT, no second path to keep in step. Override with -v VENV=<path>.
VENV="${VENV:-$REPO_ROOT/.venv}"
# TIER is exported so serve.sh serves the tier this pilot drives. Left unexported they are set
# independently, and `-v TIER=qwen8b` alone would drive the 8B pilot against a 14B server.
export PORT VENV TIER REPO_ROOT

# shellcheck source=scripts/myriad/_common.sh
source "$HERE/_common.sh"

# The sidecar serve.sh writes; exported so the manifest picks up the server-side stack (vLLM and
# torch versions, the physical GPU) that the client process has no way to observe for itself.
# Resolved through serve_env_path so the job-scoped default lives in exactly one place, then
# exported as SERVE_ENV_PATH so serve.sh's own call to it returns this identical string - the two
# scripts previously each spelled out the default and could have drifted apart silently.
export SERVE_ENV_PATH="$(serve_env_path)"
export PRECEPTX_SERVE_ENV="$SERVE_ENV_PATH"

# Checked here rather than in serve.sh so a typo'd tier fails now, not after the model has loaded.
if [[ ! -f "$REPO_ROOT/configs/model/$TIER.yaml" ]]; then
  echo "[pilot] no configs/model/$TIER.yaml - available tiers:" >&2
  ls "$REPO_ROOT/configs/model/" >&2
  exit 1
fi

# Myriad is glibc 2.17 and the lock is manylinux_2_28 throughout (DSE-051). Entering here rather
# than in serve.sh alone is what keeps this a single job: serve.sh sees APPTAINER_CONTAINER already
# set and does not nest, so the server it launches stays in this process tree and the trap below
# still names vLLM. The image is asserted, not pulled - prefetch.sh owns that, on a login node.
require_image
enter_container "$HERE/pilot.sh" "$@"

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

# Claim the port before anything is launched. The readiness loop below polls an address and takes
# any 200 it gets, which cannot distinguish our server from a co-tenant's - that is precisely how
# 244519/244520/244523 reported "endpoint live after 1s" against a Devstral server they did not
# start. Proving the port is unheld first is what makes a later 200 on it ours. A listener that is
# bound but not yet answering slips past this, and is then caught loudly by the `kill -0` check in
# the loop when our own vLLM fails to bind and exits.
if curl -sf --max-time 5 "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
  echo "[pilot] :$PORT on $(hostname) is already serving - another job or tenant holds it." >&2
  echo "[pilot] this job would have measured THEIR model. Resubmit on a free port:" >&2
  echo "[pilot]   qsub -v PORT=<free port> scripts/myriad/pilot.sh" >&2
  exit 1
fi

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

# The driver's own health check verifies the endpoint serves the model that was configured, so a
# leftover job on this port serving another tier fails here rather than being recorded as this one.
#
# The default driver is preceptx-pilot with no grid flags: conditions/difficulties/seeds stay the
# CLI defaults, which are the pre-registered E3 cell (PREREGISTRATION §6), because naming them here
# would let the two drift apart silently. DRIVER selects a different entry point - preceptx-rq1
# writes the factorial analysis and NO G1/G2/G3 verdict, which is what a characterisation grid must
# use: the gate has no attempt 3, and a driver that emits a verdict cannot be run without spending
# one. Flags after the script name reach the driver ("$@" survives the container re-exec), so a
# non-default grid needs no second jobscript.
#
#   qsub -N precept-rq1 -l h_rt=8:00:00 -v DRIVER=preceptx-rq1 scripts/myriad/pilot.sh \
#     --conditions C0,C1,C2,C3,C4 --difficulties easy,medium,hard --seeds "$(seq -s, 0 31)"
#
# --attempt exists only on preceptx-pilot, so it is appended only for that driver. One array
# holding the whole argv, rather than a separate flags array: `"${empty[@]}"` under `set -u` is an
# unbound-variable error on bash 3.2 and fine on 4.4+, and a jobscript should not depend on which
# bash the node happens to have.
DRIVER="${DRIVER:-preceptx-pilot}"
driver_args=(--model "$TIER" --base-url "http://localhost:${PORT}/v1")
case "$DRIVER" in
  preceptx-pilot) driver_args+=(--root "$RUNS_ROOT" --attempt "$ATTEMPT") ;;
  # RQ3a reads a fetched corpus instead of writing episodes, so its --root is the CORPUS root and
  # RUNS_ROOT would silently point it at an empty tree. --judge is implied because it is the only
  # part of RQ3a that makes a model call: an rq3a run without it needs no GPU and belongs on a
  # login node, so a GPU job that omitted it would hold an A100 to run sklearn.
  preceptx-rq3a) driver_args+=(--root "${RQ3A_ROOT:?RQ3A_ROOT unset; -v RQ3A_ROOT=\$HOME/Scratch/rq3a}" --judge) ;;
  *) driver_args+=(--root "$RUNS_ROOT") ;;
esac

"$DRIVER" "${driver_args[@]}" "$@"

echo "[pilot] $(date -u +%FT%TZ) complete"
