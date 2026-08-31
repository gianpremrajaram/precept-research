# shellcheck shell=bash
# Shared helpers, sourced (never executed) by serve.sh, pilot.sh and prefetch.sh.
#
# configs/model/<TIER>.yaml is the same file the run manifest records `name` and `revision` from,
# so deriving them here means a job cannot serve one checkpoint while the manifest claims another.
# That mismatch has no other detector: the client's health check compares the served model id, but
# /v1/models carries no revision at all, so a wrong one would survive every check in the repo.
#
# Sets MODEL and REVISION. MODEL/REVISION already in the environment win - the 70B-AWQ tier has no
# config file until DSE-005 pins its repo id - but an override contradicting the config is
# announced, because a deliberate override and a typo look identical in a job log otherwise.
#
# Usage: resolve_tier <path-to-tier-yaml> <log-tag>
#        cache_to_scratch

resolve_tier() {
  local tier_file="$1" tag="$2" cfg_model cfg_revision var want

  if [[ ! -f "$tier_file" ]]; then
    # An unpinned revision is refused rather than defaulted: it would put a moving target under a
    # manifest that claims to pin everything (the featuriser refuses the same).
    MODEL="${MODEL:?no $tier_file; set MODEL and REVISION explicitly}"
    REVISION="${REVISION:?no $tier_file; set MODEL and REVISION explicitly}"
    return 0
  fi

  read -r cfg_model cfg_revision < <(python - "$tier_file" <<'PYCFG'
import sys

import yaml

model = yaml.safe_load(open(sys.argv[1]))["model"]
print(model["name"], model["revision"])
PYCFG
  )

  for var in MODEL REVISION; do
    if [[ "$var" == "MODEL" ]]; then want="$cfg_model"; else want="$cfg_revision"; fi
    if [[ -n "${!var:-}" && "${!var}" != "$want" ]]; then
      echo "[$tag] WARNING: $var override '${!var}' differs from $tier_file ('$want')." >&2
      echo "[$tag] WARNING: the manifest records the config value. Only do this deliberately." >&2
    fi
  done

  MODEL="${MODEL:-$cfg_model}"
  REVISION="${REVISION:-$cfg_revision}"
}


# Every cache these jobs touch defaults into $HOME and counts against the same 1 TB quota; the 14B
# weights alone are ~28 GB. A full quota does not fail cleanly - the job dies creating its .o/.e
# files, which reads as a scheduler fault rather than an out-of-space one. prefetch.sh and serve.sh
# must agree on these paths exactly, or the prefetch populates a cache the server never reads.
cache_to_scratch() {
  export HF_HOME="${HF_HOME:-$HOME/Scratch/hf-home}"
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/Scratch/cache}"
  export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$XDG_CACHE_HOME/vllm}"
  export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$XDG_CACHE_HOME/triton}"
  mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR"
}

# Serving-environment sidecar. vLLM's and torch's versions and the physical GPU exist only in the
# server process on the compute node; the client that writes the run manifest cannot import them or
# see the card. Echoing them to the job log made the run of record's server-side stack recoverable
# only by a human copying lines out of precept-pilot.o<jobid>. This writes them where
# manifest.serve_env() can read them, via PRECEPTX_SERVE_ENV.
#
# `head -1 | awk ...` is deliberately avoided in the fields below. Under `set -o pipefail` head
# closes the pipe on a multi-line producer, the producer dies of SIGPIPE, and the pipeline reports
# 141 even though awk already printed the right answer - so `|| echo unknown` fires as well and the
# field ends up two lines long. `awk NR==1` reads all of stdin and cannot lose that race (DSE-052).
#
# Usage: write_serve_env <path>   (serve.sh writes it; pilot.sh exports the same path)
serve_env_path() {
  # Job-scoped by default. Every job used to write ONE shared runs/serve_env.json on a filesystem
  # every node mounts: job 244523 truncated it on node-l00a-003 at 23:21Z and was killed inside the
  # write, and job 244522 - mid-sweep on node-u00a-001 - read the zero bytes 21 minutes later, after
  # all 20 parquet parts were written, and died in build_sweep_manifest with the episodes paid for.
  # $JOB_ID is SGE's; the $$ fallback keeps a local run off a concurrent shell's file.
  echo "${SERVE_ENV_PATH:-$REPO_ROOT/runs/serve_env.${JOB_ID:-local-$$}.json}"
}

write_serve_env() {
  local out="$1"
  mkdir -p "$(dirname "$out")"
  # Staged and renamed, never written in place. `cat >"$out"` truncates at redirection setup, and
  # the heredoc's `import vllm` / `import torch` substitutions then hold the file at zero bytes for
  # several seconds; a reader in that window - or a kill inside it, which is what happened - sees an
  # empty capture. rename(2) within one directory is atomic, so a reader gets either the previous
  # capture or the complete new one, never a half.
  local tmp="$out.$$.tmp"
  cat >"$tmp" <<JSON
{
  "tier": "$TIER",
  "model": "$MODEL",
  "revision": "$REVISION",
  "vllm": "$(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo unknown)",
  "torch": "$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo unknown)",
  "gpu": "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | paste -sd, - || echo unknown)",
  "driver": "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | awk 'NR==1{print}' || echo unknown)",
  "host": "$(hostname)",
  "container_source": "$CONTAINER_SOURCE",
  "container_sif": "$SIF",
  "container_sif_sha256": "$(cat "$SIF.sha256" 2>/dev/null || echo unknown)",
  "structured_outputs_backend": "${GUIDED_BACKEND:-unknown}",
  "cc": "$(command -v "${CC:-cc}" 2>/dev/null || echo unknown)",
  "glibc": "$(ldd --version 2>/dev/null | awk 'NR==1{print $NF}' || echo unknown)",
  "job_id": "${JOB_ID:-}",
  "captured_at": "$(date -u +%FT%TZ)"
}
JSON
  mv -f "$tmp" "$out"
}


# --- Container runtime (DSE-051) ---------------------------------------------------------------
# Myriad's login AND compute nodes are both RHEL 7.9 / glibc 2.17 (verified 25 Aug 2026 on login12
# and node-l00a-006). Every wheel this project locks is manylinux_2_28 or newer - not just torch and
# vLLM but pandas, pyarrow, scipy and scikit-learn - and torch publishes no sdist, so `uv sync`
# cannot build a working environment on the bare node at all.
#
# The fix is to run inside a container, not to move the lock backwards. Downgrading to the last
# glibc-2.17-compatible pair (vLLM 0.8.5 / torch 2.6.0, April 2025) would also drag scipy and
# scikit-learn - the estimator's own numerical stack - back with it, making every measurement taken
# before the change incomparable with every one taken after. The container changes zero packages:
# uv.lock stays the reproducibility anchor and simply executes where its wheels are valid.
#
# The image is pinned by digest, not tag: `python:3.11` is mutable and a verdict-of-record run
# cannot rest on whatever Docker Hub served that day. Debian bookworm carries glibc 2.36, which
# clears manylinux_2_28 and vLLM's manylinux_2_31. The full image is used rather than -slim because
# manifest.git_sha() shells out to `git` and raises ManifestError when it is missing, so a run in a
# git-less image would fail after the episodes were already paid for.
CONTAINER_SOURCE="${CONTAINER_SOURCE:-docker://python@sha256:a8677eb08a56d04e75df938f9d2af3d50c0f0fba17af8eb9c8e41b65fa32938d}"
SIF="${SIF:-$HOME/Scratch/containers/precept-python311.sif}"
UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"

# UCL's apptainer module points its build directory at /run/user/<uid>, a small RAM-backed tmpfs on
# the login nodes; a ~1 GB image pull through it can fail on space. Scratch is the only location
# sized for it, and the pull is the one operation large enough for this to matter.
container_paths() {
  export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$HOME/Scratch/.apptainer}"
  export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$APPTAINER_CACHEDIR/tmp}"
  mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$(dirname "$SIF")"
}

load_apptainer() {
  if ! command -v apptainer >/dev/null 2>&1; then
    module load "${APPTAINER_MODULE:-apptainer/1.2.4-1}"
  fi
}

# The digest of the file actually executed, recorded beside the source reference: the source pins
# what was asked for, this pins what is on disk. Computed once and cached - it is a ~1 GB hash.
record_sif_digest() {
  if [[ ! -f "$SIF.sha256" ]]; then
    sha256sum "$SIF" | awk '{print $1}' >"$SIF.sha256"
  fi
}

# Login-node side of prefetch.sh: pull the image if it is not already there.
ensure_image() {
  container_paths
  load_apptainer
  if [[ -f "$SIF" ]]; then
    echo "[image] present: $SIF"
  else
    echo "[image] pulling $CONTAINER_SOURCE"
    echo "[image] -> $SIF (~1 GB, once; login node only, needs outbound network)"
    apptainer pull "$SIF" "$CONTAINER_SOURCE"
  fi
  record_sif_digest
  echo "[image] sha256=$(cat "$SIF.sha256")"
}

# serve.sh / pilot.sh side: assert, never pull. Pulling an image while holding an A100 would spend
# GPU allocation on network I/O, which is the same mistake prefetch.sh exists to prevent.
require_image() {
  # A host-side precondition only: inside the container there is no module system and the image is
  # by definition already there. Without this guard pilot.sh's nested serve.sh would try to
  # `module load apptainer` from inside the container and die.
  if [[ -n "${APPTAINER_CONTAINER:-}" ]]; then
    return 0
  fi
  container_paths
  if [[ ! -f "$SIF" ]]; then
    echo "[container] no image at $SIF" >&2
    echo "[container] run 'bash scripts/myriad/prefetch.sh' on a login node first" >&2
    exit 1
  fi
  load_apptainer
  record_sif_digest
}

# Myriad's login shells load default-modules/2018, which pulls in compilers/intel/2018/update3 and
# exports CC=icc / CXX=icpc. Apptainer passes the host environment straight through, so those follow
# us into an image that has no Intel compiler - and torch/Triton JIT-compile a small CUDA support
# module at engine start, reading CC from the environment with no existence check
# (triton/runtime/build.py). The result was `FileNotFoundError: 'icc'` minutes into vLLM startup,
# with the A100 already allocated (DSE-053).
#
# Overridden UNCONDITIONALLY. `${CC:-gcc}` would be worse than useless here: the leaked value is
# already set, so the default never fires and the broken state is preserved exactly.
#
# LD_LIBRARY_PATH is deliberately NOT touched - `apptainer --nv` manages it to expose the host
# driver libraries, and clearing it would break CUDA to fix a compiler. PYTHONPATH/PYTHONHOME are
# cleared because the venv is self-contained and nothing in this repo sets them, so a leaked value
# can only point at host site-packages built against glibc 2.17.
container_toolchain() {
  export CC=gcc CXX=g++
  unset PYTHONPATH PYTHONHOME
  local tool
  for tool in gcc g++; do
    if ! command -v "$tool" >/dev/null 2>&1; then
      echo "[container] the image has no $tool; torch/Triton JIT needs a C compiler at runtime" >&2
      echo "[container] CONTAINER_SOURCE must be a full python image, never -slim" >&2
      exit 1
    fi
  done
}

# Re-exec THIS script inside the container, once.
#
# Deliberately a re-exec rather than wrapping each command in `apptainer exec`. pilot.sh launches
# serve.sh in the background, captures $! and traps it so the server dies on the scheduler's
# wallclock SIGTERM. Two separate `apptainer exec` invocations would put them in different process
# namespaces and $! would name the wrapper rather than vLLM, so the guarantee that no orphaned
# server outlives the job would quietly stop holding. Re-execing keeps serve and drive in one
# process tree in one container, and every existing line downstream runs unchanged.
#
# $HOME is a symlink into /myriadfs here, so it is bound resolved-source-to-original-destination:
# that makes $HOME, $HOME/Scratch, the repo, the caches and the uv binary all resolve inside the
# container exactly as they do outside. --pwd preserves the working directory for the same reason -
# `#$ -cwd` plus a relative RUNS_ROOT means a dropped cwd would send artefacts to a read-only /runs.
enter_container() {
  if [[ -n "${APPTAINER_CONTAINER:-}" ]]; then
    container_toolchain
    return 0
  fi
  local script="$1"
  shift
  local bind
  bind="$(readlink -f "$HOME"):$HOME"
  # No GPU on a login node, where --nv fails looking for driver libraries that are not there.
  # The sentinel is a variable purely so tests/unit/scripts can exercise both branches; nothing
  # outside the test suite should ever set it.
  if [[ -e "${NV_SENTINEL:-/dev/nvidiactl}" ]]; then
    exec apptainer exec --nv --bind "$bind" --pwd "$PWD" "$SIF" bash "$script" "$@"
  fi
  exec apptainer exec --bind "$bind" --pwd "$PWD" "$SIF" bash "$script" "$@"
}

# Built with the image's own interpreter and only ever from inside the container. `uv sync` rather
# than `uv pip install`, so uv.lock stays the anchor. The import check is the assertion that the
# environment is the container's: these four are exactly the wheels that cannot exist at glibc 2.17,
# so importing them proves the container is real. Checking platform.libc_ver() instead would report
# the interpreter's own build-time libc and mislead.
ensure_venv() {
  if [[ -x "$VENV/bin/python" ]] && ! "$VENV/bin/python" -c 'import pyarrow' >/dev/null 2>&1; then
    echo "[venv] $VENV cannot import the locked wheels (built outside the container?); rebuilding"
    rm -rf "$VENV"
  fi
  echo "[venv] uv sync --extra serving --extra embed"
  "$UV_BIN" sync --extra serving --extra embed --python /usr/local/bin/python3.11
  "$VENV/bin/python" - <<'PYVENV'
import pandas, pyarrow, sklearn, torch

print(f"[venv] ok torch={torch.__version__} pyarrow={pyarrow.__version__} "
      f"pandas={pandas.__version__} sklearn={sklearn.__version__}")
PYVENV
}
