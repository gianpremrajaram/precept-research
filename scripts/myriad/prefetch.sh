#!/bin/bash -l
# Login-node pre-pull: fetch everything a GPU job would otherwise download while holding a GPU.
#
# Run this BEFORE the first qsub. It is not a jobscript - no SGE directives, no GPU, run it
# directly on a login node:
#
#   bash scripts/myriad/prefetch.sh              # the 14B workhorse
#   TIER=qwen8b bash scripts/myriad/prefetch.sh  # the V100-class fallback tier
#
# It now also prepares the two things a GPU job cannot build for itself: the Apptainer image and
# the virtual environment inside it (DSE-051). Both are idempotent - a second run re-uses them.
#
# Two reasons this exists, in order of severity:
#
#  1. Compute nodes may have no outbound internet (docs/myriad.md §10 - unconfirmed, and the single
#     unknown that can write off a whole session). If they do not, a GPU job that downloads is not
#     slow, it is dead, and it dies after the queue wait. Pulling here moves that failure to a
#     login node where it costs nothing and is fixable.
#  2. Even with internet, ~28 GB of weights downloaded inside the job is GPU allocation spent on
#     network I/O.
#
# The embedding encoder matters as much as the weights and is easier to forget: it is downloaded at
# ANALYSIS time, after every episode has already been paid for.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$HERE/../.." && pwd)}"
TIER="${TIER:-qwen14b}"
# uv's own default location, so `uv sync` populates exactly what these jobs activate - no
# UV_PROJECT_ENVIRONMENT, no second path to keep in step. Override with -v VENV=<path>.
VENV="${VENV:-$REPO_ROOT/.venv}"

# shellcheck source=scripts/myriad/_common.sh
source "$HERE/_common.sh"

# Myriad's nodes are glibc 2.17 and every wheel in uv.lock is manylinux_2_28 or newer, so the
# environment can only be built inside the container (DSE-051; see _common.sh for the full note).
# The image pull and the venv build are login-node work for the same reason the weights are: doing
# either while holding an A100 spends GPU allocation on network I/O.
if [[ -z "${APPTAINER_CONTAINER:-}" ]]; then
  # Quota first, and on the host: `gquota` is a Myriad login-node command that does not exist inside
  # the image, and what it guards against - a download that fills the quota and leaves the *next*
  # job unable to write its own .o/.e files - is decided before anything is pulled.
  if command -v gquota >/dev/null 2>&1; then
    gquota
  else
    echo "[prefetch] gquota not found; check your quota headroom manually (~28 GB for the 14B tier)"
  fi
  ensure_image
  enter_container "$HERE/prefetch.sh" "$@"
fi
# ---- everything below runs inside the container ----
ensure_venv
cache_to_scratch

# shellcheck disable=SC1091
source "$VENV/bin/activate"
resolve_tier "$REPO_ROOT/configs/model/$TIER.yaml" prefetch

echo "[prefetch] HF_HOME=$HF_HOME"
echo "[prefetch] tier=$TIER model=$MODEL revision=$REVISION"

echo "[prefetch] pulling weights - tens of GB, and resumable if interrupted"
hf download "$MODEL" --revision "$REVISION"

# Pinned in PREREGISTRATION; EncoderConfig's defaults are the single source of the pin, so this
# cannot drift from what the analysis will actually load.
echo "[prefetch] pulling the embedding encoder"
python - <<'PYENC'
from preceptx.measure.featuriser import EncoderConfig, Featuriser

cfg = EncoderConfig()
print(f"[prefetch] encoder={cfg.name}@{cfg.revision}")
Featuriser(cfg).embed_texts(["warm"])
PYENC

echo "[prefetch] done. Next: the interactive smoke test in docs/myriad.md §7 step 3."
