#!/bin/bash -l
# Bring a run home from Myriad: the irreproducible artefacts first, the regenerable ones on request.
#
# This exists because the 29 Aug results bundle was assembled by hand and took only
# runs/<hash>/*.parquet, leaving both v9 arms' manifests on the cluster. That is the one file no
# amount of local work can rebuild - `preceptx-analyse` reconstructs the analysis, but the model
# revision, the exact command and the serving environment live only in the manifest the sweep wrote.
# A documented rsync that must be retyped is exactly what got dropped, so it is a script now.
#
#   scripts/myriad/fetch.sh                    # every run's manifests -> runs/myriad/
#   scripts/myriad/fetch.sh 86ecbbdf35322dc3   # that run's manifests AND its Parquet, ready to analyse
#
# HOST defaults to the `myriad` ssh alias from docs/myriad.md section 2; override for a bare hostname.
set -euo pipefail

HOST="${HOST:-myriad}"
REMOTE="${REMOTE:-\$HOME/Scratch/precept-research/runs/}"   # expanded on the far side, not here
HASH="${1:-}"

# The four irreproducible files. `serve_env.json` sits at the runs/ root, the rest under <hash>-run/.
args=(--include='*/' --include='manifest.json' --include='summary.json' --include='serve_env.json')

if [[ -n "$HASH" ]]; then
  # Parquet is regenerable only by re-running the sweep on a GPU, so for a named run it comes too.
  # Restricting to the one hash keeps a 96-part dataset from arriving whenever anyone fetches.
  args+=(--include="${HASH}/***")
  echo "[fetch] $HASH: manifests + Parquet"
else
  echo "[fetch] manifests only (pass a dataset hash to include its Parquet)"
fi
args+=(--exclude='*')

mkdir -p runs/myriad
rsync -av --prune-empty-dirs "${args[@]}" "${HOST}:${REMOTE}" runs/myriad/

echo
echo "[fetch] manifests under runs/myriad/<hash>-run/manifest.json"
if [[ -n "$HASH" ]]; then
  # analyse reads runs/<hash>/, not runs/myriad/<hash>/, so put the dataset where it looks.
  mkdir -p "runs/$HASH"
  cp -f "runs/myriad/$HASH"/*.parquet "runs/$HASH/" 2>/dev/null || true
  echo "[fetch] dataset staged at runs/$HASH - analyse with:"
  echo "          uv run preceptx-analyse --dataset-hash $HASH"
fi
