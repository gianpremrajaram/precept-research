#!/usr/bin/env bash
# Fetch the three RQ3a corpora into a local root (DSE-041 / E9).
#
#   scripts/fetch_rq3a.sh ~/data/rq3a
#
# Plain HTTP against HuggingFace's `resolve` endpoints - all three datasets are public and
# ungated, so no token and no `datasets` dependency are needed. TraceElephant ships as one 569 MB
# zip that `datasets.load_dataset` cannot open at all, which is why fetching is a script rather
# than a library call. Roughly 800 MB on disk; keep it OUT of the repo.
set -euo pipefail

ROOT="${1:?usage: fetch_rq3a.sh <destination root>}"
HF="https://huggingface.co/datasets"

mkdir -p "$ROOT"/{traceelephant,who_and_when,mast}

echo "==> TraceElephant (569 MB)"
curl -fL --progress-bar -C - -o "$ROOT/traceelephant/data.zip" \
  "$HF/TraceElephant/TraceElephant/resolve/main/data.zip"
# The loader reads the unzipped tree at <root>/traceelephant/data/<family>/<task>/.
unzip -q -o "$ROOT/traceelephant/data.zip" -d "$ROOT/traceelephant"

echo "==> Who&When (2 MB)"
for f in Algorithm-Generated.parquet Hand-Crafted.parquet; do
  curl -fL --progress-bar -o "$ROOT/who_and_when/$f" \
    "$HF/Kevin355/Who_and_When/resolve/main/$f"
done

echo "==> MAST-Data (200 MB)"
curl -fL --progress-bar -o "$ROOT/mast/MAD_full_dataset.json" \
  "$HF/mcemri/MAST-Data/resolve/main/MAD_full_dataset.json"

echo
echo "Done. Counts:  uv run python -m preceptx.experiments.rq3a_load --root $ROOT"
