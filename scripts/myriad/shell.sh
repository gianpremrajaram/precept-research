#!/bin/bash -l
# Interactive (or one-shot) shell inside the DSE-051 container, with the venv already active.
#
# serve.sh, pilot.sh and prefetch.sh enter the container for themselves. This is for everything
# else: the dry-run hash checks, the two-episode smoke, poking at a failed run. Without it the
# alternative is hand-typing a 150-character `apptainer exec --nv --bind ... --pwd ...` line on a
# cluster where a mistyped one costs a queue wait.
#
#   bash scripts/myriad/shell.sh                          # interactive
#   bash scripts/myriad/shell.sh -c 'preceptx-pilot ...'  # one-shot
#
# --nv is applied only where there is a GPU, so this works unchanged on a login node.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$HERE/../.." && pwd)}"
VENV="${VENV:-$REPO_ROOT/.venv}"

# shellcheck source=scripts/myriad/_common.sh
source "$HERE/_common.sh"

require_image
enter_container "$HERE/shell.sh" "$@"

# ---- inside the container ----
if [[ ! -f "$VENV/bin/activate" ]]; then
  echo "[shell] no venv at $VENV - run 'bash scripts/myriad/prefetch.sh' on a login node first" >&2
  exit 1
fi
cache_to_scratch
# shellcheck disable=SC1091
source "$VENV/bin/activate"
exec bash "$@"
