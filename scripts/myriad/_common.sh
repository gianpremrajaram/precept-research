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
