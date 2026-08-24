"""CLI: benchmark one served tier and rewrite the ladder comparison table (DSE-005).

Thin wrapper over ``preceptx.serving.benchmark`` (the typed, tested logic lives in ``src/``). Only
one tier is served at a time, so run this once per endpoint; every run appends a row and rewrites
the table and recommendation note from all rows collected so far.

    # Local pilot tier, LM Studio on :1234
    PRECEPTX_SERVING_SUBSTRATE=local-lmstudio uv run python scripts/benchmark_models.py \
        --tier 8b-4bit --model mlx-community/Qwen3-8B-4bit \
        --revision 545dc4251c05440727734bcd94334791f6ab0192 \
        --base-url http://localhost:1234/v1 --structured-mode response_format

    # Myriad workhorse, vLLM on :8000
    PRECEPTX_SERVING_SUBSTRATE=myriad-a100 uv run python scripts/benchmark_models.py \
        --tier 14b --model Qwen/Qwen3-14B --revision <sha>
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os

from preceptx.config import ConfigError, ModelConfig
from preceptx.serving.benchmark import (
    append_result,
    begin_invocation,
    benchmark_tier,
    write_invocation,
    write_report,
)
from preceptx.serving.client import LLMClient, ServingConfig, ServingError


def main() -> int:
    p = argparse.ArgumentParser(description="Benchmark one served tier into the ladder table.")
    p.add_argument("--tier", required=True, help="ladder tier label, e.g. 8b / 14b / 8b-4bit")
    p.add_argument("--model", required=True, help="the served model identifier")
    p.add_argument("--revision", required=True, help="pinned commit SHA of the served repository")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument(
        "--structured-mode", choices=["guided_json", "response_format"], default="guided_json"
    )
    p.add_argument("--thinking-switch", default="", help="e.g. /no_think for LM Studio + Qwen3")
    p.add_argument("--out", default="runs/bench", help="ladder directory (accumulates rows)")
    p.add_argument("--root", default="runs/bench/smoke", help="dataset root for the smoke episodes")
    p.add_argument("--episodes", type=int, default=10, help="C0 easy smoke episodes")
    p.add_argument("--schema-calls", type=int, default=20, help="schema-adherence sample size")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s"
    )
    substrate = os.environ.get("PRECEPTX_SERVING_SUBSTRATE")
    if not substrate:
        raise ConfigError(
            "PRECEPTX_SERVING_SUBSTRATE is unset; a ladder row that does not say where it was "
            "measured cannot be compared with one from another substrate"
        )

    # Provenance is written BEFORE anything is served. --model and --revision are free text, and
    # the served checkpoint is precisely what no later check can recover: /v1/models carries no
    # revision, so a row recorded against the wrong one is undetectable afterwards. Writing first
    # makes persistence a precondition of running rather than a courtesy after the fact.
    invocation = begin_invocation(
        tier=args.tier,
        model=args.model,
        revision=args.revision,
        substrate=substrate,
        args={k: str(v) for k, v in sorted(vars(args).items())},
    )
    record = write_invocation(invocation, args.out)
    print(f"invocation recorded at {record}")

    client = LLMClient(
        ServingConfig(
            model=args.model,
            base_url=args.base_url,
            structured_mode=args.structured_mode,
            thinking_switch=args.thinking_switch,
        )
    )
    if not client.health_check():
        raise ServingError(f"no healthy endpoint at {args.base_url} for model {args.model!r}")

    with client:
        result = benchmark_tier(
            client,
            ModelConfig(name=args.model, revision=args.revision, tier=args.tier),
            substrate=substrate,
            root=args.root,
            n_smoke_episodes=args.episodes,
            n_schema_calls=args.schema_calls,
        )
    out = write_report(append_result(result, args.out), args.out)
    # Finalise the same record. A crashed run needs no handler: it simply keeps the exit_status of
    # None it was written with, which reads as "started, never finished" - accurate, and one fewer
    # broad except clause than recording the failure explicitly would cost.
    write_invocation(
        invocation.model_copy(
            update={
                "ended_at": dt.datetime.now(dt.UTC).isoformat(),
                "exit_status": 0,
                "artefacts": [str(out), str(args.root)],
            }
        ),
        args.out,
    )
    print(f"ladder table written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
