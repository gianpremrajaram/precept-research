"""CLI: probe a served model's fixed-seed determinism and print a variance report.

Thin wrapper over ``preceptx.determinism.run_determinism_check`` (the typed, tested logic lives in
``src/`` so mypy/pytest cover it). Run against a live vLLM endpoint on Myriad:

    python scripts/determinism_check.py --model Qwen/Qwen3-14B-Instruct --k 20
"""

from __future__ import annotations

import argparse
import json
import logging

from preceptx.determinism import run_determinism_check
from preceptx.serving.client import ChatMessage, LLMClient, ServingConfig

# A small fixed structured request standing in for one handoff action until the runner lands.
_SCHEMA = {
    "type": "object",
    "properties": {
        "dx": {"type": "number"},
        "dy": {"type": "number"},
        "rotate": {"type": "number"},
    },
    "required": ["dx", "dy", "rotate"],
}
_MESSAGES = [
    ChatMessage(role="system", content="You output a single nudge action as JSON."),
    ChatMessage(role="user", content="Nudge the load slightly towards the goal at (1.0, 0.0)."),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=20)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    config = ServingConfig(model=args.model, base_url=args.base_url, seed=args.seed)
    with LLMClient(config) as client:
        report = run_determinism_check(client, _MESSAGES, _SCHEMA, k=args.k)
    print(json.dumps(report.model_dump(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
