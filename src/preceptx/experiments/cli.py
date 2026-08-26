"""Command-line entry points for the pilot and RQ1 drivers (DSE-031).

``run_grid``, ``run_pilot`` and ``run_rq1`` are library functions with no console entry, so nothing
downstream of a served model could be launched from a shell. This module is that entry point and
nothing more: parse flags, let Hydra compose the config tree, validate it into an
``ExperimentConfig``, build the sweep, construct the client(s), run, analyse, write the report.

Two conventions are load-bearing here. First, **Hydra composes and Pydantic validates**: the raw
``DictConfig`` never leaves ``_resolve_cell``. Second, **the entry point owns logging** - library
code attaches no handlers, so ``logging.basicConfig`` is called exactly once, here.

``--dry-run`` is the pre-flight: it prints the expanded cell count, the upper-bound model-call
count and the resolved hashes without constructing a client or issuing a single call.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from preceptx.config import ConfigError, ExperimentConfig, load_config
from preceptx.data.writer import load_records
from preceptx.experiments.pilot import run_pilot, write_pilot_report
from preceptx.experiments.rq1 import CONDITION_ORDER, run_rq1, write_rq1
from preceptx.experiments.runner import run_grid
from preceptx.experiments.sweep import SweepConfig, dataset_hash_for, expand, sweep_hash
from preceptx.measure.featuriser import EncoderConfig, Featuriser
from preceptx.serving.client import LLMClient, ServingConfig, ServingError

logger = logging.getLogger(__name__)

# The config tree ships beside the source, not inside the package: this repo is always run from an
# editable checkout (CLAUDE.md pins `uv pip install -e .`). --config-dir overrides it if that ever
# stops being true.
_DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"

# The pilot's documented E3 cell (docs/EXPERIMENTS.md): the clean channel, one mild degradation and
# the hardest one, crossed with the two ends of the difficulty axis, over at least three seeds.
# C3 is in the cell because it is the only condition with a real observation asymmetry and it is
# in the headline design: a pilot that never exercises it certifies an instrument the main sweep
# will not use (PREREGISTRATION §6). An earlier rationale here - "CPVI is near zero by construction
# without C3" - was falsified by E3-local, which measured +0.19 bits in C0 (design log, D20).
_PILOT_CONDITIONS = ["C0", "C1", "C3", "C4"]
_PILOT_DIFFICULTIES = ["easy", "hard"]
# Seeds 0-9, amended again for attempt 2 (PREREGISTRATION §6, 2026-08-26, before F0). At seeds 0-4
# G1 rested on five easy-C0 episodes and read 2/5: a Wilson 95% interval of [0.12, 0.77], and a
# design whose true rate is exactly the 0.5 threshold fails the gate half the time. Doubling to ten
# also doubles G2's success half, which passed attempt 1 by exactly zero margin (2/10 against 1/10 -
# one episode). This is a precision change, not a retune: it moves no threshold and no estimator,
# and because the attempt-1 point estimate sits BELOW the threshold, added n moves the expected
# verdict toward FAIL, which is the opposite direction from optional stopping.
_PILOT_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def _parser(name: str, description: str) -> argparse.ArgumentParser:
    """The flags both drivers share; each driver adds its own defaults for the grid axes."""
    p = argparse.ArgumentParser(prog=name, description=description)
    p.add_argument("--model", default="qwen14b", help="Hydra model group (configs/model/*.yaml)")
    p.add_argument("--conditions", help="comma-separated conditions, e.g. C0,C1,C4")
    p.add_argument("--serialisations", help="comma-separated serialisations")
    p.add_argument("--difficulties", help="comma-separated difficulties")
    p.add_argument("--seeds", help="comma-separated integer seeds")
    p.add_argument("--concurrency", type=int, default=4, help="parallel episodes")
    # The 60 s default suits a cluster endpoint. A local runtime usually serves one request at a
    # time, so concurrent episodes queue behind each other and a slow tier can exceed it - which
    # fails the episode loud, as it should, but for a queueing reason rather than a broken endpoint.
    p.add_argument("--timeout", type=float, default=60.0, help="per-request timeout in seconds")
    p.add_argument("--root", type=Path, default=Path("runs"), help="dataset root")
    p.add_argument("--out", type=Path, help="report directory (default: <root>/<hash>-report)")
    p.add_argument("--base-url", default=ServingConfig.model_fields["base_url"].default)
    p.add_argument("--base-url-b", help="second endpoint, when agent B is served separately")
    p.add_argument("--model-b", help="Hydra model group serving agent B (DSE-049)")
    p.add_argument(
        "--structured-mode",
        choices=["guided_json", "response_format"],
        default="guided_json",
        help="guided_json for vLLM; response_format for an OpenAI-compatible local runtime",
    )
    p.add_argument(
        "--thinking-switch",
        default="",
        help="in-band no-thinking token for runtimes that ignore chat_template_kwargs "
        "(LM Studio + Qwen3: /no_think)",
    )
    p.add_argument("--config-dir", type=Path, default=_DEFAULT_CONFIG_DIR)
    p.add_argument("--overrides", nargs="*", default=[], help="extra Hydra overrides (key=value)")
    p.add_argument("--dry-run", action="store_true", help="print the plan; issue no model calls")
    p.add_argument("--verbose", action="store_true", help="DEBUG-level logging")
    return p


def _resolve_cell(config_dir: Path, overrides: list[str]) -> ExperimentConfig:
    """Compose the Hydra config tree and validate it. The raw DictConfig goes no further."""
    if not config_dir.is_dir():
        raise ConfigError(f"config directory not found: {config_dir}")
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="experiment", overrides=overrides)
    return load_config(cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True)))


def _csv(value: str | None, default: list[Any]) -> list[Any]:
    return default if value is None else [v.strip() for v in value.split(",") if v.strip()]


def _build_sweep(args: argparse.Namespace, defaults: dict[str, list[Any]]) -> SweepConfig:
    """Resolve the model block through Hydra, then cross it with the grid axes from the flags."""
    cell = _resolve_cell(args.config_dir, [f"model={args.model}", *args.overrides])
    model_b = (
        None
        if args.model_b is None
        else _resolve_cell(args.config_dir, [f"model={args.model_b}", *args.overrides]).model
    )
    return SweepConfig(
        conditions=_csv(args.conditions, defaults["conditions"]),
        serialisations=_csv(args.serialisations, defaults["serialisations"]),
        difficulties=_csv(args.difficulties, defaults["difficulties"]),
        seeds=[int(s) for s in _csv(args.seeds, defaults["seeds"])],
        model=cell.model,
        model_b=model_b,
        concurrency=args.concurrency,
    )


def _print_plan(sweep: SweepConfig) -> None:
    """The --dry-run pre-flight: what would run, and what it would cost in model calls."""
    cells = expand(sweep)
    # Two calls per step (A's message, B's structured action) x the per-difficulty step budget.
    # An upper bound: episodes that reach the goal terminate early.
    calls = 2 * sum(sweep.max_steps[c.difficulty] for c in cells)
    s_hash = sweep_hash(sweep)
    print(f"cells:            {len(cells)}")
    print(f"model calls:      {calls} (upper bound; early success shortens episodes)")
    print(f"sweep hash:       {s_hash}")
    print(f"dataset hash:     {dataset_hash_for(sweep)}")
    print(f"model (A):        {sweep.model.name}@{sweep.model.revision}")
    if sweep.model_b is not None:
        print(f"model (B):        {sweep.model_b.name}@{sweep.model_b.revision}")


def _client(model: str, base_url: str, args: argparse.Namespace) -> LLMClient:
    """Build a client and prove the endpoint is live before any episode is attempted."""
    client = LLMClient(
        ServingConfig(
            model=model,
            base_url=base_url,
            structured_mode=args.structured_mode,
            thinking_switch=args.thinking_switch,
            timeout=args.timeout,
        )
    )
    if not client.health_check():
        raise ServingError(f"no healthy endpoint at {base_url} for model {model!r}")
    return client


@contextlib.contextmanager
def _clients(
    sweep: SweepConfig, args: argparse.Namespace
) -> Iterator[tuple[LLMClient, LLMClient | None]]:
    """The per-role clients (DSE-049). B's client exists only when B has its own model block."""
    with contextlib.ExitStack() as stack:
        client_a = stack.enter_context(_client(sweep.model.name, args.base_url, args))
        client_b = (
            None
            if sweep.model_b is None
            else stack.enter_context(
                _client(sweep.model_b.name, args.base_url_b or args.base_url, args)
            )
        )
        yield client_a, client_b


def _prepare(args: argparse.Namespace, defaults: dict[str, list[Any]]) -> SweepConfig:
    """Shared start-up: logging, config resolution, and the substrate label every dataset needs."""
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )
    sweep = _build_sweep(args, defaults)
    if not args.dry_run and not os.environ.get("PRECEPTX_SERVING_SUBSTRATE"):
        # Not a warning: interim local-pilot data and Myriad data must stay permanently
        # distinguishable, and an unlabelled dataset cannot be told apart after the fact.
        raise ConfigError(
            "PRECEPTX_SERVING_SUBSTRATE is unset; label the substrate (e.g. 'local-lmstudio' or "
            "'myriad-a100') so the manifest records where the episodes were served"
        )
    return sweep


def _report_dir(args: argparse.Namespace, d_hash: str) -> Path:
    return cast(Path, args.out) if args.out else cast(Path, args.root) / f"{d_hash}-report"


def pilot(argv: list[str] | None = None) -> int:
    """``preceptx-pilot``: run the E3 gate cells and write the G1/G2/G3 go/no-go report."""
    parser = _parser("preceptx-pilot", "Run the pilot gate cells and write the G1/G2/G3 report.")
    parser.add_argument("--attempt", type=int, default=1, help="1 = first pass, 2 = the one retune")
    parsed = parser.parse_args(argv)
    sweep = _prepare(
        parsed,
        {
            "conditions": _PILOT_CONDITIONS,
            "serialisations": ["numeric"],
            "difficulties": _PILOT_DIFFICULTIES,
            "seeds": _PILOT_SEEDS,
        },
    )
    if parsed.dry_run:
        _print_plan(sweep)
        return 0

    with _clients(sweep, parsed) as (client_a, client_b):
        run_grid(sweep, client_a, client_b, root=parsed.root)
    d_hash = dataset_hash_for(sweep)
    report = run_pilot(
        load_records(d_hash, root=parsed.root),
        Featuriser(EncoderConfig()),
        dataset_hash=d_hash,
        attempt=parsed.attempt,
    )
    out = write_pilot_report(report, _report_dir(parsed, d_hash))
    logger.info("pilot report written to %s (%s)", out, report.recommendation)
    return 0


def rq1(argv: list[str] | None = None) -> int:
    """``preceptx-rq1``: run the information-gradient factorial and write the analysis."""
    parser = _parser("preceptx-rq1", "Run the RQ1 information-gradient sweep and analyse it.")
    parsed = parser.parse_args(argv)
    sweep = _prepare(
        parsed,
        {
            "conditions": CONDITION_ORDER,
            "serialisations": ["numeric"],
            "difficulties": ["hard"],
            "seeds": [0, 1, 2, 3, 4],
        },
    )
    if parsed.dry_run:
        _print_plan(sweep)
        return 0

    with _clients(sweep, parsed) as (client_a, client_b):
        result, scores = run_rq1(
            sweep, client_a, Featuriser(EncoderConfig()), client_b=client_b, root=parsed.root
        )
    out = write_rq1(result, _report_dir(parsed, dataset_hash_for(sweep)), scores=scores)
    logger.info("rq1 analysis written to %s", out)
    return 0


if __name__ == "__main__":  # `python -m preceptx.experiments.cli` mirrors the console script
    sys.exit(pilot())
