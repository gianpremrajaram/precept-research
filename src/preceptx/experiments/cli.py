"""Command-line entry points for the pilot, RQ1 and RQ2 drivers (DSE-031).

``run_grid``, ``run_pilot`` and ``run_rq1`` are library functions with no console entry, so nothing
downstream of a served model could be launched from a shell. This module is that entry point and
nothing more: parse flags, let Hydra compose the config tree, validate it into an
``ExperimentConfig``, build the sweep, construct the client(s), run, analyse, write the report.

Two conventions are load-bearing here. First, **Hydra composes and Pydantic validates**: the raw
``DictConfig`` never leaves ``_resolve_cell``. Second, **the entry point owns logging** - library
code attaches no handlers, so ``logging.basicConfig`` is called exactly once, here.

``preceptx-rq2`` is the odd one out: it analyses a dataset that already exists, so it takes no
model flags, resolves no Hydra tree, and needs no live endpoint or serving-substrate label.

``preceptx-rq3a`` sits between the two. It reads a fetched corpus off disk like ``preceptx-rq2``,
but its three judge replications cost model calls, so the endpoint, the Hydra model block and the
substrate label are demanded only under ``--judge``. Without that flag the whole driver is offline
and the judge rows come back ``unavailable`` with their reason, which is a supported mode rather
than a degraded one (DSE-064).

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

from preceptx.agents.channel import ChannelConfig
from preceptx.config import ConfigError, ExperimentConfig, load_config
from preceptx.data.writer import load_records
from preceptx.experiments.pilot import run_pilot, write_pilot_report
from preceptx.experiments.rq1 import CONDITION_ORDER, analyse_rq1, run_rq1, write_rq1
from preceptx.experiments.rq2 import analyse_rq2, write_rq2
from preceptx.experiments.rq3a import RQ3aConfig, write_rq3a
from preceptx.experiments.rq3a_load import count_handoff_corpus
from preceptx.experiments.rq3a_run import (
    VLLMJudge,
    load_corpus,
    projected_judge_calls,
    run_rq3a,
    write_rq3a_manifest,
)
from preceptx.experiments.rq3b import GATE_MODES, rq3b_sweeps, run_rq3b, write_rq3b
from preceptx.experiments.runner import run_grid
from preceptx.experiments.sweep import SweepConfig, dataset_hash_for, expand, sweep_hash
from preceptx.gate.calibration import (
    CalibrationConfig,
    CalibrationReport,
    calibrate,
    fit_statistics,
    write_report,
)
from preceptx.gate.statistics import resolve_statistic_key, save_statistic
from preceptx.measure.featuriser import EncoderConfig, Featuriser, second_encoder_config
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
    # ChannelConfig was unreachable from the shell until now, so every dataset recorded to date
    # ran at its defaults (8 / 2 / 0.4). `channel` is inside sweep_hash, so a changed parameter
    # keys its own dataset and cannot append into the run it is a control for.
    p.add_argument(
        "--c1-max-tokens",
        type=int,
        help="C1 whitespace-token cap (default 8). A coherent length cap, not word-level noise: "
        "this is the knob the post-hoc length control for C4 turns",
    )
    p.add_argument(
        "--c3-window-rows",
        type=int,
        help="C3 grid rows kept either side of the load (default 2)",
    )
    p.add_argument(
        "--c4-dropout",
        type=float,
        help="C4 per-token drop probability (default 0.4)",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        help="broadcast one step budget to every difficulty, overriding the certified "
        "feasibility budgets (sim/feasibility.py STEP_BUDGETS). The certificate bounds an "
        "OPTIMAL policy; agents need slack to recover from their own detours",
    )
    p.add_argument(
        "--thinking",
        action="store_true",
        help="enable Qwen3 thinking mode and raise max_tokens to fit the trace",
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
    # Only the flags actually given are passed, so an untouched axis keeps ChannelConfig's default
    # and hashes byte-identically to every dataset recorded before these flags existed.
    channel = {
        k: v
        for k, v in (
            ("c1_max_tokens", args.c1_max_tokens),
            ("c3_window_rows", args.c3_window_rows),
            ("c4_dropout", args.c4_dropout),
        )
        if v is not None
    }
    return SweepConfig(
        conditions=_csv(args.conditions, defaults["conditions"]),
        serialisations=_csv(args.serialisations, defaults["serialisations"]),
        difficulties=_csv(args.difficulties, defaults["difficulties"]),
        seeds=[int(s) for s in _csv(args.seeds, defaults["seeds"])],
        model=cell.model,
        model_b=model_b,
        concurrency=args.concurrency,
        thinking=args.thinking,
        **({} if not channel else {"channel": ChannelConfig(**channel)}),
        # Omitted rather than passed as None: the field's default_factory holds the certified
        # budgets, and None would fail the validator instead of falling back to them.
        **({} if args.max_steps is None else {"max_steps": args.max_steps}),
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
    if sweep.channel != ChannelConfig():
        print(f"channel:          {sweep.channel.model_dump()} (NON-DEFAULT)")


def _client(model: str, base_url: str, args: argparse.Namespace) -> LLMClient:
    """Build a client and prove the endpoint is live before any episode is attempted."""
    # `getattr`: --thinking is a sweep-parser flag, and rq3a builds its own namespace for the judge.
    thinking = getattr(args, "thinking", False)
    client = LLMClient(
        ServingConfig(
            model=model,
            base_url=base_url,
            structured_mode=args.structured_mode,
            thinking_switch=args.thinking_switch,
            timeout=args.timeout,
            chat_template_kwargs={"enable_thinking": thinking},
            # 512 truncates a Qwen3 thinking trace mid-token and the structured action never
            # arrives; 2048 fits the traces seen in the pilot with headroom.
            **({"max_tokens": 2048} if thinking else {}),
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
    parser.add_argument(
        "--no-analysis",
        action="store_true",
        help="run the episodes and stop; analyse later with preceptx-analyse (frees the GPU)",
    )
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

    d_hash = dataset_hash_for(sweep)
    if parsed.no_analysis:
        # The analysis is pure CPU on episodes already written, but run in-job it holds the GPU for
        # its whole duration - 2h37m of an A100 on job 227886, spent in statsmodels. It is also the
        # only part that can fail *after* the episodes are paid for. Splitting it frees the node at
        # the last episode and makes the analysis re-runnable without re-running the sweep
        # (DSE-062).
        with _clients(sweep, parsed) as (client_a, client_b):
            run_grid(sweep, client_a, client_b, root=parsed.root)
        logger.info("episodes complete; analyse with: preceptx-analyse --dataset-hash %s", d_hash)
        return 0

    with _clients(sweep, parsed) as (client_a, client_b):
        result, scores = run_rq1(
            sweep, client_a, Featuriser(EncoderConfig()), client_b=client_b, root=parsed.root
        )
    out = write_rq1(result, _report_dir(parsed, d_hash), scores=scores)
    logger.info("rq1 analysis written to %s", out)
    return 0


def rq3b(argv: list[str] | None = None) -> int:
    """``preceptx-rq3b``: run the four H6 gate arms over one grid and test the causal claim.

    The calibration is **imported, never fitted here**. Its threshold was chosen against realised
    failure on a different dataset (the R5 circularity guard), and re-deriving it from the arms'
    own outcomes would let the treatment pick the operating point it is about to be judged at.
    """
    parser = _parser("preceptx-rq3b", "Run the RQ3b causal-gate arms (H6) and analyse them.")
    parser.add_argument(
        "--calibration",
        type=Path,
        required=True,
        help="calibration.json from preceptx-pilot / gate.calibration.write_report",
    )
    parser.add_argument(
        "--calibration-dataset",
        required=True,
        help="dataset hash the statistic re-fits on - the calibration set, not an arm's episodes",
    )
    parser.add_argument("--statistic", default="cosine", help="gate statistic key (DSE-061)")
    parser.add_argument("--max-retries", type=int, default=1, help="re-prompts per blocked handoff")
    parser.add_argument("--random-rate", type=float, default=0.2, help="random-trigger block rate")
    parser.add_argument("--gate-seed", type=int, default=0, help="salt for the control draws")
    parsed = parser.parse_args(argv)
    sweep = _prepare(
        parsed,
        {
            "conditions": ["C0", "C4"],
            "serialisations": ["numeric"],
            "difficulties": ["easy"],
            "seeds": [0, 1, 2, 3, 4],
        },
    )
    if parsed.dry_run:
        # Four arms over the same grid, so the cost is four times what the plan prints - stated
        # rather than silently folded in, because the printed hash is one arm's, not the run's.
        _print_plan(sweep)
        print(f"arms:             {len(GATE_MODES)} (x the cost above)")
        for mode, arm in rq3b_sweeps(
            sweep,
            statistic_key=parsed.statistic,
            max_retries=parsed.max_retries,
            random_rate=parsed.random_rate,
            gate_seed=parsed.gate_seed,
        ).items():
            print(f"  {mode:<16}{dataset_hash_for(arm)}")
        return 0

    report = CalibrationReport.model_validate_json(parsed.calibration.read_text())
    featuriser = Featuriser(EncoderConfig())
    with _clients(sweep, parsed) as (client_a, _client_b):
        result = run_rq3b(
            sweep,
            client_a,
            root=parsed.root,
            report=report,
            calibration_records=load_records(parsed.calibration_dataset, root=parsed.root),
            featuriser=featuriser,
            statistic_key=parsed.statistic,
            max_retries=parsed.max_retries,
            random_rate=parsed.random_rate,
            gate_seed=parsed.gate_seed,
        )
    out = write_rq3b(result, _report_dir(parsed, f"{dataset_hash_for(sweep)}-rq3b"))
    logger.info("rq3b analysis written to %s: %s", out, result.verdict)
    return 0


def calibrate_cmd(argv: list[str] | None = None) -> int:
    """``preceptx-calibrate``: fit and persist the gate statistics from an existing dataset.

    Offline, no GPU, shaped like ``preceptx-analyse``. It is the only producer of the two artefacts
    the rest of the stack consumes and neither of which anything wrote before: ``calibration.json``
    (thresholds and orientations, which ``preceptx-rq3b`` imports rather than re-deriving) and the
    per-statistic joblib (which the RQ3a transfer regime loads and applies to log corpora).

    The target is realised episode failure, never CPVI - the R5 circularity guard lives in
    ``gate.calibration``; this entry point only chooses which dataset supplies the outcomes.
    """
    parser = argparse.ArgumentParser(
        prog="preceptx-calibrate",
        description="Calibrate and persist the runtime gate statistics (offline).",
    )
    parser.add_argument("--dataset-hash", required=True, help="dataset supplying the outcomes")
    parser.add_argument("--root", type=Path, default=Path("runs"), help="dataset root")
    parser.add_argument("--out", type=Path, help="output dir (default: <root>/<hash>-calibration)")
    parser.add_argument(
        "--firing-rate-budget",
        type=float,
        default=CalibrationConfig.model_fields["firing_rate_budget"].default,
        help="max fraction of handoffs the chosen threshold may block",
    )
    parser.add_argument("--verbose", action="store_true", help="DEBUG-level logging")
    parsed = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if parsed.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )

    d_hash = cast(str, parsed.dataset_hash)
    cfg = CalibrationConfig(firing_rate_budget=parsed.firing_rate_budget)
    records = load_records(d_hash, root=parsed.root)
    featuriser = Featuriser(EncoderConfig())
    report = calibrate(records, featuriser, dataset_hash=d_hash, cfg=cfg)

    out = cast(Path, parsed.out) if parsed.out else Path(parsed.root) / f"{d_hash}-calibration"
    write_report(report, out)
    for stat in fit_statistics(records, featuriser, cfg=cfg):
        save_statistic(stat, encoder=featuriser.cfg, train_dataset_hash=d_hash, dir=out)
    logger.info(
        "calibration written to %s (n=%d, keys=%s)",
        out,
        report.n,
        ", ".join(f"{s.key} auroc={s.auroc}" for s in report.statistics),
    )
    return 0


def analyse(argv: list[str] | None = None) -> int:
    """``preceptx-analyse``: run the RQ1 analysis over an existing dataset. Offline, no GPU.

    The other half of ``preceptx-rq1 --no-analysis``. Shaped like ``preceptx-rq2`` and for the same
    reason: it reads episodes from disk, so resolving a model through Hydra or demanding a serving
    substrate would be ceremony for a computation that makes no model calls.
    """
    parser = argparse.ArgumentParser(
        prog="preceptx-analyse", description="Analyse an existing RQ1 dataset (offline)."
    )
    parser.add_argument("--dataset-hash", required=True, help="the RQ1 run's dataset hash")
    parser.add_argument("--root", type=Path, default=Path("runs"), help="dataset root")
    parser.add_argument("--out", type=Path, help="report directory (default: <root>/<hash>-report)")
    parser.add_argument("--verbose", action="store_true", help="DEBUG-level logging")
    parsed = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if parsed.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )
    d_hash = cast(str, parsed.dataset_hash)
    result, scores = analyse_rq1(
        load_records(d_hash, root=parsed.root), Featuriser(EncoderConfig()), dataset_hash=d_hash
    )
    out = write_rq1(result, parsed.out or Path(parsed.root) / f"{d_hash}-report", scores=scores)
    logger.info("rq1 analysis written to %s", out)
    return 0


def rq2(argv: list[str] | None = None) -> int:
    """``preceptx-rq2``: analyse an existing RQ1 dataset. Offline - no endpoint, no model calls.

    Deliberately not built on ``_parser``/``_prepare``: those resolve a model through Hydra and
    refuse to run without ``PRECEPTX_SERVING_SUBSTRATE``, both of which are meaningless for an
    analysis that reads episodes already on disk. The dataset hash the RQ1 driver printed is the
    only handle needed.
    """
    parser = argparse.ArgumentParser(
        prog="preceptx-rq2", description="Analyse an RQ1 dataset for H3, H4 and the label check."
    )
    parser.add_argument("--dataset-hash", required=True, help="the RQ1 run's dataset hash")
    parser.add_argument("--root", type=Path, default=Path("runs"), help="dataset root")
    parser.add_argument("--out", type=Path, help="report directory (default: <root>/<hash>-rq2)")
    parser.add_argument(
        "--skip-second-encoder",
        action="store_true",
        help="skip the sensitivity rescore, which downloads and runs a second encoder",
    )
    parser.add_argument("--verbose", action="store_true", help="DEBUG-level logging")
    parsed = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if parsed.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )

    encoder = EncoderConfig()
    d_hash = cast(str, parsed.dataset_hash)
    result, scores = analyse_rq2(
        load_records(d_hash, root=parsed.root),
        Featuriser(encoder),
        dataset_hash=d_hash,
        second_featuriser=(
            None if parsed.skip_second_encoder else Featuriser(second_encoder_config(encoder))
        ),
    )
    out = cast(Path, parsed.out) if parsed.out else cast(Path, parsed.root) / f"{d_hash}-rq2"
    write_rq2(result, out, scores=scores)
    logger.info("rq2 analysis written to %s (recommended Y: %s)", out, result.recommended_y)
    return 0


def _transfer_config(dir: Path | None, key: str) -> tuple[RQ3aConfig, str | None]:
    """Build the RQ3a config, reading the transfer arm's orientation off the calibration report.

    The sign is looked up, never typed: ``orientation`` is what the calibration measured against
    realised failure, and a hand-entered ``-1`` would silently invert every localisation number in
    the table. A ``--transfer`` directory with no report for the requested key is a wiring error and
    raises, rather than falling back to an unavailable arm that looks like an absent one. The
    persisted joblib is checked here for the same reason: ``transfer_scores`` degrades a missing one
    to ``unavailable``, which on a judge run would spend the GPU allocation and lose the row.
    """
    if dir is None:
        return RQ3aConfig(), None
    path = dir / "calibration.json"
    if not path.exists():
        raise ConfigError(f"no calibration.json under {dir}; produce it with preceptx-calibrate")
    report = CalibrationReport.model_validate_json(path.read_text())
    resolved = resolve_statistic_key(key)
    cal = next((s for s in report.statistics if s.key == resolved), None)
    if cal is None:
        available = ", ".join(sorted(s.key for s in report.statistics))
        raise ConfigError(f"{path} has no statistic {resolved!r} (has: {available})")
    if not (dir / f"{resolved}.manifest.json").exists():
        raise ConfigError(
            f"{dir} carries a report but no persisted {resolved!r} statistic. --transfer wants the "
            "*-calibration directory preceptx-calibrate wrote, not a frozen run directory: the "
            "joblib is a trained probe and is gitignored, so refit it with preceptx-calibrate "
            f"--dataset-hash {report.dataset_hash} first."
        )
    cfg = RQ3aConfig(transfer_dir=dir, transfer_key=resolved, transfer_orientation=cal.orientation)
    return cfg, report.dataset_hash


def rq3a(argv: list[str] | None = None) -> int:
    """``preceptx-rq3a``: score every localisation method on a real multi-agent corpus (DSE-064).

    The corpus comes off disk (``scripts/fetch_rq3a.sh``), so the offline arms - the two surface
    baselines, both CPVI regimes and the MAST secondary - need no endpoint at all. ``--judge`` adds
    the three Who&When procedure replications, which are the only methods that cost model calls and
    the only reason this driver ever needs a served model.
    """
    parser = argparse.ArgumentParser(
        prog="preceptx-rq3a", description="Score localisation methods on a real MAS corpus (H5)."
    )
    parser.add_argument("--root", type=Path, required=True, help="corpus root (fetch_rq3a.sh)")
    parser.add_argument(
        "--corpus",
        choices=["traceelephant", "who_and_when"],
        default="traceelephant",
        help="the per-step substrate; TraceElephant is primary (it records input_context)",
    )
    parser.add_argument("--out", type=Path, help="report directory (default: <root>/<corpus>-rq3a)")
    parser.add_argument(
        "--judge",
        action="store_true",
        help="replicate the three Who&When procedures against the served tier (costs model calls)",
    )
    parser.add_argument("--no-mast", action="store_true", help="drop the trace-level MAST arm")
    parser.add_argument(
        "--transfer",
        type=Path,
        help="calibration dir from preceptx-calibrate: enables the cpvi_transfer arm",
    )
    parser.add_argument(
        "--transfer-key",
        default=RQ3aConfig.model_fields["transfer_key"].default,
        help="which persisted statistic to transfer (DSE-061 retired 'info')",
    )
    parser.add_argument("--model", default="qwen14b", help="Hydra model group serving the judge")
    parser.add_argument("--base-url", default=ServingConfig.model_fields["base_url"].default)
    parser.add_argument("--timeout", type=float, default=60.0, help="per-request timeout (s)")
    parser.add_argument(
        "--structured-mode", choices=["guided_json", "response_format"], default="guided_json"
    )
    parser.add_argument("--thinking-switch", default="", help="in-band no-thinking token")
    parser.add_argument("--config-dir", type=Path, default=_DEFAULT_CONFIG_DIR)
    parser.add_argument("--overrides", nargs="*", default=[], help="extra Hydra overrides")
    parser.add_argument("--dry-run", action="store_true", help="print the plan; issue no calls")
    parser.add_argument("--verbose", action="store_true", help="DEBUG-level logging")
    parsed = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if parsed.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )

    cfg, train_hash = _transfer_config(parsed.transfer, parsed.transfer_key)
    corpus = cast(str, parsed.corpus)
    if parsed.dry_run:
        # Loading is not a model call, and the counts are the whole point of a pre-flight here:
        # the judge's cost is a function of trace lengths, which only the corpus knows.
        records = load_corpus(parsed.corpus, parsed.root)
        counts = count_handoff_corpus(records)
        print(f"corpus:           {corpus}")
        print(f"traces:           {counts.traces}")
        print(f"steps:            {counts.steps}")
        print(f"handoffs:         {counts.handoffs}")
        print(f"failures:         {counts.failures} ({counts.non_failures} non-failures)")
        print(f"reconstructed s:  {counts.reconstructed_observations}")
        calls = projected_judge_calls(records, handoffs_only=cfg.handoffs_only)
        print(f"judge calls:      {calls if parsed.judge else 0} (upper bound; --judge)")
        return 0

    judge = None
    with contextlib.ExitStack() as stack:
        if parsed.judge:
            if not os.environ.get("PRECEPTX_SERVING_SUBSTRATE"):
                raise ConfigError(
                    "PRECEPTX_SERVING_SUBSTRATE is unset; label the substrate (e.g. 'myriad-a100') "
                    "so the manifest records where the judge replication was served"
                )
            block = _resolve_cell(parsed.config_dir, [f"model={parsed.model}", *parsed.overrides])
            client = stack.enter_context(_client(block.model.name, parsed.base_url, parsed))
            judge = VLLMJudge(client, revision=block.model.revision)
        run = run_rq3a(
            parsed.corpus,
            parsed.root,
            Featuriser(EncoderConfig()),
            cfg=cfg,
            judge=judge,
            with_mast=not parsed.no_mast,
            command=list(sys.argv),
            transfer_train_dataset_hash=train_hash,
        )

    out = cast(Path, parsed.out) if parsed.out else cast(Path, parsed.root) / f"{corpus}-rq3a"
    write_rq3a(run.result, out)
    write_rq3a_manifest(run.manifest, out)
    logger.info(
        "rq3a analysis written to %s (corpus %s, digest %s)",
        out,
        run.result.corpus,
        run.manifest.corpus_digest,
    )
    return 0


if __name__ == "__main__":  # `python -m preceptx.experiments.cli` mirrors the console script
    sys.exit(pilot())
