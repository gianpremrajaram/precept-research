"""The batch sweep executor: expand a grid, run episodes with bounded concurrency, write the handoff
dataset resumably, and persist a run manifest + summary (DSE-012).

Concurrency is on the LLM-bound episode execution (a bounded thread pool); record writes funnel
through one lock so the append-only Parquet writer never races on its part index (its part name is
``len(glob("part-*"))``, which two concurrent writers would collide on). The sweep is resumable:
completed episodes (by ``episode_id``) are read once up front and skipped, so an interrupted run
restarts without duplicating cells.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from preceptx.agents.graph import EpisodeRunner
from preceptx.agents.prompts import PROMPT_VERSION
from preceptx.config import ExperimentConfig
from preceptx.data.writer import (
    dataset_hash as derive_dataset_hash,
)
from preceptx.data.writer import (
    load_records,
    register_dataset,
    write_handoffs,
)
from preceptx.experiments.sweep import (
    RunSummary,
    SweepConfig,
    build_sweep_manifest,
    episode_id,
    expand,
    sweep_hash,
)
from preceptx.serving.client import LLMClient

logger = logging.getLogger(__name__)


def _completed_ids(dataset_hash: str, root: Path) -> set[str]:
    """Episode ids already in the dataset (read once for resumability); empty if none yet."""
    if not (root / dataset_hash).exists():
        return set()
    return {r.episode_id for r in load_records(dataset_hash, root=root)}


def run_grid(sweep: SweepConfig, client: LLMClient, *, root: Path | str) -> RunSummary:
    """Run the full grid under ``root``, writing handoffs plus a manifest + summary alongside."""
    root = Path(root)
    d_hash = derive_dataset_hash(sweep_hash(sweep))
    cells = expand(sweep)
    done_ids = _completed_ids(d_hash, root)
    pending = [c for c in cells if episode_id(c) not in done_ids]
    logger.info(
        "sweep %s: %d cells, %d pending (%d already complete)",
        d_hash,
        len(cells),
        len(pending),
        len(cells) - len(pending),
    )

    runner = EpisodeRunner(client, max_steps=sweep.max_steps, channel_cfg=sweep.channel)
    write_lock = threading.Lock()

    def _run_one(cell: ExperimentConfig) -> None:
        records = runner.run_episode(cell, episode_id(cell))
        with write_lock:  # serialise writes: the append writer derives its part index from a glob
            write_handoffs(records, root=root, dataset_hash=d_hash)

    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=sweep.concurrency) as pool:
        list(pool.map(_run_one, pending))  # propagate any episode error (fail loud)
    wall = time.monotonic() - start

    summary = _summarise(cells, d_hash, root, wall)
    # Run artefacts live beside the dataset dir, not inside it: load_records reads the whole dataset
    # dir as one parquet table, so a stray JSON file there would break the read.
    run_dir = root / f"{d_hash}-run"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_sweep_manifest(
        sweep, dataset_hash=d_hash, prompt_version=PROMPT_VERSION
    ).model_copy(update={"summary": summary})
    (run_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    (run_dir / "summary.json").write_text(summary.model_dump_json(indent=2))
    register_dataset(
        root=root,
        dataset_hash=d_hash,
        config_hash=sweep_hash(sweep),
        manifest_path=run_dir / "manifest.json",
    )
    logger.info("sweep %s complete: %s", d_hash, summary.model_dump())
    return summary


def _summarise(
    cells: list[ExperimentConfig], dataset_hash: str, root: Path, wall_s: float
) -> RunSummary:
    """Roll up the whole dataset (incl. episodes completed on an earlier run) into a summary."""
    records = load_records(dataset_hash, root=root)
    episodes: dict[str, bool] = {}
    for r in records:
        # y_terminal_success: reaches the goal at this step or any later one -> episode success.
        episodes[r.episode_id] = episodes.get(r.episode_id, False) or bool(r.y_terminal_success)
    n_success = sum(episodes.values())
    return RunSummary(
        n_cells=len(cells),
        n_episodes=len(episodes),
        n_handoffs=len(records),
        success_rate=n_success / len(episodes) if episodes else 0.0,
        wall_time_s=wall_s,
    )
