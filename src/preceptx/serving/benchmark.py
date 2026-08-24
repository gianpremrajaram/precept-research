"""Model-ladder benchmark: what a tier costs and whether it can do the task (DSE-005).

Given one served endpoint, measure the four numbers that decide the workhorse tier - throughput,
time to first token, JSON-schema adherence, and a short C0 capability smoke on the real loop - and
accumulate one row per tier into a single comparison table. Only one tier is served at a time (one
model per GPU job on Myriad), so the harness is *append-then-render*: run it once per endpoint, and
every run rewrites the table and the recommendation note from all rows collected so far.

Two of the numbers deserve their caveats up front. **Schema adherence is measured, not assumed**: a
4-bit local model may miss the action schema more often than a bf16 served one, and the constraining
engine differs between substrates (llama.cpp grammars / Outlines locally, xgrammar under vLLM), so
the rate is a property of the pair and belongs in the table rather than a footnote. And **the
capability smoke is not a gate**: it is C0 at easy difficulty on a handful of seeds, enough to
tell a tier that can do the task from one that cannot, not enough to stand in for G1 (DSE-019).
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import shutil
import subprocess
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from preceptx.agents.graph import Action
from preceptx.config import ModelConfig
from preceptx.experiments.runner import run_grid
from preceptx.experiments.sweep import SweepConfig
from preceptx.serving.client import ChatMessage, LLMClient, ServingError

logger = logging.getLogger(__name__)

# A short, fully specified prompt: throughput must measure the endpoint, not the model's willingness
# to keep talking, so the generation length is bounded by max_tokens rather than by the request.
_THROUGHPUT_PROMPT = "Describe a rectangular room in plain prose. Do not stop early."
_SCHEMA_PROMPT = "Choose one macro-action for the load. Reply with the action only."
_LATENCY_PROMPT = "Reply with the single word: ready."


class TierResult(BaseModel):
    """One row of the ladder table: a tier, where it was served, and what it measured."""

    model_config = ConfigDict(extra="forbid")

    tier: str
    model: str
    revision: str
    base_url: str
    substrate: str
    structured_mode: str
    tokens_per_s: float
    ttft_s: float
    peak_memory_mb: float | None  # None off-GPU (a laptop has no nvidia-smi), never a fabricated 0
    schema_adherence: float = Field(ge=0, le=1)
    smoke_success_rate: float = Field(ge=0, le=1)
    n_smoke_episodes: int
    timestamp: str


def parse_nvidia_smi(output: str) -> float | None:
    """Peak used memory in MiB from ``nvidia-smi`` CSV output; None when nothing parses.

    Kept pure so the parse is testable against captured fixture text with no GPU present.
    """
    values = [
        float(line.strip())
        for line in output.splitlines()
        if line.strip() and line.strip().replace(".", "", 1).isdigit()
    ]
    return max(values) if values else None


def peak_memory_mb() -> float | None:
    """Query the local GPU, or None when there is no ``nvidia-smi`` (the local pilot substrate)."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, OSError) as exc:
        logger.warning("nvidia-smi query failed; recording no memory figure: %s", exc)
        return None
    return parse_nvidia_smi(out.stdout)


def measure_throughput(client: LLMClient, *, n_calls: int = 3, max_tokens: int = 128) -> float:
    """Generated tokens per wall-clock second, averaged over ``n_calls`` bounded completions.

    Tokens are counted as whitespace-delimited words rather than by the served tokeniser: the ladder
    comparison only needs the tiers on one consistent scale, and a per-tier tokeniser would make the
    rows less comparable, not more.
    """
    messages = [ChatMessage(role="user", content=_THROUGHPUT_PROMPT)]
    start = time.monotonic()
    tokens = sum(len(client.chat(messages, max_tokens=max_tokens).split()) for _ in range(n_calls))
    elapsed = time.monotonic() - start
    return tokens / elapsed if elapsed > 0 else 0.0


def measure_ttft(client: LLMClient, *, n_calls: int = 3, max_tokens: int = 8) -> float:
    """Mean latency of a very short completion, in seconds - the stand-in for time to first token.

    ponytail: a short round trip is prefill plus a handful of decode steps, which needs no streaming
    API; swap in a streamed first-chunk timestamp if the prefill/decode split ever has to be
    separated. It cannot be a *one*-token completion: the client rejects empty content (a runtime in
    thinking mode returns exactly that), and one token is often whitespace.
    """
    messages = [ChatMessage(role="user", content=_LATENCY_PROMPT)]
    start = time.monotonic()
    for _ in range(n_calls):
        client.chat(messages, max_tokens=max_tokens)
    return (time.monotonic() - start) / n_calls


def measure_schema_adherence(client: LLMClient, *, n_calls: int = 20) -> float:
    """Fraction of schema-constrained calls returning a valid ``Action``.

    A malformed response is the measurement here, not an error: both the transport-level
    ``ServingError`` (unparseable JSON) and a schema-invalid object count as a miss, and nothing is
    re-tried, because a retried rate would flatter the tier.
    """
    schema = Action.model_json_schema()
    messages = [ChatMessage(role="user", content=_SCHEMA_PROMPT)]
    valid = 0
    for _ in range(n_calls):
        try:
            Action.model_validate(client.structured(messages, schema))
        except (ServingError, ValidationError):
            continue
        valid += 1
    return valid / n_calls


def capability_smoke(
    client: LLMClient, model: ModelConfig, *, root: Path | str, n_episodes: int = 10
) -> float:
    """Episode success rate over ``n_episodes`` C0 easy-difficulty episodes on the real loop.

    Indicative only: C0 at easy difficulty is the least demanding cell in the design, so this
    separates a tier that can drive the loop from one that cannot. G1 is DSE-019's job.
    """
    sweep = SweepConfig(
        conditions=["C0"],
        serialisations=["numeric"],
        difficulties=["easy"],
        seeds=list(range(n_episodes)),
        model=model,
        concurrency=1,  # a benchmark measures one endpoint's latency, not its saturation behaviour
    )
    return run_grid(sweep, client, root=root).success_rate


def benchmark_tier(
    client: LLMClient,
    model: ModelConfig,
    *,
    substrate: str,
    root: Path | str,
    n_smoke_episodes: int = 10,
    n_schema_calls: int = 20,
) -> TierResult:
    """Run the whole battery against one served endpoint and return its table row."""
    logger.info("benchmarking tier %s (%s) at %s", model.tier, model.name, client.config.base_url)
    tokens_per_s = measure_throughput(client)
    ttft = measure_ttft(client)
    adherence = measure_schema_adherence(client, n_calls=n_schema_calls)
    smoke = capability_smoke(client, model, root=root, n_episodes=n_smoke_episodes)
    return TierResult(
        tier=model.tier,
        model=model.name,
        revision=model.revision,
        base_url=client.config.base_url,
        substrate=substrate,
        structured_mode=client.config.structured_mode,
        tokens_per_s=tokens_per_s,
        ttft_s=ttft,
        peak_memory_mb=peak_memory_mb(),
        schema_adherence=adherence,
        smoke_success_rate=smoke,
        n_smoke_episodes=n_smoke_episodes,
        timestamp=dt.datetime.now(dt.UTC).isoformat(),
    )


def render_table(results: list[TierResult]) -> str:
    """The ladder comparison as one Markdown table, tiers in the order they were measured."""
    header = (
        "| Tier | Model | Substrate | Mode | tok/s | TTFT (s) | Peak mem (MiB) | Schema | "
        "C0 smoke |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    rows = [
        f"| {r.tier} | `{r.model}` | {r.substrate} | {r.structured_mode} | {r.tokens_per_s:.1f} | "
        f"{r.ttft_s:.2f} | {'n/a' if r.peak_memory_mb is None else f'{r.peak_memory_mb:.0f}'} | "
        f"{r.schema_adherence:.0%} | {r.smoke_success_rate:.0%} ({r.n_smoke_episodes} ep) |"
        for r in results
    ]
    return "\n".join([header, *rows])


def recommend(
    results: list[TierResult], *, min_smoke: float = 0.5, min_schema: float = 0.95
) -> str:
    """A short note naming the cheapest tier that clears both floors, or saying none does."""
    clearing = [
        r for r in results if r.smoke_success_rate >= min_smoke and r.schema_adherence >= min_schema
    ]
    lines = [
        "# Ladder recommendation",
        "",
        f"Floors applied: C0 smoke >= {min_smoke:.0%}, schema adherence >= {min_schema:.0%}.",
        "",
    ]
    if not clearing:
        lines += [
            "**No tier clears both floors.** Do not pick a workhorse from this table: either the "
            "prompts need the budgeted iteration first (E1), or the ladder needs a larger tier.",
        ]
    else:
        # Cheapest = the fastest of the tiers that clear, which on a single-GPU envelope is also the
        # smallest; throughput is the operative cost here because the GPU-hour budget is the limit.
        best = max(clearing, key=lambda r: r.tokens_per_s)
        lines += [
            f"**Workhorse: `{best.model}` ({best.tier}, {best.substrate}).** It clears both floors "
            f"at {best.tokens_per_s:.1f} tok/s and {best.smoke_success_rate:.0%} C0 smoke.",
            "",
            "Tiers clearing the floors: " + ", ".join(f"`{r.model}`" for r in clearing) + ".",
        ]
    lines += [
        "",
        "Any tier in the roadmap ladder with no row above is a **placeholder** and must be treated "
        "as one in the thesis. A local 4-bit row is indicative of throughput and schema behaviour "
        "only; capability claims are made at bf16 on the cluster.",
    ]
    return "\n".join(lines)


def append_result(result: TierResult, dir: Path | str) -> list[TierResult]:
    """Append one row to the accumulating ladder and return every row collected so far."""
    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)
    rows_path = dir / "tiers.jsonl"
    with rows_path.open("a") as fh:
        fh.write(result.model_dump_json() + "\n")
    return [TierResult.model_validate_json(line) for line in rows_path.read_text().splitlines()]


def write_report(results: list[TierResult], dir: Path | str) -> Path:
    """Rewrite the table (Markdown + CSV) and the recommendation note from all rows. Returns dir."""
    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)
    (dir / "ladder.md").write_text(render_table(results) + "\n")
    fields = list(TierResult.model_fields)
    csv_rows = [",".join(fields)] + [
        ",".join("" if (v := getattr(r, f)) is None else str(v) for f in fields) for r in results
    ]
    (dir / "ladder.csv").write_text("\n".join(csv_rows) + "\n")
    (dir / "recommendation.md").write_text(recommend(results) + "\n")
    (dir / "ladder.json").write_text(json.dumps([r.model_dump() for r in results], indent=2))
    return dir
