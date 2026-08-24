# precept-research

Thesis research and experimentation supervised by Prof. Jun Wang & Prof. Philip Treleaven.

Measures **conditional pointwise V-usable information (CPVI)** at the natural-language boundary
between two coordinating LLM agents on a T-shaped cooperative-transport task in a Pymunk arena under
a degradable communication channel. The repo is **standalone** — it does not depend on or import
precept.

## Setup

Requires [uv](https://docs.astral.sh/uv/); uv will fetch Python 3.11.

```bash
uv venv --python 3.11
uv sync --extra dev          # core + dev tooling (no GPU, no torch, no vllm)
uv run pytest
```

Optional extras: `--extra embed` (sentence-transformers, pulls torch), `--extra data` (HuggingFace
datasets), `--extra serving` (vLLM, GPU nodes only). Serving runs on Myriad GPU nodes — the model
ladder and wire format are in [`docs/serving.md`](docs/serving.md), the cluster itself in
[`docs/myriad.md`](docs/myriad.md).

## Running an experiment

```bash
uv run preceptx-pilot --dry-run              # cells, upper-bound model calls, hashes; issues no calls
export PRECEPTX_SERVING_SUBSTRATE=local-lmstudio   # required: unlabelled datasets are refused
uv run preceptx-pilot --model qwen8b --base-url http://localhost:1234/v1 \
  --structured-mode response_format --thinking-switch /no_think --root runs/local
uv run preceptx-rq1 --help                   # the RQ1 factorial driver
```

The free local pilot (LM Studio, no cluster, no cost) is set out in
[`docs/serving.md`](docs/serving.md); the cluster path — access, node classes, resource requests and
the first-session runbook — in [`docs/myriad.md`](docs/myriad.md). On Myriad the pilot is one job:
`qsub -P <project> scripts/myriad/pilot.sh`, after a login-node `bash scripts/myriad/prefetch.sh`. The stage plan and every recorded
result live in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

## Checks (run before handing a branch back)

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy --strict src/preceptx && uv run pytest
```

## Where things live

- **[`RESEARCH_ROADMAP.md`](RESEARCH_ROADMAP.md)** — the design authority: research questions (RQ1
  information gradient, RQ2 measurement primitive, RQ3a external validity, RQ3b causal gate), system
  architecture, the phase plan, and the model/compute envelope. Read it before starting a ticket.
- **[`ISSUES.md`](ISSUES.md)** — the implementation backlog (DSE-001 … DSE-049), mirrored to GitHub
  Issues.
- **[`PREREGISTRATION.md`](PREREGISTRATION.md)** — the F0 artefact: hypotheses, the outcome *Y*, the
  probe family *V*, gate thresholds and the analysis protocol, fixed before the main sweep. Currently
  **v0 (draft)**; it freezes as v1 on the E3 `proceed` verdict.
- **[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)** — the stage plan (S0-S6), the experiment specs
  (E0-E11), the freeze register, and the results log.
- **[`docs/experiment_design_log.md`](docs/experiment_design_log.md)** — why the design changed and
  what risk each change closed (as distinct from the CHANGELOG's what-the-code-does).
- **[`docs/myriad.md`](docs/myriad.md)** — the cluster runbook: SSH-gateway access, node classes and
  the `-ac allow=` codes, the per-slot `mem` rule, quota, the uv bootstrap, an ordered first-session
  checklist, and what is documented but not yet verified on the box.
- **[`docs/serving.md`](docs/serving.md)** — the model ladder, the vLLM/LM Studio wire formats, and
  the free local-pilot path.
- **[`DEPENDENCIES.md`](DEPENDENCIES.md)** — critical path, ticket dependency graph, risk register,
  phase gates.
- **[`CLAUDE.md`](CLAUDE.md)** — operating guide (code style, architectural constraints,
  reproducibility discipline).
- **`src/preceptx/`** — `sim` (Pymunk arena), `agents` (LangGraph loop + channel), `serving` (vLLM
  client), `data` (handoff schema), `measure` (CPVI), `gate` (runtime statistics + in-repo gate),
  `experiments`, `analysis`.
