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
datasets), `--extra serving` (vLLM, GPU nodes only). Serving runs on Myriad GPU nodes — see
[`docs/serving.md`](docs/serving.md).

## Checks (run before handing a branch back)

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy --strict src/preceptx && uv run pytest
```

## Where things live

- **[`RESEARCH_ROADMAP.md`](RESEARCH_ROADMAP.md)** — the design authority: research questions (RQ1
  information gradient, RQ2 measurement primitive, RQ3a external validity, RQ3b causal gate), system
  architecture, the phase plan, and the model/compute envelope. Read it before starting a ticket.
- **[`ISSUES.md`](ISSUES.md)** — the implementation backlog (DSE-001 … DSE-030), mirrored to GitHub
  Issues.
- **[`DEPENDENCIES.md`](DEPENDENCIES.md)** — critical path, ticket dependency graph, risk register,
  phase gates.
- **[`CLAUDE.md`](CLAUDE.md)** — operating guide (code style, architectural constraints,
  reproducibility discipline).
- **`src/preceptx/`** — `sim` (Pymunk arena), `agents` (LangGraph loop + channel), `serving` (vLLM
  client), `data` (handoff schema), `measure` (CPVI), `gate` (runtime statistics + in-repo gate),
  `experiments`, `analysis`.
