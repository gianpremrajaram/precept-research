# Serving on Myriad (vLLM)

> Cluster access, node classes, resource-request rules and the first-session runbook are in
> **[`docs/myriad.md`](myriad.md)**. This file is the model ladder and the wire format.

Serving is decoupled from the analysis code: one model per GPU job behind vLLM's OpenAI-compatible
server, with the LangGraph client (`preceptx.serving.LLMClient`) pointed at the local endpoint. The
analysis code installs and runs anywhere; only serving needs a GPU node.

## Model ladder → GPU mapping

| Tier | Model (default) | Serving mem | Fits | `serve.sh` overrides |
|---|---|---|---|---|
| Pilot / fast | Qwen/Qwen3-8B | ~16–18 GB bf16 | any GPU (8B at 4-bit on V100-16GB) | `-v MODEL=…` |
| Workhorse (default) | Qwen/Qwen3-14B | ~28–30 GB bf16 | A100-40GB, L40S-48GB, A100-80GB | none |
| Strong | Qwen/Qwen3-32B | ~64 GB bf16 / ~20 GB AWQ | A100-80GB (bf16) or 40GB+ (AWQ) | `-v MODEL=Qwen/Qwen3-32B` |
| Scale / heterogeneous | 70B-AWQ (repo id unverified — pin via DSE-005) | ~40 GB | 1× A100-80GB or 2× A100-40GB (TP=2) | `-l gpu=2 -v MODEL=…,QUANT=awq,TP=2` |

Dense Qwen3 checkpoints carry **no `-Instruct` suffix** and default to hybrid *thinking* mode;
`ServingConfig.chat_template_kwargs` disables it per request (`{"enable_thinking": false}`), and
`LLMClient.chat` fails loud if a `<think>` block ever reaches a message (P0-3). Pinned revisions
live in `configs/model/*.yaml`.

(See `RESEARCH_ROADMAP.md` §0 for the authoritative table and licensing notes.)

## Queues: Free vs priority

- **Free** allocation: longer, less predictable queue latency; fine for development and small smokes.
  Budget seeds/conditions conservatively.
- **Priority** (three-monthly) allocation: shorter latency; reserve it for the main RQ1 sweeps.
- **No `-P` project code is needed**: the Free allocation is the default for UCL internal users
  (verified live, 25–26 Aug 2026). A future priority allocation would go on the qsub line — an SGE
  directive cannot read the environment, so the jobscripts deliberately carry none.

Myriad is single-node: tensor-parallelism is capped by GPUs-per-node (≤ 4). Multi-node serving,
autoscaling and non-vLLM backends are out of scope.

## Launch

The served name and revision come from `configs/model/<TIER>.yaml` — the same file the run manifest
records them from — so a job cannot serve one checkpoint while the manifest claims another. That
mismatch has no other detector: the health check compares the served model *id*, but `/v1/models`
carries no revision at all. Nothing needs supplying — no `-P` project code; Free is the default.

```bash
# Workhorse (bf16 14B), the default tier:
qsub scripts/myriad/serve.sh

# The 8B tier on the V100 class:
qsub -ac allow=EF -v TIER=qwen8b scripts/myriad/serve.sh

# 70B-AWQ across 2 GPUs — the one tier with no config file, so it still takes both by hand
# (repo id is a placeholder until DSE-005 verifies and pins it):
qsub -l gpu=2 -v MODEL=<70B-AWQ-repo-id>,REVISION=<sha>,QUANT=awq,TP=2 \
  scripts/myriad/serve.sh
```

`MODEL`/`REVISION` remain overridable for exactly that last case. An override that contradicts the
tier config prints a warning naming both values, because in a job log a deliberate override and a
typo are otherwise indistinguishable.

The node class is set in the jobscript (`#$ -ac allow=L`, the 40 GB A100s the bf16 14B needs);
override it on the qsub line with `-ac allow=EF` for the V100-fitting 8B tier or `-ac allow=U` /
`-ac allow=V` for 80 GB A100s.

`serve.sh` serves and nothing more. To serve *and* run the pilot against the endpoint, submit
**`scripts/myriad/pilot.sh`** instead — one job, because a login node driving a compute node's
`localhost` reaches the wrong machine (`docs/myriad.md` §8).

Note `-l mem` is **per slot**: the jobscripts' `-pe smp 8 -l mem=4G` is 32 GB in total, and a
`mem=32G` request would ask for 256 GB and queue forever rather than fail.

Determinism: the server pins `--seed` and `--revision`; greedy decoding (`temperature=0`) is enforced
by `LLMClient`. Batched inference is **not** bit-exact — report seed sensitivity, never claim exact
reproducibility.

## Health check and teardown

```python
from preceptx.serving import LLMClient, ServingConfig

client = LLMClient(ServingConfig(model="Qwen/Qwen3-14B"))  # base_url defaults to :8000/v1
assert client.health_check()        # /v1/models reachable + a smoke completion
client.close()                      # close client connections
```

Tear the **served job** down with SGE: `qstat` to find the job id, then `qdel <jobid>`. The client's
`close()` only releases local HTTP connections; it does not stop the GPU job.

## Switching tiers

`LLMClient` is model-agnostic — change `ServingConfig.model` (and point `base_url` at the right
endpoint) to swap tiers. No code change. Keep the served `--model`, `ServingConfig.model`, and the
run manifest's recorded revision consistent.

## The free local pilot (LM Studio, no cluster)

Everything before the Myriad re-gate runs against a local OpenAI-compatible endpoint, so the pilot
costs nothing and needs no allocation. The only wire-level difference from vLLM is how the JSON
schema is sent, which `ServingConfig.structured_mode` selects (DSE-032):

| | Myriad (vLLM) | Local pilot (LM Studio) |
|---|---|---|
| `structured_mode` | `guided_json` (default) | `response_format` |
| Where the schema rides | `extra_body.guided_json` | `response_format.json_schema.schema` |
| Constraining engine | xgrammar | llama.cpp grammars / Outlines |
| `base_url` | `http://localhost:8000/v1` | `http://localhost:1234/v1` |
| `PRECEPTX_SERVING_SUBSTRATE` | `myriad-<node class>` | `local-lmstudio` |

The schema object itself is byte-identical across the two paths, so the constraint is the same one;
only the request shape and the engine enforcing it differ. That engine difference is why
**schema-adherence rate is measured, not assumed** — it is a reported row in the DSE-005 table, and
a 4-bit local model may miss the schema more often than a bf16 served one.

Drive LM Studio headless from its CLI — no GUI session required:

```bash
brew install --cask lm-studio          # once; `lms` ships inside the app bundle
open -g -a "LM Studio"                 # once: the CLI only bootstraps after a first launch
lms get mlx-community/Qwen3-8B-4bit    # ~4.6 GB, fits an 18 GB machine
lms server start --port 1234
# Serve under the FULL repository id, so the wire id, the manifest and the runtime all agree:
lms load qwen3-8b --identifier mlx-community/Qwen3-8B-4bit -y
lms ps                                 # confirm the model is loaded under that identifier
```

Two runtime quirks, both found on the first live call and both worth knowing before you lose an hour
to them.

**LM Studio ignores `chat_template_kwargs`.** Qwen3 therefore stays in thinking mode, the reasoning
is routed into a non-standard `reasoning_content` field, and `content` comes back **empty with HTTP
200**. Pass `--thinking-switch /no_think` (`ServingConfig.thinking_switch`): Qwen3's in-band switch
selects the same non-thinking branch the cluster selects via the template kwarg. `chat` now raises on
empty content and `health_check` pings for 16 tokens, so a runtime left in thinking mode fails before
a sweep starts rather than at handoff one.

**Load with `--identifier`.** LM Studio otherwise serves the model as the bare key `qwen3-8b`, while
the manifest must record the quantised repository. Setting the identifier at load time makes the two
the same string instead of relying on a flag to reconcile them.

Then point a run at it, labelling the substrate so the dataset stays distinguishable from cluster
data for good:

```bash
export PRECEPTX_SERVING_SUBSTRATE=local-lmstudio
uv run preceptx-pilot --dry-run --model qwen8b        # cost the sweep first, no calls issued
uv run preceptx-pilot --model qwen8b \
  --base-url http://localhost:1234/v1 \
  --structured-mode response_format \
  --thinking-switch /no_think \
  --overrides model.name=mlx-community/Qwen3-8B-4bit \
             model.revision=545dc4251c05440727734bcd94334791f6ab0192 model.tier=8b-4bit \
  --root runs/local
```

Two identity caveats for the manifest. The served **model identifier is the quantised repository**
(`mlx-community/Qwen3-8B-4bit` @ `545dc4251c05440727734bcd94334791f6ab0192`), not the bf16
`Qwen/Qwen3-8B` the Hydra group names — hence the overrides above, so a local run records the tier it
actually served. And a local **4-bit G1 verdict is indicative only**: the verdict of record is the
bf16 re-gate on Myriad.

## Ladder benchmark (DSE-005)

`scripts/benchmark_models.py` measures one served tier and rewrites the ladder table. Only one model
is served per GPU job, so it is **append-then-render**: run it once per endpoint and every run
regenerates `ladder.md`, `ladder.csv` and `recommendation.md` from all rows collected so far.

```bash
# Local pilot tier (free, no cluster)
PRECEPTX_SERVING_SUBSTRATE=local-lmstudio uv run python scripts/benchmark_models.py \
  --tier 8b-4bit --model mlx-community/Qwen3-8B-4bit \
  --revision 545dc4251c05440727734bcd94334791f6ab0192 \
  --base-url http://localhost:1234/v1 --structured-mode response_format

# Myriad workhorse
PRECEPTX_SERVING_SUBSTRATE=myriad-a100 uv run python scripts/benchmark_models.py \
  --tier 14b --model Qwen/Qwen3-14B --revision 40c069824f4251a91eefaf281ebe4c544efd3e18
```

Five numbers per tier: throughput (tok/s), time to first token, peak GPU memory (`n/a` off-GPU —
never a fabricated zero), JSON-schema adherence over N constrained calls, and a ten-episode C0 easy
capability smoke on the real loop. The recommendation note names the fastest tier clearing both
floors (smoke ≥ 50%, schema ≥ 95%) or refuses to pick one, and it repeats the two standing caveats:
a tier with no row is a **placeholder**, and a local 4-bit row speaks to throughput and schema
behaviour only — capability claims are made at bf16 on the cluster.

Read `ttft_s` as a proxy: it is the latency of a very short (8-token) completion, not a streamed
first-chunk timestamp. It cannot be a one-token completion — the client rejects empty content, which
is exactly what a one-token request returns from a runtime left in thinking mode.
