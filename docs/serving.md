# Serving on Myriad (vLLM)

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
- Set the project on the qsub line or uncomment `#$ -P <project>` in `serve.sh`.

Myriad is single-node: tensor-parallelism is capped by GPUs-per-node (≤ 4). Multi-node serving,
autoscaling and non-vLLM backends are out of scope.

## Launch

```bash
# Workhorse (bf16 14B):
qsub scripts/myriad/serve.sh

# 70B-AWQ across 2 GPUs (repo id is a placeholder until DSE-005 verifies and pins it):
qsub -l gpu=2 -v MODEL=<70B-AWQ-repo-id>,QUANT=awq,TP=2 scripts/myriad/serve.sh
```

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
