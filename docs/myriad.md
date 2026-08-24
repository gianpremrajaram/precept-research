# Running on Myriad

Everything the repository needs from UCL Research Computing, and the order to do it in on a first
session. Written from the [UCL RC documentation](https://www.rc.ucl.ac.uk/docs/) before any live
session, so §9 lists what is *documented* but not yet *verified on the box* — check those first and
correct this file in place.

`docs/serving.md` covers the model ladder and the vLLM wire format. This file covers the cluster.

---

## 1. Myriad, not Kathleen

UCL runs two internal clusters and only one of them can do this work.

| | Myriad | Kathleen |
|---|---|---|
| GPUs | V100, A100 40 GB, A100 80 GB | **none** |
| Local disk | `$TMPDIR` up to 1500 G | **diskless** — `-l tmpfs` is a hard error |
| Built for | high-throughput, single-node, GPU, large-memory | large multi-node MPI |

Serving a 14B model behind vLLM is a single-node GPU job, so **Myriad is the only fit**. Kathleen
has no GPU to request; the analysis half of this repo would run there, but there is no reason to
split a job across two clusters to save nothing. Young and Michael are Tier-2 services for other
consortia and are not available to this project.

## 2. Getting on

UCL services live behind the UCL firewall. From outside, you go through the SSH gateway, and **since
23 March 2026 the gateway requires SSH keys** — password login works only from the UCL VPN or
Desktop@UCL Anywhere. So the first-ever connection has to happen from the VPN.

```bash
# 1. On the VPN, once: create a key (use a real passphrase) and put it on the gateway.
ssh-keygen -t ed25519
ssh-copy-id <ucl-id>@ssh-gateway.ucl.ac.uk

# 2. The gateway is two machines with unshared home directories. Sync, or half your logins
#    will ask for a password:
ssh <ucl-id>@ssh-gateway.ucl.ac.uk
copy-ssh-keys -copy -verbose
```

Then make the two hops one, in `~/.ssh/config`:

```
Host myriad
    User <ucl-id>
    HostName myriad.rc.ucl.ac.uk
    ProxyJump <ucl-id>@ssh-gateway.ucl.ac.uk
```

`ssh myriad` and `scp file myriad:~/Scratch/` now work from anywhere. `myriad.rc.ucl.ac.uk` is
load-balanced across `login12` and `login13`; address a login node **directly** if you want to
reattach to a `tmux` session, because the alias may not send you back to the same one.

Login nodes are shared. Under 15 minutes and light is fine — `--dry-run`, `git`, `module avail`,
editing. Anything else is a job, and long-running processes on a login node get killed.

## 3. Nodes, GPUs, and which one this project wants

| Class | GPUs | RAM usable | Cores | Nodes | `-ac allow=` | Our use |
|---|---|---|---|---|---|---|
| L | 4 × A100 **40 GB** | 160 G | 36 | 6 | `L` | **the default** — bf16 14B workhorse (~28–30 GB) |
| U, V | 4 × A100 **80 GB** | 223 G | 48 | 3 | `U` or `V` | 32B bf16 (~64 GB) |
| E, F | 2 × V100 16 GB | 160 G | 36 | 19 | `EF` | 8B tier only; most nodes, shortest queue |
| D | none | 160 G | 36 | 342 | — | CPU-only re-scoring, if ever wanted |

`-l gpu=1` or `2` can land on any class; `gpu=4` forces a four-GPU node. Compute Capability is 8.0
on the A100s and 7.0 on the V100s — a `device kernel image is invalid` error means the build does
not target the card you got.

**Pick the class explicitly.** Without `-ac allow=`, a bf16 14B job can be scheduled onto a V100
node and fail on memory after waiting in the queue.

## 4. Resource requests — the three that bite

**`-l mem` is per slot, not per job.** This is the one that costs a day. `-pe smp 8 -l mem=32G` is a
request for **256 GB**, which no Myriad node can satisfy; the job is not rejected, it simply queues
forever. Both jobscripts here use `-pe smp 8 -l mem=4G` = 32 GB total. Multiply before you submit.

**Wallclock is capped by core count**, and the cap is not negotiable once the job is running:

| Cores | Max `h_rt` |
|---|---|
| 1 | 72 h |
| 2–36 | 48 h |
| 2–64 (T nodes) | 48 h |

`pilot.sh` asks for 6 h. Over-asking costs nothing on the Free queue and you cannot extend a running
job, so round up.

**Quota is 1 TB, shared between home and Scratch**, and running out does not fail cleanly — the job
dies creating its `.o`/`.e` files, which looks like a scheduler fault. `gquota` shows usage. The
jobscripts point `HF_HOME`, `XDG_CACHE_HOME`, `VLLM_CACHE_ROOT` and `TRITON_CACHE_DIR` at Scratch
for exactly this reason: model weights alone are ~28 GB for the 14B tier and ~45 GB across the
ladder.

*Not* requested, deliberately: `-l tmpfs`. Nothing here writes to `$TMPDIR` — weights and caches are
on Scratch, where they persist between jobs instead of being re-downloaded. Requesting tmpfs would
add a scheduling constraint that buys nothing.

## 5. Where the repository lives

Check out under **`~/Scratch`**, not `$HOME`:

```bash
cd ~/Scratch && git clone <repo-url> precept-research
```

UCL's example jobscripts all say compute nodes cannot write to `$HOME`. Myriad's own page says home
*is* scratch and shares the 1 TB quota, which softens that — but both jobscripts use `#$ -cwd`, so
the job's working directory is wherever you submit from, and Scratch is correct under either
reading. The `.o`/`.e` job logs land there too.

## 6. Environment

The repo pins Python 3.11. Myriad has a `python3/3.11` module, but **uv is the better route** — it
is the project's package manager and `uv.lock` is the reproducibility anchor, so an interpreter uv
manages reproduces the lockfile exactly.

```bash
# uv installs as a single static binary into user space; no module, no root.
curl -LsSf https://astral.sh/uv/install.sh | sh

cd ~/Scratch/precept-research
uv sync --extra serving --extra embed
```

`uv sync` rather than `uv pip install -e .`: sync installs the resolved versions from `uv.lock`,
which is the reproducibility anchor, while `uv pip install` re-resolves against whatever PyPI serves
that day. The cluster run is the run of record, so it is the last place to install off-lock. The
lock already carries vLLM's `manylinux_2_31_x86_64` wheel, so nothing compiles from source.

Sync populates `.venv` in the repo — uv's default, and what all three scripts activate. Override
with `-v VENV=<path>` if you keep environments elsewhere.

**CUDA.** vLLM's wheels bundle their own CUDA runtime through torch, so the module is a convenience
rather than a requirement. When it is loaded the name must be one Myriad has — UCL's are versioned
like `cuda/12.2.2/gnu-10.2.0`, and the GPU-node driver is on the 12.2 branch, so a mismatched
toolkit is a genuine source of runtime errors. Check before your first submit:

```bash
module avail cuda
```

Then either pass the exact name (`-v CUDA_MODULE=<name>`) or skip it (`-v CUDA_MODULE=none`).

## 7. First session, in order

Do the first vLLM launch **interactively**. Debugging a serving failure through the batch queue is a
multi-hour loop; interactively it is a multi-second one.

```bash
# 1. Login node: environment, and the cheap checks that need no GPU.
module avail cuda                      # note the exact 12.x name
uv run preceptx-pilot --dry-run --model qwen14b    # cells, call count, hashes; no model calls

# 2. Pre-pull weights AND the embedding encoder on the login node. Checks quota first; resumable.
bash scripts/myriad/prefetch.sh        # TIER=qwen8b for the V100-class tier

# 3. Interactive GPU node: prove vLLM serves before trusting a batch job with it.
qrsh -P <project> -l gpu=1,h_rt=2:00:00,mem=4G -pe smp 8 -ac allow=L
#    ... on the node:
nvidia-smi                             # which card did you actually get
source .venv/bin/activate
bash scripts/myriad/serve.sh
#    ... in a second shell on the same node:
curl -s localhost:8000/v1/models       # the served id must equal configs/model/qwen14b.yaml's name

# 4. Batch: the real re-gate.
qsub -P <project> scripts/myriad/pilot.sh
```

Step 3 is the one that catches the cluster-specific failures — a wrong CUDA module, a memory
request that will not schedule, a missing wheel — while they still cost seconds.

**No `REVISION=` anywhere.** Since DSE-050 the served name and revision are read from
`configs/model/<TIER>.yaml`, the same file the manifest records them from, so the two cannot
disagree. Passing them by hand was the one remaining way to serve one checkpoint and record
another — undetectable after the fact, because `/v1/models` carries no revision for the health
check to compare. `MODEL`/`REVISION` still override, for the 70B-AWQ tier that has no config file
yet, and an override that contradicts the config prints a warning naming both values.

## 8. Running, watching, stopping

`scripts/myriad/pilot.sh` is a **single job that serves and drives**: it starts `serve.sh` in the
background, waits for the endpoint, warms the embedding encoder, runs the pre-registered E3 cell,
and kills the server on every exit path including the scheduler's wallclock SIGTERM. A login node
cannot drive a compute node's `localhost`, which is why the two halves share a job.

```bash
qsub -P <project> scripts/myriad/pilot.sh                      # submit
qstat                                                          # running jobs
qstat -f -j <job-id>                                           # why is it still queued
qdel <job-id>                                                  # stop it
jobhist                                                        # finished jobs (module load userscripts)
tail -f precept-pilot.o<job-id>                                # live log; -j y merges stderr in
```

Useful `-v` overrides: `TIER` (Hydra model group), `ATTEMPT=2` (the one permitted retune),
`SERVE_TIMEOUT` (default 1800 s, sized for a cold weights cache), `RUNS_ROOT`, `PORT`, `VENV`.
Add `-m be -M <email>` to be told when it starts and ends.

`serve.sh` alone still works when you want a long-lived endpoint to poke at by hand; tear it down
with `qdel`, since the client's `close()` only drops local HTTP connections.

**The substrate label is automatic.** `pilot.sh` derives `PRECEPTX_SERVING_SUBSTRATE` from
`nvidia-smi`, so the manifest records the card that actually served rather than the class that was
requested. Cluster data is therefore never poolable with `local-lmstudio` pilot data by accident —
which is the point, since the local 4-bit G1 reading is indicative only and the **verdict of record
is the bf16 re-gate run here**.

## 9. Not yet verified on the cluster

Documented, plausible, and unconfirmed. Check these in the first session and correct this file.

- [ ] The exact CUDA module name (`module avail cuda`); `cuda/12.2.2/gnu-10.2.0` is what UCL's docs
      name, and the jobscript default.
- [ ] **Whether compute nodes have outbound internet.** If not, `scripts/myriad/prefetch.sh` (§7
      step 2) is mandatory rather than merely thrifty — it pulls both the weights and the embedding
      encoder on a login node, which does have a route out. Run it either way: it is the only
      preparation that makes this answer stop mattering. `pilot.sh` additionally warms the encoder
      before the sweep, so a node with no route out fails in seconds rather than after a full GPU
      hour with the dataset already paid for.
- [ ] Queue latency for `-ac allow=L` on the Free allocation. There are only 6 L-type nodes; if the
      wait is bad, the 8B tier on `EF` has 19 nodes and is the pressure valve.
- [ ] That `vllm` **runs** against the cluster's driver. Installation itself is no longer in
      question: `uv.lock` carries the `manylinux_2_31_x86_64` wheel, so `uv sync --extra serving`
      fetches a prebuilt binary and compiles nothing. What is unverified is the wheel's bundled
      CUDA runtime against driver 550.127.05 — which is what §7 step 3 exists to find out.
- [ ] Wallclock actually needed for the 40-episode E3 cell at bf16. `h_rt=6:00:00` is a guess from a
      local 4-bit run (5 episodes in ~7 min) and should be replaced with a measured number.
- [ ] Whether `-P <project>` is needed at all — most UCL internal users default to `AllUsers`.

Two items that were on this list are now closed by construction rather than by checking: the served
revision can no longer disagree with the manifest (it is read from the tier config), and the venv
can no longer diverge from the lockfile (`uv sync` into the repo's `.venv`, which is what every
script activates).

## 10. Cost

Nothing here is billed. The Free allocation trades queue latency for cost, which suits development
and the pilot; the three-monthly priority allocation is worth saving for the main RQ1 sweep. Every
model call in this project is either local open-weight inference or the Myriad allocation — no
hosted API is ever called.
