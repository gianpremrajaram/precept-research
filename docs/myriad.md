# Running on Myriad

Everything the repository needs from UCL Research Computing, and the order to do it in on a first
session. Originally written from the [UCL RC documentation](https://www.rc.ucl.ac.uk/docs/) before
any live session; §10 now records what the **25 August 2026** session verified on the box and the
four items still open. Correct this file in place as the rest are settled.

**The short version of what that session found:** Myriad is RHEL 7.9 / glibc 2.17 on login *and*
compute nodes, and every wheel in `uv.lock` needs glibc 2.28 or newer — so the environment runs
inside an Apptainer container and the lockfile does not move. See §6.

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
job, so round up — but not past a maintenance window (below): the scheduler will not start a job
whose wallclock crosses one.

**The cluster has scheduled downtime.** The second Tuesday of every month is a maintenance day —
clusters are at risk from 08:00, and jobs that would still be running then may be held until after.
Check [Planned Outages](https://www.rc.ucl.ac.uk/docs/Planned_Outages/) before queueing anything
long or planning a sweep week. As of 26 Aug 2026 the page lists a **full Myriad outage on 24–25
Sept 2026** (network modernisation — no access, jobs drained ahead of it). The RQ1 main sweep and
its freeze want to be finished before that window, with the 8 Sept maintenance day in the middle.

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
cd ~/Scratch && git clone https://github.com/gianpremrajaram/precept-research.git
```

HTTPS, not SSH: the repository is public, so nothing needs a key on Myriad. Results do **not** come
back by pushing from the cluster — run artefacts are gitignored by design. See §9 for how to
retrieve them.

UCL's example jobscripts all say compute nodes cannot write to `$HOME`. Myriad's own page says home
*is* scratch and shares the 1 TB quota, which softens that — but both jobscripts use `#$ -cwd`, so
the job's working directory is wherever you submit from, and Scratch is correct under either
reading. The `.o`/`.e` job logs land there too.

## 6. Environment — and why it runs in a container

Myriad's login **and** compute nodes are Red Hat Enterprise Linux 7.9 with **glibc 2.17** (verified
25 Aug 2026 on `login12` and on `node-l00a-006`; the cluster is homogeneous, so there is no node
where this is not true). Every wheel in `uv.lock` is `manylinux_2_28` or newer. That is not a torch
problem — it is a pandas, pyarrow, scipy and scikit-learn problem:

| package | cp311 x86_64 tags in `uv.lock` |
|---|---|
| pandas 2.3.3 | `manylinux_2_24` / `2_28` |
| pyarrow 21.0.0 | `manylinux_2_28` |
| scipy 1.17.1 | `manylinux_2_27` / `2_28` |
| scikit-learn 1.9.0 | `manylinux_2_27` / `2_28` |
| torch 2.10.0 | `manylinux_2_28`, **and no sdist** |
| vllm 0.18.1 | `manylinux_2_31` |

`uv sync` reports torch first only because torch is the one package publishing no source
distribution. The other four have sdists, so uv would fall through to compiling them with gcc 4.8.5
on RHEL 7 — which for pyarrow, needing Arrow C++ and C++17, does not happen. **`uv sync` with no
extras at all cannot produce a working environment on a bare Myriad node.**

Downgrading is not the fix. The newest torch with a glibc-2.17 wheel is **2.6.0**; the newest vLLM
pinning ≤ 2.6.0 is **0.8.5**, from April 2025:

| vLLM | torch pin | installs at glibc 2.17? |
|---|---|---|
| 0.8.5 | `==2.6.0` | yes — the last one |
| 0.9.x | `==2.7.0` | no |
| 0.11.2 | `==2.9.0` | no |
| 0.18.1 (locked) | `2.10.0` | no |

And that path still leaves pandas, pyarrow, scipy and scikit-learn unsolved. Since scipy and
scikit-learn sit directly under the CPVI estimator, moving them would make every measurement taken
before the change incomparable with every one taken after — for a sixteen-month-old server. So the
environment runs in a container instead, and **`uv.lock` does not move**.

### The image

```bash
docker://python@sha256:a8677eb08a56d04e75df938f9d2af3d50c0f0fba17af8eb9c8e41b65fa32938d
```

Pinned by **digest, not tag**: `python:3.11` is mutable, and a verdict-of-record run cannot rest on
whatever Docker Hub served that day. Debian bookworm carries glibc 2.36, clearing `manylinux_2_28`
and vLLM's `manylinux_2_31`. The **full** image, not `-slim`, because `manifest.git_sha()` shells
out to `git` and raises rather than defaulting when it is absent — a git-less image would fail
*after* the episodes were paid for.

Nothing is built: `apptainer pull` converts an existing image, so no `--fakeroot` and no `.def`
file. `scripts/myriad/prefetch.sh` pulls it to `~/Scratch/containers/` and records its SHA-256
beside it. Both the source digest and the `.sif` digest land in every run manifest via
`serve_env.json`, alongside the glibc version the run actually saw.

### Getting set up

```bash
# uv installs as a single static binary into user space; no module, no root. It targets glibc 2.17,
# so the same binary runs on the bare node and inside the container.
curl -LsSf https://astral.sh/uv/install.sh | sh

cd ~/Scratch/precept-research
bash scripts/myriad/prefetch.sh        # image, then venv, then weights, then the encoder

# Add the RQ3a corpora (~800 MB) if this session will run the judge replication:
RQ3A_ROOT=$HOME/Scratch/rq3a bash scripts/myriad/prefetch.sh
```

`scripts/myriad/shell.sh` is the entry point for everything that is *not* a jobscript — the dry-run
hash checks, the smoke, poking at a failed run. `bash scripts/myriad/shell.sh` drops you into an
interactive shell inside the container with `.venv` active; `-c '<cmd>'` runs one command and
exits. It applies `--nv` only where there is a GPU, so the same invocation works on a login node.


`prefetch.sh` now owns all four, in that order, and each step is idempotent. It checks `gquota`
before pulling anything, optionally pulls the RQ3a corpora when `RQ3A_ROOT` is set (on the host,
where `curl` and `unzip` live), pulls the image, re-execs itself inside it, then builds `.venv` with
`uv sync --extra serving --extra embed --python /usr/local/bin/python3.11` and asserts that
pandas, pyarrow, scikit-learn and torch import — which is the check that proves the environment is
the container's and not the host's.

`uv sync` rather than `uv pip install -e .`: sync installs the resolved versions from `uv.lock`,
while `uv pip install` re-resolves against whatever PyPI serves that day. The cluster run is the run
of record, so it is the last place to install off-lock.

> If you already ran `uv sync` on the bare login node, it left a `.venv` that cannot import the
> locked wheels. `ensure_venv` detects exactly that and rebuilds it; you do not need to clean up.

**CUDA.** `CUDA_MODULE` now defaults to `none`. There is no module system inside the image, and
nothing is lost: torch's bundled `cu12` libraries are the CUDA userspace and `apptainer --nv`
injects the host driver. The locked stack is `cu12` throughout (`nvidia-cublas-cu12 12.8.4.1`) and
the L-node driver is **550.127.05 / CUDA 12.4**, so CUDA minor-version compatibility covers it. The
override survives for a bare run on some future non-RHEL7 node.

## 7. First session, in order

Do the first vLLM launch **interactively**. Debugging a serving failure through the batch queue is a
multi-hour loop; interactively it is a multi-second one.

```bash
# 1. Login node: build everything. Image, then venv, then weights, then the encoder.
#    `bash <script>` is not a login shell, so load the module explicitly rather than relying on
#    `module` having been exported as a function into the environment.
module load apptainer/1.2.4-1
bash scripts/myriad/prefetch.sh        # TIER=qwen8b for the V100-class tier

# 2. Login node: prove the container before trusting anything else to it. Seconds, no GPU.
bash scripts/myriad/shell.sh -c 'pwd; git rev-parse --short HEAD; ldd --version | head -1; python -c "import torch, pyarrow; print(torch.__version__)"'
#    expect: the repo path, the commit SHA, glibc 2.36, a torch version

# 3. Login node: the plan hashes. These go through the container too — the CLI imports pandas and
#    pyarrow, so even --dry-run cannot run on the bare node.
bash scripts/myriad/shell.sh -c 'preceptx-pilot --dry-run --model qwen14b'
#    expect: cells 80, calls 4080, dataset hash eddd19c654515bb2
#    (attempt 1 was cells 40, calls 2040, hash 1c994b87bbca8257 - prompt v5 and seeds 0-9
#     re-key the dataset, so attempt 2 cannot append to or resume attempt 1's data)

#    The serve flag, without a GPU. `vllm serve --help` does NOT work here: building the parser
#    instantiates VllmConfig's defaults, which raises "Failed to infer device type" on a login
#    node. Reading the dataclass needs no device.
bash scripts/myriad/shell.sh -c 'python -c "
import dataclasses
from vllm.config import VllmConfig
names = [f.name for f in dataclasses.fields(VllmConfig)]
assert \"structured_outputs_config\" in names, names
print(\"structured_outputs_config: present\")"'

# 4. Interactive GPU node: prove vLLM serves, then drive two episodes end to end.
qrsh -l gpu=1,h_rt=2:00:00,mem=4G -pe smp 8 -ac allow=L
#    ... on the node (see the smoke recipe below)

# 5. Batch: the real re-gate. ATTEMPT=2 is the one permitted retune (PREREGISTRATION SS6).
qsub -v ATTEMPT=2 scripts/myriad/pilot.sh
```

Step 2 is the cheapest possible failure: if the bind, the working directory, git or the wheels are
wrong, it costs seconds on a login node rather than a queue wait and an A100.

Step 4 is what catches the cluster-specific serving failures — a wheel that will not run against
driver 550.127.05, a memory request that will not schedule — while they still cost seconds.

### The two-episode smoke (step 4, on the GPU node)

```bash
cd ~/Scratch/precept-research
nvidia-smi                             # which card did you actually get
module load apptainer/1.2.4-1          # so the scripts find it without relying on `module`

bash scripts/myriad/serve.sh > runs/smoke-serve.log 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null' EXIT

until curl -sf localhost:8000/v1/models >/dev/null; do
  kill -0 $SERVER_PID 2>/dev/null || { tail -60 runs/smoke-serve.log; break; }
  sleep 5
done
curl -s localhost:8000/v1/models       # the served id must equal configs/model/qwen14b.yaml's name

PRECEPTX_SERVE_ENV="$PWD/runs/serve_env.json" \
PRECEPTX_SERVING_SUBSTRATE=myriad-smoke \
bash scripts/myriad/shell.sh -c 'preceptx-pilot --model qwen14b \
  --base-url http://localhost:8000/v1 \
  --conditions C0,C4 --difficulties easy --seeds 0 \
  --root runs/smoke --concurrency 2' 2>&1 | tee runs/smoke-pilot.log
```

C4 is not optional: the pilot analysis contrasts a degraded condition against C0, so a C0-only
smoke runs its episodes and then fails in the report. The G1/G2/G3 numbers this produces are **not
evidence** — two episodes only prove the analysis path executes. Artefacts land under
`runs/smoke/05fcef471b8b9726…`, a different dataset hash from the re-gate's
`eddd19c654515bb2`, so the smoke cannot mix with or resume the real run.

**No `-P <project>` anywhere.** The 25 Aug 2026 session confirmed `qrsh -l gpu=1 -ac allow=L`
succeeds with no project code; the Free allocation is the default.

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
qsub scripts/myriad/pilot.sh                                   # submit
qstat                                                          # running jobs
qstat -f -j <job-id>                                           # why is it still queued
qdel <job-id>                                                  # stop it
jobhist                                                        # finished jobs (module load userscripts)
tail -f precept-pilot.o<job-id>                                # live log; -j y merges stderr in
```

Useful `-v` overrides: `TIER` (Hydra model group), `ATTEMPT=2` (the one permitted retune),
`SERVE_TIMEOUT` (default 1800 s, sized for a cold weights cache), `RUNS_ROOT`, `PORT`, `VENV`,
`SIF` (a different container image) and `APPTAINER_MODULE` (if UCL renames the module).
Add `-m be -M <email>` to be told when it starts and ends.

**Both jobscripts enter the container themselves.** They re-exec into it once, at the top, rather
than wrapping each command — `pilot.sh` launches `serve.sh` in the background and traps `$!`, so
two separate `apptainer exec` calls would put them in different process namespaces and the trap
would name the wrapper rather than vLLM. Re-execing keeps both halves in one process tree in one
container: `serve.sh` sees `APPTAINER_CONTAINER` already set and does not nest. Neither jobscript
will pull an image — that is `prefetch.sh`'s job, on a login node, because pulling while holding an
A100 spends GPU allocation on network I/O. Submit before prefetching and the job exits immediately
saying so.

`serve.sh` alone still works when you want a long-lived endpoint to poke at by hand; tear it down
with `qdel`, since the client's `close()` only drops local HTTP connections.

**The substrate label is automatic.** `pilot.sh` derives `PRECEPTX_SERVING_SUBSTRATE` from
`nvidia-smi`, so the manifest records the card that actually served rather than the class that was
requested. Cluster data is therefore never poolable with `local-lmstudio` pilot data by accident —
which is the point, since the local 4-bit G1 reading is indicative only and the **verdict of record
is the bf16 re-gate run here**.

## 9. The end-to-end run plan — everything left on the queue

**The gate track is closed.** E3 returned `fallback` on 29 August (`af50c7c12d65540f`, job 238085);
`PREREGISTRATION.md` §6 permits no attempt 3 and no second rung-2 attempt, and the verdict is frozen
at `runs/rq1/af50c7c12d65540f/`. **Nothing below is a re-gate.** Every run here is either a declared
amendment (§8b), a pre-declared causal arm (§8c), or a characterisation cell — so all of them use
`DRIVER=preceptx-rq1` or `preceptx-rq3b`, never `preceptx-pilot`, which would emit a verdict there is
no ledger left to spend.

Budget is not the constraint: the allocation holds multiple concurrent jobs and the quota is 39.5 of
1024 GiB. **The constraint is the two offline steps that gate two of the jobs, and the freeze-writing
that turns returned data into results.** Submit wide, then write while the queue works.

### 9.0 Order of operations

| # | Step | Where | Blocks |
|---|---|---|---|
| 1 | Commit the working tree, push | laptop | *everything* — the run must be at a real merged SHA, and `test_no_committed_manifest_records_a_dirty_tree` only evaluates on a clean tree |
| 2 | `git pull` on Myriad | login node | every `qsub` below |
| 3 | `fetch.sh af50c7c12d65540f` then `preceptx-calibrate --dataset-hash af50c7c12d65540f` | login node, no GPU | **R5 (RQ3b)** and **R6 (RQ3a judge)** — both need the joblib, which is gitignored and must be refit on the cluster |
| 4 | Submit R1–R4 | login node | nothing; they are independent |
| 5 | Submit R5, R6 once step 3 has produced `runs/af50c7c12d65540f-calibration/` | login node | — |
| 6 | Re-freeze RQ3a on the now-clean tree (below) | laptop | CI going green |

**Diff every dry-run hash against the table before submitting.** A hash that does not match means the
config moved since this was written, and the run would write to a directory nothing here describes.

### 9.1 The runs

Common preamble on every job: `cd ~/Scratch/precept-research && git pull`. All of R1–R4 run at
`--max-steps 50`, matching the E3 verdict of record and its logged deviation (§6, D29) — one task
parameterisation across the whole family.

#### R1 — the A2 length-holding content swap *(PREREGISTRATION §8b A2; declared, decision rule fixed)*

Arbitrates the two readings of C4's advantage. 40 whitespace tokens is C4's measured median delivered
length exactly (C4: 40 tokens / 220 chars; a 40-token prefix of C0: 40 tokens / 217 chars), and
whitespace tokens of the delivered message is the covariate `_length_matched` already stratifies on.
The cap is a **prefix**, so it keeps every number and severs the directive — the exact mirror of
dropout, which destroys the numbers and leaves the directive readable.

```bash
qsub -N precept-a2-lengthswap -l h_rt=4:00:00 -v DRIVER=preceptx-rq1 scripts/myriad/pilot.sh \
  --conditions C0,C1 --difficulties easy,hard --seeds "$(seq -s, 0 9)" \
  --max-steps 50 --c1-max-tokens 40 --no-analysis
```

**Expect `1f0d58944caa7fc3`** (sweep `7b65f40f633311a9`), 40 cells, 4,000 calls, ~35 min. The dry run
prints `channel: {...} (NON-DEFAULT)` — if it does not, the flag did not reach the sweep and the run
would silently be a duplicate of the real C1. **Read it on `action_agreement.csv`, not on success**;
the decision rule is in §8b and the arm is powered for the agreement limb.

#### R2 — serialisation A/B on C0 *(characterisation; the cheapest test of the mechanism's scope)*

The E3 mechanism says the receiver does not read the pose. Every episode behind that claim is
`numeric`. This asks whether that is a property of the *receiver* or of the *serialisation*, using the
same agreement limb and no new code — and it is the first question a reviewer asks about a
state-blindness finding. `grid` and `nl` already exist in `configs/serialisation/`.

```bash
qsub -N precept-serialisation-ab -l h_rt=6:00:00 -v DRIVER=preceptx-rq1 scripts/myriad/pilot.sh \
  --conditions C0 --serialisations numeric,grid,nl --difficulties easy,hard \
  --seeds "$(seq -s, 0 9)" --max-steps 50 --no-analysis
```

**Expect `a86599bae274eac1`** (sweep `9d09bbcece5978f8`), 60 cells, 6,000 calls, ~55 min. The
`numeric` third is seed-paired against the E3 cell's C0 arm, so the contrast is within-run *and*
anchored to a frozen result.

#### R3 — the 32B capability row *(characterisation; the other half of the same question)*

Is the state-blindness a 14B limit? Same cell as the E3 C0 arm, one model larger, so the agreement
limb reads the two directly against each other. **Needs an 80 GB A100 — `-ac allow=U,V` on the qsub
line overrides the jobscript's `allow=L` directive**, the same mechanism `-ac allow=EF` uses for the
8B tier.

```bash
qsub -N precept-32b-c0 -l h_rt=8:00:00 -ac allow=U,V -v TIER=qwen32b,DRIVER=preceptx-rq1 \
  scripts/myriad/pilot.sh --conditions C0 --difficulties easy,hard \
  --seeds "$(seq -s, 0 9)" --max-steps 50 --no-analysis
```

**Expect `86b89727699c88fd`** (sweep `64e8181c8c95b825`), 20 cells, 2,000 calls, model
`Qwen/Qwen3-32B`@`9216db5`. Time it generously: 32B bf16 is slower per token and the model load alone
is longer. This also supplies the cluster row E2 (DSE-005) never got.

#### R4 — the A3 thinking probe *(appendix; **the declared seed-pairing has to change**)*

`§9b` as written paired this against A2 on seeds 0–11. **A2 ran on the T load, and the task is now the
bar** — that pairing no longer holds anything constant except the seed integer. Re-pair it against the
E3 cell's C0 arm instead: same task, same budget, same seeds, decoding as the only changed variable.
Still an appendix probe and never inside the C0→C4 gradient: thinking is a capability manipulation,
and Qwen discourages greedy decoding in thinking mode, so it cannot meet the determinism standard.

```bash
qsub -N precept-a3-thinking -l h_rt=8:00:00 -v DRIVER=preceptx-rq1 scripts/myriad/pilot.sh \
  --conditions C0 --difficulties easy,hard --seeds "$(seq -s, 0 9)" \
  --max-steps 50 --thinking --no-analysis
```

**Expect `0ea4878b6e97f59f`** (sweep `a9940b95df220708`), 20 cells, 2,000 calls. Budget roughly four
times a non-thinking episode: a thinking trace is several times the tokens, and `max_tokens` rises to
2048 to fit it.

#### R5 — RQ3b, the causal gate *(PREREGISTRATION §8c; **direction declared before submission**)*

Four arms over one grid — gate-active, matched-firing-rate, random-trigger, off. The prediction is a
**null**, and it is on the record: a gate that blocks low-information handoffs cannot repair a
receiver that ignores the state it is handed. What would falsify it is gate-active beating
matched-random with an interval excluding zero.

Needs step 3 first. The threshold is **imported, never re-derived** from the arms' own outcomes.

```bash
# login node, after fetch.sh — no GPU:
uv run preceptx-calibrate --dataset-hash af50c7c12d65540f
# -> fail AUROC 0.906 (thr 0.977), cosine AUROC 0.766 (thr -0.647), firing rate 0.200 at n=3419

qsub -N precept-rq3b -l h_rt=8:00:00 -v DRIVER=preceptx-rq3b scripts/myriad/pilot.sh \
  --calibration runs/af50c7c12d65540f-calibration/calibration.json \
  --calibration-dataset af50c7c12d65540f --statistic cosine \
  --conditions C0,C4 --difficulties easy --seeds "$(seq -s, 0 9)" --max-steps 50
```

**Expect four datasets** off sweep `8c2f87cff96e6232`: `3f34698d1fdf06d8` (active),
`7e7bb390464b9179` (matched_random), `44b3ba3826d739fc` (random_trigger), `8658f40ba8eba68d` (off).
20 cells per arm, **8,000 calls total** — four times the printed 2,000 — plus re-prompt retries on the
two blocking arms. `--statistic cosine` deliberately: it is the probe-independent one, so a positive
result could not be dismissed as the probe grading its own homework, and at AUROC 0.766 on this corpus
it is finally strong enough to gate on (it read 0.569 on the C0-only corpus). **Easy only**: hard has
almost no headroom on this task (C0 1/10, C3 0/10 on the E3 cell), so pooling would dilute a real
effect into a floor.

#### R6 — the RQ3a judge replication *(the last unrun arm of the H5 comparison)*

Unchanged in substance from the old §9c, with one amendment: **point `--transfer` at the E3
calibration**, per §8b **A5**. The offline arms are frozen and cost zero model calls; this is the only
part needing a GPU. Until it runs, `judge` and `agreement` stay `null` and **no comparison to the
published 53.5 % / 14.2 % is stated**, because those figures come from LLM-judge procedures and
nothing in the current table is one.

```bash
# corpus must already be on disk: scripts/myriad/prefetch.sh with RQ3A_ROOT set.
# cost it first on a login node — no GPU, no model calls:
#   preceptx-rq3a --root ~/Scratch/rq3a --corpus traceelephant --judge --dry-run
qsub -N precept-rq3a-judge -l h_rt=8:00:00 \
  -v DRIVER=preceptx-rq3a,RQ3A_ROOT=$HOME/Scratch/rq3a scripts/myriad/pilot.sh \
  --corpus traceelephant --transfer runs/af50c7c12d65540f-calibration
```

**Upper-bound judge calls, measured not guessed:** 3,428 for TraceElephant (220 traces, 2,488
handoffs) and 4,380 for Who&When (184 traces, 3,505 handoffs) — upper bounds because step-by-step
stops at its first yes. The calls are prefill-dominated, each carrying a whole transcript, so run
TraceElephant first and time it before committing to Who&When. The driver refuses a `--transfer`
directory it cannot load the statistic out of, so a mistake there costs a second rather than the
reservation.

### 9.2 What each run has to produce before it counts

Every returned dataset follows the same close-out, and none of it needs a GPU:

```bash
scripts/myriad/fetch.sh <hash>
uv run preceptx-analyse --dataset-hash <hash>     # rq1.json, action_agreement.csv, scores.parquet
uv run preceptx-rq2     --dataset-hash <hash>     # where a condition contrast exists
```

then a freeze directory under `runs/rq1/<hash>/` carrying `manifest.json`, `summary.json` and a
README with the full reading, a closed row in `lineage.csv`, an `docs/EXPERIMENTS.md` results-log
entry, and a design-log entry **only where the run changes an interpretation**. The pattern is
`runs/rq1/af50c7c12d65540f/`; copy its shape.

### 9.3 Owed on the laptop, after the commit

```bash
uv run preceptx-rq3a --root ~/data/rq3a --corpus traceelephant \
  --transfer runs/af50c7c12d65540f-calibration --out runs/rq3a/traceelephant
uv run preceptx-rq3a --root ~/data/rq3a --corpus who_and_when \
  --transfer runs/af50c7c12d65540f-calibration --out runs/rq3a/who_and_when
```

This does two jobs at once: it clears `git_dirty: true` from both RQ3a manifests — stale for four
sessions, and now guarded by `test_no_committed_manifest_records_a_dirty_tree`, which **skips on a
dirty tree and bites in CI** — and it executes amendment **A5**, re-fitting the transferred statistic
on the better-calibrated corpus. Both old and new numbers go in the re-freeze note. **CI will not be
green until this runs on a clean tree**; that is the guard working, not a regression.


## 10. Getting the results back

`runs/` is gitignored on both ends, so nothing leaves the cluster on its own. **Use
`scripts/myriad/fetch.sh`, not a hand-rolled `rsync` or `tar`.** The 29 Aug results bundle was
assembled by hand, took only `runs/<hash>/*.parquet`, and left both v9 arms' manifests behind — the
one artefact that cannot be rebuilt locally, because `preceptx-analyse` reconstructs the *analysis*
(encoder revision, probe config) and not the model revision, the exact command or the serving
environment.

```bash
# Every run's manifests + summaries + serve_env - small, and the part you cannot regenerate.
scripts/myriad/fetch.sh

# One run, with its Parquet, staged where preceptx-analyse looks for it.
scripts/myriad/fetch.sh 86ecbbdf35322dc3
uv run preceptx-analyse --dataset-hash 86ecbbdf35322dc3
```

`HOST` defaults to the `myriad` ssh alias from §2; set `HOST=user@myriad.rc.ucl.ac.uk` without one.

Deliberately excluded: `handoffs.jsonl`, probes and embedding caches. All are large and
regenerable, and none are committed. Parquet is regenerable only by re-running the sweep on a GPU,
so it comes for a named hash and is skipped otherwise — one 96-part dataset should not arrive every
time someone fetches a manifest.

**Run it after every job, including the G1 confirmation.** The confirmation writes a *new* manifest,
and a gate-defining run whose manifest lives only in `~/Scratch` is one purge from unrecoverable.

## 11. Verified on the cluster, and what is still open

Checked in the first live session, **25 August 2026** (`login12`, `node-l00a-006`):

- [x] **`-P <project>` is not needed.** `qrsh -l gpu=1,h_rt=0:20:0,mem=4G -ac allow=L` was granted
      with no project code. The Free allocation is the default for UCL internal users.
- [x] **The GPU class is right.** `-ac allow=L` yielded an **NVIDIA A100-PCIE-40GB**, driver
      **550.127.05**, CUDA 12.4, on `node-l00a-006`. The queue returned it in seconds.
- [x] **Compute nodes have outbound internet — untested, and now irrelevant.** `prefetch.sh` pulls
      the image, the weights and the encoder on a login node, which demonstrably does (PyPI, Hugging
      Face and GitHub all returned 200). `pilot.sh` still warms the encoder before the sweep, so a
      node with no route out fails in seconds rather than after a full GPU hour.
- [x] **The CUDA module question is closed by the container.** `CUDA_MODULE` defaults to `none`;
      torch's bundled `cu12` libraries are the CUDA userspace and `--nv` injects the host driver.
      `cuda/12.2.2/gnu-10.2.0` does exist on Myriad if a bare run ever needs it.
- [x] **glibc is 2.17 on login *and* compute.** RHEL 7.9 throughout. This is what forced the
      container; see §6.
- [x] **Docker Hub is reachable from a login node** and the container runs. `prefetch.sh` pulled
      the digest-pinned image (366 MB compressed as a SIF), built the venv inside it, and the
      weights and encoder landed in `~/Scratch/hf-home` (28 GB). vLLM 0.18.1 reached engine
      initialisation on the A100 with torch 2.10.0+cu128 against driver 550.127.05 — so the
      glibc, CUDA and driver questions are all answered in the affirmative.
- [x] **The host environment leaks into the container.** `default-modules/2018` loads
      `compilers/intel/2018/update3`, which exports `CC=icc`; Apptainer passes the environment
      through, and torch/Triton JIT-compile a CUDA support module at engine start using `CC`
      verbatim. `container_toolchain` now pins `CC=gcc`/`CXX=g++` and clears `PYTHONPATH` on entry
      (DSE-053). Assume any host variable is present inside the container unless overridden.
- [x] **Apptainer is available** as `apptainer/1.2.4-1` on both login and compute nodes, with
      `singularity-env/1.0.0` alongside it. UCL's module points `APPTAINER_CACHEDIR` at Scratch but
      its build directory at `/run/user/<uid>`, a small RAM-backed tmpfs — `_common.sh` overrides
      `APPTAINER_TMPDIR` onto Scratch, since a ~1 GB image pull through tmpfs can fail on space.

Still open:

- [ ] **That Qwen3-14B bf16 loads and serves.** vLLM 0.18.1 now gets as far as engine
      initialisation against driver 550.127.05, which closes the CUDA-compatibility question, but
      no weights have been resident yet. `--gpu-memory-utilization 0.90` on a 40 GB card for a
      ~28-30 GB model is the next thing to be proven, and Triton's first JIT compile will be slow.
- [ ] **That the request path works end to end.** Two removed vLLM request fields were caught by
      inspection rather than by a run (DSE-052); `chat_template_kwargs` and the `<think>`-block
      guard are the two remaining client-side assumptions that only fire once a model responds.
- [ ] Wallclock actually needed for the 40-episode E3 cell at bf16. `h_rt=6:00:00` is a guess from a
      local 4-bit run (5 episodes in ~7 min) and should be replaced with a measured number.
- [ ] Queue latency for `-ac allow=L` under load. One grant in seconds is not a distribution; if the
      wait ever bites, the 8B tier on `EF` has 19 nodes and is the pressure valve.

Two items are closed by construction rather than by checking: the served revision can no longer
disagree with the manifest (it is read from the tier config), and the venv can no longer diverge
from the lockfile (`uv sync` into the repo's `.venv`, which is what every script activates).

## 12. Cost

Nothing here is billed. The Free allocation trades queue latency for cost, which suits development
and the pilot; the three-monthly priority allocation is worth saving for the main RQ1 sweep. Every
model call in this project is either local open-weight inference or the Myriad allocation — no
hosted API is ever called.

## 13. Changing these scripts

`tests/unit/scripts/test_myriad_container.py` runs the real scripts against a fake cluster: a stub
`apptainer` on PATH that records how it was invoked and then runs the payload in-process with
`APPTAINER_CONTAINER` set, which is what the real one does from the scripts' point of view. It
needs no Apptainer, no GPU and no network, and it runs in the normal `pytest` tier.

It exists because these scripts only ever execute somewhere a mistake costs a queue wait and an
A100 rather than a red test. Every case in it guards a defect that actually shipped:

| Test | Incident |
|---|---|
| `test_the_script_enters_the_container_exactly_once` | a second `apptainer exec` would break `pilot.sh`'s trap on `$!` |
| `test_no_nv_flag_without_a_gpu` | `--nv` on a login node fails looking for absent driver libraries |
| `test_home_is_bound_resolved_and_the_working_directory_survives` | `$HOME` is a symlink into `/myriadfs`; a dropped cwd sends artefacts to a read-only `/runs` |
| `test_leaked_intel_compiler_is_overridden_inside_the_container` | `CC=icc` from `default-modules/2018` killed vLLM at engine start (DSE-053) |
| `test_serve_env_is_valid_json_with_single_line_fields` | `head -1` under `pipefail` recorded `"glibc": "2.41\nunknown"` (DSE-052) |
| `test_a_spooled_jobscript_still_finds_common_sh` | SGE runs a spooled copy, so `BASH_SOURCE` named `/var/opt/sge/.../job_scripts/<jobid>` and `source "$HERE/_common.sh"` failed (DSE-054) |

Two details in there are load-bearing rather than incidental, and deleting either would leave a
test that passes against broken code:

- The stub `ldd` emits **twenty thousand** lines. `head -1` only closes the pipe early enough to
  kill the producer when there is more to write, so a one-line stub hides the SIGPIPE bug. macOS
  has no `ldd` at all, which would otherwise limit the guard to CI.
- `write_serve_env` is invoked under `set -euo pipefail`. **pipefail is the precondition** for that
  same bug: without it the pipeline reports success and the assertions pass either way.

`NV_SENTINEL` exists purely so both branches of the `--nv` decision are reachable from a test.
Nothing outside the suite should set it.

When you change a script, add the case that fails before your fix and passes after. The quickest
way to check a guard is real is to revert the fix and watch it go red.
