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
```

`scripts/myriad/shell.sh` is the entry point for everything that is *not* a jobscript — the dry-run
hash checks, the smoke, poking at a failed run. `bash scripts/myriad/shell.sh` drops you into an
interactive shell inside the container with `.venv` active; `-c '<cmd>'` runs one command and
exits. It applies `--nv` only where there is a GPU, so the same invocation works on a login node.


`prefetch.sh` now owns all four, in that order, and each step is idempotent. It checks `gquota`
before pulling anything, pulls the image, re-execs itself inside it, then builds `.venv` with
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
#    expect: cells 40, calls 2040, dataset hash 1c994b87bbca8257

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

# 5. Batch: the real re-gate.
qsub scripts/myriad/pilot.sh
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
`1c994b87bbca8257`, so the smoke cannot mix with or resume the real run.

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

## 9. Getting the results back

The pilot writes its verdict to Scratch and `runs/` is gitignored, so nothing leaves the cluster on
its own. Pull the two small directories the run of record actually consists of:

```bash
# From your laptop, after the job finishes. <hash> is printed in the job log ("sweep <hash> ...").
rsync -av --prune-empty-dirs \
  --include='*/' --include='manifest.json' --include='summary.json' \
  --include='pilot.json' --include='pilot.md' --include='serve_env.json' --exclude='*' \
  myriad:~/Scratch/precept-research/runs/ ./runs/myriad/
```

That is deliberately not the whole tree: `handoffs.jsonl`, the Parquet parts, probes and embedding
caches are large and regenerable, and none of them are committed. The manifest, the summary, the
gate report and the serving-environment capture are what a frozen result is made of.

Bring the Parquet dataset back too (`--include='*.parquet'`) when you want to re-run the analysis
locally rather than trust the on-node run — it is the same estimator either way, but a local re-run
is how you check that.

## 10. Verified on the cluster, and what is still open

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

## 11. Cost

Nothing here is billed. The Free allocation trades queue latency for cost, which suits development
and the pilot; the three-monthly priority allocation is worth saving for the main RQ1 sweep. Every
model call in this project is either local open-weight inference or the Myriad allocation — no
hosted API is ever called.

## 12. Changing these scripts

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
