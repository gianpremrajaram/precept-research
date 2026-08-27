# Changelog

All notable changes to **precept-research** are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). This repo has no semver public API; the
stable contract is the `RunManifest` and `ExperimentConfig` schemas (see CLAUDE.md). Behaviour- or
result-affecting changes get an entry; result-affecting changes also re-freeze the affected result.

## [Unreleased]

### Changed
- **Prompt surface v5 — the state carries an action history (`PROMPT_VERSION` v4 → v5, DSE-055).**
  The one retune PREREGISTRATION §6 permits, spent on the E3 attempt-1 failure. Result-affecting: it
  re-keys every dataset.
  - `sim/serialise.history_line(recent)` renders the last `HISTORY_WINDOW = 4` `(action, geodesic
    gain)` pairs plus their net as one line, e.g.
    `recent=((ROT+, +0.00), (ROT-, -0.00), …)  # … net +0.00 over the last 4`. **Four** is the
    smallest window that displays a period-2 limit cycle twice — the dominant attempt-1 failure.
    The line reports **fact, never advice**: a directive would make the retune a behavioural
    intervention rather than an observability fix and the two would be inseparable in the result.
    A unit test asserts the absence of instruction words.
  - `agents/graph.agent_a` now splits the prompt into the *scene* (what the channel may restrict)
    and the *history* (what it may not), appending the history **after** `apply_channel`. C3 windows
    B's view of the world; B's memory of its own actions is not the world. This keeps
    `apply_channel` touching exactly what it touched before, keeps the history identical across all
    three serialisations (a per-form whitelist could not), and preserves the standing invariant that
    `observation == state_str` in C0/C1/C2/C4.
  - `agents/prompts._SYSTEM_A` names the `recent` line. B's prompt is untouched.
  - **Accepted risk, recorded:** a receiver that can self-correct from its own history needs the
    message less, so this may depress CPVI. G2's CPVI half is where it will show; it currently
    carries +0.243 bits over a directional threshold.
- **The pilot cell widens to seeds 0–9** (`experiments/cli._PILOT_SEEDS`, 40 → **80 episodes**,
  4,080 upper-bound calls). Precision, not retune: no threshold and no estimator moves. Attempt 1's
  G1 read 2/5 (Wilson 95% [0.12, 0.77]; a design on the 0.5 threshold fails half the time) and G2's
  success half passed by exactly zero margin (2/10 vs 1/10 — one episode). Not optional stopping:
  the point estimate lies *below* the threshold, so added n moves the expected verdict toward FAIL.
- **`sim/actions.detect_stuck` compares the window's endpoints instead of its span**, window 3 → 5
  states (4 actions, so a period-2 cycle closes twice; an odd window cannot tell `N,S,N` from one
  net step north). The span form detected only *immobility* and scored `stuck=False` for all 18
  handoffs of an episode alternating `N,S,N,S` against a wall, and for one pushing `E` 33 times into
  a wall it could not pass (contact jitter exceeds `move_eps`). **Diagnostic only** — no gate reads
  `stuck`, and `graph.py` exits on success or budget alone — so this changes what a run records
  about itself, never what it does.
- **G3's truth set excludes the v5 action-history line** (`experiments/pilot._geometry_of`,
  `sim/serialise.HISTORY_PREFIX`). Found by review before attempt 2 ran; caught nothing in attempt
  1, which predates v5.
  - `_record_grounding` builds its truth set from every number the sender was shown, which was
    geometry-only until v5 put per-action geodesic gains into `state_str`. With `g3_abs_tol = 0.5`
    and gains clustering in 0-1.5, a fabricated small-magnitude geometric claim then matched a
    gain and scored *grounded*: on a synthetic record, a message asserting an offset of 0.85 that
    no wall, slit, load or goal coordinate supports reads **0.0** against the geometry and **1.0**
    against geometry-plus-history.
  - The drift is **single-sided** — it can only credit, never penalise — so it degrades exactly the
    property G3 certifies, and PREREGISTRATION §6 fixes the construct as "match true geometry".
    Excluding costs the reverse error (a message correctly quoting a gain scored ungrounded); that
    is the direction a gate should fail in, and A's prompt asks for position and intent, not for
    gains.
  - Keyed on `HISTORY_PREFIX`, exported from the module that owns the line's shape, so the two
    modules cannot drift apart silently. Regression test pins both directions.
- **The per-difficulty step budget stays at 2.5 × the oracle optimum** (18/33/33). All 34 attempt-1
  failures spent their full budget, which reads as starvation and is not: mean geodesic distance
  still to run at the end of a failure was **7.02** against a goal radius of 0.8, only **1 of 34**
  ended within 1.5 of the goal, and every success finished in 8–12 steps of an 18-step budget. More
  steps buy more cycling. Recorded because the rejection is evidence, not an omission.
- **Myriad jobs run inside an Apptainer container** (`scripts/myriad/_common.sh`,
  `prefetch.sh`, `serve.sh`, `pilot.sh`, `shell.sh`, `docs/myriad.md`, DSE-051). The first live cluster session
  (25 Aug 2026) found Myriad is RHEL 7.9 / **glibc 2.17** on login *and* compute nodes, while every
  wheel in `uv.lock` is `manylinux_2_28` or newer. `uv sync` cannot build a working environment on
  a bare node — not only torch and vLLM but pandas, pyarrow, scipy and scikit-learn have no
  compatible wheel. **`uv.lock`, `pyproject.toml`, `src/` and every schema are unchanged**: the
  lock now executes where its wheels are valid instead of being moved backwards.
  - *Why not downgrade.* The newest torch with a glibc-2.17 wheel is 2.6.0 and the newest vLLM
    pinning it is 0.8.5 (April 2025) — and that still leaves pandas/pyarrow/scipy/scikit-learn
    unsolved. scipy and scikit-learn sit directly under the CPVI estimator, so moving them would
    make measurements taken before and after the change incomparable, and would have forced a
    re-freeze of E3-local for a sixteen-month-old server.
  - `CONTAINER_SOURCE` is pinned by **digest**, not tag:
    `docker://python@sha256:a8677eb0…32938d`. Debian bookworm carries glibc 2.36, clearing
    `manylinux_2_28` and vLLM's `manylinux_2_31`. The **full** image, not `-slim`, because
    `manifest.git_sha()` shells out to `git` and raises when it is missing — a git-less image would
    have failed *after* the episodes were paid for. Nothing is built, so no `--fakeroot` is needed.
  - `enter_container` **re-execs the script** into the container once, rather than wrapping each
    command. `pilot.sh` launches `serve.sh` in the background and traps `$!` to kill the server on
    the scheduler's wallclock SIGTERM; two `apptainer exec` calls would put them in different
    process namespaces and the trap would name the wrapper, not vLLM. Re-execing keeps serve and
    drive in one process tree in one container — `serve.sh` sees `APPTAINER_CONTAINER` set and does
    not nest — so every line downstream, including the trap, is untouched.
  - `--nv` is passed **only when `/dev/nvidiactl` exists**: `prefetch.sh` runs on a login node,
    where `--nv` fails looking for driver libraries that are not there.
  - `$HOME` is a symlink into `/myriadfs`, so the bind is resolved-source-to-original-destination
    (`$(readlink -f "$HOME"):$HOME`) and `--pwd "$PWD"` is passed explicitly. Without both, `#$ -cwd`
    plus a relative `RUNS_ROOT` would send artefacts to a read-only `/runs`.
  - `APPTAINER_TMPDIR` is forced onto Scratch. UCL's module points the build directory at
    `/run/user/<uid>`, a small RAM-backed tmpfs on the login nodes, where a ~1 GB pull can fail.
  - `prefetch.sh` now owns image → venv → weights → encoder, in that order, each idempotent. The
    `gquota` check moved **host-side**, since `gquota` does not exist inside the image and what it
    guards against is decided before anything is pulled. `ensure_venv` rebuilds a `.venv` that
    cannot import the locked wheels — exactly the state a bare-node `uv sync` leaves behind — and
    asserts pandas, pyarrow, scikit-learn and torch import, which is what proves the environment is
    the container's rather than the host's.
  - `scripts/myriad/shell.sh` is the container entry point for everything that is *not* a
    jobscript — the dry-run hash checks, the two-episode smoke, poking at a failed run. It reuses
    `require_image`/`enter_container` and activates the venv, so `bash scripts/myriad/shell.sh -c
    '<cmd>'` replaces hand-typing a 150-character `apptainer exec --nv --bind … --pwd …` line on a
    cluster where a mistyped one costs a queue wait. Interactive when given no arguments.
  - `require_image` **asserts, never pulls**: pulling an image while holding an A100 would spend GPU
    allocation on network I/O. Submitting before prefetching exits immediately with the fix.
  - `serve_env.json` gains `container_source`, `container_sif`, `container_sif_sha256` and `glibc`,
    so the run manifest records both what was asked for and what was on disk. `ServeEnv.values` is
    a free `dict[str, str]`, so this is not a schema change.
  - `CUDA_MODULE` now defaults to `none` — there is no module system inside the image, and torch's
    bundled `cu12` libraries plus `--nv` are the whole CUDA story. The locked stack is `cu12`
    throughout (`nvidia-cublas-cu12` 12.8.4.1) against driver **550.127.05 / CUDA 12.4**, which
    CUDA minor-version compatibility covers. The override survives for a future non-RHEL7 node.
  - `docs/myriad.md` §6 and §7 rewritten; §10 now records what the live session verified —
    **no `-P` project code is needed**, `-ac allow=L` yields an A100-PCIE-40GB — and the four items
    still open, of which "vLLM 0.18.1 actually runs against driver 550.127.05" is the load-bearing
    one.

- **vLLM structured-output API migration** (`serving/client.py`, `scripts/myriad/serve.sh`,
  `_common.sh`, `RESEARCH_ROADMAP.md`, DSE-052). The first cluster serve attempt (25 Aug 2026)
  exited 2 at argument parsing: `vllm: error: unrecognized arguments: --guided-decoding-backend
  xgrammar`. vLLM removed the whole `guided_*` family in **v0.12.0**; the locked version is 0.18.1.
  Dataset hashes are unchanged (`1c994b87bbca8257` / `05fcef471b8b9726`) — `ServingConfig` and
  `structured_mode` live on `SweepManifest`, not `SweepConfig`, so `sweep_hash` never saw them.
  - `scripts/myriad/serve.sh` now passes `--structured-outputs-config.backend`. Explicit `xgrammar`
    is kept rather than falling back to the default `auto`, which selects a backend per request:
    the constraining engine is a property of the run of record, not something to leave to the
    server's judgement.
  - `LLMClient._structured_kwargs` sends `{"structured_outputs": {"json": schema}}` in place of
    `{"guided_json": …, "guided_decoding_backend": …}`. **This was the more dangerous of the two.**
    The CLI flag fails loudly at parse time, before the model loads; the request field would have
    failed *after* the weights were resident and the episodes were being paid for — and if vLLM
    ignores unknown `extra_body` keys rather than rejecting them, it would not have failed at all,
    it would have decoded the action channel unconstrained and produced a passing-looking run.
  - `ServingConfig.guided_decoding_backend` is **removed**. The backend is no longer a per-request
    choice, so a client-side field claiming one would have been written into every manifest as
    provenance it no longer controls. It is replaced by `structured_outputs_backend` in
    `serve_env.json`, written by the server that actually selects it. `ServeEnv.values` is a free
    `dict[str, str]`, so the manifest schema is unchanged.
  - `structured_mode` keeps its `guided_json` value deliberately. It names which endpoint dialect
    the branch speaks, not the field, and it is a `--structured-mode` CLI choice recorded in every
    `SweepManifest`; renaming it would churn a config contract for a label. The frozen E3-local
    data used the `response_format` branch, which is untouched — **no re-freeze**.
  - Six test stubs dispatched on `b"guided_json"` to tell an action call from a message call, so
    they kept passing against the removed field. They now sniff `b"structured_outputs"`, and
    `test_structured_parses_json_object` asserts the removed keys are absent rather than
    only asserting the new one is present.

### Fixed
- **SGE jobscripts resolve the checkout, not the spool directory** (`scripts/myriad/pilot.sh`,
  `serve.sh`, `tests/unit/scripts/test_myriad_container.py`, `docs/myriad.md`, DSE-054). The first
  real `qsub scripts/myriad/pilot.sh` (job 212796, 26 Aug 2026) exited 1 in the same second it
  started, on `source "$HERE/_common.sh"`.
  - *Cause.* SGE does not execute the submitted file: it spools a copy to
    `/var/opt/sge/<node>/job_scripts/<jobid>` and runs that. `${BASH_SOURCE[0]}` therefore names the
    spool directory, where none of `scripts/myriad/` exists. Every previous invocation had been
    `bash scripts/myriad/<script>` — run in place — so `qsub` was the first execution of this path.
  - *Blast radius beyond the source line.* `HERE` also supplies `enter_container "$HERE/pilot.sh"`,
    which would have re-exec'd a spool path with no bind mount inside the container. `REPO_ROOT`
    derives from `HERE` too, and with it `VENV`, `PRECEPTX_SERVE_ENV` and the tier-config check.
  - *Fix.* One line per script: `[[ -f "$HERE/_common.sh" ]] || HERE="$PWD/scripts/myriad"`.
    `#$ -cwd` lands the job in the submit directory, which is the repo root — an assumption the
    scripts already make, since `RUNS_ROOT` defaults to the relative `runs`. Running a script in
    place keeps the first branch, so interactive behaviour is byte-identical. A second guard
    fails loud when the fallback cannot resolve either — a job submitted from outside the repo
    root exits 1 naming the fix (`submit from the repo root`) instead of repeating the opaque
    `_common.sh: No such file or directory`.
  - `serve.sh` carried the same defect at its own `source` line and is documented as separately
    submittable (`qsub scripts/myriad/serve.sh`); it is fixed in the same pass rather than left as
    a second identical incident waiting for its first `qsub`.
  - *Guard.* `test_a_spooled_jobscript_still_finds_common_sh` copies `pilot.sh` to a spool-shaped
    directory and runs it from the repo root with a bogus `TIER` — the first exit *after* the
    source, so reaching it proves the source resolved. Verified to go red with the fix reverted.
- **Myriad's Intel compiler leaking into the container** (`scripts/myriad/_common.sh`, DSE-053).
  The second cluster serve attempt got vLLM 0.18.1 all the way to engine initialisation on the
  A100 and then died with `InductorError: FileNotFoundError: [Errno 2] No such file or directory:
  'icc'`. Myriad's login shells load `default-modules/2018`, which pulls in
  `compilers/intel/2018/update3` and exports `CC=icc`; Apptainer passes the host environment
  straight through, and torch/Triton JIT-compile a small CUDA support module at engine start,
  reading `CC` from the environment with no existence check (`triton/runtime/build.py`).
  - `container_toolchain` pins `CC=gcc` and `CXX=g++` **unconditionally** on container entry.
    `${CC:-gcc}` would have been worse than useless: the leaked value is already set, so the
    default never fires and the broken state is preserved exactly.
  - It is called from `enter_container`'s already-inside branch, which every script reaches
    exactly once on the way in, so the fix applies to `serve.sh`, `pilot.sh`, `prefetch.sh` and
    `shell.sh` from one place and stays idempotent under `pilot.sh`'s nested `serve.sh`.
  - A preflight asserts `gcc` and `g++` resolve, and names `-slim` as the likely cause if they do
    not. Without it a missing compiler surfaces as a forty-line Triton traceback minutes into
    startup, on the GPU, rather than as one line on a login node.
  - `LD_LIBRARY_PATH` is deliberately **not** touched — `apptainer --nv` manages it to expose the
    host driver libraries, and clearing it would break CUDA in order to fix a compiler.
    `PYTHONPATH`/`PYTHONHOME` are cleared: the venv is self-contained and nothing in this repo sets
    them, so a leaked value can only point at host site-packages built against glibc 2.17.
  - `serve_env.json` gains `cc`, the resolved absolute path of the compiler Triton will actually
    use. The generalisable lesson, recorded in `docs/myriad.md` §10: assume **every** host
    variable is present inside the container unless it is explicitly overridden.

- **`serve_env.json` fields truncated by a pipefail/SIGPIPE race** (`_common.sh`, DSE-052). The
  live run recorded `"glibc": "2.41\nunknown"` — a two-line JSON value. Under `set -o pipefail`,
  `ldd --version | head -1 | awk …` has `head` close the pipe, `ldd` die of SIGPIPE, and the
  pipeline report 141 even though `awk` already printed the right answer, so `|| echo unknown`
  appended a second line. Replaced with `awk 'NR==1{…}'`, which reads all of stdin and cannot lose
  the race. The `driver` field had the same latent bug and survived only because `nvidia-smi`
  produces one short line; it is fixed identically. Reproduced and verified before and after.

### Added
- **`scripts/diagnose_cycles.py` — terminal limit-cycle and confound diagnostic for a pilot dataset.**
  Written to answer a question `pilot.md` does not: did the v5 retune reduce the pathology it
  targeted? Takes one or more dataset directories and prints them side by side, so attempt 1 and
  attempt 2 are compared on one table rather than by eye across two reports.
  - **Cycle detection** walks back from the last action and measures the trailing period-1 or
    period-2 repeat. `MIN_CYCLE = 4` repeats, not 3: three would fire on `ROT+,ROT-,ROT+`, which is a
    plausible two-step correction rather than a policy that has stopped moving. A period-2 run of one
    repeated action is not double-counted as both periods.
  - **Per-condition step-level table** (`stuck_rate`, `mean_progress`, `y_binary_rate`) is the part
    that identifies a *confound* rather than a pathology: if degradation is acting as something other
    than information loss, the mangled conditions get less stuck and make more progress than the
    clean one. It does. C3, which degrades the observation rather than the message, moves the other
    way — that contrast is what the diagnostic exists to expose.
  - Reproduces the attempt-1 finding it was validated against (62 % of failed episodes ending in a
    cycle) and is the source of the E3 attempt-2 numbers in `docs/EXPERIMENTS.md`. Its output belongs
    in a results chapter, which is why it is a committed script and not a shell one-liner.
- **`scripts/check_rotation_need.py` — the rung-2 acceptance check, on CPU with no model in the loop.**
  PREREGISTRATION §6 fixes the rung-2 criterion as "the A\* optimum must contain ≥ 1 rotation and
  finish strictly inside budget, for every jittered seed at every difficulty"; this decides it in
  seconds, before any GPU time is spent.
  - **The geometric half** measures the T outline's y-extent over a full turn: **1.300 minimum at 0°,
    1.553 maximum at −146.8°**. A slit wider than 1.553 admits the load head-on whatever its
    orientation, which is what makes rotation *incapable of being necessary* rather than merely
    unused. Easy's 1.8 is such a slit; medium's 1.2 and hard's 1.1 are not.
  - **The behavioural half** runs a rotation-free policy (close the y gap to within the goal radius,
    then push east) against the real physics on each jittered seed. Easy: **10/10 solved**.
    Medium and hard: **0/10**.
  - Exits non-zero when any difficulty fails the criterion, so it can gate a re-gate submission.
    It currently returns REJECTED on easy, which is the defect rung 2 exists to close.
- **A test tier for the Myriad scripts** (`tests/unit/scripts/test_myriad_container.py`, DSE-053).
  Three tickets of container plumbing had been verified only by out-of-band stub harnesses, so
  nothing in the repo would have caught a regression in it. The suite runs the real scripts against
  a fake cluster: a stub `apptainer` on PATH that records its invocation and then runs the payload
  in-process with `APPTAINER_CONTAINER` set, which is what the real one does from the scripts'
  point of view. No Apptainer, no GPU, no network; it runs in the normal `pytest` tier.
  - Eight cases, each guarding a defect that actually shipped: enter-once (or `pilot.sh`'s trap
    stops naming vLLM), `--nv` only with a GPU, the resolved-`$HOME` bind and `--pwd`, the
    `CC=icc` override, the compiler preflight, the missing-image message, and `serve_env.json`
    being valid JSON with single-line fields.
  - Each guard was checked by **reverting its fix and confirming the test goes red** — which caught
    two tests that would otherwise have passed against broken code. The stub `ldd` emits 20 000
    lines, because `head -1` only kills the producer when there is more to write (and macOS has no
    `ldd`, which would have limited the guard to CI); and `write_serve_env` is invoked under
    `set -euo pipefail`, since pipefail is the precondition for the bug rather than decoration.
  - `NV_SENTINEL` (default `/dev/nvidiactl`) is a seam so both branches of the `--nv` decision are
    reachable. Nothing outside the suite should set it.
  - `docs/myriad.md` §12 records the rule for changing these scripts, and §7 now gives a **GPU-free
    way to verify the serve flag**: `vllm serve --help` cannot run on a login node, because
    building the parser instantiates `VllmConfig`'s defaults and raises `Failed to infer device
    type`. Reading `dataclasses.fields(VllmConfig)` needs no device.

- **RQ3a corpus loaders and the E9 substrate spike** (`data/logs.py`,
  `experiments/rq3a_load.py`, `scripts/fetch_rq3a.sh`, `docs/rq3a_schema_mapping.md`, DSE-041).
  The pre-planned fallback that can carry the dissertation alone had never been opened; its
  substrate was assumed from papers, not measured. All three corpora are now parsed from the real
  files and the assumptions are either confirmed with numbers or corrected in place.
  - `LogHandoffRecord` (Pydantic, `extra="forbid"`, `LOG_SCHEMA_VERSION` 1) is **separate from**
    `HandoffRecord` and physics fields are **absent, not nullable** — a log row can never be read
    as a degraded episode row. `trace_id` is the cross-fit grouping key, the exact analogue of
    `episode_id`; the leakage discipline is unchanged, only the name.
  - `LogTraceRecord` for corpora that publish a trace as one unsegmented transcript.
  - `mark_handoffs(agents)` returns `(receiver, is_handoff)` per step: a step is an inter-agent
    handoff when the component acting at `i+1` differs from the one at `i`, and the final step has
    no successor. Intra-agent tool turns are **kept** in the dataset — dropping them would shift
    the per-step base rate and make the handoff subset incomparable to the simulator's.
  - `load_traceelephant` reads the unzipped `data/<family>/<task>/` tree. `trace_id` is
    `family/task` so two families sharing a task id stay distinct groups.
  - `load_who_and_when` reads both parquet splits, resolves the two spellings of the correctness
    column (`is_correct` / `is_corrected`), falls back from `history[i].name` to `.role` for the
    `Hand-Crafted` split which carries no `name`, and sets `reconstructed_observation=True` on
    **every** row it emits.
  - `load_mast` emits trace-level rows only, and counts the non-failure class rather than assuming
    it.
  - `count_handoff_corpus` / `count_trace_corpus` produce the per-corpus counts table; failure
    counts are per *trace*, step counts per *step*. Never pooled across corpora.
  - `CorpusError` on a missing file or an unexpected shape — raised, never skipped, because a
    silently dropped trace changes every count downstream.
  - Loaders take **local paths only** and never touch the network, so the 20 unit tests run offline
    against hand-built fixtures mirroring the verified layouts, and CI never downloads 800 MB.
    `scripts/fetch_rq3a.sh` does the fetching out of band with plain `curl` against HuggingFace
    `resolve` endpoints.
  - **Three mapping decisions that change what a number means**, all recorded in
    `docs/rq3a_schema_mapping.md` §5: the observation is the *whole* context prefix rather than the
    last turn (truncating would shrink the state-only baseline and inflate CPVI); tool calls are
    part of the message (a `content: ""` step with populated `tool_calls` is TraceElephant's common
    case, and scoring it empty would repeat the local-pilot fail-open bug); and TraceElephant's
    `trace_failed` is derived from `tests_status` and **never** from the annotation triple, which
    means it is `None` on the 176 traces with no harness result rather than being filled in.
  - **Measured counts, which falsify a roadmap claim.** TraceElephant is **220 traces, 5,960 steps,
    2,488 inter-agent handoffs — and 0 non-failures**: every trace carries a `mistake_agent`
    annotation, only the 44 `swe-agent` traces carry `tests_status`, and 0 of those 44 pass. Roadmap
    §3.4's "380 executions of which about 220 are annotated failures … it ships non-failing
    executions too" is corrected in place. Who&When: 184 traces, 4,092 steps, 3,505 handoffs, 184
    failures, 0 non-failures. MAST: 1,642 traces, 1,237 failures, **405 non-failures (24.7%)** —
    the only two-class outcome of the three, and confounded with system identity (AG2 52.1%
    non-failure against OpenManus 3.3%), so a probe can score by recognising the system rather than
    by reading the message. Consequence for the design is in `docs/experiment_design_log.md`:
    DSE-042's counterfactual replay moves from an upgrade path to the load-bearing route to a
    two-class step-level *Y*.

- **Dataset identity carries the simulated world** (`sim/fingerprint.py`, `data/writer.py`,
  `experiments/sweep.py`, `experiments/runner.py`). The pre-registered retune lever sat outside the
  dataset hash: `_DIFFICULTY_SLITS`, `ArenaGeometry`, the T dimensions, `LOAD_MASS`, `GOAL_RADIUS`,
  `DAMPING`/frictions and `GridConfig.cell` are module constants, and `ExperimentConfig` carries
  `difficulty` only as a *label*. Widening a slit therefore left `dataset_hash_for` unchanged, so
  `run_grid` read the pre-retune `episode_id`s, logged `0 pending`, ran nothing, and let the driver
  re-report the old verdict as the post-retune result — silently, on the verdict of record, on the
  path PREREGISTRATION §6 and roadmap §3.1 prescribe.
  - `simulation_fingerprint()` returns a typed `SimulationFingerprint` grouped by owning module
    (slit map / arena / load / grid) plus `ENVIRONMENT_SCHEMA_VERSION`, an escape hatch for a
    behavioural change that leaves every constant identical. `digest()` is sha256 over sorted-key
    JSON, 16 hex — the same shape as `sweep_hash`.
  - `dataset_hash` gains `simulation_digest`, folded in exactly as `prompt_version` already is
    (`:w<digest>` suffix, empty default preserving the old value for callers with no world surface);
    `dataset_hash_for` never omits it.
  - **Deliberately not re-fingerprinted:** `ScenarioJitter`, `StepConfig`, `OutcomeConfig` and the
    step budgets are `SweepConfig` fields already inside `sweep_hash` (P0-2, P1-6) — hashing them
    twice would put one guarantee in two places. Derived values (`load.HALF_H`, `load.COG_Y`) are
    omitted as pure functions of dimensions that *are* hashed.
  - `SweepManifest` records the **payload beside the digest** (`simulation`, `simulation_digest`):
    the digest prevents an unsafe resume, the payload is the only thing that explains *why* an
    identity changed without the source tree to hand. `SWEEP_MANIFEST_VERSION` 1 → 2.
  - `run_grid._assert_simulation_matches` is defence in depth behind the hash, not the mechanism:
    a recorded fingerprint disagreeing with this process raises `ConfigError`. It deliberately does
    **not** fail closed on a *missing* manifest — the manifest is written when a sweep finishes, so
    its absence is the ordinary killed-at-wallclock case that resumability exists to serve.
  - `arena.slit_widths()` returns a copy, so the fingerprint observes the map without being able to
    mutate it.
  - **Re-keys every existing dataset.** `runs/local/*` and `runs/bench/smoke/*` are no longer
    resumable; their findings survive in `docs/EXPERIMENTS.md`, the committed `runs/bench/ladder.*`
    table is append-only, and both local pilots were indicative by pre-registration.
- **The server-side serving environment reaches the manifest** (`manifest.py`,
  `scripts/myriad/_common.sh`, `serve.sh`, `pilot.sh`). vLLM's and torch's versions and the physical
  GPU exist only in the server process on the compute node; `_TRACKED_DEPS` is client-side, and the
  values were previously echoed to the job log alone — recoverable on the run of record only by a
  human copying four lines out of `precept-pilot.o<jobid>`.
  - `write_serve_env` (in `_common.sh`, so serve/pilot cannot disagree on the path) writes tier,
    model, revision, vLLM, torch, GPU name, **driver version**, host, `JOB_ID` and capture time to a
    sidecar; `serve.sh` still echoes it so the job log stands alone.
  - `manifest.serve_env()` reads the sidecar named by `PRECEPTX_SERVE_ENV` into a typed `ServeEnv`
    (path, 16-hex digest over the bytes, values). `None` off the cluster — a local run has no
    separate server process to describe. A sidecar that is *named but unreadable or corrupt* raises
    `ManifestError`: that means the job wrote one and we lost it, which is not the same as absent.
  - `SweepManifest.serve_env` carries it; `pilot.sh` exports the path so a cluster run picks it up
    with no extra flag.
- **`BenchmarkInvocation` — provenance for a hand-launched ladder row** (`serving/benchmark.py`,
  `scripts/benchmark_models.py`). One model is served per GPU job, so DSE-005 rows are launched by
  hand with free-text `--model`/`--revision`, and the served checkpoint is exactly what no later
  check recovers — `/v1/models` carries no revision at all.
  - Written to `runs/bench/<run_id>/benchmark-invocation.json` **before the first model call**, so
    persistence is a precondition of serving rather than a courtesy afterwards; rewritten once on
    completion with `ended_at`, `exit_status` and artefact paths.
  - Carries tier, model, resolved revision, substrate, full argv and resolved args, git SHA **and
    dirty-tree flag** (a dirty tree means the SHA does not describe the code that ran), the
    simulation fingerprint and digest, the `ServeEnv` it ran against, host, `JOB_ID` and timestamps.
  - **A crashed run needs no handler**: it keeps the `exit_status: null` it was written with, which
    reads as "started, never finished" — accurate, and one fewer broad `except` than recording the
    failure explicitly would have cost.
  - `manifest.git_dirty()` added beside `git_sha()`.

- **DSE-044 — The second pre-registered length control** (`analysis/stats.py`,
  `experiments/rq1.py`). PREREGISTRATION §5 pre-registers *two* controls for the length/condition
  confound and states that both are reported; only the covariate one existed.
  - `overlap_restricted_contrast(value, covariate, treated, n_bins=3, min_per_cell=2)` stratifies
    episodes into equal-count quantile bins of the covariate and differences the two groups **only
    inside bins holding at least `min_per_cell` of each**, size-weighting across the retained bins.
    Returns `OverlapRestrictedContrast`: the delta, the unrestricted delta beside it, `n_bins` /
    `n_kept` / `n_total` (bookkeeping is part of the result — a delta read without them is
    unreadable), an `interpretable` flag, and a note.
  - **Quantile strata, not a nearest-neighbour caliper.** At the E3 cell's six episodes per
    condition a caliper has too little support and can collapse the comparison to one or two
    idiosyncratic pairs; coarse strata degrade visibly instead (`n_bins` falls). Where the two
    length distributions do not meet at all, the result is `interpretable=False` with a NaN delta
    rather than an extrapolated number.
  - Heavy ties collapse quantile bins by construction, so `n_bins` reports strata **retained**, not
    strata requested — pinned by a test.
  - `RQ1Result.length_matched` carries one `LengthMatchedContrast` per Ck, holding the restricted
    contrast on **both** success and CPVI. Both share one stratification (bins depend only on length
    and condition), so their `n_kept`/`n_bins` always agree. `RQ1Config` gains `length_bins=3` and
    `length_min_per_cell=2`. Persisted automatically via `write_rq1`'s `rq1.json`.
  - **Framed as an overlap-restricted, length-adjusted *sensitivity analysis*, never as a clean
    estimate of the channel effect with length removed** — the overlap region is a non-random subset
    of both arms. Carried in the docstring, `ANALYSIS_PROTOCOL["length_control"]` and
    PREREGISTRATION §5 so it cannot be quietly upgraded to a causal claim.
- **`scripts/myriad/prefetch.sh` — login-node pre-pull** (new script). `docs/myriad.md` §9 flags
  "compute nodes may have no outbound internet" as the single unknown that can write off a whole
  session; a GPU job that downloads is then not slow but dead, after the queue wait.
  - Pulls the tier's weights **and** the embedding encoder (`EncoderConfig`'s pinned default, so it
    cannot drift from what the analysis loads) into the Scratch caches a job reads.
  - Runs `gquota` first: the failure this guards against is a download that fills the quota and
    leaves the *next* job unable to write its own `.o`/`.e` files, which reads as a scheduler fault.
- **`scripts/myriad/_common.sh` — shared jobscript helpers** (new file). `resolve_tier` and
  `cache_to_scratch`, sourced by `serve.sh`, `pilot.sh` and `prefetch.sh`, so the prefetch cannot
  populate a cache the server does not read.
- **DSE-050 — `scripts/myriad/pilot.sh`, a single job that serves and drives** (new script). Nothing
  in the repo could run an experiment on the cluster: `serve.sh` starts vLLM on a compute node, and
  a login node running `preceptx-pilot --base-url localhost:8000` resolves `localhost` to the login
  node and finds nothing.
  - **One job, not two**, because co-locating serve and drive makes the endpoint a loopback address
    again — which is what `LLMClient` already assumes. Splitting them would require discovering the
    compute node's hostname, holding a port open between nodes, and waiting in the queue twice.
  - **`serve.sh` stays the single launch path.** `pilot.sh` runs it as a background child rather
    than duplicating the vLLM command line; because `serve.sh` `exec`s vllm, `$!` *is* the server,
    so one `trap ... EXIT` tears it down on success, failure and the scheduler's wallclock SIGTERM
    alike.
  - **Readiness wait with a liveness check**: polls `/v1/models` to a `SERVE_TIMEOUT` (default
    1800 s, sized for a cold weights cache downloading ~28 GB) but aborts immediately if the server
    process is gone, so a bad revision or an OOM fails in seconds rather than burning the timeout.
  - **The embedding encoder is warmed before any GPU time is spent.** `Featuriser` loads its encoder
    lazily, i.e. at *analysis* time — after every episode has run. On a node with no outbound
    network that failure would land at the end of a full GPU hour with the dataset already paid for.
  - **`PRECEPTX_SERVING_SUBSTRATE` is derived from `nvidia-smi`**, so the manifest records the card
    that actually served rather than the node class that was requested.
  - Grid axes are **not** named in the jobscript: they come from the CLI defaults, which are the
    pre-registered E3 cell, so the two cannot drift apart silently.
- **`docs/myriad.md` — cluster runbook and findings** (new doc). Access via the SSH gateway (keys
  mandatory from outside UCL since 23 March 2026) with a `ProxyJump` stanza; the node-class table
  (L = 4×A100-40 GB, U/V = 4×A100-80 GB, E/F = 2×V100, D = CPU) mapped to our tiers; the per-slot
  `mem` rule; wallclock caps by core count; the 1 TB shared home/Scratch quota; the uv + Python 3.11
  bootstrap; an ordered first-session runbook that does the **first vLLM launch interactively under
  `qrsh`**, not through the batch queue; and a §9 checklist of what is documented but **not yet
  verified on the cluster** — the exact CUDA module name, whether compute nodes have outbound
  internet, `allow=L` queue latency, and the wallclock the 40-episode E3 cell actually needs.
  - Records why **Kathleen is not usable** for this project (no GPUs, diskless, built for multi-node
    MPI), so it is not re-evaluated later.
- **DSE-045 — Gate retry-feedback template** (`agents/prompts.py`, `experiments/sweep.py`). Under
  greedy decoding a re-prompt is a fixed point: same prompt, same message, same statistic, same
  block, for every bounded retry. DSE-018 as written would have passed its unit tests and been
  vacuous live, measuring the cost of stalling rather than the value of blocking.
  - `GATE_FEEDBACK` instructs A to state the push direction, whether the load must rotate first and
    which way, and the goal direction — appended to A's user turn on a blocked retry and nowhere
    else. `prompt_a(state, gate_feedback=False)` is byte-identical to the previous `prompt_a(state)`
    (pinned by a test), so no existing dataset shifts.
  - `GATE_FEEDBACK_VERSION` is versioned **separately from `PROMPT_VERSION`** — the template is part
    of the RQ3b *treatment*, not the base task, so a wording change re-shapes the causal arm while
    leaving every ungated dataset untouched.
  - Recorded in `SweepManifest` but **deliberately not folded into `dataset_hash_for`**: the gate is
    unbuilt, so the template reaches no model today and hashing it would re-key every existing
    dataset over a string nothing reads. It must join the dataset hash when DSE-018 makes retries
    live — noted in the code and the design log so the step is deferred, not lost.
  - **Nonzero temperature on retry is rejected** (recorded in PREREGISTRATION §6 before any gate data
    exists): it breaks the determinism story mid-episode and confounds the gate's effect with an
    increase in sampling entropy, when H6's four arms are built to differ in one thing at a time.
- **`shellcheck` pre-commit hook** (`shellcheck-py`, the pip wrapper — no Docker, so it works in CI
  and a bare `pre-commit install`). Both jobscripts now carry real control flow and only ever run on
  the cluster, where a typo costs a queue slot rather than a test run. Both pass clean.
- **DSE-043 — Control tasks and probe selectivity** (`measure/pvi_cpvi.py`, `experiments/rq1.py`,
  `experiments/pilot.py`). Probe accuracy alone cannot separate "the representation encodes this"
  from "the probe learned the task" (Hewitt & Liang, EMNLP 2019), and at 1,536-dimensional
  concatenated features on pilot-scale N that is live — the held-out AUROC monitor does not address
  it.
  - `control_labels(y, seed)` draws i.i.d. random labels at `y`'s observed base rate,
    seed-reproducible; `control_task_cpvi(...)` re-estimates CPVI against them through the **same
    splitter and the same probe family**, so the comparison is like for like.
  - `CpviResult` gains `control_mean_cpvi` and `selectivity` (`mean_cpvi − control_mean_cpvi`);
    `ConditionSummary` gains `mean_control_cpvi` and `selectivity`; `RQ1Result` gains the pooled
    pair; G2's detail block carries both, because the pre-registered capacity rule is read off the
    re-gate report.
  - The CPVI figure caption carries pooled selectivity.
  - **Directional expectation, pre-registered before measurement:** control CPVI is `≤ 0` — against
    random labels neither probe generalises out of fold and `g_cond` carries twice the features, so
    it overfits the noise harder. Measured on E3-local v4: **−0.006 bits** pooled (every condition
    between −0.011 and −0.003), so the capacity ladder does not fire. The diagnostic is not inert:
    an almost-unregularised probe at n=30, d=128 reads **+0.93**, and an over-capacity MLP **+1.39**.
- **DSE-044 — Repeated cross-fit stabilisation and length control** (`measure/pvi_cpvi.py`,
  `analysis/stats.py`, `experiments/rq1.py`).
  - `ProbeConfig.n_repeats` (default **1**) and `cpvi_with_sd(...)` returning per-instance
    `(mean, across-repeat SD)`. **Repeat 0 is the canonical fold assignment** and repeats 1…R−1
    reshuffle the grouped folds under distinct seeds, so `n_repeats=1` reproduces the unrepeated
    estimator bit for bit — pinned by a test. `_make_splitter` gained the `fold_seed` knob that
    makes this possible: `StratifiedGroupKFold` was previously unshuffled, so naive repeats would
    have been identical copies.
  - `cpvi_sd` and `msg_tokens` are persisted in `scores.parquet`. On E3-local v4 the mean
    per-handoff across-repeat SD is **0.042 bits** — comparable to C3's whole mean CPVI (+0.058),
    which is precisely why a single cross-fit is not enough for scores that feed H2 and RQ2.
  - `analysis/stats.py::partial_spearman(x, y, control)`: first-order partial rank correlation by
    residualising ranks. A control that fully explains a rank leaves floating-point dust rather than
    exact zeros, so the degenerate guard compares residual spread **relative to** the pre-residual
    spread — an absolute `== 0` check reported a spurious ±1.
  - `analyse_rq1` reports `partial_spearman_length` (CPVI vs progress, delivered-message token
    length partialled out) and `MixedModelSummary.path_b_length_controlled` (path *b* refit with
    episode-mean message length as a covariate), reported alongside the uncontrolled path. Length is
    counted in **whitespace tokens of the delivered message** — the same unit the C1 cap operates
    in, so the covariate measures exactly what the channel manipulates. Undefined (NaN) rather than
    fitted when every episode's mean length is identical.
  - **The pre-registered R = 5 is wired into `PilotConfig.cpvi_probe` and `RQ1Config.probe`;**
    `ProbeConfig`'s own default stays 1.
- **Last pass before Myriad — `analysis/stats.py::cluster_bootstrap_ci`**: percentile bootstrap for
  the mean of a per-handoff quantity that resamples whole **episodes** and pools their handoffs.
  Handoffs within an episode share a start pose, a trajectory and overlapping next-k label windows,
  so the iid handoff resample understates uncertainty — on the E3-local v4 dataset it read roughly
  **half the honest width** (C0 CPVI [+0.141, +0.243] handoff-level vs [+0.059, +0.307] cluster).
  Percentile rather than BCa: the draw space is too discrete for the jackknife acceleration at
  pilot cluster counts. Degenerate cases pinned by tests (single cluster collapses to the point;
  misaligned inputs raise).
- **`PilotReport.provenance` (`AnalysisProvenance`)**: the pilot report embeds the encoder name +
  revision, probe config and git SHA behind its G2 CPVI number, and `pilot.md` renders them — the
  Myriad re-gate verdict is a result of record, and a result with an unrecorded revision is not a
  result. `None` only on artefacts written before this change.
- **DSE-001** — Repository scaffolding: PEP 621 `pyproject.toml` (uv-managed, pip-installable),
  `src/preceptx/` package layout with typed subpackages (`sim`, `agents`, `serving`, `data`,
  `measure`, `gate`, `experiments`, `analysis`), floor-and-ceiling pinned dependencies with
  `serving`/`embed`/`data`/`dev` extras, GitHub Actions CI (lint / typecheck / test on Python 3.11),
  `.pre-commit-config.yaml`, `.gitignore`, and a README setup stub.
- **DSE-002** — Serving harness: `scripts/myriad/serve.sh` parameterised SGE jobscript (vLLM greedy
  decoding, fixed seed, pinned revision, xgrammar guided decoding, per-tier GPU mapping);
  `src/preceptx/serving/client.py` `LLMClient` (chat + structured JSON-schema calls, temperature 0,
  seed, retries, health check, graceful shutdown); mock-endpoint unit tests; `docs/serving.md`.
- Planning scaffolding: `CHANGELOG.md`, `DEPENDENCIES.md` (critical path, ticket dependency graph,
  runtime deps, risk register, cross-cutting concerns, phase/freeze gates).
- **DSE-003** — Config, seeding, manifest, determinism: Hydra config tree under `configs/`
  (condition × serialisation × difficulty × model × seed, composed via `@_global_` groups);
  `src/preceptx/config.py` (`ExperimentConfig`/`ModelConfig` Pydantic models, `load_config` wrapping
  validation errors as `ConfigError`, mandatory pinned model revision); `src/preceptx/seeding.py`
  (`set_global_seed` for Python/NumPy/torch, with documented LLM-determinism limits);
  `src/preceptx/manifest.py` (`RunManifest` with git SHA, config hash, dep versions, command, seed,
  revisions, metrics; `build_manifest`/`write_manifest`/`read_manifest`); `src/preceptx/determinism.py`
  + `scripts/determinism_check.py` (repeat a fixed-seed structured call and report agreement rate and
  numeric variance); unit + property + integration tests.
- **DSE-004** — Handoff dataset: `src/preceptx/data/schema.py` (`HandoffRecord` Pydantic contract,
  `SCHEMA_VERSION`, four `Y`-label placeholders for DSE-009); `src/preceptx/data/writer.py`
  (append-safe hash-stamped Parquet parts with a pinned Arrow schema, `register_dataset` index,
  `load_dataset` frame + `load_records` exact round-trip, `dataset_hash`);
  `src/preceptx/data/otel_capture.py` (`emit_handoff` via the vanilla OpenTelemetry SDK, fail-open,
  no precept dependency); `docs/handoff_schema.md`; unit, property, and OTel tests.
- **DSE-006** — Arena and T-shaped load construction (`src/preceptx/sim/`):
  - `load.py::add_t_load` — one dynamic Pymunk body carrying two box `Poly` shapes (a horizontal
    bar + a vertical stem forming a T). Mass is split between the boxes by area and the moment is
    summed over both (`moment_for_poly` per box). **Key invariant:** the boxes are placed so the
    body's vertical extent is symmetric about its position (`min/max y = ∓HALF_H`, where
    `HALF_H = (T_THICK + T_STEM)/2 = 0.65`), so placing the body at a slit's y-centre centres the
    load on the gap — the slit-fit logic depends on this. Bar `1.4 × 0.3`, stem `0.3 × 1.0`,
    friction `0.6` on both shapes.
  - `arena.py::build_arena(slit_width, geometry)` — a top-down (`gravity=(0, 0)`), damped
    (`damping=0.2`, quasi-static so the load does not coast) `pymunk.Space`: four outer boundary
    segments plus two internal vertical walls at `x = chamber_w` and `2·chamber_w`, each split into
    a lower and an upper `Segment` around a slit gap of height `slit_width` centred at
    `geometry.slit_y`. Chambers run left→right along +x.
  - `ArenaGeometry` / `Goal` — Pydantic specs (`extra="forbid"`) for the static dimensions
    (`chamber_w=4`, `chamber_h=6`, `wall_radius=0.05`, `slit_y=3`) and the circular goal region in
    chamber three. `Scenario` is a `NamedTuple(space, load, goal)` bundling the live handles —
    Pydantic can't cleanly hold a live `Space`, so the serialisable specs are Pydantic and the
    bundle is a NamedTuple.
  - `make_scenario(difficulty)` — maps `easy/medium/hard` → slit width `1.8 / 1.0 / 0.7`. The
    load's y-extent is `T_THICK + T_STEM = 1.3`, so **easy clears a head-on push and hard jams it
    (the T must rotate to pass)**; the load starts centred in chamber one, the goal sits at the
    centre of chamber three (`radius=0.8`).
  - Tests: wall/slit/goal coordinates; physics sanity (wide slit passes under a scripted nudge,
    narrow slit jams below the wall) driven by raw `space.step()` to keep DSE-006 independent of the
    DSE-007 action API; deterministic reconstruction (two `make_scenario` builds are identical).
- **DSE-007** — Action interface and physics step (`src/preceptx/sim/actions.py`):
  - `apply_macro_action` — realises `MacroAction` (`N/S/E/W/ROT+/ROT-/WAIT`) as a **world-frame
    impulse (or angular kick) applied at the COM**, then settles the space. Translations use
    `apply_impulse_at_world_point` at `local_to_world(center_of_gravity)` (world-aligned, no spurious
    torque); rotations increment `angular_velocity` by `angular_impulse / body.moment`; `WAIT` only
    settles.
  - `StepConfig` (Pydantic) — stepping/stability parameters with documented defaults: `dt=1/60`
    split into `substeps=4` sub-steps per settle step (anti-tunnelling through thin walls),
    `settle_steps=30`, `linear_impulse=3.0`, `angular_impulse=2.0`, `quasi_static=True`.
    **Quasi-static settling zeroes residual velocity after each action**, so the load is
    nudged-and-comes-to-rest each turn (matching the damped top-down regime) and inverse actions
    cancel cleanly.
  - `read_state` → `BodyState` (Pydantic) — COM (`local_to_world(center_of_gravity)`), angle,
    linear/angular velocity, and an `in_contact` flag (from `body.each_arbiter`). `model_dump` feeds
    `HandoffRecord.state`, keeping the state schema typed end to end.
  - `apply_force_handles` — higher-fidelity two-grip interface (impulses at the two bar ends),
    selectable behind a flag: equal forces translate, opposed forces apply a couple (rotation).
  - `detect_collision` (the contact flag) and `detect_stuck` — **position-based, not velocity-based:
    under quasi-static settling velocity is zeroed each turn, so a jam shows up as the COM failing to
    advance over the last `window` post-action states rather than as low speed**.
  - Tests: per-direction motion, `WAIT` no-op, an inverse-action property test (returns near origin
    within tolerance), both detectors, force-handle translate/rotate, and a fixed-action-sequence
    determinism check (identical trajectory across two runs).
- **DSE-008** — State serialisation, three prompt forms (`src/preceptx/sim/serialise.py`):
  - `serialise(scene, mode)` dispatches on the `Serialisation` literal (`numeric` / `grid` / `nl`).
    The three forms are **isomorphic in information** — each exposes the same load pose and goal,
    differing only in surface form — so the serialisation factor stays a clean A/B over
    *representation*, not over information content (the sibling of "the channel degrades one thing").
  - `SceneState` (Pydantic, `extra="forbid"`) — a **frozen, plain-float snapshot** (load `BodyState`
    + `ArenaGeometry` + `Goal` + `slit_width`), distinct from the live `Scenario`/pymunk handles. It
    exists because the grid must draw the **correct slit gap for the active difficulty**, which the
    live `Scenario` does not carry; it is reconstructable from the dataset for the featuriser (DSE-013).
  - `numeric` — typed `load`/`vel`/`contact`/`goal` tuples at `.4f` precision (round-trips exactly).
  - `grid` — ASCII occupancy at `GridConfig.cell=0.25`. **Key signal:** 0.25 keeps the T's 0.3-thick
    members ~1 cell wide so the **rotate-to-clear-the-slit affordance is visible**; 0.5 aliases the
    thin members away and was rejected for that reason (costs ~4× tokens, accepted). The load
    footprint is rasterised via `load.point_in_t_local` in the body-local frame, with the **body
    origin reconstructed from the COM** using the new `load.COG_Y` (the centre of gravity sits +y of
    the origin because the bar is above the stem, so a COM-only read-back must be un-offset before
    drawing). Cell priority `T > G > # > .`; `+y` printed upward; internal walls render ~2 cells thick
    at a cell boundary (flagged, faithful within resolution).
  - `nl` — **templated hybrid (qualitative + quantitative)**, deterministic, no model call: chamber,
    coordinates, orientation tag, goal direction and nearest-slit distance. The relations are strictly
    *derived from the same numbers* `numeric` exposes, keeping information matched across the arms.
  - `deserialise_check(scene, mode)` — an **information-loss guard** for `numeric`/`grid` only:
    numeric round-trips COM+angle to print precision; grid recovers the COM as the load-cell centroid
    within ~1 cell. **Angle is certified by the occupancy-correctness tests, not recovered from the
    grid** — principal-axis recovery on a coarse, near-symmetric raster is fragile and would be either
    flaky in CI or too loose to certify anything on a rotation task. `nl` is one-way prose → fails loud.
  - Shared helpers added to existing modules: `load.COG_Y` + `load.point_in_t_local` (the canonical
    footprint test now lives with the geometry); `arena.chamber_of` (a COM→chamber query, with
    **boundaries assigned to the right-hand chamber** so the DSE-009 geodesic is continuous at a slit).
  - Tests: per-mode determinism; numeric round-trip; grid occupancy on a known pose (specific
    `T`/`G`/`#`/`.` cells); grid draws the active slit (hard gap < easy gap); grid COM recovery incl. a
    rotated pose; NL is templated and carries coordinates; NL `deserialise_check` raises; a hypothesis
    property test that no serialiser raises on valid **or** extreme/off-grid poses.
- **DSE-009** — Outcome labeller and the four candidate `Y` (`src/preceptx/sim/outcomes.py`):
  - `geodesic_distance(com, goal, geometry)` — a **chain waypoint graph through the two slit centres**
    (`(chamber_w, slit_y)`, `(2·chamber_w, slit_y)`): from the COM's chamber, hop through each
    remaining slit centre then straight to the goal. A point-COM model (per the roadmap's "waypoint
    graph through slit centres"); **routes around the internal walls** rather than straight through
    them, and is continuous across a slit because `chamber_of` assigns the boundary rightward.
  - `reached_goal` (COM within the goal radius) and `step_progress` (signed geodesic reduction,
    positive = toward goal) — the live per-step primitives the runner (DSE-012) will call.
  - `label_episode(records, goal, geometry, cfg)` — a **post-episode pass** filling the four `Y` on
    each `HandoffRecord` via `model_copy` (forward-looking labels need the whole trajectory):
    - `y_binary_progress` — net geodesic progress over the next `k` steps is positive;
    - `y_continuous_displacement` — that **same signed net progress, unthresholded** (the deliberate
      continuous twin of the binary label, so the analysis can ask if the continuous form carries more
      usable info than its binarisation);
    - `y_discrete_config` — the chamber bucket `{1, 2, 3}` at the handoff;
    - `y_terminal_success` — the goal is reached at this step or any later one.
    The window **anchors on `pre_state` and ends on the post-state `k` actions on**, so `k=1` recovers
    exactly `step_progress` for that handoff.
  - `OutcomeConfig.k` (default 3) is the **only free knob**; the discrete bucketing is geometry-derived
    (chamber index), not a hidden degree of freedom. `k` is fixed from the pilot and documented before
    the main sweep, with k-sensitivity reported (the researcher-DoF guard).
  - Tests: geodesic decreases down a scripted solving trajectory and increases when pushed away; the
    routed path exceeds the straight line (proving it routes through the slits); `reached_goal` fires
    only in the region; `step_progress` sign; the four labels on a solving episode; terminal-false +
    backward-progress on a pushed-away step; every label populated; labeller determinism.
- **DSE-010** — Two-agent episode graph (`src/preceptx/agents/graph.py`, `prompts.py`):
  - `graph.py` — a LangGraph `StateGraph` wiring `agent_A` (emits a natural-language handoff via
    `LLMClient.chat`) → `agent_B` (chooses a structured `Action` via `LLMClient.structured` guided
    decoding) → `apply` (steps the simulator and records one `HandoffRecord`), with a conditional
    edge looping to `agent_A` until the goal is reached or the step budget is spent. **Key signal:**
    termination is *our* step-budget logic, not LangGraph's — `recursion_limit` is set to
    `3·max_steps + 10` so the route function (not a `GraphRecursionError`) ends the episode.
  - `EpisodeRunner` — holds the injected `LLMClient` + fixed channel/step/outcome configs;
    `run_episode(cell, episode_id)` builds the per-episode scenario, compiles a fresh graph over it,
    runs to termination, then fills the four `Y` labels via `label_episode` (DSE-009). A mock client
    makes the whole loop testable with no live model.
  - **Framework-thin by design:** LangGraph only sequences nodes; static handles (pymunk
    `space`/`load`, geometry, goal, slit) are closure-bound and only a minimal dynamic `TypedDict`
    crosses the graph, so the langgraph-as-`Any` boundary is contained to one explicit `cast` at
    `invoke`. A LangGraph API change touches only this module.
  - `Action` (Pydantic, `extra="forbid"`) = `{action: MacroAction}`; its `model_json_schema()` is the
    guided-decoding constraint. **Invalid action despite the schema → default `WAIT` + log** — the one
    ticket-sanctioned fail-*soft* (an out-of-enum value raises `ValidationError`, caught at the node).
  - The A→B message passes through a single `apply_channel` choke point — the seam the runtime gate
    (DSE-018) later intercepts.
  - `prompts.py` — versioned `PROMPT_A`/`PROMPT_B` (`PROMPT_VERSION = "v1"`). A wording change is
    result-affecting, so the version is recorded in the **run manifest, not the record** (the frozen
    `HandoffRecord` schema has no prompt field).
  - Tests (mock LLM via `respx`, scripting A-chat vs B-structured by inspecting for `guided_json`):
    loops to budget on `WAIT`; terminates on success (7 east pushes clear the easy goal); invalid
    action falls back to `WAIT`; fixed responses give an identical trajectory (determinism); C1
    delivery is captured on the record.
- **DSE-011** — Communication channel (`src/preceptx/agents/channel.py`):
  - `apply_channel(message, condition, …)` — the degradation ladder applied to the A→B message and
    B's observation **only** (never physics or the action path): **C0** passthrough; **C1** whitespace-
    token cap; **C2** one-step delivery delay (a sentinel `"(no message yet)"` at step 0, the final
    message dropped); **C3** observation window (the message is left intact); **C4** seeded token
    dropout. Each is selected by the cell's `condition`.
  - `ChannelResult` (`NamedTuple`, mirroring `Scenario`) = `(message_delivered, observation_override,
    new_buffer)` — the `observation_override` (C3) and `new_buffer` (C2) are how the channel signals
    the graph without reaching into physics.
  - **C3 is the sanctioned observation exception:** it restricts B's view, not the message —
    grid → a row band `±c3_window_rows` around the load `T`; numeric → drop the `goal=` line; nl →
    keep only the self-state sentence. This is what **forces the message to carry the goal/global
    layout** (the asymmetry RQ depends on).
  - **Determinism:** C4 dropout draws from `default_rng([seed, step])`, so a degraded message is a
    reproducible function of the seed. *ponytail:* C1 caps on whitespace tokens, not the model
    tokenizer (noted as the upgrade path).
  - `ChannelConfig` — `c1_max_tokens`, `c3_window_rows`, `c4_dropout`, and **`c5_enabled` (a real
    `bool` field, default `False`)** so the supervisor-relay stub (full impl DSE-026) is auditable in
    the manifest rather than a comment/env-var.
  - Tests: each condition transforms as specified; C2 delays by exactly one step incl. the step-0
    edge; C4 dropout is seed-deterministic; C3 windows the observation (grid/numeric/nl) while
    leaving the message intact; C5 off by default.
- **DSE-012** — Episode runner and batch sweep executor (`src/preceptx/experiments/sweep.py`,
  `runner.py`):
  - `sweep.py::SweepConfig` — the RQ1 grid as axis lists (`conditions × serialisations × difficulties
    × seeds`) plus the fixed `model`, `channel`, `max_steps`, `concurrency`. `expand` takes the
    Cartesian product into validated single-cell `ExperimentConfig`s. **Key decision:** **one episode
    per cell, with replication carried by the seed axis** — greedy decoding + deterministic physics
    make repeated identical cells pointless, so there is no separate `n_episodes` knob.
  - `episode_id(cell)` — the deterministic resume key; `sweep_hash` — the content hash feeding the
    dataset hash. `RunSummary` (cells / episodes / handoffs / success rate / wall time) and
    `SweepManifest` — the **run-level reproducibility record for a grid** (the per-cell `RunManifest`
    in `manifest.py` models a single cell), carrying the resolved sweep + hash + `prompt_version` +
    summary.
  - `runner.py::run_grid(sweep, client, root)` — bounded-concurrent episode execution (a
    `ThreadPoolExecutor` sized by `sweep.concurrency`, suited to the sync `LLMClient`), with **record
    writes funnelled through one `threading.Lock`** so the append-only writer never races its
    `len(glob("part-*"))` part index — concurrency sits on the LLM-bound work, serialisation on the
    cheap write, and `write_handoffs`/DSE-004 is untouched. **Resumable:** completed `episode_id`s are
    read **once** up front and skipped (idempotent); the summary rolls up the **whole** dataset incl.
    earlier-run episodes. Fail-loud: an episode error propagates out of `pool.map`.
  - **Key interaction signal:** run artefacts (`manifest.json`/`summary.json`) are written to a
    **sibling `<dataset_hash>-run/` dir, not inside the dataset dir** — `load_records` reads the whole
    dataset dir as one parquet table, so a stray JSON there breaks the read (caught by the
    close-the-loop test). `register_dataset` links the manifest path.
  - Supporting surgical edits: `arena.slit_width_for(difficulty)` exposed (the graph needs the active
    slit to build the `SceneState`, which `make_scenario` does not return); `manifest._git_sha` /
    `_dep_versions` promoted to public `git_sha` / `dep_versions` for reuse by `SweepManifest`
    (mirroring the existing `config_hash` function/field idiom; no `RunManifest` schema change).
  - Tests (mock LLM): a small grid writes one record set per cell with unique ids; **concurrency is
    safe** (4 workers, no dropped/duplicated records — the write-lock test); resume skips completed
    cells and does not duplicate. Integration `test_spine_closes_loop` — real runner output flows
    through the featuriser (stub encoder) into `cpvi`, returning a finite score per handoff (hard +
    east push moves-then-jams to carry both `y_binary_progress` classes for the group folds).
- **DSE-013** — Embedding featuriser (`src/preceptx/measure/featuriser.py`):
  - `Featuriser` — turns `HandoffRecord`s into the aligned `(e_s, e_m)` arrays the estimator
    consumes (state from `state_str`, message from `message_delivered`), row-for-row in record
    order. **On-disk cache is content-addressed** by `sha256(revision + text)` → one `.npy` per
    vector, so it is safe to share one cache dir across the whole sweep and re-fitting probes never
    re-encodes. Output is cast to `float64` (the real encoder returns `float32`).
  - **Lazy, optional encoder.** `sentence-transformers` is the only torch puller (the optional
    `embed` extra), so it is imported inside `_load`; the module imports — and its unit tests run —
    with an injected stub encoder and no torch installed (`EncoderBackend` Protocol is the seam). A
    missing extra fails loud with an install hint.
  - `EncoderConfig` — default `BAAI/bge-base-en-v1.5` (768-dim retrieval embedder; **768 over a
    1024-dim model deliberately, to curb probe overfit on the pilot N** that the V-information
    estimator is sensitive to); `second_encoder` `all-mpnet-base-v2` (a different training family at
    matched dim) reserved for the DSE-022 sensitivity check; `normalize=True`. **`revision` defaults
    to the moving `"main"` and the real-encoder load path warns until it is pinned to a commit SHA
    before the Phase-2 freeze** (the manifest already carries `encoder_revision`). Not yet nested in
    `ExperimentConfig` — threaded in with the sweep driver (DSE-020), mirroring `GridConfig`.
  - Tests: deterministic vectors per text; a cache hit returns identical vectors with no re-encode
    (asserted via an encode counter); a partial cache encodes only the misses; `(e_s, e_m)` shapes
    align to the input records.
- **DSE-014** — PVI/CPVI estimator and probe training (`src/preceptx/measure/pvi_cpvi.py`):
  - `cpvi` (conditional: `g_cond` on `[e_s ; e_m]` minus `g_base` on `[e_s]`) and `pvi`
    (unconditional: message-probe minus the cross-fitted class-prior null), both per-instance
    `log2`-likelihood differences of the true label with the roadmap `eps=1e-9` floor; `estimate`
    returns a `CpviResult` (mean CPVI, mean PVI, the **`PVI − CPVI` gap**, held-out AUROC of
    `g_cond` vs `g_base`, plus the in-sample `auroc_train_cond` as the overfit monitor) alongside the
    per-instance scores (row-aligned to the source handoffs = the analysis join key).
  - **Leakage discipline is structural.** Probes are cross-fitted with `StratifiedGroupKFold` keyed
    on `episode_id` so **no episode ever spans train and test** — a random handoff split would leak
    the shared trajectory and inflate CPVI (the R6 guard). `n_splits=None` selects
    leave-one-episode-out for small pilots; an ungrouped run on ≥ 50 instances warns. Positive class
    is `classes[1]` (np.unique sorts ascending).
  - `ProbeConfig` — the probe family V: L2 logistic (default, `C=1.0`, `max_iter=1000`) or a 2-layer
    MLP behind `probe="mlp"`. Continuous twin: `cpvi_continuous` is a Gaussian `log2`-likelihood
    difference, **homoscedastic-per-probe** (σ² from each fold's train residuals). **Recorded
    deviation:** roadmap §2.4 pins a *heteroscedastic* regressor as the continuous default; we ship
    homoscedastic and record the choice in `ProbeConfig.variance_model` (→ the run manifest), with
    `variance_model="heteroscedastic"` raising `NotImplementedError` (reserved) so the deviation is
    auditable at the result level rather than silent.
  - Tests (synthetic ground truth — the mandated determinism fixture): a noise message → CPVI ≈ 0,
    an informative message → CPVI > 0, a state-echo message → PVI > CPVI (gap > 0); AUROC uplift
    (`cond > base`, `train ≥ held-out`); split discipline (no episode in both folds); continuous
    sign; heteroscedastic reserved; LOGO path; the ungrouped warning; the MLP path runs finite;
    `n_splits < 2` rejected; a hypothesis property that CPVI is finite across class balance.
- **DSE-015** — Retrospective/prospective twin and divergence proxy (`src/preceptx/measure/`):
  - `twin.py` — `predictive_distributions` is the shared `g_cond`/`g_base` out-of-fold substrate;
    `retrospective_cpvi` scores with the realised Y; **`prospective_twin` is the expected
    information `KL(g_cond ‖ g_base)` in bits and takes no Y at all — the no-Y discipline is
    structural** (its signature excludes Y, so the call path cannot reach the outcome). KL is clipped
    at `KL_CAP_BITS = 10` so one mis-calibrated probe can't dominate the Bland-Altman limits, and the
    capped count is surfaced as a calibration diagnostic. `twin_agreement` → `TwinAgreement` (Pearson,
    Spearman, Bland-Altman bias + 1.96-SD limits); retrospective and prospective share the bits scale,
    which is what makes the H3 agreement meaningful.
  - `divergence.py` — `jsd` (per-row Jensen-Shannon divergence in bits, the *bounded, symmetric*
    bridge to the runtime proxy DSE-016) and `embedding_cosine` (message-vs-state cosine, the
    probe-independent state-echo statistic).
  - Tests: the prospective signature has no `y` and is invariant while the retrospective score moves
    under a Y-relabelling (the dual no-Y check — signature *and* call-path); twin agreement is high
    with near-zero B-A bias on an informative fixture; the KL cap is applied and counted; JSD is 0 on
    identical and 1 bit on disjoint binary distributions and is symmetric; cosine on known vectors.
- **DSE-016** — Target-free runtime statistics (`src/preceptx/gate/statistics.py`):
  - `Statistic` ABC — **`fit(e_s, e_m, y)` may use Y; `score(e_s, e_m)` never does**, so the no-outcome
    guarantee is structural (a test asserts `score`'s signature is exactly `(self, e_s, e_m)`). Each
    statistic owns the label it predicts via `label(records)`, so a caller cannot feed the wrong Y to
    the wrong statistic; `key` is a stable string (`"info"`/`"fail"`/`"cosine"`) so DSE-018 loads by
    key, not by Python class name (which a rename would silently break).
  - `InfoStatistic` (`s_info`) — Shannon entropy `H(g_cond)` in bits of the offline probe's predicted
    outcome distribution; reuses the CPVI estimator's `_fit_classifier`, so it is the *same* probe
    family as `g_cond`. A one-class fold → `None` probe → entropy 0 (no crash); `n_classes` is stored
    for threshold interpretability (entropy is bounded `[0, log2 K]`, so a raw threshold depends on K).
  - `FailStatistic` (`s_fail`) — `P(fail)` from a failure-risk probe on `[e_s;e_m]` against
    `¬y_terminal_success`; a one-class fold falls back to the base-rate constant predictor.
  - `CosineStatistic` (`s_cos`) — `cos(e_m, e_s)` (message vs pre-handoff state), **reusing
    `divergence.embedding_cosine`** (the DSE-015 state-echo bridge, zero-norm safe). Probe-independent
    (no fit, no Y) — the statistic that answers the circularity objection.
  - `score_records` returns `(scores, episode_groups)` so the DSE-017 cross-fit join lives in one
    place; `_require_labelled` fails loud (`ConfigError`) on any `y_terminal_success=None`.
  - `save_statistic`/`load_statistic` — joblib blob + a `StatisticManifest` (key, encoder name +
    revision, probe config, train-dataset hash, `n_classes`, `git_sha`, timestamp), gitignored; the
    key is validated on load. A linked provenance record rather than a second `RunManifest` schema.
  - Tests: the no-Y signature check on all three; cosine probe-independence and zero-norm → 0; entropy
    bounded and lower when the outcome is predictable; `s_fail` learns on a separable fixture;
    single-class degeneracies; save/load round-trip + key-mismatch guard. Gate coverage 94%.
- **DSE-017** — Offline gate calibration (`src/preceptx/gate/calibration.py`):
  - `calibrate(records, featuriser)` validates each statistic against **realised failure**
    (`¬y_terminal_success`), **never CPVI** — `target` is the literal `"realised_failure"` and the
    entry point has no CPVI parameter (the D10 circularity guard, R5). The "predict low-CPVI" tracking
    is deliberately left to DSE-022 so it can never feed the gate threshold.
  - Honest held-out scores via `_oof_scores`: GroupKFold by episode (no episode spans train/test).
    A per-handoff random split would let the probe memorise an episode's shared state and inflate
    AUROC (R6); a quantitative test asserts `auroc_random − auroc_grouped > 0.1`.
  - `_orient` flips a statistic anti-correlated with failure (AUROC < 0.5) and records the orientation;
    the threshold sits on the **raw oriented score** so DSE-018 applies it with one comparison.
  - `_choose_threshold` — most aggressive threshold with firing rate ≤ budget (default 0.2), i.e. max
    failures-caught within budget; deterministic. A tie mass at the budget quantile steps just above
    the tie (never overshoots the budget); if that empties, the no-op threshold fires nothing. The
    firing rate is recorded for DSE-018's matched-firing-rate control.
  - ECE / reliability — a **report-only** Platt map (1-D logistic) fit on the *held-out* scores (not
    in-sample, or the ECE flatters itself) → `P(fail)`, then equal-width `n_bins` ECE. Empty bins are
    skipped (keeps the JSON nan-free and round-trippable); per-bin counts are surfaced;
    `ece_reliable=False` plus a log warning below N=200, where 10-bin ECE is high-variance.
  - `write_report` — JSON always (the load-bearing artefact); a reliability PNG only with the `viz`
    extra (`matplotlib` lazy-imported in `_render_figure`, skipped with a log line when absent). Added
    the `viz` optional extra; `uv.lock` regenerated; DEPENDENCIES.md §3 updated.
  - Tests: the group-vs-random leakage check (quantitative); AUROC perfect / single-class; orientation
    flip; threshold budget + reproducibility + tie handling; Platt ECE low on an informative fixture
    and `None` on a single-class one; the against-CPVI signature guard; an integration
    calibrate → `write_report` → reload round-trip on a torch-free fixture.
- **DSE-019** — Pilot gate harness (`src/preceptx/experiments/pilot.py`):
  - Three go/no-go gates as stateless functions over the loaded handoff records (only G2 needs the
    featuriser): **G1 capability** (C0 self-play episode success rate vs a configurable floor),
    **G2 signal** (the C0-to-hard gap must clear thresholds in *both* outcome and CPVI — CPVI scored
    held-out on the C0∪hard subset, with a single-outcome-class guard that returns a note rather than
    crashing), **G3 groundedness** (fraction of a message's numeric tokens that match a true `state`
    number within abs/rel tolerance; a message citing no numbers is vacuously grounded). `ponytail`
    ceiling on G3: number-matching grounds load-pose mentions only — not directional lies or the
    goal/slit (not persisted per record); upgrade path is per-serialisation entity parsing.
  - "hard" is the highest-index condition present (C0<…<C4); fails loud (`ConfigError`) on unlabelled
    episodes or a missing C0/contrast (a local 3-line labelled-data guard keeps the Phase-1 pilot
    decoupled from the Phase-5 gate module).
  - `run_pilot` → `PilotReport` whose recommendation escalates by `attempt`
    (proceed → retune_once → fallback; exactly one retune allowed, per the roadmap);
    `render_report`/`write_pilot_report` emit a one-page Markdown report + JSON with the fallback
    ladder (elevate RQ3a; simplify the task; reframe as a diagnostic negative) always spelled out.
  - Tests: each gate's known-answer fixture (grounded vs hallucinated coords; a constructed C0/C4
    gradient for the CPVI gap; the single-class guard), the attempt→recommendation ladder, the report
    render, and an integration run over a stub `run_grid` sweep that emits a report.
- **DSE-020** — RQ1 information-gradient driver (`src/preceptx/experiments/rq1.py`):
  - `rq1_sweep` assembles the C0–C4 × serialisation × difficulty × seed factorial; `analyse_rq1`
    (runner-free, fixture-testable) scores per-handoff CPVI and PVI, summarises each condition
    (success rate, steps, collisions, mean CPVI — each with a bootstrap CI — and the mandatory
    **PVI−CPVI gap**), and builds Ck-vs-C0 contrasts (Cliff's δ + a two-sample bootstrap CI, with
    Holm/BH-corrected mixed-model p-values). `run_rq1` = grid + analyse.
  - **Mixed-effects model**: a linear-probability `statsmodels` MixedLM of the per-handoff progress
    outcome on condition, with **seed as the group random intercept and episode as a variance
    component nested within it** — the only level where random effects for both seed and episode plus
    a per-handoff CPVI mediator fit one model (the episode id encodes the seed, so episode ⊂ seed).
    The descriptive headline stays the episode-level success gradient (H1). H2 mediation is the
    Baron-Kenny attenuation step (refit with CPVI; report the condition-coefficient shrinkage).
    `ponytail`: LPM not GLMM, attenuation not a bootstrapped indirect effect — both upgrade paths noted.
  - MixedLM fit warnings (e.g. `ConvergenceWarning` on small fits) are captured and surfaced as
    WARNING log lines, not crashes (a degraded mode, under `filterwarnings=error`); `analyse_rq1`
    fails loud on unlabelled or single-class progress data.
  - `write_rq1` persists the analysis JSON, a per-condition results CSV, and the
    outcome-vs-condition / CPVI-vs-condition figures (viz-guarded). No new dependency —
    `statsmodels`/`scipy`/`pandas` are already core.
  - Tests: the factorial assembles the expected cells; a synthetic C0→C4 known-answer fixture
    (success + CPVI gradients, a negative C4 coefficient, corrected contrast p-values); the
    table/JSON persistence; an integration mock-grid run end to end.
- **DSE-028** — Shared analysis library (`src/preceptx/analysis/`):
  - `stats.py` — generic primitives on plain arrays (functions, not classes): `cohens_d` (pooled-SD,
    0.0 on zero spread), `cliffs_delta` (distribution-free, O(na·nb) pairwise sign — the right effect
    size for skewed steps/CPVI), `bootstrap_ci` (seed-deterministic percentile bootstrap,
    parameterised by statistic), `correct_pvalues` (wraps `statsmodels.multipletests` for Holm/BH —
    never reimplemented; a property test pins corrected ≥ raw), `seed_sensitivity` (across-seed
    mean/sd/spread — the mandatory LLM-non-determinism companion), `load_analysis_frame` (reuses
    `data.writer.load_dataset`, adds a nullable `failure` that stays `None` on unlabelled episodes),
    and `ANALYSIS_PROTOCOL` (the report-citable test-per-hypothesis map, frozen with Y/V).
  - `figures.py` — one house style + a reusable `ci_plot` (asymmetric error bars so bootstrap
    intervals render faithfully); `matplotlib` guarded behind the optional `viz` extra exactly like
    the calibration figure, so the JSON/table artefacts stay load-bearing and figure calls no-op
    (with a log line) when the extra is absent.
  - Tests: effect sizes / bootstrap CI / corrections on known inputs, a Hypothesis property that
    correction never increases significance, seed-sensitivity aggregation, the nullable-`failure`
    loader, and the figure no-op-vs-render branches.

- **Documentation re-centralisation (23 Aug 2026)** — three new documents consolidating direction that
  had been spread across five files inside and outside the repository. No code change; this entry
  records what now exists and what each is authoritative for.
  - `docs/methodology.md` — the working source of truth for the dissertation's literature-review,
    methodology and experimental-design chapters, moved into the repository so that the method text
    and the code implementing it can be diffed against each other. Supersedes the standalone
    `Precept Literature review 25 July LitReview_Methodology_ExperimentalDesign.md`, which is retained
    outside the repo as a dated snapshot. Sections 1-7 carried forward with the failure-attribution
    baselines re-anchored; sections 8-10 rewritten against the shipped implementation, roughly tripling
    the methodology half. Load-bearing additions: §8.3 states the **receiver-conditioning** decision and
    why sender-conditioning makes C3 arithmetically inert; §8.4 states the **grouped cross-fit** leakage
    argument and why a random handoff split inflates the reported quantity; §8.5 separates the three
    nulls (shuffled-message, control-task selectivity, train-vs-held-out AUROC) that answer three
    different objections; §8.6 pre-registers the **length controls** and shows how C1 inverts the
    confound into a dissociation; §8.7 states that the prospective twin is a **KL divergence and
    therefore non-negative where CPVI is signed**, and treats that one-sidedness as a finding;
    §10.5 is an **18-row deviation register** mapping every departure from the design as first
    specified to its evidence.
  - `docs/EXPERIMENTS.md` — the operational run ledger: eleven experiments E0-E11 with their
    preconditions, exact commands, emitted artefacts, pre-registered analyses and null readings;
    seven execution stages S0-S6; a table of what must exist in code before each stage; a fixed
    **result-recording template** so results chapters are assembled from entries written the day a run
    completes rather than reconstructed in September; and a five-row freeze register (F0-F4).
    Opens by stating that **zero episodes have been recorded** and everything to date ran against stubs.
  - `docs/ROADMAP_VISUAL.md` — the visual roadmap and running change record. A six-row "what changed
    since 25 July" table; a section recording two 2026 papers found during the cross-reference and
    absent from all prior planning docs (**AgentForesight**, arXiv:2605.08715, which occupies the
    pre-outcome setting and forces the contribution onto three axes rather than prospectivity alone;
    and **Causal Agent Replay**, arXiv:2606.08275, which strengthens the replay-defined outcome rather
    than threatening it); and four Mermaid diagrams — programme map with gates, calendar, critical path
    with the new tickets, and the measurement architecture annotated with its three invariants.
- **Eleven new tickets in `ISSUES.md`** (GitHub Issues remains the backlog source of truth).
  - **DSE-031/032/033** — the three unticketed blockers between the repository and its first result:
    driver entry points (`run_grid`/`run_pilot`/`run_rq1` are library functions and `pyproject`
    declares no `[project.scripts]`); a structured-output mode for non-vLLM endpoints (the client sends
    vLLM's `guided_json` in `extra_body`, an OpenAI-compatible local server expects
    `response_format.json_schema`); and pinning the encoder revision, which currently defaults to an
    unpinned sentinel and only warns.
  - **DSE-041/042** — TraceElephant loader with a separate `LogHandoffRecord` (physics fields **absent,
    not nullable**; `trace_id` as the cross-fit group key) and the counterfactual-replay outcome
    labeller (majority vote over n replays, reported agreement rate, agreement floor, stratified
    sampling recorded in the manifest, hard spend cap, dry-run projection, and a signature test proving
    the labeller cannot read the annotations).
  - **DSE-043/044** — control-task selectivity and repeated cross-fits plus length control. Both are
    marked **blocking the Y/V freeze**, because each changes what the freeze covers.
  - **DSE-045** — the gate retry-feedback template, which exists because under greedy decoding an
    unchanged re-prompt is a fixed point: DSE-018 as previously written would pass its unit tests and
    be vacuous in a live run. Marked **blocking DSE-018**.
  - **DSE-046/047/048** — the absent-versus-unused decomposition (replacing the cut SocialJax arm),
    the baselines re-anchoring and framing ticket, and the stretch gate-against-a-real-MAS adapter.
- **DSE-049** — per-role clients on the episode runner (`client_a`, optional `client_b`; omitted
  `client_b` means self-play, identical to today's single-client path). Ticketed from the ground-up
  status review's flag that the heterogeneous cell (DSE-021) was blocked on an unticketed refactor;
  batched with DSE-031–033 because it is cheaper before call sites accrete.
- **DSE-032** — second wire format for schema-constrained decoding, so the pre-cluster pilot can run
  free on a local OpenAI-compatible endpoint:
  - `ServingConfig.structured_mode: Literal["guided_json", "response_format"]`, defaulting to
    `guided_json` so every existing vLLM path is byte-for-byte unchanged. `response_format` emits the
    OpenAI-standard `response_format.json_schema` block (`name="action"`, `strict=true`) and sends
    **none** of the vLLM-only keys — a local runtime rejects `guided_json`/`guided_decoding_backend`.
  - **The schema object is passed through untouched in both branches** (asserted by a unit test and a
    hypothesis round-trip), so the constraint the two backends enforce is identical; only the request
    shape and the engine enforcing it (xgrammar vs llama.cpp grammars / Outlines) differ. That engine
    difference is exactly why schema-adherence rate is a *reported* DSE-005 row, not an assumption.
  - The mode is recorded in the sweep manifest (`structured_mode`), so a local-pilot dataset stays
    permanently distinguishable from a Myriad one even at identical config hashes.
  - `docs/serving.md` gains a local-pilot section: the vLLM-vs-LM-Studio table, the headless `lms`
    commands (`brew install --cask lm-studio`, `lms get mlx-community/Qwen3-8B-4bit`,
    `lms server start --port 1234`), the `PRECEPTX_SERVING_SUBSTRATE=local-lmstudio` label, and the
    two identity caveats (the served id is the *quantised* repo; a 4-bit G1 verdict is indicative
    only, the bf16 Myriad re-gate is the verdict of record).
- **DSE-031** — `src/preceptx/experiments/cli.py` plus `[project.scripts]`: `preceptx-pilot` and
  `preceptx-rq1` are now runnable from a shell (`uv run preceptx-pilot --dry-run` works from a clean
  checkout). This was the literal blocker between a served model and a pilot verdict.
  - **Hydra composes, Pydantic validates**: `_resolve_cell` composes `configs/experiment.yaml` with
    `model=<group>` plus any `--overrides`, and the raw `DictConfig` never leaves that function — it
    is validated into an `ExperimentConfig` before the sweep is built. Grid axes come from
    comma-separated flags (`--conditions C0,C4`), each validated by `SweepConfig`.
  - `--dry-run` prints cells, an **upper-bound** model-call count (2 calls per step × the certified
    per-difficulty budget; early success shortens episodes), the sweep hash, the dataset hash and the
    per-role model identities — and constructs no client, so it issues no calls at all.
  - Defaults are the documented E3 cell, not invented ones: pilot runs C0/C1/C4 × easy/hard × seeds
    0-2; rq1 runs C0-C4 × hard × seeds 0-4.
  - **Fail loud at start-up**: an unset `PRECEPTX_SERVING_SUBSTRATE` raises `ConfigError` before any
    episode runs (an unlabelled dataset cannot be told apart from a Myriad one afterwards), and a
    failed `health_check` raises `ServingError` rather than recording a run of degraded WAITs.
  - The entry point owns `logging.basicConfig`; library code still attaches no handlers.
- **DSE-033** — encoder revisions pinned to resolved commit SHAs, verified against the HuggingFace
  API on 24 Aug 2026: `BAAI/bge-base-en-v1.5` → `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`, and the
  new `EncoderConfig.second_encoder_revision` for `all-mpnet-base-v2` →
  `e8c3b32edf5434bc2275fc9bab85f82640a19130`.
  - The real load path now **raises `ConfigError` instead of warning** when the revision is not a
    40-character commit SHA, so a branch name cannot reach a recorded run. A branch/tag is rejected
    by shape, not by name, so `v1.5` and `refs/heads/x` fail the same way `main` does.
  - `AnalysisProvenance` carries both encoder identities, so a DSE-022 sensitivity re-run is
    distinguishable from the primary fit by the artefact alone. Stub-backed tests are untouched (the
    check lives on the real load path only), so unit tests still run with no torch installed.
- **DSE-005** — model-ladder benchmark harness: `src/preceptx/serving/benchmark.py` (typed, tested)
  plus the `scripts/benchmark_models.py` wrapper, and a `docs/serving.md` section.
  - Five numbers per tier: throughput (tok/s over bounded completions), time to first token, peak GPU
    memory, JSON-schema adherence over N constrained calls, and a ten-episode C0/easy capability
    smoke **driven through the real loop** (`run_grid`), not a scripted stub.
  - **Append-then-render**: one model is served per GPU job, so each invocation appends a row to
    `tiers.jsonl` and rewrites `ladder.md`, `ladder.csv`, `ladder.json` and `recommendation.md` from
    every row collected so far. That is what lets one table span substrates measured weeks apart.
  - Honest absences: peak memory is `None`/`n/a` where there is no `nvidia-smi` (a laptop), never a
    fabricated zero; schema misses are **not retried**, because a retried rate would flatter the tier;
    `ttft_s` is documented as a one-token-round-trip proxy, not a streamed first-chunk timestamp.
  - `recommend()` names the fastest tier clearing both floors (smoke >= 0.5, schema >= 0.95) or
    **refuses to pick one**, and always repeats that an unmeasured tier is a placeholder and that a
    local 4-bit row cannot carry a capability claim.
- **`PREREGISTRATION.md` v0** (draft, not frozen) at the repository root — the F0 artefact named by
  `docs/EXPERIMENTS.md` §6, drafted at **zero recorded episodes** so no choice in it can be
  outcome-contingent. Fixes H1-H6; the task geometry, budgets and jitter; the C0-C4 parameters; the
  four Y labels with `k = 3` and the receiver-conditioning semantics; the probe family with `R = 5`
  repeated cross-fits, the control-task expectation and the two length controls; both pinned encoder
  revisions; the G1/G2/G3 thresholds and the runtime-gate calibration rule; the sweep scale and its
  power basis; and the analysis protocol. Records **one anticipated v0 → v1 change** — G2's CPVI half
  is directional until the pilot reveals the bit-scale — and opens a prospective deviation log that
  takes over from `docs/methodology.md` §10.5 at the freeze.

### Changed
- `docs/myriad.md`: new §9 **"Getting the results back"** — `runs/` is gitignored and the pilot
  writes its verdict to Scratch, so nothing left the cluster on its own and the runbook never said
  how to retrieve it. Gives the `rsync` filter for the small set a frozen result is made of
  (manifest, summary, `pilot.{json,md}`, `serve_env.json`) and notes when to pull the Parquet too.
  §5 names the real HTTPS clone URL and states that results do not come back by pushing. Old §9/§10
  renumbered to §10/§11.
- `experiments/pilot.py`: the fallback ladder's first rung now reads "TraceElephant external
  validity" rather than the superseded "Who&When / MAST" (DSE-041 supersedes DSE-023).

### Fixed
- **`scripts/myriad/serve.sh` requested 256 GB and could never have been scheduled** (DSE-002).
  UCL SGE's `-l mem` is **per slot**, not per job, so `-pe smp 8 -l mem=32G` asked for 8 × 32 = 256 GB
  against an `-ac allow=L` node's 160 GB usable. SGE does not reject an unsatisfiable request — the
  job queues forever. Now `-l mem=4G` (= 32 GB total), with the arithmetic in a comment in both
  jobscripts. Found by review against the UCL documentation before the first submission; nothing in
  the repo or CI could have caught it.
- **`serve.sh` named a CUDA module Myriad does not have.** The default was `cuda/12.4`; UCL's
  modules are versioned like `cuda/12.2.2/gnu-10.2.0`, and the GPU-node driver is on the 12.2
  branch, so a mismatched toolkit is a real runtime-error source. `module load` failing under
  `set -e` kills the job before vLLM starts. The default is corrected, the failure now prints the
  fix (`module avail cuda`), and `CUDA_MODULE=none` skips the load entirely — vLLM's wheels bundle
  their own CUDA runtime through torch, so the module is a convenience rather than a requirement.
- **`serve.sh` left torch, triton and vLLM caches in `$HOME`.** `HF_HOME` was already on Scratch, but
  the others default into the home directory and count against the same 1 TB quota the ~45 GB of
  weights do. A full quota does not fail cleanly: the job dies creating its `.o`/`.e` files, which
  reads as a scheduler fault. `XDG_CACHE_HOME`, `VLLM_CACHE_ROOT` and `TRITON_CACHE_DIR` now point
  at Scratch.
- **`LLMClient.health_check` fetched the served model list and discarded it** (DSE-002 acceptance
  criterion). Pointed at a leftover job serving a different tier, every call would have succeeded
  and the manifest would have recorded the tier that was *configured* rather than the one that
  *answered* — a wrong recorded revision, which is worse than a missing one. The served ids are now
  compared against `ServingConfig.model` and the mismatch is rejected before the smoke completion is
  even attempted.
- **The pre-commit hooks could not pass, and had not been passing on `main`.** Found while adding
  the shellcheck hook; verified against a clean checkout of `d31f159`, which fails identically.
  Two independent causes, both "the hook runs a different tool than the project does":
  - The **mypy** hook's isolated env listed only `pydantic` and `openai`, so `numpy` was absent and
    `FloatArray = NDArray[np.float64]` degraded to a plain variable — **187 `not valid as a type`
    errors across 8 files**, against a tree that `mypy --strict src/` passes clean. Added `numpy`
    and `pandas-stubs`.
  - The **ruff** hook was pinned at `v0.6.9` while the venv resolves `0.15.20`; `UP038` was dropped
    from ruff's defaults in between, so the hook flagged `pilot.py` code that `ruff check .`
    accepts. Bumped the rev to match, and moved to the current `ruff-check` hook id.
  - CI was unaffected — it runs `uv run mypy --strict src/preceptx` in the real venv — so the gate
    that mattered was green throughout and no result is implicated. The practical effect was local:
    the hooks could only be satisfied with `--no-verify`, which CLAUDE.md forbids.
- **The pilot CLI's default seed axis contradicted the pre-registration.** `_PILOT_SEEDS` was
  `[0, 1, 2]` while PREREGISTRATION §6 was amended (2026-08-24, before the bf16 re-gate and before
  F0) to seeds 0–4 / 40 episodes, because at 24 episodes a single flipped easy-C0 episode moves G1
  across its 0.5 threshold about a third of the time. A bare `preceptx-pilot` would have run a cell
  the analysis plan does not describe. Now `[0, 1, 2, 3, 4]`; the E3 cell is 20 cells per the
  dry-run plan.
- **The last carriers of the falsified "CPVI is near zero by construction" claim** (review pass;
  docs/comments only, no behaviour change): the `_PILOT_CONDITIONS` comment in `experiments/cli.py`
  and `RESEARCH_ROADMAP.md` §2.3's C3 line now state the corrected rationale (C3 varies the
  conditioning set; E3-local measured +0.19 bits in C0 with the receiver holding the full state);
  methodology §8.3's sender-conditioning argument is softened to the question-selection claim it
  actually is; the `y_discrete_config` "zero by construction" phrasing in `sim/outcomes.py` and
  `PREREGISTRATION.md` §4 is weakened to "collapses toward zero" for the same probe-relative
  reason.
- **G2 refitted its CPVI probe on a two-condition subset** (`experiments/pilot.py`;
  result-affecting). `g2_signal` selected the C0-plus-hardest rows, featurised those and fitted the
  probe on them. Pointwise V-usable information is per-instance scores from **one** fitted probe;
  refitting per contrast discards the other conditions' rows and shifts the class balance the probe
  sees — and it is not the estimator the RQ1 analysis uses. On the same 24-episode data the subset
  fit read the C0−C4 CPVI gap as **+0.012 bits** and the whole-cell fit reads **+0.211**. G2 now
  featurises and fits over all records and contrasts the resulting per-instance scores. The gate was
  not measuring a weak gradient; it was measuring a strong one badly.
- **G3 scored a sender as hallucinating geometry it had been shown** (`experiments/pilot.py`;
  result-affecting). `_record_grounding` drew its truth set from `rec.state`, which carries the load
  body only, while the serialiser prints the wall abscissae and slit interval (v3) and the load
  dimensions (v4). A message correctly citing "the slit runs 2.1 to 3.9" was counted as fabricating
  both numbers, and G3 **failed the v4 pilot at 0.720** on messages that invented essentially
  nothing. Truth is now the union of `state`'s numeric leaves and every number in `state_str` —
  what the sender was actually shown — which reads **0.977** on that run and **0.999** on the v3 run
  (was 0.811), and cannot rot when the serialiser gains a key.
- **G1 scored the wrong population** (`experiments/pilot.py`) — `g1_capability` averaged C0 episode
  success across *every* difficulty present, while the pre-registration and the roadmap both scope the
  capability floor to **easy**. On the E3 cell (C0/C1/C4 × easy/hard) that mixes a solvable geometry
  with one designed to be hard: a pair solving every easy episode and no hard one scored exactly 0.5
  and **passed the 0.5 floor by arithmetic accident**. Now filters to `difficulty == "easy"`, raises
  `ConfigError` when the dataset has no easy C0 episodes rather than averaging whatever is present,
  and reports `n_easy_c0_episodes` / `n_easy_c0_success` in the gate detail. Re-gating E3-local left
  the verdict unchanged (0/3 easy versus 0/6 mixed, both 0.000).
- **C3's numeric observation restriction had rotted against the v3 prompt surface**
  (`agents/channel.py`) — `_restrict` blacklisted the `goal=` line, so the `walls_x=` and `slit_y=`
  lines introduced by the v3 serialiser were still delivered to B and C3's receiver kept the full
  arena layout; the asymmetry the condition exists to create was nominal, in the one condition the
  design relies on to keep CPVI off the floor. Replaced with a **whitelist** of B's own state,
  `_C3_NUMERIC_KEEP = ("load=", "contact=")`, so a serialiser that gains a key fails closed instead of
  leaking it. The `nl` branch was checked and is unaffected (v3 touched `_numeric` only). No C0/C1/C4
  record in any run to date is affected.
- **`render_transcript` attributed a whole multi-episode dump to its first episode.** It emitted one
  header and one table regardless of how many episodes the records spanned, so a five-episode E1
  dump claimed "75 handoffs | terminal success: True" under episode s0's name. It now emits one
  section per episode in first-appearance order; single-episode output is unchanged. This is the
  function the E1 transcript read depends on, so a misleading header defeats the stage's purpose.
- **The ladder benchmark's TTFT probe asked for one token**, which the new empty-content guard
  rejects — a one-token completion is often whitespace, and is exactly empty from a runtime left in
  thinking mode. It now times an 8-token completion off a prompt that guarantees a short answer, and
  `docs/serving.md` states the proxy precisely.
- **DSE-004** — `write_handoffs` now writes each Parquet part to a hidden temp (`.part-NNNNN.parquet.tmp`)
  and atomically renames it into place (`os.replace`, atomic within one directory). A crash mid-write
  previously left a truncated `part-*.parquet` that poisoned every subsequent whole-dir read
  (`pq.read_table(dataset_dir)`) and resume. The temp is invisible to both the `part-*.parquet` glob
  (wrong prefix/suffix, so it never inflates the next part index) and pyarrow's directory discovery
  (which skips `.`-prefixed files); its name is keyed on the part index, so a resume overwrites any
  stale leftover. Surfaced by the DSE-012 close-the-loop smoke.
- Post-merge review hardening of the DSE-013/014/015 measurement spine (no CPVI/PVI values change on
  the default path — pure correctness/perf):
  - `Featuriser.embed_texts`: classify cache hits/misses in a single pass. The prior code built a
    `miss_idx` *list* then tested `i not in miss_idx` per row (O(n·m)) and called `_cache_path`/
    `Path.exists()` twice per text; the sweep-scale cost is now O(n) with one stat per text.
  - `_fit_regressor` (continuous CPVI path) now wires `ProbeConfig.c` to the Ridge regulariser as
    `alpha = 1.0 / c` (alpha is direct-strength, `C` its inverse), so the config knob takes effect on
    the regressor as it already did on the logistic probe. Default `c=1.0 → alpha=1.0` is unchanged.
  - `twin_agreement` returns `nan` Pearson/Spearman on a single handoff (`n<2`) instead of raising an
    opaque scipy `ValueError`; correlation is undefined there.
- Post-merge review hardening of the DSE-016/017 runtime gate (a log line + docs only; no threshold,
  score, or report values change):
  - `_choose_threshold` now emits a `WARNING` when the firing-rate budget is infeasible — degenerate
    or constant oriented scores, where stepping above the budget-quantile tie empties the candidate
    set — and the threshold falls back to no-op (fires nothing). The silent version handed DSE-018 a
    never-firing gate with no diagnostic; the chosen threshold value itself is unchanged.
- Post-merge review hardening of the DSE-019/020/028 analysis stack (result-*shape*-affecting; no RQ1
  result is frozen yet, so nothing to re-freeze):
  - **H2 mediation moved to the episode level (DSE-020).** The old test entered per-handoff CPVI as a
    covariate on the per-handoff *progress* outcome and reported only the condition-coefficient
    attenuation — a different DV from H1's episode success, and one that conflated within- vs
    between-episode CPVI variance (the "easier episodes carry more CPVI" confound). `analyse_rq1` now
    runs a full Baron-Kenny mediation on the **headline** DV: episode success on condition with the
    mediator aggregated to **episode-mean CPVI** — path *a* (`cpvi~condition`), path *b*/*c'*
    (`success~condition+cpvi`), total *c*, and the **indirect effect a·b per condition with a
    percentile bootstrap CI** over episodes (degenerate resamples — a dropped condition or single-class
    outcome — are skipped). The handoff-level CPVI attenuation is retained as a labelled *within-episode
    diagnostic*, not the H2 test. H1's inferential model is unchanged (handoff LPM, seed RE + episode
    VC — still the only level where both random effects fit). New `RQ1Config.n_boot_mediation` (default
    400; the model-refit bootstrap is far costlier per draw than the one-sample CIs). New
    `EpisodeMediation` model; `MixedModelSummary` reshaped — drops `coef_with_cpvi`, renames
    `cpvi_coef`→`diagnostic_cpvi_coef`, adds the mediation block (`path_b`, `mediations`,
    `mediation_outcome`) and `converged`/`mediation_converged` flags. `ponytail`: both fits stay LPMs;
    logistic path *b* + delta-method indirect noted as the upgrade.
  - **`bootstrap_ci` → BCa (DSE-028).** Bias-corrected-accelerated bootstrap via `scipy.stats.bootstrap`
    (corrects the percentile method's small-sample bias/skew — the right default for the small, skewed
    pilot samples), with a guarded fall-back to the plain percentile interval for the cases BCa cannot
    handle: a constant sample (collapses to the point) and n < 3 (the jackknife acceleration is
    undefined). Deterministic via `seed`. No new dependency — `scipy` is already core.
  - **Mixed-model convergence is now persisted** into `rq1.json` (`converged`, `mediation_converged`),
    so a non-converged fit is auditable, not just a transient WARNING line.
  - **Seed sensitivity now tracks the gradient, not a collapsed metric (DSE-020).** `analyse_rq1` feeds
    `seed_sensitivity` the per-seed **C0-minus-hardest success gap**, so its spread answers "is the
    C0→C4 ordering seed-stable?" rather than "does overall success vary across seeds?" (which it always
    does, from LLM non-determinism). `seed_sensitivity` itself is unchanged.
  - **Pilot proceed-verdict seed floor (DSE-019).** `PilotConfig.min_seeds_for_proceed` (default 3)
    downgrades an all-gates-pass verdict from `proceed` to `retune_once` with a `recommendation_note`
    when too few seeds ran — a single-/few-seed pass is LLM noise, not a stable gradient. `PilotReport`
    gains `n_seeds` and `recommendation_note` (both surfaced in the rendered report).
  - **G2 CPVI-gap floor documented as deliberately directional (DSE-019).** The `g2_min_cpvi_gap=0.0`
    default ("any positive CPVI gap") is now annotated as intentional: CPVI's bit-scale is uncalibrated
    until the pilot runs, so a magnitude floor cannot be honestly pre-specified — the 0.1 success-gap
    carries the magnitude; re-set to a pre-registered positive floor once the pilot reveals the scale.
- **Implementation-review correction pass** (`docs/review/2026-07-11-implementation-review.md`,
  Bundles 1–2: measurement validity + serving readiness, against `54e7b85`). Result-shape-affecting,
  but no result is frozen and zero real datasets exist — nothing to re-freeze, no v1-loader shim kept:
  - **CPVI now conditions on the receiver-observed state (P0-1).** `HandoffRecord` gains a required
    `observation` field — B's delivered view at the handoff (equals `state_str` under C0/C1/C2/C4, the
    restricted window under C3) — persisted by `graph.apply_node`, and `Featuriser.featurise` embeds
    it (not `state_str`) for `e_s`. Pre-registered semantics documented in both modules: *the
    conditioning state s is the state observable to the receiver at the handoff; under C3 that is the
    windowed view, by design* (previously C3's construct was arithmetically inert — s contained what
    the window hid). All consumers (estimator, twin, runtime statistics, calibration, G2) inherit
    through the single featurise choke point — deliberate: the runtime statistics must also condition
    on what B saw. `state_str` (A's full view) stays on the record, giving C3 a free dual-baseline
    diagnostic.
  - **Schema v2 — one deliberate bump.** `SCHEMA_VERSION = 2`; besides `observation`, the record gains
    the DSE-018 gate fields now so the contract changes once (`gate_blocked: bool = False`,
    `gate_retries: int = 0`, `message_blocked: str | None = None`) and the labeller's
    `y_window_truncated: bool | None`. The Arrow schema in `writer.py` is extended to match;
    `dataset_hash` keys on `SCHEMA_VERSION`, so dataset hashes roll automatically.
    `docs/handoff_schema.md` updated.
  - **The seed axis is now true replication (P0-2).** Under greedy decoding + a fixed scenario, seeds
    only varied the C4 dropout mask — C0–C3 "replicates" were the same episode re-run. New
    `arena.ScenarioJitter` (Pydantic): the start pose is sampled per seed with `x ∈ (1.2, 2.8)`,
    `y ∈ (1.5, 4.5)`, `θ ∈ (−π/2, π/2)`; the max T-vertex radius from the COM is ≈ 0.9, so every
    sample keeps clearance from walls, with belt-and-braces rejection sampling via `space.shape_query`
    that fails loud (`ConfigError`) after 100 attempts. `make_scenario(rng=None)` keeps the legacy
    fixed pose, so scripted physics tests stay meaningful. `EpisodeRunner` derives the rng as
    `default_rng([cell.seed, 2**16])` — the salt cannot collide with the channel's `[seed, step]`
    streams, and keying on the seed alone preserves cross-condition pairing (same seed → the same
    scenario instance in every condition). The realised pose needs no schema field — it is step-0
    `pre_state`. `SweepConfig.jitter` flows into `sweep_hash` and the manifest; the goal stays fixed
    (the review's optional second knob, skipped as minimum change).
  - **Serving actually starts (P0-3).** Dense Qwen3 checkpoints have no `-Instruct` suffix:
    `configs/model/*.yaml` corrected to `Qwen/Qwen3-{8B,14B,32B}` and pinned to HF-API-verified
    commit SHAs (fetched 2026-07-18; the `-Instruct` variants confirmed 404). Qwen3 hybrid thinking is
    disabled per request: `ServingConfig.chat_template_kwargs` (default `{"enable_thinking": false}`,
    override to `{}` for endpoints that reject unknown keys) rides `extra_body` on both `chat` and
    `structured`; a `<think>` tag in returned chat content raises `ServingError` — a CoT message is a
    category error, never a degraded mode. `serve.sh` default model fixed, the 70B example marked
    placeholder pending DSE-005, and the jobscript now echoes vLLM/torch versions, GPU name and
    model+revision into the job log (P2-10 rider). `docs/serving.md` updated.
  - **Transport errors fail the episode loud (P1-3).** `agent_b` now catches `ValidationError` only
    (the ticket-sanctioned invalid-action → WAIT path); `ServingError` propagates, so a dead endpoint
    can no longer record an episode of passing-looking WAITs.
  - **Grid legend + numeric vel drop — one serialisation bump (P1-5, RD-7).** `_grid` prepends the
    constant header `legend: T=load G=goal #=wall .=free | top row = north (+y)` (constant text across
    cells preserves the information-isomorphism argument). Wrinkle the review missed: the legend
    contains a literal `T`, so `channel._window_grid` splits the header off before windowing the body
    rows and `serialise._grid_load_centroid` skips it. `_numeric` drops the dead `vel=` line (always
    ≈ 0 under quasi-static settling). `PROMPT_VERSION = "v2"` records both changes in the manifest.
  - **Outcome labeller (P1-10, P1-12).** `y_discrete_config` is now the chamber at the **window end**
    (the roadmap's "bucketed next pose region") — the pre-state chamber was a state feature with
    CPVI ≡ 0 by construction. The final k−1 handoffs, whose forward window is clamped at episode end,
    are flagged `y_window_truncated=True` (flag only; exclusion vs k-sensitivity stays a
    pre-registration decision, not code).
  - **G2 measures the headline construct (P1-2).** The pilot's CPVI gap now scores per-handoff
    `y_binary_progress` (the label RQ1's headline uses) instead of episode terminal success, failing
    loud on unlabelled records. Side benefit: progress varies within episodes, so the
    single-class-in-C0∪hard degeneracy largely disappears. G1/G3 and the success-gap half of G2 stay
    on episode success — they are about outcomes, not the probe construct.
  - **Sweep/manifest provenance (P1-6, P1-9, §7-7).** `SweepConfig` gains `step: StepConfig` and
    `outcome: OutcomeConfig`, so `k = 3` and the impulse parameters reach `sweep_hash` and the
    manifest (P1-6). `manifest._TRACKED_DEPS` += pymunk, scipy, statsmodels, joblib,
    sentence-transformers — the deps that shape trajectories, statistics and persisted artefacts
    (P1-9). `SweepManifest` records `serving_substrate` (from `PRECEPTX_SERVING_SUBSTRATE`, default
    `"unspecified"`) and `endpoint_base_url` (from the client) — deliberately **not** in `sweep_hash`
    (an environment property, not experiment identity), so interim-GPU pilot data stays permanently
    distinguishable from Myriad data (§7-7).
  - **Analysis protocol + provenance (P1-7, P1-8, P1-11, P1-17, P2-6).** `ANALYSIS_PROTOCOL["H2"]`
    rewritten to describe the shipped episode-level Baron-Kenny test (the old string described the
    abandoned per-handoff covariate design); new `"H1_efficiency"` entry. New shared
    `AnalysisProvenance` model (encoder name + revision, `ProbeConfig`, git SHA, timestamp) embedded
    in `RQ1Result` and `CalibrationReport` (replacing the bare `git_sha` field) so both artefacts are
    self-describing (P1-8). `Contrast` gains `steps_delta` + CI: Cliff's δ on steps-to-goal, Ck vs C0
    — failures sit at `steps == max_steps`, so the rank statistic treats them as the
    censored-at-budget mass (P1-11). `analyse_rq1` returns `(RQ1Result, scores)` and `write_rq1`
    persists `scores.parquet` (`episode_id, step, condition, seed, cpvi, pvi`, row-aligned to the
    dataset) — the RQ2 join key (P1-17). `EpisodeMediation.indirect_n_draws` records retained
    bootstrap draws, so a CI from 40/400 degenerate-skipped draws is visibly flagged (P2-6).
  - **Embedding cache key includes the encoder name (P1-16).** `_cache_path` digests
    `sha256(name ⧵0 revision ⧵0 text)`; two encoders both at revision `"main"` can no longer poison
    each other's cache. No migration — no cache exists.
- **Implementation-review correction pass (Bundle 3 tooling + hygiene).** Pre-Myriad diagnostics
  and repo hygiene; the difficulty/rotation retune this surfaced is under *Changed* below.
  - **Feasibility certificates (`sim/feasibility.py`, P1-4).** An A\* search over the six
    pose-changing macro-actions on the deterministic sim — Markovian on the load *pose* under
    quasi-static settling (nodes restore by placement, not trajectory replay), states deduped on a
    0.15-unit / 18° grid, guided by the geodesic-to-goal distance. `solve(difficulty)` returns the
    shortest oracle path, its length, and a budget = `ceil(2.5 × optimum)`; `certify()` /
    `python -m preceptx.sim.feasibility` print the whole ladder. Frozen `STEP_BUDGETS` = {easy 18,
    medium 33, hard 33}. A unit test certifies easy solvability + **path soundness** (replaying the
    returned path on the real sim reaches the goal); an integration test certifies medium/hard stay
    feasible within budget. This tool caught the infeasible-hard finding (see *Changed* and
    `docs/experiment_design_log.md`).
  - **Episode renderer + transcript (`analysis/render.py`, §7-2 / the DSE-029 demo trace).**
    `render_episode(records)` draws a per-step arena / T-pose / goal PNG grid (guarded on the `viz`
    extra exactly like the other figures — no-op with a log line when absent); `render_transcript`
    emits a markdown transcript pairing each state with the raw → delivered message and the action
    (pure text, always available, channel degradation visible). Both reconstruct from persisted
    records alone. The T polygons use the newly-public `load.t_shape_verts` with world =
    `com + R(angle)·vert` (no COG offset — the read-back COM is the body origin), so the drawing
    cannot drift from the physics body.
  - **Shuffled-message audit (`measure/pvi_cpvi.shuffled_message_cpvi`, RD-15).** Permutes messages
    *within condition* and recomputes CPVI; the null mean must collapse toward 0 — the pre-registered
    "the estimator isn't hallucinating signal" manipulation check. Surfaced in `analyse_rq1` as
    `RQ1Result.shuffled_message_audit` (real `mean_cpvi` vs `null_mean_cpvi`/`null_std_cpvi`),
    controlled by `RQ1Config.n_shuffle` (default 20; 0 disables).
  - **CI + repo hygiene.** Coverage `--cov-fail-under=80` **scoped to the load-bearing core**
    (`preceptx.sim` / `measure` / `gate` / `agents.channel` / `experiments.runner`; currently ~97%);
    a `bandit -r src/preceptx -ll` security job; a weekly scheduled `pip-audit` workflow
    (`.github/workflows/audit.yml`). Added a `.gitattributes` **LFS allowlist** (final figures + the
    demo trace only, per CLAUDE.md) and `CITATION.cff`.
  - **Experiment Design Log (`docs/experiment_design_log.md`) + CLAUDE.md rule.** A dated log of
    experiment/research-design decisions — complementary to, not a replacement for, the CHANGELOG —
    with the difficulty-ladder fix as its first entry. CLAUDE.md §7 (ticket workflow) and "always
    do" §8 now require an entry when a change alters the experiment/research design.

- **Planning documents cross-referenced and brought current (23 Aug 2026).** All four planning documents
  had drifted from the implementation and from each other; each is now corrected against the shipped
  code and the August design decisions.
  - `RESEARCH_ROADMAP.md` — §0 rewritten from "the one open decision" to **compute resolved**: the UCL
    Myriad allocation was approved 23 Aug 2026 (1 GPU >=40GB, single node, 8h wall, 8 cores, 32GB RAM,
    ~45GB weights on scratch, ~250 episodes / 6-12k calls / single-digit GPU-hours), with three
    concrete jobscript gaps named — no node-class directive, no project code, and `HF_HOME` unset, which
    would download ~45GB into the home quota. §1 and §3.4 rewritten for the TraceElephant substrate and
    the replay-defined outcome. §3.1 now specifies the **free local pilot before the cluster**. §3.6
    rewritten: SocialJax cut on evidence with the replacement named, C5 deferred with a structural
    reason, and the middleware stretch documented. §4's phase table gains a status column stating
    plainly which phases are built-but-not-run. §5 gains R7-R10. A document-set index and a revision
    note were added at the top.
  - `DEPENDENCIES.md` — §1 replaced with the current work order, leading with the three blockers. §2
    gains the eleven new tickets and records that DSE-023 is superseded, DSE-024 rescoped and DSE-027
    cut. §4's risk register **renumbered to mirror the roadmap**, resolving a pre-existing collision in
    which both files used R1-R7 for different risks. §7 gains the real dataset identifiers and licences,
    the local pilot runtime (explicitly not a Python dependency), and the replay budget. §8 records the
    Phase-0 gate as passed and enumerates what the Y/V freeze actually covers.
  - `ISSUES.md` — execution guide rewritten with the merged/not-built state, the immediate critical
    path, and the two ordering constraints that are easy to violate (043+044 before the freeze; 045
    before 018).
  - `docs/experiment_design_log.md` — three dated entries for 23 Aug 2026: the RQ3a re-founding, the
    probe-validity hardening, and the contribution re-framing. Each records trigger, finding, impact if
    uncaught, risk reduced, correction path, fix, result and takeaway, per the log's template.
  - **Supervisor attribution corrected** across the roadmap and methodology to Prof. Philip Treleaven
    supervising, advised by Prof. Jun Wang, matching the approved research-computing record.

### Changed
- **Myriad jobscripts derive the served identity from the tier config** (`scripts/myriad/*.sh`).
  `serve.sh` and `pilot.sh` took `REVISION` on the `qsub` line while the manifest recorded the
  revision from `configs/model/<tier>.yaml`, and **nothing compared them** — the health check
  compares the served model *id*, but `/v1/models` carries no revision, so a typo or a stale
  copy-paste would serve one checkpoint and record another with every artefact well-formed.
  - `resolve_tier` now reads `name` and `revision` from the same file the manifest reads them from.
    `qsub -P <project> scripts/myriad/pilot.sh` is the whole command; `-v TIER=qwen8b` the whole
    fallback. `TIER` is also now **exported** to the child `serve.sh`, which it was not: `-v
    TIER=qwen8b` alone previously drove an 8B pilot against a 14B server.
  - `MODEL`/`REVISION` survive as overrides for the 70B-AWQ tier, which has no config file until
    DSE-005 pins its repo id. An override contradicting the config prints a warning naming both
    values, because in a job log a deliberate override and a typo look identical.
  - A unit test asserts every `configs/model/*.yaml` carries a name and a full 40-character commit
    SHA — the invariant the shell now depends on.
- **The cluster environment installs from `uv.lock`, not from `pyproject.toml`.** `docs/myriad.md`
  §6 said `uv pip install -e '.[serving,embed]'`, which re-resolves against whatever PyPI serves
  that day; the cluster run is the run of record and is the last place to install off-lock. Now
  `uv sync --extra serving --extra embed`, which also removes the separate `~/venvs/...` path: sync
  populates the repo's `.venv`, which is what all three scripts now default `VENV` to. The lock
  already carries vLLM's `manylinux_2_31_x86_64` wheel, so nothing compiles from source.
- **`SweepManifest` gains `gate_feedback_version`** (DSE-045). The manifest schema is one of this
  repo's two stable contracts, so this is a deliberate, recorded bump; it defaults to the live
  `GATE_FEEDBACK_VERSION`, so existing readers are unaffected and no result re-freezes (the field
  is not part of any dataset hash).
- **Planning documents reconciled with the built state** — they had drifted into describing finished
  work as pending, which is the failure mode that makes a planning doc worse than none.
  - `ISSUES.md`: **DSE-050 added** as a proper ticket (it was built this session from a proposal
    that existed only in conversation); the dated state snapshot corrected — it listed 031–033,
    041, 043–046 and 049 as "not built" when all are built, and claimed **zero episodes recorded**
    when E1 and E3-local v3/v4 exist at the local 4-bit tier; the immediate-critical-path paragraph
    rewritten, because what now stands before the Y/V freeze is **a run, not a build**.
  - `DEPENDENCIES.md`: DSE-050 added to the dependency table (**blocks every cluster run**);
    DSE-045 marked built with the hash caveat; a new §1 item 6 naming the bf16 re-gate as the one
    remaining pre-freeze step.
  - `RESEARCH_ROADMAP.md`: the "three practical consequences not yet reflected in the jobscript"
    block had two items that the jobscript now closes (node class + `-P`, and `HF_HOME`); both
    struck through with what closed them, and the two defects found *while* closing them — the
    256 GB memory request and the wrong CUDA module name — recorded as items 4 and 5.
  - `docs/EXPERIMENTS.md`: E3 gains its execution path (`qsub ... scripts/myriad/pilot.sh`) and the
    note that the grid axes come from the CLI defaults, not the jobscript, so the executed cell and
    the pre-registered one cannot drift apart.
  - `README.md`: `docs/myriad.md` and `docs/serving.md` added to the document index; the two
    "serving runs on Myriad — see serving.md" pointers split, since serving.md is now the ladder and
    wire format while myriad.md is the cluster.
- **`docs/serving.md`** now points at `docs/myriad.md` for cluster mechanics and keeps only the
  model ladder and the wire format; the node-class note is corrected from "edit the jobscript" to
  the `-ac allow=` qsub override, and it carries the per-slot `mem` warning.
- **`ShuffledMessageAudit` implements the corrected permutation-test criterion** (`experiments/rq1.py`).
  The model gains `null_max_cpvi` and `p_value` = `(1 + #{null ≥ real}) / (n_perm + 1)`; its
  docstring no longer claims the null "must collapse toward 0". The pre-registration was corrected
  to the permutation test earlier in this cycle, but the code still computed only mean and spread,
  so the criterion PREREGISTRATION §8 states had no implementation. E3-local v4 under the frozen
  estimator: real **+0.066** against a null of **+0.033 ± 0.006**, max **+0.046**, **p = 1/21**.
- **`qsub` examples repaired** (`scripts/myriad/serve.sh`, `docs/serving.md`). Making `REVISION`
  mandatory earlier in this cycle invalidated every documented launch line; all now pass
  `-P <project>` and `-v REVISION=<sha from configs/model/*.yaml>`, and `serving.md` no longer tells
  the reader to uncomment a `#$ -P` directive that no longer exists.
- **`PilotReport` provenance line carries the repeat count** (`R=5`), now that it is a
  pre-registered, load-bearing part of the estimator rather than a default.
- **RQ1 per-condition CPVI/PVI intervals are episode-cluster bootstrap** (`experiments/rq1.py`;
  result-affecting for every reported interval on a per-handoff quantity). `_condition_summary` now
  derives per-condition episode groups and calls `cluster_bootstrap_ci`; episode-level quantities
  (success) keep the plain episode bootstrap. The E3 results table in `docs/EXPERIMENTS.md` §7 is
  corrected in place with the method named — the two claims that survive the honest intervals are
  the ones the entry leads with (C0's CPVI excludes zero; the C0−C4 gap +0.211 bits holds
  [+0.060, +0.349]).
- **`sweep_hash` excludes `concurrency`** (`experiments/sweep.py`; identity-affecting for future
  datasets). Concurrency is an execution knob, and hashing it re-keyed the dataset whenever a
  resumed run changed worker count — orphaning every completed episode under the old hash, the
  expensive failure on an 8-hour Myriad wall clock. Recorded local dataset hashes in the docs
  remain the correct identifiers of what was run under the old derivation.
- **PREREGISTRATION §8's shuffled-message control is now a permutation test** (pre-freeze wording
  correction, recorded there and in the design log). The old criterion — shuffling "must collapse
  CPVI" — is structurally unattainable: within-condition permutation preserves the condition-level
  signatures message *style* carries (an 8-token C1 message stays recognisably C1) and per-handoff
  progress base rates differ by condition (E3-local: C0 0.255, C1 0.735, C3 0.379, C4 0.575), so a
  permuted message still betrays its condition. New criterion: the real pooled mean CPVI must
  exceed every permutation (E3-local: real +0.078 bits vs null +0.043 ± 0.006, max +0.057 — passes
  at p ≈ 1/21); the null's height reads as the *identity* component of CPVI, the excess as
  per-handoff message content.
- **`scripts/myriad/serve.sh` hardened against the approved allocation** (infra): requests the A100
  node class (`#$ -ac allow=L`; EF for the V100-fitting 8B tier and U/V for 32B bf16 documented in
  place), **requires** a pinned `REVISION` (the `main` default is gone — same rule as the
  featuriser's pin check), and exports `HF_HOME` to `~/Scratch/hf-home` so ~45 GB of weights land
  on scratch rather than the home quota. `-P` stays a qsub-line argument: no usable default exists.
- **E3 pilot cell now includes C3, and G2 gained a third verdict state** (`experiments/cli.py`,
  `experiments/pilot.py`; result-affecting — it changes what the pilot can certify).
  - `_PILOT_CONDITIONS` is `["C0", "C1", "C3", "C4"]` (24 episodes at three seeds, was 18). C3 is the
    only condition carrying a genuine observation asymmetry and it is in the headline design; a pilot
    that never exercises it certifies an instrument the main sweep will not use. Measured CPVI on the
    v4 cell: **+0.051 bits** (episode-cluster interval [+0.002, +0.103]).
  - `GateResult.assessable: bool = True`, for the case that genuinely admits no verdict: every
    handoff carrying the same progress label, so CPVI has nothing to predict. `_recommendation` holds
    an unassessable gate at `retune_once` **on any attempt** — it never yields `proceed` and never
    escalates to the fallback ladder, because an absence of data is not evidence about the design.
    `render_report` prints `UNASSESSABLE` rather than `FAIL`.
- **Prompt surface v4** (`sim/serialise.py`, `agents/prompts.py`; result-affecting, bump **two of the
  three** budgeted pre-E3 bumps).
  - The numeric form gained `load_size=(1.4000, 1.3000)  # (bar length, height across bar+stem)`.
    The state named the gap's extent and never the object's, so "aligned with the slit" was
    underdetermined: the threading band for a 1.3-tall load in a 1.8 gap is ±0.25 about the centre,
    not the full ±0.9, and that constant was not in the prompt. E3-local watched A call
    `com_y = 2.0074` aligned with a `(2.1, 3.9)` gap and push east into the wall for the remaining
    budget. The dimensions are **constants of the load, not a derived pass band** — the alternative
    of printing the threading band was considered and rejected as performing the agents' inference.
  - `_SYSTEM_A` now states that the **whole** load must fit the slit, its centre being inside the
    range not being enough, and points at `load_size`.
  - Dataset identity moves with it: the re-run writes to `c0bd4d7499f01d97` (sweep `c163d616d1608140`),
    not the v3 dataset, so no v3 and v4 episodes can pool.
- **Prompt surface bumped v2 → v3 (E1 transcript read).** The first five real episodes produced
  **7 distinct A-messages across 75 handoffs** and action `E` **75 times out of 75**; the cause was
  that the `numeric` serialisation carried **no wall or slit geometry at all**, so A had nothing
  state-specific it could say. One of the three budgeted pre-E3 bumps. See
  `docs/experiment_design_log.md` (2026-08-24) for why a flat gradient here would have been
  indistinguishable from an honest null.
  - `sim/serialise._numeric` gains `walls_x=(4.0000, 8.0000)` and `slit_y=(lo, hi)` with the slit
    width in the trailing comment. This **de-confounds the serialisation axis**: the grid *drew* the
    walls and slits and the NL form *named* the nearest slit centre, so the module's stated
    information-isomorphism was false in exactly one of three branches, and a numeric-vs-grid
    difference would have been partly an information difference reported as a representation effect.
    The `load=` line stays first, so `_parse_numeric_load` is unaffected.
  - **A's system prompt** now states the convention (+x toward the goal, +y north), states that
    passage depends on the load's y matching the slit's y-range, and asks for the load's position
    *relative to the next slit* plus the next move — "use the actual numbers in front of you; do not
    give generic advice". The old double instruction to be brief is gone.
  - **B's system prompt** now says A sees more of the scene and to follow A's instruction unless its
    own observation plainly contradicts it; the action hint spells out the axis meanings
    (`N` = +y, `E` = +x, …), closing E1's coordinate-convention check.
- **The manifest now records how a run was decoded, not just where it was served.**
  `SweepManifest.serving_a` / `serving_b` carry the resolved `ServingConfig` per role with the api
  key replaced by `REDACTED`. Temperature, decoding seed, token budget, retry count and the thinking
  switch all shape what the model emits and none of them live in `SweepConfig`, so two datasets
  differing only in `max_tokens` were previously indistinguishable after the fact.
- **Dataset identity now moves with the prompt version.** `data.writer.dataset_hash` takes an
  optional `prompt_version` (empty default preserves every pre-v3 hash), and the new
  `experiments.sweep.dataset_hash_for(sweep)` — the single derivation the runner, both drivers, the
  CLI and the tests all use — folds `PROMPT_VERSION` in. `sweep_hash` covers the sweep config, which
  carries no prompt version, so **without this a prompt bump resumed into the previous prompt's
  dataset directory and silently pooled two prompt surfaces into one set of episodes**. A unit test
  pins the guarantee.
- **S1 substrate adapters — found on the first live local call, not in review.** LM Studio's MLX
  runtime ignores `chat_template_kwargs`, so Qwen3 stayed in thinking mode: the reasoning went to a
  non-standard `reasoning_content` field and `content` came back **empty with HTTP 200**.
  - `ServingConfig.thinking_switch` (default `""`, leaving the vLLM path untouched) appends an
    in-band switch — `/no_think` for Qwen3 — to the **final user turn only**; system turns are never
    touched. Both routes select the same non-thinking template branch, so this is a substrate
    adapter rather than a prompt change, and the difference is recorded per run.
  - `LLMClient.chat` now **raises on empty or whitespace-only content**, not just on `None`. An
    episode of empty A-messages would otherwise have looked like a completed run rather than a
    failed one — the exact fail-open shape CLAUDE.md forbids. The error names the likely cause.
  - `health_check`'s ping asks for 16 tokens instead of 1, so a runtime stuck in thinking mode fails
    the check **before** a sweep starts rather than mid-episode.
- **DSE-049 / DSE-032 — sweep schema and manifest surface** (result-affecting by CLAUDE.md's
  reproducibility-contract rule; no results were frozen, zero episodes exist, so nothing re-freezes):
  - `SweepConfig` gains `model_b: ModelConfig | None = None` (the optional second serving block) —
    it is inside `sweep_hash`, so a heterogeneous-pair dataset can never collide with a self-play one.
  - `SweepManifest` gains `model_b_name`, `model_b_revision`, `endpoint_base_url_b` and
    `structured_mode`; all are outside the hash, matching the existing treatment of
    `serving_substrate` (environment properties separate datasets by root, not by hash).
  - `EpisodeRunner(client_a, client_b=None, ...)` and `run_grid(sweep, client_a, client_b=None, *,
    root)`: agent A's message and agent B's structured action now go to their own clients. Omitting
    `client_b` points both roles at one client and reproduces the previous path exactly (asserted by
    a record-for-record equality test). `run_rq1` takes `client_b` as a keyword.
  - `run_grid` **fails loud when `sweep.model_b` and `client_b` disagree**: a second endpoint with no
    declared model identity would leave the manifest lying about what served role B.


- **Review pass on the 23 Aug doc set (24 Aug 2026)** — surgical tightenings after cross-checking the
  ground-up status PDF against the shipped docs; no design change.
  - The pre-registration artefact is now **named**: `PREREGISTRATION.md` at the repo root, v0 drafted
    during S1 while the system is fresh, v1 committed on a `proceed` verdict (EXPERIMENTS.md E3/§6,
    methodology §9.10).
  - Numeric gate defaults recorded in the run ledger, all frozen or re-set at F0 (E3: G1 success floor
    0.5 on easy C0, G3 groundedness floor 0.8, G2 directional until the pilot fixes the bit-scale;
    E7: firing-rate budget 0.2, ECE unreliable below N = 200).
  - S1 names the local runtime: LM Studio driven headless via `lms`, substrate label `local-lmstudio`,
    manifest model identity = the quantised repo + revision SHA, and the constrained-decoding engine
    difference (Outlines/llama.cpp locally vs xgrammar under vLLM) as a second reason the local
    schema-adherence rate is its own number.
  - Methodology §10.5 preamble now states the freeze-clock bound: zero episodes recorded, so no
    register entry can be outcome-contingent, and the register closes at F0 (later deviations go to
    `PREREGISTRATION.md`'s own deviation log).
  - Methodology synced to the v3 finding (24 Aug 2026): §9.3 records that the *numeric* form named
    neither wall nor slit until the first transcript read — the matched-information claim was false
    in the arm a reader would assume most complete — and the deviation register gains **D19** with
    the before/after message counts. §9.6 now declares the non-thinking decoding regime (reasoning
    mode switched off explicitly and recorded in the manifest) so the text scored at the boundary is
    the text the receiver saw.
  - Citations: Who&When corrected to Zhang et al. 2025**a** in §2; the three 2026 preprint
    author-list placeholders resolved against arXiv (Ao, Gao and Simchi-Levi — reliability limits;
    Zhang, B. et al. — AgentForesight; Shah — Causal Agent Replay) and the entries re-alphabetised.
- **Difficulty ladder + rotation retuned pre-freeze (P1-4 finding; result-affecting).** The
  feasibility search showed the shipped ladder was partly infeasible: a rigid T (bar 1.4
  perpendicular to stem 1.0) cannot thread a thin-wall gap narrower than its **shorter member (the
  stem, 1.0) at any orientation** — rotation does not shrink the threading cross-section — so medium
  (slit 1.0) was zero-clearance and hard (0.7) impossible, while `rq1_sweep` defaults its headline to
  hard. Fix: slit widths **1.0/0.7 → 1.2/1.1** (easy 1.8 kept), all above the 1.0 threshold and graded
  by threading clearance; `StepConfig.angular_impulse` **2.0 → 0.5** (~34°/action — the old value spun
  the small-moment T ~135°, reachable only at 45° multiples, too coarse to aim). `SweepConfig.max_steps`
  is now a per-difficulty `dict[Difficulty, int]` (default = certified `STEP_BUDGETS`; a bare int
  broadcasts to all difficulties) wired through `EpisodeRunner` and `rq1_sweep`, replacing the single
  `max_steps=12` that under-fed hard — a `SweepConfig`-schema change, so `sweep_hash` rolls. No result
  is frozen and no dataset exists, so nothing to re-freeze. Full first-principles rationale in
  `docs/experiment_design_log.md` (2026-07-25).
- Repositioned `ISSUES.md` and `RESEARCH_ROADMAP.md` to the **standalone** posture mandated by
  CLAUDE.md: the repo does not depend on or import precept. OTel capture (DSE-004) uses a vanilla
  OpenTelemetry SDK exporter; the runtime gate (DSE-018) is the in-repo `RuntimeGate`
  (`gate/integration.py`, formerly `precept_integration.py`). Dissertation/project naming and the
  "upstream to precept later" framing are unchanged.
- Renamed planning docs to the canonical names CLAUDE.md references:
  `ISSUES - 15 June.md` → `ISSUES.md`, `RESEARCH_ROADMAP-15 June.md` → `RESEARCH_ROADMAP.md`.
- Isolated `sentence-transformers` (and its `torch` dependency) to an `embed` extra rather than core,
  so the analysis install and CI stay fast and torch-free. Deviation from the roadmap's primary-deps
  list, noted in `DEPENDENCIES.md` §3.
- CI runs Python 3.11 only (CLAUDE.md's pinned-single-version rule), narrowing DSE-001's stated
  3.11/3.12 matrix.

### Removed
- **`langchain-openai`** — declared as a core dependency but never imported (the code uses the raw
  `openai` client). Dropped from `pyproject.toml`, the mypy per-module override list,
  `DEPENDENCIES.md` §3, and `uv.lock`. `bandit` added to the `dev` extra for the new CI security job.

### Notes
- DSE-002's live-on-Myriad verification (one tier served + health check passing on the cluster)
  is deferred until cluster access is available; all authorable parts (script, client, mock tests,
  docs) are complete.
- DSE-003's determinism harness is verified against a mocked endpoint; the real fixed-seed run on the
  served 8B tier (DSE-003 acceptance) is deferred until Myriad access, like DSE-002's live check. The
  config-tree model revisions are placeholder `main` and must be pinned to commit SHAs before any
  recorded run (`ModelConfig` already rejects an empty revision).
- The review-correction pass resolves the DSE-003 note above: the config-tree model revisions are no
  longer placeholder `main` — all three yamls pin HF commit SHAs. The *encoder* revision
  (`EncoderConfig.revision="main"`) remains to be pinned before the Phase-2 freeze, as already noted
  in DSE-013.
- Bundle 3 tooling (BFS feasibility/P1-4, renderer, shuffled-message audit) and Bundle 4 hygiene
  (coverage gate, bandit/pip-audit, dependency prune, `.gitattributes`, `CITATION.cff`) have now
  landed (above). Still deliberately deferred: the **DSE-005 model-ladder harness** (its value needs
  a served endpoint; write-and-run once compute is up); **P1-14 per-role clients** (lands with its
  only consumer, DSE-021); **P1-1 `InfoStatistic` label repoint** (one line at Y-freeze); the
  research one-pagers (`PREREGISTRATION.md`, the RQ3a Y-on-logs note, the Who&When/MAST loader spike);
  and the thesis-text drift items (D-1..D-6).
- A pre-existing `UP038` lint (`isinstance(x, (int, float))` → `isinstance(x, int | float)`) in
  `determinism.py` (DSE-003) was fixed in passing on the DSE-006/007 branch: pre-commit's pinned
  ruff enforces the rule while the uv-installed ruff (where it is deprecated) did not, so the commit
  hook failed on otherwise-green code. No behaviour change. Aligning the two ruff versions is a
  separate follow-up.
