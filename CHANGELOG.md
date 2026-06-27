# Changelog

All notable changes to **precept-research** are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). This repo has no semver public API; the
stable contract is the `RunManifest` and `ExperimentConfig` schemas (see CLAUDE.md). Behaviour- or
result-affecting changes get an entry; result-affecting changes also re-freeze the affected result.

## [Unreleased]

### Added
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

### Changed
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

### Notes
- DSE-002's live-on-Myriad verification (one tier served + health check passing on the cluster)
  is deferred until cluster access is available; all authorable parts (script, client, mock tests,
  docs) are complete.
- DSE-003's determinism harness is verified against a mocked endpoint; the real fixed-seed run on the
  served 8B tier (DSE-003 acceptance) is deferred until Myriad access, like DSE-002's live check. The
  config-tree model revisions are placeholder `main` and must be pinned to commit SHAs before any
  recorded run (`ModelConfig` already rejects an empty revision).
- A pre-existing `UP038` lint (`isinstance(x, (int, float))` → `isinstance(x, int | float)`) in
  `determinism.py` (DSE-003) was fixed in passing on the DSE-006/007 branch: pre-commit's pinned
  ruff enforces the rule while the uv-installed ruff (where it is deprecated) did not, so the commit
  hook failed on otherwise-green code. No behaviour change. Aligning the two ruff versions is a
  separate follow-up.
