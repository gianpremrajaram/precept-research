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
