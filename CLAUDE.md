# CLAUDE.md

> Operating guide for Claude Code sessions on the **precept-research** repository. Read this end-to-end before starting any ticket.

---

## What precept-research is

A standalone, end-to-end research project measuring **conditional pointwise V-usable information (CPVI)** at the natural-language boundary between two coordinating LLM agents, using a T-shaped cooperative-transport task in a Pymunk arena under a degradable communication channel.

- **Four components that compose cleanly:** (1) a deterministic 2D physics simulator, (2) a two-agent LangGraph negotiation loop, (3) locally served open-weight models behind a vLLM OpenAI endpoint, (4) a measurement-and-gate stack that scores each handoff offline (CPVI) and online (a target-free statistic) and can block low-information handoffs.
- **Four research questions in priority order:** RQ1 information gradient, RQ2 measurement primitive, RQ3a external validity (Who&When/MAST), RQ3b causal gate. The roadmap (`RESEARCH_ROADMAP.md`) is the design authority; do not duplicate it here, point to it.
- **Standalone by design - this repo does NOT depend on precept.** The runtime gate is implemented in this repo. Findings may later be upstreamed to precept, but that work lives in the precept repository, not here. Sequence is fixed: finalise the research, then incorporate precept.
- **NOT in scope:** precept integration, a published library or stable public API, production serving, anything beyond the dissertation and its September paper.

---

## Authoritative planning documents

**GitHub Issues is the source of truth for the backlog.** The local `ISSUES.md` and `DEPENDENCIES.md` files mirror GitHub for offline reference and are large. **Do not read them wholesale** - pull only the slice you need.

If a ticket on GitHub conflicts with this CLAUDE.md, the GitHub ticket wins for that specific ticket. Raise the conflict explicitly rather than silently picking one.

### Lookup index

Use the narrowest read that answers the question.

| Need | How to get it |
|---|---|
| Active ticket details | `gh issue view <number> --json title,body,labels,state` (DSE-XXX is the title prefix, not the number) |
| Find a ticket by ID | `gh issue list --search "DSE-XXX in:title" --state all --json number,title,state` |
| Open backlog | `gh issue list --state open --limit 50` |
| Recent merges | `git log --oneline -20` |
| Dep graph for a ticket's prerequisites | `grep -n "DSE-XXX" DEPENDENCIES.md` then read that section only |
| Risk register | DEPENDENCIES.md §4 (read just that section) |
| Cross-cutting concerns | DEPENDENCIES.md §5 |
| Phase / result-freeze gates | DEPENDENCIES.md §8 |
| Critical-path priorities | DEPENDENCIES.md §1 |
| Architectural constraints | this file, "Critical architectural constraints" section |
| Code style rules | this file, "Code style and architecture" section |
| Offline ticket lookup (no network) | `grep -n "^## DSE-XXX" ISSUES.md` then read that line range only |

---

## Ticket workflow

1. **Pick a ticket.** Fetch it: `gh issue view <number> --json title,body,labels,state`. Confirm its listed dependencies are merged on `main` (cross-check DEPENDENCIES.md §2 only if the listed deps look ambiguous). If a blocker isn't merged, stop and work on the blocker first.
2. **Branch.** One branch per unit of work handed back. Single ticket: `<type>/DSE-XXX-short-slug`. Types: `feat`, `fix`, `docs`, `infra`, `test`, `research`, `exp`. Example: `exp/DSE-021-rq1-sweep`. **Multiple tickets picked up in one go go on a *single* branch**, named `<type>/DSE-XXX-YYY-theme` for a contiguous range or `<type>/DSE-XXX+ZZZ-theme` when non-contiguous, so the completed work pushes once. Example: `infra/DSE-001-002-foundation`. Use the `<type>` of the earliest/dominant ticket and a slug naming the shared theme. **Do not create stacked per-ticket branches for work handed back together** - that forces multiple pushes and PRs for one deliverable.
3. **Scope discipline.** Do exactly what the Acceptance Criteria list, nothing more. If you spot scope creep mid-ticket ("while I'm here, let me also..."), write it as a follow-up issue - do not pull it into the current branch.
4. **Tests alongside code.** Every acceptance-criteria test listed in "Testing Requirements" must be written and passing before the branch is handed back.
5. **Commit style.** Conventional Commits, **one commit per ticket** even when several tickets share a branch (keeps per-ticket history reviewable). Example: `feat(sim): add T-load and macro-action interface (DSE-009)`. Include the ticket ID in the body if not in the title.
6. **PR title and body (for the human maintainer to open).** Title: `DSE-XXX: <one-line summary>`, or `DSE-XXX-YYY: <summary>` for a grouped branch. Body: Summary, Linked issues (list every ticket, e.g. `Closes #1, #2`), Testing, Checklist (tests pass, types check, lint clean, CHANGELOG updated). You prepare the branch; you do not open the PR (see "Things to never do").
7. **CHANGELOG.md.** Update the `[Unreleased]` section under the appropriate category (Added, Changed, Deprecated, Removed, Fixed). Every behaviour or result-affecting change gets an entry.
8. **Definition of Done is a gate.** Do not mark a ticket done until every DoD item is verified. "Tests pass" means "pytest exits 0 locally AND CI is green".

---

## Code style and architecture

### When to use classes vs functions

Use OOP when: (1) a component holds injected dependencies or mutable state (`Evaluator`, `EpisodeRunner`), (2) multiple concrete implementations share a stable interface (`Statistic` ABC, `Channel` ABC), or (3) you need lifecycle methods. Use module-level functions when logic is stateless and composable (featurisers, estimators, serialisers). Prefer Pydantic models over dataclasses for all config and metadata types. Numeric arrays are plain `numpy`, not wrapped in classes. Never create a class just to namespace functions - use a module for that.

### Concrete application in this repo

| Component | Class or function? | Reason |
|---|---|---|
| `ExperimentConfig`, `RunManifest`, `EpisodeRecord`, `HandoffRecord`, `CpviResult` | Pydantic model | Config and metadata; validation; `model_dump` for serialisation. |
| `Channel`, `Statistic`, `Probe` | ABC | Multiple concretes sharing a stable interface. |
| `LengthCapChannel`, `DelayChannel`, `NoiseChannel`, `AsymmetricChannel` | Class inheriting `Channel` | Each implements one degradation of the A-to-B message. |
| `InfoStatistic`, `FailStatistic`, `CosineStatistic` | Class inheriting `Statistic` | Each is a runtime score computable at the handoff. |
| `RuntimeGate` | Class | Holds an injected `Statistic` and a calibrated threshold. |
| `EpisodeRunner` | Class | Holds the compiled LangGraph, the simulator handle and the logger. |
| `Arena` / simulator | Class | Holds the Pymunk `Space` and load body; mutable physics state. |
| `build_arena`, `add_t_load`, `apply_macro_action`, `read_state` | Module-level function | Pure-ish space transforms; no cross-call state of their own. |
| `serialize_state` (numeric / grid / nl) | Module-level function | Stateless state-to-prompt transform; the mode is a config flag. |
| `featurize`, `fit_probe`, `cpvi`, `pvi` | Module-level function | Stateless numeric transforms on frozen embeddings. |

### Pydantic and config conventions

- Pydantic v2 only. Use `ConfigDict`, `field_validator`, `model_dump`, `model_validate`.
- All config and metadata types get `model_config = ConfigDict(extra="forbid")` unless there is an explicit reason to allow extras.
- **Hydra composes, Pydantic validates.** Sweep configuration is composed with Hydra/OmegaConf; at every entry point the resolved config is validated into an `ExperimentConfig` via `ExperimentConfig.model_validate(OmegaConf.to_container(cfg, resolve=True))`. Experiment code consumes the validated `ExperimentConfig`, never a raw `DictConfig`.
- Never use dataclasses for data that crosses module boundaries. `from __future__ import annotations` at the top of every module.

### Type discipline

- `mypy --strict` is mandatory on `src/`. CI will reject branches that fail it.
- Untyped third-party libraries are handled the standard way: declare `ignore_missing_imports = true` per-module in `[[tool.mypy.overrides]]` (for `pymunk.*`, `sklearn.*`, `langgraph.*`, `langchain_openai.*`, etc.). **Never** use that override to silence our own code.
- Type numeric arrays with `numpy.typing.NDArray[np.float64]` in signatures; do not let arrays degrade to bare `Any`.
- No `Any` in our public function signatures unless genuinely polymorphic (a serialised prompt-field value is the only legitimate case). Prefer `typing.Literal` for enum-like strings (`mode: Literal["numeric", "grid", "nl"]`, `condition: Literal["C0", "C1", "C2", "C3", "C4"]`).

### Reproducibility surface (this repo has no semver API)

This repo is not a published library, so there is no `__all__` semver commitment. The contract that must stay stable instead is **the `RunManifest` schema and the `ExperimentConfig` schema**. Changing either is a result-affecting change: bump it deliberately, note it in the CHANGELOG, and re-freeze any results that depended on the old shape.

### Error handling - fail loud, not fail-open

- Research code **fails loud**. A broken experiment must crash visibly, not silently produce a passing-looking run. This is the deliberate inverse of a production library's fail-open posture.
- Raise specific, named exceptions (`ConfigError`, `GateBlockedError`, `GroundingError`). Wrap and re-raise third-party exceptions at the module boundary.
- Never use bare `except:` or `except Exception:`. Catch specific exception types.

### Logging

- Module-level loggers only: `logger = logging.getLogger(__name__)`.
- DEBUG for per-step trace detail; INFO for lifecycle events (run start, model load, sweep cell complete); WARNING for degraded modes; never ERROR (errors are exceptions, not log lines).
- **Do not log full prompts, messages, or physics state at INFO or above.** Those belong in the per-run artefacts (`handoffs.jsonl`), not the console stream.
- Library/experiment code never configures handlers; the run entry point owns `logging.config`.

### Naming

- UK spelling is NOT used in code identifiers (Python convention is US: `color`, `serialize`). UK spelling IS used in docstrings, comments, markdown and user-facing strings. *(If existing research code already uses UK identifiers, flip this rule to match the code rather than churning it.)*
- Classes: PascalCase. Functions/variables: snake_case. Constants: UPPER_SNAKE_CASE. Private helpers: `_leading_underscore`.
- Module names: lowercase, short, no underscores unless needed (`channel.py` not `comm_channel.py`).

### Implementation discipline

- **Surface assumptions before coding.** If a request admits multiple reasonable interpretations, name them and ask - do not pick silently. Ambiguity hidden upfront becomes rework at review.
- **Minimum code that satisfies the Acceptance Criteria.** No speculative abstractions, no configurability nobody asked for, no error paths for states that cannot occur. If a 200-line diff could be 50, rewrite it.
- **Surgical edits.** When modifying existing code, touch only what the change requires. Do not reformat adjacent code, do not refactor working code, do not delete pre-existing dead code (flag it in the handback notes instead). Every changed line should trace to the ticket.
- **Orphan cleanup is yours; pre-existing cruft is not.** Remove imports, variables and symbols that YOUR change leaves unused. Leave unrelated dead code alone unless the ticket scope covers it.

---

## Toolchain

### Python

- **Python 3.11, pinned single version.** Reproducibility and the vLLM / torch / pymunk matrix on Myriad favour one interpreter over a test matrix; this is a solo research repo, not a distributed library.

### Package management

- **uv** is the package manager. `uv venv` to create the environment, `uv pip install -e .` for dev, `uv lock` / `uv sync` for the reproducible lockfile. On Myriad, install uv into user space (single static binary).
- `uv.lock` is the reproducibility anchor and is committed. Regenerate it via `uv lock` when deps change.
- Never add a runtime dependency without updating `pyproject.toml`, regenerating `uv.lock`, AND updating DEPENDENCIES.md §3.
- Pin external dependencies with both floor AND upper bound (e.g. `pydantic>=2.5,<3`).

### Lint, format, type

- `ruff check .` for linting, `ruff format .` for formatting. **`black` is NOT used** - `ruff format` is its drop-in replacement; installing black creates a dual-formatter conflict.
- `mypy --strict src/` for type checking.
- `bandit -r src/ -ll` for security scanning; `pip-audit` (against the exported requirements) weekly via a scheduled Action.
- `pre-commit install` on first clone. Hooks auto-run on commit; never bypass with `--no-verify`.

### Testing

- `pytest` with coverage: `pytest --cov=src --cov-report=term-missing`.
- Test layout mirrors source layout: `src/sim/arena.py` -> `tests/unit/sim/test_arena.py`.
- Three tiers:
  - `tests/unit/` - no I/O, single module, runs in < 30s total, executed on every commit. Includes the **determinism tests**: a fixed seed must yield an identical simulator trajectory; the CPVI estimator must return the known answer on a synthetic fixture where the message is constructed to carry (or not carry) information.
  - `tests/integration/` - exercises the full `EpisodeRunner` plus the measurement stack on a tiny fixture and a stub LLM, may take 1-3 minutes, runs on every PR.
  - `tests/e2e/` - real vLLM + LLM calls, manual runs only.
- Property-based tests via `hypothesis` for input-validation logic (config schema, channel transforms, serialisers).
- **Coverage gate: ≥ 80% on the load-bearing core** - `src/sim/`, `src/channel/`, `src/measure/` (CPVI estimator and runtime statistics), and `src/runner/`. A silent bug in these invalidates results; everything else does not chase coverage.

---

## Compute and serving (Myriad + vLLM)

- Serve one model per GPU job behind vLLM's OpenAI-compatible server; the LangGraph client points at the local endpoint. The model ladder and GPU envelope are fixed in the roadmap §0 - do not duplicate the table here.
- **Determinism settings are not optional.** Greedy decoding (`temperature=0`), fixed `seed`, pinned model revision. Structured action output uses vLLM guided decoding (xgrammar/outlines) against a JSON schema, so parser brittleness is removed from the action channel.
- Batched LLM inference is **not bit-exact** across runs. Determinism in this repo means "low-variance, seed-pinned, revision-pinned", never "exactly reproducible". Report seed sensitivity; never claim exact reproducibility of LLM runs.
- SGE jobscripts live under `infra/` and request the GPU explicitly. The serving command, model revision and seed are recorded in the run manifest (below), not just in the jobscript.

---

## Critical architectural constraints

### Determinism and pinning
- Every run pins and records: random seed, model revision, embedding-encoder revision, and the resolved config. These go in the manifest. A result with an unrecorded revision is not a result.
- Embeddings are computed **once** with a pinned encoder and cached by content hash; probes fit on frozen embeddings. Do not re-embed per probe fit.

### The channel degrades one thing only
- `apply_channel` touches the A-to-B message and nothing else. Any outcome difference between conditions C0-C4 must be attributable to the channel, not to a task or model change. Do not let a channel implementation reach into physics state or the action path.

### Y and V are frozen before the main runs
- The outcome variable Y and the probe family V are chosen and **frozen** before the RQ1 main sweep (roadmap Phase 2 gate). Re-selecting Y or V after seeing main-run results is a forbidden researcher degree-of-freedom. If a pilot forces a change, it happens before freeze and is logged.

### The runtime statistic never sees the realised outcome
- The gate thresholds a statistic computable **at the handoff** (`InfoStatistic`, `FailStatistic`, `CosineStatistic`). It is calibrated offline against **realised outcomes**, never against CPVI - calibrating against CPVI is the circularity error and is banned. `CosineStatistic` is probe-independent and exists to answer the circularity objection directly.

### CPVI is always the conditioned quantity
- CPVI = the state-plus-message probe minus the state-only probe. **Always report the `PVI − CPVI` gap** (how much apparent message value was an echo of the shared state). Never report message value without the state-only baseline.

### The gate is in-repo and pluggable
- `RuntimeGate` is implemented in this repository. **precept is not a dependency and is not imported.** The gate may later be swapped for a better-known method or a small purpose-built one; keep `Statistic` and `RuntimeGate` cleanly separable so that swap is a new concrete, not a rewrite.

---

## Specific named constraints

### Run manifest is mandatory
Every run writes `manifest.json` containing **at least**: git commit; dataset or condition identifier; resolved config (inline or path + content hash); seed; the exact command; model revision; encoder revision; key metrics; and artefact paths. A run without a complete manifest is not audit-usable and does not count as done.

### Run-directory layout
```
runs/<experiment>/<run_id>/      # run_id = UTC timestamp + short git SHA + config hash
  config.yaml      # resolved config            (committed only when the run is frozen)
  manifest.json    # mandatory fields above      (committed only when the run is frozen)
  metrics.json     # key metrics                 (committed only when the run is frozen)
  handoffs.jsonl   # per-handoff raw records     (gitignored)
  probes/          # trained probes              (gitignored)
  embeddings/      # cached embeddings           (gitignored)
  figures/         # final figures               (committed via the LFS allowlist)
```

### Pre-registration before main runs
Primary hypotheses and the analysis plan are fixed before the RQ1 main sweep. Deviations are logged with a reason. This is what separates a confirmatory result from a fished one.

### Statistical reporting
Report **effect sizes and uncertainty intervals, not just significance**. Control family-wise error across the condition contrasts (Holm or Benjamini-Hochberg). Use a mixed-effects model of outcome on condition with random effects for seed and episode; enter CPVI as a mediator to test H2. Always report seed sensitivity given LLM non-determinism.

### Phase-1 gates are hard go/no-go
G1 capability (self-play solves C0 above the floor), G2 signal (a measurable C0-to-hard gap in both outcome and CPVI), G3 groundedness (messages reflect true state). Failing a gate after one retune triggers the fallback ladder (elevate RQ3a to the headline), **not** a scramble. RQ3a is the pre-planned fallback that can carry the dissertation alone.

---

## Things to never do

1. **Never import precept or add it as a dependency.** This repo is standalone; the gate lives here.
2. **Never add a runtime dependency** without updating `pyproject.toml`, regenerating `uv.lock`, AND updating DEPENDENCIES.md §3.
3. **Never freeze Y or V, or choose the gate threshold, after seeing the test outcomes.** That is leakage / researcher degrees-of-freedom.
4. **Never calibrate the runtime statistic against CPVI.** Calibrate against realised outcomes only (circularity guard).
5. **Never report message value without the state-only baseline** (no PVI without its CPVI and the gap).
6. **Never claim exact reproducibility of LLM runs.** Seed-pinned, revision-pinned, low-variance only.
7. **Never silently swallow an experiment error.** Fail loud; a passing-looking broken run is the worst outcome.
8. **Never use `yaml.load`** - `yaml.safe_load` only. **Never `eval`/`exec`/`compile` on data.**
9. **Never `pickle.load` data you did not produce in this repo.** For caching your own arrays/probes use `numpy.save` or `joblib`, kept gitignored. Never commit a pickle.
10. **Never commit raw logs, trained probes, embedding caches or model weights.** Only frozen artefacts (configs, seeds, manifests, metrics, final figures, the demo trace) are committed.
11. **Never spread Git LFS beyond the strict allowlist.** LFS is restricted in `.gitattributes` to exactly the final figure files and the demo trace - nothing else, ever. LFS has storage and bandwidth quotas with overage charges; keep it minimal.
12. **Never commit credentials, tokens or API keys** - even in fixtures. Use `os.environ` or `pytest.fixture` mocks. (The vLLM endpoint uses `api_key="EMPTY"`; that is fine.)
13. **Never push to any remote and never open pull requests.** Commit locally on the feature branch and stop. All pushes and PR creation are reserved for the human maintainer; offering to push is out of scope.
14. **Never bypass pre-commit or CI** with `--no-verify`, `[skip ci]`, or equivalent. Fix the underlying issue.
15. **Never mark a ticket Done without verifying every Definition-of-Done item.** "Almost done" is not done.

---

## Things to always do

1. **Always read the ticket end-to-end** before starting. Acceptance criteria + testing requirements are the checklist.
2. **Always check the ticket's listed prerequisites are merged** (cross-check DEPENDENCIES.md §2 only if the dep list looks ambiguous).
3. **Always write tests first or alongside code**, never after. Include the determinism test for any change to the simulator, channel, estimator or runner.
4. **Always run `ruff check . && ruff format --check . && mypy --strict src/ && pytest` locally** before handing the branch back.
5. **Always write the complete run manifest** (every mandatory field) on every run.
6. **Always set the seed and record model + encoder revisions** in the manifest for any run that touches an LLM or an embedding.
7. **Always report effect sizes, uncertainty intervals and seed sensitivity** - never bare significance.
8. **Always update CHANGELOG.md** for behaviour- or result-affecting changes, under `[Unreleased]`.
9. **Always use `from __future__ import annotations`** at the top of every Python module.
10. **Always prefer composition over inheritance** - the ABCs are interface contracts, not code reuse.

---

## Result-freezing discipline (the research analogue of a release)

- A result is **frozen** when its sweep is complete, the manifest is written, the analysis has run, effect sizes and intervals are reported, and the figure/table is committed.
- A frozen result is not silently re-run. If it must change (a bug, a corrected config), treat it like a CHANGELOG migration note and re-freeze explicitly; do not overwrite quietly.
- Frozen-result gates map to the roadmap phases (e.g. "RQ1 result frozen for write-up"). Timeline does not override correctness: a result delayed for a failing determinism or grounding check beats a fast wrong one.

---

## Scope boundaries

Do not pull these into a current branch; open an issue instead.

- **precept incorporation is out of scope for the research track.** The sequence is finalise the research, then incorporate precept. Do not add precept imports, integration code, or precept-shaped abstractions here.
- **Optional arms are first to cut:** the C5 supervisor (principal-agent) arm and the SocialJax MARL comparison. Cut SocialJax first; it is a contrast, not load-bearing.
- **RQ priority is RQ1 > RQ2 > (RQ3a in parallel with RQ3b).** Breadth beyond a frozen RQ1+RQ2 and at least one of RQ3a/RQ3b is upside, removable without touching the core.

---

## When in doubt

1. Re-fetch the ticket: `gh issue view <number>`. Acceptance Criteria + Testing Requirements answer most "should I do X?" questions.
2. Grep the relevant section of DEPENDENCIES.md (§4 risks, §5 cross-cutting). Do NOT read it wholesale.
3. If still unclear, pin the question at the top of the branch handback notes. Do not guess on architectural or methodological decisions.
4. For anything on the critical path (DEPENDENCIES.md §1) or anything touching Y/V freezing, calibration, or determinism, err heavily towards explicit clarification.

---

## Acknowledgements

The behavioural-discipline principles in the "Code style and architecture -> Implementation discipline" section above - *surface assumptions before coding, minimum code that satisfies the criteria, surgical edits, orphan cleanup* - are adapted from [`multica-ai/andrej-karpathy-skills`](https://github.com/multica-ai/andrej-karpathy-skills), itself derived from [Andrej Karpathy's observations on LLM coding pitfalls](https://x.com/karpathy/status/2015883857489522876). The upstream repository carried no licence at the time of adoption; the borrowings here are in good faith on the basis of the upstream's publicly stated purpose of distribution. The remainder of this CLAUDE.md is project-specific and original to precept-research.

---

*End of CLAUDE.md. Keep this file current as the repo evolves; stale operating guides are worse than no guide at all.*
