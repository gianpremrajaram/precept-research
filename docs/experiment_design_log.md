# Experiment Design Log

A dated, reverse-chronological record of **experiment- and research-design changes** — decisions
that alter what the study measures or how the task/measurement is constructed, not routine code
edits. It is deliberately separate from `CHANGELOG.md`: the changelog tracks *code and behaviour* at
the component level; this log tracks *design and research-question* decisions (task geometry, outcome
and probe definitions, difficulty semantics, conditioning choices, gate design) with the evidence and
the risk each one closed. The two do not replace each other — a single change often appears in both,
the changelog saying *what the code now does* and this log saying *why the design changed and what it
de-risked*. Update this log whenever a change would affect the interpretation of a result or a
reviewer's read of the method.

Each entry: **date · area · trigger · finding · impact · risk reduced · correction path · the fix ·
result of the fix · so-what/takeaways.** Keep entries roughly one page.

---

## 2026-07-25 — Difficulty ladder was partly infeasible; retuned pre-freeze

- **Area:** task geometry — difficulty ladder (slit widths) and the rotation action.
- **Status:** corrected pre-freeze (no results frozen, no datasets recorded; nothing to re-freeze).
- **Trigger:** building the P1-4 feasibility tool (an A\* search over macro-actions on the
  deterministic simulator) while preparing the pre-Myriad pilot tooling. Running it across the ladder
  immediately returned *easy solvable, medium/hard not*.

### Finding

The task asks a rigid **T-load** (a bar of length 1.4 perpendicular to a stem of length 1.0, both
0.3 thick) to pass a gap of height *w* in a thin internal wall, moving along +x. The design intent
(DSE-006) was *"the T must rotate to pass a narrow slit."* That intent is geometrically false, and
the search proved it:

- To cross a thin wall, at the instant each member of the T crosses the wall plane its vertical
  (y) slice must fit inside the gap. The bar and stem are **perpendicular**, so whichever member is
  aligned with travel crosses at its 0.3 thickness while the *other* member is then perpendicular to
  travel and presents its **full length** across the gap.
- Rotation only swaps which member jams. The tightest slit the T can thread is therefore the
  **shorter member's length — the stem, 1.0** — and it is **invariant to rotation**. A slit ≥ the
  full y-extent (bar thickness + stem = 1.3) clears head-on with no maneuver at all.
- The shipped ladder was **1.8 / 1.0 / 0.7**. So **easy (1.8)** was trivial, **medium (1.0)** sat at
  exactly zero clearance (stem length == slit), and **hard (0.7) was impossible** — below the 1.0
  threshold at every orientation. The A\* search confirmed it: easy solved in 7 pushes; medium only
  via a fragile ~17-step thread found at fine resolution; hard exhausted its reachable state space
  unsolved.
- Secondary finding: the rotation action was **uncontrollable** — one `ROT+` turned the T ~135°
  (angular impulse 2.0 against the T's small moment ~0.29), leaving only 45°-multiple orientations
  reachable, far too coarse to aim the T for a threading maneuver.

### Impact (had it not been caught)

`rq1_sweep` defaults its headline difficulty to **hard**, and the pilot runs easy + hard. On the
shipped geometry the pilot's **G1 (self-play solves the task)** and **G2 (a measurable C0→hard gap)**
would have failed on hard — *not because the models cannot coordinate, but because the task is
unsolvable.* That is precisely the "nobody can do it" misdiagnosis P1-4 was written to prevent: a
confusing null read as "no usable information transfer," triggering the fallback ladder for the wrong
reason and burning the (imminent) Myriad allocation on an uninterpretable result.

### Risk reduced

This is the concrete pay-off of the front-loaded pilot and of **early testing enabling a pivot during
architecture design**. Because the feasibility check was built and run *before* any recorded run
(CPU-only, no cluster, no LLM), a task-design flaw in the headline difficulty was caught and fixed
while it was still free to fix — no frozen result, no committed dataset, no re-run. Post-freeze or
post-Myriad, the same finding would have cost a re-specification of the difficulty axis and the
regeneration of every dependent result. (Maps to roadmap risks R1/R2: capability floor and
measurable-signal gates.)

### Correction path

Feasibility search flags infeasibility → first-principles slice geometry confirms it is real, not a
search artifact → candidate ladders and rotation magnitudes tested empirically against the search
until all three difficulties solve with a graded, sensible path length → budgets frozen from the
verified optima → wired per-difficulty.

### The fix

- **Slit ladder 1.8 / 1.0 / 0.7 → 1.8 / 1.2 / 1.1.** Easy unchanged (head-on trivial); medium 1.2
  and hard 1.1 both sit in the *threading* regime above the 1.0 stem threshold, graded by clearance
  (difficulty now increases via how precise the threading maneuver must be, a cleaner information
  gradient than the broken "rotate to pass").
- **Rotation `angular_impulse` 2.0 → 0.5** (~34° per action), so an agent can aim the T.
- **Per-difficulty step budgets** certified at ~2.5× the oracle optimum and wired through
  `SweepConfig` / `EpisodeRunner` / `rq1_sweep` (previously a single budget of 12 for all
  difficulties, which under-fed hard).

### Result of the fix

All three difficulties are now solvable with a clean gradient — **easy 7 steps, medium 13, hard 13**
(oracle optima); frozen budgets **18 / 33 / 33**. Rotation is controllable at ~34°/action. A
regression test re-derives and certifies the tight difficulties stay feasible within budget, so a
future physics change that breaks feasibility fails loudly rather than silently. The full suite,
type-check and lint stay green.

### So what / takeaways

Caught at the best possible moment: pre-freeze, pre-Myriad, at ~zero compute cost, exactly as the
front-loaded pilot was designed to enable. It converts the (now imminent, ~21 July) Myriad window
into headline sweeps rather than debugging, and it validates building the cheap CPU-side
feasibility/diagnostic tooling *before* the cluster arrives. The difficulty axis is now a defensible,
graded threading-clearance construct — worth stating plainly in the methodology, since a reviewer
would otherwise ask why "rotation" is the difficulty knob when rotation does not change the threading
cross-section.
