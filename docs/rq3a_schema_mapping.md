# RQ3a corpus schema mapping

Field-by-field mapping from the three external log corpora onto `LogHandoffRecord` /
`LogTraceRecord` (`src/preceptx/data/logs.py`), plus the counts they actually contain. Loaders live
in `src/preceptx/experiments/rq3a_load.py`; `scripts/fetch_rq3a.sh` fetches the files.

Everything below was **measured against the real files** during the E9 spike (24 August 2026), not
read off a paper. Where a measurement contradicts what the ticket assumed, the measurement wins and
the contradiction is stated.

---

## 1. Why a separate record type

`HandoffRecord` is the simulator's reproducibility contract and carries physics. A log record has no
physics at all. Widening `HandoffRecord` with nullable physics fields would let a log row be read as
a degraded episode row — so physics fields here are **absent, not nullable**, and
`LOG_SCHEMA_VERSION` runs on its own counter so a change to one substrate never re-keys the other.

`trace_id` is the cross-fitting grouping key, the exact analogue of `episode_id`. The leakage
discipline is identical; only the name changes.

The featuriser needs no change: it embeds the observation and message slots, and its content-addressed
cache is keyed on encoder name, revision and text. Simulator and log embeddings share one cache
safely — the vectors are genuinely the same function of text, so **no domain field is added to the
cache key**.

---

## 2. TraceElephant — the primary substrate

`TraceElephant/TraceElephant` on HuggingFace, CC-BY-4.0. Published as a **single 569 MB `data.zip`**,
not a loadable dataset: `datasets.load_dataset()` cannot open it, which is why fetching is a shell
script rather than a library call. The repository is auto-tagged `format:imagefolder` /
`modality:image` on HuggingFace; that tag is wrong for our purposes — it is triggered by screenshot
PNGs inside some web-agent traces, and the substrate itself is per-step JSON.

Layout: `data/<run family>/<task id>/{step_records.json, trace_metadata.json}`.

### `step_records.json` → `LogHandoffRecord`

| Source field | Target | Notes |
|---|---|---|
| *(directory)* `<family>/<task>` | `trace_id` | Family is included so two families sharing a task id stay distinct groups. |
| `step_id` | `step` | The corpus's own ordinal, not a re-index. |
| `agent_id` | `agent_id` | Stringified. |
| `agent_name` | `agent_name` | The acting component (e.g. `str_replace_editor`, `bash`). |
| `input.messages` | `observation` | Rendered role-prefixed. **The whole prefix, not the last turn** — see §5. |
| `output.choices[0].message.content` + `.tool_calls` | `message` | Tool calls are part of the message, not metadata (§5). |
| `trace_metadata.tests_status` | `trace_failed` | Annotation-free outcome (§5). |
| `trace_metadata.{mistake_agent,mistake_step,mistake_reason}` | `annotations` | Evaluation target only, never a feature. |
| — | `reconstructed_observation` | Always `False`: the input context is recorded, not rebuilt. |

**The ticket's assumed field names were `input_context` and `output_content`. The corpus calls them
`input` and `output`, and both are structured objects rather than strings** — `input` is an
OpenAI-shape `{messages, model, stream}` and `output` is a full `ChatCompletion`. The substance the
design depends on is present: the receiver-observed context genuinely exists per step. Only the
mapping needed writing.

### TraceElephant is failure-only — the second reason for the substrate move does not hold

Roadmap §3.4 gives two reasons for moving RQ3a off Who&When: it records the receiver's input
context, and it "ships non-failing executions too, so trace-level outcome is genuinely two-class".
The first is confirmed. **The second is false.**

- The corpus is **220 traces, not 380**, and every one of the 220 carries a populated
  `mistake_agent` / `mistake_step` / `mistake_reason`. It is a failure-attribution corpus and every
  trace in it is an attributed failure. The "380 executions of which about 220 are annotated
  failures" reading treated 220 as the failure subset of a larger pool; 220 is the whole pool.
- Only the 44 `swe-agent-runs-swe-bench` traces carry an annotation-free outcome at all
  (`tests_status`). The other 176 have no outcome field — `captain-*` and `magentic-*` carry a
  `ground_truth` string but no recorded final answer to compare it against, and a scan of the last
  three steps' output finds the ground-truth string in only 1 of 20 sampled traces, so it cannot be
  recovered by matching either.
- Of those 44, **0 pass every test.** The annotation-free outcome is single-class too.

So on TraceElephant the refit arm is undefined for exactly the reason it is undefined on
Who&When. See §5 for what this does to *Y*.

---

## 3. Who&When — secondary, observability-caveated

`Kevin355/Who_and_When`, two parquet splits. Confirmed: **184 traces, all 184 failures.** The
non-failure class is empty, so the refit arm is *undefined* on this corpus rather than merely thin.

| Source field | Target | Notes |
|---|---|---|
| `<split>/question_ID` | `trace_id` | |
| *(list index)* | `step` | The corpus records no step ordinal. |
| `history[i].name` else `.role` | `agent_name` | `Algorithm-Generated` has `name`; `Hand-Crafted` has only `role`, which doubles as the component identity (`Orchestrator (thought)`, `WebSurfer`). |
| `history[0..i-1]` | `observation` | **Reconstructed** — no per-step input context exists. |
| `history[i].content` | `message` | |
| `is_correct` / `is_corrected` | `trace_failed` | The two splits spell the column differently. |
| `mistake_{agent,step,reason}` | `annotations` | |
| — | `reconstructed_observation` | Always `True`, on every row this loader emits. |

The flag exists so downstream analysis can never silently pool approximated conditioning state with
recorded conditioning state. Any result mixing the two must report the split.

---

## 4. MAST-Data — trace-level secondary only

`mcemri/MAST-Data`, `MAD_full_dataset.json`. **`trace.trajectory` is one unsegmented string per
trace**, formatted differently by each of the seven systems present (AG2, MetaGPT, ChatDev, Magentic,
OpenManus, AppWorld, HyperAgent), with a median of ~9.7k characters and a maximum over 2M. Per-step
extraction would need seven bespoke parsers and would invent step boundaries the corpus does not
record, so MAST loads as `LogTraceRecord` and never as a per-step record. That is the honest
resolution, not a lesser one.

| Source field | Target |
|---|---|
| `mas_name/benchmark_name/trace_id` | `trace_id` |
| `mas_name` | `system_name` |
| `llm_name` | `model_name` |
| `benchmark_name` | `benchmark` |
| `trace.trajectory` | `trace_text` |
| `mast_annotation` (any of 14 modes flagged) | `trace_failed` |
| `mast_annotation` | `annotations` |

### The non-failure proportion, counted

DSE-047 flagged this number as resting on all-zero annotation rows seen in a preview rather than on a
count. Counted: **405 of 1642 traces (24.7%) carry an all-zero annotation**, i.e. the non-failure
class. The assumption holds — but the class is **severely unbalanced across systems**:

| System | Non-failure | Total | Rate |
|---|---:|---:|---:|
| AG2 | 311 | 597 | 52.1% |
| Magentic | 42 | 195 | 21.5% |
| HyperAgent | 3 | 30 | 10.0% |
| ChatDev | 25 | 330 | 7.6% |
| MetaGPT | 22 | 430 | 5.1% |
| AppWorld | 1 | 30 | 3.3% |
| OpenManus | 1 | 30 | 3.3% |
| **Total** | **405** | **1642** | **24.7%** |

**This is a confound, not a curiosity.** A probe fitted on pooled MAST traces can reach most of the
available accuracy by recognising *which system produced the trace* — AG2 is a coin flip, OpenManus
is nearly always a failure — without reading the message at all. It is the same defect the
simulator-side shuffled-message audit found, where condition identity leaked into CPVI through
message style. Any MAST refit arm must therefore either stratify by `system_name` or report the
system-identity-only baseline alongside, exactly as the control task does for the simulator arm.

---

## 5. Three mapping decisions that change what a number means

**The observation is the whole context prefix, not the last turn.** A component's usable state at a
step is everything in its context window. Truncating to the most recent message would shrink the
state-only baseline and inflate CPVI — the conditioning would be weaker than what the agent actually
had, and the message would appear to add information it did not add.

**Tool calls are part of the message.** On a tool-using system the call *is* what the next component
receives. A step whose `content` is `""` with a populated `tool_calls` is the common case in
TraceElephant, not an error; scoring it as an empty message would repeat the local-pilot fail-open
bug where empty completions were counted as real handoffs.

**The TraceElephant outcome reads tests, never annotations.** `trace_failed` is derived from
`tests_status` — a SWE-bench harness result — and is `True` when any test in any bucket is in its
`failure` list. It never touches `mistake_agent` / `mistake_step` / `mistake_reason`, and it is
`None` on the 176 traces that have no `tests_status` rather than being filled in from the
annotation. That `None` is the honest value: **the corpus supplies no annotation-free outcome for
four of its five run families, and single-class failure for the fifth.**

The consequence for *Y* is the opposite of convenient. Y1 (trace success) is degenerate on the
primary corpus — a constant cannot be predicted — so **DSE-042's counterfactual replay is
load-bearing, not an upgrade path**: it is the only route to a within-trace two-class target where
the per-step conditioning state actually exists. Y2 (annotation-as-Y) remains forbidden for
circularity. MAST is the only corpus here with a genuine two-class trace outcome (405 / 1,237), and
it is trace-level only and confounded with system identity (§4).

**Intra-agent tool turns stay in the dataset.** `is_handoff` marks the inter-agent boundaries but
non-handoff steps are retained: dropping them would change the per-step base rate and make the
handoff subset's CPVI incomparable to the simulator's.

---

## 6. Counts

Produced by `uv run python -m preceptx.experiments.rq3a_load --root <corpus root>` on
24 August 2026, at the revisions in §7.

| Corpus | Traces | Steps | Inter-agent handoffs | Failures | Non-failures | Observations |
|---|---:|---:|---:|---:|---:|---|
| TraceElephant | 220 | 5,960 | 2,488 | 220 by annotation; 44 annotation-free | **0** | recorded |
| Who&When | 184 | 4,092 | 3,505 | 184 | **0** | reconstructed |
| MAST-Data | 1,642 | — | — | 1,237 | **405** | trace-level only |

`trace_failed` is populated for 44 TraceElephant traces (all failing) and `None` for the other 176:
it reads `tests_status` and never the annotation, so it is absent where the corpus records no
harness result. The 220 figure counts the native `mistake_agent` annotation, which every trace has.

**Two corpora out of three are single-class at trace level.** That is the substantive result of this
spike, and §5 records what it does to *Y*.

TraceElephant by run family — the handoff structure is genuinely multi-agent, not agent-to-tool
only, and the two `captain-*` families are the densest inter-agent boundaries in the corpus:

| Family | System | Traces | Steps | Handoffs | Distinct components | Annotation-free outcome |
|---|---|---:|---:|---:|---:|---|
| `captain-runs-assistantbench` | captain-agent | 12 | 187 | 140 | 6 | *none* |
| `captain-runs-gaia` | captain-agent | 73 | 1,559 | 1,348 | 6 | *none* |
| `magentic-runs-assistant-bench` | magentic-one | 17 | 603 | 105 | 3 | *none* |
| `magentic-runs-gaia` | magentic-one | 74 | 2,060 | 400 | 7 | *none* |
| `swe-agent-runs-swe-bench` | swe-agent | 44 | 1,551 | 495 | 7 | `tests_status` (0/44 pass) |

Handoff density varies by an order of magnitude across families — `captain-runs-gaia` hands off at
86% of steps (an orchestrator alternating with experts), `magentic-runs-assistant-bench` at 17% (a
single `WebSurfer` running long tool sequences). Any pooled per-handoff rate is therefore a mixture
over five very different interaction topologies, and family must be reported alongside it.

---

## 7. Provenance

| Corpus | HuggingFace id | Revision (SHA) | Licence | Size |
|---|---|---|---|---|
| TraceElephant | `TraceElephant/TraceElephant` | `a78a57cdcdf74a080b1bec0f56f85228d86acbac` | CC-BY-4.0 | 569 MB (zip) |
| Who&When | `Kevin355/Who_and_When` | `59b9fcba1aaed7bbf206b5f4d3c68b8face2f49c` | see dataset card | 1.9 MB |
| MAST-Data | `mcemri/MAST-Data` | `95118ac951421753cf1deb87ddea3b01e693c41b` | see dataset card | 200 MB |

No corpus file is committed to this repository. Unit tests run against hand-built fixtures that
mirror the layouts above (`tests/unit/experiments/test_rq3a_load.py`), so the loaders are testable
offline and CI never downloads 800 MB.
