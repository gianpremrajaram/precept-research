# RQ3a baselines and the contribution axis (DSE-047)

> The published attribution numbers this dissertation is measured against, each with its method,
> substrate and **date**; and the statement of what the boundary measure claims relative to them.
> Written because the lit review anchored H5 to Who&When's 53.5% / 14.2%, and both figures have
> been beaten more than once since. Writing to a superseded floor is a viva liability; so is
> claiming to beat a tracer at a task a measure is not for. This document fixes both.
>
> Verified 29 August 2026 against each paper's own abstract or results table. Rows marked
> **secondary** were not read from the table and must be checked before submission.

---

## 1. The baselines table

Accuracies are the best reported by each paper for that substrate, except where a row names a
narrower cell for commensurability — the Who&When Pro row is its **text** modality, the only one of
its three that Who&When has; the same model and method reach 70.4% and 74.9% agent accuracy on
image and video traces (Table 4), which are not comparable to a text-only benchmark at all.
**The rows are not mutually comparable**: the substrates differ in observability, in how the
decisive step is defined, and in whether the failure was naturally occurring or injected. Section 2
says what may be compared.

| Method / benchmark | Agent acc. | Step acc. | Substrate | Date | Source |
|---|---|---|---|---|---|
| Who&When — best of all-at-once / binary search / step-by-step | **53.5%** | **14.2%** | 127 MAS failure logs, outputs only, 184 instances, all failures | May 2025 (ICML 2025 spotlight) | [arXiv:2505.00212](https://arxiv.org/abs/2505.00212) |
| Frontier reasoning LLMs, generic | — | **< 10%** | Who&When | Sep 2025 | [arXiv:2509.03312](https://arxiv.org/abs/2509.03312) |
| AgenTracer-8B | \+18.18% over Gemini-2.5-Pro / Claude-4-Sonnet | — | Who&When; trained on TracerTraj (counterfactual replay + programmed fault injection) | Sep 2025 | [arXiv:2509.03312](https://arxiv.org/abs/2509.03312) |
| TraceElephant — all-at-once | 62.2% | — | 380 executions, 220 annotated failures, **full execution traces** | Apr 2026 (ACL 2026) | [arXiv:2604.22708](https://arxiv.org/abs/2604.22708) |
| TraceElephant — binary search | 38.9% | — | as above | Apr 2026 | as above |
| TraceElephant — step-by-step | 60.9% | — | as above | Apr 2026 | as above |
| TraceElephant — static agentic | 65.9% | — | as above, with ground truth | Apr 2026 | as above |
| TraceElephant — dynamic agentic | **66.7%** *(secondary)* | **33.3%** | as above, with ground truth | Apr 2026 | as above |
| TraceElephant — dynamic agentic, no ground truth | 60.6% | 27.6% | as above, ground truth withheld | Apr 2026 | as above |
| AgentForesight-7B | \+19.9% over GPT-4.1 / DeepSeek-V4-Pro | 3× lower step-localisation error | AFTraj-2K: 2,276 trajectories (1,162 safe, 1,114 unsafe), **online, prefix-only** | May 2026 | [arXiv:2605.08715](https://arxiv.org/abs/2605.08715) |
| Who&When Pro — all-at-once, Qwen3.5-122B, **text traces** | 57.5% | **73.9%** | 12,326 **injected-fault** trajectories, 3 modalities, 26 benchmarks, all failures | Jul 2026 | [arXiv:2607.09996](https://arxiv.org/abs/2607.09996) |
| Long-Horizon Agent Trajectory Attribution | Hit@1 0.537 / MRR 0.713 | — | 1,300+ trajectories; **includes non-failures** (30.3% task-aligned, 39.4% unsafe, 30.3% refusals) | Aug 2026 | [arXiv:2608.06909](https://arxiv.org/abs/2608.06909) |

**Two readings of this table are wrong and must be pre-empted.**

*Who&When Pro's 73.9% step accuracy does not mean step attribution was solved.* Its pipeline
"injects a failure only after exactly replaying a successful prefix", so the decisive step is
placed by construction and is recoverable by construction. Who&When's 14.2% is over naturally
occurring decisive errors. The same all-at-once protocol produces both figures — but **not the same
model, and not model-matched**: Who&When Pro reports no original-Who&When row (its Table 1 compares
dataset statistics, not performance), so the 14.2% is 2025 models and the 73.9% is Qwen3.5-122B in
2026. The five-fold jump is therefore *suggestive* of a substrate effect rather than a clean
demonstration of one. What makes model capability an implausible sole explanation is the middle row
of the table: generic frontier reasoning models were still below 10% on Who&When step attribution
in September 2025, so the intervening capability gain would have to be extraordinary to carry it.

*Step accuracy exceeding agent accuracy on Who&When Pro is a property of injected faults*, and is
the inverse of the ordering on every naturally annotated corpus in the table. Quoting the 73.9% as
the live floor for H5 would be the error this document exists to prevent.

**Corpus boundary.** The survey above is closed at 29 August 2026 and covers the methods that
define the floor for H5. Further 2026 work on adjacent framings — POIROT
([arXiv:2606.02282](https://arxiv.org/abs/2606.02282)), span-level error localisation
([arXiv:2606.02060](https://arxiv.org/abs/2606.02060)), spectrum-analysis attribution
([arXiv:2509.13782](https://arxiv.org/abs/2509.13782)) and causal-inference attribution
([arXiv:2509.08682](https://arxiv.org/abs/2509.08682)) — is noted as existing and is not tabulated,
because none of it is a baseline this study reports against.

---

## 2. The contribution statement, restated on the axis that survives

**The claim is not that a probe beats a purpose-trained tracer at localisation.** It very likely
will not, a logistic probe over frozen embeddings is not a competitor to a reinforcement-trained
8B tracer, and a measure that had to win that comparison to be worth having would be a badly
motivated measure. Anchoring the contribution to raw accuracy also makes it perishable: it was
anchored to 53.5% / 14.2% and those were beaten three times inside fifteen months.

The claim is made on two axes instead, both of which are properties of the *construct* rather than
of a leaderboard position:

**Prospective versus retrospective.** Every method in §1 reads a completed or partial trajectory
and asks which step already went wrong. CPVI is computed from the message and the receiver's
observation **at the moment of the handoff**, from quantities that exist before the outcome does.
That is what makes it a candidate gate input (RQ3b) rather than a post-mortem, and no accuracy
figure in the table above is measured on that setting.

**Cost.** The measure is one forward pass of a frozen encoder plus a logistic probe, per handoff:
**zero model calls**. The published procedures in §1 cost one LLM call per trace (all-at-once),
⌈log₂ n⌉ (binary search) or up to *n* (step-by-step) — on this study's own corpora that projects to
3,428 calls for TraceElephant and 4,380 for Who&When. The right comparison is localisation obtained
per unit of compute, and whether any of it is available before the outcome exists.

**What is therefore reported.** Both regimes separately and never pooled — *transfer* (the
simulator-fitted statistic applied unchanged) and *refit* (probes fitted on the logs, grouped by
trace) — against schema validity and mean embedding cosine, with the published methods tabulated at
their reported accuracies and dates as in §1. The three Who&When procedures are additionally
re-implemented against the served open-weight tier so that a same-substrate, same-corpus comparison
exists; `JudgeIdentity` marks those rows as a re-implementation and **not** the published figures.

---

## 3. The online-auditing adjacency (AgentForesight)

AgentForesight ([arXiv:2605.08715](https://arxiv.org/abs/2605.08715), Zhang, B. et al., 9 May 2026)
reframes attribution as *online auditing*: at each step the auditor sees only the current prefix and
must continue or alarm at the earliest decisive error. It is the nearest work to the prospective
half of this dissertation's claim, and **methodology §7 carries the full treatment** — measure with
units versus detector with a verdict, several orders of magnitude of cost, and intervention against
a matched-firing-rate control versus an alarm. That paragraph is canonical; this section adds only
what the table above makes visible and one consequence §7 leaves implicit.

**What the adjacency costs, stated plainly.** Online auditing now exists in the literature, so
prospectivity *alone* is no longer novel. The prospective axis in §2 is therefore stated as a
property of the construct, not as a claim to be first.

**The consequence §7 leaves implicit: a prefix-reading auditor cannot be an H2 mediator.**
AgentForesight scores over everything in the trajectory prefix. CPVI is computed at one inter-agent
boundary from exactly two objects — the receiver's observation and the message. That is a strictly
weaker input, deliberately, and it is what makes the resulting number attributable to *the channel*:
a statistic that reads the manipulated and unmanipulated parts of a trajectory together cannot be
entered as the mediator in a mediation analysis of a channel manipulation (H2), nor be the input to
a gate whose effect is being isolated against a matched control (H6). The narrower input is the
reason the quantity is usable in the other two experiments, not a limitation of it.

## 4. Provenance

Numbers read on 29 August 2026 from each paper's abstract or results table via its arXiv page.
The **secondary** mark in §1 flags the one figure whose table row two reads disagreed on
(TraceElephant dynamic-agentic agent accuracy, 65.9% vs 66.7%); both readings place it above the
static-agentic row, and the exact value must be taken from Table 2 of the paper before submission.
Every other tabulated figure was read consistently.

Re-verified 29 August 2026 against the HTML full text after a cross-model review challenged one row.
Who&When Pro's 57.5% / 73.9% is confirmed as the Qwen3.5-122B all-at-once **text-modality** cell of
Table 4, and that paper reports **no** original-Who&When performance row — which is why §1's
substrate reading is now stated as suggestive rather than as a same-model demonstration. Who&When's
53.5% / 14.2% and AgenTracer's "+18.18% over Gemini-2.5-Pro and Claude-4-Sonnet" are verbatim from
their abstracts.
