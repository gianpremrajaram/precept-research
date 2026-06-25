# Precept Dissertation: End-to-End Research Roadmap

**Prepared for:** Gian Prem Rajaram (UCL MSc CS) | **Supervisor:** Prof. Jun Wang (+ Prof. Philip Treleaven) | **Window:** mid-June 2026 to dissertation submission (target late August, deliberately early of the hard deadline to absorb slippage) | **Paper:** workshop/preprint submission 21 September 2026 | **Status:** execution plan built on Research Design v3 and the literature-review chapter. Companion: `ISSUES.md` (the same work as Claude-Code-pointable tickets).

**How to use this.** This is the execution spine, not a restatement of the design. It fixes the architecture, the code, the experiment programme with alternatives, and the research milestone at each phase. Where a decision is still open (chiefly compute), it is isolated in §0 so the rest of the plan does not depend on it. Formatting follows the agreed conventions: BLUF, UK spelling, tables for multi-option comparisons, prose over bullet-spam.

**Abbreviations.** CPVI = conditional pointwise V-usable information; PVI = pointwise V-usable information; IB = information bottleneck; MI = mutual information; JSD = Jensen-Shannon divergence; Y = the physics outcome variable a probe predicts; V = the probe model family; LLM = large language model; vLLM = the high-throughput inference server; SGE = Sun Grid Engine (Myriad's scheduler); AWQ = activation-aware weight quantisation (4-bit); TP = tensor parallelism; AUROC = area under the ROC curve; ECE = expected calibration error; FA = failure attribution; COM = centre of mass; DoD = definition of done; G1/G2/G3 = the three pilot gates (capability, signal, groundedness).

---

## BLUF

The dissertation stands or falls on one measurement applied in one place: conditional usable information (CPVI) at the text boundary between two LLM agents jointly manoeuvring a T-shaped load through a Pymunk arena under a degraded channel. The engineering is four components that compose cleanly: a deterministic 2D physics simulator, a two-agent LangGraph negotiation loop, locally served open-weight models behind a vLLM OpenAI endpoint, and a measurement-and-gate stack that scores each handoff offline (CPVI) and online (a target-free statistic) and can block low-information handoffs. The research is four questions answered in priority order: RQ1 the information gradient (does success and efficiency fall as the channel is degraded, and does CPVI track it), RQ2 the measurement primitive (do a retrospective and a prospective CPVI twin agree, and does a cheap runtime proxy track the offline ground truth), RQ3a external validity (does boundary CPVI localise the responsible step in real MAS failure logs), and RQ3b the causal arm (does gating low-CPVI handoffs improve outcomes under matched-firing-rate and random-trigger controls). The plan front-loads a one-to-two week pilot with hard go/no-go gates so the central feasibility risk - that the models cannot ground 2D geometry well enough - is discovered in week two, not week eight, with MAST/Who&When external validity as the pre-planned fallback that can carry the dissertation alone. Two arms (the principal-agent supervisor C5 and the SocialJax MARL comparison) are optional and have explicit cut-lines. The single biggest lever on quality is finishing RQ1 and RQ2 cleanly and writing them up as both the dissertation's core and the September paper; everything else is breadth that can be trimmed without unpicking the core.

---

## 0. Constraints and the one open decision (compute and models)

This section isolates the only material unknown so the rest of the roadmap is decision-independent. Everything here can be frontloaded: secure compute, stand up serving, and lock the model ladder in Phase 0 before the science starts.

Signals: Myriad is single-node, high-throughput, SGE-scheduled, up to 4 GPUs and 36 cores per job, drawing on Free or three-monthly priority allocation. The GPU envelope spans V100-16GB (E/F nodes), P100 (J), A100-40GB (L), A100-80GB (U/V) and L40S-48GB. The open decision is which allocation you actually get, which sets the model-size ceiling and the seed budget. The plan therefore specifies a model ladder that degrades gracefully across the whole envelope, so any allocation yields a viable headline plus a robustness tier.

**Open decision (resolve in Phase 0, track as ticket DSE-014).** Confirm (a) Free vs priority Myriad allocation, which sets realistic queue latency and how many seeds and conditions are affordable, and (b) the largest GPU you can reliably get (40GB vs 80GB A100, or L40S), which sets whether the strong tier runs unquantised or 4-bit. Until resolved, default to the 14B workhorse path, which fits every Myriad GPU.

The model ladder below is the recommendation. Licences favour reproducibility: Qwen3 is Apache 2.0 and has the most stable tool calling and JSON adherence in mid-2026; Llama is research-permissive; Gemma and Mistral Small are clean enough for academic use. Table abbreviations: bf16 = 16-bit weights; ctx = context length used; node = the Myriad node type that fits it.

| Tier | Model (default) | Memory (serving) | Fits | Role |
|---|---|---|---|---|
| Pilot / fast sweeps | Llama-3.1-8B-Instruct or Qwen3-8B | ~16-18GB bf16 | Any GPU incl. V100-16GB (8B at 4-bit) | High-throughput pilot, many-seed RQ1 sweeps |
| Workhorse (primary) | Qwen3-14B-Instruct | ~28-30GB bf16 | A100-40GB, L40S-48GB, A100-80GB | Headline RQ1/RQ2 runs; both agents (self-play) |
| Strong tier | Qwen3-32B-Instruct | ~64GB bf16 / ~20GB AWQ | A100-80GB (bf16) or any 40GB+ (AWQ) | Robustness; "does the gradient hold with a stronger model" |
| Scale / heterogeneous partner | Llama-3.3-70B-Instruct-AWQ (INT4) | ~40GB | One A100-80GB or 2x A100-40GB (TP=2) | Heterogeneous-pair cell; scale confirmation |
| Cross-family tie-breaker | Gemma-4 (or Mistral Small 4) | fits 40-48GB | A100-40GB, L40S | One cross-family robustness cell only |

Serving and determinism. Serve one model per GPU job behind vLLM's OpenAI-compatible server; the LangGraph client points at the local endpoint. Use greedy decoding (`temperature=0`) with a fixed seed and pinned model revisions; note in the thesis that batched LLM inference is not bit-exact across runs, so determinism is "low-variance, seed-pinned, revision-pinned", and report seed sensitivity rather than claiming exact reproducibility. Structured action output uses vLLM guided decoding (xgrammar/outlines) against a JSON schema, which removes parser brittleness from the action channel.

```bash
# Myriad SGE jobscript sketch (serve.sh) - request 1 GPU, 80GB A100 if available
#$ -l gpu=1
#$ -l h_rt=8:00:00
#$ -pe smp 8
#$ -P <project>            # Free or priority project
module load cuda/12.x
source ~/venvs/precept/bin/activate
# bf16 14B workhorse; for 70B-AWQ add: --quantization awq --tensor-parallel-size 2
vllm serve Qwen/Qwen3-14B-Instruct \
  --port 8000 --dtype bfloat16 --max-model-len 8192 \
  --gpu-memory-utilization 0.90 --seed 0 \
  --guided-decoding-backend xgrammar
```

---

## 1. Research questions, hypotheses, and what the literature says is open

Signals: the four RQs map one-to-one onto the unoccupied cell identified in the review (conditional usable information measured and enforced at an LLM boundary against an objective outcome). The novelty is the conditional construct plus the detection-gating-audit posture, not any single component.

1. **RQ1 - information gradient.** As the communication channel is degraded across conditions C0-C4, do task success and efficiency fall, and does mean per-handoff CPVI fall with them? H1: success and efficiency degrade monotonically with bandwidth; H2: CPVI tracks the degradation and explains variance in outcomes beyond shared state. The human-side analogue is Dreyer et al. (2025), where restricted-communication groups did worse than individuals. The novelty over IMAC (Wang et al. 2020) is that the message is scored, not trained.

2. **RQ2 - measurement primitive.** Do a retrospective CPVI (scored after the outcome) and a prospective twin (scored from the handoff and shared state before the outcome) agree, and does a cheap runtime proxy track the offline CPVI ground truth? H3: the twins correlate strongly; H4: the proxy (a target-free statistic) tracks CPVI closely enough to gate on. The conditioning is the Hewitt et al. (2021) move; the divergence analysis (JSD over probe predictions, plus embedding cosine) is the cheap-proxy bridge.

3. **RQ3a - external validity.** On real MAS failure logs (Who&When primary; MAST-Data secondary), does boundary CPVI localise the responsible step or trace better than schema validity or mean embedding cosine? H5: low-CPVI handoffs coincide with the annotated decisive error step or inter-agent-misalignment trace label, beating the weak published attribution baselines. This is the bridge from a synthetic task to the field, and the headroom is real (best Who&When baseline is 53.5% agent / 14.2% step).

4. **RQ3b - causal gate.** Does blocking low-CPVI handoffs at runtime improve outcomes, and does the improvement survive a matched-firing-rate control (block the same number of random handoffs) and a random-trigger control? H6: gating improves success/efficiency over both controls. This is precisely the intervention Lowe et al. (2019) demand; correlation between CPVI and outcomes is not enough.

Optional. **C5 principal-agent supervisor arm** tests the buried assumption in Rauba et al. (2026) that aligned-incentive asymmetry is benign, by routing handoffs through a supervisor and measuring residual agency loss. **SocialJax MARL comparison** runs the information-gradient idea in a learned-message setting for contrast. Both are first to cut.

---

## 2. System architecture and the code to build it

This is the concrete build. Four components, each independently testable, wired into one episode runner that the experiments call. The handoff message between the two agents is the object the measurement stack scores; everything else exists to generate and label those handoffs.

### 2.1 Simulator (Pymunk): the transposed piano-movers task

Signals: a top-down (gravity-free) 2D rigid-body arena with three chambers joined by two narrow slits, a T-shaped dynamic load, a discrete action interface, and a progress-based outcome. Pymunk (Chipmunk2D) is the right tool: mature, deterministic per seed, fast headless, and it gives true contact dynamics so "the move was physically blocked" is a real event rather than a scripted one.

Design decisions that matter:
1. **Top-down, damped.** Set `space.gravity = (0, 0)` and a `space.damping` below 1.0 so the load bleeds momentum and the puzzle is quasi-static, matching the ant/human task where the load does not coast.
2. **Arena.** Three chambers built from static `Segment` walls leaving two narrow slit gaps; the load starts in chamber one, the goal is chamber three. Slit width is the difficulty knob.
3. **Load.** One dynamic `Body` carrying two box `Poly` shapes (stem + crossbar) to form a T; mass and moment summed so rotation is realistic. The T shape forces non-trivial rotation through the slits, which is the cognitive core of the task.
4. **Action interface (primary).** Each turn the two agents must agree a single macro-action from a small discrete set (translate by a step in one of four directions, rotate by a fixed +/- angle, or wait); the agreed action is realised as an impulse and the space is stepped to settle. This keeps the negotiation causally coupled to the outcome and makes handoffs clean.
5. **Action interface (alternative, higher fidelity).** Each agent holds a grip point on the load and chooses a discrete force vector; the net wrench moves the load. Closer to the ant model, but introduces a second confound (force composition) and is the fallback if the macro-action design proves too coarse.
6. **State and serialisation.** The state is COM (x, y), angle theta, velocities, slit and goal geometry. How this is written into the prompt is an experimental factor (see RQ1): numeric tuples, an ASCII occupancy grid, or a natural-language description. Serialisation is its own module so the A/B test is a config flag.
7. **Outcome.** Success = COM reaches the goal chamber within a step budget. Progress = reduction in geodesic distance-to-goal through the slits; "stuck" and "collision" are detected from contact and velocity. The outcome variable Y for CPVI is derived from this (see §2.4).

```python
import pymunk

def build_arena(slit_width=60.0, wall=5.0):
    space = pymunk.Space()
    space.gravity = (0, 0)
    space.damping = 0.25            # quasi-static: load stops when not pushed
    static = space.static_body
    walls = []
    # ... place Segment walls forming 3 chambers with 2 slit gaps of `slit_width`
    # each: s = pymunk.Segment(static, p_a, p_b, wall); s.friction = 0.6; walls.append(s)
    space.add(*walls)
    return space

def add_T_load(space, pos, mass=1.0):
    stem = (40, 120); bar = (120, 40)          # box dims (w,h)
    moment = (pymunk.moment_for_box(mass/2, stem) +
              pymunk.moment_for_box(mass/2, bar))
    body = pymunk.Body(mass, moment)
    body.position = pos
    s1 = pymunk.Poly.create_box(body, stem)
    s2 = pymunk.Poly.create_box(body, bar)     # offset to form a T via vertices
    for s in (s1, s2): s.friction = 0.6
    space.add(body, s1, s2)
    return body

def apply_macro_action(space, body, action, impulse=400.0, dtheta=0.20, substeps=20, dt=1/60):
    if action in ("N","S","E","W"):
        v = {"N":(0,1),"S":(0,-1),"E":(1,0),"W":(-1,0)}[action]
        body.apply_impulse_at_world_point((v[0]*impulse, v[1]*impulse), body.position)
    elif action == "ROT+": body.angular_velocity += dtheta*10
    elif action == "ROT-": body.angular_velocity -= dtheta*10
    for _ in range(substeps): space.step(dt)   # settle under damping
```

### 2.2 Agent loop (LangGraph): two agents, a degradable channel, structured actions

Signals: a `StateGraph` with nodes for propose, respond, aggregate-into-action, apply, and check-done, looping until success or budget. The single inter-agent message is captured as the handoff payload the measurement stack scores. Communication conditions degrade that channel only.

```python
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

class Action(BaseModel):
    move: str          # one of N,S,E,W,ROT+,ROT-,WAIT
    rationale: str

class GS(TypedDict):
    physics: dict             # COM, theta, velocities, geometry
    msg_A: Optional[str]      # A->B handoff (the scored boundary)
    proposal_B: Optional[Action]
    agreed: Optional[Action]
    step: int
    done: bool

llm = ChatOpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY",
                 model="Qwen/Qwen3-14B-Instruct", temperature=0, seed=0)

def agent_A(state: GS) -> dict:
    obs = serialise(state["physics"], mode=CFG.serialisation)   # numeric|grid|nl
    raw = llm.invoke(PROMPT_A.format(obs=obs)).content          # natural-language handoff
    return {"msg_A": apply_channel(raw, condition=CFG.condition)}  # degrade here only

def agent_B(state: GS) -> dict:
    obs = serialise(state["physics"], mode=CFG.serialisation)
    act = llm.with_structured_output(Action).invoke(
        PROMPT_B.format(obs=obs, message=state["msg_A"]))
    return {"proposal_B": act, "agreed": act}    # primary: B realises the agreed macro-action

def apply_node(state: GS) -> dict:
    apply_macro_action(SPACE, LOAD, state["agreed"].move)
    phys = read_state(SPACE, LOAD)
    return {"physics": phys, "step": state["step"]+1,
            "done": reached_goal(phys) or state["step"]+1 >= CFG.budget}

g = StateGraph(GS)
for n,f in [("A",agent_A),("B",agent_B),("apply",apply_node)]: g.add_node(n,f)
g.set_entry_point("A"); g.add_edge("A","B"); g.add_edge("B","apply")
g.add_conditional_edges("apply", lambda s: END if s["done"] else "A")
runner = g.compile()
```

The handoff capture and logging are in-repo: each `msg_A` plus the pre-action `physics` is a record emitted through a vanilla OpenTelemetry exporter, and the runtime gate (when active in RQ3b) is the in-repo `RuntimeGate` evaluating that boundary before `apply`. This is the "active control layer" contribution; the gate is exercised end to end rather than described.

### 2.3 Communication conditions C0-C5

Signals: a single degradation ladder applied only to the A-to-B channel, so any outcome difference is attributable to the channel, not to task or model changes. `apply_channel` implements each.

| Condition | Channel manipulation | What it isolates |
|---|---|---|
| C0 | Full natural-language message, no limit | Ceiling / upper bound |
| C1 | Token/length cap (e.g. <= 30 tokens) | Bandwidth |
| C2 | Delayed delivery (message arrives one step late) | Latency (motivated by the 2026 comm-gain-and-delay result) |
| C3 | Asymmetric visibility (B sees only a local window; A sees all) | Whether the message must carry non-state information |
| C4 | Lexical/semantic noise (token dropout or paraphrase corruption) | Robustness of usable information to corruption |
| C5 (optional) | Supervisor relay (principal-agent) intermediates the handoff | Residual agency loss under aligned incentives |

C3 is the structural guard against the floor effect: it forces the message to carry information the shared state does not, so CPVI cannot be near-zero by construction.

### 2.4 Measurement stack (CPVI): outcome variable, probe family, estimator

Signals: CPVI subtracts a state-only baseline from a state-plus-message probe, both fitted offline on frozen embeddings; the choice of outcome variable Y and probe family V are the two methodological decisions, both pinned here with alternatives.

Outcome variable Y - all options, with tradeoffs. Default is the binary-plus-continuous twin, which yields both a classification AUROC and a regression-style PVI and hedges against a degenerate label.

| Y option | Definition | Pro | Con |
|---|---|---|---|
| Binary progress (default A) | Did the next k steps net positive geodesic progress? | Stable, interpretable, AUROC-ready | Coarse; can saturate on easy states |
| Continuous displacement (default B, twin) | Signed change in distance-to-goal / angle | Fine-grained; Gaussian-NLL PVI | Noisier; needs variance modelling |
| Discrete next-configuration | Bucketed next pose region | Captures rotation structure | Bucket design is arbitrary |
| Terminal success-in-horizon | Did the episode ultimately succeed from here? | Closest to the real objective | Long credit path; weak per-handoff signal |

Probe family V - options. Default is L2-regularised logistic regression (classification) and a heteroscedastic linear/MLP regressor (continuous) on concatenated sentence-embeddings, because the dimensionality argument that killed MI demands a tractable, model-relative estimator, and a small fitted head is exactly V-information's intended use. A 2-layer MLP is the fallback if logistic underfits; fine-tuning a small LM is the high-capacity option held in reserve.

CPVI estimator (conditional on shared state, per Hewitt et al. 2021):
- Featurise: `e_s = embed(serialise(state))`, `e_m = embed(message)` with a pinned sentence-transformer (start with a strong retrieval embedder; report sensitivity to the encoder).
- Fit two probes on the training split: `g_cond` on `[e_s ; e_m]` predicting Y, and `g_base` on `[e_s]` predicting Y.
- Per held-out instance: `CPVI = log2 g_cond([e_s;e_m])[y] - log2 g_base([e_s])[y]`. Mean over instances is the conditional V-information; the distribution is the per-handoff score.
- Report `PVI - CPVI` (the gap is how much apparent message value was an echo of the state) and the classification AUROC of `g_cond` vs `g_base`.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

def fit_probe(X, y):                      # V = L2 logistic; swap for MLP if underfit
    return LogisticRegression(max_iter=1000, C=1.0).fit(X, y)

def cpvi(e_s_tr, e_m_tr, y_tr, e_s_te, e_m_te, y_te, eps=1e-9):
    g_cond = fit_probe(np.hstack([e_s_tr, e_m_tr]), y_tr)
    g_base = fit_probe(e_s_tr, y_tr)
    p_cond = g_cond.predict_proba(np.hstack([e_s_te, e_m_te]))
    p_base = g_base.predict_proba(e_s_te)
    idx = [list(g_cond.classes_).index(y) for y in y_te]
    pc = np.array([p_cond[i, j] for i, j in enumerate(idx)])
    pb = np.array([p_base[i, list(g_base.classes_).index(y_te[i])] for i in range(len(y_te))])
    return np.log2(pc + eps) - np.log2(pb + eps)     # per-instance CPVI
```

The retrospective/prospective twin (RQ2): the retrospective probe trains and scores with the realised Y; the prospective twin is the same `g_cond` applied at the handoff using only `[e_s ; e_m]` (no Y at inference) to produce a predictive distribution whose agreement with the retrospective score is the H3 test.

### 2.5 Runtime gate (target-free statistics) and the causal arm

Signals: the gate cannot threshold CPVI because CPVI needs the realised outcome; it thresholds a statistic computable at the handoff, calibrated offline against realised outcomes (the D10 circularity fix). Three statistics, one probe-independent.

1. **s_info** - the trained `g_cond`'s predictive confidence or entropy about Y given `[e_s ; e_m]` at the handoff. Uses the offline-trained probe but not the realised outcome, so it is legitimate at runtime.
2. **s_fail** - a dedicated failure-risk classifier trained to predict episode failure from `[e_s ; e_m]`; outputs P(fail) at the handoff.
3. **s_cos** - cosine between `e_m` and a reference embedding (pre-handoff state, or the partner's expected-information template). Probe-independent, so it is the statistic that answers the circularity objection.

Calibration: choose the gate operating point by validating each statistic against realised outcomes on a held-out split (AUROC, reliability curves, ECE), not against CPVI. RQ3b then blocks handoffs below the chosen threshold and re-prompts, and compares against (a) a matched-firing-rate control that blocks the same number of randomly chosen handoffs and (b) a random-trigger control. Gating wins only if it beats both.

---

## 3. Experiment programme (multiple approaches, evaluative)

Signals: one pilot with hard gates, four core experiments in priority order, two optional arms with cut-lines, and a single statistical plan. Approaches that are alternatives (self-play vs heterogeneous, serialisation modes) are run as designed cells, not afterthoughts.

### 3.1 Pilot (Phase 1, 1-2 weeks): de-risk before committing

Three gates decide whether the headline task is viable; failing a gate triggers the fallback ladder, not a scramble.
1. **G1 capability** - on C0 (full channel), do self-play agents solve the task above a floor success rate (e.g. >= 60% within budget on easy slit widths)? If not, the models cannot do the task and the headline pivots to RQ3a.
2. **G2 signal** - between C0 and a hard condition (C1 or C4), is there a measurable success/efficiency gap and a measurable CPVI difference on a small sample? If not, the task has no gradient to study and difficulty/serialisation is retuned once, then re-gated.
3. **G3 groundedness** - do messages reflect the true state (a hallucinated-geometry check, motivated by LLM-Coordination)? If messages are ungrounded, CPVI is measuring noise; tighten the prompt/serialisation once, then re-gate.

Fallback ladder if gates fail after one retune: (1) elevate Who&When/MAST RQ3a to the headline (CPVI as a failure localiser on real logs, no simulator dependency); (2) simplify the task (wider slits, fewer DOF, the macro-action interface); (3) reframe a clean negative result in which CPVI diagnoses absent-vs-unused signal (the Eccles et al. 2019 distinction), which is itself reportable.

### 3.2 RQ1 - information gradient (primary headline)

Design: factorial over conditions C0-C4, serialisation modes (numeric/grid/NL), and difficulty (slit width), self-play with the workhorse model, with seeds for power. Approaches written as cells:
1. **Self-play (primary)** - same model both agents; cleanest isolation of the channel effect; the headline result.
2. **Heterogeneous pair (robustness cell)** - workhorse vs Llama-3.3-70B-AWQ; tests whether the gradient is a single-model artefact; one cell, not the full factorial.
3. **Serialisation A/B** - numeric vs grid vs NL state description; quantifies how much "spatial reasoning" is really prompt formatting (the RoCo lesson) and chooses the serialisation for the rest of the study.
Metrics: success rate, steps-to-goal, collisions, mean and distribution of per-handoff CPVI, and the PVI-minus-CPVI gap. Analysis: mixed-effects model of outcome on condition with random effects for seed and episode; CPVI entered as a mediator to test H2.

### 3.3 RQ2 - measurement primitive

Design: on the RQ1 episodes, compute the retrospective CPVI and the prospective twin per handoff, plus the three runtime statistics. Tests: H3 retrospective-prospective agreement (correlation, Bland-Altman); H4 proxy-vs-CPVI tracking (rank correlation, AUROC of each statistic for predicting low-CPVI and for predicting failure); the divergence analysis (JSD over probe predictive distributions and embedding cosine as the cheap bridge). Output: the calibrated operating point used by RQ3b, and the encoder-sensitivity check.

### 3.4 RQ3a - external validity (Who&When primary, MAST secondary, TRAIL fallback)

Design: extract agent-to-agent handoffs and the surrounding state from real failure logs, compute CPVI per handoff (re-fitting probes on a held-out portion of the logs, or transferring the simulator-trained probe and reporting both), and test whether low CPVI localises the annotated decisive error step (Who&When) or the inter-agent-misalignment trace label (MAST). Baselines to beat: schema validity, mean embedding cosine, and the published Who&When attribution methods (all-at-once, binary search, step-by-step). Labelling per the decision in Q8: use the released MAST LLM-as-judge annotator and the Who&When annotations as ground truth, with a small human-agreement audit on a sample (report kappa); avoid full manual annotation. This is the bridge that turns a synthetic finding into a field claim, and the weak published baselines give real headroom.

### 3.5 RQ3b - causal gate

Design: re-run a subset of RQ1 conditions with the runtime gate active, blocking handoffs whose runtime statistic falls below the RQ2-calibrated threshold and re-prompting, against the matched-firing-rate and random-trigger controls. Test H6 (gating beats both controls on success/efficiency). This is the interventional answer to the Lowe et al. (2019) critique and the demonstration of the active-control-layer contribution.

### 3.6 Optional arms with cut-lines

C5 supervisor (principal-agent): route the handoff through a supervisor relay and measure residual agency loss under aligned incentives, testing the Rauba et al. (2026) assumption. Cut if Phase 1-4 run late. SocialJax MARL comparison: run the information-gradient idea with learned messages for contrast. Cut first; it is a "nice contrast", not load-bearing.

Statistical plan. Pre-register the primary hypotheses and the analysis. Power: pilot estimates the effect size for the C0-to-hard gap; size the seed count to detect it at 80% power; a practical default is tens of episodes per cell across multiple seeds, scaled by the compute decision in §0. Control the family-wise error across the condition contrasts (Holm or Benjamini-Hochberg). Report effect sizes and uncertainty intervals, not just significance, and report seed sensitivity given LLM non-determinism.

---

## 4. Phase-by-phase plan with research milestones

Signals: roughly ten to eleven weeks from mid-June to a late-August dissertation, then about three weeks to the 21 September paper. Each phase has a research milestone (what must be true scientifically) and an engineering deliverable (what must exist in the repo). The pilot gates front-load the central risk. Table abbreviations: R-milestone = the research claim/decision the phase must produce; gate = the go/no-go.

| Phase | Weeks (indicative) | Research milestone (R) | Engineering deliverable | Gate / decision |
|---|---|---|---|---|
| 0. Foundation | wk 1 | Compute and model ladder fixed; reproducibility baseline | Experiments repo, env, vLLM serving on Myriad, CI, data versioning, reproducibility gaps closed | DSE-014 compute decision resolved |
| 1. Pilot | wk 2-3 | Task viability and presence of an information gradient established | Simulator, agent loop, conditions C0-C4, episode runner, logging | G1/G2/G3 pass, or invoke fallback ladder |
| 2. Measurement stack | wk 3-4 | Y definition and probe family locked; CPVI estimator validated on pilot data | Featuriser, PVI/CPVI estimator, probe training, divergence proxy, runtime statistics | Y and V frozen; encoder chosen |
| 3. RQ1 main runs | wk 4-6 | H1/H2 result: gradient measured, CPVI tracks it | Full factorial sweep executed and logged; self-play + heterogeneous + serialisation cells | RQ1 result frozen for write-up |
| 4. RQ2 + gate calibration | wk 6-7 | H3/H4 result: twins agree, proxy tracks CPVI; gate operating point chosen | Twin and proxy analysis; calibrated gate threshold | Gate ready for RQ3b |
| 5. RQ3b causal arm | wk 7-8 | H6 result: gating beats both controls (or a clean null) | Gate-active runs + controls executed | Causal claim frozen |
| 6. RQ3a external validity | wk 7-9 (parallel) | H5 result: CPVI localises real failures vs baselines | Who&When/MAST extraction, CPVI on logs, baseline comparison, human-agreement audit | External-validity claim frozen |
| 7. Dissertation assembly | wk 9-11 | Coherent distinction-grade narrative across all frozen results | Full draft, figures, reproducibility appendix, examiner-runnable artefact | Submit (late August) |
| 8. Paper | post-submission to 21 Sep | RQ1+RQ2 (and gate if ready) packaged | arXiv preprint / workshop submission | Submit 21 September |

Narrative on sequencing. Phases 5 (RQ3b) and 6 (RQ3a) run partly in parallel because RQ3a depends only on the measurement stack (Phase 2), not on the gate; this is the slack that protects the late-August target. If Phase 1 fails its gates, Phase 6 becomes the headline and the simulator phases shrink to a documented negative result, which the schedule absorbs because RQ3a was always a parallel track.

---

## 5. Risks and mitigations

Signals: the live risks are the lit-review red-team's plus execution risk; each has a pre-agreed response so a risk firing is a branch, not a crisis.
1. **Models cannot ground 2D geometry (R1).** Most likely failure. Mitigated by the Phase 1 gates, the serialisation A/B, the discrete macro-action interface, and the RQ3a fallback headline. Managed, not eliminated; the thesis states this honestly.
2. **Probe circularity against CPVI.** Mitigated by validating runtime statistics against realised outcomes and by reporting the probe-independent s_cos.
3. **Floor effect (CPVI near zero).** Mitigated structurally by C3 asymmetric visibility and by reporting the PVI-minus-CPVI gap as a finding.
4. **Compute scarcity / queue latency.** Mitigated by the 14B workhorse default, frontloading the model ladder in Phase 0, and sizing seeds to the resolved allocation.
5. **Frontier closes before submission.** Mitigated by finishing RQ1+RQ2 first and shipping the September paper; the conditional construct and the gate demonstration are the durable claim.
6. **Scope creep / late phases.** Mitigated by the explicit cut-lines on C5 and SocialJax and the parallel RQ3a track.

---

## 6. Dissertation assembly (what a distinction needs)

Signals: a distinction-grade thesis is a coherent argument from a single defensible claim, with reproducible evidence and an honest treatment of the central risk; it is not maximal breadth. The chapter map below maps each experiment to its place, and the writing is incremental so nothing is left to the final fortnight.

Chapter map: (1) introduction and the political-economy-to-multi-agent throughline; (2) the literature-review chapter already drafted; (3) the task, simulator, and agent architecture (from §2.1-2.3); (4) the measurement stack and the CPVI construct (from §2.4, with the IB framing and the conditioning rationale); (5) RQ1 results; (6) RQ2 results and the gate calibration; (7) RQ3a external validity; (8) RQ3b causal arm; (9) discussion, limitations (R1 front and centre), and future work (the C5 incentive-divergence half, the sender-side compression question); appendices for reproducibility (seeds, revisions, jobscripts) and the harness. Writing schedule: draft chapters 3 and 4 during Phases 2-3 while the system is fresh; draft each results chapter in the week its result freezes; reserve Phase 7 for synthesis and polish, not first drafts. Examiner-facing reproducibility: a committed demonstration trace renders (handoff + gate-decision log), and the repo gaps from the audit (pinned encoder revision, CITATION.cff and .bib, package layout) are closed in Phase 0 so the artefact is runnable.

---

## 7. Paper plan (concise)

The September paper is RQ1+RQ2 (and the gate if Phase 5 lands in time), framed as the first measurement and runtime enforcement of conditional usable information at an LLM agent boundary, targeted at a NeurIPS/ICLR/AAMAS 2026 workshop or an arXiv preprint. It reuses the dissertation's chapters 3-6 compressed to workshop length and cites the same verified bibliography. It is explicitly lower priority than finalising the dissertation; the only hard constraints are the late-August submission and the 21 September paper date, and the phase plan hits both.

---

## 8. Definition of done (dissertation)

The dissertation is done when: RQ1 and RQ2 have frozen, written-up results with effect sizes and seed-sensitivity; at least one of RQ3a or RQ3b has a frozen result (both if the schedule holds); the central R1 risk is addressed honestly with either a positive task result or a clean diagnostic negative; the artefact is examiner-runnable with pinned dependencies and a rendered demonstration trace; and the full draft has been through synthesis and polish with the supervisor's sign-off. Anything beyond this - the optional arms, the second external-validity substrate, the cross-family cells - is upside, removable without touching the core.
