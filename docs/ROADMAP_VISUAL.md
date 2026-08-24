# Visual roadmap and change record

> Companion to `RESEARCH_ROADMAP.md`, which carries the prose. This file is the picture and the
> running record of what changed and why. Diagrams are Mermaid and render on GitHub.
>
> **Last revised:** 23 August 2026.

---

## 1. What changed

### 1.1 Since 25 July 2026

Six changes, ordered by how much each moves the dissertation. Every one is a decision already taken,
not an option still open; the evidence for each sits in `docs/experiment_design_log.md` and the
resulting text in `docs/methodology.md` §10.5.

| # | Change | From | To | Why it matters |
|---|---|---|---|---|
| 1 | **Compute resolved** | One open decision blocking the whole plan | UCL Myriad allocation **approved 23 Aug 2026**: 1 GPU ≥40 GB, single node, 8 h wall, 8 cores, 32 GB RAM, ~45 GB weights on scratch | The roadmap's only material unknown is closed. The plan is now execution-bound, not decision-bound |
| 2 | **RQ3a substrate migrated** | Who&When primary | **TraceElephant primary**; Who&When retained as a flagged transfer-only anchor; MAST secondary | Who&When records agent *outputs* only, so it cannot supply the receiver's input context the conditional construct requires, and it is single-class so the refit arm is undefined on it |
| 3 | **Y on real logs defined** | Localise the annotated decisive step | **Counterfactual replay** defines *Y* interventionally; the annotation is named as considered-and-rejected | Using the annotation as *Y* trains the probe on the label the claim is evaluated against. Replay puts the external-validity and causal claims on one epistemology |
| 4 | **Probe methodology hardened** | Shuffled-message audit only | Adds **control-task selectivity**, **repeated cross-fits**, and **pre-registered length controls** | Three different objections need three different nulls. Selectivity is the one an NLP examiner asks for by name |
| 5 | **SocialJax cut on evidence** | Retained as a learned-message contrast | **Cut**, with the reason moved from the schedule to the limitations | The suite ships no communication channel and no communication algorithm, so there is no message to score. Replaced by the absent-versus-unused decomposition, which uses data the main sweep already produces |
| 6 | **Baselines re-anchored** | H5 positioned against 53.5% agent / 14.2% step | Positioned on the **prospective-versus-retrospective and cost axes**; 2025 figures kept as a dated floor | Those numbers have been beaten. Writing to them was the most likely hostile viva question |

### 1.2 Found during this cross-reference, and not in any prior document

Two 2026 papers sit close to the contribution and were absent from the 25 July review. Both are now in
`docs/methodology.md`; the first changes how the contribution must be framed.

- **AgentForesight** (arXiv:2605.08715) reframes failure attribution as *online auditing*: an auditor
  sees only the prefix of an unfolding trajectory and must continue or alarm at the earliest decisive
  error. It therefore occupies the pre-outcome setting, and a contribution claim resting on
  prospectivity alone would now be contested. Three differences survive and are argued in §7 of the
  methodology: it is a **detector, not a measure** — a 7B RL-trained model emitting a verdict, against a
  log-loss difference between two logistic probes; it audits a **trajectory prefix**, not a single
  boundary; and it **raises an alarm without blocking or rewriting** the message, and is not validated
  against a matched-firing-rate control. The causal arm here does exactly that.
- **Causal Agent Replay** (arXiv:2606.08275) formalises step-level counterfactual intervention as a
  structural causal model with a replay-based estimator. It strengthens the citation base for the
  replay-defined *Y* rather than threatening it; what remains novel is using replay to define the
  *target of an information-theoretic boundary measure*, which it does not do.

### 1.3 The state nobody should misread

The engineering is substantially built and the science has not started. Nineteen of thirty original
tickets are merged, the spine from arena to gate calibration is tested, and **zero episodes have been
recorded**. Everything to date has run against stub models. The binding constraint has never been
Myriad access; it is that the pilot has never run, and that there is no command-line driver, so even
with a served model today about thirty lines stand between the repository and its first result.

---

## 2. Programme map

```mermaid
flowchart TD
    S0["S0 · Certify the task<br/>CPU only, no model<br/>A* oracle vs frozen budgets"]
    S1["S1 · Free local pilot<br/>8B on the laptop, zero cost<br/>smoke, transcripts, prompts"]
    G["Pilot gates<br/>G1 capability · G2 signal · G3 groundedness"]
    RT["Retune once<br/>prompts or serialisation"]
    FB["Fallback branch<br/>RQ3a becomes the headline"]
    FZ["FREEZE F0 · Pre-registration<br/>Y, k, V, encoder, serialisation,<br/>channel params, prompts, thresholds"]
    S2["S2 · Myriad re-gate<br/>+ model-ladder benchmark"]
    S3["S3 · RQ1 main sweep<br/>~250 episodes · H1, H2"]
    S4["S4 · RQ2 + calibration<br/>no new compute · H3, H4"]
    S5["S5 · RQ3b causal gate<br/>4 modes · H6"]
    S6["S6 · RQ3a external validity<br/>parallel from day one · H5"]
    W["Dissertation assembly<br/>then the paper"]

    S0 --> S1 --> G
    G -->|proceed| FZ
    G -->|retune once| RT --> G
    G -->|fail after retune| FB --> S6
    FZ --> S2 --> S3 --> S4 --> S5 --> W
    S6 --> W
    S4 -.->|operating point| S5

    classDef gate fill:#fde68a,stroke:#b45309,color:#1c1917
    classDef freeze fill:#bbf7d0,stroke:#15803d,color:#1c1917
    classDef fallback fill:#fecaca,stroke:#b91c1c,color:#1c1917
    class G gate
    class FZ freeze
    class FB fallback
```

The two structural facts in this picture. **S6 hangs off the measurement stack, not off the gate and
not off the sweep**, which is why it runs in parallel and why it can absorb a pilot failure. And
**F0 sits between the pilot and every main run** — choosing *Y* or *V* after seeing main-run results
is the forbidden move, so the freeze is a gate rather than a milestone.

---

## 3. Calendar

```mermaid
gantt
    title precept-research · execution to late September 2026
    dateFormat YYYY-MM-DD
    axisFormat %d %b

    section Unblocking
    Driver + local structured-output adapter   :a1, 2026-08-24, 2d
    S0 task certification                      :a2, 2026-08-24, 1d
    Pin encoder revision                       :a3, after a1, 1d

    section Pilot
    S1 free local pilot, smoke and transcripts :b1, after a1, 3d
    Prompt iteration, max 3 versions           :b2, after b1, 2d
    E3 formal pilot gate run                   :b3, after b2, 2d
    FREEZE F0 pre-registration                 :milestone, crit, after b3, 0d

    section Cluster
    Myriad onboarding and serve.sh dry run     :c1, 2026-08-24, 4d
    S2 re-gate + model-ladder benchmark        :c2, after b3, 2d
    S3 RQ1 main sweep                          :c3, after c2, 7d
    FREEZE F1 RQ1                              :milestone, crit, after c3, 0d

    section Measurement and gate
    S4 RQ2 + gate calibration                  :d1, after c3, 4d
    FREEZE F2 RQ2                              :milestone, crit, after d1, 0d
    S5 RQ3b causal arm                         :d2, after d1, 5d
    FREEZE F3 RQ3b                             :milestone, crit, after d2, 0d

    section External validity, parallel
    E9 corpus spike and counts                 :e1, 2026-08-25, 3d
    Loaders and schema mapping                 :e2, after e1, 4d
    E10 replay labelling, budgeted             :e3, after e2, 5d
    E11 localisation and baselines             :e4, after e3, 4d
    FREEZE F4 RQ3a                             :milestone, crit, after e4, 0d

    section Write-up
    Draft method and system chapters           :f1, after b3, 10d
    Results chapters, as each freezes          :f2, after c3, 18d
    Synthesis and polish                       :f3, after f2, 7d
```

Dates are indicative and the ordering is the load-bearing part. The one hard sequencing constraint is
that **F0 precedes S3**; everything else can slip without invalidating a result.

---

## 4. Critical path and the new tickets

```mermaid
flowchart LR
    subgraph sgdone ["Merged"]
        D1["001-004 foundation"]
        D2["006-009 simulator"]
        D3["010-012 agent spine"]
        D4["013-015 CPVI + twin"]
        D5["016-017 statistics + calibration"]
        D6["019-020 pilot + RQ1 drivers"]
        D7["028 analysis library"]
    end

    subgraph sgblock ["Blocking the first run"]
        N1["DRIVER<br/>console entry point"]
        N2["ADAPTER<br/>response_format for<br/>non-vLLM endpoints"]
        N3["PIN<br/>encoder revision"]
    end

    subgraph sgnew ["New · from the August review"]
        T43["043 control tasks<br/>+ selectivity · P0"]
        T44["044 repeated cross-fits<br/>+ length control · P1"]
        T45["045 gate feedback template<br/>greedy fixed-point escape · P0"]
        T46["046 absent-vs-unused<br/>decomposition · P1"]
        T47["047 re-anchor baselines<br/>+ framing · P0 · docs"]
        T41["041 TraceElephant loader<br/>+ schema mapping · P0"]
        T42["042 counterfactual-replay<br/>labeller · P0"]
        T48["048 gate vs real MAS<br/>via middleware · P2 · stretch"]
    end

    subgraph sgruns ["Runs"]
        R19["E3 pilot gates"]
        RF0["FREEZE F0"]
        R20["E4 RQ1 sweep"]
        R22["E6 RQ2"]
        R17["E7 gate calibration"]
        R25["E8 RQ3b"]
        R24["E11 RQ3a"]
    end

    D6 --> N1 --> R19
    N2 --> R19
    D4 --> T43 --> RF0
    T44 --> RF0
    N3 --> RF0
    R19 --> RF0 --> R20
    R20 --> T46
    R20 --> R22
    R20 --> R17
    D5 --> R17
    R17 -->|operating point| R25
    T45 --> R25
    T41 --> T42 --> R24
    T47 --> R24
    R25 -.-> T48

    classDef blk fill:#fecaca,stroke:#b91c1c,color:#1c1917
    classDef frz fill:#bbf7d0,stroke:#15803d,color:#1c1917
    class N1,N2,N3 blk
    class RF0 frz
```

Read this for two things. The three red boxes are the entire distance between "built" and "running",
and none of them is large. And **DSE-043 and DSE-044 must land before F0**, because both change what
gets frozen — adding a selectivity check after the freeze would mean either re-freezing or reporting a
number the freeze does not cover.

---

## 5. Measurement architecture

```mermaid
flowchart TD
    ST["Simulator state"] -->|serialise: numeric, grid, NL| OA["A's observation"]
    ST --> OB0["B's observation"]
    OA --> A["Agent A"]
    A -->|raw message m| CH{"Channel<br/>C0 pass · C1 cap 8 · C2 delay 1<br/>C3 window ±2 rows · C4 dropout 0.4"}
    OB0 --> CH
    CH -->|delivered message| B["Agent B"]
    CH -->|receiver observation s| B
    B -->|guided JSON action| SIM["Physics step"]
    SIM --> Y["Outcome Y<br/>net geodesic progress over k=3"]

    CH --> EMB["Frozen embeddings<br/>pinned encoder, content-hash cached"]
    EMB --> GC["g_cond on s and m"]
    EMB --> GB["g_base on s alone"]
    Y --> GC
    Y --> GB
    GC --> CPVI["CPVI = log2 g_cond(y) − log2 g_base(y)<br/>report PVI and the PVI − CPVI gap"]
    GB --> CPVI
    GC --> TWIN["Prospective twin<br/>KL(g_cond ‖ g_base), no Y"]
    GB --> TWIN
    GC --> SI["s_info · entropy"]
    SF["s_fail · failure-risk probe"]
    EMB --> SCOS["s_cos · cosine, probe-independent"]
    EMB --> SF

    SI --> CAL["Calibration<br/>against REALISED FAILURE<br/>never against CPVI"]
    SF --> CAL
    SCOS --> CAL
    Y --> CAL
    CAL -->|operating point| GATE["RuntimeGate<br/>block + versioned retry feedback"]
    GATE -.->|blocks before B acts| B

    classDef offline fill:#e0e7ff,stroke:#4338ca,color:#1c1917
    classDef online fill:#fed7aa,stroke:#c2410c,color:#1c1917
    classDef ban fill:#fecaca,stroke:#b91c1c,color:#1c1917
    class CPVI,TWIN offline
    class SI,SF,SCOS,GATE online
    class CAL ban
```

Three invariants this diagram exists to make visible. **The channel touches the message and B's
observation, and nothing else** — no path runs from the channel into physics or the action. **CPVI
consumes Y and the twin does not**, which is why one is an offline audit quantity and the other can
inform a runtime decision. And **calibration reads realised failure, never CPVI**: the red box is the
circularity guard, and `s_cos` exists specifically so that at least one runtime statistic depends on no
fitted probe at all.
