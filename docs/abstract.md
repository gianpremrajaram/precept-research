# Usable Information at LLM Agent Boundaries

Gian Prem Rajaram | UCL MSc Computer Science | Supervisor: Prof. Philip Treleaven, advised by Prof. Jun Wang

> **Provenance.** Moved into the repository 23 August 2026 and revised against the shipped
> implementation and the August design decisions, so that the abstract, the methodology
> (`docs/methodology.md`) and the code cannot drift apart unnoticed. Supersedes the standalone
> `Thesis_Abstract.md`, retained outside the repo as a dated snapshot. Changes in this revision:
> the runtime statistics now match what is implemented, the RQ3a substrate and outcome are updated,
> the SocialJax contrast is removed, and the outcome horizon is stated explicitly.

**Abstract**

This thesis investigates the measurement and runtime enforcement of usable information at the natural-language boundary between two coordinating large language model (LLM) agents. A multi-agent task is split across agents because no single context window holds it all, so they coordinate through handoff messages, and that boundary, not the reasoning of any one agent, is where coordination quietly fails, a shortfall already seen in restricted-communication collectives (Dreyer et al., 2025) and in delegation under aligned incentives (Rauba et al., 2026). Each handoff is scored for the usable information it adds about the next step outcome beyond the shared state: conditional pointwise V-usable information (CPVI), the log-loss reduction of a probe reading state and message over one reading state alone (Hewitt et al., 2021; Xu et al., 2020). The outcome is physics-defined: whether the next short horizon of joint actions makes net geodesic progress (a binary label) and by how much (a continuous twin). The conditioning state is the state observable to the **receiver**, not to the sender, which is what makes the score a statement about what the agent about to act did not already know.

Existing measures read communication too generously or target the wrong object: correlational metrics over-read, registering a signal where the message has no causal effect (Lowe et al., 2019); population-level synergy measures a whole collective, not a single boundary (Riedl, 2025); and failure-attribution methods label a trace after the fact without a deployable boundary score (Cemri et al., 2025). V-usable information replaces mutual information, which is not estimable at sentence-embedding dimensionality, and conditioning on the shared state makes the measure honest: it subtracts the part of a message that only echoes the board. The construct is tested on the Pymunk 2D rigid-body engine, where two agents manoeuvre a T-shaped load through a three-chamber arena under a single degradable channel, with locally served open-weight models in self-play (Qwen3-14B), so the channel is the only manipulated variable. The throughline: political economy and multi-agent coordination are one problem, how agents with partial, asymmetric information produce a collective outcome.

This thesis comprises three investigations, believed by the author to be original contributions, presented as follows and discussed in more detail in §1.3:

**Experiment 1: the usable-information gradient under channel degradation.** Degrading the handoff channel across four conditions (a length cap, delayed delivery, asymmetric visibility, and randomised text redaction) tests whether task success and step efficiency fall, and whether mean per-handoff CPVI falls with them and explains outcome variance beyond the shared state. A serialisation A/B (numeric, grid, natural language) separates spatial reasoning from prompt formatting. A pre-registered secondary analysis splits handoffs on usable information against realised progress to separate an absent signal, which the sender failed to encode, from an unused one, which the receiver failed to act on (Eccles et al., 2019) — a distinction a single correlational number cannot make, and one that converts a weak result into a reportable finding about which half of the channel failed.

**Experiment 2: the measurement primitive and a runtime proxy.** This study tests whether a retrospective CPVI, scored after the outcome, agrees with a prospective twin scored before it from the handoff and shared state, and whether a target-free decision-time statistic tracks the offline ground truth closely enough to act on. Three statistics are calibrated against realised outcomes, not against CPVI: the entropy of the message-reading probe's predictive distribution, a dedicated failure-risk probability, and a probe-independent embedding cosine that answers the circularity objection. The Jensen-Shannon divergence between the two probes' predictions is reported as the analytical bridge between the offline and runtime quantities rather than as a gate statistic. The prospective twin is a Kullback-Leibler divergence and so non-negative where the retrospective score is signed, and that one-sidedness is reported as a finding about the limits of decision-time measurement.

**Experiment 3: external and causal validity of the boundary measure.** Boundary CPVI is computed on real multi-agent failure logs, with TraceElephant primary because it records each agent's input context and therefore the state the conditional measure requires, Who&When retained as a flagged comparability anchor, and MAST-Data secondary. The per-step outcome is defined by counterfactual replay rather than by the human attribution annotation, since training the probe on the label the localisation claim is evaluated against would be circular; the measure is tested against schema validity and embedding cosine, and positioned against published attribution methods on cost and on availability before the outcome exists rather than on raw localisation accuracy. For causal validity, low-information handoffs are blocked and rewritten before the receiver acts, and success is compared against a matched-firing-rate control (the same number of random handoffs blocked) and a random-trigger control; an optional supervisor relay, deferred because a third agent creates a second boundary, would test whether aligned-incentive asymmetry alone loses information.

This thesis presents the following original contributions, discussed further in §1.4:

1. **Conditional usable information measured at an inter-agent text boundary.** To the author's knowledge, the first application of conditional pointwise V-usable information to a coordination handoff. The measure scores what a message adds about an objective outcome beyond the shared state, separating signal from an echo of the shared state, and yields a per-handoff distribution, not an aggregate.

2. **A target-free runtime measure and a causal test of its value.** A decision-time statistic, calibrated against realised outcomes, that tracks the offline CPVI without the realised label; and an interventional result settling whether low usable information causes failure, since blocking low-information handoffs improves outcomes over matched-firing-rate and random-trigger controls.

3. **External validity and a principal-agent account of information loss.** Evidence about whether the boundary measure localises the decisive failure step in real logs, against an outcome defined by intervention rather than annotation, and reported on the operating characteristic — what localisation is obtained per unit of compute, and whether any of it exists before the outcome does — rather than against a purpose-trained tracer's raw accuracy. Framed through the principal-agent view: aligned incentives do not guarantee that a constrained channel preserves the information the principal's outcome depends on, so the residual loss is measurable at the boundary.

---

## Indicative key references

Harvard author-year; the verified full set with BibTeX is held in the literature-review chapter.

1. Tishby, N., Pereira, F. & Bialek, W. (1999) The information bottleneck method. *37th Allerton Conference on Communication, Control and Computing*.
2. Williams, P. L. & Beer, R. D. (2010) *Nonnegative decomposition of multivariate information*. arXiv:1004.2515.
3. Xu, Y., Zhao, S., Song, J., Stewart, R. & Ermon, S. (2020) A theory of usable information under computational constraints. *ICLR*.
4. Ethayarajh, K., Choi, Y. & Swayamdipta, S. (2022) Understanding dataset difficulty with V-usable information. *ICML* (Outstanding Paper).
5. Hewitt, J., Ethayarajh, K., Liang, P. & Manning, C. D. (2021) Conditional probing: measuring usable information beyond a baseline. *EMNLP*.
6. Lu, S. et al. (2023) Measuring pointwise V-usable information in-context-ly. *EMNLP Findings*.
7. Lowe, R., Foerster, J., Boureau, Y.-L., Pineau, J. & Dauphin, Y. (2019) On the pitfalls of measuring emergent communication. *AAMAS*.
8. Eccles, T., Bachrach, Y., Lever, G., Lazaridou, A. & Graepel, T. (2019) Biases for emergent communication in multi-agent reinforcement learning. *NeurIPS*.
9. Wang, R. et al. (2020) Learning efficient multi-agent communication: an information bottleneck approach. *ICML*.
10. Dreyer, B., Haluts, A., Korman, A., Gov, N., Fonio, E. & Feinerman, O. (2025) Comparing cooperative geometric puzzle solving in ants versus humans. *PNAS*.
11. Mandi, Z., Jain, S. & Song, S. (2023) *RoCo: dialectic multi-robot collaboration with large language models*. arXiv:2307.04738.
12. Agashe, S. et al. (2024) *LLM-Coordination: evaluating and analyzing multi-agent coordination abilities in large language models*. (NAACL 2025 Findings.)
13. Zhang, H. et al. (2024) Building cooperative embodied agents modularly with large language models. *ICLR*.
14. Riedl, C. (2025) *Emergent coordination in multi-agent language models*. arXiv:2510.05174.
15. Ashery, A. F., Aiello, L. M. & Baronchelli, A. (2025) Emergent social conventions and collective bias in LLM populations. *Science Advances*.
16. SocialJax (2026) *An evaluation suite for multi-agent reinforcement learning in sequential social dilemmas*. arXiv:2503.14576.
17. Cemri, M. et al. (2025) *Why do multi-agent LLM systems fail?* arXiv:2503.13657.
18. Rauba, P. et al. (2026) *Multi-agent systems should be treated as principal-agent problems*. arXiv:2601.23211.
19. Lin, Y., Li, W., Zha, H. & Wang, B. (2023) Information design in multi-agent reinforcement learning. *NeurIPS*.
20. Liu, W. et al. (2024) Autonomous agents for collaborative task under information asymmetry. *NeurIPS*.
