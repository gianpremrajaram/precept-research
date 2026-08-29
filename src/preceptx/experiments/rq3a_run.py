"""The RQ3a driver: corpus identity, the judge backend, and the run manifest (DSE-064, DSE-065).

DSE-031 shipped the pilot, RQ1 and RQ2 entry points and listed "the gate and RQ3 drivers" as out of
scope. The gate driver arrived with DSE-018; this is the other half. Until it existed, the RQ3a
loaders, scorers and audits could only be reached from a test - roughly 1,800 lines of analysis with
no way to point it at a corpus.

Three things live here and nothing else:

* **Corpus identity.** The simulator side keys a dataset on ``dataset_hash_for``; a fetched public
  corpus has no such handle. Identity here is a content digest over the *loaded records*, which is
  the exact surface the analysis sees - narrower than the download (a re-zip with the same contents
  is the same corpus) and wider than a filename (a silently revised HuggingFace upload is not).
* **:class:`VLLMJudge`.** The three published Who&When procedures need an annotator, and every
  model call in this project is local or on the Myriad allocation, so the judge is the served
  open-weight tier and ``JudgeIdentity`` records it as a replication rather than a reproduction.
  Abstention is *decoded*, never caught: the schemas offer ``-1`` and ``"unsure"``, so a refusal is
  an answer the model gave, while a broken endpoint stays an exception (CLAUDE.md: fail loud).
* **:class:`RQ3aManifest`.** Neither ``RunManifest`` (which requires an ``ExperimentConfig``) nor
  ``SweepManifest`` (which requires a simulation fingerprint) describes a run over real logs.
  Fabricating either would write invented physics into a manifest - the same weakening DSE-041
  rejected when it kept ``LogHandoffRecord`` separate from ``HandoffRecord``. This is the
  log-substrate analogue, versioned on its own counter for the same reason.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from preceptx.config import ConfigError
from preceptx.data.logs import LogHandoffRecord, LogTraceRecord
from preceptx.experiments.rq3a import (
    JudgeBackend,
    RQ3aConfig,
    RQ3aResult,
    analyse_rq3a,
    manifest_metrics,
)
from preceptx.experiments.rq3a_load import (
    CorpusCounts,
    count_handoff_corpus,
    count_trace_corpus,
    load_mast,
    load_traceelephant,
    load_who_and_when,
)
from preceptx.manifest import ServeEnv, dep_versions, git_dirty, git_sha, serve_env
from preceptx.measure.featuriser import Featuriser
from preceptx.serving.client import ChatMessage, LLMClient

logger = logging.getLogger(__name__)

# 2 (DSE-024): the transfer regime went live, so the manifest records which arena dataset
# trained the statistic that was applied here. A v1 artefact predates the transfer arm.
RQ3A_MANIFEST_VERSION = 2
"""Bump on any breaking change to :class:`RQ3aManifest`.

On its own counter, independent of ``MANIFEST_VERSION`` and ``SWEEP_MANIFEST_VERSION``: the log
substrate and the simulator substrate are versioned separately so a change to one never invalidates
the other's artefacts.
"""

HandoffCorpus = Literal["traceelephant", "who_and_when"]
"""The two per-step corpora. MAST is trace-level and rides along as a secondary, never alone."""


# --------------------------------------------------------------------------------------------
# Corpus identity and loading
# --------------------------------------------------------------------------------------------


def corpus_digest(records: list[LogHandoffRecord] | list[LogTraceRecord]) -> str:
    """A content digest over the loaded records - the corpus's analogue of ``dataset_hash``.

    Taken over what the analysis consumes rather than over the downloaded bytes, so it is stable
    across a re-download or a re-zip and moves the moment the upstream corpus is revised.

    Records are **sorted before hashing**, not taken in load order. Only ``load_traceelephant``
    walks a sorted glob; ``load_who_and_when`` follows parquet row order and ``load_mast`` the
    JSON array's, both of which are deterministic for a given file but not canonical across a
    rewrite of it. Since this digest is the identity stamped into every RQ3a manifest, it has to
    be a function of the corpus's *content* and not of how the file happened to be laid out.
    """
    if not records:
        raise ConfigError("cannot digest an empty corpus")
    h = hashlib.sha256()
    for record in sorted(records, key=_sort_key):
        h.update(json.dumps(record.model_dump(mode="json"), sort_keys=True).encode())
    return h.hexdigest()[:16]


def _sort_key(record: LogHandoffRecord | LogTraceRecord) -> tuple[str, int]:
    """``(trace_id, step)`` - the pair that identifies a row in both log schemas.

    ``LogTraceRecord`` is one row per trace and carries no step, so it sorts on ``trace_id`` alone
    with a constant second term rather than needing a second code path.
    """
    return (record.trace_id, getattr(record, "step", 0))


def corpus_paths(root: Path) -> dict[str, Path]:
    """Where each corpus lands under a root populated by ``scripts/fetch_rq3a.sh``."""
    return {
        "traceelephant": root / "traceelephant" / "data",
        "who_and_when": root / "who_and_when",
        "mast": root / "mast" / "MAD_full_dataset.json",
    }


def load_corpus(corpus: HandoffCorpus, root: Path) -> list[LogHandoffRecord]:
    """Load one per-step corpus from a fetched root."""
    paths = corpus_paths(root)
    if corpus == "traceelephant":
        return load_traceelephant(paths["traceelephant"])
    return load_who_and_when(paths["who_and_when"])


def projected_judge_calls(records: list[LogHandoffRecord], *, handoffs_only: bool) -> int:
    """Upper bound on the judge's model calls across the three procedures.

    All-at-once is one call per trace, binary search ``ceil(log2 n)``, and step-by-step at most *n*
    - fewer in practice, because it stops at the first yes. An upper bound is the right shape for a
    pre-flight: it is the number that must fit in the job's wall clock.
    """
    lengths: dict[str, int] = {}
    for r in records:
        if handoffs_only and not r.is_handoff:
            continue
        lengths[r.trace_id] = lengths.get(r.trace_id, 0) + 1
    return sum(1 + math.ceil(math.log2(max(n, 2))) + n for n in lengths.values())


# --------------------------------------------------------------------------------------------
# The judge backend
# --------------------------------------------------------------------------------------------

# Abstention is a value in the schema, not a parse failure. `-1` and "unsure" are reachable answers,
# so a model that genuinely cannot tell says so and is recorded as an abstention, while a refusal to
# emit valid JSON at all remains a ServingError. Guided decoding makes both branches total.
_SELECT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"step": {"type": "integer"}},
    "required": ["step"],
    "additionalProperties": False,
}
_YES_NO_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"answer": {"type": "string", "enum": ["yes", "no", "unsure"]}},
    "required": ["answer"],
    "additionalProperties": False,
}

_SYSTEM = (
    "You audit transcripts of multi-agent AI systems that failed their task. You identify the "
    "single step where the system first went decisively wrong - the step that caused the failure, "
    "not a later step that merely inherited it. Answer only in the requested JSON form. If the "
    "transcript does not let you tell, say so rather than guessing."
)


class VLLMJudge(JudgeBackend):
    """The three Who&When procedures against the served open-weight tier.

    A replication, not a reproduction: the published baselines were run against a hosted frontier
    annotator, and ``JudgeIdentity`` carries that caveat into every artefact this backend touches.
    The backend only answers - the procedures (all-at-once, binary search, step-by-step) stay in
    ``rq3a.py``, so no answer parsing leaks into the analysis.
    """

    def __init__(self, client: LLMClient, *, revision: str) -> None:
        self._client = client
        cfg = client.config
        self.model_name = cfg.model
        self.model_revision = revision
        self.decoding = f"temperature={cfg.temperature} seed={cfg.seed} {cfg.structured_mode}"

    def _ask(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        return self._client.structured(
            [
                ChatMessage(role="system", content=_SYSTEM),
                ChatMessage(role="user", content=prompt),
            ],
            schema,
        )

    def select_step(self, transcript: str, n_steps: int) -> int | None:
        """All-at-once: the whole transcript, one question, one index back."""
        answer = self._ask(
            f"{transcript}\n\n"
            f"This trace has {n_steps} steps, numbered 0 to {n_steps - 1}. Which step is the "
            'decisive mistake? Reply {"step": <index>}, or {"step": -1} if you cannot tell.',
            _SELECT_SCHEMA,
        )
        step = answer.get("step")
        # Out of range is an abstention rather than an error: the procedure in rq3a.py already
        # treats an unusable index as one, and a judge that points off the end has not answered.
        return None if not isinstance(step, int) or not 0 <= step < n_steps else step

    def contains_error(self, transcript: str) -> bool | None:
        """Binary search: does this contiguous segment contain the decisive step?"""
        answer = self._ask(
            f"{transcript}\n\n"
            "Does this segment of the trace contain the decisive mistake? "
            'Reply {"answer": "yes"}, {"answer": "no"}, or {"answer": "unsure"}.',
            _YES_NO_SCHEMA,
        )
        return _tri(answer.get("answer"))

    def is_error(self, transcript: str, step_text: str) -> bool | None:
        """Step-by-step: is this one step, in its preceding context, the decisive mistake?"""
        answer = self._ask(
            f"Preceding context:\n{transcript or '(none - this is the first step)'}\n\n"
            f"The step under review:\n{step_text}\n\n"
            "Is the step under review the decisive mistake? "
            'Reply {"answer": "yes"}, {"answer": "no"}, or {"answer": "unsure"}.',
            _YES_NO_SCHEMA,
        )
        return _tri(answer.get("answer"))


def _tri(value: object) -> bool | None:
    """yes/no/unsure -> True/False/None. Anything else is an abstention, never a default."""
    return {"yes": True, "no": False}.get(str(value))


# --------------------------------------------------------------------------------------------
# The manifest
# --------------------------------------------------------------------------------------------


class RQ3aManifest(BaseModel):
    """The reproducibility record for one RQ3a run over real logs."""

    model_config = ConfigDict(extra="forbid")

    manifest_version: int = RQ3A_MANIFEST_VERSION
    git_sha: str
    git_dirty: bool
    # Corpus identity: the log substrate's analogue of `dataset_hash`, plus the counts that say
    # what was actually in it. Both primary and secondary are recorded, so a run that omitted the
    # MAST arm is distinguishable from one where MAST was empty.
    corpus: str
    corpus_digest: str
    corpus_root: str
    counts: CorpusCounts
    mast_digest: str | None = None
    mast_counts: CorpusCounts | None = None
    # The analysis is fully described by RQ3aResult.provenance (encoder, probe family, git SHA);
    # the revision is lifted out here so the manifest alone answers "which encoder produced this".
    encoder_revision: str
    cfg: RQ3aConfig
    # Which arena dataset trained the transferred statistic. `cfg` records the directory and key;
    # this records the substrate, so the manifest alone answers "transferred from what?" without
    # resolving a path that may not outlive the run.
    transfer_train_dataset_hash: str | None = None
    # Judge identity, absent on a no-judge run. `serving_substrate` is demanded of any run that
    # makes model calls, for the reason DSE-031 gives: local-pilot and Myriad data must stay
    # permanently distinguishable, and an unlabelled artefact cannot be told apart after the fact.
    judge_model: str | None = None
    judge_revision: str | None = None
    judge_decoding: str | None = None
    serving_substrate: str = "unspecified"
    serve_env: ServeEnv | None = None
    command: list[str]
    dep_versions: dict[str, str]
    timestamp: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    artefact_paths: dict[str, str] = Field(default_factory=dict)


def build_rq3a_manifest(
    result: RQ3aResult,
    *,
    corpus_root: Path,
    digest: str,
    counts: CorpusCounts,
    cfg: RQ3aConfig,
    command: list[str],
    mast_digest: str | None = None,
    mast_counts: CorpusCounts | None = None,
    transfer_train_dataset_hash: str | None = None,
    artefact_paths: dict[str, str] | None = None,
) -> RQ3aManifest:
    """Assemble the manifest from the finished result plus the live environment."""
    judge = result.judge
    return RQ3aManifest(
        git_sha=git_sha(),
        git_dirty=git_dirty(),
        corpus=result.corpus,
        corpus_digest=digest,
        corpus_root=str(corpus_root),
        counts=counts,
        mast_digest=mast_digest,
        mast_counts=mast_counts,
        encoder_revision=result.provenance.encoder_revision,
        cfg=cfg,
        transfer_train_dataset_hash=transfer_train_dataset_hash,
        judge_model=None if judge is None else judge.model_name,
        judge_revision=None if judge is None else judge.model_revision,
        judge_decoding=None if judge is None else judge.decoding,
        serving_substrate=os.environ.get("PRECEPTX_SERVING_SUBSTRATE", "unspecified"),
        serve_env=serve_env(),
        command=command,
        dep_versions=dep_versions(),
        timestamp=dt.datetime.now(dt.UTC).isoformat(),
        metrics=manifest_metrics(result),
        artefact_paths=artefact_paths or {},
    )


# --------------------------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------------------------


class RQ3aRun(BaseModel):
    """Everything one run produced, ready to be written."""

    model_config = ConfigDict(extra="forbid")

    result: RQ3aResult
    manifest: RQ3aManifest


def run_rq3a(
    corpus: HandoffCorpus,
    root: Path,
    featuriser: Featuriser,
    *,
    cfg: RQ3aConfig | None = None,
    judge: JudgeBackend | None = None,
    with_mast: bool = True,
    command: list[str] | None = None,
    transfer_train_dataset_hash: str | None = None,
) -> RQ3aRun:
    """Load one corpus, score every available method on it, and build the manifest.

    ``judge=None`` skips the three replications - the only methods that cost model calls - and
    ``with_mast=False`` drops the trace-level secondary. Both omissions are visible in the result
    and the manifest rather than silently changing what the comparison contains.
    """
    cfg = cfg or RQ3aConfig()
    records = load_corpus(corpus, root)
    counts = count_handoff_corpus(records)
    logger.info(
        "%s: %d traces, %d steps, %d handoffs (%d failures, %d non-failures)",
        corpus,
        counts.traces,
        counts.steps,
        counts.handoffs,
        counts.failures,
        counts.non_failures,
    )
    mast_records = load_mast(corpus_paths(root)["mast"]) if with_mast else None
    result = analyse_rq3a(records, featuriser, cfg=cfg, judge=judge, mast_traces=mast_records)
    return RQ3aRun(
        result=result,
        manifest=build_rq3a_manifest(
            result,
            corpus_root=root,
            digest=corpus_digest(records),
            counts=counts,
            cfg=cfg,
            command=command or [],
            mast_digest=None if mast_records is None else corpus_digest(mast_records),
            mast_counts=None if mast_records is None else count_trace_corpus(mast_records),
            transfer_train_dataset_hash=transfer_train_dataset_hash,
            artefact_paths={"rq3a": "rq3a.json", "table": "rq3a_localisation.csv"},
        ),
    )


def write_rq3a_manifest(manifest: RQ3aManifest, dir: Path | str) -> Path:
    """Persist the manifest beside the analysis artefacts ``write_rq3a`` already wrote."""
    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)
    path = dir / "manifest.json"
    path.write_text(manifest.model_dump_json(indent=2))
    return path
