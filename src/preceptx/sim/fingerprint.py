"""Content fingerprint of the simulated world - the constants that shape a trajectory but that no
caller can set through ``SweepConfig``.

``SweepConfig`` already carries every result-shaping knob a caller *can* set (jitter, step, outcome,
the step budgets), so those reach ``sweep_hash`` and the manifest already (P0-2, P1-6). The world
itself does not: slit widths, arena dimensions, load geometry and the grid resolution are module
constants. A retune to any of them therefore left ``dataset_hash_for`` unchanged, so the resume path
read completed episode ids out of the *pre-retune* dataset, found the grid complete, scheduled
nothing, and let the driver re-report the old verdict against the new geometry.

That is the exact path the pre-registration prescribes - PREREGISTRATION section 6 allows one
retune, RESEARCH_ROADMAP section 3.1 names difficulty as the lever - so the failure would have
fired on the cluster, on the verdict of record, looking like a success. Folding this digest into
dataset identity makes that state unrepresentable: changed geometry writes to a different
directory, and there is nothing to mistake for a completed run.

Deliberately NOT fingerprinted here: ``ScenarioJitter``, ``StepConfig``, ``OutcomeConfig`` and the
per-difficulty step budgets. They are ``SweepConfig`` fields and are already inside ``sweep_hash``;
hashing them twice would put one guarantee in two places to keep in step. Derived values
(``load.HALF_H``, ``load.COG_Y``) are likewise omitted - they are pure functions of the T dimensions
that *are* hashed, so they cannot change independently.
"""

from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, ConfigDict

from preceptx.sim import arena, load, serialise

# Bump to force a deliberate re-key when a change cannot be captured structurally - a behavioural
# change in pymunk's stepping, say, that leaves every constant here identical.
ENVIRONMENT_SCHEMA_VERSION = 1


class SimulationFingerprint(BaseModel):
    """The world constants, grouped by the module that owns them so an audit reads top-down.

    Persisted whole in the sweep manifest alongside its ``digest()``: the digest is what prevents
    an unsafe resume, but the payload is what lets someone six weeks later say *why* a dataset
    changed identity rather than only that it did.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int
    slit_widths: dict[str, float]
    arena: dict[str, float]
    load: dict[str, float]
    grid: dict[str, float]

    def digest(self) -> str:
        """sha256 over canonical (sorted-key) JSON, 16 hex - the same shape as ``sweep_hash``."""
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def simulation_fingerprint() -> SimulationFingerprint:
    """Snapshot the live world constants.

    Pure and process-independent: every value is a module constant or a default-constructed
    Pydantic model dumped to primitives, so the same source tree yields the same digest in any
    process. No ``repr()``, no object identity, no runtime state.
    """
    geometry = arena.ArenaGeometry().model_dump()
    return SimulationFingerprint(
        schema_version=ENVIRONMENT_SCHEMA_VERSION,
        slit_widths={k: float(v) for k, v in arena.slit_widths().items()},
        arena={
            **{k: float(v) for k, v in geometry.items()},
            "damping": arena.DAMPING,
            "load_mass": arena.LOAD_MASS,
            "goal_radius": arena.GOAL_RADIUS,
            "wall_friction": arena.WALL_FRICTION,
        },
        load={
            "thick": load.T_THICK,
            "bar": load.T_BAR,
            "stem": load.T_STEM,
            "friction": load.T_FRICTION,
        },
        grid={k: float(v) for k, v in serialise.GridConfig().model_dump().items()},
    )
