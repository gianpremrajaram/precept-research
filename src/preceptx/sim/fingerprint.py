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

``StepConfig`` IS fingerprinted, via the ``actions`` field. An earlier version of this docstring
said otherwise while the code hashed it anyway (corrected in DSE-059). Hashing it here is the right
call and the redundancy with ``sweep_hash`` is deliberate: ``hold_orientation`` and
``angular_impulse`` decide whether contact can rotate the load and how far one action turns it, so a
dataset must re-key on them whether or not a caller happened to route them through ``SweepConfig``.

Deliberately NOT fingerprinted here: ``ScenarioJitter``, ``OutcomeConfig`` and the per-difficulty
step budgets. They are ``SweepConfig`` fields and are already inside ``sweep_hash``. Derived values
(``load.LOAD_COG_Y``, ``actions.ANGULAR_IMPULSE``) are likewise omitted - they are pure functions of
values that *are* hashed, so they cannot change independently.
"""

from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, ConfigDict

from preceptx.sim import actions, arena, load, serialise

# Bump to force a deliberate re-key when a change cannot be captured structurally - a behavioural
# change in pymunk's stepping, say, that leaves every constant here identical.
ENVIRONMENT_SCHEMA_VERSION = 3


class SimulationFingerprint(BaseModel):
    """The world constants, grouped by the module that owns them so an audit reads top-down.

    Persisted whole in the sweep manifest alongside its ``digest()``: the digest is what prevents
    an unsafe resume, but the payload is what lets someone six weeks later say *why* a dataset
    changed identity rather than only that it did.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int
    # The load's shape family, not a dimension: a 1.4x0.3 bar and a T whose bar is 1.4x0.3 are
    # different worlds that could otherwise agree on every recorded number (DSE-058).
    load_shape: str
    slit_widths: dict[str, float]
    arena: dict[str, float]
    load: dict[str, float]
    # Action-model parameters that change outcomes without being arena or load dimensions - notably
    # `hold_orientation`, which decides whether contact can rotate the load (DSE-058).
    actions: dict[str, float]
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
        load_shape="bar",
        slit_widths={k: float(v) for k, v in arena.slit_widths().items()},
        arena={
            **{k: float(v) for k, v in geometry.items()},
            "damping": arena.DAMPING,
            "collision_slop": arena.COLLISION_SLOP,
            "load_mass": arena.LOAD_MASS,
            "goal_radius": arena.GOAL_RADIUS,
            "wall_friction": arena.WALL_FRICTION,
        },
        load={
            "len": load.BAR_LEN,
            "thick": load.BAR_THICK,
            "friction": load.T_FRICTION,
        },
        actions={
            k: float(v)
            for k, v in actions.StepConfig().model_dump().items()
            if isinstance(v, (int, float, bool))
        },
        grid={k: float(v) for k, v in serialise.GridConfig().model_dump().items()},
    )
