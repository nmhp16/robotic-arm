"""Jar-aware runtime template (sibling of `_runtime`).

This template extends the bundled pick-and-place archetype with an
INSERT phase that descends below the target's z (into a confined
cavity such as a jar), and a RETRACT phase that lifts the gripper out
of the cavity before the episode ends.

The env / mimic / mdp / robot side does not change -- those are
re-exported from `_runtime` so the env_cfg builder stays in one place.
Only the oracle's state machine differs.

Tasks select this template via the YAML key:

    template: _runtime_jar
"""

from . import oracle  # noqa: F401
