"""Re-export the default env_cfg builder.

`_runtime_jar` only diverges from `_runtime` in the oracle's state
machine; the env / actions / observations are identical, so we reuse
the same builder rather than duplicate ~200 lines.
"""

from arm_act.tasks._runtime.env_cfg import build_env_cfg  # noqa: F401
