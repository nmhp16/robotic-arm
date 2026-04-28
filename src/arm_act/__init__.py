"""arm-act: Action Chunking Transformer imitation pipeline for Isaac Lab.

UR5 + parallel-jaw, sim-only, one task per checkpoint. New tasks live as a
single ``src/arm_act/tasks/<name>.yaml``; the runtime in
``arm_act.tasks._runtime`` reads it and builds the env, oracle, and mimic
config dynamically.

Public entry points:

* ``arm_act.training.train_act``  — ACT trainer
* ``arm_act.eval.rollout``        — sim rollout evaluator
* ``arm_act.cli.*``               — thin CLI wrappers for the shell scripts
* ``arm_act.config``              — defaults + per-task YAML loader
"""

__version__ = "0.1.0"
