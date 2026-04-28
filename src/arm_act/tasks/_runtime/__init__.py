"""Parametric template for pick-and-place tasks.

Builders here turn a task spec dict (loaded from ``tasks/<name>.yaml``)
into Isaac Lab configclasses + a Mimic env. End users edit the YAML;
they don't import from this package.

Modules:

* ``env_cfg.build_env_cfg``           — env_cfg subclass with the spec applied
* ``mimic_env_cfg.build_mimic_env_cfg`` — env_cfg + MimicEnvCfg with subtasks
* ``mimic_env.build_mimic_env``       — mimic env class with subtask signals
* ``robot_cfg.build_robot_cfg``       — robot ArticulationCfg lookup
* ``oracle.main(spec)``               — scripted demo collector
* ``smoke.main(spec)``                — scene preview
* ``mdp``                             — generic MDP terms (obs + termination funcs)
* ``events``                          — generic reset randomization
* ``base_env_cfg``                    — shape-only base configclasses
"""
