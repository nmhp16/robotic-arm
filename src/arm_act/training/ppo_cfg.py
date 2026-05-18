"""Default rsl_rl PPO agent config for the RL-finetune gym envs.

Used by ``~/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py``: when
that script does ``gym.spec(task).kwargs["rsl_rl_cfg_entry_point"]``, it
gets a path that resolves to ``DefaultPPORunnerCfg`` below.

Why this lives in arm_act, not the task yaml:

  - The agent config is a *training-side* artifact (network shape, PPO
    hyperparams, rollout horizon) — not part of the env. Putting it in
    Python lets us reuse one ``RslRlOnPolicyRunnerCfg`` across every
    ``Isaac-…-RL-v0`` task without copying yaml.
  - Per-task hyperparameter overrides happen at the CLI:
    ``rsl_rl/train.py --task X --num_envs 1024 --max_iterations 3000``.

Network/PPO knobs picked for short-horizon visuomotor manipulation
(4-DOF SCARA, 4-D action, ~30 D proprio obs). They are NOT tuned for
high-DOF locomotion — a humanoid PPO setup would want bigger nets and
more learning epochs."""

from __future__ import annotations

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class DefaultPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Default PPO runner for any of the ``Isaac-<Task>-…-RL-v0`` envs.

    Override on the CLI for per-task tuning; the defaults are a sensible
    starting point for ~4-DOF arm pick-and-place tasks at 64-1024 envs.
    """

    # --- runner -----------------------------------------------------------
    num_steps_per_env = 24                    # PPO rollout horizon (env steps)
    max_iterations = 3000                     # total PPO iterations
    save_interval = 25                        # checkpoint every N iters
                                              # (cheap insurance against
                                              # PPO crashes mid-run)
    experiment_name = "arm_act_rl"            # subdir under runner.log_dir
    empirical_normalization = False           # let RslRl normalize obs

    # The env exposes three obs groups: "policy" (state), "rgb_camera"
    # (always empty for RL — see _apply_reward_params), and
    # "subtask_terms" (grasp/place booleans). The MLP actor-critic uses
    # only the state vector. If you switch to a CNN actor later, add
    # "rgb_camera" to the actor's list and re-enable image obs.
    obs_groups = {
        "actor": ["policy"],
        "critic": ["policy"],
    }

    # --- policy network ---------------------------------------------------
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,                   # initial action stddev
        actor_obs_normalization=True,         # state input has different scales
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],     # ~30D obs → 4D action
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    # --- PPO algorithm ----------------------------------------------------
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,                       # standard PPO clip range
        entropy_coef=0.01,                    # slightly higher than locomotion;
                                              # manipulation rewards are sparser
                                              # and need more exploration.
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",                  # KL-targeted LR adjustment
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
