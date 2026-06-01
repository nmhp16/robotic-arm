"""Decisive form-closure test: does the stem collar hold the plant under NOISY
control where pure friction slips?

The wall (confirmed 4 ways — BC/DAgger/RL/RL+curriculum): the round-stem friction
grip holds under smooth *scripted* control but slips under any *learned/exploring*
controller's lateral noise. Form-closure (a wide collar resting on the prong tops)
should carry the load by geometry instead.

Protocol per world, pre-gripped (init_mode="grasped", auto_grip): a clean vertical
LIFT phase (both collar on/off should hold — validates the grip+init), then a
LATERAL-NOISE carry phase (random ±MAXD xy deltas = worst-case RL exploration).
Report the fraction of worlds still holding the plant lifted at the end.

  ~/newton-probe/bin/python scripts/newton/test_form_closure.py [N]
"""
import sys, pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import mujoco_urdf_test as M          # noqa: E402


def run(form_closure: bool, N: int, shear: float, lift_steps=12, shear_steps=40, seed=0):
    M.FORM_CLOSURE = form_closure
    from arm_act.newton.rl_env import NewtonVecEnv
    env = NewtonVecEnv(num_envs=N, spawn_jitter=0.006,
                       max_ctrl=lift_steps + shear_steps, init_mode="grasped")
    env.reset()
    pz0 = env._plant()[:, 2].copy()

    rng = np.random.RandomState(seed)
    # Phase 1: clean vertical lift (no lateral) — both should hold.
    for _ in range(lift_steps):
        a = np.zeros((N, 4), np.float32); a[:, 2] = 0.02
        env.step(a)
    pz_lift = env._plant()[:, 2].copy()

    # Phase 2: carry with lateral exploration noise of magnitude `shear`.
    for _ in range(shear_steps):
        a = np.zeros((N, 4), np.float32)
        a[:, 0] = rng.uniform(-shear, shear, N)
        a[:, 1] = rng.uniform(-shear, shear, N)
        a[:, 2] = rng.uniform(-0.005, 0.01, N)   # small vertical jitter
        env.step(a)
    pz_end = env._plant()[:, 2].copy()

    lifted_clean = (pz_lift - pz0) > 0.02               # rose during clean lift
    held = ((pz_end - pz0) > 0.02) & lifted_clean       # still up after shear
    return dict(pz0=pz0.mean(), pz_lift=pz_lift.mean(), pz_end=pz_end.mean(),
                lifted_clean=lifted_clean.mean(), held=held.mean())


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 48
    print(f"shear  FRICTION-held   FORM-CLOSURE-held   (N={N}, ±mm/step lateral)")
    for shear in (0.005, 0.010, 0.020, 0.040):
        rf = run(False, N, shear); rc = run(True, N, shear)
        print(f"±{shear*1000:4.0f}   {rf['held']*100:6.1f}%        "
              f"{rc['held']*100:6.1f}%      "
              f"(end z: fric={rf['pz_end']*1000:.0f}  fc={rc['pz_end']*1000:.0f} mm)",
              flush=True)
