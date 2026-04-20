"""Closed-loop Opus planner for the UR10 pick-and-place env.

Opus plans at the primitive level (move_above, descend_to, grip, lift_by,
release). Python executes each primitive via the env's IK-relative action
space. After the plan runs, we check success and, if it failed, ask Opus
to revise the plan based on what actually happened. Up to N retries.

This is LLM-as-task-planner, not LLM-as-controller. Opus is called a
handful of times per episode, not every sim step.

    ./scripts/opus_plan.sh --max-attempts 3 --out media/opus_planner.mp4
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from typing import Any

from isaaclab.app import AppLauncher


INSTRUCTION = "pick up the blue cube and place it on the green pad"

PRIMITIVES_DOC = """You plan robot primitives. Available primitives:

  move_above(xyz)    Move TCP to world-frame xyz (m). Use for travel /
                     hovering. ALWAYS use a SAFE z (>= 0.12 m) — never
                     move_above directly onto an object.
  descend_to(z)      Lower TCP to world z (m), keeping xy fixed. Use this
                     ONLY after grip() has succeeded, to lower the held
                     object near a target. DO NOT call before grip().
  lift_by(dz)        Raise TCP by dz meters (dz > 0).
  grip()             Engage the suction. Auto-aligns xy to the cube and
                     auto-descends to contact. Requires TCP already
                     hovering ABOVE the cube at a safe height (z >= 0.12).
  release()          Release the suction.

The table is at z = 0.0. The cube center is at ~z = 0.02. Safe hover is
z >= 0.12.

Recommended pattern for pick-and-place (6 steps):
  1. move_above(cube_x, cube_y, 0.15)
  2. grip()                              # handles descent + suction
  3. lift_by(0.10)
  4. move_above(target_x, target_y, 0.15)
  5. descend_to(target_z + 0.06)         # lower held cube near target
  6. release()

OUTPUT a JSON object on ONE LINE:
{"plan": [{"fn": "<name>", "args": {...}}, ...], "reason": "<one sentence>"}

Omit "args" entirely for grip and release.
"""


@dataclass
class PlanStep:
    fn: str
    args: dict[str, Any]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--instruction", type=str, default=INSTRUCTION)
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("media/opus_planner.mp4"))
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--timeout", type=float, default=60.0)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if shutil.which("claude") is None:
        print("error: `claude` CLI not on PATH.", file=sys.stderr)
        return 2

    app = AppLauncher(headless=True, enable_cameras=True).app

    try:
        import gymnasium as gym
        import numpy as np
        import torch
        from PIL import Image

        import arm_vla.tasks.ur10_pick_place  # noqa: F401
        from arm_vla.tasks.ur10_pick_place.pick_place_ur10_env_cfg import UR10PickPlaceEnvCfg

        cfg = UR10PickPlaceEnvCfg()
        cfg.scene.num_envs = 1
        env = gym.make("Isaac-PickPlace-UR10-IK-Rel-v0", cfg=cfg)
        obs, _ = env.reset()
        device = env.unwrapped.device

        tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix="opus_plan_"))
        frames: list = []
        attempt_logs: list[dict] = []

        success = False
        plan_history: list[dict] = []

        for attempt in range(1, args.max_attempts + 1):
            state = _state_summary(obs)
            frame_path = tmp_dir / f"attempt_{attempt}_start.png"
            _save_table_frame(obs, frame_path)

            print(f"\n=== attempt {attempt}/{args.max_attempts} ===", flush=True)
            plan_obj = _ask_opus_for_plan(
                args.instruction,
                frame_path,
                state,
                plan_history,
                timeout=args.timeout,
            )
            if "error" in plan_obj:
                print(f"  planning failed: {plan_obj['error']}", flush=True)
                break

            plan = _parse_plan(plan_obj.get("plan", []))
            print(f"  plan ({len(plan)} steps): {plan_obj.get('reason', '')[:80]}", flush=True)
            for i, step in enumerate(plan):
                print(f"    {i + 1}. {step.fn}({step.args})", flush=True)

            gripper_closed = False
            obs, gripper_closed, exec_log = _execute_plan(
                env, obs, device, plan, gripper_closed, frames=frames
            )

            final_state = _state_summary(obs)
            success = _is_placed(obs)
            print(f"  result: {'SUCCESS' if success else 'FAIL'}  ({len(exec_log)} primitives run)", flush=True)
            print(f"  final state: {final_state}", flush=True)

            attempt_logs.append({
                "attempt": attempt,
                "plan": [{"fn": s.fn, "args": s.args} for s in plan],
                "reason": plan_obj.get("reason", ""),
                "exec_log": exec_log,
                "final_state": final_state,
                "success": success,
            })

            if success:
                break

            plan_history.append({
                "attempt": attempt,
                "plan": [{"fn": s.fn, "args": s.args} for s in plan],
                "outcome": {
                    "succeeded": False,
                    "final_state": final_state,
                    "notes": [log.get("result") for log in exec_log],
                },
            })

        _save_video(args.out, frames, args.fps)
        summary_path = args.out.with_suffix(".json")
        with open(summary_path, "w") as f:
            json.dump(
                {"success": success, "attempts": attempt_logs, "total_frames": len(frames)},
                f,
                indent=2,
            )
        print(f"\nsummary → {summary_path}", flush=True)
        print(f"final: {'SUCCESS' if success else 'FAIL'}", flush=True)

        try:
            env.close()
        except Exception:
            pass

    except Exception:
        traceback.print_exc()
        return 1
    finally:
        app.close()

    return 0


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #

def _execute_plan(env, obs, device, plan: list[PlanStep], gripper_closed: bool, frames: list):
    """Run each primitive in sequence. Returns final obs + gripper state + log."""
    import numpy as np

    log: list[dict] = []
    for step in plan:
        try:
            if step.fn == "move_above":
                xyz = np.asarray(step.args["xyz"], dtype="float32")
                obs, ok = _move_to_xyz(env, obs, device, xyz, gripper_closed, frames)
            elif step.fn == "descend_to":
                z = float(step.args["z"])
                obs, ok = _descend_to_z(env, obs, device, z, gripper_closed, frames)
            elif step.fn == "lift_by":
                dz = float(step.args["dz"])
                eef = obs["policy"]["eef_pos"][0].cpu().numpy()
                obs, ok = _descend_to_z(env, obs, device, float(eef[2] + dz), gripper_closed, frames)
            elif step.fn == "grip":
                obs, ok = _do_grasp(env, obs, device, frames)
                if ok:
                    gripper_closed = True
            elif step.fn == "release":
                obs, ok = _run_gripper(env, obs, device, close=False, frames=frames)
                if ok:
                    gripper_closed = False
            else:
                log.append({"fn": step.fn, "result": f"unknown primitive"})
                continue
            log.append({"fn": step.fn, "args": step.args, "result": "ok" if ok else "timeout"})
        except Exception as e:
            log.append({"fn": step.fn, "args": step.args, "result": f"error: {e}"})
            break
    return obs, gripper_closed, log


def _move_to_xyz(env, obs, device, target_xyz, gripper_closed: bool, frames: list,
                 step_size: float = 0.03, tol: float = 0.01, max_steps: int = 120):
    import numpy as np
    import torch

    for _ in range(max_steps):
        eef = obs["policy"]["eef_pos"][0].cpu().numpy()
        diff = target_xyz - eef
        if float(np.linalg.norm(diff)) < tol:
            return obs, True
        step = np.clip(diff, -step_size, step_size).astype("float32")
        action = _make_action(step, gripper_closed)
        obs = _apply(env, device, action, frames, obs)
    return obs, False


def _descend_to_z(env, obs, device, target_z: float, gripper_closed: bool, frames: list,
                  step_size: float = 0.025, tol: float = 0.005, max_steps: int = 120):
    import numpy as np
    import torch

    for _ in range(max_steps):
        eef = obs["policy"]["eef_pos"][0].cpu().numpy()
        dz = target_z - float(eef[2])
        if abs(dz) < tol:
            return obs, True
        step = np.array([0.0, 0.0, float(np.clip(dz, -step_size, step_size))], dtype="float32")
        action = _make_action(step, gripper_closed)
        obs = _apply(env, device, action, frames, obs)
    return obs, False


def _run_gripper(env, obs, device, close: bool, frames: list, steps: int = 15):
    """Emit close/open command for N steps so the suction state settles."""
    import numpy as np

    for _ in range(steps):
        action = _make_action(np.zeros(3, dtype="float32"), gripper_closed=close)
        obs = _apply(env, device, action, frames, obs)
    return obs, True


def _do_grasp(env, obs, device, frames: list,
              descend_speed: float = 0.004, xy_gain: float = 0.3,
              max_steps: int = 250,
              close_depth: float = 0.015, hold_depth: float = 0.05,
              hold_steps: int = 60):
    """Contact-aware suction grip driven by the wrist-camera depth.

    Per step: read median center-pixel depth from the wrist cam (distance
    to nearest surface along the tool axis). Descend until depth drops
    into the ``close_depth`` window, then hold position while the suction
    commands engagement. Micro-corrects xy using ground truth to keep the
    TCP centered over the cube (this is cheap; the contact trigger itself
    is perception-driven).

    Returns (obs, True) the moment gripper state flips closed.
    """
    import numpy as np

    held = 0
    last_log = -10
    for t in range(max_steps):
        eef = obs["policy"]["eef_pos"][0].cpu().numpy()
        cube = obs["policy"]["cube_pos"][0].cpu().numpy()
        grip_state = float(obs["policy"]["gripper_pos"][0, 0].cpu())
        depth = float(obs["policy"]["wrist_depth"][0, 0].cpu())

        if grip_state >= 0.5:
            print(f"    grip: engaged at step {t} (depth={depth:.3f}, tcp_z={eef[2]:.3f})", flush=True)
            return obs, True

        in_contact_band = depth <= close_depth
        above_contact = depth <= hold_depth

        if in_contact_band:
            held += 1
            z_step = 0.0
        elif above_contact:
            z_step = -descend_speed * 0.3
        else:
            z_step = -descend_speed

        xy_err = cube[:2] - eef[:2]
        xy_step = np.clip(xy_gain * xy_err, -0.01, 0.01)

        step = np.array([xy_step[0], xy_step[1], z_step], dtype="float32")
        action = _make_action(step, gripper_closed=True)
        obs = _apply(env, device, action, frames, obs)

        if t - last_log >= 20:
            print(
                f"    grip t={t:>3d} depth={depth:.3f} tcp_z={eef[2]:.3f} "
                f"cube_z={cube[2]:.3f} state={grip_state:.2f} band={in_contact_band}",
                flush=True,
            )
            last_log = t

        if in_contact_band and held >= hold_steps:
            break

    grip_state = float(obs["policy"]["gripper_pos"][0, 0].cpu())
    print(f"    grip: ended state={grip_state:.2f} (last depth={depth:.3f})", flush=True)
    return obs, grip_state >= 0.5


def _make_action(translation, gripper_closed: bool):
    """Build a 7-D action: 3-D translation, 3-D rotation (zero), 1-D gripper."""
    import numpy as np
    action = np.zeros(7, dtype="float32")
    action[:3] = translation
    action[6] = 1.0 if gripper_closed else -1.0
    return action


def _apply(env, device, action, frames: list, obs):
    import torch
    a = torch.from_numpy(action).to(device).unsqueeze(0)
    obs, *_ = env.step(a)
    table = obs["policy"]["table_cam"][0].cpu().numpy().astype("uint8")
    wrist = obs["policy"]["wrist_cam"][0].cpu().numpy().astype("uint8")
    import numpy as np
    frames.append(np.concatenate([table, wrist], axis=1))
    return obs


# --------------------------------------------------------------------------- #
# State + success
# --------------------------------------------------------------------------- #

def _state_summary(obs) -> dict:
    eef = obs["policy"]["eef_pos"][0].cpu().tolist()
    cube = obs["policy"]["cube_pos"][0].cpu().tolist()
    target = obs["policy"]["target_pos"][0].cpu().tolist()
    grip = float(obs["policy"]["gripper_pos"][0, 0].cpu())
    depth = float(obs["policy"]["wrist_depth"][0, 0].cpu())
    return {
        "tcp_xyz": [round(v, 3) for v in eef],
        "cube_xyz": [round(v, 3) for v in cube],
        "target_xyz": [round(v, 3) for v in target],
        "suction": "closed" if grip >= 0.5 else "open",
        "wrist_depth_m": round(depth, 3),
    }


def _is_placed(obs, xy_tol: float = 0.05, z_tol: float = 0.06) -> bool:
    import numpy as np
    cube = obs["policy"]["cube_pos"][0].cpu().numpy()
    target = obs["policy"]["target_pos"][0].cpu().numpy()
    grip = float(obs["policy"]["gripper_pos"][0, 0].cpu())
    xy = float(np.linalg.norm(cube[:2] - target[:2]))
    dz = float(cube[2] - target[2])
    placed = (xy < xy_tol) and (-0.02 < dz < z_tol)
    released = grip < 0.5
    return placed and released


# --------------------------------------------------------------------------- #
# Opus interaction
# --------------------------------------------------------------------------- #

def _ask_opus_for_plan(instruction: str, frame_path: pathlib.Path, state: dict,
                      history: list[dict], timeout: float) -> dict:
    history_blob = ""
    if history:
        history_blob = "\nPrevious attempts (all failed):\n" + json.dumps(history, indent=2) + "\n"

    prompt = (
        f"{PRIMITIVES_DOC}\n\n"
        f"Task: {instruction}\n"
        f"Open the image with the Read tool: {frame_path}\n"
        f"Current state: {json.dumps(state)}\n"
        f"{history_blob}\n"
        "Return a plan as JSON only, on one line."
    )
    return _call_claude(prompt, timeout)


def _call_claude(prompt: str, timeout: float) -> dict:
    try:
        result = subprocess.run(
            [
                "claude",
                "-p",
                prompt,
                "--output-format",
                "json",
                "--allowedTools",
                "Read",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}

    if result.returncode != 0:
        return {"error": f"exit {result.returncode}"}

    try:
        envelope = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"error": "non-JSON envelope"}

    text = envelope.get("result") or ""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return {"error": "no JSON", "raw": text[:120]}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {"error": "bad JSON", "raw": text[start : end + 1][:120]}


def _parse_plan(plan_list: list) -> list[PlanStep]:
    steps: list[PlanStep] = []
    for item in plan_list:
        if not isinstance(item, dict) or "fn" not in item:
            continue
        fn = str(item["fn"]).strip()
        args = item.get("args", {}) if isinstance(item.get("args"), dict) else {}
        # Opus sometimes emits args as {"x":..., "y":..., "z":...} instead
        # of {"xyz": [...]}. Normalize either form.
        if "xyz" not in args and {"x", "y", "z"}.issubset(args.keys()):
            args = {**args, "xyz": [args["x"], args["y"], args["z"]]}
        steps.append(PlanStep(fn=fn, args=args))
    return steps


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #

def _save_table_frame(obs, path: pathlib.Path) -> None:
    from PIL import Image
    frame = obs["policy"]["table_cam"][0].cpu().numpy().astype("uint8")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(path)


def _save_video(out_path: pathlib.Path, frames: list, fps: int) -> None:
    if not frames:
        print("no frames to save", file=sys.stderr)
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio
        imageio.mimsave(str(out_path), frames, fps=fps)
        print(f"wrote {out_path} ({len(frames)} frames @ {fps} fps)", flush=True)
    except ImportError:
        print("imageio not installed", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
