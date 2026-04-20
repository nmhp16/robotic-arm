"""Drive the UR10 pick-and-place env with Claude Opus 4.7 via the claude CLI.

Each step the current table_cam frame is saved to disk, passed to
``claude -p`` with the task instruction and current proprio state, and the
response is parsed back into a 7-D Delta-pose + gripper action.

This is a demo baseline, not a real policy. Claude is a language model, not
a VLA; expect reasoning that is directionally correct but not precise
enough to grasp reliably. Useful as a before/after comparison against a
fine-tuned OpenVLA checkpoint.

    ./scripts/opus.sh --steps 50 --out media/opus_rollout.mp4
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

from isaaclab.app import AppLauncher


DEFAULT_INSTRUCTION = "pick up the blue cube and place it on the green pad"

SYSTEM_PROMPT = """You are driving a robot arm in a simulation. Each step \
you receive a camera image (third-person) and the current end-effector \
pose. You output a small Delta-pose action that moves the arm toward \
completing the task.

Output STRICT JSON on ONE LINE, nothing else:
{"dx": float, "dy": float, "dz": float, "droll": float, "dpitch": float, "dyaw": float, "gripper": "open" | "close", "reason": "<one short sentence>"}

Units: dx/dy/dz in meters, clamp to [-0.05, 0.05]. Rotations in radians, \
clamp to [-0.2, 0.2]. Gripper open = release suction, close = engage. \
Positive dz lifts the TCP; positive dx is the robot's forward direction. \
Reason must be <= 15 words."""


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--instruction", type=str, default=DEFAULT_INSTRUCTION)
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("media/opus_rollout.mp4"))
    p.add_argument("--fps", type=int, default=4)
    p.add_argument("--timeout", type=float, default=60.0)
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    if shutil.which("claude") is None:
        print("error: `claude` CLI not on PATH. Install Claude Code and run `claude login`.", file=sys.stderr)
        return 2

    app = AppLauncher(headless=True, enable_cameras=True).app

    try:
        import gymnasium as gym
        import numpy as np
        import torch
        from PIL import Image

        import arm_vla.tasks.ur10_pick_place  # noqa: F401  registers gym id
        from arm_vla.tasks.ur10_pick_place.pick_place_ur10_env_cfg import UR10PickPlaceEnvCfg

        cfg = UR10PickPlaceEnvCfg()
        cfg.scene.num_envs = 1
        env = gym.make("Isaac-PickPlace-UR10-IK-Rel-v0", cfg=cfg)
        obs, _ = env.reset()
        device = env.unwrapped.device

        frames = []
        tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix="opus_frames_"))
        t0 = time.time()

        for step in range(args.steps):
            table = obs["policy"]["table_cam"][0].cpu().numpy().astype("uint8")
            wrist = obs["policy"]["wrist_cam"][0].cpu().numpy().astype("uint8")
            eef_pos = obs["policy"]["eef_pos"][0].cpu().numpy()
            cube_pos = obs["policy"]["cube_pos"][0].cpu().numpy()
            target_pos = obs["policy"]["target_pos"][0].cpu().numpy()
            gripper = float(obs["policy"]["gripper_pos"][0, 0].cpu())

            frame_path = tmp_dir / f"step_{step:03d}.png"
            Image.fromarray(table).save(frame_path)

            prompt = _build_prompt(
                args.instruction,
                frame_path,
                eef_pos,
                cube_pos,
                target_pos,
                gripper,
            )
            action_dict = _call_claude(prompt, timeout=args.timeout)
            action = _action_from_dict(action_dict)

            print(
                f"step {step:>3d} | eef={eef_pos.round(3).tolist()} "
                f"cube={cube_pos.round(3).tolist()} "
                f"action=[{action[0]:+.3f},{action[1]:+.3f},{action[2]:+.3f},"
                f"{action[6]:+.1f}] reason={action_dict.get('reason','')[:60]}",
                flush=True,
            )

            action_t = torch.as_tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
            obs, _, terminated, truncated, info = env.step(action_t)

            frames.append(np.concatenate([table, wrist], axis=1))

            if bool(terminated[0]) or bool(truncated[0]):
                print(f"episode ended at step {step + 1}", flush=True)
                break

        # Save video BEFORE env.close() — close can crash on the Isaac Lab
        # SurfaceGripper cleanup bug and we don't want to lose the rollout.
        _save_video(args.out, frames, args.fps)
        dt = time.time() - t0
        print(f"\ndone in {dt:.1f} s ({dt / max(1, len(frames)):.1f} s/step)", flush=True)
        try:
            env.close()
        except Exception as close_exc:
            print(f"env.close() raised (ignored): {close_exc}", flush=True)

    except Exception:
        traceback.print_exc()
        return 1
    finally:
        app.close()

    return 0


def _build_prompt(
    instruction: str,
    frame_path: pathlib.Path,
    eef_pos,
    cube_pos,
    target_pos,
    gripper: float,
) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Task: {instruction}\n"
        f"Image: open and read this file with the Read tool -> {frame_path}\n"
        f"TCP position (x,y,z meters): {eef_pos.tolist()}\n"
        f"Cube position: {cube_pos.tolist()}\n"
        f"Target pad position: {target_pos.tolist()}\n"
        f"Gripper state: {'closed' if gripper >= 0.5 else 'open'}\n\n"
        "Respond with JSON only."
    )


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

    text = envelope.get("result") or envelope.get("response") or ""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return {"error": "no JSON in response", "raw": text[:120]}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {"error": "bad JSON in response", "raw": text[start : end + 1][:120]}


def _action_from_dict(d: dict):
    import numpy as np

    def _f(key: str, lo: float, hi: float) -> float:
        v = d.get(key, 0.0)
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = 0.0
        return float(np.clip(v, lo, hi))

    gripper_cmd = -1.0
    gripper_val = d.get("gripper", "open")
    if isinstance(gripper_val, str) and gripper_val.strip().lower().startswith("close"):
        gripper_cmd = 1.0
    elif isinstance(gripper_val, (int, float)) and gripper_val > 0:
        gripper_cmd = 1.0

    return np.array(
        [
            _f("dx", -0.05, 0.05),
            _f("dy", -0.05, 0.05),
            _f("dz", -0.05, 0.05),
            _f("droll", -0.2, 0.2),
            _f("dpitch", -0.2, 0.2),
            _f("dyaw", -0.2, 0.2),
            gripper_cmd,
        ],
        dtype="float32",
    )


def _save_video(out_path: pathlib.Path, frames: list, fps: int) -> None:
    if not frames:
        print("no frames to save", file=sys.stderr)
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio
        imageio.mimsave(str(out_path), frames, fps=fps)
        print(f"wrote {out_path} ({len(frames)} frames @ {fps} fps)")
    except ImportError:
        from PIL import Image
        stem = out_path.with_suffix("")
        stem.mkdir(parents=True, exist_ok=True)
        for i, fr in enumerate(frames):
            Image.fromarray(fr).save(stem / f"{i:04d}.png")
        print(f"wrote {len(frames)} PNGs under {stem}/ (imageio missing)")


if __name__ == "__main__":
    sys.exit(main())
