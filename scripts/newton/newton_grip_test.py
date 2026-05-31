#!/usr/bin/env python3
"""
Minimal Newton (SolverMuJoCo, GPU) grasp-hold probe.

Isolates the ONE question the whole engine investigation is about:
can Newton's MuJoCo-Warp convex solver HOLD a thin round stem in a
horizontal parallel-jaw friction pinch against gravity — the exact
axial slip that PhysX (TGS, every contact lever exhausted) could not
sustain, but standalone MuJoCo could (see memory: mujoco-grasp-works,
newton-the-convex-solver-with-isaac).

No arm / IK / URDF / transport — just the contact physics:
  world --prismatic Z--> hand_base
  hand_base --prismatic X--> finger_left   (closes toward centre)
  hand_base --prismatic X--> finger_right
  free vertical stem cylinder (r=3mm) standing between the jaws.

Phases: settle (jaws pinch) -> lift hand -> hold. We compare the stem's
world-z against the hand's lift. If the stem rises with the hand it is
HELD; if it stays behind / falls it SLIPPED.

Run (in the isolated newton probe venv):
    ~/newton-probe/bin/python scripts/newton_grip_test.py
"""
from __future__ import annotations

import numpy as np
import warp as wp
import newton
from newton import JointTargetMode

# ── stem geometry (matches the round stem that slips in PhysX) ───────────
STEM_R      = 0.003     # 3 mm radius round stem
STEM_HALF   = 0.020     # 40 mm tall
STEM_MASS   = 0.002     # 2 g (as in the MuJoCo test)
STEM_MU     = 0.8       # tangential friction (MuJoCo test used 0.8)
STEM_MU_T   = 0.05      # torsional  (condim=6 analog)
STEM_MU_R   = 0.01      # rolling    (condim=6 analog)

# ── jaw geometry ─────────────────────────────────────────────────────────
# Fingers START near-touching the 3mm-radius stem (faces at ±3.5mm, 0.5mm
# clearance) so there is NO slam impulse; then we RAMP a gentle symmetric
# clamp inward so the free-standing rod is not knocked over.
JAW_HX, JAW_HY, JAW_HZ = 0.004, 0.006, 0.010
GAP_OPEN    = 0.0075    # finger-centre |x| at start -> inner face at ∓3.5mm (0.5mm clearance)
CLAMP_DELTA = 0.0011    # ramp faces inward to ∓2.4mm (~0.6mm penetration -> clamp force)
HAND_Z0     = 0.060     # hand_base start height
JAW_Z_OFF   = -0.035    # finger z offset below hand_base so jaw faces sit at stem mid (~0.025)

LIFT_AMT    = 0.040     # raise hand 40 mm (clears a 45mm vial in the real task)

# ── control gains (gentle finger clamp so a 2g rod is not ejected) ─────────
FINGER_KE, FINGER_KD = 500.0, 5.0
LIFT_KE,   LIFT_KD   = 5000.0, 80.0

# ── timing ─────────────────────────────────────────────────────────────────
DT          = 0.0005
SETTLE_STEPS = 400     # let the rod settle, jaws open-touching
CLAMP_STEPS  = 1200    # slow symmetric clamp ramp
LIFT_STEPS   = 1200
HOLD_STEPS   = 4000     # long hold (2.0s) to rule out PhysX-style slow axial creep


def build():
    b = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)

    jaw_cfg  = newton.ModelBuilder.ShapeConfig(mu=STEM_MU, mu_torsional=STEM_MU_T,
                                               mu_rolling=STEM_MU_R, density=500.0)
    stem_cfg = newton.ModelBuilder.ShapeConfig(mu=STEM_MU, mu_torsional=STEM_MU_T,
                                               mu_rolling=STEM_MU_R, density=0.0)

    # hand_base: prismatic Z off world
    hand = b.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, HAND_Z0)), mass=0.5, label="hand")
    jz = b.add_joint_prismatic(parent=-1, child=hand, axis=newton.Axis.Z,
                               target_pos=0.0, target_ke=LIFT_KE, target_kd=LIFT_KD,
                               actuator_mode=JointTargetMode.POSITION)

    # fingers: prismatic X off hand_base. Left starts at -GAP_OPEN, right at +GAP_OPEN.
    # Each finger joint's parent_xform anchors q=0 at the finger's rest offset
    # from the hand (±GAP_OPEN in x, JAW_Z_OFF in z). The prismatic X coordinate
    # then moves the finger inward/outward from that rest pose. Without this the
    # joint-zero would place the finger origin AT the hand origin.
    fl = b.add_link(xform=wp.transform(wp.vec3(-GAP_OPEN, 0.0, HAND_Z0 + JAW_Z_OFF)), mass=0.02, label="fl")
    jfl = b.add_joint_prismatic(parent=hand, child=fl, axis=newton.Axis.X,
                                parent_xform=wp.transform(wp.vec3(-GAP_OPEN, 0.0, JAW_Z_OFF)),
                                target_pos=0.0, target_ke=FINGER_KE, target_kd=FINGER_KD,
                                actuator_mode=JointTargetMode.POSITION)
    b.add_shape_box(fl, hx=JAW_HX, hy=JAW_HY, hz=JAW_HZ, cfg=jaw_cfg, color=wp.vec3(0.2, 0.3, 0.8))

    fr = b.add_link(xform=wp.transform(wp.vec3(GAP_OPEN, 0.0, HAND_Z0 + JAW_Z_OFF)), mass=0.02, label="fr")
    jfr = b.add_joint_prismatic(parent=hand, child=fr, axis=newton.Axis.X,
                                parent_xform=wp.transform(wp.vec3(+GAP_OPEN, 0.0, JAW_Z_OFF)),
                                target_pos=0.0, target_ke=FINGER_KE, target_kd=FINGER_KD,
                                actuator_mode=JointTargetMode.POSITION)
    b.add_shape_box(fr, hx=JAW_HX, hy=JAW_HY, hz=JAW_HZ, cfg=jaw_cfg, color=wp.vec3(0.2, 0.3, 0.8))

    # stem: free vertical cylinder standing on the ground, centred between jaws
    stem = b.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, STEM_HALF), wp.quat_identity()),
                      mass=STEM_MASS, label="stem")
    jfree = b.add_joint_free(child=stem)
    # cylinder default axis is local Z -> already vertical. centre at body origin.
    b.add_shape_cylinder(stem, radius=STEM_R, half_height=STEM_HALF, cfg=stem_cfg,
                         color=wp.vec3(0.35, 0.6, 0.2))

    # group joints into articulations (hand assembly + free stem)
    b.add_articulation([jz, jfl, jfr], label="hand_assembly")
    b.add_articulation([jfree], label="stem")

    b.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))

    model = b.finalize()
    return model, dict(jz=jz, jfl=jfl, jfr=jfr, hand=hand, fl=fl, fr=fr, stem=stem)


def main():
    wp.init()
    dev = wp.get_device()
    print(f"device: {dev} | cuda: {dev.is_cuda}")
    model, idx = build()
    print(f"bodies={model.body_count} joints={model.joint_count} dofs={model.joint_dof_count} "
          f"shapes={model.shape_count}")

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    # match the MuJoCo working-test solver settings as closely as the API allows
    solver = newton.solvers.SolverMuJoCo(
        model,
        integrator="implicitfast",
        cone="elliptic",          # better friction cone (MuJoCo default is pyramidal)
        iterations=100,
        ls_iterations=50,
        impratio=10.0,            # favour friction stiffness
    )

    def dump(tag, st):
        q = st.body_q.numpy()
        for nm, i in [("hand", idx["hand"]), ("fl", idx["fl"]), ("fr", idx["fr"]), ("stem", idx["stem"])]:
            print(f"    {tag} {nm:4s} pos=({q[i][0]*1000:7.2f},{q[i][1]*1000:7.2f},{q[i][2]*1000:7.2f}) mm")
    dump("INIT", state_0)

    # joint dof order follows creation: [hand_z, fl_x, fr_x, stem_free(6)]
    # find the target array on the control object
    tgt = control.joint_target_pos if hasattr(control, "joint_target_pos") else control.joint_target
    tgt_np = tgt.numpy()
    ndof = tgt_np.shape[-1]
    print(f"control target dofs: {ndof}  (initial {np.round(tgt_np,4).tolist()})")

    # dof indices for the 3 actuated prismatic joints (free joint dofs come after)
    HAND_DOF, FL_DOF, FR_DOF = 0, 1, 2

    def set_targets(hand_z, fl_x, fr_x):
        v = tgt.numpy()
        v[..., HAND_DOF] = hand_z
        v[..., FL_DOF]   = fl_x
        v[..., FR_DOF]   = fr_x
        tgt.assign(v)

    def body_z(state, body_i):
        q = state.body_q.numpy()        # (nbody, 7) transform
        return float(q[body_i][2])

    # manual double-buffered state swap
    s0, s1 = state_0, state_1

    def run(n):
        nonlocal s0, s1
        for _ in range(n):
            s0.clear_forces()
            solver.step(s0, s1, control, None, DT)
            s0, s1 = s1, s0

    # NOTE: the world->hand prismatic-Z zero is at the world origin, so the hand
    # Z target is ABSOLUTE world height (0.060 holds at grip height, +LIFT lifts).
    # finger-X targets are RELATIVE displacements from their rest offset.
    HAND_Z_HOLD = HAND_Z0

    # ── Phase 0a: settle — jaws open-touching, rod stabilises ──────────────
    set_targets(HAND_Z_HOLD, 0.0, 0.0)
    run(SETTLE_STEPS)

    # ── Phase 0b: slow symmetric clamp ramp ────────────────────────────────
    #   fl moves +x by CLAMP_DELTA, fr moves -x by CLAMP_DELTA -> faces pinch the stem
    for i in range(CLAMP_STEPS):
        d = CLAMP_DELTA * (i + 1) / CLAMP_STEPS
        set_targets(HAND_Z_HOLD, +d, -d)
        s0.clear_forces()
        solver.step(s0, s1, control, None, DT)
        s0, s1 = s1, s0
    z_stem_pre = body_z(s0, idx["stem"])
    z_hand_pre = body_z(s0, idx["hand"])
    dump("CLAMP", s0)
    print(f"[after clamp ] stem_z={z_stem_pre*1000:7.2f} mm  hand_z={z_hand_pre*1000:7.2f} mm")

    # ── Phase 1: lift the hand ─────────────────────────────────────────────
    for i in range(LIFT_STEPS):
        set_targets(HAND_Z_HOLD + LIFT_AMT * (i + 1) / LIFT_STEPS,
                    +CLAMP_DELTA, -CLAMP_DELTA)
        s0.clear_forces()
        solver.step(s0, s1, control, None, DT)
        s0, s1 = s1, s0
    z_stem_lift = body_z(s0, idx["stem"])
    z_hand_lift = body_z(s0, idx["hand"])
    print(f"[after lift  ] stem_z={z_stem_lift*1000:7.2f} mm  hand_z={z_hand_lift*1000:7.2f} mm  "
          f"(commanded +{LIFT_AMT*1000:.0f} mm)")

    # ── Phase 2: hold at height ────────────────────────────────────────────
    run(HOLD_STEPS)
    z_stem_hold = body_z(s0, idx["stem"])
    z_hand_hold = body_z(s0, idx["hand"])
    print(f"[after hold  ] stem_z={z_stem_hold*1000:7.2f} mm  hand_z={z_hand_hold*1000:7.2f} mm")

    # ── verdict ────────────────────────────────────────────────────────────
    stem_rise = z_stem_hold - z_stem_pre
    hand_rise = z_hand_hold - z_hand_pre
    # held if the stem followed the hand up (within slip tolerance) AND didn't fall back to ground
    held = (stem_rise > 0.6 * hand_rise) and (z_stem_hold > z_stem_pre + 0.010)
    print("\n── Newton (SolverMuJoCo, GPU) grasp-hold ──")
    print(f"  stem rise : {stem_rise*1000:7.2f} mm")
    print(f"  hand rise : {hand_rise*1000:7.2f} mm")
    print(f"  slip      : {(hand_rise - stem_rise)*1000:7.2f} mm")
    print(f"  VERDICT   : {'HELD ✓' if held else 'SLIPPED ✗'}")
    return 0 if held else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
