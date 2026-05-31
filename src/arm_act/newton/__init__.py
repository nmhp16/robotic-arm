"""Newton (MuJoCo-Warp convex solver) demo-collection backend for arm_act.

Route (a): a standalone Newton oracle that generates vial->vial demos with the
high-friction tip-pad grasp (100% success, deployable — see memory
newton-the-convex-solver-with-isaac) and writes them in the EXACT HDF5 schema the
existing Isaac Lab pipeline (mimic annotate -> LeRobot -> ACT/SmolVLA) consumes.

Newton holds the round-stem friction grasp that Isaac/PhysX could not, so demos
are generated here while the rest of the pipeline is unchanged. Because PhysX
cannot reproduce the grasp, success is also measured in Newton (eval here), with
the real robot (soft high-friction fingertip pads) as the deployment target.

Runs under the ~/newton-probe venv (newton, mujoco-warp, warp, h5py, pyyaml).
"""
