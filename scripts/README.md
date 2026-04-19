# CLI wrappers

Thin shell scripts over the Python entry points. See `../PLAN.md` for the full pipeline.

Each script picks the right Python env:

| Script | Env | Purpose |
|---|---|---|
| `teleop.sh` | Isaac Lab | Record keyboard demos → HDF5 |
| `mimic.sh` | Isaac Lab | Augment demos via curobo |
| `convert.sh` | training | HDF5 → RLDS TFDS |
| `train.sh` | training | OpenVLA LoRA fine-tune |
| `eval.sh` | Isaac Lab | Sim rollouts of fine-tuned checkpoint |

Not yet implemented — stubs will land alongside each phase.
