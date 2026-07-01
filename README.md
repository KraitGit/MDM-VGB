# MDM-VGB

Paper-code release for **VGB for Masked Diffusion Model: Efficient Test-time
Scaling for Reward Satisfaction and Sample Editing**.

Status: Bug-fix (Dyck / QM9) in progress (Jul 1. ~).

## Layout

- `configs/main.yaml`: shared defaults using paper notation (`N`, `L_f`, `L_b`, `K`, `B`, `chi`).
- `configs/<task>/<task>_{base_model_training,inference,rollout,verifier_training}.yaml`: task-specific configs for each run stage.
- `base_model_training.py`: train a task base model if necessary.
- `base_model_rollout.py`: collect base-model rollouts.
- `verifier_training.py`: train a task verifier from rollout data.
- `inference.py`: run `Base`, `BoN`, `VGR`, `VGB`, or `VGB-Momentum` and print paper metrics.
- `src/algorithms/`: shared VGR/VGB/VGB-Momentum logic, including MDM-style block proposals.
- `src/tasks/`: task-specific models, harnesses, verifiers, and metrics.

## Pipeline Stages

Use these stages when reproducing a task from scratch:

| Task | Base model | `base_model_training.py` | `base_model_rollout.py` | `verifier_training.py` |
| --- | --- | --- | --- | --- |
| `dyck` | trained in this repo | required | required | required |
| `sudoku` | trained in this repo | required | uses heuristic process verifier | not needed |
| `qm9` | trained in this repo | required | required | required |
| `dna_deepstarr` | pretrained D3LM | not used | required | required |
| `protein_scaffold` | pretrained EvoDiff OADM | not used | required | required |
| `letter` | pretrained Qwen diffusion MDLM | not used | uses heuristic process verifier | not needed |

`base_model_rollout.py` writes rollout snapshots for learned-verifier tasks.
Tasks with task-provided heuristic process verifiers can run inference directly
after the base model/assets are available.

## Setup

Install the repo once in editable mode so the top-level scripts can import the
`src` packages without runtime path injection.

```bash
pip install -e .
```

## Usage

```bash
python base_model_training.py --task dyck
python base_model_rollout.py --task dyck
python verifier_training.py --task dyck
python inference.py --task dyck --algorithm VGB
```

For multi-GPU rollout, verifier training, and inference, use standalone
`torchrun` commands as below:

```bash
torchrun --standalone --nproc_per_node=2 base_model_rollout.py --task dyck
torchrun --standalone --nproc_per_node=2 verifier_training.py --task dyck
torchrun --standalone --nproc_per_node=2 inference.py --task dyck --algorithm VGB
```

`base_model_rollout.py`, `verifier_training.py`, and `inference.py` default to
`qm9`; `base_model_training.py` defaults to `sudoku`. Pass `--task <name>` to
use another config folder.
Set result paths in yaml with `output`. In inference configs, `{algorithm}` is
expanded to names like `base`, `bon`, `vgb`, and `vgb_momentum`.

For applicable tasks, pretrained models and datasets are downloaded through
Hugging Face or external links:

```bash
python prepare_assets.py --tasks qm9 letter dna_deepstarr
```

For multi-GPU rollout or inference, use standard torchrun and select GPUs with
`CUDA_VISIBLE_DEVICES`. Rollout ranks are merged into the requested `.pt`
artifact; inference ranks are merged into the requested JSONL results file.

```bash
CUDA_VISIBLE_DEVICES=0,3,4 torchrun --standalone --nproc_per_node=2 base_model_rollout.py
CUDA_VISIBLE_DEVICES=0,3,4 torchrun --standalone --nproc_per_node=3 inference.py --task dyck --algorithm VGB
```

## License

This code is released under the MIT License.
