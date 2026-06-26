import math
import os
from pathlib import Path

import datasets
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import QED
from tqdm.auto import tqdm

from tasks.common import decode_with_harness, length_from_example, masked_initial_state, prompt_from_example, result_output


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_DIR = Path(os.environ.get("QM9_DATASET_DIR", REPO_ROOT / "data" / "qm9" / "dataset"))
DEFAULT_CACHE_DIR = Path(os.environ.get("QM9_DATA_CACHE", REPO_ROOT / "data" / "qm9" / "cache"))
EVAL_SPLIT_SIZE = 0.05
REWARD_MODE = "quantile_indicator"
REWARD_BETA = 0.0
REWARD_CENTER = 0.5
REWARD_THRESHOLD = 0.5614854131280362
REWARD_QUANTILE = 95.0
SEQ_LEN = 32

_TRAIN_SMILES = None

Chem.rdBase.DisableLog("rdApp.error")


def clean_smiles(text):
    text = text.replace("<bos>", "")
    if "<eos>" in text:
        text = text.split("<eos>", 1)[0]
    for token in ("<pad>", "<mask>", "<unk>"):
        text = text.replace(token, "")
    return text.strip()


def _qed_reward(text):
    smiles = clean_smiles(text)
    if not smiles:
        return 0.0, smiles, ""
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        if mol is None:
            return 0.0, smiles, ""
        return float(QED.qed(mol)), smiles, Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return 0.0, smiles, ""


def _split():
    if DEFAULT_DATASET_DIR.exists():
        dataset = datasets.load_from_disk(str(DEFAULT_DATASET_DIR))
    else:
        DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        dataset = datasets.load_dataset("yairschiff/qm9", cache_dir=str(DEFAULT_CACHE_DIR), split="train")
    split = dataset.train_test_split(test_size=EVAL_SPLIT_SIZE, seed=42)
    return split["train"], split["test"], dataset


def _train_smiles():
    global _TRAIN_SMILES
    if _TRAIN_SMILES is None:
        train_ds, _, _ = _split()
        _TRAIN_SMILES = set(str(x) for x in train_ds["smiles"])
    return _TRAIN_SMILES


def _dataset_qed_threshold(quantile):
    train_ds, _, _ = _split()
    values = []
    for smiles in train_ds["smiles"]:
        qed, _, _ = _qed_reward(smiles)
        if qed > 0:
            values.append(qed)
    if not values:
        raise ValueError("QM9 training split has no valid QED values")
    return float(np.percentile(np.asarray(values, dtype=np.float32), float(quantile)))


def configure(config):
    global REWARD_MODE, REWARD_BETA, REWARD_CENTER, REWARD_THRESHOLD, REWARD_QUANTILE
    reward_cfg = dict(config.get("reward", {}) or {})
    REWARD_MODE = str(reward_cfg.get("mode", REWARD_MODE))
    REWARD_BETA = float(reward_cfg.get("beta", REWARD_BETA))
    REWARD_CENTER = float(reward_cfg.get("center", REWARD_CENTER))
    REWARD_QUANTILE = float(reward_cfg.get("quantile", REWARD_QUANTILE))
    if reward_cfg.get("threshold") is not None:
        REWARD_THRESHOLD = float(reward_cfg["threshold"])
    elif reward_cfg.get("threshold_source") == "dataset_quantile":
        REWARD_THRESHOLD = _dataset_qed_threshold(REWARD_QUANTILE)


def load_examples(split):
    train_ds, val_ds, _ = _split()
    if split == "train":
        ds = train_ds
    elif split == "val":
        ds = val_ds
    else:
        raise KeyError(f"Unknown QM9 split: {split}. Expected 'train' or 'val'.")
    rows = []
    for idx, row in enumerate(ds):
        rows.append(
            {
                "id": "qm9-%s-%d" % (split, idx),
                "prompt": "",
                "answer": row.get("smiles", ""),
                "length": SEQ_LEN,
            }
        )
    return rows


def make_prompt(example):
    return prompt_from_example(example)


def default_length(example):
    return length_from_example(example, SEQ_LEN)


def initial_state(example, harness):
    return masked_initial_state(default_length(example), harness)


def decode_state(state, harness):
    return decode_with_harness(state, harness)


def _raw_qed(text):
    return _qed_reward(text)


def _shape_reward(qed):
    if qed <= 0:
        return 0.0
    if REWARD_MODE == "tail_exp":
        return math.exp(REWARD_BETA * max(0.0, qed - REWARD_THRESHOLD))
    if REWARD_MODE == "quantile_indicator":
        return 1.0 if qed >= REWARD_THRESHOLD else 0.0
    if REWARD_BETA == 0.0:
        return float(qed)
    return math.exp(REWARD_BETA * (float(qed) - REWARD_CENTER))


def reward(example, output):
    del example
    qed, _, _ = _raw_qed(output)
    return _shape_reward(qed)


def reward_result(example, result, harness):
    del harness
    return reward(example, result_output(result))


def reward_state(example, state, harness):
    return reward(example, decode_state(state, harness))


def terminal_accept(example, output):
    del example
    return bool(row_info(None, output)["valid"])


def row_info(example, output):
    del example
    qed, raw, canonical = _raw_qed(output)
    shaped = _shape_reward(qed)
    valid = bool(qed > 0.0 and canonical)
    return {
        "smiles": raw,
        "canonical_smiles": canonical,
        "valid": valid,
        "qed": float(qed),
        "reward": float(shaped),
        "pass95": bool(valid and qed >= REWARD_THRESHOLD),
    }


def collect_rollout_rows(harness, examples, config, rank=0):
    configure(config)
    generation = dict(config.get("generation", {}))
    rollout = dict(config.get("rollout", {}))
    batch_size = int(generation.get("batch_size", 256))
    steps = int(generation.get("steps", 128))
    max_new_tokens = int(generation.get("max_new_tokens", SEQ_LEN))
    snapshots_per_rollout = int(rollout.get("snapshots_per_rollout", 3))
    data_split = str(config.get("data", {}).get("split", "train"))

    rows = []
    starts = range(0, len(examples), batch_size)
    if rank == 0:
        starts = tqdm(starts, desc="QM9 verifier rollouts")
    for start in starts:
        batch = examples[start:start + batch_size]
        final_ids, snapshots, snapshot_steps = harness.generate_batch_with_snapshots(
            len(batch),
            steps=steps,
            snapshots_per_rollout=snapshots_per_rollout,
            max_new_tokens=max_new_tokens,
        )
        final_states = final_ids.detach().cpu().tolist()
        snapshot_states = snapshots.detach().cpu().tolist()
        snapshot_step_values = snapshot_steps.detach().cpu().tolist()
        outputs = [decode_state(state, harness) for state in final_states]

        for idx, (example, output, final_state) in enumerate(zip(batch, outputs, final_states)):
            info = row_info(example, output)
            rollout_id = example.get("id", f"qm9-rollout-{start + idx}") if isinstance(example, dict) else f"qm9-rollout-{start + idx}"
            for snapshot_idx, state in enumerate(snapshot_states[idx]):
                state_ids = [int(token) for token in state]
                row = {
                    "split": data_split,
                    "example": example,
                    "rollout_id": rollout_id,
                    "snapshot_index": int(snapshot_idx),
                    "snapshot_step": int(snapshot_step_values[idx][snapshot_idx]),
                    "state_ids": state_ids,
                    "state": state_ids,
                    "final_state_ids": [int(token) for token in final_state],
                    "output": output,
                    "sample": output,
                    "reward": float(info["reward"]),
                    "stats": {
                        "steps": steps,
                        "snapshot_step": int(snapshot_step_values[idx][snapshot_idx]),
                    },
                }
                row.update(info)
                rows.append(row)
    return rows


def serialize_rollout_rows(rows, config=None):
    del config
    state_ids = torch.tensor([row["state_ids"] for row in rows], dtype=torch.long)
    qeds = torch.tensor([float(row.get("qed", 0.0)) for row in rows], dtype=torch.float32)
    shaped_rewards = torch.tensor([float(row.get("reward", 0.0)) for row in rows], dtype=torch.float32)
    snapshot_indices = torch.tensor([int(row.get("snapshot_index", 0)) for row in rows], dtype=torch.long)
    snapshot_steps = torch.tensor([int(row.get("snapshot_step", 0)) for row in rows], dtype=torch.long)
    valid = torch.tensor([bool(row.get("valid", False)) for row in rows], dtype=torch.bool)
    pass95 = torch.tensor([bool(row.get("pass95", False)) for row in rows], dtype=torch.bool)
    return {
        "format": "qm9_rollout_v1",
        "max_length": int(state_ids.shape[1]) if state_ids.ndim == 2 else SEQ_LEN,
        "state_ids": state_ids,
        "rewards": qeds,
        "qeds": qeds,
        "shaped_rewards": shaped_rewards,
        "valid": valid,
        "pass95": pass95,
        "outputs": [str(row.get("output", "")) for row in rows],
        "splits": [str(row.get("split", "train")) for row in rows],
        "rollout_ids": [str(row.get("rollout_id", f"qm9-rollout-{idx}")) for idx, row in enumerate(rows)],
        "snapshot_indices": snapshot_indices,
        "snapshot_steps": snapshot_steps,
        "reward_mode": REWARD_MODE,
        "reward_threshold": float(REWARD_THRESHOLD),
        "num_rows": len(rows),
    }


def _stat_mean(rows, key):
    values = []
    for row in rows:
        stats = row.get("stats", {}) or {}
        if key in stats:
            values.append(float(stats.get(key, 0.0) or 0.0))
    return float(np.mean(values)) if values else 0.0


def _stat_median(rows, key):
    values = []
    for row in rows:
        stats = row.get("stats", {}) or {}
        if key in stats:
            values.append(float(stats.get(key, 0.0) or 0.0))
    return float(np.median(values)) if values else 0.0


def _step_values(rows):
    values = []
    for row in rows:
        stats = row.get("stats", {}) or {}
        if "steps" in stats:
            values.append(float(stats.get("steps", 0.0) or 0.0))
            continue
        total = 0.0
        for key in ("forward", "backward", "switch_up", "switch_down", "leaf_backtrack", "force_forward"):
            total += float(stats.get(key, 0.0) or 0.0)
        if total <= 0 and "nfe" in stats:
            total = float(stats.get("nfe", 0.0) or 0.0)
        values.append(total)
    return values


def metrics(rows):
    if not rows:
        return {
            "num_samples": 0,
            "valid_ratio": 0.0,
            "constraint_violation_ratio": 0.0,
            "qed_mean": 0.0,
            "reward_mean": 0.0,
        }

    valid = []
    invalid = []
    qeds = []
    rewards = []
    pass95 = 0
    for row in rows:
        output = row.get("output", "")
        qed = row.get("qed")
        canonical = row.get("canonical_smiles")
        if qed is None or canonical is None:
            qed, _, canonical = _raw_qed(output)
        qed = float(qed or 0.0)
        shaped = _shape_reward(qed)
        rewards.append(shaped)
        if qed > 0.0 and canonical:
            valid.append(canonical)
            qeds.append(qed)
            if qed >= REWARD_THRESHOLD:
                pass95 += 1
        else:
            invalid.append(output)

    train_smiles = _train_smiles()
    unique_valid = len(set(valid))
    novel_valid = len(set(valid) - train_smiles)
    total = len(rows)
    valid_count = len(valid)
    step_values = _step_values(rows)
    return {
        "num_samples": total,
        "valid": valid_count,
        "invalid": len(invalid),
        "valid_ratio": valid_count / total if total else 0.0,
        "constraint_violation_ratio": len(invalid) / total if total else 0.0,
        "qed_mean": float(np.mean(qeds)) if qeds else 0.0,
        "qed_median": float(np.median(qeds)) if qeds else 0.0,
        "pass95": pass95 / total if total else 0.0,
        "reward_mean": float(np.mean(rewards)) if rewards else 0.0,
        "reward_median": float(np.median(rewards)) if rewards else 0.0,
        "reward_mode": REWARD_MODE,
        "reward_threshold": REWARD_THRESHOLD,
        "unique_valid": unique_valid,
        "unique_ratio_of_valid": unique_valid / valid_count if valid_count else 0.0,
        "novel_valid": novel_valid,
        "novel_ratio_of_valid": novel_valid / valid_count if valid_count else 0.0,
        "nfe_mean": _stat_mean(rows, "nfe"),
        "nfe_median": _stat_median(rows, "nfe"),
        "steps_mean": float(np.mean(step_values)) if step_values else 0.0,
        "steps_median": float(np.median(step_values)) if step_values else 0.0,
        "forward_mean": _stat_mean(rows, "forward"),
        "backward_mean": _stat_mean(rows, "backward"),
        "switch_up_mean": _stat_mean(rows, "switch_up"),
        "switch_down_mean": _stat_mean(rows, "switch_down"),
        "forced_ratio": _stat_mean(rows, "forced"),
    }
