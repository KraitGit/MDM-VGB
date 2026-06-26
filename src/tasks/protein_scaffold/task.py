import os
import random
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from .structure import clean_sequence
from .structure import esmfold_version
from .structure import low_complexity
from .structure import load_omegafold_model
from .structure import motif_preserved
from .structure import omegafold_version
from .structure import read_cache
from .structure import score_sequence
from tasks.common import decode_with_harness, masked_initial_state, result_output
from utils import read_jsonl


DEFAULT_TASKS = Path("data/evodiff_scaffold/tasks.jsonl")
DEFAULT_CACHE_DIR = Path("data/evodiff_scaffold/score_cache")
DEFAULT_BACKEND = "omegafold"
REWARD_RMSD_SCALE = 1.0
SUCCESS_RMSD = 1.0
EXACT_LEAF_VALUE = False
HANDLES_ROLLOUT_REPEATS = True
_CONFIG = {}
_FOLD_MODEL = None


def configure(config):
    global _CONFIG
    _CONFIG = dict(config or {})


def _fold_model(scoring_cfg):
    global _FOLD_MODEL
    if str(scoring_cfg.get("backend", DEFAULT_BACKEND)) != "omegafold":
        return None
    if not bool(scoring_cfg.get("inprocess", True)):
        return None
    if _FOLD_MODEL is None:
        _FOLD_MODEL = load_omegafold_model(
            device=scoring_cfg.get("device"),
            weights_file=scoring_cfg.get("weights_file"),
            subbatch_size=scoring_cfg.get("subbatch_size"),
            num_cycle=scoring_cfg.get("num_cycle"),
            model_num=int(scoring_cfg.get("omegafold_model", 1)),
        )
    return _FOLD_MODEL


def load_examples(split):
    data_cfg = dict(_CONFIG.get("data", {}))
    path = data_cfg.get("path") or os.environ.get("EVODIFF_SCAFFOLD_TASKS") or DEFAULT_TASKS
    rows = read_jsonl(path)
    if split is not None:
        rows = [row for row in rows if not row.get("split") or row.get("split") == split]
    count = _CONFIG.get("data", {}).get("count")
    if count is None:
        return rows
    count = int(count)
    if count <= len(rows):
        return rows[:count]
    out = []
    for idx in range(count):
        row = dict(rows[idx % len(rows)])
        row["id"] = f"{row.get('id', row.get('task_id', 'protein'))}-{idx:06d}"
        out.append(row)
    return out


def make_prompt(example):
    return example


def default_length(example):
    return int(example.get("length", len(example.get("native_sequence", ""))))


def decode_state(state, harness):
    return decode_with_harness(state, harness)


def _motif_tokens(example, harness):
    token_to_id = getattr(harness, "token_to_id", {})
    motif = {}
    for pos, aa in (example.get("motif_text") or {}).items():
        aa = str(aa).upper()
        if aa in token_to_id:
            motif[int(pos)] = int(token_to_id[aa])
    if motif:
        return motif
    motif_map = example.get("motif") or {}
    return {int(pos): int(token) for pos, token in motif_map.items()}


def initial_state(example, harness):
    state = masked_initial_state(default_length(example), harness)
    for pos, token in _motif_tokens(example, harness).items():
        if 0 <= int(pos) < len(state):
            state[int(pos)] = int(token)
    return state


def locked_positions(example, state, harness):
    del state
    return set(_motif_tokens(example, harness))


def _cache_dir(example, cache_dir=None):
    scoring_cfg = dict(_CONFIG.get("scoring", {}))
    return (
        cache_dir
        or scoring_cfg.get("cache_dir")
        or example.get("score_cache_dir")
        or os.environ.get("EVODIFF_SCAFFOLD_CACHE")
        or DEFAULT_CACHE_DIR
    )


def _backend_version(example, backend, backend_version=None):
    if backend_version:
        return backend_version
    if example.get("backend_version"):
        return example.get("backend_version")
    if backend == "esmfold":
        return esmfold_version()
    if backend == "omegafold":
        return omegafold_version()
    return "unknown"


def cached_score(example, output, cache_dir=None, backend=None, backend_version=None):
    sequence = clean_sequence(output)
    backend = backend or example.get("folding_backend") or DEFAULT_BACKEND
    version = _backend_version(example, backend, backend_version=backend_version)
    return read_cache(_cache_dir(example, cache_dir=cache_dir), sequence, backend=backend, backend_version=version)


def reward_result(example, result, harness=None, cache_dir=None, backend=None, backend_version=None):
    del harness
    output = result_output(result)
    scoring_cfg = dict(_CONFIG.get("scoring", {}))
    cache_dir = cache_dir or scoring_cfg.get("cache_dir")
    backend = backend or scoring_cfg.get("backend")
    backend_version = backend_version or scoring_cfg.get("backend_version")
    record = cached_score(example, output, cache_dir=cache_dir, backend=backend, backend_version=backend_version)
    if record is None:
        if bool(scoring_cfg.get("fold", False)):
            return score_output(
                example,
                output,
                cache_dir=cache_dir,
                folded_dir=scoring_cfg.get("folded_dir"),
                fold=True,
                model=_fold_model(scoring_cfg),
                backend=backend,
                backend_version=backend_version,
                device=scoring_cfg.get("device"),
                chunk_size=scoring_cfg.get("chunk_size"),
            )
        raise FileNotFoundError(
            "missing EvoDiff scaffold score cache for sequence; run fold_evodiff_scaffold_sequences.py "
            "and score_motif_scaffold.py first, or call score_output(..., fold=True)"
        )
    return record


def reward(example, output):
    return float(reward_result(example, output).get("reward", 0.0))


def success(example, output):
    return bool(reward_result(example, output).get("success", False))


def terminal_accept(example, output):
    return success(example, output)


def score_output(example, output, cache_dir=None, folded_dir=None, fold=False, model=None, device=None, chunk_size=None, backend=None, backend_version=None):
    backend = backend or example.get("folding_backend") or DEFAULT_BACKEND
    version = _backend_version(example, backend, backend_version=backend_version)
    return score_sequence(
        example,
        output,
        _cache_dir(example, cache_dir=cache_dir),
        folded_dir=folded_dir,
        backend=backend,
        backend_version=version,
        fold=fold,
        model=model,
        device=device,
        chunk_size=chunk_size,
        atom_names=example.get("atom_names") or "CA",
        rmsd_scale=REWARD_RMSD_SCALE,
        success_threshold=SUCCESS_RMSD,
    )


def row_info(example, output):
    sequence = clean_sequence(output)
    scoring_cfg = dict(_CONFIG.get("scoring", {}))
    info = {
        "sequence": sequence,
        "motif_preserved": bool(motif_preserved(example, sequence)),
    }
    info.update(low_complexity(sequence))
    record = cached_score(
        example,
        sequence,
        cache_dir=scoring_cfg.get("cache_dir"),
        backend=scoring_cfg.get("backend"),
        backend_version=scoring_cfg.get("backend_version"),
    )
    if record is not None:
        for key in [
            "motif_rmsd",
            "reward",
            "success",
            "plddt_motif",
            "plddt_all",
            "motif_atoms",
            "cache_key",
        ]:
            if key in record:
                info[key] = record[key]
    return info


def reward_state(example, state, harness=None, vocab=None, mask_id=None):
    if harness is not None and hasattr(harness, "decode_state"):
        return reward(example, decode_state(state, harness))
    if vocab is None:
        vocab = list("ACDEFGHIKLMNPQRSTVWY")
    if mask_id is None:
        mask_id = -1
    pieces = []
    for token in state:
        token = int(token)
        if token == int(mask_id):
            pieces.append("X")
        else:
            pieces.append(vocab[token % len(vocab)])
    return reward(example, "".join(pieces))


def _sample_token(logits, temperature):
    logits = torch.as_tensor(logits, dtype=torch.float32)
    if float(temperature) <= 0:
        return int(torch.argmax(logits).item())
    probs = torch.softmax(logits / float(temperature), dim=-1)
    return int(torch.multinomial(probs, 1).item())


def _mask_ratio(state, mask_id, locked):
    editable = [idx for idx in range(len(state)) if idx not in locked]
    if not editable:
        return 0.0
    masked = sum(1 for idx in editable if int(state[idx]) == int(mask_id))
    return float(masked / len(editable))


def _snapshot_indices(num_states, snapshots, rng):
    num_states = int(num_states)
    snapshots = max(1, int(snapshots))
    if num_states <= 0:
        return []
    if snapshots >= num_states:
        return list(range(num_states))
    return sorted(rng.sample(range(num_states), snapshots))


def collect_rollout_rows(harness, examples, config, rank=0):
    configure(config)
    generation_cfg = dict(config.get("generation", {}))
    rollout_cfg = dict(config.get("rollout", {}))
    temperature = float(generation_cfg.get("temperature", 1.0))
    snapshots = int(rollout_cfg.get("snapshots_per_rollout", 3))
    num_rollouts = int(rollout_cfg.get("num_rollouts", -1))
    repeats = 1 if num_rollouts < 0 else max(1, num_rollouts)
    val_frac = float(rollout_cfg.get("val_frac", 0.1))
    rng = random.Random(int(config.get("seed", 0)) + int(rank))
    rollout_examples = [
        (example, repeat_idx)
        for example in examples
        for repeat_idx in range(repeats)
    ]
    iterator = tqdm(rollout_examples, desc="Protein verifier rollouts", disable=int(rank) != 0)
    rows = []
    for rollout_idx, (example, repeat_idx) in enumerate(iterator):
        prompt = make_prompt(example)
        locked = locked_positions(example, None, harness)
        state = initial_state(example, harness)
        trajectory = [list(state)]
        actions = []
        while True:
            positions = [idx for idx, token in enumerate(state) if int(token) == int(harness.mask_id) and idx not in locked]
            if not positions:
                break
            pos = rng.choice(positions)
            logits = harness.logits(prompt, state)[pos]
            token = _sample_token(logits, temperature)
            state = list(state)
            state[int(pos)] = int(token)
            actions.append({"pos": int(pos), "token": int(token)})
            trajectory.append(list(state))

        final_state = list(state)
        output = decode_state(final_state, harness)
        record = reward_result(example, output)
        split = "val" if rng.random() < val_frac else "train"
        partials = trajectory[:-1] or trajectory
        example_id = example.get("id", example.get("task_id", "protein"))
        rollout_id = f"{example_id}-rollout-{repeat_idx:06d}"
        for snapshot_idx, trajectory_idx in enumerate(_snapshot_indices(len(partials), snapshots, rng)):
            snapshot = list(partials[trajectory_idx])
            motif_rmsd = record.get("motif_rmsd")
            row = {
                "id": f"{rollout_id}-snapshot-{snapshot_idx:02d}",
                "rollout_id": rollout_id,
                "snapshot_index": int(snapshot_idx),
                "trajectory_index": int(trajectory_idx),
                "split": split,
                "task": "evodiff_scaffold",
                "task_id": example.get("task_id") or example.get("id") or example.get("pdb"),
                "pdb": example.get("pdb"),
                "length": int(default_length(example)),
                "motif_positions": [int(x) for x in example.get("motif_positions", [])],
                "motif_text": example.get("motif_text", {}),
                "state": [int(x) for x in snapshot],
                "state_ids": [int(x) for x in snapshot],
                "final_state": [int(x) for x in final_state],
                "actions": actions,
                "output": output,
                "sequence": clean_sequence(output),
                "reward": float(record.get("reward", 0.0)),
                "success": bool(record.get("success", False)),
                "mask_ratio": _mask_ratio(snapshot, harness.mask_id, locked),
                "motif_preserved": bool(motif_preserved(example, output)),
            }
            if motif_rmsd is not None:
                row["reward_sigma4"] = float(np.exp(-((float(motif_rmsd) / 4.0) ** 2)))
            for key in ["motif_rmsd", "plddt_motif", "plddt_all", "motif_atoms", "cache_key"]:
                if key in record:
                    row[key] = record[key]
            rows.append(row)
    return rows


def _mean(values):
    values = [float(x) for x in values if x is not None]
    if not values:
        return 0.0
    return float(np.mean(values))


def _median(values):
    values = [float(x) for x in values if x is not None]
    if not values:
        return 0.0
    return float(np.median(values))


def _diversity(sequences):
    sequences = [clean_sequence(seq) for seq in sequences if clean_sequence(seq)]
    if len(sequences) < 2:
        return 0.0
    values = []
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            a = sequences[i]
            b = sequences[j]
            n = min(len(a), len(b))
            if n == 0:
                continue
            diff = sum(1 for x, y in zip(a[:n], b[:n]) if x != y)
            diff += abs(len(a) - len(b))
            values.append(diff / max(len(a), len(b), 1))
    return _mean(values)


def metrics(rows):
    scored = [row for row in rows if "reward" in row or "motif_rmsd" in row]
    sequences = [row.get("sequence") or row.get("output", "") for row in rows]
    successes = [row for row in rows if bool(row.get("success", False))]
    stats = [row.get("stats") or {} for row in rows]
    return {
        "num_samples": len(rows),
        "mean_tau": _mean(row.get("reward") for row in scored),
        "median_tau": _median(row.get("reward") for row in scored),
        "success_at_1A": _mean(1.0 if row.get("success") else 0.0 for row in scored),
        "motif_rmsd_mean": _mean(row.get("motif_rmsd") for row in scored),
        "motif_rmsd_median": _median(row.get("motif_rmsd") for row in scored),
        "plddt_motif_mean": _mean(row.get("plddt_motif") for row in scored),
        "plddt_all_mean": _mean(row.get("plddt_all") for row in scored),
        "motif_preserved_rate": _mean(1.0 if row.get("motif_preserved") else 0.0 for row in rows),
        "diversity": _diversity(sequences),
        "unique_sequences": len(set(clean_sequence(seq) for seq in sequences if clean_sequence(seq))),
        "unique_successful_scaffolds": len(set(clean_sequence(row.get("sequence") or row.get("output", "")) for row in successes)),
        "avg_nfe": _mean(stat.get("nfe", 0.0) for stat in stats),
        "forced_rate": _mean(1.0 if stat.get("forced", 0) else 0.0 for stat in stats),
        "low_complexity_top_aa": _mean(row.get("top_aa_frac") for row in rows),
        "low_complexity_longest_run": _mean(row.get("longest_run_frac") for row in rows),
    }
