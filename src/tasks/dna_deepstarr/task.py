import json
import math
import os
import random

from .harness import center_crop, clean_dna
from .deepstarr_oracle import gc_content
from .deepstarr_oracle import max_homopolymer
from tasks.common import decode_with_harness, masked_initial_state, prompt_from_example, result_state


DEFAULT_N_TOKENS = 48
DEFAULT_TARGET_NT = 249
_ORACLE = None
_THRESHOLDS = None
_CONFIG = {}


def configure(config):
    global _CONFIG, _ORACLE, _THRESHOLDS
    _CONFIG = dict(config or {})
    reward_cfg = dict(_CONFIG.get("reward", {}))
    oracle_cfg = dict(_CONFIG.get("oracle", {}))
    thresholds_path = reward_cfg.get("thresholds_path")
    if thresholds_path:
        os.environ["D3LM_DEEPSTARR_THRESHOLDS"] = str(thresholds_path)
    model_path = oracle_cfg.get("model_path")
    if model_path:
        os.environ["DEEPSTARR_MODEL_PATH"] = str(model_path)
    _ORACLE = None
    _THRESHOLDS = None


def _oracle():
    global _ORACLE
    if _ORACLE is None:
        from .deepstarr_oracle import DeepSTARRDevOracle

        _ORACLE = DeepSTARRDevOracle(
            path=os.environ.get("DEEPSTARR_MODEL_PATH"),
            device=os.environ.get("DEEPSTARR_DEVICE"),
        )
    return _ORACLE


def _thresholds():
    global _THRESHOLDS
    if _THRESHOLDS is not None:
        return _THRESHOLDS
    path = os.environ.get("D3LM_DEEPSTARR_THRESHOLDS")
    if not path:
        _THRESHOLDS = {}
        return _THRESHOLDS
    with open(path, "r", encoding="utf-8") as f:
        _THRESHOLDS = json.load(f)
    return _THRESHOLDS


def load_examples(split):
    count = int(_CONFIG.get("data", {}).get("count", 100))
    return [
        {
            "id": f"d3lm-deepstarr-{split}-{idx}",
            "split": split,
            "length": DEFAULT_N_TOKENS,
            "target_nt": DEFAULT_TARGET_NT,
        }
        for idx in range(count)
    ]


def make_prompt(example):
    return prompt_from_example(example)


def default_length(example):
    return int(example.get("length", example.get("n_tokens", DEFAULT_N_TOKENS)))


def decode_state(state, harness):
    return decode_with_harness(state, harness)


def initial_state(example, harness):
    return masked_initial_state(default_length(example), harness)


def sequence_from_output(output):
    seq = center_crop(output, DEFAULT_TARGET_NT)
    if seq is not None:
        return seq
    seq = clean_dna(output)
    return seq if len(seq) == DEFAULT_TARGET_NT else None


def sequence_from_state(state, harness):
    if hasattr(harness, "decode_to_249nt"):
        return harness.decode_to_249nt([int(x) for x in state if int(x) != int(harness.mask_id)])
    return sequence_from_output(harness.decode_state(state))


def reward(example, output):
    del example
    seq = sequence_from_output(output)
    if seq is None:
        return 0.0
    return float(_oracle().score([seq])[0]["dev"])


def reward_result(example, result, harness):
    state = result_state(result)
    if state is not None:
        seq = sequence_from_state(state, harness)
        if seq is None:
            return 0.0
        return float(_oracle().score([seq])[0]["dev"])
    return reward(example, result.get("output", ""))


def _reward_values(dev):
    thresholds = _thresholds()
    if thresholds.get("mean") is None:
        z = float(dev) if dev == dev else 0.0
        tau = z
        return z, tau, 1.0 if z > 0.0 else 0.0
    std = float(thresholds.get("std", 1.0) or 1.0)
    if std <= 0:
        std = 1.0
    eta = float(_CONFIG.get("reward", {}).get("eta", 1.0))
    z = (float(dev) - float(thresholds["mean"])) / std
    z = max(-3.0, min(3.0, z))
    tau = float(math.exp(eta * z))
    success95 = 1.0 if thresholds.get("t95") is not None and float(dev) >= float(thresholds["t95"]) else 0.0
    return float(z), tau, success95


def _shape_reward(dev):
    zdev, tau, success95 = _reward_values(dev)
    del zdev
    mode = str(_CONFIG.get("reward", {}).get("mode", "tau"))
    if mode == "success95":
        return success95
    return tau


def _random_mask_state(token_ids, mask_id, rng):
    state = [int(x) for x in token_ids]
    if not state:
        return state, 0.0
    count = int(round(rng.random() * len(state)))
    if count <= 0:
        return state, 0.0
    for pos in rng.sample(range(len(state)), min(count, len(state))):
        state[int(pos)] = int(mask_id)
    return state, count / len(state)


def collect_rollout_rows(harness, examples, config, rank=0):
    configure(config)
    generation_cfg = dict(config.get("generation", {}))
    batch_size = max(1, int(generation_cfg.get("batch_size", 16)))
    n_tokens = int(generation_cfg.get("max_new_tokens", DEFAULT_N_TOKENS))
    steps = int(generation_cfg.get("steps", 50))
    temperature = float(generation_cfg.get("temperature", 1.0))
    remasking = generation_cfg.get("remasking", "random")
    snapshots = int(config.get("rollout", {}).get("snapshots_per_rollout", 4))
    val_frac = float(config.get("rollout", {}).get("val_frac", 0.1))
    rng = random.Random(int(config.get("seed", 0)) + int(rank))

    rows = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        generated = harness.generate_tokens(
            n_tokens=n_tokens,
            batch_size=len(batch),
            steps=steps,
            temperature=temperature,
            alg="random" if remasking in (None, "random") else remasking,
        )
        sequences = [row.get("sequence_249") for row in generated if row.get("sequence_249")]
        scores = _oracle().score(sequences)
        score_idx = 0
        for local_idx, (example, generated_row) in enumerate(zip(batch, generated)):
            seq = generated_row.get("sequence_249")
            if seq:
                scored = scores[score_idx]
                score_idx += 1
                dev = float(scored["dev"])
                hk = float(scored["hk"])
            else:
                dev = 0.0
                hk = 0.0
            zdev, tau, success95 = _reward_values(dev) if seq else (0.0, 0.0, 0.0)
            split = "val" if rng.random() < val_frac else "train"
            token_ids = generated_row.get("token_ids", [])
            for snap_idx in range(snapshots):
                state, mask_ratio = _random_mask_state(token_ids, harness.mask_id, rng)
                rows.append(
                    {
                        "id": f"{example['id']}-{snap_idx}",
                        "rollout_id": example["id"],
                        "snapshot_index": snap_idx,
                        "split": split,
                        "state_ids": state,
                        "final_token_ids": [int(x) for x in token_ids],
                        "output": seq or generated_row.get("decoded_nt", ""),
                        "sequence_249": seq,
                        "valid_249": bool(seq),
                        "deepstarr_dev": dev,
                        "deepstarr_hk": hk,
                        "zdev": zdev,
                        "tau": tau,
                        "success_95_reward": success95,
                        "reward": _shape_reward(dev) if seq else 0.0,
                        "mask_ratio": mask_ratio,
                        "mask_id": int(harness.mask_id),
                        "vocab_size": int(harness.vocab_size),
                        "n_tokens": n_tokens,
                    }
                )
    return rows


def terminal_accept(example, output):
    score = reward(example, output)
    thresholds = _thresholds()
    if thresholds.get("t95") is None:
        return score > 0.0
    return score >= float(thresholds["t95"])


def row_info(example, output):
    seq = sequence_from_output(output)
    if seq is None:
        return {
            "sequence_249": None,
            "valid_249": False,
            "deepstarr_dev": 0.0,
            "deepstarr_hk": 0.0,
            "success_95": False,
            "success_99": False,
        }
    scores = _oracle().score([seq])[0]
    thresholds = _thresholds()
    t95 = thresholds.get("t95")
    t99 = thresholds.get("t99")
    return {
        "sequence_249": seq,
        "valid_249": True,
        "deepstarr_dev": float(scores["dev"]),
        "deepstarr_hk": float(scores["hk"]),
        "gc_content": float(gc_content(seq)),
        "max_homopolymer": int(max_homopolymer(seq)),
        "success_95": bool(t95 is not None and float(scores["dev"]) >= float(t95)),
        "success_99": bool(t99 is not None and float(scores["dev"]) >= float(t99)),
    }


def metrics(rows):
    if not rows:
        return {"num_examples": 0, "success_95": 0.0, "success_99": 0.0, "mean_dev": 0.0}
    seqs = [row.get("sequence_249") for row in rows if row.get("sequence_249")]
    unique = len(set(seqs)) / max(1, len(seqs))
    return {
        "num_examples": len(rows),
        "valid_249_rate": sum(1 for row in rows if row.get("valid_249")) / len(rows),
        "success_95": sum(1 for row in rows if row.get("success_95")) / len(rows),
        "success_99": sum(1 for row in rows if row.get("success_99")) / len(rows),
        "mean_dev": sum(float(row.get("deepstarr_dev", 0.0)) for row in rows) / len(rows),
        "mean_hk": sum(float(row.get("deepstarr_hk", 0.0)) for row in rows) / len(rows),
        "gc": sum(float(row.get("gc_content", 0.0)) for row in rows) / len(rows),
        "unique_fraction": unique,
        "duplicate_rate": 1.0 - unique,
    }
