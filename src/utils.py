"""Small shared utilities for MDM-VGB runs."""


import copy
import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def repo_root():
    return Path(__file__).resolve().parents[1]


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected a mapping in {path}")
    return data


def deep_update(base, update):
    out = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(config_path):
    path = Path(config_path)
    cfg = load_yaml(path)
    defaults = cfg.pop("defaults", None)
    if not defaults:
        return cfg
    if isinstance(defaults, (str, Path)):
        defaults = [defaults]
    merged = {}
    for item in defaults:
        item_path = Path(item)
        if not item_path.is_absolute():
            item_path = path.parent / item_path
        merged = deep_update(merged, load_config(item_path))
    return deep_update(merged, cfg)


def load_task_config(
    task,
    config = None,
    stage = "inference",
):
    base = load_config(repo_root() / "configs" / "main.yaml")
    if config is not None:
        return deep_update(base, load_config(config))
    if task is None:
        return base
    task_path = Path(task)
    if not task_path.suffix:
        task_path = repo_root() / "configs" / str(task) / f"{task}_{stage}.yaml"
    elif not task_path.is_absolute():
        task_path = repo_root() / task_path
    return deep_update(base, load_config(task_path))


def setup_logging(level = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("aoar_vgb")


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_rollout_payload(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        raise ValueError(f"Rollout inputs must use .pt, got {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def rollout_rows_from_payload(payload):
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported rollout payload type: {type(payload).__name__}")
    if "rows" in payload:
        return list(payload["rows"])
    raise TypeError(f"Rollout payload does not contain rows: {payload.get('format', 'unknown')}")


def read_rollout(path):
    return rollout_rows_from_payload(read_rollout_payload(path))


def write_rollout(
    path,
    rows,
    task_module = None,
    config = None,
):
    path = Path(path)
    rows = list(rows)
    if path.suffix == ".jsonl":
        raise ValueError(f"Rollout outputs must use .pt, got {path}")
    serialize = getattr(task_module, "serialize_rollout_rows", None) if task_module is not None else None
    payload = serialize(rows, config or {}) if callable(serialize) else {"format": "rollout_rows_v1", "rows": rows}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device = None):
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pad_1d(sequences, pad_value, device = None):
    max_len = max((len(seq) for seq in sequences), default=0)
    out = torch.full((len(sequences), max_len), int(pad_value), dtype=torch.long, device=device)
    mask = torch.zeros((len(sequences), max_len), dtype=torch.bool, device=device)
    for idx, seq in enumerate(sequences):
        if seq:
            out[idx, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
            mask[idx, : len(seq)] = True
    return out, mask


def verifier_training_config(config, device = None):
    task = str(config.get("task", "task"))
    training_cfg = dict(config.get("training", {}))
    data_path = training_cfg.get("data")
    output_dir = training_cfg.get("output_dir")
    if not data_path:
        data_path = f"outputs/{task}/base_rollout.pt"
    if not output_dir:
        output_dir = f"outputs/{task}/verifier"

    train_cfg = {}
    train_cfg.update(config.get("verifier", {}))
    train_cfg.update(training_cfg.get("verifier", {}))
    train_cfg["seed"] = train_cfg.get("seed", config.get("seed", 2026))
    if device:
        train_cfg["device"] = device
    train_cfg = {key: value for key, value in train_cfg.items() if value is not None}
    return str(data_path), str(output_dir), train_cfg


def supported_kwargs(fn, kwargs):
    allowed = set(fn.__code__.co_varnames[: fn.__code__.co_argcount])
    return {key: value for key, value in kwargs.items() if key in allowed}


def per_rank_batch_size(config, world_size):
    global_batch_size = config.get("global_batch_size", config.get("batch_size"))
    if global_batch_size is None:
        raise KeyError("missing global_batch_size")
    global_batch_size = int(global_batch_size)
    world_size = int(world_size)
    if world_size <= 0:
        raise ValueError(f"invalid world_size={world_size}")
    if global_batch_size % world_size != 0:
        raise ValueError(f"global_batch_size={global_batch_size} must be divisible by world_size={world_size}")
    return max(1, global_batch_size // world_size)


def init_distributed():
    device = "cuda:0" if torch.cuda.is_available() else None
    if not torch.distributed.is_available():
        return {"rank": 0, "world_size": 1, "local_rank": 0, "distributed": False, "device": device}
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        return {"rank": 0, "world_size": 1, "local_rank": 0, "distributed": False, "device": device}
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    if not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if backend == "nccl":
            try:
                torch.distributed.init_process_group(backend=backend, device_id=torch.device(device))
            except TypeError:
                torch.distributed.init_process_group(backend=backend)
        else:
            torch.distributed.init_process_group(backend=backend)
    return {
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "distributed": world_size > 1,
        "device": device,
    }


def cleanup_distributed(info = None):
    del info
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def barrier(info = None):
    del info
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.cuda.is_available():
            torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
        else:
            torch.distributed.barrier()


def shard_items(items, info):
    if not info.get("distributed"):
        return items
    rank = int(info["rank"])
    world_size = int(info["world_size"])
    per_rank = (len(items) + world_size - 1) // world_size
    start = rank * per_rank
    return items[start:min(len(items), start + per_rank)]


def gather_objects(obj, info):
    if not info.get("distributed"):
        return [obj]
    gathered = [None for _ in range(int(info["world_size"]))]
    torch.distributed.all_gather_object(gathered, obj)
    return gathered


def mean(values):
    xs = [float(value) for value in values if value is not None]
    return float(sum(xs) / len(xs)) if xs else 0.0


def infer_nfe(row):
    stats = row.get("stats") or {}
    if "nfe" in stats:
        return float(stats["nfe"])
    if "model_forwards" in stats:
        return float(stats["model_forwards"])
    if "d3lm_model_forwards_est" in stats:
        return float(stats["d3lm_model_forwards_est"])
    return 1.0


def verifier_evals(row):
    stats = row.get("stats") or {}
    for key in ("verifier_state_evals", "verifier_evals", "value_evals"):
        if key in stats:
            return float(stats[key])
    return 0.0


def add_nfe_metrics(metrics, rows, verifier_cost = 0.0):
    out = dict(metrics)
    raw_nfe = mean(infer_nfe(row) for row in rows)
    value_evals = mean(verifier_evals(row) for row in rows)
    out["raw_nfe"] = raw_nfe
    out["adjusted_nfe"] = raw_nfe + float(verifier_cost) * value_evals
    return out


def log_metrics(metrics, logger = None):
    text = json.dumps(metrics, sort_keys=True)
    if logger is None:
        print(text)
    else:
        logger.info("metrics %s", text)
