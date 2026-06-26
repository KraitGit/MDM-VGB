import json
import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from tqdm.auto import tqdm

from .verifier import ProteinStateValueVerifier
from .verifier import build_batch
from .verifier import task_key
from utils import per_rank_batch_size, read_rollout, verifier_training_config


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def ranks(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values)
    out = np.empty(len(values), dtype=np.float64)
    out[order] = np.arange(len(values), dtype=np.float64)
    return out


def corr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def metrics(pred, labels):
    pred = np.asarray(pred, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if len(pred) == 0:
        return {"mse": 0.0, "mae": 0.0, "pearson": 0.0, "spearman": 0.0, "log_spearman": 0.0, "num": 0}
    log_pred = np.log(np.clip(pred, 1e-12, 1.0))
    log_labels = np.log(np.clip(labels, 1e-12, 1.0))
    return {
        "mse": float(np.mean((pred - labels) ** 2)),
        "mae": float(np.mean(np.abs(pred - labels))),
        "pearson": corr(pred, labels),
        "spearman": corr(ranks(pred), ranks(labels)),
        "log_spearman": corr(ranks(log_pred), ranks(log_labels)),
        "mean_pred": float(np.mean(pred)),
        "mean_label": float(np.mean(labels)),
        "std_pred": float(np.std(pred)),
        "std_label": float(np.std(labels)),
        "collapsed": bool(float(np.std(pred)) < 0.05 * max(float(np.std(labels)), 1e-12)),
        "num": int(len(pred)),
    }


def infer_vocab_size(rows):
    max_token = 0
    for row in rows:
        for token in row["state"]:
            token = int(token)
            if token >= 0:
                max_token = max(max_token, token)
    return max_token + 1


def task_mapping(rows):
    tasks = sorted(set(task_key(row) for row in rows))
    return {task: idx for idx, task in enumerate(tasks)}


def split_by_reward(rows, threshold):
    pos = [row for row in rows if float(row.get("reward", 0.0)) > float(threshold)]
    neg = [row for row in rows if float(row.get("reward", 0.0)) <= float(threshold)]
    return pos, neg


def row_target(row, args):
    field = str(args.target_field)
    if field not in row:
        return float(row.get("reward", 0.0))
    return float(row.get(field, 0.0))


def target_tensor(rows, args, device):
    values = [row_target(row, args) for row in rows]
    labels = torch.tensor(values, dtype=torch.float32, device=device)
    if args.target_mode == "logtau":
        return torch.log(labels.clamp_min(1e-12)).clamp(min=-20.0, max=0.0)
    return labels


def predict_for_loss(model, batch, args):
    out = model(
        batch["states"],
        batch["is_motif"],
        batch["is_editable"],
        batch["task_ids"],
        valid_mask=batch["valid_mask"],
    )
    if args.target_mode == "value":
        return torch.exp(out)
    return out


def loss_value(pred, labels, args):
    del args
    return F.mse_loss(pred, labels)


def sample_batch(rows, batch_size, rng, pos_rows=None, neg_rows=None, pos_fraction=0.0):
    if len(rows) <= int(batch_size):
        return list(rows)
    if float(pos_fraction) > 0 and pos_rows and neg_rows:
        n_pos = int(round(int(batch_size) * float(pos_fraction)))
        n_pos = max(0, min(int(batch_size), n_pos))
        n_neg = int(batch_size) - n_pos
        batch = []
        if n_pos:
            batch.extend(pos_rows[idx] for idx in rng.choices(range(len(pos_rows)), k=n_pos))
        if n_neg:
            batch.extend(neg_rows[idx] for idx in rng.choices(range(len(neg_rows)), k=n_neg))
        rng.shuffle(batch)
        return batch
    return [rows[idx] for idx in rng.choices(range(len(rows)), k=int(batch_size))]


def best_metric_value(row, metric):
    return float((row.get("val") or {}).get(metric, 0.0))


def is_better(row, best, metric, mode):
    if best is None:
        return True
    value = best_metric_value(row, metric)
    best_value = best_metric_value(best, metric)
    if mode == "min":
        return value < best_value
    return value > best_value


def setup_distributed(args):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        if torch.cuda.is_available() and (args.device is None or str(args.device).startswith("cuda")):
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            backend = "nccl"
        else:
            device = torch.device(args.device or "cpu")
            backend = "gloo"
        dist.init_process_group(backend=backend)
    else:
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    return distributed, rank, world_size, local_rank, device


@torch.no_grad()
def evaluate(model, rows, task_to_idx, args, device, limit=8192):
    model.eval()
    rows = rows[:min(len(rows), int(limit))]
    preds = []
    labels = []
    losses = []
    for start in range(0, len(rows), int(args.batch_size)):
        batch_rows = rows[start:start + int(args.batch_size)]
        batch = build_batch(batch_rows, task_to_idx, device)
        target = target_tensor(batch_rows, args, device)
        pred_loss = predict_for_loss(model, batch, args)
        loss = loss_value(pred_loss, target, args)
        losses.append(float(loss.detach().float().cpu().item()))
        if args.target_mode == "logtau":
            values = torch.exp(pred_loss)
            target_values = torch.exp(target)
        else:
            values = pred_loss
            target_values = target
        preds.extend(values.detach().float().cpu().tolist())
        labels.extend(target_values.detach().float().cpu().tolist())
    out = metrics(preds, labels)
    out["loss"] = float(np.mean(losses)) if losses else 0.0
    return out


def train_from_config(config, device=None):
    data_path, output_dir, train_cfg = verifier_training_config(config, device)
    verifier_cfg = dict(config.get("verifier", {}))
    args = SimpleNamespace(
        data=data_path,
        out_dir=output_dir,
        dim=verifier_cfg.get("dim", verifier_cfg.get("hidden_dim", 192)),
        layers=verifier_cfg.get("layers", verifier_cfg.get("num_layers", 4)),
        heads=verifier_cfg.get("heads", verifier_cfg.get("num_heads", 4)),
        dropout=verifier_cfg.get("dropout", 0.1),
        batch_size=train_cfg.get("batch_size", 256),
        iterations=train_cfg.get("iterations", 1000),
        lr=train_cfg.get("lr", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
        eval_every=train_cfg.get("eval_every", 100),
        seed=train_cfg.get("seed", 2026),
        device=train_cfg.get("device"),
        max_length=verifier_cfg.get("max_length"),
        target_field=train_cfg.get("target_field", "reward"),
        target_mode=train_cfg.get("target_mode", "value"),
        pooling=verifier_cfg.get("pooling", train_cfg.get("pooling", "triple")),
        positive_fraction=train_cfg.get("positive_fraction", 0.0),
        positive_threshold=train_cfg.get("positive_threshold", 1e-3),
        global_batch_size=train_cfg.get("global_batch_size"),
        best_metric=train_cfg.get("best_metric", "mse"),
        best_mode=train_cfg.get("best_mode"),
    )
    return train_from_args(args)


def train_from_args(args):
    distributed, rank, world_size, local_rank, device = setup_distributed(args)
    random.seed(args.seed + rank)
    np.random.seed((args.seed + rank) % (2**32 - 1))
    torch.manual_seed(args.seed + rank)
    rows = read_rollout(args.data)
    train = [row for row in rows if row.get("split") == "train"]
    val = [row for row in rows if row.get("split") == "val"]
    if not train:
        train = rows
    if not val:
        val = rows[:min(1024, len(rows))]
    pos_train, neg_train = split_by_reward(train, args.positive_threshold)
    max_length = int(args.max_length or max(len(row["state"]) for row in rows))
    vocab_size = infer_vocab_size(rows)
    task_to_idx = task_mapping(rows)
    model = ProteinStateValueVerifier(
        vocab_size=vocab_size,
        max_length=max_length,
        num_tasks=len(task_to_idx),
        dim=args.dim,
        layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
        pooling=args.pooling,
        output_mode="logv",
    ).to(device)
    if distributed:
        ddp_kwargs = {}
        if device.type == "cuda":
            ddp_kwargs = {"device_ids": [local_rank], "output_device": local_rank}
        model = DistributedDataParallel(model, **ddp_kwargs)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    out_dir = Path(args.out_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    model_config = {
        "architecture": "protein_state_transformer",
        "vocab_size": int(vocab_size),
        "mask_id": -1,
        "max_length": int(max_length),
        "num_tasks": len(task_to_idx),
        "task_to_idx": task_to_idx,
        "dim": int(args.dim),
        "layers": int(args.layers),
        "heads": int(args.heads),
        "dropout": float(args.dropout),
        "pooling": args.pooling,
        "output_mode": "logv",
        "head_params": sum(param.numel() for param in model.parameters() if param.requires_grad),
        "train_examples": len(train),
        "val_examples": len(val),
        "positive_train_examples": len(pos_train),
        "negative_train_examples": len(neg_train),
        "target": args.target_field,
        "target_mode": args.target_mode,
        "distributed": bool(distributed),
        "world_size": int(world_size),
        "args": vars(args),
    }
    if rank == 0:
        write_json(out_dir / "config.json", model_config)
        print(json.dumps(model_config, sort_keys=True), flush=True)
    rng = random.Random(args.seed + rank)
    history = []
    best = None
    best_metric = str(args.best_metric)
    best_mode = args.best_mode
    if best_mode is None:
        best_mode = "min" if best_metric in ("mse", "mae", "loss") else "max"
    train_batch_size = per_rank_batch_size({"global_batch_size": args.global_batch_size if args.global_batch_size is not None else args.batch_size}, world_size)
    steps = range(1, int(args.iterations) + 1)
    pbar = tqdm(steps, desc="protein value", disable=rank != 0)
    for step in pbar:
        model.train()
        batch_rows = sample_batch(
            train,
            train_batch_size,
            rng,
            pos_rows=pos_train,
            neg_rows=neg_train,
            pos_fraction=args.positive_fraction,
        )
        batch = build_batch(batch_rows, task_to_idx, device)
        labels = target_tensor(batch_rows, args, device)
        pred = predict_for_loss(model, batch, args)
        loss = loss_value(pred, labels, args)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if rank == 0:
            pbar.set_postfix(loss=float(loss.detach().cpu().item()))
        if step % int(args.eval_every) == 0 or step == int(args.iterations):
            if distributed:
                dist.barrier()
            if rank == 0:
                eval_model = model.module if distributed else model
                train_eval = evaluate(eval_model, train, task_to_idx, args, device)
                val_eval = evaluate(eval_model, val, task_to_idx, args, device)
                row = {
                    "step": int(step),
                    "loss": float(loss.detach().cpu().item()),
                    "train": train_eval,
                    "val": val_eval,
                }
                history.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
                payload = {
                    "state_dict": eval_model.state_dict(),
                    "config": model_config,
                    "metrics": row,
                }
                torch.save(payload, out_dir / "last.pt")
                if is_better(row, best, best_metric, best_mode):
                    best = row
                    torch.save(payload, out_dir / "best.pt")
                write_json(out_dir / "metrics.json", {"best": best, "history": history})
            if distributed:
                dist.barrier()
    if distributed:
        dist.destroy_process_group()
    return {"best": best, "output_dir": str(out_dir), "history": history} if rank == 0 else {"rank": rank, "done": True}
