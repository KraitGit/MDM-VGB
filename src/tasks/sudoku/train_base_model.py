import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

from .dit import DIT
from .ema import ExponentialMovingAverage
from .noise_schedule import get_noise
from .harness import build_config
from utils import load_task_config, per_rank_batch_size, repo_root, require_known_keys


DATA_KEYS = {"train_csv", "valid_csv"}
TRAINING_KEYS = {
    "beta1",
    "beta2",
    "dropout",
    "ema",
    "eps",
    "global_batch_size",
    "log_every",
    "lr",
    "max_steps",
    "num_workers",
    "output_dir",
    "save_every",
    "seed",
    "train_limit",
    "weight_decay",
}


class SudokuDataset(Dataset):
    def __init__(self, csv_path, limit=0):
        data = np.loadtxt(csv_path, delimiter=",", dtype=np.int64)
        if limit and limit > 0:
            data = data[:limit]
        self.data = torch.from_numpy(data).long()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]


def resolve_path(path):
    path = Path(path)
    return str(path if path.is_absolute() else repo_root() / path)


def setup_distributed(device=None):
    if "RANK" not in os.environ:
        device_obj = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        if device_obj.type == "cuda":
            torch.cuda.set_device(device_obj.index or 0)
        return 0, 1, device_obj.index or 0, device_obj
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank, torch.device(f"cuda:{local_rank}")


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def seed_everything(seed, rank):
    full_seed = int(seed) + int(rank)
    np.random.seed(full_seed)
    torch.manual_seed(full_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(full_seed)


def save_checkpoint(path, model, optimizer, ema, step, args, best_acc=0.0):
    raw_model = model.module if isinstance(model, DDP) else model
    state = {
        "step": int(step),
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_acc": float(best_acc),
        "args": vars(args),
    }
    if ema is not None:
        state["ema"] = ema.state_dict()
    torch.save(state, path)


def sample_t(batch_size, device, eps=1e-3):
    u = torch.rand(batch_size, device=device)
    offset = torch.arange(batch_size, device=device) / batch_size
    u = (u / batch_size + offset) % 1
    return (1 - eps) * u + eps


def compute_pretrain_loss(model, noise, batch, mask_index):
    x0 = batch
    t = sample_t(x0.shape[0], x0.device)
    sigma, _ = noise(t)
    move_chance = 1 - torch.exp(-sigma[:, None])
    move_indices = torch.rand(x0.shape, device=x0.device) < move_chance
    xt = torch.where(move_indices, mask_index, x0)
    mask_positions = xt == mask_index
    logits = model(xt, sigma)
    if not mask_positions.any():
        return logits.sum() * 0
    reweighting = 1 / t.unsqueeze(-1).expand_as(x0)
    losses = F.cross_entropy(logits[mask_positions], x0[mask_positions], reduction="none")
    return (losses * reweighting[mask_positions]).sum() / (x0.shape[0] * x0.shape[1])


def make_loader(dataset, batch_size, workers, sampler, shuffle, drop_last):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        drop_last=drop_last,
    )


def require_keys(config, keys, section):
    missing = [key for key in keys if config.get(key) is None]
    if missing:
        raise KeyError(f"Missing Sudoku base-model {section} keys: {missing}")


def train(args, device=None):
    rank, world_size, local_rank, device = setup_distributed(device)
    del local_rank
    seed_everything(args.seed, rank)

    args.train_csv = resolve_path(args.train_csv)
    args.valid_csv = resolve_path(args.valid_csv)
    args.output_dir = resolve_path(args.output_dir)
    if not os.path.exists(args.train_csv):
        raise FileNotFoundError(f"missing Sudoku train CSV: {args.train_csv}")
    if not os.path.exists(args.valid_csv):
        raise FileNotFoundError(f"missing Sudoku validation CSV: {args.valid_csv}")

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_data = SudokuDataset(args.train_csv, limit=args.train_limit)
    per_rank_batch = per_rank_batch_size(vars(args), world_size)
    train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    train_loader = make_loader(train_data, per_rank_batch, args.num_workers, train_sampler, False, True)

    cfg = build_config(dropout=args.dropout)
    model = DIT(cfg, vocab_size=11).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[device.index], output_device=device.index, broadcast_buffers=False)
    raw_model = model.module if isinstance(model, DDP) else model
    noise = get_noise(cfg).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )
    ema = ExponentialMovingAverage(raw_model.parameters(), decay=args.ema) if args.ema > 0 else None
    if ema is not None:
        ema.move_shadow_params_to_device(device)

    step = 0
    best_acc = 0.0

    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    model.train()
    last_log = time.time()
    running = 0.0
    running_count = 0
    epoch = 0
    while step < args.max_steps:
        train_sampler.set_epoch(epoch)
        for batch in train_loader:
            if step >= args.max_steps:
                break
            step += 1
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = compute_pretrain_loss(model, noise, batch, 0)
            loss.backward()
            optimizer.step()
            if ema is not None:
                ema.update(raw_model.parameters())
            running += float(loss.detach().item())
            running_count += 1
            if step % args.log_every == 0 and rank == 0:
                print(json.dumps({"step": step, "loss": running / max(running_count, 1), "dt": time.time() - last_log}))
                last_log = time.time()
                running = 0.0
                running_count = 0
            if step % args.save_every == 0 and rank == 0:
                save_checkpoint(os.path.join(checkpoint_dir, "last.pt"), model, optimizer, ema, step, args, best_acc)
        epoch += 1

    if rank == 0:
        last_path = os.path.join(checkpoint_dir, "last.pt")
        best_path = os.path.join(checkpoint_dir, "best.pt")
        save_checkpoint(last_path, model, optimizer, ema, step, args, best_acc)
        save_checkpoint(best_path, model, optimizer, ema, step, args, best_acc)
        with open(os.path.join(args.output_dir, "done.json"), "w") as f:
            json.dump({"step": step, "best_acc": best_acc}, f, indent=2)
    cleanup_distributed()
    return {"step": step, "output_dir": args.output_dir}


def train_from_config(config, device=None):
    data_cfg = dict(config.get("data", {}).get("base_model", {}))
    train_cfg = dict(config.get("training", {}).get("base_model", {}))
    require_known_keys(data_cfg, DATA_KEYS, "Sudoku base-model data")
    require_known_keys(train_cfg, TRAINING_KEYS, "Sudoku base-model training")
    require_keys(data_cfg, ["train_csv", "valid_csv"], "data")
    require_keys(
        train_cfg,
        [
            "output_dir",
            "seed",
            "global_batch_size",
            "max_steps",
            "save_every",
            "log_every",
            "lr",
            "weight_decay",
            "beta1",
            "beta2",
            "eps",
            "dropout",
            "ema",
            "train_limit",
            "num_workers",
        ],
        "training",
    )
    args = argparse.Namespace(**data_cfg, **train_cfg)
    return train(args, device=device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="sudoku")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    config = load_task_config(args.task, stage="base_model_training")
    print(train_from_config(config, device=args.device))


if __name__ == "__main__":
    main()
