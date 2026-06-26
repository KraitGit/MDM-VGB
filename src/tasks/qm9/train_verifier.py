import json
import os

import torch
from torch.nn.parallel import DistributedDataParallel
import torch.nn.functional as F
from tqdm.auto import tqdm

from utils import barrier, cleanup_distributed, get_device, init_distributed, per_rank_batch_size, read_rollout_payload, rollout_rows_from_payload, set_seed
from utils import supported_kwargs, verifier_training_config
from .verifier import QM9Verifier, save_verifier


def _train_mask(splits, count):
    if not splits:
        return torch.ones(count, dtype=torch.bool)
    mask = torch.tensor([str(split) == "train" for split in splits], dtype=torch.bool)
    return mask if mask.any() else torch.ones(count, dtype=torch.bool)


def _rank_shard(input_ids, rewards, rank, world_size):
    if world_size <= 1:
        return input_ids, rewards
    boundaries = torch.linspace(0, input_ids.shape[0], world_size + 1, dtype=torch.long)
    start = int(boundaries[int(rank)].item())
    end = int(boundaries[int(rank) + 1].item())
    return input_ids[start:end].contiguous(), rewards[start:end].contiguous()


def load_snapshot_tensors(data_path, seq_len, reward_fn, rank=0, world_size=1, shard_by_rank=True):
    payload = read_rollout_payload(data_path)
    if isinstance(payload, dict) and payload.get("format") == "qm9_rollout_v1":
        input_ids = payload["state_ids"].long()
        if input_ids.ndim != 2 or input_ids.shape[1] != int(seq_len):
            raise ValueError(f"QM9 verifier expects fixed state length {seq_len}")
        raw_rewards = payload.get("qeds", payload["rewards"]).float()
        if reward_fn is None:
            rewards = raw_rewards
        else:
            rewards = torch.tensor([float(reward_fn({"qed": float(qed)})) for qed in raw_rewards.tolist()], dtype=torch.float32)
        mask = _train_mask(payload.get("splits", []), int(input_ids.shape[0]))
        input_ids = input_ids[mask].contiguous()
        rewards = rewards[mask].contiguous()
        if shard_by_rank:
            input_ids, rewards = _rank_shard(input_ids, rewards, rank, world_size)
        return input_ids, rewards

    rows = rollout_rows_from_payload(payload)
    train_rows = [row for row in rows if row.get("split", "train") == "train"] or rows
    input_ids = torch.tensor([row["state_ids"] for row in train_rows], dtype=torch.long)
    if input_ids.ndim != 2 or input_ids.shape[1] != int(seq_len):
        raise ValueError(f"QM9 verifier expects fixed state length {seq_len}")
    rewards = torch.tensor(
        [float(reward_fn(row) if reward_fn is not None else row["reward"]) for row in train_rows],
        dtype=torch.float32,
    )
    if shard_by_rank:
        input_ids, rewards = _rank_shard(input_ids, rewards, rank, world_size)
    return input_ids.contiguous(), rewards.contiguous()


def train(
    data_path,
    output_dir,
    vocab_size,
    seq_len=32,
    hidden_dim=256,
    d_model=None,
    num_layers=4,
    num_heads=4,
    dropout=0.1,
    epochs=100,
    batch_size=512,
    global_batch_size=None,
    lr=1e-4,
    weight_decay=0.0,
    output_activation="sigmoid",
    seed=2026,
    device=None,
    reward_fn=None,
    shard_by_rank=True,
):
    dist_info = init_distributed()
    set_seed(int(seed) + int(dist_info["rank"]))
    if dist_info["rank"] == 0:
        os.makedirs(output_dir, exist_ok=True)

    if device is None and dist_info["distributed"]:
        device = f"cuda:{dist_info['local_rank']}"
    device = get_device(device)

    hidden_dim = int(hidden_dim if hidden_dim is not None else d_model)
    model = QM9Verifier(
        vocab_size=int(vocab_size),
        seq_len=int(seq_len),
        hidden_dim=hidden_dim,
        num_layers=int(num_layers),
        num_heads=int(num_heads),
        dropout=float(dropout),
        output_activation=str(output_activation),
    ).to(device)
    train_model = model
    if dist_info["distributed"]:
        train_model = DistributedDataParallel(model, device_ids=[dist_info["local_rank"]] if device.type == "cuda" else None)

    input_ids, rewards = load_snapshot_tensors(
        data_path,
        seq_len,
        reward_fn,
        rank=int(dist_info["rank"]),
        world_size=int(dist_info["world_size"]),
        shard_by_rank=bool(shard_by_rank),
    )
    n = int(input_ids.shape[0])
    batch_size = per_rank_batch_size({"global_batch_size": global_batch_size if global_batch_size is not None else batch_size}, dist_info["world_size"])
    steps_per_epoch = (n + batch_size - 1) // batch_size
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    history = []

    for epoch in range(1, int(epochs) + 1):
        perm = torch.randperm(n)
        train_model.train()
        total_loss = 0.0
        seen = 0
        iterator = range(steps_per_epoch)
        if dist_info["rank"] == 0:
            iterator = tqdm(iterator, desc=f"QM9 verifier epoch {epoch}", leave=False)
        for step in iterator:
            idx = perm[step * batch_size:(step + 1) * batch_size]
            x = input_ids[idx].to(device=device, dtype=torch.long)
            y = rewards[idx].to(device=device, dtype=torch.float32)
            pred = train_model(x)
            loss = F.mse_loss(pred, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item()) * int(x.shape[0])
            seen += int(x.shape[0])

        local = torch.tensor([total_loss, seen], dtype=torch.float64, device=device)
        if dist_info["distributed"]:
            torch.distributed.all_reduce(local, op=torch.distributed.ReduceOp.SUM)
        row = {"epoch": epoch, "mse": float(local[0].item() / max(1.0, local[1].item())), "num_snapshots": int(local[1].item())}
        if dist_info["rank"] == 0:
            history.append(row)
            print(json.dumps(row), flush=True)
            with open(os.path.join(output_dir, "history.json"), "w", encoding="utf-8") as handle:
                json.dump(history, handle, indent=2)
        barrier(dist_info)

    if dist_info["rank"] == 0:
        path = os.path.join(output_dir, "verifier.pt")
        save_verifier(model, path, output_activation=output_activation)
        print(json.dumps({"saved_verifier": path}), flush=True)
    cleanup_distributed(dist_info)
    return {"history": history, "world_size": dist_info["world_size"]} if dist_info["rank"] == 0 else {"rank": dist_info["rank"], "done": True}


def train_from_config(config, device=None):
    from . import task

    task.configure(config)

    def reward_fn(row):
        if "qed" in row:
            return task._shape_reward(float(row["qed"]))
        return task.reward(None, row.get("output", ""))

    data_path, output_dir, train_cfg = verifier_training_config(config, device)
    train_cfg["reward_fn"] = reward_fn
    return train(data_path, output_dir, **supported_kwargs(train, train_cfg))
