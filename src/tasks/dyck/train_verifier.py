import json
import os

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from utils import read_rollout
from utils import barrier, cleanup_distributed, init_distributed
from utils import set_seed
from utils import get_device, pad_1d
from utils import per_rank_batch_size
from utils import supported_kwargs, verifier_training_config
from .verifier import TokenVerifier


class SnapshotDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        return {"state_ids": row["state_ids"], "reward": float(row["reward"])}


def collate(batch, pad_id):
    ids, mask = pad_1d([row["state_ids"] for row in batch], pad_id)
    rewards = torch.tensor([row["reward"] for row in batch], dtype=torch.float)
    return {"input_ids": ids, "attention_mask": mask, "reward": rewards}


def evaluate(model, loader, device):
    model.eval()
    losses = []
    preds = []
    labels = []
    loss_fn = torch.nn.MSELoss()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            reward = batch["reward"].to(device)
            logits = model(input_ids, attention_mask)
            pred = torch.sigmoid(logits)
            loss = loss_fn(pred, reward)
            losses.append(float(loss.item()))
            preds.extend(pred.cpu().tolist())
            labels.extend(reward.cpu().tolist())
    if not labels:
        return {"loss": 0.0, "acc": 0.0}
    acc = sum((p >= 0.5) == (y >= 0.5) for p, y in zip(preds, labels)) / len(labels)
    mse = sum((p - y) ** 2 for p, y in zip(preds, labels)) / len(labels)
    return {"loss": sum(losses) / max(1, len(losses)), "acc": acc, "mse": mse}


def train(data_path, output_dir, vocab_size, mask_id, max_length=512, d_model=256, num_layers=4, num_heads=4, dropout=0.25, epochs=10, batch_size=64, global_batch_size=None, eval_batch_size=None, lr=1e-4, seed=2026, device=None):
    dist_info = init_distributed()
    set_seed(seed)
    if dist_info["rank"] == 0:
        os.makedirs(output_dir, exist_ok=True)
    rows = read_rollout(data_path)
    train_rows = [row for row in rows if row.get("split", "train") == "train"]
    val_rows = [row for row in rows if row.get("split") == "val"]
    if not val_rows:
        val_rows = train_rows[: max(1, len(train_rows) // 10)]
    if device is None and dist_info["distributed"]:
        device = f"cuda:{dist_info['local_rank']}"
    device = get_device(device)
    model = TokenVerifier(vocab_size, mask_id, max_length, d_model, num_layers, num_heads, dropout).to(device)
    train_batch_size = per_rank_batch_size({"global_batch_size": global_batch_size if global_batch_size is not None else batch_size}, dist_info["world_size"])
    eval_batch_size = int(eval_batch_size or (global_batch_size if global_batch_size is not None else batch_size))
    train_dataset = SnapshotDataset(train_rows)
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if dist_info["distributed"] else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=lambda b: collate(b, mask_id),
    )
    val_loader = DataLoader(SnapshotDataset(val_rows), batch_size=eval_batch_size, shuffle=False, collate_fn=lambda b: collate(b, mask_id))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    train_model = model
    if dist_info["distributed"]:
        train_model = DistributedDataParallel(model, device_ids=[dist_info["local_rank"]] if device.type == "cuda" else None)
    loss_fn = torch.nn.MSELoss()
    best = None
    history = []

    for epoch in range(1, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_model.train()
        total = 0.0
        count = 0
        iterator = tqdm(train_loader, desc=f"epoch {epoch}", leave=False) if dist_info["rank"] == 0 else train_loader
        for batch in iterator:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            reward = batch["reward"].to(device)
            logits = train_model(input_ids, attention_mask)
            loss = loss_fn(torch.sigmoid(logits), reward)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item())
            count += 1
        barrier(dist_info)
        val = evaluate(model, val_loader, device) if dist_info["rank"] == 0 else {"loss": 0.0, "acc": 0.0, "mse": 0.0}
        row = {"epoch": epoch, "train_loss": total / max(1, count)}
        row.update({f"val_{k}": v for k, v in val.items()})
        if dist_info["rank"] == 0:
            history.append(row)
            if best is None or val["loss"] < best["val_loss"]:
                best = {"epoch": epoch, "val_loss": val["loss"]}
                torch.save({"config": model.config, "state_dict": model.state_dict(), "metrics": row}, os.path.join(output_dir, "best.pt"))
            with open(os.path.join(output_dir, "history.json"), "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
        barrier(dist_info)
    cleanup_distributed(dist_info)
    return {"best": best, "history": history, "world_size": dist_info["world_size"]} if dist_info["rank"] == 0 else {"rank": dist_info["rank"], "done": True}


def train_from_config(config, device=None):
    data_path, output_dir, train_cfg = verifier_training_config(config, device)
    return train(data_path, output_dir, **supported_kwargs(train, train_cfg))
