import json
import random
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from .verifier import SmallDNAValueVerifier
from utils import read_rollout
from utils import barrier, cleanup_distributed, get_device, init_distributed, per_rank_batch_size, set_seed
from utils import verifier_training_config


class StateDataset(Dataset):
    def __init__(self, rows):
        self.rows = list(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        return {"state_ids": row["state_ids"], "tau": float(row.get("reward", row.get("tau", 0.0)))}


def collate(batch, pad_id):
    max_len = max(len(row["state_ids"]) for row in batch)
    input_ids = torch.full((len(batch), max_len), int(pad_id), dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    tau = torch.tensor([row["tau"] for row in batch], dtype=torch.float)
    for idx, row in enumerate(batch):
        ids = [int(x) for x in row["state_ids"]]
        input_ids[idx, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[idx, : len(ids)] = True
    return {"input_ids": input_ids, "attention_mask": attention_mask, "tau": tau}


def rankdata(values):
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0 for _ in values]
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + j - 1) / 2.0
        for k in range(i, j):
            ranks[order[k]] = rank
        i = j
    return ranks


def spearman(preds, labels):
    if len(preds) < 2:
        return 0.0
    rx = rankdata(preds)
    ry = rankdata(labels)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    num = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
    den_x = sum((x - mx) ** 2 for x in rx) ** 0.5
    den_y = sum((y - my) ** 2 for y in ry) ** 0.5
    if den_x <= 0 or den_y <= 0:
        return 0.0
    return num / (den_x * den_y)


def pairwise_auc(preds, labels, max_pairs=20000, seed=2026):
    if len(preds) < 2:
        return 0.0
    rng = random.Random(seed)
    correct = 0
    total = 0
    for _ in range(min(int(max_pairs), len(preds) * (len(preds) - 1) // 2)):
        i = rng.randrange(len(preds))
        j = rng.randrange(len(preds))
        if i == j or labels[i] == labels[j]:
            continue
        total += 1
        if (preds[i] - preds[j]) * (labels[i] - labels[j]) > 0:
            correct += 1
        elif preds[i] == preds[j]:
            correct += 0.5
    return correct / max(1, total)


def move_batch(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def positive_weight(rows):
    positives = sum(1 for row in rows if float(row.get("reward", row.get("tau", 0.0))) > 0.5)
    negatives = len(rows) - positives
    if positives <= 0:
        return 1.0
    return negatives / positives


def predict_values(model, batch, loss_mode):
    logits = model(batch["input_ids"], batch["attention_mask"])
    if loss_mode == "bce":
        return torch.sigmoid(logits), logits
    pred = torch.exp(logits.clamp(-4.0, 4.0))
    return pred, pred


def training_loss(pred, logits, labels, loss_mode, pos_weight):
    if loss_mode == "bce":
        weight = torch.tensor(float(pos_weight), dtype=torch.float, device=labels.device)
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, pos_weight=weight)
    if loss_mode == "huber":
        return torch.nn.functional.huber_loss(pred, labels, delta=1.0)
    return torch.nn.functional.mse_loss(pred, labels)


def evaluate(model, loader, device, seed, loss_mode, pos_weight):
    model.eval()
    preds = []
    labels = []
    losses = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            pred, logits = predict_values(model, batch, loss_mode)
            loss = training_loss(pred, logits, batch["tau"], loss_mode, pos_weight)
            losses.append(float(loss.item()))
            preds.extend(pred.detach().cpu().tolist())
            labels.extend(batch["tau"].detach().cpu().tolist())
    if not labels:
        return {"mse": 0.0, "spearman": 0.0, "pairwise_auc": 0.0, "mean_pred": 0.0, "mean_tau": 0.0}
    mse = sum((p - y) ** 2 for p, y in zip(preds, labels)) / len(labels)
    return {
        "loss": sum(losses) / max(1, len(losses)),
        "mse": mse,
        "spearman": spearman(preds, labels),
        "pairwise_auc": pairwise_auc(preds, labels, seed=seed),
        "mean_pred": sum(preds) / len(preds),
        "mean_tau": sum(labels) / len(labels),
    }


def infer_config(train_rows, args):
    vocab_size = args.vocab_size
    mask_id = args.mask_id
    for row in train_rows:
        if vocab_size is None and row.get("vocab_size") is not None:
            vocab_size = int(row["vocab_size"])
        if mask_id is None and row.get("mask_id") is not None:
            mask_id = int(row["mask_id"])
    if vocab_size is None or mask_id is None:
        raise ValueError("vocab_size and mask_id must be provided or present in rows")
    return {
        "vocab_size": int(vocab_size),
        "mask_id": int(mask_id),
        "pad_id": int(args.pad_id if args.pad_id is not None else vocab_size),
        "max_len": int(args.max_len),
        "d_model": int(args.d_model),
        "n_layers": int(args.n_layers),
        "n_heads": int(args.n_heads),
        "ffn_dim": int(args.ffn_dim),
        "dropout": float(args.dropout),
        "pooling": args.pooling,
    }


def train_from_config(config, device=None):
    data_path, output_dir, train_cfg = verifier_training_config(config, device)
    verifier_cfg = dict(config.get("verifier", {}))
    output_dir = Path(output_dir)
    args = SimpleNamespace(
        train=data_path,
        val=train_cfg.get("val_data", data_path),
        output=str(output_dir / "best.pt"),
        metrics=str(output_dir / "metrics.json"),
        vocab_size=verifier_cfg.get("vocab_size"),
        mask_id=verifier_cfg.get("mask_id"),
        pad_id=verifier_cfg.get("pad_id"),
        max_len=verifier_cfg.get("max_len", verifier_cfg.get("context_length", 64)),
        d_model=verifier_cfg.get("d_model", verifier_cfg.get("hidden_dim", 192)),
        n_layers=verifier_cfg.get("n_layers", verifier_cfg.get("num_layers", 4)),
        n_heads=verifier_cfg.get("n_heads", verifier_cfg.get("num_heads", 6)),
        ffn_dim=verifier_cfg.get("ffn_dim", 768),
        dropout=verifier_cfg.get("dropout", 0.1),
        pooling=verifier_cfg.get("pooling", "mean"),
        batch_size=train_cfg.get("batch_size", 512),
        global_batch_size=train_cfg.get("global_batch_size"),
        eval_batch_size=train_cfg.get("eval_batch_size", 1024),
        epochs=train_cfg.get("epochs", 10),
        lr=train_cfg.get("lr", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
        grad_clip=train_cfg.get("grad_clip", 1.0),
        loss=train_cfg.get("loss", "huber"),
        pos_weight=train_cfg.get("pos_weight", "auto"),
        device=train_cfg.get("device"),
        seed=train_cfg.get("seed", 2026),
    )
    return train_from_args(args)


def train_from_args(args):
    dist_info = init_distributed()
    set_seed(args.seed + int(dist_info["rank"]))
    if args.device is None and dist_info["distributed"]:
        args.device = f"cuda:{dist_info['local_rank']}"
    device = get_device(args.device)
    train_payload_rows = read_rollout(args.train)
    if args.val == args.train:
        train_rows = [row for row in train_payload_rows if row.get("split", "train") == "train"] or train_payload_rows
        val_rows = [row for row in train_payload_rows if row.get("split") == "val"] or train_rows
    else:
        train_rows = [row for row in train_payload_rows if row.get("split", "train") == "train"] or train_payload_rows
        val_payload_rows = read_rollout(args.val)
        val_rows = [row for row in val_payload_rows if row.get("split") == "val"] or val_payload_rows
    config = infer_config(train_rows, args)
    model = SmallDNAValueVerifier(**config).to(device)
    train_model = model
    if dist_info["distributed"]:
        train_model = DistributedDataParallel(model, device_ids=[dist_info["local_rank"]] if device.type == "cuda" else None)
    loss_mode = str(args.loss or "huber")
    if loss_mode not in {"huber", "mse", "bce"}:
        raise ValueError(f"unknown DNA verifier loss: {loss_mode}")
    pos_weight = positive_weight(train_rows) if args.pos_weight == "auto" else float(args.pos_weight)
    train_batch_size = per_rank_batch_size({"global_batch_size": args.global_batch_size if args.global_batch_size is not None else args.batch_size}, dist_info["world_size"])
    train_dataset = StateDataset(train_rows)
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if dist_info["distributed"] else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=lambda batch: collate(batch, config["pad_id"]),
    )
    val_loader = DataLoader(
        StateDataset(val_rows),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate(batch, config["pad_id"]),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = None
    history = []
    for epoch in range(1, int(args.epochs) + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_model.train()
        total = 0.0
        seen = 0
        iterator = tqdm(train_loader, desc=f"dna verifier epoch {epoch}", disable=dist_info["rank"] != 0)
        for batch in iterator:
            batch = move_batch(batch, device)
            pred, logits = predict_values(train_model, batch, loss_mode)
            loss = training_loss(pred, logits, batch["tau"], loss_mode, pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            opt.step()
            total += float(loss.item()) * int(batch["tau"].shape[0])
            seen += int(batch["tau"].shape[0])
        local = torch.tensor([total, seen], dtype=torch.float64, device=device)
        if dist_info["distributed"]:
            torch.distributed.all_reduce(local, op=torch.distributed.ReduceOp.SUM)
        if dist_info["rank"] == 0:
            val = evaluate(model, val_loader, device, args.seed + epoch, loss_mode, pos_weight)
            row = {"epoch": epoch, "pos_weight": float(pos_weight), "train_loss": float(local[0].item() / max(1.0, local[1].item()))}
            row.update({f"val_{key}": value for key, value in val.items()})
            history.append(row)
            if loss_mode == "bce":
                is_best = best is None or val["pairwise_auc"] > best["val_pairwise_auc"]
            else:
                is_best = best is None or val["mse"] < best["val_mse"]
            if is_best:
                best = {"epoch": epoch, "val_mse": val["mse"], "val_pairwise_auc": val["pairwise_auc"]}
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "verifier_type": "small_dna_value",
                        "value_mode": "binary_logprob" if loss_mode == "bce" else "exp",
                        "config": config,
                        "state_dict": model.state_dict(),
                        "metrics": row,
                        "eval_batch_size": int(args.eval_batch_size),
                    },
                    args.output,
                )
            Path(args.metrics).parent.mkdir(parents=True, exist_ok=True)
            with open(args.metrics, "w", encoding="utf-8") as f:
                json.dump({"best": best, "history": history}, f, indent=2, sort_keys=True)
            print(json.dumps(row, sort_keys=True))
        barrier(dist_info)
    result = {"best": best, "output": args.output, "metrics": args.metrics}
    if dist_info["rank"] == 0:
        print(json.dumps(result, sort_keys=True))
    cleanup_distributed(dist_info)
    return result if dist_info["rank"] == 0 else {"rank": dist_info["rank"], "done": True}
