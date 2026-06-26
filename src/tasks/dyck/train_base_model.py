
import copy
import itertools
import json
import math
import pickle
import random
import shutil
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm
from transformers import BertConfig, BertForMaskedLM

from .model import BOS_ID, EOS_ID, MASK_ID, VOCAB_SIZE
from .prepare_data import make_split, normalize_probs, save_pickle
from utils import (
    barrier,
    cleanup_distributed,
    init_distributed,
    per_rank_batch_size,
    repo_root,
    require_known_keys,
    set_seed,
)


DATA_KEYS = {"dev_path", "train_path"}
TRAINING_KEYS = {
    "checkpoint_name",
    "dev_count",
    "final_checkpoint_name",
    "global_batch_size",
    "hidden_dim",
    "kind",
    "length",
    "lr",
    "mask_prob",
    "max_depth",
    "num_heads",
    "num_iters",
    "num_layers",
    "num_types",
    "output_dir",
    "prefix_length",
    "seed",
    "train_count",
    "type_probs",
    "validation_every_steps",
    "warmup",
    "weight_decay",
}


def _path(path):
    path = Path(path)
    return path if path.is_absolute() else repo_root() / path


def _load_pickle(path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def _data_config(config):
    return dict(config.get("data", {}).get("base_model", {}))


def _training_config(config):
    return dict(config.get("training", {}).get("base_model", {}))


def _load_or_generate_split(name, path, count, seed, train_cfg):
    if path.exists():
        return _load_pickle(path)
    type_probs = normalize_probs(train_cfg.get("type_probs"), int(train_cfg.get("num_types", 2)))
    rows = make_split(
        name,
        int(count),
        int(seed),
        int(train_cfg.get("length", 32)),
        int(train_cfg.get("num_types", 2)),
        type_probs,
        int(train_cfg.get("max_depth", 12)),
    )
    save_pickle(path, rows)
    return rows


def _batch_iter(samples, batch_size, shuffle, rng):
    indices = list(range(len(samples)))
    if shuffle:
        rng.shuffle(indices)
    for start in range(0, len(indices), int(batch_size)):
        yield [samples[idx] for idx in indices[start:start + int(batch_size)]]


def _make_generator(device, seed):
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def _mask_batch(batch, prefix_length, mask_prob, device, generator, num_mask=None):
    seqs = torch.tensor([sample["tokens"] for sample in batch], dtype=torch.long, device=device)
    labels = torch.full_like(seqs, -100)
    inputs = seqs.clone()

    start = int(prefix_length) + 1
    suffix_len = seqs.shape[1] - start
    if suffix_len <= 0:
        raise ValueError("prefix_length leaves no maskable Dyck suffix")

    batch_size = seqs.shape[0]
    if num_mask is None:
        suffix_mask = torch.rand((batch_size, suffix_len), device=device, generator=generator) < float(mask_prob)
        empty_rows = ~suffix_mask.any(dim=1)
        if empty_rows.any():
            picks = torch.randint(suffix_len, (int(empty_rows.sum().item()),), device=device, generator=generator)
            suffix_mask[empty_rows] = False
            suffix_mask[empty_rows, picks] = True
    else:
        k = min(int(num_mask), suffix_len)
        scores = torch.rand((batch_size, suffix_len), device=device, generator=generator)
        selected = scores.topk(k, dim=1).indices
        suffix_mask = torch.zeros((batch_size, suffix_len), dtype=torch.bool, device=device)
        suffix_mask.scatter_(1, selected, True)

    mask = torch.zeros_like(seqs, dtype=torch.bool)
    mask[:, start:] = suffix_mask
    labels[mask] = seqs[mask]
    inputs[mask] = MASK_ID
    return inputs, labels


def _loss_and_acc(model, input_ids, labels):
    logits = model(input_ids).logits
    mask = labels != -100
    if not mask.any():
        return None, 0, 0
    masked_logits = logits[mask]
    masked_labels = labels[mask]
    loss = torch.nn.functional.cross_entropy(masked_logits, masked_labels)
    correct = (masked_logits.argmax(dim=-1) == masked_labels).sum().item()
    return loss, correct, masked_labels.numel()


@torch.no_grad()
def _evaluate(model, rows, train_cfg, device, seed):
    model.eval()
    generator = _make_generator(device, int(seed) + 12345)
    correct = 0
    total = 0
    batch_size = int(train_cfg.get("global_batch_size", 32))
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        input_ids, labels = _mask_batch(
            batch,
            prefix_length=int(train_cfg.get("prefix_length", 16)),
            mask_prob=float(train_cfg.get("mask_prob", 0.3)),
            device=device,
            generator=generator,
            num_mask=1,
        )
        _, batch_correct, batch_total = _loss_and_acc(model, input_ids, labels)
        correct += batch_correct
        total += batch_total
    model.train()
    return correct / total if total else 0.0


def _build_model(train_cfg):
    model_config = BertConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=int(train_cfg.get("hidden_dim", 512)),
        intermediate_size=int(train_cfg.get("hidden_dim", 512)) * 2,
        num_hidden_layers=int(train_cfg.get("num_layers", 6)),
        num_attention_heads=int(train_cfg.get("num_heads", 8)),
        max_position_embeddings=int(train_cfg.get("length", 32)) + 2,
        pad_token_id=0,
        bos_token_id=BOS_ID,
        eos_token_id=EOS_ID,
        type_vocab_size=1,
    )
    return BertForMaskedLM(model_config)


def _scheduler(optimizer, warmup, total_steps):
    warmup = int(warmup)
    total_steps = int(total_steps)

    def lr_lambda(step_idx):
        if warmup > 0 and step_idx < warmup:
            return (step_idx + 1) / warmup
        if total_steps <= warmup:
            return 1.0
        return max((total_steps - (step_idx + 1)) / (total_steps - warmup), 0.0)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _save_model(path, model):
    raw_model = model.module if hasattr(model, "module") else model
    path.parent.mkdir(parents=True, exist_ok=True)
    save_pickle(path, copy.deepcopy(raw_model).cpu())


def train_from_config(config, device=None):
    info = init_distributed()
    data_cfg = _data_config(config)
    train_cfg = _training_config(config)
    require_known_keys(data_cfg, DATA_KEYS, "Dyck base-model data")
    require_known_keys(train_cfg, TRAINING_KEYS, "Dyck base-model training")
    if str(train_cfg.get("kind", "aoar")) != "aoar":
        raise ValueError("Dyck base-model training currently matches the vgb_package AOAR LM only.")

    try:
        rank = int(info["rank"])
        world_size = int(info["world_size"])
        seed = int(train_cfg.get("seed", config.get("seed", 0)))
        set_seed(seed + rank)
        device_obj = torch.device(device or info.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))

        train_path = _path(data_cfg.get("train_path", "data/dyck/base_dataset/train.pkl"))
        dev_path = _path(data_cfg.get("dev_path", "data/dyck/base_dataset/dev.pkl"))
        if info.get("distributed") and rank != 0:
            barrier(info)
        train_rows = _load_or_generate_split("train", train_path, int(train_cfg.get("train_count", 300000)), seed, train_cfg)
        dev_rows = _load_or_generate_split("dev", dev_path, int(train_cfg.get("dev_count", 10000)), seed + 1, train_cfg)
        if info.get("distributed") and rank == 0:
            barrier(info)

        model = _build_model(train_cfg).to(device_obj)
        if info.get("distributed"):
            device_ids = [int(info["local_rank"])] if device_obj.type == "cuda" else None
            model = DDP(model, device_ids=device_ids)
        raw_model = model.module if isinstance(model, DDP) else model
        model.train()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(train_cfg.get("lr", 3e-4)),
            weight_decay=float(train_cfg.get("weight_decay", 0.1)),
        )
        batch_size = per_rank_batch_size(train_cfg, world_size)
        total_steps = int(train_cfg.get("num_iters", 0))
        if total_steps <= 0:
            total_steps = max(math.ceil(len(train_rows) / int(train_cfg.get("global_batch_size", 32))), 1)
        local_rows = train_rows[rank::world_size]
        if not local_rows:
            raise RuntimeError(f"rank {rank} received no Dyck training examples")
        batches = itertools.cycle(_batch_iter(local_rows, batch_size, shuffle=True, rng=random.Random(seed + rank)))
        scheduler = _scheduler(optimizer, int(train_cfg.get("warmup", 100)), total_steps)

        train_generator = _make_generator(device_obj, seed + 999 + rank)
        eval_every = int(train_cfg.get("validation_every_steps", 100))
        output_dir = _path(train_cfg.get("output_dir", "model_data/dyck/base_model"))
        best_path = output_dir / str(train_cfg.get("checkpoint_name", "aoar_best.pkl"))
        final_path = output_dir / str(train_cfg.get("final_checkpoint_name", "aoar_final.pkl"))
        best_acc = float("-inf")
        last_loss = 0.0
        last_eval = 0.0

        progress = tqdm(range(total_steps), total=total_steps, desc="dyck:aoar", disable=rank != 0)
        for step_idx in progress:
            batch = next(batches)
            input_ids, labels = _mask_batch(
                batch,
                prefix_length=int(train_cfg.get("prefix_length", 16)),
                mask_prob=float(train_cfg.get("mask_prob", 0.3)),
                device=device_obj,
                generator=train_generator,
            )
            loss, _, _ = _loss_and_acc(model, input_ids, labels)
            if loss is None:
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            last_loss = float(loss.item())
            if rank == 0:
                progress.set_postfix(loss=f"{last_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}", best=f"{max(best_acc, 0.0):.4f}")

            step = step_idx + 1
            if step % eval_every == 0 or step == total_steps:
                if rank == 0:
                    last_eval = _evaluate(raw_model, dev_rows, train_cfg, device_obj, seed)
                    if last_eval > best_acc:
                        best_acc = last_eval
                        _save_model(best_path, raw_model)
                    print(json.dumps({"step": step, "train_loss": last_loss, "eval_acc": last_eval, "best_eval_acc": best_acc}), flush=True)
                barrier(info)
                model.train()

        if rank == 0:
            _save_model(final_path, raw_model)
            if best_path != output_dir / "aoar_best.pkl":
                shutil.copyfile(best_path, output_dir / "aoar_best.pkl")
        barrier(info)
        return {"output": str(best_path), "final": str(final_path), "best_eval_acc": float(best_acc), "final_eval_acc": float(last_eval)}
    finally:
        cleanup_distributed(info)
