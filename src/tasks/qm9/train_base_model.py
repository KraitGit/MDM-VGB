
import json
import math
import random
from pathlib import Path

import datasets
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

from .harness import loglinear_sigma, subs_log_probs
from utils import cleanup_distributed, init_distributed, per_rank_batch_size, repo_root, set_seed


def _path(path):
    path = Path(path)
    return path if path.is_absolute() else repo_root() / path


def _training_config(config):
    return dict(config.get("training", {}).get("base_model", {}))


def _load_qm9_dataset(config, eval_split_size):
    data_cfg = dict(config.get("data", {}))
    dataset_dir = _path(data_cfg.get("dataset_dir", "data/qm9/dataset"))
    cache_dir = _path(data_cfg.get("cache_dir", "data/qm9/cache"))
    if dataset_dir.exists():
        dataset = datasets.load_from_disk(str(dataset_dir))
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)
        dataset = datasets.load_dataset("yairschiff/qm9", split="train", cache_dir=str(cache_dir))
    split = dataset.train_test_split(test_size=float(eval_split_size), seed=42)
    return split["train"]


def _tokenize(tokenizer, smiles, seq_len):
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    return tokenizer(
        smiles,
        max_length=int(seq_len),
        padding="max_length",
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )


def _build_scheduler(optimizer, lr, min_lr, train_steps, warmup_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return max(1e-8, step / max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, train_steps - warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
        min_ratio = min_lr / lr
        return min_ratio + (1 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _batch_size(train_cfg, world_size):
    return per_rank_batch_size({"global_batch_size": train_cfg.get("global_batch_size", 64)}, world_size)


def _load_model_and_tokenizer(config, train_cfg, device):
    model_cfg = dict(config.get("model", {}))
    tokenizer_path = _path(model_cfg["tokenizer_path"])
    model_source = _path(train_cfg.get("model_source", model_cfg["model_name"]))
    seq_len = int(model_cfg.get("seq_len", train_cfg.get("seq_len", 32)))

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)
    if bool(train_cfg.get("init_from_scratch", True)):
        model_config = AutoConfig.from_pretrained(str(model_source), trust_remote_code=True)
        model_config.vocab_size = int(tokenizer.vocab_size)
        model_config.model_length = seq_len
        model_config.hidden_dim = int(train_cfg.get("hidden_dim", 768))
        model_config.cond_dim = int(train_cfg.get("cond_dim", 128))
        model_config.n_blocks = int(train_cfg.get("n_blocks", 12))
        model_config.n_heads = int(train_cfg.get("n_heads", 12))
        model_config.dropout = float(train_cfg.get("dropout", 0.1))
        model_config.time_conditioning = str(train_cfg.get("model_kind", "mdlm")) == "udlm"
        model_config.return_dict = True
        model = AutoModelForMaskedLM.from_config(model_config, trust_remote_code=True)
    else:
        model = AutoModelForMaskedLM.from_pretrained(str(model_source), trust_remote_code=True)

    model.to(device)
    model.train()
    return model, tokenizer, seq_len


def _train_step(model, tokenizer, smiles, optimizer, scheduler, device, use_bf16, seq_len):
    x0 = _tokenize(tokenizer, smiles, seq_len)["input_ids"].to(device)
    batch_size = x0.shape[0]
    t = torch.rand(batch_size, device=device).clamp_min(1e-3)
    sigma = loglinear_sigma(t)
    move_chance = 1 - torch.exp(-sigma[:, None])
    corrupt = torch.rand_like(x0.float()) < move_chance
    xt = torch.where(corrupt, int(tokenizer.mask_token_id), x0)

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
        out = model(input_ids=xt, timesteps=sigma, return_dict=True)
        log_probs = subs_log_probs(out.logits, xt, int(tokenizer.mask_token_id))
        token_loss = -torch.gather(log_probs, -1, x0[:, :, None]).squeeze(-1)
        loss = (token_loss / t[:, None]).sum() / (x0.shape[0] * x0.shape[1])

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    return float(loss.item())


def train_from_config(config, device = None):
    info = init_distributed()
    train_cfg = _training_config(config)
    model_kind = str(train_cfg.get("model_kind", "mdlm"))
    if model_kind != "mdlm":
        cleanup_distributed(info)
        raise ValueError("QM9 base-model training supports model_kind=mdlm.")

    device_name = device or info.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    device_obj = torch.device(device_name)
    rank = int(info["rank"])
    world_size = int(info["world_size"])

    try:
        seed = int(train_cfg.get("seed", config.get("seed", 1)))
        set_seed(seed + rank)
        model, tokenizer, seq_len = _load_model_and_tokenizer(config, train_cfg, device_obj)
        if info.get("distributed"):
            device_ids = [int(info["local_rank"])] if device_obj.type == "cuda" else None
            model = DDP(model, device_ids=device_ids)
            raw_model = model.module
        else:
            raw_model = model

        train_ds = _load_qm9_dataset(config, float(train_cfg.get("eval_split_size", 0.05)))
        smiles = list(train_ds["canonical_smiles"])
        per_rank = len(smiles) // world_size
        start = rank * per_rank
        end = len(smiles) if rank == world_size - 1 else (rank + 1) * per_rank
        local_smiles = smiles[start:end]
        if not local_smiles:
            raise RuntimeError(f"rank {rank} received no QM9 training examples")

        lr = float(train_cfg.get("lr", 3e-4))
        train_steps = int(train_cfg.get("train_steps", 25_000))
        optimizer = torch.optim.AdamW(
            raw_model.parameters(),
            lr=lr,
            weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        )
        scheduler = _build_scheduler(
            optimizer,
            lr=lr,
            min_lr=float(train_cfg.get("min_lr", 3e-6)),
            train_steps=train_steps,
            warmup_steps=int(train_cfg.get("warmup_steps", 1000)),
        )
        batch_size = _batch_size(train_cfg, world_size)
        save_every = int(train_cfg.get("save_every", 5000))
        log_every = int(train_cfg.get("log_every", 20))
        use_bf16 = str(config.get("model", {}).get("dtype", train_cfg.get("dtype", "bf16"))) == "bf16" and device_obj.type == "cuda"
        output_dir = _path(train_cfg.get("output_dir", "model_data/qm9/mdlm_qm9"))

        for step in range(1, train_steps + 1):
            batch = random.sample(local_smiles, batch_size)
            loss = _train_step(model, tokenizer, batch, optimizer, scheduler, device_obj, use_bf16, seq_len)
            if rank == 0 and step % log_every == 0:
                print(json.dumps({"step": step, "loss": loss, "lr": scheduler.get_last_lr()[0]}), flush=True)
            if rank == 0 and save_every > 0 and step % save_every == 0:
                output_dir.mkdir(parents=True, exist_ok=True)
                raw_model.save_pretrained(output_dir / f"checkpoint-step-{step}")

        if rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            raw_model.save_pretrained(output_dir / "last")
    finally:
        cleanup_distributed(info)
