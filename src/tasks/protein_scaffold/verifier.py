import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProteinStateValueVerifier(nn.Module):
    def __init__(
        self,
        vocab_size=20,
        max_length=256,
        num_tasks=1,
        dim=192,
        layers=4,
        heads=4,
        dropout=0.1,
        pooling="scaffold",
        output_mode="logv",
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.mask_token = int(vocab_size)
        self.pad_token = int(vocab_size) + 1
        self.max_length = int(max_length)
        self.num_tasks = max(1, int(num_tasks))
        self.pooling = str(pooling or "scaffold")
        self.output_mode = str(output_mode or "logv")
        self.config = {
            "vocab_size": self.vocab_size,
            "max_length": self.max_length,
            "num_tasks": self.num_tasks,
            "dim": int(dim),
            "layers": int(layers),
            "heads": int(heads),
            "dropout": float(dropout),
            "pooling": self.pooling,
            "output_mode": self.output_mode,
        }
        self.task_to_idx = {"task0": 0}
        self.batch_size = 256
        self.stats = {"requests": 0, "state_evals": 0, "batch_calls": 0}
        self.token_embed = nn.Embedding(int(vocab_size) + 2, int(dim))
        self.position_embed = nn.Embedding(int(max_length), int(dim))
        self.task_embed = nn.Embedding(self.num_tasks, int(dim))
        self.flag_proj = nn.Linear(3, int(dim))
        layer = nn.TransformerEncoderLayer(
            d_model=int(dim),
            nhead=int(heads),
            dim_feedforward=int(dim) * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(layers), enable_nested_tensor=False)
        self.norm = nn.LayerNorm(int(dim))
        pooled_dim = int(dim)
        if self.pooling == "triple":
            pooled_dim = int(dim) * 3
        self.out = nn.Sequential(
            nn.Linear(pooled_dim + 1, int(dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(dim), 1),
        )

    def forward(self, states, is_motif, is_editable, task_ids, valid_mask=None):
        states = states.long()
        if valid_mask is None:
            valid_mask = torch.ones_like(states, dtype=torch.bool)
        token_ids = states.clone()
        token_ids = torch.where(token_ids < 0, torch.full_like(token_ids, self.mask_token), token_ids)
        token_ids = torch.where(valid_mask, token_ids, torch.full_like(token_ids, self.pad_token))
        token_ids = token_ids.clamp(min=0, max=self.pad_token)

        length = states.shape[1]
        if length > self.max_length:
            raise ValueError(f"sequence length {length} exceeds max_length {self.max_length}")
        position_ids = torch.arange(length, device=states.device).unsqueeze(0).expand_as(states)
        is_mask = states.eq(-1) & valid_mask
        flags = torch.stack(
            [
                is_motif.float(),
                is_editable.float(),
                is_mask.float(),
            ],
            dim=-1,
        )
        task_ids = task_ids.long().clamp(min=0, max=self.num_tasks - 1)
        x = (
            self.token_embed(token_ids)
            + self.position_embed(position_ids)
            + self.task_embed(task_ids)[:, None, :]
            + self.flag_proj(flags)
        )
        pad_mask = ~valid_mask.bool()
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        x = self.norm(x)
        scaffold_mask = is_editable.bool() & valid_mask.bool()
        motif_mask = is_motif.bool() & valid_mask.bool()
        global_mask = valid_mask.bool()

        def masked_mean(mask):
            denom = mask.float().sum(dim=1).clamp_min(1.0)
            return (x * mask[:, :, None].float()).sum(dim=1) / denom[:, None]

        scaffold_pool = masked_mean(scaffold_mask)
        empty_scaffold = scaffold_mask.float().sum(dim=1).eq(0)
        if empty_scaffold.any():
            scaffold_pool = torch.where(empty_scaffold[:, None], masked_mean(global_mask), scaffold_pool)
        if self.pooling == "triple":
            motif_pool = masked_mean(motif_mask)
            empty_motif = motif_mask.float().sum(dim=1).eq(0)
            if empty_motif.any():
                motif_pool = torch.where(empty_motif[:, None], masked_mean(global_mask), motif_pool)
            pooled = torch.cat([scaffold_pool, motif_pool, masked_mean(global_mask)], dim=-1)
        else:
            pooled = scaffold_pool
        mask_ratio = (is_mask.float() * is_editable.float()).sum(dim=1) / is_editable.float().sum(dim=1).clamp_min(1.0)
        raw = self.out(torch.cat([pooled, mask_ratio[:, None]], dim=-1)).squeeze(-1)
        if self.output_mode == "logit":
            return torch.clamp(raw, min=-20.0, max=20.0)
        logv = -F.softplus(raw)
        logv = torch.clamp(logv, min=-20.0, max=0.0)
        return logv

    def tensor_value(self, states, is_motif, is_editable, task_ids, valid_mask=None):
        if self.output_mode == "logit":
            return torch.sigmoid(self.forward(states, is_motif, is_editable, task_ids, valid_mask=valid_mask))
        return torch.exp(self.forward(states, is_motif, is_editable, task_ids, valid_mask=valid_mask))

    def value(self, states_or_example, is_motif_or_state=None, is_editable_or_harness=None, task_ids=None, valid_mask=None):
        if torch.is_tensor(states_or_example):
            return self.tensor_value(states_or_example, is_motif_or_state, is_editable_or_harness, task_ids, valid_mask=valid_mask)
        return self.values([states_or_example], [is_motif_or_state], is_editable_or_harness)[0]

    @classmethod
    def from_checkpoint(cls, checkpoint, device=None):
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = dict(payload["config"])
        model = cls(
            vocab_size=config.get("vocab_size", 20),
            max_length=config.get("max_length", 256),
            num_tasks=config.get("num_tasks", 1),
            dim=config.get("dim", 192),
            layers=config.get("layers", 4),
            heads=config.get("heads", 4),
            dropout=config.get("dropout", 0.1),
            pooling=config.get("pooling", "scaffold"),
            output_mode=config.get("output_mode", "logv"),
        )
        model.load_state_dict(payload["state_dict"])
        model.task_to_idx = dict(payload.get("task_to_idx") or config.get("task_to_idx") or {"task0": 0})
        model.batch_size = int(payload.get("batch_size", 256))
        model.to(device)
        model.eval()
        return model

    def reset_stats(self):
        for key in self.stats:
            self.stats[key] = 0

    def get_stats(self):
        return dict(self.stats)

    def _row(self, example, state):
        row = dict(example)
        row["state"] = [int(x) for x in state]
        return row

    def values(self, examples, states, harness):
        del harness
        rows = [self._row(example, state) for example, state in zip(examples, states)]
        self.stats["requests"] += len(rows)
        self.stats["state_evals"] += len(rows)
        self.stats["batch_calls"] += math.ceil(len(rows) / max(1, int(self.batch_size))) if rows else 0
        device = next(self.parameters()).device
        return values_for_rows(self, rows, self.task_to_idx, device, batch_size=self.batch_size)


def task_key(row):
    return str(row.get("task_id") or row.get("pdb") or "task0")


def build_batch(rows, task_to_idx, device, pad_value=-2):
    max_len = max(len(row["state"]) for row in rows)
    states = []
    motif = []
    editable = []
    valid = []
    task_ids = []
    for row in rows:
        state = [int(x) for x in row["state"]]
        n = len(state)
        motif_positions = set(int(x) for x in row.get("motif_positions", []))
        state_pad = state + [int(pad_value)] * (max_len - n)
        motif_row = [idx in motif_positions for idx in range(n)] + [False] * (max_len - n)
        valid_row = [True] * n + [False] * (max_len - n)
        editable_row = [(idx not in motif_positions) for idx in range(n)] + [False] * (max_len - n)
        task = task_key(row)
        states.append(state_pad)
        motif.append(motif_row)
        editable.append(editable_row)
        valid.append(valid_row)
        task_ids.append(int(task_to_idx.get(task, 0)))
    return {
        "states": torch.tensor(states, dtype=torch.long, device=device),
        "is_motif": torch.tensor(motif, dtype=torch.bool, device=device),
        "is_editable": torch.tensor(editable, dtype=torch.bool, device=device),
        "valid_mask": torch.tensor(valid, dtype=torch.bool, device=device),
        "task_ids": torch.tensor(task_ids, dtype=torch.long, device=device),
    }


def values_for_rows(model, rows, task_to_idx, device, batch_size=256):
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(rows), int(batch_size)):
            batch_rows = rows[start:start + int(batch_size)]
            batch = build_batch(batch_rows, task_to_idx, device)
            values = model.tensor_value(
                batch["states"],
                batch["is_motif"],
                batch["is_editable"],
                batch["task_ids"],
                valid_mask=batch["valid_mask"],
            )
            out.extend(values.detach().float().cpu().tolist())
    return out


def load(checkpoint=None, device=None, **kwargs):
    del kwargs
    if not checkpoint:
        raise ValueError("Protein scaffold VGB requires verifier.checkpoint")
    return ProteinStateValueVerifier.from_checkpoint(checkpoint, device=device)
