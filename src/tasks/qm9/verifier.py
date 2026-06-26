import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class QM9Verifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_length,
        hidden_dim=256,
        num_layers=4,
        num_heads=4,
        dropout=0.1,
        output_activation="sigmoid",
    ):
        super().__init__()
        self.verifier_kind = "token_transformer"
        self.max_length = int(max_length)
        self.output_activation = str(output_activation)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)
        hidden_dim = int(hidden_dim)
        self.token_emb = nn.Embedding(int(vocab_size), hidden_dim)
        self.pos_emb = nn.Embedding(self.max_length, hidden_dim)
        self.batch_size = 8192
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=self.dropout,
            batch_first=True,
            activation="gelu",
        )
        try:
            self.encoder = nn.TransformerEncoder(layer, num_layers=self.num_layers, enable_nested_tensor=False)
        except TypeError:
            self.encoder = nn.TransformerEncoder(layer, num_layers=self.num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, input_ids):
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        x = self.token_emb(input_ids.long()) + self.pos_emb(positions)[None, :, :]
        x = self.encoder(x)
        score = self.head(x.mean(dim=1)).squeeze(-1)
        if self.output_activation == "softplus":
            return F.softplus(score)
        if self.output_activation == "identity":
            return score
        return torch.sigmoid(score)

    @classmethod
    def from_checkpoint(cls, checkpoint, device=None, batch_size=8192):
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("verifier_kind", "token_transformer") != "token_transformer":
            raise ValueError("QM9 only supports token_transformer verifier checkpoints")
        model = cls(
            int(payload["vocab_size"]),
            int(payload["max_length"]),
            hidden_dim=int(payload.get("hidden_dim", 256)),
            num_layers=int(payload.get("num_layers", 4)),
            num_heads=int(payload.get("num_heads", 4)),
            dropout=float(payload.get("dropout", 0.1)),
            output_activation=str(payload.get("output_activation", "sigmoid")),
        )
        model.load_state_dict(payload["state_dict"])
        model.batch_size = int(payload.get("eval_batch_size", batch_size))
        model.to(device)
        model.eval()
        return model

    @torch.no_grad()
    def values(self, examples, states, harness):
        del examples, harness
        if not states:
            return []
        device = next(self.parameters()).device
        out = []
        for start in range(0, len(states), max(1, self.batch_size)):
            chunk = states[start:start + self.batch_size]
            ids = torch.tensor(chunk, dtype=torch.long, device=device)
            if ids.ndim != 2 or ids.shape[1] != self.max_length:
                raise ValueError(f"QM9 verifier expects states with length {self.max_length}")
            out.extend(float(score) for score in self(ids).detach().cpu().tolist())
        return out

    def value(self, example, state, harness):
        return self.values([example], [state], harness)[0]


def save_verifier(model, path, output_activation="sigmoid"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw_model = model.module if hasattr(model, "module") else model
    torch.save(
        {
            "verifier_kind": "token_transformer",
            "state_dict": raw_model.state_dict(),
            "vocab_size": int(raw_model.token_emb.num_embeddings),
            "max_length": int(raw_model.max_length),
            "hidden_dim": int(raw_model.token_emb.embedding_dim),
            "num_layers": int(raw_model.num_layers),
            "num_heads": int(raw_model.num_heads),
            "dropout": float(raw_model.dropout),
            "output_activation": str(output_activation),
        },
        path,
    )


def load(checkpoint=None, device=None, harness=None, batch_size=8192):
    del harness
    if not checkpoint:
        raise ValueError("QM9 VGB requires verifier.checkpoint")
    return QM9Verifier.from_checkpoint(checkpoint, device=device, batch_size=batch_size)
