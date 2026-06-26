import torch
import torch.nn as nn

from tasks.common import ConstantVerifier


class TokenVerifier(nn.Module):
    def __init__(self, vocab_size, mask_id, max_length=512, d_model=256, num_layers=4, num_heads=4, dropout=0.25):
        super().__init__()
        self.config = {
            "vocab_size": vocab_size,
            "mask_id": mask_id,
            "max_length": max_length,
            "d_model": d_model,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "dropout": dropout,
        }
        self.token_emb = nn.Embedding(vocab_size + 1, d_model, padding_idx=mask_id)
        self.pos_emb = nn.Embedding(max_length, d_model)
        self.batch_size = 8192
        self.stats = {"requests": 0, "state_evals": 0, "batch_calls": 0}
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, input_ids, attention_mask=None):
        batch, length = input_ids.shape
        positions = torch.arange(length, device=input_ids.device).unsqueeze(0).expand(batch, length)
        x = self.token_emb(input_ids) + self.pos_emb(positions.clamp(max=self.config["max_length"] - 1))
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        if attention_mask is None:
            pooled = x.mean(dim=1)
        else:
            denom = attention_mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
            pooled = (x * attention_mask.unsqueeze(-1).float()).sum(dim=1) / denom
        return self.head(pooled).squeeze(-1)

    @classmethod
    def from_checkpoint(cls, checkpoint, device=None):
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = cls(**dict(payload["config"]))
        model.load_state_dict(payload["state_dict"])
        model.batch_size = int(payload.get("eval_batch_size", 8192))
        model.to(device)
        model.eval()
        return model

    def reset_stats(self):
        for key in self.stats:
            self.stats[key] = 0

    def get_stats(self):
        return dict(self.stats)

    @torch.no_grad()
    def values(self, examples, states, harness):
        del examples, harness
        if not states:
            return []
        self.stats["requests"] += len(states)
        device = next(self.parameters()).device
        out = []
        batch_size = max(1, int(self.batch_size))
        for start in range(0, len(states), batch_size):
            chunk = [[int(x) for x in state] for state in states[start:start + batch_size]]
            self.stats["state_evals"] += len(chunk)
            self.stats["batch_calls"] += 1
            max_len = max(len(state) for state in chunk)
            pad_id = int(self.config.get("mask_id", 0))
            ids = torch.full((len(chunk), max_len), pad_id, dtype=torch.long, device=device)
            mask = torch.zeros((len(chunk), max_len), dtype=torch.bool, device=device)
            for row, state in enumerate(chunk):
                ids[row, : len(state)] = torch.tensor(state, dtype=torch.long, device=device)
                mask[row, : len(state)] = True
            out.extend(float(score) for score in torch.sigmoid(self(ids, mask)).detach().cpu().tolist())
        return out

    def value(self, example, state, harness):
        return self.values([example], [state], harness)[0]


def load(checkpoint=None, device=None, constant=None, **kwargs):
    del kwargs
    if constant is not None:
        return ConstantVerifier(constant)
    if not checkpoint:
        raise ValueError("Dyck VGB requires verifier.checkpoint or verifier.constant")
    return TokenVerifier.from_checkpoint(checkpoint, device=device)
