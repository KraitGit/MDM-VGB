import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallDNAValueVerifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        mask_id,
        pad_id=None,
        max_len=64,
        d_model=192,
        n_layers=4,
        n_heads=6,
        ffn_dim=768,
        dropout=0.1,
        pooling="mean",
    ):
        super().__init__()
        if pad_id is None:
            pad_id = int(vocab_size)
        self.config = {
            "vocab_size": int(vocab_size),
            "mask_id": int(mask_id),
            "pad_id": int(pad_id),
            "max_len": int(max_len),
            "d_model": int(d_model),
            "n_layers": int(n_layers),
            "n_heads": int(n_heads),
            "ffn_dim": int(ffn_dim),
            "dropout": float(dropout),
            "pooling": str(pooling),
        }
        emb_size = max(int(vocab_size), int(mask_id) + 1, int(pad_id) + 1)
        self.token_emb = nn.Embedding(emb_size, d_model, padding_idx=int(pad_id))
        self.pos_emb = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, enable_nested_tensor=False)
        self.batch_size = 4096
        self.value_mode = "exp"
        self.cache = {}
        self.stats = {
            "requests": 0,
            "cache_hits": 0,
            "state_evals": 0,
            "batch_calls": 0,
        }
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
        positions = positions.clamp(max=self.config["max_len"] - 1)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        if self.config.get("pooling") == "cls":
            pooled = x[:, 0]
        else:
            if attention_mask is None:
                pooled = x.mean(dim=1)
            else:
                mask = attention_mask.float().unsqueeze(-1)
                pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.head(pooled).squeeze(-1)

    def log_value(self, input_ids, attention_mask=None):
        return self.forward(input_ids, attention_mask).clamp(-4.0, 4.0)

    def tensor_value(self, input_ids, attention_mask=None):
        return torch.exp(self.log_value(input_ids, attention_mask))

    def value(self, input_ids_or_example, attention_mask_or_state=None, harness=None):
        if torch.is_tensor(input_ids_or_example):
            return self.tensor_value(input_ids_or_example, attention_mask_or_state)
        return self.values([input_ids_or_example], [attention_mask_or_state], harness)[0]

    @classmethod
    def from_checkpoint(cls, path, device=None):
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = torch.load(path, map_location="cpu")
        model = cls(**dict(checkpoint["config"]))
        model.load_state_dict(checkpoint["state_dict"])
        model.batch_size = int(checkpoint.get("eval_batch_size", 4096))
        model.value_mode = checkpoint.get("value_mode", "exp")
        model.cache = {}
        model.to(device)
        model.eval()
        return model

    def reset_stats(self):
        for key in self.stats:
            self.stats[key] = 0

    def get_stats(self):
        return dict(self.stats)

    def _tensor_batch(self, states):
        pad_id = int(self.config["pad_id"])
        max_len = max(len(state) for state in states)
        device = next(self.parameters()).device
        input_ids = torch.full((len(states), max_len), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((len(states), max_len), dtype=torch.bool, device=device)
        for row, state in enumerate(states):
            ids = [int(x) for x in state]
            input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
            attention_mask[row, : len(ids)] = True
        return input_ids, attention_mask

    @torch.no_grad()
    def log_values(self, states):
        if not states:
            return []
        self.stats["requests"] += len(states)
        outputs = [None for _ in states]
        uncached = []
        uncached_indices = []
        uncached_keys = []
        for idx, state in enumerate(states):
            key = tuple(int(x) for x in state)
            if key in self.cache:
                outputs[idx] = self.cache[key]
                self.stats["cache_hits"] += 1
            else:
                uncached.append(list(key))
                uncached_indices.append(idx)
                uncached_keys.append(key)
        for start in range(0, len(uncached), self.batch_size):
            chunk = uncached[start : start + self.batch_size]
            self.stats["state_evals"] += len(chunk)
            self.stats["batch_calls"] += 1
            input_ids, attention_mask = self._tensor_batch(chunk)
            if self.value_mode == "binary_logprob":
                logits = self.forward(input_ids, attention_mask)
                scores = F.logsigmoid(logits).clamp(min=-20.0, max=0.0).detach().cpu().tolist()
            else:
                scores = self.log_value(input_ids, attention_mask).detach().cpu().tolist()
            for idx, key, score in zip(uncached_indices[start : start + self.batch_size], uncached_keys[start : start + self.batch_size], scores):
                value = float(score)
                self.cache[key] = value
                outputs[idx] = value
        return [float(x) for x in outputs]

    @torch.no_grad()
    def values(self, examples, states, harness):
        del examples, harness
        return [float(torch.exp(torch.tensor(x)).item()) for x in self.log_values(states)]


def load(checkpoint=None, device=None, **kwargs):
    del kwargs
    if not checkpoint:
        raise ValueError("DNA DeepSTARR VGB requires verifier.checkpoint")
    return SmallDNAValueVerifier.from_checkpoint(checkpoint, device=device)
