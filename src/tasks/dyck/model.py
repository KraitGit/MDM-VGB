import torch
import torch.nn as nn


TOKENS = ["(", "[", ")", "]", "B", "E", "P", "S"]
TOKEN_TO_ID = {token: i for i, token in enumerate(TOKENS)}
ID_TO_TOKEN = {i: token for token, i in TOKEN_TO_ID.items()}
MASK_ID = TOKEN_TO_ID["P"]
BOS_ID = TOKEN_TO_ID["B"]
EOS_ID = TOKEN_TO_ID["E"]
VOCAB_SIZE = len(TOKENS)


class DyckTransformer(nn.Module):
    def __init__(self, vocab_size, max_length, hidden_dim, num_layers, num_heads, dropout=0.1):
        super().__init__()
        hidden_dim = int(hidden_dim)
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_length, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, causal=False):
        length = input_ids.shape[1]
        pos = torch.arange(length, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        mask = None
        if causal:
            mask = torch.triu(torch.ones(length, length, device=input_ids.device, dtype=torch.bool), diagonal=1)
        x = self.encoder(x, mask=mask)
        return self.head(self.norm(x))


def build_model(config):
    return DyckTransformer(
        vocab_size=config.get("vocab_size", VOCAB_SIZE),
        max_length=config.get("max_length", 34),
        hidden_dim=config.get("hidden_dim", 256),
        num_layers=config.get("num_layers", 4),
        num_heads=config.get("num_heads", 4),
        dropout=config.get("dropout", 0.1),
    )


def ids_to_text(ids, skip_mask=True):
    out = []
    for token in ids:
        token = int(token)
        if skip_mask and token == MASK_ID:
            continue
        out.append(ID_TO_TOKEN.get(token, ""))
    return "".join(out)
