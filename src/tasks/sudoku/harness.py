from pathlib import Path

import torch
from omegaconf import OmegaConf

from .dit import DIT
from .ema import ExponentialMovingAverage


PACKAGE_ROOT = Path(__file__).resolve().parents[3]


def build_config(dropout=0.1):
    return OmegaConf.create(
        {
            "parameterization": "subs",
            "time_conditioning": False,
            "noise": {
                "type": "loglinear",
                "sigma_min": 1e-4,
                "sigma_max": 20.0,
            },
            "model": {
                "hidden_size": 512,
                "cond_dim": 128,
                "length": 89,
                "n_blocks": 8,
                "n_heads": 8,
                "scale_by_sigma": True,
                "dropout": dropout,
                "tie_word_embeddings": False,
            },
            "sampling": {
                "steps": 51,
                "num_initial_masks": 51,
            },
        }
    )


def strip_module_prefix(state_dict):
    out = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            out[key[7:]] = value
        else:
            out[key] = value
    return out


class SudokuTokenizer:
    def __init__(self):
        self.vocab_size = 11
        self.mask_token_id = 0
        self.eol_token_id = 10

    def decode(self, token_ids):
        pieces = []
        for token_id in token_ids:
            token_id = int(token_id)
            if token_id == self.mask_token_id:
                pieces.append(".")
            elif token_id == self.eol_token_id:
                pieces.append("\n")
            elif 1 <= token_id <= 9:
                pieces.append(str(token_id))
            else:
                pieces.append("?")
        return "".join(pieces)


class SudokuHarness:
    kind = "aoar"

    def __init__(self, checkpoint=None, device=None, dtype="bf16"):
        del dtype
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = SudokuTokenizer()
        self.mask_id = self.tokenizer.mask_token_id
        self.eol_id = self.tokenizer.eol_token_id
        self.eos_id = None
        self.vocab_size = self.tokenizer.vocab_size
        self.valid_vocab_size = self.vocab_size
        self.id_to_token = {0: ".", 10: "\n"}
        self.id_to_token.update({idx: str(idx) for idx in range(1, 10)})
        self.token_to_id = {value: key for key, value in self.id_to_token.items()}
        self.config = build_config()
        self.model = DIT(self.config, vocab_size=self.vocab_size).to(self.device)
        self.model.eval()
        if checkpoint:
            self.load_checkpoint(checkpoint)

    def load_checkpoint(self, checkpoint):
        path = Path(str(checkpoint))
        if not path.is_absolute():
            path = PACKAGE_ROOT / path
        checkpoint_data = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(strip_module_prefix(checkpoint_data["model"]))
        if "ema" in checkpoint_data:
            ema = ExponentialMovingAverage(self.model.parameters(), decay=0.9999)
            ema.load_state_dict(checkpoint_data["ema"])
            ema.move_shadow_params_to_device(self.device)
            ema.copy_to(self.model.parameters())
        self.model.eval()

    def encode_text(self, text):
        out = []
        for ch in text:
            if ch in "123456789":
                out.append(int(ch))
            elif ch in ".0":
                out.append(self.mask_id)
            elif ch == "\n":
                out.append(self.eol_id)
        return out

    def decode_state(self, state):
        return self.tokenizer.decode(state)

    def _sanitize_logits(self, logits):
        logits = logits.detach().float().clone()
        logits[..., self.mask_id] = -float("inf")
        return logits

    @torch.no_grad()
    def logits_batch(self, prompts, states):
        del prompts
        x = torch.tensor(states, dtype=torch.long, device=self.device)
        sigma = torch.zeros(x.shape[0], dtype=torch.float32, device=self.device)
        logits = self.model(x, sigma)
        return self._sanitize_logits(logits)

    def logits(self, prompt, state):
        return self.logits_batch([prompt], [state])[0]

    def generate(self, prompt, max_new_tokens=89, temperature=1.0):
        del prompt
        state = [self.mask_id for _ in range(int(max_new_tokens))]
        for idx in range(9, min(len(state), 89), 10):
            state[idx] = self.eol_id
        while self.mask_id in state:
            logits = self.logits("", state)
            positions = [idx for idx, token in enumerate(state) if int(token) == self.mask_id]
            pos = positions[0]
            token_logits = logits[pos]
            if temperature <= 0:
                token = int(torch.argmax(token_logits).item())
            else:
                probs = torch.softmax(token_logits.float() / float(temperature), dim=-1)
                token = int(torch.multinomial(probs, 1).item())
            state[pos] = token
        return self.decode_state(state)


def load(device=None, checkpoint=None):
    return SudokuHarness(checkpoint=checkpoint, device=device)
