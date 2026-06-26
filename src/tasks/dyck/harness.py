from pathlib import Path
import pickle

import torch
import torch.nn as nn

from . import task
from .model import (
    BOS_ID,
    EOS_ID,
    ID_TO_TOKEN,
    MASK_ID,
    TOKEN_TO_ID,
    TOKENS,
    VOCAB_SIZE,
    build_model,
    ids_to_text,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_AOAR_CHECKPOINT = "model_data/dyck/base_model/aoar_best.pkl"
DEFAULT_AR_CHECKPOINT = "model_data/dyck/base_model/ar_best.pkl"
DYCK_CHECKPOINT_FORMAT = "aoar_vgb_dyck_transformer_v1"


class _LogitsAdapter(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, causal=False):
        try:
            out = self.model(input_ids, causal=causal)
        except TypeError:
            out = self.model(input_ids, attention_mask=torch.ones_like(input_ids))
        if hasattr(out, "logits"):
            return out.logits
        return out


def resolve_package_model_path(path, kind):
    if path is None:
        path = DEFAULT_AR_CHECKPOINT if kind == "ar" else DEFAULT_AOAR_CHECKPOINT
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PACKAGE_ROOT / candidate
    candidate = candidate.resolve()
    root = PACKAGE_ROOT.resolve()
    if root not in candidate.parents and candidate != root:
        raise ValueError(f"model path must stay inside this repository: {candidate}")
    if not candidate.exists():
        raise FileNotFoundError(
            f"missing Dyck {kind} checkpoint inside this repository: {candidate}. "
            "Pass model.checkpoint in the task config or train the task base model first."
        )
    return candidate


def _load_pickled_model(path):
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"]
    return payload


def _mask_dyck_logits(logits):
    if logits.shape[-1] != VOCAB_SIZE:
        return logits
    out = logits.clone()
    seq_len = out.shape[-2]
    device = out.device
    for pos in range(seq_len):
        allowed = [EOS_ID] if pos == task.TOTAL_LENGTH - 1 else [0, 1, 2, 3]
        mask = torch.ones(VOCAB_SIZE, dtype=torch.bool, device=device)
        mask[allowed] = False
        if out.dim() == 2:
            out[pos, mask] = float("-inf")
        else:
            out[:, pos, mask] = float("-inf")
    return out


class DyckHarness:
    def __init__(self, checkpoint=None, device=None, kind="aoar"):
        self.tokens = TOKENS
        self.token_to_id = TOKEN_TO_ID
        self.id_to_token = ID_TO_TOKEN
        self.mask_id = MASK_ID
        self.bos_id = BOS_ID
        self.eos_id = EOS_ID
        self.vocab_size = VOCAB_SIZE
        self.kind = kind
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        path = resolve_package_model_path(checkpoint, kind)
        if path.suffix == ".pkl":
            try:
                raw_model = _load_pickled_model(path)
            except Exception:
                raw_model = None
        else:
            raw_model = None
        if raw_model is None:
            ckpt = torch.load(path, map_location="cpu")
            if ckpt.get("format") != DYCK_CHECKPOINT_FORMAT:
                raise ValueError(f"unsupported Dyck checkpoint format: {path}")
            ckpt_kind = ckpt.get("kind", kind)
            if ckpt_kind != kind:
                raise ValueError(f"Dyck checkpoint kind mismatch: requested {kind}, checkpoint is {ckpt_kind}")
            raw_model = build_model(ckpt["config"])
            raw_model.load_state_dict(ckpt["state_dict"])
        self.model = _LogitsAdapter(raw_model)
        self.model.to(self.device)
        self.model.eval()

    def encode_text(self, text):
        return [self.token_to_id[ch] for ch in text]

    def decode_state(self, state):
        return ids_to_text(state, skip_mask=True)

    @torch.no_grad()
    def logits(self, prompt, state):
        del prompt
        input_ids = torch.tensor([state], dtype=torch.long, device=self.device)
        out = self.model(input_ids, causal=(self.kind == "ar"))
        if len(state) == task.TOTAL_LENGTH:
            out = _mask_dyck_logits(out)
        return out[0].detach().float().cpu()

    @torch.no_grad()
    def logits_batch(self, prompts, states):
        del prompts
        input_ids = torch.tensor(states, dtype=torch.long, device=self.device)
        out = self.model(input_ids, causal=(self.kind == "ar"))
        if input_ids.shape[1] == task.TOTAL_LENGTH:
            out = _mask_dyck_logits(out)
        return out.detach().float().cpu()

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=34, temperature=1.0):
        del prompt
        if self.kind != "ar":
            from algorithms.algorithm_utils import force_complete

            example = {"id": "dyck-generate", "prefix": task.DEFAULT_PREFIX, "length": max_new_tokens}
            state = task.initial_state(example, self)
            state = force_complete(
                self,
                [state],
                [{"prompt": task.make_prompt(example), "temperature": temperature, "stop_on_eos": False}],
            )[0]
            return task.decode_state(state, self)

        state = self.encode_text(task.DEFAULT_PREFIX)
        while len(state) < max_new_tokens:
            input_ids = torch.tensor([state], dtype=torch.long, device=self.device)
            logits = self.model(input_ids, causal=True)[0, -1].float()
            if temperature <= 0:
                token = int(torch.argmax(logits).item())
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                token = int(torch.multinomial(probs, 1).item())
            state.append(token)
            if token == self.eos_id:
                break
        return self.decode_state(state)


def load(device=None, checkpoint=None, kind="aoar"):
    return DyckHarness(checkpoint=checkpoint, device=device, kind=kind)
