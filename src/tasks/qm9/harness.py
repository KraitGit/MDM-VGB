from pathlib import Path

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


def _require_path(path, label):
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"missing QM9 {label}: {resolved}. Run prepare_assets.py --tasks qm9.")
    return str(resolved)


def loglinear_sigma(t, eps=1e-3):
    return -torch.log1p(-(1 - eps) * t)


def sigma_from_mask_ratio(mask_ratio):
    return -torch.log1p(-torch.clamp(mask_ratio, min=1e-5, max=0.999))


def sample_categorical(probs):
    noise = 1e-10 - (torch.rand_like(probs) + 1e-10).log()
    return (probs / noise).argmax(dim=-1)


def token_log_probs(logits, mask_index):
    logits = logits.clone()
    logits[..., mask_index] = -1_000_000.0
    return logits.log_softmax(dim=-1)


def subs_log_probs(logits, xt, mask_index):
    logits = logits.clone()
    neg_inf = -1_000_000.0
    logits[..., mask_index] = neg_inf
    unmasked = xt != mask_index
    logits[unmasked] = neg_inf
    logits[unmasked, xt[unmasked]] = 0
    return logits.log_softmax(dim=-1)


def absorbing_posterior(x_theta, xt, mask_index, move_chance_t, move_chance_s):
    probs = x_theta * (move_chance_t - move_chance_s)
    probs[:, :, mask_index] = move_chance_s[:, :, 0]
    probs = probs / move_chance_t.clamp_min(1e-20)
    copy = xt != mask_index
    probs[copy] = 0.0
    probs[copy, xt[copy]] = 1.0
    probs = probs.clamp_min(0)
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-20)


def _snapshot_steps(batch_size, steps, k, device):
    k = max(1, int(k))
    if k == 1:
        steps_tensor = torch.full((1,), int(steps) // 2, dtype=torch.long, device=device)
    else:
        steps_tensor = torch.linspace(0, int(steps) - 1, k + 2, device=device)[1:-1].round().long()
    return steps_tensor.unsqueeze(0).repeat(int(batch_size), 1)


@torch.no_grad()
def sample_mdlm(model, batch_size, max_length, steps, device, use_bf16, mask_id, snapshots_per_rollout=0):
    xt = torch.full((batch_size, max_length), mask_id, dtype=torch.long, device=device)
    timesteps = torch.linspace(1, 1e-5, steps + 1, device=device)
    dt = (1 - 1e-5) / steps
    k = max(0, int(snapshots_per_rollout))
    snapshots = snapshot_steps = filled = None
    if k:
        snapshot_steps = _snapshot_steps(batch_size, steps, k, device)
        snapshots = torch.empty((batch_size, k, max_length), dtype=torch.long, device=device)
        filled = torch.zeros((batch_size, k), dtype=torch.bool, device=device)

    cache = None
    for step, t in enumerate(timesteps[:-1]):
        if k:
            rows, cols = torch.nonzero(snapshot_steps == step, as_tuple=True)
            if rows.numel():
                snapshots[rows, cols] = xt[rows]
                filled[rows, cols] = True
        sigma_t = loglinear_sigma(t).expand(batch_size)
        sigma_s = loglinear_sigma(t - dt).expand(batch_size)
        move_t = (1 - torch.exp(-sigma_t))[:, None, None]
        move_s = (1 - torch.exp(-sigma_s))[:, None, None]
        if cache is None:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                out = model(input_ids=xt, timesteps=sigma_t, return_dict=True)
            cache = subs_log_probs(out.logits, xt, mask_id)
        xs = sample_categorical(absorbing_posterior(cache.exp(), xt, mask_id, move_t, move_s))
        if not torch.equal(xs, xt):
            cache = None
        xt = xs

    if k and not filled.all():
        rows, cols = torch.nonzero(~filled, as_tuple=True)
        snapshots[rows, cols] = xt[rows]
    return (xt, snapshots, snapshot_steps) if k else xt


class QM9MdlmHarness:
    def __init__(self, model_name, tokenizer_path, max_length=32, device=None, dtype="bf16"):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model_path = _require_path(model_name, "base model")
        tokenizer_path = _require_path(tokenizer_path, "tokenizer")
        self.max_length = int(max_length)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        self.model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()
        self.mask_id = int(self.tokenizer.mask_token_id)
        self.vocab_size = int(getattr(self.tokenizer, "vocab_size", self.model.config.vocab_size))
        model_vocab_size = int(self.model.config.vocab_size)
        if model_vocab_size != self.vocab_size:
            raise ValueError(
                f"QM9 model/tokenizer vocab mismatch: model={model_vocab_size}, tokenizer={self.vocab_size}. "
                "Train a clean QM9 base model with `python base_model_training.py --task qm9` before rollout or inference."
            )
        self.use_bf16 = bool(dtype == "bf16" and self.device.type == "cuda")

    def encode_text(self, text):
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def decode_state(self, state):
        return self.tokenizer.decode([int(x) for x in state])

    @torch.no_grad()
    def logits_batch(self, prompts, states):
        del prompts
        if not states:
            return torch.empty((0, 0, self.vocab_size), dtype=torch.float)
        xt = torch.tensor(states, dtype=torch.long, device=self.device)
        mask_ratio = xt.eq(self.mask_id).sum(dim=1).float() / float(xt.shape[1])
        sigma = sigma_from_mask_ratio(mask_ratio)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
            out = self.model(input_ids=xt, timesteps=sigma, return_dict=True)
        return token_log_probs(out.logits, self.mask_id).detach()

    @torch.no_grad()
    def generate_batch(
        self,
        prompts,
        max_new_tokens=32,
        steps=128,
        temperature=1.0,
        remasking="low_confidence",
    ):
        del temperature, remasking
        batch_size = len(prompts) if not isinstance(prompts, str) else 1
        ids = sample_mdlm(
            self.model,
            batch_size,
            int(max_new_tokens or self.max_length),
            int(steps),
            self.device,
            self.use_bf16,
            self.mask_id,
        )
        return self.tokenizer.batch_decode(ids.detach().cpu().tolist())

    def generate(self, prompt, max_new_tokens=32, steps=128, temperature=1.0):
        return self.generate_batch([prompt], max_new_tokens=max_new_tokens, steps=steps, temperature=temperature)[0]

    @torch.no_grad()
    def generate_batch_with_snapshots(self, batch_size, steps=128, snapshots_per_rollout=3, max_new_tokens=None):
        max_length = int(max_new_tokens or self.max_length)
        return sample_mdlm(
            self.model,
            int(batch_size),
            max_length,
            int(steps),
            self.device,
            self.use_bf16,
            self.mask_id,
            snapshots_per_rollout=int(snapshots_per_rollout),
        )


def load(model_name, tokenizer_path, max_length=32, device=None, dtype="bf16"):
    return QM9MdlmHarness(model_name=model_name, tokenizer_path=tokenizer_path, max_length=max_length, device=device, dtype=dtype)
