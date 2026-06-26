from pathlib import Path

import torch
import torch.nn.functional as F


MDLM_QWEN_06B = "dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1"
PACKAGE_ROOT = Path(__file__).resolve().parents[3]


def _is_repo_local_path(model_name):
    text = str(model_name)
    return Path(text).is_absolute() or text.startswith((".", "model_data/", "data/")) or Path(text).exists()


def validate_model_source(model_name):
    if model_name is None:
        return None
    path = Path(str(model_name))
    if not _is_repo_local_path(model_name):
        return model_name
    candidate = path if path.is_absolute() else PACKAGE_ROOT / path
    candidate = candidate.resolve()
    root = PACKAGE_ROOT.resolve()
    if root not in candidate.parents and candidate != root:
        raise ValueError(f"model path must stay inside this repository: {candidate}")
    if not candidate.exists():
        raise FileNotFoundError(f"missing model inside this repository: {candidate}. Run prepare_assets.py.")
    return str(candidate)


class MaskedLMHarness:
    def __init__(self, model_name=MDLM_QWEN_06B, device=None, dtype="bf16"):
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.model_name = validate_model_source(model_name)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        torch_dtype = torch.bfloat16 if dtype == "bf16" and self.device.type == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True, local_files_only=True)
        self.model = AutoModelForMaskedLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            local_files_only=True,
        )
        self.model.to(self.device)
        self.model.eval()
        self.mask_id = self._mask_id()
        self.eos_id = self.tokenizer.eos_token_id
        self.vocab_size = len(self.tokenizer)
        self.valid_vocab_size = len(self.tokenizer)
        self.invalid_token_ids = [] if self.mask_id is None else [int(self.mask_id)]

    def _mask_id(self):
        if self.tokenizer.mask_token_id is not None:
            return int(self.tokenizer.mask_token_id)
        token_id = self.tokenizer.convert_tokens_to_ids(getattr(self.tokenizer, "mask_token", None) or "<|mask|>")
        if token_id is None or token_id == self.tokenizer.unk_token_id:
            token_id = len(self.tokenizer) - 1
        return int(token_id)

    def _pad_id(self):
        return self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0

    def _sanitize_logits(self, logits):
        logits = logits.detach().float().clone()
        if self.valid_vocab_size < logits.shape[-1]:
            logits[..., self.valid_vocab_size:] = -float("inf")
        invalid = [idx for idx in self.invalid_token_ids if 0 <= idx < logits.shape[-1]]
        if invalid:
            logits[..., torch.tensor(invalid, dtype=torch.long, device=logits.device)] = -float("inf")
        return logits

    def _prompt_ids(self, prompt):
        cache = getattr(self, "_vgb_prompt_ids_cache", None)
        if cache is None:
            cache = {}
            self._vgb_prompt_ids_cache = cache
        key = str(prompt)
        if key not in cache:
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt},
            ]
            cache[key] = tuple(
                self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    enable_thinking=False,
                )
            )
        return list(cache[key])

    def _input_ids(self, prompt, state):
        prompt_ids = self._prompt_ids(prompt)
        return prompt_ids + [int(token) for token in state], len(prompt_ids)

    def decode_state(self, state):
        tokens = [int(token) for token in state if int(token) != int(self.mask_id)]
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def logits(self, prompt, state):
        return self.logits_batch([prompt], [state])[0]

    @torch.no_grad()
    def logits_batch(self, prompts, states):
        if isinstance(prompts, str):
            prompts = [prompts for _ in states]
        encoded = []
        offsets = []
        for prompt, state in zip(prompts, states):
            ids, offset = self._input_ids(prompt, state)
            encoded.append(ids)
            offsets.append(offset)
        max_len = max(len(ids) for ids in encoded)
        input_ids = torch.full((len(encoded), max_len), self._pad_id(), dtype=torch.long, device=self.device)
        attention_mask = torch.zeros((len(encoded), max_len), dtype=torch.long, device=self.device)
        for row, ids in enumerate(encoded):
            input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=self.device)
            attention_mask[row, : len(ids)] = 1
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        rows = []
        for row, offset in enumerate(offsets):
            logits = out.logits[row, offset : offset + len(states[row])]
            rows.append(self._sanitize_logits(logits))
        return torch.stack(rows, dim=0)

    def _num_transfer_tokens(self, mask_index, steps):
        mask_num = mask_index.sum(dim=1, keepdim=True)
        base = mask_num // steps
        remainder = mask_num % steps
        out = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.long) + base
        for row in range(mask_num.size(0)):
            out[row, : remainder[row]] += 1
        return out

    def _gumbel_argmax_scores(self, logits, temperature):
        if temperature <= 0:
            return logits
        logits = logits.to(torch.float64)
        noise = torch.rand_like(logits, dtype=torch.float64)
        return logits.exp() / ((-torch.log(noise)) ** temperature)

    @torch.no_grad()
    def generate_batch(self, prompts, max_new_tokens=256, steps=256, block_size=64, temperature=0.0, remasking="low_confidence"):
        max_new_tokens = int(max_new_tokens)
        block_size = max(1, min(int(block_size), max_new_tokens))
        if max_new_tokens % block_size != 0:
            block_size = max_new_tokens
        steps = max(int(steps), 1)
        num_blocks = max_new_tokens // block_size
        steps_per_block = max(1, steps // num_blocks)

        encoded = [self._prompt_ids(prompt) for prompt in prompts]
        prompt_lens = torch.tensor([len(ids) for ids in encoded], dtype=torch.long, device=self.device)
        max_prompt_len = int(prompt_lens.max().item())
        total_length = max_prompt_len + max_new_tokens
        x = torch.full((len(encoded), total_length), self._pad_id(), dtype=torch.long, device=self.device)
        for row, ids in enumerate(encoded):
            x[row, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=self.device)
            x[row, len(ids) : len(ids) + max_new_tokens] = int(self.mask_id)

        positions = torch.arange(total_length, device=self.device)
        for block in range(num_blocks):
            block_start = prompt_lens + block * block_size
            block_end = block_start + block_size
            init_mask = (
                (positions.unsqueeze(0) >= block_start.unsqueeze(1))
                & (positions.unsqueeze(0) < block_end.unsqueeze(1))
                & (x == int(self.mask_id))
            )
            num_transfer = self._num_transfer_tokens(init_mask, steps_per_block)
            for step in range(steps_per_block):
                block_mask = (
                    (positions.unsqueeze(0) >= block_start.unsqueeze(1))
                    & (positions.unsqueeze(0) < block_end.unsqueeze(1))
                    & (x == int(self.mask_id))
                )
                if not block_mask.any():
                    continue
                logits = self.model(input_ids=x, attention_mask=(x != self._pad_id()).long()).logits
                logits = self._sanitize_logits(logits)
                sampled = torch.argmax(self._gumbel_argmax_scores(logits, float(temperature)), dim=-1)
                if remasking == "low_confidence":
                    probs = F.softmax(logits.float(), dim=-1)
                    confidence = torch.gather(probs, dim=-1, index=sampled.unsqueeze(-1)).squeeze(-1)
                elif remasking == "random":
                    confidence = torch.rand_like(sampled, dtype=torch.float)
                else:
                    raise ValueError(f"unknown remasking: {remasking}")
                confidence = torch.where(block_mask, confidence, torch.full_like(confidence, -float("inf")))
                transfer = torch.zeros_like(block_mask)
                for row in range(confidence.shape[0]):
                    k = min(int(num_transfer[row, step].item()), int(block_mask[row].sum().item()))
                    if k > 0:
                        _, selected = torch.topk(confidence[row], k=k)
                        transfer[row, selected] = True
                x[transfer] = sampled[transfer]

        outputs = []
        for row in range(x.size(0)):
            new_tokens = x[row, prompt_lens[row] : prompt_lens[row] + max_new_tokens].tolist()
            outputs.append(self.tokenizer.decode(new_tokens, skip_special_tokens=True))
        return outputs

    def generate(self, prompt, max_new_tokens=256, temperature=0.0):
        return self.generate_batch([prompt], max_new_tokens=max_new_tokens, steps=max_new_tokens, temperature=temperature)[0]


def load(device=None, model_name=None, checkpoint=None, dtype="bf16"):
    return MaskedLMHarness(model_name=model_name or checkpoint or MDLM_QWEN_06B, device=device, dtype=dtype)
