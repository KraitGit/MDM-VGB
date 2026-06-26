import sys
from contextlib import nullcontext
from pathlib import Path

import torch


D3LM_FROM_NT = "Hengchang-Liu/D3LM-from-nt"
REPO_ROOT = Path(__file__).resolve().parents[3]


def _is_repo_local_path(model_id):
    text = str(model_id)
    return Path(text).is_absolute() or text.startswith((".", "model_data/", "data/")) or Path(text).exists()


def resolve_model_source(model_id):
    path = Path(str(model_id))
    if _is_repo_local_path(model_id):
        candidate = path if path.is_absolute() else REPO_ROOT / path
        candidate = candidate.resolve()
        root = REPO_ROOT.resolve()
        if root not in candidate.parents and candidate != root:
            raise ValueError(f"model path must stay inside this repository: {candidate}")
        if not candidate.exists():
            raise FileNotFoundError(f"missing model inside this repository: {candidate}. Run prepare_assets.py.")
        return str(candidate)
    return model_id


def clean_dna(text):
    return "".join(ch for ch in str(text).upper().replace(" ", "") if ch in "ACGT")


def center_crop(seq, length=249):
    seq = clean_dna(seq)
    if len(seq) < int(length):
        return None
    start = (len(seq) - int(length)) // 2
    return seq[start : start + int(length)]


def _dtype_for_device(dtype, device):
    if str(dtype).lower() in ("bf16", "bfloat16") and torch.device(device).type == "cuda":
        return torch.bfloat16
    if str(dtype).lower() in ("fp16", "float16") and torch.device(device).type == "cuda":
        return torch.float16
    return torch.float32


def _generation_config_class(model):
    module = sys.modules.get(type(model).__module__)
    if module is None or not hasattr(module, "MDMGenerationConfig"):
        raise ValueError("D3LM model module does not expose MDMGenerationConfig")
    return getattr(module, "MDMGenerationConfig")


class D3LMWrapper:
    def __init__(
        self,
        model_id=D3LM_FROM_NT,
        device="cuda",
        dtype="bf16",
        local_files_only=False,
        target_nt=249,
    ):
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        if device in (None, "auto"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.torch_dtype = _dtype_for_device(dtype, self.device)
        self.model_id = resolve_model_source(model_id)
        self.target_nt = int(target_nt)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        self.model = AutoModelForMaskedLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            dtype=self.torch_dtype,
            local_files_only=local_files_only,
        )
        self.model.to(self.device)
        self.model.eval()
        self.MDMGenerationConfig = _generation_config_class(self.model)
        self.mask_id = int(self.tokenizer.mask_token_id)
        self.pad_id = self.tokenizer.pad_token_id
        self.eos_id = self.tokenizer.eos_token_id
        self.vocab_size = len(self.tokenizer)
        self.kind = "aoar"
        self.invalid_token_ids = self._invalid_token_ids()

    def _autocast(self):
        if self.device.type == "cuda" and self.torch_dtype in (torch.bfloat16, torch.float16):
            return torch.autocast("cuda", dtype=self.torch_dtype)
        return nullcontext()

    def _invalid_token_ids(self):
        ids = set(getattr(self.tokenizer, "all_special_ids", []) or [])
        ids.add(int(self.mask_id))
        if self.pad_id is not None:
            ids.add(int(self.pad_id))
        return {int(x) for x in ids if x is not None and int(x) >= 0}

    def _config(
        self,
        length,
        steps=50,
        temperature=1.0,
        top_p=0.9,
        top_k=0,
        alg="random",
        alg_temp=0.9,
        num_return_sequences=1,
        return_history=False,
    ):
        return self.MDMGenerationConfig(
            mask_token_id=self.mask_id,
            max_length=int(length),
            steps=int(steps),
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            alg=str(alg),
            alg_temp=float(alg_temp),
            num_return_sequences=int(num_return_sequences),
            return_dict_in_generate=True,
            output_history=bool(return_history),
        )

    def decode_tokens(self, token_ids):
        text = self.tokenizer.decode([int(x) for x in token_ids], skip_special_tokens=True)
        return clean_dna(text)

    def decode_to_249nt(self, token_ids):
        return center_crop(self.decode_tokens(token_ids), self.target_nt)

    def decode_state(self, state):
        ids = [int(x) for x in state if int(x) != int(self.mask_id)]
        seq = self.decode_to_249nt(ids)
        if seq is not None:
            return seq
        return self.decode_tokens(ids)

    def _sanitize_logits(self, logits):
        logits = logits.clone()
        for token_id in self.invalid_token_ids:
            if 0 <= int(token_id) < logits.shape[-1]:
                logits[..., int(token_id)] = -float("inf")
        return logits[..., : self.vocab_size]

    @torch.no_grad()
    def logits_batch(self, prompts, states):
        del prompts
        rows = [[int(x) for x in state] for state in states]
        max_len = max(len(row) for row in rows)
        input_ids = torch.full((len(rows), max_len), int(self.mask_id), dtype=torch.long, device=self.device)
        attention_mask = torch.zeros((len(rows), max_len), dtype=torch.long, device=self.device)
        for row_idx, row in enumerate(rows):
            input_ids[row_idx, : len(row)] = torch.tensor(row, dtype=torch.long, device=self.device)
            attention_mask[row_idx, : len(row)] = 1
        with self._autocast():
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return self._sanitize_logits(out.logits.detach().float())

    def logits(self, prompt, state):
        return self.logits_batch([prompt], [state])[0]

    @torch.no_grad()
    def diffusion_generate(
        self,
        n_tokens=48,
        batch_size=1,
        steps=50,
        temperature=1.0,
        top_p=0.9,
        top_k=0,
        alg="random",
        alg_temp=0.9,
        return_history=False,
    ):
        x = torch.full((int(batch_size), int(n_tokens)), int(self.mask_id), dtype=torch.long, device=self.device)
        config = self._config(
            n_tokens,
            steps=steps,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            alg=alg,
            alg_temp=alg_temp,
            return_history=return_history,
        )
        with self._autocast():
            return self.model.diffusion_generate(inputs=x, generation_config=config)

    def generate_tokens(
        self,
        n_tokens=48,
        batch_size=1,
        steps=50,
        temperature=1.0,
        top_p=0.9,
        top_k=0,
        alg="random",
        alg_temp=0.9,
        return_history=False,
    ):
        outputs = self.diffusion_generate(
            n_tokens=n_tokens,
            batch_size=batch_size,
            steps=steps,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            alg=alg,
            alg_temp=alg_temp,
            return_history=return_history,
        )
        sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs
        rows = []
        for token_ids in sequences.detach().cpu().tolist():
            seq = self.decode_tokens(token_ids)
            rows.append(
                {
                    "token_ids": [int(x) for x in token_ids],
                    "decoded_nt": seq,
                    "sequence_249": center_crop(seq, self.target_nt),
                }
            )
        if return_history and getattr(outputs, "history", None) is not None:
            for row, hist in zip(rows, getattr(outputs, "history", [])):
                row["history"] = hist.detach().cpu().tolist()
        return rows

    def generate(
        self,
        prompt="",
        max_new_tokens=48,
        temperature=1.0,
        steps=50,
        remasking="random",
        top_p=0.9,
        top_k=0,
        alg_temp=0.9,
        return_history=False,
    ):
        del prompt
        alg = "random" if remasking in (None, "random") else remasking
        rows = self.generate_tokens(
            n_tokens=max_new_tokens,
            batch_size=1,
            steps=steps,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            alg=alg,
            alg_temp=alg_temp,
            return_history=return_history,
        )
        return rows[0]["sequence_249"] or rows[0]["decoded_nt"]

    def generate_batch(
        self,
        prompts,
        max_new_tokens=48,
        steps=50,
        temperature=1.0,
        remasking="random",
        top_p=0.9,
        top_k=0,
        alg_temp=0.9,
    ):
        alg = "random" if remasking in (None, "random") else remasking
        rows = self.generate_tokens(
            n_tokens=max_new_tokens,
            batch_size=len(prompts),
            steps=steps,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            alg=alg,
            alg_temp=alg_temp,
        )
        return [row["sequence_249"] or row["decoded_nt"] for row in rows]


D3LMHarness = D3LMWrapper


def load(device=None, checkpoint=None, model_id=None, dtype="bf16", local_files_only=False, target_nt=249):
    return D3LMWrapper(
        model_id=model_id or checkpoint or D3LM_FROM_NT,
        device=device or "cuda",
        dtype=dtype,
        local_files_only=local_files_only,
        target_nt=target_nt,
    )
