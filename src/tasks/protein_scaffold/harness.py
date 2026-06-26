import torch

from .structure import clean_sequence


AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
MASK_ID = -1


class EvoDiffOADMModel:
    def __init__(self, model, tokenizer, length, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.length = int(length)
        self.vocab = AMINO_ACIDS
        self.vocab_size = len(self.vocab)
        self.mask_id = MASK_ID
        self.aa_token_ids = [int(tokenizer.tokenize(aa)[0]) for aa in self.vocab]
        self.evodiff_mask_id = int(tokenizer.mask_id)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.model.eval()
        self.backend = "evodiff_oadm"

    def _to_evodiff_ids(self, state):
        ids = []
        for token in state:
            token = int(token)
            ids.append(self.evodiff_mask_id if token == self.mask_id else self.aa_token_ids[token % self.vocab_size])
        return ids

    @torch.no_grad()
    def logits(self, target, state):
        del target
        ids = torch.tensor([self._to_evodiff_ids(state)], dtype=torch.long, device=self.device)
        num_mask = sum(1 for token in state if int(token) == self.mask_id)
        timestep = torch.tensor([max(1, int(num_mask))], dtype=torch.long, device=self.device)
        out = self.model(ids, timestep)
        return out[0].detach().float().cpu()[:, self.aa_token_ids]

    def decode(self, state):
        pieces = []
        for token in state:
            token = int(token)
            pieces.append("X" if token == self.mask_id else self.vocab[token % self.vocab_size])
        return "".join(pieces)


def load_evodiff_backend(model_type, length, device=None):
    try:
        from evodiff import pretrained
    except Exception as exc:
        raise ImportError("protein_scaffold requires the evodiff package; install this repository with pip install -e .") from exc
    if model_type == "oa_dm_640M":
        model, collater, tokenizer, scheme = pretrained.OA_DM_640M()
    else:
        model, collater, tokenizer, scheme = pretrained.OA_DM_38M()
    del collater, scheme
    return EvoDiffOADMModel(model, tokenizer, length, device=device)


class ProteinScaffoldHarness:
    kind = "aoar"

    def __init__(self, model, length=256, device=None):
        self.model = model
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.vocab = AMINO_ACIDS
        self.vocab_size = len(self.vocab)
        self.mask_id = MASK_ID
        self.eos_id = None
        self.id_to_token = {idx: aa for idx, aa in enumerate(self.vocab)}
        self.token_to_id = {aa: idx for idx, aa in self.id_to_token.items()}

    def _target(self, prompt):
        return prompt if isinstance(prompt, dict) else {}

    def logits(self, prompt, state):
        return torch.as_tensor(self.model.logits(self._target(prompt), state), dtype=torch.float32)

    def logits_batch(self, prompts, states):
        return torch.stack([self.logits(prompt, state) for prompt, state in zip(prompts, states)], dim=0)

    def decode_state(self, state):
        return clean_sequence(self.model.decode(state))

    def generate(self, prompt, max_new_tokens=256, temperature=1.0):
        target = self._target(prompt)
        state = target.get("initial_state")
        if state is None:
            state = [self.mask_id for _ in range(int(target.get("length", max_new_tokens)))]
            for pos, token in (target.get("motif") or {}).items():
                if 0 <= int(pos) < len(state):
                    state[int(pos)] = int(token)
            for pos, aa in (target.get("motif_text") or {}).items():
                aa = str(aa).upper()
                if 0 <= int(pos) < len(state) and aa in self.token_to_id:
                    state[int(pos)] = int(self.token_to_id[aa])
        state = [int(x) for x in state]
        while self.mask_id in state:
            pos = state.index(self.mask_id)
            logits = self.logits(target, state)[pos]
            if temperature <= 0:
                token = int(torch.argmax(logits).item())
            else:
                probs = torch.softmax(logits / float(temperature), dim=-1)
                token = int(torch.multinomial(probs, 1).item())
            state[pos] = token
        return self.decode_state(state)


def load(device=None, model_type="oa_dm_38M", length=256):
    model = load_evodiff_backend(model_type, length, device=device)
    return ProteinScaffoldHarness(model=model, length=length, device=device)
