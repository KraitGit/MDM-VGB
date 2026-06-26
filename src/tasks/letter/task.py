from tasks.common import decode_with_harness, length_from_example, masked_initial_state, prompt_from_example, result_state


LETTER = "e"
PROMPT = "Generate a one-sentence story without using the letter 'e':\n"
DEFAULT_COUNT = 100
DEFAULT_LENGTH = 32
HARD_ZERO_STATE_VALUE = True
FILTER_FORWARD_TOKENS = True
_CONFIG = {}
_TOKEN_MASK_CACHE = {}


def configure(config):
    _CONFIG.clear()
    _CONFIG.update(config or {})


def load_examples(split):
    count = int(_CONFIG.get("data", {}).get("count", DEFAULT_COUNT))
    return [
        {
            "id": f"letter-{split}-{i}",
            "prompt": PROMPT,
            "letter": LETTER,
            "length": DEFAULT_LENGTH,
        }
        for i in range(count)
    ]


def make_prompt(example):
    return prompt_from_example(example, PROMPT)


def default_length(example):
    return length_from_example(example, DEFAULT_LENGTH)


def decode_state(state, harness):
    return decode_with_harness(state, harness)


def _letter(example):
    return str(example.get("letter", LETTER)).lower()


def initial_state(example, harness):
    return masked_initial_state(default_length(example), harness)


def _token_text(harness, token_id):
    tokenizer = getattr(harness, "tokenizer", None)
    if tokenizer is None:
        id_to_token = getattr(harness, "id_to_token", {})
        return str(id_to_token.get(int(token_id), token_id))
    return tokenizer.decode([int(token_id)], skip_special_tokens=False)


def _forward_token_mask(harness, letter):
    tokenizer = getattr(harness, "tokenizer", None)
    vocab_size = int(getattr(harness, "valid_vocab_size", getattr(harness, "vocab_size", 0)))
    key = (id(tokenizer), vocab_size, str(letter).lower())
    if key in _TOKEN_MASK_CACHE:
        return _TOKEN_MASK_CACHE[key]

    special_ids = set(getattr(tokenizer, "all_special_ids", []) or []) if tokenizer is not None else set()
    if getattr(harness, "mask_id", None) is not None:
        special_ids.add(int(harness.mask_id))
    if getattr(harness, "eos_id", None) is not None:
        special_ids.add(int(harness.eos_id))

    allowed = []
    for token_id in range(vocab_size):
        if token_id in special_ids:
            allowed.append(False)
            continue
        token_text = _token_text(harness, token_id)
        if str(letter).lower() in token_text.lower():
            allowed.append(False)
            continue
        if tokenizer is not None and tokenizer.decode([token_id], skip_special_tokens=True) == "":
            allowed.append(False)
            continue
        allowed.append(True)
    _TOKEN_MASK_CACHE[key] = allowed
    return allowed


def forward_token_mask(example, harness):
    return _forward_token_mask(harness, _letter(example))


def _state_valid_tokens(example, state, harness, require_complete=False):
    letter = _letter(example)
    mask_id = int(harness.mask_id)
    tokenizer = getattr(harness, "tokenizer", None)
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or []) if tokenizer is not None else set()
    eos_id = getattr(harness, "eos_id", None)
    for token in state:
        token = int(token)
        if token == mask_id:
            if require_complete:
                return False
            continue
        if eos_id is not None and token == int(eos_id):
            return False
        if token in special_ids:
            return False
        if letter in _token_text(harness, token).lower():
            return False
    return True


def state_value(example, state, harness):
    return 1.0 if _state_valid_tokens(example, state, harness, require_complete=False) else 0.0


def reward_state(example, state, harness):
    return 1.0 if _state_valid_tokens(example, state, harness, require_complete=True) else 0.0


def reward_result(example, result, harness):
    state = result_state(result)
    if state is not None:
        return reward_state(example, state, harness)
    output = result.get("output")
    if output is not None:
        return reward(example, output)
    return 0.0


def reward(example, output):
    text = str(output).lower()
    if not text.strip():
        return 0.0
    return 0.0 if _letter(example) in text else 1.0


def row_info(example, output):
    text = str(output)
    letter = _letter(example)
    return {
        "letter": letter,
        "contains_forbidden_letter": letter in text.lower(),
        "output_length": len(text),
    }


def metrics(rows):
    if not rows:
        return {"accuracy": 0.0, "avoid_rate": 0.0}
    accuracy = sum(row["reward"] for row in rows) / len(rows)
    avoid_rate = sum(
        1
        for row in rows
        if not row.get("contains_forbidden_letter", False) and str(row.get("output", "")).strip()
    ) / len(rows)
    return {"accuracy": accuracy, "avoid_rate": avoid_rate}
