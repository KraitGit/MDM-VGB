from tasks.common import decode_with_harness, length_from_example, masked_initial_state, prompt_from_example, result_state


LETTER = "e"
PROMPT = "Generate a one-sentence story without using the letter 'e':\n"
DEFAULT_COUNT = 100
DEFAULT_LENGTH = 32

def load_examples(split):
    return [
        {
            "id": f"letter-{split}-{i}",
            "prompt": PROMPT,
            "letter": LETTER,
            "length": DEFAULT_LENGTH,
        }
        for i in range(DEFAULT_COUNT)
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


def _revealed_text(state, harness):
    mask_id = int(harness.mask_id)
    tokenizer = getattr(harness, "tokenizer", None)
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or []) if tokenizer is not None else set()
    tokens = []
    for token in state:
        token = int(token)
        if token == mask_id:
            continue
        if token in special_ids:
            return None
        if tokenizer is not None and tokenizer.decode([token], skip_special_tokens=True) == "":
            return None
        tokens.append(token)
    if not tokens:
        return ""
    if tokenizer is not None:
        return tokenizer.decode(tokens, skip_special_tokens=True)
    id_to_token = getattr(harness, "id_to_token", {})
    return "".join(id_to_token.get(token, str(token)) for token in tokens)


def state_value(example, state, harness):
    letter = _letter(example)
    text = _revealed_text(state, harness)
    if text is None:
        return 0.0
    if letter in text.lower():
        return 0.0
    return 1.0


value_state = state_value


def reward_state(example, state, harness):
    if any(int(token) == int(harness.mask_id) for token in state):
        return 0.0
    return state_value(example, state, harness)


def reward_result(example, result, harness):
    output = result.get("output")
    if output is not None:
        return reward(example, output)
    state = result_state(result)
    if state is not None:
        return reward_state(example, state, harness)
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
