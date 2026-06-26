from pathlib import Path

from tasks.common import binary_accuracy, decode_with_harness
from utils import read_jsonl, repo_root


NUM_TYPES = 2
LENGTH = 32
TOTAL_LENGTH = LENGTH + 2
DEFAULT_PREFIX = "B((((())((((((()("
_CONFIG = {}


def configure(config):
    global _CONFIG
    _CONFIG = dict(config or {})


def load_examples(split):
    data_cfg = dict(_CONFIG.get("data", {}))
    path = data_cfg.get("path")
    if path:
        path = Path(path)
        if not path.is_absolute():
            path = repo_root() / path
        rows = read_jsonl(path)
        if split is not None:
            rows = [row for row in rows if not row.get("split") or row.get("split") == split]
        count = data_cfg.get("count")
        return rows if count is None else rows[: int(count)]

    count = int(data_cfg.get("count", 100))
    return [{"id": f"dyck-{split}-{i}", "prefix": DEFAULT_PREFIX, "length": TOTAL_LENGTH} for i in range(count)]


def make_prompt(example):
    return "dyck"


def default_length(example):
    if example.get("initial_state") is not None:
        return len(example["initial_state"])
    return int(example.get("length", TOTAL_LENGTH))


def decode_state(state, harness):
    return decode_with_harness(state, harness)


def tokenize(text, harness):
    return [harness.token_to_id[ch] for ch in text]


def initial_state(example, harness):
    if "initial_state" in example:
        return [int(token) for token in example["initial_state"]]
    length = default_length(example)
    if "partial_sequence" in example:
        return tokenize(example["partial_sequence"], harness)
    state = [harness.mask_id] * length
    prefix = example.get("prefix", "B")
    prefix_ids = tokenize(prefix, harness)
    state[: len(prefix_ids)] = prefix_ids
    return state


def locked_positions(example, state, harness):
    del state
    if "editable_start" in example and "editable_end" in example:
        start = int(example["editable_start"])
        end = int(example["editable_end"])
        return [idx for idx in range(default_length(example)) if idx < start or idx >= end]
    prefix = example.get("prefix", "B")
    return range(len(tokenize(prefix, harness)))


def accept_ids(seq, harness):
    bos = harness.token_to_id["B"]
    eos = harness.token_to_id["E"]
    if len(seq) != TOTAL_LENGTH or seq[0] != bos or seq[-1] != eos:
        return False
    stack = []
    for token in seq[1:-1]:
        token = int(token)
        if token < NUM_TYPES:
            stack.append(token)
        elif token < 2 * NUM_TYPES:
            if not stack or stack.pop() + NUM_TYPES != token:
                return False
        else:
            return False
    return len(stack) == 0


def output_to_ids(output):
    mapping = {"(": 0, "[": 1, ")": 2, "]": 3, "B": 4, "E": 5}
    return [mapping[ch] for ch in output if ch in mapping]


def reward(example, output):
    return 1.0 if accept_ids(output_to_ids(output), _DummyHarness()) else 0.0


def row_info(example, output):
    del output
    keys = [
        "source",
        "corrupted",
        "editable_start",
        "editable_end",
        "corrupt_start",
        "corrupt_end",
    ]
    return {key: example[key] for key in keys if key in example}


def metrics(rows):
    if not rows:
        return {"accuracy": 0.0}
    return {"accuracy": binary_accuracy(rows)}


def _snapshot_indices(trace, count):
    count = max(1, int(count))
    if not trace:
        return [0] * count
    if count == 1:
        return [len(trace) - 1]
    return [round(i * (len(trace) - 1) / (count - 1)) for i in range(count)]


def collect_rollout_rows(harness, examples, config, rank=0):
    from algorithms.algorithm_utils import force_complete
    from tqdm.auto import tqdm

    rollout_cfg = dict(config.get("rollout", {}))
    generation_cfg = dict(config.get("generation", {}))
    snapshots_per_rollout = int(rollout_cfg.get("snapshots_per_rollout", 1))
    temperature = float(generation_cfg.get("temperature", 1.0))
    stop_on_eos = bool(generation_cfg.get("stop_on_eos", False))
    iterator = tqdm(examples, desc="Dyck verifier rollouts") if int(rank) == 0 else examples
    rows = []
    for idx, example in enumerate(iterator):
        prompt = make_prompt(example)
        initial = initial_state(example, harness)
        finals, traces = force_complete(
            harness,
            [initial],
            [{"prompt": prompt, "temperature": temperature, "stop_on_eos": stop_on_eos}],
            return_trace=True,
        )
        final_state = finals[0]
        trace = traces[0] or [final_state]
        output = decode_state(final_state, harness)
        rollout_reward = float(reward(example, output))
        rollout_id = str(example.get("id", f"dyck-rollout-{idx}"))
        for snapshot_id, trace_idx in enumerate(_snapshot_indices(trace, snapshots_per_rollout)):
            rows.append(
                {
                    "rollout_id": rollout_id,
                    "snapshot_id": snapshot_id,
                    "split": config.get("data", {}).get("split", "train"),
                    "example": dict(example),
                    "state_ids": [int(token) for token in trace[trace_idx]],
                    "output": output,
                    "sample": output,
                    "reward": rollout_reward,
                    "stats": {"nfe": len(trace), "snapshot_step": int(trace_idx)},
                }
            )
    return rows


class _DummyHarness:
    def __init__(self):
        self.token_to_id = {"(": 0, "[": 1, ")": 2, "]": 3, "B": 4, "E": 5}
