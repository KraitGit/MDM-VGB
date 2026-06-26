from pathlib import Path

import numpy as np

from tasks.common import binary_accuracy, decode_with_harness, prompt_from_example, result_state


VALID = set("123456789")
DEFAULT_DATA = Path("data/sudoku/hard_sudokus_valid.csv")
EOL_POSITIONS = set(range(9, 89, 10))
_CONFIG = {}


def configure(config):
    global _CONFIG
    _CONFIG = dict(config or {})


def load_examples(split):
    del split
    data = np.loadtxt(DEFAULT_DATA, delimiter=",", dtype=np.int64)
    count = _CONFIG.get("data", {}).get("count")
    if count is not None:
        data = data[: int(count)]
    rows = []
    for idx, puzzle in enumerate(data.tolist()):
        rows.append(
            {
                "id": f"sudoku-{idx}",
                "puzzle": [int(x) for x in puzzle],
                "prompt": "",
            }
        )
    return rows


def make_prompt(example):
    return prompt_from_example(example)


def default_length(example):
    if example.get("initial_state") is not None:
        return len(example.get("initial_state", []))
    return len(example.get("puzzle", [])) or 89


def decode_state(state, harness):
    return decode_with_harness(state, harness)


def initial_state(example, harness):
    if example.get("initial_state") is not None:
        return [int(x) for x in example["initial_state"]]
    puzzle = example.get("puzzle")
    if puzzle is not None:
        return [int(x) for x in puzzle]
    state = [int(harness.mask_id) for _ in range(89)]
    for pos in EOL_POSITIONS:
        state[pos] = int(getattr(harness, "eol_id", 10))
    return state


def locked_positions(example, state, harness):
    del harness
    puzzle = example.get("puzzle") or []
    length = len(state) if state is not None else len(puzzle) or 89
    locked = {pos for pos in EOL_POSITIONS if pos < length}
    for pos, clue in enumerate(puzzle[:length]):
        if int(clue) != 0:
            locked.add(pos)
    return locked


def _grid_positions(length=89):
    return [pos for pos in range(length) if pos not in EOL_POSITIONS]


def _state_grid(state, harness, allow_masks=True):
    if len(state) != 89:
        return None
    eol_id = int(getattr(harness, "eol_id", 10))
    mask_id = int(harness.mask_id)
    for pos in EOL_POSITIONS:
        if int(state[pos]) != eol_id:
            return None
    grid = []
    for pos in _grid_positions(len(state)):
        token = int(state[pos])
        if token == mask_id:
            if allow_masks:
                grid.append(None)
            else:
                return None
        elif 1 <= token <= 9:
            grid.append(str(token))
        else:
            return None
    if len(grid) != 81:
        return None
    return grid


def _unit_ok(values, complete=False):
    seen = set()
    for value in values:
        if value is None:
            if complete:
                return False
            continue
        if value in seen:
            return False
        seen.add(value)
    if complete:
        return seen == VALID
    return True


def _grid_valid(grid, complete=False):
    if grid is None or len(grid) != 81:
        return False
    for row in range(9):
        if not _unit_ok(grid[row * 9 : (row + 1) * 9], complete=complete):
            return False
    for col in range(9):
        if not _unit_ok([grid[row * 9 + col] for row in range(9)], complete=complete):
            return False
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            block = []
            for row in range(br, br + 3):
                for col in range(bc, bc + 3):
                    block.append(grid[row * 9 + col])
            if not _unit_ok(block, complete=complete):
                return False
    return True


def _clue_ok(example, state):
    puzzle = example.get("puzzle")
    if puzzle is None:
        return True
    for pos, clue in enumerate(puzzle):
        clue = int(clue)
        if clue != 0 and int(state[pos]) != clue:
            return False
    return True


def valid_grid(text):
    digits = [ch for ch in text if ch in VALID]
    if len(digits) < 81:
        return False
    grid = digits[:81]
    return _grid_valid(grid, complete=True)


def reward(example, output):
    del example
    return 1.0 if valid_grid(output) else 0.0


def state_value(example, state, harness):
    if not _clue_ok(example, state):
        return 0.0
    grid = _state_grid(state, harness, allow_masks=True)
    return 1.0 if _grid_valid(grid, complete=False) else 0.0


def reward_state(example, state, harness):
    if any(int(token) == int(harness.mask_id) for token in state):
        return 0.0
    if not _clue_ok(example, state):
        return 0.0
    grid = _state_grid(state, harness, allow_masks=False)
    return 1.0 if _grid_valid(grid, complete=True) else 0.0


def reward_result(example, result, harness):
    state = result_state(result)
    if state is not None and state:
        return reward_state(example, state, harness)
    return reward(example, result.get("output", ""))


def row_info(example, output):
    del example
    return {"valid_grid": valid_grid(output)}


def metrics(rows):
    if not rows:
        return {"accuracy": 0.0}
    return {"accuracy": binary_accuracy(rows)}
