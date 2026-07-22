import unittest

import torch

from inference import build_row, run_base
from tasks.sudoku import harness as sudoku_harness
from tasks.sudoku import task as sudoku_task


def solved_state():
    rows = (
        "123456789",
        "456789123",
        "789123456",
        "234567891",
        "567891234",
        "891234567",
        "345678912",
        "678912345",
        "912345678",
    )
    state = []
    for row_idx, row in enumerate(rows):
        state.extend(int(token) for token in row)
        if row_idx < len(rows) - 1:
            state.append(10)
    return state


class FakeSudokuHarness:
    mask_id = 0
    eol_id = 10
    eos_id = None

    def __init__(self, target):
        self.target = list(target)

    def logits_batch(self, prompts, states):
        del prompts
        logits = torch.full((len(states), len(self.target), 11), -100.0)
        for batch_idx in range(len(states)):
            for pos, token in enumerate(self.target):
                logits[batch_idx, pos, token] = 100.0
        return logits

    def decode_state(self, state):
        pieces = []
        for token in state:
            pieces.append("\n" if int(token) == self.eol_id else str(int(token)))
        return "".join(pieces)


class SudokuBaseTest(unittest.TestCase):
    def test_public_harness_has_no_unconditioned_generate_path(self):
        self.assertFalse(hasattr(sudoku_harness.SudokuHarness, "generate"))

    def test_base_preserves_clues_uses_state_reward_and_counts_nfe(self):
        target = solved_state()
        puzzle_a = list(target)
        puzzle_b = list(target)
        for pos in (0, 1, 20):
            puzzle_a[pos] = 0
        for pos in (3, 4):
            puzzle_b[pos] = 0
        examples = [
            {"id": "a", "prompt": "", "puzzle": puzzle_a},
            {"id": "b", "prompt": "", "puzzle": puzzle_b},
            {"id": "complete", "prompt": "", "puzzle": target},
        ]
        harness = FakeSudokuHarness(target)
        config = {"generation": {"batch_size": 3, "temperature": 0.0}}

        rows = run_base(sudoku_task, harness, examples, config, rank=1)

        self.assertEqual([row["state_ids"] for row in rows], [target, target, target])
        self.assertEqual([row["reward"] for row in rows], [1.0, 1.0, 1.0])
        self.assertEqual([row["stats"]["nfe"] for row in rows], [3, 2, 0])
        self.assertEqual([row["stats"]["forward"] for row in rows], [3, 2, 0])

    def test_state_reward_takes_precedence_over_valid_text(self):
        target = solved_state()
        puzzle = list(target)
        harness = FakeSudokuHarness(target)
        bad_state = list(target)
        bad_state[0] = 2
        text = harness.decode_state(target)

        row = build_row(
            sudoku_task,
            {"id": "bad-clue", "prompt": "", "puzzle": puzzle},
            harness,
            text,
            bad_state,
            {"nfe": 0, "forward": 0, "backward": 0, "forced": 0},
        )

        self.assertTrue(row["valid_grid"])
        self.assertEqual(row["reward"], 0.0)


if __name__ == "__main__":
    unittest.main()
