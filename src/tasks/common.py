"""Reusable building blocks for task modules.

Task files intentionally remain small plain modules.  This module holds the
boring glue so each task can focus on its domain-specific state, reward, and
metrics logic.
"""




def prompt_from_example(example, default = ""):
    return str(example.get("prompt") or default)


def length_from_example(example, default, state_key = "initial_state"):
    if example.get(state_key) is not None:
        return len(example[state_key])
    return int(example.get("length", default))


def decode_with_harness(state, harness):
    return harness.decode_state(state)


def masked_initial_state(length, harness):
    return [int(harness.mask_id) for _ in range(int(length))]


def result_output(result):
    if isinstance(result, dict):
        return str(result.get("output", ""))
    return str(result)


def result_state(result):
    if isinstance(result, dict):
        return result.get("state")
    return None


def binary_accuracy(rows, key = "reward"):
    if not rows:
        return 0.0
    return sum(float(row.get(key, 0.0)) for row in rows) / len(rows)


class ConstantVerifier:
    def __init__(self, value = 1.0):
        self.constant = float(value)

    def value(self, example, state, harness):
        del example, state, harness
        return self.constant

    def values(self, examples, states, harness):
        del examples, harness
        return [self.constant for _ in states]


class ExactStateVerifier:
    def __init__(self, task_module):
        self.task = task_module

    def value(self, example, state, harness):
        if hasattr(self.task, "state_value"):
            return float(self.task.state_value(example, state, harness))
        if hasattr(self.task, "value_state"):
            return float(self.task.value_state(example, state, harness))
        if hasattr(self.task, "reward_state"):
            return float(self.task.reward_state(example, state, harness))
        text = self.task.decode_state(state, harness)
        return float(self.task.reward(example, text))

    def values(self, examples, states, harness):
        return [self.value(example, state, harness) for example, state in zip(examples, states)]
