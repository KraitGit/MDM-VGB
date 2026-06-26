REQUIRED_TASK_FUNCTIONS = (
    "load_examples",
    "make_prompt",
    "default_length",
    "initial_state",
    "decode_state",
    "reward",
    "metrics",
)

REQUIRED_HARNESS_FUNCTIONS = ("load",)
REQUIRED_VERIFIER_FUNCTIONS = ("load",)
REQUIRED_TRAINING_FUNCTIONS = ("train_from_config",)


def missing_functions(module, required):
    return [name for name in required if not callable(getattr(module, name, None))]
