"""Task registry for paper tasks."""


import importlib

from .interfaces import (
    REQUIRED_HARNESS_FUNCTIONS,
    REQUIRED_TASK_FUNCTIONS,
    REQUIRED_TRAINING_FUNCTIONS,
    REQUIRED_VERIFIER_FUNCTIONS,
    missing_functions,
)
from utils import get_device


TASKS = ("dyck", "letter", "sudoku", "qm9", "dna_deepstarr", "protein_scaffold")
BASE_MODEL_TRAINING_TASKS = ("dyck", "sudoku", "qm9")
PRETRAINED_BASE_MODEL_TASKS = tuple(task for task in TASKS if task not in BASE_MODEL_TRAINING_TASKS)


def normalize_task_name(name):
    task = str(name).lower()
    if task not in TASKS:
        raise KeyError(f"Unknown task: {task}")
    return task


def require_module_functions(module, required, label):
    missing = missing_functions(module, required)
    if missing:
        raise AttributeError(f"{label} must expose: {', '.join(missing)}")


def require_task_function(module, function_name, label):
    require_module_functions(module, (function_name,), label)


def load_task_module(name):
    name = normalize_task_name(name)
    module = importlib.import_module(f"tasks.{name}.task")
    require_module_functions(module, REQUIRED_TASK_FUNCTIONS, f"tasks.{name}.task")
    return module


def load_task_component(name, component):
    name = normalize_task_name(name)
    module = importlib.import_module(f"tasks.{name}.{component}")
    required = {
        "harness": REQUIRED_HARNESS_FUNCTIONS,
        "verifier": REQUIRED_VERIFIER_FUNCTIONS,
        "train_base_model": REQUIRED_TRAINING_FUNCTIONS,
        "train_verifier": REQUIRED_TRAINING_FUNCTIONS,
    }.get(component)
    if required is not None:
        require_module_functions(module, required, f"tasks.{name}.{component}")
    return module


def load_harness(name, config, device = None):
    module = load_task_component(name, "harness")
    model_cfg = {key: value for key, value in config.get("model", {}).items() if value is not None}
    return module.load(device=device or str(get_device()), **model_cfg)


def load_verifier(name, config, harness = None, device = None):
    module = load_task_component(name, "verifier")
    verifier_cfg = {key: value for key, value in config.get("verifier", {}).items() if value is not None}
    return module.load(harness=harness, device=device or str(get_device()), **verifier_cfg)


def evaluate_rows(name, rows, config = None):
    module = load_task_module(name)
    if config is not None and hasattr(module, "configure"):
        module.configure(config)
    return module.metrics(rows)
