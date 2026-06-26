#!/usr/bin/env python
import argparse

from tasks import BASE_MODEL_TRAINING_TASKS, PRETRAINED_BASE_MODEL_TASKS, load_task_component, normalize_task_name
from utils import load_task_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="sudoku", help="Task name under configs/<task>.")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    task = normalize_task_name(args.task)
    if task in PRETRAINED_BASE_MODEL_TASKS:
        raise ValueError(f"{task} does not train a base model in this repo; a pretrained model is available.")
    if task not in BASE_MODEL_TRAINING_TASKS:
        raise KeyError(f"Unknown base-model training task: {task}")
    config = load_task_config(task, stage="base_model_training")
    task = normalize_task_name(config.get("task", task))
    module = load_task_component(task, "train_base_model")
    module.train_from_config(config, device=args.device)


if __name__ == "__main__":
    main()
