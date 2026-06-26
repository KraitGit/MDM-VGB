#!/usr/bin/env python
import argparse

from tasks import load_task_component, normalize_task_name
from utils import load_task_config, setup_logging, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="qm9", help="Task name under configs/<task>.")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_task_config(args.task, stage="verifier_training")
    set_seed(config.get("seed"))
    logger = setup_logging(config.get("logging", {}).get("level", "INFO"))
    task = normalize_task_name(config.get("task", args.task))

    try:
        module = load_task_component(task, "train_verifier")
    except ModuleNotFoundError as exc:
        expected = f"tasks.{task}.train_verifier"
        if exc.name != expected:
            raise
        logger.info("%s uses an exact verifier; no verifier training step is required", task)
        return
    module.train_from_config(config=config, device=args.device)


if __name__ == "__main__":
    main()
