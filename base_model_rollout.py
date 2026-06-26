#!/usr/bin/env python
import argparse
import os
import random

from inference import load_examples, resolve_output
from tasks import evaluate_rows, load_harness, load_task_module, normalize_task_name, require_task_function
from utils import barrier, cleanup_distributed, init_distributed, load_task_config, log_metrics, read_rollout, setup_logging, set_seed
from utils import shard_items, write_rollout

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="qm9", help="Task name under configs/<task>.")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def select_rollout_examples(examples, config, seed, task_module):
    if getattr(task_module, "HANDLES_ROLLOUT_REPEATS", False):
        return examples
    num_rollouts = config.get("rollout", {}).get("num_rollouts", -1)
    if num_rollouts is None:
        return examples
    num_rollouts = int(num_rollouts)
    if num_rollouts < 0 or num_rollouts >= len(examples):
        return examples
    indices = list(range(len(examples)))
    random.Random(0 if seed is None else int(seed)).shuffle(indices)
    return [examples[idx] for idx in indices[:num_rollouts]]


def write_and_merge_rollout_rows(
    rows,
    output,
    dist_info,
    task_module,
    config,
):
    if not output:
        return rows if int(dist_info["rank"]) == 0 else []
    if not dist_info.get("distributed"):
        write_rollout(output, rows, task_module=task_module, config=config)
        return rows

    rank = int(dist_info["rank"])
    world_size = int(dist_info["world_size"])
    part_path = f"{output}.rank{rank:05d}.tmp"
    write_rollout(part_path, rows)
    barrier(dist_info)
    if rank != 0:
        barrier(dist_info)
        return []

    merged = []
    for part_rank in range(world_size):
        part = f"{output}.rank{part_rank:05d}.tmp"
        merged.extend(read_rollout(part))
    write_rollout(output, merged, task_module=task_module, config=config)
    for part_rank in range(world_size):
        try:
            os.remove(f"{output}.rank{part_rank:05d}.tmp")
        except FileNotFoundError:
            pass
    barrier(dist_info)
    return merged


def main():
    args = parse_args()
    config = load_task_config(args.task, stage="rollout")
    task_name = normalize_task_name(config.get("task", args.task))

    dist_info = init_distributed()
    base_seed = config.get("seed")
    rank_seed = base_seed
    if rank_seed is not None and dist_info.get("distributed"):
        rank_seed = int(rank_seed) + int(dist_info["rank"])
    set_seed(rank_seed)
    logger = setup_logging(config.get("logging", {}).get("level", "INFO"))

    try:
        task_module = load_task_module(task_name)
        examples = load_examples(task_module, config)
        examples = select_rollout_examples(examples, config, base_seed, task_module)
        examples = shard_items(examples, dist_info)
        harness = load_harness(task_name, config, device=args.device or dist_info.get("device"))
        require_task_function(task_module, "collect_rollout_rows", f"tasks.{task_name}.task rollout")
        rows = task_module.collect_rollout_rows(harness, examples, config, rank=int(dist_info["rank"]))
        output = resolve_output(config, task_name, "Base", "rollout")
        rows = write_and_merge_rollout_rows(rows, output, dist_info, task_module, config)
        if int(dist_info["rank"]) != 0:
            return
        if output:
            logger.info("wrote %d snapshot rows to %s", len(rows), output)
        final_rows = {}
        for row in rows:
            final_rows.setdefault(row["rollout_id"], row)
        metrics = evaluate_rows(task_name, list(final_rows.values()), config=config)
        num_rollouts = len(final_rows)
        snapshots_per_rollout = int(config.get("rollout", {}).get("snapshots_per_rollout", 1))
        metrics.update(
            {
                "num_rollouts": num_rollouts,
                "num_snapshots": num_rollouts * snapshots_per_rollout,
                "snapshots_per_rollout": snapshots_per_rollout,
            }
        )
        log_metrics(metrics, logger)
    finally:
        cleanup_distributed(dist_info)


if __name__ == "__main__":
    main()
