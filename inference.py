#!/usr/bin/env python
import argparse
import inspect
import os
import time

from tqdm.auto import tqdm

from algorithms.algorithm_utils import force_complete
from algorithms.config import (
    ALGORITHMS,
    apply_budget_overrides,
    algorithm_slug,
    value_guidance_config,
    value_guidance_section,
)
from algorithms.vgb import sample_vgb
from algorithms.vgb_momentum import sample_vgb_momentum
from algorithms.vgr import sample_vgr
from tasks import evaluate_rows, load_harness, load_task_module, load_verifier, normalize_task_name
from utils import (
    add_nfe_metrics,
    cleanup_distributed,
    gather_objects,
    init_distributed,
    load_task_config,
    log_metrics,
    read_jsonl,
    setup_logging,
    set_seed,
    shard_items,
    write_jsonl,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="qm9", help="Task name under configs/<task>.")
    parser.add_argument("--stage", default="inference", choices=["inference", "rollout", "verifier_training"], help=argparse.SUPPRESS)
    parser.add_argument("--algorithm", default="Base", choices=sorted(ALGORITHMS))
    parser.add_argument("--device", default=None)
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--N", type=int, default=None, help="BoN candidate count or VGB maximum Markov-step budget factor.")
    parser.add_argument("--K", default=None)
    parser.add_argument("--max-steps-multiplier", type=int, default=None, help="Set the VGB maximum Markov-step budget factor directly.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--verifier", default=None)
    return parser.parse_args()


def maybe_set_forward_token_filter(task_module, harness, examples, cfg):
    enabled = cfg.get("filter_forward_tokens")
    if enabled is None:
        enabled = bool(getattr(task_module, "FILTER_FORWARD_TOKENS", False))
    if not hasattr(harness, "set_forward_token_mask"):
        return
    if not enabled or not examples:
        harness.set_forward_token_mask(None)
        return
    if not hasattr(task_module, "forward_token_mask"):
        harness.set_forward_token_mask(None)
        return
    harness.set_forward_token_mask(task_module.forward_token_mask(examples[0], harness))


def load_examples(task_module, config):
    if hasattr(task_module, "configure"):
        task_module.configure(config)
    return task_module.load_examples(split=config["data"]["split"])


def resolve_output(config, task, algorithm, stage):
    output = config.get("output")
    if output is None:
        return None
    return str(output).format(
        algorithm=algorithm_slug(algorithm),
        algorithm_name=algorithm,
        stage=stage,
        task=task,
    )


def example_json(task_module, example):
    if hasattr(task_module, "example_to_json"):
        return task_module.example_to_json(example)
    return example


def reward_result(
    task_module,
    example,
    harness,
    text,
    state=None,
):
    result = {"output": text, "state": state}
    if hasattr(task_module, "reward_result"):
        value = task_module.reward_result(example, result, harness)
        if isinstance(value, dict):
            out = dict(value)
            out.setdefault("reward", float(out.get("reward", 0.0)))
            return out
        return {"reward": float(value), "ok": bool(float(value) > 0.0)}
    reward = float(task_module.reward(example, text))
    return {"reward": reward, "ok": bool(reward > 0.0)}


def build_row(task_module, example, harness, text, state, stats):
    result = reward_result(task_module, example, harness, text, state=state)
    row = {
        "example": example_json(task_module, example),
        "output": text,
        "sample": text,
        "reward": float(result.get("reward", 0.0)),
        "result": result,
        "stats": stats,
    }
    for key, value in result.items():
        if key not in row:
            row[key] = value
    if state is not None:
        row["state_ids"] = [int(x) for x in state]
    if hasattr(task_module, "row_info"):
        row.update(task_module.row_info(example, text))
    return row


def _base_nfe(generation_cfg):
    if generation_cfg.get("nfe") is not None:
        return max(1, int(generation_cfg["nfe"]))
    if generation_cfg.get("steps") is not None:
        return max(1, int(generation_cfg["steps"]))
    return 1


def _base_stats(generation_cfg):
    nfe = _base_nfe(generation_cfg)
    return {"nfe": nfe, "forward": nfe, "backward": 0, "forced": 0}


def base_sample(task_module, harness, example, config):
    generation_cfg = dict(config.get("generation", {}))
    generation_cfg.pop("batch_size", None)
    prompt = task_module.make_prompt(example)
    state_rollout = bool(generation_cfg.pop("state_rollout", False))
    if hasattr(harness, "generate") and not state_rollout:
        generation_cfg.setdefault("max_new_tokens", task_module.default_length(example))
        text = harness.generate(prompt, **_generation_kwargs(harness.generate, generation_cfg))
        return text, None, _base_stats(generation_cfg)
    state = task_module.initial_state(example, harness)
    final_states, counts = force_complete(harness, [state], [{"prompt": prompt, **generation_cfg}], return_counts=True)
    final_state = final_states[0]
    text = task_module.decode_state(final_state, harness)
    count = int(counts[0]) if counts else 0
    return text, final_state, {"nfe": count, "forward": count, "backward": 0, "forced": 0}


def _generation_kwargs(fn, generation_cfg):
    signature = inspect.signature(fn)
    accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
    if accepts_kwargs:
        return generation_cfg
    return {key: value for key, value in generation_cfg.items() if key in signature.parameters}


def run_base(task_module, harness, examples, config, rank=0):
    generation_cfg = dict(config.get("generation", {}))
    batch_size = max(1, int(generation_cfg.pop("batch_size", 1)))
    state_rollout = bool(generation_cfg.pop("state_rollout", False))
    if hasattr(harness, "generate_batch") and not state_rollout and batch_size > 1:
        rows = []
        starts = range(0, len(examples), batch_size)
        iterator = tqdm(starts, desc="Base inference", total=(len(examples) + batch_size - 1) // batch_size) if rank == 0 else starts
        for start in iterator:
            batch = examples[start:start + batch_size]
            prompts = [task_module.make_prompt(example) for example in batch]
            generation_args = dict(generation_cfg)
            if "max_new_tokens" not in generation_args:
                generation_args["max_new_tokens"] = max(task_module.default_length(example) for example in batch)
            outputs = harness.generate_batch(prompts, **_generation_kwargs(harness.generate_batch, generation_args))
            stats = _base_stats(generation_cfg)
            for example, text in zip(batch, outputs):
                rows.append(build_row(task_module, example, harness, text, None, dict(stats)))
        return rows

    if hasattr(task_module, "initial_state") and (state_rollout or not hasattr(harness, "generate")):
        rows = []
        starts = range(0, len(examples), batch_size)
        iterator = tqdm(starts, desc="Base inference", total=(len(examples) + batch_size - 1) // batch_size) if rank == 0 else starts
        for start in iterator:
            batch = examples[start:start + batch_size]
            states = [task_module.initial_state(example, harness) for example in batch]
            configs = []
            for example in batch:
                prompt = task_module.make_prompt(example)
                configs.append({"prompt": prompt, **generation_cfg})
            final_states, counts = force_complete(harness, states, configs, return_counts=True)
            for example, state, count in zip(batch, final_states, counts):
                text = task_module.decode_state(state, harness)
                count = int(count)
                stats = {"nfe": count, "forward": count, "backward": 0, "forced": 0}
                rows.append(build_row(task_module, example, harness, text, state, stats))
        return rows

    rows = []
    iterator = tqdm(examples, desc="Base inference") if rank == 0 else examples
    for example in iterator:
        text, state, stats = base_sample(task_module, harness, example, config)
        rows.append(build_row(task_module, example, harness, text, state, stats))
    return rows


def run_bon(task_module, harness, examples, config, rank=0):
    cfg = value_guidance_config(config, "BoN")
    n = int(cfg.get("N", 1))
    generation_cfg = dict(config.get("generation", {}))
    batch_size = max(1, int(generation_cfg.get("batch_size", 64)))
    state_rollout = bool(generation_cfg.pop("state_rollout", False))
    generation_cfg.pop("batch_size", None)

    best_rows = [None for _ in examples]
    nfe_totals = [0 for _ in examples]

    def keep_best(example_idx, row):
        best_row = best_rows[example_idx]
        if best_row is None or float(row.get("reward", 0.0)) > float(best_row.get("reward", 0.0)):
            best_rows[example_idx] = row

    if hasattr(harness, "generate_batch") and not state_rollout:
        candidate_nfe = _base_nfe(generation_cfg)
        candidate_stats = _base_stats(generation_cfg)
        total = len(examples) * n
        starts = range(0, total, batch_size)
        iterator = tqdm(starts, desc="BoN inference", total=(total + batch_size - 1) // batch_size) if rank == 0 else starts
        for start in iterator:
            end = min(start + batch_size, total)
            indexed = [(flat_idx // n, examples[flat_idx // n]) for flat_idx in range(start, end)]
            prompts = [task_module.make_prompt(example) for _, example in indexed]
            generation_args = dict(generation_cfg)
            if "max_new_tokens" not in generation_args:
                generation_args["max_new_tokens"] = max(task_module.default_length(example) for _, example in indexed)
            outputs = harness.generate_batch(prompts, **_generation_kwargs(harness.generate_batch, generation_args))
            for (example_idx, example), text in zip(indexed, outputs):
                nfe_totals[example_idx] += candidate_nfe
                keep_best(example_idx, build_row(task_module, example, harness, text, None, dict(candidate_stats)))
    elif hasattr(task_module, "initial_state"):
        grouped = {}
        for example_idx, example in enumerate(examples):
            grouped.setdefault(int(task_module.default_length(example)), []).append((example_idx, example))
        total = sum(len(group) * n for group in grouped.values())
        progress = tqdm(total=(total + batch_size - 1) // batch_size, desc="BoN inference") if rank == 0 else None
        for length, group in grouped.items():
            group_total = len(group) * n
            for start in range(0, group_total, batch_size):
                end = min(start + batch_size, group_total)
                indexed = [group[flat_idx // n] for flat_idx in range(start, end)]
                states = [task_module.initial_state(example, harness) for _, example in indexed]
                configs = []
                for _, example in indexed:
                    prompt = task_module.make_prompt(example)
                    gen_cfg = dict(generation_cfg)
                    gen_cfg.setdefault("max_new_tokens", length)
                    configs.append({"prompt": prompt, **gen_cfg})
                final_states, counts = force_complete(harness, states, configs, return_counts=True)
                for (example_idx, example), state, count in zip(indexed, final_states, counts):
                    text = task_module.decode_state(state, harness)
                    count = int(count)
                    nfe_totals[example_idx] += count
                    keep_best(
                        example_idx,
                        build_row(
                            task_module,
                            example,
                            harness,
                            text,
                            state,
                            {"forward": count, "backward": 0, "forced": 0, "nfe": count},
                        ),
                    )
                if progress is not None:
                    progress.update(1)
        if progress is not None:
            progress.close()
    else:
        iterator = tqdm(examples, desc="BoN inference") if rank == 0 else examples
        for example_idx, example in enumerate(iterator):
            for _ in range(n):
                text, state, stats = base_sample(task_module, harness, example, config)
                nfe_totals[example_idx] += int(stats.get("nfe", 1))
                keep_best(example_idx, build_row(task_module, example, harness, text, state, stats))

    rows = []
    for example_idx, (example, row) in enumerate(zip(examples, best_rows)):
        assert row is not None
        row["example"] = example_json(task_module, example)
        total_nfe = int(nfe_totals[example_idx] or n)
        row["stats"]["nfe"] = total_nfe
        row["stats"]["forward"] = total_nfe
        row["stats"].setdefault("backward", 0)
        row["stats"].setdefault("forced", 0)
        row["stats"]["bon_candidates"] = n
        rows.append(row)
    return rows


def run_value_guided(
    task_module,
    harness,
    verifier,
    examples,
    config,
    algorithm,
    rank=0,
):
    cfg = value_guidance_config(config, algorithm)
    maybe_set_forward_token_filter(task_module, harness, examples, cfg)
    cfg["progress"] = rank == 0
    cfg["progress_desc"] = f"{algorithm} inference"
    rows = []
    if algorithm == "VGR":
        sampling_fn = sample_vgr
    elif algorithm == "VGB":
        sampling_fn = sample_vgb
    else:
        sampling_fn = sample_vgb_momentum
    grouped = {}
    for idx, example in enumerate(examples):
        grouped.setdefault(int(task_module.default_length(example)), []).append((idx, example))
    if len(grouped) <= 1:
        samples = sampling_fn(harness, task_module, examples, verifier=verifier, config=cfg)
    else:
        samples = [None for _ in examples]
        for length, group in grouped.items():
            group_indices = [idx for idx, _ in group]
            group_examples = [example for _, example in group]
            group_cfg = dict(cfg, length=length)
            group_samples = sampling_fn(harness, task_module, group_examples, verifier=verifier, config=group_cfg)
            for idx, sample in zip(group_indices, group_samples):
                samples[idx] = sample
        if any(sample is None for sample in samples):
            raise RuntimeError("value-guided sampling function did not return a sample for every example")
    for example, sample in zip(examples, samples):
        text = sample["output"]
        stats = dict(sample.get("stats", {}))
        rows.append(build_row(task_module, example, harness, text, sample.get("state"), stats))
    return rows


def wait_for_rank_outputs(output, world_size, timeout_sec=86400.0):
    deadline = time.time() + timeout_sec
    missing = [f"{output}.rank{rank:05d}.tmp.done" for rank in range(world_size)]
    while missing:
        missing = [path for path in missing if not os.path.exists(path)]
        if not missing:
            return
        if time.time() >= deadline:
            raise TimeoutError(f"Timed out waiting for distributed outputs: {missing}")
        time.sleep(1.0)


def collect_rows(rows, output, dist_info):
    if not dist_info.get("distributed"):
        if output:
            write_jsonl(output, rows)
        return rows
    rank = int(dist_info["rank"])
    world_size = int(dist_info["world_size"])
    if output:
        part_path = f"{output}.rank{rank:05d}.tmp"
        write_jsonl(part_path, rows)
        done_path = f"{part_path}.done"
        write_jsonl(done_path, [{"rank": rank, "rows": len(rows)}])
        if rank != 0:
            return []
        wait_for_rank_outputs(output, world_size)
        merged = []
        for part_rank in range(world_size):
            part_path = f"{output}.rank{part_rank:05d}.tmp"
            done_path = f"{part_path}.done"
            merged.extend(read_jsonl(part_path))
            os.remove(part_path)
            os.remove(done_path)
        write_jsonl(output, merged)
        return merged
    gathered = gather_objects(rows, dist_info)
    if rank != 0:
        return []
    return [row for part in gathered for row in part]


def main():
    args = parse_args()
    config = load_task_config(args.task, stage=args.stage)
    if args.count is not None:
        config.setdefault("data", {})["count"] = int(args.count)
    value_guidance_cfg = value_guidance_section(config)
    apply_budget_overrides(
        config,
        args.algorithm,
        n=args.N,
        max_steps_multiplier=args.max_steps_multiplier,
    )
    if args.K is not None:
        value_guidance_cfg["K"] = args.K if str(args.K).lower() == "all" else int(args.K)
    if args.output is not None:
        config["output"] = args.output
    if args.verifier is not None:
        config.setdefault("verifier", {})["checkpoint"] = args.verifier
    dist_info = init_distributed()
    seed = config.get("seed")
    if seed is not None and dist_info.get("distributed"):
        seed = int(seed) + int(dist_info["rank"])
    set_seed(seed)
    logger = setup_logging(config.get("logging", {}).get("level", "INFO"))

    task_name = normalize_task_name(config.get("task", args.task))
    task_module = load_task_module(task_name)
    examples = shard_items(load_examples(task_module, config), dist_info)
    device = args.device or dist_info.get("device")
    try:
        harness = load_harness(task_name, config, device=device)

        rank = int(dist_info["rank"])
        if args.algorithm == "Base":
            rows = run_base(task_module, harness, examples, config, rank=rank)
        elif args.algorithm == "BoN":
            rows = run_bon(task_module, harness, examples, config, rank=rank)
        else:
            verifier = load_verifier(task_name, config, harness=harness, device=device)
            rows = run_value_guided(task_module, harness, verifier, examples, config, args.algorithm, rank=rank)

        output = resolve_output(config, task_name, args.algorithm, args.stage)
        rows = collect_rows(rows, output, dist_info)
        if int(dist_info["rank"]) != 0:
            return
        if output:
            logger.info("wrote %d rows to %s", len(rows), output)

        metrics = evaluate_rows(task_name, rows, config=config)
        metrics = add_nfe_metrics(metrics, rows)
        log_metrics(metrics, logger)
    finally:
        cleanup_distributed(dist_info)


if __name__ == "__main__":
    main()
