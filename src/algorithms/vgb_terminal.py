import math

import torch
from tqdm.auto import tqdm

from .algorithm_utils import (
    active_batch,
    force_complete_indices,
    initial_states_and_prompts,
    is_eos_choice,
    is_leaf,
    is_leaf_accept,
    length_max_steps,
    locked_positions,
    logits,
    sampler_outputs,
    sample_grouped_choices,
    state_has_eos,
    terminal_on_eos,
    terminal_state,
    values_for_states,
)
from .vgb_candidates import (
    build_mdm_candidates,
    build_state_value_candidates,
    build_state_value_forward_candidates,
    value_epsilon,
)


def budgeted_max_steps(config, task, examples, states, harness):
    value = config.get("max_steps")
    if value is not None:
        return [int(value) for _ in states]
    multiplier = config.get("max_steps_multiplier", 8)
    if multiplier is None or int(multiplier) <= 0:
        return None
    unit = str(config.get("budget_unit", "length"))
    budgets = []
    for example, state in zip(examples, states):
        if unit == "editable_tokens":
            locked = locked_positions(task, example, state, harness)
            count = sum(1 for pos in range(len(state)) if pos not in locked)
        else:
            count = len(state)
        budgets.append(max(1, int(count)) * int(multiplier))
    return budgets


def budget_value(max_steps, idx):
    if isinstance(max_steps, list):
        return max_steps[int(idx)]
    return max_steps


def progress_total(max_steps):
    if isinstance(max_steps, list):
        return max(max_steps, default=None)
    return max_steps


def resolve_current_values(task, harness, verifier, examples, states, cached_values, config):
    if any(value is None for value in cached_values):
        return values_for_states(
            task,
            harness,
            verifier,
            examples,
            states,
            config=config,
        )
    return list(cached_values)


def state_value_stats():
    return {
        "forward": 0,
        "backward": 0,
        "switch_up": 0,
        "switch_down": 0,
        "forced": 0,
        "leaf_reached": False,
        "full_leaf_reached": False,
        "eos_reached": False,
        "eos_accepted": False,
        "eos_rejected": 0,
        "accepted": False,
        "terminal_checks": 0,
        "terminal_accepts": 0,
        "terminal_rejects": 0,
        "mass_steps": 0,
        "f_mass_sum": 0.0,
        "b_mass_sum": 0.0,
        "fb_log_ratio_sum": 0.0,
        "f_mass_gt_b_mass": 0,
        "b_mass_gt_f_mass": 0,
        "nfe": 0,
    }


def positive_mass(value):
    return max(float(value), 0.0)


def switch_mass(opposite_mass, same_mass, chi):
    opposite_mass = positive_mass(opposite_mass)
    same_mass = positive_mass(same_mass)
    chi = max(0.0, min(1.0, float(chi if chi is not None else 1.0)))
    return max(opposite_mass - chi * min(opposite_mass, same_mass), 0.0)


def directed_choices(direction, state, fwd, f_mass, bwd, b_mass, chi):
    if direction == "down":
        choices = list(fwd)
        switch = switch_mass(b_mass, f_mass, chi)
        if switch > 0.0:
            choices.append({"kind": "switch_up", "state": state, "weight": switch})
        return choices
    choices = list(bwd)
    switch = switch_mass(f_mass, b_mass, chi)
    if switch > 0.0:
        choices.append({"kind": "switch_down", "state": state, "weight": switch})
    return choices


def add_mass_stats(stats, f_mass, b_mass):
    f_mass = positive_mass(f_mass)
    b_mass = positive_mass(b_mass)
    stats["mass_steps"] += 1
    stats["f_mass_sum"] += f_mass
    stats["b_mass_sum"] += b_mass
    stats["fb_log_ratio_sum"] += math.log((f_mass + 1e-30) / (b_mass + 1e-30))
    if f_mass > b_mass:
        stats["f_mass_gt_b_mass"] += 1
    elif b_mass > f_mass:
        stats["b_mass_gt_f_mass"] += 1


def note_terminal(stats, harness, state, stop_on_eos):
    stats["leaf_reached"] = True
    if is_leaf(state, harness.mask_id):
        stats["full_leaf_reached"] = True
    if stop_on_eos and state_has_eos(harness, state):
        stats["eos_reached"] = True


def score_terminal_state(task, harness, example, state, config):
    eps = value_epsilon(config)
    try:
        if hasattr(task, "reward_state"):
            return max(float(task.reward_state(example, state, harness)), eps)
        text = harness.decode_state(state)
        if hasattr(task, "reward"):
            return max(float(task.reward(example, text)), eps)
    except Exception:
        return eps
    return eps


def _force_complete_unfinished_state_value(task, harness, verifier, examples, prompts, states, stats, config):
    forced = []
    stop_on_eos = terminal_on_eos(config)
    for idx, state in enumerate(states):
        if terminal_state(harness, state, config):
            note_terminal(stats[idx], harness, state, stop_on_eos)
            if is_leaf_accept(task, harness, examples[idx], state, config):
                stats[idx]["accepted"] = True
                if stop_on_eos and state_has_eos(harness, state):
                    stats[idx]["eos_accepted"] = True
        else:
            stats[idx]["forced"] = 1
            forced.append(idx)

    if not forced:
        return states

    states = force_complete_indices(task, harness, verifier, examples, prompts, states, forced, stats, config, build_state_value_forward_candidates)
    for idx in forced:
        state = states[idx]
        if terminal_state(harness, state, config):
            note_terminal(stats[idx], harness, state, stop_on_eos)
            if is_leaf_accept(task, harness, examples[idx], state, config):
                stats[idx]["accepted"] = True
    return states


def sample_state_value_mdm(harness, task, examples, verifier=None, config=None, momentum=False):
    config = dict(config or {})
    states, prompts = initial_states_and_prompts(task, harness, examples, config)
    directions = ["down" for _ in examples]
    state_values = [None for _ in examples]
    finished = [False for _ in examples]
    terminal_checked = [False for _ in examples]
    best_terminal_states = [None for _ in examples]
    best_terminal_values = [-float("inf") for _ in examples]
    trajectories = [[] for _ in examples]
    stats = [state_value_stats() for _ in examples]
    stop_on_eos = terminal_on_eos(config)
    max_steps = budgeted_max_steps(config, task, examples, states, harness)

    while True:
        active = [idx for idx in range(len(states)) if not finished[idx]]
        if max_steps is not None:
            active = [idx for idx in active if float(stats[idx].get("nfe", 0.0)) < float(max_steps[idx])]
        if not active:
            break

        ready = []
        for idx in active:
            state = states[idx]
            if terminal_state(harness, state, config) and not terminal_checked[idx]:
                note_terminal(stats[idx], harness, state, stop_on_eos)
                stats[idx]["terminal_checks"] += 1
                value = score_terminal_state(task, harness, examples[idx], state, config)
                if value >= best_terminal_values[idx]:
                    best_terminal_values[idx] = float(value)
                    best_terminal_states[idx] = list(state)
                if is_leaf_accept(task, harness, examples[idx], state, config):
                    finished[idx] = True
                    state_values[idx] = float(value)
                    stats[idx]["accepted"] = True
                    stats[idx]["terminal_accepts"] += 1
                    if stop_on_eos and state_has_eos(harness, state):
                        stats[idx]["eos_accepted"] = True
                    continue
                stats[idx]["terminal_rejects"] += 1
                if stop_on_eos and state_has_eos(harness, state):
                    stats[idx]["eos_rejected"] += 1
                terminal_checked[idx] = True
                directions[idx] = "up"
                state_values[idx] = float(value)
            ready.append(idx)

        if not ready:
            continue

        for idx in ready:
            trajectories[idx].append(list(states[idx]))
        active_examples, active_prompts, active_states = active_batch(examples, prompts, states, ready)
        logit_tensor = logits(harness, active_prompts, active_states)
        for idx in ready:
            stats[idx]["nfe"] += 1

        current_values = values_for_states(task, harness, verifier, active_examples, active_states, config=config)
        for local_idx, idx in enumerate(ready):
            if state_values[idx] is not None:
                current_values[local_idx] = max(float(state_values[idx]), value_epsilon(config))

        fwd_groups, f_masses, bwd_groups, b_masses = build_state_value_candidates(
            task,
            harness,
            verifier,
            active_examples,
            active_states,
            logit_tensor,
            current_values,
            config,
        )
        grouped = []
        for local_idx, idx in enumerate(ready):
            add_mass_stats(stats[idx], f_masses[local_idx], b_masses[local_idx])
            if momentum:
                grouped.append(
                    directed_choices(
                        directions[idx],
                        states[idx],
                        fwd_groups[local_idx],
                        f_masses[local_idx],
                        bwd_groups[local_idx],
                        b_masses[local_idx],
                        config.get("chi"),
                    )
                )
            else:
                grouped.append(list(fwd_groups[local_idx]) + list(bwd_groups[local_idx]))

        choices = sample_grouped_choices(grouped)
        for local_idx, idx in enumerate(ready):
            choice = choices[local_idx]
            if choice is None:
                if momentum:
                    directions[idx] = "up" if directions[idx] == "down" else "down"
                continue

            kind = choice["kind"]
            if kind == "switch_up":
                directions[idx] = "up"
                stats[idx]["switch_up"] += 1
                continue
            if kind == "switch_down":
                directions[idx] = "down"
                stats[idx]["switch_down"] += 1
                continue

            states[idx] = choice["state"]
            state_values[idx] = choice.get("state_value")
            terminal_checked[idx] = False
            stats[idx][kind] += 1
            if kind == "backward":
                directions[idx] = "up"
            elif kind == "forward" and momentum:
                directions[idx] = "down"
            if is_eos_choice(harness, config, choice):
                terminal_checked[idx] = False

    for idx, state in enumerate(states):
        if not finished[idx] and terminal_state(harness, state, config) and not terminal_checked[idx]:
            note_terminal(stats[idx], harness, state, stop_on_eos)
            stats[idx]["terminal_checks"] += 1
            value = score_terminal_state(task, harness, examples[idx], state, config)
            if value >= best_terminal_values[idx]:
                best_terminal_values[idx] = float(value)
                best_terminal_states[idx] = list(state)
            if is_leaf_accept(task, harness, examples[idx], state, config):
                finished[idx] = True
                stats[idx]["accepted"] = True
                stats[idx]["terminal_accepts"] += 1
            else:
                stats[idx]["terminal_rejects"] += 1

    for idx, best in enumerate(best_terminal_states):
        if best is not None and not stats[idx]["accepted"]:
            states[idx] = list(best)
            state_values[idx] = max(float(best_terminal_values[idx]), value_epsilon(config))

    if config.get("terminal_force_complete", True):
        states = _force_complete_unfinished_state_value(task, harness, verifier, examples, prompts, states, stats, config)

    return sampler_outputs(harness, states, stats, trajectories)


def _edge_stats():
    return {
        "forward": 0,
        "backward": 0,
        "switch_up": 0,
        "switch_down": 0,
        "leaf_backtrack": 0,
        "forced": 0,
        "leaf_reached": False,
        "full_leaf_reached": False,
        "accepted": False,
        "nfe": 0,
    }


def _move_count(stats):
    return sum(int(stats.get(key, 0) or 0) for key in ("forward", "backward", "leaf_backtrack"))


def leaf_backtrack(task, harness, example, state, mask_id, config):
    locked = locked_positions(task, example, state, harness)
    revealed = [idx for idx, token in enumerate(state) if int(token) != int(mask_id) and idx not in locked]
    if not revealed:
        return list(state), False
    block_size = min(max(1, int(config.get("B", 1) or 1)), len(revealed))
    positions = torch.randperm(len(revealed))[:block_size].tolist()
    parent = list(state)
    for item in positions:
        parent[revealed[int(item)]] = int(mask_id)
    return parent, True


def _handle_edge_walk_leaves(task, harness, examples, states, finished, stats, active, config, max_steps):
    nonleaf = []
    for idx in active:
        state = states[idx]
        if not is_leaf(state, harness.mask_id):
            nonleaf.append(idx)
            continue

        stats[idx]["leaf_reached"] = True
        stats[idx]["full_leaf_reached"] = True
        if is_leaf_accept(task, harness, examples[idx], state, config):
            finished[idx] = True
            stats[idx]["accepted"] = True
            continue
        if max_steps is not None and _move_count(stats[idx]) >= max_steps:
            finished[idx] = True
            continue

        parent, moved = leaf_backtrack(task, harness, examples[idx], state, harness.mask_id, config)
        if moved:
            states[idx] = parent
            stats[idx]["leaf_backtrack"] += 1
            stats[idx]["nfe"] += 1
            continue
        finished[idx] = True
    return nonleaf


def sample_edge_walk_mdm(harness, task, examples, verifier=None, config=None):
    config = dict(config or {})
    states, prompts = initial_states_and_prompts(task, harness, examples, config)
    finished = [False for _ in examples]
    trajectories = [[] for _ in examples]
    stats = [_edge_stats() for _ in examples]
    max_steps = length_max_steps(config, states)

    progress = tqdm(total=max_steps, desc=str(config.get("progress_desc", "VGB inference")), dynamic_ncols=True) if config.get("progress") else None
    shown_step = 0
    while True:
        active = [idx for idx, done in enumerate(finished) if not done]
        if not active:
            break
        if progress is not None:
            step_now = max((_move_count(item) for item in stats), default=shown_step)
            if step_now > shown_step:
                progress.update(step_now - shown_step)
                shown_step = step_now
            progress.set_postfix(active=len(active))

        active = _handle_edge_walk_leaves(task, harness, examples, states, finished, stats, active, config, max_steps)
        if not active:
            continue

        for idx in active:
            trajectories[idx].append(list(states[idx]))
        active_examples, active_prompts, active_states = active_batch(examples, prompts, states, active)
        logit_tensor = logits(harness, active_prompts, active_states)
        for idx in active:
            stats[idx]["nfe"] += 1

        current_values = values_for_states(task, harness, verifier, active_examples, active_states, config=config)
        fwd_groups, _, bwd_groups, _ = build_mdm_candidates(
            task,
            harness,
            verifier,
            active_examples,
            active_states,
            logit_tensor,
            current_values,
            config,
        )
        grouped = []
        for local_idx, idx in enumerate(active):
            if max_steps is not None and _move_count(stats[idx]) >= max_steps:
                grouped.append(list(fwd_groups[local_idx]))
            else:
                grouped.append(list(fwd_groups[local_idx]) + list(bwd_groups[local_idx]))

        choices = sample_grouped_choices(grouped)
        for local_idx, idx in enumerate(active):
            choice = choices[local_idx]
            if choice is None:
                if max_steps is not None and _move_count(stats[idx]) < max_steps:
                    parent, moved = leaf_backtrack(task, harness, examples[idx], states[idx], harness.mask_id, config)
                    if moved:
                        states[idx] = parent
                        stats[idx]["leaf_backtrack"] += 1
                        stats[idx]["nfe"] += 1
                        continue
                finished[idx] = True
                continue
            states[idx] = choice["state"]
            stats[idx][choice["kind"]] += 1

    if progress is not None:
        progress.close()
    return sampler_outputs(harness, states, stats, trajectories)
