import math
import random

from tqdm.auto import tqdm

from .algorithm_utils import (
    force_complete,
    force_complete_value,
    initial_state,
    is_eos_choice,
    is_leaf,
    is_leaf_accept,
    locked_positions,
    logits,
    sample_grouped_choices,
    state_has_eos,
    terminal_on_eos,
    terminal_state,
    values_for_states,
)
from .vgb import (
    _candidates_mdm,
    _forward_candidates_mdm,
    _sample_state_value_mdm,
)


def _active_indices(states, mask_id, finished):
    del mask_id
    return [idx for idx, _ in enumerate(states) if not finished[idx]]


def _max_steps(config, states, task=None, examples=None, harness=None):
    value = config.get("max_steps")
    if value is not None:
        return int(value)
    multiplier = config.get("max_steps_multiplier", config.get("N", 8))
    if multiplier is None or int(multiplier) <= 0:
        return None
    scope = str(config.get("max_steps_scope", config.get("budget_scope", "length")))
    if scope in {"mutable", "editable"} and task is not None and examples is not None and harness is not None:
        budgets = []
        for example, state in zip(examples, states):
            locked = locked_positions(task, example, state, harness)
            mutable = sum(1 for pos in range(len(state)) if pos not in locked)
            budgets.append(max(1, int(mutable)) * int(multiplier))
        return budgets
    max_len = max(len(state) for state in states) if states else 0
    return int(max_len * int(multiplier))


def _budget_value(max_steps, idx):
    if isinstance(max_steps, list):
        return max_steps[int(idx)]
    return max_steps


def _progress_total(max_steps):
    if isinstance(max_steps, list):
        return max(max_steps, default=None)
    return max_steps


def _stats():
    return {
        "forward": 0,
        "backward": 0,
        "switch_up": 0,
        "switch_down": 0,
        "leaf_backtrack": 0,
        "forced": 0,
        "leaf_reached": False,
        "full_leaf_reached": False,
        "eos_reached": False,
        "eos_accepted": False,
        "eos_rejected": 0,
        "accepted": False,
        "mass_steps": 0,
        "f_mass_sum": 0.0,
        "b_mass_sum": 0.0,
        "fb_log_ratio_sum": 0.0,
        "f_mass_gt_b_mass": 0,
        "b_mass_gt_f_mass": 0,
        "nfe": 0,
    }


def _positive(value):
    return max(float(value), 0.0)


def _switch_mass(opposite_mass, same_mass, chi):
    opposite_mass = _positive(opposite_mass)
    same_mass = _positive(same_mass)
    if chi is None:
        return max(opposite_mass - same_mass, 0.0)
    chi = max(0.0, min(1.0, float(chi)))
    return max(opposite_mass - chi * min(opposite_mass, same_mass), 0.0)


def _directed_choices(direction, state, fwd, f_mass, bwd, b_mass, chi):
    if direction == "down":
        choices = list(fwd)
        switch = _switch_mass(b_mass, f_mass, chi)
        if switch > 0.0:
            choices.append({"kind": "switch_up", "state": state, "weight": switch})
        return choices

    choices = list(bwd)
    switch = _switch_mass(f_mass, b_mass, chi)
    if switch > 0.0:
        choices.append({"kind": "switch_down", "state": state, "weight": switch})
    return choices


def _add_mass_stats(stats, f_mass, b_mass):
    f_mass = _positive(f_mass)
    b_mass = _positive(b_mass)
    stats["mass_steps"] += 1
    stats["f_mass_sum"] += f_mass
    stats["b_mass_sum"] += b_mass
    stats["fb_log_ratio_sum"] += math.log((f_mass + 1e-30) / (b_mass + 1e-30))
    if f_mass > b_mass:
        stats["f_mass_gt_b_mass"] += 1
    elif b_mass > f_mass:
        stats["b_mass_gt_f_mass"] += 1


def _current_values(task, harness, verifier, examples, states, cached_values, config):
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


def _note_terminal(stats, harness, state, stop_on_eos):
    stats["leaf_reached"] = True
    if is_leaf(state, harness.mask_id):
        stats["full_leaf_reached"] = True
    if stop_on_eos and state_has_eos(harness, state):
        stats["eos_reached"] = True


def _leaf_backtrack(task, harness, example, state, mask_id, config):
    locked = locked_positions(task, example, state, harness)
    revealed = [idx for idx, token in enumerate(state) if int(token) != int(mask_id) and idx not in locked]
    if not revealed:
        return list(state), False
    block_size = min(max(1, int(config.get("B", 1) or 1)), len(revealed))
    positions = random.sample(revealed, block_size)
    parent = list(state)
    for pos in positions:
        parent[pos] = int(mask_id)
    return parent, True


def _handle_leaf_states(
    task,
    harness,
    examples,
    states,
    directions,
    state_values,
    finished,
    stats,
    active,
    config,
    max_steps,
    stop_on_eos,
):
    nonleaf = []
    for idx in active:
        state = states[idx]
        if not is_leaf(state, harness.mask_id):
            nonleaf.append(idx)
            continue

        _note_terminal(stats[idx], harness, state, stop_on_eos)
        if is_leaf_accept(task, harness, examples[idx], state, config):
            finished[idx] = True
            state_values[idx] = 1.0
            stats[idx]["accepted"] = True
            continue
        budget = _budget_value(max_steps, idx) if max_steps is not None else None
        if budget is not None and float(stats[idx].get("nfe", 0.0)) >= float(budget):
            finished[idx] = True
            state_values[idx] = 0.0
            continue

        parent, moved = _leaf_backtrack(task, harness, examples[idx], state, harness.mask_id, config)
        if moved:
            states[idx] = parent
            state_values[idx] = None
            directions[idx] = "up"
            stats[idx]["nfe"] += 1
            stats[idx]["leaf_backtrack"] = stats[idx].get("leaf_backtrack", 0) + 1
            continue
        finished[idx] = True
        state_values[idx] = 0.0
    return nonleaf


def _force_complete_unfinished(
    task,
    harness,
    verifier,
    examples,
    prompts,
    states,
    stats,
    config,
):
    forced = []
    stop_on_eos = terminal_on_eos(config)
    for idx, state in enumerate(states):
        if is_leaf(state, harness.mask_id):
            _note_terminal(stats[idx], harness, state, stop_on_eos)
            if is_leaf_accept(task, harness, examples[idx], state, config):
                stats[idx]["accepted"] = True
        elif stop_on_eos and state_has_eos(harness, state):
            _note_terminal(stats[idx], harness, state, stop_on_eos)
            if is_leaf_accept(task, harness, examples[idx], state, config):
                stats[idx]["accepted"] = True
                stats[idx]["eos_accepted"] = True
        else:
            stats[idx]["forced"] = 1
            forced.append(idx)

    if not forced:
        return states

    forced_states = [states[idx] for idx in forced]
    if getattr(harness, "kind", None) == "ar":
        completed, extra_forward = force_complete_value(
            task,
            harness,
            verifier,
            [examples[idx] for idx in forced],
            forced_states,
            [prompts[idx] for idx in forced],
            config,
            _forward_candidates_mdm,
        )
    else:
        completed, extra_forward = force_complete(
            harness,
            forced_states,
            [dict(config, prompt=prompts[idx]) for idx in forced],
            return_counts=True,
        )

    for idx, count in zip(forced, extra_forward):
        stats[idx]["forward"] += count
        stats[idx]["nfe"] += count

    for idx, state in zip(forced, completed):
        states[idx] = state
        if terminal_state(harness, state, config):
            _note_terminal(stats[idx], harness, state, stop_on_eos)
            if is_leaf_accept(task, harness, examples[idx], state, config):
                stats[idx]["accepted"] = True
                stats[idx]["eos_accepted"] = bool(state_has_eos(harness, state))
            elif stop_on_eos and state_has_eos(harness, state):
                stats[idx]["eos_rejected"] += 1
    return states


def sample_vgb_momentum(harness, task, examples, verifier=None, config=None, candidate_builder=None):
    config = dict(config or {})
    candidate_builder = candidate_builder or _candidates_mdm

    length = config.get("length")
    states = [
        initial_state(
            task,
            example,
            harness,
            int(task.default_length(example) if length is None else length),
        )
        for example in examples
    ]
    prompts = [task.make_prompt(example) for example in examples]
    directions = ["down" for _ in examples]
    state_values = [None for _ in examples]
    finished = [False for _ in examples]
    trajectories = [[] for _ in examples]
    stats = [_stats() for _ in examples]

    stop_on_eos = terminal_on_eos(config)
    max_steps = _max_steps(config, states, task=task, examples=examples, harness=harness)

    def handle_terminal(idx, count_rejection=False):
        state = states[idx]
        _note_terminal(stats[idx], harness, state, stop_on_eos)
        if is_leaf_accept(task, harness, examples[idx], state, config):
            finished[idx] = True
            state_values[idx] = 1.0
            stats[idx]["accepted"] = True
            if stop_on_eos and state_has_eos(harness, state):
                stats[idx]["eos_accepted"] = True
            return True

        if count_rejection and stop_on_eos and state_has_eos(harness, state):
            stats[idx]["eos_rejected"] += 1
        directions[idx] = "up"
        finished[idx] = True
        return False

    step = 0
    progress = tqdm(
        total=_progress_total(max_steps),
        desc=str(config.get("progress_desc", "VGB-Momentum inference")),
        dynamic_ncols=True,
    ) if config.get("progress") else None
    while True:
        active = _active_indices(states, harness.mask_id, finished)
        if not active:
            break

        active = _handle_leaf_states(
            task,
            harness,
            examples,
            states,
            directions,
            state_values,
            finished,
            stats,
            active,
            config,
            max_steps,
            stop_on_eos,
        )
        if not active:
            continue

        at_budget = {
            idx: max_steps is not None
            and float(stats[idx].get("nfe", 0.0)) >= float(_budget_value(max_steps, idx))
            for idx in active
        }
        step += 1
        if progress is not None:
            progress.update(1)
            progress.set_postfix(active=len(active))
        for idx in active:
            trajectories[idx].append(list(states[idx]))

        active_states = [states[idx] for idx in active]
        active_prompts = [prompts[idx] for idx in active]
        active_examples = [examples[idx] for idx in active]
        logit_tensor = logits(harness, active_prompts, active_states)
        for idx in active:
            stats[idx]["nfe"] += 1

        current_values = _current_values(
            task,
            harness,
            verifier,
            active_examples,
            active_states,
            [state_values[idx] for idx in active],
            config,
        )
        fwd_groups, f_masses, bwd_groups, b_masses = candidate_builder(
            task,
            harness,
            verifier,
            active_examples,
            active_states,
            logit_tensor,
            current_values,
            config,
        )

        grouped_choices = []
        for local_idx, idx in enumerate(active):
            _add_mass_stats(stats[idx], f_masses[local_idx], b_masses[local_idx])
            if at_budget[idx]:
                grouped_choices.append(list(fwd_groups[local_idx]))
            else:
                grouped_choices.append(
                    _directed_choices(
                        directions[idx],
                        states[idx],
                        fwd_groups[local_idx],
                        f_masses[local_idx],
                        bwd_groups[local_idx],
                        b_masses[local_idx],
                        config.get("chi"),
                    )
                )

        choices = sample_grouped_choices(grouped_choices)

        for local_idx, idx in enumerate(active):
            choice = choices[local_idx]
            if choice is None:
                if at_budget[idx]:
                    finished[idx] = True
                    state_values[idx] = 0.0
                    continue
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
            state_values[idx] = choice.get("state_value", choice.get("value"))
            stats[idx][kind] += 1

            if is_eos_choice(harness, config, choice):
                handle_terminal(idx, count_rejection=True)
                continue
            if is_leaf(states[idx], harness.mask_id):
                _note_terminal(stats[idx], harness, states[idx], stop_on_eos)

    if config.get("force_complete_at_end", True):
        states = _force_complete_unfinished(
            task,
            harness,
            verifier,
            examples,
            prompts,
            states,
            stats,
            config,
        )
    if progress is not None:
        progress.close()

    return [
        {
            "output": harness.decode_state(state),
            "state": state,
            "trajectory": trajectories[idx],
            "stats": stats[idx],
        }
        for idx, state in enumerate(states)
    ]


def sample_vgb_momentum_state_value(harness, task, examples, verifier=None, config=None, candidate_builder=None):
    del candidate_builder
    return _sample_state_value_mdm(harness, task, examples, verifier=verifier, config=config, momentum=True)
