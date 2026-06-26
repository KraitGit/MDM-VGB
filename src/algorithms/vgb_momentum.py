from tqdm.auto import tqdm

from .algorithm_utils import (
    active_batch,
    force_complete_indices,
    initial_states_and_prompts,
    is_eos_choice,
    is_leaf,
    is_leaf_accept,
    logits,
    sampler_outputs,
    sample_grouped_choices,
    state_has_eos,
    terminal_on_eos,
    terminal_state,
)
from .vgb_candidates import (
    build_mdm_candidates,
    build_mdm_forward_candidates,
)
from .vgb_terminal import (
    add_mass_stats,
    budget_value,
    budgeted_max_steps,
    directed_choices,
    leaf_backtrack,
    note_terminal,
    progress_total,
    resolve_current_values,
    sample_state_value_mdm,
    state_value_stats,
)


def _active_indices(states, finished):
    return [idx for idx, _ in enumerate(states) if not finished[idx]]


def _stats():
    stats = state_value_stats()
    stats["leaf_backtrack"] = 0
    return stats


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

        note_terminal(stats[idx], harness, state, stop_on_eos)
        if is_leaf_accept(task, harness, examples[idx], state, config):
            finished[idx] = True
            state_values[idx] = 1.0
            stats[idx]["accepted"] = True
            continue
        budget = budget_value(max_steps, idx) if max_steps is not None else None
        if budget is not None and float(stats[idx].get("nfe", 0.0)) >= float(budget):
            finished[idx] = True
            state_values[idx] = 0.0
            continue

        parent, moved = leaf_backtrack(task, harness, examples[idx], state, harness.mask_id, config)
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
            note_terminal(stats[idx], harness, state, stop_on_eos)
            if is_leaf_accept(task, harness, examples[idx], state, config):
                stats[idx]["accepted"] = True
        elif stop_on_eos and state_has_eos(harness, state):
            note_terminal(stats[idx], harness, state, stop_on_eos)
            if is_leaf_accept(task, harness, examples[idx], state, config):
                stats[idx]["accepted"] = True
                stats[idx]["eos_accepted"] = True
        else:
            stats[idx]["forced"] = 1
            forced.append(idx)

    if not forced:
        return states

    states = force_complete_indices(task, harness, verifier, examples, prompts, states, forced, stats, config, build_mdm_forward_candidates)
    for idx in forced:
        state = states[idx]
        if terminal_state(harness, state, config):
            note_terminal(stats[idx], harness, state, stop_on_eos)
            if is_leaf_accept(task, harness, examples[idx], state, config):
                stats[idx]["accepted"] = True
                stats[idx]["eos_accepted"] = bool(state_has_eos(harness, state))
            elif stop_on_eos and state_has_eos(harness, state):
                stats[idx]["eos_rejected"] += 1
    return states


def sample_vgb_momentum(harness, task, examples, verifier=None, config=None):
    config = dict(config or {})
    if config.get("terminal_keep_best"):
        return sample_state_value_mdm(harness, task, examples, verifier=verifier, config=config, momentum=True)

    states, prompts = initial_states_and_prompts(task, harness, examples, config)
    directions = ["down" for _ in examples]
    state_values = [None for _ in examples]
    finished = [False for _ in examples]
    trajectories = [[] for _ in examples]
    stats = [_stats() for _ in examples]

    stop_on_eos = terminal_on_eos(config)
    max_steps = budgeted_max_steps(config, task, examples, states, harness)

    def handle_terminal(idx, count_rejection=False):
        state = states[idx]
        note_terminal(stats[idx], harness, state, stop_on_eos)
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

    progress = tqdm(
        total=progress_total(max_steps),
        desc=str(config.get("progress_desc", "VGB-Momentum inference")),
        dynamic_ncols=True,
    ) if config.get("progress") else None
    while True:
        active = _active_indices(states, finished)
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
            and float(stats[idx].get("nfe", 0.0)) >= float(budget_value(max_steps, idx))
            for idx in active
        }
        if progress is not None:
            progress.update(1)
            progress.set_postfix(active=len(active))
        for idx in active:
            trajectories[idx].append(list(states[idx]))

        active_examples, active_prompts, active_states = active_batch(examples, prompts, states, active)
        logit_tensor = logits(harness, active_prompts, active_states)
        for idx in active:
            stats[idx]["nfe"] += 1

        current_values = resolve_current_values(
            task,
            harness,
            verifier,
            active_examples,
            active_states,
            [state_values[idx] for idx in active],
            config,
        )
        fwd_groups, f_masses, bwd_groups, b_masses = build_mdm_candidates(
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
            add_mass_stats(stats[idx], f_masses[local_idx], b_masses[local_idx])
            if at_budget[idx]:
                grouped_choices.append(list(fwd_groups[local_idx]))
            else:
                grouped_choices.append(
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
            state_values[idx] = choice.get("state_value")
            stats[idx][kind] += 1

            if is_eos_choice(harness, config, choice):
                handle_terminal(idx, count_rejection=True)
                continue
            if is_leaf(states[idx], harness.mask_id):
                note_terminal(stats[idx], harness, states[idx], stop_on_eos)

    if config.get("terminal_force_complete", True):
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

    return sampler_outputs(harness, states, stats, trajectories)
