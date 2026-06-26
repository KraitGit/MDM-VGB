from tqdm.auto import tqdm

from .algorithm_utils import (
    active_batch,
    force_complete_indices,
    initial_states_and_prompts,
    is_eos_choice,
    is_leaf,
    is_leaf_accept,
    length_max_steps,
    logits,
    sampler_outputs,
    sample_grouped_choices,
    state_has_eos,
    terminal_state,
)
from .vgb_candidates import build_mdm_candidates, build_mdm_forward_candidates
from .vgb_terminal import sample_edge_walk_mdm, sample_state_value_mdm


def sample_vgb(harness, task, examples, verifier=None, config=None):
    config = dict(config or {})
    if config.get("terminal_keep_best"):
        return sample_state_value_mdm(harness, task, examples, verifier=verifier, config=config, momentum=False)
    if config.get("terminal_reject") == "backtrack":
        return sample_edge_walk_mdm(harness, task, examples, verifier=verifier, config=config)

    states, prompts = initial_states_and_prompts(task, harness, examples, config)
    max_steps = length_max_steps(config, states)
    trajectories = [[] for _ in examples]
    finished = [False for _ in examples]
    stats = [
        {
            "forward": 0,
            "backward": 0,
            "forced": 0,
            "leaf_reached": False,
            "full_leaf_reached": False,
            "eos_reached": False,
            "eos_accepted": False,
            "eos_rejected": 0,
            "accepted": False,
            "nfe": 0,
        }
        for _ in examples
    ]
    step = 0
    progress = tqdm(
        total=max_steps,
        desc=str(config.get("progress_desc", "VGB inference")),
        dynamic_ncols=True,
    ) if config.get("progress") else None
    while max_steps is None or step < max_steps:
        active = [idx for idx, state in enumerate(states) if not finished[idx] and not is_leaf(state, harness.mask_id)]
        if not active:
            break
        step += 1
        if progress is not None:
            progress.update(1)
            progress.set_postfix(active=len(active))
        for idx in active:
            trajectories[idx].append(list(states[idx]))

        active_examples, active_prompts, active_states = active_batch(examples, prompts, states, active)
        logit_tensor = logits(harness, active_prompts, active_states)
        for idx in active:
            stats[idx]["nfe"] += 1
        fwd_groups, f_masses, bwd_groups, b_masses = build_mdm_candidates(
            task,
            harness,
            verifier,
            active_examples,
            active_states,
            logit_tensor,
            None,
            config,
        )
        del f_masses, b_masses

        grouped = [fwd + bwd for fwd, bwd in zip(fwd_groups, bwd_groups)]
        choices = sample_grouped_choices(grouped)
        for local_idx, idx in enumerate(active):
            choice = choices[local_idx]
            if choice is None:
                continue
            states[idx] = choice["state"]
            stats[idx][choice["kind"]] += 1
            if is_eos_choice(harness, config, choice):
                stats[idx]["eos_reached"] = True
                stats[idx]["leaf_reached"] = True
                if is_leaf_accept(task, harness, examples[idx], states[idx], config):
                    finished[idx] = True
                    stats[idx]["eos_accepted"] = True
                    stats[idx]["accepted"] = True
                else:
                    stats[idx]["eos_rejected"] += 1
                    finished[idx] = True

    forced_indices = []
    for idx, state in enumerate(states):
        if is_leaf(state, harness.mask_id):
            stats[idx]["full_leaf_reached"] = True
            stats[idx]["leaf_reached"] = True
        elif finished[idx]:
            stats[idx]["leaf_reached"] = True
        else:
            stats[idx]["forced"] = 1
            forced_indices.append(idx)

    if forced_indices:
        states = force_complete_indices(task, harness, verifier, examples, prompts, states, forced_indices, stats, config, build_mdm_forward_candidates)
        for idx in forced_indices:
            state = states[idx]
            if is_leaf(state, harness.mask_id):
                stats[idx]["full_leaf_reached"] = True
                stats[idx]["leaf_reached"] = True
            if state_has_eos(harness, state):
                stats[idx]["eos_reached"] = True
                stats[idx]["leaf_reached"] = True
            if terminal_state(harness, state, config) and is_leaf_accept(task, harness, examples[idx], state, config):
                stats[idx]["eos_accepted"] = bool(state_has_eos(harness, state))
                stats[idx]["accepted"] = True
            elif state_has_eos(harness, state):
                stats[idx]["eos_rejected"] += 1
    if progress is not None:
        progress.close()

    return sampler_outputs(harness, states, stats, trajectories)
