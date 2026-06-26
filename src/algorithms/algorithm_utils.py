import random
import torch


def masked_positions(state, mask_id):
    return [idx for idx, token in enumerate(state) if int(token) == int(mask_id)]


def is_leaf(state, mask_id):
    return all(int(token) != int(mask_id) for token in state)


def initial_state(task, example, harness, length):
    if hasattr(task, "initial_state"):
        return task.initial_state(example, harness)
    return [harness.mask_id for _ in range(length)]


def initial_states_and_prompts(task, harness, examples, config=None):
    config = config or {}
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
    return states, prompts


def length_max_steps(config, states):
    if config.get("max_steps") is not None:
        return int(config["max_steps"])
    multiplier = config.get("max_steps_multiplier", 8)
    if multiplier is None or int(multiplier) <= 0:
        return None
    return max((len(state) for state in states), default=0) * int(multiplier)


def sampler_outputs(harness, states, stats, trajectories=None):
    rows = []
    for idx, state in enumerate(states):
        row = {
            "output": harness.decode_state(state),
            "state": state,
            "stats": stats[idx],
        }
        if trajectories is not None:
            row["trajectory"] = trajectories[idx]
        rows.append(row)
    return rows


def active_batch(examples, prompts, states, indices):
    return (
        [examples[idx] for idx in indices],
        [prompts[idx] for idx in indices],
        [states[idx] for idx in indices],
    )


def locked_positions(task, example, state, harness):
    if not hasattr(task, "locked_positions"):
        return set()
    return {int(pos) for pos in task.locked_positions(example, state, harness)}


def logits(harness, prompts, states):
    if hasattr(harness, "logits_batch"):
        return harness.logits_batch(prompts, states)
    return torch.stack([harness.logits(prompt, state) for prompt, state in zip(prompts, states)], dim=0)


def terminal_on_eos(config=None):
    config = config or {}
    return bool(config.get("stop_on_eos", True))


def eos_token_ids(harness):
    cached = getattr(harness, "_vgb_eos_token_ids", None)
    if cached is not None:
        return cached
    ids = set()
    eos_id = getattr(harness, "eos_id", None)
    if eos_id is not None:
        ids.add(int(eos_id))
    tokenizer = getattr(harness, "tokenizer", None)
    if tokenizer is not None:
        tok_eos = getattr(tokenizer, "eos_token_id", None)
        if isinstance(tok_eos, (list, tuple)):
            ids.update(int(x) for x in tok_eos if x is not None)
        elif tok_eos is not None:
            ids.add(int(tok_eos))
    mask_id = getattr(harness, "mask_id", None)
    if mask_id is not None:
        ids.discard(int(mask_id))
    try:
        harness._vgb_eos_token_ids = ids
    except Exception:
        pass
    return ids


def state_has_eos(harness, state):
    ids = eos_token_ids(harness)
    return bool(ids) and any(int(token) in ids for token in state)


def terminal_state(harness, state, config=None):
    return is_leaf(state, harness.mask_id) or (terminal_on_eos(config) and state_has_eos(harness, state))


def is_eos_choice(harness, config, choice):
    token = choice.get("token")
    if token is None or not bool(config.get("stop_on_eos", True)):
        return False
    ids = eos_token_ids(harness)
    if isinstance(token, (list, tuple)):
        return any(int(item) in ids for item in token)
    return int(token) in ids


def decoded_terminal_value(task, harness, example, state):
    text = harness.decode_state(state)
    if hasattr(task, "terminal_accept"):
        return 1.0 if task.terminal_accept(example, text) else 0.0
    reward = task.reward(example, text)
    if reward is None:
        return None
    return float(reward)


def clamp_value(value):
    if value is None:
        return 1.0
    return max(float(value), 1e-8)


def clamp_state_value(task, value):
    if value is None:
        return 1.0
    value = float(value)
    if getattr(task, "HARD_ZERO_STATE_VALUE", False) and value <= 0.0:
        return 0.0
    return clamp_value(value)


def terminal_exact_value_enabled(task, config=None):
    config = config or {}
    if "terminal_value" in config:
        return str(config.get("terminal_value")) == "exact"
    return bool(getattr(task, "EXACT_LEAF_VALUE", True))


def terminal_candidate_exact_value_enabled(task, config=None):
    config = config or {}
    if "terminal_candidate_value" in config:
        return str(config.get("terminal_candidate_value")) == "exact"
    return terminal_exact_value_enabled(task, config)


def state_value_fn(task):
    if hasattr(task, "state_value"):
        return task.state_value
    return None


def is_leaf_accept(task, harness, example, state, config=None):
    if terminal_on_eos(config) and state_has_eos(harness, state):
        reward = decoded_terminal_value(task, harness, example, state)
        return reward is not None and reward >= 1.0 - 1e-12
    text = harness.decode_state(state)
    if hasattr(task, "terminal_accept"):
        return bool(task.terminal_accept(example, text))
    if not terminal_exact_value_enabled(task, config):
        return bool((config or {}).get("terminal_accept_without_exact", True))
    if hasattr(task, "reward_state"):
        return float(task.reward_state(example, state, harness)) >= 1.0 - 1e-12
    reward = task.reward(example, text)
    return reward is not None and float(reward) >= 1.0 - 1e-12


def values_for_states(task, harness, verifier, examples, states, config=None):
    values = [None for _ in states]
    need_examples = []
    need_states = []
    need_indices = []
    terminal_eos = terminal_on_eos(config)
    state_value = state_value_fn(task)
    exact_terminal_candidate = terminal_candidate_exact_value_enabled(task, config)
    for idx, state in enumerate(states):
        if terminal_eos and state_has_eos(harness, state):
            reward = decoded_terminal_value(task, harness, examples[idx], state)
            if reward is not None:
                values[idx] = clamp_value(reward)
                continue
        if state_value is not None and not is_leaf(state, harness.mask_id):
            values[idx] = clamp_state_value(task, state_value(examples[idx], state, harness))
            continue
        if is_leaf(state, harness.mask_id) and exact_terminal_candidate:
            text = harness.decode_state(state)
            if hasattr(task, "reward_state"):
                values[idx] = float(task.reward_state(examples[idx], state, harness))
                continue
            reward = task.reward(examples[idx], text)
            if reward is not None:
                values[idx] = float(reward)
                continue
        if verifier is None:
            values[idx] = 1.0
        else:
            need_examples.append(examples[idx])
            need_states.append(state)
            need_indices.append(idx)

    if need_states:
        if hasattr(verifier, "values"):
            scored = verifier.values(need_examples, need_states, harness)
        else:
            scored = [verifier.value(example, state, harness) for example, state in zip(need_examples, need_states)]
        for idx, value in zip(need_indices, scored):
            values[idx] = clamp_value(value)
    return values


def normalize_forward_mode(mode):
    mode = mode or "random"
    if mode not in {"random", "high_conf", "top_margin"}:
        raise ValueError(f"unknown forward_selection: {mode}")
    return mode


def normalize_backward_mode(mode):
    mode = mode or "random"
    if mode not in {"random", "low_conf"}:
        raise ValueError(f"unknown backward_selection: {mode}")
    return mode


def position_scores(logit_tensor, state_tensor, mask_id, mode):
    mode = normalize_forward_mode(mode)
    masked = state_tensor.eq(mask_id)
    if mode == "random":
        scores = torch.rand(masked.shape, device=state_tensor.device)
    elif mode == "top_margin":
        if logit_tensor.shape[-1] >= 10:
            probs = torch.softmax(logit_tensor.float()[..., 1:10], dim=-1)
        else:
            probs = torch.softmax(logit_tensor.float(), dim=-1)
        if probs.shape[-1] >= 2:
            top2 = torch.topk(probs, k=2, dim=-1).values
            scores = top2[..., 0] - top2[..., 1]
        else:
            scores = probs.squeeze(-1)
    else:
        probs = torch.softmax(logit_tensor.float(), dim=-1)
        scores = probs.max(dim=-1).values
    return scores.masked_fill(~masked, float("-inf"))


def choose_positions(logit_tensor, positions, limit, mode):
    if not positions:
        return []
    limit = min(int(limit), len(positions))
    mode = normalize_forward_mode(mode)
    if mode == "random":
        return random.sample(positions, limit)
    if mode == "top_margin":
        return _top_margin_positions(logit_tensor, positions, limit)
    return _top_confidence_positions(logit_tensor, positions, limit)


def sample_grouped_choices(groups):
    choices = []
    for group in groups:
        group = [item for item in group if float(item.get("weight", 0.0)) > 0.0]
        if not group:
            choices.append(None)
            continue
        weights = torch.tensor([float(item["weight"]) for item in group], dtype=torch.float)
        pick = int(torch.multinomial(weights / weights.sum(), 1).item())
        choices.append(group[pick])
    return choices


def force_complete(harness, states, configs, return_counts=False, return_trace=False):
    outs = [list(state) for state in states]
    counts = [0 for _ in outs]
    traces = [[] for _ in outs]
    if not outs:
        if return_counts and return_trace:
            return outs, counts, traces
        if return_counts:
            return outs, counts
        if return_trace:
            return outs, traces
        return outs

    max_len = max(len(state) for state in outs)
    for _ in range(max_len):
        active = []
        active_prompts = []
        active_states = []
        active_positions = []
        for idx, out in enumerate(outs):
            config = configs[idx]
            if config.get("stop_on_eos", True) and state_has_eos(harness, out):
                continue
            positions = masked_positions(out, harness.mask_id)
            if not positions:
                continue
            active.append(idx)
            active_prompts.append(config.get("prompt", ""))
            active_states.append(out)
            active_positions.append(positions)
        if not active:
            break

        logit_tensor = logits(harness, active_prompts, active_states)
        for local_idx, idx in enumerate(active):
            out = outs[idx]
            config = configs[idx]
            positions = active_positions[local_idx]
            forward_mode = normalize_forward_mode(config.get("forward_selection", "high_conf"))
            forward_selection = "random" if forward_mode == "random" else "high_conf"
            temperature = float(config.get("temperature", 1.0))
            pos = choose_positions(logit_tensor[local_idx], positions, 1, forward_selection)[0]
            if temperature <= 0:
                token = int(logit_tensor[local_idx, pos].argmax().item())
            else:
                probs = torch.softmax(logit_tensor[local_idx, pos].float() / temperature, dim=-1)
                token = int(torch.multinomial(probs, 1).item())
            out[pos] = token
            counts[idx] += 1
            if return_trace:
                traces[idx].append(list(out))
    if return_counts and return_trace:
        return outs, counts, traces
    if return_counts:
        return outs, counts
    if return_trace:
        return outs, traces
    return outs


def force_complete_value(task, harness, verifier, examples, states, prompts, config, forward_candidates):
    outs = [list(state) for state in states]
    counts = [0 for _ in outs]
    if not outs:
        return outs, counts

    max_len = max(len(state) for state in outs)
    for _ in range(max_len):
        active = []
        active_examples = []
        active_prompts = []
        active_states = []
        for idx, out in enumerate(outs):
            if config.get("stop_on_eos", True) and state_has_eos(harness, out):
                continue
            if not masked_positions(out, harness.mask_id):
                continue
            active.append(idx)
            active_examples.append(examples[idx])
            active_prompts.append(prompts[idx])
            active_states.append(out)
        if not active:
            break

        logit_tensor = logits(harness, active_prompts, active_states)
        fwd_groups, _ = forward_candidates(
            task,
            harness,
            verifier,
            active_examples,
            active_states,
            logit_tensor,
            config,
        )
        choices = _sample_forward_choices(fwd_groups)
        for local_idx, idx in enumerate(active):
            choice = choices[local_idx]
            if choice is None:
                continue
            outs[idx] = choice["state"]
            counts[idx] += 1
    return outs, counts


def force_complete_indices(task, harness, verifier, examples, prompts, states, indices, stats, config, forward_candidates):
    if not indices:
        return states

    forced_states = [states[idx] for idx in indices]
    if getattr(harness, "kind", None) == "ar":
        completed, extra_forward = force_complete_value(
            task,
            harness,
            verifier,
            [examples[idx] for idx in indices],
            forced_states,
            [prompts[idx] for idx in indices],
            config,
            forward_candidates,
        )
    else:
        completed, extra_forward = force_complete(
            harness,
            forced_states,
            [dict(config, prompt=prompts[idx]) for idx in indices],
            return_counts=True,
        )

    for idx, count in zip(indices, extra_forward):
        stats[idx]["forward"] += int(count)
        stats[idx]["nfe"] += int(count)
    for idx, state in zip(indices, completed):
        states[idx] = state
    return states


def _sample_forward_choices(groups):
    choices = sample_grouped_choices(groups)
    for idx, choice in enumerate(choices):
        if choice is None and groups[idx]:
            choices[idx] = groups[idx][0]
    return choices


def _top_confidence_positions(logit_tensor, positions, limit):
    scored = []
    for pos in positions:
        probs = torch.softmax(logit_tensor[pos].float(), dim=-1)
        scored.append((float(torch.max(probs).item()), pos))
    scored.sort(reverse=True)
    return [pos for _, pos in scored[:limit]]


def _top_margin_positions(logit_tensor, positions, limit):
    scored = []
    for pos in positions:
        row = logit_tensor[pos].float()
        probs = torch.softmax(row[1:10] if row.shape[-1] >= 10 else row, dim=-1)
        if probs.shape[-1] >= 2:
            top2 = torch.topk(probs, k=2).values
            score = float((top2[0] - top2[1]).item())
        else:
            score = float(probs.max().item())
        scored.append((score, pos))
    scored.sort(reverse=True)
    return [pos for _, pos in scored[:limit]]
