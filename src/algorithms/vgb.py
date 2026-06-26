import itertools
import math

import torch
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
    normalize_backward_mode,
    normalize_forward_mode,
    position_scores,
    sample_grouped_choices,
    state_has_eos,
    terminal_on_eos,
    terminal_state,
    values_for_states,
)


def sample_vgb(harness, task, examples, verifier=None, config=None, candidate_builder=None):
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
    max_steps = config.get("max_steps")
    if max_steps is None:
        max_len = max(len(state) for state in states) if states else 0
        multiplier = config.get("max_steps_multiplier", config.get("N", 8))
        if multiplier is None or int(multiplier) <= 0:
            max_steps = None
        else:
            max_steps = int(max_len * int(multiplier))
    else:
        max_steps = int(max_steps)
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

        active_states = [states[idx] for idx in active]
        active_prompts = [prompts[idx] for idx in active]
        logit_tensor = logits(harness, active_prompts, active_states)
        for idx in active:
            stats[idx]["nfe"] += 1
        active_examples = [examples[idx] for idx in active]
        fwd_groups, f_masses, bwd_groups, b_masses = candidate_builder(
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
        forced_states = [states[idx] for idx in forced_indices]
        if getattr(harness, "kind", None) == "ar":
            forced_examples = [examples[idx] for idx in forced_indices]
            forced_prompts = [prompts[idx] for idx in forced_indices]
            completed, extra_forward = force_complete_value(
                task,
                harness,
                verifier,
                forced_examples,
                forced_states,
                forced_prompts,
                config,
                _forward_candidates_mdm,
            )
            for idx, count in zip(forced_indices, extra_forward):
                stats[idx]["forward"] += count
                stats[idx]["nfe"] += count
        else:
            forced_configs = [dict(config, prompt=prompts[idx]) for idx in forced_indices]
            completed, extra_forward = force_complete(harness, forced_states, forced_configs, return_counts=True)
            for idx, count in zip(forced_indices, extra_forward):
                stats[idx]["forward"] += count
                stats[idx]["nfe"] += count
        for idx, state in zip(forced_indices, completed):
            states[idx] = state
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

    outputs = []
    for idx, state in enumerate(states):
        outputs.append({"output": harness.decode_state(state), "state": state, "trajectory": trajectories[idx], "stats": stats[idx]})
    return outputs

def _B(config):
    return max(1, int(config.get("B", 1) or 1))


def _log_mdm_coeff(k, n, r):
    if r <= 0 or k < 0 or k > n - r:
        return -float("inf")
    return -(math.lgamma(n - r + 1) - math.lgamma(k + 1) - math.lgamma(n - r - k + 1))


def _state_log_ref(k, n, r):
    logs = []
    if k + r <= n:
        logs.append(_log_mdm_coeff(k, n, r))
    if k - r >= 0:
        logs.append(_log_mdm_coeff(k - r, n, r))
    return max(logs) if logs else 0.0


def _relative_coeff(k, n, r, ref):
    log_coeff = _log_mdm_coeff(k, n, r)
    if not math.isfinite(log_coeff) or not math.isfinite(ref):
        return 0.0
    return math.exp(log_coeff - ref)


def _comb(n, r):
    if r < 0 or r > n:
        return 0
    return math.comb(n, r)


def _finite_top_positions(scores, limit):
    if limit <= 0:
        return []
    limit = min(int(limit), int(scores.numel()))
    values, positions = torch.topk(scores, k=limit)
    out = []
    for value, pos in zip(values.tolist(), positions.tolist()):
        if not math.isfinite(float(value)):
            continue
        out.append(int(pos))
    return out


def _editable_mask(task, harness, examples, states, state_tensor):
    locked = torch.zeros(state_tensor.shape, dtype=torch.bool, device=state_tensor.device)
    for row, (example, state) in enumerate(zip(examples, states)):
        positions = [pos for pos in locked_positions(task, example, state, harness) if 0 <= pos < state_tensor.shape[1]]
        if positions:
            locked[row, torch.tensor(positions, dtype=torch.long, device=state_tensor.device)] = True
    return ~locked


def _chunks(items, size):
    out = []
    for start in range(0, len(items), size):
        block = items[start : start + size]
        if len(block) == size:
            out.append(block)
    return out


def _sample_block_assignments(row_logits, block, K, temperature):
    if not block:
        return {}, 1
    if str(K).lower() == "all":
        vocab_size = int(row_logits.shape[-1])
        total = vocab_size ** len(block)
        if total <= 100000:
            return {tuple(tokens): 1 for tokens in itertools.product(range(vocab_size), repeat=len(block))}, total
        K = 100000
    if temperature <= 0:
        tokens = tuple(int(row_logits[pos].argmax().item()) for pos in block)
        return {tokens: 1}, 1
    K = max(1, int(K))
    samples = []
    for pos in block:
        probs = torch.softmax(row_logits[pos].float() / temperature, dim=-1)
        sampled = torch.multinomial(probs, num_samples=K, replacement=True)
        samples.append(sampled.tolist())
    counts = {}
    for idx in range(K):
        assignment = tuple(int(sample[idx]) for sample in samples)
        counts[assignment] = counts.get(assignment, 0) + 1
    return counts, K


def _position_blocks(scores, available, limit, r):
    selected = _finite_top_positions(scores, min(limit, available))
    blocks = _chunks(selected, r)
    if not blocks and len(selected) >= r:
        blocks = [selected[:r]]
    return blocks


def _geometric_lambda(config):
    return max(0.0, float(config.get("lambda", 0.0) or 0.0))


def _as_float(value, default):
    if value is None:
        return float(default)
    return float(value)


def _gamma(config):
    return _as_float(config.get("gamma"), 1.0)


def _value_lambda(config):
    return _as_float(config.get("lambda"), 0.0)


def _value_eps(config):
    return _as_float(config.get("value_eps"), 1e-4)


def _k_tokens(config):
    value = config.get("K", 8)
    if str(value).lower() == "all":
        return "all"
    return max(1, int(value))


def _top_tokens(logits, k, temperature):
    probs = torch.softmax(logits.float() / float(temperature), dim=-1)
    if k == "all":
        tokens = torch.arange(probs.shape[-1], device=probs.device)
    else:
        tokens = torch.topk(probs, k=min(int(k), probs.shape[-1]), dim=-1).indices
    return [(int(token.item()), float(torch.log(probs[token].clamp_min(1e-30)).item())) for token in tokens]


def _sample_block_assignments_logq(row_logits, block, num_candidates, temperature):
    if not block:
        return {}
    samples = []
    for pos in block:
        probs = torch.softmax(row_logits[pos].float() / float(temperature), dim=-1)
        sampled = torch.multinomial(probs, num_samples=int(num_candidates), replacement=True)
        samples.append(sampled.tolist())
    counts = {}
    for idx in range(int(num_candidates)):
        assignment = tuple(int(sample[idx]) for sample in samples)
        counts[assignment] = counts.get(assignment, 0) + 1
    return counts


def _cleanup_log_weight_groups(groups):
    for group in groups:
        for item in group:
            item.pop("_log_weight", None)
    return groups


def _rescale_log_weight_groups(fwd_groups, bwd_groups):
    f_masses = []
    b_masses = []
    for fwd, bwd in zip(fwd_groups, bwd_groups):
        logs = [float(item["_log_weight"]) for item in fwd + bwd if math.isfinite(float(item["_log_weight"]))]
        if not logs:
            for item in fwd + bwd:
                item["weight"] = 0.0
            f_masses.append(0.0)
            b_masses.append(0.0)
            continue
        offset = max(logs)
        f_mass = 0.0
        b_mass = 0.0
        for item in fwd:
            weight = math.exp(float(item["_log_weight"]) - offset) if math.isfinite(float(item["_log_weight"])) else 0.0
            item["weight"] = float(weight)
            f_mass += float(weight)
        for item in bwd:
            weight = math.exp(float(item["_log_weight"]) - offset) if math.isfinite(float(item["_log_weight"])) else 0.0
            item["weight"] = float(weight)
            b_mass += float(weight)
        f_masses.append(float(f_mass))
        b_masses.append(float(b_mass))
    return _cleanup_log_weight_groups(fwd_groups), f_masses, _cleanup_log_weight_groups(bwd_groups), b_masses


def _forward_candidates_mdm(task, harness, verifier, examples, states, logit_tensor, config, current_values=None):
    if not states:
        return [], []
    mask_id = harness.mask_id
    logit_tensor = torch.as_tensor(logit_tensor)
    state_tensor = torch.tensor(states, dtype=torch.long, device=logit_tensor.device)
    editable = _editable_mask(task, harness, examples, states, state_tensor)
    masked = state_tensor.eq(mask_id) & editable
    masked_count = masked.sum(dim=1)
    editable_count = editable.sum(dim=1)
    selected_limit = min(int(config.get("L_f", 8)), int(editable_count.max().item()) if states else 0)
    K = config.get("K", 8)
    if selected_limit <= 0 or int(masked_count.max().item()) == 0:
        return [[] for _ in states], [0.0 for _ in states]

    forward_mode = normalize_forward_mode(config.get("forward_selection", "random"))
    scores = position_scores(
        logit_tensor,
        state_tensor,
        mask_id,
        forward_mode,
    )
    scores = scores.masked_fill(~masked, float("-inf"))

    child_states = []
    child_examples = []
    metas = []
    target_r = _B(config)
    lambda_value = _geometric_lambda(config)
    for row, state in enumerate(states):
        available = int(masked_count[row].item())
        r = min(target_r, available)
        if r <= 0:
            continue
        blocks = _position_blocks(scores[row].detach(), available, selected_limit, r)
        if not blocks:
            continue

        length = int(editable_count[row].item())
        k = length - available
        ref = _state_log_ref(k, length, r)
        scale = _relative_coeff(k, length, r, ref)
        possible_blocks = max(1, _comb(available, r))
        if len(blocks) < possible_blocks:
            scale *= float(possible_blocks) / float(len(blocks))

        for block in blocks:
            counts, denominator = _sample_block_assignments(logit_tensor[row], block, K, config.get("temperature", 1.0))
            for assignment, count in counts.items():
                child = list(state)
                for pos, token in zip(block, assignment):
                    child[pos] = int(token)
                child_states.append(child)
                child_examples.append(examples[row])
                current_gate = 1.0
                if current_values is not None and lambda_value > 0.0:
                    current_gate = max(float(current_values[row]), 1e-8) ** lambda_value
                metas.append((row, list(block), tuple(assignment), float(count), float(denominator), scale, current_gate))

    if not child_states:
        return [[] for _ in states], [0.0 for _ in states]

    child_values = values_for_states(
        task,
        harness,
        verifier,
        child_examples,
        child_states,
        config=config,
    )

    groups = [[] for _ in states]
    for idx, child in enumerate(child_states):
        row, block, assignment, count, denominator, scale, current_gate = metas[idx]
        groups[row].append(
            {
                "kind": "forward",
                "state": child,
                "weight": float((scale / denominator) * count * current_gate * float(child_values[idx])),
                "value": float(child_values[idx]),
                "state_value": child_values[idx],
                "pos": block,
                "token": assignment,
            }
        )
    return groups, [sum(item["weight"] for item in group) for group in groups]


def _backward_scores(task, harness, examples, states, logit_tensor, config):
    mask_id = harness.mask_id
    device = torch.as_tensor(logit_tensor).device if logit_tensor is not None else torch.device("cpu")
    state_tensor = torch.tensor(states, dtype=torch.long, device=device)
    revealed = state_tensor.ne(mask_id)
    for row, (example, state) in enumerate(zip(examples, states)):
        locked = [pos for pos in locked_positions(task, example, state, harness) if 0 <= pos < state_tensor.shape[1]]
        if locked:
            revealed[row, torch.tensor(locked, dtype=torch.long, device=device)] = False
    choice = normalize_backward_mode(config.get("backward_selection", "random"))
    if choice == "low_conf" and logit_tensor is not None:
        logit_tensor = torch.as_tensor(logit_tensor)
        if logit_tensor.device != device:
            logit_tensor = logit_tensor.to(device)
        probs = torch.softmax(logit_tensor.float(), dim=-1)
        token_ids = state_tensor.unsqueeze(-1).clamp(min=0)
        token_probs = probs.gather(-1, token_ids).squeeze(-1)
        scores = (-token_probs).masked_fill(~revealed, float("-inf"))
    else:
        scores = torch.rand(revealed.shape, device=device).masked_fill(~revealed, float("-inf"))
    return scores, revealed


def _backward_candidates_mdm(task, harness, verifier, examples, states, current_values, config, logit_tensor=None):
    if not states:
        return [], []
    mask_id = harness.mask_id
    scores, revealed = _backward_scores(task, harness, examples, states, logit_tensor, config)
    state_tensor = torch.tensor(states, dtype=torch.long, device=scores.device)
    editable = _editable_mask(task, harness, examples, states, state_tensor)
    masked_editable = state_tensor.eq(mask_id) & editable
    revealed_count = revealed.sum(dim=1)
    editable_count = revealed_count + masked_editable.sum(dim=1)
    selected_limit = config.get("L_b") or config.get("L_f", 8)
    selected_limit = min(int(selected_limit), int(editable_count.max().item()) if states else 0)
    if selected_limit <= 0 or int(revealed_count.max().item()) == 0:
        return [[] for _ in states], [0.0 for _ in states]

    parent_states = []
    parent_examples = []
    metas = []
    target_r = _B(config)
    lambda_value = _geometric_lambda(config)
    for row, state in enumerate(states):
        available = int(revealed_count[row].item())
        r = min(target_r, available)
        if r <= 0:
            continue
        blocks = _position_blocks(scores[row].detach(), available, selected_limit, r)
        if not blocks:
            continue

        length = int(editable_count[row].item())
        ref = _state_log_ref(available, length, r)
        scale = _relative_coeff(available - r, length, r, ref)
        possible_blocks = max(1, _comb(available, r))
        if len(blocks) < possible_blocks:
            scale *= float(possible_blocks) / float(len(blocks))

        for block in blocks:
            parent = list(state)
            for pos in block:
                parent[pos] = mask_id
            parent_states.append(parent)
            parent_examples.append(examples[row])
            metas.append((row, list(block), scale))

    if not parent_states:
        return [[] for _ in states], [0.0 for _ in states]

    if lambda_value > 0.0:
        parent_values = values_for_states(task, harness, verifier, parent_examples, parent_states, config=config)
    else:
        parent_values = [1.0 for _ in parent_states]

    groups = [[] for _ in states]
    for idx, parent in enumerate(parent_states):
        row, block, scale = metas[idx]
        parent_gate = float(parent_values[idx]) ** lambda_value
        parent_value = None if lambda_value == 0.0 else float(parent_values[idx])
        item = {
            "kind": "backward",
            "state": parent,
            "weight": float(scale * float(current_values[row]) * parent_gate),
            "pos": block,
        }
        if parent_value is not None:
            item["value"] = parent_value
            item["state_value"] = parent_value
        groups[row].append(item)
    return groups, [sum(item["weight"] for item in group) for group in groups]


def _candidates_mdm(task, harness, verifier, examples, states, logit_tensor, current_values, config):
    if current_values is None:
        current_values = values_for_states(
            task,
            harness,
            verifier,
            examples,
            states,
            config=config,
        )

    fwd_groups, f_masses = _forward_candidates_mdm(
        task,
        harness,
        verifier,
        examples,
        states,
        logit_tensor,
        config,
        current_values=current_values,
    )

    bwd_groups, b_masses = _backward_candidates_mdm(
        task,
        harness,
        verifier,
        examples,
        states,
        current_values,
        config,
        logit_tensor=logit_tensor,
    )
    return fwd_groups, f_masses, bwd_groups, b_masses


def _forward_candidates_mdm_state_value(task, harness, verifier, examples, states, logit_tensor, config, current_values=None):
    if not states:
        return [], []
    mask_id = int(harness.mask_id)
    logit_tensor = torch.as_tensor(logit_tensor)
    device = logit_tensor.device
    state_tensor = torch.tensor(states, dtype=torch.long, device=device)
    editable = _editable_mask(task, harness, examples, states, state_tensor)
    masked = state_tensor.eq(mask_id) & editable
    masked_count = masked.sum(dim=1)
    revealed_count = (state_tensor.ne(mask_id) & editable).sum(dim=1)
    editable_count = editable.sum(dim=1)
    selected_limit = min(int(config.get("L_f", 8)), state_tensor.shape[1])
    if selected_limit <= 0 or int(masked_count.max().item()) == 0:
        return [[] for _ in states], [0.0 for _ in states]

    forward_mode = normalize_forward_mode(config.get("forward_selection", "random"))
    scores = position_scores(logit_tensor, state_tensor, mask_id, forward_mode).masked_fill(~editable, float("-inf"))
    target_r = _B(config)
    gamma = _gamma(config)
    lam = _value_lambda(config)
    eps = _value_eps(config)
    temperature = float(config.get("temperature", 1.0))
    k_tokens = _k_tokens(config)

    child_states = []
    child_examples = []
    metas = []
    for row, state in enumerate(states):
        available = int(masked_count[row].item())
        r = min(target_r, available)
        if r <= 0:
            continue
        blocks = _position_blocks(scores[row].detach(), available, selected_limit, r)
        if not blocks:
            continue
        k = int(revealed_count[row].item())
        n = max(1, int(editable_count[row].item()))
        ref = _state_log_ref(k, n, r)
        scale = _relative_coeff(k, n, r, ref)
        total_blocks = max(1, _comb(available, r))
        scale *= float(total_blocks) / float(len(blocks))
        if scale <= 0.0:
            continue
        log_scale = math.log(float(scale))
        log_current = 0.0
        if current_values is not None and lam > 0.0:
            log_current = math.log(max(float(current_values[row]), eps))

        for block in blocks:
            if r == 1:
                pos = int(block[0])
                assignments = [((token,), logq) for token, logq in _top_tokens(logit_tensor[row, pos], k_tokens, temperature)]
            else:
                count = int(config.get("mdm_num_candidates", config.get("K", 32) if str(config.get("K", "")).lower() != "all" else 32))
                counts = _sample_block_assignments_logq(logit_tensor[row], block, count, temperature)
                assignments = [(assignment, math.log(max(1, sample_count))) for assignment, sample_count in counts.items()]
            for assignment, logq in assignments:
                child = list(state)
                for pos, token in zip(block, assignment):
                    child[int(pos)] = int(token)
                child_states.append(child)
                child_examples.append(examples[row])
                metas.append((row, list(block), tuple(assignment), float(logq), log_scale, log_current))

    if not child_states:
        return [[] for _ in states], [0.0 for _ in states]

    child_values = values_for_states(task, harness, verifier, child_examples, child_states, config=config)
    groups = [[] for _ in states]
    for idx, child in enumerate(child_states):
        row, block, assignment, logq, log_scale, log_current = metas[idx]
        value = max(float(child_values[idx]), eps)
        log_weight = log_scale + float(logq) + gamma * math.log(value) + gamma * lam * log_current
        groups[row].append(
            {
                "kind": "forward",
                "state": child,
                "weight": 0.0,
                "_log_weight": float(log_weight),
                "value": float(value),
                "state_value": float(value),
                "pos": block,
                "token": assignment,
            }
        )
    return groups, [0.0 for _ in groups]


def _backward_candidates_mdm_state_value(task, harness, verifier, examples, states, current_values, config, logit_tensor=None):
    if not states:
        return [], []
    scores, revealed = _backward_scores(task, harness, examples, states, logit_tensor, config)
    state_tensor = torch.tensor(states, dtype=torch.long, device=scores.device)
    editable = _editable_mask(task, harness, examples, states, state_tensor)
    revealed = revealed & editable
    revealed_count = revealed.sum(dim=1)
    editable_count = editable.sum(dim=1)
    selected_limit = config.get("L_b") or config.get("L_f", 8)
    selected_limit = min(int(selected_limit), revealed.shape[1])
    if selected_limit <= 0 or int(revealed_count.max().item()) == 0:
        return [[] for _ in states], [0.0 for _ in states]

    parent_states = []
    parent_examples = []
    metas = []
    target_r = _B(config)
    gamma = _gamma(config)
    lam = _value_lambda(config)
    eps = _value_eps(config)
    up_prob = float(config.get("up_prob", 1.0))
    for row, state in enumerate(states):
        available = int(revealed_count[row].item())
        r = min(target_r, available)
        if r <= 0:
            continue
        blocks = _position_blocks(scores[row].detach(), available, selected_limit, r)
        if not blocks:
            continue
        k = int(revealed_count[row].item())
        n = max(1, int(editable_count[row].item()))
        ref = _state_log_ref(k, n, r)
        candidate_factor = float(config.get("mdm_num_candidates", 32)) if r > 1 else 1.0
        scale = up_prob * candidate_factor * _relative_coeff(k - r, n, r, ref)
        total_blocks = max(1, _comb(k, r))
        scale *= float(total_blocks) / float(len(blocks))
        if scale <= 0.0:
            continue
        log_scale = math.log(float(scale))
        log_current = math.log(max(float(current_values[row]), eps))
        for block in blocks:
            parent = list(state)
            for pos in block:
                parent[int(pos)] = int(harness.mask_id)
            parent_states.append(parent)
            parent_examples.append(examples[row])
            metas.append((row, list(block), log_scale, log_current))

    if not parent_states:
        return [[] for _ in states], [0.0 for _ in states]

    parent_values = values_for_states(task, harness, verifier, parent_examples, parent_states, config=config) if lam > 0.0 else [1.0 for _ in parent_states]
    groups = [[] for _ in states]
    for idx, parent in enumerate(parent_states):
        row, block, log_scale, log_current = metas[idx]
        parent_value = max(float(parent_values[idx]), eps)
        log_weight = log_scale + gamma * log_current + gamma * lam * math.log(parent_value)
        groups[row].append(
            {
                "kind": "backward",
                "state": parent,
                "weight": 0.0,
                "_log_weight": float(log_weight),
                "value": float(parent_value),
                "state_value": float(parent_value),
                "pos": block,
            }
        )
    return groups, [0.0 for _ in groups]


def _candidates_mdm_state_value(task, harness, verifier, examples, states, logit_tensor, current_values, config):
    if current_values is None:
        current_values = values_for_states(task, harness, verifier, examples, states, config=config)
    fwd_groups, _ = _forward_candidates_mdm_state_value(
        task, harness, verifier, examples, states, logit_tensor, config, current_values=current_values
    )
    bwd_groups, _ = _backward_candidates_mdm_state_value(
        task, harness, verifier, examples, states, current_values, config, logit_tensor=logit_tensor
    )
    return _rescale_log_weight_groups(fwd_groups, bwd_groups)


def _empty_stats():
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


def _leaf_backtrack(task, harness, example, state, mask_id, config):
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


def _original_like_max_steps(config, states):
    if config.get("max_steps") is not None:
        return int(config["max_steps"])
    multiplier = config.get("max_steps_multiplier", config.get("N", 8))
    if multiplier is None or int(multiplier) <= 0:
        return None
    return max((len(state) for state in states), default=0) * int(multiplier)


def _handle_original_like_leaves(task, harness, examples, states, finished, stats, active, config, max_steps):
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

        parent, moved = _leaf_backtrack(task, harness, examples[idx], state, harness.mask_id, config)
        if moved:
            states[idx] = parent
            stats[idx]["leaf_backtrack"] += 1
            stats[idx]["nfe"] += 1
            continue
        finished[idx] = True
    return nonleaf


def sample_vgb_original_like(harness, task, examples, verifier=None, config=None, candidate_builder=None):
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
    finished = [False for _ in examples]
    trajectories = [[] for _ in examples]
    stats = [_empty_stats() for _ in examples]
    max_steps = _original_like_max_steps(config, states)

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

        active = _handle_original_like_leaves(task, harness, examples, states, finished, stats, active, config, max_steps)
        if not active:
            continue

        for idx in active:
            trajectories[idx].append(list(states[idx]))
        active_states = [states[idx] for idx in active]
        active_examples = [examples[idx] for idx in active]
        active_prompts = [prompts[idx] for idx in active]
        logit_tensor = logits(harness, active_prompts, active_states)
        for idx in active:
            stats[idx]["nfe"] += 1

        current_values = values_for_states(task, harness, verifier, active_examples, active_states, config=config)
        fwd_groups, _, bwd_groups, _ = candidate_builder(
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
                    parent, moved = _leaf_backtrack(task, harness, examples[idx], states[idx], harness.mask_id, config)
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
    return [
        {
            "output": harness.decode_state(state),
            "state": state,
            "trajectory": trajectories[idx],
            "stats": stats[idx],
        }
        for idx, state in enumerate(states)
    ]


def _state_value_max_steps(config, task, examples, states, harness):
    value = config.get("max_steps", config.get("budget_steps"))
    if value is not None:
        return [int(value) for _ in states]
    multiplier = config.get("max_steps_multiplier", config.get("multiplier", config.get("N", 8)))
    if multiplier is None or int(multiplier) <= 0:
        return None
    scope = str(config.get("max_steps_scope", config.get("budget_scope", "length")))
    budgets = []
    for example, state in zip(examples, states):
        if scope in {"mutable", "editable"}:
            locked = locked_positions(task, example, state, harness)
            unit = sum(1 for pos in range(len(state)) if pos not in locked)
        else:
            unit = len(state)
        budgets.append(max(1, int(unit)) * int(multiplier))
    return budgets


def _state_value_stats():
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


def _positive(value):
    return max(float(value), 0.0)


def _switch_mass(opposite_mass, same_mass, chi):
    opposite_mass = _positive(opposite_mass)
    same_mass = _positive(same_mass)
    chi = max(0.0, min(1.0, float(chi if chi is not None else 1.0)))
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


def _note_terminal(stats, harness, state, stop_on_eos):
    stats["leaf_reached"] = True
    if is_leaf(state, harness.mask_id):
        stats["full_leaf_reached"] = True
    if stop_on_eos and state_has_eos(harness, state):
        stats["eos_reached"] = True


def _terminal_value(task, harness, example, state, config):
    eps = _value_eps(config)
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
            _note_terminal(stats[idx], harness, state, stop_on_eos)
            if is_leaf_accept(task, harness, examples[idx], state, config):
                stats[idx]["accepted"] = True
                if stop_on_eos and state_has_eos(harness, state):
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
            _forward_candidates_mdm_state_value,
        )
    else:
        completed, extra_forward = force_complete(
            harness,
            forced_states,
            [dict(config, prompt=prompts[idx]) for idx in forced],
            return_counts=True,
        )
    for idx, count in zip(forced, extra_forward):
        stats[idx]["forward"] += int(count)
        stats[idx]["nfe"] += int(count)
    for idx, state in zip(forced, completed):
        states[idx] = state
        if terminal_state(harness, state, config):
            _note_terminal(stats[idx], harness, state, stop_on_eos)
            if is_leaf_accept(task, harness, examples[idx], state, config):
                stats[idx]["accepted"] = True
    return states


def sample_vgb_state_value(harness, task, examples, verifier=None, config=None, candidate_builder=None):
    del candidate_builder
    return _sample_state_value_mdm(harness, task, examples, verifier=verifier, config=config, momentum=False)


def _sample_state_value_mdm(harness, task, examples, verifier=None, config=None, momentum=False):
    config = dict(config or {})
    candidate_builder = _candidates_mdm_state_value
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
    terminal_checked = [False for _ in examples]
    best_terminal_states = [None for _ in examples]
    best_terminal_values = [-float("inf") for _ in examples]
    trajectories = [[] for _ in examples]
    stats = [_state_value_stats() for _ in examples]
    stop_on_eos = terminal_on_eos(config)
    max_steps = _state_value_max_steps(config, task, examples, states, harness)

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
                _note_terminal(stats[idx], harness, state, stop_on_eos)
                stats[idx]["terminal_checks"] += 1
                value = _terminal_value(task, harness, examples[idx], state, config)
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
        active_states = [states[idx] for idx in ready]
        active_prompts = [prompts[idx] for idx in ready]
        active_examples = [examples[idx] for idx in ready]
        logit_tensor = logits(harness, active_prompts, active_states)
        for idx in ready:
            stats[idx]["nfe"] += 1

        current_values = values_for_states(task, harness, verifier, active_examples, active_states, config=config)
        for local_idx, idx in enumerate(ready):
            if state_values[idx] is not None:
                current_values[local_idx] = max(float(state_values[idx]), _value_eps(config))

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
        grouped = []
        for local_idx, idx in enumerate(ready):
            _add_mass_stats(stats[idx], f_masses[local_idx], b_masses[local_idx])
            if momentum:
                grouped.append(
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
            state_values[idx] = choice.get("state_value", choice.get("value"))
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
            _note_terminal(stats[idx], harness, state, stop_on_eos)
            stats[idx]["terminal_checks"] += 1
            value = _terminal_value(task, harness, examples[idx], state, config)
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
            state_values[idx] = max(float(best_terminal_values[idx]), _value_eps(config))

    if config.get("force_complete_at_end", True):
        states = _force_complete_unfinished_state_value(task, harness, verifier, examples, prompts, states, stats, config)

    return [
        {
            "output": harness.decode_state(state),
            "state": state,
            "trajectory": trajectories[idx],
            "stats": stats[idx],
        }
        for idx, state in enumerate(states)
    ]
