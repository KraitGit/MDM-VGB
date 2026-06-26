import itertools
import math

import torch

from .algorithm_utils import (
    locked_positions,
    normalize_backward_mode,
    normalize_forward_mode,
    position_scores,
    values_for_states,
)


def _block_size(config):
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
    if not hasattr(task, "locked_positions"):
        return torch.ones(state_tensor.shape, dtype=torch.bool, device=state_tensor.device)
    locked = torch.zeros(state_tensor.shape, dtype=torch.bool, device=state_tensor.device)
    for row, (example, state) in enumerate(zip(examples, states)):
        positions = [pos for pos in locked_positions(task, example, state, harness) if 0 <= pos < state_tensor.shape[1]]
        if positions:
            locked[row, torch.tensor(positions, dtype=torch.long, device=state_tensor.device)] = True
    return ~locked


def _candidate_context(task, harness, examples, states, device):
    state_tensor = torch.tensor(states, dtype=torch.long, device=device)
    return state_tensor, _editable_mask(task, harness, examples, states, state_tensor)


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


def value_epsilon(config):
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
    counts, _ = _sample_block_assignments(row_logits, block, num_candidates, temperature)
    return [(assignment, math.log(max(1, count))) for assignment, count in counts.items()]


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


def build_mdm_forward_candidates(task, harness, verifier, examples, states, logit_tensor, config, current_values=None, state_tensor=None, editable=None):
    if not states:
        return [], []
    mask_id = harness.mask_id
    logit_tensor = torch.as_tensor(logit_tensor)
    if state_tensor is None or editable is None:
        state_tensor, editable = _candidate_context(task, harness, examples, states, logit_tensor.device)
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
    target_r = _block_size(config)
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
                "state_value": float(child_values[idx]),
                "pos": block,
                "token": assignment,
            }
        )
    return groups, [sum(item["weight"] for item in group) for group in groups]


def _backward_scores(task, harness, examples, states, logit_tensor, config, state_tensor=None, editable=None):
    mask_id = harness.mask_id
    device = torch.as_tensor(logit_tensor).device if logit_tensor is not None else torch.device("cpu")
    if state_tensor is None:
        state_tensor = torch.tensor(states, dtype=torch.long, device=device)
    revealed = state_tensor.ne(mask_id)
    if editable is not None:
        revealed = revealed & editable
    elif hasattr(task, "locked_positions"):
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


def _build_mdm_backward_candidates(task, harness, verifier, examples, states, current_values, config, logit_tensor=None, state_tensor=None, editable=None):
    if not states:
        return [], []
    mask_id = harness.mask_id
    scores, revealed = _backward_scores(task, harness, examples, states, logit_tensor, config, state_tensor=state_tensor, editable=editable)
    if state_tensor is None or editable is None:
        state_tensor, editable = _candidate_context(task, harness, examples, states, scores.device)
    masked_editable = state_tensor.eq(mask_id) & editable
    revealed_count = revealed.sum(dim=1)
    editable_count = revealed_count + masked_editable.sum(dim=1)
    selected_limit = config.get("L_b", 8)
    selected_limit = min(int(selected_limit), int(editable_count.max().item()) if states else 0)
    if selected_limit <= 0 or int(revealed_count.max().item()) == 0:
        return [[] for _ in states], [0.0 for _ in states]

    parent_states = []
    parent_examples = []
    metas = []
    target_r = _block_size(config)
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
            item["state_value"] = parent_value
        groups[row].append(item)
    return groups, [sum(item["weight"] for item in group) for group in groups]


def build_mdm_candidates(task, harness, verifier, examples, states, logit_tensor, current_values, config):
    if current_values is None:
        current_values = values_for_states(
            task,
            harness,
            verifier,
            examples,
            states,
            config=config,
        )
    logit_tensor = torch.as_tensor(logit_tensor)
    state_tensor, editable = _candidate_context(task, harness, examples, states, logit_tensor.device)

    fwd_groups, f_masses = build_mdm_forward_candidates(
        task,
        harness,
        verifier,
        examples,
        states,
        logit_tensor,
        config,
        current_values=current_values,
        state_tensor=state_tensor,
        editable=editable,
    )

    bwd_groups, b_masses = _build_mdm_backward_candidates(
        task,
        harness,
        verifier,
        examples,
        states,
        current_values,
        config,
        logit_tensor=logit_tensor,
        state_tensor=state_tensor,
        editable=editable,
    )
    return fwd_groups, f_masses, bwd_groups, b_masses


def build_state_value_forward_candidates(task, harness, verifier, examples, states, logit_tensor, config, current_values=None, state_tensor=None, editable=None):
    if not states:
        return [], []
    mask_id = int(harness.mask_id)
    logit_tensor = torch.as_tensor(logit_tensor)
    device = logit_tensor.device
    if state_tensor is None or editable is None:
        state_tensor, editable = _candidate_context(task, harness, examples, states, device)
    masked = state_tensor.eq(mask_id) & editable
    masked_count = masked.sum(dim=1)
    revealed_count = (state_tensor.ne(mask_id) & editable).sum(dim=1)
    editable_count = editable.sum(dim=1)
    selected_limit = min(int(config.get("L_f", 8)), state_tensor.shape[1])
    if selected_limit <= 0 or int(masked_count.max().item()) == 0:
        return [[] for _ in states], [0.0 for _ in states]

    forward_mode = normalize_forward_mode(config.get("forward_selection", "random"))
    scores = position_scores(logit_tensor, state_tensor, mask_id, forward_mode).masked_fill(~editable, float("-inf"))
    target_r = _block_size(config)
    gamma = _gamma(config)
    lam = _value_lambda(config)
    eps = value_epsilon(config)
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
                count = _k_tokens(config)
                assignments = _sample_block_assignments_logq(logit_tensor[row], block, count, temperature)
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
                "state_value": float(value),
                "pos": block,
                "token": assignment,
            }
        )
    return groups, [0.0 for _ in groups]


def _build_state_value_backward_candidates(task, harness, verifier, examples, states, current_values, config, logit_tensor=None, state_tensor=None, editable=None):
    if not states:
        return [], []
    scores, revealed = _backward_scores(task, harness, examples, states, logit_tensor, config, state_tensor=state_tensor, editable=editable)
    if state_tensor is None or editable is None:
        state_tensor, editable = _candidate_context(task, harness, examples, states, scores.device)
    revealed = revealed & editable
    revealed_count = revealed.sum(dim=1)
    editable_count = editable.sum(dim=1)
    selected_limit = config.get("L_b", 8)
    selected_limit = min(int(selected_limit), revealed.shape[1])
    if selected_limit <= 0 or int(revealed_count.max().item()) == 0:
        return [[] for _ in states], [0.0 for _ in states]

    parent_states = []
    parent_examples = []
    metas = []
    target_r = _block_size(config)
    gamma = _gamma(config)
    lam = _value_lambda(config)
    eps = value_epsilon(config)
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
        candidate_factor = float(_k_tokens(config)) if r > 1 else 1.0
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
                "state_value": float(parent_value),
                "pos": block,
            }
        )
    return groups, [0.0 for _ in groups]


def build_state_value_candidates(task, harness, verifier, examples, states, logit_tensor, current_values, config):
    if current_values is None:
        current_values = values_for_states(task, harness, verifier, examples, states, config=config)
    logit_tensor = torch.as_tensor(logit_tensor)
    state_tensor, editable = _candidate_context(task, harness, examples, states, logit_tensor.device)
    fwd_groups, _ = build_state_value_forward_candidates(
        task, harness, verifier, examples, states, logit_tensor, config, current_values=current_values, state_tensor=state_tensor, editable=editable
    )
    bwd_groups, _ = _build_state_value_backward_candidates(
        task, harness, verifier, examples, states, current_values, config, logit_tensor=logit_tensor, state_tensor=state_tensor, editable=editable
    )
    return _rescale_log_weight_groups(fwd_groups, bwd_groups)
