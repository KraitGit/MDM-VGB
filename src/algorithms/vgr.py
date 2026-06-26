import torch
from tqdm.auto import tqdm

from .algorithm_utils import initial_state, logits, values_for_states
from .vgb import _forward_candidates_mdm, sample_vgb


def _candidates_vgr(task, harness, verifier, examples, states, logit_tensor, current_values, config):
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
    return fwd_groups, f_masses, [[] for _ in fwd_groups], [0.0 for _ in fwd_groups]


def sample_forward_vgr(harness, task, examples, verifier=None, config=None):
    config = dict(config or {})
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
    mask_id = int(harness.mask_id)
    L_f = max(1, int(config.get("L_f", 8)))
    K_config = config.get("K", 8)
    K = "all" if str(K_config).lower() == "all" else max(1, int(K_config))
    temperature = float(config.get("temperature", 1.0))
    max_steps = max((len(state) for state in states), default=0)
    stats = [{"forward": 0, "backward": 0, "switch_up": 0, "switch_down": 0, "forced": 0, "nfe": 0} for _ in examples]

    progress = tqdm(range(max_steps), desc=str(config.get("progress_desc", "VGR inference"))) if config.get("progress") else range(max_steps)
    for _ in progress:
        active = [idx for idx, state in enumerate(states) if any(int(token) == mask_id for token in state)]
        if not active:
            break
        active_states = [states[idx] for idx in active]
        active_prompts = [prompts[idx] for idx in active]
        child_states = []
        child_examples = []
        owners = []
        logit_tensor = logits(harness, active_prompts, active_states)
        for row, idx in enumerate(active):
            masked = [pos for pos, token in enumerate(states[idx]) if int(token) == mask_id]
            if not masked:
                continue
            order = torch.randperm(len(masked), device=logit_tensor.device)[: min(L_f, len(masked))].detach().cpu().tolist()
            for item in order:
                pos = masked[int(item)]
                if temperature <= 0:
                    repeat = int(logit_tensor.shape[-1]) if K == "all" else int(K)
                    tokens = [int(logit_tensor[row, pos].argmax().item())] * repeat
                else:
                    probs = torch.softmax(logit_tensor[row, pos].float() / temperature, dim=-1)
                    if K == "all":
                        tokens = list(range(int(probs.shape[-1])))
                    else:
                        tokens = torch.multinomial(probs, num_samples=int(K), replacement=True).detach().cpu().tolist()
                for token in tokens:
                    child = list(states[idx])
                    child[pos] = int(token)
                    child_states.append(child)
                    child_examples.append(examples[idx])
                    owners.append(idx)

        if not child_states:
            break
        values = values_for_states(task, harness, verifier, child_examples, child_states, config=config)
        grouped = {idx: [] for idx in active}
        for child, owner, value in zip(child_states, owners, values):
            grouped[owner].append((child, max(float(value), 0.0)))

        for idx in active:
            group = grouped.get(idx, [])
            if not group:
                continue
            weights = torch.tensor([weight for _, weight in group], dtype=torch.float)
            if float(weights.sum().item()) <= 0.0 or not torch.isfinite(weights).all():
                choice = int(torch.randint(len(group), (1,)).item())
            else:
                choice = int(torch.multinomial(weights / weights.sum(), 1).item())
            states[idx] = group[choice][0]
            stats[idx]["forward"] += 1
            stats[idx]["nfe"] += 1

    return [{"output": harness.decode_state(state), "state": state, "stats": stats[idx]} for idx, state in enumerate(states)]


def sample_vgr(harness, task, examples, verifier=None, config=None):
    cfg = dict(config or {})
    cfg["L_b"] = 0
    return sample_vgb(harness, task, examples, verifier=verifier, config=cfg, candidate_builder=_candidates_vgr)
