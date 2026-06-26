ALGORITHMS = {"Base", "BoN", "VGR", "VGB", "VGB-Momentum"}
VALUE_GUIDED_ALGORITHMS = {"BoN", "VGR", "VGB", "VGB-Momentum"}
ALGORITHM_CONFIG_KEYS = {"BoN", "value_guidance"}
VALUE_GUIDANCE_SECTION_KEYS = {"backward", "momentum", "state_value", "terminal", "budget"}
VALUE_GUIDANCE_BASE_KEYS = {
    "N",
    "L_f",
    "K",
    "filter_forward_tokens",
    "forward_selection",
    "length",
    "max_steps",
    "max_steps_multiplier",
    "stop_on_eos",
    "temperature",
}
VGR_CONFIG_KEYS = {
    "L_f",
    "K",
    "filter_forward_tokens",
    "forward_selection",
    "length",
    "temperature",
}
BACKWARD_CONFIG_KEYS = {"L_b", "B", "lambda", "backward_selection"}
MOMENTUM_CONFIG_KEYS = {"chi"}
STATE_VALUE_CONFIG_KEYS = {"gamma", "up_prob", "value_eps"}
TERMINAL_CONFIG_KEYS = {
    "accept_without_exact",
    "candidate_value",
    "force_complete",
    "keep_best",
    "reject",
    "value",
}
BUDGET_CONFIG_KEYS = {"unit"}
MISPLACED_ALGORITHM_ROOT_KEYS = (
    ALGORITHM_CONFIG_KEYS
    | VALUE_GUIDANCE_BASE_KEYS
    | BACKWARD_CONFIG_KEYS
    | MOMENTUM_CONFIG_KEYS
    | STATE_VALUE_CONFIG_KEYS
)
VALUE_GUIDANCE_SECTION_ALLOWED_KEYS = {
    "backward": BACKWARD_CONFIG_KEYS,
    "momentum": MOMENTUM_CONFIG_KEYS,
    "state_value": STATE_VALUE_CONFIG_KEYS,
    "terminal": TERMINAL_CONFIG_KEYS,
    "budget": BUDGET_CONFIG_KEYS,
}


def dict_section(config, key):
    value = config.get(key)
    return dict(value) if isinstance(value, dict) else {}


def _section(config, key):
    value = config.setdefault(key, {})
    if not isinstance(value, dict):
        raise TypeError(f"{key} must be a mapping")
    return value


def value_guidance_section(config):
    return _section(_section(config, "algorithms"), "value_guidance")


def algorithm_slug(algorithm):
    return str(algorithm).lower().replace("-", "_")


def _algorithms_config(config):
    misplaced = sorted(key for key in MISPLACED_ALGORITHM_ROOT_KEYS if key in config)
    if misplaced:
        raise ValueError(f"algorithm config keys must live under algorithms: {misplaced}")
    algorithms = dict_section(config, "algorithms")
    unknown = sorted(key for key in algorithms if key not in ALGORITHM_CONFIG_KEYS)
    if unknown:
        raise ValueError(f"unknown algorithms sections: {unknown}")
    return algorithms


def _value_guidance_root(config):
    return dict_section(_algorithms_config(config), "value_guidance")


def _value_guidance_base(config):
    root = _value_guidance_root(config)
    misplaced = sorted((BACKWARD_CONFIG_KEYS | MOMENTUM_CONFIG_KEYS | STATE_VALUE_CONFIG_KEYS) & set(root))
    if misplaced:
        raise ValueError(f"move algorithms.value_guidance keys into their section: {misplaced}")
    unknown = sorted(key for key in root if key not in VALUE_GUIDANCE_BASE_KEYS and key not in VALUE_GUIDANCE_SECTION_KEYS)
    if unknown:
        raise ValueError(f"unknown algorithms.value_guidance keys: {unknown}")
    for name, allowed in VALUE_GUIDANCE_SECTION_ALLOWED_KEYS.items():
        section = dict_section(root, name)
        extra = sorted(key for key in section if key not in allowed)
        if extra:
            raise ValueError(f"algorithms.value_guidance.{name} only supports {sorted(allowed)}; got {extra}")
    return {key: value for key, value in root.items() if key not in VALUE_GUIDANCE_SECTION_KEYS}


def _validate_bon_config(cfg):
    extra = sorted(key for key in cfg if key != "N")
    if extra:
        raise ValueError(f"algorithms.BoN only supports N; got {extra}")


def _merge_section(cfg, root, name):
    section = dict_section(root, name)
    cfg.update(section)


def _normalize_terminal_policy(cfg):
    terminal = cfg.pop("terminal", None)
    if terminal is None:
        return
    if not isinstance(terminal, dict):
        raise TypeError("algorithms.value_guidance.terminal must be a mapping")

    value = terminal.get("value")
    if value is not None:
        value = str(value)
        if value not in {"exact", "learned"}:
            raise ValueError(f"unknown terminal.value: {value}")
        cfg["terminal_value"] = value

    candidate_value = terminal.get("candidate_value")
    if candidate_value is not None:
        candidate_value = str(candidate_value)
        if candidate_value not in {"exact", "learned"}:
            raise ValueError(f"unknown terminal.candidate_value: {candidate_value}")
        cfg["terminal_candidate_value"] = candidate_value

    if "accept_without_exact" in terminal:
        cfg["terminal_accept_without_exact"] = bool(terminal["accept_without_exact"])
    if "force_complete" in terminal:
        cfg["terminal_force_complete"] = bool(terminal["force_complete"])
    if "reject" in terminal:
        cfg["terminal_reject"] = str(terminal["reject"])
    if "keep_best" in terminal:
        cfg["terminal_keep_best"] = bool(terminal["keep_best"])


def _normalize_budget_policy(cfg):
    budget = cfg.pop("budget", None)
    if budget is None:
        return
    if not isinstance(budget, dict):
        raise TypeError("algorithms.value_guidance.budget must be a mapping")
    unit = budget.get("unit")
    if unit is None:
        return
    unit = str(unit)
    if unit == "editable_tokens":
        cfg["budget_unit"] = "editable_tokens"
    elif unit == "length":
        cfg["budget_unit"] = "length"
    else:
        raise ValueError(f"unknown algorithms.value_guidance budget unit: {unit}")


def _strip_vgr_unused_config(cfg):
    for key in tuple(cfg):
        if key not in VGR_CONFIG_KEYS:
            cfg.pop(key, None)


def value_guidance_config(config, algorithm):
    if algorithm not in VALUE_GUIDED_ALGORITHMS:
        raise ValueError(f"unknown value-guided algorithm: {algorithm}")
    if algorithm == "BoN":
        cfg = dict_section(_algorithms_config(config), "BoN")
        _validate_bon_config(cfg)
        cfg["algorithm"] = algorithm
        cfg.setdefault("N", 1)
        return cfg

    cfg = _value_guidance_base(config)
    root = _value_guidance_root(config)
    for section_name in ("terminal", "budget"):
        section = dict_section(root, section_name)
        if section:
            cfg[section_name] = section
    if algorithm != "VGR":
        _merge_section(cfg, root, "backward")
    if algorithm == "VGB-Momentum":
        _merge_section(cfg, root, "momentum")
    if algorithm != "VGR":
        _merge_section(cfg, root, "state_value")
    _normalize_terminal_policy(cfg)
    _normalize_budget_policy(cfg)
    if algorithm == "VGR":
        _strip_vgr_unused_config(cfg)
    cfg["algorithm"] = algorithm
    if algorithm != "VGR":
        cfg.setdefault("N", 8)
    cfg.setdefault("L_f", 8)
    cfg.setdefault("K", 8)
    if algorithm != "VGR":
        cfg.setdefault("L_b", 8)
        cfg.setdefault("B", 1)
    if algorithm == "VGB-Momentum":
        cfg.setdefault("chi", None)
    return cfg
