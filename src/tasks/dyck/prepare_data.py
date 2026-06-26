import argparse
import math
import pickle
import random
from pathlib import Path

from tqdm.auto import tqdm

from .model import BOS_ID, EOS_ID, ID_TO_TOKEN


PACKAGE_ROOT = Path(__file__).resolve().parents[4]


def resolve_inside_package(path):
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PACKAGE_ROOT / candidate
    candidate = candidate.resolve()
    root = PACKAGE_ROOT.resolve()
    if root not in candidate.parents and candidate != root:
        raise ValueError(f"data path must stay inside this repository: {candidate}")
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def normalize_probs(probs, num_types):
    if probs is None:
        return [1.0 / num_types for _ in range(num_types)]
    probs = [float(x) for x in probs]
    if len(probs) != num_types:
        raise ValueError("type_probs length must match num_types")
    total = sum(probs)
    if total <= 0:
        raise ValueError("type_probs must sum to a positive value")
    return [x / total for x in probs]


def accept_ids(seq, num_types, length):
    if len(seq) != length + 2 or seq[0] != BOS_ID or seq[-1] != EOS_ID:
        return False
    stack = []
    for token in seq[1:-1]:
        if 0 <= token < num_types:
            stack.append(token)
        elif num_types <= token < 2 * num_types:
            if not stack or stack.pop() + num_types != token:
                return False
        else:
            return False
    return len(stack) == 0


def ids_to_text(seq):
    return "".join(ID_TO_TOKEN[int(x)] for x in seq)


def sample_open_type(rng, type_probs):
    return rng.choices(range(len(type_probs)), weights=type_probs, k=1)[0]


def generate_one(rng, length, num_types, type_probs, max_depth):
    stack = []
    seq = []
    log_prob = 0.0
    prefix_log_probs = []

    for step in range(length):
        remaining = length - step
        if not stack:
            token = sample_open_type(rng, type_probs)
            seq.append(token)
            stack.append(token)
            log_prob += math.log(type_probs[token])
        else:
            pop_prob = 0.5
            if remaining == len(stack) or (max_depth is not None and len(stack) == max_depth):
                pop_prob = 1.0
            if rng.random() < pop_prob:
                token = stack.pop() + num_types
                seq.append(token)
                log_prob += math.log(pop_prob)
            else:
                token = sample_open_type(rng, type_probs)
                seq.append(token)
                stack.append(token)
                log_prob += math.log(1.0 - pop_prob) + math.log(type_probs[token])
        prefix_log_probs.append(log_prob)

    tokens = [BOS_ID] + seq + [EOS_ID]
    if not accept_ids(tokens, num_types, length):
        raise RuntimeError("internal Dyck generator produced an invalid sequence")
    return {"tokens": tokens, "text": ids_to_text(tokens), "log_prob": log_prob, "prefix_log_probs": prefix_log_probs}


def make_split(name, count, seed, length, num_types, type_probs, max_depth):
    rng = random.Random(seed)
    rows = []
    iterator = tqdm(range(count), total=count, desc=f"dyck:{name}")
    for _ in iterator:
        rows.append(generate_one(rng, length, num_types, type_probs, max_depth))
    return rows


def save_pickle(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as handle:
        pickle.dump(rows, handle)
    tmp_path.replace(path)


def write_split(output_dir, name, rows):
    path = output_dir / f"{name}.pkl"
    save_pickle(path, rows)
    return str(path)


def prepare_dyck_data(args):
    output_dir = resolve_inside_package(args.output_dir)
    type_probs = normalize_probs(args.type_probs, args.num_types)
    splits = [
        ("train", args.train_count, args.seed, type_probs),
        ("dev", args.dev_count, args.seed + 1, type_probs),
        ("ood_eval", args.ood_count, args.seed + 2, normalize_probs(args.ood_type_probs, args.num_types)),
    ]
    outputs = {}
    for name, count, seed, probs in splits:
        if count <= 0:
            continue
        rows = make_split(name, count, seed, args.length, args.num_types, probs, args.max_depth)
        outputs[name] = write_split(output_dir, name, rows)
    meta = {
        "num_types": args.num_types,
        "length": args.length,
        "max_depth": args.max_depth,
        "type_probs": type_probs,
        "ood_type_probs": normalize_probs(args.ood_type_probs, args.num_types),
        "seed": args.seed,
        "outputs": outputs,
    }
    save_pickle(output_dir / "meta.pkl", meta)
    return meta


def parse_probs(text):
    if text is None:
        return None
    return [float(x) for x in text.split(",")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/dyck/base_dataset")
    parser.add_argument("--train-count", type=int, default=300000)
    parser.add_argument("--dev-count", type=int, default=10000)
    parser.add_argument("--ood-count", type=int, default=10000)
    parser.add_argument("--length", type=int, default=32)
    parser.add_argument("--num-types", type=int, default=2)
    parser.add_argument("--type-probs", default="0.2,0.8")
    parser.add_argument("--ood-type-probs", default="0.8,0.2")
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    args.type_probs = parse_probs(args.type_probs)
    args.ood_type_probs = parse_probs(args.ood_type_probs)
    print(prepare_dyck_data(args))


if __name__ == "__main__":
    main()
