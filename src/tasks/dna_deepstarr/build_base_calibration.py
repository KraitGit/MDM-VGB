import argparse
import json
from pathlib import Path

import numpy as np

from .harness import D3LMWrapper
from .deepstarr_oracle import DeepSTARRDevOracle
from .deepstarr_oracle import diversity
from utils import write_jsonl
from utils import set_seed


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def generate_scored(d3lm, oracle, count, args):
    rows = []
    remaining = int(count)
    sample_id = 0
    while remaining > 0:
        batch = min(int(args.batch_size), remaining)
        generated = d3lm.generate_tokens(
            n_tokens=args.n_tokens,
            batch_size=batch,
            steps=args.steps,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            alg=args.alg,
        )
        valid = [row["sequence_249"] for row in generated if row.get("sequence_249")]
        scores = oracle.score(valid)
        score_idx = 0
        for row in generated:
            seq = row.get("sequence_249")
            if seq:
                scored = scores[score_idx]
                score_idx += 1
            else:
                scored = {"dev": None, "hk": None, "valid": False, "gc": None, "max_homopolymer": None}
            out = {
                "sample_id": sample_id,
                "token_ids": row["token_ids"],
                "decoded_nt": row["decoded_nt"],
                "decoded_length": len(row["decoded_nt"]),
                "sequence_249": seq,
                "valid_249": bool(seq),
                "deepstarr_dev": scored.get("dev"),
                "deepstarr_hk": scored.get("hk"),
                "gc_content": scored.get("gc"),
                "max_homopolymer": scored.get("max_homopolymer"),
            }
            rows.append(out)
            sample_id += 1
        remaining -= batch
    return rows


def summarize(rows, thresholds=None):
    thresholds = thresholds or {}
    valid = [row for row in rows if row.get("valid_249")]
    dev = [float(row["deepstarr_dev"]) for row in valid]
    hk = [float(row["deepstarr_hk"]) for row in valid]
    seqs = [row["sequence_249"] for row in valid]
    if not rows:
        return {}
    summary = {
        "num_samples": len(rows),
        "valid_249_rate": len(valid) / len(rows),
        "decoded_length_mean": float(np.mean([row["decoded_length"] for row in rows])),
        "decoded_length_min": int(min(row["decoded_length"] for row in rows)),
        "duplicate_rate": 1.0 - len(set(seqs)) / max(1, len(seqs)),
        "unique_fraction": len(set(seqs)) / max(1, len(seqs)),
        "diversity": float(diversity(seqs, max_pairs=200)),
    }
    if dev:
        summary.update(
            {
                "mean_DeepSTARR_Dev": float(np.mean(dev)),
                "mean_DeepSTARR_Hk": float(np.mean(hk)),
                "gc": float(np.mean([row["gc_content"] for row in valid])),
                "median_max_homopolymer": float(np.median([row["max_homopolymer"] for row in valid])),
            }
        )
    if thresholds.get("t95") is not None and dev:
        summary["success_95"] = float(np.mean([score >= thresholds["t95"] for score in dev]))
    if thresholds.get("t99") is not None and dev:
        summary["success_99"] = float(np.mean([score >= thresholds["t99"] for score in dev]))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Hengchang-Liu/D3LM-from-nt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--n-calib", type=int, default=1000)
    parser.add_argument("--n-eval", type=int, default=1000)
    parser.add_argument("--n-tokens", type=int, default=48)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--alg", default="random")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--deepstarr-model", default=None)
    parser.add_argument("--output-dir", default="artifacts/d3lm_deepstarr/base")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    d3lm = D3LMWrapper(args.model_id, device=args.device, dtype=args.dtype)
    oracle = DeepSTARRDevOracle(path=args.deepstarr_model, device=args.device)

    calib = generate_scored(d3lm, oracle, args.n_calib, args)
    eval_rows = generate_scored(d3lm, oracle, args.n_eval, args)
    write_jsonl(str(out_dir / "calib_samples.jsonl"), calib)
    write_jsonl(str(out_dir / "eval_base_samples.jsonl"), eval_rows)

    calib_dev = [float(row["deepstarr_dev"]) for row in calib if row.get("valid_249")]
    if not calib_dev:
        raise ValueError("no valid 249nt calibration samples")
    thresholds = {
        "t95": float(np.quantile(calib_dev, 0.95)),
        "t99": float(np.quantile(calib_dev, 0.99)),
        "mean": float(np.mean(calib_dev)),
        "std": float(np.std(calib_dev) if np.std(calib_dev) > 0 else 1.0),
        "n_tokens": int(args.n_tokens),
        "reward": "DeepSTARR_Dev",
    }
    write_json(out_dir / "thresholds.json", thresholds)
    metrics = {
        "calib": summarize(calib, thresholds),
        "base_eval": summarize(eval_rows, thresholds),
        "thresholds": thresholds,
    }
    write_json(out_dir / "base_metrics.json", metrics)
    print(json.dumps(metrics["base_eval"], sort_keys=True))


if __name__ == "__main__":
    main()
