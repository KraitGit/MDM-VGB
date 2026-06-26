#!/usr/bin/env python
"""Download or copy task assets into data/ and model_data/."""


import argparse
import shutil
from pathlib import Path
from urllib.request import urlretrieve

from datasets import load_dataset
from huggingface_hub import snapshot_download


REPO_ROOT = Path(__file__).resolve().parent
DATA_ROOT = REPO_ROOT / "data"
MODEL_ROOT = REPO_ROOT / "model_data"

QM9_DATASET_ID = "yairschiff/qm9"
QM9_TOKENIZER_ID = "yairschiff/qm9-tokenizer"
LETTER_MODEL_ID = "dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1"
DNA_MODEL_ID = "Hengchang-Liu/D3LM-from-nt"
DEEPSTARR_MODEL_URL = "https://zenodo.org/record/5502060/files/DeepSTARR.model.h5?download=1"


def replace_dir(path, force):
    if not path.exists() and not path.is_symlink():
        return True
    if not force:
        print(f"skip existing {path}")
        return False
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)
    return True


def download_model(repo_id, dst, force):
    if not replace_dir(dst, force):
        return
    dst.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo_id, repo_type="model", local_dir=str(dst))
    print(f"downloaded model {repo_id} -> {dst}")


def download_file(url, dst, force):
    if dst.exists() and dst.stat().st_size > 0 and not force:
        print(f"skip existing {dst}")
        return
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, dst)
    print(f"downloaded {url} -> {dst}")


def prepare_qm9(force):
    dataset_dir = DATA_ROOT / "qm9" / "dataset"
    cache_dir = DATA_ROOT / "qm9" / "cache"
    if replace_dir(dataset_dir, force):
        cache_dir.mkdir(parents=True, exist_ok=True)
        dataset = load_dataset(QM9_DATASET_ID, split="train", cache_dir=str(cache_dir))
        dataset.save_to_disk(str(dataset_dir))
        print(f"downloaded dataset {QM9_DATASET_ID} -> {dataset_dir}")

    download_model(QM9_TOKENIZER_ID, MODEL_ROOT / "qm9" / "qm9-tokenizer", force)


def prepare_letter(force):
    download_model(LETTER_MODEL_ID, MODEL_ROOT / "letter" / "Qwen3-0.6B-diffusion-mdlm-v0.1", force)


def prepare_dna_deepstarr(force):
    download_model(DNA_MODEL_ID, MODEL_ROOT / "dna_deepstarr" / "D3LM-from-nt", force)
    download_file(DEEPSTARR_MODEL_URL, MODEL_ROOT / "dna_deepstarr" / "deepstarr" / "DeepSTARR.model.h5", force)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["qm9"], choices=["all", "qm9", "letter", "dna_deepstarr"])
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    tasks = {"qm9", "letter", "dna_deepstarr"} if "all" in args.tasks else set(args.tasks)
    if "qm9" in tasks:
        prepare_qm9(args.force)
    if "letter" in tasks:
        prepare_letter(args.force)
    if "dna_deepstarr" in tasks:
        prepare_dna_deepstarr(args.force)


if __name__ == "__main__":
    main()
