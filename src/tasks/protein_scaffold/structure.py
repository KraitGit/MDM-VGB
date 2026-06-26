import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np


AA3_TO_AA1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "MSE": "M",
}


def parse_atom_names(text):
    if text is None:
        return ["CA"]
    if isinstance(text, (list, tuple)):
        return [str(x).strip() for x in text if str(x).strip()]
    return [item.strip() for item in str(text).replace(";", ",").split(",") if item.strip()]


def clean_sequence(sequence):
    return "".join(ch for ch in str(sequence).strip().upper() if ch.isalpha())


def parse_pdb_residues(path, chain=None):
    chain = None if chain in (None, "", "*") else str(chain)
    residues = []
    by_key = {}
    selected_chain = chain
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            altloc = line[16].strip()
            if altloc not in ("", "A", "1"):
                continue
            resname = line[17:20].strip().upper()
            if resname not in AA3_TO_AA1:
                continue
            line_chain = line[21].strip() or "_"
            if selected_chain is None:
                selected_chain = line_chain
            if line_chain != selected_chain:
                continue
            atom = line[12:16].strip()
            try:
                resseq = int(line[22:26])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except Exception:
                continue
            icode = line[26].strip()
            try:
                bfactor = float(line[60:66])
            except Exception:
                bfactor = 0.0
            key = (line_chain, resseq, icode)
            if key not in by_key:
                row = {
                    "chain": line_chain,
                    "resseq": resseq,
                    "icode": icode,
                    "resname": resname,
                    "aa": AA3_TO_AA1[resname],
                    "atoms": {},
                }
                by_key[key] = row
                residues.append(row)
            by_key[key]["atoms"][atom] = {
                "coord": [x, y, z],
                "bfactor": bfactor,
            }
    return residues


def motif_atom_table(path, positions, chain=None, atom_names=None):
    atom_names = parse_atom_names(atom_names)
    residues = parse_pdb_residues(path, chain=chain)
    table = {}
    for pos in positions:
        pos = int(pos)
        if pos < 0 or pos >= len(residues):
            continue
        residue = residues[pos]
        for atom in atom_names:
            if atom not in residue["atoms"]:
                continue
            entry = residue["atoms"][atom]
            table[(pos, atom)] = {
                "coord": np.asarray(entry["coord"], dtype=np.float64),
                "bfactor": float(entry.get("bfactor", 0.0)),
                "aa": residue["aa"],
                "chain": residue["chain"],
                "resseq": residue["resseq"],
                "icode": residue["icode"],
            }
    return table


def all_atom_bfactors(path, chain=None, atom_names=None):
    atom_names = parse_atom_names(atom_names)
    values = []
    for residue in parse_pdb_residues(path, chain=chain):
        for atom in atom_names:
            if atom in residue["atoms"]:
                values.append(float(residue["atoms"][atom].get("bfactor", 0.0)))
    return values


def kabsch_rmsd(pred_coords, native_coords):
    pred = np.asarray(pred_coords, dtype=np.float64)
    native = np.asarray(native_coords, dtype=np.float64)
    if pred.shape != native.shape:
        raise ValueError(f"coordinate shape mismatch: {pred.shape} vs {native.shape}")
    if pred.ndim != 2 or pred.shape[1] != 3 or pred.shape[0] == 0:
        raise ValueError(f"invalid coordinate array: {pred.shape}")
    pred_center = pred.mean(axis=0)
    native_center = native.mean(axis=0)
    pred0 = pred - pred_center
    native0 = native - native_center
    cov = pred0.T @ native0
    v, _, wt = np.linalg.svd(cov)
    det = np.linalg.det(v @ wt)
    fix = np.eye(3)
    if det < 0:
        fix[-1, -1] = -1.0
    rot = v @ fix @ wt
    aligned = pred0 @ rot
    diff = aligned - native0
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def motif_rmsd(predicted_pdb, native_pdb, positions, native_chain=None, predicted_chain=None, atom_names=None):
    atom_names = parse_atom_names(atom_names)
    native = motif_atom_table(native_pdb, positions, chain=native_chain, atom_names=atom_names)
    pred = motif_atom_table(predicted_pdb, positions, chain=predicted_chain, atom_names=atom_names)
    labels = [label for label in sorted(native) if label in pred]
    if not labels and atom_names != ["CA"]:
        return motif_rmsd(predicted_pdb, native_pdb, positions, native_chain, predicted_chain, ["CA"])
    if not labels:
        raise ValueError("no common motif atoms between predicted and native structures")
    native_coords = np.stack([native[label]["coord"] for label in labels], axis=0)
    pred_coords = np.stack([pred[label]["coord"] for label in labels], axis=0)
    plddt_values = [float(pred[label].get("bfactor", 0.0)) for label in labels]
    return {
        "motif_rmsd": kabsch_rmsd(pred_coords, native_coords),
        "motif_atoms": len(labels),
        "plddt_motif": float(np.mean(plddt_values)) if plddt_values else 0.0,
    }


def low_complexity(sequence):
    sequence = clean_sequence(sequence)
    if not sequence:
        return {
            "aa_entropy": 0.0,
            "top_aa_frac": 1.0,
            "longest_run_frac": 1.0,
            "adjacent_repeat_frac": 1.0,
            "unique_aa": 0.0,
        }
    counts = {}
    for ch in sequence:
        counts[ch] = counts.get(ch, 0) + 1
    total = len(sequence)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log(p)
    longest = 1
    run = 1
    for idx in range(1, total):
        if sequence[idx] == sequence[idx - 1]:
            run += 1
        else:
            longest = max(longest, run)
            run = 1
    longest = max(longest, run)
    adjacent = sum(1 for idx in range(1, total) if sequence[idx] == sequence[idx - 1])
    return {
        "aa_entropy": float(entropy / math.log(20)),
        "top_aa_frac": float(max(counts.values()) / total),
        "longest_run_frac": float(longest / total),
        "adjacent_repeat_frac": float(adjacent / max(1, total - 1)),
        "unique_aa": float(len(counts)),
    }


def motif_preserved(task, sequence):
    sequence = clean_sequence(sequence)
    positions = task.get("motif_positions", [])
    motif = str(task.get("motif_sequence", ""))
    if len(positions) != len(motif):
        motif_text = task.get("motif_text", {})
        for pos, aa in motif_text.items():
            pos = int(pos)
            if pos >= len(sequence) or sequence[pos] != aa:
                return False
        return bool(motif_text)
    for idx, pos in enumerate(positions):
        pos = int(pos)
        if pos >= len(sequence) or sequence[pos] != motif[idx]:
            return False
    return True


def dense_reward_from_rmsd(value, scale=1.0):
    value = float(value)
    scale = max(1e-8, float(scale))
    return float(math.exp(-((value / scale) ** 2)))


def sequence_cache_key(sequence, backend="esmfold", backend_version="unknown", context=None):
    text = clean_sequence(sequence) + "|" + str(backend) + "|" + str(backend_version)
    if context is not None:
        text += "|" + json.dumps(context, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def cache_path(cache_dir, sequence, backend="esmfold", backend_version="unknown", context=None):
    key = sequence_cache_key(sequence, backend=backend, backend_version=backend_version, context=context)
    return Path(cache_dir) / key[:2] / f"{key}.json"


def folded_pdb_path(folded_dir, sequence, backend="esmfold", backend_version="unknown"):
    key = sequence_cache_key(sequence, backend=backend, backend_version=backend_version)
    return Path(folded_dir) / key[:2] / f"{key}.pdb"


def read_cache(cache_dir, sequence, backend="esmfold", backend_version="unknown", context=None):
    path = cache_path(cache_dir, sequence, backend=backend, backend_version=backend_version, context=context)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_cache(cache_dir, sequence, record, backend="esmfold", backend_version="unknown", context=None):
    path = cache_path(cache_dir, sequence, backend=backend, backend_version=backend_version, context=context)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(record)
    record["cache_key"] = path.stem
    record["cache_path"] = str(path)
    tmp_path = str(path) + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, sort_keys=True)
    os.replace(tmp_path, path)
    return record


def score_cache_context(task, atom_names=None, rmsd_scale=1.0, success_threshold=1.0):
    return {
        "task_id": task.get("id") or task.get("task_id") or task.get("pdb"),
        "pdb": task.get("pdb"),
        "length": int(task.get("length", 0) or 0),
        "native_pdb": str(task.get("native_pdb") or task.get("pdb_path") or ""),
        "native_motif_coords_path": str(task.get("native_motif_coords_path") or ""),
        "motif_positions": [int(x) for x in task.get("motif_positions", [])],
        "atom_names": parse_atom_names(atom_names or task.get("atom_names") or "CA"),
        "rmsd_scale": float(rmsd_scale),
        "success_threshold": float(success_threshold),
    }


def esmfold_version():
    try:
        import esm

        return getattr(esm, "__version__", "esm_unknown")
    except Exception:
        return "esm_unavailable"


def omegafold_version():
    try:
        import omegafold

        return getattr(omegafold, "__version__", "omegafold_unknown")
    except Exception:
        return "omegafold_unavailable"


def default_omegafold_weights_file(model_num=1):
    env_path = os.environ.get("OMEGAFOLD_WEIGHTS_FILE") or os.environ.get("OMEGAFOLD_WEIGHTS")
    if env_path:
        return str(env_path)
    repo_root = Path(__file__).resolve().parents[4]
    name = "model2.pt" if int(model_num) == 2 else "model.pt"
    return str(repo_root / "cache" / "omegafold" / name)


def omegafold_weights_url(model_num=1):
    if int(model_num) == 2:
        return "https://helixon.s3.amazonaws.com/release2.pt"
    return "https://helixon.s3.amazonaws.com/release1.pt"


def omegafold_backend_version(weights_file=None):
    version = omegafold_version()
    if weights_file:
        path = Path(weights_file)
        if path.exists():
            return f"{version}|weights={path.name}:{path.stat().st_size}"
        return f"{version}|weights={path.name}:missing"
    return version


def load_esmfold_model(device=None, chunk_size=None):
    import torch
    import esm

    model = esm.pretrained.esmfold_v1()
    if chunk_size is not None and hasattr(model, "set_chunk_size"):
        model.set_chunk_size(int(chunk_size))
    model = model.eval()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return model


def fold_sequence_esmfold(sequence, out_pdb, model=None, device=None, chunk_size=None):
    import torch

    sequence = clean_sequence(sequence)
    out_pdb = Path(out_pdb)
    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    if model is None:
        model = load_esmfold_model(device=device, chunk_size=chunk_size)
    with torch.no_grad():
        pdb_text = model.infer_pdb(sequence)
    with open(out_pdb, "w", encoding="utf-8") as f:
        f.write(pdb_text)
    return str(out_pdb)


class OmegaFoldRunner:
    def __init__(self, device=None, weights_file=None, subbatch_size=None, num_cycle=None, model_num=1):
        import argparse
        import torch
        import omegafold as of
        from omegafold import pipeline

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.num_cycle = int(num_cycle or 10)
        self.forward_config = argparse.Namespace(subbatch_size=subbatch_size, num_recycle=self.num_cycle)
        if hasattr(pipeline, "_set_precision"):
            pipeline._set_precision(True)
        model = of.OmegaFold(of.make_config(int(model_num)))
        if weights_file is None:
            weights_file = default_omegafold_weights_file(model_num)
        state_dict = pipeline._load_weights(omegafold_weights_url(model_num), str(weights_file))
        if "model" in state_dict:
            state_dict = state_dict.pop("model")
        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)
        self.model = model
        self.weights_file = str(weights_file)
        self.backend_version = omegafold_backend_version(weights_file)

    def predict(self, sequence):
        import tempfile
        import torch
        from omegafold import pipeline

        sequence = clean_sequence(sequence)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            fasta = tmp / "input.fasta"
            raw_out = tmp / "omegafold"
            raw_out.mkdir(parents=True, exist_ok=True)
            write_fasta(fasta, [("seq_0", sequence)])
            inputs = pipeline.fasta2inputs(
                str(fasta),
                num_pseudo_msa=15,
                output_dir=str(raw_out),
                device=self.device,
                mask_rate=0.12,
                num_cycle=self.num_cycle,
            )
            input_data, _ = next(inputs)
            with torch.no_grad():
                output = self.model(input_data, predict_with_confidence=True, fwd_cfg=self.forward_config)
        return output, input_data

    def fold_sequence(self, sequence, out_pdb):
        from omegafold import pipeline

        output, input_data = self.predict(sequence)
        out_pdb = Path(out_pdb)
        out_pdb.parent.mkdir(parents=True, exist_ok=True)
        pipeline.save_pdb(
            pos14=output["final_atom_positions"],
            b_factors=output["confidence"] * 100,
            sequence=input_data[0]["p_msa"][0],
            mask=input_data[0]["p_msa_mask"][0],
            save_path=str(out_pdb),
            model=0,
        )
        return str(out_pdb)

    def score_sequence(self, task, sequence, atom_names=None, rmsd_scale=1.0, success_threshold=1.0):
        atom_names = parse_atom_names(atom_names or task.get("atom_names") or "CA")
        if atom_names != ["CA"] or not task.get("native_motif_coords_path"):
            return None
        output, _ = self.predict(sequence)
        positions = [int(x) for x in task.get("motif_positions", [])]
        if not positions:
            raise ValueError("task is missing motif_positions")
        native_coords = np.load(task["native_motif_coords_path"])
        pred = output["final_atom_positions"].detach().float().cpu().numpy()
        confidence = output["confidence"].detach().float().cpu().numpy() * 100.0
        pred_coords = pred[positions, 1, :]
        plddt_motif = confidence[positions]
        motif_rmsd_value = kabsch_rmsd(pred_coords, native_coords)
        reward = dense_reward_from_rmsd(motif_rmsd_value, scale=rmsd_scale)
        out = {
            "sequence": clean_sequence(sequence),
            "pdb": task.get("pdb"),
            "task_id": task.get("id"),
            "native_pdb": str(task.get("native_pdb") or task.get("pdb_path")),
            "motif_rmsd": float(motif_rmsd_value),
            "motif_atoms": int(len(positions)),
            "reward": float(reward),
            "success": bool(float(motif_rmsd_value) <= float(success_threshold)),
            "plddt_motif": float(np.mean(plddt_motif)) if len(plddt_motif) else 0.0,
            "plddt_all": float(np.mean(confidence)) if len(confidence) else 0.0,
            "motif_preserved": bool(motif_preserved(task, sequence)),
            "rmsd_scale": float(rmsd_scale),
            "success_threshold": float(success_threshold),
            "atom_names": atom_names,
            "scoring_path": "omegafold_tensor",
        }
        out.update(low_complexity(sequence))
        return out


def load_omegafold_model(device=None, weights_file=None, subbatch_size=None, num_cycle=None, model_num=1):
    return OmegaFoldRunner(
        device=device,
        weights_file=weights_file,
        subbatch_size=subbatch_size,
        num_cycle=num_cycle,
        model_num=model_num,
    )


def write_fasta(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for name, sequence in rows:
            f.write(f">{name}\n")
            seq = clean_sequence(sequence)
            for start in range(0, len(seq), 80):
                f.write(seq[start:start + 80] + "\n")


def fold_sequences_omegafold(sequences, out_dir, backend_version=None, device=None, subbatch_size=None, num_cycle=None, weights_file=None):
    import shutil
    import subprocess
    import sys
    import tempfile

    executable = shutil.which("omegafold")
    if executable is None:
        candidate = Path(sys.executable).resolve().parent / "omegafold"
        if candidate.exists():
            executable = str(candidate)
    if executable is None:
        raise FileNotFoundError("omegafold executable not found; install OmegaFold first")
    if weights_file is None:
        weights_file = default_omegafold_weights_file()
    backend_version = backend_version or omegafold_backend_version(weights_file)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sequence_rows = []
    sequence_to_path = {}
    for idx, sequence in enumerate(sequences):
        sequence = clean_sequence(sequence)
        path = folded_pdb_path(out_dir, sequence, backend="omegafold", backend_version=backend_version)
        sequence_to_path[sequence] = str(path)
        if not path.exists():
            sequence_rows.append((f"seq_{idx}", sequence))
    if not sequence_rows:
        return sequence_to_path
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        fasta = tmp / "input.fasta"
        raw_out = tmp / "omegafold"
        write_fasta(fasta, sequence_rows)
        cmd = [executable, str(fasta), str(raw_out)]
        if device is not None:
            cmd.extend(["--device", str(device)])
        if subbatch_size is not None:
            cmd.extend(["--subbatch_size", str(subbatch_size)])
        if num_cycle is not None:
            cmd.extend(["--num_cycle", str(num_cycle)])
        if weights_file is not None:
            cmd.extend(["--weights_file", str(weights_file)])
        subprocess.run(cmd, check=True)
        for name, sequence in sequence_rows:
            srcs = list(raw_out.glob(f"{name}*.pdb"))
            if not srcs:
                srcs = list(raw_out.glob("*.pdb"))
            if not srcs:
                raise FileNotFoundError(f"OmegaFold did not produce PDB for {name}")
            dst = folded_pdb_path(out_dir, sequence, backend="omegafold", backend_version=backend_version)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(sorted(srcs)[0]), str(dst))
    return sequence_to_path


def score_predicted_pdb(task, sequence, predicted_pdb, atom_names=None, rmsd_scale=1.0, success_threshold=1.0):
    sequence = clean_sequence(sequence)
    native_pdb = task.get("native_pdb") or task.get("pdb_path")
    if not native_pdb:
        raise ValueError("task is missing native_pdb/pdb_path")
    positions = [int(x) for x in task.get("motif_positions", [])]
    if not positions:
        raise ValueError("task is missing motif_positions")
    native_chain = task.get("chain") or task.get("native_chain")
    predicted_chain = task.get("predicted_chain")
    atom_names = parse_atom_names(atom_names or task.get("atom_names") or "CA")
    plddt_all = all_atom_bfactors(predicted_pdb, chain=predicted_chain, atom_names=atom_names)
    native_coords_path = task.get("native_motif_coords_path")
    if atom_names == ["CA"] and native_coords_path:
        native_coords = np.load(native_coords_path)
        pred = motif_atom_table(predicted_pdb, positions, chain=predicted_chain, atom_names=atom_names)
        labels = [(int(pos), "CA") for pos in positions if (int(pos), "CA") in pred]
        if len(labels) != len(native_coords):
            raise ValueError(
                f"predicted motif atoms ({len(labels)}) do not match native motif coords ({len(native_coords)})"
            )
        pred_coords = np.stack([pred[label]["coord"] for label in labels], axis=0)
        plddt_motif = [float(pred[label].get("bfactor", 0.0)) for label in labels]
        rmsd_info = {
            "motif_rmsd": kabsch_rmsd(pred_coords, native_coords),
            "motif_atoms": len(labels),
            "plddt_motif": float(np.mean(plddt_motif)) if plddt_motif else 0.0,
        }
    else:
        rmsd_info = motif_rmsd(
            predicted_pdb,
            native_pdb,
            positions,
            native_chain=native_chain,
            predicted_chain=predicted_chain,
            atom_names=atom_names,
        )
    reward = dense_reward_from_rmsd(rmsd_info["motif_rmsd"], scale=rmsd_scale)
    out = {
        "sequence": sequence,
        "pdb": task.get("pdb"),
        "task_id": task.get("id"),
        "native_pdb": str(native_pdb),
        "predicted_pdb": str(predicted_pdb),
        "motif_rmsd": float(rmsd_info["motif_rmsd"]),
        "motif_atoms": int(rmsd_info["motif_atoms"]),
        "reward": float(reward),
        "success": bool(float(rmsd_info["motif_rmsd"]) <= float(success_threshold)),
        "plddt_motif": float(rmsd_info.get("plddt_motif", 0.0)),
        "plddt_all": float(np.mean(plddt_all)) if plddt_all else 0.0,
        "motif_preserved": bool(motif_preserved(task, sequence)),
        "rmsd_scale": float(rmsd_scale),
        "success_threshold": float(success_threshold),
        "atom_names": atom_names,
    }
    out.update(low_complexity(sequence))
    return out


def score_sequence(task, sequence, cache_dir, folded_dir=None, backend="esmfold", backend_version=None, fold=False, model=None, device=None, chunk_size=None, atom_names=None, rmsd_scale=1.0, success_threshold=1.0, subbatch_size=None, num_cycle=None, weights_file=None):
    if backend == "esmfold":
        backend_version = backend_version or esmfold_version()
    elif backend == "omegafold":
        backend_version = backend_version or omegafold_version()
    else:
        backend_version = backend_version or "unknown"
    if backend == "omegafold" and model is not None and hasattr(model, "backend_version"):
        backend_version = model.backend_version
    cache_context = score_cache_context(
        task,
        atom_names=atom_names,
        rmsd_scale=rmsd_scale,
        success_threshold=success_threshold,
    )
    cached = read_cache(cache_dir, sequence, backend=backend, backend_version=backend_version, context=cache_context)
    if cached is not None:
        return cached
    if folded_dir is None:
        folded_dir = Path(cache_dir) / "folded_pdbs"
    pdb_path = folded_pdb_path(folded_dir, sequence, backend=backend, backend_version=backend_version)
    if not pdb_path.exists():
        if not fold:
            raise FileNotFoundError(f"missing folded PDB for sequence and fold=False: {pdb_path}")
        if backend == "esmfold":
            fold_sequence_esmfold(sequence, pdb_path, model=model, device=device, chunk_size=chunk_size)
        elif backend == "omegafold":
            if model is not None and hasattr(model, "score_sequence"):
                record = model.score_sequence(
                    task,
                    sequence,
                    atom_names=atom_names,
                    rmsd_scale=rmsd_scale,
                    success_threshold=success_threshold,
                )
                if record is not None:
                    record["backend"] = backend
                    record["backend_version"] = backend_version
                    return write_cache(
                        cache_dir,
                        sequence,
                        record,
                        backend=backend,
                        backend_version=backend_version,
                        context=cache_context,
                    )
            if model is not None and hasattr(model, "fold_sequence"):
                model.fold_sequence(sequence, pdb_path)
            else:
                fold_sequences_omegafold(
                    [sequence],
                    folded_dir,
                    backend_version=backend_version,
                    device=device,
                    subbatch_size=subbatch_size,
                    num_cycle=num_cycle,
                    weights_file=weights_file,
                )
        else:
            raise ValueError(f"unsupported folding backend: {backend}")
    record = score_predicted_pdb(
        task,
        sequence,
        pdb_path,
        atom_names=atom_names,
        rmsd_scale=rmsd_scale,
        success_threshold=success_threshold,
    )
    record["backend"] = backend
    record["backend_version"] = backend_version
    return write_cache(
        cache_dir,
        sequence,
        record,
        backend=backend,
        backend_version=backend_version,
        context=cache_context,
    )
