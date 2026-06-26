import math

from .harness import clean_dna
from .deepstarr import DeepSTARRScorer


def gc_content(seq):
    seq = clean_dna(seq)
    if not seq:
        return 0.0
    return sum(1 for ch in seq if ch in "GC") / len(seq)


def max_homopolymer(seq):
    seq = clean_dna(seq)
    if not seq:
        return 0
    best = 1
    cur = 1
    for prev, ch in zip(seq, seq[1:]):
        if prev == ch:
            cur += 1
        else:
            cur = 1
        best = max(best, cur)
    return best


def edit_distance(a, b):
    a = clean_dna(a)
    b = clean_dna(b)
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ch_a in enumerate(a, start=1):
        cur = [i]
        for j, ch_b in enumerate(b, start=1):
            cost = 0 if ch_a == ch_b else 1
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def diversity(sequences, max_pairs=2000):
    sequences = [clean_dna(seq) for seq in sequences if clean_dna(seq)]
    if len(sequences) < 2:
        return 0.0
    total = 0.0
    count = 0
    for i, seq_i in enumerate(sequences):
        for seq_j in sequences[i + 1 :]:
            total += edit_distance(seq_i, seq_j) / max(1, max(len(seq_i), len(seq_j)))
            count += 1
            if count >= int(max_pairs):
                return total / count
    return total / max(1, count)


class DeepSTARRDevOracle:
    def __init__(self, path=None, device=None, batch_size=256):
        self.scorer = DeepSTARRScorer(path=path, device=device)
        self.batch_size = int(batch_size)
        self.cache = {}

    def _valid_seq(self, seq):
        seq = clean_dna(seq)
        if len(seq) != 249:
            return None
        return seq

    def score(self, sequences):
        outputs = []
        uncached = []
        for seq in sequences:
            seq = self._valid_seq(seq)
            if seq is None:
                outputs.append({"dev": float("-inf"), "hk": float("-inf"), "valid": False})
                continue
            if seq in self.cache:
                outputs.append(self.cache[seq])
                continue
            outputs.append(None)
            uncached.append(seq)
        if uncached:
            preds = self.scorer.predict(uncached, batch_size=self.batch_size).tolist()
            for seq, pred in zip(uncached, preds):
                self.cache[seq] = {
                    "dev": float(pred[0]),
                    "hk": float(pred[1]),
                    "valid": True,
                    "gc": float(gc_content(seq)),
                    "max_homopolymer": int(max_homopolymer(seq)),
                }
        for idx, value in enumerate(outputs):
            if value is None:
                seq = self._valid_seq(sequences[idx])
                outputs[idx] = self.cache[seq]
        return [dict(item) for item in outputs]

    def tau(self, dev_score, mean, std, eta=1.0):
        if std <= 0 or not math.isfinite(dev_score):
            return 0.0
        z = max(-3.0, min(3.0, (float(dev_score) - float(mean)) / float(std)))
        return float(math.exp(float(eta) * z))
