from pathlib import Path
from urllib.request import urlretrieve

import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F


MODEL_URL = "https://zenodo.org/record/5502060/files/DeepSTARR.model.h5?download=1"
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = REPO_ROOT / "model_data" / "dna_deepstarr" / "deepstarr" / "DeepSTARR.model.h5"


class DeepSTARR(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 256, 7, padding=3)
        self.bn1 = nn.BatchNorm1d(256, eps=0.001)
        self.conv2 = nn.Conv1d(256, 60, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(60, eps=0.001)
        self.conv3 = nn.Conv1d(60, 60, 5, padding=2)
        self.bn3 = nn.BatchNorm1d(60, eps=0.001)
        self.conv4 = nn.Conv1d(60, 120, 3, padding=1)
        self.bn4 = nn.BatchNorm1d(120, eps=0.001)
        self.fc1 = nn.Linear(1800, 256)
        self.bn5 = nn.BatchNorm1d(256, eps=0.001)
        self.fc2 = nn.Linear(256, 256)
        self.bn6 = nn.BatchNorm1d(256, eps=0.001)
        self.dev = nn.Linear(256, 1)
        self.hk = nn.Linear(256, 1)

    def forward(self, x):
        x = F.max_pool1d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool1d(F.relu(self.bn2(self.conv2(x))), 2)
        x = F.max_pool1d(F.relu(self.bn3(self.conv3(x))), 2)
        x = F.max_pool1d(F.relu(self.bn4(self.conv4(x))), 2)
        x = torch.flatten(x.transpose(1, 2), 1)
        x = F.relu(self.bn5(self.fc1(x)))
        x = F.relu(self.bn6(self.fc2(x)))
        return torch.cat([self.dev(x), self.hk(x)], dim=1)


def ensure_model(path=None):
    path = Path(path or DEFAULT_MODEL_PATH)
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(MODEL_URL, path)
    return path


def _dataset(group, suffix):
    for key in group:
        item = group[key]
        if hasattr(item, "shape") and key.endswith(suffix):
            return item[()]
    raise KeyError(suffix)


def _load_conv(layer, group):
    kernel = torch.tensor(_dataset(group, "kernel:0")).permute(2, 1, 0)
    bias = torch.tensor(_dataset(group, "bias:0"))
    layer.weight.data.copy_(kernel)
    layer.bias.data.copy_(bias)


def _load_dense(layer, group):
    kernel = torch.tensor(_dataset(group, "kernel:0")).t()
    bias = torch.tensor(_dataset(group, "bias:0"))
    layer.weight.data.copy_(kernel)
    layer.bias.data.copy_(bias)


def _load_bn(layer, group):
    layer.weight.data.copy_(torch.tensor(_dataset(group, "gamma:0")))
    layer.bias.data.copy_(torch.tensor(_dataset(group, "beta:0")))
    layer.running_mean.data.copy_(torch.tensor(_dataset(group, "moving_mean:0")))
    layer.running_var.data.copy_(torch.tensor(_dataset(group, "moving_variance:0")))


def load_model(path=None, device=None):
    path = ensure_model(path)
    model = DeepSTARR()
    with h5py.File(path, "r") as f:
        _load_conv(model.conv1, f["Conv1D_1st"]["Conv1D_1st_11"])
        _load_conv(model.conv2, f["Conv1D_2"]["Conv1D_2_11"])
        _load_conv(model.conv3, f["Conv1D_3"]["Conv1D_3_11"])
        _load_conv(model.conv4, f["Conv1D_4"]["Conv1D_4_10"])
        _load_dense(model.fc1, f["Dense_1"]["Dense_1_8"])
        _load_dense(model.fc2, f["Dense_2"]["Dense_2_8"])
        _load_dense(model.dev, f["Dense_Dev"]["Dense_Dev_8"])
        _load_dense(model.hk, f["Dense_Hk"]["Dense_Hk_8"])
        _load_bn(model.bn1, f["batch_normalization_60"]["batch_normalization_60"])
        _load_bn(model.bn2, f["batch_normalization_61"]["batch_normalization_61"])
        _load_bn(model.bn3, f["batch_normalization_62"]["batch_normalization_62"])
        _load_bn(model.bn4, f["batch_normalization_63"]["batch_normalization_63"])
        _load_bn(model.bn5, f["batch_normalization_64"]["batch_normalization_64"])
        _load_bn(model.bn6, f["batch_normalization_65"]["batch_normalization_65"])
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model


def one_hot(sequences, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    alphabet = {"A": 0, "C": 1, "G": 2, "T": 3}
    x = torch.zeros((len(sequences), 4, 249), dtype=torch.float32, device=device)
    for row, seq in enumerate(sequences):
        seq = str(seq).upper()[:249]
        for pos, ch in enumerate(seq):
            idx = alphabet.get(ch)
            if idx is not None:
                x[row, idx, pos] = 1.0
    return x


class DeepSTARRScorer:
    def __init__(self, path=None, device=None):
        self.model = load_model(path=path, device=device)
        self.device = next(self.model.parameters()).device

    def predict(self, sequences, batch_size=256):
        rows = []
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start : start + batch_size]
            x = one_hot(batch, device=self.device)
            with torch.no_grad():
                rows.append(self.model(x).detach().cpu())
        if not rows:
            return torch.empty((0, 2))
        return torch.cat(rows, dim=0)
