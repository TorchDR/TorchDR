#!/usr/bin/env bash
set -euo pipefail

# EB spectral landmarking smoke test for TorchDR PHATE.
# Defaults: full EB, n_landmarks=2000, spectral landmarking (random_landmarking=False).
#
# Example:
#   bash run_eb_phate_spectral_landmarking.sh
#   DEVICE=cuda N_SAMPLES=3000 N_LANDMARKS=2000 bash run_eb_phate_spectral_landmarking.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing python: ${PYTHON_BIN}" >&2
  exit 1
fi

EB_PATH="${EB_PATH:-/home/mila/m/matthew.scicluna/ActiveProjects/PHATE/data/EBdata.mat}"
OUTDIR="${OUTDIR:-${ROOT}/eb_spectral_landmarking}"

SEED="${SEED:-2}"
N_SAMPLES="${N_SAMPLES:-0}"        # 0 => full EB
N_PCA="${N_PCA:-50}"
N_LANDMARKS="${N_LANDMARKS:-2000}"
K="${K:-5}"
DECAY="${DECAY:-40}"
T="${T:-30}"
MAX_ITER="${MAX_ITER:-400}"
THRESH="${THRESH:-1e-4}"
DEVICE="${DEVICE:-auto}"           # auto|cpu|cuda

mkdir -p "${OUTDIR}"

echo "[eb-spectral] repo=${ROOT}"
echo "[eb-spectral] git_rev=$(git -C "${ROOT}" rev-parse HEAD)"
echo "[eb-spectral] config: seed=${SEED}, n_samples=${N_SAMPLES}, n_pca=${N_PCA}, n_landmarks=${N_LANDMARKS}, k=${K}, decay=${DECAY}, t=${T}, max_iter=${MAX_ITER}, thresh=${THRESH}, device=${DEVICE}"

export ROOT EB_PATH OUTDIR SEED N_SAMPLES N_PCA N_LANDMARKS K DECAY T MAX_ITER THRESH DEVICE
"${PYTHON_BIN}" -u - <<'PY'
import inspect
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import sparse
from scipy.io import loadmat
from sklearn.decomposition import PCA

root = Path(os.environ["ROOT"])
eb_path = Path(os.environ["EB_PATH"])
outdir = Path(os.environ["OUTDIR"])
seed = int(os.environ["SEED"])
n_samples = int(os.environ["N_SAMPLES"])
n_pca = int(os.environ["N_PCA"])
n_landmarks = int(os.environ["N_LANDMARKS"])
k = int(os.environ["K"])
decay = float(os.environ["DECAY"])
t = int(os.environ["T"])
max_iter = int(os.environ["MAX_ITER"])
thresh = float(os.environ["THRESH"])
device = os.environ["DEVICE"]

if device == "auto":
    device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("DEVICE=cuda requested but CUDA is unavailable")

import sys
sys.path.insert(0, str(root))
from torchdr import PHATE

sig = inspect.signature(PHATE.__init__)
params = sig.parameters
if "n_landmarks" not in params:
    raise RuntimeError(
        "Current TorchDR checkout does not expose `n_landmarks`. "
        "Run this script on the landmarking branch."
    )

mat = loadmat(eb_path)
X = mat["data"]
y = np.asarray(mat["cells"]).reshape(-1)
if sparse.issparse(X):
    X = X.toarray()
X = np.asarray(X, dtype=np.float32)

# Same preprocessing as other EB comparisons.
libsize = np.maximum(X.sum(axis=1, keepdims=True), 1e-12)
X = X / libsize * np.median(libsize)
X = np.sqrt(X).astype(np.float32, copy=False)

if n_samples > 0 and n_samples < X.shape[0]:
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=n_samples, replace=False)
    X = X[idx]
    y = y[idx]
else:
    idx = np.arange(X.shape[0])

if n_pca > 0 and n_pca < X.shape[1]:
    X = PCA(n_components=n_pca, svd_solver="randomized", random_state=seed).fit_transform(X)
    X = X.astype(np.float32, copy=False)

Xt = torch.tensor(X, dtype=torch.float32, device=device)

kwargs = {
    "k": k,
    "decay": decay,
    "t": t,
    "backend": None,
    "device": device,
    "random_state": seed,
    "max_iter": max_iter,
    "n_landmarks": n_landmarks,
    "random_landmarking": False,
    "verbose": True,
}
if "mds_solver" in params:
    kwargs["mds_solver"] = "sgd"
if "thresh" in params:
    kwargs["thresh"] = thresh

print(f"[eb-spectral] n={X.shape[0]}, d={X.shape[1]}, device={device}")
print(f"[eb-spectral] kwargs={kwargs}")

start = time.perf_counter()
model = PHATE(**kwargs)
emb = model.fit_transform(Xt)
runtime = time.perf_counter() - start

emb_np = emb.detach().cpu().numpy().astype(np.float32)
if not np.isfinite(emb_np).all():
    raise RuntimeError("Non-finite values detected in embedding.")

np.save(outdir / "embedding.npy", emb_np)
np.save(outdir / "labels.npy", y)
np.save(outdir / "idx.npy", idx)

plt.figure(figsize=(7, 6))
plt.scatter(emb_np[:, 0], emb_np[:, 1], c=y, s=2, cmap="tab20", linewidths=0)
plt.title(f"TorchDR PHATE EB (spectral landmarks={n_landmarks})")
plt.xlabel("PHATE-1")
plt.ylabel("PHATE-2")
plt.tight_layout()
plt.savefig(outdir / "embedding.png", dpi=220)

meta = {
    "runtime_s": float(runtime),
    "n_iter": int(model.n_iter_.item()) if hasattr(model, "n_iter_") else None,
    "device": device,
    "kwargs": kwargs,
    "n": int(X.shape[0]),
    "d": int(X.shape[1]),
}
with open(outdir / "meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print(f"[eb-spectral] done in {runtime:.2f}s")
print(f"[eb-spectral] saved: {outdir / 'embedding.png'}")
PY
