from sklearn.datasets import make_moons
import torch

def toy_dataset(n=300, dtype="float32", return_tensor=False):
    r"""Generate a toy dataset for testing purposes."""
    X, y = make_moons(n_samples=n, noise=0.05, random_state=0)
    if return_tensor:
        X = torch.tensor(X, dtype=dtype)
        y = torch.tensor(y, dtype=dtype)
        return X, y
    return X.astype(dtype), y
