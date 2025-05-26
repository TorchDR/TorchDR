from sklearn.datasets import make_moons


def toy_dataset(n=300, dtype="float32"):
    r"""Generate a toy dataset for testing purposes."""
    X, y = make_moons(n_samples=n, noise=0.05, random_state=0)
    return X.astype(dtype), y
