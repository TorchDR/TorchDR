from sklearn.datasets import make_moons
from sklearn.datasets import load_iris


def toy_dataset(n=300, dtype="float32"):
    r"""Generate a toy dataset for testing purposes."""
    X, y = make_moons(n_samples=n, noise=0.05, random_state=0)
    return X.astype(dtype), y


def iris_dataset(dtype="float32"):
    r"""Iris dataset for testing purposes."""
    iris = load_iris()
    return iris.data.astype(dtype), iris.target
