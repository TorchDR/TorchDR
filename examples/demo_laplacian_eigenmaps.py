"""
Laplacian Eigenmaps on Iris
===========================

Embed Iris using a dense Gaussian graph and a normalized graph Laplacian.
This small example uses no downloaded data. Dense storage is quadratic in
sample count; it is intended for small datasets rather than large k-NN graphs.
"""

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

from torchdr import LaplacianEigenmaps

X, labels = load_iris(return_X_y=True)
embedding = LaplacianEigenmaps(n_components=2).fit_transform(X)
plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, s=15)
plt.xlabel("First nontrivial eigenvector")
plt.ylabel("Second nontrivial eigenvector")
plt.title("Laplacian Eigenmaps")
plt.show()
