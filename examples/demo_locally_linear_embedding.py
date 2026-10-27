"""
Locally Linear Embedding of an S-curve
======================================

Unroll a synthetic curved surface using standard LLE. The dense implementation
is intended for small data; it uses quadratic storage and no external datasets.
"""

import matplotlib.pyplot as plt
from sklearn.datasets import make_s_curve

from torchdr import LocallyLinearEmbedding

X, color = make_s_curve(200, random_state=0)
Y = LocallyLinearEmbedding(n_neighbors=12).fit_transform(X)
plt.scatter(Y[:, 0], Y[:, 1], c=color, s=15)
plt.title("Standard Locally Linear Embedding")
plt.show()
