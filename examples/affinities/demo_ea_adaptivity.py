r"""
Entropic Affinities can adapt to varying noise levels
=====================================================

We show the adaptivity property of entropic affinities on a toy
simulated dataset with heteroscedastic noise.

"""

import torch
import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np
from torchdr import (
    GibbsAffinity,
    EntropicAffinity,
)

# %%
# We generate three Gaussian clusters of points with different noise levels.

seed = 2
torch.manual_seed(seed)
n = 20

X1 = torch.Tensor([-8, -8])[None, :] + torch.normal(
    0, 1, size=(n, 2), dtype=torch.double
)
X2 = torch.Tensor([0, 8])[None, :] + torch.normal(0, 3, size=(n, 2), dtype=torch.double)
X3 = torch.Tensor([8, -8])[None, :] + torch.normal(
    0, 2, size=(n, 2), dtype=torch.double
)
X = torch.cat([X1, X2, X3], 0)


def plot_affinity_graph(G, X, n):
    for i in range(3 * n):
        for j in range(i):
            plt.plot(
                [X[i, 0], X[j, 0]],
                [X[i, 1], X[j, 1]],
                color="black",
                alpha=G[i, j].item(),
            )


# %%
# Row-normalised Gibbs affinity with constant bandwidth
# -----------------------------------------------------
#
#

K = GibbsAffinity(
    sigma=1, normalization_dim=1, keops=False, nodiag=False
).fit_transform(X)

plt.figure(1, (6, 3))

plt.subplot(1, 2, 1)
plt.imshow(K, cmap=cm.Blues, interpolation="none")
plt.title("Gibbs Affinity Matrix")

plt.subplot(1, 2, 2)
plot_affinity_graph(K, X, n)
plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
plt.title("Gibbs Affinity Graph")
plt.show()

# %%
# Entropic affinity (adaptive bandwidth)
# --------------------------------------
#
#

EA = EntropicAffinity(
    perplexity=5, keops=False, verbose=False, nodiag=False
).fit_transform(X)

plt.figure(1, (6, 3))

plt.subplot(1, 2, 1)
plt.imshow(EA, cmap=cm.Oranges, interpolation="none", vmax=1)
plt.title("Entropic Affinity Matrix")

plt.subplot(1, 2, 2)
plot_affinity_graph(EA, X, n)
plt.scatter(X[:, 0], X[:, 1], alpha=0.5, c="orange")
plt.title("Entropic Affinity Graph")
plt.show()
