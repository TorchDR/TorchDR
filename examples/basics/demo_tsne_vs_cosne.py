r"""
TSNE vs COSNE : Euclidean vs Hyperbolic
=======================================

We compare in this example two dimensionalty reduction methods:
T-SNE and CO-SNE on a synthetic hierarchical toy dataset and on
singlecell data. The first method computes an embedding in a 2D
Euclidean space while the second one operates in the Hyperbolic
Poincaré Ball model.
"""

import numpy as np
from torchdr.utils.visu import plot_disk
from torchdr import TSNE, COSNE
import urllib.request
import matplotlib.pylab as plt


# %%
# Load the SNARE-seq dataset (gene expression) with cell type labels
# -------------------------------------------------------------------


def load_numpy_from_url(url, delimiter="\t"):
    """
    Load a numpy array from a URL.

    Parameters
    ----------
    url : str
        URL to load data from.
    delimiter : str, default="\t"
        Delimiter used in the data file.

    Returns
    -------
    numpy.ndarray
        Loaded data as a numpy array.
    """
    response = urllib.request.urlopen(url)
    data = response.read().decode("utf-8")
    data = data.split("\n")
    data = [row.split(delimiter) for row in data if row]
    numpy_array = np.array(data, dtype=float)
    return numpy_array


url_x = "https://rsinghlab.github.io/SCOT/data/snare_rna.txt"
snare_data = load_numpy_from_url(url_x) / 100

url_y = "https://rsinghlab.github.io/SCOT/data/SNAREseq_types.txt"
snare_labels = load_numpy_from_url(url_y)

# %%
# Computing TSNE and COSNE on SNARE-seq data
# -----------------------------------------
#
# We can now proceed to computing the two DR methods and visualizing
# the results on the SNARE-seq dataset.

tsne_model = TSNE(verbose=True, max_iter=500)
out_tsne = tsne_model.fit_transform(snare_data)

cosne_model = COSNE(lr=1e-1, verbose=True, gamma=0.5, lambda1=0.01, max_iter=500)
out_cosne = cosne_model.fit_transform(snare_data)


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
axes[0].scatter(*out_tsne.T, c=snare_labels.squeeze(1), cmap=plt.get_cmap("rainbow"))
axes[0].set_xticks([])
axes[0].set_yticks([])
axes[0].set_title("T-SNE", fontsize=24)
plot_disk(axes[1])
axes[1].scatter(*out_cosne.T, c=snare_labels.squeeze(1), cmap=plt.get_cmap("rainbow"))
axes[1].axis("off")
axes[1].set_title("CO-SNE", fontsize=24)
plt.show()
