r"""
TSNE vs COSNE : Euclidean vs Hyperbolic
=======================================

We compare in this example two dimensionalty reduction methods:
T-SNE and CO-SNE on a synthetic hierarchical toy dataset and on
singlecell data. The first method computes an embedding in a 2D
Euclidean space while the second one operates in the Hyperbolic
Poincaré Ball model.
"""

# %%
# Designing the synthetic hierarchical dataset
# ---------------
#
# We first construct a synthetic hierarchical dataset with the following class

import numpy as np
from torchdr.utils.visu import plotGrid
from torchdr import TSNE, COSNE
from torchdr import pairwise_distances
import torch
import itertools
import urllib.request
import matplotlib.pylab as plt
from torchdr.utils import geoopt


class SyntheticDataset(torch.utils.data.Dataset):
    """
    Implementation of a synthetic dataset by hierarchical diffusion.

    Adopted from https://github.com/emilemathieu/pvae/

    Parameters
    ----------
    ball : torchdr.utils.geoopt.PoincareBall
        The Poincaré ball used for generating the dataset.
    dim : int
        Dimension of the input sample.
    depth : int
        Depth of the tree; the root corresponds to the depth 0.
    num_children : int
        Number of children of each node in the tree.
    dist_children : float
        Distance parameter for children nodes.
    sigma_sibling : float
        Noise parameter for sibling nodes.
    num_siblings : int
        Number of noisy observations obtained from the nodes of the tree.
    """

    def __init__(
        self,
        ball,
        dim,
        depth,
        num_children=2,
        dist_children=1,
        sigma_sibling=2,
        num_siblings=1,
    ):
        assert num_children == 2
        self.dim = int(dim)
        self.ball = ball
        self.root = ball.origin(self.dim)
        self.sigma_sibling = sigma_sibling
        self.depth = int(depth)
        self.dist_children = dist_children
        self.num_children = int(num_children)
        self.num_siblings = int(num_siblings)
        self.__class_counter = itertools.count()
        self.origin_data, self.origin_labels, self.data, self.labels = map(
            torch.detach, self.bst()
        )
        self.num_classes = self.origin_labels.max().item() + 1

    def __len__(self):
        """
        Return the total number of samples/nodes.

        Returns
        -------
        int
            Number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Generate one sample.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        tuple
            Contains (data, labels, max_label) for the requested index.
        """
        data, labels = self.data[idx], self.labels[idx]
        return data, labels, labels.max(-1).values

    def get_children(self, parent_value, parent_label, current_depth, offspring=True):
        """
        Generate children nodes or noisy observations from a parent node.

        Parameters
        ----------
        parent_value : torch.Tensor
            1D array representing the parent node value.
        parent_label : torch.Tensor
            1D array representing the parent node label.
        current_depth : int
            Current depth in the tree.
        offspring : bool, default=True
            If True, the parent node gives birth to num_children nodes.
            If False, the parent node gives birth to num_siblings noisy observations.

        Returns
        -------
        list
            List of 2-tuples containing the value and label of each child of the
            parent node. Length depends on offspring parameter.
        """
        if offspring:
            num_children = self.num_children
            sigma = self.dist_children
        else:
            num_children = self.num_siblings
            sigma = self.sigma_sibling
        if offspring:
            direction = torch.randn_like(parent_value)
            parent_value_n = parent_value / parent_value.norm().clamp_min(1e-15)
            direction -= parent_value_n @ direction * parent_value_n
            child_value_1 = self.ball.geodesic_unit(
                torch.tensor(sigma), parent_value, direction
            )
            child_value_2 = self.ball.geodesic_unit(
                torch.tensor(sigma), parent_value, -direction
            )
            child_label_1 = parent_label.clone()
            child_label_1[current_depth] = next(self.__class_counter)
            child_label_2 = parent_label.clone()
            child_label_2[current_depth] = next(self.__class_counter)
            children = [(child_value_1, child_label_1), (child_value_2, child_label_2)]
        else:
            children = []
            for i in range(num_children):
                child_value = self.ball.random(
                    self.dim, mean=parent_value, std=sigma**0.5
                )
                child_label = parent_label.clone()
                children.append((child_value, child_label))
        return children

    def bst(self):
        """
        Generate all nodes of a level before proceeding to the next level.

        This method builds the hierarchical tree structure level by level.

        Returns
        -------
        tuple
            Contains (images, labels_visited, values_clones, labels_clones)
            representing the original data points, their labels, and the
            noisy observations with their labels.
        """
        label = -torch.ones(self.depth + 1, dtype=torch.long)
        label[0] = next(self.__class_counter)
        queue = [(self.root, label, 0)]
        visited = []
        labels_visited = []
        values_clones = []
        labels_clones = []
        while len(queue) > 0:
            current_node, current_label, current_depth = queue.pop(0)
            visited.append(current_node)
            labels_visited.append(current_label)
            if current_depth < self.depth:
                children = self.get_children(current_node, current_label, current_depth)
                for child in children:
                    queue.append((child[0], child[1], current_depth + 1))
            if current_depth <= self.depth:
                clones = self.get_children(
                    current_node, current_label, current_depth, False
                )
                for clone in clones:
                    values_clones.append(clone[0])
                    labels_clones.append(clone[1])
        length = int(
            ((self.num_children) ** (self.depth + 1) - 1) / (self.num_children - 1)
        )
        images = torch.cat([i for i in visited]).reshape(length, self.dim)
        labels_visited = torch.cat([i for i in labels_visited]).reshape(
            length, self.depth + 1
        )[:, : self.depth]
        values_clones = torch.cat([i for i in values_clones]).reshape(
            self.num_siblings * length, self.dim
        )
        labels_clones = torch.cat([i for i in labels_clones]).reshape(
            self.num_siblings * length, self.depth + 1
        )
        return images, labels_visited, values_clones, labels_clones


# %%
# Generating the data
# -----------
#
# Let us now generate some data of interest. The dimension of the input space is
# set to 50


ball = geoopt.PoincareBall()

dataset = SyntheticDataset(
    ball, 50, 2, num_siblings=100, sigma_sibling=0.05, dist_children=0.7
)
data_points = dataset.data
data_points = data_points - data_points.mean(axis=0)

labels = dataset.labels
colors = dataset.labels.max(-1).values

# %%
# Visualization of the original similarities
# -----------------------
#
# We can observe the hierarchical nature of the input data by examining the
# pairwaise distance matrix in the input space

dist_matrix, _ = pairwise_distances(data_points, data_points, metric="sqeuclidean")

plt.figure()
plt.imshow(dist_matrix)
plt.title("Distance matrix in the input space")
plt.show()


# %%
# Computing TSNE and COSNE
# -----------------------
#
# We can now proceed to computing the two DR methods and visualizing
# the results


tsne_model = TSNE(verbose=True, max_iter=500)
out_tsne = tsne_model.fit_transform(data_points)

cosne_model = COSNE(lr=1e-1, verbose=True, gamma=0.5, lambda1=0.01, max_iter=500)
out_cosne = cosne_model.fit_transform(data_points)


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
axes[0].scatter(*out_tsne.T, c=colors, cmap=plt.get_cmap("rainbow"))
axes[0].set_xticks([])
axes[0].set_yticks([])
axes[0].set_title("T-SNE", fontsize=24)
plotGrid(axes[1])
axes[1].scatter(*out_cosne.T, c=colors, cmap=plt.get_cmap("rainbow"))
axes[1].axis("off")
axes[1].set_title("CO-SNE", fontsize=24)
plt.show()


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
# the results on the SNARE-seq dataset

tsne_model = TSNE(verbose=True, max_iter=500)
out_tsne = tsne_model.fit_transform(snare_data)

cosne_model = COSNE(lr=1e-1, verbose=True, gamma=0.5, lambda1=0.01, max_iter=500)
out_cosne = cosne_model.fit_transform(snare_data)


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
axes[0].scatter(*out_tsne.T, c=snare_labels.squeeze(1), cmap=plt.get_cmap("rainbow"))
axes[0].set_xticks([])
axes[0].set_yticks([])
axes[0].set_title("T-SNE", fontsize=24)
plotGrid(axes[1])
axes[1].scatter(*out_cosne.T, c=snare_labels.squeeze(1), cmap=plt.get_cmap("rainbow"))
axes[1].axis("off")
axes[1].set_title("CO-SNE", fontsize=24)
plt.show()
