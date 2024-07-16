r"""
TSNE and COSNE  via AffinityMatcher
===================================

We compare in this example two dimensionalty reduction methods:
T-SNE and CO-SNE on a synthetic hierarchical toy dataset The first
method computes an emebdding in a 2D Euclidean space while the
second one operates in the Hyperbolic Poincaré Ball model.


"""

# %%
# Desiging the synthetic hierarchical dataset
# ---------------
#
# We first construct a synthetic hierarchical dataset with the following class

from torchdr.utils.visu import plotGrid
from torchdr import TSNE, COSNE
from torchdr import pairwise_distances
import torch
import itertools
import matplotlib.pylab as plt
import geoopt


class SyntheticDataset(torch.utils.data.Dataset):
    '''
    Adopted from https://github.com/emilemathieu/pvae/

    Implementation of a synthetic dataset by hierarchical diffusion.
    Args:
    :param int dim: dimension of the input sample
    :param int depth: depth of the tree; the root corresponds to the depth 0
    :param int :numberOfChildren: Number of children of each node in the tree
    :param int :numberOfsiblings: Number of noisy observations obtained from the
    nodes of the tree
    :param float sigma_children: noise
    :param int param: integer by which :math:`\\sigma_children` is divided at each
    deeper level of the tree
    '''

    def __init__(self, ball, dim, depth, numberOfChildren=2, dist_children=1,
                 sigma_sibling=2, param=1, numberOfsiblings=1):
        assert numberOfChildren == 2
        self.dim = int(dim)
        self.ball = ball
        self.root = ball.origin(self.dim)
        self.sigma_sibling = sigma_sibling
        self.depth = int(depth)
        self.dist_children = dist_children
        self.numberOfChildren = int(numberOfChildren)
        self.numberOfsiblings = int(numberOfsiblings)
        self.__class_counter = itertools.count()
        self.origin_data, self.origin_labels, self.data, self.labels = map(
            torch.detach, self.bst())
        self.num_classes = self.origin_labels.max().item()+1

    def __len__(self):
        '''
        this method returns the total number of samples/nodes
        '''
        return len(self.data)

    def __getitem__(self, idx):
        '''
        Generates one sample
        '''
        data, labels = self.data[idx], self.labels[idx]
        return data, labels, labels.max(-1).values

    def get_children(self, parent_value, parent_label, current_depth, offspring=True):
        '''
        :param 1d-array parent_value
        :param 1d-array parent_label
        :param int current_depth
        :param  Boolean offspring:
        if True the parent node gives birth to numberOfChildren nodes
        if False the parent node gives birth to numberOfsiblings noisy observations
        :return: list of 2-tuples containing the value and label of each child of a
        parent node
        :rtype: list of length numberOfChildren
        '''
        if offspring:
            numberOfChildren = self.numberOfChildren
            sigma = self.dist_children
        else:
            numberOfChildren = self.numberOfsiblings
            sigma = self.sigma_sibling
        if offspring:
            direction = torch.randn_like(parent_value)
            parent_value_n = parent_value / parent_value.norm().clamp_min(1e-15)
            direction -= parent_value_n @ direction * parent_value_n
            child_value_1 = self.ball.geodesic_unit(
                torch.tensor(sigma), parent_value, direction)
            child_value_2 = self.ball.geodesic_unit(
                torch.tensor(sigma), parent_value, -direction)
            child_label_1 = parent_label.clone()
            child_label_1[current_depth] = next(self.__class_counter)
            child_label_2 = parent_label.clone()
            child_label_2[current_depth] = next(self.__class_counter)
            children = [
                (child_value_1, child_label_1),
                (child_value_2, child_label_2)
            ]
        else:
            children = []
            for i in range(numberOfChildren):
                child_value = self.ball.random(
                    self.dim, mean=parent_value, std=sigma ** .5)
                child_label = parent_label.clone()
                children.append((child_value, child_label))
        return children

    def bst(self):
        '''
        This method generates all the nodes of a level before going to the next level
        '''
        label = -torch.ones(self.depth+1, dtype=torch.long)
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
                    current_node, current_label, current_depth, False)
                for clone in clones:
                    values_clones.append(clone[0])
                    labels_clones.append(clone[1])
        length = int(((self.numberOfChildren) ** (self.depth + 1) - 1) /
                     (self.numberOfChildren - 1))
        images = torch.cat([i for i in visited]).reshape(length, self.dim)
        labels_visited = torch.cat([i for i in labels_visited]).reshape(
            length, self.depth+1)[:, :self.depth]
        values_clones = torch.cat([i for i in values_clones]).reshape(
            self.numberOfsiblings*length, self.dim)
        labels_clones = torch.cat([i for i in labels_clones]).reshape(
            self.numberOfsiblings*length, self.depth+1)
        return images, labels_visited, values_clones, labels_clones

# %%
# Generating the data
# -----------
#
# Let us now generate some data of interest. The dimension of the input space is
# set to 50


ball = geoopt.PoincareBall()

dataset = SyntheticDataset(ball, 50, 2, numberOfsiblings=100,
                           sigma_sibling=0.05, dist_children=.7)
HXs = dataset.data.double()
HXs = HXs - HXs.mean(axis=0)

ys = dataset.labels
colors = dataset.labels.max(-1).values

# %%
# Visualization of the original similarities
# -----------------------
#
# We can observe the hierarchical nature of the input data by examining the
# pairwaise distance matrix in the input space

D = pairwise_distances(HXs, HXs, metric='sqeuclidean')

plt.figure()
plt.imshow(D)
plt.title("Distance matrix in the input space")
plt.show()


# %%
# Computing TSNE and COSNE
# -----------------------
#
# We can now proceed to computing the two DR methods and visualizing
# the results


myTSNE = TSNE(lr=1e-1, coeff_attraction=1, verbose=True)
out_tsne = myTSNE.fit_transform(HXs)

myCOSNE = COSNE(lr=1e-1, verbose=True, optimizer='RAdam', gamma=.5, lambda1=0.1,
                metric_out='sqhyperbolic')
out_cosne = myCOSNE.fit_transform(HXs)


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
axes[0].scatter(*out_tsne.T, c=colors, cmap=plt.get_cmap("rainbow"))
axes[0].set_xticks([])
axes[0].set_yticks([])
axes[0].set_title('T-SNE', fontsize=24)
plotGrid(axes[1])
axes[1].scatter(*out_cosne.T, c=colors, cmap=plt.get_cmap("rainbow"))
axes[1].axis('off')
axes[1].set_title('CO-SNE', fontsize=24)
plt.show()