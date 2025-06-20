from typing import Optional

from torchdr.affinity.unnormalized import NegativeCostAffinity, NegPotentialAffinity
from torchdr.affinity_matcher import AffinityMatcher


class PHATE(AffinityMatcher):
    r"""Implementation of PHATE introduced in :cite:`moon2019visualizing`.

    PHATE is a diffusion map-based method that uses a potential distance
    matrix to embed the data.

    Parameters
    ----------
    n_neighbors : int
        Number of nearest neighbors.
    n_components : int, optional
        Dimension of the embedding space.
    t : int
        Diffusion time parameter.
    eps : float, optional
        Small value to avoid division by zero in the affinity matrix.
        Default is 1e-5.
    backend : {"keops", "faiss", None}, optional
        Which backend to use for handling sparsity and memory efficiency.
        Default is None.
    device : str
        Device to use for computations. Default is "cpu".
    """

    def __init__(
        self,
        n_neighbors: int = 10,
        n_components: int = 2,
        t: int = 5,
        eps: float = 1e-5,
        backend: Optional[str] = None,
        device: str = "cpu",
        **kwargs,
    ):
        affinity_in = NegPotentialAffinity(
            backend=backend, device=device, eps=eps, t=t, K=n_neighbors
        )
        affinity_out = NegativeCostAffinity(backend=backend, device=device)
        loss_fn = "l2_loss"
        init_scaling = 1.0
        init = "pca"
        super().__init__(
            affinity_in=affinity_in,
            affinity_out=affinity_out,
            n_components=n_components,
            loss_fn=loss_fn,
            init_scaling=init_scaling,
            init=init,
            **kwargs,
        )
