from torchdr.affinity.unnormalized import NegativeCostAffinity
from torchdr.affinity.knn_normalized import NegPotentialAffinity
from torchdr.affinity_matcher import AffinityMatcher

class PHATE(AffinityMatcher):
    def __init__(self, n_neighbors:int = 10, n_components: int = 2, t:int = 5, eps: float = 1e-5, keops: bool = False, device:str="cpu", **kwargs):
        r"""Implementation of PHATE introduced in [?]_.

        PHATE is a diffusion map-based method that uses a potential distance
        matrix to embed the data.

        Parameters
        ----------
        n_neighbors : int
            Number of nearest neighbors.
        n_components : int, optional
            Dimension of the embedding space.
        a : float, optional
            Parameter for the Student t-distribution.
        b : float, optional
            Parameter for the Student t-distribution.
        eps : float, optional
            Small value to avoid numerical issues.
        keops : bool, optional
            Whether to use KeOps for computations
        """
        affinity_in = NegPotentialAffinity(keops=keops, device=device, t=t, eps=eps, K=n_neighbors)    
        affinity_out = NegativeCostAffinity(keops=keops, device=device)
        loss_fn = "l2_loss"
        init_scaling = 1.0
        init = "pca"
        super().__init__(affinity_in=affinity_in, affinity_out=affinity_out, n_components=n_components, loss_fn=loss_fn, init_scaling=init_scaling, init=init, **kwargs)
