from typing import Optional

from torchdr.affinity import NegativeCostAffinity, PHATEAffinity
from torchdr.affinity_matcher import AffinityMatcher
from torchdr.utils import square_loss


class PHATE(AffinityMatcher):
    r"""Implementation of PHATE introduced in :cite:`moon2019visualizing`.

    PHATE is a diffusion map-based method that uses a potential distance
    matrix to embed the data.

    Parameters
    ----------
    k : int, optional
        Number of nearest neighbors. Default is 5.
    n_components : int, optional
        Dimension of the embedding space. Default is 2.
    t : int, optional
        Diffusion time parameter. Default is 5.
    alpha : float, optional
        Exponent for the alpha-decay kernel. Default is 10.0.
    backend : {"keops", None}, optional
        Which backend to use for handling sparsity and memory efficiency.
        Default is None.
    optimizer : str or torch.optim.Optimizer, optional
        Name of an optimizer from torch.optim or an optimizer class.
        Default is "Adam".
    optimizer_kwargs : dict, optional
        Additional keyword arguments for the optimizer.
    lr : float or 'auto', optional
        Learning rate for the optimizer. Default is 1e0.
    scheduler : str or torch.optim.lr_scheduler.LRScheduler, optional
        Name of a scheduler from torch.optim.lr_scheduler or a scheduler class.
        Default is None (no scheduler).
    scheduler_kwargs : dict, optional
        Additional keyword arguments for the scheduler.
    min_grad_norm : float, optional
        Tolerance for stopping criterion. Default is 1e-7.
    max_iter : int, optional
        Maximum number of iterations. Default is 1000.
    init : str, torch.Tensor, or np.ndarray, optional
        Initialization method for the embedding. Default is "pca".
    init_scaling : float, optional
        Scaling factor for the initial embedding. Default is 1e-4.
    device : str, optional
        Device to use for computations. Default is "auto".
    backend : {"keops", None}, optional
        Which backend to use for handling sparsity and memory efficiency.
        Default is None.
    verbose : bool, optional
        Verbosity of the optimization process. Default is False.
    random_state : float, optional
        Random seed for reproducibility. Default is None.
    check_interval : int, optional
        Number of iterations between two checks for convergence. Default is 50.
    """  # noqa: E501

    def __init__(
        self,
        k: int = 5,
        n_components: int = 2,
        t: int = 5,
        alpha: float = 10.0,
        optimizer: str = "Adam",
        optimizer_kwargs: dict = {},
        lr: float = 1e0,
        scheduler: Optional[str] = None,
        scheduler_kwargs: dict = {},
        min_grad_norm: float = 1e-7,
        max_iter: int = 1000,
        init: str = "pca",
        init_scaling: float = 1.0,
        device: str = "auto",
        backend: Optional[str] = None,
        verbose: bool = False,
        random_state: Optional[float] = None,
        check_interval: int = 50,
        metric_in: str = "euclidean",
        metric_out: str = "euclidean",
    ):
        if backend == "faiss" or backend == "keops":
            raise ValueError(
                f"[TorchDR] ERROR : {self.__class__.__name__} class does not support backend {backend}."
            )

        self.metric_in = metric_in
        self.metric_out = metric_out
        self.k = k
        self.t = t
        self.alpha = alpha

        affinity_in = PHATEAffinity(
            k=k,
            t=t,
            alpha=alpha,
            metric=metric_in,
            backend=backend,
            device=device,
        )
        affinity_out = NegativeCostAffinity(
            backend=backend, device=device, metric=metric_out
        )
        super().__init__(
            affinity_in=affinity_in,
            affinity_out=affinity_out,
            n_components=n_components,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr=lr,
            scheduler=scheduler,
            scheduler_kwargs=scheduler_kwargs,
            min_grad_norm=min_grad_norm,
            max_iter=max_iter,
            init=init,
            init_scaling=init_scaling,
            device=device,
            backend=backend,
            verbose=verbose,
            random_state=random_state,
            check_interval=check_interval,
        )

    def _loss(self):
        Q = self.affinity_out(self.embedding_)
        loss = square_loss(self.PX_, Q)
        loss = loss / (self.PX_**2).sum()
        return loss.sqrt()
