"""Arithmetic mixtures of directed entropic affinities."""

# License: BSD 3-Clause License

import math
from numbers import Real

import torch
import torch.nn.functional as F

from torchdr.affinity.entropic import EntropicAffinity
from torchdr.utils import check_neighbor_param


class MultiscaleEntropicAffinity(EntropicAffinity):
    r"""Average directed entropic affinities at several perplexities.

    This is the explicit-scale arithmetic mixture motivated by
    :cite:`lee2015multiscale`: solve one normalized Gaussian neighborhood per
    perplexity, then average the resulting probabilities, not the perplexities
    or log probabilities. Each row sums to ``1 / n_samples`` and total mass is
    one. The mixture is directed and has no single prescribed row entropy.
    This class is an affinity component, not a complete multi-scale SNE method
    or an automatic selection of all scales from the paper.

    Sparse scales retain their own ``3 * perplexity`` neighbors within a single
    maximum-k search. Missing entries contribute zero probability. Ties at a
    scale's cutoff follow the shared search's ordering, which may differ from
    separate k-NN searches. Dense mode averages full distance-based affinities.

    Parameters
    ----------
    perplexities : sequence of int, optional
        Nonempty sequence of finite integer-valued perplexities >= 2.
        Default is ``(5, 30)``. Each scale is clamped to the input's valid
        neighbor range by the same rule as :class:`torchdr.EntropicAffinity`.
        Repeated scales retain their multiplicity in the average.
    max_iter : int, optional
        Maximum root-search iterations for each scale. Default is 1000.
    sparsity : bool, optional
        Use sparse neighborhoods rather than the full matrix. Default is True.
    metric : str, optional
        Pairwise distance metric. Default is ``"sqeuclidean"``.
    zero_diag : bool, optional
        Exclude self-neighbors. Default is True.
    device : str, optional
        Compute device; ``"auto"`` follows the input. Default is ``"auto"``.
    backend : {None, "faiss"} or FaissConfig, optional
        Tensor distance backend. Default is None (PyTorch). FAISS requires sparse
        mode. KeOps mixtures are
        not implemented; requesting that backend raises ValueError.
    verbose : bool, optional
        Enable logging. Default is False.
    compile : bool, optional
        Must be False. Compiled mixtures are not implemented.
    distributed : bool or 'auto', optional
        Forwarded to the sparse distance backend. Default is ``"auto"``.
    _pre_processed : bool, optional
        Skip input conversion for internal callers. Default is False.

    Attributes
    ----------
    eps_ : torch.Tensor of shape (n_scales, n_local_rows)
        Fitted bandwidths, one per scale and data row.
    log_normalization_ : torch.Tensor of shape (n_scales, n_local_rows, 1)
        Per-scale log normalizers before the final total-mass normalization.

    Notes
    -----
    The inherited call returns ``(affinity, indices)`` by default. Use
    ``return_indices=False`` for only the matrix, and ``log=True`` for its log.
    Sparse output has shape ``(n_local_rows, k_max)``; dense output has shape
    ``(n_samples, n_samples)`` on one device. ``k_max`` is the largest clamped
    neighborhood size. As with EntropicAffinity, no symmetrization is applied.
    """

    def __init__(
        self,
        perplexities=(5, 30),
        max_iter=1000,
        sparsity=True,
        metric="sqeuclidean",
        zero_diag=True,
        device="auto",
        backend=None,
        verbose=False,
        compile=False,
        distributed="auto",
        _pre_processed=False,
    ):
        try:
            scales = tuple(perplexities)
        except TypeError as exc:
            raise ValueError("perplexities must be a nonempty sequence") from exc
        if not scales or any(
            not isinstance(p, Real)
            or isinstance(p, bool)
            or not math.isfinite(p)
            or p < 2
            or int(p) != p
            for p in scales
        ):
            raise ValueError("perplexities must contain finite integers >= 2")
        if compile:
            raise ValueError("MultiscaleEntropicAffinity does not support compile=True")
        if backend == "keops":
            raise ValueError(
                "MultiscaleEntropicAffinity does not support the KeOps backend"
            )
        self.perplexities = scales
        super().__init__(
            perplexity=max(scales),
            max_iter=max_iter,
            sparsity=sparsity,
            metric=metric,
            zero_diag=zero_diag,
            device=device,
            backend=backend,
            verbose=verbose,
            compile=False,
            distributed=distributed,
            _pre_processed=_pre_processed,
        )

    def _compute_sparse_log_affinity(self, X, return_indices=True, **kwargs):
        if self.compile:
            raise ValueError("MultiscaleEntropicAffinity does not support compile=True")
        n = self._get_n_samples(X)
        if n < 4:
            raise ValueError("MultiscaleEntropicAffinity requires at least 4 samples")
        device = self._get_compute_device(X)
        dtype = self._get_dtype(X)
        n_tensor = torch.tensor(n, device=device, dtype=dtype)
        scales = [check_neighbor_param(p, n_tensor) for p in self.perplexities]
        if self.sparsity:
            counts = [check_neighbor_param(3 * p, n_tensor) for p in scales]
            C, indices = self._distance_matrix(X, k=max(counts), return_indices=True)
        else:
            counts = [None] * len(scales)
            C, indices = self._distance_matrix(X, return_indices=True)
        log_mixture = None
        bandwidths = []
        normalizations = []
        for perplexity, k in zip(scales, counts):
            distances = C if k is None else C[:, :k]
            log_P = self._solve_entropic_affinity(
                distances, perplexity, n, device, dtype
            )
            bandwidths.append(self.eps_)
            normalizations.append(self.log_normalization_)
            if self.sparsity:
                log_P = F.pad(log_P, (0, C.shape[1] - log_P.shape[1]), value=-torch.inf)
            log_mixture = (
                log_P if log_mixture is None else torch.logaddexp(log_mixture, log_P)
            )
        log_mixture = log_mixture - math.log(len(scales))
        self.register_buffer("eps_", torch.stack(bandwidths), persistent=False)
        self.register_buffer(
            "log_normalization_", torch.stack(normalizations), persistent=False
        )
        return (log_mixture, indices) if return_indices else log_mixture
