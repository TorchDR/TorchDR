"""PHATE algorithm."""

# Author: Guillaume Huguet @guillaumehu
#         Danqi Liao @Danqi7
#         Hugues Van Assel @huguesva
#         Matthew Scicluna <mattcscicluna@gmail.com>
#
# License: BSD 3-Clause License

import math
import time
import warnings
from typing import Optional, Union

import torch
import numpy as np
from torch.utils.data import DataLoader

from torchdr.affinity import PHATEAffinity
from torchdr.affinity_matcher import AffinityMatcher
from torchdr.utils import check_NaNs, to_torch
from torchdr.distance import FaissConfig


class PHATE(AffinityMatcher):
    r"""Implementation of PHATE introduced in :cite:`moon2019visualizing`.

    PHATE is a diffusion map-based method that uses a potential affinity matrix :math:`\mathbf{P}` implemented in :class:`~torchdr.PHATEAffinity` as input.

    The loss function is defined as:

    .. math::

        \sqrt{\sum_{i,j} (P_{ij} - \|\mathbf{z}_i - \mathbf{z}_j\|)^2 / \sum_{i,j} P_{ij}^2} \:.

    Parameters
    ----------
    k : int, optional
        Number of nearest neighbors. Default is 5.
    n_components : int, optional
        Dimension of the embedding space. Default is 2.
    t : int, optional
        Diffusion time parameter. Default is 100.
    alpha : float, optional
        Exponent for the alpha-decay kernel. Default is 10.0.
    decay : float, optional
        Alias for ``alpha`` for compatibility with the CPU PHATE API.
        If provided, overrides ``alpha``.
    knn_max : int or None, optional
        Maximum number of neighbors queried when expanding thresholded
        alpha-decay neighborhoods. If None, expansion can grow up to
        ``n_samples - 1``.
    thresh : float or None, optional
        Threshold used to keep alpha-decay affinities in the pre-diffusion
        kernel. Default is 1e-4. If None, keeps strict kNN-only entries.
    backend : {"faiss", "keops", None} or FaissConfig, optional
        Backend used to build a sparse kNN kernel before diffusion:
        - None: PyTorch backend (current default)
        - "faiss": reserved for future FAISS support
        - "keops": reserved for future KeOps support
        - FaissConfig: kNN with a custom FAISS configuration
    n_landmarks : int or None, optional
        Number of landmarks for landmark PHATE. If None, landmarking is disabled.
    random_landmarking : bool, optional
        If True, use random landmark assignment; otherwise use spectral landmarking.
    max_iter : int, optional
        Maximum number of iterations. Default is 1000.
    init : str, torch.Tensor, or np.ndarray, optional
        Initialization method for the embedding. Default is "pca".
    init_scaling : float, optional
        Scaling factor for the initial embedding. Default is 1e-4.
    device : str, optional
        Device to use for computations. Default is "auto".
    verbose : bool, optional
        Verbosity of the optimization process. Default is False.
    random_state : float, optional
        Random seed for reproducibility. Default is None.
    check_interval : int, optional
        Number of iterations between two checks for convergence. Default is 50.
    mds_solver : {"sgd"}, optional
        Solver used for the PHATE MDS optimization step.
        Only ``"sgd"`` is supported.
    pairs_per_iter : int or None, optional
        Number of pairs sampled per iteration when ``mds_solver="sgd"``.
        If None, uses ``2 * n_samples * log(n_samples)``.
    sgd_learning_rate : float, optional
        Base learning rate for ``mds_solver="sgd"``. Default is 1e-3.
    sgd_stress_tol : float, optional
        Relative stress tolerance for early stopping when ``mds_solver="sgd"``.
        Default is 1e-6.
    """  # noqa: E501

    def __init__(
        self,
        k: int = 5,
        n_components: int = 2,
        t: int = 100,
        alpha: float = 10.0,
        decay: Optional[float] = None,
        knn_max: Optional[int] = None,
        thresh: Optional[float] = 1e-4,
        backend: Union[str, FaissConfig, None] = None,
        n_landmarks: Optional[int] = None,
        random_landmarking: bool = False,
        max_iter: int = 1000,
        init: str = "pca",
        init_scaling: float = 1e-4,
        device: str = "auto",
        verbose: bool = False,
        random_state: Optional[float] = None,
        check_interval: int = 50,
        metric_in: str = "euclidean",
        mds_solver: str = "sgd",
        pairs_per_iter: Optional[int] = None,
        sgd_learning_rate: float = 1e-3,
        sgd_stress_tol: float = 1e-6,
    ):
        if isinstance(backend, FaissConfig) or backend in {"faiss", "keops"}:
            raise ValueError(
                f"[TorchDR] ERROR : backend={backend} is not implemented yet for PHATE."
            )

        self.metric_in = metric_in
        self.k = k
        self.t = t
        if decay is not None:
            alpha = decay
        self.alpha = alpha
        self.knn_max = knn_max
        self.thresh = thresh
        self.knn_backend = backend
        self.n_landmarks = n_landmarks
        self.random_landmarking = random_landmarking
        self.mds_solver = mds_solver
        self.pairs_per_iter = pairs_per_iter
        self.sgd_learning_rate = sgd_learning_rate
        self.sgd_stress_tol = sgd_stress_tol

        if self.mds_solver != "sgd":
            raise ValueError(
                f"[TorchDR] ERROR : mds_solver must be 'sgd'. Got {self.mds_solver}."
            )
        if self.pairs_per_iter is not None and self.pairs_per_iter <= 0:
            raise ValueError(
                f"[TorchDR] ERROR : pairs_per_iter must be positive or None. Got {self.pairs_per_iter}."
            )
        if self.sgd_learning_rate <= 0:
            raise ValueError(
                f"[TorchDR] ERROR : sgd_learning_rate must be positive. Got {self.sgd_learning_rate}."
            )
        if self.sgd_stress_tol <= 0:
            raise ValueError(
                f"[TorchDR] ERROR : sgd_stress_tol must be positive. Got {self.sgd_stress_tol}."
            )
        if self.n_landmarks is not None and self.n_landmarks <= 1:
            raise ValueError(
                "[TorchDR] ERROR : n_landmarks must be > 1 or None. "
                f"Got {self.n_landmarks}."
            )
        if self.random_landmarking and self.n_landmarks is None:
            warnings.warn(
                "random_landmarking=True has no effect when n_landmarks=None.",
                RuntimeWarning,
            )

        affinity_in = PHATEAffinity(
            k=k,
            t=t,
            alpha=alpha,
            knn_max=knn_max,
            thresh=thresh,
            backend=backend,
            n_landmarks=n_landmarks,
            random_landmarking=random_landmarking,
            metric=metric_in,
            device=device,
            verbose=verbose,
            random_state=random_state,
        )
        super().__init__(
            affinity_in=affinity_in,
            affinity_out=None,
            n_components=n_components,
            optimizer="Adam",
            optimizer_kwargs=None,
            lr=1e0,
            scheduler=None,
            scheduler_kwargs=None,
            min_grad_norm=1e-7,
            max_iter=max_iter,
            init=init,
            init_scaling=init_scaling,
            device=device,
            backend=None,
            verbose=verbose,
            random_state=random_state,
            check_interval=check_interval,
        )

    def _fit_transform(self, X, y=None):
        return self._fit_transform_sgd(X, y=y)

    def _fit_transform_sgd(self, X, y=None):
        # Sampled-pair SGD-MDS-style solver specialized for PHATE.
        t_total = time.perf_counter()
        if isinstance(X, DataLoader):
            raise NotImplementedError(
                "[TorchDR] ERROR : PHATE with mds_solver='sgd' does not support "
                "DataLoader input. Pass a tensor/ndarray instead."
            )
        if not isinstance(X, torch.Tensor):
            raise TypeError(
                "[TorchDR] ERROR : PHATE expects a torch.Tensor input in "
                f"_fit_transform_sgd. Got type={type(X).__name__}."
            )
        if X.ndim != 2:
            raise ValueError(
                f"[TorchDR] ERROR : expected 2D input tensor, got shape={tuple(X.shape)}."
            )

        self.n_samples_in_, self.n_features_in_ = X.shape
        self.device_ = X.device if self.device == "auto" else self.device
        if X.device != self.device_:
            X = X.to(self.device_)

        t0 = time.perf_counter()
        self.on_affinity_computation_start()
        affinity_matrix = self.affinity_in(X)
        if self.affinity_in.backend == "keops":
            self.affinity_in_ = affinity_matrix
        else:
            self.register_buffer("affinity_in_", affinity_matrix, persistent=False)
        self.on_affinity_computation_end()
        t_affinity = time.perf_counter() - t0

        # Target metric distances are stored as negative affinities.
        target_dist = (-self.affinity_in_).clamp_min(0.0)
        d_max = target_dist.max()
        if d_max > 0:
            target_dist = target_dist / d_max
        target_dist = target_dist.to(X.dtype)
        n = target_dist.shape[0]
        if n < 2:
            return target_dist.new_zeros((n, self.n_components))

        t0 = time.perf_counter()
        self._init_embedding_sgd(target_dist)
        t_init = time.perf_counter() - t0

        n = self.embedding_.shape[0]

        if self.pairs_per_iter is None:
            pairs_per_iter = max(16, int(2 * n * math.log(max(n, 2))))
        else:
            pairs_per_iter = min(self.pairs_per_iter, n * n)

        total_pairs = max(n * (n - 1) / 2, 1.0)
        sampling_ratio = max(pairs_per_iter / total_pairs, 1e-12)
        eta_max = float(self.sgd_learning_rate) * math.sqrt(1.0 / sampling_ratio)
        eta_min = eta_max * 0.01
        decay = math.log(eta_max / eta_min) / max(self.max_iter - 1, 1)

        prev_stress = None
        t_opt = time.perf_counter()
        for step in range(self.max_iter):
            self.n_iter_.fill_(step)
            lr_step = eta_max * math.exp(-decay * step)

            i = torch.randint(0, n, (pairs_per_iter,), device=self.embedding_.device)
            j = torch.randint(0, n, (pairs_per_iter,), device=self.embedding_.device)
            valid = i != j
            i = i[valid]
            j = j[valid]
            if i.numel() == 0:
                continue

            y_i = self.embedding_[i]
            y_j = self.embedding_[j]
            diff = y_i - y_j
            dist = diff.norm(dim=1).clamp_min(1e-10)
            target = target_dist[i, j]
            errors = target - dist

            weights = -2.0 * errors / dist
            grad_contrib = diff * weights[:, None]
            gradients = torch.zeros_like(self.embedding_)
            gradients.index_add_(0, i, grad_contrib)
            gradients.index_add_(0, j, -grad_contrib)

            with torch.no_grad():
                grad_norm = gradients.norm(2)
                if torch.isfinite(grad_norm) and grad_norm > 1e6:
                    gradients.mul_(1e6 / grad_norm)
                self.embedding_.sub_(lr_step * gradients)

            check_NaNs(
                self.embedding_,
                msg="[TorchDR] ERROR PHATE (sgd) : NaNs in the embeddings "
                f"at iter {step}.",
            )

            stress = (errors * errors).mean()
            if self.verbose and step % self.check_interval == 0:
                self.logger.info(
                    f"[{step}/{self.max_iter}] Stress: {stress.item():.2e} | "
                    f"Grad norm: {grad_norm.item():.2e} | LR: {lr_step:.2e}"
                )

            if prev_stress is not None and step > 50:
                rel_change = torch.abs(stress - prev_stress) / (prev_stress + 1e-10)
                if rel_change.item() < self.sgd_stress_tol:
                    if self.verbose:
                        self.logger.info(
                            f"Convergence reached at iter {step} with relative stress change "
                            f"{rel_change.item():.2e}."
                        )
                    break
            prev_stress = stress

        t_opt = time.perf_counter() - t_opt
        if d_max > 0:
            with torch.no_grad():
                self.embedding_.mul_(d_max)

        transitions = getattr(self.affinity_in, "transitions_", None)
        if transitions is not None:
            if self.verbose:
                self.logger.info(
                    "Interpolating landmark embedding back to full data "
                    f"(n={transitions.shape[0]}, n_landmarks={transitions.shape[1]})."
                )
            transitions = transitions.to(
                device=self.embedding_.device, dtype=self.embedding_.dtype
            )
            self.landmark_embedding_ = self.embedding_
            self.embedding_ = transitions @ self.landmark_embedding_

        if self.verbose:
            self.logger.info(
                "PHATE(sgd) timing (s): affinity=%.2f, init=%.2f, optimize=%.2f, total=%.2f",
                t_affinity,
                t_init,
                t_opt,
                time.perf_counter() - t_total,
            )

        self.clear_memory()
        return self.embedding_

    def _init_embedding_sgd(self, target_dist: torch.Tensor):
        n = target_dist.shape[0]
        dtype = target_dist.dtype
        device = target_dist.device

        if isinstance(self.init, (torch.Tensor, np.ndarray)):
            embedding_ = to_torch(self.init).to(device=device, dtype=dtype)
        elif self.init in ("normal", "random"):
            embedding_ = torch.randn((n, self.n_components), device=device, dtype=dtype)
        elif self.init == "pca":
            # For SGD-MDS, mimic CPU PHATE by using classical-MDS-style init from
            # target distances (not PCA on raw high-dimensional input).
            if self.verbose:
                self.logger.info(
                    "Initializing SGD-MDS with classical MDS (randomized SVD)."
                )
            try:
                D2 = target_dist.to(torch.float64) ** 2
                row_mean = D2.mean(dim=1, keepdim=True)
                col_mean = D2.mean(dim=0, keepdim=True)
                grand_mean = D2.mean()
                B = -0.5 * (D2 - row_mean - col_mean + grand_mean)

                q = min(max(self.n_components + 2, 4), max(2, n - 1))
                U, S, _ = torch.pca_lowrank(B, q=q, center=False, niter=2)
                vals = S[: self.n_components].clamp_min(0).sqrt()
                embedding_ = (U[:, : self.n_components] * vals.unsqueeze(0)).to(
                    device=device, dtype=dtype
                )
            except RuntimeError as err:
                warnings.warn(
                    f"Classical-MDS init failed ({err}). Falling back to random init.",
                    RuntimeWarning,
                )
                embedding_ = torch.randn(
                    (n, self.n_components), device=device, dtype=dtype
                )
        else:
            raise ValueError(
                f"[TorchDR] ERROR : init {self.init} not supported in {self.__class__.__name__}."
            )

        with torch.no_grad():
            std = embedding_[:, 0].std().clamp_min(1e-12)
            # CPU SGD-MDS uses unit-scale initialization; avoid tiny Adam-style
            # defaults (1e-4) for SGD-MDS dynamics.
            init_scale = 1.0 if self.init_scaling == 1e-4 else float(self.init_scaling)
            embedding_ = init_scale * embedding_ / std

        self.embedding_ = embedding_.requires_grad_(True)
        return self.embedding_
