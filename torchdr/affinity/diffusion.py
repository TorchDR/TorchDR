import torch
from torchdr.utils.kernel_approx import expm_multiply, compute_chebychev_coeff_all
from torchdr.affinity.base import Affinity

EPS = 1e-6

def norm_sym_laplacian(A: torch.Tensor):
    deg = A.sum(dim=1)
    deg_sqrt_inv = torch.diag(1.0 / torch.sqrt(deg + EPS))
    id = torch.eye(A.shape[0], device=A.device)
    return id - deg_sqrt_inv @ A @ deg_sqrt_inv


class Diffusion(Affinity):
    def __init__(self, graph_fn, t: float, order: int = 30, **kwargs):
        super().__init__(**kwargs)
        self.graph_fn = graph_fn
        self.t = t
        self.order = order

    def compute_heat_from_laplacian(self, L: torch.Tensor):
        n = L.shape[0]
        val = torch.linalg.eigvals(L).real
        max_eigval = val.max()
        cheb_coeff = compute_chebychev_coeff_all(
            0.5 * max_eigval, self.t, self.order
        )
        heat_kernel = expm_multiply(
            L, torch.eye(n, device=L.device), cheb_coeff, 0.5 * max_eigval
        )
        return heat_kernel
    
    def sym_clip(self, heat_kernel: torch.Tensor):
        heat_kernel = (heat_kernel + heat_kernel.T) / 2
        heat_kernel[heat_kernel < 0] = 0.0 + EPS
        return heat_kernel
    
    def _compute_affinity(self, X: torch.Tensor):
        adj = self.graph_fn(X)
        L = norm_sym_laplacian(adj)
        heat_kernel = self.compute_heat_from_laplacian(L)
        heat_kernel = self.sym_clip(heat_kernel)
        return heat_kernel
