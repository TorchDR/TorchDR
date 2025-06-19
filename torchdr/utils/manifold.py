# -*- coding: utf-8 -*-
"""Manifold classes for Riemannian optimization."""

# Author: Nicolas Courty <ncourty@irisa.fr>
# adapted from geoopt library
# License: BSD 3-Clause License

from torch.nn import Parameter

import torch


class Manifold:
    """Abstract class to define operations on a manifold."""

    def __init__(self):
        super().__init__()
        self.eps = 10e-8

    def sqdist(self, p1, p2, c):
        """Squared distance between pairs of points."""
        raise NotImplementedError

    def egrad2rgrad(self, p, dp, c):
        """Converts Euclidean Gradient to Riemannian Gradients."""
        raise NotImplementedError

    def proj(self, p, c):
        """Projects point p on the manifold."""
        raise NotImplementedError

    def proj_tan(self, u, p, c):
        """Projects u on the tangent space of p."""
        raise NotImplementedError

    def proj_tan0(self, u, c):
        """Projects u on the tangent space of the origin."""
        raise NotImplementedError

    def expmap(self, u, p, c):
        """Exponential map of u at point p."""
        raise NotImplementedError

    def logmap(self, p1, p2, c):
        """Logarithmic map of point p1 at point p2."""
        raise NotImplementedError

    def expmap0(self, u, c):
        """Exponential map of u at the origin."""
        raise NotImplementedError

    def logmap0(self, p, c):
        """Logarithmic map of point p at the origin."""
        raise NotImplementedError

    def mobius_add(self, x, y, c, dim=-1):
        """Adds points x and y."""
        raise NotImplementedError

    def mobius_matvec(self, m, x, c):
        """Performs hyperbolic matrix-vector multiplication."""
        raise NotImplementedError

    def init_weights(self, w, c, irange=1e-5):
        """Initializes random weights on the manifold."""
        raise NotImplementedError

    def inner(self, p, c, u, v=None, keepdim=False):
        """Inner product for tangent vectors at point x."""
        raise NotImplementedError

    def ptransp(self, x, y, u, c):
        """Parallel transport of u from x to y."""
        raise NotImplementedError

    def ptransp0(self, x, u, c):
        """Parallel transport of u from the origin to y."""
        raise NotImplementedError


class ManifoldParameter(Parameter):
    """Subclass of torch.nn.Parameter for Riemannian optimization."""

    def __new__(cls, data, requires_grad, manifold, c):
        """Create a new instance of ManifoldParameter."""
        return Parameter.__new__(cls, data, requires_grad)

    def __init__(self, data, requires_grad, manifold, c):
        self.c = c
        self.manifold = manifold

    def __repr__(self):
        return (
            "{} Parameter containing:\n".format(self.manifold.name)
            + super(Parameter, self).__repr__()
        )


class EuclideanManifold(Manifold):
    """Euclidean Manifold class."""

    def __init__(self):
        super(EuclideanManifold, self).__init__()
        self.name = "Euclidean"

    def normalize(self, p):
        """Normalizes the parameter p to have unit norm."""
        dim = p.size(-1)
        p.view(-1, dim).renorm_(2, 0, 1.0)
        return p

    def sqdist(self, p1, p2, c):
        """Squared distance between pairs of points."""
        return (p1 - p2).pow(2).sum(dim=-1)

    def egrad2rgrad(self, p, dp, c):
        """Converts Euclidean Gradient to Riemannian Gradients."""
        return dp

    def proj(self, p, c):
        """Projects point p on the manifold."""
        return p

    def proj_tan(self, u, p, c):
        """Projects u on the tangent space of p."""
        return u

    def proj_tan0(self, u, c):
        """Projects u on the tangent space of the origin."""
        return u

    def expmap(self, u, p, c):
        """Exponential map of u at point p."""
        return p + u

    def logmap(self, p1, p2, c):
        """Logarithmic map of point p1 at point p2."""
        return p2 - p1

    def expmap0(self, u, c):
        """Exponential map of u at the origin."""
        return u

    def logmap0(self, p, c):
        """Logarithmic map of point p at the origin."""
        return p

    def mobius_add(self, x, y, c, dim=-1):
        """Adds points x and y."""
        return x + y

    def mobius_matvec(self, m, x, c):
        """Performs Euclidean matrix-vector multiplication."""
        mx = x @ m.transpose(-1, -2)
        return mx

    def init_weights(self, w, c, irange=1e-5):
        """Initializes random weights on the manifold."""
        w.data.uniform_(-irange, irange)
        return w

    def inner(self, p, c, u, v=None, keepdim=False):
        """Inner product on the Euclidean manifold."""
        if v is None:
            v = u
        return (u * v).sum(dim=-1, keepdim=keepdim)

    def ptransp(self, x, y, v, c):
        """Parallel transport of v from x to y."""
        return v

    def ptransp0(self, x, v, c):
        """Parallel transport of v from the origin to x."""
        return x + v


class Artanh(torch.autograd.Function):
    """The inverse of the tanh function.

    It is defined as:

    .. math::
        \text{artanh}(x) = 0.5 \cdot (\log(1 + x) - \log(1 - x))

    This function is numerically stable for inputs close to -1 or 1.
    It is used to compute the hyperbolic distance in the Poincare Ball model.
    """

    @staticmethod
    def forward(ctx, x):
        # Clamp input to avoid numerical issues with log(0)
        x = x.clamp(-1 + 1e-15, 1 - 1e-15)
        ctx.save_for_backward(x)
        z = x.double()
        return (torch.log_(1 + z).sub_(torch.log_(1 - z))).mul_(0.5).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve the saved input tensor
        # and compute the gradient of the artanh function
        (input,) = ctx.saved_tensors
        return grad_output / (1 - input**2)


def tanh(x, clamp=15):
    """Hyperbolic tangent function with clamping to avoid overflow.

    This function is used to ensure numerical stability when computing
    the hyperbolic tangent in the Poincare Ball model.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    clamp : float, optional
        The value at which to clamp the input tensor to avoid overflow,
        by default 15.
    Returns
    -------
    torch.Tensor
        The hyperbolic tangent of the input tensor, clamped to the range [-clamp, clamp].
    """
    return x.clamp(-clamp, clamp).tanh()


def artanh(x):
    """Inverse hyperbolic tangent function.

    This function is used to compute the hyperbolic distance in the Poincare Ball model.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor, should be in the range (-1, 1).

    Returns
    -------
    torch.Tensor
        The inverse hyperbolic tangent of the input tensor.
    """
    return Artanh.apply(x)


class PoincareBallManifold(Manifold):
    """PoincareBall Manifold class.

    We use the following convention: x0^2 + x1^2 + ... + xd^2 < 1 / c

    Note that 1/sqrt(c) is the Poincare ball radius.
    """

    def __init__(
        self,
    ):
        super(PoincareBallManifold, self).__init__()
        self.name = "PoincareBall"
        self.min_norm = 1e-15
        self.eps = {torch.float32: 4e-3, torch.float64: 1e-5}

    def sqdist(self, p1, p2, c):
        sqrt_c = c**0.5
        dist_c = artanh(
            sqrt_c
            * self.mobius_add(-p1, p2, c, dim=-1).norm(dim=-1, p=2, keepdim=False)
        )
        dist = dist_c * 2 / sqrt_c
        return dist**2

    def _lambda_x(self, x, c):
        x_sqnorm = torch.sum(x.data.pow(2), dim=-1, keepdim=True)
        return 2 / (1.0 - c * x_sqnorm).clamp_min(self.min_norm)

    def egrad2rgrad(self, p, dp, c):
        lambda_p = self._lambda_x(p, c)
        dp /= lambda_p.pow(2)
        return dp

    def proj(self, x, c):
        norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), self.min_norm)
        maxnorm = (1 - self.eps[x.dtype]) / (c**0.5)
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x)

    def proj_tan(self, u, p, c):
        return u

    def proj_tan0(self, u, c):
        return u

    def expmap(self, u, p, c):
        sqrt_c = c**0.5
        u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        second_term = (
            tanh(sqrt_c / 2 * self._lambda_x(p, c) * u_norm) * u / (sqrt_c * u_norm)
        )
        gamma_1 = self.mobius_add(p, second_term, c)
        return gamma_1

    def logmap(self, p1, p2, c):
        sub = self.mobius_add(-p1, p2, c)
        sub_norm = sub.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        lam = self._lambda_x(p1, c)
        sqrt_c = c**0.5
        return 2 / sqrt_c / lam * artanh(sqrt_c * sub_norm) * sub / sub_norm

    def expmap0(self, u, c):
        sqrt_c = c**0.5
        u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), self.min_norm)
        gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
        return gamma_1

    def logmap0(self, p, c):
        sqrt_c = c**0.5
        p_norm = p.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        scale = 1.0 / sqrt_c * artanh(sqrt_c * p_norm) / p_norm
        return scale * p

    def mobius_add(self, x, y, c, dim=-1):
        x2 = x.pow(2).sum(dim=dim, keepdim=True)
        y2 = y.pow(2).sum(dim=dim, keepdim=True)
        xy = (x * y).sum(dim=dim, keepdim=True)
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denom = 1 + 2 * c * xy + c**2 * x2 * y2
        return num / denom.clamp_min(self.min_norm)

    def mobius_matvec(self, m, x, c):
        sqrt_c = c**0.5
        x_norm = x.norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)
        mx = x @ m.transpose(-1, -2)
        mx_norm = mx.norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)
        res_c = (
            tanh(mx_norm / x_norm * artanh(sqrt_c * x_norm)) * mx / (mx_norm * sqrt_c)
        )
        cond = (mx == 0).prod(-1, keepdim=True, dtype=torch.uint8)
        res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
        res = torch.where(cond, res_0, res_c)
        return res

    def init_weights(self, w, c, irange=1e-5):
        w.data.uniform_(-irange, irange)
        return w

    def _gyration(self, u, v, w, c, dim: int = -1):
        u2 = u.pow(2).sum(dim=dim, keepdim=True)
        v2 = v.pow(2).sum(dim=dim, keepdim=True)
        uv = (u * v).sum(dim=dim, keepdim=True)
        uw = (u * w).sum(dim=dim, keepdim=True)
        vw = (v * w).sum(dim=dim, keepdim=True)
        c2 = c**2
        a = -c2 * uw * v2 + c * vw + 2 * c2 * uv * vw
        b = -c2 * vw * u2 - c * uw
        d = 1 + 2 * c * uv + c2 * u2 * v2
        return w + 2 * (a * u + b * v) / d.clamp_min(self.min_norm)

    def inner(self, x, c, u, v=None, keepdim=False):
        if v is None:
            v = u
        lambda_x = self._lambda_x(x, c)
        return lambda_x**2 * (u * v).sum(dim=-1, keepdim=keepdim)

    def ptransp(self, x, y, u, c):
        lambda_x = self._lambda_x(x, c)
        lambda_y = self._lambda_x(y, c)
        return self._gyration(y, -x, u, c) * lambda_x / lambda_y

    def ptransp_(self, x, y, u, c):
        lambda_x = self._lambda_x(x, c)
        lambda_y = self._lambda_x(y, c)
        return self._gyration(y, -x, u, c) * lambda_x / lambda_y

    def ptransp0(self, x, u, c):
        lambda_x = self._lambda_x(x, c)
        return 2 * u / lambda_x.clamp_min(self.min_norm)

    def to_hyperboloid(self, x, c):
        K = 1.0 / c
        sqrtK = K**0.5
        sqnorm = torch.norm(x, p=2, dim=1, keepdim=True) ** 2
        return sqrtK * torch.cat([K + sqnorm, 2 * sqrtK * x], dim=1) / (K - sqnorm)
