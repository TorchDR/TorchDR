"""Euclidean manifold."""

from torchdr.manifold.manifold import Manifold


class Euclidean(Manifold):
    """
    Euclidean Manifold class.
    """

    def __init__(self):
        super(Euclidean, self).__init__()
        self.name = 'Euclidean'

    def normalize(self, p):
        """Normalizes the parameter p to have unit norm."""
        dim = p.size(-1)
        p.view(-1, dim).renorm_(2, 0, 1.)
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
