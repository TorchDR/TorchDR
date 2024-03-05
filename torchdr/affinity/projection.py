# -*- coding: utf-8 -*-
"""
Projections to construct affinity matrices
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#         Titouan Vayer <titouan.vayer@inria.fr>
#
# License: BSD 3-Clause License

import torch
import math
from tqdm import tqdm
from abc import abstractmethod

from torchdr.utils.root_finding import false_position
from torchdr.utils.operator import entropy


OPTIMIZERS = {'SGD': torch.optim.SGD,
              'Adam': torch.optim.Adam,
              'NAdam': torch.optim.NAdam
              }


class NanError(Exception):
    pass


class BadPerplexity(Exception):
    pass


class BaseAffinity():
    def __init__(self):
        self.log_ = {}  # BaseAffinity contains a dictionary of different results

    @abstractmethod
    def compute_affinity(self, X):
        pass


class GramAffinity(BaseAffinity):
    def __init__(self, centering=False):
        super(GramAffinity, self).__init__()
        self.centering = centering

    def compute_affinity(self, X):
        if self.centering:
            X = X - X.mean(0)
        self.affinity_matrix = X @ X.T
        return self.affinity_matrix


class LogAffinity(BaseAffinity):
    @abstractmethod
    def compute_log_affinity(self, X):
        pass

    def compute_affinity(self, X):
        """Computes an affinity matrix from an affinity matrix in log space.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Data on which affinity is computed.

        Returns
        -------
        P : torch.Tensor of shape (n_samples, n_samples)
            Affinity matrix.
        """
        log_P = self.compute_log_affinity(X)
        self.log_affinity_matrix = log_P
        self.affinity_matrix = torch.exp(log_P)
        return self.affinity_matrix


class NormalizedGaussianAndStudentAffinity(LogAffinity):
    """
    This class computes the normalized affinity associated to a Gaussian or t-Student
    kernel. The affinity matrix is normalized by given axis.

    Parameters
    ----------
    student : bool, optional
        If True, computes a t-Student kernel (default False).
    sigma : float, optional
        The length scale of the Gaussian kernel (default 1.0).
    p : int, optional
        p value for the p-norm distance to calculate between each vector pair 
        (default 2).
    """

    def __init__(self, student=False, sigma=1.0, p=2):
        self.student = student
        self.sigma = sigma
        self.p = p
        super(NormalizedGaussianAndStudentAffinity, self).__init__()

    def compute_log_affinity(self, X, axis=(0, 1)):
        """
        Computes the pairwise affinity matrix in log space and normalize it by given
        axis.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Data on which affinity is computed.

        Returns
        -------
        log_P : torch.Tensor of shape (n_samples, n_samples)
            Affinity matrix in log space.
        """
        C = torch.cdist(X, X, self.p) ** 2
        if self.student:
            log_P = - torch.log(1 + C)
        else:
            log_P = - C / (2 * self.sigma)
        return log_P - torch.logsumexp(log_P, dim=axis)


class EntropicAffinity(LogAffinity):
    """
    This class computes the entropic affinity used in SNE and tSNE in log domain. 
    It also corresponds to the Pe matrix in [1] in log domain (see also [2]). 
    When normalize_as_sne = True, the affinity is symmetrized as (Pe + Pe.T)/2.

    Parameters
    ----------
    perplexity : float
        Perplexity parameter, related to the number of 'effective' nearest neighbors
        that is used in SNE/t-SNE. Consider selecting a value between 2 and the number
        of samples. Different values can result in significantly different results.
    tol : float, optional
        Precision threshold at which the root finding algorithm stops.
    max_iter : int, optional
        Number of maximum iterations for the root finding algorithm.
    verbose : bool, optional
        Verbosity.
    begin : float or torch.Tensor, optional
        Initial lower bound of the root (default None).
    end : float or torch.Tensor, optional
        Initial upper bound of the root (default None).
    normalize_as_sne : bool, optional
        If True the entropic affinity is symmetrized as (Pe + Pe.T) / 2 (default True).

    References
    ----------
    [1] SNEkhorn: Dimension Reduction with Symmetric Entropic Affinities, Hugues Van
    Assel, Titouan Vayer, Rémi Flamary, Nicolas Courty, NeurIPS 2023.
    [2] Entropic Affinities: Properties and Efficient Numerical Computation, Max
    Vladymyrov, Miguel A. Carreira-Perpinan, ICML 2013.
    """

    def __init__(self,
                 perplexity,
                 tol=1e-5,
                 max_iter=1000,
                 verbose=True,
                 begin=None,
                 end=None,
                 normalize_as_sne=True):

        self.perplexity = perplexity
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.begin = begin
        self.end = end
        self.normalize_as_sne = normalize_as_sne
        super(EntropicAffinity, self).__init__()

    def compute_log_affinity(self, X):
        """
        Computes the pairwise entropic affinity matrix in log space. If
        normalize_as_sne is True returns the symmetrized version.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Data on which affinity is computed

        Returns
        -------
        log_P : torch.Tensor of shape (n_samples, n_samples)
            Affinity matrix in log space. If normalize_as_sne is True returns the
            symmetrized affinty in log space
        """
        C = torch.cdist(X, X, 2)**2
        log_P = self._solve_dual(C)
        if self.normalize_as_sne:  # does P+P.T/2 in log space
            log_P_SNE = torch.logsumexp(torch.stack(
                [log_P, log_P.T], 0), 0, keepdim=False) - math.log(2)
            return log_P_SNE
        else:
            return log_P

    def _solve_dual(self, C):
        """
        Performs a binary search to solve the dual problem of entropic affinities in
        log space.
        It solves the problem (EA) in [1] and returns the entropic affinity matrix in
        log space (which is **not** symmetric).

        Parameters
        ----------
        C : torch.Tensor of shape (n_samples, n_samples)
            Distance matrix between the samples.

        References
        ----------
        [1] SNEkhorn: Dimension Reduction with Symmetric Entropic Affinities, Hugues
        Van Assel, Titouan Vayer, Rémi Flamary, Nicolas Courty, NeurIPS 2023.
        """
        target_entropy = math.log(self.perplexity) + 1
        n = C.shape[0]

        if not 1 <= self.perplexity <= n:
            BadPerplexity(
                'The perplexity parameter must be between 1 and number of samples')

        def f(eps):
            return entropy(log_Pe(C, eps), log=True) - target_entropy

        eps_star, _, _ = false_position(
            f=f, n=n, begin=self.begin, end=self.end, tol=self.tol,
            max_iter=self.max_iter, verbose=self.verbose, device=C.device
            )
        log_affinity = log_Pe(C, eps_star)

        return log_affinity


class SymmetricEntropicAffinity(LogAffinity):
    def __init__(self,
                 perp,
                 lr=1e0,
                 square_parametrization=False,
                 tol=1e-3,
                 max_iter=500,
                 optimizer='Adam',
                 verbose=True,
                 tolog=False):
        """
        This class computes the solution to the symmetric entropic affinity problem
        described in [1], in log space. 
        More precisely, it solves equation (SEA) in [1] with the dual ascent procedure
        described in the paper and returns the log of the affinity matrix.

        Parameters
        ----------
        perp : int
            Perplexity parameter, related to the number of nearest neighbors that is
            used in other manifold learning algorithms. 
            Larger datasets usually require a larger perplexity. Consider selecting a
            value between 5 and the number of samples. 
            Different values can result in significantly different results. The
            perplexity must be less than the number of samples.
        lr : float, optional
            Learning rate used to update dual variables.
        square_parametrization : bool, optional
            Whether to optimize on the square of the dual variables. May be more stable
            in practice.
        tol : float, optional
            Precision threshold at which the algorithm stops, by default 1e-5.
        max_iter : int, optional
            Number of maximum iterations for the algorithm, by default 500.
        optimizer : {'SGD', 'Adam', 'NAdam'}, optional
            Which pytorch optimizer to use (default 'Adam').
        verbose : bool, optional
            Verbosity (default True).
        tolog : bool, optional
            Whether to store intermediate result in a dictionary (default False).

        Attributes
        ----------
        log_ : dictionary
            Contains the loss and the dual variables at each iteration of the 
            optimization algorithm when tolog = True.
        n_iter_ : int
            Number of iterations run.
        eps_ : torch.Tensor of shape (n_samples)
            Dual variable associated to the entropy constraint.
        mu_ : torch.Tensor of shape (n_samples)
            Dual variable associated to the marginal constraint.

        References
        ----------
        [1] SNEkhorn: Dimension Reduction with Symmetric Entropic Affinities, Hugues
        Van Assel, Titouan Vayer, Rémi Flamary, Nicolas Courty, NeurIPS 2023.
        """
        self.perp = perp
        self.lr = lr
        self.tol = tol
        self.max_iter = max_iter
        self.optimizer = optimizer
        self.verbose = verbose
        self.tolog = tolog
        self.n_iter_ = 0
        self.square_parametrization = square_parametrization
        super(SymmetricEntropicAffinity, self).__init__()

    def compute_log_affinity(self, X):
        """
        Computes the pairwise symmetric entropic affinity matrix in log space.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Data on which affinity is computed.

        Returns
        -------
        log_P : torch.Tensor of shape (n_samples, n_samples)
            Affinity matrix in log space. 
        """
        C = torch.cdist(X, X, 2) ** 2
        log_P = self._solve_dual(C)
        return log_P

    def _solve_dual(self, C):
        """
        Solves the dual optimization problem (Dual-SEA) in [1] and returns the
        corresponding symmetric entropic affinty in log space.

        Parameters
        ----------
        C : torch.Tensor of shape (n_samples, n_samples)
            Distance matrix between samples.

        Returns
        -------
        log_P : torch.Tensor of shape (n_samples, n_samples)
            Affinity matrix in log space. 

        References
        ----------
        [1] SNEkhorn: Dimension Reduction with Symmetric Entropic Affinities, Hugues
        Van Assel, Titouan Vayer, Rémi Flamary, Nicolas Courty, NeurIPS 2023.
        """
        device = C.device
        n = C.shape[0]
        if not 1 < self.perp <= n:
            BadPerplexity(
                'The perplexity parameter must be between 1 and number of samples')
        target_entropy = math.log(self.perp) + 1
        # dual variable corresponding to the entropy constraint
        eps = torch.ones(n, dtype=torch.double, device=device)
        # dual variable corresponding to the marginal constraint
        mu = torch.zeros(n, dtype=torch.double, device=device)
        log_P = log_Pse(C, eps, mu, to_square=self.square_parametrization)

        optimizer = OPTIMIZERS[self.optimizer]([eps, mu], lr=self.lr)

        if self.tolog:
            self.log_['eps'] = [eps.clone().detach().cpu()]
            self.log_['mu'] = [mu.clone().detach().cpu()]
            self.log_['loss'] = []

        if self.verbose:
            print(
                '---------- Computing the symmetric entropic affinity matrix \
                    ----------')

        one = torch.ones(n, dtype=torch.double, device=device)
        pbar = tqdm(range(self.max_iter), disable=not self.verbose)
        for k in pbar:
            with torch.no_grad():
                optimizer.zero_grad()
                H = entropy(log_P, log=True)

                if self.square_parametrization:
                    # the Jacobian must be corrected by 2 * diag(eps) in the case of
                    # square parametrization.
                    eps.grad = 2 * eps.clone().detach() * (H - target_entropy)
                else:
                    eps.grad = H - target_entropy

                P_sum = torch.exp(torch.logsumexp(log_P, -1, keepdim=False))
                mu.grad = P_sum - one
                optimizer.step()
                if not self.square_parametrization:  # optimize on eps > 0
                    eps.clamp_(min=0)

                log_P = log_Pse(
                    C, eps, mu, to_square=self.square_parametrization)

                if torch.isnan(eps).any() or torch.isnan(mu).any():
                    raise NanError(
                        f'NaN in dual variables at iteration {k}, consider decreasing \
                            the learning rate of SymmetricEntropicAffinity')

                if self.tolog:
                    eps0 = eps.clone().detach()
                    mu0 = mu.clone().detach()
                    self.log_['eps'].append(eps0)
                    self.log_['mu'].append(mu0)
                    if self.square_parametrization:
                        self.log_[
                            'loss'].append(
                                -Lagrangian(C, torch.exp(log_P.clone().detach()),
                                            eps0**2, mu0, self.perp).item())
                    else:
                        self.log_[
                            'loss'].append(
                                -Lagrangian(C, torch.exp(log_P.clone().detach()),
                                            eps0, mu0, self.perp).item())

                perps = torch.exp(H - 1)
                if self.verbose:
                    pbar.set_description(
                        f'PERPLEXITY:{float(perps.mean().item()): .2e} 
                        (std:{float(perps.std().item()): .2e}), '
                        f'MARGINAL:{float(P_sum.mean().item()): .2e} 
                        (std:{float(P_sum.std().item()): .2e})')

                if (torch.abs(H - math.log(self.perp)-1) < self.tol).all() \
                and (torch.abs(P_sum - one) < self.tol).all():
                    self.log_['n_iter'] = k
                    self.n_iter_ = k
                    if self.verbose:
                        print(f'breaking at iter {k}')
                    break

                if k == self.max_iter-1 and self.verbose:
                    print(
                        '---------- Warning: max iter attained, algorithm stops but \
                            may not have converged ----------')

        self.eps_ = eps.clone().detach()
        self.mu_ = mu.clone().detach()

        return log_P


class BistochasticAffinity(LogAffinity):
    """
    This class computes the symmetric doubly stochastic affinity matrix 
    in log domain with Sinkhorn algorithm.
    It normalizes a Gaussian RBF kernel or t-Student kernel to satisfy 
    the doubly stochasticity constraints.

    Parameters
    ----------
    eps : float, optional
        Regularization parameter for the Sinkhorn algorithm. 
        It corresponds to the square root of the length scale of the Gaussian kernel when student = False.
    f : torch.Tensor of shape (n_samples), optional
        Initialization for the dual variable of the Sinkhorn algorithm (default None).
    tol : float, optional
        Precision threshold at which the algorithm stops.    
    max_iter : int, optional
        Number of maximum iterations for the algorithm.
    student : bool, optional
        Whether to use a t-Student kernel instead of a Gaussian kernel.
    verbose : bool, optional
        Verbosity.
    tolog : bool, optional
        Whether to store intermediate result in a dictionary.

    Attributes
    ----------
    log_ : dictionary
        Contains the dual variables at each iteration of the optimization algorithm
        when tolog = True.
    n_iter_ : int
        Number of iterations run.

    """

    def __init__(self,
                 eps=1.0,
                 f=None,
                 tol=1e-5,
                 max_iter=100,
                 student=False,
                 verbose=False,
                 tolog=False):
        self.eps = eps
        self.f = f
        self.tol = tol
        self.max_iter = max_iter
        self.student = student
        self.tolog = tolog
        self.n_iter_ = 0
        self.verbose = verbose
        super(BistochasticAffinity, self).__init__()

    def compute_log_affinity(self, X):
        """
        Computes the doubly stochastic affinity matrix in log space. 
        Returns the log of the transport plan at convergence.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Data on which affinity is computed.

        Returns
        -------
        log_P : torch.Tensor of shape (n_samples, n_samples)
            Affinity matrix in log space. 
        """
        C = torch.cdist(X, X, 2)**2
        # If student is True, considers the Student-t kernel instead of Gaussian RBF
        if self.student:
            C = torch.log(1+C)
        log_P = self._solve_dual(C)
        return log_P

    def _solve_dual(self, C):
        """
        Performs Sinkhorn iterations in log domain to solve the entropic "self" (or
        "symmetric") OT problem with symmetric cost C and entropic regularization 
        eps.

        Parameters
        ----------
        C : torch.Tensor of shape (n_samples, n_samples)
            Distance matrix between samples.

        Returns
        -------
        log_P : torch.Tensor of shape (n_samples, n_samples)
            Affinity matrix in log space. 
        """

        if self.verbose:
            print(
                '---------- Computing the doubly stochastic affinity matrix ----------')
        device = C.device
        n = C.shape[0]

        # Allows a warm-start if a dual variable f is provided
        f = torch.zeros(n, device=device) if self.f is None else self.f

        if self.tolog:
            self.log_['f'] = [f.clone()]

        # Sinkhorn iterations
        for k in range(self.max_iter+1):
            f = 0.5 * (f - self.eps*torch.logsumexp((f - C) / self.eps, -1))

            if self.tolog:
                self.log_['f'].append(f.clone())

            if torch.isnan(f).any():
                raise NanError(
                    f'NaN in self-Sinkhorn dual variable at iteration {k}')

            log_T = (f[:, None] + f[None, :] - C) / self.eps
            if (torch.abs(torch.exp(torch.logsumexp(log_T, -1))-1) < self.tol).all():
                if self.verbose:
                    print(f'breaking at iter {k}')
                break

            if k == self.max_iter-1 and self.verbose:
                print('---------- Max iter attained for Sinkhorn algorithm ----------')

        self.n_iter_ = k

        return (f[:, None] + f[None, :] - C) / self.eps


def log_Pe(C, eps):
    """
    Returns the log of the directed affinity matrix of SNE with prescribed kernel 
    bandwidth.

    Parameters
    ----------
    C : torch.Tensor of shape (n_samples, n_samples)
        Distance matrix between samples.
    eps : torch.Tensor of shape (n_samples)
        Kernel bandwidths vector.

    Returns
    -------
    log_P : torch.Tensor of shape (n_samples, n_samples)
        log of the directed affinity matrix of SNE.
    """
    log_P = - C / (eps[:, None])
    return log_P - torch.logsumexp(log_P, -1, keepdim=True)


def log_Pse(C, eps, mu, to_square=False):
    """
    Returns the log of the symmetric entropic affinity matrix with specified parameters 
    epsilon and mu.

    Parameters
    ----------
    C : torch.Tensor of shape (n_samples, n_samples)
        Distance matrix between samples.
    eps : torch.Tensor of shape (n_samples)
        Symmetric entropic affinity dual variables associated to the entropy constraint.
    mu : torch.Tensor of shape (n_samples)
        Symmetric entropic affinity dual variables associated to the marginal 
        constraint.
    to_square : bool, optional
        Whether to use the square of the dual variables associated to the entropy 
        constraint, by default False. 
    """
    if to_square:
        return (mu[:, None] + mu[None, :] - 2*C) / (eps[:, None]**2 + eps[None, :]**2)
    else:
        return (mu[:, None] + mu[None, :] - 2*C) / (eps[:, None] + eps[None, :])


def Lagrangian(C, log_P, eps, mu, perp):
    """
    Computes the Lagrangian associated to the symmetric entropic affinity optimization 
    problem.

    Parameters
    ----------
    C : torch.Tensor of shape (n_samples, n_samples)
        Distance matrix between samples.
    log_P : torch.Tensor of shape (n_samples, n_samples)
        log of the symmetric entropic affinity matrix.
    eps : torch.Tensor of shape (n_samples)
        Dual variable associated to the entropy constraint.
    mu : torch.Tensor of shape (n_samples)
        Dual variable associated to the marginal constraint.
    perp : int
        Perplexity parameter.

    Returns
    -------
    cost : float
        Value of the Lagrangian.
    """
    one = torch.ones(C.shape[0], dtype=C.dtype, device=C.device)
    target_entropy = math.log(perp) + 1
    HP = entropy(log_P, log=True, ax=1)
    linear_cost = torch.exp(torch.logsumexp(log_P + torch.log(C), (0, 1)))
    dual_entropy = torch.inner(eps, (target_entropy - HP))
    dual_marginal = torch.inner(mu, (one - torch.exp(torch.logsumexp(log_P, -1))))
    return linear_cost + dual_entropy + dual_marginal
