"""Riemannian adam optimizer."""

# Author: Nicolas Courty <ncourty@irisa.fr>
#
# License: BSD 3-Clause License

# inspired by geoopt implementation (https://github.com/geoopt/)

import torch.optim
from torchdr.utils import EuclideanManifold, ManifoldParameter

# in order not to create it at each iteration
_default_manifold = EuclideanManifold()


class OptimMixin(object):
    """Mixin for Riemannian optimizer."""

    def __init__(self, *args, stabilize=None, **kwargs):
        self._stabilize = stabilize
        super().__init__(*args, **kwargs)

    def stabilize_group(self, group):
        pass

    def stabilize(self):
        """Stabilize parameters if they are off-manifold due to numerical reasons."""
        for group in self.param_groups:
            self.stabilize_group(group)


def copy_or_set_(dest, source):
    """A workaround to respect strides of :code:`dest` when copying :code:`source`.

    (https://github.com/geoopt/geoopt/issues/70)

    Parameters
    ----------
    dest : torch.Tensor
        Destination tensor where to store new data.
    source : torch.Tensor
        Source data to put in the new tensor.

    Returns
    -------
    dest
        torch.Tensor, modified inplace.
    """
    if dest.stride() != source.stride():
        return dest.copy_(source)
    else:
        return dest.set_(source)


class RiemannianAdam(OptimMixin, torch.optim.Adam):
    r"""Riemannian Adam with the same API as :class:`torch.optim.Adam`.

    Parameters
    ----------
    params : iterable
        iterable of parameters to optimize or dicts defining
        parameter groups
    lr : float (optional)
        learning rate (default: 1e-3)
    betas : Tuple[float, float] (optional)
        coefficients used for computing
        running averages of gradient and its square (default: (0.9, 0.999))
    eps : float (optional)
        term added to the denominator to improve
        numerical stability (default: 1e-8)
    weight_decay : float (optional)
        weight decay (L2 penalty) (default: 0)
    amsgrad : bool (optional)
        whether to use the AMSGrad variant of this
        algorithm from the paper `On the Convergence of Adam and Beyond`_
        (default: False)

    Other Parameters
    ----------------
    stabilize : int
        Stabilize parameters if they are off-manifold due to numerical
        reasons every ``stabilize`` steps (default: ``None`` -- no stabilize)
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments
        ---------
        closure : callable (optional)
            A closure that reevaluates the model
            and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        with torch.no_grad():
            for group in self.param_groups:
                if "step" not in group:
                    group["step"] = 0
                betas = group["betas"]
                weight_decay = group["weight_decay"]
                eps = group["eps"]
                learning_rate = group["lr"]
                amsgrad = group["amsgrad"]
                for point in group["params"]:
                    grad = point.grad
                    if grad is None:
                        continue
                    if isinstance(point, (ManifoldParameter)):
                        manifold = point.manifold
                        c = point.c
                    else:
                        manifold = _default_manifold
                        c = None
                    if grad.is_sparse:
                        raise RuntimeError(
                            "Riemannian Adam does not support sparse gradients yet (PR is welcome)"
                        )

                    state = self.state[point]

                    # State initialization
                    if len(state) == 0:
                        state["step"] = 0
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(point)
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = torch.zeros_like(point)
                        if amsgrad:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state["max_exp_avg_sq"] = torch.zeros_like(point)
                    # make local variables for easy access
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    # actual step
                    grad.add_(point, alpha=weight_decay)
                    grad = manifold.egrad2rgrad(point, grad, c)
                    exp_avg.mul_(betas[0]).add_(grad, alpha=1 - betas[0])
                    exp_avg_sq.mul_(betas[1]).add_(
                        manifold.inner(point, c, grad, keepdim=True), alpha=1 - betas[1]
                    )
                    if amsgrad:
                        max_exp_avg_sq = state["max_exp_avg_sq"]
                        # Maintains the maximum of all 2nd moment running avg. till now
                        torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                        # Use the max. for normalizing running avg. of gradient
                        denom = max_exp_avg_sq.sqrt().add_(eps)
                    else:
                        denom = exp_avg_sq.sqrt().add_(eps)
                    group["step"] += 1
                    bias_correction1 = 1 - betas[0] ** group["step"]
                    bias_correction2 = 1 - betas[1] ** group["step"]
                    step_size = learning_rate * bias_correction2**0.5 / bias_correction1

                    # copy the state, we need it for retraction
                    # get the direction for ascend
                    direction = exp_avg / denom
                    # transport the exponential averaging to the new point
                    new_point = manifold.proj(
                        manifold.expmap(-step_size * direction, point, c), c
                    )
                    exp_avg_new = manifold.ptransp(point, new_point, exp_avg, c)
                    # use copy only for user facing point
                    copy_or_set_(point, new_point)
                    exp_avg.set_(exp_avg_new)

                    group["step"] += 1
                if self._stabilize is not None and group["step"] % self._stabilize == 0:
                    self.stabilize_group(group)
        return loss

    @torch.no_grad()
    def stabilize_group(self, group):
        for p in group["params"]:
            if not isinstance(p, ManifoldParameter):
                continue
            state = self.state[p]
            if not state:  # due to None grads
                continue
            manifold = p.manifold
            c = p.c
            exp_avg = state["exp_avg"]
            copy_or_set_(p, manifold.proj(p, c))
            exp_avg.set_(manifold.proj_tan(exp_avg, p, c))
