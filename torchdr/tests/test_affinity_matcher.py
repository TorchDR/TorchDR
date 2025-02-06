import pytest
import torch

from torchdr.affinity import Affinity
from torchdr.affinity_matcher import AffinityMatcher


def test_invalid_optimizer():
    with pytest.raises(ValueError):
        AffinityMatcher(
            affinity_in=Affinity(), affinity_out=Affinity(), optimizer="invalid"
        )


def test_invalid_loss_fn():
    with pytest.raises(ValueError):
        AffinityMatcher(
            affinity_in=Affinity(), affinity_out=Affinity(), loss_fn="invalid_loss"
        )


def test_invalid_affinity_out():
    with pytest.raises(ValueError):
        AffinityMatcher(affinity_in=Affinity(), affinity_out=None)


def test_invalid_affinity_in():
    with pytest.raises(ValueError):
        AffinityMatcher(affinity_in=None, affinity_out=Affinity())


def test_affinity_in_precomputed_shape_error():
    model = AffinityMatcher(affinity_in="precomputed", affinity_out=Affinity())
    with pytest.raises(ValueError):
        model._fit(torch.rand(5, 4))


def test_convergence_reached(capfd):
    class TestAffinity(Affinity):
        def __call__(self, X, **kwargs):
            return X @ X.T

    model = AffinityMatcher(
        affinity_in=TestAffinity(),
        affinity_out=TestAffinity(),
        min_grad_norm=1e0,
        max_iter=3,
        verbose=True,
    )
    model._fit(torch.rand(5, 2))
    captured = capfd.readouterr()
    assert "Convergence reached" in captured.out


def test_scheduler_not_set_optimizer():
    model = AffinityMatcher(affinity_in=Affinity(), affinity_out=Affinity())
    with pytest.raises(ValueError):
        model._set_scheduler()


def test_scheduler_invalid_type():
    model = AffinityMatcher(
        affinity_in=Affinity(), affinity_out=Affinity(), scheduler="invalid_scheduler"
    )
    with pytest.raises(ValueError):
        model._set_scheduler()


def test_lr_auto_warning():
    with pytest.warns(UserWarning):
        model = AffinityMatcher(
            affinity_in=Affinity(), affinity_out=Affinity(), lr="auto", verbose=True
        )
        model._set_learning_rate()


def test_init_embedding_invalid():
    model = AffinityMatcher(
        affinity_in=Affinity(), affinity_out=Affinity(), init="invalid_init"
    )
    with pytest.raises(ValueError):
        model._init_embedding(torch.rand(5, 2))
