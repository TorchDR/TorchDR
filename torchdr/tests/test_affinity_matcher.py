import pytest
import torch
import numpy as np
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR, ExponentialLR

from torchdr.affinity import (
    Affinity,
    ScalarProductAffinity,
    SparseLogAffinity,
    NormalizedGaussianAffinity,
    GaussianAffinity,
)
from torchdr.affinity_matcher import AffinityMatcher


def test_invalid_loss_fn():
    with pytest.raises(ValueError):
        AffinityMatcher(
            affinity_in=GaussianAffinity(),
            affinity_out=GaussianAffinity(),
            loss_fn="invalid_loss",
        )


def test_invalid_affinity_out():
    with pytest.raises(ValueError):
        AffinityMatcher(affinity_in=GaussianAffinity(), affinity_out=None)


def test_invalid_affinity_in():
    with pytest.raises(ValueError):
        AffinityMatcher(affinity_in=None, affinity_out=GaussianAffinity())


def test_affinity_in_precomputed_shape_error():
    model = AffinityMatcher(affinity_in="precomputed", affinity_out=GaussianAffinity())
    with pytest.raises(ValueError):
        model._fit(torch.rand(5, 4))


def test_convergence_reached(capfd):
    class TestAffinity(Affinity):
        def __call__(self, X, **kwargs):
            return X @ X.T

        def _compute_affinity(self, X):
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
    model = AffinityMatcher(
        affinity_in=GaussianAffinity(), affinity_out=GaussianAffinity()
    )
    with pytest.raises(ValueError):
        model._set_scheduler()


def test_scheduler_invalid_type():
    model = AffinityMatcher(
        affinity_in=GaussianAffinity(),
        affinity_out=GaussianAffinity(),
        scheduler="invalid_scheduler",
    )
    with pytest.raises(ValueError):
        model._set_scheduler()


def test_lr_auto_warning():
    with pytest.warns(UserWarning):
        model = AffinityMatcher(
            affinity_in=GaussianAffinity(),
            affinity_out=GaussianAffinity(),
            lr="auto",
            verbose=True,
        )
        model._set_learning_rate()


def test_init_embedding_invalid():
    model = AffinityMatcher(
        affinity_in=GaussianAffinity(),
        affinity_out=GaussianAffinity(),
        init="invalid_init",
    )
    with pytest.raises(ValueError):
        model._init_embedding(torch.rand(5, 2))


def test_optimizer_invalid_string():
    model = AffinityMatcher(
        affinity_in=GaussianAffinity(),
        affinity_out=GaussianAffinity(),
        optimizer="InvalidOptimizer",
    )
    model._init_embedding(torch.rand(5, 2))
    model._set_params()
    model._set_learning_rate()
    with pytest.raises(ValueError):
        model._set_optimizer()


def test_optimizer_invalid_class():
    class NotAnOptimizer:
        pass

    model = AffinityMatcher(
        affinity_in=GaussianAffinity(),
        affinity_out=GaussianAffinity(),
        optimizer=NotAnOptimizer,
    )
    model._init_embedding(torch.rand(5, 2))
    model._set_params()
    model._set_learning_rate()
    with pytest.raises(ValueError):
        model._set_optimizer()


def test_init_embedding_methods():
    X = torch.rand(5, 2)

    # Test normal initialization
    model_normal = AffinityMatcher(
        affinity_in=GaussianAffinity(), affinity_out=GaussianAffinity(), init="normal"
    )
    model_normal._init_embedding(X)
    assert model_normal.embedding_.shape == (5, 2)

    # Test random initialization (alias for normal)
    model_random = AffinityMatcher(
        affinity_in=GaussianAffinity(), affinity_out=GaussianAffinity(), init="random"
    )
    model_random._init_embedding(X)
    assert model_random.embedding_.shape == (5, 2)

    # Test PCA initialization
    model_pca = AffinityMatcher(
        affinity_in=GaussianAffinity(), affinity_out=GaussianAffinity(), init="pca"
    )
    model_pca._init_embedding(X)
    assert model_pca.embedding_.shape == (5, 2)

    # Test tensor initialization
    init_tensor = torch.ones((5, 2))
    model_tensor = AffinityMatcher(
        affinity_in=GaussianAffinity(),
        affinity_out=GaussianAffinity(),
        init=init_tensor,
    )
    model_tensor._init_embedding(X)
    assert model_tensor.embedding_.shape == (5, 2)

    # Test numpy array initialization
    init_array = np.ones((5, 2))
    model_array = AffinityMatcher(
        affinity_in=GaussianAffinity(), affinity_out=GaussianAffinity(), init=init_array
    )
    model_array._init_embedding(X)
    assert model_array.embedding_.shape == (5, 2)


def test_different_optimizers():
    # Test string optimizer
    model_adam = AffinityMatcher(
        affinity_in=GaussianAffinity(),
        affinity_out=GaussianAffinity(),
        optimizer="Adam",
    )
    model_adam._init_embedding(torch.rand(5, 2))
    model_adam._set_params()
    model_adam._set_learning_rate()
    model_adam._set_optimizer()
    assert isinstance(model_adam.optimizer_, torch.optim.Adam)

    # Test optimizer class
    model_sgd = AffinityMatcher(
        affinity_in=GaussianAffinity(), affinity_out=GaussianAffinity(), optimizer=SGD
    )
    model_sgd._init_embedding(torch.rand(5, 2))
    model_sgd._set_params()
    model_sgd._set_learning_rate()
    model_sgd._set_optimizer()
    assert isinstance(model_sgd.optimizer_, SGD)

    # Test optimizer with kwargs
    model_sgd_kwargs = AffinityMatcher(
        affinity_in=GaussianAffinity(),
        affinity_out=GaussianAffinity(),
        optimizer="SGD",
        optimizer_kwargs={"momentum": 0.9},
    )
    model_sgd_kwargs._init_embedding(torch.rand(5, 2))
    model_sgd_kwargs._set_params()
    model_sgd_kwargs._set_learning_rate()
    model_sgd_kwargs._set_optimizer()
    assert isinstance(model_sgd_kwargs.optimizer_, SGD)


def test_different_schedulers():
    # Test string scheduler
    model_step = AffinityMatcher(
        affinity_in=GaussianAffinity(),
        affinity_out=GaussianAffinity(),
        scheduler="StepLR",
        scheduler_kwargs={"step_size": 10},
    )
    model_step._init_embedding(torch.rand(5, 2))
    model_step._set_params()
    model_step._set_learning_rate()
    model_step._set_optimizer()
    model_step._set_scheduler()
    assert isinstance(model_step.scheduler_, StepLR)

    # Test scheduler class
    model_exp = AffinityMatcher(
        affinity_in=GaussianAffinity(),
        affinity_out=GaussianAffinity(),
        scheduler=ExponentialLR,
        scheduler_kwargs={"gamma": 0.9},
    )
    model_exp._init_embedding(torch.rand(5, 2))
    model_exp._set_params()
    model_exp._set_learning_rate()
    model_exp._set_optimizer()
    model_exp._set_scheduler()
    assert isinstance(model_exp.scheduler_, ExponentialLR)


def test_loss_with_different_functions():
    X = torch.rand(5, 2)

    # Test square loss
    model_square = AffinityMatcher(
        affinity_in=GaussianAffinity(),
        affinity_out=GaussianAffinity(),
        loss_fn="square_loss",
    )
    model_square._init_embedding(X)
    model_square.PX_ = torch.rand(5, 5)
    loss = model_square._loss()
    assert isinstance(loss, torch.Tensor)

    # Test cross entropy loss
    model_ce = AffinityMatcher(
        affinity_in=GaussianAffinity(),
        affinity_out=ScalarProductAffinity(),
        loss_fn="cross_entropy_loss",
    )
    model_ce._init_embedding(X)
    model_ce.PX_ = torch.rand(5, 5)
    loss = model_ce._loss()
    assert isinstance(loss, torch.Tensor)

    # Test cross entropy loss with LogAffinity
    model_ce_log = AffinityMatcher(
        affinity_in=GaussianAffinity(),
        affinity_out=GaussianAffinity(),
        loss_fn="cross_entropy_loss",
    )
    model_ce_log._init_embedding(X)
    model_ce_log.PX_ = torch.rand(5, 5)
    loss = model_ce_log._loss()
    assert isinstance(loss, torch.Tensor)


def test_sparse_affinity_warning():
    class TestSparseAffinity(SparseLogAffinity):
        def __init__(self):
            super().__init__()
            self._sparsity = True

        def _compute_affinity(self, X):
            return torch.rand(X.shape[0], X.shape[0])

    with pytest.warns(UserWarning):
        AffinityMatcher(
            affinity_in=TestSparseAffinity(), affinity_out=NormalizedGaussianAffinity()
        )

    # No warning when using UnnormalizedAffinity
    sparse_affinity = TestSparseAffinity()
    sparse_affinity._sparsity = True
    # Just construct the affinity matcher without assigning to unused variable
    AffinityMatcher(affinity_in=sparse_affinity, affinity_out=GaussianAffinity())
    assert sparse_affinity._sparsity  # Use truth value directly instead of == True


def test_fit_and_transform():
    X = torch.rand(5, 2)

    # Test fit_transform
    model = AffinityMatcher(
        affinity_in=GaussianAffinity(),
        affinity_out=GaussianAffinity(),
        max_iter=2,  # Small value for quick test
    )
    embedding = model.fit_transform(X)
    assert embedding.shape == (5, 2)
    assert hasattr(model, "embedding_")

    # Test fit
    model2 = AffinityMatcher(
        affinity_in=GaussianAffinity(),
        affinity_out=GaussianAffinity(),
        max_iter=2,  # Small value for quick test
    )
    model2.fit(X)
    assert hasattr(model2, "embedding_")


def test_precomputed_affinity():
    # Create a precomputed affinity matrix
    X_affinity = torch.rand(5, 5)

    model = AffinityMatcher(
        affinity_in="precomputed",
        affinity_out=GaussianAffinity(),
        max_iter=2,  # Small value for quick test
    )
    model.fit_transform(X_affinity)
    assert hasattr(model, "PX_")
    assert model.PX_ is X_affinity


def test_sparse_affinity_with_indices():
    class TestSparseAffinity(SparseLogAffinity):
        def __call__(self, X, return_indices=False, **kwargs):
            if return_indices:
                indices = torch.randint(0, X.shape[0], (X.shape[0], 3))
                return torch.rand(X.shape[0], 3), indices
            return torch.rand(X.shape[0], 3)

        def _compute_affinity(self, X):
            return torch.rand(X.shape[0], X.shape[0])

    X = torch.rand(5, 2)
    model = AffinityMatcher(
        affinity_in=TestSparseAffinity(),
        affinity_out=GaussianAffinity(),
        max_iter=2,
    )
    model.fit(X)
    assert hasattr(model, "NN_indices_")

    # Test that _loss uses indices
    loss = model._loss()
    assert isinstance(loss, torch.Tensor)


def test_additional_updates():
    # This is a placeholder test for _additional_updates method
    # which currently does nothing in the base class
    model = AffinityMatcher(
        affinity_in=GaussianAffinity(), affinity_out=GaussianAffinity()
    )
    # Just ensure it doesn't raise an error
    model._additional_updates()
