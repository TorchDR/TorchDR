import pytest
import torch
import numpy as np
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from torch.testing import assert_close

from torchdr.affinity import (
    Affinity,
    ScalarProductAffinity,
    SparseLogAffinity,
    GaussianAffinity,
    LogAffinity,
    EntropicAffinity,
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
    # Test that affinity_out must be an Affinity instance when not None
    with pytest.raises(ValueError):
        AffinityMatcher(affinity_in=GaussianAffinity(), affinity_out="invalid_affinity")


def test_invalid_affinity_in():
    with pytest.raises(ValueError):
        AffinityMatcher(affinity_in=None, affinity_out=GaussianAffinity())


def test_affinity_in_precomputed_shape_error():
    model = AffinityMatcher(affinity_in="precomputed", affinity_out=GaussianAffinity())
    with pytest.raises(ValueError):
        model.fit_transform(torch.rand(5, 4))


def test_convergence_reached():
    class TestAffinity(Affinity):
        def __call__(self, X, **kwargs):
            return X @ X.T

        def _compute_affinity(self, X):
            return X @ X.T

    model = AffinityMatcher(
        affinity_in=TestAffinity(),
        affinity_out=TestAffinity(),
        min_grad_norm=1e10,  # high value to ensure convergence
        max_iter=3,
        verbose=False,
        check_interval=1,
    )
    model.fit_transform(torch.rand(5, 2))
    assert model.n_iter_ < 2  # should converge in less than 2 iterations


def test_scheduler_not_configure_optimizer():
    model = AffinityMatcher(
        affinity_in=GaussianAffinity(), affinity_out=GaussianAffinity()
    )
    with pytest.raises(ValueError):
        model._configure_scheduler()


def test_scheduler_invalid_type():
    model = AffinityMatcher(
        affinity_in=GaussianAffinity(),
        affinity_out=GaussianAffinity(),
        scheduler="invalid_scheduler",
    )
    with pytest.raises(ValueError):
        model._configure_scheduler()


def test_lr_auto_warning():
    model = AffinityMatcher(
        affinity_in=GaussianAffinity(),
        affinity_out=GaussianAffinity(),
        lr="auto",
        verbose=True,
    )
    model._set_learning_rate()
    assert model.lr_ == 1.0


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
        model._configure_optimizer()


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
        model._configure_optimizer()


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
    model_adam._configure_optimizer()
    assert isinstance(model_adam.optimizer_, torch.optim.Adam)

    # Test optimizer class
    model_sgd = AffinityMatcher(
        affinity_in=GaussianAffinity(), affinity_out=GaussianAffinity(), optimizer=SGD
    )
    model_sgd._init_embedding(torch.rand(5, 2))
    model_sgd._set_params()
    model_sgd._set_learning_rate()
    model_sgd._configure_optimizer()
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
    model_sgd_kwargs._configure_optimizer()
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
    model_step._configure_scheduler()
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
    model_exp._configure_scheduler()
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
    model_square.affinity_in_ = torch.rand(5, 5)
    loss = model_square._compute_loss()
    assert isinstance(loss, torch.Tensor)

    # Test cross entropy loss
    model_ce = AffinityMatcher(
        affinity_in=GaussianAffinity(),
        affinity_out=ScalarProductAffinity(),
        loss_fn="cross_entropy_loss",
    )
    model_ce._init_embedding(X)
    model_ce.affinity_in_ = torch.rand(5, 5)
    loss = model_ce._compute_loss()
    assert isinstance(loss, torch.Tensor)

    # Test cross entropy loss with LogAffinity
    model_ce_log = AffinityMatcher(
        affinity_in=GaussianAffinity(),
        affinity_out=GaussianAffinity(),
        loss_fn="cross_entropy_loss",
    )
    model_ce_log._init_embedding(X)
    model_ce_log.affinity_in_ = torch.rand(5, 5)
    loss = model_ce_log._compute_loss()
    assert isinstance(loss, torch.Tensor)


def test_sparse_affinity_warning():
    affinity_in = EntropicAffinity(sparsity=True, verbose=False)
    assert affinity_in.sparsity
    AffinityMatcher(
        affinity_in=affinity_in,
        affinity_out=LogAffinity(verbose=False),  # Not UnnormalizedAffinity
        verbose=True,
    )
    # The warning is logged, and sparsity is set to False
    assert not affinity_in.sparsity


def test_fit_and_transform():
    # Test that fit_transform returns the embedding
    model = AffinityMatcher(
        affinity_in=GaussianAffinity(),
        affinity_out=GaussianAffinity(),
        max_iter=2,
    )
    X = torch.rand(5, 2)
    embedding = model.fit_transform(X)
    assert isinstance(embedding, torch.Tensor)
    assert embedding.shape == (5, 2)
    assert_close(embedding, model.embedding_)


def test_precomputed_affinity():
    # Create a precomputed affinity matrix
    X_affinity = torch.rand(5, 5)

    model = AffinityMatcher(
        affinity_in="precomputed",
        affinity_out=GaussianAffinity(),
        max_iter=2,  # Small value for quick test
    )
    embedding = model.fit_transform(X_affinity)
    # After fit_transform, we should have an embedding
    assert embedding is not None
    assert embedding.shape == (5, 2)  # n_samples x n_components


def test_sparse_affinity_with_indices():
    class TestSparseAffinity(SparseLogAffinity):
        def __call__(self, X, return_indices=False, **kwargs):
            if return_indices:
                indices = torch.randint(0, X.shape[0], (X.shape[0], 3))
                return torch.rand(X.shape[0], 3), indices
            return torch.rand(X.shape[0], 3)

        def _compute_affinity(self, X):
            return torch.exp(-(X @ X.T))

    model = AffinityMatcher(
        affinity_in=TestSparseAffinity(), affinity_out=GaussianAffinity()
    )
    embedding = model.fit_transform(torch.rand(5, 2))
    assert isinstance(embedding, torch.Tensor)
    assert embedding.shape == (5, 2)
    assert_close(embedding, model.embedding_)


def test_after_step():
    # This is a placeholder test for _additional_updates method
    # which currently does nothing in the base class
    model = AffinityMatcher(
        affinity_in=GaussianAffinity(), affinity_out=GaussianAffinity()
    )
    # Just ensure it doesn't raise an error
    model.on_training_step_end()


def test_affinity_out_none_requires_custom_loss():
    # Test that affinity_out=None requires a custom _loss method
    model = AffinityMatcher(affinity_in=GaussianAffinity(), affinity_out=None)
    X = torch.rand(5, 2)
    # Just do minimal setup needed for _loss() to be callable
    model._init_embedding(X)
    model.affinity_in_ = torch.rand(5, 5)  # Mock the fitted affinity
    with pytest.raises(ValueError, match="affinity_out is not set"):
        model._compute_loss()


def test_affinity_out_none_with_custom_loss():
    # Test that affinity_out=None works with custom _loss method
    class CustomAffinityMatcher(AffinityMatcher):
        def _compute_loss(self):
            # Simple custom loss that uses the embedding
            return (self.embedding_**2).sum()

    model = CustomAffinityMatcher(
        affinity_in=GaussianAffinity(), affinity_out=None, max_iter=2
    )
    X = torch.rand(5, 2)
    embedding = model.fit_transform(X)
    assert isinstance(embedding, torch.Tensor)
    assert embedding.shape == (5, 2)


def test_affinity_out_none_default():
    # Test that affinity_out defaults to None when not specified
    model = AffinityMatcher(affinity_in=GaussianAffinity())
    assert model.affinity_out is None


def test_affinity_out_invalid_type():
    # Test that affinity_out must be an Affinity instance when not None
    with pytest.raises(ValueError, match="affinity_out must be an Affinity instance"):
        AffinityMatcher(affinity_in=GaussianAffinity(), affinity_out=42)


def test_affinity_out_none_fit_without_custom_loss():
    # Test that fitting with affinity_out=None fails if no custom _loss is provided
    model = AffinityMatcher(
        affinity_in=GaussianAffinity(), affinity_out=None, max_iter=1
    )
    X = torch.rand(5, 2)
    with pytest.raises(ValueError, match="affinity_out is not set"):
        model.fit_transform(X)
