"""Tests for parametric (encoder-based) neighbor embeddings."""

# License: BSD 3-Clause License

import pytest
import torch
import torch.nn as nn

from torchdr import TSNE, UMAP, LargeVis


@pytest.fixture
def data():
    """Generate a small random dataset."""
    torch.manual_seed(0)
    n, d = 50, 10
    return torch.randn(n, d)


def _make_encoder(in_features, out_features):
    """Create a simple MLP encoder."""
    return nn.Sequential(
        nn.Linear(in_features, 32),
        nn.ReLU(),
        nn.Linear(32, out_features),
    )


class TestParametricTSNE:
    """Test parametric TSNE (autograd path)."""

    def test_fit_transform_shape(self, data):
        n, d = data.shape
        n_components = 2
        encoder = _make_encoder(d, n_components)
        model = TSNE(
            n_components=n_components,
            perplexity=5,
            max_iter=10,
            optimizer="Adam",
            lr=1e-3,
            encoder=encoder,
            random_state=0,
        )
        embedding = model.fit_transform(data)
        assert embedding.shape == (n, n_components)

    def test_transform_new_data(self, data):
        n, d = data.shape
        n_components = 2
        encoder = _make_encoder(d, n_components)
        model = TSNE(
            n_components=n_components,
            perplexity=5,
            max_iter=10,
            optimizer="Adam",
            lr=1e-3,
            encoder=encoder,
            random_state=0,
        )
        model.fit_transform(data)

        X_new = torch.randn(5, d)
        out = model.transform(X_new)
        assert out.shape == (5, n_components)


class TestParametricUMAP:
    """Test parametric UMAP (direct gradients path)."""

    def test_fit_transform_shape(self, data):
        n, d = data.shape
        n_components = 2
        encoder = _make_encoder(d, n_components)
        model = UMAP(
            n_neighbors=5,
            n_components=n_components,
            max_iter=10,
            backend=None,
            encoder=encoder,
            random_state=0,
        )
        embedding = model.fit_transform(data)
        assert embedding.shape == (n, n_components)

    def test_transform_new_data(self, data):
        n, d = data.shape
        n_components = 2
        encoder = _make_encoder(d, n_components)
        model = UMAP(
            n_neighbors=5,
            n_components=n_components,
            max_iter=10,
            backend=None,
            encoder=encoder,
            random_state=0,
        )
        model.fit_transform(data)

        X_new = torch.randn(5, d)
        out = model.transform(X_new)
        assert out.shape == (5, n_components)


class TestParametricLargeVis:
    """Test parametric LargeVis (autograd path with negative sampling)."""

    def test_fit_transform_shape(self, data):
        n, d = data.shape
        n_components = 2
        encoder = _make_encoder(d, n_components)
        model = LargeVis(
            perplexity=5,
            n_components=n_components,
            max_iter=10,
            backend=None,
            encoder=encoder,
            random_state=0,
        )
        embedding = model.fit_transform(data)
        assert embedding.shape == (n, n_components)


class TestTransform:
    """Test transform behavior with and without encoder."""

    def test_transform_none_returns_training_embedding(self, data):
        n, d = data.shape
        n_components = 2
        encoder = _make_encoder(d, n_components)
        model = LargeVis(
            perplexity=5,
            n_components=n_components,
            max_iter=10,
            backend=None,
            encoder=encoder,
            random_state=0,
        )
        training_embedding = model.fit_transform(data)
        result = model.transform()
        assert torch.equal(result, training_embedding)

    def test_transform_without_encoder_raises(self, data):
        model = TSNE(
            n_components=2,
            perplexity=5,
            max_iter=10,
            random_state=0,
        )
        model.fit_transform(data)
        with pytest.raises(NotImplementedError):
            model.transform(torch.randn(5, data.shape[1]))

    def test_transform_before_fit_raises(self, data):
        n_components = 2
        encoder = _make_encoder(data.shape[1], n_components)
        model = TSNE(
            n_components=n_components,
            perplexity=5,
            max_iter=10,
            encoder=encoder,
        )
        with pytest.raises(ValueError, match="not fitted"):
            model.transform(data)


class TestEncoderValidation:
    """Test encoder validation errors."""

    def test_encoder_dim_mismatch(self, data):
        wrong_dim_encoder = _make_encoder(data.shape[1], 5)
        model = TSNE(
            n_components=2,
            perplexity=5,
            max_iter=10,
            encoder=wrong_dim_encoder,
            random_state=0,
        )
        with pytest.raises(ValueError, match="encoder output dim"):
            model.fit_transform(data)

    def test_encoder_not_module(self, data):
        model = TSNE(
            n_components=2,
            perplexity=5,
            max_iter=10,
            encoder="not_a_module",
            random_state=0,
        )
        with pytest.raises(TypeError, match="nn.Module"):
            model.fit_transform(data)


class TestBackwardCompatibility:
    """Verify encoder=None (default) works identically to before."""

    def test_tsne_default(self, data):
        model = TSNE(
            n_components=2,
            perplexity=5,
            max_iter=10,
            random_state=0,
        )
        embedding = model.fit_transform(data)
        assert embedding.shape == (data.shape[0], 2)

    def test_umap_default(self, data):
        model = UMAP(
            n_neighbors=5,
            n_components=2,
            max_iter=10,
            backend=None,
            random_state=0,
        )
        embedding = model.fit_transform(data)
        assert embedding.shape == (data.shape[0], 2)
