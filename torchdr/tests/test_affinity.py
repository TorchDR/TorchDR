import torch
import pytest

from pykeops.torch import LazyTensor

from torchdr.affinity.entropic import (
    entropy,
    EntropicAffinity,
    L2SymmetricEntropicAffinity,
    SymmetricEntropicAffinity,
    DoublyStochasticEntropic,
)

lst_types = [torch.double, torch.float]


@pytest.mark.parametrize("dtype", lst_types)
def test_entropic_affinities(dtype):
    n, p = 10, 2
    perp = 5

    X = torch.randn(n, p, dtype=dtype)

    # --- Test Entropic Affinity ---

    # Without keops
    affinity_ea = EntropicAffinity(perplexity=perp, keops=False)
    P_ea = affinity_ea.compute_affinity(X)
    assert isinstance(P_ea, torch.Tensor), "Affinity matrix is not a torch.Tensor"
    assert P_ea.shape == (n, n), "Affinity matrix shape is incorrect"
    assert torch.allclose(
        P_ea.sum(1), torch.ones(n, dtype=dtype), atol=1e-5
    ), "Affinity matrix is not normalized row-wise"
    H_ea = entropy(P_ea, log=False, dim=1)
    assert torch.allclose(
        torch.exp(H_ea - 1), perp * torch.ones(n, dtype=dtype), atol=1e-5
    ), "Exp(Entropy-1) is not equal to the perplexity"

    # With keops
    affinity_ea_keops = EntropicAffinity(perplexity=perp, keops=True)
    P_ea_keops = affinity_ea_keops.compute_affinity(X)
    assert isinstance(P_ea_keops, LazyTensor), "Affinity matrix is not a LazyTensor"
    assert P_ea_keops.shape == (n, n), "Affinity matrix shape is incorrect"
    assert torch.allclose(
        P_ea_keops.sum(1), torch.ones(n, dtype=dtype), atol=1e-5
    ), "Affinity matrix is not normalized row-wise"
    H_ea_keops = entropy(P_ea_keops, log=False, dim=1)
    assert torch.allclose(
        torch.exp(H_ea_keops - 1), perp * torch.ones(n, dtype=dtype), atol=1e-5
    ), "Exp(Entropy-1) is not equal to the perplexity"

    # --- Test L2-Symmetric Entropic Affinity ---

    # Without keops
    affinity_l2ea = L2SymmetricEntropicAffinity(perplexity=perp, keops=False)
    P_l2ea = affinity_l2ea.compute_affinity(X)
    assert isinstance(P_l2ea, torch.Tensor), "Affinity matrix is not a torch.Tensor"
    assert P_l2ea.shape == (n, n), "Affinity matrix shape is incorrect"
    assert torch.allclose(
        P_l2ea, P_l2ea.T, atol=1e-5
    ), "Affinity matrix is not symmetric"

    # With keops
    affinity_l2ea_keops = L2SymmetricEntropicAffinity(perplexity=perp, keops=True)
    P_l2ea_keops = affinity_l2ea_keops.compute_affinity(X)
    assert isinstance(
        P_l2ea_keops, torch.Tensor
    ), "Affinity matrix is not a torch.Tensor"
    assert P_l2ea_keops.shape == (n, n), "Affinity matrix shape is incorrect"
    assert ((P_l2ea_keops - P_l2ea_keops.T) ** 2).sum(
        0, 1
    ) < 1e-5, "Affinity matrix is not symmetric"

    # --- Test Symmetric Entropic Affinity ---

    # Without keops
    affinity_sea = SymmetricEntropicAffinity(perplexity=perp, keops=False)
    P_sea = affinity_sea.compute_affinity(X)
    assert isinstance(P_sea, torch.Tensor), "Affinity matrix is not a torch.Tensor"
    assert P_sea.shape == (n, n), "Affinity matrix shape is incorrect"
    assert torch.allclose(P_sea, P_sea.T, atol=1e-5), "Affinity matrix is not symmetric"
    assert torch.allclose(
        P_sea.sum(1), torch.ones(n, dtype=dtype), atol=1e-5
    ), "Affinity matrix is not normalized row-wise"
    H_sea = entropy(P_sea, log=False, dim=1)
    assert torch.allclose(
        torch.exp(H_sea - 1), perp * torch.ones(10, dtype=dtype), atol=1e-5
    ), "Exp(Entropy-1) is not equal to the perplexity"

    # With keops
    affinity_sea_keops = SymmetricEntropicAffinity(perplexity=perp, keops=True)
    P_sea_keops = affinity_sea_keops.compute_affinity(X)
    assert isinstance(P_sea_keops, LazyTensor), "Affinity matrix is not a LazyTensor"
    assert P_sea_keops.shape == (n, n), "Affinity matrix shape is incorrect"
    assert ((P_sea_keops - P_sea_keops.T) ** 2).sum(
        0, 1
    ) < 1e-5, "Affinity matrix is not symmetric"
    assert torch.allclose(
        P_sea_keops.sum(1), torch.ones(n, dtype=dtype), atol=1e-5
    ), "Affinity matrix is not normalized row-wise"
    H_sea_keops = entropy(P_sea_keops, log=False, dim=1)
    assert torch.allclose(
        torch.exp(H_sea_keops - 1), perp * torch.ones(10, dtype=dtype), atol=1e-5
    ), "Exp(Entropy-1) is not equal to the perplexity"
