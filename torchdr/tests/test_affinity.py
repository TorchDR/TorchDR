import torch
import pytest

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

    affinity = EntropicAffinity(perplexity=perp, keops=False)
    P = affinity.compute_affinity(X)
    assert isinstance(P, torch.Tensor)
    assert P.shape == (n, n)
    assert torch.allclose(P.sum(1), torch.ones(n, dtype=dtype), atol=1e-5)
    H = entropy(P, log=False, dim=1)
    assert torch.allclose(
        torch.exp(H - 1), perp * torch.ones(10, dtype=dtype), atol=1e-5
    )

    affinity = L2SymmetricEntropicAffinity(perplexity=perp, keops=False)
    P = affinity.compute_affinity(X)
    assert isinstance(P, torch.Tensor)
    assert P.shape == (n, n)
    assert torch.allclose(P, P.T, atol=1e-5)

    affinity = SymmetricEntropicAffinity(perplexity=perp, keops=False)
    P = affinity.compute_affinity(X)
    assert isinstance(P, torch.Tensor)
    assert P.shape == (n, n)
    assert torch.allclose(P, P.T, atol=1e-5)
    assert torch.allclose(P.sum(1), torch.ones(n, dtype=dtype), atol=1e-5)
    H = entropy(P, log=False, dim=1)
    assert torch.allclose(
        torch.exp(H - 1), perp * torch.ones(10, dtype=dtype), atol=1e-5
    )
