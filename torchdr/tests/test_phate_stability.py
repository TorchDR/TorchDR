"""PHATE stability regression tests."""

import torch

from torchdr import PHATE


def test_phate_regression_no_nan_on_fixed_seed_gaussian():
    """Regression test for NaNs at iter 0 on a deterministic synthetic matrix."""
    seed = 4
    g = torch.Generator().manual_seed(seed)
    X = (1.0 + torch.randn(128, 64, generator=g)).to(torch.float32)

    phate = PHATE(
        backend=None,
        device="cpu",
        k=5,
        t=30,
        max_iter=1,
        random_state=seed,
    )

    embedding = phate.fit_transform(X)
    assert torch.isfinite(embedding).all(), "PHATE produced NaN/Inf embedding."
