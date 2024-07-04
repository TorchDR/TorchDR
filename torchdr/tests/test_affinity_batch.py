# -*- coding: utf-8 -*-
"""
Tests for batched affinity matrices.
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import pytest
import torch
from torch.testing import assert_close

from torchdr.affinity import (
    ScalarProductAffinity,
    GibbsAffinity,
    NormalizedGibbsAffinity,
    SelfTuningGibbsAffinity,
    StudentAffinity,
    EntropicAffinity,
    SymmetricEntropicAffinity,
    SinkhornAffinity,
    DoublyStochasticQuadraticAffinity,
    UMAPAffinityIn,
    UMAPAffinityOut,
)

DEVICE = "cpu"


@pytest.mark.parametrize(
    "Affinity, kwargs",
    [
        (ScalarProductAffinity, {}),
        (GibbsAffinity, {}),
        (StudentAffinity, {}),
        (NormalizedGibbsAffinity, {"normalization_dim": None}),
        (NormalizedGibbsAffinity, {"normalization_dim": 0}),
        (NormalizedGibbsAffinity, {"normalization_dim": 1}),
        (NormalizedGibbsAffinity, {"normalization_dim": (0, 1)}),
        (SelfTuningGibbsAffinity, {"normalization_dim": None}),
        (SelfTuningGibbsAffinity, {"normalization_dim": 0}),
        (SelfTuningGibbsAffinity, {"normalization_dim": 1}),
        (SelfTuningGibbsAffinity, {"normalization_dim": (0, 1)}),
        (EntropicAffinity, {}),
        (SymmetricEntropicAffinity, {"lr": 1e-6}),
        (SinkhornAffinity, {}),
        (DoublyStochasticQuadraticAffinity, {"max_iter": 100}),
        (UMAPAffinityIn, {}),
        (UMAPAffinityOut, {}),
    ],
)
def test_get_batch_affinity(Affinity, kwargs):
    n_batch = 2
    batch_size = 5
    n = n_batch * batch_size

    X = torch.randn(n, 3).to(device=DEVICE)
    torch.manual_seed(0)
    indices = torch.randperm(n).reshape(-1, batch_size)

    Aff = Affinity(keops=False, device=DEVICE, **kwargs)
    P = Aff.fit_transform(X)

    # extract batch from full matrix
    P_subset = torch.zeros((n_batch, batch_size, batch_size)).to(
        device=X.device, dtype=X.dtype
    )
    for b in range(n_batch):
        ind = indices[b]
        P_subset[b] = P[ind][:, ind]
    # use get_batch
    P_batch = Aff.get_batch(indices)

    assert_close(
        P_subset,
        P_batch,
        msg="P_subset and P_batch are different.",
    )

    # test consistency with keops:
    Aff_keops = Affinity(keops=True, device=DEVICE, **kwargs)
    Aff_keops.fit(X)
    P_batch_keops = Aff_keops.get_batch(indices)
    assert (
        (P_batch.sum(1) - P_batch_keops.sum(1).squeeze()) ** 2
    ).sum() < 1e-6, "get_batch not consistent with KeOps."
