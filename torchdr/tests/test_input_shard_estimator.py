"""Single-process coverage for the ``input_layout='sharded'`` estimator path.

These tests exercise the sharded-input wiring that does *not* require a live
process group: the ``ShardLayout.owner_boundaries`` table, the configuration
guards that reject layouts the first vertical slice does not support, and the
degenerate one-rank case where a sharded run must reproduce the replicated run
exactly (a single shard is the whole dataset, so nothing may change). The real
multi-rank NCCL/FAISS-GPU vertical slice is covered by
``test_distributed_umap_input_shard.py``.
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import numpy as np
import pytest
import torch

from torchdr import SNE, UMAP
from torchdr.affinity import UMAPAffinity
from torchdr.affinity_matcher import AffinityMatcher
from torchdr.distance import FaissConfig, FaissPlanConfig
from torchdr.distributed.input_contract import ShardLayout
from torchdr.utils import faiss, seed_everything

pytestmark = pytest.mark.skipif(
    faiss is None or faiss is False, reason="faiss not installed"
)


def _blobs(n, d, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=6.0, size=(4, d))
    labels = rng.integers(0, 4, size=n)
    return (centers[labels] + rng.normal(scale=1.0, size=(n, d))).astype(np.float32)


# --------------------------------------------------------------------------- #
# ShardLayout.owner_boundaries
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "counts",
    [(10,), (5, 5), (15, 55, 30, 20), (80, 40), (0, 7, 3)],
)
def test_owner_boundaries_matches_prefix_sum(counts):
    """The boundary table is the length W+1 rank-major prefix sum."""
    layout = ShardLayout(
        rank=0,
        world_size=len(counts),
        local_count=counts[0],
        global_count=sum(counts),
        local_offset=0,
        counts=tuple(counts),
    )
    boundaries = layout.owner_boundaries()
    expected = [0]
    for c in counts:
        expected.append(expected[-1] + c)
    assert boundaries.dtype == torch.long
    assert boundaries.tolist() == expected

    # Each interior global row must map back to the shard that owns it.
    for rank, c in enumerate(counts):
        for local in range(c):
            g = expected[rank] + local
            owner = int(torch.bucketize(torch.tensor(g), boundaries, right=True)) - 1
            assert owner == rank


# --------------------------------------------------------------------------- #
# Configuration guards
# --------------------------------------------------------------------------- #


def test_invalid_input_layout_rejected():
    with pytest.raises(ValueError, match="input_layout"):
        UMAP(input_layout="partitioned")


def test_sharded_rejects_pca_init():
    X = _blobs(60, 8)
    model = UMAP(
        input_layout="sharded",
        init="pca",
        backend="faiss",
        device="cpu",
        n_neighbors=10,
    )
    with pytest.raises(NotImplementedError, match="init="):
        model.fit_transform(X)


def test_sharded_rejects_tensor_init():
    X = _blobs(60, 8)
    init = torch.randn(60, 2)
    model = UMAP(
        input_layout="sharded", init=init, backend="faiss", device="cpu", n_neighbors=10
    )
    with pytest.raises(NotImplementedError, match="init="):
        model.fit_transform(X)


def test_sharded_rejects_precomputed_affinity():
    P = torch.rand(20, 20)
    P = P + P.T
    model = AffinityMatcher(
        affinity_in="precomputed", init="random", input_layout="sharded", device="cpu"
    )
    with pytest.raises(NotImplementedError, match="precomputed"):
        model.fit_transform(P)


def test_sharded_rejects_faiss_plan_config():
    with pytest.raises(ValueError, match="FaissPlanConfig"):
        UMAPAffinity(
            input_layout="sharded", backend=FaissPlanConfig(), distributed=False
        )


def test_sharded_rejects_non_flat_faiss_config():
    with pytest.raises(NotImplementedError, match="Flat"):
        UMAPAffinity(
            input_layout="sharded",
            backend=FaissConfig(index_type="IVF", nlist=16),
            distributed=False,
        )


def test_sharded_rejects_explicit_non_faiss_backend():
    with pytest.raises(ValueError, match="requires backend='faiss'"):
        UMAPAffinity(input_layout="sharded", backend="keops", distributed=False)


def test_estimator_and_affinity_layouts_must_match():
    with pytest.raises(ValueError, match="must match affinity_in.input_layout"):
        SNE(input_layout="sharded")


def test_sharded_accepts_explicit_flat_faiss_config():
    """An explicit exact-Flat ``FaissConfig`` is the one accepted backend override.

    The non-Flat and ``FaissPlanConfig`` rejections above only prove the guard
    fires; this pins the positive branch so a user who passes ``FaissConfig()``
    (whose ``index_type`` defaults to ``"Flat"``) keeps their own config object
    rather than having it silently replaced.
    """
    config = FaissConfig()
    aff = UMAPAffinity(
        n_neighbors=10,
        backend=config,
        device="cpu",
        input_layout="sharded",
        distributed=False,
    )
    assert aff._sharded_faiss_config_ is config


def test_sharded_rejects_dataloader():
    """A DataLoader has no addressable row shard, so the layout is rejected."""
    from torch.utils.data import DataLoader, TensorDataset

    loader = DataLoader(TensorDataset(torch.as_tensor(_blobs(40, 8))), batch_size=8)
    model = UMAP(input_layout="sharded", backend="faiss", device="cpu", n_neighbors=10)
    with pytest.raises(NotImplementedError, match="DataLoader"):
        model.fit_transform(loader)


def test_sharded_rejects_encoder():
    """An encoder maps rows to a learned space, breaking the raw-row shard contract."""
    model = AffinityMatcher(
        affinity_in=UMAPAffinity(
            n_neighbors=10,
            backend="faiss",
            device="cpu",
            input_layout="sharded",
        ),
        init="random",
        input_layout="sharded",
        device="cpu",
        encoder=torch.nn.Linear(8, 4),
    )
    with pytest.raises(NotImplementedError, match="encoder"):
        model.fit_transform(_blobs(60, 8))


def test_sharded_forces_process_duplicates_off():
    model = UMAP(input_layout="sharded", process_duplicates=True, device="cpu")
    assert model.process_duplicates is False
    # The replicated default is untouched.
    assert UMAP(process_duplicates=True).process_duplicates is True


# --------------------------------------------------------------------------- #
# Degenerate one-rank equivalence: a single shard is the whole dataset.
# --------------------------------------------------------------------------- #


def _dense(t):
    """Densify a sparse tensor of any layout; pass dense tensors through."""
    return t if t.layout == torch.strided else t.to_dense()


def _fit(layout, X, seed=0):
    seed_everything(seed)
    model = UMAP(
        input_layout=layout,
        init="random",
        backend="faiss",
        device="cpu",
        n_neighbors=10,
        max_iter=25,
        random_state=seed,
    )
    emb = model.fit_transform(X)
    return model, torch.as_tensor(emb)


def _umap_affinity(layout, X):
    aff = UMAPAffinity(
        n_neighbors=10, backend="faiss", device="cpu", input_layout=layout
    )
    P, idx = aff(X, return_indices=True)
    return aff, P, idx


def test_single_process_sharded_affinity_matches_replicated():
    """The deterministic symmetrized affinity is identical in the one-rank case.

    A single shard is the whole dataset, so resolving the layout and running the
    per-shard Flat search must reproduce the replicated exact search bit-for-bit.
    Comparing here (rather than on the fitted UMAP) avoids UMAP's post-build
    deletion of ``affinity_in_``/``NN_indices_``.
    """
    X = _blobs(120, 16, seed=1)

    aff_rep, P_rep, idx_rep = _umap_affinity("replicated", X)
    aff_shard, P_shard, idx_shard = _umap_affinity("sharded", X)

    # The sharded affinity records the global sample total.
    assert aff_shard.n_global_ == X.shape[0]

    # Symmetrized memberships (order-independent) must coincide, and the neighbor
    # sets per row must match (sorted to ignore any column permutation).
    torch.testing.assert_close(_dense(P_shard), _dense(P_rep), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        idx_shard.sort(dim=1).values, idx_rep.sort(dim=1).values, rtol=0, atol=0
    )


def test_single_process_sharded_embedding_matches_replicated():
    """End-to-end: with no process group the fitted embedding coincides."""
    X = _blobs(120, 16, seed=1)

    rep_model, rep_emb = _fit("replicated", X)
    shard_model, shard_emb = _fit("sharded", X)

    # Global-N bookkeeping is unchanged in the one-rank case.
    assert shard_model.n_samples_in_ == X.shape[0]
    assert shard_emb.shape == (X.shape[0], 2)

    # Same seed + identical affinity + identical single-process optimization ->
    # the embeddings coincide.
    torch.testing.assert_close(shard_emb, rep_emb, rtol=1e-4, atol=1e-4)


def test_single_process_sharded_embedding_is_finite():
    X = _blobs(90, 12, seed=3)
    _, emb = _fit("sharded", X)
    assert torch.isfinite(emb).all()
