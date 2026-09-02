"""Tests for sparse utility functions."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import threading
from types import SimpleNamespace

import pytest
import torch

import torchdr.utils.sparse as sparse_utils
from torchdr.distributed import DistributedContext
from torchdr.utils.sparse import (
    distributed_symmetrize_sparse,
    flatten_sparse,
    merge_symmetry,
    pack_to_rowwise,
    symmetrize_sparse,
    _combine_P_PT,
)


def _rowwise_to_dense(values, indices, n_columns):
    """Convert TorchDR's padded row-wise representation to a dense tensor."""
    dense = torch.zeros((values.shape[0], n_columns), dtype=values.dtype)
    valid = indices >= 0
    dense.scatter_add_(1, indices.clamp_min(0), values * valid)
    return dense


class _SimulatedProcessGroup:
    """Thread-backed stand-in for the all-to-all collectives.

    Every simulated rank runs the production code path in its own thread and the
    collectives rendezvous on a barrier, so each rank receives exactly what a
    real process group would deliver. The barrier is bounded so a rank that
    stops participating surfaces as a test failure instead of a hang.
    """

    def __init__(self, world_size, timeout=60.0):
        self.world_size = world_size
        self._barrier = threading.Barrier(world_size, timeout=timeout)
        self._slots = [None] * world_size
        self._state = threading.local()

    def set_rank(self, rank):
        self._state.rank = rank

    def abort(self):
        self._barrier.abort()

    def all_to_all_single(self, output, input_):
        rank = self._state.rank
        self._slots[rank] = input_.clone()
        self._barrier.wait()
        received = torch.stack(
            [self._slots[source][rank] for source in range(self.world_size)]
        )
        self._barrier.wait()
        output.copy_(received)

    def all_to_all(self, outputs, inputs):
        rank = self._state.rank
        self._slots[rank] = [tensor.clone() for tensor in inputs]
        self._barrier.wait()
        for source in range(self.world_size):
            outputs[source].copy_(self._slots[source][rank])
        self._barrier.wait()


def _run_multi_rank_symmetrize(monkeypatch, values, indices, world_size, mode):
    """Run the distributed path on every rank and return the per-rank outputs.

    Rows are partitioned with the same helper the library uses, so the exchange
    exercises the real agreement between ``compute_chunk_bounds`` and
    ``get_rank_for_indices``.
    """
    n_total = values.shape[0]
    group = _SimulatedProcessGroup(world_size)
    outputs = [None] * world_size
    failures = [None] * world_size

    def worker(rank):
        group.set_rank(rank)
        try:
            chunk_start, chunk_end = DistributedContext.compute_chunk_bounds(
                SimpleNamespace(rank=rank, world_size=world_size), n_total
            )
            outputs[rank] = distributed_symmetrize_sparse(
                values[chunk_start:chunk_end].clone(),
                indices[chunk_start:chunk_end].clone(),
                chunk_start=chunk_start,
                chunk_size=chunk_end - chunk_start,
                n_total=n_total,
                mode=mode,
            )
        except BaseException as error:  # noqa: BLE001 - re-raised by the caller
            failures[rank] = error
            group.abort()

    monkeypatch.setattr(sparse_utils.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(sparse_utils.dist, "get_world_size", lambda: world_size)
    monkeypatch.setattr(sparse_utils.dist, "all_to_all_single", group.all_to_all_single)
    monkeypatch.setattr(sparse_utils.dist, "all_to_all", group.all_to_all)

    threads = [
        threading.Thread(target=worker, args=(rank,)) for rank in range(world_size)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120.0)
        assert not thread.is_alive(), "a simulated rank did not finish"

    for error in failures:
        if error is not None:
            raise error
    return outputs


def _gathered_dense(outputs, n_total):
    """Concatenate the per-rank row blocks into the full dense matrix."""
    return torch.cat(
        [_rowwise_to_dense(values, indices, n_total) for values, indices in outputs]
    )


def _random_sparse_graph(n, k, seed, dtype=torch.float32):
    """Build a deterministic k-nearest-neighbour style sparse matrix."""
    generator = torch.Generator().manual_seed(seed)
    indices = torch.stack(
        [torch.randperm(n, generator=generator)[:k] for _ in range(n)]
    ).long()
    values = torch.rand((n, k), generator=generator, dtype=dtype).clamp_min(1e-3)
    return values, indices


@pytest.fixture
def mock_single_rank_collectives(monkeypatch):
    """Replace distributed collectives with one-rank copies."""
    exchanged_dtypes = []

    def fake_all_to_all_single(output, input_):
        output.copy_(input_)

    def fake_all_to_all(outputs, inputs):
        exchanged_dtypes.append(tuple(tensor.dtype for tensor in inputs))
        for output, input_ in zip(outputs, inputs):
            output.copy_(input_)

    monkeypatch.setattr(sparse_utils.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(sparse_utils.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(sparse_utils.dist, "all_to_all_single", fake_all_to_all_single)
    monkeypatch.setattr(sparse_utils.dist, "all_to_all", fake_all_to_all)
    return exchanged_dtypes


class TestFlattenSparse:
    """Tests for flatten_sparse function."""

    def test_basic(self):
        """Test basic flattening of sparse representation."""
        values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        indices = torch.tensor([[0, 1], [2, 3]])

        i, j, v = flatten_sparse(values, indices)

        assert i.tolist() == [0, 0, 1, 1]
        assert j.tolist() == [0, 1, 2, 3]
        assert v.tolist() == [1.0, 2.0, 3.0, 4.0]

    def test_single_row(self):
        """Test with single row."""
        values = torch.tensor([[1.0, 2.0, 3.0]])
        indices = torch.tensor([[5, 10, 15]])

        i, j, v = flatten_sparse(values, indices)

        assert i.tolist() == [0, 0, 0]
        assert j.tolist() == [5, 10, 15]
        assert v.tolist() == [1.0, 2.0, 3.0]


class TestMergeSymmetry:
    """Tests for merge_symmetry function."""

    def test_symmetric_entries(self):
        """Test merging with symmetric (duplicate) entries."""
        # P has (0,1)=1.0 and (1,0)=2.0
        i = torch.tensor([0, 1])
        j = torch.tensor([1, 0])
        v = torch.tensor([1.0, 2.0])

        i_out, j_out, vP, vPT = merge_symmetry(i, j, v, n=2)

        # Should have unique position (0,1) with P=1.0, PT=2.0
        # and (1,0) with P=2.0, PT=1.0
        assert i_out.numel() == 2
        assert vP.sum().item() == 3.0  # 1.0 + 2.0
        assert vPT.sum().item() == 3.0  # 2.0 + 1.0

    def test_diagonal_entries(self):
        """Test that diagonal entries are handled correctly."""
        # Diagonal entry (i,i) should have P=PT
        i = torch.tensor([0])
        j = torch.tensor([0])
        v = torch.tensor([5.0])

        i_out, j_out, vP, vPT = merge_symmetry(i, j, v, n=2)

        assert i_out.tolist() == [0]
        assert j_out.tolist() == [0]
        assert vP.item() == 5.0
        assert vPT.item() == 5.0


class TestPackToRowwise:
    """Tests for pack_to_rowwise function."""

    def test_basic(self):
        """Test basic packing to row-wise format."""
        i_out = torch.tensor([0, 0, 1])
        j_out = torch.tensor([1, 2, 0])
        v_out = torch.tensor([1.0, 2.0, 3.0])

        values, indices = pack_to_rowwise(i_out, j_out, v_out, n=2)

        assert values.shape == (2, 2)
        assert indices.shape == (2, 2)
        assert values[0].tolist() == [1.0, 2.0]
        assert values[1, 0].item() == 3.0

    def test_empty(self):
        """Test with empty input."""
        i_out = torch.tensor([], dtype=torch.long)
        j_out = torch.tensor([], dtype=torch.long)
        v_out = torch.tensor([])

        values, indices = pack_to_rowwise(i_out, j_out, v_out, n=3)

        assert values.shape == (3, 0)
        assert indices.shape == (3, 0)


class TestCombinePPT:
    """Tests for _combine_P_PT helper function."""

    def test_sum_mode(self):
        """Test sum mode: P + P^T."""
        vP = torch.tensor([1.0, 2.0])
        vPT = torch.tensor([3.0, 4.0])

        result = _combine_P_PT(vP, vPT, mode="sum")

        assert result.tolist() == [4.0, 6.0]

    def test_sum_minus_prod_mode(self):
        """Test sum_minus_prod mode: P + P^T - P*P^T."""
        vP = torch.tensor([0.5, 0.2])
        vPT = torch.tensor([0.3, 0.4])

        result = _combine_P_PT(vP, vPT, mode="sum_minus_prod")

        expected = [0.5 + 0.3 - 0.5 * 0.3, 0.2 + 0.4 - 0.2 * 0.4]
        torch.testing.assert_close(result, torch.tensor(expected))

    def test_invalid_mode(self):
        """Test that invalid mode raises error."""
        with pytest.raises(ValueError, match="Unsupported mode"):
            _combine_P_PT(torch.tensor([1.0]), torch.tensor([1.0]), mode="invalid")


class TestSymmetrizeSparse:
    """Tests for symmetrize_sparse function."""

    def test_basic_symmetrization(self):
        """Test basic symmetrization of a sparse matrix."""
        # 3x3 matrix with edges (0,1) and (1,2)
        values = torch.tensor([[1.0], [2.0], [0.0]])
        indices = torch.tensor([[1], [2], [0]])

        values_out, indices_out = symmetrize_sparse(values, indices, mode="sum")

        # Should have symmetric edges
        assert values_out.shape[0] == 3

    def test_sum_mode(self):
        """Test sum mode produces P + P^T."""
        # Edge (0,1)=1.0 and (1,0)=2.0
        values = torch.tensor([[1.0], [2.0]])
        indices = torch.tensor([[1], [0]])

        values_out, indices_out = symmetrize_sparse(values, indices, mode="sum")

        # After symmetrization, (0,1) and (1,0) should both be 3.0
        assert values_out.shape[0] == 2

    def test_sum_minus_prod_preserves_range(self):
        """Test that sum_minus_prod keeps values in [0, 1] for inputs in [0, 1]."""
        torch.manual_seed(42)
        n, k = 10, 3
        values = torch.rand(n, k)
        indices = torch.randint(0, n, (n, k))

        values_out, _ = symmetrize_sparse(values, indices, mode="sum_minus_prod")

        # With inputs in [0,1], sum_minus_prod should be in [0,1]
        assert values_out.min() >= -1e-6
        assert values_out.max() <= 2.0 + 1e-6


class TestDistributedSymmetrizeSparse:
    """Tests for the distributed symmetrization path."""

    def test_requires_initialized_process_group(self, monkeypatch):
        """Distributed symmetrization should fail before any collective."""
        monkeypatch.setattr(sparse_utils.dist, "is_initialized", lambda: False)

        with pytest.raises(RuntimeError, match="torch.distributed"):
            distributed_symmetrize_sparse(
                torch.ones(2, 1),
                torch.tensor([[1], [0]]),
                chunk_start=0,
                chunk_size=2,
                n_total=2,
            )

    def test_matches_centralized_result_and_preserves_dtypes(
        self, mock_single_rank_collectives
    ):
        """Reciprocal edges must retain distinct P and P-transpose values."""
        values = torch.tensor(
            [[0.2, 0.1], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]],
            dtype=torch.float64,
        )
        indices = torch.tensor([[1, 3], [0, 2], [1, 3], [2, 0]])

        actual_values, actual_indices = distributed_symmetrize_sparse(
            values,
            indices,
            chunk_start=0,
            chunk_size=4,
            n_total=4,
            mode="sum_minus_prod",
        )
        expected_values, expected_indices = symmetrize_sparse(
            values, indices, mode="sum_minus_prod"
        )

        torch.testing.assert_close(
            _rowwise_to_dense(actual_values, actual_indices, 4),
            _rowwise_to_dense(expected_values, expected_indices, 4),
        )
        assert actual_values.dtype == torch.float64
        assert actual_indices.dtype == torch.int64
        assert mock_single_rank_collectives == [
            (torch.int64,),
            (torch.float64,),
        ]

    def test_large_indices_are_not_rounded(self, mock_single_rank_collectives):
        """Collectives must preserve indices above float32's exact range."""
        large_index = 2**24 + 1
        values = torch.tensor([[0.125]], dtype=torch.float64)
        indices = torch.tensor([[large_index]], dtype=torch.long)

        values_out, indices_out = distributed_symmetrize_sparse(
            values,
            indices,
            chunk_start=0,
            chunk_size=1,
            n_total=large_index + 1,
            mode="sum",
        )

        assert mock_single_rank_collectives == [
            (torch.int64,),
            (torch.float64,),
        ]
        assert values_out.dtype == torch.float64
        assert indices_out[0, 0].item() == large_index


class TestDistributedSymmetrizeSparseMultiRank:
    """Multi-rank equivalence between the distributed and centralized paths.

    A single rank routes every edge back to itself, so it cannot observe whether
    the forward and reverse contributions of a cross-rank pair stay in separate
    streams. These cases run the production code on two or more ranks that
    genuinely exchange edges.
    """

    def test_cross_rank_pair_keeps_the_product_term(self, monkeypatch):
        """Reciprocal edges split across ranks must not lose ``P * Pᵀ``.

        Rank 0 owns rows 0-1 and rank 1 owns rows 2-3, so the reciprocal pair
        ``(0, 2) = 0.2`` / ``(2, 0) = 0.4`` crosses the rank boundary while the
        one-way edge ``(1, 0) = 0.3`` stays local.
        """
        values = torch.tensor(
            [[0.2, 0.0], [0.3, 0.0], [0.4, 0.0], [0.0, 0.0]], dtype=torch.float64
        )
        indices = torch.tensor([[2, 3], [0, 3], [0, 1], [1, 2]])

        outputs = _run_multi_rank_symmetrize(
            monkeypatch, values, indices, world_size=2, mode="sum_minus_prod"
        )
        actual = _gathered_dense(outputs, 4)

        # Cross-rank reciprocal pair: 0.2 + 0.4 - 0.2 * 0.4.
        assert actual[0, 2].item() == pytest.approx(0.52)
        # Same-rank one-way edge must be counted exactly once.
        assert actual[1, 0].item() == pytest.approx(0.30)

        expected_values, expected_indices = symmetrize_sparse(
            values, indices, mode="sum_minus_prod"
        )
        torch.testing.assert_close(
            actual, _rowwise_to_dense(expected_values, expected_indices, 4)
        )

    @pytest.mark.parametrize("mode", ["sum", "sum_minus_prod"])
    @pytest.mark.parametrize(
        "n,k,world_size",
        [
            (16, 3, 2),  # even partition
            (17, 4, 3),  # uneven partition, 6/6/5 rows
            (23, 5, 4),  # uneven partition, 6/6/6/5 rows
            (64, 8, 2),
        ],
    )
    def test_matches_centralized_reference(self, monkeypatch, n, k, world_size, mode):
        """Gathered multi-rank output must equal the single-process result."""
        values, indices = _random_sparse_graph(n, k, seed=n * 10 + world_size)

        outputs = _run_multi_rank_symmetrize(
            monkeypatch, values, indices, world_size=world_size, mode=mode
        )
        expected_values, expected_indices = symmetrize_sparse(
            values, indices, mode=mode
        )

        torch.testing.assert_close(
            _gathered_dense(outputs, n),
            _rowwise_to_dense(expected_values, expected_indices, n),
        )

    @pytest.mark.parametrize(
        "structure", ["self_loops", "duplicate_columns", "one_way_ring"]
    )
    def test_degenerate_edge_structures(self, monkeypatch, structure):
        """Self-loops, repeated columns and one-way edges must stay exact."""
        n, world_size = 12, 3
        if structure == "self_loops":
            values, indices = _random_sparse_graph(n, 3, seed=101)
            indices[:, 0] = torch.arange(n)
        elif structure == "duplicate_columns":
            values, indices = _random_sparse_graph(n, 4, seed=102)
            indices[:, 1] = indices[:, 0]
        else:
            indices = torch.stack(
                [torch.tensor([(row + 1) % n, (row + 2) % n]) for row in range(n)]
            )
            generator = torch.Generator().manual_seed(103)
            values = torch.rand((n, 2), generator=generator) + 0.1

        outputs = _run_multi_rank_symmetrize(
            monkeypatch, values, indices, world_size=world_size, mode="sum_minus_prod"
        )
        expected_values, expected_indices = symmetrize_sparse(
            values, indices, mode="sum_minus_prod"
        )

        torch.testing.assert_close(
            _gathered_dense(outputs, n),
            _rowwise_to_dense(expected_values, expected_indices, n),
        )

    def test_hub_heavy_graph_matches_reference(self, monkeypatch):
        """A few hub columns concentrate the exchange on a single owner rank."""
        n, k, world_size = 48, 7, 3
        generator = torch.Generator().manual_seed(104)
        weights = 1.0 / (torch.arange(n, dtype=torch.float64) + 1.0)
        indices = torch.multinomial(
            weights.expand(n, n), k, replacement=False, generator=generator
        ).long()
        values = torch.rand((n, k), generator=generator).clamp_min(1e-3)

        outputs = _run_multi_rank_symmetrize(
            monkeypatch, values, indices, world_size=world_size, mode="sum_minus_prod"
        )
        expected_values, expected_indices = symmetrize_sparse(
            values, indices, mode="sum_minus_prod"
        )

        torch.testing.assert_close(
            _gathered_dense(outputs, n),
            _rowwise_to_dense(expected_values, expected_indices, n),
        )

    def test_preserves_dtypes_across_ranks(self, monkeypatch):
        """float64 values and int64 indices must survive the exchange."""
        n, world_size = 31, 3
        values, indices = _random_sparse_graph(n, 6, seed=105, dtype=torch.float64)

        outputs = _run_multi_rank_symmetrize(
            monkeypatch, values, indices, world_size=world_size, mode="sum"
        )

        for rank_values, rank_indices in outputs:
            assert rank_values.dtype == torch.float64
            assert rank_indices.dtype == torch.int64

        expected_values, expected_indices = symmetrize_sparse(
            values, indices, mode="sum"
        )
        torch.testing.assert_close(
            _gathered_dense(outputs, n),
            _rowwise_to_dense(expected_values, expected_indices, n),
        )

    def test_is_deterministic_across_repetitions(self, monkeypatch):
        """Repeated runs of the same partition must be bitwise identical."""
        n, world_size = 40, 3
        values, indices = _random_sparse_graph(n, 5, seed=106)

        reference = None
        for _ in range(3):
            outputs = _run_multi_rank_symmetrize(
                monkeypatch,
                values,
                indices,
                world_size=world_size,
                mode="sum_minus_prod",
            )
            gathered = _gathered_dense(outputs, n)
            if reference is None:
                reference = gathered
            else:
                assert torch.equal(reference, gathered)
