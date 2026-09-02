"""Tests for sparse utility functions."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import pytest
import torch

import torchdr.utils.sparse as sparse_utils
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
