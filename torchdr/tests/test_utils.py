"""Tests for functions in utils module."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import numpy as np
import pytest
import torch
from torch.testing import assert_close

from torchdr.distance import (
    LIST_METRICS_TORCH,
    LIST_METRICS_KEOPS,
    LIST_METRICS_FAISS,
    pairwise_distances,
    symmetric_pairwise_distances_indices,
)
from torchdr.utils import (
    binary_search,
    center_kernel,
    validate_tensor,
    check_shape,
    check_similarity,
    check_similarity_torch_keops,
    false_position,
    to_torch,
    pykeops,
    faiss,
    RiemannianAdam,
    ManifoldParameter,
    EuclideanManifold,
    PoincareBallManifold,
    matrix_power,
    identity_matrix,
)

lst_types = [torch.double, torch.float]


# ====== test root finding methods ======


@pytest.mark.parametrize("dtype", lst_types)
def test_binary_search(dtype):
    def f(x):
        return x**2 - 1

    # --- test 1D, with begin as scalar ---
    begin = 0.5
    end = 2.0

    m = binary_search(f, 1, begin=begin, end=end, max_iter=1000, dtype=dtype)
    assert_close(m, torch.tensor([1.0], dtype=dtype), rtol=1e-5, atol=1e-5)

    # --- test 2D, with begin as tensor ---
    begin = 0.5 * torch.ones(2, dtype=torch.float16)
    end = 2.0

    m = binary_search(f, 2, begin=begin, end=end, max_iter=1000, dtype=dtype)
    assert_close(m, torch.tensor([1.0, 1.0], dtype=dtype), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dtype", lst_types)
def test_false_position(dtype):
    def f(x):
        return x**2 - 1

    # --- test 1D, with end as scalar ---
    begin = 0.5
    end = 2

    m = false_position(f, 1, begin=begin, end=end, max_iter=1000, dtype=dtype)
    assert_close(m, torch.tensor([1.0], dtype=dtype), rtol=1e-5, atol=1e-5)

    # --- test 2D, with end as tensor ---
    begin = 0.5
    end = 2 * torch.ones(2, dtype=torch.int)

    m = false_position(f, 2, begin=begin, end=end, max_iter=1000, dtype=dtype)
    assert_close(m, torch.tensor([1.0, 1.0], dtype=dtype), rtol=1e-5, atol=1e-5)


# ====== test pairwise distance matrices ======


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("metric", LIST_METRICS_TORCH)
def test_pairwise_distances(dtype, metric):
    n, m, p = 100, 50, 10
    x = torch.randn(n, p, dtype=dtype)
    y = torch.randn(m, p, dtype=dtype)

    x = x / x.max() - 0.1
    y = y / y.max() - 0.1

    C = pairwise_distances(x, y, metric=metric, backend=None)
    check_shape(C, (n, m))


@pytest.mark.skipif(not pykeops, reason="pykeops is not available")
@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("metric", LIST_METRICS_KEOPS)
def test_pairwise_distances_keops(dtype, metric):
    n, m, p = 100, 50, 10
    x = torch.randn(n, p, dtype=dtype)
    y = torch.randn(m, p, dtype=dtype)

    # --- check consistency between torch and keops ---
    C = pairwise_distances(x, y, metric=metric, backend=None)
    C_keops = pairwise_distances(x, y, metric=metric, backend="keops")
    check_shape(C_keops, (n, m))

    check_similarity_torch_keops(C, C_keops, K=10)

    # --- check consistency between torch and keops with kNN search ---
    k = 10
    C = pairwise_distances(x, y, k=k, metric=metric, backend=None)
    C_keops = pairwise_distances(x, y, k=k, metric=metric, backend="keops")
    check_shape(C_keops, (n, k))

    torch.testing.assert_close(C, C_keops, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not faiss, reason="faiss is not available")
@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("metric", LIST_METRICS_FAISS)
@pytest.mark.parametrize("exclude_diag", [True, False])
def test_pairwise_distances_faiss(dtype, metric, exclude_diag):
    n, m, p = 100, 50, 10
    x = torch.randn(n, p, dtype=dtype)
    y = torch.randn(m, p, dtype=dtype)

    # --- check consistency between torch and faiss ---
    k = 10
    C = pairwise_distances(
        x, y, k=k, metric=metric, backend=None, exclude_diag=exclude_diag
    )
    C_faiss = pairwise_distances(
        x, y, k=k, metric=metric, backend="faiss", exclude_diag=exclude_diag
    )
    check_shape(C_faiss, (n, k))

    torch.testing.assert_close(C, C_faiss, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("metric", LIST_METRICS_TORCH)
def test_symmetric_pairwise_distances_indices(dtype, metric):
    n, p = 100, 20
    x = torch.randn(n, p, dtype=dtype)
    indices = torch.randint(0, n, (n, 10))

    # --- check consistency with symmetric_pairwise_distances ---
    C_indices = symmetric_pairwise_distances_indices(x, indices, metric=metric)
    check_shape(C_indices, (n, 10))

    C_full = pairwise_distances(x, metric=metric, backend=None)
    C_full_indices = C_full.gather(1, indices)

    check_similarity(C_indices, C_full_indices)


# ====== test center_kernel ======


@pytest.mark.parametrize("dtype", lst_types)
def test_center_kernel(dtype):
    torch.manual_seed(0)
    X = torch.randn(20, 30, dtype=dtype)
    K = X @ X.T
    K_c = center_kernel(K)
    n = K.shape[0]
    ones_n = torch.ones(n, dtype=dtype)
    H = torch.eye(n, dtype=dtype) - torch.outer(ones_n, ones_n) / n
    torch.testing.assert_close(K_c, H @ K @ H)


# ====== test radam functions ======


class TestRiemannianAdam:
    """Test RiemannianAdam optimizer."""

    @pytest.mark.parametrize("dtype", lst_types)
    def test_init_default_params(self, dtype):
        """Test RiemannianAdam initialization with default parameters."""
        param = torch.randn(5, 3, dtype=dtype, requires_grad=True)
        optimizer = RiemannianAdam([param])

        assert len(optimizer.param_groups) == 1
        assert optimizer.param_groups[0]["lr"] == 1e-3
        assert optimizer.param_groups[0]["betas"] == (0.9, 0.999)
        assert optimizer.param_groups[0]["eps"] == 1e-8
        assert optimizer.param_groups[0]["weight_decay"] == 0
        assert optimizer.param_groups[0]["amsgrad"] is False
        assert optimizer._stabilize is None

    @pytest.mark.parametrize("dtype", lst_types)
    def test_init_custom_params(self, dtype):
        """Test RiemannianAdam initialization with custom parameters."""
        param = torch.randn(5, 3, dtype=dtype, requires_grad=True)
        optimizer = RiemannianAdam(
            [param],
            lr=1e-2,
            betas=(0.8, 0.99),
            eps=1e-6,
            weight_decay=1e-4,
            amsgrad=True,
            stabilize=5,
        )

        assert optimizer.param_groups[0]["lr"] == 1e-2
        assert optimizer.param_groups[0]["betas"] == (0.8, 0.99)
        assert optimizer.param_groups[0]["eps"] == 1e-6
        assert optimizer.param_groups[0]["weight_decay"] == 1e-4
        assert optimizer.param_groups[0]["amsgrad"] is True
        assert optimizer._stabilize == 5

    @pytest.mark.parametrize("dtype", lst_types)
    def test_step_euclidean_manifold(self, dtype):
        """Test optimization step with Euclidean manifold (standard parameters)."""
        # Create a simple quadratic loss function
        x = torch.randn(5, 3, dtype=dtype, requires_grad=True)
        target = torch.randn(5, 3, dtype=dtype)

        optimizer = RiemannianAdam([x], lr=1e-2)

        initial_x = x.clone()
        loss = torch.sum((x - target) ** 2)
        loss.backward()

        # Ensure gradients are not None
        assert x.grad is not None

        optimizer.step()

        # Parameter should have changed
        assert not torch.allclose(x, initial_x)

        # Check optimizer state was created
        assert len(optimizer.state) == 1
        state = optimizer.state[x]
        assert "step" in state
        assert "exp_avg" in state
        assert "exp_avg_sq" in state
        assert state["step"] == 0  # step counter is tracked in param_groups

    @pytest.mark.parametrize("dtype", lst_types)
    def test_step_manifold_parameter(self, dtype):
        """Test optimization step with ManifoldParameter."""
        manifold = EuclideanManifold()
        data = torch.randn(3, 2, dtype=dtype)
        param = ManifoldParameter(data, requires_grad=True, manifold=manifold, c=None)
        target = torch.randn(3, 2, dtype=dtype)

        optimizer = RiemannianAdam([param], lr=1e-2)

        initial_param = param.clone()
        loss = torch.sum((param - target) ** 2)
        loss.backward()

        optimizer.step()

        # Parameter should have changed
        assert not torch.allclose(param, initial_param)

    @pytest.mark.parametrize("dtype", lst_types)
    def test_step_poincare_manifold_parameter(self, dtype):
        """Test optimization step with PoincareBall ManifoldParameter."""
        manifold = PoincareBallManifold()
        # Initialize points in the Poincare ball (norm < 1)
        data = torch.randn(3, 2, dtype=dtype) * 0.1  # Small values to stay in ball
        param = ManifoldParameter(data, requires_grad=True, manifold=manifold, c=1.0)
        target = torch.randn(3, 2, dtype=dtype) * 0.1

        optimizer = RiemannianAdam([param], lr=1e-3)

        initial_param = param.clone()
        loss = torch.sum((param - target) ** 2)
        loss.backward()

        optimizer.step()

        # Parameter should have changed
        assert not torch.allclose(param, initial_param)

    @pytest.mark.parametrize("dtype", lst_types)
    def test_step_with_amsgrad(self, dtype):
        """Test optimization step with AMSGrad variant."""
        x = torch.randn(3, 2, dtype=dtype, requires_grad=True)
        target = torch.randn(3, 2, dtype=dtype)

        optimizer = RiemannianAdam([x], lr=1e-2, amsgrad=True)

        loss = torch.sum((x - target) ** 2)
        loss.backward()

        optimizer.step()

        # Check that max_exp_avg_sq is created in state
        state = optimizer.state[x]
        assert "max_exp_avg_sq" in state

    def test_step_with_none_gradients(self):
        """Test optimization step when some parameters have None gradients."""
        x1 = torch.randn(2, 2, requires_grad=True)
        x2 = torch.randn(2, 2, requires_grad=True)

        optimizer = RiemannianAdam([x1, x2])

        # Only compute gradient for x1
        loss = torch.sum(x1**2)
        loss.backward()

        # x2.grad should be None
        assert x2.grad is None

        # Should not raise error
        optimizer.step()

        # Only x1 should have optimizer state
        assert x1 in optimizer.state
        assert x2 not in optimizer.state

    def test_step_with_sparse_gradients(self):
        """Test that sparse gradients raise appropriate error."""
        x = torch.randn(5, 3, requires_grad=True)
        optimizer = RiemannianAdam([x])

        # Create a sparse gradient manually
        indices = torch.LongTensor([[0, 1], [2, 3]])
        values = torch.FloatTensor([1, 2])
        x.grad = torch.sparse_coo_tensor(indices, values, (5, 3))

        with pytest.raises(
            RuntimeError, match="Riemannian Adam does not support sparse gradients"
        ):
            optimizer.step()

    @pytest.mark.parametrize("dtype", lst_types)
    def test_step_with_closure(self, dtype):
        """Test optimization step with closure function."""
        x = torch.randn(3, 2, dtype=dtype, requires_grad=True)
        target = torch.randn(3, 2, dtype=dtype)

        optimizer = RiemannianAdam([x], lr=1e-2)

        def closure():
            optimizer.zero_grad()
            loss = torch.sum((x - target) ** 2)
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        assert loss is not None
        assert isinstance(loss, torch.Tensor)

    @pytest.mark.parametrize("dtype", lst_types)
    def test_multiple_steps(self, dtype):
        """Test multiple optimization steps to ensure state is properly maintained."""
        x = torch.randn(3, 2, dtype=dtype, requires_grad=True)
        target = torch.randn(3, 2, dtype=dtype)

        optimizer = RiemannianAdam([x], lr=1e-2)

        losses = []
        for i in range(5):
            optimizer.zero_grad()
            loss = torch.sum((x - target) ** 2)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should generally decrease
        assert losses[-1] < losses[0]

        # Check step count
        assert (
            optimizer.param_groups[0]["step"] == 10
        )  # 2 * 5 (incremented twice per step)

    @pytest.mark.parametrize("dtype", lst_types)
    def test_stabilize_group(self, dtype):
        """Test stabilize_group method with ManifoldParameters."""
        manifold = EuclideanManifold()
        data = torch.randn(3, 2, dtype=dtype)
        param = ManifoldParameter(data, requires_grad=True, manifold=manifold, c=None)

        optimizer = RiemannianAdam([param], lr=1e-2)

        # Create some optimizer state
        loss = torch.sum(param**2)
        loss.backward()
        optimizer.step()

        # Test stabilize_group method
        group = optimizer.param_groups[0]
        optimizer.stabilize_group(group)  # Should not raise error

    @pytest.mark.parametrize("dtype", lst_types)
    def test_stabilize_group_empty_state(self, dtype):
        """Test stabilize_group method with empty state."""
        manifold = EuclideanManifold()
        data = torch.randn(3, 2, dtype=dtype)
        param = ManifoldParameter(data, requires_grad=True, manifold=manifold, c=None)

        optimizer = RiemannianAdam([param], lr=1e-2)

        # Test stabilize_group with empty state
        group = optimizer.param_groups[0]
        optimizer.stabilize_group(group)  # Should not raise error

    @pytest.mark.parametrize("dtype", lst_types)
    def test_stabilize_group_regular_parameter(self, dtype):
        """Test stabilize_group method with regular (non-manifold) parameters."""
        param = torch.randn(3, 2, dtype=dtype, requires_grad=True)
        optimizer = RiemannianAdam([param], lr=1e-2)

        # Create some optimizer state
        loss = torch.sum(param**2)
        loss.backward()
        optimizer.step()

        # Test stabilize_group method - should skip regular parameters
        group = optimizer.param_groups[0]
        optimizer.stabilize_group(group)  # Should not raise error

    @pytest.mark.parametrize("dtype", lst_types)
    def test_stabilize_automatic(self, dtype):
        """Test automatic stabilization based on stabilize parameter."""
        manifold = EuclideanManifold()
        data = torch.randn(3, 2, dtype=dtype)
        param = ManifoldParameter(data, requires_grad=True, manifold=manifold, c=None)

        # Set stabilize to trigger every 2 steps
        optimizer = RiemannianAdam([param], lr=1e-2, stabilize=2)

        # Mock the stabilize_group method to track calls
        stabilize_calls = 0
        original_stabilize_group = optimizer.stabilize_group

        def mock_stabilize_group(group):
            nonlocal stabilize_calls
            stabilize_calls += 1
            return original_stabilize_group(group)

        optimizer.stabilize_group = mock_stabilize_group

        # Run several optimization steps
        for i in range(5):
            optimizer.zero_grad()
            loss = torch.sum(param**2)
            loss.backward()
            optimizer.step()

        # Stabilize should have been called when step count is divisible by stabilize value
        # Note: step count is incremented twice per step, so after 5 steps we have 10 total steps
        # Stabilization should occur at steps 2, 4, 6, 8, 10
        assert stabilize_calls == 5

    @pytest.mark.parametrize("dtype", lst_types)
    def test_weight_decay(self, dtype):
        """Test weight decay functionality."""
        # Create initial data that we'll use for both optimizers
        initial_data = torch.randn(3, 2, dtype=dtype)
        target = torch.randn(3, 2, dtype=dtype)

        # Create two separate leaf tensors with the same initial values
        x_with_decay = initial_data.clone().detach().requires_grad_(True)
        x_no_decay = initial_data.clone().detach().requires_grad_(True)

        # Create optimizers
        optimizer_with_decay = RiemannianAdam(
            [x_with_decay], lr=1e-2, weight_decay=1e-2
        )
        optimizer_no_decay = RiemannianAdam([x_no_decay], lr=1e-2, weight_decay=0)

        # Compute gradients for both (should be identical initially)
        loss_with_decay = torch.sum((x_with_decay - target) ** 2)
        loss_with_decay.backward()

        loss_no_decay = torch.sum((x_no_decay - target) ** 2)
        loss_no_decay.backward()

        # Store initial positions
        initial_x_with_decay = x_with_decay.clone().detach()
        initial_x_no_decay = x_no_decay.clone().detach()

        # Perform optimization steps
        optimizer_with_decay.step()
        optimizer_no_decay.step()

        # The parameters should have moved differently due to weight decay
        # With weight decay, the effective gradient includes an additional term
        assert not torch.allclose(x_with_decay, x_no_decay, atol=1e-6)

        # Both should have moved from their initial positions
        assert not torch.allclose(x_with_decay, initial_x_with_decay)
        assert not torch.allclose(x_no_decay, initial_x_no_decay)


# ====== test manifold functions ======


@pytest.mark.parametrize("dtype", lst_types)
def test_manifold_parameter(dtype):
    """Test ManifoldParameter creation and basic functionality."""
    manifold = EuclideanManifold()
    data = torch.randn(3, 2, dtype=dtype)
    c = 1.0

    # Test creation
    param = ManifoldParameter(data, requires_grad=True, manifold=manifold, c=c)

    # Test attributes
    assert param.manifold is manifold
    assert param.c == c
    assert param.requires_grad is True
    assert param.shape == (3, 2)
    assert param.dtype == dtype

    # Test __repr__ method
    repr_str = repr(param)
    assert "Euclidean Parameter containing:" in repr_str


class TestEuclideanManifold:
    """Test EuclideanManifold class."""

    @pytest.mark.parametrize("dtype", lst_types)
    def test_init(self, dtype):
        """Test EuclideanManifold initialization."""
        manifold = EuclideanManifold()
        assert manifold.name == "Euclidean"
        assert manifold.eps == 10e-8

    @pytest.mark.parametrize("dtype", lst_types)
    def test_normalize(self, dtype):
        """Test normalize method."""
        manifold = EuclideanManifold()
        p = torch.randn(5, 3, dtype=dtype) * 10  # Large values to test normalization

        normalized = manifold.normalize(p.clone())
        norms = torch.norm(normalized, dim=-1)

        # All vectors should have unit norm
        torch.testing.assert_close(norms, torch.ones_like(norms), rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("dtype", lst_types)
    def test_sqdist(self, dtype):
        """Test squared distance computation."""
        manifold = EuclideanManifold()
        p1 = torch.randn(4, 3, dtype=dtype)
        p2 = torch.randn(4, 3, dtype=dtype)
        c = None

        sqdist = manifold.sqdist(p1, p2, c)
        expected = torch.sum((p1 - p2) ** 2, dim=-1)

        torch.testing.assert_close(sqdist, expected)

    @pytest.mark.parametrize("dtype", lst_types)
    def test_egrad2rgrad(self, dtype):
        """Test Euclidean to Riemannian gradient conversion."""
        manifold = EuclideanManifold()
        p = torch.randn(3, 2, dtype=dtype)
        dp = torch.randn(3, 2, dtype=dtype)
        c = None

        rgrad = manifold.egrad2rgrad(p, dp, c)

        # For Euclidean manifold, Riemannian gradient equals Euclidean gradient
        torch.testing.assert_close(rgrad, dp)

    @pytest.mark.parametrize("dtype", lst_types)
    def test_proj(self, dtype):
        """Test projection onto manifold."""
        manifold = EuclideanManifold()
        p = torch.randn(3, 2, dtype=dtype)
        c = None

        projected = manifold.proj(p, c)

        # For Euclidean manifold, projection is identity
        torch.testing.assert_close(projected, p)

    @pytest.mark.parametrize("dtype", lst_types)
    def test_proj_tan(self, dtype):
        """Test projection onto tangent space."""
        manifold = EuclideanManifold()
        u = torch.randn(3, 2, dtype=dtype)
        p = torch.randn(3, 2, dtype=dtype)
        c = None

        projected = manifold.proj_tan(u, p, c)

        # For Euclidean manifold, tangent space projection is identity
        torch.testing.assert_close(projected, u)

    @pytest.mark.parametrize("dtype", lst_types)
    def test_proj_tan0(self, dtype):
        """Test projection onto tangent space at origin."""
        manifold = EuclideanManifold()
        u = torch.randn(3, 2, dtype=dtype)
        c = None

        projected = manifold.proj_tan0(u, c)

        # For Euclidean manifold, tangent space projection is identity
        torch.testing.assert_close(projected, u)

    @pytest.mark.parametrize("dtype", lst_types)
    def test_expmap(self, dtype):
        """Test exponential map."""
        manifold = EuclideanManifold()
        u = torch.randn(3, 2, dtype=dtype)
        p = torch.randn(3, 2, dtype=dtype)
        c = None

        result = manifold.expmap(u, p, c)
        expected = p + u

        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("dtype", lst_types)
    def test_logmap(self, dtype):
        """Test logarithmic map."""
        manifold = EuclideanManifold()
        p1 = torch.randn(3, 2, dtype=dtype)
        p2 = torch.randn(3, 2, dtype=dtype)
        c = None

        result = manifold.logmap(p1, p2, c)
        expected = p2 - p1

        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("dtype", lst_types)
    def test_expmap0(self, dtype):
        """Test exponential map at origin."""
        manifold = EuclideanManifold()
        u = torch.randn(3, 2, dtype=dtype)
        c = None

        result = manifold.expmap0(u, c)

        # For Euclidean manifold, exp map at origin is identity
        torch.testing.assert_close(result, u)

    @pytest.mark.parametrize("dtype", lst_types)
    def test_logmap0(self, dtype):
        """Test logarithmic map at origin."""
        manifold = EuclideanManifold()
        p = torch.randn(3, 2, dtype=dtype)
        c = None

        result = manifold.logmap0(p, c)

        # For Euclidean manifold, log map at origin is identity
        torch.testing.assert_close(result, p)

    @pytest.mark.parametrize("dtype", lst_types)
    def test_mobius_add(self, dtype):
        """Test Möbius addition."""
        manifold = EuclideanManifold()
        x = torch.randn(3, 2, dtype=dtype)
        y = torch.randn(3, 2, dtype=dtype)
        c = None

        result = manifold.mobius_add(x, y, c)
        expected = x + y

        # For Euclidean manifold, Möbius addition is regular addition
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("dtype", lst_types)
    def test_mobius_matvec(self, dtype):
        """Test Möbius matrix-vector multiplication."""
        manifold = EuclideanManifold()
        m = torch.randn(3, 4, dtype=dtype)  # m shape: (3, 4)
        x = torch.randn(2, 4, dtype=dtype)  # x shape: (2, 4)
        c = None

        # x @ m.transpose(-1, -2) = (2, 4) @ (4, 3) = (2, 3)
        result = manifold.mobius_matvec(m, x, c)
        expected = x @ m.transpose(-1, -2)

        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("dtype", lst_types)
    def test_init_weights(self, dtype):
        """Test weight initialization."""
        manifold = EuclideanManifold()
        w = torch.zeros(3, 2, dtype=dtype)
        c = None
        irange = 0.1

        result = manifold.init_weights(w, c, irange=irange)

        # Check that weights are initialized within range
        assert torch.all(result >= -irange)
        assert torch.all(result <= irange)
        assert result is w  # Should modify in place

    @pytest.mark.parametrize("dtype", lst_types)
    def test_inner(self, dtype):
        """Test inner product."""
        manifold = EuclideanManifold()
        p = torch.randn(3, 2, dtype=dtype)
        u = torch.randn(3, 2, dtype=dtype)
        v = torch.randn(3, 2, dtype=dtype)
        c = None

        # Test with two vectors
        result = manifold.inner(p, c, u, v)
        expected = torch.sum(u * v, dim=-1)
        torch.testing.assert_close(result, expected)

        # Test with one vector (should compute ||u||^2)
        result_single = manifold.inner(p, c, u)
        expected_single = torch.sum(u * u, dim=-1)
        torch.testing.assert_close(result_single, expected_single)

        # Test keepdim
        result_keepdim = manifold.inner(p, c, u, v, keepdim=True)
        assert result_keepdim.shape == (3, 1)

    @pytest.mark.parametrize("dtype", lst_types)
    def test_ptransp(self, dtype):
        """Test parallel transport."""
        manifold = EuclideanManifold()
        x = torch.randn(3, 2, dtype=dtype)
        y = torch.randn(3, 2, dtype=dtype)
        v = torch.randn(3, 2, dtype=dtype)
        c = None

        result = manifold.ptransp(x, y, v, c)

        # For Euclidean manifold, parallel transport is identity
        torch.testing.assert_close(result, v)

    @pytest.mark.parametrize("dtype", lst_types)
    def test_ptransp0(self, dtype):
        """Test parallel transport from origin."""
        manifold = EuclideanManifold()
        x = torch.randn(3, 2, dtype=dtype)
        v = torch.randn(3, 2, dtype=dtype)
        c = None

        result = manifold.ptransp0(x, v, c)
        expected = x + v

        torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("dtype", lst_types)
def test_artanh_function(dtype):
    """Test artanh function and Artanh autograd function."""
    from torchdr.utils.manifold import artanh

    # Test normal values
    x = torch.tensor([0.0, 0.5, -0.3, 0.9], dtype=dtype, requires_grad=True)
    result = artanh(x)

    # Test against expected values (computed with numpy for reference)
    expected_approx = torch.tensor([0.0, 0.5493, -0.3095, 1.4722], dtype=dtype)
    torch.testing.assert_close(result, expected_approx, rtol=1e-3, atol=1e-3)

    # Test gradient computation
    loss = result.sum()
    loss.backward()

    # Gradient of artanh(x) is 1/(1-x^2)
    expected_grad = 1 / (1 - x**2)
    torch.testing.assert_close(x.grad, expected_grad, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dtype", lst_types)
def test_artanh_edge_cases(dtype):
    """Test artanh function with edge cases."""
    from torchdr.utils.manifold import artanh

    # Test values close to boundaries
    x_edge = torch.tensor([0.99999, -0.99999], dtype=dtype)
    result = artanh(x_edge)

    # Should not produce inf or nan
    assert torch.all(torch.isfinite(result))


@pytest.mark.parametrize("dtype", lst_types)
def test_tanh_with_clamp(dtype):
    """Test tanh function with clamping."""
    from torchdr.utils.manifold import tanh

    # Test normal values
    x = torch.randn(5, dtype=dtype)
    result = tanh(x)
    expected = torch.tanh(x)
    torch.testing.assert_close(result, expected)

    # Test large values that would cause overflow
    x_large = torch.tensor([100.0, -100.0], dtype=dtype)
    result_large = tanh(x_large, clamp=15)

    # Should be clamped to reasonable values
    assert torch.all(torch.abs(result_large) <= 1.0)
    assert torch.all(torch.isfinite(result_large))


class TestPoincareBallManifold:
    """Test PoincareBallManifold class."""

    @pytest.mark.parametrize("dtype", lst_types)
    def test_init(self, dtype):
        """Test PoincareBallManifold initialization."""
        manifold = PoincareBallManifold()
        assert manifold.name == "PoincareBall"
        assert manifold.min_norm == 1e-15
        assert dtype in manifold.eps

    @pytest.mark.parametrize("dtype", lst_types)
    def test_lambda_x(self, dtype):
        """Test _lambda_x computation."""
        manifold = PoincareBallManifold()
        x = torch.randn(3, 2, dtype=dtype) * 0.1  # Small values to stay in ball
        c = 1.0

        lambda_x = manifold._lambda_x(x, c)

        # Should be positive and finite
        assert torch.all(lambda_x > 0)
        assert torch.all(torch.isfinite(lambda_x))

    @pytest.mark.parametrize("dtype", lst_types)
    def test_proj(self, dtype):
        """Test projection onto Poincare ball."""
        manifold = PoincareBallManifold()
        c = 1.0

        # Test points inside the ball (should remain unchanged)
        x_inside = torch.randn(3, 2, dtype=dtype) * 0.1
        projected_inside = manifold.proj(x_inside, c)
        torch.testing.assert_close(projected_inside, x_inside, rtol=1e-5, atol=1e-5)

        # Test points outside the ball (should be projected to boundary)
        x_outside = torch.randn(3, 2, dtype=dtype) * 2  # Large values
        projected_outside = manifold.proj(x_outside, c)

        # Projected points should be within the ball
        norms = torch.norm(projected_outside, dim=-1)
        max_norm = (1 - manifold.eps[dtype]) / (c**0.5)
        assert torch.all(
            norms <= max_norm + 1e-6
        )  # Small tolerance for numerical errors

    @pytest.mark.parametrize("dtype", lst_types)
    def test_egrad2rgrad(self, dtype):
        """Test Euclidean to Riemannian gradient conversion."""
        manifold = PoincareBallManifold()
        p = torch.randn(3, 2, dtype=dtype) * 0.1
        dp = torch.randn(3, 2, dtype=dtype)
        c = 1.0

        rgrad = manifold.egrad2rgrad(p, dp.clone(), c)

        # Should scale the gradient
        assert rgrad.shape == dp.shape
        assert torch.all(torch.isfinite(rgrad))

    @pytest.mark.parametrize("dtype", lst_types)
    def test_expmap_logmap_consistency(self, dtype):
        """Test that expmap and logmap are inverse operations."""
        manifold = PoincareBallManifold()
        c = 1.0

        # Start with points in the ball
        p = torch.randn(3, 2, dtype=dtype) * 0.1
        u = torch.randn(3, 2, dtype=dtype) * 0.1

        # Apply expmap then logmap
        q = manifold.expmap(u, p, c)
        u_recovered = manifold.logmap(p, q, c)

        # Should recover original tangent vector (approximately)
        torch.testing.assert_close(u_recovered, u, rtol=1e-3, atol=1e-4)

    @pytest.mark.parametrize("dtype", lst_types)
    def test_expmap0_logmap0_consistency(self, dtype):
        """Test that expmap0 and logmap0 are inverse operations."""
        manifold = PoincareBallManifold()
        c = 1.0

        # Start with small tangent vectors
        u = torch.randn(3, 2, dtype=dtype) * 0.1

        # Apply expmap0 then logmap0
        p = manifold.expmap0(u, c)
        u_recovered = manifold.logmap0(p, c)

        # Should recover original tangent vector (approximately)
        torch.testing.assert_close(u_recovered, u, rtol=1e-3, atol=1e-4)

    @pytest.mark.parametrize("dtype", lst_types)
    def test_mobius_add_properties(self, dtype):
        """Test properties of Möbius addition."""
        manifold = PoincareBallManifold()
        c = 1.0

        x = torch.randn(2, 2, dtype=dtype) * 0.1
        y = torch.randn(2, 2, dtype=dtype) * 0.1

        # Test that Möbius addition keeps points in the ball
        result = manifold.mobius_add(x, y, c)
        norms = torch.norm(result, dim=-1)
        max_norm = 1.0 / (c**0.5)

        # Results should be in the ball (with some numerical tolerance)
        assert torch.all(norms < max_norm - 1e-6)

        # Test with very small values where commutativity should be more stable
        x_small = torch.tensor([[0.01, 0.01], [0.02, -0.01]], dtype=dtype)
        y_small = torch.tensor([[0.01, -0.01], [-0.01, 0.02]], dtype=dtype)

        result_small = manifold.mobius_add(x_small, y_small, c)
        result_small_reversed = manifold.mobius_add(y_small, x_small, c)

        # Test approximate commutativity with small values
        torch.testing.assert_close(
            result_small, result_small_reversed, rtol=0.1, atol=0.01
        )

    @pytest.mark.parametrize("dtype", lst_types)
    def test_inner_product_positive_definite(self, dtype):
        """Test that inner product is positive definite."""
        manifold = PoincareBallManifold()
        c = 1.0

        x = torch.randn(3, 2, dtype=dtype) * 0.1
        u = torch.randn(3, 2, dtype=dtype)

        # Inner product of a vector with itself should be positive
        inner_uu = manifold.inner(x, c, u)
        assert torch.all(inner_uu >= 0)

        # Should be zero only if u is zero
        u_zero = torch.zeros_like(u)
        inner_zero = manifold.inner(x, c, u_zero)
        torch.testing.assert_close(
            inner_zero, torch.zeros_like(inner_zero), rtol=1e-5, atol=1e-8
        )

    @pytest.mark.parametrize("dtype", lst_types)
    def test_ptransp_preserves_inner_product(self, dtype):
        """Test that parallel transport preserves inner products (approximately)."""
        manifold = PoincareBallManifold()
        c = 1.0

        x = (
            torch.randn(2, 2, dtype=dtype) * 0.05
        )  # Very small to avoid numerical issues
        y = torch.randn(2, 2, dtype=dtype) * 0.05
        u = torch.randn(2, 2, dtype=dtype) * 0.1
        v = torch.randn(2, 2, dtype=dtype) * 0.1

        # Compute inner product at x
        inner_x = manifold.inner(x, c, u, v)

        # Transport vectors to y
        u_transported = manifold.ptransp(x, y, u, c)
        v_transported = manifold.ptransp(x, y, v, c)

        # Compute inner product at y
        inner_y = manifold.inner(y, c, u_transported, v_transported)

        # Should be approximately preserved (with some tolerance for numerical errors)
        torch.testing.assert_close(inner_x, inner_y, rtol=5e-2, atol=1e-2)

    @pytest.mark.parametrize("dtype", lst_types)
    def test_init_weights(self, dtype):
        """Test weight initialization."""
        manifold = PoincareBallManifold()
        w = torch.zeros(3, 2, dtype=dtype)
        c = 1.0
        irange = 0.01

        result = manifold.init_weights(w, c, irange=irange)

        # Check that weights are initialized within range
        assert torch.all(result >= -irange)
        assert torch.all(result <= irange)
        assert result is w  # Should modify in place

        # Check that initialized weights are in the ball
        norms = torch.norm(result, dim=-1)
        max_norm = 1.0 / (c**0.5)
        assert torch.all(norms < max_norm)

    @pytest.mark.parametrize("dtype", lst_types)
    def test_to_hyperboloid(self, dtype):
        """Test conversion to hyperboloid model."""
        manifold = PoincareBallManifold()
        c = 1.0

        x = torch.randn(3, 2, dtype=dtype) * 0.1
        hyperboloid = manifold.to_hyperboloid(x, c)

        # Check dimensions
        assert hyperboloid.shape == (3, 3)  # Should add one dimension

        # Check that result satisfies hyperboloid constraint (approximately)
        # For hyperboloid: -x_0^2 + x_1^2 + x_2^2 = -K where K = 1/c
        constraint = -(hyperboloid[:, 0:1] ** 2) + torch.sum(
            hyperboloid[:, 1:] ** 2, dim=1, keepdim=True
        )
        expected_constraint = -torch.ones_like(constraint) / c
        torch.testing.assert_close(
            constraint, expected_constraint, rtol=1e-4, atol=1e-5
        )


# ====== test matrix_power function ======


cases = [
    # Dense backend (torch.Tensor)
    pytest.param(False, 0, None, id="dense_pow0"),
    pytest.param(False, 1, None, id="dense_pow1"),
    pytest.param(False, 3, None, id="dense_pow3"),
    pytest.param(False, 2.5, None, id="dense_pow2.5"),
    pytest.param(
        False,
        -1,
        (ValueError, r"Negative matrix powers are not supported"),
        id="dense_neg",
    ),
    # KeOps‐lazy backend (LazyTensor) - not supported
    pytest.param(
        True,
        0,
        (
            NotImplementedError,
            r"matrix powers are not supported with KeOps backend",
        ),
        marks=pytest.mark.skipif(not pykeops, reason="pykeops is not available"),
        id="lazy_pow0",
    ),
    pytest.param(
        True,
        2,
        (
            NotImplementedError,
            r"matrix powers are not supported with KeOps backend",
        ),
        marks=pytest.mark.skipif(not pykeops, reason="pykeops is not available"),
        id="lazy_pow2",
    ),
    pytest.param(
        True,
        2.5,
        (
            NotImplementedError,
            r"matrix powers are not supported with KeOps backend",
        ),
        marks=pytest.mark.skipif(not pykeops, reason="pykeops is not available"),
        id="lazy_pow2.5",
    ),
]


@pytest.mark.parametrize("use_lazy,power,exc", cases)
def test_matrix_power(use_lazy, power, exc):
    """Combined float32 tests for dense and KeOps‐lazy matrix_power."""
    dtype = torch.float32

    if use_lazy:
        A = identity_matrix(3, keops=True, device="cpu", dtype=dtype)
    else:
        A = torch.randn(3, 3, dtype=dtype)
        A = A @ A.T + torch.eye(3, dtype=dtype) * 0.1

    if exc:
        err_type, msg = exc
        with pytest.raises(err_type, match=msg):
            matrix_power(A, power)
        return

    result = matrix_power(A, power)

    if not use_lazy:
        # dense: integer vs float
        if power == int(power):
            ip = int(power)
            if ip == 0:
                expected = torch.eye(3, dtype=dtype)
            elif ip == 1:
                expected = A
            else:
                expected = torch.linalg.matrix_power(A, ip)
        else:
            vals, vecs = torch.linalg.eigh(A)
            vals = torch.clamp(vals, min=1e-12) ** power
            expected = vecs @ torch.diag_embed(vals) @ vecs.transpose(-2, -1)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-6)


# ====== Test validate_tensor ======
class TestValidateTensor:
    def test_validate_tensor_expects_tensor(self):
        X = np.random.randn(10, 5)
        with pytest.raises(ValueError, match="validate_tensor expects a torch.Tensor"):
            validate_tensor(X)

    def test_validate_tensor_tensor(self):
        X = torch.randn(10, 5)
        X_validated = validate_tensor(X)
        assert X is X_validated  # Should be the same object

    def test_ensure_2d(self):
        X = torch.randn(10)
        X_2d = validate_tensor(X, ensure_2d=True)
        assert X_2d.ndim == 2
        assert X_2d.shape == (10, 1)

    def test_min_samples_features(self):
        X = torch.randn(1, 1)
        with pytest.raises(ValueError):
            validate_tensor(X, ensure_min_samples=2)
        with pytest.raises(ValueError):
            validate_tensor(X, ensure_min_features=2)

    def test_accept_sparse(self):
        X = torch.randn(10, 5).to_sparse()
        with pytest.raises(ValueError):
            validate_tensor(X, accept_sparse=False)
        X_sparse = validate_tensor(X, accept_sparse=True)
        assert X_sparse.is_sparse


# ====== Test to_torch ======
class TestToTorch:
    def test_to_torch_numpy(self):
        X = np.random.randn(10, 5)
        X_torch, backend, device = to_torch(X, return_backend_device=True)
        assert isinstance(X_torch, torch.Tensor)
        assert backend == "numpy"
        assert device == "cpu"
        assert_close(torch.from_numpy(X), X_torch)

    def test_to_torch_tensor(self):
        X = torch.randn(10, 5)
        X_torch, backend, device = to_torch(X, return_backend_device=True)
        assert backend == "torch"
        assert device == X.device

    def test_validate_complex_error(self):
        X = torch.randn(10, 5, dtype=torch.cfloat)
        with pytest.raises(ValueError, match="complex tensors are not supported"):
            validate_tensor(X)

    def test_validate_infinite_error(self):
        X = torch.tensor([1.0, float("inf")])
        with pytest.raises(ValueError, match="infinite values"):
            validate_tensor(X)
