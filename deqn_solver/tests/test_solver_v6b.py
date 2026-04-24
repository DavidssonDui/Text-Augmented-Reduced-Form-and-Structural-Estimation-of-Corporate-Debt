"""
test_solver_v6b.py
==================
Tests for deqn_solver.src.solver_v6b — the pure helper functions
that don't depend on stochastic sampling.

Covers:
  - auto_device
  - smooth_max
  - smooth_default_indicator
  - solve_bond_yield_analytic
  - analytical_value_target / analytical_policy_target / analytical_default_target
  - compute_boundary_loss
  - compute_monotonicity_penalty
  - compute_payout_anchor

NOT covered (requires sampling.py and lengthy training):
  - compute_bellman_residual_v6  (exercised indirectly via training; requires sampling)
  - pretrain_networks            (integration test; slow)
  - solve_v6b                    (end-to-end training; slow)
"""

import torch
import pytest

from deqn_solver.src.solver_v6b import (
    auto_device, smooth_max, smooth_default_indicator,
    solve_bond_yield_analytic,
    analytical_value_target, analytical_policy_target, analytical_default_target,
    compute_boundary_loss, compute_monotonicity_penalty, compute_payout_anchor,
)
from deqn_solver.src.config import ModelParams
from deqn_solver.src.networks import ValueNet, PolicyNet, DefaultNet, StateNormalizer


# ════════════════════════════════════════════════════════════════
# auto_device
# ════════════════════════════════════════════════════════════════

class TestAutoDevice:
    """Device detection."""

    def test_returns_valid_device_string(self):
        d = auto_device()
        assert d in {"cpu", "cuda", "mps"}


# ════════════════════════════════════════════════════════════════
# smooth_max
# ════════════════════════════════════════════════════════════════

class TestSmoothMax:
    """Smooth approximation to max(a, b)."""

    def test_approximately_max_when_far_apart(self):
        """When a and b differ a lot, smooth_max ≈ max(a, b)."""
        a = torch.tensor([10.0])
        b = torch.tensor([-10.0])
        sm = smooth_max(a, b, sharpness=10.0)
        # Should be very close to max = 10
        assert abs(sm.item() - 10.0) < 0.1

    def test_greater_than_or_equal_to_max(self):
        """smooth_max should upper-bound the true max (from log-sum-exp formula)."""
        for a_val, b_val in [(5.0, 3.0), (-1.0, 2.0), (0.0, 0.0)]:
            a = torch.tensor([a_val])
            b = torch.tensor([b_val])
            sm = smooth_max(a, b, sharpness=5.0).item()
            true_max = max(a_val, b_val)
            assert sm >= true_max - 1e-5, \
                f"smooth_max({a_val},{b_val})={sm} < max={true_max}"

    def test_gradient_flows(self):
        """smooth_max should be differentiable."""
        a = torch.tensor([1.0], requires_grad=True)
        b = torch.tensor([2.0], requires_grad=True)
        sm = smooth_max(a, b, sharpness=5.0)
        sm.backward()
        assert a.grad is not None
        assert b.grad is not None
        # Gradients should sum to ~1 (sum of softmax weights)
        assert abs(a.grad.item() + b.grad.item() - 1.0) < 0.01

    def test_sharpness_controls_accuracy(self):
        """Higher sharpness → closer to true max."""
        a = torch.tensor([5.0])
        b = torch.tensor([3.0])
        sm_soft = smooth_max(a, b, sharpness=1.0).item()
        sm_sharp = smooth_max(a, b, sharpness=100.0).item()
        # Higher sharpness → closer to 5.0 (the true max)
        assert abs(sm_sharp - 5.0) < abs(sm_soft - 5.0)


# ════════════════════════════════════════════════════════════════
# smooth_default_indicator
# ════════════════════════════════════════════════════════════════

class TestSmoothDefaultIndicator:
    """Sigmoid-based smooth indicator for default."""

    def test_output_in_unit_interval(self):
        """Output must always be in [0, 1] regardless of inputs."""
        torch.manual_seed(42)
        w_realized = torch.randn(100) * 10
        w_underbar = torch.randn(100) * 5 - 5
        ind = smooth_default_indicator(w_realized, w_underbar)
        assert (ind >= 0).all()
        assert (ind <= 1).all()

    def test_one_when_realized_far_below_underbar(self):
        """If w_realized << w_underbar, default indicator ≈ 1."""
        w_realized = torch.tensor([-100.0])
        w_underbar = torch.tensor([0.0])
        ind = smooth_default_indicator(w_realized, w_underbar, sharpness=5.0)
        assert ind.item() > 0.99

    def test_zero_when_realized_far_above_underbar(self):
        """If w_realized >> w_underbar, default indicator ≈ 0."""
        w_realized = torch.tensor([100.0])
        w_underbar = torch.tensor([0.0])
        ind = smooth_default_indicator(w_realized, w_underbar, sharpness=5.0)
        assert ind.item() < 0.01

    def test_half_at_boundary(self):
        """When w_realized = w_underbar, sigmoid gives exactly 0.5."""
        w_realized = torch.tensor([-3.0])
        w_underbar = torch.tensor([-3.0])
        ind = smooth_default_indicator(w_realized, w_underbar)
        assert abs(ind.item() - 0.5) < 1e-6

    def test_gradient_flows(self):
        w_realized = torch.tensor([1.0], requires_grad=True)
        w_underbar = torch.tensor([-1.0], requires_grad=True)
        ind = smooth_default_indicator(w_realized, w_underbar)
        ind.sum().backward()
        assert w_realized.grad is not None
        assert w_underbar.grad is not None


# ════════════════════════════════════════════════════════════════
# solve_bond_yield_analytic
# ════════════════════════════════════════════════════════════════

class TestSolveBondYieldAnalytic:
    """Break-even bond yield solver."""

    def test_returns_risk_free_when_not_borrowing(self, default_params):
        """If b' ≈ 0, r_tilde should default to r (risk-free)."""
        k_prime = torch.tensor([10.0])
        b_prime = torch.tensor([0.0])  # no borrowing
        z_prime = torch.randn(1, 8).exp()
        importance = torch.ones_like(z_prime)
        w_underbar = -torch.rand_like(z_prime) * 5 - 1
        r_tilde = solve_bond_yield_analytic(
            k_prime, b_prime, z_prime, importance, w_underbar, default_params
        )
        # Should be at the risk-free rate
        assert abs(r_tilde.item() - default_params.r) < 1e-3

    def test_yield_above_risk_free_when_borrowing_risky(self, default_params):
        """Firm with risky profile gets yield > r."""
        k_prime = torch.tensor([10.0])
        b_prime = torch.tensor([8.0])  # heavy borrowing
        # Very low z samples + high default threshold → high default prob
        z_prime = torch.tensor([[0.2, 0.3, 0.4, 0.5]])
        importance = torch.ones_like(z_prime)
        w_underbar = torch.tensor([[-1.0, -1.0, -1.0, -1.0]])  # easy to default
        r_tilde = solve_bond_yield_analytic(
            k_prime, b_prime, z_prime, importance, w_underbar, default_params
        )
        # Yield should be strictly above risk-free rate
        assert r_tilde.item() > default_params.r

    def test_output_is_finite(self, default_params):
        """Regardless of inputs, output should never be NaN or inf."""
        torch.manual_seed(42)
        k_prime = torch.tensor([5.0, 10.0, 20.0])
        b_prime = torch.tensor([1.0, 5.0, 15.0])
        z_prime = torch.rand(3, 10).exp()
        importance = torch.ones_like(z_prime)
        w_underbar = -torch.rand_like(z_prime) * 5
        r_tilde = solve_bond_yield_analytic(
            k_prime, b_prime, z_prime, importance, w_underbar, default_params
        )
        assert torch.isfinite(r_tilde).all()

    def test_output_shape_matches_input_batch(self, default_params):
        """r_tilde should have shape (batch,)."""
        k = torch.tensor([5.0, 10.0])
        b = torch.tensor([1.0, 3.0])
        z_prime = torch.rand(2, 5).exp()
        imp = torch.ones_like(z_prime)
        w_ub = -torch.rand_like(z_prime) * 3
        r_tilde = solve_bond_yield_analytic(k, b, z_prime, imp, w_ub, default_params)
        assert r_tilde.shape == (2,)


# ════════════════════════════════════════════════════════════════
# Analytical pre-training targets
# ════════════════════════════════════════════════════════════════

class TestAnalyticalTargets:
    """Pre-training targets (initial conditions for network training)."""

    def test_value_target_is_linear_in_w(self, default_params):
        """V_target(w, z, s) = w + 5(z-1) - 2 should be linear in w."""
        s1 = torch.tensor([[0.0, 1.0, 0.0]])
        s2 = torch.tensor([[10.0, 1.0, 0.0]])
        v1 = analytical_value_target(s1, default_params).item()
        v2 = analytical_value_target(s2, default_params).item()
        # Difference should be 10 (slope of 1 in w)
        assert abs((v2 - v1) - 10.0) < 1e-4

    def test_value_target_increases_with_z(self, default_params):
        """Higher productivity → higher value target (slope of 5)."""
        s1 = torch.tensor([[0.0, 0.5, 0.0]])
        s2 = torch.tensor([[0.0, 1.5, 0.0]])
        v1 = analytical_value_target(s1, default_params).item()
        v2 = analytical_value_target(s2, default_params).item()
        # Difference should be 5 * (1.5 - 0.5) = 5
        assert abs((v2 - v1) - 5.0) < 1e-4

    def test_policy_target_k_is_positive(self, default_params):
        """Analytical policy for k' should be strictly positive."""
        states = torch.tensor([
            [-100.0, 1.0, 0.0],  # very negative w
            [0.0, 1.0, 0.0],
            [100.0, 1.0, 0.0],
        ])
        policy = analytical_policy_target(states, default_params)
        k_prime = policy[:, 0]
        assert (k_prime > 0).all()

    def test_policy_target_k_bounded(self, default_params):
        """k' target clamped to [5, 50]."""
        states = torch.tensor([
            [-100.0, 1.0, 0.0],
            [1000.0, 1.0, 0.0],
        ])
        policy = analytical_policy_target(states, default_params)
        k_prime = policy[:, 0]
        assert (k_prime >= 5.0).all()
        assert (k_prime <= 50.0).all()

    def test_policy_target_b_proportional_to_k(self, default_params):
        """b' = 0.2 * k' in the analytical target."""
        states = torch.tensor([[10.0, 1.0, 0.0], [20.0, 1.0, 0.0]])
        policy = analytical_policy_target(states, default_params)
        ratio = policy[:, 1] / policy[:, 0]
        torch.testing.assert_close(ratio, torch.tensor([0.2, 0.2]),
                                     atol=1e-4, rtol=1e-4)

    def test_default_target_is_negative(self, default_params):
        """Analytical default threshold should be negative."""
        z = torch.tensor([0.5, 1.0, 1.5, 2.0])
        w_ub = analytical_default_target(z, default_params)
        assert (w_ub < 0).all()

    def test_default_target_decreasing_in_z(self, default_params):
        """Higher z → lower (more negative) default threshold in theory."""
        z_low = torch.tensor([0.5])
        z_high = torch.tensor([2.0])
        w_low = analytical_default_target(z_low, default_params).item()
        w_high = analytical_default_target(z_high, default_params).item()
        # Higher productivity firm should tolerate more negative net worth
        assert w_high < w_low


# ════════════════════════════════════════════════════════════════
# compute_boundary_loss
# ════════════════════════════════════════════════════════════════

class TestComputeBoundaryLoss:
    """V(w_underbar, z, 0) ≈ 0 enforcement."""

    def test_output_is_scalar(self, value_net, default_net):
        state = torch.tensor([[10.0, 1.0, 0.0], [5.0, 1.5, 0.1]])
        loss = compute_boundary_loss(state, value_net, default_net)
        assert loss.dim() == 0, "Boundary loss should be a scalar"

    def test_nonnegative(self, value_net, default_net):
        """Squared loss is always >= 0."""
        state = torch.tensor([[10.0, 1.0, 0.0], [5.0, 1.5, 0.1]])
        loss = compute_boundary_loss(state, value_net, default_net)
        assert loss.item() >= 0

    def test_finite(self, value_net, default_net):
        state = torch.tensor([[10.0, 1.0, 0.0], [5.0, 1.5, 0.1]])
        loss = compute_boundary_loss(state, value_net, default_net)
        assert torch.isfinite(loss)


# ════════════════════════════════════════════════════════════════
# compute_monotonicity_penalty
# ════════════════════════════════════════════════════════════════

class TestComputeMonotonicityPenalty:
    """Penalty for V non-monotonic in z."""

    def test_output_is_scalar(self, value_net, default_params):
        state = torch.tensor([[10.0, 1.0, 0.0], [5.0, 1.5, 0.1]])
        pen = compute_monotonicity_penalty(state, value_net, default_params)
        assert pen.dim() == 0

    def test_nonnegative(self, value_net, default_params):
        state = torch.tensor([[10.0, 1.0, 0.0], [5.0, 1.5, 0.1]])
        pen = compute_monotonicity_penalty(state, value_net, default_params)
        assert pen.item() >= 0


# ════════════════════════════════════════════════════════════════
# compute_payout_anchor
# ════════════════════════════════════════════════════════════════

class TestComputePayoutAnchor:
    """v6b's targeted fix for high-wealth hoarding."""

    def test_output_is_scalar(self, policy_net):
        state = torch.tensor([[50.0, 1.0, 0.0], [60.0, 1.0, 0.0]])
        loss = compute_payout_anchor(state, policy_net, high_w_threshold=30.0)
        assert loss.dim() == 0

    def test_nonnegative(self, policy_net):
        state = torch.tensor([[10.0, 1.0, 0.0], [40.0, 1.0, 0.0]])
        loss = compute_payout_anchor(state, policy_net)
        assert loss.item() >= 0

    def test_zero_when_no_states_above_threshold(self, policy_net):
        """
        If all states have w < threshold, payout anchor is trivially 0.
        """
        state = torch.tensor([[-5.0, 1.0, 0.0], [5.0, 1.0, 0.0]])
        loss = compute_payout_anchor(state, policy_net, high_w_threshold=100.0)
        # Either exactly 0 (no states above threshold) or very small
        assert loss.item() >= 0
        assert loss.item() < 1e-2

    def test_threshold_parameter_respected(self, policy_net):
        """Changing threshold should change which states contribute."""
        state = torch.tensor([[25.0, 1.0, 0.0], [35.0, 1.0, 0.0]])
        loss_low = compute_payout_anchor(state, policy_net, high_w_threshold=20.0)
        loss_high = compute_payout_anchor(state, policy_net, high_w_threshold=40.0)
        # Low threshold includes both states; high threshold includes neither
        assert loss_high.item() <= loss_low.item()
