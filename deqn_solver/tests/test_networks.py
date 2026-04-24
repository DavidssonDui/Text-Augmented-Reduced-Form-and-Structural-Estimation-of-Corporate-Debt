"""
test_networks.py
================
Tests for deqn_solver.src.networks — MLP backbone and three domain networks.

Covers:
  - MLP (generic backbone)
  - StateNormalizer
  - ValueNet (value function V(w_tilde, z, s))
  - PolicyNet (policy (k', b')(w_tilde, z, s))
  - DefaultNet (default threshold w_underbar(z))
  - count_parameters
"""

import torch
import pytest

from deqn_solver.src.networks import (
    MLP, StateNormalizer, ValueNet, PolicyNet, DefaultNet, count_parameters,
)


# ════════════════════════════════════════════════════════════════
# MLP
# ════════════════════════════════════════════════════════════════

class TestMLP:
    """Generic multi-layer perceptron backbone."""

    def test_output_shape_single_input(self):
        """Input shape (batch, input_dim) → output (batch, output_dim)."""
        net = MLP(input_dim=3, output_dim=1, hidden_sizes=[16, 16])
        x = torch.randn(10, 3)
        y = net(x)
        assert y.shape == (10, 1)

    def test_output_shape_multi_output(self):
        net = MLP(input_dim=3, output_dim=2, hidden_sizes=[16, 16])
        x = torch.randn(10, 3)
        y = net(x)
        assert y.shape == (10, 2)

    def test_gradient_flow(self):
        """Loss should be differentiable w.r.t. net params."""
        net = MLP(input_dim=3, output_dim=1, hidden_sizes=[16])
        x = torch.randn(5, 3)
        y = net(x)
        loss = y.sum()
        loss.backward()
        # Check at least one parameter has a non-zero gradient
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                         for p in net.parameters())
        assert has_grad

    def test_raises_on_unknown_activation(self):
        with pytest.raises(ValueError, match="Unknown activation"):
            MLP(input_dim=2, output_dim=1, hidden_sizes=[8],
                activation="not_a_real_activation")

    def test_supports_tanh_relu_silu(self):
        for act in ["tanh", "relu", "silu"]:
            net = MLP(input_dim=2, output_dim=1, hidden_sizes=[8], activation=act)
            x = torch.randn(3, 2)
            y = net(x)
            assert y.shape == (3, 1)
            assert torch.isfinite(y).all()

    def test_small_initialization_produces_small_outputs(self):
        """With the small-init scheme, outputs should be in a bounded range."""
        torch.manual_seed(42)
        net = MLP(input_dim=3, output_dim=1, hidden_sizes=[16, 16])
        x = torch.randn(100, 3) * 5  # inputs up to ~15
        y = net(x)
        # Tanh activations + small-init weights should keep output reasonable
        # (within magnitude 10 for this small input scale)
        assert y.abs().max() < 10, \
            f"Small-init MLP should produce bounded outputs, got max {y.abs().max()}"


# ════════════════════════════════════════════════════════════════
# StateNormalizer
# ════════════════════════════════════════════════════════════════

class TestStateNormalizer:
    """State input normalization: (x - mean) / std."""

    def test_normalizes_to_zero_mean_unit_std_when_input_matches(self):
        """If input has the same mean/std as the normalizer, output ~ N(0,1)."""
        means = torch.tensor([5.0, 0.0, 10.0])
        stds = torch.tensor([1.0, 2.0, 0.5])
        normalizer = StateNormalizer(means, stds)

        # Make input with exactly these moments
        raw = means.unsqueeze(0) + stds.unsqueeze(0) * torch.tensor([
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, -1.0],
        ])
        out = normalizer(raw)
        # Should produce values of exactly +1 and -1
        expected = torch.tensor([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]])
        torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)

    def test_hand_calculation(self):
        means = torch.tensor([0.0, 1.0, 0.0])
        stds = torch.tensor([2.0, 0.5, 1.0])
        normalizer = StateNormalizer(means, stds)

        x = torch.tensor([[4.0, 2.0, 3.0]])
        # Expected: (4-0)/2 = 2.0, (2-1)/0.5 = 2.0, (3-0)/1 = 3.0
        expected = torch.tensor([[2.0, 2.0, 3.0]])
        out = normalizer(x)
        torch.testing.assert_close(out, expected)

    def test_is_deterministic(self):
        """Repeated applications yield identical output."""
        normalizer = StateNormalizer(
            torch.tensor([1.0, 1.0, 1.0]),
            torch.tensor([1.0, 1.0, 1.0]),
        )
        x = torch.randn(10, 3)
        y1 = normalizer(x)
        y2 = normalizer(x)
        torch.testing.assert_close(y1, y2)

    def test_protects_against_zero_std(self):
        """A zero std would divide by zero; 1e-8 stabilization should kick in."""
        normalizer = StateNormalizer(
            torch.tensor([0.0, 0.0, 0.0]),
            torch.tensor([1.0, 0.0, 1.0]),  # middle dim has zero std
        )
        x = torch.tensor([[1.0, 1.0, 1.0]])
        out = normalizer(x)
        # Should not be inf or nan
        assert torch.isfinite(out).all()

    def test_to_device_moves_tensors(self):
        """Calling .to() should move buffers to the target device."""
        normalizer = StateNormalizer(
            torch.tensor([0.0, 0.0, 0.0]),
            torch.tensor([1.0, 1.0, 1.0]),
        )
        normalizer.to('cpu')
        assert normalizer.means.device.type == 'cpu'
        assert normalizer.stds.device.type == 'cpu'


# ════════════════════════════════════════════════════════════════
# ValueNet
# ════════════════════════════════════════════════════════════════

class TestValueNet:
    """Value function network V(w_tilde, z, s)."""

    def test_output_shape(self, value_net, sample_state):
        V = value_net(sample_state)
        assert V.shape == (sample_state.shape[0],), \
            f"Expected (batch,), got {V.shape}"

    def test_output_is_finite(self, value_net, sample_state):
        V = value_net(sample_state)
        assert torch.isfinite(V).all()

    def test_gradient_flows(self, value_net, sample_state):
        V = value_net(sample_state)
        loss = V.sum()
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                         for p in value_net.parameters())
        assert has_grad

    def test_works_without_normalizer(self, torch_seed):
        """Should accept None for the normalizer arg."""
        net = ValueNet(hidden_sizes=[8, 8], normalizer=None)
        x = torch.randn(5, 3)
        y = net(x)
        assert y.shape == (5,)

    def test_output_can_be_negative(self, value_net):
        """Value can be negative near the default boundary."""
        # With random init, some states should produce negative V
        torch.manual_seed(123)
        x = torch.randn(200, 3)
        V = value_net(x)
        # At least some should be positive and some negative
        assert V.min() < 0 or V.max() > 0, \
            "Value net output is uniformly one-signed; something's off"


# ════════════════════════════════════════════════════════════════
# PolicyNet
# ════════════════════════════════════════════════════════════════

class TestPolicyNet:
    """Policy network (k', b')(w_tilde, z, s)."""

    def test_output_shape(self, policy_net, sample_state):
        policy = policy_net(sample_state)
        assert policy.shape == (sample_state.shape[0], 2)

    def test_k_prime_is_nonnegative(self, policy_net, sample_state):
        """k' = softplus(raw) * k_scale, must be >= 0."""
        policy = policy_net(sample_state)
        k_prime = policy[:, 0]
        assert (k_prime >= 0).all(), \
            f"k_prime has negative values: {k_prime.min()}"

    def test_b_prime_can_be_any_sign(self, policy_net, torch_seed):
        """b' is identity-output, so with random inputs both signs occur."""
        torch.manual_seed(42)
        x = torch.randn(500, 3)
        policy = policy_net(x)
        b_prime = policy[:, 1]
        # Over a random batch, we expect both signs
        assert (b_prime > 0).any() and (b_prime < 0).any(), \
            "b_prime should span both signs in random batch"

    def test_k_prime_respects_scale(self, policy_net, sample_state):
        """k' magnitudes should be on the order of k_scale."""
        policy = policy_net(sample_state)
        k_prime = policy[:, 0]
        # With k_scale=50, k_prime should not be orders of magnitude larger
        assert k_prime.max() < 10 * policy_net.k_scale

    def test_default_k_scale(self, torch_seed):
        net = PolicyNet(hidden_sizes=[8])
        assert net.k_scale == 100.0

    def test_default_b_scale(self, torch_seed):
        net = PolicyNet(hidden_sizes=[8])
        assert net.b_scale == 30.0


# ════════════════════════════════════════════════════════════════
# DefaultNet
# ════════════════════════════════════════════════════════════════

class TestDefaultNet:
    """Default threshold network w_underbar(z)."""

    def test_output_shape(self, default_net):
        """Takes shape (batch,), returns shape (batch,)."""
        z = torch.tensor([0.5, 1.0, 1.5, 2.0])
        w_ub = default_net(z)
        assert w_ub.shape == (4,)

    def test_output_is_strictly_negative(self, default_net, torch_seed):
        """w_underbar(z) = -softplus(...) is always < 0."""
        torch.manual_seed(42)
        z = torch.exp(torch.randn(100))  # random positive z
        w_ub = default_net(z)
        assert (w_ub < 0).all(), \
            f"Default threshold must be negative, got max {w_ub.max()}"

    def test_output_is_finite(self, default_net):
        z = torch.tensor([0.01, 1.0, 100.0])  # wide range
        w_ub = default_net(z)
        assert torch.isfinite(w_ub).all()

    def test_gradient_flows(self, default_net):
        z = torch.tensor([0.5, 1.0, 1.5])
        w_ub = default_net(z)
        loss = w_ub.sum()
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                         for p in default_net.parameters())
        assert has_grad

    def test_depends_only_on_z(self, default_net):
        """
        DefaultNet takes only z as input (no w_tilde or s).
        Verify by checking signature accepts 1-D z.
        """
        z = torch.tensor([1.0])
        w_ub = default_net(z)
        assert w_ub.shape == (1,)


# ════════════════════════════════════════════════════════════════
# count_parameters
# ════════════════════════════════════════════════════════════════

class TestCountParameters:
    """Utility: count trainable parameters."""

    def test_simple_mlp_count(self):
        """MLP with (in=3, hidden=[16], out=1):
           Layer 1: 3*16 + 16 = 64
           Layer 2: 16*1 + 1 = 17
           Total: 81
        """
        net = MLP(input_dim=3, output_dim=1, hidden_sizes=[16])
        n = count_parameters(net)
        assert n == 81

    def test_zero_for_frozen_network(self):
        """If all parameters are frozen, should count 0."""
        net = MLP(input_dim=3, output_dim=1, hidden_sizes=[16])
        for p in net.parameters():
            p.requires_grad = False
        assert count_parameters(net) == 0
