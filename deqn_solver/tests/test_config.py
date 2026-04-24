"""
test_config.py
==============
Tests for deqn_solver.src.config — dataclass configs for model,
network, training, and top-level Config.
"""

import pytest

from deqn_solver.src.config import (
    ModelParams, NetworkConfig, TrainingConfig, Config,
    default_config, hw07_baseline_config, augmented_config,
)


# ════════════════════════════════════════════════════════════════
# ModelParams
# ════════════════════════════════════════════════════════════════

class TestModelParams:
    """Structural parameters dataclass."""

    def test_default_values_match_hw07_table1(self):
        """HW07 Table I, full-sample estimates."""
        p = ModelParams()
        assert p.alpha == 0.627
        assert p.rho == 0.684
        assert p.sigma_eps == 0.118
        assert p.lambda_1 == 0.091
        assert p.lambda_2 == 0.0004
        assert p.xi == 0.104
        assert p.phi == 0.732

    def test_default_lambda_text_is_zero(self):
        """Default: no text augmentation → HW07 baseline."""
        p = ModelParams()
        assert p.lambda_text == 0.0

    def test_beta_computed_correctly(self):
        """beta = 1 / (1 + r * (1 - tau_i))"""
        p = ModelParams(r=0.025, tau_i=0.29)
        expected = 1.0 / (1.0 + 0.025 * (1.0 - 0.29))
        assert abs(p.beta - expected) < 1e-9

    def test_beta_changes_with_r(self):
        p1 = ModelParams(r=0.02)
        p2 = ModelParams(r=0.05)
        # Higher r → lower beta (more discounting)
        assert p2.beta < p1.beta

    def test_can_override_any_parameter(self):
        p = ModelParams(alpha=0.5, lambda_text=0.7, sigma_eta=1.5)
        assert p.alpha == 0.5
        assert p.lambda_text == 0.7
        assert p.sigma_eta == 1.5
        # Unchanged params remain at defaults
        assert p.rho == 0.684


# ════════════════════════════════════════════════════════════════
# NetworkConfig
# ════════════════════════════════════════════════════════════════

class TestNetworkConfig:

    def test_defaults(self):
        n = NetworkConfig()
        assert n.state_dim == 3
        assert n.value_output_dim == 1
        assert n.policy_output_dim == 2
        assert n.hidden_sizes == (64, 64, 64)
        assert n.activation == "tanh"

    def test_hidden_sizes_is_tuple(self):
        """hidden_sizes should be immutable."""
        n = NetworkConfig()
        assert isinstance(n.hidden_sizes, tuple)


# ════════════════════════════════════════════════════════════════
# TrainingConfig
# ════════════════════════════════════════════════════════════════

class TestTrainingConfig:

    def test_defaults_positive(self):
        t = TrainingConfig()
        assert t.learning_rate > 0
        assert t.batch_size > 0
        assert t.num_epochs > 0
        assert t.num_mc_samples > 0

    def test_state_bounds_ordered(self):
        """w_tilde_min must be strictly less than w_tilde_max."""
        t = TrainingConfig()
        assert t.w_tilde_min < t.w_tilde_max


# ════════════════════════════════════════════════════════════════
# Factory functions
# ════════════════════════════════════════════════════════════════

class TestFactoryFunctions:

    def test_default_config_returns_config_instance(self):
        c = default_config()
        assert isinstance(c, Config)
        assert isinstance(c.model, ModelParams)

    def test_hw07_baseline_has_zero_lambda_text(self):
        c = hw07_baseline_config()
        assert c.model.lambda_text == 0.0

    def test_augmented_default(self):
        c = augmented_config()
        assert c.model.lambda_text == 0.3

    def test_augmented_custom_lambda(self):
        c = augmented_config(lambda_text=0.7)
        assert c.model.lambda_text == 0.7

    def test_augmented_keeps_other_defaults(self):
        """augmented_config should only change lambda_text."""
        c = augmented_config(0.5)
        assert c.model.alpha == 0.627  # HW07 default


# ════════════════════════════════════════════════════════════════
# Config
# ════════════════════════════════════════════════════════════════

class TestConfig:

    def test_default_seed(self):
        c = Config()
        assert c.seed == 42

    def test_default_device_is_auto(self):
        c = Config()
        assert c.device == "auto"

    def test_repr_returns_string(self):
        c = Config()
        s = repr(c)
        assert isinstance(s, str)
        # Should mention key fields
        assert 'Model' in s or 'model' in s

    def test_each_factory_produces_independent_configs(self):
        """
        Check that default_factory creates separate instances
        (not a shared reference).
        """
        c1 = Config()
        c2 = Config()
        c1.model.alpha = 0.9
        assert c2.model.alpha == 0.627, \
            "Config instances should not share model state"
