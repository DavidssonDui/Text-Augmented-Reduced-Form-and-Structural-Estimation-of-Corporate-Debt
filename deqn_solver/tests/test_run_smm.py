"""
test_run_smm.py
===============
Tests for the pure helper functions in scripts/run_smm.py.

The script uses sys.path manipulation to import its dependencies,
which makes direct package import awkward. We work around this by
loading the source file as a module via importlib.

Covers:
  - PARAM_NAMES / N_PARAMS / PARAM_BOUNDS / INIT_PARAMS — module constants
  - clip_params: bounds enforcement
  - apply_params_to_config: parameter array → Config
  - compute_loss: weighted sum of squares
  - build_weights: diagonal W = 1/m_data²

NOT covered (requires solver and lengthy runs):
  - evaluate_theta: trains a v6b model
  - run_nelder_mead: full optimization
"""

import importlib.util
import os
import sys

import numpy as np
import pytest

# Load the run_smm.py file as a module since it lives in scripts/
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# Make 'src' resolve to deqn_solver.src for the relative imports inside run_smm.py
sys.path.insert(0, os.path.join(ROOT, 'deqn_solver'))


def _find_run_smm():
    """Locate run_smm.py — try several layouts."""
    candidates = [
        os.path.join(ROOT, 'deqn_solver', 'scripts', 'run_smm.py'),
        os.path.join(ROOT, 'deqn_solver', 'src', 'run_smm.py'),
        # When tests are run from inside deqn_solver/, ROOT is deqn_solver itself
        os.path.join(ROOT, 'scripts', 'run_smm.py'),
        os.path.join(ROOT, 'src', 'run_smm.py'),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"run_smm.py not found in any of: {candidates}")


RUN_SMM_PATH = _find_run_smm()


def _load_run_smm():
    """Load run_smm.py as a fresh module for each test session."""
    spec = importlib.util.spec_from_file_location("run_smm", RUN_SMM_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def run_smm_module():
    """Load run_smm.py once per test module."""
    return _load_run_smm()


# ════════════════════════════════════════════════════════════════
# Module constants
# ════════════════════════════════════════════════════════════════

class TestModuleConstants:
    """The PARAM_NAMES, INIT_PARAMS, PARAM_BOUNDS structure."""

    def test_param_names_length_matches_n_params(self, run_smm_module):
        assert len(run_smm_module.PARAM_NAMES) == run_smm_module.N_PARAMS

    def test_n_params_is_9(self, run_smm_module):
        """The estimation has 9 parameters."""
        assert run_smm_module.N_PARAMS == 9

    def test_init_params_keys_match_param_names(self, run_smm_module):
        """INIT_PARAMS must have an entry for each PARAM_NAMES entry."""
        for name in run_smm_module.PARAM_NAMES:
            assert name in run_smm_module.INIT_PARAMS, \
                f"INIT_PARAMS missing entry for {name}"

    def test_bounds_keys_match_param_names(self, run_smm_module):
        for name in run_smm_module.PARAM_NAMES:
            assert name in run_smm_module.PARAM_BOUNDS, \
                f"PARAM_BOUNDS missing entry for {name}"

    def test_init_values_within_bounds(self, run_smm_module):
        """Initial values must satisfy their own bounds."""
        for name in run_smm_module.PARAM_NAMES:
            init = run_smm_module.INIT_PARAMS[name]
            lo, hi = run_smm_module.PARAM_BOUNDS[name]
            assert lo <= init <= hi, \
                f"Initial value {name}={init} violates bounds [{lo}, {hi}]"

    def test_bounds_are_well_formed(self, run_smm_module):
        """For each bound (lo, hi), lo < hi."""
        for name, (lo, hi) in run_smm_module.PARAM_BOUNDS.items():
            assert lo < hi, f"Bounds for {name}: lo={lo} >= hi={hi}"

    def test_lambda_text_starts_at_03(self, run_smm_module):
        """The methodology specifies INIT lambda_text = 0.3."""
        assert run_smm_module.INIT_PARAMS['lambda_text'] == 0.3

    def test_lambda_text_bounds_are_unit_interval(self, run_smm_module):
        """lambda_text ∈ [0, 1] by economic interpretation."""
        lo, hi = run_smm_module.PARAM_BOUNDS['lambda_text']
        assert lo == 0.0
        assert hi == 1.0


# ════════════════════════════════════════════════════════════════
# clip_params
# ════════════════════════════════════════════════════════════════

class TestClipParams:
    """Enforce parameter bounds."""

    def test_within_bounds_unchanged(self, run_smm_module):
        """Theta inside bounds should be unchanged."""
        theta = np.array([
            run_smm_module.INIT_PARAMS[n] for n in run_smm_module.PARAM_NAMES
        ])
        clipped = run_smm_module.clip_params(theta)
        np.testing.assert_array_equal(clipped, theta)

    def test_below_bound_clipped_to_lo(self, run_smm_module):
        """Theta below the lower bound should be clipped to lo."""
        theta = np.zeros(run_smm_module.N_PARAMS)
        theta[0] = -1.0  # alpha can't be negative
        clipped = run_smm_module.clip_params(theta)
        lo_alpha, _ = run_smm_module.PARAM_BOUNDS['alpha']
        assert clipped[0] == lo_alpha

    def test_above_bound_clipped_to_hi(self, run_smm_module):
        """Theta above the upper bound should be clipped to hi."""
        theta = np.array([
            run_smm_module.INIT_PARAMS[n] for n in run_smm_module.PARAM_NAMES
        ])
        # alpha is at index 0; push it above its upper bound
        _, hi_alpha = run_smm_module.PARAM_BOUNDS['alpha']
        theta[0] = hi_alpha + 1.0
        clipped = run_smm_module.clip_params(theta)
        assert clipped[0] == hi_alpha

    def test_lambda_text_clipped_to_unit(self, run_smm_module):
        """If lambda_text would be 1.5, clip to 1.0."""
        theta = np.array([
            run_smm_module.INIT_PARAMS[n] for n in run_smm_module.PARAM_NAMES
        ])
        idx = run_smm_module.PARAM_NAMES.index('lambda_text')
        theta[idx] = 1.5
        clipped = run_smm_module.clip_params(theta)
        assert clipped[idx] == 1.0

    def test_does_not_mutate_input(self, run_smm_module):
        theta_orig = np.array([
            run_smm_module.INIT_PARAMS[n] for n in run_smm_module.PARAM_NAMES
        ])
        theta_orig[0] = 100.0  # extreme
        theta_copy = theta_orig.copy()
        run_smm_module.clip_params(theta_orig)
        np.testing.assert_array_equal(theta_orig, theta_copy)


# ════════════════════════════════════════════════════════════════
# apply_params_to_config
# ════════════════════════════════════════════════════════════════

class TestApplyParamsToConfig:
    """Set a parameter array onto a Config object."""

    def test_returns_config_with_updated_model(self, run_smm_module):
        from deqn_solver.src.config import Config
        cfg = Config()
        theta = np.array([
            0.5, 0.7, 0.15, 0.1, 0.001, 0.12, 0.7, 0.4, 1.2,
        ])  # 9 parameters
        result = run_smm_module.apply_params_to_config(cfg, theta)
        # The function modifies in place AND returns the cfg
        assert result is cfg
        assert cfg.model.alpha == 0.5

    def test_writes_lambda_text_correctly(self, run_smm_module):
        from deqn_solver.src.config import Config
        cfg = Config()
        theta = np.array([
            run_smm_module.INIT_PARAMS[n] for n in run_smm_module.PARAM_NAMES
        ])
        idx = run_smm_module.PARAM_NAMES.index('lambda_text')
        theta[idx] = 0.7
        run_smm_module.apply_params_to_config(cfg, theta)
        assert cfg.model.lambda_text == 0.7

    def test_writes_all_9_params(self, run_smm_module):
        """Each of the 9 parameters in PARAM_NAMES should be set."""
        from deqn_solver.src.config import Config
        cfg = Config()
        theta = np.array([
            run_smm_module.INIT_PARAMS[n] for n in run_smm_module.PARAM_NAMES
        ])
        # Modify every position
        for i in range(len(theta)):
            theta[i] *= 0.9 if theta[i] != 0 else 1.0
        run_smm_module.apply_params_to_config(cfg, theta)
        # Check each parameter on the model object
        for i, name in enumerate(run_smm_module.PARAM_NAMES):
            actual = getattr(cfg.model, name)
            assert abs(actual - theta[i]) < 1e-9, \
                f"{name}: expected {theta[i]}, got {actual}"


# ════════════════════════════════════════════════════════════════
# compute_loss
# ════════════════════════════════════════════════════════════════

class TestComputeLoss:
    """Weighted sum of squares: (m_sim - m_data)' W (m_sim - m_data)."""

    def test_zero_when_perfect_match(self, run_smm_module):
        m_data = np.array([1.0, 2.0, 0.5])
        m_sim = m_data.copy()
        W = np.ones(3)
        loss = run_smm_module.compute_loss(m_sim, m_data, W)
        assert loss == 0.0

    def test_hand_calculation(self, run_smm_module):
        """
        Loss = sum((sim - data)^2 * W).
        diff = [0.1, -0.2, 0.0]
        squared = [0.01, 0.04, 0.0]
        weights = [1.0, 2.0, 3.0]
        loss = 0.01 + 0.08 + 0.0 = 0.09
        """
        m_data = np.array([1.0, 2.0, 0.5])
        m_sim = np.array([1.1, 1.8, 0.5])
        W = np.array([1.0, 2.0, 3.0])
        expected = 0.01 + 0.08 + 0.0
        assert abs(run_smm_module.compute_loss(m_sim, m_data, W) - expected) < 1e-9

    def test_always_nonnegative(self, run_smm_module):
        rng = np.random.default_rng(42)
        for _ in range(20):
            m_data = rng.normal(0, 1, 22)
            m_sim = rng.normal(0, 1, 22)
            W = np.abs(rng.normal(1, 0.5, 22))  # weights should be positive
            assert run_smm_module.compute_loss(m_sim, m_data, W) >= 0

    def test_returns_float(self, run_smm_module):
        m_data = np.zeros(3)
        m_sim = np.array([1.0, 2.0, 3.0])
        W = np.ones(3)
        loss = run_smm_module.compute_loss(m_sim, m_data, W)
        assert isinstance(loss, float)


# ════════════════════════════════════════════════════════════════
# build_weights
# ════════════════════════════════════════════════════════════════

class TestBuildWeights:
    """Diagonal W = 1/m_data^2, with floor for tiny values."""

    def test_basic_inverse_squared(self, run_smm_module):
        """For non-tiny values, W[i] = 1/m_data[i]^2."""
        m_data = np.array([1.0, 2.0, 0.5])
        W = run_smm_module.build_weights(m_data)
        expected = 1.0 / (m_data ** 2)
        np.testing.assert_allclose(W, expected, rtol=1e-9)

    def test_floor_protects_against_zero(self, run_smm_module):
        """If m_data[i] = 0, W[i] should be finite (use floor instead)."""
        m_data = np.array([1.0, 0.0, 2.0])
        W = run_smm_module.build_weights(m_data, floor=1e-4)
        assert np.all(np.isfinite(W))
        # The zero element should use the floor
        expected_zero = 1.0 / (1e-4 ** 2)
        assert abs(W[1] - expected_zero) < 1e-3

    def test_floor_protects_against_tiny_negative(self, run_smm_module):
        """Negative values smaller in magnitude than the floor → use floor."""
        m_data = np.array([1.0, -1e-8, 2.0])
        W = run_smm_module.build_weights(m_data, floor=1e-4)
        assert np.all(np.isfinite(W))
        expected = 1.0 / (1e-4 ** 2)
        assert abs(W[1] - expected) < 1e-3

    def test_uses_absolute_value(self, run_smm_module):
        """Sign doesn't matter — only magnitude."""
        m_data_pos = np.array([2.0])
        m_data_neg = np.array([-2.0])
        W_pos = run_smm_module.build_weights(m_data_pos)
        W_neg = run_smm_module.build_weights(m_data_neg)
        np.testing.assert_allclose(W_pos, W_neg)

    def test_output_shape_matches_input(self, run_smm_module):
        m_data = np.zeros(22)
        W = run_smm_module.build_weights(m_data)
        assert W.shape == (22,)
