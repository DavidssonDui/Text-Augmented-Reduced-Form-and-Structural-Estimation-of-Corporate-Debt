"""
test_sim_moments.py
===================
Tests for deqn_solver.src.sim_moments.

Covers:
  - MOMENT_KEYS / N_MOMENTS consistency
  - compute_sim_moments_from_panel: the moment computation given a
    pre-built panel DataFrame (the deterministic part of the pipeline)

NOT covered (requires trained networks and lengthy simulation):
  - simulate_panel: depends on solver_v6b and sampling
  - compute_sim_moments: end-to-end wrapper
"""

import numpy as np
import pandas as pd
import pytest

from deqn_solver.src.sim_moments import (
    MOMENT_KEYS, N_MOMENTS, compute_sim_moments_from_panel,
)
from deqn_solver.src.config import ModelParams


# ════════════════════════════════════════════════════════════════
# Module-level constants
# ════════════════════════════════════════════════════════════════

class TestMomentKeys:
    """Sanity checks on the canonical moment list."""

    def test_n_moments_equals_22(self):
        assert N_MOMENTS == 22, "Spec says 22 moments"

    def test_keys_match_n(self):
        assert len(MOMENT_KEYS) == N_MOMENTS

    def test_keys_unique(self):
        assert len(set(MOMENT_KEYS)) == N_MOMENTS, "MOMENT_KEYS must be unique"

    def test_keys_sortable_by_prefix(self):
        """Each key starts with mNN_ where NN is its 1-indexed position."""
        for i, key in enumerate(MOMENT_KEYS):
            expected_prefix = f"m{i+1:02d}_"
            assert key.startswith(expected_prefix), \
                f"Key {key} should start with {expected_prefix}"


# ════════════════════════════════════════════════════════════════
# Synthetic panel for testing moment computation
# ════════════════════════════════════════════════════════════════

@pytest.fixture
def sim_panel():
    """
    Build a small simulated panel with all columns that
    compute_sim_moments_from_panel expects to read.
    """
    rng = np.random.default_rng(42)
    n_firms = 50
    t_sim = 20
    rows = []
    for i in range(n_firms):
        z_t = 1.0
        for t in range(t_sim):
            # AR(1) productivity
            z_t = 0.7 * np.log(z_t) + 0.1 * rng.normal()
            z_t = np.exp(z_t)

            k_prime = max(10.0 + 5 * rng.normal(), 1.0)  # ensure positive
            b_prime = 2.0 + rng.normal() * 3.0  # can be positive or negative
            r_tilde = 0.05 + 0.02 * abs(rng.normal())
            spread = r_tilde - 0.025
            w_realized = 5.0 + rng.normal() * 2.0
            w_tilde = max(w_realized, -5.0)
            s = 0.3 * np.log(z_t) + rng.normal() * 0.5
            financing_need = max(rng.normal() * 5.0, 0.0) - 2.0
            investment = (k_prime - 0.85 * 10) / 10
            book_lev = max(b_prime / k_prime, 0.0)
            cash_ratio = max(-b_prime / k_prime, 0.0)
            prof = (z_t * (k_prime ** 0.6) - 0.15 * 10) / 10
            tobins_q = 1.5 + rng.normal() * 0.3
            default = float(w_realized < -3)

            rows.append({
                'firm_id': i,
                'period': t,
                'w_tilde': w_tilde,
                'z': z_t,
                's': s,
                'k_prime': k_prime,
                'b_prime': b_prime,
                'r_tilde': r_tilde,
                'spread': spread,
                'spread_bps': spread * 10000.0,
                'w_realized': w_realized,
                'default': default,
                'investment': investment,
                'financing_need': financing_need,
                'book_leverage_sim': book_lev,
                'cash_ratio_sim': cash_ratio,
                'profitability_sim': prof,
                'tobins_q_sim': tobins_q,
            })
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════
# compute_sim_moments_from_panel
# ════════════════════════════════════════════════════════════════

class TestComputeSimMomentsFromPanel:
    """Moment computation from a simulated panel."""

    def test_returns_array_of_length_22(self, sim_panel):
        params = ModelParams()
        m = compute_sim_moments_from_panel(sim_panel, params)
        assert isinstance(m, np.ndarray)
        assert m.shape == (22,)

    def test_all_finite(self, sim_panel):
        params = ModelParams()
        m = compute_sim_moments_from_panel(sim_panel, params)
        assert np.all(np.isfinite(m))

    def test_default_rate_target_is_constant(self, sim_panel):
        """
        m11 (default rate target) should always be 0.02 — the Moody's
        external benchmark, NOT computed from the panel.
        """
        params = ModelParams()
        m = compute_sim_moments_from_panel(sim_panel, params)
        idx_m11 = MOMENT_KEYS.index('m11_default_rate_target')
        assert abs(m[idx_m11] - 0.02) < 1e-9, \
            f"Default rate target should be exactly 0.02; got {m[idx_m11]}"

    def test_book_leverage_mean_in_unit_interval(self, sim_panel):
        """Book leverage by construction is in [0, 1]; mean must be too."""
        params = ModelParams()
        m = compute_sim_moments_from_panel(sim_panel, params)
        idx = MOMENT_KEYS.index('m01_book_leverage_mean')
        assert 0 <= m[idx] <= 1

    def test_correlations_in_unit_interval(self, sim_panel):
        """
        Correlation moments (m07, m08, m15, m17, m18, m19, m20) must be
        in [-1, 1].
        """
        params = ModelParams()
        m = compute_sim_moments_from_panel(sim_panel, params)
        corr_indices = [MOMENT_KEYS.index(k) for k in MOMENT_KEYS
                          if any(c in k for c in ['_ar1', 'corr_'])]
        for i in corr_indices:
            assert -1.0 <= m[i] <= 1.0, \
                f"Moment {MOMENT_KEYS[i]} = {m[i]} outside [-1, 1]"

    def test_default_rates_in_unit_interval(self, sim_panel):
        """Quintile default rates should be in [0, 1]."""
        params = ModelParams()
        m = compute_sim_moments_from_panel(sim_panel, params)
        for k in ['m21_default_rate_bottom_quintile',
                   'm22_default_rate_top_quintile']:
            idx = MOMENT_KEYS.index(k)
            assert 0.0 <= m[idx] <= 1.0, \
                f"{k} = {m[idx]} not in [0, 1]"

    def test_deterministic_for_same_input(self, sim_panel):
        """Calling twice with the same panel should give identical output."""
        params = ModelParams()
        m1 = compute_sim_moments_from_panel(sim_panel.copy(), params)
        m2 = compute_sim_moments_from_panel(sim_panel.copy(), params)
        np.testing.assert_allclose(m1, m2, rtol=1e-12)

    def test_moment_18_equals_moment_17(self, sim_panel):
        """
        In simulation, m18 should equal m17 since there's only one signal s
        in the model (no separate FinBERT/LM).
        """
        params = ModelParams()
        m = compute_sim_moments_from_panel(sim_panel, params)
        idx_17 = MOMENT_KEYS.index('m17_corr_s_default_next')
        idx_18 = MOMENT_KEYS.index('m18_corr_other_signal_default_next')
        assert m[idx_17] == m[idx_18], \
            "Sim m18 should equal m17 (same signal in sim)"

    def test_handles_panel_with_no_borrowing(self, sim_panel):
        """If no firms borrow (all b_prime <= 0), spread moments should still produce a valid array."""
        # Force all b_prime to be <= 0 (cash holders only)
        panel = sim_panel.copy()
        panel['b_prime'] = -panel['b_prime'].abs()
        params = ModelParams()
        m = compute_sim_moments_from_panel(panel, params)
        assert m.shape == (22,)
        assert np.all(np.isfinite(m))
        # Spread moments should be exactly zero
        idx_hy = MOMENT_KEYS.index('m12_spread_hy_mean')
        assert m[idx_hy] == 0.0

    def test_single_firm_handled(self):
        """Edge case: panel with only one firm."""
        rows = []
        for t in range(20):
            rows.append({
                'firm_id': 0, 'period': t,
                'w_tilde': 5.0 + t, 'z': 1.0, 's': 0.0,
                'k_prime': 10.0, 'b_prime': 2.0,
                'r_tilde': 0.05, 'spread': 0.025, 'spread_bps': 250,
                'w_realized': 4.0, 'default': 0.0,
                'investment': 0.05, 'financing_need': 1.0,
                'book_leverage_sim': 0.2, 'cash_ratio_sim': 0.0,
                'profitability_sim': 0.1, 'tobins_q_sim': 1.5,
            })
        panel = pd.DataFrame(rows)
        params = ModelParams()
        m = compute_sim_moments_from_panel(panel, params)
        assert m.shape == (22,)
        # Most moments will compute; correlations may be 0 since no variance
        assert np.all(np.isfinite(m))
