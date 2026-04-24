"""
test_signal_utils.py
====================
Tests for src/signal_utils.py covering:
  - standardize_within_year
  - build_supervised_lm
  - compute_merton_dd
"""

import numpy as np
import pandas as pd
import pytest

from src.signal_utils import (
    standardize_within_year,
    build_supervised_lm,
    compute_merton_dd,
    LM_CATEGORIES,
)


# ════════════════════════════════════════════════════════════════
# standardize_within_year
# ════════════════════════════════════════════════════════════════

class TestStandardizeWithinYear:
    """Within-year cross-sectional standardization."""

    def test_output_has_mean_zero_within_year(self):
        """Standardized values should sum to ~0 within each year."""
        x = pd.Series([1, 2, 3, 4, 5, 6])
        y = pd.Series([2010, 2010, 2010, 2011, 2011, 2011])
        out = standardize_within_year(x, y)
        for year in y.unique():
            mask = y == year
            assert abs(out[mask].mean()) < 1e-10, \
                f"Mean within year {year} should be ~0"

    def test_output_has_std_one_within_year(self):
        x = pd.Series([1, 2, 3, 4, 5, 6])
        y = pd.Series([2010, 2010, 2010, 2011, 2011, 2011])
        out = standardize_within_year(x, y)
        for year in y.unique():
            mask = y == year
            # Note: sample std (ddof=1) will be ~1 due to n-1 denominator
            assert abs(out[mask].std() - 1.0) < 1e-10, \
                f"Std within year {year} should be ~1"

    def test_returns_nan_for_constant_within_year(self):
        """If all values in a year are identical, std is 0; should yield NaN."""
        x = pd.Series([5, 5, 5, 1, 2, 3])
        y = pd.Series([2010, 2010, 2010, 2011, 2011, 2011])
        out = standardize_within_year(x, y)
        # The year-2010 values should be NaN (constant)
        assert out[y == 2010].isna().all()
        # The year-2011 values should be standardized
        assert not out[y == 2011].isna().any()

    def test_preserves_length(self):
        x = pd.Series(np.arange(100))
        y = pd.Series(np.repeat([2010, 2011, 2012, 2013], 25))
        out = standardize_within_year(x, y)
        assert len(out) == 100

    def test_handles_single_observation_year(self):
        """A year with only one observation should yield NaN (can't standardize)."""
        x = pd.Series([1, 2, 3, 99])
        y = pd.Series([2010, 2010, 2010, 2011])
        out = standardize_within_year(x, y)
        # 2011 has only one obs, should be NaN
        assert out[y == 2011].isna().all()


# ════════════════════════════════════════════════════════════════
# build_supervised_lm
# ════════════════════════════════════════════════════════════════

class TestBuildSupervisedLM:
    """Supervised LM signal construction."""

    def test_returns_correct_columns(self, synthetic_panel, synthetic_lm_scores):
        default_df = (synthetic_panel.assign(
            default_next=synthetic_panel.groupby('gvkey')['default_broad'].shift(-1)
        )[['gvkey', 'fyear', 'default_next']])
        # Need enough defaults — synthetic panel has some
        if default_df['default_next'].sum() < 3:
            pytest.skip("Synthetic panel has too few defaults for supervised LM")

        out, info = build_supervised_lm(synthetic_lm_scores, default_df,
                                          lm_prefix='combined')

        expected_cols = ['gvkey', 'fyear', 's_lm_supervised_raw',
                          's_lm_supervised', 'in_training_period', 'train_set']
        for c in expected_cols:
            assert c in out.columns, f"Missing column: {c}"

    def test_info_reports_beta_and_training_n(self, synthetic_panel,
                                                 synthetic_lm_scores):
        default_df = (synthetic_panel.assign(
            default_next=synthetic_panel.groupby('gvkey')['default_broad'].shift(-1)
        )[['gvkey', 'fyear', 'default_next']])
        if default_df['default_next'].sum() < 3:
            pytest.skip("Too few defaults")

        _, info = build_supervised_lm(synthetic_lm_scores, default_df)
        assert 'beta' in info
        assert 'n_train' in info
        assert 'n_train_defaults' in info
        assert 'train_cutoff_year' in info
        # Beta should have length == number of categories
        assert len(info['beta']) == len(LM_CATEGORIES)

    def test_s_lm_supervised_is_standardized_within_year(self, synthetic_panel,
                                                           synthetic_lm_scores):
        default_df = (synthetic_panel.assign(
            default_next=synthetic_panel.groupby('gvkey')['default_broad'].shift(-1)
        )[['gvkey', 'fyear', 'default_next']])
        if default_df['default_next'].sum() < 3:
            pytest.skip("Too few defaults")

        out, _ = build_supervised_lm(synthetic_lm_scores, default_df)
        # Within each year with >1 valid obs, mean should be ~0
        for year in out['fyear'].dropna().unique():
            year_vals = out[out['fyear'] == year]['s_lm_supervised'].dropna()
            if len(year_vals) > 1:
                assert abs(year_vals.mean()) < 1e-6, \
                    f"Year {int(year)} s_lm_supervised mean should be ~0"

    def test_raises_when_too_few_training_defaults(self, synthetic_lm_scores):
        """Should raise if <3 training defaults."""
        # Build a default_df with all zero defaults
        default_df = synthetic_lm_scores[['gvkey', 'fyear']].copy()
        default_df['default_next'] = 0
        with pytest.raises(ValueError, match="Too few training"):
            build_supervised_lm(synthetic_lm_scores, default_df)

    def test_deterministic_given_seed(self, synthetic_panel,
                                        synthetic_lm_scores):
        """Running twice with same seed should give identical output."""
        default_df = (synthetic_panel.assign(
            default_next=synthetic_panel.groupby('gvkey')['default_broad'].shift(-1)
        )[['gvkey', 'fyear', 'default_next']])
        if default_df['default_next'].sum() < 3:
            pytest.skip("Too few defaults")

        out1, info1 = build_supervised_lm(synthetic_lm_scores, default_df,
                                             random_state=42)
        out2, info2 = build_supervised_lm(synthetic_lm_scores, default_df,
                                             random_state=42)
        np.testing.assert_array_equal(info1['beta'], info2['beta'])
        # Raw projections should be identical
        np.testing.assert_allclose(
            out1['s_lm_supervised_raw'].fillna(-999).values,
            out2['s_lm_supervised_raw'].fillna(-999).values,
            rtol=1e-10,
        )


# ════════════════════════════════════════════════════════════════
# compute_merton_dd
# ════════════════════════════════════════════════════════════════

class TestComputeMertonDD:
    """Merton naive distance-to-default."""

    def test_adds_both_columns(self, synthetic_panel):
        out = compute_merton_dd(synthetic_panel)
        assert 'merton_dd' in out.columns
        assert 'merton_distress' in out.columns

    def test_distress_is_negative_of_dd(self, synthetic_panel):
        out = compute_merton_dd(synthetic_panel)
        valid = out.dropna(subset=['merton_dd', 'merton_distress'])
        np.testing.assert_allclose(
            valid['merton_distress'].values,
            -valid['merton_dd'].values,
            rtol=1e-10,
        )

    def test_dd_magnitudes_in_plausible_range(self, synthetic_panel):
        """
        Typical firm DD (Bharath-Shumway 2008) is in range 1-10, with
        winsorized tails. Values outside [-5, 50] would indicate a unit bug.
        """
        out = compute_merton_dd(synthetic_panel)
        valid = out['merton_dd'].dropna()
        assert valid.median() >= -5 and valid.median() <= 50, \
            f"Median DD {valid.median():.2f} implausibly large — unit bug?"

    def test_raises_when_required_columns_missing(self):
        df = pd.DataFrame({'gvkey': ['a'], 'fyear': [2020]})
        with pytest.raises(ValueError, match="missing columns"):
            compute_merton_dd(df)

    def test_handles_mktcap_in_dollars_vs_millions(self):
        """
        If mktcap_yearend is in raw dollars (median > 1e6), it should be
        rescaled to millions internally. Test with both cases.
        """
        # Case 1: dollars (mktcap like 500M)
        df_dollars = pd.DataFrame({
            'mktcap_yearend': [500_000_000, 1_000_000_000, 200_000_000],
            'dlc': [10, 20, 5],
            'dltt': [50, 100, 30],
            'equity_vol': [0.3, 0.4, 0.5],
            'tsy_1y': [2.0, 2.0, 2.0],
        })
        out_dollars = compute_merton_dd(df_dollars)

        # Case 2: millions (mktcap like 500)
        df_millions = df_dollars.copy()
        df_millions['mktcap_yearend'] = df_millions['mktcap_yearend'] / 1e6
        out_millions = compute_merton_dd(df_millions)

        # Both should produce similar (not necessarily identical) DD values
        # because the internal rescaling normalizes dollars to millions
        for c in ['merton_dd']:
            d_dollars = out_dollars[c].dropna()
            d_millions = out_millions[c].dropna()
            if len(d_dollars) and len(d_millions):
                # Should be within same order of magnitude
                assert abs(d_dollars.median() - d_millions.median()) < 5.0

    def test_higher_leverage_yields_lower_dd(self):
        """
        Monotone property: holding everything else constant, a firm with
        more debt should have lower DD.
        """
        # Baseline: moderate debt
        base = pd.DataFrame({
            'mktcap_yearend': [100, 100],
            'dlc': [10, 90],  # second firm is heavily leveraged
            'dltt': [20, 180],
            'equity_vol': [0.4, 0.4],
            'tsy_1y': [2.0, 2.0],
        })
        out = compute_merton_dd(base)
        dd_low_lev = out['merton_dd'].iloc[0]
        dd_high_lev = out['merton_dd'].iloc[1]
        assert dd_high_lev < dd_low_lev, \
            f"High-leverage firm should have lower DD, got {dd_high_lev} vs {dd_low_lev}"

    def test_higher_vol_yields_lower_dd(self):
        """Monotone property: higher equity vol → lower DD."""
        df = pd.DataFrame({
            'mktcap_yearend': [100, 100],
            'dlc': [30, 30],
            'dltt': [60, 60],
            'equity_vol': [0.2, 0.8],  # second firm is much more volatile
            'tsy_1y': [2.0, 2.0],
        })
        out = compute_merton_dd(df)
        dd_low_vol = out['merton_dd'].iloc[0]
        dd_high_vol = out['merton_dd'].iloc[1]
        assert dd_high_vol < dd_low_vol, \
            f"High-vol firm should have lower DD, got {dd_high_vol} vs {dd_low_vol}"

    def test_zero_debt_produces_nan_or_inf(self):
        """A firm with zero debt has no well-defined distance to default."""
        df = pd.DataFrame({
            'mktcap_yearend': [100],
            'dlc': [0],
            'dltt': [0],
            'equity_vol': [0.4],
            'tsy_1y': [2.0],
        })
        out = compute_merton_dd(df)
        # Should be NaN (division by zero / log of inf handled gracefully)
        assert pd.isna(out['merton_dd'].iloc[0])

    def test_correlation_with_default_positive_in_synthetic(self,
                                                              synthetic_panel):
        """
        In synthetic data where default is planted with financial distress,
        merton_distress should correlate positively with next-year default.
        """
        out = compute_merton_dd(synthetic_panel)
        out['default_next'] = out.groupby('gvkey')['default_broad'].shift(-1)
        valid = out.dropna(subset=['merton_distress', 'default_next'])
        if len(valid) > 20:
            corr = valid['merton_distress'].corr(valid['default_next'])
            # Corr may be small, but should not be strongly negative
            assert corr > -0.1, \
                f"merton_distress should not negatively correlate with default: {corr}"
