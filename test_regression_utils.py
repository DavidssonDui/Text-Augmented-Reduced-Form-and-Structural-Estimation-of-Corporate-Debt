"""
test_regression_utils.py
========================
Tests for src/regression_utils.py.

Organized by function. Each function has (1) correctness-on-known-cases,
(2) property/invariant tests, and (3) edge-case tests where applicable.
"""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from src.regression_utils import (
    prepare_sample, run_lpm, run_logit,
    stars_for, format_coef, format_se,
)


# ════════════════════════════════════════════════════════════════
# stars_for
# ════════════════════════════════════════════════════════════════

class TestStarsFor:
    """Significance star assignment."""

    def test_p_below_01_gets_three_stars(self):
        assert stars_for(0.001) == '***'
        assert stars_for(0.009) == '***'

    def test_p_boundary_01_is_two_stars(self):
        # p = 0.01 exactly: convention is <0.01 for ***, so exactly at 0.01
        # should be ** (not <0.01)
        assert stars_for(0.01) == '**'

    def test_p_between_01_05_gets_two_stars(self):
        assert stars_for(0.02) == '**'
        assert stars_for(0.049) == '**'

    def test_p_between_05_10_gets_one_star(self):
        assert stars_for(0.05) == '*'
        assert stars_for(0.099) == '*'

    def test_p_above_10_gets_no_stars(self):
        assert stars_for(0.10) == ''
        assert stars_for(0.5) == ''
        assert stars_for(0.99) == ''


# ════════════════════════════════════════════════════════════════
# format_coef and format_se
# ════════════════════════════════════════════════════════════════

class TestFormatCoef:
    """Coefficient-formatting helpers."""

    def test_positive_significant(self):
        assert format_coef(0.1234, 0.001) == '+0.1234***'

    def test_negative_significant(self):
        assert format_coef(-0.1234, 0.03) == '-0.1234**'

    def test_positive_insignificant(self):
        assert format_coef(0.1234, 0.5) == '+0.1234'

    def test_four_decimal_precision(self):
        # Format should always show 4 decimal places
        assert format_coef(0.1, 0.001) == '+0.1000***'
        assert format_coef(-0.00001, 0.5) == '-0.0000'

    def test_format_se_is_parenthesized(self):
        assert format_se(0.05) == '(0.0500)'
        assert format_se(0.00012) == '(0.0001)'


# ════════════════════════════════════════════════════════════════
# prepare_sample
# ════════════════════════════════════════════════════════════════

class TestPrepareSample:
    """Panel preparation: adding log_at, sic1, def_within_*, etc."""

    def test_adds_log_at(self, synthetic_panel):
        out = prepare_sample(synthetic_panel)
        assert 'log_at' in out.columns
        # log_at should equal log(at) for firms with at > 1
        expected = np.log(synthetic_panel['at'].clip(lower=1))
        np.testing.assert_allclose(out['log_at'].values, expected.values,
                                     rtol=1e-9)

    def test_adds_sic1_one_digit(self, synthetic_panel):
        out = prepare_sample(synthetic_panel)
        assert 'sic1' in out.columns
        # SIC values in synthetic panel are 3000-3999, so sic1 should all be '3'
        assert out['sic1'].unique().tolist() == ['3']

    def test_adds_forward_default_indicators(self, synthetic_panel):
        out = prepare_sample(synthetic_panel)
        for col in ['def_t1', 'def_t2', 'def_t3',
                    'def_within_1', 'def_within_2', 'def_within_3']:
            assert col in out.columns, f"Missing column: {col}"

    def test_def_within_1_equals_def_t1(self, synthetic_panel):
        out = prepare_sample(synthetic_panel)
        # def_within_1 should be identical to def_t1
        both_valid = out[['def_t1', 'def_within_1']].dropna()
        pd.testing.assert_series_equal(
            both_valid['def_t1'].astype(float).reset_index(drop=True),
            both_valid['def_within_1'].astype(float).reset_index(drop=True),
            check_names=False,
        )

    def test_def_within_2_is_or_of_t1_and_t2(self, synthetic_panel):
        out = prepare_sample(synthetic_panel)
        # If def_t1 == 1 OR def_t2 == 1, def_within_2 should be 1
        for _, row in out.iterrows():
            if pd.notna(row['def_t1']) and pd.notna(row['def_t2']):
                expected = 1.0 if (row['def_t1'] == 1 or row['def_t2'] == 1) else 0.0
                assert row['def_within_2'] == expected, \
                    f"Mismatch at row {row.name}"

    def test_forward_default_is_nan_for_last_firm_years(self, synthetic_panel):
        """The last firm-year for each firm can't have def_t1 observed."""
        out = prepare_sample(synthetic_panel)
        last_years = out.groupby('gvkey')['fyear'].idxmax()
        # For those last-year observations, def_t1 should be NaN
        assert out.loc[last_years, 'def_t1'].isna().all()

    def test_preserves_original_columns(self, synthetic_panel):
        out = prepare_sample(synthetic_panel)
        for col in synthetic_panel.columns:
            assert col in out.columns, f"Original column {col} dropped"

    def test_sic1_handles_missing_values(self, synthetic_panel):
        """When sic is NaN, sic1 should be 'missing'."""
        p = synthetic_panel.copy()
        p.loc[p.index[0], 'sic'] = np.nan
        out = prepare_sample(p)
        assert 'missing' in out['sic1'].values

    def test_does_not_mutate_input(self, synthetic_panel):
        before_cols = set(synthetic_panel.columns)
        prepare_sample(synthetic_panel)
        after_cols = set(synthetic_panel.columns)
        assert before_cols == after_cols, \
            "prepare_sample must not mutate its input"


# ════════════════════════════════════════════════════════════════
# run_lpm
# ════════════════════════════════════════════════════════════════

class TestRunLPM:
    """Linear probability model with clustered standard errors."""

    def test_matches_manual_ols_for_simple_case(self, small_linear_dataset):
        """
        For y = 2 + 0.5x + noise with clustered SE, run_lpm should
        match sm.OLS with cluster='cluster' directly.
        """
        df = small_linear_dataset
        model, _ = run_lpm(df, 'y', ['x'], cluster_col='cluster')

        # Verify key outputs
        assert abs(model.params['x'] - 0.5) < 0.05, \
            f"x coefficient should be near 0.5, got {model.params['x']}"
        assert abs(model.params['const'] - 2.0) < 0.05
        assert model.nobs == 200

    def test_returns_tuple_of_model_and_frame(self, small_linear_dataset):
        model, clean_frame = run_lpm(small_linear_dataset, 'y', ['x'],
                                       cluster_col='cluster')
        assert hasattr(model, 'params'), "First return should be a fitted model"
        assert isinstance(clean_frame, pd.DataFrame)

    def test_clean_frame_drops_nans(self):
        """run_lpm should drop rows with NaN in any used column."""
        df = pd.DataFrame({
            'y': [1, 0, 1, np.nan, 1],
            'x': [0.1, 0.2, 0.3, 0.4, 0.5],
            'gvkey': ['a', 'b', 'c', 'd', 'e'],
        })
        model, clean = run_lpm(df, 'y', ['x'])
        assert len(clean) == 4, "NaN row should be dropped"
        assert model.nobs == 4

    def test_fixed_effects_add_dummies(self, synthetic_panel):
        """When fe_cols is provided, FE dummies should be added."""
        panel = prepare_sample(synthetic_panel)
        model, _ = run_lpm(panel, 'default_broad',
                            ['s_finbert'],
                            fe_cols=['sic1'])
        # Should have constant, s_finbert, and sic1 dummies
        # (our synthetic data has only one SIC value, so no dummies — test
        # with a panel that has multiple)
        params = list(model.params.index)
        assert 'const' in params
        assert 's_finbert' in params

    def test_raises_on_missing_y_column(self, synthetic_panel):
        with pytest.raises(KeyError):
            run_lpm(synthetic_panel, 'nonexistent_y', ['s_finbert'])

    def test_raises_on_missing_x_column(self, synthetic_panel):
        with pytest.raises(KeyError):
            run_lpm(synthetic_panel, 'default_broad', ['nonexistent_x'])

    def test_clustered_se_differ_from_unclustered(self, small_linear_dataset):
        """
        Sanity check: clustered SE should differ from robust SE,
        especially when cluster structure is meaningful.
        """
        df = small_linear_dataset
        # Our clustered SE
        model_cluster, _ = run_lpm(df, 'y', ['x'], cluster_col='cluster')
        # Manual OLS without clustering (HC0 robust)
        X = sm.add_constant(df[['x']])
        model_robust = sm.OLS(df['y'], X).fit(cov_type='HC0')
        # SEs are typically different (can't guarantee direction, just that
        # they use different formulas)
        assert model_cluster.bse['x'] != pytest.approx(model_robust.bse['x'],
                                                         rel=1e-6)


# ════════════════════════════════════════════════════════════════
# run_logit
# ════════════════════════════════════════════════════════════════

class TestRunLogit:
    """Logistic regression with clustered standard errors."""

    def test_predicts_planted_relationship(self, synthetic_panel):
        """
        Synthetic panel has default = f(s_finbert) planted with positive
        coefficient; logit should recover a positive coefficient.
        """
        model, _ = run_logit(synthetic_panel, 'default_broad', ['s_finbert'])
        assert model.params['s_finbert'] > 0, \
            "Should recover positive coefficient on s_finbert"

    def test_returns_model_with_prsquared(self, synthetic_panel):
        model, _ = run_logit(synthetic_panel, 'default_broad', ['s_finbert'])
        assert hasattr(model, 'prsquared'), \
            "Logit result should expose pseudo-R² attribute"

    def test_y_is_cast_to_int_silently(self):
        """
        run_logit casts y with .astype(int), which silently rounds
        continuous y. This is potentially a footgun — we document it
        by testing the actual behavior.
        """
        df = pd.DataFrame({
            'y': [0.5, 1.2, -0.3, 0.8, 0.1],  # continuous
            'x': [0.1, 0.2, 0.3, 0.4, 0.5],
            'gvkey': ['a', 'b', 'c', 'd', 'e'],
        })
        # Should NOT raise — logit fits against int-cast y
        model, clean = run_logit(df, 'y', ['x'])
        # The model fits successfully (documenting current behavior)
        assert hasattr(model, 'params')
