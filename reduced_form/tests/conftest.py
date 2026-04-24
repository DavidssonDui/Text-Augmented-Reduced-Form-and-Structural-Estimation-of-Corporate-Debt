"""
conftest.py
===========
Shared pytest fixtures for the reduced-form analysis test suite.

Fixtures:
  - synthetic_panel: a small deterministic panel for testing regression
    machinery without needing the full data
  - synthetic_lm_scores: minimal LM-scores frame matching the real schema
  - small_linear_dataset: y = 2 + 0.5*x + noise, for OLS hand-verification
  - hand_ar1_series: series with known AR(1) autocorrelation
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Make src importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# ────────────────────────────────────────────────────────────────
# Deterministic synthetic data
# ────────────────────────────────────────────────────────────────

@pytest.fixture
def rng():
    """A seeded numpy RNG for deterministic tests."""
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_panel(rng):
    """
    Small panel with all the columns reduced-form utils expect.
    10 firms × 10 years = 100 firm-years.

    Constructed to have:
      - Some firms with defaults (about 5% of firm-years)
      - Non-trivial variation in controls
      - A planted relationship between s_finbert and default
    """
    n_firms = 10
    years = list(range(2010, 2020))
    n_years = len(years)
    n_obs = n_firms * n_years

    firm_ids = np.repeat(np.arange(n_firms) + 1000, n_years)
    year_ids = np.tile(years, n_firms)

    # Controls
    log_at = rng.normal(6, 1.5, n_obs)
    book_leverage_w = np.clip(rng.normal(0.3, 0.2, n_obs), 0, 1)
    tobins_q_fix_w = np.clip(rng.normal(1.5, 0.5, n_obs), 0, 5)
    profitability_w = rng.normal(0.05, 0.1, n_obs)
    cash_ratio_w = np.clip(rng.normal(0.2, 0.15, n_obs), 0, 1)

    # Text signals standardized within year
    s_finbert_raw = rng.normal(0, 1, n_obs)
    s_lm_raw = rng.normal(0, 1, n_obs)

    # Default: rare, but with a planted relationship with s_finbert
    # Higher s_finbert → higher probability of default
    # Use a higher intercept so we get ~10-20 defaults in our 100 obs panel
    logit = -3 + 0.8 * s_finbert_raw + 0.3 * book_leverage_w
    prob = 1 / (1 + np.exp(-logit))
    default = (rng.uniform(0, 1, n_obs) < prob).astype(int)

    # Build Merton inputs
    mktcap_yearend = np.exp(log_at + 14) + 1e6  # in dollars
    dlc = np.clip(rng.normal(30, 20, n_obs), 0, None)
    dltt = np.clip(rng.normal(150, 100, n_obs), 0, None)
    equity_vol = np.clip(rng.normal(0.4, 0.15, n_obs), 0.05, 2.0)
    tsy_1y = np.tile(rng.uniform(0.5, 4.0, n_years), n_firms)  # varies by year

    panel = pd.DataFrame({
        'gvkey': firm_ids.astype(str),
        'fyear': year_ids,
        'sic': 3000 + rng.integers(0, 999, n_obs),  # industry range
        'at': np.exp(log_at),
        'book_leverage_w': book_leverage_w,
        'tobins_q_fix_w': tobins_q_fix_w,
        'profitability_w': profitability_w,
        'cash_ratio_w': cash_ratio_w,
        's_finbert': s_finbert_raw,
        's_lm': s_lm_raw,
        'mktcap_yearend': mktcap_yearend,
        'dlc': dlc,
        'dltt': dltt,
        'equity_vol': equity_vol,
        'tsy_1y': tsy_1y,
        'default_broad': default,
    })

    # Sort for stability
    return panel.sort_values(['gvkey', 'fyear']).reset_index(drop=True)


@pytest.fixture
def synthetic_lm_scores(rng, synthetic_panel):
    """
    LM scores matching the schema of the real lm_scores.csv, aligned to
    synthetic_panel on (gvkey, fyear).
    """
    panel = synthetic_panel
    n = len(panel)

    # Generate realistic-looking word-count fractions
    rows = []
    for _, r in panel.iterrows():
        # Generate correlated neg/unc/wk_modal fractions
        base = rng.uniform(0.005, 0.04)
        rows.append({
            'gvkey': r['gvkey'],
            'cik': int(r['gvkey']) * 10,
            'fyear': r['fyear'],
            'combined_negative_fraction': base + rng.normal(0, 0.005),
            'combined_uncertainty_fraction': base * 0.8 + rng.normal(0, 0.003),
            'combined_weak_modal_fraction': base * 0.5 + rng.normal(0, 0.002),
            'combined_positive_fraction': 0.01 + rng.uniform(0, 0.01),
            'combined_litigious_fraction': 0.005 + rng.uniform(0, 0.005),
            'combined_strong_modal_fraction': 0.003 + rng.uniform(0, 0.002),
            'combined_word_count': rng.integers(5000, 15000),
        })
    return pd.DataFrame(rows)


@pytest.fixture
def small_linear_dataset(rng):
    """
    A trivial dataset where y = 2 + 0.5*x + noise.
    Used to verify run_lpm against sm.OLS hand-calculation.
    """
    n = 200
    x = rng.uniform(-2, 2, n)
    y = 2.0 + 0.5 * x + rng.normal(0, 0.1, n)
    return pd.DataFrame({
        'y': y, 'x': x,
        'cluster': np.repeat(np.arange(20), n // 20),  # for clustering
        'gvkey': np.repeat(np.arange(20), n // 20).astype(str),
    })


@pytest.fixture
def hand_ar1_series():
    """
    A simple series with known lag-1 autocorrelation ≈ 0.7.
    """
    rng = np.random.default_rng(123)
    n = 500
    x = [0.0]
    for _ in range(n - 1):
        x.append(0.7 * x[-1] + rng.normal(0, 1))
    return pd.Series(x)
