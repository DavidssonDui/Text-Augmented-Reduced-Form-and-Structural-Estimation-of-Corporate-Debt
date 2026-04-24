"""
regression_utils.py
===================
Shared regression machinery for the reduced-form analysis package.

Functions:
  - run_lpm(df, y, X, cluster, fe): linear probability model with
    cluster-robust standard errors and optional fixed effects
  - run_logit(df, y, X, cluster, fe): analogous logit with robust SE
  - prepare_sample(df): add standard derived columns (log_at, sic1, etc.)

All functions are deterministic (no random state) and take/return
pandas DataFrames to keep composition transparent.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit


# ────────────────────────────────────────────────────────────────
# Sample preparation
# ────────────────────────────────────────────────────────────────

def prepare_sample(df):
    """
    Add derived columns used across analyses.

    Adds: log_at, sic1 (1-digit SIC), fyear_cat, forward-default indicators
    (def_t1, def_t2, def_t3, def_within_2, def_within_3).

    Assumes df is sorted or can be sorted by (gvkey, fyear).
    """
    df = df.copy()
    df = df.sort_values(['gvkey', 'fyear']).reset_index(drop=True)

    # Log assets
    if 'at' in df.columns:
        df['log_at'] = np.log(df['at'].clip(lower=1))

    # 1-digit SIC
    if 'sic' in df.columns:
        sic_numeric = pd.to_numeric(df['sic'], errors='coerce')
        sic1 = (sic_numeric // 1000)
        # Fill NaN with a sentinel and convert to string
        df['sic1'] = sic1.fillna(-1).astype(int).astype(str)
        df['sic1'] = df['sic1'].replace({'-1': 'missing'})

    # Year as categorical
    if 'fyear' in df.columns:
        df['fyear_cat'] = df['fyear'].astype(int).astype(str)

    # Forward-default indicators
    if 'default_broad' in df.columns:
        for h in [1, 2, 3]:
            df[f'def_t{h}'] = df.groupby('gvkey')['default_broad'].shift(-h)

        df['def_within_1'] = df['def_t1']
        df['def_within_2'] = ((df['def_t1'] == 1) | (df['def_t2'] == 1)).astype('float')
        df['def_within_3'] = ((df['def_t1'] == 1) | (df['def_t2'] == 1)
                              | (df['def_t3'] == 1)).astype('float')
        # Handle missing future values correctly
        for h in [2, 3]:
            observed = df[[f'def_t{i}' for i in range(1, h + 1)]].notna().any(axis=1)
            df.loc[~observed, f'def_within_{h}'] = np.nan

    return df


# ────────────────────────────────────────────────────────────────
# Linear probability model
# ────────────────────────────────────────────────────────────────

def _build_design_matrix(df, x_cols, fe_cols):
    """Build the design matrix (with dummies for FEs)."""
    X = df[x_cols].copy()
    if fe_cols:
        for fe in fe_cols:
            dummies = pd.get_dummies(df[fe], prefix=fe, drop_first=True, dtype=float)
            X = pd.concat([X, dummies], axis=1)
    X = X.astype(float)
    return sm.add_constant(X)


def run_lpm(df, y_col, x_cols, cluster_col='gvkey', fe_cols=None):
    """
    Linear probability model with firm-clustered standard errors.

    Returns (fitted_model, clean_frame) where clean_frame is the subset
    of df used for estimation (after dropping NaNs).
    """
    cols_needed = [y_col] + list(x_cols) + [cluster_col]
    if fe_cols:
        cols_needed += [c for c in fe_cols if c not in cols_needed]

    df_clean = df[cols_needed].dropna()
    X = _build_design_matrix(df_clean, x_cols, fe_cols)
    y = df_clean[y_col].astype(float).values

    model = sm.OLS(y, X).fit(
        cov_type='cluster',
        cov_kwds={'groups': df_clean[cluster_col].values},
    )
    return model, df_clean


# ────────────────────────────────────────────────────────────────
# Logit model
# ────────────────────────────────────────────────────────────────

def run_logit(df, y_col, x_cols, cluster_col='gvkey', fe_cols=None):
    """
    Logistic regression with firm-clustered standard errors.

    Returns (fitted_model, clean_frame).
    """
    cols_needed = [y_col] + list(x_cols) + [cluster_col]
    if fe_cols:
        cols_needed += [c for c in fe_cols if c not in cols_needed]

    df_clean = df[cols_needed].dropna()
    X = _build_design_matrix(df_clean, x_cols, fe_cols)
    y = df_clean[y_col].astype(int).values

    model = Logit(y, X).fit(
        disp=False,
        method='bfgs',
        cov_type='cluster',
        cov_kwds={'groups': df_clean[cluster_col].values},
        maxiter=100,
    )
    return model, df_clean


# ────────────────────────────────────────────────────────────────
# Formatting helpers
# ────────────────────────────────────────────────────────────────

def stars_for(pval):
    """Return Latex-safe significance stars."""
    if pval < 0.01:
        return '***'
    elif pval < 0.05:
        return '**'
    elif pval < 0.1:
        return '*'
    return ''


def format_coef(coef, pval):
    return f"{coef:+.4f}{stars_for(pval)}"


def format_se(se):
    return f"({se:.4f})"
