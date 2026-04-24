"""
signal_utils.py
===============
Construction of derived signals used across tables:
  - Supervised LM signal (projection of LM categories onto default direction)
  - Altman Z-score (classic distress predictor)
  - Within-year standardization helper
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


# ────────────────────────────────────────────────────────────────
# Within-year standardization
# ────────────────────────────────────────────────────────────────

def standardize_within_year(series, year_col):
    """
    Standardize a pandas Series within each year.

    series: the Series to standardize
    year_col: a Series of equal length indicating the year for each row

    Returns: a Series of equal length, mean 0 std 1 within each year.
    """
    s = pd.Series(series).reset_index(drop=True)
    y = pd.Series(year_col).reset_index(drop=True)
    out = pd.Series(np.full(len(s), np.nan), index=s.index)

    for year in y.dropna().unique():
        mask = y == year
        vals = s[mask]
        if vals.std() > 0 and len(vals.dropna()) > 1:
            out[mask] = (vals - vals.mean()) / vals.std()
    return out


# ────────────────────────────────────────────────────────────────
# Supervised LM signal
# ────────────────────────────────────────────────────────────────

# Standard-order LM categories used in our construction
LM_CATEGORIES = ['negative', 'uncertainty', 'weak_modal']


def build_supervised_lm(lm_scores, default_df,
                         lm_prefix='combined',
                         categories=LM_CATEGORIES,
                         train_cutoff_year=None,
                         random_state=42):
    """
    Construct a supervised LM signal by the same procedure as FinBERT:
    logistic regression of default_next on LM category fractions,
    with L2 penalty and balanced class weights.

    Arguments:
        lm_scores: DataFrame with columns gvkey, fyear, and
            {prefix}_{cat}_fraction for each category
        default_df: DataFrame with columns gvkey, fyear, default_next
            where default_next is 1 if firm defaults in fyear+1
        lm_prefix: which text section ('item1a', 'item7', 'combined')
        categories: which LM categories to include
        train_cutoff_year: if None, use median year of default_df as split
        random_state: seed for reproducibility

    Returns:
        DataFrame with columns (gvkey, fyear, s_lm_supervised_raw,
                                 s_lm_supervised, train_set, in_training_period)
        where s_lm_supervised is standardized within year.
    """
    # Build feature matrix columns
    feature_cols = [f'{lm_prefix}_{cat}_fraction' for cat in categories]

    # Ensure types are consistent
    lm = lm_scores.copy()
    lm['gvkey'] = lm['gvkey'].astype(str).str.strip()
    lm['fyear'] = pd.to_numeric(lm['fyear'], errors='coerce')

    defaults = default_df.copy()
    defaults['gvkey'] = defaults['gvkey'].astype(str).str.strip()
    defaults['fyear'] = pd.to_numeric(defaults['fyear'], errors='coerce')

    # Merge
    merged = lm.merge(defaults[['gvkey', 'fyear', 'default_next']],
                       on=['gvkey', 'fyear'], how='left')

    # Valid subset: has default_next AND all feature values
    valid_mask = merged['default_next'].notna() & merged[feature_cols].notna().all(axis=1)
    valid = merged[valid_mask].copy()
    X_valid = valid[feature_cols].values
    y_valid = valid['default_next'].astype(int).values

    if train_cutoff_year is None:
        train_cutoff_year = int(valid['fyear'].median())

    train_mask = valid['fyear'] <= train_cutoff_year
    X_train = X_valid[train_mask.values]
    y_train = y_valid[train_mask.values]

    if len(X_train) < 10 or y_train.sum() < 3:
        raise ValueError(
            f"Too few training observations or defaults for supervised LM: "
            f"n={len(X_train)}, defaults={int(y_train.sum())}"
        )

    # Fit logistic regression
    clf = LogisticRegression(
        penalty='l2', C=1.0, max_iter=1000,
        class_weight='balanced', solver='lbfgs',
        random_state=random_state,
    )
    clf.fit(X_train, y_train)
    beta = clf.coef_[0]
    intercept = clf.intercept_[0]

    # Project ALL rows that have feature values (not just valid)
    feat_complete = merged[feature_cols].notna().all(axis=1)
    proj = np.zeros(len(merged))
    proj[:] = np.nan
    proj[feat_complete.values] = merged.loc[feat_complete, feature_cols].values @ beta

    out = merged[['gvkey', 'fyear']].copy()
    out['s_lm_supervised_raw'] = proj
    out['in_training_period'] = merged['fyear'] <= train_cutoff_year
    out['train_set'] = train_mask.reindex(out.index).fillna(False).values

    # Standardize within year
    out['s_lm_supervised'] = standardize_within_year(
        out['s_lm_supervised_raw'], out['fyear']
    ).values

    return out, {'beta': beta, 'intercept': intercept,
                 'feature_cols': feature_cols,
                 'train_cutoff_year': train_cutoff_year,
                 'n_train': len(X_train),
                 'n_train_defaults': int(y_train.sum())}


# ────────────────────────────────────────────────────────────────
# Merton naive distance-to-default (Bharath-Shumway 2008)
# ────────────────────────────────────────────────────────────────

def compute_merton_dd(panel, ratio_name='merton_dd', distress_name='merton_distress'):
    r"""
    Compute the naive Merton distance-to-default following
    Bharath and Shumway (2008, RFS).

    Formula:
        DD_naive = [ln(V/F) + (r - 0.5 * sigma_V^2) * T] / (sigma_V * sqrt(T))

    Where:
        V       = equity market value + book debt
        F       = short-term debt + 0.5 * long-term debt (default point)
        T       = 1 year
        r       = 1-year Treasury yield (risk-free)
        sigma_V = (E/V) * sigma_E + (F/V) * sigma_D
        sigma_D = 0.05 + 0.25 * sigma_E (standard approximation)

    Arguments:
        panel: DataFrame with columns mktcap_yearend, dlc, dltt,
               equity_vol, tsy_1y
        ratio_name: column name for distance-to-default (higher = safer)
        distress_name: column name for -DD (higher = more distressed),
            which is what you'd use as a regressor to match the sign
            convention of FinBERT (higher = more distress)

    Returns:
        Copy of panel with two new columns (DD and -DD).
        Missing components yield NaN values.
    """
    df = panel.copy()

    req = ['mktcap_yearend', 'dlc', 'dltt', 'equity_vol', 'tsy_1y']
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Panel missing columns for Merton DD: {missing}")

    import numpy as np

    # Convert interest rate from percentage to fraction.
    # FRED 1yr Treasury values are reported as percent (e.g., 2.5 for 2.5%).
    # Typical values during our sample are 0.1%-5%, always below 10.
    # We divide by 100 whenever the max is clearly above 1 (i.e., percent).
    r = df['tsy_1y'].copy()
    r = r / 100.0  # FRED convention: always percent

    # Ensure equity_vol is in annualized fraction form (not percent)
    sigma_E = df['equity_vol'].copy()
    if sigma_E.dropna().median() > 2:
        # Values are big — likely percent or daily vol scaled differently
        # Leave as-is and note; user can verify
        pass

    # Unit normalization: mktcap_yearend is in raw dollars while dlc/dltt
    # are in millions (Compustat convention). Rescale mktcap to millions
    # if its magnitude suggests it's in raw dollars (median > 10^6).
    # This is a defensive check; ideally the panel would already have
    # consistent units.
    E_raw = df['mktcap_yearend']
    mktcap_median = E_raw.dropna().median()
    if mktcap_median > 1e6:
        E = E_raw / 1e6
    else:
        E = E_raw

    # Face value of debt (already in millions)
    F = df['dlc'].fillna(0) + 0.5 * df['dltt'].fillna(0)
    # Firm value (naive): equity + debt, now in consistent units
    V = E + F

    # Avoid taking log of zero / negative
    V = V.where(V > 0)
    F_clip = F.where(F > 0)

    # Debt volatility (Bharath-Shumway approximation)
    sigma_D = 0.05 + 0.25 * sigma_E

    # Asset volatility (weighted)
    sigma_V = (E / V) * sigma_E + (F / V) * sigma_D

    # Horizon
    T = 1.0

    # Distance to default
    DD = (np.log(V / F_clip) + (r - 0.5 * sigma_V ** 2) * T) / (sigma_V * np.sqrt(T))

    # Winsorize at 1/99 — raw DD can have extreme values
    if DD.dropna().size > 100:
        lo, hi = DD.quantile([0.01, 0.99])
        DD = DD.clip(lower=lo, upper=hi)

    df[ratio_name] = DD
    df[distress_name] = -DD

    return df


# ────────────────────────────────────────────────────────────────
# Altman Z-score (kept for reference, requires more Compustat fields)
# ────────────────────────────────────────────────────────────────

def compute_altman_z(panel, ratio_name='altman_z'):
    """
    Compute the classic Altman (1968) Z-score for public firms:

        Z = 1.2 · WC/TA + 1.4 · RE/TA + 3.3 · EBIT/TA
          + 0.6 · ME/TL + 1.0 · Sales/TA

    Where:
      WC   = working capital = act - lct  (current assets - current liabilities)
      RE   = retained earnings (re)
      EBIT = earnings before interest and taxes (ebit, or oiadp as fallback)
      ME   = market value of equity (mktcap)
      TL   = total liabilities (lt, or at - ceq as fallback)
      TA   = total assets (at)
      Sales = revenue (sale)

    Adds column `ratio_name` to a copy of panel. Winsorizes at 1/99%
    to reduce outlier impact.

    Returns: the panel with the new column. Missing ratio components
    yield NaN Z-score.
    """
    df = panel.copy()

    # EBIT: prefer ebit, fall back to oiadp
    ebit = df.get('ebit')
    if ebit is None or ebit.isna().all():
        ebit = df.get('oiadp')

    # TL: prefer lt, fall back to at - ceq
    tl = df.get('lt')
    if tl is None or tl.isna().all():
        if 'at' in df and 'ceq' in df:
            tl = df['at'] - df['ceq']

    # WC = act - lct
    if 'act' in df and 'lct' in df:
        wc = df['act'] - df['lct']
    else:
        wc = None

    at = df.get('at')
    re = df.get('re')
    me = df.get('mktcap')
    sale = df.get('sale')

    # Build Z-score, NaN-safe
    def safe_div(num, den):
        if num is None or den is None:
            return pd.Series(np.nan, index=df.index)
        return num / den.replace(0, np.nan)

    z = (1.2 * safe_div(wc, at)
         + 1.4 * safe_div(re, at)
         + 3.3 * safe_div(ebit, at)
         + 0.6 * safe_div(me, tl)
         + 1.0 * safe_div(sale, at))

    # Winsorize at 1/99 to reduce outlier effects
    if z.dropna().size > 100:
        lo, hi = z.quantile([0.01, 0.99])
        z = z.clip(lower=lo, upper=hi)

    df[ratio_name] = z
    return df
