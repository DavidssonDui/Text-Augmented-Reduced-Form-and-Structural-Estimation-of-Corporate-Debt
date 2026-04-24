"""
build_panel.py
================
Merge Compustat Annual, CRSP Monthly, LoPucki BRD, and FRED spreads
into a unified firm-year panel for the structural estimation.

Produces two output files:
  1. full_panel.csv     - Complete dataset for SMM estimation
  2. sample_firms.csv   - Minimal subset for the NLP pipeline (gvkey, cik, fyear, conm, default_next_year)

INPUTS (place these in data/raw/):
  comp_funda.csv               - Compustat Annual (from WRDS comp.funda)
  crsp_msf.csv                 - CRSP Monthly Stock File (from WRDS crsp.msf)
  crsp_msedelist.csv           - CRSP Monthly Delisting File (from WRDS crsp.msedelist)
  ccm_linktable.csv            - CRSP-Compustat linking table (from WRDS crsp.ccmxpf_lnkhist)
  lopucki_brd.xlsx             - LoPucki Bankruptcy Research Database
  fred_corporate_spreads.csv   - FRED ICE BofA spread indices (BAMLC0A4CBBB, etc.)
  fred_treasuries.csv          - FRED Treasury yields (DGS10, DGS1, etc.)

USAGE:
  python build_panel.py
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================

RAW_DIR = "/Users/computerboi/Downloads/nlp_pipeline/data/raw"
OUT_DIR = "/Users/computerboi/Downloads/nlp_pipeline/data"

START_YEAR = 2006
END_YEAR = 2022

# Industry exclusions (financials and regulated utilities)
EXCLUDE_SIC_RANGES = [
    (6000, 6999),  # Financials
    (4900, 4999),  # Utilities
]


# ============================================================
# STEP 1: LOAD AND CLEAN COMPUSTAT
# ============================================================

def load_compustat():
    """
    Load Compustat Annual and apply filters following Hennessy-Whited (2007).
    """
    print("\n" + "="*60)
    print("STEP 1: COMPUSTAT ANNUAL")
    print("="*60)

    df = pd.read_csv(os.path.join(RAW_DIR, "comp_funda.csv"))
    df.columns = df.columns.str.lower()
    print(f"Initial: {len(df):,} observations")

    # Standard Compustat filters
    if 'indfmt' in df.columns:
        df = df[df['indfmt'] == 'INDL']
    if 'datafmt' in df.columns:
        df = df[df['datafmt'] == 'STD']
    if 'popsrc' in df.columns:
        df = df[df['popsrc'] == 'D']
    if 'consol' in df.columns:
        df = df[df['consol'] == 'C']
    print(f"After standard filters: {len(df):,}")

    # Year filter
    df = df[(df['fyear'] >= START_YEAR - 1) & (df['fyear'] <= END_YEAR)]
    print(f"After year filter ({START_YEAR-1}-{END_YEAR}): {len(df):,}")

    # Industry exclusions (drop financials and utilities)
    for low, high in EXCLUDE_SIC_RANGES:
        df = df[~((df['sic'] >= low) & (df['sic'] <= high))]
    print(f"After excluding financials/utilities: {len(df):,}")

    # Require positive assets and non-missing key variables
    df = df[df['at'] > 0]
    df = df.dropna(subset=['at', 'dltt', 'dlc'])
    print(f"After requiring at>0 and non-missing debt: {len(df):,}")

    # Clean CIK (Compustat stores it as string with leading zeros)
    if 'cik' in df.columns:
        df['cik'] = pd.to_numeric(df['cik'], errors='coerce')

    # ── Construct Compustat-derived variables ──
    df['total_debt'] = df['dltt'].fillna(0) + df['dlc'].fillna(0)
    df['book_leverage'] = df['total_debt'] / df['at']
    df['profitability'] = df['oibdp'] / df['at']
    df['investment_rate'] = df['capx'] / df['at']
    df['equity_issuance'] = df['sstk'] / df['at']
    df['payout'] = (df['dv'].fillna(0) + df['prstkc'].fillna(0)) / df['at']
    df['debt_issuance_net'] = (
        df['dltis'].fillna(0) - df['dltr'].fillna(0)
    ) / df['at']
    df['cash_ratio'] = df['che'] / df['at']

    # Lagged values for HW07-style ratios
    df = df.sort_values(['gvkey', 'fyear'])
    df['at_lag'] = df.groupby('gvkey')['at'].shift(1)
    df['ppegt_lag'] = df.groupby('gvkey')['ppegt'].shift(1)
    df['investment_rate_lag'] = df['capx'] / df['at_lag']
    df['depreciation_rate'] = df['dp'] / df['ppegt_lag']

    print(f"\nFinal Compustat sample: {len(df):,} firm-years")
    print(f"Unique firms (gvkey): {df['gvkey'].nunique():,}")
    print(f"Year range: {df['fyear'].min()}-{df['fyear'].max()}")

    return df


# ============================================================
# STEP 2: LOAD AND CLEAN CRSP
# ============================================================

def load_crsp():
    """
    Load CRSP Monthly Stock File and Delisting File.
    Compute annual market cap, equity volatility, and default indicators.
    """
    print("\n" + "="*60)
    print("STEP 2: CRSP MONTHLY")
    print("="*60)

    # ── Load Monthly Stock File ──
    msf = pd.read_csv(os.path.join(RAW_DIR, "crsp_msf.csv"))
    msf.columns = msf.columns.str.lower()
    print(f"Loaded {len(msf):,} monthly observations")

    # CRSP encodes missing returns with letter codes (B, C, etc.)
    # Coerce to numeric — non-numeric values become NaN
    msf['ret'] = pd.to_numeric(msf['ret'], errors='coerce')
    msf['prc'] = pd.to_numeric(msf['prc'], errors='coerce')
    msf['shrout'] = pd.to_numeric(msf['shrout'], errors='coerce')

    # CRSP price is sometimes negative (bid-ask midpoint when no trade)
    # Always take absolute value
    msf['prc'] = msf['prc'].abs()

    # Parse date
    msf['date'] = pd.to_datetime(msf['date'])
    msf['year'] = msf['date'].dt.year
    msf['month'] = msf['date'].dt.month

    # Compute market cap (shrout is in thousands)
    msf['mktcap'] = msf['prc'] * msf['shrout'] * 1000  # in dollars

    # ── Load Delisting File ──
    dlst = pd.read_csv(os.path.join(RAW_DIR, "crsp_msedelist.csv"))
    dlst.columns = dlst.columns.str.lower()
    dlst['dlstdt'] = pd.to_datetime(dlst['dlstdt'])
    dlst['dlst_year'] = dlst['dlstdt'].dt.year

    # Coerce dlret to numeric (CRSP also uses letter codes here)
    dlst['dlret'] = pd.to_numeric(dlst['dlret'], errors='coerce')

    # Apply Shumway (1997) -0.30 fix for missing delisting returns on distress delistings
    distress_codes = (dlst['dlstcd'] >= 400) & (dlst['dlstcd'] <= 490)
    dlst.loc[distress_codes & dlst['dlret'].isna(), 'dlret'] = -0.30

    # Default indicator from CRSP (codes 400-490 = financial distress)
    dlst['default_crsp'] = distress_codes.astype(int)

    print(f"Loaded {len(dlst):,} delisting records")
    print(f"Distress delistings (codes 400-490): {dlst['default_crsp'].sum():,}")

    # ── Merge delisting returns into MSF ──
    # When a stock delists, replace the last month's return with the delisting return
    msf = msf.merge(
        dlst[['permno', 'dlstdt', 'dlret', 'dlstcd']],
        on='permno',
        how='left'
    )

    # If month matches the delisting month, use dlret
    msf['ret_adj'] = msf['ret']
    delist_month_mask = (
        msf['dlstdt'].notna() &
        (msf['date'].dt.year == msf['dlstdt'].dt.year) &
        (msf['date'].dt.month == msf['dlstdt'].dt.month)
    )
    msf.loc[delist_month_mask, 'ret_adj'] = msf.loc[delist_month_mask, 'dlret']

    # ── Aggregate to annual frequency ──
    # For each permno-year, compute year-end market cap and annual equity volatility
    annual_crsp = msf.groupby(['permno', 'year']).agg(
        mktcap_yearend=('mktcap', 'last'),
        ret_std=('ret_adj', 'std'),
        n_months=('ret_adj', 'count'),
    ).reset_index()

    # Annualize volatility (sqrt(12) scaling)
    annual_crsp['equity_vol'] = annual_crsp['ret_std'] * np.sqrt(12)

    # Require at least 6 months of returns
    annual_crsp = annual_crsp[annual_crsp['n_months'] >= 6]

    # ── Add default flag from delisting ──
    delist_annual = dlst[['permno', 'dlst_year', 'default_crsp', 'dlstcd']].copy()
    delist_annual.columns = ['permno', 'year', 'default_crsp', 'dlstcd']
    annual_crsp = annual_crsp.merge(delist_annual, on=['permno', 'year'], how='left')
    annual_crsp['default_crsp'] = annual_crsp['default_crsp'].fillna(0).astype(int)

    print(f"Annual CRSP records: {len(annual_crsp):,}")
    print(f"Unique permnos: {annual_crsp['permno'].nunique():,}")

    return annual_crsp


# ============================================================
# STEP 3: MERGE COMPUSTAT AND CRSP VIA CCM LINKING TABLE
# ============================================================

def merge_compustat_crsp(comp, crsp):
    """
    Merge Compustat and CRSP using the CCM linking table.
    """
    print("\n" + "="*60)
    print("STEP 3: MERGE COMPUSTAT-CRSP")
    print("="*60)

    # Load CCM linking table
    ccm = pd.read_csv(os.path.join(RAW_DIR, "ccm_linktable.csv"))
    ccm.columns = ccm.columns.str.lower()

    # Some WRDS exports name the linked permno as 'lpermno', others as 'permno'
    if 'lpermno' not in ccm.columns and 'permno' in ccm.columns:
        ccm = ccm.rename(columns={'permno': 'lpermno'})

    print(f"Loaded {len(ccm):,} linking table records")

    # Filter to valid linktypes (LC = link confirmed, LU = link unconfirmed but usable)
    ccm = ccm[ccm['linktype'].isin(['LC', 'LU'])]

    # Filter to primary security indicators (P or C)
    ccm = ccm[ccm['linkprim'].isin(['P', 'C'])]

    # Parse link dates
    ccm['linkdt'] = pd.to_datetime(ccm['linkdt'])
    # linkenddt is 'E' for ongoing links; replace with future date
    ccm['linkenddt'] = ccm['linkenddt'].replace('E', '2099-12-31')
    ccm['linkenddt'] = pd.to_datetime(ccm['linkenddt'], errors='coerce')
    ccm['linkenddt'] = ccm['linkenddt'].fillna(pd.Timestamp('2099-12-31'))

    print(f"After filtering linktypes/linkprim: {len(ccm):,}")

    # ── Merge Compustat -> CCM ──
    # For each Compustat firm-year, find the matching permno
    comp = comp.copy()
    comp['datadate'] = pd.to_datetime(comp['datadate']) if 'datadate' in comp.columns else pd.NaT

    # If datadate is missing, construct from fyear and fyr
    if comp['datadate'].isna().all():
        # Default to December 31 of fyear
        comp['datadate'] = pd.to_datetime(comp['fyear'].astype(str) + '-12-31')

    merged = comp.merge(
        ccm[['gvkey', 'lpermno', 'linkdt', 'linkenddt']],
        on='gvkey',
        how='left'
    )

    # Keep only rows where datadate falls within the link validity window
    merged = merged[
        (merged['datadate'] >= merged['linkdt']) &
        (merged['datadate'] <= merged['linkenddt'])
    ]

    merged = merged.rename(columns={'lpermno': 'permno'})
    print(f"Compustat-CCM merged: {len(merged):,}")

    # ── Merge with CRSP annual data ──
    # Match on permno and year (use fyear as the calendar year for the CRSP match)
    merged['year'] = merged['datadate'].dt.year
    merged = merged.merge(
        crsp[['permno', 'year', 'mktcap_yearend', 'equity_vol', 'default_crsp', 'dlstcd']],
        on=['permno', 'year'],
        how='left'
    )

    # ── Construct market leverage ──
    merged['market_leverage'] = merged['total_debt'] / (
        merged['total_debt'] + merged['mktcap_yearend']
    )

    # ── Tobin's Q ──
    # Q = (market value of equity + total debt) / total assets
    merged['tobins_q'] = (merged['mktcap_yearend'] + merged['total_debt']) / merged['at']

    print(f"Final Compustat-CRSP merged: {len(merged):,}")
    print(f"With market cap (CRSP match): {merged['mktcap_yearend'].notna().sum():,}")

    return merged


# ============================================================
# STEP 4: ADD LOPUCKI BANKRUPTCY DATA
# ============================================================

def add_lopucki(panel):
    """
    Merge in LoPucki Bankruptcy Research Database.
    LoPucki captures large public company bankruptcies that may not appear
    cleanly in CRSP delistings (e.g., firms that were acquired in distress).
    """
    print("\n" + "="*60)
    print("STEP 4: LOPUCKI BANKRUPTCY DATA")
    print("="*60)

    try:
        lopucki = pd.read_excel(os.path.join(RAW_DIR, "lopucki_brd.xlsx"))
        print(f"Loaded {len(lopucki):,} LoPucki records")
    except FileNotFoundError:
        print("LoPucki file not found. Skipping.")
        panel['default_lopucki'] = 0
        return panel

    # Key columns in LoPucki BRD
    # GvkeyBefore: Compustat gvkey for the firm before bankruptcy
    # DateFiled:   Date of bankruptcy filing
    # YearFiled:   Year of bankruptcy filing
    # Chapter:     Chapter 7 (liquidation) or Chapter 11 (reorganization)

    if 'GvkeyBefore' not in lopucki.columns:
        print("WARNING: GvkeyBefore column not found in LoPucki data")
        panel['default_lopucki'] = 0
        return panel

    # Standardize column names
    lopucki = lopucki.rename(columns={
        'GvkeyBefore': 'gvkey',
        'YearFiled': 'bankruptcy_year',
    })

    # Drop missing gvkeys
    lopucki = lopucki.dropna(subset=['gvkey', 'bankruptcy_year'])
    lopucki['gvkey'] = pd.to_numeric(lopucki['gvkey'], errors='coerce')
    lopucki['bankruptcy_year'] = pd.to_numeric(lopucki['bankruptcy_year'], errors='coerce').astype('Int64')

    lopucki = lopucki.dropna(subset=['gvkey', 'bankruptcy_year'])

    # Build a flag table: (gvkey, year) -> default flag
    lopucki['default_lopucki'] = 1
    lopucki_flags = lopucki[['gvkey', 'bankruptcy_year', 'default_lopucki']].drop_duplicates()
    lopucki_flags = lopucki_flags.rename(columns={'bankruptcy_year': 'fyear'})

    # Merge into panel
    panel = panel.merge(lopucki_flags, on=['gvkey', 'fyear'], how='left')
    panel['default_lopucki'] = panel['default_lopucki'].fillna(0).astype(int)

    print(f"LoPucki defaults matched to panel: {panel['default_lopucki'].sum():,}")

    return panel


# ============================================================
# STEP 5: COMBINE DEFAULT INDICATORS AND COMPUTE NEXT-YEAR DEFAULT
# ============================================================

def construct_default_indicators(panel):
    """
    Combine CRSP delisting defaults and LoPucki bankruptcies into a single
    default indicator. Construct the next-year default indicator that
    the text signal predicts.
    """
    print("\n" + "="*60)
    print("STEP 5: DEFAULT INDICATORS")
    print("="*60)

    # Combined default = CRSP delisting OR LoPucki bankruptcy
    panel['default'] = (
        (panel['default_crsp'].fillna(0) == 1) |
        (panel['default_lopucki'].fillna(0) == 1)
    ).astype(int)

    # Sort and compute next-year default
    panel = panel.sort_values(['gvkey', 'fyear'])
    panel['default_next_year'] = panel.groupby('gvkey')['default'].shift(-1)

    # Also create a "default within next 2 years" version
    # because bankruptcy filing often lags warning signs
    panel['default_2yr'] = (
        panel.groupby('gvkey')['default'].shift(-1).fillna(0) +
        panel.groupby('gvkey')['default'].shift(-2).fillna(0)
    ) > 0
    panel['default_2yr'] = panel['default_2yr'].astype(int)

    n_default = panel['default'].sum()
    n_default_next = panel['default_next_year'].sum()
    print(f"Current-year defaults:   {n_default:,} ({n_default/len(panel)*100:.2f}%)")
    print(f"Next-year defaults:      {n_default_next:,} ({n_default_next/len(panel)*100:.2f}%)")
    print(f"Default within 2 years:  {panel['default_2yr'].sum():,}")

    return panel


# ============================================================
# STEP 6: ADD FRED SPREAD DATA
# ============================================================

def load_one_fred_file(filename):
    """
    Load a single FRED CSV. Handles various date column name conventions
    (DATE, date, observation_date) and '.' missing-value codes.
    """
    fred = pd.read_csv(os.path.join(RAW_DIR, filename))

    # FRED uses different date column names depending on download source:
    #   - 'DATE' (older API exports)
    #   - 'observation_date' (current FRED website downloads)
    #   - 'date' (some other tools)
    date_col = None
    for candidate in ['observation_date', 'DATE', 'date', 'Date']:
        if candidate in fred.columns:
            date_col = candidate
            break

    if date_col is None:
        # Last resort: assume the first column is the date
        date_col = fred.columns[0]
        print(f"  WARNING: No standard date column in {filename}, using '{date_col}'")

    fred['date'] = pd.to_datetime(fred[date_col])

    # Identify data columns (everything except the date columns)
    data_cols = [c for c in fred.columns if c not in ['date', date_col]]

    # Convert to numeric (FRED uses '.' for missing observations)
    for col in data_cols:
        fred[col] = pd.to_numeric(fred[col], errors='coerce')

    return fred[['date'] + data_cols]


def add_fred_spreads(panel):
    """
    Merge in FRED credit spread indices and Treasury yields at annual frequency.
    These come in two separate files because FRED groups corporate spreads
    and Treasury yields under different release categories.
    """
    print("\n" + "="*60)
    print("STEP 6: FRED CREDIT SPREADS & TREASURY YIELDS")
    print("="*60)

    fred_frames = []

    # ── Load corporate bond spreads ──
    try:
        spreads = load_one_fred_file("fred_corporate_spreads.csv")
        print(f"Loaded corporate spreads: {len(spreads):,} daily observations, "
              f"columns: {[c for c in spreads.columns if c != 'date']}")
        fred_frames.append(spreads)
    except FileNotFoundError:
        print("WARNING: fred_corporate_spreads.csv not found")

    # ── Load Treasury yields ──
    try:
        treasuries = load_one_fred_file("fred_treasuries.csv")
        print(f"Loaded Treasury yields: {len(treasuries):,} daily observations, "
              f"columns: {[c for c in treasuries.columns if c != 'date']}")
        fred_frames.append(treasuries)
    except FileNotFoundError:
        print("WARNING: fred_treasuries.csv not found")

    if not fred_frames:
        print("No FRED files loaded. Skipping.")
        return panel

    # ── Merge the two FRED files on date ──
    # Outer join because the series may have different date coverage
    fred = fred_frames[0]
    for f in fred_frames[1:]:
        fred = fred.merge(f, on='date', how='outer')

    fred['year'] = fred['date'].dt.year

    # Identify all data columns
    data_cols = [c for c in fred.columns if c not in ['date', 'year']]

    # Aggregate to annual averages
    fred_annual = fred.groupby('year')[data_cols].mean().reset_index()
    fred_annual = fred_annual.rename(columns={'year': 'fyear'})

    # Rename FRED series codes to descriptive names
    rename_map = {
        'BAMLC0A0CM':   'spread_ig',     # Investment grade
        'BAMLC0A4CBBB': 'spread_bbb',    # BBB
        'BAMLH0A0HYM2': 'spread_hy',     # High yield
        'BAMLH0A1HYBB': 'spread_bb',     # BB
        'BAMLH0A2HYB':  'spread_b',      # B
        'BAMLH0A3HYC':  'spread_ccc',    # CCC
        'DGS10':        'tsy_10y',       # 10-year Treasury
        'DGS1':         'tsy_1y',        # 1-year Treasury
        'DGS3MO':       'tsy_3m',        # 3-month Treasury (in case included)
        'DGS5':         'tsy_5y',        # 5-year Treasury (in case included)
    }
    fred_annual = fred_annual.rename(columns=rename_map)

    # Merge with panel
    panel = panel.merge(fred_annual, on='fyear', how='left')

    added_cols = [v for k, v in rename_map.items() if v in panel.columns]
    print(f"FRED columns added to panel: {added_cols}")

    return panel


# ============================================================
# STEP 7: WINSORIZE AND FINALIZE
# ============================================================

def finalize_panel(panel):
    """
    Final cleanup: winsorize key variables, restrict to estimation period,
    keep only relevant columns.
    """
    print("\n" + "="*60)
    print("STEP 7: FINALIZE")
    print("="*60)

    # Restrict to estimation period
    panel = panel[(panel['fyear'] >= START_YEAR) & (panel['fyear'] <= END_YEAR)]

    # Winsorize key ratios at the 1st and 99th percentiles
    winsorize_vars = [
        'profitability', 'investment_rate', 'book_leverage',
        'market_leverage', 'tobins_q', 'equity_issuance',
        'cash_ratio', 'equity_vol',
    ]
    for var in winsorize_vars:
        if var in panel.columns:
            lo, hi = panel[var].quantile([0.01, 0.99])
            panel[var] = panel[var].clip(lower=lo, upper=hi)

    print(f"Final panel: {len(panel):,} firm-years")
    print(f"Unique firms: {panel['gvkey'].nunique():,}")
    print(f"Year coverage: {panel['fyear'].min()}-{panel['fyear'].max()}")
    print(f"Mean default rate: {panel['default'].mean()*100:.2f}%")

    return panel


# ============================================================
# STEP 8: BUILD SAMPLE_FIRMS.CSV FOR NLP PIPELINE
# ============================================================

def build_sample_firms(panel):
    """
    Extract the minimal columns needed for the NLP pipeline.
    """
    print("\n" + "="*60)
    print("STEP 8: SAMPLE FIRMS FOR NLP PIPELINE")
    print("="*60)

    sample = panel[['gvkey', 'cik', 'fyear', 'conm', 'default_next_year']].copy()

    # Drop firms without CIK (can't query EDGAR without it)
    sample = sample.dropna(subset=['cik'])
    sample['cik'] = sample['cik'].astype(int)

    # Drop the last year for each firm (no next-year default available)
    sample = sample.dropna(subset=['default_next_year'])
    sample['default_next_year'] = sample['default_next_year'].astype(int)

    print(f"Sample firms file: {len(sample):,} firm-years")
    print(f"Unique firms: {sample['gvkey'].nunique():,}")
    print(f"With default in next year: {sample['default_next_year'].sum():,}")

    return sample


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Step 1: Compustat
    comp = load_compustat()

    # Step 2: CRSP
    crsp = load_crsp()

    # Step 3: Merge Compustat-CRSP
    panel = merge_compustat_crsp(comp, crsp)

    # Step 4: LoPucki bankruptcies
    panel = add_lopucki(panel)

    # Step 5: Default indicators
    panel = construct_default_indicators(panel)

    # Step 6: FRED spreads
    panel = add_fred_spreads(panel)

    # Step 7: Finalize
    panel = finalize_panel(panel)

    # Step 8: Sample firms for NLP
    sample_firms = build_sample_firms(panel)

    # Save outputs
    full_path = os.path.join(OUT_DIR, "full_panel.csv")
    sample_path = os.path.join(OUT_DIR, "sample_firms.csv")

    panel.to_csv(full_path, index=False)
    sample_firms.to_csv(sample_path, index=False)

    print("\n" + "="*60)
    print("DONE")
    print("="*60)
    print(f"Full panel saved to:    {full_path}")
    print(f"  ({len(panel):,} firm-years, {len(panel.columns)} columns)")
    print(f"Sample firms saved to:  {sample_path}")
    print(f"  ({len(sample_firms):,} firm-years for NLP pipeline)")
    print("\nNext steps:")
    print("  1. Copy data/sample_firms.csv into nlp_pipeline/data/")
    print("  2. Run the NLP pipeline scripts (01 through 05)")
    print("  3. Merge output/text_signal.csv back into full_panel.csv on (gvkey, fyear)")
    print("  4. Use the merged panel for SMM estimation")


if __name__ == "__main__":
    main()
