"""
table7_merton_incremental.py
============================
Table 7: Incremental predictive power of FinBERT over Merton-naive
distance-to-default (Bharath-Shumway 2008).

Merton DD is a classic market-based distress predictor. This table
tests whether FinBERT adds information beyond it.

Specifications (DV: default within 2 years):
  (1) merton_distress alone (= -DD; higher = more distress)
  (2) s_finbert alone
  (3) both together
  (4) both + financial controls
  (5) both + financial controls + industry FE + year FE

Inputs:
    data/panel_smm_ready.csv

Outputs:
    output/tables/table7_merton_incremental.tex
    output/tables/table7_merton_incremental.txt

Usage:
    python scripts/table7_merton_incremental.py
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.regression_utils import prepare_sample, run_lpm
from src.signal_utils import compute_merton_dd
from src.latex_utils import format_reg_table_latex, write_latex_table


CONTROLS = ['log_at', 'book_leverage_w', 'tobins_q_fix_w',
            'profitability_w', 'cash_ratio_w']


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    panel_path = os.path.join(root, 'data', 'panel_smm_ready.csv')

    print(f"Loading {panel_path}...")
    panel = pd.read_csv(panel_path)
    panel = prepare_sample(panel)
    panel = compute_merton_dd(panel)

    # Standardize merton_distress within year (matches s_finbert convention)
    from src.signal_utils import standardize_within_year
    panel['merton_distress_std'] = standardize_within_year(
        panel['merton_distress'], panel['fyear']
    ).values

    valid_merton = panel['merton_distress_std'].notna().sum()
    print(f"  Rows: {len(panel):,}; merton_distress valid: {valid_merton:,}")
    print(f"  Defaults (2-yr): {int(panel['def_within_2'].sum())}")

    # Specifications
    specs = []

    # (1) Merton only
    m1, _ = run_lpm(panel, 'def_within_2', ['merton_distress_std'])
    specs.append(m1)

    # (2) FinBERT only
    m2, _ = run_lpm(panel, 'def_within_2', ['s_finbert'])
    specs.append(m2)

    # (3) both
    m3, _ = run_lpm(panel, 'def_within_2',
                    ['merton_distress_std', 's_finbert'])
    specs.append(m3)

    # (4) both + controls
    m4, _ = run_lpm(panel, 'def_within_2',
                    ['merton_distress_std', 's_finbert'] + CONTROLS)
    specs.append(m4)

    # (5) both + controls + FE
    m5, _ = run_lpm(panel, 'def_within_2',
                    ['merton_distress_std', 's_finbert'] + CONTROLS,
                    fe_cols=['sic1', 'fyear_cat'])
    specs.append(m5)

    # Console report
    print("\n" + "=" * 70)
    print(" TABLE 7: Incremental over Merton DD (DV = def_within_2)")
    print("=" * 70)
    focal = ['merton_distress_std', 's_finbert']
    print(f" {'Variable':<25s} "
          f"{'(1)':>10s} {'(2)':>10s} {'(3)':>10s} {'(4)':>10s} {'(5)':>10s}")
    print(" " + "-" * 81)
    for var in focal:
        row = f" {var:<25s}"
        for m in specs:
            if var in m.params.index:
                c = m.params[var]
                p = m.pvalues[var]
                stars = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))
                row += f" {c:+.4f}{stars:<3s}"
            else:
                row += f" {'':>10s}"
        print(row)
    print(" " + "-" * 81)
    print(f" {'N':<25s}", end='')
    for m in specs:
        print(f" {int(m.nobs):>10,d}", end='')
    print()
    print(f" {'R²':<25s}", end='')
    for m in specs:
        print(f" {m.rsquared:>10.4f}", end='')
    print()

    # LaTeX output
    latex = format_reg_table_latex(
        specs,
        column_names=['(1)', '(2)', '(3)', '(4)', '(5)'],
        focal_vars=['merton_distress_std', 's_finbert'],
        control_vars=CONTROLS,
        focal_display_names={
            'merton_distress_std': r'Merton distress ($-DD$)',
            's_finbert': r'$s^{\mathrm{FB}}$',
        },
        control_display_names={
            'log_at': 'Log assets',
            'book_leverage_w': 'Book leverage',
            'tobins_q_fix_w': "Tobin's Q",
            'profitability_w': 'Profitability',
            'cash_ratio_w': 'Cash/assets',
        },
        label='tab:t7_merton',
        caption=r"Incremental predictive power of FinBERT over Merton-naive "
                r"distance-to-default (Bharath and Shumway, 2008). The "
                r"dependent variable is default within two years (binary). "
                r"Merton distress is defined as $-DD$, where "
                r"$DD = [\ln(V/F) + (r-0.5\sigma_V^2)T]/(\sigma_V \sqrt{T})$ "
                r"with $V$ equity plus book debt, $F$ default point "
                r"$=dlc + 0.5 \cdot dltt$, $T=1$ year, $r$ the 1-year "
                r"Treasury yield, and $\sigma_V$ the Bharath-Shumway naive "
                r"approximation. Both Merton distress and $s^{\mathrm{FB}}$ "
                r"are standardized to mean zero and unit variance within each "
                r"year. Coefficients are marginal effects on default "
                r"probability per standard deviation increase. Industry FE "
                r"use 1-digit SIC.",
    )

    out_dir = os.path.join(root, 'output', 'tables')
    tex_path = os.path.join(out_dir, 'table7_merton_incremental.tex')
    write_latex_table(tex_path, latex)

    # Plain-text
    txt_path = os.path.join(out_dir, 'table7_merton_incremental.txt')
    lines = []
    lines.append("=" * 88)
    lines.append("TABLE 7: Incremental over Merton DD (DV = def_within_2)")
    lines.append("=" * 88)
    lines.append("")
    lines.append(f"{'':25s} {'(1)':>12s} {'(2)':>12s} {'(3)':>12s} {'(4)':>12s} {'(5)':>12s}")
    lines.append("-" * 90)
    for var in focal:
        row = f"{var:<25s}"
        se_row = " " * 25
        for m in specs:
            if var in m.params.index:
                c = m.params[var]
                p = m.pvalues[var]
                stars = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))
                row += f" {c:+.4f}{stars:<4s}"
                se_row += f" {'('+f'{m.bse[var]:.4f}'+')':>12s}"
            else:
                row += f" {'':>12s}"
                se_row += f" {'':>12s}"
        lines.append(row)
        lines.append(se_row)
    lines.append("-" * 90)
    for label, attr in [('N', 'nobs'), ('R²', 'rsquared')]:
        row = f"{label:<25s}"
        for m in specs:
            val = getattr(m, attr)
            if label == 'N':
                row += f" {int(val):>12,d}"
            else:
                row += f" {val:>12.4f}"
        lines.append(row)
    lines.append("")
    lines.append("Merton naive distance-to-default following Bharath-Shumway (2008).")
    lines.append("Cluster-robust SE by firm. *** p<0.01, ** p<0.05, * p<0.1")

    with open(txt_path, 'w') as f:
        f.write("\n".join(lines))

    print(f"\n✓ Saved: {tex_path}")
    print(f"✓ Saved: {txt_path}")


if __name__ == "__main__":
    main()
