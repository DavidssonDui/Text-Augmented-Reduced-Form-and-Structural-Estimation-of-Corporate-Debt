"""
table6_horse_race.py
====================
Table 6: Horse race between FinBERT and LM signals.

Tests whether FinBERT subsumes LM or whether they carry independent
information by including both signals in the same regression.

Specifications (DV: default within 2 years):
  (1) s_lm only
  (2) s_finbert only
  (3) both signals
  (4) both signals + financial controls
  (5) both signals + financial controls + industry FE + year FE

If s_finbert's coefficient falls when s_lm is added, the two signals
overlap. If it's unchanged, FinBERT captures something LM doesn't.

Inputs:
    data/panel_smm_ready.csv

Outputs:
    output/tables/table6_horse_race.tex
    output/tables/table6_horse_race.txt

Usage:
    python scripts/table6_horse_race.py

Deterministic: set random_state=None; all regressions are deterministic
given the input data.
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.regression_utils import prepare_sample, run_lpm
from src.latex_utils import format_reg_table_latex, write_latex_table


CONTROLS = ['log_at', 'book_leverage_w', 'tobins_q_fix_w',
            'profitability_w', 'cash_ratio_w']


def main():
    # Paths
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    panel_path = os.path.join(root, 'data', 'panel_smm_ready.csv')

    print(f"Loading {panel_path}...")
    panel = pd.read_csv(panel_path)
    panel = prepare_sample(panel)
    print(f"  Rows: {len(panel):,}, defaults (2-yr): {int(panel['def_within_2'].sum())}")

    # Run the five specifications
    specs = []

    # (1) s_lm only
    m1, _ = run_lpm(panel, 'def_within_2', ['s_lm'])
    specs.append(m1)

    # (2) s_finbert only
    m2, _ = run_lpm(panel, 'def_within_2', ['s_finbert'])
    specs.append(m2)

    # (3) both
    m3, _ = run_lpm(panel, 'def_within_2', ['s_lm', 's_finbert'])
    specs.append(m3)

    # (4) both + controls
    m4, _ = run_lpm(panel, 'def_within_2', ['s_lm', 's_finbert'] + CONTROLS)
    specs.append(m4)

    # (5) both + controls + FE
    m5, _ = run_lpm(panel, 'def_within_2', ['s_lm', 's_finbert'] + CONTROLS,
                    fe_cols=['sic1', 'fyear_cat'])
    specs.append(m5)

    # Report to console
    print("\n" + "=" * 60)
    print(" TABLE 6: Horse race FinBERT vs LM (DV = def_within_2)")
    print("=" * 60)
    print()
    print(f" {'Variable':<22s} {'(1)':>10s} {'(2)':>10s} {'(3)':>10s} "
          f"{'(4)':>10s} {'(5)':>10s}")
    print(" " + "-" * 78)
    for var in ['s_lm', 's_finbert']:
        row = f" {var:<22s}"
        for m in specs:
            if var in m.params.index:
                c = m.params[var]
                p = m.pvalues[var]
                stars = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))
                row += f" {c:+.4f}{stars:<3s}"
            else:
                row += f" {'':>10s}"
        print(row)
    print(" " + "-" * 78)
    for m in specs:
        pass
    print(f" {'N':<22s}", end='')
    for m in specs:
        print(f" {int(m.nobs):>10,d}", end='')
    print()
    print(f" {'R²':<22s}", end='')
    for m in specs:
        print(f" {m.rsquared:>10.4f}", end='')
    print()

    # LaTeX output
    latex = format_reg_table_latex(
        specs,
        column_names=['(1)', '(2)', '(3)', '(4)', '(5)'],
        focal_vars=['s_lm', 's_finbert'],
        control_vars=CONTROLS,
        focal_display_names={
            's_lm': r'$s^{\mathrm{LM}}$',
            's_finbert': r'$s^{\mathrm{FB}}$',
        },
        control_display_names={
            'log_at': 'Log assets',
            'book_leverage_w': 'Book leverage',
            'tobins_q_fix_w': "Tobin's Q",
            'profitability_w': 'Profitability',
            'cash_ratio_w': 'Cash/assets',
        },
        label='tab:t6_horse_race',
        caption=r"Horse race between FinBERT and LM text signals. The "
                r"dependent variable is default within two years (binary). "
                r"Both signals are standardized to mean zero and unit "
                r"variance within each year; coefficients are thus "
                r"per-standard-deviation marginal effects on default "
                r"probability. Industry FE use 1-digit SIC.",
    )

    # Write outputs
    out_dir = os.path.join(root, 'output', 'tables')
    tex_path = os.path.join(out_dir, 'table6_horse_race.tex')
    write_latex_table(tex_path, latex)

    # Also a plain-text version for readability
    txt_path = os.path.join(out_dir, 'table6_horse_race.txt')
    lines = []
    lines.append("=" * 80)
    lines.append("TABLE 6: Horse race FinBERT vs LM (DV = def_within_2)")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"{'':22s} {'(1)':>12s} {'(2)':>12s} {'(3)':>12s} {'(4)':>12s} {'(5)':>12s}")
    lines.append("-" * 88)
    for var in ['s_lm', 's_finbert']:
        row = f"{var:<22s}"
        se_row = " " * 22
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
    lines.append("-" * 88)
    for label, attr in [('N', 'nobs'), ('R²', 'rsquared')]:
        row = f"{label:<22s}"
        for m in specs:
            val = getattr(m, attr)
            if label == 'N':
                row += f" {int(val):>12,d}"
            else:
                row += f" {val:>12.4f}"
        lines.append(row)

    with open(txt_path, 'w') as f:
        f.write("\n".join(lines))

    print(f"\n✓ Saved: {tex_path}")
    print(f"✓ Saved: {txt_path}")


if __name__ == "__main__":
    main()
