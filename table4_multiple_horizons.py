"""
table4_multiple_horizons.py
===========================
Table 4: FinBERT predictive power across multiple forecast horizons.

DV varies: default_within_1, default_within_2, default_within_3.
For each horizon: spec with FinBERT + financial controls + FE.

Shows the FinBERT result is not specific to a particular horizon.

Outputs:
    output/tables/table4_multiple_horizons.tex
    output/tables/table4_multiple_horizons.txt
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
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    panel_path = os.path.join(root, 'data', 'panel_smm_ready.csv')
    panel = pd.read_csv(panel_path)
    panel = prepare_sample(panel)

    specs = []
    col_names = []

    # For each horizon, run: FinBERT + LM + controls + FE
    for h in [1, 2, 3]:
        y = f'def_within_{h}'
        m, _ = run_lpm(panel, y, ['s_finbert', 's_lm'] + CONTROLS,
                        fe_cols=['sic1', 'fyear_cat'])
        specs.append(m)
        col_names.append(f'{h}-yr')

    print("=" * 80)
    print(" TABLE 4: FinBERT predictive power across forecast horizons")
    print("=" * 80)
    print(f" {'Horizon (DV: default within)':<32s} "
          f"{col_names[0]:>12s} {col_names[1]:>12s} {col_names[2]:>12s}")
    print(" " + "-" * 75)
    for var in ['s_finbert', 's_lm']:
        row = f" {var:<32s}"
        for m in specs:
            c, p = m.params[var], m.pvalues[var]
            stars = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))
            row += f" {c:+.4f}{stars:<4s}"
        print(row)
    print(" " + "-" * 75)
    print(f" {'N':<32s}", end='')
    for m in specs: print(f" {int(m.nobs):>12,d}", end='')
    print()
    print(f" {'R²':<32s}", end='')
    for m in specs: print(f" {m.rsquared:>12.4f}", end='')
    print()
    # Number of defaults at each horizon
    print(f" {'# defaults in test sample':<32s}", end='')
    for h in [1, 2, 3]:
        # Count defaults in the SAME subsample used in the regression
        cols_needed = [f'def_within_{h}', 's_finbert', 's_lm'] + CONTROLS + ['gvkey']
        n_def = int(panel[cols_needed].dropna()[f'def_within_{h}'].sum())
        print(f" {n_def:>12,d}", end='')
    print()

    # LaTeX
    latex = format_reg_table_latex(
        specs,
        column_names=['1-year', '2-year', '3-year'],
        focal_vars=['s_finbert', 's_lm'],
        control_vars=CONTROLS,
        focal_display_names={
            's_finbert': r'$s^{\mathrm{FB}}$',
            's_lm': r'$s^{\mathrm{LM}}$',
        },
        control_display_names={
            'log_at': 'Log assets',
            'book_leverage_w': 'Book leverage',
            'tobins_q_fix_w': "Tobin's Q",
            'profitability_w': 'Profitability',
            'cash_ratio_w': 'Cash/assets',
        },
        label='tab:t4_horizons',
        caption=r"FinBERT predictive power across forecast horizons. The "
                r"dependent variable is default within $h$ years where "
                r"$h \in \{1, 2, 3\}$ across columns. All specifications "
                r"include both $s^{\mathrm{FB}}$ and $s^{\mathrm{LM}}$, "
                r"financial controls, industry FE (1-digit SIC), and year "
                r"FE. Coefficients give the marginal effect on default "
                r"probability per standard deviation increase in the "
                r"signal. Cluster-robust standard errors by firm.",
    )

    out_dir = os.path.join(root, 'output', 'tables')
    write_latex_table(os.path.join(out_dir, 'table4_multiple_horizons.tex'), latex)

    txt_lines = ["=" * 80,
                 "TABLE 4: FinBERT across forecast horizons",
                 "=" * 80, "",
                 f"{'Horizon':<28s} {'1-yr':>12s} {'2-yr':>12s} {'3-yr':>12s}",
                 "-" * 70]
    for var in ['s_finbert', 's_lm']:
        row = f"{var:<28s}"
        se_row = " " * 28
        for m in specs:
            c, p = m.params[var], m.pvalues[var]
            stars = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))
            row += f" {c:+.4f}{stars:<4s}"
            se_row += f" {'('+f'{m.bse[var]:.4f}'+')':>12s}"
        txt_lines.append(row)
        txt_lines.append(se_row)
    txt_lines.append("-" * 70)
    txt_lines.append(f"{'N':<28s}" + "".join(f" {int(m.nobs):>12,d}" for m in specs))
    txt_lines.append(f"{'R²':<28s}" + "".join(f" {m.rsquared:>12.4f}" for m in specs))
    txt_lines.append("")
    txt_lines.append("All specs include log_at, leverage, Q, profit, cash, industry FE, year FE.")
    txt_lines.append("Cluster-robust SE by firm. *** p<0.01, ** p<0.05, * p<0.1")

    txt_path = os.path.join(out_dir, 'table4_multiple_horizons.txt')
    with open(txt_path, 'w') as f:
        f.write("\n".join(txt_lines))

    print(f"\n✓ Saved: {os.path.join(out_dir, 'table4_multiple_horizons.tex')}")
    print(f"✓ Saved: {txt_path}")


if __name__ == "__main__":
    main()
