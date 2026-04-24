"""
table2_baseline_finbert.py
==========================
Table 2: Baseline FinBERT regressions.

DV: default within 2 years.

Specifications:
  (1) FinBERT alone
  (2) + log assets
  (3) + financial controls (leverage, Q, profitability, cash)
  (4) + industry FE
  (5) + industry FE + year FE

This is the headline result establishing that FinBERT predicts default.

Outputs:
    output/tables/table2_baseline_finbert.tex
    output/tables/table2_baseline_finbert.txt
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.regression_utils import prepare_sample, run_lpm
from src.latex_utils import format_reg_table_latex, write_latex_table


CONTROLS = ['book_leverage_w', 'tobins_q_fix_w', 'profitability_w', 'cash_ratio_w']


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    panel_path = os.path.join(root, 'data', 'panel_smm_ready.csv')
    panel = pd.read_csv(panel_path)
    panel = prepare_sample(panel)

    specs = []
    # (1) FinBERT alone
    m1, _ = run_lpm(panel, 'def_within_2', ['s_finbert'])
    specs.append(m1)
    # (2) + log assets
    m2, _ = run_lpm(panel, 'def_within_2', ['s_finbert', 'log_at'])
    specs.append(m2)
    # (3) + financial controls
    m3, _ = run_lpm(panel, 'def_within_2', ['s_finbert', 'log_at'] + CONTROLS)
    specs.append(m3)
    # (4) + industry FE
    m4, _ = run_lpm(panel, 'def_within_2', ['s_finbert', 'log_at'] + CONTROLS,
                     fe_cols=['sic1'])
    specs.append(m4)
    # (5) + industry FE + year FE
    m5, _ = run_lpm(panel, 'def_within_2', ['s_finbert', 'log_at'] + CONTROLS,
                     fe_cols=['sic1', 'fyear_cat'])
    specs.append(m5)

    # Console
    print("=" * 70)
    print(" TABLE 2: Baseline FinBERT regressions (DV = def_within_2)")
    print("=" * 70)
    print(f" {'Variable':<22s} {'(1)':>10s} {'(2)':>10s} {'(3)':>10s} "
          f"{'(4)':>10s} {'(5)':>10s}")
    print(" " + "-" * 75)
    focals = ['s_finbert', 'log_at'] + CONTROLS
    for var in focals:
        row = f" {var:<22s}"
        for m in specs:
            if var in m.params.index:
                c, p = m.params[var], m.pvalues[var]
                stars = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))
                row += f" {c:+.4f}{stars:<3s}"
            else:
                row += f" {'':>10s}"
        print(row)
    print(" " + "-" * 75)
    print(f" {'N':<22s}", end='')
    for m in specs: print(f" {int(m.nobs):>10,d}", end='')
    print()
    print(f" {'R²':<22s}", end='')
    for m in specs: print(f" {m.rsquared:>10.4f}", end='')
    print()

    # LaTeX
    latex = format_reg_table_latex(
        specs,
        column_names=['(1)', '(2)', '(3)', '(4)', '(5)'],
        focal_vars=['s_finbert'],
        control_vars=['log_at'] + CONTROLS,
        focal_display_names={'s_finbert': r'$s^{\mathrm{FB}}$'},
        control_display_names={
            'log_at': 'Log assets',
            'book_leverage_w': 'Book leverage',
            'tobins_q_fix_w': "Tobin's Q",
            'profitability_w': 'Profitability',
            'cash_ratio_w': 'Cash/assets',
        },
        label='tab:t2_baseline_finbert',
        caption=r"Baseline FinBERT regressions. The dependent variable is "
                r"default within two years (binary). The FinBERT signal "
                r"$s^{\mathrm{FB}}$ is the projection of [CLS] embeddings "
                r"of 10-K text onto the default direction, standardized to "
                r"mean zero and unit variance within each fiscal year. "
                r"Coefficients give the marginal effect on the probability "
                r"of default within two years per standard deviation "
                r"increase in $s^{\mathrm{FB}}$. Industry FE use 1-digit "
                r"SIC.",
    )

    out_dir = os.path.join(root, 'output', 'tables')
    write_latex_table(os.path.join(out_dir, 'table2_baseline_finbert.tex'), latex)

    # Plain text
    txt_lines = ["=" * 80,
                 "TABLE 2: Baseline FinBERT regressions (DV = def_within_2)",
                 "=" * 80, "",
                 f"{'':22s} {'(1)':>12s} {'(2)':>12s} {'(3)':>12s} {'(4)':>12s} {'(5)':>12s}",
                 "-" * 88]
    for var in focals:
        row = f"{var:<22s}"
        se_row = " " * 22
        for m in specs:
            if var in m.params.index:
                c, p = m.params[var], m.pvalues[var]
                stars = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))
                row += f" {c:+.4f}{stars:<4s}"
                se_row += f" {'('+f'{m.bse[var]:.4f}'+')':>12s}"
            else:
                row += f" {'':>12s}"
                se_row += f" {'':>12s}"
        txt_lines.append(row)
        txt_lines.append(se_row)
    txt_lines.append("-" * 88)
    txt_lines.append(f"{'N':<22s}" + "".join(f" {int(m.nobs):>12,d}" for m in specs))
    txt_lines.append(f"{'R²':<22s}" + "".join(f" {m.rsquared:>12.4f}" for m in specs))
    txt_lines.append("")
    txt_lines.append("Cluster-robust SE by firm. *** p<0.01, ** p<0.05, * p<0.1")

    txt_path = os.path.join(out_dir, 'table2_baseline_finbert.txt')
    with open(txt_path, 'w') as f:
        f.write("\n".join(txt_lines))

    print(f"\n✓ Saved: {os.path.join(out_dir, 'table2_baseline_finbert.tex')}")
    print(f"✓ Saved: {txt_path}")


if __name__ == "__main__":
    main()
