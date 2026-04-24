"""
table9_supervised_lm.py
=======================
Table 9: Supervised LM signal comparison with FinBERT.

The baseline LM signal is unsupervised (first PC of dictionary category
fractions). FinBERT is supervised (logistic projection onto default
direction). This asymmetry confounds the LM-vs-FinBERT comparison.

This table constructs a SUPERVISED LM signal using the same procedure
as FinBERT — logistic regression of default_next on LM category
fractions (L2 penalty, balanced class weights, split train/test).

If FinBERT's advantage persists against supervised-LM, the neural
language representation is genuinely carrying information beyond
dictionary categories. If supervised-LM closes the gap, the earlier
FinBERT-vs-LM result was really "supervision matters" not
"neural vs dictionary".

Specifications (DV: default within 2 years):
  (1) s_lm (unsupervised PCA baseline)
  (2) s_lm_supervised (new supervised variant)
  (3) s_finbert
  (4) s_lm_supervised + s_finbert + controls + FE

Inputs:
    data/panel_smm_ready.csv
    data/lm_scores.csv

Outputs:
    output/tables/table9_supervised_lm.tex
    output/tables/table9_supervised_lm.txt

Usage:
    python scripts/table9_supervised_lm.py
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.regression_utils import prepare_sample, run_lpm
from src.signal_utils import build_supervised_lm
from src.latex_utils import format_reg_table_latex, write_latex_table


CONTROLS = ['log_at', 'book_leverage_w', 'tobins_q_fix_w',
            'profitability_w', 'cash_ratio_w']


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    panel_path = os.path.join(root, 'data', 'panel_smm_ready.csv')
    lm_path = os.path.join(root, 'data', 'lm_scores.csv')

    print(f"Loading {panel_path}...")
    panel = pd.read_csv(panel_path)
    panel = prepare_sample(panel)

    print(f"Loading {lm_path}...")
    lm = pd.read_csv(lm_path)
    print(f"  LM scores: {len(lm):,} rows")

    # Build supervised LM signal
    print("\nBuilding supervised LM signal...")
    default_df = panel[['gvkey', 'fyear', 'def_t1']].rename(
        columns={'def_t1': 'default_next'}
    )
    sup_lm, info = build_supervised_lm(
        lm, default_df, lm_prefix='combined',
        categories=['negative', 'uncertainty', 'weak_modal'],
    )
    print(f"  Training period: fyear <= {info['train_cutoff_year']}")
    print(f"  Training N: {info['n_train']:,} "
          f"(defaults: {info['n_train_defaults']})")
    print(f"  Beta (negative, uncertainty, weak_modal): "
          f"{info['beta'].round(2).tolist()}")

    # Merge s_lm_supervised into panel
    sup_lm['gvkey'] = sup_lm['gvkey'].astype(str).str.strip()
    sup_lm['fyear'] = pd.to_numeric(sup_lm['fyear'], errors='coerce')
    panel['gvkey'] = panel['gvkey'].astype(str).str.strip()
    panel['fyear'] = pd.to_numeric(panel['fyear'], errors='coerce')

    panel = panel.merge(sup_lm[['gvkey', 'fyear', 's_lm_supervised']],
                        on=['gvkey', 'fyear'], how='left')
    n_with_suplm = panel['s_lm_supervised'].notna().sum()
    print(f"  s_lm_supervised merged: {n_with_suplm:,} non-null in panel")

    # Run specifications
    specs = []

    # (1) Unsupervised LM baseline
    m1, _ = run_lpm(panel, 'def_within_2', ['s_lm'])
    specs.append(m1)

    # (2) Supervised LM
    m2, _ = run_lpm(panel, 'def_within_2', ['s_lm_supervised'])
    specs.append(m2)

    # (3) FinBERT
    m3, _ = run_lpm(panel, 'def_within_2', ['s_finbert'])
    specs.append(m3)

    # (4) All three + controls + FE
    m4, _ = run_lpm(panel, 'def_within_2',
                    ['s_lm_supervised', 's_finbert'] + CONTROLS,
                    fe_cols=['sic1', 'fyear_cat'])
    specs.append(m4)

    # Console report
    print("\n" + "=" * 70)
    print(" TABLE 9: Supervised LM vs FinBERT (DV = def_within_2)")
    print("=" * 70)
    focal = ['s_lm', 's_lm_supervised', 's_finbert']
    print(f" {'Variable':<25s} "
          f"{'(1) LM_PC':>14s} {'(2) LM_sup':>14s} "
          f"{'(3) FinBERT':>14s} {'(4) combined':>14s}")
    print(" " + "-" * 83)
    for var in focal:
        row = f" {var:<25s}"
        for m in specs:
            if var in m.params.index:
                c = m.params[var]
                p = m.pvalues[var]
                stars = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))
                row += f" {c:+.4f}{stars:<5s}"
            else:
                row += f" {'':>14s}"
        print(row)
    print(" " + "-" * 83)
    print(f" {'N':<25s}", end='')
    for m in specs:
        print(f" {int(m.nobs):>14,d}", end='')
    print()
    print(f" {'R²':<25s}", end='')
    for m in specs:
        print(f" {m.rsquared:>14.4f}", end='')
    print()

    # LaTeX
    latex = format_reg_table_latex(
        specs,
        column_names=['(1)', '(2)', '(3)', '(4)'],
        focal_vars=['s_lm', 's_lm_supervised', 's_finbert'],
        control_vars=CONTROLS,
        focal_display_names={
            's_lm': r'$s^{\mathrm{LM,unsup}}$ (PCA)',
            's_lm_supervised': r'$s^{\mathrm{LM,sup}}$',
            's_finbert': r'$s^{\mathrm{FB}}$',
        },
        control_display_names={
            'log_at': 'Log assets',
            'book_leverage_w': 'Book leverage',
            'tobins_q_fix_w': "Tobin's Q",
            'profitability_w': 'Profitability',
            'cash_ratio_w': 'Cash/assets',
        },
        label='tab:t9_supervised_lm',
        caption=r"Fair comparison of LM dictionary and FinBERT signals. "
                r"The supervised LM signal ($s^{\mathrm{LM,sup}}$) is "
                r"constructed by projecting the three LM category "
                r"fractions (negative, uncertainty, weak modal) onto the "
                r"default direction identified by logistic regression on "
                r"firms in the first half of the sample "
                r"($\text{fyear} \le \text{median}$), using the same "
                r"procedure as $s^{\mathrm{FB}}$. The unsupervised variant "
                r"$s^{\mathrm{LM,unsup}}$ is the first principal component "
                r"of the same three categories, standardized within year. "
                r"All signals are standardized to mean zero and unit "
                r"variance within each year. The dependent variable is "
                r"default within two years (binary). Industry FE use "
                r"1-digit SIC.",
    )

    out_dir = os.path.join(root, 'output', 'tables')
    tex_path = os.path.join(out_dir, 'table9_supervised_lm.tex')
    write_latex_table(tex_path, latex)

    # Plain text
    txt_lines = []
    txt_lines.append("=" * 86)
    txt_lines.append("TABLE 9: Supervised LM vs FinBERT (DV = def_within_2)")
    txt_lines.append("=" * 86)
    txt_lines.append("")
    txt_lines.append(f"{'':26s} {'(1) LM_PC':>14s} {'(2) LM_sup':>14s} "
                      f"{'(3) FinBERT':>14s} {'(4) combined':>14s}")
    txt_lines.append("-" * 86)
    for var in focal:
        row = f"{var:<26s}"
        se_row = " " * 26
        for m in specs:
            if var in m.params.index:
                c = m.params[var]
                p = m.pvalues[var]
                stars = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))
                row += f" {c:+.4f}{stars:<6s}"
                se_row += f" {'('+f'{m.bse[var]:.4f}'+')':>14s}"
            else:
                row += f" {'':>14s}"
                se_row += f" {'':>14s}"
        txt_lines.append(row)
        txt_lines.append(se_row)
    txt_lines.append("-" * 86)
    for label, attr in [('N', 'nobs'), ('R²', 'rsquared')]:
        row = f"{label:<26s}"
        for m in specs:
            val = getattr(m, attr)
            if label == 'N':
                row += f" {int(val):>14,d}"
            else:
                row += f" {val:>14.4f}"
        txt_lines.append(row)
    txt_lines.append("")
    txt_lines.append(f"Supervised LM trained on fyear <= {info['train_cutoff_year']} "
                     f"({info['n_train']:,} obs, {info['n_train_defaults']} defaults)")
    txt_lines.append(f"Logit beta (neg, unc, wk_modal): "
                     f"{info['beta'].round(2).tolist()}")

    txt_path = os.path.join(out_dir, 'table9_supervised_lm.txt')
    with open(txt_path, 'w') as f:
        f.write("\n".join(txt_lines))

    print(f"\n✓ Saved: {tex_path}")
    print(f"✓ Saved: {txt_path}")


if __name__ == "__main__":
    main()
