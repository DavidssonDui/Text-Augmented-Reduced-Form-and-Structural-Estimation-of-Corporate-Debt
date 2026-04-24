"""
table1_sample_summary.py
========================
Table 1: Sample summary statistics.

Panel A: Sample composition (firms, firm-years, defaults, coverage by year)
Panel B: Variable-level distribution stats

Outputs:
    output/tables/table1_sample_summary.tex
    output/tables/table1_sample_summary.txt
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.regression_utils import prepare_sample


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    panel_path = os.path.join(root, 'data', 'panel_smm_ready.csv')
    panel = pd.read_csv(panel_path)
    panel = prepare_sample(panel)

    out_dir = os.path.join(root, 'output', 'tables')
    os.makedirs(out_dir, exist_ok=True)

    # ── Panel A: Sample composition ──
    n_firms = panel['gvkey'].nunique()
    n_obs = len(panel)
    yr_min = int(panel['fyear'].min())
    yr_max = int(panel['fyear'].max())
    n_def_broad = int(panel['default_broad'].sum())
    n_def_1 = int(panel['def_within_1'].sum())
    n_def_2 = int(panel['def_within_2'].sum())
    n_def_3 = int(panel['def_within_3'].sum())
    n_finbert = int(panel['s_finbert'].notna().sum())
    n_lm = int(panel['s_lm'].notna().sum())
    pct_finbert = 100 * n_finbert / n_obs
    pct_lm = 100 * n_lm / n_obs

    # ── Panel B: Variable distributions ──
    # Variables to summarize
    vars_to_summarize = [
        ('at', 'Total assets (\\$M)'),
        ('book_leverage_w', 'Book leverage'),
        ('tobins_q_fix_w', "Tobin's Q"),
        ('profitability_w', 'Profitability'),
        ('cash_ratio_w', 'Cash/assets'),
        ('s_finbert', 'FinBERT signal'),
        ('s_lm', 'LM signal (PC1)'),
    ]

    panel_b_rows = []
    for var, disp in vars_to_summarize:
        if var not in panel.columns:
            continue
        s = panel[var].dropna()
        if len(s) == 0:
            continue
        panel_b_rows.append({
            'variable': disp,
            'n': len(s),
            'mean': s.mean(),
            'sd': s.std(),
            'p25': s.quantile(0.25),
            'p50': s.quantile(0.50),
            'p75': s.quantile(0.75),
        })
    panel_b = pd.DataFrame(panel_b_rows)

    # ── Console output ──
    print("=" * 70)
    print(" TABLE 1: Sample summary statistics")
    print("=" * 70)
    print()
    print(" Panel A: Sample composition")
    print(" " + "-" * 50)
    print(f"  Sample period            : {yr_min}-{yr_max}")
    print(f"  Unique firms             : {n_firms:,}")
    print(f"  Firm-year observations   : {n_obs:,}")
    print(f"  Default events (broad)   : {n_def_broad}")
    print(f"  Defaults within 1 year   : {n_def_1}")
    print(f"  Defaults within 2 years  : {n_def_2}")
    print(f"  Defaults within 3 years  : {n_def_3}")
    print(f"  FinBERT signal coverage  : {n_finbert:,} ({pct_finbert:.1f}%)")
    print(f"  LM signal coverage       : {n_lm:,} ({pct_lm:.1f}%)")
    print()
    print(" Panel B: Variable distributions")
    print(" " + "-" * 80)
    print(f" {'Variable':<22s} {'N':>10s} {'Mean':>10s} {'SD':>10s} "
          f"{'P25':>10s} {'P50':>10s} {'P75':>10s}")
    print(" " + "-" * 80)
    for r in panel_b_rows:
        print(f" {r['variable']:<22s} {r['n']:>10,d} {r['mean']:>10.4f} "
              f"{r['sd']:>10.4f} {r['p25']:>10.4f} {r['p50']:>10.4f} {r['p75']:>10.4f}")
    print()

    # ── LaTeX output ──
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Sample summary statistics.}")
    lines.append(r"\label{tab:t1_sample_summary}")
    lines.append(r"\small")

    # Panel A
    lines.append(r"\begin{tabular}{lr}")
    lines.append(r"\multicolumn{2}{l}{\textit{Panel A: Sample composition}} \\")
    lines.append(r"\toprule")
    lines.append(r"Sample period & " + f"{yr_min}--{yr_max}" + r" \\")
    lines.append(r"Unique firms & " + f"{n_firms:,}" + r" \\")
    lines.append(r"Firm-year observations & " + f"{n_obs:,}" + r" \\")
    lines.append(r"\midrule")
    lines.append(r"Default events (broad sample) & " + f"{n_def_broad}" + r" \\")
    lines.append(r"\quad Defaults within 1 year & " + f"{n_def_1}" + r" \\")
    lines.append(r"\quad Defaults within 2 years & " + f"{n_def_2}" + r" \\")
    lines.append(r"\quad Defaults within 3 years & " + f"{n_def_3}" + r" \\")
    lines.append(r"\midrule")
    lines.append(r"FinBERT signal coverage & " + "{:,} ({:.1f}\\%)".format(n_finbert, pct_finbert) + r" \\")
    lines.append(r"LM signal coverage & " + "{:,} ({:.1f}\\%)".format(n_lm, pct_lm) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    lines.append(r"\vspace{1em}")

    # Panel B
    lines.append(r"\begin{tabular}{lrrrrrr}")
    lines.append(r"\multicolumn{7}{l}{\textit{Panel B: Variable distributions}} \\")
    lines.append(r"\toprule")
    lines.append(r"Variable & N & Mean & SD & P25 & Median & P75 \\")
    lines.append(r"\midrule")
    for r in panel_b_rows:
        # Special formatting for total assets (large numbers)
        if 'assets' in r['variable'].lower():
            fmt_mean = f"{r['mean']:,.0f}"
            fmt_sd = f"{r['sd']:,.0f}"
            fmt_p25 = f"{r['p25']:,.0f}"
            fmt_p50 = f"{r['p50']:,.0f}"
            fmt_p75 = f"{r['p75']:,.0f}"
        else:
            fmt_mean = f"{r['mean']:.3f}"
            fmt_sd = f"{r['sd']:.3f}"
            fmt_p25 = f"{r['p25']:.3f}"
            fmt_p50 = f"{r['p50']:.3f}"
            fmt_p75 = f"{r['p75']:.3f}"
        lines.append(
            f"{r['variable']} & {r['n']:,} & {fmt_mean} & {fmt_sd} & "
            f"{fmt_p25} & {fmt_p50} & {fmt_p75} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    lines.append(r"\\[-1ex]")
    lines.append(r"\begin{flushleft}")
    lines.append(r"\footnotesize")
    lines.append(r"\textit{Notes}: Sample of US public firms (excluding "
                 r"financials and utilities) covering fiscal years "
                 + f"{yr_min}--{yr_max}. " +
                 r"Default events combine LoPucki Bankruptcy Research "
                 r"Database filings (Chapter 7 and Chapter 11) with CRSP "
                 r"delisting codes 400--499 (liquidations) and 552 "
                 r"(insufficient capital). Defaults within $h$ years are "
                 r"forward-looking from the observation date. Both text "
                 r"signals (FinBERT and LM) are standardized to mean "
                 r"zero and unit variance within each fiscal year. "
                 r"Financial variables are winsorized at the 1st and 99th "
                 r"percentiles within year.")
    lines.append(r"\end{flushleft}")
    lines.append(r"\end{table}")

    tex_path = os.path.join(out_dir, 'table1_sample_summary.tex')
    with open(tex_path, 'w') as f:
        f.write("\n".join(lines))

    # Plain text
    txt_lines = ["TABLE 1: Sample summary statistics", "=" * 70, "",
                 "Panel A: Sample composition",
                 f"  Period: {yr_min}-{yr_max}, Firms: {n_firms:,}, "
                 f"Obs: {n_obs:,}",
                 f"  Defaults: {n_def_broad} broad, {n_def_1}/{n_def_2}/{n_def_3} "
                 f"within 1/2/3 years",
                 f"  Coverage: FinBERT {n_finbert:,} ({pct_finbert:.1f}%), "
                 f"LM {n_lm:,} ({pct_lm:.1f}%)", "",
                 "Panel B: Variable distributions",
                 f"{'Variable':<22s} {'N':>10s} {'Mean':>10s} {'SD':>10s} "
                 f"{'P25':>10s} {'P50':>10s} {'P75':>10s}"]
    for r in panel_b_rows:
        if 'assets' in r['variable'].lower():
            txt_lines.append(f"{r['variable']:<22s} {r['n']:>10,d} "
                              f"{r['mean']:>10,.0f} {r['sd']:>10,.0f} "
                              f"{r['p25']:>10,.0f} {r['p50']:>10,.0f} {r['p75']:>10,.0f}")
        else:
            txt_lines.append(f"{r['variable']:<22s} {r['n']:>10,d} "
                              f"{r['mean']:>10.4f} {r['sd']:>10.4f} "
                              f"{r['p25']:>10.4f} {r['p50']:>10.4f} {r['p75']:>10.4f}")

    txt_path = os.path.join(out_dir, 'table1_sample_summary.txt')
    with open(txt_path, 'w') as f:
        f.write("\n".join(txt_lines))

    print(f"✓ Saved: {tex_path}")
    print(f"✓ Saved: {txt_path}")


if __name__ == "__main__":
    main()
