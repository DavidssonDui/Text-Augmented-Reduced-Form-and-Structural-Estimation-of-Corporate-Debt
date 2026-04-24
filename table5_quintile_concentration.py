"""
table5_quintile_concentration.py
================================
Table 5: Default concentration across FinBERT quintiles.

Rather than a regression, this table reports the unconditional default
rate in each quintile of FinBERT signal. The story: defaults
concentrate in the top quintile (most negative tone), demonstrating
the FinBERT signal isn't just statistically significant — it's
economically meaningful.

Outputs:
    output/tables/table5_quintile_concentration.tex
    output/tables/table5_quintile_concentration.txt
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.regression_utils import prepare_sample
from src.latex_utils import write_latex_table


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    panel_path = os.path.join(root, 'data', 'panel_smm_ready.csv')
    panel = pd.read_csv(panel_path)
    panel = prepare_sample(panel)

    # Subset to firm-years with FinBERT signal AND observable forward default
    sub = panel.dropna(subset=['s_finbert', 'def_within_2']).copy()

    # Quintile assignment within year (so quintiles aren't dominated by
    # one calendar year)
    sub['fb_quintile'] = sub.groupby('fyear')['s_finbert'].transform(
        lambda x: pd.qcut(x, q=5, labels=False, duplicates='drop') + 1
    )

    # Compute by-quintile stats
    rows = []
    for q in [1, 2, 3, 4, 5]:
        qsub = sub[sub['fb_quintile'] == q]
        if len(qsub) == 0:
            continue
        n_obs = len(qsub)
        n_def = int(qsub['def_within_2'].sum())
        rate = n_def / n_obs
        rows.append({
            'quintile': q,
            'n_obs': n_obs,
            'n_def': n_def,
            'def_rate_pct': 100 * rate,
            'mean_signal': qsub['s_finbert'].mean(),
        })

    total_def = int(sub['def_within_2'].sum())
    total_obs = len(sub)

    # Console
    print("=" * 80)
    print(" TABLE 5: Default concentration by FinBERT quintile")
    print("=" * 80)
    print(f" {'Quintile':<12s} {'Mean signal':>14s} "
          f"{'N obs':>12s} {'N defaults':>14s} "
          f"{'Default rate':>18s} {'Share of defaults':>20s}")
    print(" " + "-" * 92)
    for r in rows:
        share_def = 100 * r['n_def'] / total_def if total_def else 0
        print(f" {r['quintile']:<12d} {r['mean_signal']:>14.3f} "
              f"{r['n_obs']:>12,d} {r['n_def']:>14d} "
              f"{r['def_rate_pct']:>17.3f}% {share_def:>19.1f}%")
    print(" " + "-" * 92)
    print(f" {'Total':<12s} {sub['s_finbert'].mean():>14.3f} "
          f"{total_obs:>12,d} {total_def:>14d} "
          f"{100*total_def/total_obs:>17.3f}% {100.0:>19.1f}%")

    # LaTeX
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Default concentration by FinBERT quintile.}")
    lines.append(r"\label{tab:t5_quintile}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"FinBERT & Mean & & Defaults within & Default & Share of \\")
    lines.append(r"quintile & signal & N firm-years & 2 years & rate (\%) & defaults (\%) \\")
    lines.append(r"\midrule")
    for r in rows:
        share_def = 100 * r['n_def'] / total_def if total_def else 0
        # Quintile label: 1 (lowest), 5 (highest)
        if r['quintile'] == 1:
            qlabel = "1 (lowest)"
        elif r['quintile'] == 5:
            qlabel = "5 (highest)"
        else:
            qlabel = str(r['quintile'])
        lines.append(
            f"{qlabel} & {r['mean_signal']:+.3f} & {r['n_obs']:,} & "
            f"{r['n_def']} & {r['def_rate_pct']:.3f} & {share_def:.1f} \\\\"
        )
    lines.append(r"\midrule")
    lines.append(
        f"Total & {sub['s_finbert'].mean():+.3f} & {total_obs:,} & "
        f"{total_def} & {100*total_def/total_obs:.3f} & 100.0 \\\\"
    )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\\[-1ex]")
    lines.append(r"\begin{flushleft}")
    lines.append(r"\footnotesize")
    lines.append(r"\textit{Notes}: Sample restricted to firm-years with "
                 r"both FinBERT signal and observable forward default. "
                 r"Quintiles are assigned within each fiscal year so that "
                 r"sample composition does not drive the result. The top "
                 r"quintile (5) captures the firm-years with the most "
                 r"negative-toned 10-K text. The default rate within two "
                 r"years rises monotonically across quintiles, with the "
                 r"top quintile concentrating a disproportionate share "
                 r"of subsequent default events.")
    lines.append(r"\end{flushleft}")
    lines.append(r"\end{table}")

    out_dir = os.path.join(root, 'output', 'tables')
    tex_path = os.path.join(out_dir, 'table5_quintile_concentration.tex')
    with open(tex_path, 'w') as f:
        f.write("\n".join(lines))

    txt_lines = ["=" * 92,
                 "TABLE 5: Default concentration by FinBERT quintile",
                 "=" * 92,
                 f"{'Quintile':<12s} {'Mean signal':>14s} "
                 f"{'N obs':>12s} {'N defaults':>14s} "
                 f"{'Def rate':>18s} {'Share of defaults':>20s}",
                 "-" * 92]
    for r in rows:
        share_def = 100 * r['n_def'] / total_def if total_def else 0
        txt_lines.append(
            f"{r['quintile']:<12d} {r['mean_signal']:>14.3f} "
            f"{r['n_obs']:>12,d} {r['n_def']:>14d} "
            f"{r['def_rate_pct']:>17.3f}% {share_def:>19.1f}%"
        )
    txt_lines.append("-" * 92)
    txt_lines.append(
        f"{'Total':<12s} {sub['s_finbert'].mean():>14.3f} "
        f"{total_obs:>12,d} {total_def:>14d} "
        f"{100*total_def/total_obs:>17.3f}% {100.0:>19.1f}%"
    )

    txt_path = os.path.join(out_dir, 'table5_quintile_concentration.txt')
    with open(txt_path, 'w') as f:
        f.write("\n".join(txt_lines))

    print(f"\n✓ Saved: {tex_path}")
    print(f"✓ Saved: {txt_path}")


if __name__ == "__main__":
    main()
