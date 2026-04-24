"""
figure1_quintile_chart.py
=========================
Figure 1: Default rate by FinBERT quintile (bar chart visualization
of Table 5).

Outputs:
    output/figures/figure1_quintile_chart.pdf
    output/figures/figure1_quintile_chart.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 11

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.regression_utils import prepare_sample


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    panel = pd.read_csv(os.path.join(root, 'data', 'panel_smm_ready.csv'))
    panel = prepare_sample(panel)

    sub = panel.dropna(subset=['s_finbert', 'def_within_2']).copy()
    sub['fb_quintile'] = sub.groupby('fyear')['s_finbert'].transform(
        lambda x: pd.qcut(x, q=5, labels=False, duplicates='drop') + 1
    )

    rates = []
    counts = []
    for q in [1, 2, 3, 4, 5]:
        qsub = sub[sub['fb_quintile'] == q]
        rates.append(100 * qsub['def_within_2'].mean())
        counts.append(int(qsub['def_within_2'].sum()))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    # Left panel: default rate
    bars1 = ax1.bar(range(1, 6), rates,
                      color=['#4575b4', '#74add1', '#abd9e9',
                             '#f46d43', '#a50026'],
                      edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('FinBERT quintile (1=most positive, 5=most negative)')
    ax1.set_ylabel('Default rate within 2 years (%)')
    ax1.set_title('Panel A: Default rate by FinBERT quintile')
    ax1.set_xticks(range(1, 6))
    for bar, val in zip(bars1, rates):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                  f'{val:.2f}%', ha='center', va='bottom', fontsize=10)
    ax1.set_ylim(0, max(rates) * 1.15)
    ax1.grid(axis='y', linestyle=':', alpha=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Right panel: count of defaults
    bars2 = ax2.bar(range(1, 6), counts,
                      color=['#4575b4', '#74add1', '#abd9e9',
                             '#f46d43', '#a50026'],
                      edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('FinBERT quintile')
    ax2.set_ylabel('Number of defaults')
    ax2.set_title('Panel B: Default count by FinBERT quintile')
    ax2.set_xticks(range(1, 6))
    for bar, val in zip(bars2, counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                  f'{val}', ha='center', va='bottom', fontsize=10)
    ax2.set_ylim(0, max(counts) * 1.15)
    ax2.grid(axis='y', linestyle=':', alpha=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()

    out_dir = os.path.join(root, 'output', 'figures')
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, 'figure1_quintile_chart.pdf')
    png_path = os.path.join(out_dir, 'figure1_quintile_chart.png')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.savefig(png_path, bbox_inches='tight', dpi=150)
    plt.close()

    print(f"✓ Saved: {pdf_path}")
    print(f"✓ Saved: {png_path}")


if __name__ == "__main__":
    main()
