"""
figure2_sample_over_time.py
============================
Figure 2: Sample composition and default events over time.

Three-panel figure:
  Panel A: Number of firm-years per fiscal year (sample composition)
  Panel B: FinBERT signal coverage rate per year
  Panel C: Number of default events per year

Outputs:
    output/figures/figure2_sample_over_time.pdf
    output/figures/figure2_sample_over_time.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 10

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.regression_utils import prepare_sample


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    panel = pd.read_csv(os.path.join(root, 'data', 'panel_smm_ready.csv'))
    panel = prepare_sample(panel)

    yearly = panel.groupby('fyear').agg(
        n_obs=('gvkey', 'count'),
        n_finbert=('s_finbert', lambda x: x.notna().sum()),
        n_default=('default_broad', 'sum'),
    ).reset_index()
    yearly['coverage_pct'] = 100 * yearly['n_finbert'] / yearly['n_obs']

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))

    # Panel A: firm-years per year
    axes[0].bar(yearly['fyear'], yearly['n_obs'], color='#377eb8',
                  edgecolor='black', linewidth=0.5)
    axes[0].set_xlabel('Fiscal year')
    axes[0].set_ylabel('Firm-year observations')
    axes[0].set_title('Panel A: Sample size by year')
    axes[0].grid(axis='y', linestyle=':', alpha=0.5)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].tick_params(axis='x', rotation=45)

    # Panel B: coverage
    axes[1].plot(yearly['fyear'], yearly['coverage_pct'],
                   color='#e41a1c', marker='o', linewidth=2)
    axes[1].set_xlabel('Fiscal year')
    axes[1].set_ylabel('FinBERT coverage rate (%)')
    axes[1].set_title('Panel B: Text signal coverage')
    axes[1].set_ylim(0, 100)
    axes[1].grid(axis='y', linestyle=':', alpha=0.5)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].tick_params(axis='x', rotation=45)

    # Panel C: defaults per year
    axes[2].bar(yearly['fyear'], yearly['n_default'], color='#a50026',
                  edgecolor='black', linewidth=0.5)
    axes[2].set_xlabel('Fiscal year')
    axes[2].set_ylabel('Default events')
    axes[2].set_title('Panel C: Default events by year')
    axes[2].grid(axis='y', linestyle=':', alpha=0.5)
    axes[2].spines['top'].set_visible(False)
    axes[2].spines['right'].set_visible(False)
    axes[2].tick_params(axis='x', rotation=45)

    plt.tight_layout()

    out_dir = os.path.join(root, 'output', 'figures')
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, 'figure2_sample_over_time.pdf')
    png_path = os.path.join(out_dir, 'figure2_sample_over_time.png')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.savefig(png_path, bbox_inches='tight', dpi=150)
    plt.close()

    print(f"✓ Saved: {pdf_path}")
    print(f"✓ Saved: {png_path}")


if __name__ == "__main__":
    main()
