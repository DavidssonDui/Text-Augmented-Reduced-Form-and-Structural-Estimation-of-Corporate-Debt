"""
table8_out_of_sample.py
=======================
Table 8: Out-of-sample prediction performance.

Splits the sample chronologically: train on fyear <= 2014,
test on fyear >= 2015. Fits logit models and evaluates test-set AUC.

This addresses the concern that FinBERT's in-sample predictive power
might reflect overfitting — since the FinBERT signal itself was
constructed via supervised learning on default labels, any in-sample
predictive power could be driven by information leakage.

The OOS test uses data the FinBERT projection never saw. Only the
2015-2022 subsample is evaluated. We also restrict to years after the
FinBERT training cutoff (which was median year ~2014) to ensure the
test data was never used in projection construction.

Models evaluated (all fitted on train, evaluated on test):
  (1) Financial controls only (log_at, leverage, Q, profit, cash)
  (2) + s_lm
  (3) + s_finbert
  (4) + s_lm + s_finbert

Inputs:
    data/panel_smm_ready.csv

Outputs:
    output/tables/table8_out_of_sample.tex
    output/tables/table8_out_of_sample.txt

Usage:
    python scripts/table8_out_of_sample.py
"""

import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.regression_utils import prepare_sample
from src.latex_utils import write_latex_table


CONTROLS = ['log_at', 'book_leverage_w', 'tobins_q_fix_w',
            'profitability_w', 'cash_ratio_w']

TRAIN_CUTOFF = 2014  # inclusive
RANDOM_STATE = 42


def fit_and_evaluate(train_df, test_df, y_col, x_cols):
    """
    Fit logit on train, evaluate on test.
    Returns dict with AUC, average precision, and other diagnostics.
    """
    # Clean both sets using same columns
    cols = [y_col] + x_cols
    train_clean = train_df[cols].dropna()
    test_clean = test_df[cols].dropna()

    X_train = sm.add_constant(train_clean[x_cols].astype(float))
    y_train = train_clean[y_col].astype(int)

    X_test = sm.add_constant(test_clean[x_cols].astype(float),
                             has_constant='add')
    y_test = test_clean[y_col].astype(int)

    # Fit logit
    try:
        model = sm.Logit(y_train, X_train).fit(disp=False, maxiter=200)
    except Exception as e:
        return {'error': str(e)}

    # Predict on test
    p_test = model.predict(X_test)

    # AUC and AP (requires some positives in test)
    out = {
        'n_train': int(len(y_train)),
        'n_test': int(len(y_test)),
        'pos_train': int(y_train.sum()),
        'pos_test': int(y_test.sum()),
    }

    if y_test.sum() > 0 and y_test.sum() < len(y_test):
        out['auc_test'] = float(roc_auc_score(y_test, p_test))
        out['ap_test'] = float(average_precision_score(y_test, p_test))
        # In-sample for comparison
        p_train = model.predict(X_train)
        out['auc_train'] = float(roc_auc_score(y_train, p_train))
    else:
        out['auc_test'] = np.nan
        out['ap_test'] = np.nan
        out['auc_train'] = np.nan

    return out


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    panel_path = os.path.join(root, 'data', 'panel_smm_ready.csv')

    print(f"Loading {panel_path}...")
    panel = pd.read_csv(panel_path)
    panel = prepare_sample(panel)

    # Split
    train = panel[panel['fyear'] <= TRAIN_CUTOFF].copy()
    test = panel[panel['fyear'] > TRAIN_CUTOFF].copy()
    print(f"  Train: fyear <= {TRAIN_CUTOFF}, {len(train):,} rows, "
          f"{int(train['def_within_2'].sum())} defaults")
    print(f"  Test:  fyear > {TRAIN_CUTOFF}, {len(test):,} rows, "
          f"{int(test['def_within_2'].sum())} defaults")
    print()

    y_col = 'def_within_2'

    # Run the 4 specifications
    specs = []
    spec_defs = [
        ('Controls only', CONTROLS),
        ('+ s_lm', CONTROLS + ['s_lm']),
        ('+ s_finbert', CONTROLS + ['s_finbert']),
        ('+ both', CONTROLS + ['s_lm', 's_finbert']),
    ]

    for name, x_cols in spec_defs:
        result = fit_and_evaluate(train, test, y_col, x_cols)
        result['name'] = name
        specs.append(result)
        print(f"  {name}:")
        if 'error' in result:
            print(f"    ERROR: {result['error']}")
        else:
            print(f"    n_train={result['n_train']:,}, n_test={result['n_test']:,}")
            print(f"    test defaults: {result['pos_test']}, "
                  f"train defaults: {result['pos_train']}")
            print(f"    AUC train: {result['auc_train']:.4f}")
            print(f"    AUC test:  {result['auc_test']:.4f}")
            print(f"    AP  test:  {result['ap_test']:.4f}")
        print()

    # Format LaTeX output manually since this table has different structure
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Out-of-sample prediction performance.}")
    lines.append(r"\label{tab:t8_oos}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r" & Controls & $+ s^{\mathrm{LM}}$ & "
                 r"$+ s^{\mathrm{FB}}$ & $+$ both \\")
    lines.append(r" & (1) & (2) & (3) & (4) \\")
    lines.append(r"\midrule")
    # AUC test row
    row = "Test-set AUC"
    for s in specs:
        if 'error' not in s:
            row += f" & {s['auc_test']:.4f}"
        else:
            row += " & --"
    lines.append(row + r" \\")

    # AP test row
    row = "Test-set avg. precision"
    for s in specs:
        if 'error' not in s:
            row += f" & {s['ap_test']:.4f}"
        else:
            row += " & --"
    lines.append(row + r" \\")

    # Training AUC for reference
    row = "In-sample AUC (train)"
    for s in specs:
        if 'error' not in s:
            row += f" & {s['auc_train']:.4f}"
        else:
            row += " & --"
    lines.append(row + r" \\")

    lines.append(r"\midrule")
    # N train
    row = "N (train)"
    for s in specs:
        row += f" & {s.get('n_train', '--'):,}" if s.get('n_train') else " & --"
    lines.append(row + r" \\")
    # N test
    row = "N (test)"
    for s in specs:
        row += f" & {s.get('n_test', '--'):,}" if s.get('n_test') else " & --"
    lines.append(row + r" \\")
    # Defaults test
    row = "Defaults (test)"
    for s in specs:
        row += f" & {s.get('pos_test', 0)}"
    lines.append(row + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\\[-2ex]")
    lines.append(r"\begin{flushleft}")
    lines.append(r"\footnotesize")
    lines.append(r"\textit{Notes}: Logistic regression models fit on firm-years "
                 r"with $\text{fyear} \le 2014$ and evaluated on firm-years "
                 r"with $\text{fyear} \ge 2015$. Controls are log assets, "
                 r"book leverage, Tobin's Q, profitability, and cash/assets. "
                 r"The text signals $s^{\mathrm{LM}}$ and $s^{\mathrm{FB}}$ "
                 r"are standardized within year. The dependent variable is "
                 r"default within two years. AUC $=$ area under the ROC "
                 r"curve; AP $=$ average precision (area under "
                 r"precision-recall curve).")
    lines.append(r"\end{flushleft}")
    lines.append(r"\end{table}")

    latex = "\n".join(lines)

    out_dir = os.path.join(root, 'output', 'tables')
    tex_path = os.path.join(out_dir, 'table8_out_of_sample.tex')
    write_latex_table(tex_path, latex)

    # Plain text
    txt_lines = []
    txt_lines.append("=" * 80)
    txt_lines.append("TABLE 8: Out-of-sample prediction performance")
    txt_lines.append(f"Train: fyear <= {TRAIN_CUTOFF}, test: fyear >= {TRAIN_CUTOFF+1}")
    txt_lines.append("=" * 80)
    txt_lines.append("")
    txt_lines.append(f"{'Metric':<26s} {'Controls':>12s} {'+s_lm':>12s} "
                      f"{'+s_finbert':>12s} {'+both':>12s}")
    txt_lines.append("-" * 82)
    for label, key, fmt in [
        ('Test AUC', 'auc_test', '.4f'),
        ('Test Avg. Precision', 'ap_test', '.4f'),
        ('Train AUC (reference)', 'auc_train', '.4f'),
    ]:
        row = f"{label:<26s}"
        for s in specs:
            if 'error' not in s:
                row += f" {s[key]:>12{fmt}}"
            else:
                row += " " * 12 + " "
        txt_lines.append(row)
    txt_lines.append("-" * 82)
    for label, key in [('N train', 'n_train'), ('N test', 'n_test'),
                       ('Defaults test', 'pos_test')]:
        row = f"{label:<26s}"
        for s in specs:
            val = s.get(key, 0)
            row += f" {val:>12,d}"
        txt_lines.append(row)

    txt_path = os.path.join(out_dir, 'table8_out_of_sample.txt')
    with open(txt_path, 'w') as f:
        f.write("\n".join(txt_lines))

    print(f"✓ Saved: {tex_path}")
    print(f"✓ Saved: {txt_path}")


if __name__ == "__main__":
    main()
