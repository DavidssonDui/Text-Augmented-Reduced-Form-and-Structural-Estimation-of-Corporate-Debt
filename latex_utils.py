"""
latex_utils.py
==============
Format regression results as publication-quality LaTeX tables using
booktabs style.

Produces:
  - Coefficient rows with stars
  - Standard errors on a row below, in parentheses
  - Control variables in a middle block
  - Fixed effects indicators
  - N and R² in bottom block
  - \\sym{*}, \\sym{**}, \\sym{***} marks for sig levels
"""

from .regression_utils import stars_for


SIG_NOTE = (r" \multicolumn{2}{l}{\footnotesize Cluster-robust standard"
            r" errors by firm. $^{*}p<0.1$, $^{**}p<0.05$, $^{***}p<0.01$}")


def _latex_escape(s):
    """Convert underscores in variable names to LaTeX-safe form."""
    return str(s).replace('_', r'\_')


def format_reg_table_latex(
    results_list,
    column_names,
    focal_vars,
    control_vars=None,
    label=None,
    caption=None,
    focal_display_names=None,
    control_display_names=None,
    additional_indicator_rows=None,
):
    """
    Format a list of statsmodels results objects as a LaTeX booktabs table.

    Arguments:
        results_list: list of fitted statsmodels results
        column_names: list of column headers (one per result)
        focal_vars: list of regressor names to show with coefficient + SE
        control_vars: list of control variables (coefficient only, no SE)
        label: LaTeX \\label to attach
        caption: LaTeX \\caption text
        focal_display_names: optional dict mapping var name → display name
        control_display_names: optional dict for controls
        additional_indicator_rows: list of (row_name, [cells]) tuples —
            e.g., [("Industry FE", ["Yes", "Yes", "No"])]

    Returns: a string containing a complete LaTeX table environment.
    """
    focal_display_names = focal_display_names or {}
    control_display_names = control_display_names or {}
    additional_indicator_rows = additional_indicator_rows or []

    n_cols = len(column_names)
    col_spec = "l" + "c" * n_cols

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    if caption:
        lines.append(f"\\caption{{{caption}}}")
    if label:
        lines.append(f"\\label{{{label}}}")
    lines.append(r"\small")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # Column headers: use the caller-supplied column_names only
    # (don't duplicate with auto-numbered "(1)", "(2)", etc.)
    header_cells = ["", *[str(n) for n in column_names]]
    lines.append(" & ".join(header_cells) + r" \\")
    lines.append(r"\midrule")

    # ── Focal rows ──
    for var in focal_vars:
        disp = focal_display_names.get(var, _latex_escape(var))

        # Coefficient row
        coef_cells = [disp]
        for r in results_list:
            if var in r.params.index:
                coef = r.params[var]
                pval = r.pvalues[var]
                coef_cells.append(f"${coef:+.4f}^{{{stars_for(pval)}}}$")
            else:
                coef_cells.append("")
        lines.append(" & ".join(coef_cells) + r" \\")

        # SE row
        se_cells = [""]
        for r in results_list:
            if var in r.params.index:
                se_cells.append(f"({r.bse[var]:.4f})")
            else:
                se_cells.append("")
        lines.append(" & ".join(se_cells) + r" \\")

    # ── Control rows (coef only) ──
    if control_vars:
        lines.append(r"\midrule")
        for var in control_vars:
            disp = control_display_names.get(var, _latex_escape(var))
            cells = [disp]
            for r in results_list:
                if var in r.params.index:
                    coef = r.params[var]
                    pval = r.pvalues[var]
                    cells.append(f"${coef:+.4f}^{{{stars_for(pval)}}}$")
                else:
                    cells.append("")
            lines.append(" & ".join(cells) + r" \\")

    # ── FE indicator rows (auto-detected) ──
    lines.append(r"\midrule")
    has_ind_fe = []
    has_yr_fe = []
    for r in results_list:
        params = list(r.params.index)
        has_ind_fe.append("Yes" if any('sic1' in p for p in params) else "No")
        has_yr_fe.append("Yes" if any('fyear_cat' in p for p in params) else "No")
    if any(x == "Yes" for x in has_ind_fe):
        lines.append(" & ".join(["Industry FE", *has_ind_fe]) + r" \\")
    if any(x == "Yes" for x in has_yr_fe):
        lines.append(" & ".join(["Year FE", *has_yr_fe]) + r" \\")

    for name, cells in additional_indicator_rows:
        lines.append(" & ".join([name, *[str(c) for c in cells]]) + r" \\")

    # ── N and R² / pseudo-R² ──
    lines.append(r"\midrule")
    lines.append(" & ".join(["Observations",
                              *[f"{int(r.nobs):,}" for r in results_list]]) + r" \\")
    r2_cells = ["R-squared"]
    for r in results_list:
        if hasattr(r, 'rsquared'):
            r2_cells.append(f"{r.rsquared:.4f}")
        elif hasattr(r, 'prsquared'):
            r2_cells.append(f"{r.prsquared:.4f} (pseudo)")
        else:
            r2_cells.append("")
    lines.append(" & ".join(r2_cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if caption or label:
        lines.append(r"\\[-2ex]")
        lines.append(r"\begin{flushleft}")
        lines.append(r"\footnotesize")
        lines.append(r"\textit{Notes}: Cluster-robust standard errors in parentheses,"
                     r" clustered by firm (gvkey). Stars indicate statistical"
                     r" significance: $^{*}p<0.1$, $^{**}p<0.05$, $^{***}p<0.01$.")
        lines.append(r"\end{flushleft}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def write_latex_table(path, latex_string):
    """Convenience wrapper to write a LaTeX table string to disk."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(latex_string)
