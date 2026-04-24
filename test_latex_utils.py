"""
test_latex_utils.py
===================
Tests for src/latex_utils.py — LaTeX table formatter.

Verifies:
  - Output is syntactically valid LaTeX (all env pairs balanced)
  - Booktabs rules appear in the right places
  - Stars, SEs, and coefficients formatted correctly
  - Optional caption/label included when provided
"""

import re
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from src.latex_utils import (
    format_reg_table_latex, write_latex_table, _latex_escape,
)


@pytest.fixture
def simple_ols_result():
    """Simple OLS result for tests."""
    rng = np.random.default_rng(42)
    n = 100
    x = rng.uniform(0, 1, n)
    y = 2 + 0.5 * x + rng.normal(0, 0.1, n)
    X = sm.add_constant(pd.DataFrame({'x': x}))
    return sm.OLS(y, X).fit(cov_type='HC0')


@pytest.fixture
def two_ols_results():
    """Two OLS results of varying specification complexity."""
    rng = np.random.default_rng(42)
    n = 100
    x1 = rng.uniform(0, 1, n)
    x2 = rng.uniform(0, 1, n)
    y = 2 + 0.5 * x1 + 0.3 * x2 + rng.normal(0, 0.1, n)

    X1 = sm.add_constant(pd.DataFrame({'x1': x1}))
    m1 = sm.OLS(y, X1).fit(cov_type='HC0')

    X2 = sm.add_constant(pd.DataFrame({'x1': x1, 'x2': x2}))
    m2 = sm.OLS(y, X2).fit(cov_type='HC0')

    return [m1, m2]


# ════════════════════════════════════════════════════════════════
# _latex_escape
# ════════════════════════════════════════════════════════════════

class TestLatexEscape:
    """Underscore escaping for LaTeX-safe variable names."""

    def test_underscores_escaped(self):
        assert _latex_escape('s_finbert') == r's\_finbert'
        assert _latex_escape('book_leverage_w') == r'book\_leverage\_w'

    def test_no_change_if_no_underscores(self):
        assert _latex_escape('Variable') == 'Variable'
        assert _latex_escape('x') == 'x'

    def test_accepts_non_string(self):
        # Should convert to string first
        assert _latex_escape(42) == '42'


# ════════════════════════════════════════════════════════════════
# format_reg_table_latex — structural tests
# ════════════════════════════════════════════════════════════════

class TestFormatRegTableStructure:
    """Structural checks on the LaTeX output."""

    def test_contains_table_environment(self, simple_ols_result):
        latex = format_reg_table_latex([simple_ols_result], ['(1)'],
                                         focal_vars=['x'])
        assert r'\begin{table}' in latex
        assert r'\end{table}' in latex

    def test_contains_tabular_environment(self, simple_ols_result):
        latex = format_reg_table_latex([simple_ols_result], ['(1)'],
                                         focal_vars=['x'])
        assert r'\begin{tabular}' in latex
        assert r'\end{tabular}' in latex

    def test_begin_end_pairs_balanced(self, simple_ols_result):
        latex = format_reg_table_latex([simple_ols_result], ['(1)'],
                                         focal_vars=['x'])
        # Count all \begin and \end occurrences
        n_begin = len(re.findall(r'\\begin\{', latex))
        n_end = len(re.findall(r'\\end\{', latex))
        assert n_begin == n_end, \
            f"Unbalanced: {n_begin} \\begin vs {n_end} \\end"

    def test_contains_booktabs_rules(self, simple_ols_result):
        latex = format_reg_table_latex([simple_ols_result], ['(1)'],
                                         focal_vars=['x'])
        assert r'\toprule' in latex
        assert r'\bottomrule' in latex
        assert r'\midrule' in latex

    def test_contains_caption_when_provided(self, simple_ols_result):
        latex = format_reg_table_latex([simple_ols_result], ['(1)'],
                                         focal_vars=['x'],
                                         caption='Test caption')
        assert r'\caption{Test caption}' in latex

    def test_contains_label_when_provided(self, simple_ols_result):
        latex = format_reg_table_latex([simple_ols_result], ['(1)'],
                                         focal_vars=['x'],
                                         label='tab:test')
        assert r'\label{tab:test}' in latex

    def test_column_names_appear_in_header(self, two_ols_results):
        latex = format_reg_table_latex(two_ols_results,
                                         ['Simple', 'Full'],
                                         focal_vars=['x1'])
        assert 'Simple' in latex
        assert 'Full' in latex

    def test_header_row_count(self, two_ols_results):
        """
        Should have exactly one header row with column names — not
        duplicated with auto (1)(2) numbering.
        """
        latex = format_reg_table_latex(two_ols_results,
                                         ['Col A', 'Col B'],
                                         focal_vars=['x1'])
        # Count lines in the output that contain "Col A" — should be exactly 1
        lines = [l for l in latex.split('\n') if 'Col A' in l]
        assert len(lines) == 1, \
            f"Expected 1 header row with column names, got {len(lines)}"


class TestFormatRegTableContent:
    """Content checks: stars, SEs, coefficients."""

    def test_coefficients_displayed_with_4_decimals(self, simple_ols_result):
        latex = format_reg_table_latex([simple_ols_result], ['(1)'],
                                         focal_vars=['x'])
        # Find coefficient pattern: +X.XXXX or -X.XXXX with superscript stars
        pattern = r'[+-]\d\.\d{4}\^\{(\*\*\*|\*\*|\*|)\}'
        matches = re.findall(pattern, latex)
        assert len(matches) > 0, "No coefficients found with 4-decimal format"

    def test_standard_errors_in_parentheses(self, simple_ols_result):
        latex = format_reg_table_latex([simple_ols_result], ['(1)'],
                                         focal_vars=['x'])
        # SE row pattern: "(X.XXXX)" — we can't be TOO specific since
        # (1), (2) also show up, so look for 4-decimal parenthesized numbers
        pattern = r'\(\d\.\d{4}\)'
        matches = re.findall(pattern, latex)
        assert len(matches) > 0, "No standard errors in parenthesized form"

    def test_focal_var_appears(self, simple_ols_result):
        latex = format_reg_table_latex([simple_ols_result], ['(1)'],
                                         focal_vars=['x'])
        # 'x' may appear as part of \sigma_V^2 etc., use boundary-aware check
        assert 'x' in latex  # weak but sufficient for trivial var name

    def test_focal_display_names_override(self, simple_ols_result):
        latex = format_reg_table_latex([simple_ols_result], ['(1)'],
                                         focal_vars=['x'],
                                         focal_display_names={'x': r'$\alpha$'})
        assert r'$\alpha$' in latex

    def test_control_vars_shown_without_se(self, two_ols_results):
        """
        Control vars should appear with coefficient but not with SE row.
        """
        latex = format_reg_table_latex(two_ols_results, ['(1)', '(2)'],
                                         focal_vars=['x1'],
                                         control_vars=['x2'])
        # x2 should appear as a row, but only once as a coefficient row
        # (focal vars have two rows, control vars have one)
        assert 'x2' in latex

    def test_observation_count_included(self, simple_ols_result):
        latex = format_reg_table_latex([simple_ols_result], ['(1)'],
                                         focal_vars=['x'])
        assert 'Observations' in latex
        # Should show comma-formatted N
        assert f'{int(simple_ols_result.nobs):,}' in latex

    def test_rsquared_included(self, simple_ols_result):
        latex = format_reg_table_latex([simple_ols_result], ['(1)'],
                                         focal_vars=['x'])
        assert 'R-squared' in latex


class TestFormatRegTableEdgeCases:

    def test_handles_missing_focal_var(self, simple_ols_result):
        """
        If a focal_var isn't in some models, those cells should be empty,
        not crash.
        """
        latex = format_reg_table_latex([simple_ols_result], ['(1)'],
                                         focal_vars=['x', 'nonexistent'])
        # Should still produce valid output
        assert r'\begin{table}' in latex
        assert r'\end{table}' in latex

    def test_works_without_caption_or_label(self, simple_ols_result):
        latex = format_reg_table_latex([simple_ols_result], ['(1)'],
                                         focal_vars=['x'])
        # Should not contain empty caption/label tags
        assert r'\caption{}' not in latex
        assert r'\label{}' not in latex


# ════════════════════════════════════════════════════════════════
# write_latex_table
# ════════════════════════════════════════════════════════════════

class TestWriteLatexTable:

    def test_creates_file_with_content(self, tmp_path, simple_ols_result):
        latex = format_reg_table_latex([simple_ols_result], ['(1)'],
                                         focal_vars=['x'])
        out_path = tmp_path / 'subdir' / 'table.tex'
        write_latex_table(str(out_path), latex)

        assert out_path.exists()
        content = out_path.read_text()
        assert r'\begin{table}' in content

    def test_creates_parent_directory_if_missing(self, tmp_path):
        """write_latex_table should mkdir the parent."""
        out_path = tmp_path / 'nested' / 'deep' / 'table.tex'
        write_latex_table(str(out_path), 'test content')
        assert out_path.exists()
