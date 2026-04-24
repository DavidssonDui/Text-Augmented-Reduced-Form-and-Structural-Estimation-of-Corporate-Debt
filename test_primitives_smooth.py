"""
test_primitives_smooth.py
=========================
Tests for deqn_solver.src.primitives_smooth.

Covers:
  - profit_fn
  - corporate_tax
  - distribution_tax
  - equity_cost (modified for Option 1: no fixed cost)
  - taxable_income
  - realized_net_worth
  - bankruptcy_recovery
  - period_payoff
"""

import math
import torch
import pytest

from deqn_solver.src.primitives_smooth import (
    profit_fn, corporate_tax, distribution_tax, equity_cost,
    taxable_income, realized_net_worth, bankruptcy_recovery, period_payoff,
)
from deqn_solver.src.config import ModelParams


# ════════════════════════════════════════════════════════════════
# profit_fn
# ════════════════════════════════════════════════════════════════

class TestProfitFn:
    """Cobb-Douglas production: y = z * k^alpha."""

    def test_hand_calculation_single_point(self):
        """profit_fn(z=1, k=10, alpha=0.5) = 1 * sqrt(10) ≈ 3.162"""
        z = torch.tensor([1.0])
        k = torch.tensor([10.0])
        result = profit_fn(z, k, alpha=0.5)
        expected = math.sqrt(10.0)
        assert abs(result.item() - expected) < 1e-6

    def test_hand_calculation_multiple_points(self):
        """Test vectorized evaluation."""
        z = torch.tensor([1.0, 2.0, 0.5])
        k = torch.tensor([4.0, 4.0, 4.0])
        result = profit_fn(z, k, alpha=0.5)
        expected = torch.tensor([2.0, 4.0, 1.0])
        torch.testing.assert_close(result, expected)

    def test_monotonic_in_k(self):
        """Higher k should yield higher output (for k > 0)."""
        z = torch.tensor([1.0])
        k_low = torch.tensor([5.0])
        k_high = torch.tensor([10.0])
        assert profit_fn(z, k_high, 0.5).item() > profit_fn(z, k_low, 0.5).item()

    def test_monotonic_in_z(self):
        """Higher z should yield higher output."""
        z_low = torch.tensor([0.5])
        z_high = torch.tensor([2.0])
        k = torch.tensor([10.0])
        assert profit_fn(z_high, k, 0.5).item() > profit_fn(z_low, k, 0.5).item()

    def test_concave_in_k_when_alpha_lt_1(self):
        """For 0 < alpha < 1, production is concave in k."""
        z = torch.tensor([1.0])
        alpha = 0.5

        # Use unequal spacing where concavity should show clearly
        # y'(k) = 0.5 * k^(-0.5) is strictly decreasing
        # So marginal from k=1 to k=2 should exceed marginal from k=10 to k=11
        k1, k2 = torch.tensor([1.0]), torch.tensor([2.0])
        k3, k4 = torch.tensor([10.0]), torch.tensor([11.0])

        marg_low_k = profit_fn(z, k2, alpha).item() - profit_fn(z, k1, alpha).item()
        marg_high_k = profit_fn(z, k4, alpha).item() - profit_fn(z, k3, alpha).item()

        assert marg_high_k < marg_low_k, \
            "Marginal product must diminish (alpha < 1)"

    def test_clamps_negative_k(self):
        """k <= 0 should not produce NaN or infinity."""
        z = torch.tensor([1.0])
        k_zero = torch.tensor([0.0])
        k_neg = torch.tensor([-1.0])
        # Should not crash; clamped to small positive value
        out_zero = profit_fn(z, k_zero, 0.5)
        out_neg = profit_fn(z, k_neg, 0.5)
        assert torch.isfinite(out_zero).all()
        assert torch.isfinite(out_neg).all()


# ════════════════════════════════════════════════════════════════
# corporate_tax
# ════════════════════════════════════════════════════════════════

class TestCorporateTax:
    """Asymmetric corporate tax: tau_c_plus on income>0, tau_c_minus on income<0."""

    def test_positive_income_uses_tau_plus(self, default_params):
        income = torch.tensor([100.0])
        result = corporate_tax(income, default_params)
        expected = default_params.tau_c_plus * 100.0
        assert abs(result.item() - expected) < 1e-6

    def test_negative_income_uses_tau_minus(self, default_params):
        income = torch.tensor([-100.0])
        result = corporate_tax(income, default_params)
        expected = default_params.tau_c_minus * (-100.0)
        assert abs(result.item() - expected) < 1e-6

    def test_zero_income_yields_zero_tax(self, default_params):
        income = torch.tensor([0.0])
        assert corporate_tax(income, default_params).item() == 0.0

    def test_symmetric_rates_when_tau_plus_equals_tau_minus(self):
        p = ModelParams(tau_c_plus=0.3, tau_c_minus=0.3)
        # With symmetric tau, tax(I) = 0.3 * I for any I
        for val in [100.0, -100.0, 50.0]:
            income = torch.tensor([val])
            expected = 0.3 * val
            # float32 precision: use 1e-4 tolerance
            assert abs(corporate_tax(income, p).item() - expected) < 1e-4

    def test_mixed_batch(self, default_params):
        income = torch.tensor([100.0, -50.0, 0.0])
        result = corporate_tax(income, default_params)
        expected = torch.tensor([
            default_params.tau_c_plus * 100.0,
            default_params.tau_c_minus * (-50.0),
            0.0,
        ])
        torch.testing.assert_close(result, expected)


# ════════════════════════════════════════════════════════════════
# distribution_tax
# ════════════════════════════════════════════════════════════════

class TestDistributionTax:
    """
    Distribution tax: tau_d_bar * X - (tau_d_bar/phi) * (1 - exp(-phi*X)).

    Properties:
    - Zero for non-positive X (only positive distributions taxed)
    - Increasing in X
    - Concave for X > 0 (progressive toward tau_d_bar asymptote)
    """

    def test_zero_distribution_is_zero_tax(self, default_params):
        X = torch.tensor([0.0])
        assert distribution_tax(X, default_params).item() == 0.0

    def test_negative_distribution_is_zero_tax(self, default_params):
        """Distribution = equity issuance (negative X) shouldn't generate tax."""
        X = torch.tensor([-10.0])
        assert distribution_tax(X, default_params).item() == 0.0

    def test_positive_distribution_yields_positive_tax(self, default_params):
        X = torch.tensor([10.0])
        assert distribution_tax(X, default_params).item() > 0.0

    def test_monotonically_increasing_in_X(self, default_params):
        X_low = torch.tensor([1.0])
        X_high = torch.tensor([10.0])
        assert (distribution_tax(X_high, default_params).item()
                > distribution_tax(X_low, default_params).item())

    def test_tax_rate_bounded_by_tau_d_bar(self, default_params):
        """Average tax rate (tax / X) should be strictly less than tau_d_bar."""
        X = torch.tensor([100.0])
        tax = distribution_tax(X, default_params).item()
        avg_rate = tax / 100.0
        assert avg_rate < default_params.tau_d_bar


# ════════════════════════════════════════════════════════════════
# equity_cost (Option 1: no fixed cost)
# ════════════════════════════════════════════════════════════════

class TestEquityCost:
    """
    Equity cost (Option 1 modification):
        Λ(x) = lambda_1 * x + lambda_2 * x^2  for x > 0
        Λ(x) = 0                               for x <= 0

    No fixed cost (lambda_0 ignored).
    """

    def test_zero_issuance_is_zero_cost(self, default_params):
        x = torch.tensor([0.0])
        assert equity_cost(x, default_params).item() == 0.0

    def test_negative_issuance_is_zero_cost(self, default_params):
        """x < 0 means no issuance; cost should be zero, not negative."""
        x = torch.tensor([-5.0])
        assert equity_cost(x, default_params).item() == 0.0

    def test_positive_issuance_hand_calculation(self, default_params):
        """Λ(x) = lambda_1 * x + lambda_2 * x^2"""
        x_val = 10.0
        x = torch.tensor([x_val])
        expected = (default_params.lambda_1 * x_val
                    + default_params.lambda_2 * x_val ** 2)
        result = equity_cost(x, default_params)
        assert abs(result.item() - expected) < 1e-6

    def test_lambda_0_is_ignored(self):
        """The modified equity_cost should ignore lambda_0."""
        p_no_fixed = ModelParams(lambda_0=0.0, lambda_1=0.1, lambda_2=0.001)
        p_with_fixed = ModelParams(lambda_0=5.0, lambda_1=0.1, lambda_2=0.001)

        x = torch.tensor([10.0])
        cost_no_fixed = equity_cost(x, p_no_fixed).item()
        cost_with_fixed = equity_cost(x, p_with_fixed).item()

        # Option 1 ignores lambda_0, so costs should be IDENTICAL
        assert abs(cost_no_fixed - cost_with_fixed) < 1e-10, \
            "equity_cost must ignore lambda_0 in Option 1"

    def test_is_continuous_at_zero(self, default_params):
        """Cost should be continuous at x=0 (no jump for Option 1)."""
        x_tiny_pos = torch.tensor([1e-6])
        x_zero = torch.tensor([0.0])
        x_tiny_neg = torch.tensor([-1e-6])

        c_pos = equity_cost(x_tiny_pos, default_params).item()
        c_zero = equity_cost(x_zero, default_params).item()
        c_neg = equity_cost(x_tiny_neg, default_params).item()

        # All three values should be ~0 (within 1e-5)
        assert abs(c_pos) < 1e-5
        assert c_zero == 0.0
        assert c_neg == 0.0

    def test_monotonically_increasing(self, default_params):
        """For x > 0, cost increases with x."""
        x_low = torch.tensor([1.0])
        x_high = torch.tensor([10.0])
        assert (equity_cost(x_high, default_params).item()
                > equity_cost(x_low, default_params).item())

    def test_convex_for_positive_x(self, default_params):
        """Quadratic cost makes Λ convex in x for x > 0."""
        x1 = torch.tensor([1.0])
        x2 = torch.tensor([5.0])
        x3 = torch.tensor([10.0])
        c1, c2, c3 = [equity_cost(x, default_params).item() for x in [x1, x2, x3]]
        # Convexity check: midpoint inequality (4 is midpoint of 1 and 7)
        # Easier: (c3 - c2) > (c2 - c1) when the spacing is equal
        # Since our spacings aren't equal, use: marginal cost increases
        marginal_12 = (c2 - c1) / (5.0 - 1.0)
        marginal_23 = (c3 - c2) / (10.0 - 5.0)
        assert marginal_23 > marginal_12


# ════════════════════════════════════════════════════════════════
# taxable_income
# ════════════════════════════════════════════════════════════════

class TestTaxableIncome:
    """Taxable income: operating - depreciation - interest."""

    def test_hand_calculation(self, default_params):
        """
        income = z * k^alpha - delta * k - r_tilde * b
        With z=1, k=10, alpha=0.627, delta=0.15, r=0.05, b=5:
             = 1 * 10^0.627 - 0.15 * 10 - 0.05 * 5
             ≈ 4.238 - 1.5 - 0.25 = 2.488
        """
        z_prime = torch.tensor([1.0])
        k_prime = torch.tensor([10.0])
        r_tilde = torch.tensor([0.05])
        b_prime = torch.tensor([5.0])
        result = taxable_income(z_prime, k_prime, r_tilde, b_prime, default_params)
        expected = (1.0 * 10.0 ** default_params.alpha
                    - default_params.delta * 10.0 - 0.05 * 5.0)
        assert abs(result.item() - expected) < 1e-4

    def test_zero_debt_removes_interest(self, default_params):
        """When b=0, interest expense is zero."""
        z_prime = torch.tensor([1.0])
        k_prime = torch.tensor([10.0])
        r_tilde = torch.tensor([0.1])  # any rate
        b_prime = torch.tensor([0.0])
        result = taxable_income(z_prime, k_prime, r_tilde, b_prime, default_params)
        # Should equal operating - depreciation
        expected = 10.0 ** default_params.alpha - default_params.delta * 10.0
        assert abs(result.item() - expected) < 1e-4


# ════════════════════════════════════════════════════════════════
# realized_net_worth
# ════════════════════════════════════════════════════════════════

class TestRealizedNetWorth:
    """
    Realized net worth at start of t+1:
        w' = (1-delta)*k + profit - tax - (1+r_tilde)*b
    """

    def test_hand_calculation(self, default_params):
        """
        With k=10, b=5, z=1, r_tilde=0.05, alpha=0.627, delta=0.15:
        income = 10^0.627 - 1.5 - 0.25 = 2.488 (positive, use tau_c_plus=0.40)
        tax    = 0.40 * 2.488 = 0.995
        w'     = 0.85*10 + 10^0.627 - 0.995 - 1.05*5
               = 8.5 + 4.238 - 0.995 - 5.25 = 6.493
        """
        k_prime = torch.tensor([10.0])
        b_prime = torch.tensor([5.0])
        z_prime = torch.tensor([1.0])
        r_tilde = torch.tensor([0.05])
        result = realized_net_worth(k_prime, b_prime, z_prime, r_tilde, default_params)

        # Hand calculation
        alpha = default_params.alpha
        delta = default_params.delta
        tau = default_params.tau_c_plus
        profit = 1.0 * 10.0 ** alpha
        income = profit - delta * 10.0 - 0.05 * 5.0
        tc = tau * income  # positive income case
        expected = (1.0 - delta) * 10.0 + profit - tc - (1.0 + 0.05) * 5.0

        assert abs(result.item() - expected) < 1e-4

    def test_monotonic_in_z(self, default_params):
        """Higher productivity z' should yield higher realized net worth."""
        k_prime = torch.tensor([10.0, 10.0])
        b_prime = torch.tensor([5.0, 5.0])
        z_prime = torch.tensor([0.5, 2.0])  # low, high productivity
        r_tilde = torch.tensor([0.05, 0.05])
        w = realized_net_worth(k_prime, b_prime, z_prime, r_tilde, default_params)
        assert w[1].item() > w[0].item(), "Higher z should yield higher w'"

    def test_higher_debt_lowers_net_worth(self, default_params):
        """Holding other things equal, more debt → lower realized net worth."""
        k_prime = torch.tensor([10.0, 10.0])
        b_prime = torch.tensor([2.0, 20.0])  # low, high debt
        z_prime = torch.tensor([1.0, 1.0])
        r_tilde = torch.tensor([0.05, 0.05])
        w = realized_net_worth(k_prime, b_prime, z_prime, r_tilde, default_params)
        assert w[1].item() < w[0].item()


# ════════════════════════════════════════════════════════════════
# bankruptcy_recovery
# ════════════════════════════════════════════════════════════════

class TestBankruptcyRecovery:
    """
    Recovery for lenders in default:
        R = (1-xi)*(1-delta)*k + profit - tax_no_interest - w_underbar
    """

    def test_higher_xi_lowers_recovery(self):
        """Bigger bankruptcy cost means smaller recovery."""
        p_low_xi = ModelParams(xi=0.05)
        p_high_xi = ModelParams(xi=0.30)
        k = torch.tensor([10.0])
        z = torch.tensor([1.0])
        w_ub = torch.tensor([-5.0])

        r_low = bankruptcy_recovery(k, z, w_ub, p_low_xi).item()
        r_high = bankruptcy_recovery(k, z, w_ub, p_high_xi).item()
        assert r_low > r_high


# ════════════════════════════════════════════════════════════════
# period_payoff
# ════════════════════════════════════════════════════════════════

class TestPeriodPayoff:
    """
    Period payoff = dist_amount - dist_tax - issue_amount - equity_cost
    where net_financing_need = k' - w_tilde - b'
    """

    def test_distribution_case(self, default_params):
        """
        If k' < w + b (net_financing_need < 0), firm distributes |gap|.
        Payoff = distribution - dist_tax.
        """
        # w=100, k'=10, b'=0 → net_financing = -90 → distribute 90
        w_tilde = torch.tensor([100.0])
        k_prime = torch.tensor([10.0])
        b_prime = torch.tensor([0.0])
        pay = period_payoff(w_tilde, k_prime, b_prime, default_params).item()
        assert pay > 0, "Distribution should yield positive payoff"
        assert pay < 90.0, "Payoff should be less than gross distribution (taxed)"

    def test_issuance_case(self, default_params):
        """
        If k' > w + b (net_financing_need > 0), firm issues.
        Payoff = -(issuance + equity_cost), which is strictly negative.
        """
        # w=0, k'=10, b'=0 → net_financing = 10 → issue 10
        w_tilde = torch.tensor([0.0])
        k_prime = torch.tensor([10.0])
        b_prime = torch.tensor([0.0])
        pay = period_payoff(w_tilde, k_prime, b_prime, default_params).item()
        assert pay < 0, "Issuance should yield negative payoff"
        # Should be less than -10 (raw issuance plus cost)
        assert pay < -10.0

    def test_break_even_case(self, default_params):
        """
        If k' exactly equals w + b, neither distributes nor issues.
        Payoff should be ~0.
        """
        w_tilde = torch.tensor([10.0])
        b_prime = torch.tensor([5.0])
        k_prime = torch.tensor([15.0])  # exactly matches w + b
        pay = period_payoff(w_tilde, k_prime, b_prime, default_params).item()
        assert abs(pay) < 1e-6
