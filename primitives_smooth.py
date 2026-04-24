"""
primitives_smooth.py
====================
MODIFIED primitives for Option 1: drop the fixed equity issuance cost.

This is a modification of the original primitives.py. Two changes:
  1. equity_cost has NO fixed component — only linear + quadratic
  2. No sigmoid smoothing of the indicator — not needed

Economic interpretation: "We abstract from fixed flotation costs to
focus on marginal costs of external equity." This is a defensible
modeling choice; many HW07-successor papers make it.

The rest of the primitives are unchanged from primitives.py.
"""

import torch
from .config import ModelParams


def profit_fn(z: torch.Tensor, k: torch.Tensor, alpha: float) -> torch.Tensor:
    """Operating profit z·k^α (unchanged)."""
    k_safe = torch.clamp(k, min=1e-6)
    return z * (k_safe ** alpha)


def corporate_tax(income: torch.Tensor, params: ModelParams) -> torch.Tensor:
    """Asymmetric corporate tax (unchanged)."""
    pos_tax = params.tau_c_plus * torch.clamp(income, min=0.0)
    neg_tax = params.tau_c_minus * torch.clamp(income, max=0.0)
    return pos_tax + neg_tax


def distribution_tax(X: torch.Tensor, params: ModelParams) -> torch.Tensor:
    """Distribution tax (unchanged)."""
    X_pos = torch.clamp(X, min=0.0)
    phi = params.phi
    tau_d_bar = params.tau_d_bar
    return tau_d_bar * X_pos - (tau_d_bar / phi) * (1.0 - torch.exp(-phi * X_pos))


def equity_cost(x: torch.Tensor, params: ModelParams) -> torch.Tensor:
    """
    MODIFIED: Smooth equity issuance cost — NO fixed component.

        Λ(x) = λ_1 · x + λ_2 · x²    for x > 0
        Λ(x) = 0                      for x ≤ 0

    The previous version had λ_0 = 0.598 as a fixed cost per issuance,
    which created a step discontinuity at x=0. This made the Bellman
    equation non-smooth in a way that broke DEQN training.

    Removing λ_0 makes the cost function C^1 (continuous gradient).
    The marginal deterrent to small issuances is now purely the λ_1
    linear cost (9.1% of issued amount).

    Note: params.lambda_0 is ignored in this version regardless of its value.
    """
    # Only positive issuance has cost; clamp to avoid negative costs
    x_pos = torch.clamp(x, min=0.0)
    # NO FIXED COST — only linear and quadratic
    return params.lambda_1 * x_pos + params.lambda_2 * x_pos ** 2


def taxable_income(z_prime, k_prime, r_tilde, b_prime, params):
    """Taxable income (unchanged)."""
    operating = profit_fn(z_prime, k_prime, params.alpha)
    depreciation = params.delta * k_prime
    interest_expense = r_tilde * b_prime
    return operating - depreciation - interest_expense


def realized_net_worth(k_prime, b_prime, z_prime, r_tilde, params):
    """Realized net worth (unchanged)."""
    income = taxable_income(z_prime, k_prime, r_tilde, b_prime, params)
    tc = corporate_tax(income, params)
    return (
        (1.0 - params.delta) * k_prime
        + profit_fn(z_prime, k_prime, params.alpha)
        - tc
        - (1.0 + r_tilde) * b_prime
    )


def bankruptcy_recovery(k_prime, z_prime, w_underbar, params):
    """Bankruptcy recovery (unchanged)."""
    operating = profit_fn(z_prime, k_prime, params.alpha)
    income_no_int = operating - params.delta * k_prime
    tc_no_int = corporate_tax(income_no_int, params)
    return (
        (1.0 - params.xi) * (1.0 - params.delta) * k_prime
        + operating
        - tc_no_int
        - w_underbar
    )


def period_payoff(w_tilde, k_prime, b_prime, params):
    """
    Period payoff using the MODIFIED equity_cost (no fixed cost).

    Otherwise identical to original.
    """
    net_financing_need = k_prime - w_tilde - b_prime
    dist_amount = torch.clamp(-net_financing_need, min=0.0)
    issue_amount = torch.clamp(net_financing_need, min=0.0)
    return (
        dist_amount - distribution_tax(dist_amount, params)
        - issue_amount - equity_cost(issue_amount, params)
    )
