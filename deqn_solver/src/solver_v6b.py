"""
solver_v6b.py
=============
v6 + ONE targeted fix for high-wealth hoarding.

v6 produced a defensible solver with: borrowing at low w̃, positive default
rates, and proper boundary condition V(w̲) ≈ 0. Its known weakness was at
high w̃ (say w̃>30) where V flattened because the firm piled up cash
rather than distributing.

v6b adds ONE loss term — the payout anchor — which discourages equity
issuance when w̃ > 30. Nothing else changes. This addresses the
cosmetic high-w̃ issue without touching the economically important
borrowing behavior at low/moderate wealth, which v7's three-change
approach inadvertently eliminated.

Tradeoff explicit:
  v6:   borrowing OK, high-w̃ weird
  v6b:  borrowing OK (expected), high-w̃ fixed
  v7:   high-w̃ clean, but no borrowing anywhere (not useful for our thesis)

For the text signal channel to work, we need firms that borrow. v6b is
the minimal change from v6 that keeps borrowing intact.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict

from .config import Config, ModelParams
from .networks import ValueNet, PolicyNet, DefaultNet, StateNormalizer, count_parameters
from .primitives_smooth import (
    realized_net_worth,
    bankruptcy_recovery,
    period_payoff,
)
from .sampling import sample_states, sample_next_shocks_conditional


def auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


# ────────────────────────────────────────────────────────────────
# SMOOTH MECHANICS (kept from v5; they're still useful for gradient flow)
# ────────────────────────────────────────────────────────────────

def smooth_max(a, b, sharpness=10.0):
    stable_max = torch.maximum(a, b).detach()
    return stable_max + (1.0 / sharpness) * torch.log(
        torch.exp(sharpness * (a - stable_max)) + torch.exp(sharpness * (b - stable_max))
    )


def smooth_default_indicator(w_realized, w_underbar, sharpness=5.0):
    return torch.sigmoid(sharpness * (w_underbar - w_realized))


# ────────────────────────────────────────────────────────────────
# BOND PRICING (same as v5 but operates with DefaultNet-provided w̲)
# ────────────────────────────────────────────────────────────────

def solve_bond_yield_analytic(
    k_prime, b_prime, z_prime, importance_weights, w_underbar, params, max_iter=5,
):
    num_samples = z_prime.shape[1]
    is_borrowing = b_prime > 1e-6
    r_tilde = torch.full_like(b_prime, params.r)
    beta = params.beta
    one_minus_taui = 1.0 - params.tau_i

    k_prime_bc = k_prime.unsqueeze(-1).expand(-1, num_samples)
    b_prime_bc = b_prime.unsqueeze(-1).expand(-1, num_samples)

    for _ in range(max_iter):
        r_tilde_bc = r_tilde.unsqueeze(-1).expand(-1, num_samples)
        w_realized = realized_net_worth(k_prime_bc, b_prime_bc, z_prime, r_tilde_bc, params)
        default_prob = smooth_default_indicator(w_realized, w_underbar, sharpness=10.0)
        recovery = bankruptcy_recovery(k_prime_bc, z_prime, w_underbar, params)
        expected_recovery = (importance_weights * default_prob * recovery).mean(dim=-1)
        p_nodefault = (importance_weights * (1.0 - default_prob)).mean(dim=-1)

        p_nodefault_safe = torch.clamp(p_nodefault, min=1e-4)
        b_prime_safe = torch.clamp(b_prime, min=1e-4)

        numerator = b_prime / beta - expected_recovery
        denominator = b_prime_safe * p_nodefault_safe
        gross_rate = numerator / denominator
        r_tilde_new = (gross_rate - 1.0) / one_minus_taui
        r_tilde_new = torch.clamp(r_tilde_new, min=params.r, max=2.0)

        r_tilde = torch.where(is_borrowing, r_tilde_new, torch.full_like(r_tilde, params.r))
    return r_tilde


# ────────────────────────────────────────────────────────────────
# PRE-TRAINING TARGETS (unchanged from v5)
# ────────────────────────────────────────────────────────────────

def analytical_value_target(state, params):
    w_tilde, z, _ = state[:, 0], state[:, 1], state[:, 2]
    return w_tilde + 5.0 * (z - 1.0) - 2.0


def analytical_policy_target(state, params):
    w_tilde, z, _ = state[:, 0], state[:, 1], state[:, 2]
    k_target = torch.clamp(0.5 * (w_tilde + 10.0), min=5.0, max=50.0)
    b_target = 0.2 * k_target
    return torch.stack([k_target, b_target], dim=-1)


def analytical_default_target(z, params):
    """
    Pre-training target for DefaultNet.

    A reasonable initial guess: w̲(z) ≈ -5·z — firms with higher
    productivity tolerate more negative net worth before defaulting.
    This pre-trains DefaultNet to a sensible starting point.
    """
    return -5.0 * z


def pretrain_networks(value_net, policy_net, default_net, cfg, num_epochs, lr, device, verbose):
    if verbose:
        print("\n" + "="*60)
        print(" STAGE 1: Pre-training (analytical targets)")
        print("="*60)

    optimizer_v = optim.Adam(value_net.parameters(), lr=lr)
    optimizer_p = optim.Adam(policy_net.parameters(), lr=lr)
    optimizer_d = optim.Adam(default_net.parameters(), lr=lr)
    mse = nn.MSELoss()

    for epoch in range(num_epochs):
        state = sample_states(cfg.training, cfg.model, cfg.training.batch_size, device=device)
        z = state[:, 1]

        # Value net target
        v_target = analytical_value_target(state, cfg.model)
        v_pred = value_net(state)
        loss_v = mse(v_pred, v_target)
        optimizer_v.zero_grad(); loss_v.backward(); optimizer_v.step()

        # Policy net target
        policy_target = analytical_policy_target(state, cfg.model)
        policy_pred = policy_net(state)
        loss_p = mse(policy_pred, policy_target)
        optimizer_p.zero_grad(); loss_p.backward(); optimizer_p.step()

        # Default net target
        d_target = analytical_default_target(z, cfg.model)
        d_pred = default_net(z)
        loss_d = mse(d_pred, d_target)
        optimizer_d.zero_grad(); loss_d.backward(); optimizer_d.step()

        if verbose and epoch % 100 == 0:
            print(f"  Epoch {epoch:5d} | V_loss={loss_v.item():.4e} | "
                  f"P_loss={loss_p.item():.4e} | D_loss={loss_d.item():.4e}")

    if verbose:
        # Report what DefaultNet now outputs
        with torch.no_grad():
            test_z = torch.tensor([0.5, 1.0, 1.5], device=device)
            test_w = default_net(test_z)
            print(f"  Post-pretrain w̲ estimates: {[f'{x:+.2f}' for x in test_w.tolist()]}")
        print("  Pre-training complete.\n")


# ────────────────────────────────────────────────────────────────
# BELLMAN RESIDUAL + BOUNDARY LOSS
# ────────────────────────────────────────────────────────────────

def compute_bellman_residual_v6(
    state, value_net, policy_net, default_net,
    params, num_mc_samples,
):
    """
    Bellman residual using DefaultNet-provided w̲.

    Key difference from v5: w̲(z) comes from default_net, not bisection.
    This means:
    - w̲ has real gradients (default_net can be trained)
    - w̲ is structurally constrained to be negative
    - Stage 3 transition doesn't exist — we're in full-dynamic mode from
      epoch 1, but DefaultNet gets a proper training signal
    """
    w_tilde, z, s = state[:, 0], state[:, 1], state[:, 2]

    policy = policy_net(state)
    k_prime, b_prime = policy[:, 0], policy[:, 1]

    z_prime, s_prime, importance_weights = sample_next_shocks_conditional(
        z, s, params, num_samples=num_mc_samples
    )

    # w̲ from DefaultNet (depends only on z)
    # Shape: (batch, num_mc_samples)
    w_underbar_zp = default_net(z_prime.flatten()).view_as(z_prime)

    r_tilde = solve_bond_yield_analytic(
        k_prime, b_prime, z_prime, importance_weights,
        w_underbar_zp, params, max_iter=5
    )

    k_prime_bc = k_prime.unsqueeze(-1).expand(-1, num_mc_samples)
    b_prime_bc = b_prime.unsqueeze(-1).expand(-1, num_mc_samples)
    r_tilde_bc = r_tilde.unsqueeze(-1).expand(-1, num_mc_samples)
    w_realized = realized_net_worth(k_prime_bc, b_prime_bc, z_prime, r_tilde_bc, params)

    w_tilde_prime = smooth_max(w_underbar_zp, w_realized, sharpness=5.0)

    next_state = torch.stack([w_tilde_prime, z_prime, s_prime], dim=-1)
    V_next = value_net(next_state.view(-1, 3)).view_as(z_prime)

    default_soft = smooth_default_indicator(w_realized, w_underbar_zp, sharpness=5.0)
    V_next_with_default = (1.0 - default_soft) * V_next + default_soft * 0.0

    expected_V = (importance_weights * V_next_with_default).mean(dim=-1)
    pp = period_payoff(w_tilde, k_prime, b_prime, params)

    v_target = pp + params.beta * expected_V
    v_pred = value_net(state)
    residual = v_pred - v_target

    return {
        "residual": residual,
        "v_pred": v_pred,
        "v_target": v_target,
        "k_prime": k_prime,
        "b_prime": b_prime,
        "r_tilde": r_tilde,
        "default_frac": default_soft.mean(dim=-1),
        "r_tilde_mean": r_tilde.mean(),
    }


def compute_boundary_loss(state, value_net, default_net):
    """
    THE KEY NEW LOSS.

    Enforces V(w̲(z), z, s) ≈ 0 for each state in the batch.

    Without this loss, the solver has no reason to prefer one
    equilibrium over another — V can drift and w̲ from bisection
    will track it. With this loss, DefaultNet is actively trained
    to match the real economic boundary where V = 0.
    """
    z = state[:, 1]
    s = state[:, 2]

    # Get DefaultNet's current w̲(z)
    w_underbar = default_net(z)

    # Evaluate V at the boundary
    boundary_state = torch.stack([w_underbar, z, s], dim=-1)
    V_at_boundary = value_net(boundary_state)

    return (V_at_boundary ** 2).mean()


def compute_monotonicity_penalty(state, value_net, params):
    batch = state.shape[0]
    device = state.device
    z_perturbation = 0.1 * torch.abs(torch.randn(batch, device=device))
    state_perturbed = state.clone()
    state_perturbed[:, 1] = state[:, 1] + z_perturbation
    v_orig = value_net(state)
    v_perturbed = value_net(state_perturbed)
    violation = torch.clamp(v_orig - v_perturbed, min=0.0)
    return (violation ** 2).mean()


def compute_payout_anchor(state, policy_net, high_w_threshold=30.0):
    """
    v6b targeted fix: at high w̃, discourage equity issuance.

    At high net worth the firm should distribute, not issue new equity.
    Penalize positive (k' - w̃ - b') at w̃ > threshold. This is the ONLY
    change from v6 — it only bites at high w̃, so it doesn't affect
    borrowing behavior at moderate/low wealth where the research
    contribution matters.

    Unlike v7, we do NOT add: slope penalty, reweighted sampling,
    reduced b_scale. Those changes interacted to eliminate borrowing
    everywhere. Here, we only touch the high-w̃ corner.
    """
    w = state[:, 0]
    policy = policy_net(state)
    k_prime, b_prime = policy[:, 0], policy[:, 1]
    net_fin_need = k_prime - w - b_prime
    issuance = torch.clamp(net_fin_need, min=0.0)
    high_w_mask = (w > high_w_threshold).float()
    penalized = (issuance ** 2) * high_w_mask
    return penalized.sum() / (high_w_mask.sum() + 1e-8)


# ────────────────────────────────────────────────────────────────
# MAIN SOLVER
# ────────────────────────────────────────────────────────────────

def solve_v6b(
    cfg: Config,
    pretrain_epochs: int = 500,
    joint_epochs: int = 2500,
    boundary_weight: float = 1.0,
    monotonicity_weight: float = 0.1,
    payout_weight: float = 0.1,
    high_w_threshold: float = 30.0,
    verbose: bool = True,
) -> Dict[str, object]:
    """
    v6b: v6 + targeted payout anchor at high w̃.

    The only difference from v6: added loss term penalizing equity
    issuance when w̃ > high_w_threshold. Nothing else changes. This
    addresses the high-w̃ hoarding without disrupting borrowing
    behavior at moderate/low w̃.

    Stage 1: Pre-train all three networks to analytical targets.
    Stage 2: Joint training minimizing:
               bellman_weight · Bellman_residual²
             + boundary_weight · V(w̲(z), z, s)²
             + monotonicity_weight · V-nondecreasing-in-z penalty
             + payout_weight · payout_anchor (NEW in v6b)
    """
    device = cfg.device
    if device == "auto":
        device = auto_device()
    if verbose:
        print(f"Using device: {device}")

    torch.manual_seed(cfg.seed)

    # Normalizer
    sigma_z = cfg.model.sigma_eps / (1.0 - cfg.model.rho ** 2) ** 0.5
    z_mean = float(torch.exp(torch.tensor(0.5 * sigma_z ** 2)))
    var_s = (cfg.model.lambda_text ** 2) * (sigma_z ** 2) + \
            ((1.0 - cfg.model.lambda_text) ** 2) * (cfg.model.sigma_eta ** 2)
    sigma_s = var_s ** 0.5
    means = torch.tensor([
        (cfg.training.w_tilde_min + cfg.training.w_tilde_max) / 2.0,
        z_mean, 0.0,
    ])
    stds = torch.tensor([
        (cfg.training.w_tilde_max - cfg.training.w_tilde_min) / 4.0,
        max(sigma_z * z_mean, 0.1),
        max(sigma_s, 0.5),
    ])
    normalizer = StateNormalizer(means, stds).to(device)

    # All three networks
    value_net = ValueNet(
        hidden_sizes=list(cfg.network.hidden_sizes),
        activation=cfg.network.activation,
        state_dim=cfg.network.state_dim,
        normalizer=normalizer,
    ).to(device)

    policy_net = PolicyNet(
        hidden_sizes=list(cfg.network.hidden_sizes),
        activation=cfg.network.activation,
        state_dim=cfg.network.state_dim,
        k_scale=100.0, b_scale=30.0,
        normalizer=normalizer,
    ).to(device)

    default_net = DefaultNet(
        hidden_sizes=(32, 32),
        activation="tanh",
        z_scale=max(sigma_z * z_mean, 0.1),
    ).to(device)

    if verbose:
        print(f"ValueNet params:   {count_parameters(value_net):,}")
        print(f"PolicyNet params:  {count_parameters(policy_net):,}")
        print(f"DefaultNet params: {count_parameters(default_net):,}")

    # ────────────────────────────────────────
    # STAGE 1: Pre-training
    # ────────────────────────────────────────
    pretrain_networks(
        value_net, policy_net, default_net, cfg,
        num_epochs=pretrain_epochs, lr=cfg.training.learning_rate,
        device=device, verbose=verbose,
    )

    # ────────────────────────────────────────
    # STAGE 2: Joint Bellman training
    # ────────────────────────────────────────
    if verbose:
        print("="*60)
        print(" STAGE 2: Joint training (Bellman + boundary + monotonicity)")
        print("="*60)

    all_params = (
        list(value_net.parameters())
        + list(policy_net.parameters())
        + list(default_net.parameters())
    )
    optimizer = optim.Adam(all_params, lr=cfg.training.learning_rate)

    loss_history = []
    start_time = time.time()

    for epoch in range(joint_epochs):
        optimizer.zero_grad()
        state = sample_states(cfg.training, cfg.model, cfg.training.batch_size, device=device)

        out = compute_bellman_residual_v6(
            state, value_net, policy_net, default_net,
            cfg.model, cfg.training.num_mc_samples,
        )
        bellman_loss = (out["residual"] ** 2).mean()
        boundary_loss = compute_boundary_loss(state, value_net, default_net)
        mono_loss = compute_monotonicity_penalty(state, value_net, cfg.model)
        payout_loss = compute_payout_anchor(
            state, policy_net, high_w_threshold=high_w_threshold
        )

        total_loss = (
            bellman_loss
            + boundary_weight * boundary_loss
            + monotonicity_weight * mono_loss
            + payout_weight * payout_loss
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=10.0)
        optimizer.step()

        loss_history.append({
            "epoch": epoch,
            "total": total_loss.item(),
            "bellman": bellman_loss.item(),
            "boundary": boundary_loss.item(),
            "mono": mono_loss.item(),
        })

        if verbose and epoch % 100 == 0:
            elapsed = time.time() - start_time
            with torch.no_grad():
                test_z = torch.tensor([0.5, 1.0, 1.5], device=device)
                w_test = default_net(test_z)
            print(
                f"  {epoch:5d} | "
                f"total={total_loss.item():.3e} "
                f"(bell={bellman_loss.item():.2e}, "
                f"bdry={boundary_loss.item():.2e}) | "
                f"V={out['v_pred'].mean().item():+.2f} | "
                f"def={out['default_frac'].mean().item():.3f} | "
                f"r̃={out['r_tilde_mean'].item():.3f} | "
                f"w̲={[f'{x:+.1f}' for x in w_test.tolist()]} | "
                f"k̄={out['k_prime'].mean().item():.1f}, "
                f"b̄={out['b_prime'].mean().item():+.1f} | "
                f"{elapsed:.0f}s"
            )

    total_time = time.time() - start_time
    if verbose:
        print(f"\nTraining complete in {total_time:.1f}s")

    return {
        "value_net": value_net,
        "policy_net": policy_net,
        "default_net": default_net,
        "normalizer": normalizer,
        "loss_history": loss_history,
        "total_time": total_time,
    }
