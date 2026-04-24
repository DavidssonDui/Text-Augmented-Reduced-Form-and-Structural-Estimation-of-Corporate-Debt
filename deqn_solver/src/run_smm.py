"""
run_smm.py
==========
SMM layer 3-4: objective function + Nelder-Mead optimizer.

Estimates 9 parameters of the HW07-augmented model:
    (α, ρ, σ_ε, λ_1, λ_2, ξ, φ, λ_text, σ_η)

Method:
  For each candidate θ:
    1. Configure v6b with θ
    2. Train v6b from scratch (~30s)
    3. Simulate panel (500 firms × 100 periods)
    4. Compute 22 simulated moments
    5. Compute weighted distance from data moments

  Nelder-Mead searches over θ to minimize the distance.

Weighting:
  Diagonal W with weights 1/m_data², so each moment contributes
  equally in relative (squared) terms.

Initialization:
  HW07 Table I values (full sample) for the 8 shared parameters,
  λ_text = 0.3 for the text signal.

Bounds (enforced via log-transformation for some parameters):
  α ∈ (0.3, 0.9), ρ ∈ (0.3, 0.95), σ_ε ∈ (0.05, 0.30),
  λ_1 ∈ (0.01, 0.30), λ_2 ∈ (0.0001, 0.01),
  ξ ∈ (0.05, 0.30), φ ∈ (0.1, 0.9),
  λ_text ∈ (0, 1), σ_η ∈ (0.3, 2.0)

Logging:
  Every iteration's (θ, loss, m_sim) saved to smm_iterations.jsonl
  Final result saved to smm_result.json

Usage:
    python scripts/run_smm.py
"""

import sys
import os
import json
import time
import numpy as np
import torch
from dataclasses import asdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import hw07_baseline_config
from src.solver_v6b import solve_v6b
from src.sim_moments import compute_sim_moments, MOMENT_KEYS, N_MOMENTS


# ────────────────────────────────────────────────────────────────
# Parameter management
# ────────────────────────────────────────────────────────────────

# Parameters we estimate, in canonical order
PARAM_NAMES = ['alpha', 'rho', 'sigma_eps', 'lambda_1', 'lambda_2',
               'xi', 'phi', 'lambda_text', 'sigma_eta']
N_PARAMS = len(PARAM_NAMES)

# Bounds as (lo, hi) for each parameter
PARAM_BOUNDS = {
    'alpha': (0.3, 0.9),
    'rho': (0.3, 0.95),
    'sigma_eps': (0.05, 0.30),
    'lambda_1': (0.01, 0.30),
    'lambda_2': (0.0001, 0.01),
    'xi': (0.05, 0.30),
    'phi': (0.1, 0.9),
    'lambda_text': (0.0, 1.0),
    'sigma_eta': (0.3, 2.0),
}

# HW07 Table I (full sample) starting values, + λ_text
INIT_PARAMS = {
    'alpha': 0.627,
    'rho': 0.684,
    'sigma_eps': 0.118,
    'lambda_1': 0.091,
    'lambda_2': 0.0004,
    'xi': 0.104,
    'phi': 0.732,
    'lambda_text': 0.3,
    'sigma_eta': 1.0,
}


def clip_params(theta_arr):
    """Clip a parameter array to stay within bounds."""
    out = np.zeros_like(theta_arr)
    for i, name in enumerate(PARAM_NAMES):
        lo, hi = PARAM_BOUNDS[name]
        out[i] = np.clip(theta_arr[i], lo, hi)
    return out


def apply_params_to_config(cfg, theta_arr):
    """Write parameter values onto a config object."""
    for i, name in enumerate(PARAM_NAMES):
        setattr(cfg.model, name, float(theta_arr[i]))
    return cfg


# ────────────────────────────────────────────────────────────────
# Loss function
# ────────────────────────────────────────────────────────────────

def compute_loss(m_sim, m_data, W_diag):
    """
    Diagonal weighted distance:
        loss = (m_sim - m_data)' W (m_sim - m_data)
    with W_diag[i] = 1 / m_data[i]² (guarded against zero).
    """
    diff = m_sim - m_data
    return float(np.sum((diff ** 2) * W_diag))


def build_weights(m_data, floor=1e-4):
    """Diagonal W = 1/m_data² with floor to guard against tiny values."""
    safe_data = np.maximum(np.abs(m_data), floor)
    return 1.0 / (safe_data ** 2)


# ────────────────────────────────────────────────────────────────
# Single evaluation: train, simulate, compute loss
# ────────────────────────────────────────────────────────────────

def evaluate_theta(theta_arr, base_cfg, m_data, W_diag, device='cpu',
                    pretrain_epochs=500, joint_epochs=1500):
    """
    For a parameter vector, train v6b, simulate, and return loss.

    pretrain/joint epochs reduced from baseline (500/2500) to (500/1500)
    to keep iteration time around 25-30s. This trades a bit of solver
    precision for SMM tractability.
    """
    theta_arr = clip_params(theta_arr)
    cfg = apply_params_to_config(base_cfg, theta_arr)

    t0 = time.time()

    try:
        result = solve_v6b(
            cfg,
            pretrain_epochs=pretrain_epochs,
            joint_epochs=joint_epochs,
            boundary_weight=1.0,
            monotonicity_weight=0.1,
            payout_weight=0.1,
            high_w_threshold=30.0,
            verbose=False,
        )
    except Exception as e:
        # Solver exploded at this θ; return high loss
        print(f"    Solver crashed: {e}")
        return 1e10, np.zeros(N_MOMENTS)

    t_train = time.time() - t0

    try:
        m_sim, _ = compute_sim_moments(
            result['value_net'], result['policy_net'], result['default_net'],
            cfg.model, n_firms=500, t_burn=50, t_sim=100, device=device,
        )
    except Exception as e:
        print(f"    Simulation crashed: {e}")
        return 1e10, np.zeros(N_MOMENTS)

    t_sim = time.time() - t0 - t_train

    # Guard against non-finite moments
    if not np.all(np.isfinite(m_sim)):
        bad = np.where(~np.isfinite(m_sim))[0].tolist()
        print(f"    Non-finite moments at indices {bad} — returning high loss")
        return 1e10, m_sim

    loss = compute_loss(m_sim, m_data, W_diag)

    return loss, m_sim


# ────────────────────────────────────────────────────────────────
# Optimizer
# ────────────────────────────────────────────────────────────────

def run_nelder_mead(
    base_cfg, m_data, W_diag,
    max_iter=100, xatol=1e-3, fatol=1e-4,
    device='cpu', log_path='smm_iterations.jsonl',
):
    """
    Nelder-Mead optimization over 9 parameters.

    scipy.optimize.minimize with method='Nelder-Mead' doesn't natively
    support bounds, so we clip inside evaluate_theta and also in the
    closure below.
    """
    from scipy.optimize import minimize

    theta_init = np.array([INIT_PARAMS[n] for n in PARAM_NAMES])

    # Open log file
    log_f = open(log_path, 'w')

    iteration_state = {'n': 0, 'best_loss': np.inf, 'best_theta': theta_init.copy()}

    def objective(theta_arr):
        iteration_state['n'] += 1
        t0 = time.time()
        loss, m_sim = evaluate_theta(
            theta_arr, base_cfg, m_data, W_diag, device=device,
        )
        elapsed = time.time() - t0

        # Log
        record = {
            'iter': iteration_state['n'],
            'theta': {n: float(t) for n, t in zip(PARAM_NAMES, theta_arr)},
            'loss': loss,
            'elapsed_s': elapsed,
        }
        log_f.write(json.dumps(record) + '\n')
        log_f.flush()

        # Console
        if loss < iteration_state['best_loss']:
            iteration_state['best_loss'] = loss
            iteration_state['best_theta'] = theta_arr.copy()
            marker = '*'
        else:
            marker = ' '

        print(f"  iter {iteration_state['n']:3d}{marker} "
              f"loss={loss:.4e} "
              f"α={theta_arr[0]:.3f} ρ={theta_arr[1]:.3f} ξ={theta_arr[5]:.3f} "
              f"λ_text={theta_arr[7]:.3f} "
              f"({elapsed:.0f}s)")

        return loss

    print()
    print("Starting Nelder-Mead optimization...")
    print(f"  Initial loss:")
    initial_loss, _ = evaluate_theta(theta_init, base_cfg, m_data, W_diag, device=device)
    print(f"  at HW07 defaults: {initial_loss:.4e}")
    print()

    res = minimize(
        objective, theta_init,
        method='Nelder-Mead',
        options={
            'maxiter': max_iter,
            'xatol': xatol,
            'fatol': fatol,
            'adaptive': True,  # adaptive Nelder-Mead for high dims
        },
    )

    log_f.close()

    return res, iteration_state


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────

def main():
    # Load data moments
    moments_path = 'data_moments_finbert.json'
    if not os.path.exists(moments_path):
        print(f"ERROR: {moments_path} not found. Run compute_data_moments.py first.")
        return

    with open(moments_path) as f:
        data_moments_dict = json.load(f)

    # Build ordered array matching MOMENT_KEYS
    m_data = np.array([data_moments_dict.get(k, 0.0) for k in MOMENT_KEYS])
    W_diag = build_weights(m_data)

    print("=" * 70)
    print(" SMM estimation — 9 parameters, 22 moments")
    print("=" * 70)
    print()
    print("Data moments:")
    for k, v, w in zip(MOMENT_KEYS, m_data, W_diag):
        print(f"  {k:<45s} {v:>+10.4f}  weight={w:.2e}")
    print()

    # Build base config
    cfg = hw07_baseline_config()
    cfg.model.lambda_0 = 0.0
    cfg.network.hidden_sizes = (32, 32)
    cfg.training.batch_size = 256
    cfg.training.w_tilde_min = -20.0
    cfg.training.w_tilde_max = 50.0

    if torch.backends.mps.is_available():
        cfg.device = 'mps'
    elif torch.cuda.is_available():
        cfg.device = 'cuda'
    else:
        cfg.device = 'cpu'

    print(f"Device: {cfg.device}")
    print(f"Max iterations: 100")
    print(f"Expected time: 45-75 min")
    print()

    # ────────────────────────────────────────────────────────
    # Pre-SMM diagnostic: evaluate loss at HW07 defaults, show
    # per-moment contributions. If the initial loss is dominated
    # by one or two moments, the SMM estimates will be distorted.
    # ────────────────────────────────────────────────────────
    print("=" * 70)
    print(" PRE-SMM DIAGNOSTIC: Loss at HW07 defaults")
    print("=" * 70)
    print(" Training v6b at HW07 parameters + computing sim moments...")
    print()

    theta_init_arr = np.array([INIT_PARAMS[n] for n in PARAM_NAMES])
    initial_loss, m_sim_init = evaluate_theta(
        theta_init_arr, cfg, m_data, W_diag, device=cfg.device,
    )

    # Per-moment contribution to total loss
    contrib = ((m_sim_init - m_data) ** 2) * W_diag

    print(f" Initial total loss: {initial_loss:.4e}")
    print()
    print(f" {'Moment':<45s} {'Sim':>10s} {'Data':>10s} {'Weight':>10s} {'Contrib':>12s}")
    print(" " + "-" * 88)
    # Sort by contribution descending to see which moments dominate
    idx_sorted = np.argsort(-contrib)
    for i in idx_sorted:
        k = MOMENT_KEYS[i]
        pct = 100 * contrib[i] / max(initial_loss, 1e-12)
        marker = " <-- DOMINATES" if pct > 30 else ""
        print(f" {k:<45s} {m_sim_init[i]:>+10.4f} {m_data[i]:>+10.4f} "
              f"{W_diag[i]:>10.2e} {contrib[i]:>12.2e} ({pct:4.1f}%){marker}")

    print()
    print(" If loss is dominated by 1-2 moments, SMM estimates will be distorted.")
    print(" Consider aborting (Ctrl-C) to drop problematic moments before running.")
    print()

    # Pause briefly so user can read and abort if needed
    import time as _time
    print(" Proceeding to SMM in 15 seconds... (Ctrl-C to abort)")
    for sec in range(15, 0, -5):
        print(f"   ...{sec}s")
        _time.sleep(5)
    print()

    # Run SMM
    res, state = run_nelder_mead(
        cfg, m_data, W_diag,
        max_iter=100,
        device=cfg.device,
        log_path='smm_iterations.jsonl',
    )

    # Report
    print()
    print("=" * 70)
    print(" SMM RESULT")
    print("=" * 70)
    print(f"  Converged: {res.success}")
    print(f"  Iterations: {res.nit}")
    print(f"  Function evals: {res.nfev}")
    print(f"  Final loss: {res.fun:.4e}")
    print()
    print(f"  ESTIMATED PARAMETERS:")
    print(f"  {'Parameter':<15s} {'Initial':>10s} {'Estimate':>10s} {'Bound':>15s}")
    print("  " + "-" * 55)
    for i, name in enumerate(PARAM_NAMES):
        init_val = INIT_PARAMS[name]
        est_val = res.x[i]
        lo, hi = PARAM_BOUNDS[name]
        bound_str = f"[{lo:.2f}, {hi:.2f}]"
        print(f"  {name:<15s} {init_val:>10.4f} {est_val:>10.4f} {bound_str:>15s}")
    print()

    # Save result
    result_out = {
        'converged': bool(res.success),
        'n_iter': int(res.nit),
        'n_feval': int(res.nfev),
        'final_loss': float(res.fun),
        'theta': {n: float(t) for n, t in zip(PARAM_NAMES, res.x)},
        'theta_init': INIT_PARAMS,
        'moments_path': moments_path,
    }
    with open('smm_result.json', 'w') as f:
        json.dump(result_out, f, indent=2)
    print(f"  ✓ Result saved to smm_result.json")
    print(f"  ✓ Iteration log: smm_iterations.jsonl")


if __name__ == "__main__":
    main()
