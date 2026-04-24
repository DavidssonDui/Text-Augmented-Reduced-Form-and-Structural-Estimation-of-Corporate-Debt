"""
Minimal stub for sampling module.

The actual sampling.py was not available when tests were written.
These stubs allow import-time success for solver modules; code paths
that call these functions are not exercised by the test suite.
"""
import torch

def sample_states(training_cfg, model_params, batch_size, device='cpu'):
    """Stub: uniform sampling over state space bounds."""
    w = torch.rand(batch_size, device=device) * (
        training_cfg.w_tilde_max - training_cfg.w_tilde_min
    ) + training_cfg.w_tilde_min
    z = torch.exp(torch.randn(batch_size, device=device) *
                   model_params.sigma_eps / (1 - model_params.rho**2)**0.5)
    sigma_s = 1.0
    s = torch.randn(batch_size, device=device) * sigma_s
    return torch.stack([w, z, s], dim=-1)


def sample_next_shocks_conditional(z, s, params, num_samples=32):
    """Stub: conditional sampling for Bellman expectation."""
    batch = z.shape[0]
    device = z.device
    ln_z = torch.log(torch.clamp(z, min=1e-6))
    ln_z_next = (params.rho * ln_z.unsqueeze(-1) +
                 params.sigma_eps * torch.randn(batch, num_samples, device=device))
    z_next = torch.exp(ln_z_next)
    if params.lambda_text > 0:
        sigma_sp = (1 - params.lambda_text) * params.sigma_eta
        eta = torch.randn(batch, num_samples, device=device) * sigma_sp
        s_next = params.lambda_text * ln_z_next + eta
    else:
        s_next = torch.randn(batch, num_samples, device=device) * params.sigma_eta
    importance_weights = torch.ones_like(z_next)
    return z_next, s_next, importance_weights
