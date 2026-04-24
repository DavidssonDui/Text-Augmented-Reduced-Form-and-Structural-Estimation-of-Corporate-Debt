"""
networks.py
===========
Neural network architectures for approximating the value function V
and the policy functions (k', b').

Design choices:

1. We use THREE separate networks (not one shared one):
   - ValueNet: V(w̃, z, s) -> scalar
   - PolicyNet: (w̃, z, s) -> (k', b')
   - DefaultNet: (z) -> w̲(z), the default threshold

   Separate networks keep each function's learning well-targeted.

2. Standardized inputs: we normalize (w̃, z, s) to roughly zero mean
   and unit variance before feeding into networks. This dramatically
   speeds up training.

3. Policy network output constraints:
   - k' must be positive → we use softplus to map raw output to [0, ∞)
   - b' can be positive or negative → identity output

4. Tanh activations throughout (smooth, bounded gradients) with
   small initialization so networks start near zero output.

All networks support forward passes on batches of states.
"""

import torch
import torch.nn as nn
from typing import List


class MLP(nn.Module):
    """
    Generic multi-layer perceptron with configurable hidden sizes and
    activation. Used as the backbone for all three networks.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: List[int],
        activation: str = "tanh",
    ):
        super().__init__()

        # Activation function lookup
        if activation == "tanh":
            act = nn.Tanh
        elif activation == "relu":
            act = nn.ReLU
        elif activation == "silu":
            act = nn.SiLU
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build layer list
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(act())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

        # Small initialization so the network starts close to zero output
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StateNormalizer:
    """
    Simple state normalizer. Stores means and stds for each state
    dimension; applies (x - mean) / std during forward pass.

    Values are set by calling .fit(samples) or by hand; they do NOT
    get updated during training (this is intentional: we want the
    network inputs to be on a stable scale).
    """

    def __init__(self, means: torch.Tensor, stds: torch.Tensor):
        self.means = means  # shape (state_dim,)
        self.stds = stds    # shape (state_dim,)

    def to(self, device: str) -> "StateNormalizer":
        """Move normalizer to a device."""
        self.means = self.means.to(device)
        self.stds = self.stds.to(device)
        return self

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply normalization. x has shape (batch, state_dim)."""
        return (x - self.means) / (self.stds + 1e-8)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.normalize(x)


class ValueNet(nn.Module):
    """
    Value function network: V(w̃, z, s) -> R.

    Takes a batch of states of shape (batch, 3), returns values of shape
    (batch, 1) (or (batch,) after .squeeze()).

    The value function in HW07 can be negative (for states near the
    default boundary), so we use an unbounded output (no softplus).
    """

    def __init__(
        self,
        hidden_sizes: List[int],
        activation: str = "tanh",
        state_dim: int = 3,
        normalizer: StateNormalizer = None,
    ):
        super().__init__()
        self.mlp = MLP(state_dim, 1, hidden_sizes, activation)
        self.normalizer = normalizer

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        state: shape (batch, state_dim) — columns are (w_tilde, z, s).
        Returns: shape (batch,)
        """
        if self.normalizer is not None:
            state = self.normalizer(state)
        return self.mlp(state).squeeze(-1)


class PolicyNet(nn.Module):
    """
    Policy network: (w̃, z, s) -> (k', b').

    Design:
    - k' output is passed through softplus to ensure positivity.
    - b' output is identity (can be negative for saving).
    - We scale outputs by k_scale and b_scale set externally, so the
      network's raw output is roughly on a unit scale and the scaling
      happens outside. This makes training more stable.

    The caller provides k_scale and b_scale (typical magnitudes of k'
    and b' in the relevant range).
    """

    def __init__(
        self,
        hidden_sizes: List[int],
        activation: str = "tanh",
        state_dim: int = 3,
        k_scale: float = 100.0,  # typical k' magnitude
        b_scale: float = 30.0,   # typical b' magnitude
        normalizer: StateNormalizer = None,
    ):
        super().__init__()
        self.mlp = MLP(state_dim, 2, hidden_sizes, activation)
        self.k_scale = k_scale
        self.b_scale = b_scale
        self.normalizer = normalizer

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        state: shape (batch, 3)
        Returns: shape (batch, 2) — columns are (k', b').
        """
        if self.normalizer is not None:
            state = self.normalizer(state)
        raw = self.mlp(state)  # shape (batch, 2)

        # k' = softplus(raw_k) * k_scale, ensures k' > 0
        k_prime = torch.nn.functional.softplus(raw[:, 0]) * self.k_scale

        # b' = raw_b * b_scale, can be any sign
        b_prime = raw[:, 1] * self.b_scale

        return torch.stack([k_prime, b_prime], dim=-1)


class DefaultNet(nn.Module):
    """
    Default threshold network: w̲(z).

    The default threshold depends only on productivity z (not on the
    current state w̃ or signal s — these are redundant information at
    the time the default decision is made, since w̲ captures the zero
    of V at the start of the next period).

    We parameterize log(-w̲(z)) since w̲ is always negative in HW07
    (firms only default when net worth has become negative). This
    ensures the output is consistent with theory.

    Wait — actually HW07 shows w̲ is strictly negative, so we'll
    output log(-w̲) and take -exp of it.
    """

    def __init__(
        self,
        hidden_sizes: List[int] = (32, 32),
        activation: str = "tanh",
        z_scale: float = 1.0,  # normalize z before input
    ):
        super().__init__()
        # Smaller network since w̲(z) is a 1D function
        self.mlp = MLP(1, 1, list(hidden_sizes), activation)
        self.z_scale = z_scale

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: shape (batch,) — productivity values (level, not log)
        Returns: shape (batch,) — default threshold w̲(z), strictly < 0
        """
        z_normalized = z.unsqueeze(-1) / self.z_scale  # shape (batch, 1)
        raw = self.mlp(z_normalized).squeeze(-1)
        # Map to negative number via -softplus
        return -torch.nn.functional.softplus(raw)


def count_parameters(net: nn.Module) -> int:
    """Count trainable parameters in a network."""
    return sum(p.numel() for p in net.parameters() if p.requires_grad)
