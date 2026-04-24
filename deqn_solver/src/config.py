"""
config.py
=========
Configuration for the DEQN solver of the augmented HW07 model.

All model parameters and network/training hyperparameters live here,
so experiments are fully described by a single Config object.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class ModelParams:
    """
    Structural parameters of the augmented Hennessy-Whited (2007) model.

    Defaults follow HW07 Table I full-sample estimates. The text signal
    parameter λ_text defaults to 0 (pure HW07 baseline).
    """
    # ── Calibrated parameters (HW07 Section III.A) ──
    r: float = 0.025           # Risk-free rate
    delta: float = 0.15        # Depreciation rate
    tau_c_plus: float = 0.40   # Corporate tax rate, positive income
    tau_c_minus: float = 0.20  # Corporate tax rate, negative income (refund)
    tau_i: float = 0.29        # Personal tax rate on interest
    tau_d_bar: float = 0.12    # Asymptotic distribution tax rate

    # ── Estimated HW07 parameters (Table I, full sample) ──
    alpha: float = 0.627       # Returns to scale in production
    rho: float = 0.684         # Productivity persistence
    sigma_eps: float = 0.118   # Productivity innovation std
    lambda_0: float = 0.598    # Fixed cost of equity issuance
    lambda_1: float = 0.091    # Linear cost of equity issuance
    lambda_2: float = 0.0004   # Quadratic cost of equity issuance
    xi: float = 0.104          # Bankruptcy cost fraction
    phi: float = 0.732         # Distribution tax convexity

    # ── Augmented model parameters ──
    lambda_text: float = 0.0   # Text signal informativeness (0 = HW07)
    sigma_eta: float = 1.0     # Signal noise std (normalized to 1)

    @property
    def beta(self) -> float:
        """After-tax discount factor 1/(1+r(1-τ_i))."""
        return 1.0 / (1.0 + self.r * (1.0 - self.tau_i))


@dataclass
class NetworkConfig:
    """
    Architecture of the neural networks used to approximate value and
    policy functions.

    We use separate networks for:
    - V(w̃, z, s): value function (1 output)
    - Policy head outputting (k', b') simultaneously (2 outputs)

    Each network is a simple feedforward MLP.
    """
    # Hidden layer sizes (number of neurons per hidden layer)
    hidden_sizes: Tuple[int, ...] = (64, 64, 64)

    # Activation: "tanh" is common in DEQN since value functions are
    # typically smooth. "relu" also works but can produce kinks.
    activation: str = "tanh"

    # Number of input dimensions to the network = state dimension
    # State = (w_tilde, z, s), so 3 inputs
    state_dim: int = 3

    # Output dimensions
    value_output_dim: int = 1  # V(state)
    policy_output_dim: int = 2  # (k', b')


@dataclass
class TrainingConfig:
    """
    Training hyperparameters for Bellman residual minimization.

    The algorithm alternates:
    1. Sample a batch of states from some distribution over (w̃, z, s).
    2. Compute the Bellman residual at each sampled state.
    3. Minimize the squared residual via gradient descent.
    """
    # Optimization
    learning_rate: float = 1e-3
    batch_size: int = 512      # Number of states sampled per gradient step
    num_epochs: int = 50000    # Total gradient steps
    log_every: int = 500        # Print diagnostics every N epochs
    checkpoint_every: int = 2000  # Save weights every N epochs

    # State sampling
    # During training, we sample states from a distribution that covers
    # the relevant state space. Bounds should encompass the region where
    # firms actually operate in equilibrium.
    w_tilde_min: float = -50.0   # Lower sampling bound for net worth
    w_tilde_max: float = 100.0   # Upper sampling bound for net worth
    # z is sampled from its unconditional distribution (log-normal stationary)
    # s is sampled from its unconditional distribution (normal with variance
    #    determined by λ_text)

    # Monte Carlo integration
    # The Bellman equation involves E[V(w̃', z', s')]. We estimate this
    # expectation by Monte Carlo over next-period shocks.
    num_mc_samples: int = 32   # Shocks per state for the expectation

    # Loss weighting
    # We train the network to satisfy the Bellman equation AND boundary
    # conditions (e.g., V >= 0 at the default boundary). Loss weights
    # control the relative emphasis.
    bellman_weight: float = 1.0
    default_weight: float = 0.1   # Penalty on V for defaulting states


@dataclass
class Config:
    """Top-level config bundling everything."""
    model: ModelParams = field(default_factory=ModelParams)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Device: "mps" for Apple Silicon, "cuda" for NVIDIA, "cpu" fallback
    device: str = "auto"  # Auto-detect

    # Random seed for reproducibility
    seed: int = 42

    # Output directory for checkpoints and diagnostics
    output_dir: str = "./checkpoints"

    def __repr__(self) -> str:
        """Compact summary."""
        return (
            f"Config(\n"
            f"  Model: α={self.model.alpha}, ξ={self.model.xi}, "
            f"λ_text={self.model.lambda_text}\n"
            f"  Net: hidden={self.network.hidden_sizes}, act={self.network.activation}\n"
            f"  Train: lr={self.training.learning_rate}, "
            f"batch={self.training.batch_size}, "
            f"epochs={self.training.num_epochs}\n"
            f"  Device: {self.device}\n"
            f")"
        )


def default_config() -> Config:
    """Return a standard default configuration."""
    return Config()


def hw07_baseline_config() -> Config:
    """HW07 baseline (no text augmentation)."""
    cfg = default_config()
    cfg.model.lambda_text = 0.0
    return cfg


def augmented_config(lambda_text: float = 0.3) -> Config:
    """Augmented model with text signal."""
    cfg = default_config()
    cfg.model.lambda_text = lambda_text
    return cfg
