"""
conftest.py
===========
Shared fixtures for the DEQN solver test suite.
"""
import os
import sys
import numpy as np
import pytest
import torch

# Make package importable from project root
HERE = os.path.dirname(os.path.abspath(__file__))
DEQN_DIR = os.path.dirname(HERE)
ROOT = os.path.dirname(DEQN_DIR)

# Two import strategies:
# 1. Running from project root with deqn_solver as a subpackage
# 2. Running from deqn_solver/ directly (then src/ is the importable path)
sys.path.insert(0, ROOT)        # for deqn_solver.src.X
sys.path.insert(0, DEQN_DIR)    # for src.X (alternative layout)

# If we're being imported from deqn_solver/ (no parent package), alias
# the deqn_solver.src.* namespace to src.* so test files using the
# fully-qualified import still work.
try:
    import deqn_solver.src.config  # noqa: F401
except ModuleNotFoundError:
    # Build the alias on the fly
    import importlib
    import types
    pkg = types.ModuleType('deqn_solver')
    pkg.__path__ = [DEQN_DIR]
    sys.modules['deqn_solver'] = pkg

    src_pkg = importlib.import_module('src')
    sys.modules['deqn_solver.src'] = src_pkg
    for sub in ['config', 'networks', 'primitives_smooth', 'solver_v6b',
                 'sim_moments', 'run_smm', 'sampling']:
        try:
            mod = importlib.import_module(f'src.{sub}')
            sys.modules[f'deqn_solver.src.{sub}'] = mod
        except ImportError:
            pass

from deqn_solver.src.config import ModelParams, Config, hw07_baseline_config
from deqn_solver.src.networks import (
    MLP, StateNormalizer, ValueNet, PolicyNet, DefaultNet, count_parameters
)


@pytest.fixture
def torch_seed():
    """Set torch seed for deterministic tests."""
    torch.manual_seed(42)
    np.random.seed(42)


@pytest.fixture
def default_params(torch_seed):
    """Standard HW07 parameters."""
    return ModelParams()


@pytest.fixture
def state_normalizer():
    """Simple state normalizer with zero mean, unit std."""
    means = torch.tensor([0.0, 1.0, 0.0])
    stds = torch.tensor([1.0, 0.3, 1.0])
    return StateNormalizer(means, stds)


@pytest.fixture
def value_net(torch_seed, state_normalizer):
    """A freshly initialized value network."""
    return ValueNet(hidden_sizes=[16, 16], normalizer=state_normalizer)


@pytest.fixture
def policy_net(torch_seed, state_normalizer):
    """A freshly initialized policy network."""
    return PolicyNet(hidden_sizes=[16, 16], normalizer=state_normalizer,
                      k_scale=50.0, b_scale=20.0)


@pytest.fixture
def default_net(torch_seed):
    """A freshly initialized default network."""
    return DefaultNet(hidden_sizes=[8, 8])


@pytest.fixture
def sample_state(torch_seed):
    """A small batch of valid states (w_tilde, z, s)."""
    return torch.tensor([
        [10.0, 1.0, 0.0],
        [5.0, 0.8, 0.5],
        [20.0, 1.2, -0.3],
        [-5.0, 0.9, 0.1],
    ])
