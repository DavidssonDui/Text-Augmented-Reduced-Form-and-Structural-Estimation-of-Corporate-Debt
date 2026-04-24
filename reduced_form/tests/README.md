# Test Suite

Unit tests for the reduced-form analysis package. More tests will be added for the DEQN solver and SMM infrastructure.

## Running the tests

From the project root:

```bash
pip install pytest
python -m pytest
```

## Current coverage

- `test_regression_utils.py` — 29 tests covering `prepare_sample`, `run_lpm`,
  `run_logit`, significance stars, coefficient formatting
- `test_signal_utils.py` — 19 tests covering `standardize_within_year`,
  `build_supervised_lm`, `compute_merton_dd`
- `test_latex_utils.py` — 22 tests covering `format_reg_table_latex`,
  `_latex_escape`, `write_latex_table`

Total: **70 tests**, all passing.

## What gets tested

Each function has tests in up to three categories:

1. **Correctness on known cases** — e.g., `run_lpm` on a synthetic
   `y = 2 + 0.5x + noise` dataset should recover coefficients near (2, 0.5).

2. **Invariant / property tests** — e.g., `standardize_within_year`
   output must have mean zero and standard deviation one within each year;
   `compute_merton_dd` must be monotonically decreasing in leverage
   and equity volatility.

3. **Edge cases and error handling** — e.g., `build_supervised_lm` must
   raise when training defaults are too few; `prepare_sample` must
   handle missing SIC codes without crashing.

## Fixtures

`conftest.py` provides shared fixtures:

- `rng` — a seeded numpy RNG for deterministic tests
- `synthetic_panel` — 10 firms × 10 years with a planted default relationship
- `synthetic_lm_scores` — aligned LM score frame matching the real schema
- `small_linear_dataset` — `y = 2 + 0.5x + noise` for hand-calibration
- `hand_ar1_series` — series with known lag-1 autocorrelation

## Markers

Selective running:

- `pytest -m "not slow"` — skip long tests (none currently)
- `pytest -m requires_data` — only tests that load the full panel

## Design notes

- All fixtures are deterministic (seeded RNG).
- Tests never modify fixture inputs — we check with `test_does_not_mutate_input`.
- Synthetic data is small enough that tests run in <10 seconds total.
- Real data files (`panel_smm_ready.csv`, `lm_scores.csv`) are not required
  for tests; all tests use synthetic fixtures.

## Known gap

The DEQN solver (`src/solver_v6b.py`, `src/primitives_smooth.py`) and SMM
infrastructure (`scripts/run_smm.py`, `src/sim_moments.py`) do not yet have
a test suite. These will be added in a follow-up once the source modules are
stable.
