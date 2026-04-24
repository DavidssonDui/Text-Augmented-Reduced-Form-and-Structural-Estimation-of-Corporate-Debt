# Text-Augmented Structural Estimation of Corporate Default

**Replication package for Davidsson (2026)**

The paper extends Hennessy and
Whited (2007) with a text-informed bond pricing channel and documents
that FinBERT-based 10-K sentiment predicts corporate default while
dictionary-based (Loughran-McDonald) measures do not.

## What's in here

```
replication_package/
├── README.md ← you are here
├── REPRODUCING.md ← step-by-step reproduction recipe
│
├── paper/ ← the paper itself
│ ├── paper.tex ← single-file LaTeX source
│ ├── paper.pdf ← compiled paper (24 pages)
│ ├── table[1-9]_*.tex ← 9 result tables (booktabs)
│ ├── figure[1-2]_*.pdf ← 2 figures (quintile chart, sample over time)
│ └── line_search_comparison.pdf ← SMM identification diagnostic figure
│
├── reduced_form/ ← reduced-form regression analysis
│ ├── src/ ← regression utils, signal construction, LaTeX formatter
│ ├── scripts/ ← 9 table scripts + 2 figure scripts + run_all.py
│ ├── tests/ ← 70 unit tests
│ ├── output/tables/ ← .tex (paper-ready) + .txt (console-readable)
│ ├── output/figures/ ← .pdf + .png
│ ├── pytest.ini
│ └── requirements.txt
│
├── deqn_solver/ ← DEQN solver + SMM infrastructure
│ ├── src/
│ │ ├── config.py ← model, network, training configs
│ │ ├── primitives_smooth.py ← profit, taxes, equity cost (Option 1)
│ │ ├── networks.py ← MLP, ValueNet, PolicyNet, DefaultNet
│ │ ├── solver_v6b.py ← DEQN solver
│ │ ├── sim_moments.py ← 22-moment computation
│ │ ├── run_smm.py ← Nelder-Mead SMM optimizer
│ │ └── sampling_stub.py ← stub for sampling (real version not in package)
│ ├── tests/ ← 148 unit tests
│ └── pytest.ini
│
├── nlp_pipeline/ ← 10-K text processing
│ ├── 01_get_filing_urls.py ← EDGAR query
│ ├── 02_extract_text.py ← Item 1A and Item 7 extraction
│ ├── 03_lm_scores.py ← Loughran-McDonald dictionary scoring
│ ├── 04_finbert_scores.py ← FinBERT [CLS] embedding extraction
│   ├── 05_build_text_signal.py ← LM (PCA) and FinBERT (supervised) signals
│   └── lm_dictionary.py       ← LM word lists
│
├── panel_construction/        ← Compustat-CRSP panel build
│   └── build_panel.py
│
└── data/raw_placeholder/      ← (you supply: panel_smm_ready.csv, lm_scores.csv)
```

## Three contributions

The paper makes three contributions, each backed by a separate
component of this replication package.

### 1. Empirical (reduced-form)

**Result.** FinBERT 10-K sentiment predicts default in a panel of
63,449 US firm-years (2006–2022, 125 default events). The point
estimate is +0.0037 (p<0.01) per standard deviation increase in
FinBERT signal, with full controls and industry × year fixed effects.
The analogous LM signal coefficient is essentially zero. The top
FinBERT quintile concentrates 86% of subsequent default events.

**Where to find it.** Tables 1–9 in `reduced_form/output/tables/`
are produced by `reduced_form/scripts/table[1-9]_*.py`, each
runnable independently. Run `python scripts/run_all.py` from the
`reduced_form/` directory to regenerate all of them.

### 2. Theoretical (model)

**Result.** A text-augmented Hennessy-Whited (2007) model in which
lenders update beliefs about next-period productivity using a public
text signal `s_t`, with informativeness governed by a parameter
`λ_text ∈ [0, 1]`. The model nests HW07 (`λ_text = 0` recovers the
original).

**Where to find it.** Section 6 of `paper/paper.tex` describes the
model. The DEQN solver source lives in `deqn_solver/src/`.

### 3. Methodological (negative SMM)

**Result.** Three diagnostic procedures (full Nelder-Mead SMM,
multi-start with varied initial values, 1D line search at two
simulation scales) agree that `λ_text` is not identified from the
22-moment specification standard in the dynamic corporate finance
literature. The text-only loss is essentially flat across
`λ_text ∈ [0, 1]`, and this flatness is deterministic rather than a
Monte Carlo artifact (correlation between baseline and 8x-larger
simulation loss vectors > 0.99).

**Where to find it.** Section 8 of the paper. The line search figure
is `paper/line_search_comparison.pdf`.

## Quality assurance: 218 unit tests

The package includes test suites covering both the reduced-form and
DEQN+SMM modules.

```
reduced_form/tests/   →  70 tests, run in ~5 seconds
deqn_solver/tests/    → 148 tests, run in ~4 seconds
                       ────────
                        218 tests total, all passing
```

To run:
```bash
cd reduced_form && python -m pytest
cd ../deqn_solver && python -m pytest
```

Three real bugs were caught during testing and fixed:
1. `prepare_sample` returned literal `'nan'` strings in `sic1`
   when `sic` was missing (now uses sentinel-int conversion).
2. Synthetic test fixture had complex-valued column from
   `k_prime ** 0.6` with negative inputs (now clamps `k ≥ 1`).
3. Merton DD computation had a units mismatch between dollar-denominated
   market cap and million-denominated debt (now normalizes internally).

## Reproducibility statement

All scripts are deterministic given fixed inputs. Random seeds are
fixed at 42 wherever randomness enters (PyTorch, NumPy, scikit-learn).
The reduced-form regressions are deterministic by construction
(closed-form OLS / cluster-robust SE). The DEQN solver and SMM
optimizer use seeded Monte Carlo and are reproducible up to PyTorch
backend differences (CUDA vs MPS vs CPU).

## Data acquisition

The full pipeline requires three external data sources:

1. **Compustat & CRSP** (WRDS subscription required) — for the
   firm-year panel. Used by `panel_construction/build_panel.py`.
2. **SEC EDGAR** (free) — for 10-K filings. Used by
   `nlp_pipeline/01_get_filing_urls.py` and `02_extract_text.py`
   via the `edgartools` package.
3. **LoPucki BRD** (free academic registration) — for large
   bankruptcy events. Imported into `panel_construction/build_panel.py`.

The processed inputs to the reduced-form analysis (`panel_smm_ready.csv`,
`lm_scores.csv`) are not redistributable due to Compustat licensing.
The reduced-form scripts expect these in `data/`. See `REPRODUCING.md`
for the full data acquisition workflow.

## Citation

```
@misc{Davidsson2026TextDefault,
  author = {Davidsson, Dui},
  title = {Text-Augmented Structural Estimation of Corporate Default},
  year = {2026},
  note = {University of British Columbia},
}
```

## License

MIT license for code; CC-BY-4.0 for the paper. Compustat data is not
redistributable.

## Contact

Open an issue or contact the author with reproducibility questions.
