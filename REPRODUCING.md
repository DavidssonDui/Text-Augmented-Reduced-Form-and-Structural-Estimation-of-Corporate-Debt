# Reproducing the Paper

This document gives the end-to-end recipe for reproducing every table,
figure, and number in the paper from raw data sources.

## Time and resource budget

| Stage | Time | Hardware | Disk |
|---|---|---|---|
| 1. Data acquisition (Compustat/CRSP/EDGAR) | 4–8 hours | any | 5 GB |
| 2. NLP pipeline (Steps 1–5) | 12–24 hours | GPU helpful for FinBERT | 10 GB |
| 3. Panel construction | 5 minutes | any | <1 GB |
| 4. Reduced-form analysis (Tables 1–9, Figures 1–2) | 2 minutes | any | <100 MB |
| 5. DEQN solver training (one solve) | 30 seconds | Apple M-series MPS or CUDA recommended | <100 MB |
| 6. SMM diagnostic (single line search) | 1 hour | same as above | <100 MB |
| 7. Paper compilation | 1 minute | any with pdflatex | <100 MB |

The reduced-form analysis (stages 1+3+4+7) is the fastest path to
reproducing the paper's empirical contribution and can be completed
in under 8 hours given the data.

## Software requirements

- Python 3.9 or newer
- PyTorch 2.0 or newer
- LaTeX distribution with pdflatex (TeX Live, MacTeX, or MiKTeX)

```bash
# Reduced-form analysis
cd reduced_form
pip install -r requirements.txt

# DEQN solver
cd ../deqn_solver
pip install torch numpy pandas scipy
```

## Stage 1: Data acquisition

### Compustat (Annual Fundamentals)

Required variables: `gvkey`, `fyear`, `at`, `dlc`, `dltt`, `ceq`,
`che`, `re`, `oibdp`, `sale`, `ppent`, `ppegt`, `xint`, `dp`, `txt`,
`prstkc`, `sstk`, `dvc`, `dvp`, `tic`, `cik`, `sic`. Filter
`indfmt = 'INDL'`, `consol = 'C'`, `datafmt = 'STD'`,
`fyear` between 2006 and 2022. Save as `compustat_annual.csv`.

### CRSP (Monthly Stock Files + Delisting File)

Required from CRSP MSF: `permno`, `permco`, `date`, `prc`, `shrout`.
Required from CRSP MSEDELIST: `permno`, `dlstdt`, `dlstcd`. Required
from CRSP-Compustat link table (CCM): `gvkey`, `permno`, `linkdt`,
`linkenddt`, `linktype`, `linkprim`. Save as `crsp_msf.csv`,
`crsp_msedelist.csv`, `ccm_link.csv`.

### LoPucki Bankruptcy Research Database

Free academic registration at <https://lopucki.law.ufl.edu/>. Download
all bankruptcies and save as `lopucki_brd.csv`. Required columns:
`Name`, `DateFiled`, `Chapter`, `GvKey` (some entries have GvKey
prepopulated; for those without, manual matching is needed).

### Treasury and credit-spread data

From FRED: 1-year Treasury yield (DGS1), 10-year Treasury yield (DGS10),
and Moody's Baa-Aaa spread (BAA10YM). Annual averages. Save as
`treasury_rates.csv` and `credit_spreads.csv`.

### SEC EDGAR (10-K filings)

Used in stage 2. No advance download needed.

## Stage 2: NLP pipeline

The NLP pipeline produces `lm_scores.csv` and `text_signal.csv` which
feed into both the panel build and the regressions.

```bash
cd nlp_pipeline

# Edit config.py to point to your Compustat-CRSP CIK list
# and your output directories

# Step 1: Get 10-K filing URLs from EDGAR (4-8 hours, rate-limited)
python 01_get_filing_urls.py

# Step 2: Extract Item 1A and Item 7 text (4-8 hours)
python 02_extract_text.py

# Step 3: Apply LM dictionary (1 hour)
python 03_lm_scores.py

# Step 4: Extract FinBERT [CLS] embeddings (4-12 hours; GPU recommended)
python 04_finbert_scores.py

# Step 5: Build LM (PCA) and FinBERT (supervised) signals (5 min)
python 05_build_text_signal.py
```

Output files:
- `lm_scores.csv` — raw LM category fractions per (gvkey, fyear)
- `text_signal.csv` — final standardized signals (s_lm, s_finbert)

## Stage 3: Panel construction

```bash
cd panel_construction

# Edit input file paths in build_panel.py header
python build_panel.py
```

Output: `panel_smm_ready.csv` — the Compustat-CRSP firm-year panel
merged with default events from CRSP delistings + LoPucki, and
text signals from Step 5.

Expected sample composition (from the paper):
- 63,449 firm-years
- 8,590 unique firms
- 125 default events (44 LoPucki + 81 CRSP delisting)
- 47,051 firm-years with non-null text signals (74.2% coverage)

## Stage 4: Reduced-form analysis

Place `panel_smm_ready.csv` and `lm_scores.csv` from Stages 2-3 into
`reduced_form/data/`. Then:

```bash
cd reduced_form

# Run all 9 tables and 2 figures
python scripts/run_all.py
# Or individual:
python scripts/table1_sample_summary.py
python scripts/table2_baseline_finbert.py
# ... through table9
python scripts/figure1_quintile_chart.py
python scripts/figure2_sample_over_time.py
```

Outputs land in `reduced_form/output/tables/` and `output/figures/`.
LaTeX `.tex` files and console-friendly `.txt` files are produced for
each table.

To verify everything works without real data:
```bash
python -m pytest tests/   # 70 tests pass in <5s
```

## Stage 5: DEQN solver (optional, for structural estimation)

The solver requires a `sampling.py` module that is not in this package
(it's environment-specific to your hardware). For testing/development
without it, copy `deqn_solver/src/sampling_stub.py` to `sampling.py`
to allow imports to succeed. The stub will not produce results
matching the paper.

```bash
cd deqn_solver

# Run unit tests (148 tests, ~4s)
python -m pytest tests/

# To actually train a solver (requires real sampling.py):
# python src/run_baseline_v6b.py  # 30-90 seconds
```

## Stage 6: SMM diagnostic (optional)

```bash
cd deqn_solver
# Single 1D line search (1-2 hours; requires real sampling.py)
# python src/line_search_lambda_text.py
```

## Stage 7: Paper compilation

The paper is a single LaTeX file with included tables and figures.

```bash
cd paper

# All required .tex tables and .pdf figures should already be present
# (they are shipped with this package as example outputs)

pdflatex paper.tex
pdflatex paper.tex   # second pass for cross-references
```

Output: `paper.pdf` (24 pages).

To rebuild with your own freshly-computed tables, copy from
`reduced_form/output/tables/` to `paper/` first:

```bash
cp ../reduced_form/output/tables/table*.tex .
cp ../reduced_form/output/figures/figure*.pdf .
pdflatex paper.tex && pdflatex paper.tex
```

## Verification: do my numbers match?

Quick checks against the paper text:

| Metric | Expected value | Source |
|---|---|---|
| Total firm-years | 63,449 | Table 1 |
| Unique firms | 8,590 | Table 1 |
| Default events | 125 | Table 1 |
| FinBERT coefficient (Table 2 col 5) | +0.0037, p<0.01 | Table 2 |
| LM coefficient (Table 3 col 5) | -0.0002, ns | Table 3 |
| Top-quintile default share | 86% | Table 5 |
| Test-set AUC (controls + FinBERT) | 0.92 | Table 8 |
| SMM α estimate | 0.434 | Section 8 of paper |
| SMM λ_text estimate | 0.303 (no movement) | Section 8 of paper |

If any of these are off by more than a rounding amount, something
is wrong with your reproduction. Check:

1. Are you using the same panel filters? (excluding financials and
   utilities; 2006-2022)
2. Did the FinBERT embeddings come out right? (Default-direction
   projection should yield positive correlation with subsequent default)
3. Did the LM signal use the three categories specified in
   `05_build_text_signal.py`? (negative, uncertainty, weak_modal)

## Known gotchas

- **`tsy_1y` is in percent**, not fraction. Confirm in `panel_smm_ready.csv`
  that median is around 0.5 (i.e., 0.5%). The Merton DD calculation in
  `signal_utils.compute_merton_dd` divides by 100 to convert.
- **`mktcap_yearend` is in raw dollars**, not millions. Other Compustat
  variables are in millions. The Merton DD function rescales internally.
- **Forward-looking defaults** are NaN for the last firm-years (no future
  observations). These observations are correctly dropped from regressions
  via `dropna()`.
- **The supervised LM training set is small** (about 19 defaults in
  `fyear ≤ 2013`). This is by design—it matches the FinBERT training
  set size for an apples-to-apples comparison. Coefficients on the
  supervised LM are still ≈0, ruling out the "FinBERT just gets
  supervision" interpretation.

## Contact

Open an issue or contact the author if you encounter problems
reproducing any specific number.
