"""
05_build_text_signal.py
Construct the final text signal s for the structural model.

Option A (Baseline): First principal component of LM negative, uncertainty,
    and weak modal fractions, standardized within each cross-section year.

Option B (Robustness): FinBERT embedding projected onto the direction
    maximally predictive of next-year default, standardized within each year.

Input:  output/lm_scores.csv
        output/finbert_embeddings.pkl
        data/sample_firms.csv (with default indicators for FinBERT projection)
Output: output/text_signal.csv (final panel with both signals)
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from config import LM_SCORES_CSV, FINBERT_SCORES_CSV, SAMPLE_FIRMS_CSV

os.chdir("/Users/computerboi/Downloads/nlp_pipeline")

def build_lm_signal(lm_scores):
    """
    Option A: Build text signal from Loughran-McDonald dictionary scores.

    Takes the first principal component of negative, uncertainty, and
    weak_modal fractions from the combined (Item 1A + Item 7) text.
    Standardizes within each cross-section year to remove time trends
    in disclosure style.
    """
    print("\n" + "="*60)
    print("OPTION A: Loughran-McDonald Text Signal")
    print("="*60)

    # Select the three category fractions for the combined text
    score_cols = [
        'combined_negative_fraction',
        'combined_uncertainty_fraction',
        'combined_weak_modal_fraction',
    ]

    # Check columns exist
    missing = [c for c in score_cols if c not in lm_scores.columns]
    if missing:
        print(f"WARNING: Missing columns: {missing}")
        return None

    # Drop rows with missing scores
    df = lm_scores[['gvkey', 'cik', 'fyear'] + score_cols].dropna()
    print(f"Firm-years with complete LM scores: {len(df)}")

    # Standardize within each cross-section year
    # This removes time trends in disclosure style (e.g., 10-Ks got longer
    # and more boilerplate over time, which would inflate raw word counts)
    df_standardized = df.copy()
    for col in score_cols:
        df_standardized[col + '_std'] = df.groupby('fyear')[col].transform(
            lambda x: (x - x.mean()) / x.std()
        )

    std_cols = [c + '_std' for c in score_cols]

    # Compute first principal component
    pca = PCA(n_components=1)
    X = df_standardized[std_cols].values
    pc1 = pca.fit_transform(X).flatten()

    # Report PCA loadings
    loadings = pca.components_[0]
    var_explained = pca.explained_variance_ratio_[0]
    print(f"\nPCA Results:")
    print(f"  Variance explained by PC1: {var_explained*100:.1f}%")
    print(f"  Loadings:")
    for col, loading in zip(score_cols, loadings):
        cat = col.replace('combined_', '').replace('_fraction', '')
        print(f"    {cat:15s}: {loading:+.4f}")

    # Standardize PC1 within each year (final text signal)
    df_standardized['s_lm_raw'] = pc1
    df_standardized['s_lm'] = df_standardized.groupby('fyear')['s_lm_raw'].transform(
        lambda x: (x - x.mean()) / x.std()
    )

    # Summary
    print(f"\nText signal s_lm summary:")
    print(f"  Mean:  {df_standardized['s_lm'].mean():.4f} (should be ~0)")
    print(f"  Std:   {df_standardized['s_lm'].std():.4f} (should be ~1)")
    print(f"  Min:   {df_standardized['s_lm'].min():.4f}")
    print(f"  Max:   {df_standardized['s_lm'].max():.4f}")

    return df_standardized[['gvkey', 'cik', 'fyear', 's_lm']].copy()


def build_finbert_signal(lm_scores, sample_firms):
    """
    Option B: Build text signal from FinBERT embeddings.

    Projects the 768-dim document embedding onto the direction that
    maximally predicts next-year default. Uses logistic regression
    with a train/test split to avoid look-ahead bias.
    """
    print("\n" + "="*60)
    print("OPTION B: FinBERT Text Signal")
    print("="*60)

    # Load embeddings
    embeddings_path = "/Users/computerboi/Downloads/nlp_pipeline/output/finbert_embeddings.pkl"
    if not os.path.exists(embeddings_path):
        print("FinBERT embeddings not found. Run 04_finbert_scores.py first.")
        return None

    with open(embeddings_path, 'rb') as f:
        embeddings_dict = pickle.load(f)
    print(f"Loaded {len(embeddings_dict)} FinBERT embeddings")

    # Build embedding matrix
    keys = []
    X_list = []
    for key, emb in embeddings_dict.items():
        parts = key.split('_')
        gvkey = parts[0]
        fyear = int(parts[1])
        keys.append({'gvkey': gvkey, 'fyear': fyear})
        X_list.append(emb)

    emb_df = pd.DataFrame(keys)
    X = np.vstack(X_list)  # Shape: (n_firms, 768)
    print(f"Embedding matrix shape: {X.shape}")

    # Merge with default indicators from sample
    # The sample_firms CSV should have a 'default_next_year' column
    # (1 if the firm defaults in year fyear+1, 0 otherwise)
    if 'default_next_year' not in sample_firms.columns:
        print("WARNING: 'default_next_year' column not found in sample_firms.csv")
        print("You need to construct this from CRSP delistings + LoPucki before running this step.")
        print("Skipping FinBERT signal construction.")
        return None

    emb_df = emb_df.merge(
        sample_firms[['gvkey', 'fyear', 'default_next_year']],
        on=['gvkey', 'fyear'],
        how='left'
    )

    # Drop rows without default indicator
    valid = emb_df['default_next_year'].notna()
    emb_df_valid = emb_df[valid].reset_index(drop=True)
    X_valid = X[valid.values]
    y = emb_df_valid['default_next_year'].values.astype(int)

    print(f"Firm-years with default indicator: {len(emb_df_valid)}")
    print(f"Default rate: {y.mean()*100:.2f}%")

    # ── Train/test split by time ──
    # Use first half of sample for training, second half for projection
    # This avoids look-ahead bias
    median_year = emb_df_valid['fyear'].median()
    train_mask = emb_df_valid['fyear'] <= median_year
    test_mask = emb_df_valid['fyear'] > median_year

    X_train, y_train = X_valid[train_mask.values], y[train_mask.values]
    X_test = X_valid[test_mask.values]

    print(f"Training period: fyear <= {int(median_year)} ({train_mask.sum()} obs, {y_train.sum()} defaults)")
    print(f"Test period: fyear > {int(median_year)} ({test_mask.sum()} obs)")

    # ── Fit logistic regression ──
    # The coefficient vector defines the "default risk direction" in embedding space
    # Use L2 regularization to handle the high dimensionality (768 >> n_defaults)
    clf = LogisticRegression(
        penalty='l2',
        C=1.0,           # Regularization strength (lower = more regularization)
        max_iter=1000,
        class_weight='balanced',  # Handle class imbalance (few defaults)
        solver='lbfgs',
        random_state=42,
    )
    clf.fit(X_train, y_train)

    # The projection direction is the coefficient vector
    beta = clf.coef_[0]  # Shape: (768,)
    print(f"\nLogistic regression fitted. Coefficient norm: {np.linalg.norm(beta):.4f}")

    # In-sample AUC
    from sklearn.metrics import roc_auc_score
    train_scores = clf.predict_proba(X_train)[:, 1]
    train_auc = roc_auc_score(y_train, train_scores)
    print(f"Training AUC: {train_auc:.4f}")

    if test_mask.sum() > 0 and y[test_mask.values].sum() > 0:
        test_scores = clf.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y[test_mask.values], test_scores)
        print(f"Test AUC: {test_auc:.4f}")

    # ── Project ALL embeddings onto the default-risk direction ──
    # (not just the valid subset — we want scores for all firm-years)
    projections = X @ beta  # Shape: (n_firms,)

    # Assign back to dataframe
    proj_df = pd.DataFrame(keys)
    proj_df['s_finbert_raw'] = projections

    # Standardize within each year
    proj_df['s_finbert'] = proj_df.groupby('fyear')['s_finbert_raw'].transform(
        lambda x: (x - x.mean()) / x.std()
    )

    print(f"\nFinBERT text signal s_finbert summary:")
    print(f"  Mean:  {proj_df['s_finbert'].mean():.4f}")
    print(f"  Std:   {proj_df['s_finbert'].std():.4f}")

    return proj_df[['gvkey', 'fyear', 's_finbert']].copy()


def main():
    # Load data
    lm_scores = pd.read_csv(LM_SCORES_CSV)
    print(f"Loaded LM scores: {len(lm_scores)} rows")

    try:
        sample_firms = pd.read_csv(SAMPLE_FIRMS_CSV)
    except FileNotFoundError:
        sample_firms = pd.DataFrame()

    # Harmonize key types only AFTER both objects exist
    lm_scores['gvkey'] = lm_scores['gvkey'].astype(str).str.strip()
    lm_scores['fyear'] = pd.to_numeric(lm_scores['fyear'], errors='coerce')

    if not sample_firms.empty:
        sample_firms['gvkey'] = sample_firms['gvkey'].astype(str).str.strip()
        sample_firms['fyear'] = pd.to_numeric(sample_firms['fyear'], errors='coerce')

    # ── Build Option A: LM signal ──
    lm_signal = build_lm_signal(lm_scores)

    # ── Build Option B: FinBERT signal ──
    finbert_signal = build_finbert_signal(lm_scores, sample_firms)

    # ── Merge both signals into final output ──
    if lm_signal is not None:
        output = lm_signal.copy()
    else:
        output = lm_scores[['gvkey', 'cik', 'fyear']].copy()

    output['gvkey'] = output['gvkey'].astype(str).str.strip()
    output['fyear'] = pd.to_numeric(output['fyear'], errors='coerce')

    if finbert_signal is not None:
        finbert_signal['gvkey'] = finbert_signal['gvkey'].astype(str).str.strip()
        finbert_signal['fyear'] = pd.to_numeric(finbert_signal['fyear'], errors='coerce')

        output = output.merge(finbert_signal, on=['gvkey', 'fyear'], how='left')

    # Save
    output.to_csv("output/text_signal.csv", index=False)

    print("\n" + "="*60)
    print("FINAL OUTPUT")
    print("="*60)
    print(f"Saved to: output/text_signal.csv")
    print(f"Rows: {len(output)}")
    print(f"Columns: {list(output.columns)}")

    if 's_lm' in output.columns:
        print(f"\ns_lm (LM dictionary signal):")
        print(f"  Non-missing: {output['s_lm'].notna().sum()}")
        print(f"  This is your BASELINE text signal for the structural model.")

    if 's_finbert' in output.columns:
        print(f"\ns_finbert (FinBERT embedding signal):")
        print(f"  Non-missing: {output['s_finbert'].notna().sum()}")
        print(f"  This is your ROBUSTNESS CHECK text signal.")

    if 's_lm' in output.columns and 's_finbert' in output.columns:
        both_valid = output[['s_lm', 's_finbert']].dropna()
        if len(both_valid) > 0:
            corr = both_valid['s_lm'].corr(both_valid['s_finbert'])
            print(f"\nCorrelation between s_lm and s_finbert: {corr:.4f}")
            print("(High correlation suggests both methods capture similar information)")

    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("1. Merge output/text_signal.csv with your Compustat-CRSP panel on (gvkey, fyear)")
    print("2. The 's_lm' column is the text signal s that enters the structural model")
    print("3. Compute the text-financial cross-moments for SMM estimation")
    print("4. Higher s_lm = more negative/uncertain/hedging language = worse outlook")


if __name__ == "__main__":
    main()
