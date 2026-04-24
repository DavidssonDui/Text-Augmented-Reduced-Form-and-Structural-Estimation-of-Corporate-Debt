"""
03_lm_scores.py
Compute Loughran-McDonald dictionary scores for all extracted 10-K text.

For each firm-year, scores Item 1A and Item 7 separately and combined.
Produces six category fractions per filing:
  negative, positive, uncertainty, litigious, strong_modal, weak_modal

Input:  data/extracted_text/*.json
Output: output/lm_scores.csv
"""

import pandas as pd
import json
import os
import glob
from lm_dictionary import LMDictionary
from config import EXTRACTED_TEXT_DIR, LM_SCORES_CSV, LM_DICTIONARY_CSV

os.chdir("/Users/computerboi/Downloads/nlp_pipeline")

def main():
    # Load the LM dictionary
    print("Loading Loughran-McDonald dictionary...")
    lm = LMDictionary(LM_DICTIONARY_CSV)

    # Find all extracted text files
    text_files = sorted(glob.glob(os.path.join(EXTRACTED_TEXT_DIR, "*.json")))
    print(f"\nFound {len(text_files)} extracted text files")

    # Score each filing
    results = []
    for i, fpath in enumerate(text_files):
        if (i + 1) % 500 == 0:
            print(f"  Scoring {i+1}/{len(text_files)}")

        with open(fpath, 'r') as f:
            doc = json.load(f)

        gvkey = doc['gvkey']
        cik = doc['cik']
        fyear = doc['fyear']

        # ── Score Item 1A ──
        item_1a_text = doc.get('item_1a', None)
        if item_1a_text and len(item_1a_text) > 100:
            scores_1a = lm.score(item_1a_text)
            scores_1a = {f'item1a_{k}': v for k, v in scores_1a.items()}
        else:
            scores_1a = {}

        # ── Score Item 7 ──
        item_7_text = doc.get('item_7', None)
        if item_7_text and len(item_7_text) > 100:
            scores_7 = lm.score(item_7_text)
            scores_7 = {f'item7_{k}': v for k, v in scores_7.items()}
        else:
            scores_7 = {}

        # ── Score Combined (Item 1A + Item 7) ──
        combined_text = ""
        if item_1a_text:
            combined_text += item_1a_text + " "
        if item_7_text:
            combined_text += item_7_text

        if len(combined_text) > 100:
            scores_combined = lm.score(combined_text)
            scores_combined = {f'combined_{k}': v for k, v in scores_combined.items()}
        else:
            scores_combined = {}

        # Assemble row
        row = {
            'gvkey': gvkey,
            'cik': cik,
            'fyear': fyear,
            **scores_1a,
            **scores_7,
            **scores_combined,
        }
        results.append(row)

    # Build output DataFrame
    df = pd.DataFrame(results)
    os.makedirs("output", exist_ok=True)
    df.to_csv(LM_SCORES_CSV, index=False)

    # Summary statistics
    print("\n" + "="*60)
    print("LM DICTIONARY SCORING SUMMARY")
    print("="*60)
    print(f"Total firm-years scored: {len(df)}")

    # Print summary for combined scores
    score_cols = [c for c in df.columns if c.startswith('combined_') and c.endswith('_fraction')]
    if score_cols:
        print(f"\nCombined text score summary (Item 1A + Item 7):")
        print("-"*50)
        for col in score_cols:
            cat = col.replace('combined_', '').replace('_fraction', '')
            series = df[col].dropna()
            if len(series) > 0:
                print(f"  {cat:15s}: mean={series.mean():.4f}, "
                      f"std={series.std():.4f}, "
                      f"min={series.min():.4f}, "
                      f"max={series.max():.4f}, "
                      f"N={len(series)}")

    print(f"\nOutput saved to {LM_SCORES_CSV}")


if __name__ == "__main__":
    main()
