"""
01_get_filing_urls.py
Find 10-K filing URLs on SEC EDGAR for all firms in the sample.

Input:  data/sample_firms.csv (columns: gvkey, cik, fyear, conm)
Output: data/filing_urls.csv   (columns: gvkey, cik, fyear, filing_date, accession_no, filing_url)
"""

import pandas as pd
import time
import os
from edgar import Company, set_identity
from config import (
    SEC_IDENTITY, SAMPLE_FIRMS_CSV, SEC_RATE_LIMIT_DELAY,
    START_YEAR, END_YEAR
)

os.chdir("/Users/computerboi/Downloads/nlp_pipeline")

def get_10k_filings_for_cik(cik, gvkey, conm):
    """
    Query EDGAR for all 10-K filings by a given CIK.
    Returns a list of dicts with filing metadata.
    """
    results = []
    try:
        company = Company(int(cik))
        filings = company.get_filings(form="10-K")

        for filing in filings:
            # filing.filing_date may be a string or a datetime.date depending on edgartools version
            try:
                fd = filing.filing_date
                if hasattr(fd, 'year'):
                    year = fd.year
                else:
                    year = int(str(fd)[:4])
            except (TypeError, ValueError, AttributeError):
                continue

            if year < START_YEAR or year > END_YEAR:
                continue

            results.append({
                'gvkey': gvkey,
                'cik': int(cik),
                'conm': conm,
                'fyear': year,
                'filing_date': str(filing.filing_date),
                'accession_no': filing.accession_no,
                'form_type': filing.form,
            })

    except Exception as e:
        print(f"  ERROR for CIK {cik} ({conm}): {e}")

    return results


def main():
    # Set SEC identity (required)
    set_identity(SEC_IDENTITY)

    # Load sample firms
    firms = pd.read_csv(SAMPLE_FIRMS_CSV)
    print(f"Loaded {len(firms)} firm-year observations")

    # Get unique CIKs
    unique_firms = firms[['gvkey', 'cik', 'conm']].drop_duplicates(subset='cik')
    unique_firms = unique_firms.dropna(subset=['cik'])
    print(f"Found {len(unique_firms)} unique CIKs to query")

    # Query EDGAR for each firm
    all_filings = []
    for i, (_, row) in enumerate(unique_firms.iterrows()):
        if (i + 1) % 100 == 0:
            print(f"Processing firm {i+1}/{len(unique_firms)}: {row['conm']}")

        filings = get_10k_filings_for_cik(row['cik'], row['gvkey'], row['conm'])
        all_filings.extend(filings)

        time.sleep(SEC_RATE_LIMIT_DELAY)

    # Save results
    df = pd.DataFrame(all_filings)
    os.makedirs("data", exist_ok=True)

    if len(df) == 0:
        print("\n" + "="*60)
        print("ERROR: No filings retrieved")
        print("="*60)
        print("Possible causes:")
        print("  1. SEC_IDENTITY in config.py is still the placeholder")
        print("     -> Open config.py and set your real name and email")
        print("  2. Your sample_firms.csv has no valid CIKs")
        print(f"     -> Check: {len(unique_firms)} unique CIKs were loaded")
        print("  3. edgartools failed to connect to EDGAR")
        print("     -> Test manually with:")
        print("        from edgar import Company, set_identity")
        print("        set_identity('Your Name your@email.com')")
        print("        Company(320193).get_filings(form='10-K')")
        print("  4. None of your firms had 10-K filings in 2006-2022")
        return

    df.to_csv("data/filing_urls.csv", index=False)
    print(f"\nSaved {len(df)} filing records to data/filing_urls.csv")
    print(f"Coverage: {df['fyear'].min()} to {df['fyear'].max()}")
    print(f"Unique firms: {df['cik'].nunique()}")


if __name__ == "__main__":
    main()