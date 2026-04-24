"""
02_extract_text.py
Extract Item 1A (Risk Factors) and Item 7 (MD&A) from 10-K filings.

Uses edgartools as the primary parser, with regex fallback for filings
that edgartools can't handle.

Input:  data/filing_urls.csv
Output: data/extracted_text/ (one JSON file per firm-year)
        data/extraction_log.csv (success/failure log)
"""

import pandas as pd
import json
import os
import re
import time
import traceback
from edgar import Company, set_identity
from config import (
    SEC_IDENTITY, EXTRACTED_TEXT_DIR, SEC_RATE_LIMIT_DELAY,
)

os.chdir("/Users/computerboi/Downloads/nlp_pipeline")

def extract_with_edgartools(filing):
    """
    Primary extraction method using edgartools' built-in parser.
    Returns dict with item_1a and item_7 text, or None on failure.
    """
    try:
        tenk = filing.obj()
        item_1a = str(tenk["Item 1A"]) if tenk["Item 1A"] else None
        item_7 = str(tenk["Item 7"]) if tenk["Item 7"] else None

        # Basic validation: sections should have substantial text
        if item_1a and len(item_1a) < 500:
            item_1a = None
        if item_7 and len(item_7) < 500:
            item_7 = None

        if item_1a or item_7:
            return {'item_1a': item_1a, 'item_7': item_7, 'method': 'edgartools'}
    except Exception:
        pass
    return None


def extract_with_regex(html_text):
    """
    Fallback extraction using regex on raw HTML/text.
    Less reliable but catches cases edgartools misses.
    """
    from bs4 import BeautifulSoup

    # Strip HTML tags
    soup = BeautifulSoup(html_text, 'html.parser')
    text = soup.get_text(separator='\n')

    # Normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)

    item_1a = None
    item_7 = None

    # ── Extract Item 1A ──
    # Look for "Item 1A" followed by "Risk Factors"
    pattern_1a_start = re.compile(
        r'(?:item\s*1a[\.\s\-\u2014\u2013]*risk\s*factors)',
        re.IGNORECASE
    )
    pattern_1a_end = re.compile(
        r'(?:item\s*1b|item\s*2[\.\s\-])',
        re.IGNORECASE
    )

    matches_1a = list(pattern_1a_start.finditer(text))
    if matches_1a:
        # Take the last match (skip table of contents entries)
        # by finding the match followed by the most text
        best_start = None
        best_length = 0
        for match in matches_1a:
            start_pos = match.end()
            end_match = pattern_1a_end.search(text, start_pos)
            end_pos = end_match.start() if end_match else len(text)
            length = end_pos - start_pos
            if length > best_length:
                best_length = length
                best_start = start_pos
                best_end = end_pos

        if best_start and best_length > 500:
            item_1a = text[best_start:best_end].strip()

    # ── Extract Item 7 ──
    # Look for "Item 7" followed by "Management's Discussion"
    pattern_7_start = re.compile(
        r'(?:item\s*7[\.\s\-\u2014\u2013]*management[\u2019\']?s?\s*discussion)',
        re.IGNORECASE
    )
    pattern_7_end = re.compile(
        r'(?:item\s*7a|item\s*8[\.\s\-])',
        re.IGNORECASE
    )

    matches_7 = list(pattern_7_start.finditer(text))
    if matches_7:
        best_start = None
        best_length = 0
        for match in matches_7:
            start_pos = match.end()
            end_match = pattern_7_end.search(text, start_pos)
            end_pos = end_match.start() if end_match else len(text)
            length = end_pos - start_pos
            if length > best_length:
                best_length = length
                best_start = start_pos
                best_end = end_pos

        if best_start and best_length > 500:
            item_7 = text[best_start:best_end].strip()

    if item_1a or item_7:
        return {'item_1a': item_1a, 'item_7': item_7, 'method': 'regex'}
    return None


def clean_text(text):
    """Clean extracted text for NLP processing."""
    if text is None:
        return None

    # Remove residual HTML entities
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)
    text = re.sub(r'&#\d+;', ' ', text)

    # Remove table-like content (rows of numbers)
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        # Skip lines that are mostly numbers/symbols (likely table rows)
        alphanumeric = re.sub(r'[^a-zA-Z]', '', line)
        if len(line) > 0 and len(alphanumeric) / max(len(line), 1) < 0.3:
            continue
        cleaned_lines.append(line)
    text = '\n'.join(cleaned_lines)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def process_one_filing(row):
    """
    Process a single filing: try edgartools first, then regex fallback.
    Returns a result dict.
    """
    cik = int(row['cik'])
    fyear = int(row['fyear'])
    gvkey = row['gvkey']
    accession = row.get('accession_no', '')

    result = {
        'gvkey': gvkey,
        'cik': cik,
        'fyear': fyear,
        'accession_no': accession,
        'item_1a_extracted': False,
        'item_7_extracted': False,
        'item_1a_length': 0,
        'item_7_length': 0,
        'method': None,
        'error': None,
    }

    try:
        # Try edgartools first
        company = Company(cik)
        filings = company.get_filings(form="10-K")

        target_filing = None
        for f in filings:
            try:
                if f.accession_no == accession:
                    target_filing = f
                    break
            except:
                continue

        # If we can't find by accession, try by year
        if target_filing is None:
            for f in filings:
                try:
                    fd = f.filing_date
                    if hasattr(fd, 'year'):
                        f_year = fd.year
                    else:
                        f_year = int(str(fd)[:4])
                    if f_year == fyear:
                        target_filing = f
                        break
                except:
                    continue

        if target_filing is None:
            result['error'] = 'Filing not found'
            return result

        # Try edgartools extraction
        extracted = extract_with_edgartools(target_filing)

        # Fallback to regex if edgartools failed
        if extracted is None:
            try:
                html = target_filing.html()
                if html:
                    extracted = extract_with_regex(html)
            except:
                pass

        if extracted is None:
            result['error'] = 'Extraction failed (both methods)'
            return result

        # Clean the text
        item_1a = clean_text(extracted.get('item_1a'))
        item_7 = clean_text(extracted.get('item_7'))

        # Save extracted text as JSON
        output = {
            'gvkey': gvkey,
            'cik': cik,
            'fyear': fyear,
            'accession_no': accession,
            'item_1a': item_1a,
            'item_7': item_7,
            'method': extracted['method'],
        }

        outpath = os.path.join(EXTRACTED_TEXT_DIR, f"{gvkey}_{fyear}.json")
        with open(outpath, 'w') as f:
            json.dump(output, f)

        result['item_1a_extracted'] = item_1a is not None
        result['item_7_extracted'] = item_7 is not None
        result['item_1a_length'] = len(item_1a) if item_1a else 0
        result['item_7_length'] = len(item_7) if item_7 else 0
        result['method'] = extracted['method']

    except Exception as e:
        result['error'] = str(e)

    return result


def main():
    set_identity(SEC_IDENTITY)
    os.makedirs(EXTRACTED_TEXT_DIR, exist_ok=True)

    # Load filing URLs
    filings = pd.read_csv("data/filing_urls.csv")
    print(f"Processing {len(filings)} filings")

    # Check which ones are already done
    done = set()
    for fname in os.listdir(EXTRACTED_TEXT_DIR):
        if fname.endswith('.json'):
            key = fname.replace('.json', '')
            done.add(key)
    print(f"Already extracted: {len(done)}")

    # Process each filing
    log = []
    for i, (_, row) in enumerate(filings.iterrows()):
        key = f"{row['gvkey']}_{int(row['fyear'])}"
        if key in done:
            continue

        if (i + 1) % 50 == 0:
            print(f"Processing {i+1}/{len(filings)}: CIK {int(row['cik'])}, year {int(row['fyear'])}")

        result = process_one_filing(row)
        log.append(result)

        time.sleep(SEC_RATE_LIMIT_DELAY)

        # Save log periodically
        if len(log) % 200 == 0:
            pd.DataFrame(log).to_csv("data/extraction_log.csv", index=False)

    # Final save
    log_df = pd.DataFrame(log)
    log_df.to_csv("data/extraction_log.csv", index=False)

    # Summary
    print("\n" + "="*60)
    print("EXTRACTION SUMMARY")
    print("="*60)
    print(f"Total filings processed: {len(log_df)}")
    print(f"Item 1A extracted: {log_df['item_1a_extracted'].sum()} ({log_df['item_1a_extracted'].mean()*100:.1f}%)")
    print(f"Item 7 extracted:  {log_df['item_7_extracted'].sum()} ({log_df['item_7_extracted'].mean()*100:.1f}%)")
    print(f"Both extracted:    {(log_df['item_1a_extracted'] & log_df['item_7_extracted']).sum()}")
    print(f"Neither extracted: {(~log_df['item_1a_extracted'] & ~log_df['item_7_extracted']).sum()}")
    if 'method' in log_df.columns:
        print(f"\nExtraction method breakdown:")
        print(log_df['method'].value_counts())


if __name__ == "__main__":
    main()