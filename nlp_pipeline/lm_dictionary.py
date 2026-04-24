"""
lm_dictionary.py
Helper module for loading and applying the Loughran-McDonald Master Dictionary.

Download the dictionary from: https://sraf.nd.edu/loughranmcdonald-master-dictionary/
Save as data/LoughranMcDonald_MasterDictionary.csv
"""

import pandas as pd
import re
from collections import Counter


class LMDictionary:
    """
    Loads the Loughran-McDonald Master Dictionary and provides
    methods for scoring text.
    """

    def __init__(self, dict_path):
        """Load the dictionary CSV and build word sets for each category."""
        df = pd.read_csv(dict_path)

        # The dictionary uses non-zero values to indicate category membership.
        # Column names vary slightly across versions; handle common variants.
        self.categories = {}

        category_columns = {
            'negative': 'Negative',
            'positive': 'Positive',
            'uncertainty': 'Uncertainty',
            'litigious': 'Litigious',
            'strong_modal': 'Strong_Modal',
            'weak_modal': 'Weak_Modal',
        }

        for cat_name, col_name in category_columns.items():
            if col_name in df.columns:
                # Words where the column value is > 0 belong to this category
                words = df[df[col_name] > 0]['Word'].str.lower().tolist()
                self.categories[cat_name] = set(words)
                print(f"  Loaded {len(self.categories[cat_name]):,} {cat_name} words")
            else:
                print(f"  WARNING: Column '{col_name}' not found in dictionary")

        print(f"  Total categories loaded: {len(self.categories)}")

    def tokenize(self, text):
        """
        Simple tokenization: lowercase, split on non-alphabetic characters.
        This matches the approach used in most LM dictionary studies.
        """
        if text is None or len(text) == 0:
            return []
        # Convert to lowercase, split on non-alpha characters
        words = re.findall(r'[a-z]+', text.lower())
        return words

    def score(self, text):
        """
        Score a text using all LM dictionary categories.

        Returns a dict with:
          - word_count: total number of words
          - {category}_count: number of words in each category
          - {category}_fraction: fraction of words in each category

        Fractions are the standard measure used in the literature
        (Loughran & McDonald 2011).
        """
        words = self.tokenize(text)
        total = len(words)

        result = {'word_count': total}

        if total == 0:
            for cat_name in self.categories:
                result[f'{cat_name}_count'] = 0
                result[f'{cat_name}_fraction'] = None
            return result

        # Count words in each category
        for cat_name, word_set in self.categories.items():
            count = sum(1 for w in words if w in word_set)
            result[f'{cat_name}_count'] = count
            result[f'{cat_name}_fraction'] = count / total

        return result

    def score_detailed(self, text):
        """
        Like score(), but also returns the top matching words per category.
        Useful for debugging and understanding what drives the scores.
        """
        words = self.tokenize(text)
        total = len(words)
        word_counts = Counter(words)

        result = self.score(text)

        # Add top matching words for each category
        for cat_name, word_set in self.categories.items():
            matching = {w: c for w, c in word_counts.items() if w in word_set}
            top_words = sorted(matching.items(), key=lambda x: -x[1])[:20]
            result[f'{cat_name}_top_words'] = top_words

        return result


def demo():
    """Quick demo showing how the dictionary works."""
    # Example text (simplified risk factors language)
    sample_text = """
    The company faces significant uncertainty regarding future demand
    for its products. Adverse economic conditions could materially
    impact our revenue and profitability. We may be unable to refinance
    our existing debt obligations, which could result in default on our
    credit facilities. Litigation related to our products could result
    in substantial liability. We cannot predict whether regulatory
    changes will adversely affect our business.
    """

    print("DEMO: Scoring sample text")
    print("-" * 50)

    # You would replace this path with your actual dictionary path
    try:
        lm = LMDictionary("data/LoughranMcDonald_MasterDictionary.csv")
        scores = lm.score_detailed(sample_text)

        print(f"\nTotal words: {scores['word_count']}")
        for cat in ['negative', 'positive', 'uncertainty', 'litigious', 'weak_modal']:
            frac = scores.get(f'{cat}_fraction', 0)
            count = scores.get(f'{cat}_count', 0)
            if frac is not None:
                print(f"\n{cat.upper()}: {count} words ({frac*100:.2f}%)")
                top = scores.get(f'{cat}_top_words', [])
                if top:
                    print(f"  Top words: {', '.join(f'{w}({c})' for w, c in top[:10])}")
    except FileNotFoundError:
        print("Dictionary file not found. Download from https://sraf.nd.edu/")


if __name__ == "__main__":
    demo()
