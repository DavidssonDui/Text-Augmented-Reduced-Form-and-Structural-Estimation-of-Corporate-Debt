"""
04_finbert_scores.py
Compute FinBERT embedding scores for all extracted 10-K text.

For each firm-year:
  1. Split text into sentences
  2. Pass each sentence through FinBERT
  3. Average sentence embeddings to get a 768-dim document embedding
  4. Save the embedding

Requires: pip install transformers torch
Recommended: GPU for reasonable speed (~2-4 hours with GPU, days without)

Input:  data/extracted_text/*.json
Output: output/finbert_embeddings.pkl   (dict of gvkey_fyear -> 768-dim array)
        output/finbert_scores.csv       (projected scores after running 05)
"""

import pandas as pd
import numpy as np
import json
import os
import re
import glob
import pickle
import torch
from transformers import AutoTokenizer, AutoModel
from config import (
    EXTRACTED_TEXT_DIR, FINBERT_SCORES_CSV,
    FINBERT_MODEL, FINBERT_BATCH_SIZE, FINBERT_MAX_LENGTH
)


def split_sentences(text):
    """
    Simple sentence splitting. Splits on period/question mark/exclamation
    followed by space and capital letter. Not perfect but good enough
    for averaging embeddings.
    """
    if text is None:
        return []
    # Split on sentence-ending punctuation followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    # Filter out very short sentences (likely artifacts)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    return sentences


def get_embeddings_batch(sentences, tokenizer, model, device, batch_size=32, max_length=512):
    """
    Get FinBERT embeddings for a list of sentences.
    Returns array of shape (n_sentences, 768).
    """
    all_embeddings = []

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]

        # Tokenize
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )

        # Move to GPU if available
        encoded = {k: v.to(device) for k, v in encoded.items()}

        # Forward pass (no gradient computation needed)
        with torch.no_grad():
            outputs = model(**encoded)

        # Use the [CLS] token embedding (first token) as the sentence embedding
        # Shape: (batch_size, 768)
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls_embeddings)

    if all_embeddings:
        return np.vstack(all_embeddings)
    return np.array([])


def main():
    # Check for GPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    print(f"Using device: {device}")
    if device.type == 'cpu':
        print("WARNING: Running on CPU. This will be very slow for large samples.")
        print("Consider using Google Colab or UBC ARC for GPU access.")

    # Load FinBERT model and tokenizer
    print(f"\nLoading FinBERT model: {FINBERT_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model = AutoModel.from_pretrained(FINBERT_MODEL)
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    print("Model loaded successfully")

    # Find all extracted text files
    text_files = sorted(glob.glob(os.path.join(EXTRACTED_TEXT_DIR, "*.json")))
    print(f"\nFound {len(text_files)} extracted text files")

    # Check for existing embeddings (resume capability)
    embeddings_path = "output/finbert_embeddings.pkl"
    if os.path.exists(embeddings_path):
        with open(embeddings_path, 'rb') as f:
            embeddings_dict = pickle.load(f)
        print(f"Loaded {len(embeddings_dict)} existing embeddings")
    else:
        embeddings_dict = {}

    # Process each filing
    os.makedirs("output", exist_ok=True)

    for i, fpath in enumerate(text_files):
        # Extract key from filename
        key = os.path.basename(fpath).replace('.json', '')

        # Skip if already done
        if key in embeddings_dict:
            continue

        if (i + 1) % 100 == 0:
            print(f"  Processing {i+1}/{len(text_files)}: {key}")

        with open(fpath, 'r') as f:
            doc = json.load(f)

        # Combine Item 1A and Item 7 text
        combined_text = ""
        if doc.get('item_1a'):
            combined_text += doc['item_1a'] + " "
        if doc.get('item_7'):
            combined_text += doc['item_7']

        if len(combined_text) < 100:
            continue

        # Split into sentences
        sentences = split_sentences(combined_text)
        if len(sentences) == 0:
            continue

        # Cap at 500 sentences to keep computation reasonable
        if len(sentences) > 500:
            sentences = sentences[:500]

        # Get embeddings
        try:
            sentence_embeddings = get_embeddings_batch(
                sentences, tokenizer, model, device,
                batch_size=FINBERT_BATCH_SIZE,
                max_length=FINBERT_MAX_LENGTH
            )

            # Average across sentences to get document embedding
            if len(sentence_embeddings) > 0:
                doc_embedding = sentence_embeddings.mean(axis=0)  # Shape: (768,)
                embeddings_dict[key] = doc_embedding

        except Exception as e:
            print(f"  ERROR processing {key}: {e}")
            continue

        # Save periodically
        if len(embeddings_dict) % 500 == 0:
            with open(embeddings_path, 'wb') as f:
                pickle.dump(embeddings_dict, f)
            print(f"  Saved {len(embeddings_dict)} embeddings")

    # Final save
    with open(embeddings_path, 'wb') as f:
        pickle.dump(embeddings_dict, f)

    print("\n" + "="*60)
    print("FINBERT EMBEDDING SUMMARY")
    print("="*60)
    print(f"Total embeddings computed: {len(embeddings_dict)}")
    print(f"Embedding dimension: 768")
    print(f"Saved to: {embeddings_path}")
    print(f"\nRun 05_build_text_signal.py to project embeddings onto risk direction")


if __name__ == "__main__":
    main()
