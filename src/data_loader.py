"""
Data Loading Module for IMDB Movie Reviews Dataset.

Handles downloading the raw IMDB dataset from Stanford AI Lab,
parsing individual review files, and creating a structured DataFrame.
Falls back to Keras built-in dataset if the download fails.
"""

import os
import sys
import urllib.request
import tarfile
import ssl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.utils import print_header, print_info, print_success, print_metric, print_subheader


def _download_progress(count, block_size, total_size):
    """Display download progress bar."""
    percent = min(int(count * block_size * 100 / total_size), 100)
    bar_len = 40
    filled = int(bar_len * percent / 100)
    bar = '#' * filled + '.' * (bar_len - filled)
    print(f"\r  [{bar}] {percent}%", end='', flush=True)


def download_from_stanford(data_dir, url):
    """Download and extract IMDB dataset from Stanford AI Lab."""
    extract_path = os.path.join(data_dir, 'aclImdb')
    tar_path = os.path.join(data_dir, 'aclImdb_v1.tar.gz')

    if os.path.exists(extract_path):
        print_info("Dataset directory already exists. Skipping download.")
        return extract_path

    os.makedirs(data_dir, exist_ok=True)

    print_info(f"Downloading IMDB dataset (~84 MB)...")
    print_info("Source: Stanford AI Lab (ai.stanford.edu)")

    try:
        urllib.request.urlretrieve(url, tar_path, reporthook=_download_progress)
        print()  # newline after progress bar
    except (urllib.error.URLError, ssl.SSLError) as e:
        print(f"\n  [WARN] Download failed: {e}")
        # Try with unverified SSL as fallback
        try:
            print_info("Retrying with relaxed SSL...")
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            opener = urllib.request.build_opener(
                urllib.request.HTTPSHandler(context=ctx)
            )
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(url, tar_path, reporthook=_download_progress)
            print()
        except Exception as e2:
            print(f"\n  [ERROR] Download failed: {e2}")
            if os.path.exists(tar_path):
                os.remove(tar_path)
            return None

    print_info("Extracting dataset...")
    try:
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(data_dir, filter='data')
    except TypeError:
        # Older Python versions don't support filter parameter
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(data_dir)

    # Clean up tar file to save disk space
    if os.path.exists(tar_path):
        os.remove(tar_path)

    print_success("Dataset downloaded and extracted successfully!")
    return extract_path


def load_reviews_from_directory(dataset_path):
    """Load reviews from extracted IMDB directory structure.

    Directory structure:
        aclImdb/
        ├── train/
        │   ├── pos/  (12,500 .txt files)
        │   └── neg/  (12,500 .txt files)
        └── test/
            ├── pos/  (12,500 .txt files)
            └── neg/  (12,500 .txt files)
    """
    reviews = []
    sentiments = []

    for split in ['train', 'test']:
        for label_dir in ['pos', 'neg']:
            folder = os.path.join(dataset_path, split, label_dir)
            label = 1 if label_dir == 'pos' else 0

            if not os.path.exists(folder):
                continue

            files = sorted(os.listdir(folder))
            total = len(files)
            for idx, filename in enumerate(files):
                if not filename.endswith('.txt'):
                    continue
                filepath = os.path.join(folder, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    reviews.append(f.read())
                    sentiments.append(label)

                if (idx + 1) % 5000 == 0:
                    print(f"\r  Reading {split}/{label_dir}: {idx+1}/{total}", end='', flush=True)

            print(f"\r  Reading {split}/{label_dir}: {total}/{total} -- done")

    return pd.DataFrame({'text': reviews, 'sentiment': sentiments})


def load_from_keras_fallback():
    """Fallback: Load IMDB dataset from Keras built-in and decode to text."""
    print_info("Using Keras built-in IMDB dataset as fallback...")

    from tensorflow.keras.datasets import imdb

    # Get word index and create reverse mapping
    word_index = imdb.get_word_index()
    word_index = {k: (v + 3) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2
    reverse_word_index = {v: k for k, v in word_index.items()}

    # Load data
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=None)

    def decode_review(encoded):
        words = []
        for i in encoded:
            word = reverse_word_index.get(i, '')
            if word not in ('<PAD>', '<START>', '<UNK>', ''):
                words.append(word)
        return ' '.join(words)

    print_info("Decoding training reviews...")
    train_texts = [decode_review(x) for x in x_train]
    print_info("Decoding test reviews...")
    test_texts = [decode_review(x) for x in x_test]

    all_texts = train_texts + test_texts
    all_labels = list(y_train) + list(y_test)

    return pd.DataFrame({'text': all_texts, 'sentiment': all_labels})


def load_imdb_dataset(data_dir, url, csv_path):
    """Main entry point: Load IMDB dataset into a pandas DataFrame.

    Tries Stanford download first, falls back to Keras built-in.
    Caches result as CSV for subsequent runs.
    """
    print_header("STEP 1: LOADING IMDB MOVIE REVIEWS DATASET")

    # Check if processed CSV already exists (cache)
    if os.path.exists(csv_path):
        print_info("Loading from cached CSV...")
        df = pd.read_csv(csv_path)
        print_success(f"Loaded {len(df):,} reviews from cache.")
    else:
        # Try primary source: Stanford
        dataset_path = download_from_stanford(data_dir, url)

        if dataset_path and os.path.exists(dataset_path):
            print_info("Parsing review files...")
            df = load_reviews_from_directory(dataset_path)
        else:
            # Fallback to Keras
            df = load_from_keras_fallback()

        # Shuffle
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Cache as CSV
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)
        print_success(f"Cached processed dataset to {os.path.basename(csv_path)}")

    # ─── Display Dataset Overview ────────────────────────────
    print_subheader("DATASET OVERVIEW")
    print_metric("Shape", f"{df.shape[0]:,} rows x {df.shape[1]} columns")
    print_metric("Columns", list(df.columns))

    print(f"\n  First 5 rows:")
    display_df = df.head().copy()
    display_df['text'] = display_df['text'].str[:80] + '...'
    print(display_df.to_string(index=True))

    print_subheader("DATASET STATISTICS")
    print_metric("Total Reviews", f"{len(df):,}")

    print(f"\n  Sentiment Distribution:")
    sentiment_map = {1: 'Positive', 0: 'Negative'}
    for val, count in df['sentiment'].value_counts().sort_index().items():
        pct = count / len(df) * 100
        print(f"    {sentiment_map.get(val, val)}: {count:,} ({pct:.1f}%)")

    print(f"\n  Missing Values:")
    missing = df.isnull().sum()
    for col, count in missing.items():
        print(f"    {col}: {count}")

    duplicates = df.duplicated().sum()
    print(f"\n  Duplicates: {duplicates}")

    if duplicates > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print_info(f"Removed {duplicates} duplicates. New shape: {df.shape}")

    print(f"\n  Data Types:")
    for col, dtype in df.dtypes.items():
        print(f"    {col}: {dtype}")

    return df


def split_dataset(df, test_size, random_state):
    """Split dataset into stratified train and test sets."""
    print_subheader("TRAIN-TEST SPLIT")

    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'],
        df['sentiment'],
        test_size=test_size,
        random_state=random_state,
        stratify=df['sentiment']
    )

    print_metric("Training Set Size", f"{len(X_train):,}")
    print_metric("Test Set Size", f"{len(X_test):,}")
    print_metric("Train Positive Ratio", f"{y_train.mean():.3f}")
    print_metric("Test Positive Ratio", f"{y_test.mean():.3f}")
    print_success("Dataset split complete.")

    return X_train, X_test, y_train, y_test
