"""
Text Preprocessing Pipeline for Sentiment Analysis.

Implements a complete NLP preprocessing pipeline with:
- HTML tag removal, URL removal, emoji handling
- Punctuation, numbers, special character removal
- Stopword removal (preserving negation words)
- Lemmatization using NLTK WordNetLemmatizer
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from src.utils import print_header, print_info, print_success, print_subheader, print_metric

# Download required NLTK resources (silent)
for resource in ['stopwords', 'wordnet', 'omw-1.4']:
    nltk.download(resource, quiet=True)

# ─── Initialize NLP tools ────────────────────────────────────
_stop_words = set(stopwords.words('english'))

# Preserve negation and intensity words (critical for sentiment)
_KEEP_WORDS = {
    'not', 'no', 'nor', 'never', 'neither', 'nobody', 'nothing',
    'nowhere', 'hardly', 'barely', 'scarcely', 'very', 'really',
    'too', 'quite', 'rather', 'most', 'more', 'few', 'little',
    'against', 'above', 'below',
}
_stop_words -= _KEEP_WORDS

_lemmatizer = WordNetLemmatizer()

# Precompiled regex patterns for performance
_HTML_RE = re.compile(r'<[^>]+>')
_URL_RE = re.compile(r'https?://\S+|www\.\S+')
_EMOJI_RE = re.compile(
    "["
    u"\U0001F600-\U0001F64F"  # emoticons
    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    u"\U0001F680-\U0001F6FF"  # transport & map
    u"\U0001F1E0-\U0001F1FF"  # flags
    u"\U00002702-\U000027B0"
    u"\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)
_NUMBERS_RE = re.compile(r'\d+')
_SPECIAL_RE = re.compile(r'[^a-zA-Z\s]')
_SPACES_RE = re.compile(r'\s+')


def preprocess_text(text):
    """Apply the full NLP preprocessing pipeline to a single text.

    Pipeline:
        1. Lowercase conversion
        2. Remove HTML tags
        3. Remove URLs
        4. Remove emojis
        5. Remove punctuation
        6. Remove numbers
        7. Remove special characters
        8. Tokenize
        9. Remove stopwords (preserving negation)
        10. Lemmatization
        11. Remove very short tokens
        12. Rejoin with single spaces

    Args:
        text: Raw input text string.

    Returns:
        Cleaned text string ready for feature extraction.
    """
    if not isinstance(text, str) or len(text) == 0:
        return ''

    # 1. Lowercase
    text = text.lower()

    # 2. Remove HTML tags  (e.g., <br />, <p>, etc.)
    text = _HTML_RE.sub(' ', text)

    # 3. Remove URLs
    text = _URL_RE.sub('', text)

    # 4. Remove emojis
    text = _EMOJI_RE.sub('', text)

    # 5. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # 6. Remove numbers
    text = _NUMBERS_RE.sub('', text)

    # 7. Remove remaining special characters
    text = _SPECIAL_RE.sub('', text)

    # 8. Tokenize
    tokens = text.split()

    # 9. Remove stopwords
    tokens = [t for t in tokens if t not in _stop_words]

    # 10. Lemmatization
    tokens = [_lemmatizer.lemmatize(t) for t in tokens]

    # 11. Remove very short tokens (single chars)
    tokens = [t for t in tokens if len(t) > 1]

    # 12. Rejoin
    return ' '.join(tokens)


def preprocess_dataset(df):
    """Apply preprocessing to the entire dataset with progress tracking.

    Args:
        df: DataFrame with 'text' column.

    Returns:
        DataFrame with added 'cleaned_text', 'original_length',
        'cleaned_length', and 'word_count' columns.
    """
    print_header("STEP 2: TEXT PREPROCESSING")

    # ─── Show Before & After Examples ────────────────────
    print_subheader("BEFORE & AFTER PREPROCESSING")

    examples = [
        'The movie was AMAZING!!! <br /><br /> Best film I\'ve EVER seen!!!',
        'I really hated this film. The acting was terrible :(  2/10',
        'Not bad, not good. Just okay. https://example.com/review 3/5 stars.',
        'WORST movie ever!!! Don\'t waste your $$$money$$$. <b>HORRIBLE</b>',
    ]

    for i, ex in enumerate(examples, 1):
        cleaned = preprocess_text(ex)
        print(f"\n  Example {i}:")
        print(f"    Original : \"{ex}\"")
        print(f"    Cleaned  : \"{cleaned}\"")

    # ─── Apply to Full Dataset ───────────────────────────
    print_info("\nPreprocessing all reviews (this may take a few minutes)...")

    total = len(df)
    batch_size = 5000
    cleaned_texts = []

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = df['text'].iloc[start:end].apply(preprocess_text)
        cleaned_texts.extend(batch.tolist())

        progress = end / total * 100
        bar_len = 30
        filled = int(bar_len * progress / 100)
        bar = '#' * filled + '.' * (bar_len - filled)
        print(f"\r  [{bar}] {progress:.0f}% ({end:,}/{total:,})", end='', flush=True)

    print()  # newline

    df = df.copy()
    df['cleaned_text'] = cleaned_texts

    # ─── Compute Statistics ──────────────────────────────
    df['original_length'] = df['text'].str.len()
    df['cleaned_length'] = df['cleaned_text'].str.len()
    df['word_count'] = df['cleaned_text'].str.split().str.len().fillna(0).astype(int)

    print_subheader("PREPROCESSING STATISTICS")
    print_metric("Avg Original Length", f"{df['original_length'].mean():.0f} chars")
    print_metric("Avg Cleaned Length", f"{df['cleaned_length'].mean():.0f} chars")
    print_metric("Avg Word Count", f"{df['word_count'].mean():.1f} words")
    print_metric("Min Word Count", f"{df['word_count'].min()}")
    print_metric("Max Word Count", f"{df['word_count'].max()}")

    # Handle empty reviews
    empty_count = (df['cleaned_text'].str.strip() == '').sum()
    print_metric("Empty Reviews After Cleaning", empty_count)

    if empty_count > 0:
        df = df[df['cleaned_text'].str.strip() != ''].reset_index(drop=True)
        print_info(f"Removed {empty_count} empty reviews. New size: {len(df):,}")

    print_success(f"Preprocessing complete! Final dataset: {len(df):,} reviews")

    return df
